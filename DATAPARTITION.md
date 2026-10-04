# Partitions, parts and pieces

A `DatatablePartition` splits a `Datatable` into folds: the fit and eval
tables of a probe, say, kept apart by patient and balanced by label. Each
fold is read as a table, a `DatatablePart`. A tab that cannot go to one fold
whole, because its rows belong to several patients or carry several labels,
is split into `DatatabPiece`s.

All three live in `dbx.datatables`:

```python
from dbx.datatables import Datacollator, DatatablePartition

split = DatatablePartition(spec=dict(
    datatable=table, fractions=[0.8, 0.2], partition_slice='tiles', seed=0,
    collator=Datacollator(spec=dict(columns={
        'groupby': ('annotations', 'annotations', 'case_id'),     # the patient
        'stratifyby': ('annotations', 'annotations', 'cohort'),   # the label
    })))).build()
fit_table, eval_table = split.part(0).build(), split.part(1).build()
```

## Terms

- **Partition**: the `DatatablePartition` block. An index: it reads the
  partition columns, decides which rows go to which fold, and writes `tabs`
  and `summary`. It writes no rows.
- **Fold**: an index *k* into `tabs`, the *k*-th of `fractions`.
- **Part**: fold *k* as a table, `partition.part(k)`, a `DatatablePart`. Its
  tabs are the table's own tabs, and pieces of them. `partition.fold(k)` is the
  name `part` had first and does the same.
- **Piece**: a `DatatabPiece`, the rows of **one** source tab that hold **one**
  combination of the partition values: the rows of `tab_000002` whose patient
  is `p4` and cohort is `B`.
- **Core partition**, **core part**, **core tab**: a `DatatableCorePartition`,
  which deals rows by cluster; a fold of it as a table, `DatatableCorePart`;
  and its tabs, `DatatableCoreTab`s, holding rows repacked from anywhere in
  the table. See "Core partitions" below.
- **Entry**: one element of a fold's list in `tabs`. Either a tab index,
  `3`, or a piece's record, `{"tab": 2, "groupby": "p4", "stratifyby": "B"}`.

## What decides a partition

Four choices, each answering one question:

- **groupby**: which rows must land in the SAME fold? Rows sharing its value,
  a patient's, are dealt as one unit. Without it, each tab or piece is a unit
  of its own.
- **stratifyby**: within which categories must the fractions hold? The
  fractions are met within each of its values, not only overall. A group must
  lie in one stratum, or the partition raises.
- **balance**: fractions of what? `'rows'` (the default) or `'tabs'`. A piece
  weighs its rows, or 1.
- **seed**: the order units are dealt in, within each stratum.

groupby and stratifyby are roles of a `Datacollator`'s `columns`, each one
column spec, `(slice, column[, key, ...])`. A column upstream of the table, a
feature table's annotations, is read only with the collator's
`recursive=True`.

`method='largest_first'` is the partition from before these choices: whole
tabs by descending rows, each to the fold furthest below target. It takes
no collator and never splits a tab.

## The flow

```
DatatablePartition.build()
 │
 ├─ 1. SCAN, in parallel, one worker per tab          _scan_, TabPartitionScanCallable
 │     reads each tab's groupby and stratifyby columns
 │       constant tab: its single value per role
 │       mixed tab:    its pieces, {values: {groupby, stratifyby}, rows: count}
 │                     counts, not row lists: a tab may hold millions of rows
 │     then, on the master:
 │       a mixed tab in a table reading upstream  → NotImplementedError
 │       slices or columns of unequal length       → ValueError
 │
 ├─ 2. ITEMS, master                                  _items_
 │     one item per constant tab, one per piece: its entry, values, rows, tag
 │
 ├─ 3. DEAL, master                                   _deal_
 │     an item with no value  → skipped, warned, recorded with its row count
 │     units: the items sharing a groupby value, across tabs and pieces
 │     a group in two strata  → ValueError
 │     per stratum: shuffle the units with the seed; each goes to the fold
 │     furthest below its target
 │
 └─ 4. WRITE
       tabs.json     [[0, 3, {"tab": 2, "groupby": "p4", "stratifyby": "B"}], [...]]
       summary.json  per fold its tags, tabs, pieces, rows, strata; what was skipped
```

A part reads its entries and builds what is missing:

```
partition.part(k)  →  DatatablePart
 │   tab_indices   fold k's entries
 │   tab(i)        an int: the table's own tab, nothing copied
 │                 a record: partition.tab(entry), a DatatabPiece
 │
 └─ part.build()   the ordinary Datatable build, in parallel
       whole tabs: already built, skipped
       pieces:     built here, each reading its source tab
```

### Why DEAL is not parallel

DEAL is a greedy pass, and each of its decisions depends on the ones before:
a unit goes to the fold furthest below its target, and that is known only
after every earlier unit has been placed. Splitting the pass across workers
would deal against stale totals, so it would lose both the balance and the
reproducibility that the seed promises.

It also has nothing to gain from it. DEAL works on counts already in memory:
one item per tab or piece, a few arithmetic operations each, and no reads.
The two passes that read data, SCAN (every
tab's partition columns) and BUILD (every piece's rows), are where the time
goes, and both run in parallel.

## Pieces

Only mixed tabs are split. A tab holding one value per role is dealt whole,
and its part reads it where it is.

A `DatatabPiece` is an ordinary `Datatab`, so everything that reads tabs
reads it unchanged: `data()`, `datastream()`, a probe. Its VAR:

- `tab`: the source tab;
- `columns`: the column spec of each role;
- `values`: the value each role holds in the piece's rows, as read.

So a piece is defined by its value, not by a list of rows. Its hash says
"these rows of that tab": readable, small however many rows it holds, and
the same whatever the deal. Its tag is the source's and the values,
`tab_000002#p4#B`, which is how a probe tells two pieces of one slide apart.

Its slices are the source tab's, declared as the source declares them, and
`source`. A topic of the source that is not a slice belongs to the whole tab,
not to a subset of its rows, and is left out.

Its build re-reads the source's partition columns and finds the rows that
hold its values. It writes those rows, in the same order, to every slice, one
slice at a time, so row *i* of each slice is still row *i* of the others.
It raises if the source's slices have different lengths, or if no row holds
the values any more, as when the source was rebuilt after it was
partitioned.

### The `source` slice

Every repartitioned tab, a piece or a core tab (below), points back with one
slice, **`source`**, declared `DATASLICE(tab='int', row='int')`. Row *i* of
it is the tab of the partitioned table, and the row of that tab, that row *i*
of every other slice was copied from. It is written in lockstep with them,
so it is read like any other slice: `tab.data('source')`, a zip with the
others, a datastream. `piece.source_rows()` and `piece.source_index()`
read it as arrays.

A slice and not a file beside the slices, so the back-pointers travel with
the rows: whatever reads a row of a repartitioned tab can read where it came
from, by the same means and in the same order. A table's slice may not be
named `source`; the partition refuses one.

The value and the indices say the same thing for different readers. The
value is how the partition names a piece: it is stable and needs no rows.
The indices are how another tab can take the same piece of itself (see
below).

## Identities

No identity moved:

- `DatatablePartition` and `DatatablePart` gained no VAR field, topic or
  version.
- A partition whose tabs are all constant writes exactly the `tabs.json` it
  always did: plain ints, sorted the same way. So its hash, its parts and its
  summary are what they were. `test_a_partition_of_constant_tabs_is_what_it_was`
  pins them, from dbx 07cd119.
- A part's BLOCK is its table's TAB, and is in the part's hash. A piece is
  not an instance of the TAB, so `DatatablePart._form_block_` checks the
  piece's source instead.

## Probes

`FeatureAffineLogisticProbe` takes its fit and eval tables ready-made: two
parts of one partition, typically. It does not split anything itself.

- `_check_disjoint_` refuses two tables that share a tab, compared by tag.
  Two pieces of one slide in different parts have different tags and pass.
  The partition put their groups apart.
- `check_probe_inputs` asks a part about its own blocks, its pieces included,
  before any worker starts. If a piece is missing it says so and says to
  build the part. A part that was never built is refused before that, by the
  probe's `validate_vars`.
- `tab_aggregation='mean'` makes each tab one sample, and refuses a tab whose
  rows carry more than one label. A piece is one sample. Stratifying the
  partition by the label column splits a tab of two labels into two pieces of
  one each, which turns a refused tab into two valid samples.

Before 2026-10-01 the probe took one `feature_table`, pooled every tab's
rows, and split them with an unseeded `np.random.permutation` at
`training_fraction`. Rows of one slide, and so one patient, landed on both
sides, class balance was left to chance, and two builds of one hash scored
differently. Fit and eval tables, `_check_disjoint_`, and partitions by group
and stratum replaced it. `FeatureAffineLogisticProber`, the standalone form
of the same split, was removed when nothing used it any more.

## Upstream and downstream tables

A feature table (`Featuretable`, a `DataslicesUpstream`) reads some slices
from its upstream table: its features are its own, its annotations are the
sample table's. Row *i* of a feature tab is the features of row *i* of its
upstream sample tab.

Splitting a mixed feature tab is **not implemented** and raises
`NotImplementedError`. A piece of it would have to be the same piece of its
upstream tab as well, or its features and its annotations would describe
different rows.

The way there is to partition once, upstream, and let downstream tables
follow:

1. Partition the sample table, the one whose tabs hold the rows. Its pieces'
   `source` slices record which rows they took.
2. A downstream part needs no SCAN and no DEAL: its entries are the upstream
   part's, mapped one to one. A whole tab maps to the downstream tab at the
   same index. A piece maps to a downstream piece, defined by its downstream
   tab and the upstream piece, that takes the rows the upstream piece's
   `source` slice names, of the downstream tab.
3. Building the downstream part builds those pieces in parallel, each
   selecting by index. The upstream and downstream parts agree by
   construction: one row selection, made once, upstream.

This still rests on row *i* of a feature tab being row *i* of its upstream
tab, which every `Featuretab` already assumes. Indices give that assumption
a place to be checked: where the feature tab carries a
`shared_upstream_column`, a downstream piece can compare it against the
upstream piece's and refuse a mismatch. The downstream piece class and the
mirrored part are not written yet.

A core partition (below) already does both halves of this for its own
kind: it copies a feature table's upstream columns into its core tabs, flat,
and a table laid out alike takes its layout without clustering anything.

## Core partitions: coresets

A `DatatableCorePartition` deals **rows**, not tabs, by what they hold. It
clusters the table's rows by their values and deals each cluster's rows to
the folds, nearest the center first. Its folds are new tables of new tabs,
`DatatableCorePart`s of `DatatableCoreTab`s, holding only the columns asked
for.

```python
from dbx.datatables import Datacollator, DatatableCorePartition

core = DatatableCorePartition(spec=dict(
    datatable=features, fractions=[0.05, 0.05], partition_slice='features',
    n_clusters=256, rows_per_tab=4096,
    collator=Datacollator(spec=dict(columns={
        'coreby':  [('features', 'final')],                      # clustered by
        'groupby': ('annotations', 'annotations', 'case_id'),    # dealt whole
        'carry':   [('annotations', 'annotations')],             # carried along
    }, recursive=True)))).build()
fit_table, eval_table = core.part(0).build(), core.part(1).build()
```

### The collator names every column it touches

By role:

- **coreby**: the columns rows are clustered by. Each row's are flattened
  and laid end to end into one vector. Required.
- **groupby**, **stratifyby**: as for `DatatablePartition`. A group's rows go
  to one fold, and the fractions hold within each stratum.
- **carry**: columns the folds hold that nothing above reads, such as labels
  for a probe.

A core tab carries the columns of every role and no others: nothing is pulled
by default. With `recursive=True` a column may be upstream of the table, a
feature table's annotations. It is copied like any other, so a core tab is a
plain tab that reads nothing upstream. Features and annotations stay row for
row together because they are copied together.

### Fractions are of the rows

Not normalized. Summing to 1, every row is dealt, and each fold samples every
cluster in proportion. Summing to less, each fold takes the rows nearest each
center: a coreset, which covers the table as its clusters do, and covers more
of it as `n_clusters` grows.

### The flow

```
DatatableCorePartition.build()
 │
 ├─ 1. SCAN, in parallel, one worker per tab           TabCoreScanCallable
 │     each row's coreby columns, flattened: one float32 vector
 │     each row's groupby and stratifyby values
 │       clustering='master':      the vectors come back to the master
 │       clustering='distributed': written to a scratch directory; a seeded
 │                                 sample comes back to start the centers from
 │
 ├─ 2. CLUSTER                                         _cluster_master_ / _cluster_distributed_
 │     'master':      MiniBatchKMeans over every vector, in the master
 │     'distributed': k-means++ on the sample, then Lloyd passes: each pass the
 │                    workers sum their tabs' rows per center (TabCoreStepCallable),
 │                    the master moves the centers; a last pass labels every row
 │     → each row's cluster and distance to its center; the scratch is removed
 │
 ├─ 3. DEAL, master                                    _deal_rows_
 │     a row missing a value → skipped, counted in summary
 │     units: groups, or rows when there is no groupby
 │     a unit's cell: (stratum, its rows' most frequent cluster)
 │     within each cell, units by their rows' mean distance, nearest first;
 │     each to the fold furthest below its target, while it brings that fold
 │     nearer; units no fold wants are left undealt
 │     a fold's target is fraction × rows, accumulated over the cells of the
 │     stratum, so a cell's rounding is made up by the next
 │
 ├─ 4. COVERAGE, workers per tab ('distributed') or master   _coverage_, TabCoverageCallable
 │     how tightly each fold fills out the table -- see "Coverage" below
 │
 └─ 5. LAYOUT, master
       each fold's rows as (tab, row, cluster), in source order (or shuffled,
       with shuffle=True), cut into core tabs of rows_per_tab
       topics: layout/<fold>.npy, tabs, centers, columns, coverage, summary

partition.part(k).build()   the core tabs, in parallel
       each reads the source tabs its rows are in, only the columns carried,
       and writes them, with `source`, in layout order
```

Source order keeps a core tab's reads local: its rows come from a few
neighbouring tabs. A shuffled layout mixes the table into every core tab,
and each core tab then reads up to as many tabs as it has rows.

### Clustering, and its parameters

All in `VAR`, with defaults:

| Field | Default | |
|---|---|---|
| `clustering` | `'master'` | `'master'`: the master holds every vector. `'distributed'`: it holds only the centers, for a table too large to gather |
| `n_clusters` | 64 | clamped to the rows there are |
| `max_iter`, `tol` | 100, 1e-4 | k-means iterations and convergence (center shift, relative) |
| `batch_size` | 4096 | `MiniBatchKMeans`'s, for `'master'` |
| `init_size` | 65536 | rows sampled to start `'distributed'`'s centers |
| `rows_per_tab` | 4096 | a core tab's rows |
| `shuffle` | False | each fold's rows in a seeded random order, not source order |
| `coverage_sample` | 10000 | table rows (and rows of each fold) sampled to measure coverage; 0 measures nothing |
| `coverage_clusters` | 3 | the nearest clusters a sampled row is compared within |
| `layout` | None | another core partition whose layout to take (below) |

`seed` seeds the clustering, the tie-breaks of the deal, and the shuffle.
`method` and `balance` are refused: a core partition deals rows, by cluster.

### Stratifying

`stratifyby` matters here for a reason of its own. The clusters stratify the
rows by what they hold, not by their label, and a cluster may mix cohorts.
Stratified, each cohort's rows in a cluster are ranked and dealt apart from
the others', so each fold takes its share of every cohort, and a coreset
takes a rare cohort's central rows even where a common one dominates the
cluster.

A fold's target accumulates over the cells of a stratum, cluster by cluster,
so what one cell rounds off the next makes up. A stratum's total is within
half a unit of its share: half a row, or half a group with `groupby`. A
group whose rows lie in two strata is refused.

### Coverage

How tightly does a fold "fill out" the table? The `coverage` topic, a
`DATADICT` (`coverage.npz`), answers with distances in the space the rows
were clustered in, the flattened `coreby` vectors.

The measure is the **covering distance**: for a row *x* of the table, the
distance from *x* to the nearest row of the fold, *d(x, C)*. If every row
of the table is close to some row of the fold, the fold fills it out, and
the largest *d(x, C)* is the radius of the worst hole. The opposite measure
is the **separation**: for a row of the fold, the distance to the nearest
*other* row of the fold. Small separations mean the fold spends rows on
near-duplicates. A good coreset has small coverage and large separation.

All pairs would cost O(N²). Instead:

- **Sampled rows.** `coverage_sample` rows of the table, seeded and spread
  over all tabs, are the queries for coverage; the same number of each
  fold's rows, for separation. The quantiles are estimates from that sample.
  The median and p90 are sound; the sample's maximum is only a lower bound
  on the table's worst hole.
- **Nearest clusters only.** A query is compared only with the rows in its
  `coverage_clusters` nearest clusters. That cuts the cost from all rows to a
  few clusters' worth, and is exact whenever a query's nearest row lies in one
  of them, the usual case for k-means clusters, though not a guarantee.
  Otherwise the distance is an over-estimate: coverage can only look worse, never better,
  than it is. With `coverage_clusters >= n_clusters` it is exact.
- **Tab by tab.** Each tab's rows are compared with the queries in one
  worker (`TabCoverageCallable`), and the master keeps the minimum over tabs.
  Under `'distributed'` the workers read the scratch vectors, which are
  removed only after this pass.

The arrays, each with one entry per quantile in `levels`
(0.5, 0.9, 0.99, 1.0):

| Key | Shape | What |
|---|---|---|
| `levels` | (4,) | the quantile levels |
| `sample` | (S, 2) | the sampled table rows, `(tab, row)` |
| `scale` | (4,) | each sampled row's distance to its nearest *other* row of the table: the data's own spacing |
| `coverage` | (F+1, 4) | *d(x, fold)* for the sampled rows; one row per fold, the last for all folds together |
| `coverage_mean` | (F+1,) | their means |
| `baseline` | (F+1, 4) | the same for a seeded random subset of the table of the fold's size |
| `baseline_mean` | (F+1,) | their means |
| `separation` | (F, 4) | each sampled fold row's distance to its nearest other row of the fold |
| `cluster_rows` | (F+1, K) | rows per cluster: the table's first, then each fold's |
| `cluster_tv` | (F,) | total variation between a fold's cluster shares and the table's: 0 is the same mix |
| `empty_clusters` | (F,) | clusters the table has rows in and the fold has none of |

How to read them:

- **Against `scale`.** A coverage median near the table's own spacing means
  the fold is nearly as dense as the table; several times it means the fold
  is coarse, as a 5% coreset must be.
- **Against `baseline`.** The same number of rows drawn at random. A coreset
  below its baseline fills the table out better than chance; above it, worse.
  Nearest-first sampling takes the rows at the centers, so with few clusters
  it covers the centers and leaves the edges. Its coverage then shows worse
  tails than random, and more clusters, approaching the core's size, fix
  that.
- **`cluster_tv` and `empty_clusters`** say whether every region is
  represented in proportion, exactly, over all rows, not a sample.
- Fractions summing to 1 put every row in some fold, so the all-folds row of
  `coverage` is 0.

A partition built from a `layout` takes the layout's coverage: it describes
the rows the layout took, measured where they were clustered.

### Reusing a layout

The layout records, for every row of every fold, the tab and row of the
partitioned table it came from. A table **laid out alike**, with the same
tabs holding the same number of rows, such as a feature table and the table
it was computed from, takes it whole:

```python
again = DatatableCorePartition(spec=dict(
    datatable=samples, fractions=[0.05, 0.05], partition_slice='tiles', layout=core,
    collator=Datacollator(spec=dict(columns={'carry': [('tiles', 'image')]}))))
```

Nothing is scanned, clustered or dealt. Its collator names only `carry`, its
core tabs hold the same rows in the same places, and their `source` slices
agree with `core`'s. Every tab's row count is checked against the layout's
table first, and a table that differs is refused.

### Probes on core parts

A core part's tabs are its own, holding every slice they read, so a probe
reads it as it reads any table, with a collator that is not `recursive`.
`check_probe_inputs` checks the core tabs themselves. With `groupby`, a
group's rows are in one fold, so fit and eval share no patient.

## Where the code is

All in `dbx/datatables.py`, except the probe check.

| What | Where |
|---|---|
| Reading a partition column, following dict keys | `_column_values_` |
| How two values are compared | `_value_key_` (sorted JSON) |
| A piece's tag | `_piece_tag_` |
| The per-tab scan worker | `TabPartitionScanCallable` |
| SCAN, ITEMS, DEAL, the summary | `DatatablePartition._scan_`, `_items_`, `_deal_`, `_summary_` |
| An entry to its tab or piece | `DatatablePartition.tab` |
| A fold as a table | `DatatablePartition.part`, `DatatablePart` |
| A part accepting pieces as blocks | `DatatablePart.tab`, `valid_tab`, `_form_block_` |
| The piece | `DatatabPiece`: `_rows_`, `__build__`, `source_rows` |
| The back-pointing slice | `SOURCE_SLICE`, `SOURCE` |
| Core: scan, cluster, deal, layout | `DatatableCorePartition`: `_cluster_and_deal_`, `_cluster_master_`, `_cluster_distributed_`, `_deal_rows_`, `_reuse_layout_` |
| Core: the workers | `TabCoreScanCallable`, `TabCoreStepCallable`, `TabRowCountCallable`, `TabCoverageCallable`, `TabVectorsCallable` |
| Core: coverage | `DatatableCorePartition._coverage_` |
| Core: a fold and its tabs | `DatatableCorePart`, `DatatableCoreTab` |
| The probe's check before reading | `check_probe_inputs` in `dbx/probes.py` |

The tests are `tests/test_partition_methods.py` (dealing, pieces, probes on
parts), `tests/test_partition_fold.py` (parts as views) and
`tests/test_core_partition.py` (core partitions).
