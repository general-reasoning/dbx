# Partitions, parts and pieces

A `DatatablePartition` splits a `Datatable` into **parts**: tables, each a
`DatatablePart`, such as a probe's fit and eval tables. Part *k* takes the
*k*-th of `fractions` of the rows (or of the tabs), and `partition.part(k)`
returns it. The index *k* is also called the part's fold; `partition.fold(k)`
is the name `part(k)` had first, and does the same. The `DatatablePartition`
itself writes no rows: it decides which rows go to which part, and writes
that down in `tabs` and `summary`.

Two optional columns constrain which rows go to which part:

- **stratifyby**: rows sharing a stratifyby value, a **stratum** (a patient, say),
  go to **one** part. A stratum never straddles parts.
- **partitionby**: rows sharing a partitionby value, a **partition** (a cohort,
  say), are spread over **all** the parts in the proportions of `fractions`,
  so every part takes its share of every partition. The partitions keep their
  proportions in every part; they are not made equal.

A tab is dealt to a part whole when all of its rows carry the same stratum
and, with partitionby, the same partition. Otherwise it is cut into
`DatatabPiece`s. "When a tab is cut" below says exactly when and how. Once
the deal is done, *coalesce* can make a cut tab whole again, and pool small
tabs and pieces into packed tabs, `DatatablePackTab`s: "What a part's tabs
are" below.

The partition and the classes it makes live in `dbx.dataparts`; the collator
naming its columns, in `dbx.datatables`:

```python
from dbx.datatables import Datacollator
from dbx.dataparts import DatatablePartition

split = DatatablePartition(spec=dict(
    datatable=table, fractions=[0.8, 0.2], partition_slice='tiles', seed=0,
    collator=Datacollator(spec=dict(columns={
        'stratifyby': ('annotations', 'annotations', 'case_id'),     # strata: patients
        'partitionby': ('annotations', 'annotations', 'cohort'),   # partitions: cohorts
    })))).build()
fit_table, eval_table = split.part(0).build(), split.part(1).build()
```

## What decides the deal

Five choices, each answering one question:

- **stratifyby**: which rows must land in the SAME part? The rows of a stratum,
  across all the tabs they lie in, are dealt as one **unit**. Without
  stratifyby, each tab or piece is a unit of its own.
- **partitionby**: within which categories must the fractions hold? Each
  partition is dealt to the parts separately, so the fractions are met within
  each partition, not only overall. A partition spans every part; a stratum spans
  one.
- **balance**: fractions of what? `'rows'` (the default) or `'tabs'`. A piece
  weighs its rows, or 1.
- **seed**: the order units are dealt in, within each partition.
- **coalesce**: what a part's tabs are, once dealt. `None` (the default):
  whole tabs and pieces, as dealt. `'whole'`: a cut tab is made whole again
  where it can be. `'pack'`: and small tabs and pieces are pooled into packed
  tabs. It never moves a row to another part. See "What a part's tabs are".

stratifyby and partitionby are roles of a `Datacollator`'s `columns`, each one
column spec, `(slice, column[, key, ...])`. A column upstream of the table, a
feature table's annotations, is read only with the collator's
`recursive=True`.

`method='largest_first'` is the partition from before these choices: whole
tabs by descending rows, each to the part furthest below target. It takes
no collator and never cuts a tab.

## When a tab is cut

Units are dealt one partition at a time, and each unit goes to one part. So
everything dealt must lie in **one** stratum, which goes to one part, and in
**one** partition, which is the pool it is dealt from. A tab that already does
is dealt whole. A tab that does not is cut into pieces, one per combination
of stratum and partition among its rows. A piece takes those rows wherever they
lie in the tab, in their order.

What that comes to depends on the roles the collator sets. The figures below
split one table of five tabs three ways, with `fractions=[0.5, 0.5]` and
`seed=4`. Each row shows its stratum, `s`, and its partition, `p`. A column
the `DatatablePartition` reads is filled with the color of the part the row
goes to; a column it does not read is grey. The strata lie in partitions: s1
and s3 in p1, the others in p2. A tab marked "(split)" has rows in both
parts. The figures are drawn from real runs by
[make_figures.py](figures/datapartition/make_figures.py).

**stratifyby only: a tab is cut by stratum.** Tabs 1, 3 and 4 hold two strata
each, so each is cut in two. Tabs 0 and 2 hold only s1 and are dealt whole,
together, as s1's unit, to part 0. Tab 1 is split: s2's piece goes to part 0
and s3's to part 1. s3 also has a piece in tab 4, and it goes to part 1 with
s3's unit. Tabs 3 and 4 are cut too, although both pieces of each go to
part 1: tabs are cut before the deal, by what they hold, not by where their
strata land. `coalesce='whole'` makes them whole again; see "What a part's
tabs are". The partitions are not read.

![stratifyby only](figures/datapartition/stratifyby.svg)

**partitionby only: a tab is cut by partition.** Tabs 1 and 4 hold two partitions
each, so each is cut in two, and both are split: their p1 rows go to part 0
and their p2 rows to part 1. Tab 3 holds two strata, s4 and s6, but one
partition, p2, and is dealt whole, since strata are not read. Each tab and
piece is a unit of its own, so nothing keeps a stratum together: s1's tabs 0
and 2 go to different parts.

![partitionby only](figures/datapartition/partitionby.svg)

**Both: a stratum goes to one part, and each partition's strata are divided
between the parts.** A stratum lies in one partition, so cutting by stratum and
partition gives the same pieces as cutting by stratum alone: these are the
pieces of the stratifyby figure. What partitionby changes is the deal. The
strata of p1 (s1, s3) are divided between the parts, and so, separately, are
the strata of p2 (s2, s4, s5, s6). Each part gets its share of each partition,
as nearly as whole strata allow. Tabs 1 and 3 end up split.

![stratifyby and partitionby](figures/datapartition/both.svg)

**A stratum in two partitions is refused.** Cutting it by partition would
give two pieces of one stratum, which must go to one part but would be dealt
from two pools. The `DatatablePartition` raises `ValueError`. Strata must lie
inside partitions: patients within cohorts work, but patients whose tiles
each carry a partition of their own do not. To use such a column, either
partition by a value of the whole stratum, or drop stratifyby, so that each
tab is cut by partition and its pieces are dealt apart.

![a stratum in two partitions](figures/datapartition/refused.svg)

## How the units are dealt

Within each partition, in sorted order:

1. The partition's weight, its rows or its tabs and pieces, sets each part's
   target: `target[k] = weight × fraction[k]`.
2. The partition's units are shuffled with the seed. One generator shuffles
   every partition in turn, so the whole deal depends on the seed.
3. Each unit in turn goes to the part furthest below its target, measured
   relatively, `(target[k] − have[k]) / target[k]`. A tie goes to the lower
   *k*.

The deal of the "both" figure:

| Partition | Unit | Its tabs and pieces | Weight | Deficits before, part 0 / 1 | To part | `have` after |
|---|---|---|---|---|---|---|
| p1, targets 5 / 5 | s3 | tab 1 #s3, tab 4 #s3 | 3 | 1.00 / 1.00 | 0 (tie) | 3 / 0 |
| | s1 | tab 0, tab 2 | 7 | 0.40 / 1.00 | 1 | 3 / 7 |
| p2, targets 4.5 / 4.5 | s5 | tab 4 #s5 | 3 | 1.00 / 1.00 | 0 (tie) | 3 / 0 |
| | s2 | tab 1 #s2 | 2 | 0.33 / 1.00 | 1 | 3 / 2 |
| | s4 | tab 3 #s4 | 2 | 0.33 / 0.56 | 1 | 3 / 4 |
| | s6 | tab 3 #s6 | 2 | 0.33 / 0.11 | 0 | 5 / 4 |

**Units are not cut to fit, so the fractions hold only as finely as the
units allow.** The deal does not ask whether a unit fits: it goes to the
part furthest below target, even if that overshoots. In partition p1 above, s1
alone holds 7 of the 10 rows, so the split is 3 : 7 against 5 : 5. A partition
of few, large strata meets its fractions loosely. `summary.json` gives the
rows each part got per partition.

## What a part's tabs are: coalesce

Tabs are cut before the deal, by what they hold, not by where their strata
land. So a tab may be cut although all its pieces go to one part, like
tabs 3 and 4 in the stratifyby figure. A piece is a copy of its rows, while a
whole tab is read where it is. *coalesce* decides, once the deal is done,
what each part's tabs are. It moves no row from one part to another: the
parts hold the same rows in every mode, and only the tabs holding them
differ.

**`coalesce=None`**, the default: the entries as dealt, whole tabs and
pieces. This is what a `DatatablePartition` did before *coalesce*.

**`coalesce='whole'`**: a cut tab whose pieces all went to one part is made
whole again in that part, the table's own tab, read where it is, whatever
strata and partitions its rows hold. A tab stays cut only when a piece of it is
in no part, skipped for holding no value: whole, the tab would bring those
rows back.

The stratifyby figure again, under `'whole'`. Tabs 3 and 4 are whole; tab 1 is
split between the parts, so it stays cut:

![coalesce='whole'](figures/datapartition/whole.svg)

**`coalesce='pack'`**: as `'whole'`, and then each part's entries of fewer
than `min_tab_rows` rows, whole tabs and pieces alike, are pooled in source
order into packed tabs of `max_tab_rows` rows (twice `min_tab_rows` unless
given). An entry may run across two packed tabs, and the last packed tab of
a part holds what is left. Larger entries stay as they are, so nothing is
copied that is not small. With `min_tab_rows=4`, tab 1's s2 piece and tab 2
are pooled into part 0's packed tab 0:

![coalesce='pack'](figures/datapartition/pack.svg)

So a tab of a part may hold the rows of several strata, partitions and source
tabs: a rejoined tab, a packed one. A probe that needs the rows of one slide
takes them by a column, its `aggregateby`, not by tab: see "Probes".

The `summary` says what was done: `rejoined`, the tags of the tabs made
whole, and, per part, `packed`, how many packed tabs hold how many entries
and rows.

## The flow

`DatatablePartition.build()` runs five steps. Each hands the next one thing:

```
the table ─SCAN─▶ a scan per tab ─ITEMS─▶ items ─DEAL─▶ each part's entries ─COALESCE─▶ each part's entries ─WRITE─▶ tabs.json, summary.json
```

1. **SCAN** reads the table, in parallel, one worker per tab
   (`_scan_`, `TabPartitionScanCallable`). Each worker reads its tab's
   stratifyby and partitionby columns and returns a *scan* of the tab: its
   row count, and the distinct values each column holds in it. A tab whose
   columns each hold one value is *constant*, and that one value per column
   is all its scan reports. A tab whose columns hold several is *mixed*, and
   its scan also lists its pieces: each combination of values its rows hold,
   with how many rows hold it. Only counts come back, never the rows, since a
   tab may hold millions of them. Once every scan is in, the master refuses
   two kinds of mixed tab: one in a table that reads slices upstream
   (`NotImplementedError`, see "Upstream and downstream tables"), and one
   whose slices or columns differ in length (`ValueError`), since a piece
   takes its rows by index.

2. **ITEMS** turns the scans into the things to be dealt (`_items_`): one
   item per constant tab, and one per piece of each mixed tab. An item
   carries its entry (the tab's index, or the piece's record), its values,
   its row count and its tag.

3. **DEAL** gives each item to a part (`_deal_`). An item missing a value
   is set aside first: in no part, warned of, and recorded in the summary,
   a piece with its row count. The rest are gathered into units, each holding every
   item of one stratum, from whichever tabs; without stratifyby, each item is
   a unit of its own. A stratum whose items lie in two partitions is refused
   (`ValueError`). Then, one partition at a time, the units are shuffled with
   the seed, and each goes to the part furthest below its target, as "How
   the units are dealt" describes. What comes out is a list of entries for
   each part.

4. **COALESCE** reshapes those lists, when *coalesce* is set (`_rejoin_`,
   `_pack_`). It moves no entry to another part. With `'whole'`, where all
   of a cut tab's pieces are in one part's list, and none was set aside,
   they are replaced there by the tab's own index. With `'pack'`, each
   part's entries of fewer than `min_tab_rows` rows are then taken out of
   its list, and replaced by packed tabs' records: each a list of segments,
   pieces of those entries, filling up to `max_tab_rows` rows. Like DEAL, it
   works on the counts alone.

5. **WRITE** stores the result. `tabs.json` holds each part's list of
   entries, for instance
   `[[{"tab": 1, "stratifyby": "s3", "partitionby": "p1"}, ...], [0, 2, ...]]`.
   `summary.json` says, for each part, which tabs it holds by tag, how many
   tabs and pieces, how many rows in all and in each partition, and what was
   packed; and, for the whole, what was set aside and what was rejoined.

Each part's list in `tabs.json` holds its **entries**: a tab index, `0`,
for a whole tab; a record, `{"tab": 3, "stratifyby": "s6", "partitionby": "p2"}`,
for a piece; and, with `coalesce='pack'`, a packed tab's record,
`{"fold": 0, "pack": 0, "segments": [[{"tab": 1, "stratifyby": "s2"}, 0, 2], [2, 0, 3]]}`:
rows *offset* to *offset + length* of each entry, in order.

A part, `partition.part(k)`, is a `DatatablePart` reading part *k*'s list.
Its tab *i* is what entry *i* names: for a tab index, the table's own tab,
with nothing copied; for a piece's record, a `DatatabPiece`; for a packed
tab's record, a `DatatablePackTab`, both made by `partition.tab(entry)`.
`part.build()` is the ordinary table build, in parallel. The whole tabs are
built already and are skipped; each piece is built from its source tab, and
each packed tab from the tabs its segments name.

### Why DEAL is not parallel

DEAL is a greedy pass, and each of its decisions depends on the ones before:
a unit goes to the part furthest below its target, and that is known only
after every earlier unit has been placed. Splitting the pass across workers
would deal against stale totals, so it would lose both the balance and the
reproducibility that the seed promises.

It also has nothing to gain from it. DEAL works on counts already in memory:
one item per tab or piece, a few arithmetic operations each, and no reads.
The two passes that read data, SCAN (every
tab's partition columns) and BUILD (every piece's and packed tab's rows),
are where the time goes, and both run in parallel. COALESCE, like DEAL,
works on counts in memory.

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
`tab_000004#s5#p2`, which is how a probe tells two pieces of one slide apart.

Its slices are the source tab's, declared as the source declares them, and
`source`. A topic of the source that is not a slice belongs to the whole tab,
not to a subset of its rows, and is left out.

Its build re-reads the source's partition columns and finds the rows that
hold its values. It writes those rows, in the same order, to every slice, one
slice at a time, so row *i* of each slice is still row *i* of the others.
It raises if the source's slices have different lengths, or if no row holds
the values any more, as when the source was rebuilt after it was
partitioned.

### Packed tabs

A `DatatablePackTab` is an ordinary `Datatab` too. Its VAR is its place,
`partition`, `fold` and `index`, as a core tab's is: what it holds is what
its partition's record says that place holds. Its tag says the same,
`pack<partition hash>_f0_000000`.

Its slices are the partitioned table's, declared as its tabs declare them,
and `source`. Its build takes its segments in order. A whole tab's segment
takes rows of all of that tab's; a piece's selects the piece's rows by value
first, as the piece does. The rows are written to every slice, one slice at
a time and a segment at a time, so only one source slice is held whole. It
raises if a segment's tab no longer holds the rows the record counted.

### The `source` slice

Every repartitioned tab, a piece, a packed tab or a core tab (below), points
back with one slice, **`source`**, declared `DATASLICE(tab='int', row='int')`. Row *i* of
it is the tab of the partitioned table, and the row of that tab, that row *i*
of every other slice was copied from. It is written in lockstep with them,
so it is read like any other slice: `tab.data('source')`, a zip with the
others, a datastream. `piece.source_rows()` reads it as an array, and so
does `source_index()`, on a piece, a packed tab or a core tab.
`part.source_index()` reads a whole part's: every row of it, as the
`(tab, row)` of the partitioned table, whatever tab holds it.

A slice and not a file beside the slices, so the back-pointers travel with
the rows: whatever reads a row of a repartitioned tab can read where it came
from, by the same means and in the same order. A table's slice may not be
named `source`; the partition refuses one.

The value and the indices say the same thing for different readers. The
value is how the partition names a piece: it is stable and needs no rows.
The indices are how another tab can take the same piece of itself (see
below).

## Identities

- **The roles were renamed on 2026-10-10.** `groupby` became `stratifyby`,
  and what was `stratifyby` became `partitionby`. A collator naming
  `stratifyby` spells, and so hashes, exactly as it did, and now means the
  opposite. So `DatatablePartition` and `DatatableCorePartition` went to
  `VERSION = 2`, and no specialization reaches a partition built before:
  partitions, their parts, packed tabs and core tabs are rebuilt. Only the
  `largest_first` partitions, which take no collator, are still reached.
  `test_a_partition_from_before_the_roles_were_renamed_is_not_taken_for_one_now`
  says so.
- **Pieces are kept.** A piece is the rows of a tab holding a value in a
  column, whichever role read the column, so its hash and its rows did not
  change. A piece built under `dbx.datatables`, before the move to
  `dbx.dataparts` on 2026-10-09, is reached through a specialization
  anchored there: a hash names no module, but a block is stored under its
  class's.
- **`coalesce`, `min_tab_rows` and `max_tab_rows`**, added on 2026-10-09,
  are in the hash at their defaults too. A core partition takes no
  *coalesce*: its VAR is built on the deal's fields alone,
  `DatatablePartition._DealVAR_`.
- A partition of constant tabs, under `coalesce=None`, deals exactly as it
  always did, and writes the same `tabs.json`: plain ints, sorted the same
  way. `test_a_partition_of_constant_tabs_is_what_it_was` pins the deal of
  dbx 07cd119.
- A part's BLOCK is its table's TAB, and is in the part's hash. A piece is
  not an instance of the TAB, so `DatatablePart._form_block_` checks the
  piece's source instead. A packed tab packs the table's own tabs, and is
  not checked again.

## Probes

`FeatureAffineLogisticProbe` takes its fit and eval tables ready-made: two
parts of one partition, typically. It does not split anything itself.

- **A sample** is a row by default. With `aggregation='mean'` it is the mean
  of the rows sharing a value of the collator's **`aggregateby`** column (a
  slide's id, say) and a label, from whichever tabs they lie in. Each worker
  sums its tab's share of each (value, label), and the probe divides once
  every share is in. The rows of one value carrying two labels are two
  samples. This is what makes rejoined, packed and core tabs usable: the
  rows of a slide are named by a column, not by the tab that happens to hold
  them.
- `signal_reduction='mean'` is the other mean, and works within a row, never
  across rows: a signal whose value per row is `(T, D)` tokens is averaged
  over `T` before it is flattened.
- `_check_disjoint_` refuses two tables that share a tab, compared by tag.
  Two pieces of one slide in different parts have different tags and pass:
  the partition put their strata apart. Packed and core tabs are tagged by
  where they are, which says nothing of what they hold. So two parts of
  **different** partitions of one table are also compared row by row, by
  `part.source_index()`, and refused if they share a row. Two parts of one
  partition are disjoint by its deal.
- With an `aggregateby` column, the two tables must share no value of it,
  whatever `aggregation` is. A row of no value is refused, unless
  `skip_missing` skips it.
- `check_probe_inputs` asks a part about its own blocks, its pieces and
  packed tabs included, before any worker starts. If a piece is missing it
  says so and says to build the part. A part that was never built is refused
  before that, by the probe's `validate_vars`.

Before 2026-10-01 the probe took one `feature_table`, pooled every tab's
rows, and split them with an unseeded `np.random.permutation` at
`training_fraction`. Rows of one slide, and so one patient, landed on both
sides, class balance was left to chance, and two builds of one hash scored
differently. Fit and eval tables, `_check_disjoint_`, and partitions by
stratifyby and partitionby replaced it. `FeatureAffineLogisticProber`, the standalone form
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
the parts, nearest the center first. Its parts are new tables of new tabs,
`DatatableCorePart`s of `DatatableCoreTab`s, holding only the columns asked
for.

```python
from dbx.datatables import Datacollator
from dbx.dataparts import DatatableCorePartition

core = DatatableCorePartition(spec=dict(
    datatable=features, fractions=[0.05, 0.05], partition_slice='features',
    n_clusters=256, rows_per_tab=4096,
    collator=Datacollator(spec=dict(columns={
        'coreby':  [('features', 'final')],                      # clustered by
        'stratifyby': ('annotations', 'annotations', 'case_id'),    # dealt whole
        'carry':   [('annotations', 'annotations')],             # carried along
    }, recursive=True)))).build()
fit_table, eval_table = core.part(0).build(), core.part(1).build()
```

### The collator names every column it touches

By role:

- **coreby**: the columns rows are clustered by. Each row's are flattened
  and laid end to end into one vector. Required.
- **stratifyby**, **partitionby**: as for `DatatablePartition`. A stratum's rows go
  to one part, and the fractions hold within each partition.
- **carry**: columns the parts hold that nothing above reads, such as labels
  for a probe.

A core tab carries the columns of every role and no others: nothing is pulled
by default. With `recursive=True` a column may be upstream of the table, a
feature table's annotations. It is copied like any other, so a core tab is a
plain tab that reads nothing upstream. Features and annotations stay row for
row together because they are copied together.

### Fractions are of the rows

Not normalized. Summing to 1, every row is dealt, and each part samples every
cluster in proportion. Summing to less, each part takes the rows nearest each
center: a coreset, which covers the table as its clusters do, and covers more
of it as `n_clusters` grows.

### The flow

`DatatableCorePartition.build()` runs five steps, and the parts then build
the core tabs:

```
the table ─SCAN─▶ a vector and values per row ─CLUSTER─▶ a cluster and distance per row ─DEAL─▶ a part per row ─COVERAGE─▶ coverage ─LAYOUT─▶ each part's rows, cut into core tabs
```

1. **SCAN** reads the table, in parallel, one worker per tab
   (`TabCoreScanCallable`). For each row it flattens the `coreby` columns
   and lays them end to end into one `float32` vector, and reads the row's
   stratifyby and partitionby values. The values always come back to the
   master. Where the vectors go depends on `clustering`: with `'master'`
   they come back too; with `'distributed'` each worker writes its tab's to
   a scratch directory, and only a seeded sample of them comes back, to
   start the centers from.

2. **CLUSTER** assigns every row to a cluster (`_cluster_master_`,
   `_cluster_distributed_`). With `'master'`, `MiniBatchKMeans` runs over
   all the vectors in the master. With `'distributed'`, k-means++ picks the
   starting centers from the sample, and Lloyd passes refine them: in each
   pass the workers sum their tabs' rows per center (`TabCoreStepCallable`),
   and the master moves the centers to the means. A last pass gives each row
   its nearest center. What comes out is each row's cluster, and its
   distance to the cluster's center.

3. **DEAL** gives rows to parts, a unit at a time (`_deal_rows_`). A row
   missing a value is set aside first, and counted in the summary. A unit is
   a stratum's rows, or a single row when there is no stratifyby; a stratum
   in two partitions is refused. Each unit belongs to a *cell*: its
   partition, and the cluster most of its rows are in. Cell by cell, the
   units are taken nearest first, by their rows' mean distance to their
   centers, and each goes to the part furthest below its target, provided
   it brings that part nearer its target rather than past it. A unit no part
   has room for is left undealt, which is how fractions summing to less
   than 1 make a coreset. A part's target is its fraction of the rows,
   accumulated over the cells of a partition, so what one cell rounds off,
   the next makes up. What comes out is each row's part, or none.

4. **COVERAGE** measures how tightly each part fills out the table
   (`_coverage_`, `TabCoverageCallable`), in the workers, tab by tab, with
   `'distributed'`, and in the master otherwise. See "Coverage" below. Only
   now, with the distances measured, is the scratch directory removed.

5. **LAYOUT** writes each part's rows down, in the master: as
   `(tab, row, cluster)`, in source order (or shuffled, with
   `shuffle=True`), cut into core tabs of `rows_per_tab` rows. The topics
   are `layout/<fold>.npy`, `tabs`, `centers`, `columns`, `coverage` and
   `summary`.

`partition.part(k).build()` then builds part *k*'s core tabs, in parallel.
Each reads the source tabs its rows are in, only the columns carried, and
writes them, with `source`, in the layout's order.

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
| `shuffle` | False | each part's rows in a seeded random order, not source order |
| `coverage_sample` | 10000 | table rows (and rows of each part) sampled to measure coverage; 0 measures nothing |
| `coverage_clusters` | 3 | the nearest clusters a sampled row is compared within |
| `layout` | None | another core partition whose layout to take (below) |

`seed` seeds the clustering, the tie-breaks of the deal, and the shuffle.
`method` and `balance` are refused: a core partition deals rows, by cluster.

### Partitioning

`partitionby` matters here for a reason of its own. The clusters group the
rows by what they hold, not by their partition, and a cluster may mix
cohorts. With `partitionby`, each cohort's rows in a cluster are ranked and
dealt apart from the others', so each part takes its share of every cohort,
and a coreset takes a rare cohort's central rows even where a common one
dominates the cluster.

A part's target accumulates over the cells of a partition, cluster by cluster,
so what one cell rounds off the next makes up. A partition's total is within
half a unit of its share: half a row, or half a stratum with `stratifyby`. A
stratum whose rows lie in two partitions is refused.

### Coverage

How tightly does a part "fill out" the table? The `coverage` topic, a
`DATADICT` (`coverage.npz`), answers with distances in the space the rows
were clustered in, the flattened `coreby` vectors.

The measure is the **covering distance**: for a row *x* of the table, the
distance from *x* to the nearest row of the part, *d(x, C)*. If every row
of the table is close to some row of the part, the part fills it out, and
the largest *d(x, C)* is the radius of the worst hole. The opposite measure
is the **separation**: for a row of the part, the distance to the nearest
*other* row of the part. Small separations mean the part spends rows on
near-duplicates. A good coreset has small coverage and large separation.

All pairs would cost O(N²). Instead:

- **Sampled rows.** `coverage_sample` rows of the table, seeded and spread
  over all tabs, are the queries for coverage; the same number of each
  part's rows, for separation. The quantiles are estimates from that sample.
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
| `coverage` | (F+1, 4) | *d(x, part)* for the sampled rows; one row per part, the last for all parts together |
| `coverage_mean` | (F+1,) | their means |
| `baseline` | (F+1, 4) | the same for a seeded random subset of the table of the part's size |
| `baseline_mean` | (F+1,) | their means |
| `separation` | (F, 4) | each sampled part row's distance to its nearest other row of the part |
| `cluster_rows` | (F+1, K) | rows per cluster: the table's first, then each part's |
| `cluster_tv` | (F,) | total variation between a part's cluster shares and the table's: 0 is the same mix |
| `empty_clusters` | (F,) | clusters the table has rows in and the part has none of |

How to read them:

- **Against `scale`.** A coverage median near the table's own spacing means
  the part is nearly as dense as the table; several times it means the part
  is coarse, as a 5% coreset must be.
- **Against `baseline`.** The same number of rows drawn at random. A coreset
  below its baseline fills the table out better than chance; above it, worse.
  Nearest-first sampling takes the rows at the centers, so with few clusters
  it covers the centers and leaves the edges. Its coverage then shows worse
  tails than random, and more clusters, approaching the core's size, fix
  that.
- **`cluster_tv` and `empty_clusters`** say whether every region is
  represented in proportion, exactly, over all rows, not a sample.
- Fractions summing to 1 put every row in some part, so the all-parts row of
  `coverage` is 0.

A partition built from a `layout` takes the layout's coverage: it describes
the rows the layout took, measured where they were clustered.

### Reusing a layout

The layout records, for every row of every part, the tab and row of the
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
`check_probe_inputs` checks the core tabs themselves. With `stratifyby`, a
stratum's rows are in one part, so fit and eval share no patient. A core tab
holds rows of many slides, so the probe takes a slide's rows by an
`aggregateby` column, with `aggregation='mean'`. Carry that column into the
core tabs, as a `carry` role.

## Where the code is

All in `dbx/dataparts.py`, except the probe.

| What | Where |
|---|---|
| Reading a partition column, following dict keys | `_column_values_` |
| How two values are compared | `_value_key_` (sorted JSON) |
| A piece's tag | `_piece_tag_` |
| The per-tab scan worker | `TabPartitionScanCallable` |
| SCAN, ITEMS, DEAL, the summary | `DatatablePartition._scan_`, `_items_`, `_deal_`, `_summary_` |
| COALESCE | `DatatablePartition._rejoin_`, `_pack_`, `_entry_rows_` |
| An entry to its tab, piece or packed tab | `DatatablePartition.tab` |
| A part as a table | `DatatablePartition.part`, `DatatablePart` |
| A part accepting pieces and packed tabs as blocks | `DatatablePart.tab`, `valid_tab`, `_form_block_` |
| A part's rows, as the table's | `DatatablePart.source_index` |
| The piece | `DatatabPiece`: `__build__`, `source_rows`, `source_index` |
| The packed tab | `DatatablePackTab`: `__build__`, `segments`, `_segment_rows_` |
| Selecting a tab's rows by value; copying a tab's slices | `_selected_rows_`, `_written_columns_`, `_slice_topics_` |
| The back-pointing slice | `SOURCE_SLICE`, `SOURCE`, `_source_index_` |
| Core: scan, cluster, deal, layout | `DatatableCorePartition`: `_cluster_and_deal_`, `_cluster_master_`, `_cluster_distributed_`, `_deal_rows_`, `_reuse_layout_` |
| Core: the workers | `TabCoreScanCallable`, `TabCoreStepCallable`, `TabRowCountCallable`, `TabCoverageCallable`, `TabVectorsCallable` |
| Core: coverage | `DatatableCorePartition._coverage_` |
| Core: a part and its tabs | `DatatableCorePart`, `DatatableCoreTab` |
| The probe's check before reading | `check_probe_inputs` in `dbx/probes.py` |
| Rows aggregated by a column; the checks by its value and by row | `aggregateby_vector`, `FeatureAffineLogisticProbe._aggregate_means_`, `_check_disjoint_` in `dbx/probes.py` |

The tests are `tests/test_partition_methods.py` (dealing, pieces, probes on
parts), `tests/test_partition_coalesce.py` (`coalesce`, packed tabs, and
probes aggregating rows by a column), `tests/test_partition_fold.py` (parts as
views) and `tests/test_core_partition.py` (core partitions). The figures are
drawn from real partitions by
[make_figures.py](figures/datapartition/make_figures.py).
