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

Its topics are the source tab's slices, declared as the source declares
them, and `source_rows`. A topic of the source that is not a slice belongs to
the whole tab, not to a subset of its rows, and is left out.

Its build re-reads the source's partition columns and finds the rows that
hold its values. It writes those rows, in the same order, to every slice, one
slice at a time, so row *i* of each slice is still row *i* of the others.
Then it records the row indices in **`source_rows`** (`piece.source_rows()`,
an int64 array, in the order written). It raises if the source's slices have
different lengths, or if no row holds the values any more, as when the
source was rebuilt after it was partitioned.

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

1. Partition the sample table, the one whose tabs hold the rows. Its pieces
   record `source_rows`.
2. A downstream part needs no SCAN and no DEAL: its entries are the upstream
   part's, mapped one to one. A whole tab maps to the downstream tab at the
   same index. A piece maps to a downstream piece, defined by its downstream
   tab and the upstream piece, that takes the upstream piece's `source_rows`
   of the downstream tab.
3. Building the downstream part builds those pieces in parallel, each
   selecting by index. The upstream and downstream parts agree by
   construction: one row selection, made once, upstream.

This still rests on row *i* of a feature tab being row *i* of its upstream
tab, which every `Featuretab` already assumes. Indices give that assumption
a place to be checked: where the feature tab carries a
`shared_upstream_column`, a downstream piece can compare it against the
upstream piece's and refuse a mismatch. The downstream piece class and the
mirrored part are not written yet.

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
| The probe's check before reading | `check_probe_inputs` in `dbx/probes.py` |

The tests are `tests/test_partition_methods.py` (dealing, pieces, probes on
parts) and `tests/test_partition_fold.py` (parts as views).
