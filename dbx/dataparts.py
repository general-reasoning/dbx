"""dataparts — partitions of a `Datatable` into parts: `DatatablePartition`, `DatatableCorePartition`, their maps, and the tables pulled back along them.

A partition is an index: it reads the table's partition columns, deals its rows
to parts, and records which rows went where. A part is the table that reads that
record: a view of the partitioned table's own tablets, pieces of them, and packed
tablets -- or, for a core partition, core tablets of rows repacked by cluster. See
DATAPARTITION.md.

TODO(untangle): three independent choices are bundled into class pairs here --
which rows go to which fold (a deal, a clustering, a template's), what a part is
(a view of the table's tablets, or copies of the rows), and what happens to
upstream data (kept through a view, so a tablet reading upstream cannot be cut;
flattened by a copy). The design: a datamap -- `PartitionDatamapTable` of `PartitionDatamapTablet`s,
each output tablet's ``[(source tablet, rows | ALL)]`` -- made by the deal, and
tables pulled back along it, layer by layer -- `MappedDatatable`,
`MappedFeaturetable` -- a whole tablet pulling back to itself. The CPTAC folds
keep their hashes. See DATAPARTITION.md, "TODO: three choices, untangled --
datamaps, and tables mapped along them".
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass

import numpy as np
from streaming import StreamingDataset

from .datablocks import DATADICT, DATADIR, DATAFILE, SAME, Datablock, Datastack, forward_property
from .databits import callable_executor
from .datastreams import SharedMemoryManager, release_shared_memory_when_collected
from .datatables import DATASLICE, SLICETOPIC, Datacollator, Layered, Datatablet, Datatable

#: How `DatatablePartition` deals tablets to folds -- see its docstring.
PARTITION_METHODS = ('random', 'largest_first')


def _column_values_(tablet, spec) -> list:
    """Every row's value of *spec* -- ``(slice, column, key, ...)`` -- in *tablet*; None where a row has none."""
    s_name, column, *keys = spec
    # Row by row: concatenated, a dict column comes back by key, and a row that is None has no keys.
    values = tablet.data((s_name, column), concat=False)[s_name][column]
    out = []
    for value in values:
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        # A numpy scalar as the Python value it is: what a deal records in JSON, and compares by.
        out.append(value.item() if isinstance(value, np.generic) else value)
    return out


def _value_key_(value) -> str:
    """A value as a sortable, hashable string: what two rows' values are compared by."""
    return json.dumps(value, sort_keys=True, default=str)


PARTITION_ROLES = ('stratifyby', 'partitionby')

#: The slice a copying tablet -- a `MappedDatatablet`, a piece, a packed or a core tablet -- points back
#: with: row i of it names the tablet and the row of the partitioned table that row i of
#: every other slice was copied from.
SOURCE_SLICE = 'source'
SOURCE = DATASLICE(tab='int', row='int')


def _selected_rows_(source, columns: dict, values: dict, what: str) -> list[int]:
    """The indices of *source*'s rows holding every role's value -- all its rows, for no roles.

    Raising when they cannot be what was dealt: rows are selected by index, so
    every slice must have as many rows, and a piece's value must still be held.
    """
    lengths = {s: int(source.n_rows(s)) for s in source.slices()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{what}: {source.tag}'s slices have different lengths, {lengths}; rows are "
                         f"selected by index, so row i must be row i of every slice. Rebuild the tablet with its "
                         f"slices written in lockstep.")
    n = next(iter(lengths.values()), 0)
    hits = [True] * n
    for role, spec in columns.items():
        found = _column_values_(source, tuple(spec))
        if len(found) != n:
            raise ValueError(f"{what}: {source.tag}'s {role} column {tuple(spec)} has {len(found)} values "
                             f"for its {n} rows")
        want = _value_key_(values[role])
        hits = [h and _value_key_(v) == want for h, v in zip(hits, found)]
    rows = [i for i, h in enumerate(hits) if h]
    if columns and not rows:
        raise ValueError(
            f"{what}: no row of {source.tag} holds {values}: the piece cannot be recovered from its tablet. Was "
            f"the tablet rebuilt since it was partitioned? Build the partition again over the table as it is now.")
    return rows


def _written_columns_(source, slice, what: str) -> dict:
    """The columns *slice* of *source* was written with, read off its index -- for a slice that declares none."""
    with source.fs.open(source.slice_index_path(slice), 'r') as f:
        shards = json.load(f)['shards']
    if not shards:
        raise ValueError(f"{what}: {source.tag}'s slice {slice!r} has no shards, and declares no columns: "
                         f"there is nothing to write it with")
    return dict(zip(shards[0]['column_names'], shards[0]['column_encodings']))


def _slice_topics_(source, what: str) -> dict:
    """*source*'s slices, declared as it declares them, nested by their ``/`` -- and ``source``: a copied tablet's TOPICS."""
    topics = {}
    for name in source.slices():
        *head, leaf = name.split('/')
        node = topics
        for h in head:
            node = node.setdefault(h, {})
        node[leaf] = source._topicnode_(*name.split('/'))
    if SOURCE_SLICE in topics:
        raise ValueError(f"{what}: {source.tag} has a slice named {SOURCE_SLICE!r}, the slice a copied tablet "
                         f"points back to its rows with; rename the slice")
    return {**topics, SOURCE_SLICE: SOURCE}


def _source_index_(tablet) -> np.ndarray:
    """``(tablet, row)`` of the partitioned table for each of *tablet*'s rows: its ``source`` slice, as an ``(N, 2)`` array."""
    src = tablet.data(SOURCE_SLICE)[SOURCE_SLICE]
    return np.stack([np.asarray(src['tab'], dtype=np.int64), np.asarray(src['row'], dtype=np.int64)], axis=1)


def _piece_tag_(tag, values: dict) -> str:
    """A piece's tag: its source tablet's, and the value of each role, ``'#'``-joined -- what a probe tells pieces apart by."""
    return '#'.join([str(tag)] + [v if isinstance(v, str) else _value_key_(v) for v in values.values()])


class TabletPartitionScanCallable:
    """Worker callable: one tablet's row count and the values each partition column holds in it.

    A tablet holding one value per role reports that value. A tablet holding several -- mixed --
    also reports its PIECES: each distinct combination of the role values, and how many rows
    hold it. Counts and not rows: a tablet may hold millions of rows, and a piece re-selects its
    own from the value when it is built, so nothing row by row comes back to the partition.
    """

    def __init__(self, partition, idx: int):
        self.partition = partition
        self.idx = idx

    def __call__(self):
        part = self.partition
        tablet = part.var.datatable.tablet(self.idx)
        out = {'idx': self.idx, 'tag': tablet.tag, 'rows': None, 'values': {}}
        if part.var.balance == 'rows' or part.var.method == 'largest_first' or getattr(part.var, 'coalesce', None) == 'pack':
            out['rows'] = int(tablet.n_rows(part._partition_slice_name_))
        columns = {}
        for name in PARTITION_ROLES:
            spec = part.column_spec(name)
            if spec is None:
                continue
            columns[name] = _column_values_(tablet, tuple(spec))
            distinct = {}
            for value in columns[name]:
                distinct.setdefault(_value_key_(value), value)
            out['values'][name] = list(distinct.values())
        if any(len(vals) > 1 for vals in out['values'].values()):
            out['column_rows'] = {name: len(vals) for name, vals in columns.items()}
            out['slice_rows'] = {s: int(tablet.n_rows(s)) for s in tablet.slices()}
            pieces = {}
            for row in zip(*columns.values()):
                values = dict(zip(columns, row))
                piece = pieces.setdefault(_value_key_(values), {'values': values, 'rows': 0})
                piece['rows'] += 1
            out['pieces'] = [pieces[k] for k in sorted(pieces)]
        return out


#: Where `DatatablePartition` and its blocks were stored before they moved here, on 2026-10-09.
_DATATABLES_ = 'dbx.datatables'

COALESCE_MODES = (None, 'whole', 'pack')


class DatatablePartition(Datablock):
    """Partitions a `Datatable`'s rows into parts according to target *fractions*.

    Tablets are dealt to parts; part *k*, `part(k)`, a `DatatablePart`, is a table
    of them -- the table's own tablets where they are, and the tablets this partition
    makes of them (see *coalesce*). The choices that decide the deal each answer
    one question:

    The columns it reads are a *collator*'s: a `Datacollator` whose
    ``columns`` hold one column spec, ``(slice, column[, key, ...])``, for
    each of the roles

    *stratifyby* -- which rows must land in the SAME part? Rows sharing its
    value, a stratum (a patient's, say), are dealt as one unit, across all the
    tablets they lie in. Default: each tablet alone.

    *partitionby* -- within which categories must the fractions hold? The
    fractions are met within each of its values, a partition (each cancer
    type, say), not only overall. Default: overall only.

    A column upstream of *datatable* -- a feature table's samples' -- is read
    only with the collator's ``recursive=True``. No collator: neither role.

    *balance* -- fractions of what? ``'rows'`` (the default): of rows, read
    from *partition_slice*. ``'tablets'``: of tablets.

    *seed* -- the order units are dealt in. Within each partition, the units
    are shuffled with it, then each goes to the part furthest below its target.

    A tablet whose *stratifyby* and *partitionby* columns hold one value throughout is
    dealt whole. A tablet holding several -- several patients in one bag, tiles
    annotated one by one -- is cut: into PIECES, one per combination of its
    values, each a `MappedDatatablet` of the rows of that tablet holding those values.
    A piece is dealt as a tablet is: with every tablet and piece of its stratum,
    within its partition.

    *coalesce* -- what a part's tablets are, once the deal is done. It never moves
    a row to another part.

    - None (the default): the dealt entries as they are -- whole tablets, and
      pieces of the cut ones. What a partition was before *coalesce*.
    - ``'whole'``: a cut tablet whose pieces all went to one part is that part's
      whole again -- the table's own tablet, read where it is, rather than pieces
      copied out of it. A tablet of which a piece was skipped stays cut: whole, it
      would bring the skipped rows back.
    - ``'pack'``: as ``'whole'``, and then a part's entries of fewer than
      *min_tablet_rows* rows -- tablets and pieces both -- are pooled, in source
      order, into packed tablets of *max_tablet_rows* rows (twice *min_tablet_rows* by
      default), each a `MappedDatatablet`. An entry may run across two packed
      tablets; the last one of a part holds what is left. Larger entries stay as
      they are, so nothing is copied that is not small.

    A tablet of a part may hold rows of several strata and partitions -- a rejoined
    tablet, a packed one -- so a reader that needs the rows of one slide, say,
    takes them by a column, not by tablet: a probe's ``aggregateby`` -- see
    `FeatureAffineLogisticProbe`.

    A tablet or a piece holding no value is skipped: in no part, said loudly, and
    recorded in ``summary`` -- a piece with the rows it leaves out. A stratum
    must lie in one partition. A table that reads slices upstream
    (`Layered`, a `Featuretable`) is cut and packed as any other: its
    pieces and packed tablets are `MappedFeaturetablet`s, the same rows of its
    own slices and, as their upstream, of its upstream's -- see `MappedDatatable`.

    *method* ``'largest_first'`` is the partition from before these choices:
    tablets in descending order of rows, each to the part with the largest deficit
    -- no strata, no partitions, no seed. Its builds are reached by a
    specialization, and it takes none of the choices above.

    Topics: ``tablets`` -- one list of entries per part, which the parts read: a
    tablet index; a piece's record, ``{"tab": i, "stratifyby": value, "partitionby":
    value}`` with the roles the collator sets, its values as read; and, with
    ``coalesce='pack'``, a packed tablet's record, ``{"fold": k, "pack": j,
    "segments": [[entry, offset, length], ...]}`` -- see `tablets_indices`;
    ``summary`` -- per part its tags and counts (tablets, rows, per partition), the
    tablets and pieces skipped and why, and what was rejoined and packed.

    The parts build the pieces and the packed tablets: ``partition.fold(k).build()``
    builds part k's, in parallel, as a table builds its tablets, and the whole tablets
    among them are already built. A probe reading a part whose tablets are not
    built says so before it starts. The partition itself stays an index: it
    writes no rows.
    """

    # 2: the collator's roles were renamed on 2026-10-10 -- stratifyby is partitionby, and what was
    # partitionby is partitionby. A collator naming partitionby hashes as it did, and means the
    # opposite, so the version keeps every partition built before from being taken for one now.
    VERSION = 2

    TOPICS = {
        'tablets': DATAFILE('tabs.json', 'one list of tablet indices per fold'),
        'summary': DATAFILE('summary.json', 'per fold its tags and counts, and the tablets skipped'),
    }
    # None: the roles were renamed (VERSION 2), and the topic `tabs` became `tablets` on 2026-10-10 -- a
    # specialization takes a topic only under the name it had, so even the partitions with no collator,
    # largest_first's, are rebuilt.
    SPECIALIZATIONS = []

    @dataclass
    class _DealVAR_(Datablock.VAR):
        """What decides the deal: a `DatatableCorePartition`'s too, which takes no *coalesce*."""
        datatable: Datatable
        fractions: list[float]
        partition_slice: int | str
        collator: Datacollator | None = None
        method: str = 'random'
        seed: int = 0
        balance: str = 'rows'

    @dataclass
    class VAR(_DealVAR_):
        coalesce: str | None = None
        min_tablet_rows: int | None = None
        max_tablet_rows: int | None = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        var = self.var
        if var.method not in PARTITION_METHODS:
            raise ValueError(f"{type(self).__name__}: unknown method {var.method!r}; expected one of {PARTITION_METHODS}")
        if var.balance not in ('rows', 'tablets'):
            raise ValueError(f"{type(self).__name__}: balance must be 'rows' or 'tablets', got {var.balance!r}")
        if var.collator is not None:
            for name in ('stratifyby', 'partitionby'):
                spec = var.collator.var.columns.get(name)
                if spec is not None and not (isinstance(spec, tuple) and len(spec) >= 2
                                             and all(isinstance(p, str) for p in spec)):
                    raise ValueError(f"{type(self).__name__}: the collator's {name} must be one column spec, "
                                     f"a tuple (slice, column[, key, ...]) of strings, got {spec!r}")
            other = [r for r in var.collator.roles if r not in ('stratifyby', 'partitionby')]
            if other:
                raise ValueError(f"{type(self).__name__}: a partition reads the roles stratifyby and partitionby; "
                                 f"its collator declares {other} too")
        if var.method == 'largest_first' and (var.collator is not None or var.balance != 'rows' or var.seed != 0):
            raise ValueError(f"{type(self).__name__}: method='largest_first' takes no collator (no stratifyby, "
                             f"partitionby), balance or seed -- it is the partition from before them; "
                             f"use method='random'")
        what = type(self).__name__
        if var.coalesce not in COALESCE_MODES:
            raise ValueError(f"{what}: coalesce must be one of {COALESCE_MODES}, got {var.coalesce!r}")
        if var.method == 'largest_first' and var.coalesce is not None:
            raise ValueError(f"{what}: method='largest_first' cuts no tablet, so there is nothing to coalesce")
        if var.coalesce == 'pack':
            if not (isinstance(var.min_tablet_rows, int) and var.min_tablet_rows >= 1):
                raise ValueError(f"{what}: coalesce='pack' pools the entries of fewer than min_tablet_rows rows, "
                                 f"which must be a positive int, got {var.min_tablet_rows!r}")
            if var.max_tablet_rows is not None and not (isinstance(var.max_tablet_rows, int)
                                                     and var.max_tablet_rows >= var.min_tablet_rows):
                raise ValueError(f"{what}: max_tablet_rows, a packed tablet's rows, must be an int of at least "
                                 f"min_tab_rows={var.min_tablet_rows}, got {var.max_tablet_rows!r}")
        elif var.min_tablet_rows is not None or var.max_tablet_rows is not None:
            raise ValueError(f"{what}: min_tablet_rows and max_tablet_rows size packed tablets, and are for "
                             f"coalesce='pack' only, not coalesce={var.coalesce!r}")

    def __build__(self):
        table = self.var.datatable
        if table is None:
            raise ValueError(f"{self.__class__.__name__}: VAR.datatable is required")
        self.log.info(f"DEAL {type(self).__name__}/{self.hash[:8]}: {type(table).__name__} into "
                      f"{len(self.var.fractions or [])} folds -- SCAN, ITEMS, DEAL, COALESCE; the parts copy, later")
        if self.var.collator is not None:
            self.var.collator.slices(table)       # raises for a column the table does not hold
        if not self.var.fractions:
            raise ValueError(f"{self.__class__.__name__}: VAR.fractions is required")
        scans = self._scan_()
        if self.valid_topic('tablets'):
            # Adopted -- a build from before the summary existed: its folds stand.
            folds_tablets, skipped = json.loads(self.fs.cat(self.path('tablets'))), []
        else:
            folds_tablets, skipped = ((self._largest_first_(scans), []) if self.var.method == 'largest_first'
                                   else self._deal_(scans))
            if self.var.coalesce is not None:
                folds_tablets = self._rejoin_(scans, folds_tablets)
            if self.var.coalesce == 'pack':
                folds_tablets = self._pack_(scans, folds_tablets)
            with self.fs.open(self.path('tablets', ensure_dirpath=True), 'w') as f:
                json.dump(folds_tablets, f)
        with self.fs.open(self.path('summary', ensure_dirpath=True), 'w') as f:
            json.dump(self._summary_(scans, folds_tablets, skipped), f, indent=1, default=str)

    def __read__(self, *topicpath):
        topicpath = self._normtopic_(topicpath)
        if topicpath in (('tablets',), ('summary',)):
            return json.loads(self.fs.cat(self.path(topicpath[0])))
        return super().__read__(*topicpath)

    # 2. Declared API ------------------------------------------------------

    def n_folds(self) -> int:
        return len(self.var.fractions)

    def tablets_indices(self, fold: int | str) -> list:
        """Part *fold*'s entries: a table's tablet index; for a piece of a mixed tablet, its record; for a packed tablet, its.

        A piece's record is ``{"tab": i, <role>: value, ...}``: the source tablet,
        and the value of each role its rows hold, as read. A packed tablet's is
        ``{"fold": k, "pack": j, "segments": [[entry, offset, length], ...]}``:
        rows *offset* to *offset + length* of each entry's, in order. Ints only,
        as ever, when no tablet was cut. `tablet` turns any of them into the block it
        names.
        """
        data = json.loads(self.fs.cat(self.path('tablets')))
        return data[int(fold)]

    def tablets(self, fold: int | str) -> list[Datatablet]:
        """Part *fold*'s tablets: its table's, where it takes them whole, and the copies of the rest."""
        part = self.fold(fold)
        return [part.tablet(i) for i in range(part.n_tablets)]

    def map(self, fold: int | str) -> PartitionDatamapTable:
        """Fold *fold*'s map: which rows of which of its table's tablets each of the fold's tablets takes.

        Pull any table laid out as this one's back along it: ``map(k).pullback(table, collator)``.
        """
        # The deal's workers: the map builds its tablets, in parallel, as a part does.
        return PartitionDatamapTable(spec=dict(partition=self, fold=fold), **self._part_kwargs_)

    def lot(self, fold: int | str, idx: int) -> list:
        """Tablet *idx* of fold *fold*'s lot: ``[(tablet, rows | None), ...]``, the rows of the table's tablets it takes.

        Resolved from the entry the deal wrote: a whole tablet, ``[(t, None)]``; a
        piece, ``[(t, rows)]``, its rows those of tablet *t* holding its values; a
        packed tablet, its segments' rows, in order.
        """
        entry = self._dealt_tablets_[int(fold)][int(idx)]
        if not isinstance(entry, dict):
            return [(int(entry), None)]
        if 'pack' in entry:
            return [(t, self._lot_rows_(e, t)[int(offset):int(offset) + int(length)])
                    for e, offset, length in entry['segments']
                    for t in (int(e['tab']) if isinstance(e, dict) else int(e),)]
        t = int(entry['tab'])
        return [(t, self._lot_rows_(entry, t))]

    def lot_tablets(self, fold: int | str, idx: int) -> list[int]:
        """The tablets tablet *idx* of fold *fold*'s lot takes rows of, in order: its entry's, no row resolved."""
        entry = self._dealt_tablets_[int(fold)][int(idx)]
        if not isinstance(entry, dict):
            return [int(entry)]
        if 'pack' in entry:
            return [int(e['tab']) if isinstance(e, dict) else int(e) for e, _, _ in entry['segments']]
        return [int(entry['tab'])]

    def whole(self, fold: int | str, idx: int) -> int | None:
        """The tablet tablet *idx* of fold *fold* is, whole -- or None, when it takes some rows only. Reads no rows."""
        entry = self._dealt_tablets_[int(fold)][int(idx)]
        return None if isinstance(entry, dict) else int(entry)

    def tag_of(self, fold: int | str, idx: int) -> str:
        """What a copy of tablet *idx* of fold *fold* is tagged: a piece by its tablet and values, a packed tablet by its place."""
        entry = self._dealt_tablets_[int(fold)][int(idx)]
        if not isinstance(entry, dict):
            return self.var.datatable.tablet(int(entry)).tag
        if 'pack' in entry:
            return f"pack{self.hash[:8]}_f{int(entry['fold'])}_{int(entry['pack']):06d}"
        return _piece_tag_(self.var.datatable.tablet(int(entry['tab'])).tag, self._entry_values_(entry))

    def dealt(self) -> bool:
        """Whether it has dealt: its ``tablets`` topic is there, built or redirected."""
        return self.valid_topic('tablets')

    def _is_copies_only_(self, fold: int | str) -> bool:
        """Whether fold *fold* is copies only, no tablet taken whole -- False until it has dealt, reading nothing.

        What a mapped table asks to know whether every tablet holds ``source`` -- see `MappedDatatable._is_copies_only_`.
        """
        if not self.dealt():
            return False
        n = len(self._dealt_tablets_[int(fold)])
        return n > 0 and all(self.whole(fold, i) is None for i in range(n))

    def fold(self, fold: int | str) -> DatatablePart:
        """Fold *fold* of its own table: the table pulled back along `map(fold)` -- a `DatatablePart`.

        Its VAR is ``{partition, fold}``, its map derived: the identity of every fold
        built, and of everything built on one. Any other table laid out alike is
        pulled back by `map`.
        """
        cls = FeaturetablePart if isinstance(self.var.datatable, Layered) else DatatablePart
        return cls(spec=dict(partition=self, fold=fold), **self._part_kwargs_)

    # 3. Accessors ---------------------------------------------------------

    @property
    def datatable(self) -> Datatable:
        return self.var.datatable

    def column_spec(self, role: str) -> 'tuple | None':
        """The column spec playing *role* -- ``'stratifyby'`` or ``'partitionby'`` -- or None."""
        collator = self.var.collator
        return None if collator is None else collator.var.columns.get(role)

    @property
    def _partition_slice_name_(self) -> str:
        p_slice = self.var.partition_slice
        return self.var.datatable.slices()[p_slice] if isinstance(p_slice, int) else p_slice

    @functools.cached_property
    def _dealt_tablets_(self) -> list:
        """Every fold's entries, as dealt: its ``tablets`` topic, read once for the lots of every tablet."""
        return json.loads(self.fs.cat(self.path('tablets')))

    @staticmethod
    def _entry_values_(entry) -> dict:
        return {role: entry[role] for role in PARTITION_ROLES if role in entry}

    def _lot_rows_(self, entry, t: int) -> np.ndarray:
        """The rows of tablet *t* an entry takes: a whole tablet's every row, a piece's those holding its values."""
        values = self._entry_values_(entry) if isinstance(entry, dict) else {}
        columns = {role: tuple(self.column_spec(role)) for role in values}
        rows = _selected_rows_(self.var.datatable.tablet(t), columns, values, type(self).__name__)
        return np.asarray(rows, dtype=np.int64)

    @property
    def _part_kwargs_(self) -> dict:
        """What a part takes from its partition: where it is, and the workers it builds its tablets with.

        Where: as a table gives its tablets its url, a fold of a partition belongs
        where the partition does, not wherever DBX_ROOT happens to point in the
        process that asks for it. The workers: each part is built by its own
        build(), one after another, and builds its tablets -- pieces, packed
        tablets, core tablets -- in parallel, as the partition scanned in parallel.
        """
        return dict(datalake=self._datalake_, storage_options=self.storage_options,
                    parallelization=getattr(self, 'parallelization', None),
                    n_workers=getattr(self, 'n_workers', 1) or 1)

    # 4. Helpers -----------------------------------------------------------

    def _scan_(self) -> list[dict]:
        """Each tablet's rows and partition-column values, read in parallel; a mixed tablet's pieces too.

        A mixed tablet is checked here, before anything is dealt: its pieces are
        selected by row, so every slice of it must have as many rows as its
        partition columns have values -- or row i of one slice is not row i of
        another, and a piece would pair one row's columns with another's.
        """
        table = self.var.datatable
        executor = callable_executor(
            getattr(self, 'parallelization', None) or 'inline',
            n_workers=getattr(self, 'n_workers', 1) or 1,
            tag=f"SCANNING {table.n_tablets} tablets [{type(self).__name__}]")
        scans = executor.exec_callables([TabletPartitionScanCallable(self, i) for i in range(table.n_tablets)])
        mixed = [s for s in scans if 'pieces' in s]
        if not mixed:
            return scans
        shown = "\n".join(f"  {s['tag']}: " + ", ".join(f"{name} holds {len(vals)} values, e.g. {vals[:3]!r}"
                                                         for name, vals in s['values'].items() if len(vals) > 1)
                          for s in mixed[:5])
        uneven = [s for s in mixed if len(set(s['column_rows'].values()) | set(s['slice_rows'].values())) > 1]
        if uneven:
            s = uneven[0]
            raise ValueError(
                f"{type(self).__name__}: {len(uneven)} mixed tablet(s) have slices or partition columns of "
                f"different lengths, e.g. {s['tag']}: slices {s['slice_rows']}, partition columns "
                f"{s['column_rows']}. A mixed tablet is split by row, so row i must be row i of every slice; "
                f"rebuild the tablet with its slices written in lockstep.")
        n_pieces = sum(len(s['pieces']) for s in mixed)
        self.log.warning(
            f"{type(self).__name__}: splitting {len(mixed)} of {len(scans)} tablets, which hold more than one "
            f"value of a partition column, into {n_pieces} pieces, one per value -- dealt as tablets are, and "
            f"built by the parts (part.build()):\n{shown}" + ("\n  ..." if len(mixed) > 5 else ""))
        return scans

    def _items_(self, scans) -> list[dict]:
        """What is dealt, in scan order: each constant tablet, and each piece of a mixed one.

        An item's *entry* is what ``tablets`` records for it -- a tablet index, or a
        piece's record -- and its *values* the value of each role, None where
        it holds none.
        """
        roles = [r for r in PARTITION_ROLES if self.column_spec(r) is not None]
        items = []
        for s in scans:
            if 'pieces' not in s:
                values = {r: (s['values'][r][0] if s['values'][r] else None) for r in roles}
                items.append({'idx': s['idx'], 'entry': s['idx'], 'tag': s['tag'], 'rows': s['rows'],
                              'values': values})
                continue
            for p in s['pieces']:
                items.append({'idx': s['idx'], 'entry': {'tab': s['idx'], **p['values']},
                              'tag': _piece_tag_(s['tag'], p['values']), 'rows': p['rows'], 'values': p['values']})
        return items

    @staticmethod
    def _entry_key_(entry) -> 'int | str':
        """An entry of ``tablets`` as a dict key: a tablet index is itself, a piece its record's `_value_key_`."""
        return _value_key_(entry) if isinstance(entry, dict) else entry

    @staticmethod
    def _entry_order_(entry) -> tuple:
        """The order a fold lists its entries in: by source tablet, a tablet's pieces by value -- ints alone as sorted()."""
        return (entry['tab'], _value_key_(entry)) if isinstance(entry, dict) else (entry, '')

    def _largest_first_(self, scans) -> list[list[int]]:
        """The partition from before: tablets by descending rows, each to the fold with the largest deficit."""
        fractions = self.var.fractions
        tablet_rows = [s['rows'] for s in scans]
        total = sum(tablet_rows)
        norm = [f / sum(fractions) for f in fractions]
        target = [total * f for f in norm]
        fold_rows = [0.0] * len(fractions)
        folds_tablets = [[] for _ in fractions]
        for t_idx in sorted(range(len(tablet_rows)), key=lambda i: tablet_rows[i], reverse=True):
            best = max(range(len(fractions)), key=lambda k: target[k] - fold_rows[k])
            folds_tablets[best].append(t_idx)
            fold_rows[best] += tablet_rows[t_idx]
        return [sorted(f) for f in folds_tablets]

    def _deal_(self, scans) -> tuple[list[list], list[dict]]:
        """Units -- strata of tablets and pieces -- dealt within each partition, in seeded order, to the fold furthest below target.

        A piece is dealt exactly as a tablet is, weighing its rows or 1: a stratum's
        unit collects every tablet and piece holding its value, whichever tablets the
        pieces came from. Without a stratifyby, each tablet is its own unit, and so
        is each piece.
        """
        fractions = [f / sum(self.var.fractions) for f in self.var.fractions]
        items = self._items_(scans)
        skipped, units = [], {}
        for item in items:
            missing = [name for name, value in item['values'].items() if value is None]
            if missing:
                why = f"no {' or '.join(missing)} value"
                skipped.append({'idx': item['idx'], 'tag': item['tag'], 'why': why} if not isinstance(item['entry'], dict)
                               else {'idx': item['idx'], 'tag': item['tag'], 'piece': item['entry'],
                                     'rows': item['rows'], 'why': f"{why} in {item['rows']} of its rows"})
                continue
            values = item['values']
            stratum = (_value_key_(values['stratifyby']) if 'stratifyby' in values else
                       f"tab:{item['idx']}" if not isinstance(item['entry'], dict) else
                       f"tab:{item['idx']}#{_value_key_(values)}")
            partition = _value_key_(values['partitionby']) if 'partitionby' in values else ''
            unit = units.setdefault(stratum, {'tablets': [], 'partitions': set(), 'weight': 0})
            unit['tablets'].append(item['entry'])
            unit['partitions'].add(partition)
            unit['weight'] += item['rows'] if self.var.balance == 'rows' else 1
        if skipped:
            n_rows = sum(s['rows'] for s in skipped if 'piece' in s)
            self.log.warning(
                f"{type(self).__name__}: skipping {len(skipped)} of {len(items)} "
                f"{'tablets' if len(items) == len(scans) else 'tablets and pieces'}, which hold no "
                f"{'/'.join(n for n in PARTITION_ROLES if self.column_spec(n) is not None)} "
                f"value -- they are in no fold: {[s['tag'] for s in skipped[:10]]}"
                + (" ..." if len(skipped) > 10 else "")
                + (f"; the pieces among them leave out {n_rows} rows" if n_rows else ""))
        straddling = {st: u['partitions'] for st, u in units.items() if len(u['partitions']) > 1}
        if straddling:
            st, partitions = next(iter(straddling.items()))
            raise ValueError(f"{type(self).__name__}: {len(straddling)} stratum(s) span more than one partition, "
                             f"e.g. stratum {st} in partitions {sorted(partitions)}: a stratum is dealt whole, so "
                             f"it must lie in one partition")
        by_partition = {}
        for st in sorted(units):
            by_partition.setdefault(next(iter(units[st]['partitions'])), []).append(st)
        rng = np.random.default_rng(self.var.seed)
        folds_tablets = [[] for _ in fractions]
        for partition in sorted(by_partition):
            strata = by_partition[partition]
            total = sum(units[st]['weight'] for st in strata)
            target = [total * f for f in fractions]
            have = [0.0] * len(fractions)
            for i in rng.permutation(len(strata)):
                unit = units[strata[i]]
                k = max(range(len(fractions)),
                        key=lambda k: (target[k] - have[k]) / target[k] if target[k] > 0 else -np.inf)
                folds_tablets[k].extend(unit['tablets'])
                have[k] += unit['weight']
        return [sorted(f, key=self._entry_order_) for f in folds_tablets], skipped

    def _rejoin_(self, scans, folds_tablets) -> list[list]:
        """Each cut tablet whose pieces all went to one part made whole there again.

        Whole, a tablet is read where it is, and nothing is copied out of it. A tablet
        of which a piece is in no part -- skipped, for holding no value -- stays
        cut: whole, it would bring those rows back.
        """
        cut = {s['idx']: s for s in scans if 'pieces' in s}
        where = {}
        for k, entries in enumerate(folds_tablets):
            for e in entries:
                if isinstance(e, dict):
                    where.setdefault(e['tab'], []).append(k)
        whole = {idx for idx, parts in where.items()
                 if len(set(parts)) == 1 and len(parts) == len(cut[idx]['pieces'])}
        return [sorted([e for e in entries if not (isinstance(e, dict) and e['tab'] in whole)]
                       + sorted({e['tab'] for e in entries if isinstance(e, dict) and e['tab'] in whole}),
                       key=self._entry_order_)
                for entries in folds_tablets]

    def _pack_(self, scans, folds_tablets) -> list[list]:
        """Each part's entries of fewer than *min_tablet_rows* rows pooled, in source order, into packed tablets' records.

        A packed tablet takes *max_tablet_rows* rows -- twice *min_tablet_rows* unless
        given -- as segments, ``[entry, offset, length]``, and an entry may run
        across two of them. The last packed tablet of a part holds what is left.
        Entries of at least *min_tablet_rows* rows stay as they are: a whole tablet is
        read where it is, a piece is a copy already.
        """
        rows = self._entry_rows_(scans)
        least = self.var.min_tablet_rows
        most = self.var.max_tablet_rows or 2 * least
        out = []
        for k, entries in enumerate(folds_tablets):
            kept, packs, segments, fill = [], [], [], 0
            for e in entries:
                n = rows[self._entry_key_(e)]
                if n >= least or n == 0:
                    kept.append(e)
                    continue
                offset = 0
                while offset < n:
                    take = min(n - offset, most - fill)
                    segments.append([e, offset, take])
                    fill, offset = fill + take, offset + take
                    if fill == most:
                        packs.append(segments)
                        segments, fill = [], 0
            if segments:
                packs.append(segments)
            out.append(kept + [{'fold': k, 'pack': j, 'segments': seg} for j, seg in enumerate(packs)])
        return out

    def _entry_rows_(self, scans) -> dict:
        """Every entry a deal can hold -- each tablet, whole, and each piece -- by `_entry_key_`: its rows."""
        rows = {}
        for s in scans:
            pieces = s.get('pieces', ())
            rows[s['idx']] = sum(p['rows'] for p in pieces) if pieces else s['rows']
            for p in pieces:
                rows[self._entry_key_({'tab': s['idx'], **p['values']})] = p['rows']
        return rows

    def _summary_(self, scans, folds_tablets, skipped) -> dict:
        """What each part holds -- tags, tablets, rows, per partition -- what was left out, and what was rejoined and packed.

        A piece counts as a tablet -- it is one of its part's -- with its own rows.
        ``pieces`` and ``n_pieces`` say how many, and appear only when a tablet was
        cut, so the summary of a partition of constant tablets is what it was. A
        packed tablet's entries count as the entries they are, and ``packed`` says
        how many packed tablets hold them; a rejoined tablet counts as the whole tablet.
        """
        items = self._items_(scans)
        by_entry = {self._entry_key_(item['entry']): [item] for item in items}
        for s in scans:
            if 'pieces' in s:      # a cut tablet made whole by coalesce: its pieces, each in its partition
                by_entry[s['idx']] = [by_entry[self._entry_key_({'tab': s['idx'], **p['values']})][0]
                                      for p in s['pieces']]
        tag_of = {s['idx']: s['tag'] for s in scans}
        split = any('pieces' in s for s in scans)
        packing = getattr(self.var, 'coalesce', None) == 'pack'

        folds = []
        for k, fold_entries in enumerate(folds_tablets):
            packs = [e for e in fold_entries if isinstance(e, dict) and 'pack' in e]
            entries = [e for e in fold_entries if not (isinstance(e, dict) and 'pack' in e)]
            pooled = list({self._entry_key_(seg[0]): seg[0] for p in packs for seg in p['segments']}.values())
            dealt = entries + pooled
            fold_items = [item for e in dealt for item in by_entry[self._entry_key_(e)]]
            partitions = {}
            for e in dealt:
                held = by_entry[self._entry_key_(e)]
                for key in {str(item['values'].get('partitionby')) for item in held}:
                    partitions.setdefault(key, {'tabs': 0, 'rows': 0})['tabs'] += 1
                for item in held:
                    partitions[str(item['values'].get('partitionby'))]['rows'] += item['rows'] or 0
            folds.append({
                'fold': k,
                'fraction': self.var.fractions[k],
                'tabs': len(dealt),
                **({'pieces': sum(isinstance(e, dict) for e in dealt)} if split else {}),
                'rows': sum(item['rows'] or 0 for item in fold_items) if self.var.balance == 'rows' or self.var.method == 'largest_first' else None,
                'partitions': dict(sorted(partitions.items())) if self.column_spec('partitionby') is not None else None,
                'tags': [tag_of[e] if isinstance(e, int) else by_entry[self._entry_key_(e)][0]['tag']
                         for e in dealt],
                **({'packed': {'tabs': len(packs), 'entries': len(pooled),
                               'rows': sum(seg[2] for p in packs for seg in p['segments'])}} if packing else {}),
            })
        coalesced = {}
        if getattr(self.var, 'coalesce', None) is not None:
            whole = {e for entries in folds_tablets for e in entries if not isinstance(e, dict)}
            whole |= {seg[0] for entries in folds_tablets for e in entries if isinstance(e, dict) and 'pack' in e
                      for seg in e['segments'] if not isinstance(seg[0], dict)}
            coalesced = {'coalesce': self.var.coalesce,
                         'rejoined': [s['tag'] for s in scans if 'pieces' in s and s['idx'] in whole]}
        return {'method': self.var.method, 'balance': self.var.balance, 'seed': self.var.seed,
                'stratifyby': self.column_spec('stratifyby'), 'partitionby': self.column_spec('partitionby'),
                'n_tablets': len(scans), **({'n_pieces': sum(len(s['pieces']) for s in scans if 'pieces' in s)} if split else {}),
                **coalesced, 'folds': folds, 'skipped': skipped}


class PartitionDatamapTablet(Datatablet):
    """The map of one tablet of a fold: its lot -- which rows of which tablets of the mapped table it takes.

    Its `lot` is ``[(tablet, rows), ...]``: runs, each a tablet *t* of the mapped
    table and the indices of its rows taken, in order -- None for all of them, a
    whole tablet. Every kind of fold tablet is a map tablet, and they differ only in
    their lot's shape: a whole tablet's is ``[(t, None)]``, a piece's ``[(t, rows)]``,
    a packed tablet's or a core tablet's several runs. A table pulled back along the
    map -- a `MappedDatatable` -- holds, as its tablet of the same index, tablet *t*
    itself for a lot ``[(t, None)]``, and otherwise a copy of the rows.

    Its build resolves its lot from its deal -- `DatatablePartition.lot` -- once: a
    piece's rows read off its tablet by its values, a core tablet's sliced out of
    the layout. It stores it as its ``lot`` slice, one row per run: the tablet, its
    rows, and whether it is taken whole. Built by its map, in parallel, or by a copy
    that needs it, first.

    Its VAR names its map, which forms it: ``datamap`` is in `TREE_SKIP_BUILDING`
    and `TREE_SKIP_VALIDATION`, so building it does not build its map, which builds
    it.
    """

    TOPICS = {'lot': DATASLICE(tab='int', rows='ndarray:int64', whole='int')}
    SPECIALIZATIONS = []
    TREE_SKIP_BUILDING = ('datamap',)
    TREE_SKIP_VALIDATION = ('datamap',)

    @dataclass
    class VAR(Datatablet.VAR):
        datamap: Datatable = None       # a PartitionDatamapTable, defined below: it names this class as its TABLET
        index: int = 0

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.var.datamap, PartitionDatamapTable):
            raise ValueError(f"{type(self).__name__}: VAR.datamap must be a PartitionDatamapTable, "
                             f"got {type(self.var.datamap).__name__}")

    def __build__(self):
        datamap = self.var.datamap
        deal = datamap.var.partition
        if not deal.dealt():
            raise ValueError(f"{type(self).__name__}: its deal has not dealt -- build {deal.anchorkeypath} first: "
                             f"a map tablet's lot is its deal's")
        with self.slice_writers() as writers:
            for t, rows in deal.lot(datamap.var.fold, self.var.index):
                # A whole tablet's rows are [-1], read as None: MDS encodes no empty array.
                writers['lot'].write({'tab': int(t), 'whole': int(rows is None),
                                      'rows': np.full(1, -1, dtype=np.int64) if rows is None
                                      else np.asarray(rows, dtype=np.int64)})

    # 3. Accessors ---------------------------------------------------------

    @functools.cached_property
    def lot(self) -> list:
        """``[(tablet, rows | None), ...]``: the rows of the mapped table's tablets this tablet takes, in order -- as built."""
        if not self.valid():
            raise ValueError(f"{type(self).__name__} {self.tag}: not built -- build it, or its map, "
                             f"{self.var.datamap.anchorkeypath}, which builds its tablets in parallel")
        lot = self.data('lot', concat=False)['lot']
        return [(int(t), None if w else np.asarray(r, dtype=np.int64))
                for t, r, w in zip(lot['tab'], lot['rows'], lot['whole'])]


class PartitionDatamapTable(Datatable):
    """The map of fold *fold* of a deal: a `PartitionDatamapTablet` per tablet of the fold.

    The layout of the fold, and nothing else: which rows of which tablets of the
    mapped table -- the deal's -- each tablet of the fold takes, its lot. The deal --
    a `DatatablePartition`, a `DatatableCorePartition` -- writes every fold at once,
    disjoint, and answers for each lot in the form it wrote it: the map only asks,
    and coordinates its tablets. A table is pulled back along it -- `pullback` --
    the map's own, or any laid out alike. See DATAPARTITION.md, "TODO: three
    choices, untangled".
    """

    TABLET = PartitionDatamapTablet
    TOPICS = Datatable.TOPICS
    SPECIALIZATIONS = []

    @dataclass
    class VAR(Datablock.VAR):
        partition: DatatablePartition
        fold: int

    # 1. Protocol and hooks ------------------------------------------------

    def _build_tag_(self, n: int) -> str:
        deal = self.var.partition
        return f"MAP: resolving {n} lots of fold {self.var.fold} of {type(deal).__name__}/{deal.hash[:8]} [{type(self).__name__}]"

    def __tablet__(self, idx: int, *, tag=None, **spec) -> PartitionDatamapTablet:
        return PartitionDatamapTablet(datalake=self._datalake_, storage_options=self.storage_options, verbose=False,
                             tag=tag if tag is not None else f"map{self.hash[:8]}_{int(idx):06d}",
                             spec=dict(datamap=self, index=int(idx)))

    # 2. Declared API ------------------------------------------------------

    def lot(self, idx: int) -> list:
        """Tablet *idx*'s lot, ``[(tablet, rows | None), ...]``, as its map tablet stores it -- see `PartitionDatamapTablet`."""
        return self.tablet(idx).lot

    def lot_tablets(self, idx: int) -> list[int]:
        """The tablets tablet *idx*'s lot takes rows of, in order, from the deal: no row resolved, nothing built."""
        return self.var.partition.lot_tablets(self.var.fold, idx)

    def whole(self, idx: int) -> int | None:
        """The tablet tablet *idx* of the fold is, whole -- or None, when it takes some rows only."""
        return self.var.partition.whole(self.var.fold, idx)

    def tag_of(self, idx: int) -> str:
        """What a tablet copying tablet *idx*'s lot is tagged, in its deal's words."""
        return self.var.partition.tag_of(self.var.fold, idx)

    def pullback(self, datatable: Datatable | None = None, collator: Datacollator | None = None,
                 **kwargs) -> MappedDatatable:
        """*datatable* -- this map's own table, by default -- pulled back along this map: a `MappedDatatable`.

        Another table must be laid out as this map's is, tablet for tablet and row
        for row -- a feature table and the table it was computed from -- and each
        copy checks it as it resolves its lot: a deal made over one table taken to
        every table laid out alike. *collator* names what the copies copy, every
        slice by default. A table reading upstream pulls back to a
        `MappedFeaturetable`, layer by layer.
        """
        if datatable is not None and datatable.hash == self.datatable.hash:
            datatable = None
        kwargs.setdefault('datalake', self._datalake_)
        kwargs.setdefault('storage_options', self.storage_options)
        return _mapped_table_(self, collator, datatable=datatable, **kwargs)

    # 3. Accessors ---------------------------------------------------------

    @functools.cached_property
    def entries(self) -> list:
        """Its fold's entries, as its deal wrote them -- see `DatatablePartition.tablets_indices`."""
        return self.var.partition.tablets_indices(self.var.fold)

    @property
    def n_tablets(self) -> int:
        return len(self.entries)

    @property
    def datatable(self) -> Datatable:
        """The mapped table: its deal's."""
        return self.var.partition.datatable


class MappedDatatablet(Datatablet):
    """A tablet of a `MappedDatatable` that copies some rows: those its map tablet's lot names.

    Its slices are the table's it is pulled back from -- those its *collator*
    names, of every role, or all of them -- declared as that table's tablets
    declare them, and it writes every one with the same rows in the same order;
    ``source`` names the tablet and the row each came from. A piece of a mixed
    tablet, a packed tablet and a core tablet are all one of these: what differs is
    the lot.

    The table is the map's own, or *datatable*, one laid out as it is: then every
    tablet the lot names is checked to hold as many rows as the map's table's, and
    it raises when they are not laid out alike.

    Its identity is its map tablet, its table and its collator: what it holds is what
    that place of the map takes, of that table.
    """

    SPECIALIZATIONS = []

    #: Whether it writes ``source``: a layered copy writes it once, at the bottom -- see `MappedFeaturetablet`.
    WRITES_SOURCE = True

    @dataclass
    class VAR(Datatablet.VAR):
        map: PartitionDatamapTablet = None
        datatable: Datatable | None = None
        collator: Datacollator | None = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.var.map, PartitionDatamapTablet):
            raise ValueError(f"{type(self).__name__}: VAR.map must be a PartitionDatamapTablet, got {type(self.var.map).__name__}")

    def __build__(self):
        what = type(self).__name__
        sources = self._sources_()
        first = sources[0][1]
        copied = self._copied_(first)
        columns = {s: {c: t for c, t in (self.declared_columns(s) or _written_columns_(first, s, what)).items()
                       if cols is None or c in cols}
                   for s, cols in copied.items()}
        # A slice at a time, and within it a source tablet at a time, so only one
        # source slice is held whole; the writers refuse unequal slices.
        if self.WRITES_SOURCE:
            columns[SOURCE_SLICE] = SOURCE.columns
        with self.slice_writers(columns) as writers:
            for s, cols in copied.items():
                for _, source, rows in sources:
                    data = source._read_slice_(s)
                    for i in rows:
                        writers[s].write(data[i] if cols is None else {c: data[i][c] for c in cols})
                    del data
            if self.WRITES_SOURCE:
                for t, _, rows in sources:
                    for i in rows:
                        writers[SOURCE_SLICE].write({'tab': int(t), 'row': int(i)})

    def build(self, *args, **kwargs):
        """Its map tablet first -- its lot, resolved once, stored -- then itself."""
        if not self.var.map.valid():
            self.var.map.build()
        return super().build(*args, **kwargs)

    # 2. Declared API ------------------------------------------------------

    def source_index(self) -> np.ndarray:
        """``(tablet, row)`` of the table it copies for each of its rows -- its ``source`` slice."""
        return _source_index_(self)

    # 3. Accessors ---------------------------------------------------------

    @property
    def source_table(self) -> Datatable:
        """The table it copies rows of: its *datatable*, or its map's own."""
        return self.var.datatable if self.var.datatable is not None else self.var.map.var.datamap.datatable

    @forward_property({})
    def TOPICS(self):
        """The slices it copies, as the tablet its first rows come from declares them, and ``source``.

        On the class, with no map to ask, nothing: even ``source`` is spelled as the slices it copies are.
        """
        mt = self.var.map
        first = self.source_table.tablet(mt.var.datamap.lot_tablets(mt.var.index)[0])
        topics = _slice_topics_(first, type(self).__name__)
        copied = self._copied_(first)
        out = {}
        for s, cols in copied.items():
            node = topics[s]
            if cols is not None:
                declared = first.declared_columns(s) or {}
                node = DATASLICE(**{c: declared[c] for c in cols}) if all(c in declared for c in cols) else DATASLICE
            out[s] = node
        if not self.WRITES_SOURCE:
            return out
        # Spelled as the slices it copies are: one declaration cannot mix the markers and the sentinels.
        legacy = bool(out) and bool(self._topic_spellings_(out)[1]) and not self._topic_spellings_(out)[0]
        return {**out, SOURCE_SLICE: SLICETOPIC if legacy else SOURCE}

    # 4. Helpers -----------------------------------------------------------

    def _copied_(self, first) -> dict:
        """``{slice: [column, ...] | None}``: what it copies -- its collator's columns, by slice; every slice, whole, by default."""
        collator, own = self.var.collator, list(first.slices())
        if collator is None:
            return {s: None for s in own}
        # Its own slices only: a layered copy's upstream takes the collator's others.
        out = {}
        for role in collator.roles:
            for pair in collator.pairs(role):
                if pair[0] not in own:
                    continue
                cols = out.setdefault(pair[0], [])
                if pair[1] not in cols:
                    cols.append(pair[1])
        return out

    def _sources_(self) -> list[tuple]:
        """``(tablet index, tablet, rows)`` of the table it copies, for each run of its lot: every row, for None.

        Of a table not the map's own, every tablet is checked to hold as many rows as the map's table's.
        """
        table, what = self.source_table, type(self).__name__
        mapped = self.var.map.var.datamap.datatable
        out = []
        for t, rows in self.var.map.lot:
            source = table.tablet(int(t))
            if self.var.datatable is not None:
                here = len(_selected_rows_(source, {}, {}, what))
                there = len(_selected_rows_(mapped.tablet(int(t)), {}, {}, what))
                if here != there:
                    raise ValueError(f"{what}: tablet {t} holds {here} rows here and {there} in the map's table: "
                                     f"they are not laid out alike, row for row")
            if rows is None:
                rows = _selected_rows_(source, {}, {}, what)
            out.append((int(t), source, [int(r) for r in rows]))
        return out


class MappedFeaturetablet(Layered, MappedDatatablet):
    """A copy of some rows of a table that reads upstream -- a `Featuretable` -- keeping its layers.

    It copies only that table's own slices -- features -- and its `upstream` is the
    same rows of that table's upstream, pulled back along the same map: a
    `MappedDatatablet`, or, if that table reads upstream too, another of these,
    and so on down. Nothing is flattened and nothing computed again: the features
    are copied, never re-evaluated. ``source`` is written once, at the bottom, and
    read through the upstream from every layer above it: the tables are laid out
    alike, so the indices are the same.
    """

    SPECIALIZATIONS = []
    WRITES_SOURCE = False

    @dataclass
    class VAR(MappedDatatablet.VAR):
        upstream: Datatablet = None


class MappedDatatable(Datatable):
    """A table pulled back along a `PartitionDatamapTable`: a tablet per tablet of the map, of the mapped table's rows.

    Its tablet *i* takes the rows map tablet *i* records: the mapped table's
    tablet *t* itself, built where it is, for a whole tablet -- the identity map
    pulls back to the identity, nothing copied -- and otherwise a
    `MappedDatatablet`, this table's to build, copying the rows. Its build() builds
    those, in parallel, as any table builds its tablets, skipping the ones built.

    Its BLOCK and TOPICS are the table's it pulls back: it is a view of its
    tablets. Its VAR is its map, the table -- None, the map's own, or one laid out
    as the map's is -- and a collator naming what a copy copies, every slice by
    default. A table that reads upstream -- a `Featuretable` -- is pulled back
    layer by layer: its copies are `MappedFeaturetablet`s, whose upstream is this
    table's upstream pulled back along the same map.
    """

    SPECIALIZATIONS = []

    @dataclass
    class VAR(Datablock.VAR):
        datamap: PartitionDatamapTable
        datatable: Datatable | None = None
        collator: Datacollator | None = None

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, *args, filter_built_tablets: bool = True, **kwargs):
        # True by default: a mapped table's whole tablets are its mapped table's, so the ones it
        # has built are checked here, in the parent, and no callable is sent to form and check
        # each one again in a worker.
        super().__init__(*args, filter_built_tablets=filter_built_tablets, **kwargs)

    def __tablet__(self, idx: int, *, tag=None, **spec) -> Datatablet:
        return self.tablet(idx)

    def __block__(self, idx: int) -> Datatablet:
        return self.tablet(idx)

    def build(self, *args, **kwargs):
        """Its map first, when it names one -- the lots, built in parallel -- then its copies, in parallel."""
        if not self.valid():
            self._log_stages_()
        datamap = getattr(self.var, 'datamap', None)
        if datamap is not None and not datamap.valid():
            datamap.build()
        # The layer below first, every copy of it, in parallel: each copy here reads its own from there.
        upstream = self._copies_upstream_()
        if upstream is not None and not upstream.valid():
            upstream.build(*args, **kwargs)
        return super().build(*args, **kwargs)

    def _build_tag_(self, n: int) -> str:
        return f"COPY: {n} tablets, layer by layer [{type(self).__name__}]"

    def _log_stages_(self):
        """The pullback's stages, said once, before it builds: the deal, the map, the copies and their layers."""
        datamap, what = self.datamap, f"{type(self).__name__}/{self.hash[:8]}"
        deal = datamap.var.partition
        n = self.n_tablets
        copies = sum(datamap.whole(i) is None for i in range(n))
        layers, table = [], self
        while table is not None:
            layers.append(f"{type(table.datatable).__name__} {list(table.slices())}")
            table = table.upstream if isinstance(table, Layered) else None
        self.log.info(
            f"PULLBACK {what}: {type(self.datatable).__name__} along fold {datamap.var.fold} of "
            f"{type(deal).__name__}/{deal.hash[:8]}\n"
            f"  DEAL {type(deal).__name__}/{deal.hash[:8]}: {'dealt, kept' if deal.dealt() else 'NOT DEALT'}\n"
            f"  MAP  {type(datamap).__name__}/{datamap.hash[:8]}: {datamap.n_tablets} lots, "
            f"{'built' if datamap.valid() else 'to build' if datamap is getattr(self.var, 'datamap', None) else 'built by each copy, as it needs its lot'}\n"
            f"  COPY {copies} of {n} tablets ({n - copies} taken whole), layers: {' <- '.join(layers)}")

    def _block_specializations_(self):
        """None for a table of copies: its block class's specializations are its source tablets', and no copy is one."""
        return [] if self._is_copies_only_() else super()._block_specializations_()

    # 2. Declared API ------------------------------------------------------

    def slices(self, recursive: bool = False):
        """The mapped table's slices it holds: those its collator names, or all; and ``source``, when every tablet is a copy.

        Its own, for a table reading upstream: the layers below answer for theirs.
        """
        table = self.datatable
        if isinstance(table, Layered):
            own = list(Layered.slices(table, recursive=False))
        else:
            own = list(table.slices(recursive=recursive))
        named = self._named_slices_()
        if named is not None:
            own = [s for s in own if s in named]
        if not isinstance(table, Layered) and self._is_copies_only_():
            own.append(SOURCE_SLICE)
        return tuple(own)

    def tablet(self, idx: int) -> Datatablet:
        """Tablet *idx*: the mapped table's tablet, taken whole, or the `MappedDatatablet` copying the rows its map names."""
        whole = self.datamap.whole(idx)
        if whole is not None:
            return self.datatable.tablet(whole)
        first = self.datatable.tablet(self.datamap.lot_tablets(idx)[0])
        spec = dict(datapoints_per_row=getattr(self.datatable.var, 'datapoints_per_row', 1),
                    map=self.datamap.tablet(idx), datatable=self._target_, collator=self.copy_collator)
        cls = MappedDatatablet
        if isinstance(first, Layered):
            # The tablets' upstream, layered or not this table's class: a part built as a DatatablePart has them too.
            cls, spec['upstream'] = MappedFeaturetablet, self._copies_upstream_().tablet(idx)
        return cls(
            # Where the map is: a copying tablet is derived from the map, and the mapped
            # table's lake may be one nothing writes to.
            datalake=self.datamap._datalake_,
            storage_options=self.storage_options,
            cache=getattr(first, 'cache', None),
            cache_limit=getattr(first, 'cache_limit', None),
            verbose=False,
            tag=self.datamap.tag_of(idx),
            spec=spec,
        )

    def valid_tablet(self, idx: int, validation: str | None = None) -> bool:
        """Whether tablet *idx* is valid: this table's manifest, or the mapped table's tablet, or the copy -- see `Datastack.valid_blocks`."""
        validation = self._validation_(validation)
        if validation == 'cross_check' and self._blocks_cross_checked_():
            return True
        whole = self.datamap.whole(idx)
        if whole is None:
            copy = self.tablet(idx)
            return bool(copy.validate() if validation == 'validate' else copy.valid())
        full = 'validate' if validation == 'validate' else 'valid'
        return self.datatable.valid_tablet(whole, validation=full)

    valid_block = valid_tablet

    def redirected_tablet(self, idx: int) -> bool:
        whole = self.datamap.whole(idx)
        return self.tablet(idx).redirected() if whole is None else self.datatable.redirected_tablet(whole)

    redirected_block = redirected_tablet

    def validate_tablet(self, idx: int, **kwargs) -> bool:
        whole = self.datamap.whole(idx)
        if whole is None:
            return Datastack.validate_block(self, idx, **kwargs)
        return self.datatable.validate_tablet(whole, **kwargs)

    validate_block = validate_tablet

    def source_index(self) -> np.ndarray:
        """``(tablet, row)`` of the table it pulls back for each of its rows, tablet by tablet -- an ``(N, 2)`` array.

        Which rows of the table it holds, however they are kept: a whole tablet is
        all of that tablet's rows, a copy what its ``source`` slice names -- so it
        must be built.
        """
        out, what = [], type(self).__name__
        for idx in range(self.n_tablets):
            whole = self.datamap.whole(idx)
            if whole is None:
                out.append(self.tablet(idx).source_index())
            else:
                n = len(_selected_rows_(self.datatable.tablet(whole), {}, {}, what))
                out.append(np.stack([np.full(n, whole, dtype=np.int64), np.arange(n, dtype=np.int64)], axis=1))
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)

    def datastream(self, slice, **kwargs) -> StreamingDataset:
        self.slice_names((slice,))
        cacheroot = self._ensure_cacheroot_(kwargs.pop('cache', None))
        cache_dir = kwargs.pop('cache_dir',
                               f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        local = os.path.join(cacheroot, cache_dir)
        os.makedirs(local, exist_ok=True)
        streams = self._tablet_streams_(slice, local)
        cache_limit = kwargs.pop('cache_limit', getattr(self, 'cache_limit', None))
        shuffle = kwargs.pop('shuffle', False)
        allow_unsafe_types = kwargs.pop('allow_unsafe_types', True)
        streaming_kwargs = dict(
            streams=streams,
            shuffle=shuffle,
            allow_unsafe_types=allow_unsafe_types,
            cache_limit=cache_limit,
            **kwargs,
        )
        try:
            return release_shared_memory_when_collected(StreamingDataset(**streaming_kwargs))
        except (ValueError, TypeError, OSError, FileExistsError, FileNotFoundError) as exc:
            SharedMemoryManager.clean_process_shared_memory()
            streaming_kwargs['streams'] = self._tablet_streams_(slice, local)
            return release_shared_memory_when_collected(StreamingDataset(**streaming_kwargs))

    # 3. Accessors ---------------------------------------------------------

    @property
    def datamap(self) -> PartitionDatamapTable:
        """The map it is pulled back along."""
        return self.var.datamap

    @property
    def copy_collator(self) -> Datacollator | None:
        """What a copying tablet copies: its collator's columns; None, every slice the mapped table's tablets hold."""
        return self.var.collator

    @functools.cached_property
    def _pulled_back_upstream_(self) -> MappedDatatable:
        """The table's upstream, pulled back along the same map -- for a table whose map and source fix it, a part's.

        A `MappedFeaturetable` holds it, as its VAR's ``upstream``; a part derives it, its
        VAR being ``{partition, fold}``. The same pullback either way: see `_mapped_table_`.
        """
        return self.datamap.pullback(self.datatable._upstream_block_(), self.copy_collator,
                                     datalake=self.datamap._datalake_, storage_options=self.storage_options)

    def _copies_upstream_(self) -> MappedDatatable | None:
        """The table a copy's upstream is a tablet of: its own upstream, layered, or the source's pulled back."""
        if self._is_whole_only_():
            return None                 # nothing copied: a whole tablet keeps its own upstream
        if isinstance(self, Layered):
            return self._upstream_block_()
        return self._pulled_back_upstream_ if isinstance(self.datatable, Layered) else None

    def _is_whole_only_(self) -> bool:
        """Whether every tablet is taken whole -- so no copy needs an upstream -- read from its deal's entries, when dealt."""
        deal = self.datamap.var.partition
        return deal.dealt() and all(self.datamap.whole(i) is not None for i in range(self.n_tablets))

    @property
    def datatable(self) -> Datatable:
        """The table it pulls back: its own *datatable*, or its map's."""
        return self._target_ if self._target_ is not None else self.datamap.datatable

    @property
    def _target_(self) -> Datatable | None:
        """The table it pulls back when not its map's own -- None, the map's."""
        return getattr(self.var, 'datatable', None)

    @property
    def datapoints_per_row(self) -> int:
        return getattr(self.datatable.var, 'datapoints_per_row')

    @forward_property(Datatablet)
    def TABLET(self):
        """The mapped table's TABLET: on the class, `Datatablet` -- what every mapped table's tablets are."""
        return getattr(self.datatable, 'TABLET', None)

    @forward_property({})
    def TOPICS(self):
        """The mapped table's TOPICS: a mapped table is a view of its tablets, and declares nothing of its own.

        On the class, with no table to ask, ``{}``.
        """
        return self.datatable.TOPICS

    @property
    def n_tablets(self) -> int:
        return self.datamap.n_tablets

    # 4. Helpers -----------------------------------------------------------

    def _named_slices_(self) -> set | None:
        """The slices its collator names, of every role -- None, with no collator: every slice."""
        collator = self.copy_collator
        if collator is None:
            return None
        return {pair[0] for role in collator.roles for pair in collator.pairs(role)}

    def _is_copies_only_(self) -> bool:
        """Whether every tablet is a copy -- so every one holds ``source`` -- and none the mapped table's own.

        Its deal's to say, and asked before anything is built -- a journal entry of a tree
        about to build records its slices -- so it must not need the deal dealt.
        """
        return self.datamap.var.partition._is_copies_only_(self.datamap.var.fold)

    def _read_slice_(self, slice, *, tablets=None, **kwargs):
        indices = range(self.n_tablets) if tablets is None else tablets
        datapoints = []
        for idx in indices:
            datapoints.extend(self.tablet(idx)._read_slice_(slice, **kwargs))
        return datapoints

    def _blocks_datalake_(self):
        """A mapped table's whole tablets are its mapped table's, and are stored where they are."""
        return self.datatable._blocks_datalake_()

    def _form_block_(self, idx: int, *, use_specializations='stack'):
        """As `Datastack._form_block_`, and a copying tablet passes for the TABLET of the tablets it copies.

        A mapped table's BLOCK is its mapped table's TABLET, and is in its hash, so it
        stays that for a mapped table holding copies; a copy copies the mapped table's
        own tablets, which are of that class by that table's own check.
        """
        if self.datamap.whole(idx) is not None:
            return super()._form_block_(idx, use_specializations=use_specializations)
        with self._block_specializations_in_force_(use_specializations):
            return self._adopt_(self.__block__(idx), keyby=True)

    def _block_class_(self):
        """A mapped table's tablets are its mapped table's, and so is its BLOCK."""
        try:
            table = self.datatable
        except Exception:
            return None
        return table._block_class_() if table is not None else None


class MappedFeaturetable(Layered, MappedDatatable):
    """A table that reads upstream -- a `Featuretable` -- pulled back along a map, keeping its layers.

    An ordinary layered table, which also carries the map that made it: its own
    slices are the table's, and its VAR's ``upstream`` is the table's upstream
    pulled back along the same map -- a `MappedDatatable`, or another of these, and
    so on down -- built first, by `build_tree`, bottom up. So it routes a request for
    an upstream slice as the table it maps does, and its copying tablets are
    `MappedFeaturetablet`s whose upstreams are its upstream's tablets:

        map.pullback(ft).tablet(i).var.upstream == map.pullback(ft.upstream).tablet(i)

    over the layers the pullback keeps: a layer its collator names nothing of is
    passed over, to the nearest one below that it does.
    """

    SPECIALIZATIONS = []

    @dataclass
    class VAR(MappedDatatable.VAR):
        upstream: Datatable = None


def _mapped_table_(datamap: PartitionDatamapTable, collator: Datacollator | None = None, datatable: Datatable | None = None,
                   **kwargs) -> MappedDatatable:
    """*datatable* -- *datamap*'s own, for None -- pulled back along it, layer by layer, from the bottom up.

    The bottom layer is always pulled back -- it holds ``source`` -- as a `MappedDatatable`.
    Each layer above is a `MappedFeaturetable` whose VAR's ``upstream`` is the pullback of
    the layer below it; a layer *collator* names nothing of is passed over, its place
    taken by the pullback below it -- so the result may be a `MappedDatatable` alone.
    """
    if datatable is not None and datatable.hash == datamap.datatable.hash:
        datatable = None            # the map's own table: one block, however it is reached
    table = datatable if datatable is not None else datamap.datatable
    if not isinstance(table, Layered):
        return MappedDatatable(spec=dict(datamap=datamap, datatable=datatable, collator=collator), **kwargs)
    below = _mapped_table_(datamap, collator, table._upstream_block_(), **kwargs)
    if collator is not None:
        named = {pair[0] for role in collator.roles for pair in collator.pairs(role)}
        if not set(Layered.slices(table, recursive=False)) & named:
            return below
    return MappedFeaturetable(spec=dict(datamap=datamap, datatable=datatable, collator=collator, upstream=below),
                              **kwargs)


class DatatablePart(MappedDatatable):
    """Part *fold* of a `DatatablePartition`: its table, pulled back along the fold's map.

    Its map is the `PartitionDatamapTable` of ``(partition, fold)``, derived, not held: its
    VAR is ``{partition, fold}``, as it has always been, and so is every identity
    built on a part. Its tablets are its table's own, where the fold takes them
    whole, and otherwise `MappedDatatablet`s -- a piece of a mixed tablet, a packed
    tablet -- this part's to build.
    """

    # None: a part is its partition's, and no partition with a collator is reached from before
    # the roles were renamed.
    SPECIALIZATIONS = []

    @dataclass
    class VAR(Datablock.VAR):
        partition: DatatablePartition
        fold: int

    # 3. Accessors ---------------------------------------------------------

    @functools.cached_property
    def datamap(self) -> PartitionDatamapTable:
        """The map of its fold: derived from ``(partition, fold)``, so that its VAR -- and every identity built on it -- is as it was."""
        return self.var.partition.map(self.var.fold)

    @property
    def copy_collator(self) -> Datacollator | None:
        """A part copies every slice."""
        return None

    @property
    def tablet_indices(self) -> list:
        """The partition's entries for this fold -- see `DatatablePartition.tablets_indices`."""
        return self.datamap.entries

    @property
    def datatable(self) -> Datatable:
        return self.var.partition.datatable

    @property
    def dealt(self) -> bool:
        """Whether its partition has dealt its tablets: the ``tablets`` topic is there, built or redirected.

        Not the partition's `valid`, which a partition owing a topic only it computes -- one adopted from
        before its ``summary`` -- is not, its tablets readable all the same.
        """
        return self.var.partition.valid_topic('tablets')

    def valid(self) -> bool:
        """Not valid while its tablets are not dealt: they are the partition's to say, and not there to ask about."""
        return self.dealt and super().valid()

    def specialize(self, *args, **kwargs):
        """Nothing to specialize while its tablets are not dealt -- see `valid`; else as any table."""
        return super().specialize(*args, **kwargs) if self.dealt else self

    def _block_class_(self):
        """A part's tablets are its table's, and so is its BLOCK."""
        table = getattr(getattr(self, 'var', None), 'partition', None)
        table = getattr(table, 'datatable', None)
        return table._block_class_() if table is not None else None


class FeaturetablePart(Layered, DatatablePart):
    """Part *fold* of a `DatatablePartition` of a table that reads upstream -- a `Featuretable` -- keeping its layers.

    As a `DatatablePart`, and its `upstream` is the partitioned table's
    upstream, pulled back along the fold's map -- derived,
    not a VAR field -- so it routes a request for an upstream slice, an
    annotation, as the feature table does.

    Until 2026-10-13 a part of a feature table was a `DatatablePart`, stored under
    that class's path; its identity is unchanged -- the class is no part of a hash
    -- and a part built then is adopted where it is, by its specialization.
    """

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, topics=SAME, anchor=f'{__name__}.DatatablePart',
            note="this very part, built as a DatatablePart, under that class's path, until 2026-10-13"),
    ]

    def _upstream_block_(self):
        return self._pulled_back_upstream_


#: The roles a `DatatableCorePartition`'s collator may hold -- see its docstring.
CORE_ROLES = ('coreby', 'stratifyby', 'partitionby', 'carry')

#: Where a `DatatableCorePartition` clusters -- see its docstring.
CORE_CLUSTERINGS = ('master', 'distributed')


def _spec_path_(pair) -> tuple:
    """A collator's normalized pair as a `_column_values_` spec: ``(slice, column, key, ...)``."""
    s_name, column, *rest = pair
    keys = () if not rest else (rest[0] if isinstance(rest[0], tuple) else (rest[0],))
    return (s_name, column, *keys)


def _row_vectors_(tablet, pairs) -> np.ndarray:
    """Each row's *pairs*, flattened and laid end to end, as float32 -- a row holding none of one is all NaN."""
    columns = [_column_values_(tablet, _spec_path_(p)) for p in pairs]
    n = len(columns[0]) if columns else 0
    rows, width = [], None
    for i in range(n):
        parts = [None if col[i] is None else np.asarray(col[i], dtype=np.float32).ravel() for col in columns]
        rows.append(None if any(p is None for p in parts) else np.concatenate(parts))
        if rows[-1] is not None:
            width = len(rows[-1])
    out = np.full((n, width or 0), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            if len(r) != width:
                raise ValueError(f"{tablet.tag}: row {i} flattens {[p for p in pairs]} to {len(r)} values, "
                                 f"others to {width}: the rows of a table must flatten alike to be clustered")
            out[i] = r
    return out


class TabletCoreScanCallable:
    """Worker callable: one tablet's rows as vectors, and each row's stratum and partition.

    ``'master'`` clustering returns the vectors; ``'distributed'`` writes them
    to the partition's scratch directory, for the clustering passes to read
    there, and returns only a seeded sample of them to start the centers from.
    """

    def __init__(self, partition, idx: int):
        self.partition = partition
        self.idx = idx

    def __call__(self):
        part, var = self.partition, self.partition.var
        tablet = var.datatable.tablet(self.idx)
        vectors = _row_vectors_(tablet, part.collator_pairs('coreby'))
        out = {'idx': self.idx, 'tag': tablet.tag, 'n': len(vectors), 'width': vectors.shape[1],
               'valid': ~np.isnan(vectors).any(axis=1) if vectors.size else np.ones(len(vectors), dtype=bool)}
        for role in PARTITION_ROLES:
            spec = part.column_spec(role)
            out[role] = None if spec is None else [None if v is None else _value_key_(v)
                                                   for v in _column_values_(tablet, tuple(spec))]
            if out[role] is not None and len(out[role]) != len(vectors):
                raise ValueError(f"{type(part).__name__}: {tablet.tag}'s {role} column has {len(out[role])} "
                                 f"values for its {len(vectors)} rows")
        if var.clustering == 'master':
            out['vectors'] = vectors
            return out
        with part.fs.open(part._scratch_path_(self.idx), 'wb') as f:
            np.save(f, vectors)
        valid = np.flatnonzero(out['valid'])
        m = min(len(valid), -(-var.init_size // max(var.datatable.n_tablets, 1)))
        rng = np.random.default_rng([var.seed, self.idx])
        out['sample'] = vectors[np.sort(rng.choice(valid, size=m, replace=False))] if m else vectors[:0]
        return out


class TabletCoreStepCallable:
    """Worker callable: one Lloyd step over one tablet's scratch vectors, or -- *final* -- its rows' clusters and distances."""

    def __init__(self, partition, idx: int, centers: np.ndarray, final: bool = False):
        self.partition = partition
        self.idx = idx
        self.centers = centers
        self.final = final

    def __call__(self):
        part, centers = self.partition, self.centers
        with part.fs.open(part._scratch_path_(self.idx), 'rb') as f:
            vectors = np.load(f)
        labels, dist = _nearest_(vectors, centers)
        if self.final:
            return {'idx': self.idx, 'labels': labels, 'dist': dist}
        valid = labels >= 0
        sums = np.zeros_like(centers, dtype=np.float64)
        np.add.at(sums, labels[valid], vectors[valid])
        counts = np.bincount(labels[valid], minlength=len(centers))
        return {'sums': sums, 'counts': counts, 'inertia': float((dist[valid] ** 2).sum())}


def _nearest_(vectors: np.ndarray, centers: np.ndarray, chunk: int = 65536):
    """Each row's nearest center and its distance -- -1 and NaN for a row of NaNs -- a chunk at a time."""
    labels = np.full(len(vectors), -1, dtype=np.int64)
    dist = np.full(len(vectors), np.nan, dtype=np.float64)
    c2 = (centers.astype(np.float64) ** 2).sum(axis=1)
    for lo in range(0, len(vectors), chunk):
        x = vectors[lo:lo + chunk].astype(np.float64)
        ok = ~np.isnan(x).any(axis=1)
        if not ok.any():
            continue
        d2 = (x[ok] ** 2).sum(axis=1)[:, None] - 2 * x[ok] @ centers.T.astype(np.float64) + c2[None, :]
        best = d2.argmin(axis=1)
        idx = np.arange(lo, lo + len(x))[ok]
        labels[idx] = best
        dist[idx] = np.sqrt(np.maximum(d2[np.arange(len(best)), best], 0))
    return labels, dist


class TabletRowCountCallable:
    """Worker callable: one tablet's rows in *slice* -- what a reused layout is checked against."""

    def __init__(self, table, idx: int, slice: str):
        self.table, self.idx, self.slice = table, idx, slice

    def __call__(self):
        return int(self.table.tablet(self.idx).n_rows(self.slice))


#: The quantiles the ``coverage`` topic reports each distance at.
COVERAGE_LEVELS = (0.5, 0.9, 0.99, 1.0)


def _min_distances_(queries: np.ndarray, rows: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """``(len(queries), len(rows))`` Euclidean distances, a chunk of queries at a time."""
    r = rows.astype(np.float64)
    r2 = (r ** 2).sum(axis=1)
    out = np.empty((len(queries), len(rows)), dtype=np.float64)
    for lo in range(0, len(queries), chunk):
        q = queries[lo:lo + chunk].astype(np.float64)
        d2 = (q ** 2).sum(axis=1)[:, None] - 2 * q @ r.T + r2[None, :]
        out[lo:lo + chunk] = np.sqrt(np.maximum(d2, 0))
    return out


class TabletCoverageCallable:
    """Worker callable: each query's distance to its nearest row of this tablet, in each named set of the tablet's rows.

    A query is compared only with the rows of its *coverage_clusters*
    nearest clusters -- what makes a pass over the table cost a fraction of
    it, and what makes each distance an upper bound of the true one, equal to
    it whenever the nearest row lies in one of those clusters.

    *jobs*: ``(queries, ids, query_clusters, names, exclude)`` -- the query
    vectors, their own ``(tablet, row)``, their candidate clusters, the sets to
    measure against, and those of them in which a query's own row does not
    count (a row is not its own nearest neighbour).
    """

    def __init__(self, partition, idx: int, labels: np.ndarray, sets: dict, jobs: list, vectors=None):
        self.partition, self.idx, self.labels, self.sets, self.jobs = partition, idx, labels, sets, jobs
        self.vectors = vectors

    def __call__(self):
        vectors = self.vectors
        if vectors is None:
            with self.partition.fs.open(self.partition._scratch_path_(self.idx), 'rb') as f:
                vectors = np.load(f)
        out = []
        for queries, ids, query_clusters, names, exclude in self.jobs:
            res = {}
            for name in names:
                best = np.full(len(queries), np.inf)
                rows = self.sets.get(name, np.zeros(0, dtype=np.int64))
                row_clusters = self.labels[rows]
                for c in np.unique(row_clusters):
                    q = np.flatnonzero((query_clusters == c).any(axis=1))
                    if not len(q):
                        continue
                    r = rows[row_clusters == c]
                    d = _min_distances_(queries[q], vectors[r])
                    if name in exclude:
                        d[(ids[q, 0][:, None] == self.idx) & (ids[q, 1][:, None] == r[None, :])] = np.inf
                    best[q] = np.minimum(best[q], d.min(axis=1))
                res[name] = best
            out.append(res)
        return out


class TabletVectorsCallable:
    """Worker callable: the scratch vectors of some of one tablet's rows."""

    def __init__(self, partition, idx: int, rows: np.ndarray):
        self.partition, self.idx, self.rows = partition, idx, rows

    def __call__(self):
        with self.partition.fs.open(self.partition._scratch_path_(self.idx), 'rb') as f:
            return np.load(f)[self.rows]


class DatatableCorePartition(DatatablePartition):
    """A `DatatablePartition` dealing ROWS, clustered by their values, into folds of fresh tablets: a coreset.

    Its collator names, by role, every column it touches:

    *coreby* -- the columns rows are clustered by: each row's, flattened and
    laid end to end, is the vector k-means clusters. Required.

    *stratifyby*, *partitionby* -- as for `DatatablePartition`: a stratum's rows
    are dealt whole, to one fold, and the fractions hold within each partition.

    *carry* -- columns the folds hold that nothing above reads: labels, for a
    probe. A fold's tablets hold the columns of every role, and no others.

    With ``recursive=True`` a column may be upstream of the table -- a feature
    table's annotations -- and it is copied into the folds' tablets like any
    other: they read nothing upstream.

    The deal. Rows are clustered into *n_clusters*, each row's distance to
    its center kept. A unit -- a stratum, or a row when there is no stratifyby --
    belongs to the cell of its partition and its rows' most frequent cluster,
    and is ranked by its rows' mean distance to their centers. Within each
    cell, units are dealt nearest first, each to the fold furthest below its
    target, ``fraction * rows in the cell``, while it brings that fold nearer
    to it. So *fractions* are of the rows, NOT normalized: summing to 1, every
    row is dealt and each fold samples every cluster; summing to less, each
    fold takes the rows nearest each center -- a coreset, filling out the
    table as its clusters do. A unit whose rows lack a value is skipped, and
    counted in ``summary``.

    *clustering* -- where k-means runs. ``'master'``: the workers read the
    vectors, the master clusters them all at once with scikit-learn's
    `MiniBatchKMeans` -- it must hold them. ``'distributed'``: the workers
    write their vectors to scratch, and the master runs Lloyd's iterations,
    each pass's sums computed by the workers tablet by tablet, holding only the
    centers -- for a table too large to gather. The scratch is removed when
    the build is done.

    The layout. A fold's rows, as ``(tablet, row, cluster)``, in source order --
    or, *shuffle*, in a seeded random one -- are cut into tablets of
    *rows_per_tablet*: copies, `MappedDatatablet`s, built by the fold's part,
    ``partition.fold(k).build()``, in parallel. Source order makes a core tablet
    read a few neighbouring tablets; a shuffled one reads up to as many tablets as
    it has rows.

    A table laid out as this one's is -- a feature table and the table it was
    computed from -- is pulled back along its maps: ``partition.map(k).pullback(table, collator)``.

    Topics: ``tablets`` -- per fold, each core tablet's ``[start, end)`` in its
    layout; ``layout`` -- per fold, ``(tablet, row, cluster)`` per row;
    ``centers``; ``columns`` -- what the core tablets carry, ``{slice: {column:
    type}}``; ``summary``.
    """

    # 2: the roles were renamed on 2026-10-10 -- see DatatablePartition.VERSION.
    VERSION = 2

    TOPICS = {
        'tablets': DATAFILE('tabs.json', "per fold, each core tablet's [start, end) in the fold's layout"),
        'layout': DATADIR('per fold, (tablet, row, cluster) of each row dealt to it, as <fold>.npy'),
        'centers': DATAFILE('centers.npy', 'the cluster centers'),
        'columns': DATAFILE('columns.json', 'the columns the core tablets carry: {slice: {column: type}}'),
        'summary': DATAFILE('summary.json', 'per fold its rows, strata, clusters and core tablets; what was skipped'),
        'coverage': DATADICT('coverage.npz', levels='ndarray', sample='ndarray', scale='ndarray',
                             coverage='ndarray', coverage_mean='ndarray', baseline='ndarray',
                             baseline_mean='ndarray', separation='ndarray', cluster_rows='ndarray',
                             cluster_tv='ndarray', empty_clusters='ndarray'),
    }
    SPECIALIZATIONS = []

    @dataclass
    class VAR(DatatablePartition._DealVAR_):
        clustering: str = 'master'
        n_clusters: int = 64
        max_iter: int = 100
        tol: float = 1e-4
        batch_size: int = 4096
        init_size: int = 65536
        rows_per_tablet: int = 4096
        shuffle: bool = False
        coverage_sample: int = 10000
        coverage_clusters: int = 3
        # Retired on 2026-10-13 for a map's pullback; kept, always None, because a
        # VAR's defaults are its identity: removed, it would move the hash of every core partition built.
        layout: DatatablePartition = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        Datablock.__post_init__(self)
        var, what = self.var, type(self).__name__
        if var.method != 'random' or var.balance != 'rows':
            raise ValueError(f"{what}: deals rows, by cluster -- it takes no method or balance")
        if var.clustering not in CORE_CLUSTERINGS:
            raise ValueError(f"{what}: clustering must be one of {CORE_CLUSTERINGS}, got {var.clustering!r}")
        for name in ('n_clusters', 'max_iter', 'batch_size', 'init_size', 'rows_per_tablet', 'coverage_clusters'):
            if not (isinstance(getattr(var, name), int) and getattr(var, name) >= 1):
                raise ValueError(f"{what}: {name} must be a positive int, got {getattr(var, name)!r}")
        if not (isinstance(var.coverage_sample, int) and var.coverage_sample >= 0):
            raise ValueError(f"{what}: coverage_sample must be an int >= 0, got {var.coverage_sample!r}")
        if not var.fractions or any(f < 0 for f in var.fractions) or sum(var.fractions) > 1 + 1e-9:
            raise ValueError(f"{what}: fractions are of the rows, and must be non-negative and sum to at most 1 "
                             f"-- less for a coreset -- got {var.fractions}")
        if var.layout is not None:
            raise ValueError(f"{what}: layout is retired; pull a table laid out alike back along a core "
                             f"partition's map: partition.map(fold).pullback(table, collator)")
        if var.collator is None:
            raise ValueError(f"{what}: a collator naming the columns, by role, is required")
        other = [r for r in var.collator.roles if r not in CORE_ROLES]
        if other:
            raise ValueError(f"{what}: a core partition reads the roles {CORE_ROLES}; its collator declares {other} too")
        if not var.collator.pairs('coreby'):
            raise ValueError(f"{what}: its collator names no 'coreby' columns to cluster the rows by")
        for name in PARTITION_ROLES:
            spec = var.collator.var.columns.get(name)
            if spec is not None and not (isinstance(spec, tuple) and len(spec) >= 2 and all(isinstance(p, str) for p in spec)):
                raise ValueError(f"{what}: the collator's {name} must be one column spec, a tuple "
                                 f"(slice, column[, key, ...]) of strings, got {spec!r}")
        self._check_carried_slices_()

    def __build__(self):
        self.log.info(f"DEAL {type(self).__name__}/{self.hash[:8]}: {type(self.var.datatable).__name__} into "
                      f"{self.n_folds()} folds -- SCAN, CLUSTER, DEAL, COVERAGE, LAYOUT; the parts copy, later")
        self.var.collator.slices(self.var.datatable)       # raises for a column the table does not hold
        layouts, centers, summary, coverage = self._lay_out_()
        with self.fs.open(self.path('coverage', ensure_dirpath=True), 'wb') as f:
            np.savez(f, **coverage)
        n = self.rows_per_tablet
        tablets = [[[lo, min(lo + n, len(fl))] for lo in range(0, len(fl), n)] for fl in layouts]
        for k, fl in enumerate(layouts):
            with self.fs.open(os.path.join(self.path('layout', ensure_dirpath=True), f'{k}.npy'), 'wb') as f:
                np.save(f, fl)
        with self.fs.open(self.path('centers', ensure_dirpath=True), 'wb') as f:
            np.save(f, centers)
        with self.fs.open(self.path('columns', ensure_dirpath=True), 'w') as f:
            json.dump(self._carried_schema_(), f)
        with self.fs.open(self.path('tablets', ensure_dirpath=True), 'w') as f:
            json.dump(tablets, f)
        for k, f_tablets in enumerate(tablets):
            summary['folds'][k]['core_tabs'] = len(f_tablets)
        with self.fs.open(self.path('summary', ensure_dirpath=True), 'w') as f:
            json.dump(summary, f, indent=1, default=str)

    def __read__(self, *topicpath):
        topicpath = self._normtopic_(topicpath)
        if topicpath in (('tablets',), ('summary',), ('columns',)):
            return json.loads(self.fs.cat(self.path(topicpath[0])))
        if topicpath == ('centers',):
            with self.fs.open(self.path('centers'), 'rb') as f:
                return np.load(f)
        if topicpath == ('layout',):
            return [self.layout(k) for k in range(self.n_folds())]
        if topicpath == ('coverage',):
            with self.fs.open(self.path('coverage'), 'rb') as f:
                with np.load(f) as data:
                    return {key: data[key] for key in data.files}
        return Datablock.__read__(self, *topicpath)

    # 2. Declared API ------------------------------------------------------

    def tablets_indices(self, fold: int | str) -> list:
        """Fold *fold*'s core tablets, each its ``[start, end)`` in the fold's `layout`."""
        return json.loads(self.fs.cat(self.path('tablets')))[int(fold)]

    def layout(self, fold: int | str) -> np.ndarray:
        """Fold *fold*'s rows, in order: ``(tablet, row, cluster)`` of the partitioned table, one row each."""
        with self.fs.open(os.path.join(self.path('layout'), f'{int(fold)}.npy'), 'rb') as f:
            return np.load(f)

    def lot(self, fold: int | str, idx: int) -> list:
        """Core tablet *idx* of fold *fold*'s lot: its ``[start, end)`` of the fold's layout, as runs of one source tablet each."""
        start, end = self._dealt_tablets_[int(fold)][int(idx)]
        layout = self._layouts_(int(fold))[int(start):int(end)]
        if not len(layout):
            return []
        cut = np.flatnonzero(np.diff(layout[:, 0])) + 1
        return [(int(run[0, 0]), run[:, 1].astype(np.int64)) for run in np.split(layout, cut)]

    def lot_tablets(self, fold: int | str, idx: int) -> list[int]:
        """The tablets core tablet *idx* of fold *fold* takes rows of, in the layout's order."""
        start, end = self._dealt_tablets_[int(fold)][int(idx)]
        tabs = self._layouts_(int(fold))[int(start):int(end), 0]
        if not len(tabs):
            return []
        return [int(t) for t in tabs[np.r_[0, np.flatnonzero(np.diff(tabs)) + 1]]]

    def whole(self, fold: int | str, idx: int) -> int | None:
        """None: a core tablet is a copy, never a whole tablet."""
        return None

    def tag_of(self, fold: int | str, idx: int) -> str:
        return f"core{self.hash[:8]}_f{int(fold)}_{int(idx):06d}"

    def dealt(self) -> bool:
        """Whether it has laid the folds out: built, or adopted."""
        return self.valid()

    def _is_copies_only_(self, fold: int | str) -> bool:
        """True: every core tablet is a copy, dealt or not."""
        return True

    def fold(self, fold: int | str) -> MappedDatatable:
        """Fold *fold* of its own table: pulled back along `map(fold)`, copying what the collator names, of every role.

        A table reading upstream keeps its layers -- see `MappedFeaturetable`. Any other table
        laid out alike is pulled back along the same map: ``partition.map(fold).pullback(table, collator)``.
        """
        return self.map(fold).pullback(None, self.var.collator, **self._part_kwargs_)

    def collator_pairs(self, role: str) -> tuple:
        """The collator's pairs playing *role*, as `Datacollator.pairs` gives them."""
        return self.var.collator.pairs(role)

    def carried_slices(self) -> list[str]:
        """The slices the core tablets hold, in the order the collator first names them: every role's."""
        return list(self._carried_columns_())

    # 3. Accessors ---------------------------------------------------------

    @property
    def rows_per_tablet(self) -> int:
        """The rows of each core tablet, the last of a fold holding what is left."""
        return self.var.rows_per_tablet

    def _layouts_(self, fold: int) -> np.ndarray:
        """Fold *fold*'s layout, read once for the lots of all its core tablets."""
        cache = self.__dict__.setdefault('_layout_cache', {})
        if fold not in cache:
            cache[fold] = self.layout(fold)
        return cache[fold]

    # 4. Helpers -----------------------------------------------------------

    def _lay_out_(self):
        """``(layouts, centers, summary, coverage)``: each fold's rows, as `layout` reads them, and how they were chosen."""
        return self._cluster_and_deal_()

    def _check_carried_slices_(self):
        what = type(self).__name__
        if SOURCE_SLICE in self.carried_slices():
            raise ValueError(f"{what}: a carried slice is named {SOURCE_SLICE!r}, the slice a core tablet points back "
                             f"with; rename it")
        if any('/' in s for s in self.carried_slices()):
            raise NotImplementedError(f"{what}: nested slices ({self.carried_slices()}) are not carried yet")

    def _carried_columns_(self) -> dict:
        """``{slice: [column, ...]}``: every column a role of the collator names, whole, in the order named."""
        out = {}
        for role in self.var.collator.roles:
            for pair in self.collator_pairs(role):
                cols = out.setdefault(pair[0], [])
                if pair[1] not in cols:
                    cols.append(pair[1])
        return out

    def _carried_schema_(self) -> dict:
        """The carried columns with their MDS types, read off the first tablet -- or the tablet its slice is upstream in."""
        tablet = self.var.datatable.tablet(0)
        owners = tablet._upstream_slices_() if hasattr(tablet, '_upstream_slices_') else {}
        schema = {}
        for s, cols in self._carried_columns_().items():
            owner = tablet if s in tablet.slices() else owners[s]
            declared = owner.declared_columns(s)
            if not declared:
                with owner.fs.open(owner.slice_index_path(s), 'r') as f:
                    shard = json.load(f)['shards'][0]
                declared = dict(zip(shard['column_names'], shard['column_encodings']))
            missing = [c for c in cols if c not in declared]
            if missing:
                raise KeyError(f"{type(self).__name__}: slice {s!r} has no column(s) {missing}; it has {list(declared)}")
            schema[s] = {c: declared[c] for c in cols}
        return schema

    def _scratch_path_(self, idx: int) -> str:
        return os.path.join(self.anchorkeypath, '.scratch', 'vectors', f'{idx}.npy')

    def _executor_(self, tag: str):
        """The executor a stage runs on: while clustering, the one whose workers every stage reuses, retagged."""
        tag = f"{tag} [{type(self).__name__}]"
        reusing = self.__dict__.get('_reusing_')
        if reusing is not None:
            reusing.tag = tag
            return reusing
        return callable_executor(getattr(self, 'parallelization', None) or 'inline',
                                 n_workers=getattr(self, 'n_workers', 1) or 1, tag=tag)

    def _cluster_and_deal_(self):
        """Scan, cluster, deal and measure: ``(layouts, centers, summary, coverage)``."""
        var, table = self.var, self.var.datatable
        if var.clustering == 'distributed':
            self.fs.makedirs(os.path.dirname(self._scratch_path_(0)), exist_ok=True)
        # One set of workers for every stage -- the scan, each of up to max_iter passes, the
        # assignment, the coverage -- rather than a set started anew for each: minutes apiece.
        self.__dict__['_reusing_'] = callable_executor(
            getattr(self, 'parallelization', None) or 'inline', n_workers=getattr(self, 'n_workers', 1) or 1,
            tag=type(self).__name__, reuse_workers=True)
        try:
            scans = self._executor_(f"SCANNING {table.n_tablets} tablets").exec_callables(
                [TabletCoreScanCallable(self, i) for i in range(table.n_tablets)])
            widths = {s['width'] for s in scans if s['valid'].any()}
            if len(widths) > 1:
                raise ValueError(f"{type(self).__name__}: the tablets' coreby columns flatten to different widths "
                                 f"{sorted(widths)}; rows must flatten alike to be clustered")
            n_valid = sum(int(s['valid'].sum()) for s in scans)
            if n_valid == 0:
                raise ValueError(f"{type(self).__name__}: no row holds every coreby column")
            k = min(var.n_clusters, n_valid)
            if k < var.n_clusters:
                self.log.warning(f"{type(self).__name__}: {n_valid} rows to cluster, fewer than n_clusters="
                                 f"{var.n_clusters}: clustering into {k}")
            self.log.info(f"{type(self).__name__}: SCANNED {n_valid} rows of width {widths.pop() if widths else 0} "
                          f"in {len(scans)} tablets; clustering them into {k} ({var.clustering})")
            centers, labels, dist, inertia = (self._cluster_master_(scans, k) if var.clustering == 'master'
                                              else self._cluster_distributed_(scans, k))
            self.log.info(f"{type(self).__name__}: DEALING the rows of {k} clusters to folds of fractions "
                          f"{list(var.fractions)}")
            layouts, summary = self._deal_rows_(scans, labels, dist)
            self.log.info(f"{type(self).__name__}: DEALT {[len(fl) for fl in layouts]} rows to the folds")
            # Before the scratch goes: the distances are measured on the vectors it holds.
            if var.coverage_sample:
                self.log.info(f"{type(self).__name__}: MEASURING coverage from {var.coverage_sample} sampled rows")
            coverage = self._coverage_(scans, labels, layouts, centers) if var.coverage_sample else {}
        finally:
            self.__dict__.pop('_reusing_').close()
            if var.clustering == 'distributed':
                scratch = os.path.dirname(os.path.dirname(self._scratch_path_(0)))
                if self.fs.exists(scratch):
                    self.fs.rm(scratch, recursive=True)
        summary.update(clustering=var.clustering, n_clusters=int(len(centers)), inertia=inertia)
        return layouts, centers, summary, coverage

    def _cluster_master_(self, scans, k):
        """Every valid vector gathered here, clustered by `MiniBatchKMeans`: ``(centers, labels, dist, inertia)`` per tablet."""
        from sklearn.cluster import MiniBatchKMeans
        var = self.var
        X = np.concatenate([s['vectors'][s['valid']] for s in scans], axis=0)
        self.log.info(f"{type(self).__name__}: CLUSTERING {X.shape[0]} x {X.shape[1]} on the master "
                      f"(MiniBatchKMeans, k={k}, max_iter={var.max_iter})")
        km = MiniBatchKMeans(n_clusters=k, batch_size=var.batch_size, max_iter=var.max_iter, tol=var.tol,
                             n_init=3, random_state=var.seed).fit(X)
        centers = km.cluster_centers_.astype(np.float32)
        self.log.info(f"{type(self).__name__}: CLUSTERED after {km.n_iter_} passes, inertia {km.inertia_:.6g}; "
                      f"ASSIGNING rows")
        labels, dist = [], []
        for s in scans:
            lab, d = _nearest_(s['vectors'], centers)
            labels.append(lab)
            dist.append(d)
        return centers, labels, dist, float(km.inertia_)

    def _cluster_distributed_(self, scans, k):
        """Lloyd's iterations, each pass's sums computed tablet by tablet by the workers: ``(centers, labels, dist, inertia)``."""
        from sklearn.cluster import kmeans_plusplus
        var, n = self.var, len(scans)
        sample = np.concatenate([s['sample'] for s in scans], axis=0)
        self.log.info(f"{type(self).__name__}: SEEDING {min(k, len(sample))} centers by k-means++ "
                      f"from {len(sample)} sampled rows")
        centers, _ = kmeans_plusplus(sample.astype(np.float64), n_clusters=min(k, len(sample)),
                                     random_state=var.seed)
        centers = centers.astype(np.float32)
        inertia, converged = None, False
        for it in range(var.max_iter):
            steps = self._executor_(f"CLUSTERING pass {it + 1}/{var.max_iter}").exec_callables(
                [TabletCoreStepCallable(self, i, centers) for i in range(n)])
            sums = sum(s['sums'] for s in steps)
            counts = sum(s['counts'] for s in steps)
            inertia = sum(s['inertia'] for s in steps)
            moved = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], centers).astype(np.float32)
            shift = float(np.sqrt(((moved - centers) ** 2).sum(axis=1)).max())
            scale = float(np.sqrt((centers ** 2).sum(axis=1)).mean()) or 1.0
            centers = moved
            converged = shift <= var.tol * scale
            self.log.info(f"{type(self).__name__}: CLUSTERING pass {it + 1}/{var.max_iter}: inertia {inertia:.6g}, "
                          f"{int((counts == 0).sum())} empty clusters, the centers moved {shift / scale:.3g} of "
                          f"their scale (tol {var.tol}){' -- converged' if converged else ''}")
            if converged:
                break
        if not converged:
            self.log.warning(f"{type(self).__name__}: k-means did not converge in max_iter={var.max_iter} passes; "
                             f"taking the centers as they are")
        self.log.info(f"{type(self).__name__}: ASSIGNING the rows to {len(centers)} centers")
        finals = self._executor_("ASSIGNING rows").exec_callables(
            [TabletCoreStepCallable(self, i, centers, final=True) for i in range(n)])
        return centers, [f['labels'] for f in finals], [f['dist'] for f in finals], inertia

    def _coverage_(self, scans, labels, layouts, centers) -> dict:
        """How tightly each fold, and all of them together, fill out the table: the ``coverage`` topic's arrays.

        Distances to the nearest row, measured from seeded samples, tablet by tablet
        -- in the workers, under ``'distributed'`` -- and only within each
        query's *coverage_clusters* nearest clusters: see the class docstring.
        """
        var, n_folds, k = self.var, len(layouts), len(centers)
        levels = np.asarray(COVERAGE_LEVELS)
        tablets = [s['idx'] for s in scans]
        valid = np.concatenate([np.stack([np.full(len(l), t), np.arange(len(l))], axis=1)[l >= 0]
                                for t, l in zip(tablets, labels)]).astype(np.int64)
        rng = np.random.default_rng([var.seed, 2])

        def pick(ids, n):
            return ids[np.sort(rng.choice(len(ids), size=min(n, len(ids)), replace=False))] if len(ids) else ids

        cores = [fl[:, :2] for fl in layouts]
        union = np.concatenate(cores) if cores else np.zeros((0, 2), dtype=np.int64)
        named = {'table': valid, 'core': union, 'base': pick(valid, len(union))}
        for f, core in enumerate(cores):
            named[f'core{f}'] = core
            named[f'base{f}'] = pick(valid, len(core))
        sets = {t: {name: ids[ids[:, 0] == t, 1] for name, ids in named.items()} for t in tablets}

        sample = pick(valid, var.coverage_sample)
        core_samples = [pick(core, var.coverage_sample) for core in cores]
        queries = [sample, *core_samples]
        vectors = self._vectors_of_(scans, np.concatenate(queries))
        bounds = np.cumsum([0] + [len(q) for q in queries])
        c = min(var.coverage_clusters, k)
        jobs = []
        for j, ids in enumerate(queries):
            q = vectors[bounds[j]:bounds[j + 1]]
            if len(q):
                d2 = (q.astype(np.float64) ** 2).sum(1)[:, None] - 2 * q.astype(np.float64) @ centers.T.astype(np.float64) \
                     + (centers.astype(np.float64) ** 2).sum(1)[None, :]
                near = np.argsort(d2, axis=1)[:, :c]
            else:
                near = np.zeros((0, c), dtype=np.int64)
            names, exclude = ((['table', 'core', 'base', *[f'core{f}' for f in range(n_folds)],
                                *[f'base{f}' for f in range(n_folds)]], ['table']) if j == 0
                              else ([f'core{j - 1}'], [f'core{j - 1}']))
            jobs.append((q, ids, near, names, exclude))
        by_tablet = {t: l for t, l in zip(tablets, labels)}
        if var.clustering == 'master':
            results = [TabletCoverageCallable(self, s['idx'], by_tablet[s['idx']], sets[s['idx']], jobs, vectors=s['vectors'])()
                       for s in scans]
        else:
            results = self._executor_("MEASURING coverage").exec_callables(
                [TabletCoverageCallable(self, t, by_tablet[t], sets[t], jobs) for t in tablets])
        nearest = [{name: np.min([r[j][name] for r in results], axis=0) if results else np.zeros(0)
                    for name in jobs[j][3]} for j in range(len(jobs))]

        def q(d):
            d = d[np.isfinite(d)] if np.isfinite(d).any() else d
            return np.quantile(d, levels) if len(d) else np.full(len(levels), np.nan)

        def mean(d):
            return float(d.mean()) if len(d) else np.nan

        order = [f'core{f}' for f in range(n_folds)] + ['core']
        based = [f'base{f}' for f in range(n_folds)] + ['base']
        table_rows = np.bincount(np.concatenate([l[l >= 0] for l in labels]), minlength=k)
        fold_rows = np.stack([np.bincount(fl[:, 2], minlength=k) for fl in layouts]) if n_folds else np.zeros((0, k))
        share = table_rows / max(table_rows.sum(), 1)
        tv = np.array([0.5 * np.abs(r / max(r.sum(), 1) - share).sum() for r in fold_rows])
        return {
            'levels': levels,
            'sample': sample,
            'scale': q(nearest[0]['table']),
            'coverage': np.stack([q(nearest[0][n]) for n in order]),
            'coverage_mean': np.array([mean(nearest[0][n]) for n in order]),
            'baseline': np.stack([q(nearest[0][n]) for n in based]),
            'baseline_mean': np.array([mean(nearest[0][n]) for n in based]),
            'separation': np.stack([q(nearest[j + 1][f'core{j}']) for j in range(n_folds)]),
            'cluster_rows': np.vstack([table_rows[None, :], fold_rows]).astype(np.int64),
            'cluster_tv': tv,
            'empty_clusters': np.array([int(((table_rows > 0) & (r == 0)).sum()) for r in fold_rows]),
        }

    def _vectors_of_(self, scans, ids: np.ndarray) -> np.ndarray:
        """The vectors of rows *ids*, ``(tablet, row)`` -- from the scans, or the workers' scratch."""
        width = next(s['width'] for s in scans if s['valid'].any())
        out = np.empty((len(ids), width), dtype=np.float32)
        tablets = np.unique(ids[:, 0]) if len(ids) else []
        if self.var.clustering == 'master':
            by_tablet = {s['idx']: s['vectors'] for s in scans}
            for t in tablets:
                at = ids[:, 0] == t
                out[at] = by_tablet[int(t)][ids[at, 1]]
            return out
        got = self._executor_("GATHERING sample vectors").exec_callables(
            [TabletVectorsCallable(self, int(t), ids[ids[:, 0] == t, 1]) for t in tablets])
        for t, vecs in zip(tablets, got):
            out[ids[:, 0] == t] = vecs
        return out

    def _deal_rows_(self, scans, labels, dist):
        """Units dealt nearest first, cell by cell, to their folds: ``(layouts, summary)``."""
        var = self.var
        fractions = [float(f) for f in var.fractions]
        tablet = np.concatenate([np.full(s['n'], s['idx'], dtype=np.int64) for s in scans])
        row = np.concatenate([np.arange(s['n'], dtype=np.int64) for s in scans])
        label, distance = np.concatenate(labels), np.concatenate(dist)
        keys = {role: (None if self.column_spec(role) is None
                       else [v for s in scans for v in s[role]]) for role in PARTITION_ROLES}
        keep = label >= 0
        skipped = {'no coreby value': int((~keep).sum())}
        for role, vals in keys.items():
            if vals is not None:
                missing = np.array([v is None for v in vals]) & keep
                skipped[f'no {role} value'] = int(missing.sum())
                keep &= ~missing
        if any(skipped.values()):
            self.log.warning(f"{type(self).__name__}: skipping rows that hold no value of a column it reads -- "
                             f"in no fold: {skipped}")
        at = np.flatnonzero(keep)
        stratum = (np.unique(np.array([keys['stratifyby'][i] for i in at], dtype=object).astype(str), return_inverse=True)[1]
                   if keys['stratifyby'] is not None else np.arange(len(at)))
        partition = (np.unique(np.array([keys['partitionby'][i] for i in at], dtype=object).astype(str), return_inverse=True)[1]
                     if keys['partitionby'] is not None else np.zeros(len(at), dtype=np.int64))
        lab, dst = label[at], distance[at]
        n_units = int(stratum.max()) + 1 if len(stratum) else 0
        weight = np.bincount(stratum, minlength=n_units)
        rank = np.bincount(stratum, weights=dst, minlength=n_units) / np.maximum(weight, 1)
        partition_of = np.full(n_units, -1, dtype=np.int64)
        for st, p in zip(stratum, partition):
            if partition_of[st] not in (-1, p):
                raise ValueError(f"{type(self).__name__}: a stratum spans more than one partition; a stratum is "
                                 f"dealt whole, so it must lie in one partition")
            partition_of[st] = p
        # A unit's cluster: the most frequent among its rows.
        pair = np.unique(np.stack([stratum, lab], axis=1), axis=0, return_counts=True)
        best = {}
        for (g, c), cnt in zip(*pair):
            if cnt > best.get(g, (-1, 0))[1] or (cnt == best[g][1] and c < best[g][0]):
                best[g] = (c, cnt)
        cluster = np.array([best[g][0] for g in range(n_units)], dtype=np.int64)
        tiebreak = np.random.default_rng(var.seed).random(n_units)
        fold_of = np.full(n_units, -1, dtype=np.int64)
        # Targets accumulate over a partition's cells, so what one cell rounds off
        # the next makes up: a partition's total is within half a unit of its share.
        partition_now = target = have = None
        for cell in sorted(set(zip(partition_of.tolist(), cluster.tolist()))):
            units = np.flatnonzero((partition_of == cell[0]) & (cluster == cell[1]))
            units = units[np.lexsort((tiebreak[units], rank[units]))]
            if cell[0] != partition_now:
                partition_now, target, have = cell[0], [0.0] * len(fractions), [0.0] * len(fractions)
            target = [t + f * weight[units].sum() for t, f in zip(target, fractions)]
            for u in units:
                w = weight[u]
                ok = [k for k in range(len(fractions)) if target[k] - have[k] >= w / 2]
                if not ok:
                    continue
                k = max(ok, key=lambda k: (target[k] - have[k]) / target[k])
                fold_of[u] = k
                have[k] += w
        row_fold = fold_of[stratum]
        layouts = []
        rng = np.random.default_rng([var.seed, 1])
        for k in range(len(fractions)):
            sel = at[row_fold == k]
            fl = np.stack([tablet[sel], row[sel], label[sel]], axis=1).astype(np.int64)
            fl = fl[np.lexsort((fl[:, 1], fl[:, 0]))] if not var.shuffle else fl[rng.permutation(len(fl))]
            layouts.append(fl)
        folds = []
        for k, fl in enumerate(layouts):
            units_k = np.flatnonzero(fold_of == k)
            folds.append({'fold': k, 'fraction': fractions[k], 'rows': int(len(fl)),
                          ('strata' if keys['stratifyby'] is not None else 'units'): int(len(units_k)),
                          'clusters': int(len(np.unique(fl[:, 2]))) if len(fl) else 0})
        summary = {'n_tablets': len(scans), 'n_rows': int(len(tablet)), 'n_dealt': int((row_fold >= 0).sum()),
                   'n_undealt': int((row_fold < 0).sum()), 'coreby': self.collator_pairs('coreby'),
                   'stratifyby': self.column_spec('stratifyby'), 'partitionby': self.column_spec('partitionby'),
                   'seed': var.seed, 'folds': folds, 'skipped': skipped}
        return layouts, summary
