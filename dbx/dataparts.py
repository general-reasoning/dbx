"""dataparts — partitions of a `Datatable` into parts: `DatatablePartition`, `DatatableCorePartition`, and their tabs.

A partition is an index: it reads the table's partition columns, deals its rows
to parts, and records which rows went where. A part is the table that reads that
record: a view of the partitioned table's own tabs, pieces of them, and packed
tabs -- or, for a core partition, core tabs of rows repacked by cluster. See
DATAPARTITION.md.
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
from .datatables import DATASLICE, Datacollator, DataslicesUpstream, Datatab, Datatable

#: How `DatatablePartition` deals tabs to folds -- see its docstring.
PARTITION_METHODS = ('random', 'largest_first')


def _column_values_(tab, spec) -> list:
    """Every row's value of *spec* -- ``(slice, column, key, ...)`` -- in *tab*; None where a row has none."""
    s_name, column, *keys = spec
    # Row by row: concatenated, a dict column comes back by key, and a row that is None has no keys.
    values = tab.data((s_name, column), concat=False)[s_name][column]
    out = []
    for value in values:
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        out.append(value)
    return out


def _value_key_(value) -> str:
    """A value as a sortable, hashable string: what two rows' values are compared by."""
    return json.dumps(value, sort_keys=True, default=str)


PARTITION_ROLES = ('stratifyby', 'partitionby')

#: The slice a repartitioned tab -- a `DatatabPiece`, a `DatatableCoreTab` -- points back
#: with: row i of it names the tab and the row of the partitioned table that row i of
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
                         f"selected by index, so row i must be row i of every slice. Rebuild the tab with its "
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
            f"{what}: no row of {source.tag} holds {values}: the piece cannot be recovered from its tab. Was "
            f"the tab rebuilt since it was partitioned? Build the partition again over the table as it is now.")
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
    """*source*'s slices, declared as it declares them, nested by their ``/`` -- and ``source``: a copied tab's TOPICS."""
    topics = {}
    for name in source.slices():
        *head, leaf = name.split('/')
        node = topics
        for h in head:
            node = node.setdefault(h, {})
        node[leaf] = source._topicnode_(*name.split('/'))
    if SOURCE_SLICE in topics:
        raise ValueError(f"{what}: {source.tag} has a slice named {SOURCE_SLICE!r}, the slice a copied tab "
                         f"points back to its rows with; rename the slice")
    return {**topics, SOURCE_SLICE: SOURCE}


def _source_index_(tab) -> np.ndarray:
    """``(tab, row)`` of the partitioned table for each of *tab*'s rows: its ``source`` slice, as an ``(N, 2)`` array."""
    src = tab.data(SOURCE_SLICE)[SOURCE_SLICE]
    return np.stack([np.asarray(src['tab'], dtype=np.int64), np.asarray(src['row'], dtype=np.int64)], axis=1)


def _piece_tag_(tag, values: dict) -> str:
    """A piece's tag: its source tab's, and the value of each role, ``'#'``-joined -- what a probe tells pieces apart by."""
    return '#'.join([str(tag)] + [v if isinstance(v, str) else _value_key_(v) for v in values.values()])


class TabPartitionScanCallable:
    """Worker callable: one tab's row count and the values each partition column holds in it.

    A tab holding one value per role reports that value. A tab holding several -- mixed --
    also reports its PIECES: each distinct combination of the role values, and how many rows
    hold it. Counts and not rows: a tab may hold millions of rows, and a piece re-selects its
    own from the value when it is built, so nothing row by row comes back to the partition.
    """

    def __init__(self, partition, idx: int):
        self.partition = partition
        self.idx = idx

    def __call__(self):
        part = self.partition
        tab = part.var.datatable.tab(self.idx)
        out = {'idx': self.idx, 'tag': tab.tag, 'rows': None, 'values': {}}
        if part.var.balance == 'rows' or part.var.method == 'largest_first' or getattr(part.var, 'coalesce', None) == 'pack':
            out['rows'] = int(tab.n_rows(part._partition_slice_name_))
        columns = {}
        for name in PARTITION_ROLES:
            spec = part.column_spec(name)
            if spec is None:
                continue
            columns[name] = _column_values_(tab, tuple(spec))
            distinct = {}
            for value in columns[name]:
                distinct.setdefault(_value_key_(value), value)
            out['values'][name] = list(distinct.values())
        if any(len(vals) > 1 for vals in out['values'].values()):
            out['column_rows'] = {name: len(vals) for name, vals in columns.items()}
            out['slice_rows'] = {s: int(tab.n_rows(s)) for s in tab.slices()}
            pieces = {}
            for row in zip(*columns.values()):
                values = dict(zip(columns, row))
                piece = pieces.setdefault(_value_key_(values), {'values': values, 'rows': 0})
                piece['rows'] += 1
            out['pieces'] = [pieces[k] for k in sorted(pieces)]
        return out


#: The values of the fields `DatatablePartition` gained with *coalesce* at which it computes what it
#: did before them: cut tabs stay pieces, and nothing is packed.
_UNCOALESCED_ = {'coalesce': None, 'min_tab_rows': None, 'max_tab_rows': None}

#: Where `DatatablePartition` and its blocks were stored before they moved here, on 2026-10-09.
_DATATABLES_ = 'dbx.datatables'

COALESCE_MODES = (None, 'whole', 'pack')


class DatatablePartition(Datablock):
    """Partitions a `Datatable`'s rows into parts according to target *fractions*.

    Tabs are dealt to parts; part *k*, `part(k)`, a `DatatablePart`, is a table
    of them -- the table's own tabs where they are, and the tabs this partition
    makes of them (see *coalesce*). The choices that decide the deal each answer
    one question:

    The columns it reads are a *collator*'s: a `Datacollator` whose
    ``columns`` hold one column spec, ``(slice, column[, key, ...])``, for
    each of the roles

    *stratifyby* -- which rows must land in the SAME part? Rows sharing its
    value, a stratum (a patient's, say), are dealt as one unit, across all the
    tabs they lie in. Default: each tab alone.

    *partitionby* -- within which categories must the fractions hold? The
    fractions are met within each of its values, a partition (each cancer
    type, say), not only overall. Default: overall only.

    A column upstream of *datatable* -- a feature table's samples' -- is read
    only with the collator's ``recursive=True``. No collator: neither role.

    *balance* -- fractions of what? ``'rows'`` (the default): of rows, read
    from *partition_slice*. ``'tabs'``: of tabs.

    *seed* -- the order units are dealt in. Within each partition, the units
    are shuffled with it, then each goes to the part furthest below its target.

    A tab whose *stratifyby* and *partitionby* columns hold one value throughout is
    dealt whole. A tab holding several -- several patients in one bag, tiles
    annotated one by one -- is cut: into PIECES, one per combination of its
    values, each a `DatatabPiece` -- the rows of that tab holding those values.
    A piece is dealt as a tab is: with every tab and piece of its stratum,
    within its partition.

    *coalesce* -- what a part's tabs are, once the deal is done. It never moves
    a row to another part.

    - None (the default): the dealt entries as they are -- whole tabs, and
      pieces of the cut ones. What a partition was before *coalesce*.
    - ``'whole'``: a cut tab whose pieces all went to one part is that part's
      whole again -- the table's own tab, read where it is, rather than pieces
      copied out of it. A tab of which a piece was skipped stays cut: whole, it
      would bring the skipped rows back.
    - ``'pack'``: as ``'whole'``, and then a part's entries of fewer than
      *min_tab_rows* rows -- tabs and pieces both -- are pooled, in source
      order, into packed tabs of *max_tab_rows* rows (twice *min_tab_rows* by
      default), each a `DatatablePackTab`. An entry may run across two packed
      tabs; the last one of a part holds what is left. Larger entries stay as
      they are, so nothing is copied that is not small.

    A tab of a part may hold rows of several strata and partitions -- a rejoined
    tab, a packed one -- so a reader that needs the rows of one slide, say,
    takes them by a column, not by tab: a probe's ``aggregateby`` -- see
    `FeatureAffineLogisticProbe`.

    A tab or a piece holding no value is skipped: in no part, said loudly, and
    recorded in ``summary`` -- a piece with the rows it leaves out. A stratum
    must lie in one partition. A table that reads slices upstream
    (`DataslicesUpstream`, a `Featuretable`) is neither cut nor packed: a piece
    of it would have to be the same piece of its upstream too, which is not
    implemented, and a mixed tab of one, or ``coalesce='pack'``, raises
    `NotImplementedError`.

    *method* ``'largest_first'`` is the partition from before these choices:
    tabs in descending order of rows, each to the part with the largest deficit
    -- no strata, no partitions, no seed. Its builds are reached by a
    specialization, and it takes none of the choices above.

    Topics: ``tabs`` -- one list of entries per part, which the parts read: a
    tab index; a piece's record, ``{"tab": i, "stratifyby": value, "partitionby":
    value}`` with the roles the collator sets, its values as read; and, with
    ``coalesce='pack'``, a packed tab's record, ``{"fold": k, "pack": j,
    "segments": [[entry, offset, length], ...]}`` -- see `tabs_indices`;
    ``summary`` -- per part its tags and counts (tabs, rows, per partition), the
    tabs and pieces skipped and why, and what was rejoined and packed.

    The parts build the pieces and the packed tabs: ``partition.part(k).build()``
    builds part k's, in parallel, as a table builds its tabs, and the whole tabs
    among them are already built. A probe reading a part whose tabs are not
    built says so before it starts. The partition itself stays an index: it
    writes no rows.
    """

    # 2: the collator's roles were renamed on 2026-10-10 -- stratifyby is partitionby, and what was
    # partitionby is partitionby. A collator naming partitionby hashes as it did, and means the
    # opposite, so the version keeps every partition built before from being taken for one now.
    VERSION = 2

    TOPICS = {
        'tabs': DATAFILE('tabs.json', 'one list of tab indices per fold'),
        'summary': DATAFILE('summary.json', 'per fold its tags and counts, and the tabs skipped'),
    }
    # Only the partitions with no collator are reached: no role names in them to have moved.
    SPECIALIZATIONS = [
        *(Datablock.Specialization(
            spec={'method': 'largest_first', 'seed': 0, 'collator': None, 'balance': 'rows', **_UNCOALESCED_},
            topics={'tabs': DATAFILE('tabs.json', 'one list of tab indices per fold')}, version=1,
            redirect_vars={'datatable': 'datapoint_table'}, anchor=anchor,
            note="the partition from before method/seed/groupby/stratifyby/balance: largest_first, by rows")
          for anchor in (SAME, f'{_DATATABLES_}.DatatablePartition')),
        *(Datablock.Specialization(
            spec={'method': 'largest_first', 'seed': 0, 'collator': None, 'balance': 'rows', **_UNCOALESCED_},
            topics={'tabs': 'tabs.json'}, version=1,
            redirect_vars={'datatable': 'datapoint_table'}, anchor=anchor,
            note="... and from before its topic was respelled DATAFILE")
          for anchor in (SAME, f'{_DATATABLES_}.DatatablePartition')),
    ]

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
        min_tab_rows: int | None = None
        max_tab_rows: int | None = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        var = self.var
        if var.method not in PARTITION_METHODS:
            raise ValueError(f"{type(self).__name__}: unknown method {var.method!r}; expected one of {PARTITION_METHODS}")
        if var.balance not in ('rows', 'tabs'):
            raise ValueError(f"{type(self).__name__}: balance must be 'rows' or 'tabs', got {var.balance!r}")
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
            raise ValueError(f"{what}: method='largest_first' cuts no tab, so there is nothing to coalesce")
        if var.coalesce == 'pack':
            if not (isinstance(var.min_tab_rows, int) and var.min_tab_rows >= 1):
                raise ValueError(f"{what}: coalesce='pack' pools the entries of fewer than min_tab_rows rows, "
                                 f"which must be a positive int, got {var.min_tab_rows!r}")
            if var.max_tab_rows is not None and not (isinstance(var.max_tab_rows, int)
                                                     and var.max_tab_rows >= var.min_tab_rows):
                raise ValueError(f"{what}: max_tab_rows, a packed tab's rows, must be an int of at least "
                                 f"min_tab_rows={var.min_tab_rows}, got {var.max_tab_rows!r}")
        elif var.min_tab_rows is not None or var.max_tab_rows is not None:
            raise ValueError(f"{what}: min_tab_rows and max_tab_rows size packed tabs, and are for "
                             f"coalesce='pack' only, not coalesce={var.coalesce!r}")

    def __build__(self):
        table = self.var.datatable
        if table is None:
            raise ValueError(f"{self.__class__.__name__}: VAR.datatable is required")
        if self.var.collator is not None:
            self.var.collator.slices(table)       # raises for a column the table does not hold
        if not self.var.fractions:
            raise ValueError(f"{self.__class__.__name__}: VAR.fractions is required")
        if self.var.coalesce == 'pack' and isinstance(table, DataslicesUpstream):
            raise NotImplementedError(
                f"{type(self).__name__}: coalesce='pack' copies rows into packed tabs, and a {type(table).__name__} "
                f"reads slices upstream: a packed tab of it would have to pack its upstream rows too. Partition "
                f"the upstream table, whose tabs hold the rows.")
        scans = self._scan_()
        if self.valid_topic('tabs'):
            # Adopted -- a build from before the summary existed: its folds stand.
            folds_tabs, skipped = json.loads(self.fs.cat(self.path('tabs'))), []
        else:
            folds_tabs, skipped = ((self._largest_first_(scans), []) if self.var.method == 'largest_first'
                                   else self._deal_(scans))
            if self.var.coalesce is not None:
                folds_tabs = self._rejoin_(scans, folds_tabs)
            if self.var.coalesce == 'pack':
                folds_tabs = self._pack_(scans, folds_tabs)
            with self.fs.open(self.path('tabs', ensure_dirpath=True), 'w') as f:
                json.dump(folds_tabs, f)
        with self.fs.open(self.path('summary', ensure_dirpath=True), 'w') as f:
            json.dump(self._summary_(scans, folds_tabs, skipped), f, indent=1, default=str)

    def __read__(self, *topicpath):
        topicpath = self._normtopic_(topicpath)
        if topicpath in (('tabs',), ('summary',)):
            return json.loads(self.fs.cat(self.path(topicpath[0])))
        return super().__read__(*topicpath)

    # 2. Declared API ------------------------------------------------------

    def n_folds(self) -> int:
        return len(self.var.fractions)

    def tabs_indices(self, fold: int | str) -> list:
        """Part *fold*'s entries: a table's tab index; for a piece of a mixed tab, its record; for a packed tab, its.

        A piece's record is ``{"tab": i, <role>: value, ...}``: the source tab,
        and the value of each role its rows hold, as read. A packed tab's is
        ``{"fold": k, "pack": j, "segments": [[entry, offset, length], ...]}``:
        rows *offset* to *offset + length* of each entry's, in order. Ints only,
        as ever, when no tab was cut. `tab` turns any of them into the block it
        names.
        """
        data = json.loads(self.fs.cat(self.path('tabs')))
        return data[int(fold)]

    def tabs(self, fold: int | str) -> list[Datatab]:
        return [self.tab(entry) for entry in self.tabs_indices(fold)]

    def tab(self, entry: 'int | dict') -> Datatab:
        """The block an entry of ``tabs`` names: the table's tab, the `DatatabPiece` of it, or a `DatatablePackTab`."""
        table = self.var.datatable
        if not isinstance(entry, dict):
            return table.tab(int(entry))
        if 'pack' in entry:
            fold, index = int(entry['fold']), int(entry['pack'])
            return DatatablePackTab(
                datalake=self._datalake_,
                storage_options=self.storage_options,
                verbose=False,
                tag=f"pack{self.hash[:8]}_f{fold}_{index:06d}",
                spec=dict(datapoints_per_row=getattr(table.var, 'datapoints_per_row', 1),
                          partition=self, fold=fold, index=index),
            )
        source = table.tab(int(entry['tab']))
        values = {role: entry[role] for role in PARTITION_ROLES if role in entry}
        return DatatabPiece(
            # Where the partition is, as its folds are: a piece is derived from
            # the partition, and the table's lake may be one nothing writes to.
            datalake=self._datalake_,
            storage_options=self.storage_options,
            cache=getattr(source, 'cache', None),
            cache_limit=getattr(source, 'cache_limit', None),
            verbose=False,
            tag=_piece_tag_(source.tag, values),
            spec=dict(datapoints_per_row=source.var.datapoints_per_row, tab=source, tab_index=int(entry['tab']),
                      columns={role: tuple(self.column_spec(role)) for role in values}, values=values),
        )

    def part(self, fold: int | str) -> DatatablePart:
        """Fold *fold* as a table: the `DatatablePart` holding its tabs and pieces.

        A fold is an index into ``tabs``; its part is the block that reads it.
        """
        return DatatablePart(
            # As a table gives its tabs its url: a fold of a partition belongs
            # where the partition does, not wherever DBX_ROOT happens to point
            # in the process that asks for it.
            datalake=self._datalake_,
            storage_options=self.storage_options,
            spec=dict(
                partition=self,
                fold=fold,
            )
        )

    def fold(self, fold: int | str) -> DatatablePart:
        """The name `part` had first."""
        return self.part(fold)

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

    # 4. Helpers -----------------------------------------------------------

    def _scan_(self) -> list[dict]:
        """Each tab's rows and partition-column values, read in parallel; a mixed tab's pieces too.

        A mixed tab is checked here, before anything is dealt: its pieces are
        selected by row, so every slice of it must have as many rows as its
        partition columns have values -- or row i of one slice is not row i of
        another, and a piece would pair one row's columns with another's.
        """
        table = self.var.datatable
        executor = callable_executor(
            getattr(self, 'parallelization', None) or 'inline',
            n_workers=getattr(self, 'n_workers', 1) or 1,
            tag=f"SCANNING {table.n_tabs} tabs [{type(self).__name__}]")
        scans = executor.exec_callables([TabPartitionScanCallable(self, i) for i in range(table.n_tabs)])
        mixed = [s for s in scans if 'pieces' in s]
        if not mixed:
            return scans
        shown = "\n".join(f"  {s['tag']}: " + ", ".join(f"{name} holds {len(vals)} values, e.g. {vals[:3]!r}"
                                                         for name, vals in s['values'].items() if len(vals) > 1)
                          for s in mixed[:5])
        if isinstance(table, DataslicesUpstream):
            raise NotImplementedError(
                f"{type(self).__name__}: {len(mixed)} tab(s) of {type(table).__name__} hold more than one value "
                f"of a partition column:\n{shown}\n"
                f"A mixed tab is split into pieces, one per value, and that is implemented for a plain Datatab "
                f"table only: a piece of a table that reads slices upstream ({type(table).__name__} is a "
                f"DataslicesUpstream) would have to be the same piece of its upstream tab too. Partition the "
                f"upstream table, whose tabs hold the rows, or give this one tabs that each hold one value.")
        uneven = [s for s in mixed if len(set(s['column_rows'].values()) | set(s['slice_rows'].values())) > 1]
        if uneven:
            s = uneven[0]
            raise ValueError(
                f"{type(self).__name__}: {len(uneven)} mixed tab(s) have slices or partition columns of "
                f"different lengths, e.g. {s['tag']}: slices {s['slice_rows']}, partition columns "
                f"{s['column_rows']}. A mixed tab is split by row, so row i must be row i of every slice; "
                f"rebuild the tab with its slices written in lockstep.")
        n_pieces = sum(len(s['pieces']) for s in mixed)
        self.log.warning(
            f"{type(self).__name__}: splitting {len(mixed)} of {len(scans)} tabs, which hold more than one "
            f"value of a partition column, into {n_pieces} pieces, one per value -- dealt as tabs are, and "
            f"built by the parts (part.build()):\n{shown}" + ("\n  ..." if len(mixed) > 5 else ""))
        return scans

    def _items_(self, scans) -> list[dict]:
        """What is dealt, in scan order: each constant tab, and each piece of a mixed one.

        An item's *entry* is what ``tabs`` records for it -- a tab index, or a
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
        """An entry of ``tabs`` as a dict key: a tab index is itself, a piece its record's `_value_key_`."""
        return _value_key_(entry) if isinstance(entry, dict) else entry

    @staticmethod
    def _entry_order_(entry) -> tuple:
        """The order a fold lists its entries in: by source tab, a tab's pieces by value -- ints alone as sorted()."""
        return (entry['tab'], _value_key_(entry)) if isinstance(entry, dict) else (entry, '')

    def _largest_first_(self, scans) -> list[list[int]]:
        """The partition from before: tabs by descending rows, each to the fold with the largest deficit."""
        fractions = self.var.fractions
        tab_rows = [s['rows'] for s in scans]
        total = sum(tab_rows)
        norm = [f / sum(fractions) for f in fractions]
        target = [total * f for f in norm]
        fold_rows = [0.0] * len(fractions)
        folds_tabs = [[] for _ in fractions]
        for t_idx in sorted(range(len(tab_rows)), key=lambda i: tab_rows[i], reverse=True):
            best = max(range(len(fractions)), key=lambda k: target[k] - fold_rows[k])
            folds_tabs[best].append(t_idx)
            fold_rows[best] += tab_rows[t_idx]
        return [sorted(f) for f in folds_tabs]

    def _deal_(self, scans) -> tuple[list[list], list[dict]]:
        """Units -- strata of tabs and pieces -- dealt within each partition, in seeded order, to the fold furthest below target.

        A piece is dealt exactly as a tab is, weighing its rows or 1: a stratum's
        unit collects every tab and piece holding its value, whichever tabs the
        pieces came from. Without a stratifyby, each tab is its own unit, and so
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
            unit = units.setdefault(stratum, {'tabs': [], 'partitions': set(), 'weight': 0})
            unit['tabs'].append(item['entry'])
            unit['partitions'].add(partition)
            unit['weight'] += item['rows'] if self.var.balance == 'rows' else 1
        if skipped:
            n_rows = sum(s['rows'] for s in skipped if 'piece' in s)
            self.log.warning(
                f"{type(self).__name__}: skipping {len(skipped)} of {len(items)} "
                f"{'tabs' if len(items) == len(scans) else 'tabs and pieces'}, which hold no "
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
        folds_tabs = [[] for _ in fractions]
        for partition in sorted(by_partition):
            strata = by_partition[partition]
            total = sum(units[st]['weight'] for st in strata)
            target = [total * f for f in fractions]
            have = [0.0] * len(fractions)
            for i in rng.permutation(len(strata)):
                unit = units[strata[i]]
                k = max(range(len(fractions)),
                        key=lambda k: (target[k] - have[k]) / target[k] if target[k] > 0 else -np.inf)
                folds_tabs[k].extend(unit['tabs'])
                have[k] += unit['weight']
        return [sorted(f, key=self._entry_order_) for f in folds_tabs], skipped

    def _rejoin_(self, scans, folds_tabs) -> list[list]:
        """Each cut tab whose pieces all went to one part made whole there again.

        Whole, a tab is read where it is, and nothing is copied out of it. A tab
        of which a piece is in no part -- skipped, for holding no value -- stays
        cut: whole, it would bring those rows back.
        """
        cut = {s['idx']: s for s in scans if 'pieces' in s}
        where = {}
        for k, entries in enumerate(folds_tabs):
            for e in entries:
                if isinstance(e, dict):
                    where.setdefault(e['tab'], []).append(k)
        whole = {idx for idx, parts in where.items()
                 if len(set(parts)) == 1 and len(parts) == len(cut[idx]['pieces'])}
        return [sorted([e for e in entries if not (isinstance(e, dict) and e['tab'] in whole)]
                       + sorted({e['tab'] for e in entries if isinstance(e, dict) and e['tab'] in whole}),
                       key=self._entry_order_)
                for entries in folds_tabs]

    def _pack_(self, scans, folds_tabs) -> list[list]:
        """Each part's entries of fewer than *min_tab_rows* rows pooled, in source order, into packed tabs' records.

        A packed tab takes *max_tab_rows* rows -- twice *min_tab_rows* unless
        given -- as segments, ``[entry, offset, length]``, and an entry may run
        across two of them. The last packed tab of a part holds what is left.
        Entries of at least *min_tab_rows* rows stay as they are: a whole tab is
        read where it is, a piece is a copy already.
        """
        rows = self._entry_rows_(scans)
        least = self.var.min_tab_rows
        most = self.var.max_tab_rows or 2 * least
        out = []
        for k, entries in enumerate(folds_tabs):
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
        """Every entry a deal can hold -- each tab, whole, and each piece -- by `_entry_key_`: its rows."""
        rows = {}
        for s in scans:
            pieces = s.get('pieces', ())
            rows[s['idx']] = sum(p['rows'] for p in pieces) if pieces else s['rows']
            for p in pieces:
                rows[self._entry_key_({'tab': s['idx'], **p['values']})] = p['rows']
        return rows

    def _summary_(self, scans, folds_tabs, skipped) -> dict:
        """What each part holds -- tags, tabs, rows, per partition -- what was left out, and what was rejoined and packed.

        A piece counts as a tab -- it is one of its part's -- with its own rows.
        ``pieces`` and ``n_pieces`` say how many, and appear only when a tab was
        cut, so the summary of a partition of constant tabs is what it was. A
        packed tab's entries count as the entries they are, and ``packed`` says
        how many packed tabs hold them; a rejoined tab counts as the whole tab.
        """
        items = self._items_(scans)
        by_entry = {self._entry_key_(item['entry']): [item] for item in items}
        for s in scans:
            if 'pieces' in s:      # a cut tab made whole by coalesce: its pieces, each in its partition
                by_entry[s['idx']] = [by_entry[self._entry_key_({'tab': s['idx'], **p['values']})][0]
                                      for p in s['pieces']]
        tag_of = {s['idx']: s['tag'] for s in scans}
        split = any('pieces' in s for s in scans)
        packing = getattr(self.var, 'coalesce', None) == 'pack'

        folds = []
        for k, fold_entries in enumerate(folds_tabs):
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
            whole = {e for entries in folds_tabs for e in entries if not isinstance(e, dict)}
            whole |= {seg[0] for entries in folds_tabs for e in entries if isinstance(e, dict) and 'pack' in e
                      for seg in e['segments'] if not isinstance(seg[0], dict)}
            coalesced = {'coalesce': self.var.coalesce,
                         'rejoined': [s['tag'] for s in scans if 'pieces' in s and s['idx'] in whole]}
        return {'method': self.var.method, 'balance': self.var.balance, 'seed': self.var.seed,
                'stratifyby': self.column_spec('stratifyby'), 'partitionby': self.column_spec('partitionby'),
                'n_tabs': len(scans), **({'n_pieces': sum(len(s['pieces']) for s in scans if 'pieces' in s)} if split else {}),
                **coalesced, 'folds': folds, 'skipped': skipped}


class DatatablePart(Datatable):
    """Part *fold* of a `DatatablePartition`: a table of the tabs dealt to it.

    Its tabs are its table's own, pieces of them -- a `DatatabPiece` for each
    entry of ``tab_indices`` that is a piece's record -- and, with
    ``coalesce='pack'``, packed tabs, a `DatatablePackTab` for each packed
    tab's record. The table's tabs are built where they are; the pieces and the
    packed tabs are this part's to build, and its build() builds them, in
    parallel, as any table builds its tabs, skipping the ones already built.
    """

    # None: a part is its partition's, and no partition with a collator is reached from before
    # the roles were renamed.
    SPECIALIZATIONS = []

    @dataclass
    class VAR(Datablock.VAR):
        partition: DatatablePartition
        fold: int

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, *args, filter_built_tabs: bool = True, **kwargs):
        # True by default: a part's tabs are its table's, so the ones the table
        # has built are checked here, in the parent, and no callable is sent
        # to form and check each one again in a worker.
        super().__init__(*args, filter_built_tabs=filter_built_tabs, **kwargs)

    def __tab__(self, idx: int, *, tag=None, **spec) -> Datatab:
        return self.tab(idx)

    def __block__(self, idx: int) -> Datatab:
        return self.tab(idx)

    # 2. Declared API ------------------------------------------------------

    def slices(self, recursive: bool = False):
        return self.datatable.slices(recursive=recursive)

    def tab(self, idx: int) -> Datatab:
        """Tab *idx* of this part: its table's tab, or the `DatatabPiece` or `DatatablePackTab` its entry records."""
        entry = self.tab_indices[idx]
        if isinstance(entry, dict):
            return self.var.partition.tab(entry)
        return self.datatable.tab(entry)

    def valid_tab(self, idx: int, validation: str | None = None) -> bool:
        """Whether tab *idx* is valid: this part's manifest, or its table's tab, or the piece -- see `Datastack.valid_blocks`."""
        validation = self._validation_(validation)
        if validation == 'cross_check' and self._blocks_cross_checked_():
            return True
        entry = self.tab_indices[idx]
        if isinstance(entry, dict):
            piece = self.tab(idx)
            return bool(piece.validate() if validation == 'validate' else piece.valid())
        full = 'validate' if validation == 'validate' else 'valid'
        return self.datatable.valid_tab(entry, validation=full)

    valid_block = valid_tab

    def redirected_tab(self, idx: int) -> bool:
        entry = self.tab_indices[idx]
        if isinstance(entry, dict):
            return self.tab(idx).redirected()
        return self.datatable.redirected_tab(entry)

    redirected_block = redirected_tab

    def validate_tab(self, idx: int, **kwargs) -> bool:
        entry = self.tab_indices[idx]
        if isinstance(entry, dict):
            return Datastack.validate_block(self, idx, **kwargs)
        return self.datatable.validate_tab(entry, **kwargs)

    validate_block = validate_tab

    def datastream(self, slice, **kwargs) -> StreamingDataset:
        self.slice_names((slice,))
        cacheroot = self._ensure_cacheroot_(kwargs.pop('cache', None))
        cache_dir = kwargs.pop('cache_dir',
                               f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        local = os.path.join(cacheroot, cache_dir)
        os.makedirs(local, exist_ok=True)
        streams = self._tab_streams_(slice, local)
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
            streaming_kwargs['streams'] = self._tab_streams_(slice, local)
            return release_shared_memory_when_collected(StreamingDataset(**streaming_kwargs))

    def source_index(self) -> np.ndarray:
        """``(tab, row)`` of the partitioned table for each of this part's rows, tab by tab -- an ``(N, 2)`` array.

        Which rows of the table it holds, however they are kept: a whole tab is
        all of that tab's rows, a piece or a packed tab what its ``source``
        slice names -- so it must be built.
        """
        out, rows_slice = [], self.var.partition._partition_slice_name_
        for idx, entry in enumerate(self.tab_indices):
            if isinstance(entry, dict):
                out.append(self.tab(idx).source_index())
            else:
                n = int(self.datatable.tab(entry).n_rows(rows_slice))
                out.append(np.stack([np.full(n, int(entry), dtype=np.int64), np.arange(n, dtype=np.int64)], axis=1))
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)

    # 3. Accessors ---------------------------------------------------------

    @functools.cached_property
    def tab_indices(self) -> list:
        """The partition's entries for this fold: tab indices, and the records of pieces -- see `DatatablePartition.tabs_indices`."""
        return self.var.partition.tabs_indices(self.var.fold)

    @property
    def datatable(self) -> Datatable:
        return self.var.partition.datatable

    @property
    def datapoints_per_row(self) -> int:
        return getattr(self.datatable.var, 'datapoints_per_row')

    @forward_property(Datatab)
    def TAB(self):
        """The partitioned table's TAB: on the class, `Datatab` -- what every part's tabs are."""
        return getattr(self.datatable, 'TAB', None)

    @forward_property({})
    def TOPICS(self):
        """The partitioned table's TOPICS: a part is a view of its tabs, and declares nothing of its own.

        On the class, with no table to ask, ``{}``.
        """
        return self.datatable.TOPICS

    @property
    def n_tabs(self) -> int:
        return len(self.tab_indices)

    # 4. Helpers -----------------------------------------------------------


    def _read_slice_(self, slice, *, tabs=None, **kwargs):
        if tabs is None:
            indices = range(self.n_tabs)
        else:
            indices = tabs
        datapoints = []
        for idx in indices:
            datapoints.extend(self.tab(idx)._read_slice_(slice, **kwargs))
        return datapoints

    def _blocks_datalake_(self):
        """A part's tabs are its table's, and are stored where they are."""
        return self.datatable._blocks_datalake_()

    def _form_block_(self, idx: int, *, use_specializations='stack'):
        """As `Datastack._form_block_`, and a piece passes for the TAB it is a piece of, a packed tab for the TAB it packs.

        A part's BLOCK is its table's TAB, and is in its hash, so it stays that
        for a part holding pieces and packed tabs; a piece is a `DatatabPiece`
        of a tab of that class, and is checked as one. A packed tab packs the
        table's own tabs, which are of that class by the table's own check.
        """
        if not isinstance(self.tab_indices[idx], dict):
            return super()._form_block_(idx, use_specializations=use_specializations)
        with self._block_specializations_in_force_(use_specializations):
            piece = self._adopt_(self.__block__(idx), keyby=True)
        if isinstance(piece, DatatablePackTab):
            return piece
        declared = self._block_class_()
        if declared is not None and not isinstance(piece.var.tab, declared):
            raise TypeError(
                f"{self.__class__.__name__}.block({idx}) is a piece of a {type(piece.var.tab).__name__}, "
                f"not of the {declared.__name__} its BLOCK declares"
            )
        return piece

    def _block_class_(self):
        """A part's tabs are its table's, and so is its BLOCK."""
        table = getattr(getattr(self, 'var', None), 'partition', None)
        table = getattr(table, 'datatable', None)
        return table._block_class_() if table is not None else None

class DatatabPiece(Datatab):
    """The rows of one tab holding one value of each partition column: a piece of a mixed tab.

    What a `DatatablePartition` deals in place of a tab whose *stratifyby* or
    *partitionby* column holds several values. A piece is defined by its value,
    not by a list of rows: *tab* is the source tab, *columns* the column spec
    of each role, ``{role: (slice, column[, key, ...])}``, and *values* the
    value each role holds in the piece's rows, as read. So its identity is
    readable -- these rows of that tab -- stable whatever the deal, and small,
    however many rows it holds; its build re-reads the columns and selects the
    rows itself.

    Its slices are the source tab's, declared as the source declares them, and
    it writes every one of them with the same rows, in the same order, so row
    i of each is still row i of the others. Only the slices: a topic of the
    source that is not a slice is the tab's, not of any subset of its rows.

    Its ``source`` slice records which rows those were: row i of it is the
    source tab's index in the partitioned table, and the row of it that row i
    of every other slice was copied from. The value says which rows a piece
    holds; the indices say it in a form another tab can apply -- a tab built
    from the source row by row (a feature tab over a sample tab) takes the
    same piece of itself by taking the same rows, without reading any
    partition column, and a check that two tabs still agree has something to
    compare.

    A piece of a tab that reads slices upstream (`DataslicesUpstream`, a
    `Featuretab`) is not implemented: it would have to be the same piece of
    its upstream tab too, and nothing says row i of the one is row i of the
    other (see the TODO on `Featuretab` about ``shared_upstream_column``).
    """

    SPECIALIZATIONS = [Datablock.Specialization(
        spec={}, topics=SAME, anchor=f'{_DATATABLES_}.DatatabPiece',
        note="this very block, stored under dbx.datatables until 2026-10-09")]

    @dataclass
    class VAR(Datatab.VAR):
        tab: Datatab = None
        tab_index: int = None
        columns: dict = None
        values: dict = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        var = self.var
        if var.tab is None or var.tab_index is None or not var.columns or var.values is None:
            raise ValueError(f"{type(self).__name__}: VAR.tab, VAR.tab_index, VAR.columns and VAR.values are required")
        if sorted(var.columns) != sorted(var.values):
            raise ValueError(f"{type(self).__name__}: columns name the roles {sorted(var.columns)}, "
                             f"values {sorted(var.values)}: each role's value is read from its column")
        if isinstance(var.tab, DataslicesUpstream):
            raise NotImplementedError(
                f"{type(self).__name__}: a piece of a {type(var.tab).__name__}, which reads slices upstream, "
                f"is not implemented -- it would have to be the same piece of its upstream tab too. Split the "
                f"upstream tab, which holds the rows.")

    def __build__(self):
        source, what = self.var.tab, type(self).__name__
        rows = _selected_rows_(source, self.var.columns, self.var.values, what)
        copied = [s for s in self.slices() if s != SOURCE_SLICE]
        columns = {s: self.declared_columns(s) or _written_columns_(source, s, what) for s in copied}
        # A slice at a time, so only one is held whole; the writers count, and refuse
        # a build that wrote the slices unequally.
        with self.slice_writers({**columns, SOURCE_SLICE: SOURCE.columns}) as writers:
            for s in copied:
                data = source._read_slice_(s)
                for i in rows:
                    writers[s].write(data[i])
                del data
            tab = int(self.var.tab_index)
            for i in rows:
                writers[SOURCE_SLICE].write({'tab': tab, 'row': int(i)})

    # 2. Declared API ------------------------------------------------------

    def source_rows(self) -> np.ndarray:
        """The rows of the source tab this piece holds, in the order its slices hold them -- its ``source`` slice's."""
        return np.asarray(self.data((SOURCE_SLICE, 'row'))[SOURCE_SLICE]['row'], dtype=np.int64)

    def source_index(self) -> np.ndarray:
        """``(tab, row)`` of the partitioned table for each of this piece's rows -- its ``source`` slice."""
        return _source_index_(self)

    # 3. Accessors ---------------------------------------------------------

    @forward_property({SOURCE_SLICE: SOURCE})
    def TOPICS(self):
        """The source tab's slices, declared as it declares them, and ``source`` -- nothing else of its topics.

        On the class, with no tab to ask, ``source`` alone.
        """
        return _slice_topics_(self.var.tab, type(self).__name__)


class DatatablePackTab(Datatab):
    """Rows of several of a part's entries, pooled into one tab: what ``coalesce='pack'`` makes of small ones.

    A `DatatablePartition` with ``coalesce='pack'`` pools each part's entries
    of fewer than *min_tab_rows* rows -- whole tabs and pieces -- into packed
    tabs, so that a part is not a great many small tabs. A packed tab's rows
    are its record's segments, ``[entry, offset, length]``, in order: rows
    *offset* to *offset + length* of the entry's, which are a whole tab's rows,
    or a piece's, selected by its value as a `DatatabPiece` selects them.

    Its slices are the partitioned table's, declared as its tabs declare them,
    and it writes every one with the same rows in the same order; ``source``
    names the tab and the row of the partitioned table each came from.

    Its identity is its place -- *partition*, *fold*, *index* -- as a core
    tab's is, since what it holds is what that partition's record says that
    place holds.

    A packed tab holds rows of several source tabs, and so, typically, of
    several slides: a probe takes the rows of one by a column, its
    ``aggregateby``.
    """

    SPECIALIZATIONS = []

    @dataclass
    class VAR(Datatab.VAR):
        partition: DatatablePartition = None
        fold: int = 0
        index: int = 0

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.var.partition, DatatablePartition):
            raise ValueError(f"{type(self).__name__}: VAR.partition must be a DatatablePartition, "
                             f"got {type(self.var.partition).__name__}")

    def __build__(self):
        what = type(self).__name__
        segments = self._segment_rows_()
        copied = [s for s in self.slices() if s != SOURCE_SLICE]
        first = segments[0][1]
        columns = {s: self.declared_columns(s) or _written_columns_(first, s, what) for s in copied}
        # A slice at a time, and within it a segment at a time, so only one
        # source slice is held whole; the writers refuse unequal slices.
        with self.slice_writers({**columns, SOURCE_SLICE: SOURCE.columns}) as writers:
            for s in copied:
                for _, source, rows in segments:
                    data = source._read_slice_(s)
                    for i in rows:
                        writers[s].write(data[i])
                    del data
            for t, _, rows in segments:
                for i in rows:
                    writers[SOURCE_SLICE].write({'tab': int(t), 'row': int(i)})

    # 2. Declared API ------------------------------------------------------

    def segments(self) -> list:
        """This packed tab's segments, ``[entry, offset, length]``, as its partition recorded them."""
        for entry in self.var.partition.tabs_indices(self.var.fold):
            if isinstance(entry, dict) and entry.get('pack') == self.var.index:
                return entry['segments']
        raise ValueError(f"{type(self).__name__}: part {self.var.fold} of {self.var.partition.tag} holds no "
                         f"packed tab {self.var.index}")

    def source_index(self) -> np.ndarray:
        """``(tab, row)`` of the partitioned table for each of this packed tab's rows -- its ``source`` slice."""
        return _source_index_(self)

    # 3. Accessors ---------------------------------------------------------

    @forward_property({SOURCE_SLICE: SOURCE})
    def TOPICS(self):
        """The partitioned table's slices, as the tab of its first segment declares them, and ``source``.

        On the class, with no partition to ask, ``source`` alone.
        """
        entry = self.segments()[0][0]
        tab = entry['tab'] if isinstance(entry, dict) else entry
        return _slice_topics_(self.var.partition.datatable.tab(int(tab)), type(self).__name__)

    # 4. Helpers -----------------------------------------------------------

    def _segment_rows_(self) -> list[tuple]:
        """``(tab index, source tab, rows)`` for each segment: the rows of that tab it takes, in order."""
        partition, what = self.var.partition, type(self).__name__
        out = []
        for entry, offset, length in self.segments():
            if isinstance(entry, dict):
                t = int(entry['tab'])
                values = {role: entry[role] for role in PARTITION_ROLES if role in entry}
                columns = {role: tuple(partition.column_spec(role)) for role in values}
            else:
                t, values, columns = int(entry), {}, {}
            source = partition.datatable.tab(t)
            rows = _selected_rows_(source, columns, values, what)[int(offset):int(offset) + int(length)]
            if len(rows) != int(length):
                raise ValueError(f"{what}: {source.tag} holds {len(rows)} of the {length} rows its segment "
                                 f"{[entry, offset, length]} takes. Was the tab rebuilt since it was partitioned? "
                                 f"Build the partition again over the table as it is now.")
            out.append((t, source, rows))
        return out


#: The roles a `DatatableCorePartition`'s collator may hold -- see its docstring.
CORE_ROLES = ('coreby', 'stratifyby', 'partitionby', 'carry')

#: Where a `DatatableCorePartition` clusters -- see its docstring.
CORE_CLUSTERINGS = ('master', 'distributed')


def _spec_path_(pair) -> tuple:
    """A collator's normalized pair as a `_column_values_` spec: ``(slice, column, key, ...)``."""
    s_name, column, *rest = pair
    keys = () if not rest else (rest[0] if isinstance(rest[0], tuple) else (rest[0],))
    return (s_name, column, *keys)


def _row_vectors_(tab, pairs) -> np.ndarray:
    """Each row's *pairs*, flattened and laid end to end, as float32 -- a row holding none of one is all NaN."""
    columns = [_column_values_(tab, _spec_path_(p)) for p in pairs]
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
                raise ValueError(f"{tab.tag}: row {i} flattens {[p for p in pairs]} to {len(r)} values, "
                                 f"others to {width}: the rows of a table must flatten alike to be clustered")
            out[i] = r
    return out


class TabCoreScanCallable:
    """Worker callable: one tab's rows as vectors, and each row's stratum and partition.

    ``'master'`` clustering returns the vectors; ``'distributed'`` writes them
    to the partition's scratch directory, for the clustering passes to read
    there, and returns only a seeded sample of them to start the centers from.
    """

    def __init__(self, partition, idx: int):
        self.partition = partition
        self.idx = idx

    def __call__(self):
        part, var = self.partition, self.partition.var
        tab = var.datatable.tab(self.idx)
        vectors = _row_vectors_(tab, part.collator_pairs('coreby'))
        out = {'idx': self.idx, 'tag': tab.tag, 'n': len(vectors), 'width': vectors.shape[1],
               'valid': ~np.isnan(vectors).any(axis=1) if vectors.size else np.ones(len(vectors), dtype=bool)}
        for role in PARTITION_ROLES:
            spec = part.column_spec(role)
            out[role] = None if spec is None else [None if v is None else _value_key_(v)
                                                   for v in _column_values_(tab, tuple(spec))]
            if out[role] is not None and len(out[role]) != len(vectors):
                raise ValueError(f"{type(part).__name__}: {tab.tag}'s {role} column has {len(out[role])} "
                                 f"values for its {len(vectors)} rows")
        if var.clustering == 'master':
            out['vectors'] = vectors
            return out
        with part.fs.open(part._scratch_path_(self.idx), 'wb') as f:
            np.save(f, vectors)
        valid = np.flatnonzero(out['valid'])
        m = min(len(valid), -(-var.init_size // max(var.datatable.n_tabs, 1)))
        rng = np.random.default_rng([var.seed, self.idx])
        out['sample'] = vectors[np.sort(rng.choice(valid, size=m, replace=False))] if m else vectors[:0]
        return out


class TabCoreStepCallable:
    """Worker callable: one Lloyd step over one tab's scratch vectors, or -- *final* -- its rows' clusters and distances."""

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


class TabRowCountCallable:
    """Worker callable: one tab's rows in *slice* -- what a reused layout is checked against."""

    def __init__(self, table, idx: int, slice: str):
        self.table, self.idx, self.slice = table, idx, slice

    def __call__(self):
        return int(self.table.tab(self.idx).n_rows(self.slice))


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


class TabCoverageCallable:
    """Worker callable: each query's distance to its nearest row of this tab, in each named set of the tab's rows.

    A query is compared only with the rows of its *coverage_clusters*
    nearest clusters -- what makes a pass over the table cost a fraction of
    it, and what makes each distance an upper bound of the true one, equal to
    it whenever the nearest row lies in one of those clusters.

    *jobs*: ``(queries, ids, query_clusters, names, exclude)`` -- the query
    vectors, their own ``(tab, row)``, their candidate clusters, the sets to
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


class TabVectorsCallable:
    """Worker callable: the scratch vectors of some of one tab's rows."""

    def __init__(self, partition, idx: int, rows: np.ndarray):
        self.partition, self.idx, self.rows = partition, idx, rows

    def __call__(self):
        with self.partition.fs.open(self.partition._scratch_path_(self.idx), 'rb') as f:
            return np.load(f)[self.rows]


class DatatableCoreTab(Datatab):
    """A tab of a `DatatableCorePart`: a run of the rows a `DatatableCorePartition` dealt to one fold.

    Rows from anywhere in the partitioned table -- a core tab repacks them --
    and only the columns its partition's collator names, each in a slice of
    the name it had, its own or upstream of the table: a core tab is a plain
    tab, reading nothing upstream, whatever the table it was cut from.

    Its ``source`` slice points back: row i of it is the tab and the row of
    the partitioned table that row i of every other slice was copied from.

    Its identity is its place -- *partition*, *fold*, *index* -- since what it
    holds is what that partition's layout says that place holds.
    """

    SPECIALIZATIONS = []

    @dataclass
    class VAR(Datatab.VAR):
        partition: DatatablePartition = None
        fold: int = 0
        index: int = 0

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.var.partition, DatatableCorePartition):
            raise ValueError(f"{type(self).__name__}: VAR.partition must be a DatatableCorePartition, "
                             f"got {type(self.var.partition).__name__}")

    def __build__(self):
        partition, var = self.var.partition, self.var
        start, end = partition.tabs_indices(var.fold)[var.index]
        layout = partition.layout(var.fold)[start:end]
        schema = partition.read('columns')
        table = partition.datatable
        # One source tab at a time, and only the rows this core tab takes of it:
        # a layout in source order makes a core tab touch a few neighbouring tabs.
        picked = {s: {c: [None] * len(layout) for c in cols} for s, cols in schema.items()}
        for t in np.unique(layout[:, 0]):
            at = np.flatnonzero(layout[:, 0] == t)
            rows = layout[at, 1]
            tab = table.tab(int(t))
            for s, cols in schema.items():
                data = tab.data((s, list(cols)), concat=False)[s]
                for c in cols:
                    column = data[c]
                    for j, r in zip(at, rows):
                        picked[s][c][j] = column[r]
                del data
        with self.slice_writers({**schema, SOURCE_SLICE: SOURCE.columns}) as writers:
            for s, cols in schema.items():
                for j in range(len(layout)):
                    writers[s].write({c: picked[s][c][j] for c in cols})
            for t, r, _ in layout:
                writers[SOURCE_SLICE].write({'tab': int(t), 'row': int(r)})

    # 2. Declared API ------------------------------------------------------

    def source_index(self) -> np.ndarray:
        """``(tab, row)`` of the partitioned table for each of this core tab's rows -- its ``source`` slice."""
        return _source_index_(self)

    # 3. Accessors ---------------------------------------------------------

    @forward_property({SOURCE_SLICE: SOURCE})
    def TOPICS(self):
        """The slices its partition carries, and ``source``.

        Declared bare: their columns are what the partitioned table holds,
        read when the partition is built -- its ``columns`` topic -- and not
        known to a core tab before then.
        """
        return {**{s: DATASLICE for s in self.var.partition.carried_slices()}, SOURCE_SLICE: SOURCE}


class DatatableCorePartition(DatatablePartition):
    """A `DatatablePartition` dealing ROWS, clustered by their values, into folds of fresh tabs: a coreset.

    Its collator names, by role, every column it touches:

    *coreby* -- the columns rows are clustered by: each row's, flattened and
    laid end to end, is the vector k-means clusters. Required.

    *stratifyby*, *partitionby* -- as for `DatatablePartition`: a stratum's rows
    are dealt whole, to one fold, and the fractions hold within each partition.

    *carry* -- columns the folds hold that nothing above reads: labels, for a
    probe. A fold's tabs hold the columns of every role, and no others.

    With ``recursive=True`` a column may be upstream of the table -- a feature
    table's annotations -- and it is copied into the folds' tabs like any
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
    each pass's sums computed by the workers tab by tab, holding only the
    centers -- for a table too large to gather. The scratch is removed when
    the build is done.

    The layout. A fold's rows, as ``(tab, row, cluster)``, in source order --
    or, *shuffle*, in a seeded random one -- are cut into tabs of
    *rows_per_tab*: `DatatableCoreTab`s, built by the fold's part,
    ``partition.part(k).build()``, in parallel. Source order makes a core tab
    read a few neighbouring tabs; a shuffled one reads up to as many tabs as
    it has rows.

    *layout* -- another core partition, whose layout this one takes: the
    same rows, in the same folds and tabs, of THIS table -- a table laid out
    as that one's is, tab for tab and row for row, such as a feature table
    and the table it was computed from. Nothing is clustered or dealt, and
    its collator names only *carry*.

    Topics: ``tabs`` -- per fold, each core tab's ``[start, end)`` in its
    layout; ``layout`` -- per fold, ``(tab, row, cluster)`` per row;
    ``centers``; ``columns`` -- what the core tabs carry, ``{slice: {column:
    type}}``; ``summary``.
    """

    # 2: the roles were renamed on 2026-10-10 -- see DatatablePartition.VERSION.
    VERSION = 2

    TOPICS = {
        'tabs': DATAFILE('tabs.json', "per fold, each core tab's [start, end) in the fold's layout"),
        'layout': DATADIR('per fold, (tab, row, cluster) of each row dealt to it, as <fold>.npy'),
        'centers': DATAFILE('centers.npy', 'the cluster centers'),
        'columns': DATAFILE('columns.json', 'the columns the core tabs carry: {slice: {column: type}}'),
        'summary': DATAFILE('summary.json', 'per fold its rows, strata, clusters and core tabs; what was skipped'),
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
        rows_per_tab: int = 4096
        shuffle: bool = False
        coverage_sample: int = 10000
        coverage_clusters: int = 3
        layout: DatatablePartition = None

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        Datablock.__post_init__(self)
        var, what = self.var, type(self).__name__
        if var.method != 'random' or var.balance != 'rows':
            raise ValueError(f"{what}: deals rows, by cluster -- it takes no method or balance")
        if var.clustering not in CORE_CLUSTERINGS:
            raise ValueError(f"{what}: clustering must be one of {CORE_CLUSTERINGS}, got {var.clustering!r}")
        for name in ('n_clusters', 'max_iter', 'batch_size', 'init_size', 'rows_per_tab', 'coverage_clusters'):
            if not (isinstance(getattr(var, name), int) and getattr(var, name) >= 1):
                raise ValueError(f"{what}: {name} must be a positive int, got {getattr(var, name)!r}")
        if not (isinstance(var.coverage_sample, int) and var.coverage_sample >= 0):
            raise ValueError(f"{what}: coverage_sample must be an int >= 0, got {var.coverage_sample!r}")
        if not var.fractions or any(f < 0 for f in var.fractions) or sum(var.fractions) > 1 + 1e-9:
            raise ValueError(f"{what}: fractions are of the rows, and must be non-negative and sum to at most 1 "
                             f"-- less for a coreset -- got {var.fractions}")
        if var.collator is None:
            raise ValueError(f"{what}: a collator naming the columns, by role, is required")
        roles = var.collator.roles
        other = [r for r in roles if r not in CORE_ROLES]
        if other:
            raise ValueError(f"{what}: a core partition reads the roles {CORE_ROLES}; its collator declares {other} too")
        if var.layout is not None:
            if not isinstance(var.layout, DatatableCorePartition):
                raise ValueError(f"{what}: layout must be a DatatableCorePartition, got {type(var.layout).__name__}")
            if [r for r in roles if r != 'carry']:
                raise ValueError(f"{what}: with a layout nothing is clustered or dealt; its collator names only "
                                 f"'carry', the columns to take, not {[r for r in roles if r != 'carry']}")
        elif not var.collator.pairs('coreby'):
            raise ValueError(f"{what}: its collator names no 'coreby' columns to cluster the rows by")
        for name in PARTITION_ROLES:
            spec = var.collator.var.columns.get(name)
            if spec is not None and not (isinstance(spec, tuple) and len(spec) >= 2 and all(isinstance(p, str) for p in spec)):
                raise ValueError(f"{what}: the collator's {name} must be one column spec, a tuple "
                                 f"(slice, column[, key, ...]) of strings, got {spec!r}")
        if SOURCE_SLICE in self.carried_slices():
            raise ValueError(f"{what}: a carried slice is named {SOURCE_SLICE!r}, the slice a core tab points back "
                             f"with; rename it")
        if any('/' in s for s in self.carried_slices()):
            raise NotImplementedError(f"{what}: nested slices ({self.carried_slices()}) are not carried yet")

    def __build__(self):
        var, table = self.var, self.var.datatable
        var.collator.slices(table)       # raises for a column the table does not hold
        if var.layout is not None:
            layouts, centers, summary, coverage = self._reuse_layout_()
        else:
            layouts, centers, summary, coverage = self._cluster_and_deal_()
        with self.fs.open(self.path('coverage', ensure_dirpath=True), 'wb') as f:
            np.savez(f, **coverage)
        tabs = [[[lo, min(lo + var.rows_per_tab, len(fl))] for lo in range(0, len(fl), var.rows_per_tab)]
                for fl in layouts]
        for k, fl in enumerate(layouts):
            with self.fs.open(os.path.join(self.path('layout', ensure_dirpath=True), f'{k}.npy'), 'wb') as f:
                np.save(f, fl)
        with self.fs.open(self.path('centers', ensure_dirpath=True), 'wb') as f:
            np.save(f, centers)
        with self.fs.open(self.path('columns', ensure_dirpath=True), 'w') as f:
            json.dump(self._carried_schema_(), f)
        with self.fs.open(self.path('tabs', ensure_dirpath=True), 'w') as f:
            json.dump(tabs, f)
        for k, f_tabs in enumerate(tabs):
            summary['folds'][k]['core_tabs'] = len(f_tabs)
        with self.fs.open(self.path('summary', ensure_dirpath=True), 'w') as f:
            json.dump(summary, f, indent=1, default=str)

    def __read__(self, *topicpath):
        topicpath = self._normtopic_(topicpath)
        if topicpath in (('tabs',), ('summary',), ('columns',)):
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

    def tabs_indices(self, fold: int | str) -> list:
        """Fold *fold*'s core tabs, each its ``[start, end)`` in the fold's `layout`."""
        return json.loads(self.fs.cat(self.path('tabs')))[int(fold)]

    def layout(self, fold: int | str) -> np.ndarray:
        """Fold *fold*'s rows, in order: ``(tab, row, cluster)`` of the partitioned table, one row each."""
        with self.fs.open(os.path.join(self.path('layout'), f'{int(fold)}.npy'), 'rb') as f:
            return np.load(f)

    def tabs(self, fold: int | str) -> list:
        return [self.core_tab(fold, i) for i in range(len(self.tabs_indices(fold)))]

    def core_tab(self, fold: int | str, index: int) -> DatatableCoreTab:
        """Core tab *index* of fold *fold*."""
        return DatatableCoreTab(
            datalake=self._datalake_,
            storage_options=self.storage_options,
            verbose=False,
            tag=f"core{self.hash[:8]}_f{int(fold)}_{int(index):06d}",
            spec=dict(partition=self, fold=int(fold), index=int(index)),
        )

    def part(self, fold: int | str) -> DatatableCorePart:
        """Fold *fold* as a table of core tabs, a `DatatableCorePart`."""
        return DatatableCorePart(datalake=self._datalake_, storage_options=self.storage_options,
                                 spec=dict(partition=self, fold=fold))

    def collator_pairs(self, role: str) -> tuple:
        """The collator's pairs playing *role*, as `Datacollator.pairs` gives them."""
        return self.var.collator.pairs(role)

    def carried_slices(self) -> list[str]:
        """The slices the core tabs hold, in the order the collator first names them: every role's."""
        return list(self._carried_columns_())

    # 4. Helpers -----------------------------------------------------------

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
        """The carried columns with their MDS types, read off the first tab -- or the tab its slice is upstream in."""
        tab = self.var.datatable.tab(0)
        owners = tab._upstream_slices_() if hasattr(tab, '_upstream_slices_') else {}
        schema = {}
        for s, cols in self._carried_columns_().items():
            owner = tab if s in tab.slices() else owners[s]
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
        return callable_executor(getattr(self, 'parallelization', None) or 'inline',
                                 n_workers=getattr(self, 'n_workers', 1) or 1, tag=f"{tag} [{type(self).__name__}]")

    def _cluster_and_deal_(self):
        """Scan, cluster, deal and measure: ``(layouts, centers, summary, coverage)``."""
        var, table = self.var, self.var.datatable
        if var.clustering == 'distributed':
            self.fs.makedirs(os.path.dirname(self._scratch_path_(0)), exist_ok=True)
        try:
            scans = self._executor_(f"SCANNING {table.n_tabs} tabs").exec_callables(
                [TabCoreScanCallable(self, i) for i in range(table.n_tabs)])
            widths = {s['width'] for s in scans if s['valid'].any()}
            if len(widths) > 1:
                raise ValueError(f"{type(self).__name__}: the tabs' coreby columns flatten to different widths "
                                 f"{sorted(widths)}; rows must flatten alike to be clustered")
            n_valid = sum(int(s['valid'].sum()) for s in scans)
            if n_valid == 0:
                raise ValueError(f"{type(self).__name__}: no row holds every coreby column")
            k = min(var.n_clusters, n_valid)
            if k < var.n_clusters:
                self.log.warning(f"{type(self).__name__}: {n_valid} rows to cluster, fewer than n_clusters="
                                 f"{var.n_clusters}: clustering into {k}")
            centers, labels, dist, inertia = (self._cluster_master_(scans, k) if var.clustering == 'master'
                                              else self._cluster_distributed_(scans, k))
            layouts, summary = self._deal_rows_(scans, labels, dist)
            # Before the scratch goes: the distances are measured on the vectors it holds.
            coverage = self._coverage_(scans, labels, layouts, centers) if var.coverage_sample else {}
        finally:
            if var.clustering == 'distributed':
                scratch = os.path.dirname(os.path.dirname(self._scratch_path_(0)))
                if self.fs.exists(scratch):
                    self.fs.rm(scratch, recursive=True)
        summary.update(clustering=var.clustering, n_clusters=int(len(centers)), inertia=inertia)
        return layouts, centers, summary, coverage

    def _cluster_master_(self, scans, k):
        """Every valid vector gathered here, clustered by `MiniBatchKMeans`: ``(centers, labels, dist, inertia)`` per tab."""
        from sklearn.cluster import MiniBatchKMeans
        var = self.var
        X = np.concatenate([s['vectors'][s['valid']] for s in scans], axis=0)
        km = MiniBatchKMeans(n_clusters=k, batch_size=var.batch_size, max_iter=var.max_iter, tol=var.tol,
                             n_init=3, random_state=var.seed).fit(X)
        centers = km.cluster_centers_.astype(np.float32)
        labels, dist = [], []
        for s in scans:
            lab, d = _nearest_(s['vectors'], centers)
            labels.append(lab)
            dist.append(d)
        return centers, labels, dist, float(km.inertia_)

    def _cluster_distributed_(self, scans, k):
        """Lloyd's iterations, each pass's sums computed tab by tab by the workers: ``(centers, labels, dist, inertia)``."""
        from sklearn.cluster import kmeans_plusplus
        var, n = self.var, len(scans)
        sample = np.concatenate([s['sample'] for s in scans], axis=0)
        centers, _ = kmeans_plusplus(sample.astype(np.float64), n_clusters=min(k, len(sample)),
                                     random_state=var.seed)
        centers = centers.astype(np.float32)
        inertia = None
        for it in range(var.max_iter):
            steps = self._executor_(f"CLUSTERING pass {it + 1}").exec_callables(
                [TabCoreStepCallable(self, i, centers) for i in range(n)])
            sums = sum(s['sums'] for s in steps)
            counts = sum(s['counts'] for s in steps)
            inertia = sum(s['inertia'] for s in steps)
            moved = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], centers).astype(np.float32)
            shift = float(np.sqrt(((moved - centers) ** 2).sum(axis=1)).max())
            scale = float(np.sqrt((centers ** 2).sum(axis=1)).mean()) or 1.0
            centers = moved
            if shift <= var.tol * scale:
                break
        finals = self._executor_("ASSIGNING rows").exec_callables(
            [TabCoreStepCallable(self, i, centers, final=True) for i in range(n)])
        return centers, [f['labels'] for f in finals], [f['dist'] for f in finals], inertia

    def _coverage_(self, scans, labels, layouts, centers) -> dict:
        """How tightly each fold, and all of them together, fill out the table: the ``coverage`` topic's arrays.

        Distances to the nearest row, measured from seeded samples, tab by tab
        -- in the workers, under ``'distributed'`` -- and only within each
        query's *coverage_clusters* nearest clusters: see the class docstring.
        """
        var, n_folds, k = self.var, len(layouts), len(centers)
        levels = np.asarray(COVERAGE_LEVELS)
        tabs = [s['idx'] for s in scans]
        valid = np.concatenate([np.stack([np.full(len(l), t), np.arange(len(l))], axis=1)[l >= 0]
                                for t, l in zip(tabs, labels)]).astype(np.int64)
        rng = np.random.default_rng([var.seed, 2])

        def pick(ids, n):
            return ids[np.sort(rng.choice(len(ids), size=min(n, len(ids)), replace=False))] if len(ids) else ids

        cores = [fl[:, :2] for fl in layouts]
        union = np.concatenate(cores) if cores else np.zeros((0, 2), dtype=np.int64)
        named = {'table': valid, 'core': union, 'base': pick(valid, len(union))}
        for f, core in enumerate(cores):
            named[f'core{f}'] = core
            named[f'base{f}'] = pick(valid, len(core))
        sets = {t: {name: ids[ids[:, 0] == t, 1] for name, ids in named.items()} for t in tabs}

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
        by_tab = {t: l for t, l in zip(tabs, labels)}
        if var.clustering == 'master':
            results = [TabCoverageCallable(self, s['idx'], by_tab[s['idx']], sets[s['idx']], jobs, vectors=s['vectors'])()
                       for s in scans]
        else:
            results = self._executor_("MEASURING coverage").exec_callables(
                [TabCoverageCallable(self, t, by_tab[t], sets[t], jobs) for t in tabs])
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
        """The vectors of rows *ids*, ``(tab, row)`` -- from the scans, or the workers' scratch."""
        width = next(s['width'] for s in scans if s['valid'].any())
        out = np.empty((len(ids), width), dtype=np.float32)
        tabs = np.unique(ids[:, 0]) if len(ids) else []
        if self.var.clustering == 'master':
            by_tab = {s['idx']: s['vectors'] for s in scans}
            for t in tabs:
                at = ids[:, 0] == t
                out[at] = by_tab[int(t)][ids[at, 1]]
            return out
        got = self._executor_("GATHERING sample vectors").exec_callables(
            [TabVectorsCallable(self, int(t), ids[ids[:, 0] == t, 1]) for t in tabs])
        for t, vecs in zip(tabs, got):
            out[ids[:, 0] == t] = vecs
        return out

    def _deal_rows_(self, scans, labels, dist):
        """Units dealt nearest first, cell by cell, to their folds: ``(layouts, summary)``."""
        var = self.var
        fractions = [float(f) for f in var.fractions]
        tab = np.concatenate([np.full(s['n'], s['idx'], dtype=np.int64) for s in scans])
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
            fl = np.stack([tab[sel], row[sel], label[sel]], axis=1).astype(np.int64)
            fl = fl[np.lexsort((fl[:, 1], fl[:, 0]))] if not var.shuffle else fl[rng.permutation(len(fl))]
            layouts.append(fl)
        folds = []
        for k, fl in enumerate(layouts):
            units_k = np.flatnonzero(fold_of == k)
            folds.append({'fold': k, 'fraction': fractions[k], 'rows': int(len(fl)),
                          ('strata' if keys['stratifyby'] is not None else 'units'): int(len(units_k)),
                          'clusters': int(len(np.unique(fl[:, 2]))) if len(fl) else 0})
        summary = {'n_tabs': len(scans), 'n_rows': int(len(tab)), 'n_dealt': int((row_fold >= 0).sum()),
                   'n_undealt': int((row_fold < 0).sum()), 'coreby': self.collator_pairs('coreby'),
                   'stratifyby': self.column_spec('stratifyby'), 'partitionby': self.column_spec('partitionby'),
                   'seed': var.seed, 'folds': folds, 'skipped': skipped}
        return layouts, summary

    def _reuse_layout_(self):
        """Another core partition's layout, over this table -- checked to be laid out as that one's is."""
        var, source = self.var, self.var.layout
        mine, theirs = var.datatable, source.datatable
        if mine.n_tabs != theirs.n_tabs:
            raise ValueError(f"{type(self).__name__}: the layout is of a table of {theirs.n_tabs} tabs, this one "
                             f"has {mine.n_tabs}: it is not laid out alike")
        counts = [self._executor_(f"COUNTING {t.n_tabs} tabs' rows").exec_callables(
            [TabRowCountCallable(t, i, s) for i in range(t.n_tabs)])
            for t, s in ((mine, self._partition_slice_name_), (theirs, source._partition_slice_name_))]
        if counts[0] != counts[1]:
            bad = next(i for i, (a, b) in enumerate(zip(*counts)) if a != b)
            raise ValueError(f"{type(self).__name__}: tab {bad} has {counts[0][bad]} rows here and {counts[1][bad]} "
                             f"in the layout's table: they are not laid out alike, row for row")
        if len(var.fractions) != source.n_folds():
            raise ValueError(f"{type(self).__name__}: the layout has {source.n_folds()} folds, fractions "
                             f"{var.fractions} name {len(var.fractions)}")
        layouts = [source.layout(k) for k in range(source.n_folds())]
        summary = {**source.read('summary'), 'layout': source.hash}
        for k, f in enumerate(summary['folds']):
            f.pop('core_tabs', None)
        # The layout's coverage: of the rows it took, measured where they were clustered.
        return layouts, source.read('centers'), summary, source.read('coverage')


class DatatableCorePart(DatatablePart):
    """Fold *fold* of a `DatatableCorePartition`: a table of `DatatableCoreTab`s.

    Unlike a `DatatablePart`, not a view of the partitioned table: its tabs
    are its own, built here -- ``build()`` builds them, in parallel -- and
    hold the slices the partition carries, and ``source``.
    """

    TAB = DatatableCoreTab
    TOPICS = Datatable.TOPICS
    SPECIALIZATIONS = []

    # 2. Declared API ------------------------------------------------------

    def slices(self, recursive: bool = False):
        return tuple(self.var.partition.carried_slices()) + (SOURCE_SLICE,)

    def source_index(self) -> np.ndarray:
        """``(tab, row)`` of the partitioned table for each of this part's rows, core tab by core tab."""
        out = [self.tab(i).source_index() for i in range(self.n_tabs)]
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)

    def tab(self, idx: int) -> Datatab:
        return self.var.partition.core_tab(self.var.fold, idx)

    def valid_tab(self, idx: int, validation: str | None = None) -> bool:
        return Datastack.valid_block(self, idx, validation=validation)

    valid_block = valid_tab

    def redirected_tab(self, idx: int) -> bool:
        return self.tab(idx).redirected()

    redirected_block = redirected_tab

    def validate_tab(self, idx: int, **kwargs) -> bool:
        return Datastack.validate_block(self, idx, **kwargs)

    validate_block = validate_tab

    # 3. Accessors ---------------------------------------------------------

    @property
    def datapoints_per_row(self) -> int:
        return 1

    # 4. Helpers -----------------------------------------------------------

    def _blocks_datalake_(self):
        """A core part's tabs are its own, stored where it is."""
        return self.datalake

    def _block_class_(self):
        return DatatableCoreTab

    def _form_block_(self, idx: int, *, use_specializations='stack'):
        return Datastack._form_block_(self, idx, use_specializations=use_specializations)
