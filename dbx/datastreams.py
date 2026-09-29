"""datastreams — Utilities for composing streaming datasets.

This module requires ``torch`` AND ``mosaicml-streaming`` at *import time*
(``torch.utils.data.Dataset`` and the ``streaming.base`` readers).  Both are
optional extras; importing without them raises an ImportError naming what to
install.  The module is intentionally **not** re-exported from
``dbx.__init__``, so the rest of ``dbx`` never forces a dependency on PyTorch
or any streaming library.

Usage::

    from dbx.datastreams import ZipStreamingDataset          # map-style
    from dbx.datastreams import ZipIterableStreamingDatasets  # iterator-style
    from dbx.datatables import Datatab, Datatable
"""

from __future__ import annotations

import contextlib
import gc
import json
import atexit
import os
import weakref
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datastreams requires PyTorch.  "
        "Install it with:  pip install datablocks[torch]"
    ) from exc

try:
    from streaming import MDSWriter, Stream, StreamingDataset
    from streaming.base.compression import decompress as mds_decompress
    from streaming.base.format import reader_from_json
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datastreams requires mosaicml-streaming.  "
        "Install it with:  pip install datablocks[streaming]"
    ) from exc

from .datablocks import DIRTOPIC, Datablock, Datastack


def _same_value_(a, b):
    """Equality that tolerates numpy/torch arrays as well as scalars."""
    if a is b:
        return True
    try:
        if hasattr(a, 'shape') or hasattr(b, 'shape'):
            import numpy as np
            return np.array_equal(np.asarray(a), np.asarray(b))
        return bool(a == b)
    except Exception:
        return False


class SharedMemoryManager:
    """PID-qualified shared memory prefix management for MosaicML StreamingDataset."""

    _patched: bool = False

    @classmethod
    def enable_pid_prefixes(cls) -> None:
        """Monkey-patch MosaicML streaming to use PID-qualified shared memory names.

        This isolates shared memory namespaces per PID (e.g. ``p12345_0000_locals``),
        preventing cross-process and cross-user shared memory collisions on shared nodes.
        """
        if cls._patched:
            return
        try:
            import streaming.base.shared.prefix as shm_prefix_module
            import streaming.base.util as shm_util_module

            def _pid_get_path(prefix_int: int, name: str) -> str:
                pid = os.getpid()
                return f'p{pid}_{prefix_int:04}_{name}'

            shm_prefix_module._get_path = _pid_get_path
            shm_util_module._get_path = _pid_get_path
            cls._patched = True
        except Exception:
            pass

    @classmethod
    def clean_process_shared_memory(cls, pid: int | None = None) -> None:
        """Clean up shared memory segments created by the specified PID (defaults to current PID)."""
        target_pid = pid or os.getpid()
        try:
            from multiprocessing.shared_memory import SharedMemory as BuiltinSharedMemory
            from streaming.base.constant import SHM_TO_CLEAN
            for prefix_int in range(100):
                for shm_name in SHM_TO_CLEAN:
                    for name in (f'p{target_pid}_{prefix_int:04}_{shm_name}', f'{prefix_int:06d}_{shm_name}'):
                        try:
                            shm = BuiltinSharedMemory(name, False, 4)
                            shm.close()
                            shm.unlink()
                        except Exception:
                            pass
        except Exception:
            pass

def release_shared_memory_when_collected(dataset):
    """*dataset*, its shared memory released when it is garbage-collected.

    mosaicml-streaming registers every `SharedMemory` it creates with
    ``atexit`` -- ``atexit.register(self.cleanup)`` -- so the registry holds
    each one, and its file descriptors, until the PROCESS exits, long after the
    `StreamingDataset` that made it is gone. A process that opens a dataset per
    tab, as a probe or a table build does, accumulates them without bound until
    the next open fails with "Too many open files" -- in whatever needed a file
    next, which is how it surfaced: Metal failing to load its shader library.

    So the dataset's segments are found now, while it holds them, and released
    -- closed, and taken off ``atexit`` -- by a finalizer when it is collected.
    """
    try:
        from streaming.base.shared.memory import SharedMemory as StreamingSharedMemory
    except Exception:
        return dataset
    found, seen = [], set()

    def walk(obj, depth):
        if id(obj) in seen or depth > 3:
            return
        seen.add(id(obj))
        if isinstance(obj, StreamingSharedMemory):
            found.append(obj)
            return
        attrs = getattr(obj, '__dict__', None)
        if attrs is None:
            return
        # The dataset itself, then only streaming's own holders: SharedArray,
        # SharedScalar, SharedBarrier -- not whatever else it references.
        if depth and not (type(obj).__module__ or '').startswith('streaming.'):
            return
        for value in list(attrs.values()):
            walk(value, depth + 1)

    walk(dataset, 0)
    if found:
        weakref.finalize(dataset, _release_streaming_shared_memory_, found)
    return dataset


def _release_streaming_shared_memory_(shms):
    for shm in shms:
        try:
            shm.cleanup()
        except Exception:
            pass
        atexit.unregister(shm.cleanup)


# Enable PID-qualified shared memory prefixing automatically on import
SharedMemoryManager.enable_pid_prefixes()


SharedMemoryManager.enable_pid_prefixes()


# ---------------------------------------------------------------------------
#  Column specs: a column, or a column and the keys of it to take
# ---------------------------------------------------------------------------

def key_path(key) -> tuple:
    """A key or path as a tuple: ``'a'`` is ``('a',)``; nested tuples are depth, flattened."""
    if isinstance(key, tuple):
        out = []
        for k in key:
            out.extend(key_path(k) if isinstance(k, tuple) else [str(k)])
        if not out:
            raise ValueError("a key path names at least one key")
        return tuple(out)
    if isinstance(key, list):
        raise ValueError(f"a list is several keys, not one path: {key!r}")
    return (str(key),)


def _path_keys_(parts) -> 'tuple | list | None':
    """The keys a sequence of path parts selects: depth throughout, and a list -- last -- branches.

    ``('site', 'code')`` is one path; ``('site', ['code', 'name'])`` two, both
    under 'site'; ``(['label', ('site', 'code')],)`` two from the top.
    """
    prefix = []
    parts = tuple(p for p in parts if p is not None)     # (column, None): the whole column
    for i, part in enumerate(parts):
        if isinstance(part, list):
            if i != len(parts) - 1:
                raise ValueError(f"several keys ({part!r}) must end a path; depth after them is ambiguous")
            return [tuple(prefix) + key_path(p) for p in part]
        prefix.extend(key_path(part))
    return tuple(prefix) if prefix else None


def column_specs(selection) -> list:
    """What a slice's column selection names, as ``[(column, keys)]``.

    LISTS are several things side by side, TUPLES are depth:

    * ``'annotations'`` -- the column;
    * ``['annotations', 'idx']`` -- two columns;
    * ``('annotations', 'site', 'code')`` -- the column, narrowed to the path
      ``value['site']['code']``; ``('annotations', ('site', 'code'))`` is the
      same, nested tuples being more depth;
    * ``('annotations', ['label', ('site', 'code')])`` -- the column narrowed
      to several keys or paths at once: the dict pruned to them,
      ``{'label': ..., 'site': {'code': ...}}``;
    * ``[('annotations', 'site', 'code'), 'idx']`` -- any of these, side by side.

    *keys* is None (the whole column), one path (a tuple), or a list of paths.
    """
    if isinstance(selection, list):
        return [spec for item in selection for spec in column_specs(item)]
    if isinstance(selection, tuple):
        if not selection:
            raise ValueError("an empty tuple selects nothing")
        head, rest = selection[0], selection[1:]
        if isinstance(head, tuple):                   # ((column, key...), key...)
            return column_specs(key_path(head) + tuple(rest))
        if isinstance(head, list):
            raise ValueError(f"a column selection starts with a column, got {selection!r}")
        return [(str(head), _path_keys_(rest))]
    return [(str(selection), None)]


def slice_spec(item) -> tuple:
    """One ``dataset()``/``data()`` request item as ``(slice, [(column, keys)] | None)``.

    ``'frames'`` is a whole slice; a tuple goes deeper -- ``(slice, selection)``
    or ``(slice, column, key, ...)`` -- where the selection is as
    `column_specs` reads it. A list item is read as a tuple, for callers that
    built one as a list.
    """
    if isinstance(item, str):
        return item, None
    if isinstance(item, (tuple, list)) and item and isinstance(item[0], str):
        s_name, rest = item[0], tuple(item[1:])
        if not rest:
            return s_name, None
        return s_name, column_specs(rest[0] if len(rest) == 1 else rest)
    raise ValueError(
        f"a slice request is a slice name, (slice, column | [columns]) or "
        f"(slice, column, key, ...), got {item!r}"
    )


def column_spec(spec) -> tuple:
    """ONE column spec as ``(column, keys)`` -- `column_specs` for a selection naming one column."""
    if isinstance(spec, tuple) and len(spec) == 2 and (spec[1] is None or isinstance(spec[1], list)):
        # Already normalized, or (column, [keys]).
        column, keys = str(spec[0]), spec[1]
        return column, (None if keys is None else [key_path(k) for k in keys])
    specs = column_specs(spec)
    if len(specs) != 1:
        raise ValueError(f"{spec!r} names {len(specs)} columns, not one")
    return specs[0]


def column_name(spec) -> str:
    """The column a spec reads: rows are keyed by it whatever part of it is taken."""
    return column_spec(spec)[0]


def merge_column_specs(specs) -> list:
    """*specs* with one entry per column, in first-asked order.

    The same column asked for twice cannot appear twice in a row, which is
    keyed by column. So a whole column absorbs any part of it, and two parts
    are their keys together -- one key and another become a list of both.
    """
    order, merged = [], {}
    for spec in specs:
        column, keys = column_spec(spec)
        if column not in merged:
            order.append(column)
            merged[column] = keys
            continue
        prev = merged[column]
        if prev is None or keys is None:
            merged[column] = None
        elif prev != keys:
            have = prev if isinstance(prev, list) else [prev]
            new = keys if isinstance(keys, list) else [keys]
            merged[column] = have + [k for k in new if k not in have]
    return [c if merged[c] is None else (c, merged[c]) for c in order]


def _at_path_(value, path, where, *, allow_none: bool = False):
    for depth, key in enumerate(path):
        if value is None and allow_none:
            return None
        if isinstance(value, list) and allow_none and all(x is None for x in value):
            return [None] * len(value)
        if isinstance(value, list) and allow_none:
            return [_at_path_(item, path[depth:], where, allow_none=True) if item is not None else None for item in value]
        if not isinstance(value, dict):
            at = '.'.join(path[:depth]) or 'the column'
            detail = ""
            if isinstance(value, list) and value and all(x is None for x in value):
                detail = f" (a list of {len(value)} None values; entries may be null/empty)"
            elif value is None:
                detail = " (value is None; entry may be null/empty)"
            raise TypeError(f"{where}: asked for {'.'.join(path)!r}, but {at} holds a "
                            f"{type(value).__name__}{detail}, which is not a dict")
        if key not in value:
            if allow_none:
                return None
            at = '.'.join(path[:depth])
            raise KeyError(f"{where}: no key(s) [{key!r}]{' under ' + repr(at) if at else ''}; "
                           f"it has {sorted(value)}")
        value = value[key]
    return value


def project_column(value, keys, *, where: str = '', allow_none: bool = False):
    """The part of one column's *value* a spec asks for: all of it, one key or path, or a pruned dict."""
    if keys is None:
        return value
    if not isinstance(keys, list):
        return _at_path_(value, key_path(keys), where, allow_none=allow_none)
    out = {}
    for key in keys:
        path = key_path(key)
        node = out
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = _at_path_(value, path, where, allow_none=allow_none)
    return out


def check_structure(value, schema: dict, *, where: str = ''):
    """Refuse a dict that does not have exactly the keys *schema* declares, at every level.

    The keys, not their types: a declared ``'int'`` is written by json, and a
    numpy integer and a Python one are the same value to it. What matters is
    the shape -- a missing key is a ``(slice, column, key)`` read that fails
    later, far from the row that caused it, and an undeclared one is data the
    hash says is not there.
    """
    if not isinstance(value, dict):
        raise TypeError(f"{where}: declared a dict of {sorted(schema)}, got a {type(value).__name__}")
    missing = [k for k in schema if k not in value]
    extra = [k for k in value if k not in schema]
    if missing or extra:
        raise ValueError(
            f"{where}: declared keys {list(schema)}"
            + (f"; missing {missing}" if missing else '')
            + (f"; not declared {extra}" if extra else '')
        )
    for key, sub in schema.items():
        if isinstance(sub, dict):
            check_structure(value[key], sub, where=f"{where}[{key!r}]")


class ZipBase:
    """Merge configuration and merge logic, shared by the two zip datasets.

    ``ZipStreamingDataset`` (map-style, index-addressed) and
    ``ZipIterableStreamingDatasets`` (iterator-style, lockstep) differ only
    in how they *obtain* one sample per source; what they then do with those
    samples is identical.  Keeping that half here means one merge policy,
    one set of error messages, and one place to change either.

    Not a dataset itself -- it defines no ``__getitem__`` and no
    ``__iter__``, and is mixed in front of whichever ``torch`` base supplies
    one.  A third way of reading the sources subclasses this and adds only
    that, calling ``_merge_(idx, samples)`` with one sample per source.

    Beyond the plain merge, two things a multi-slice
    ``Datatable`` needs:

    * **Per-source column projection** (*columns*), so a caller can say
      "frames, and only these two annotation columns" without paying to
      materialise the rest of the annotation row.
    * **Rows that cannot collide** (*names*, *nested*).  Sources written by
      one builder legitimately share bookkeeping columns, so the source's
      name is part of every key and both values survive -- rather than one
      of them winning by source ordering.
    * **Alignment checking that need not be silent** (*shared*,
      *validate_shared*), for the shared columns that are now kept side by
      side and so can be compared.

    Parameters
    ----------
    *datasets
        One or more ``StreamingDataset`` instances to zip.
    names : sequence[str]
        Source names, positionally parallel to *datasets* -- normally the
        slice names.  Required, and required to be distinct: every row is
        keyed by them under either value of *nested*.
    nested : bool
        How a merged row is keyed.  ``True`` (the default) gives
        ``{name: {column: value}}``, keeping each source's columns in their
        own sub-dict; ``False`` flattens to ``{(name, column): value}``.
        Both are collision-free -- the source name is part of every key --
        which is the point: two slices may carry the same column name and
        both values survive, where the older un-named flat row could keep
        only one of them.
    columns : sequence | None
        Per-dataset column projection, positionally parallel to *datasets*.
        Entry *i* is either ``None`` ("every column of dataset *i*") or an
        iterable of column specs to keep -- a column name, or ``(column, key)``
        / ``(column, [keys])`` for part of a dict column (see `column_spec`).
        ``None`` means no projection at all.

        Deliberately a separate argument rather than ``(dataset, columns)``
        pairs: a dataset may itself be a tuple or a list subclass, so there
        is no reliable way to tell a pair from a bare list-like dataset, and
        guessing wrong silently reinterprets the caller's data as a
        projection spec.
    shared : iterable[str] | None
        Columns expected in more than one source.  Every keying keeps all of
        them, so this only marks which ones *validate_shared* compares.
    validate_shared : bool
        Assert that every *shared* key present in more than one source holds
        an equal value at the same index -- which is what makes a mis-zipped
        stream loud instead of silently misaligned.  Off by default: it
        costs a comparison per shared key per item.
    skip_none : bool
        Drop ``None`` values while projecting, so a row does not carry a
        column the source left empty.  On by default; a projection that
        means to carry a genuinely null column wants ``skip_none=False``.
    zip_validator : callable | None
        Optional callable ``(idx, *samples) → None`` that is invoked
        with the flat index and the individual sample dicts **before**
        they are projected or merged.  It should raise on inconsistency
        (e.g. when ``bag_name`` or ``tile_index`` disagree across the
        datasets).
    """

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, *datasets, names, nested=True,
                 columns=None, shared=None,
                 validate_shared=False, skip_none=True,
                 zip_validator=None):
        what = self.__class__.__name__
        if not datasets:
            raise ValueError(f"{what} needs at least one dataset")
        lengths = [len(d) for d in datasets]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{what} requires datasets of equal length, got {lengths}"
            )
        if columns is None:
            columns = [None] * len(datasets)
        elif len(columns) != len(datasets):
            raise ValueError(
                f"columns must be positionally parallel to datasets: got "
                f"{len(columns)} entries for {len(datasets)} datasets"
            )

        # Required, not defaulted: both keyings write the source's name into
        # every row, and a positional stand-in ('0', '1') would key real data
        # by a number that silently shifts when the caller reorders the slices.
        names = [str(n) for n in names]
        if len(names) != len(datasets):
            raise ValueError(
                f"{what}: names must be positionally parallel to datasets: "
                f"got {len(names)} entries for {len(datasets)} datasets"
            )
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"{what}: duplicate source name(s) {duplicates} would key two "
                f"sources to one entry"
            )

        self.datasets = datasets
        self.names = names
        self.nested = bool(nested)
        self.columns = [None if cols is None else list(cols) for cols in columns]
        self.shared = set(shared or ())
        self.validate_shared = validate_shared
        # Whether the shared keys have been shown to be comparable -- carried by
        # two sources or more, so that something actually compares them. Settled
        # on the first row, off the row itself, since a projection can drop a
        # column the sources do hold.
        self._shared_comparable = False
        self.skip_none = skip_none
        self.zip_validator = zip_validator

    def __len__(self):
        return len(self.datasets[0])

    # 4. Helpers -----------------------------------------------------------

    def _project_(self, pos, sample, idx):
        """One source's sample as an ordered ``[(column, value)]`` list.

        Column order is the caller's projection order when there is one, and
        otherwise the order the source wrote them in -- never a set's order.
        Downstream code stacks these into an array whose axes are positional,
        so an order that varied between processes would silently permute the
        axis rather than fail.
        """
        cols = self.columns[pos]
        if cols is None:
            items = list(sample.items())
        else:
            specs = [column_spec(c) for c in cols]
            missing = [c for c, _ in specs if c not in sample]
            if missing:
                raise KeyError(
                    f"{self.__class__.__name__} source {pos} has no column(s) "
                    f"{missing} at index {idx}; it provides {sorted(sample)}"
                )
            # A spec naming key(s) of a dict column takes just those, here,
            # as the row is assembled: only that part of the value reaches
            # the caller -- or a DataLoader's collate.
            items = [(c, project_column(sample[c], keys,
                                        where=f"{self.__class__.__name__} source {pos} column {c!r}"))
                     for c, keys in specs]
        if self.skip_none:
            items = [(k, v) for k, v in items if v is not None]
        return items

    def _check_shared_(self, idx, projected):
        """Assert that every *shared* column agrees across the sources holding it.

        Alignment checking, not merging: two slices written in lockstep carry
        the same bookkeeping column, and a shuffle that mispairs them shows up
        here as a disagreement. Hoisted out of the merge because it says
        nothing about how a row is keyed -- it is equally meaningful when
        nesting keeps the two values in separate sub-dicts, where the flat
        merge that used to host this check never runs at all.
        """
        if not (self.shared and self.validate_shared):
            return
        if len(projected) < 2:
            return  # one source aligns with nothing; there is no pairing to check
        what = self.__class__.__name__
        seen, carried = {}, {}
        for pos, items in enumerate(projected):
            for key, value in items:
                if key not in self.shared:
                    continue
                carried[key] = carried.get(key, 0) + 1
                if key not in seen:
                    seen[key] = (pos, value)
                elif not _same_value_(seen[key][1], value):
                    raise ValueError(
                        f"{what}: shared key {key!r} disagrees "
                        f"between source {seen[key][0]} and source {pos} at "
                        f"index {idx} -- the streams are not aligned"
                    )
        if not self._shared_comparable:
            self._check_shared_is_comparable_(carried)

    def _check_shared_is_comparable_(self, carried):
        """Refuse a shared key that only one source carries.

        A key present in one source is compared against nothing, so the loop
        above passes it every time and the alignment it was named to check goes
        unchecked -- and reads as checked, which is worse than not asking. It is
        the failure mode of naming the shared columns once, on the block, rather
        than at each call: the block cannot know which slices a caller will read
        together, so it has to be told here that its answer did not apply.

        Once per zip, on the first row: which columns a row carries is fixed by
        the sources and the projection, both settled before any row is read.
        """
        lonely = sorted(k for k in self.shared if carried.get(k, 0) < 2)
        if lonely:
            raise ValueError(
                f"{self.__class__.__name__}: shared key(s) {lonely} are carried "
                f"by fewer than two of the {len(self.datasets)} sources read, so "
                f"nothing compares them and the alignment they name is not being "
                f"checked. Read sources that carry them, project them back in, or "
                f"stop naming them shared."
            )
        self._shared_comparable = True

    def _merge_(self, idx, samples) -> dict:
        """Project and merge one sample per source into one row.

        *idx* only labels the item in error messages and is what the
        *zip_validator* is handed.  It is a physical sample index in
        ``ZipStreamingDataset`` and a position within the stream in
        ``ZipIterableStreamingDatasets`` -- the merge itself never uses it
        to address anything, so the difference does not matter here.

        The row is keyed by :attr:`nested`. Either keying is collision-free,
        because a source's name is part of every key, so nothing here decides
        which of two values to keep -- both are kept, under distinct keys.
        """
        if self.zip_validator is not None:
            self.zip_validator(idx, *samples)

        projected = [self._project_(pos, sample, idx)
                     for pos, sample in enumerate(samples)]
        self._check_shared_(idx, projected)

        if self.nested:
            return {name: dict(items)
                    for name, items in zip(self.names, projected)}
        return {(name, key): value
                for name, items in zip(self.names, projected)
                for key, value in items}


class ZipStreamingDataset(ZipBase, Dataset):
    """Pairs multiple ``StreamingDataset`` objects by index.

    All datasets must have the same length.  ``__getitem__`` merges
    the sample dicts from all datasets into a single dict.  Merge parameters
    are documented on ``ZipBase``.

    This avoids opening per-bag tile datasets inside DataLoader
    workers (which breaks DDP barriers) by creating multiple
    rank-coordinated ``StreamingDataset`` objects at the top level.

    Map-style, so index-addressed: every source is read through
    ``StreamingDataset.get_item()``, which is a *blocking* read that
    downloads the sample's shard inline on the calling thread if it is not
    already cached.  ``StreamingDataset``'s download-ahead thread is started
    by its ``__iter__`` and so never runs here.  Two consequences worth
    knowing before choosing this over ``ZipIterableStreamingDatasets``:

    * Shard misses stall the worker for a full download round-trip rather
      than being prefetched.  For local storage there is nothing to
      download and this costs nothing; for remote storage it is the
      dominant cost.
    * Access order is the caller's, so cache locality is the caller's
      problem.  ``DataLoader(shuffle=True)`` is a full permutation and will
      thrash a bounded ``cache_limit``; use ``Datatable.sampler()``,
      which shuffles in shard-sized chunks instead.

    What it buys in exchange is genuine random access: an index means the
    same sample every time, so inspection, subsetting, a ``Subset``, or any
    sampler at all work normally.
    """

    def __getitem__(self, idx):
        return self._merge_(idx, [ds[idx] for ds in self.datasets])


class ZipIterableStreamingDatasets(ZipBase, IterableDataset):
    """Pairs multiple ``StreamingDataset`` objects by *iteration order*.

    The same merge as ``ZipStreamingDataset`` -- merge parameters are
    documented on ``ZipBase`` -- reached by iterating every source in
    lockstep rather than indexing them.  That is what puts each source back
    on its own ``__iter__``, and so back in possession of the machinery
    map-style access leaves switched off: the download-ahead thread, the
    rank/worker partitioning, ``num_canonical_nodes``, the shard-locality
    shuffle, and mid-epoch resumption via ``state_dict()``.

    On remote storage this is the difference between a blocking download per
    shard miss and a background one, which is normally the largest single
    factor in throughput.  On local storage there is no download to hide and
    the gain shrinks to access locality.

    Alignment
    ---------
    Zipping iterators is only correct if every source yields the **same
    sequence of samples**.  Sources of equal length are aligned by
    construction under ``shuffle=False``: the partition comes from
    ``get_partitions(algo, num_samples, num_canonical_nodes,
    num_physical_nodes, ranks_per_node, workers_per_rank, batch_size)``,
    which reads sample *counts* and the world, never shard structure.

    Shuffling is the part that can silently break it.  ``get_shuffle(algo,
    shard_sizes, ...)`` derives the permutation from the *per-shard sample
    counts*, and slices shard by bytes -- so a frames slice and an
    annotations slice holding the same samples split into different shards,
    shuffle differently, and pair unrelated samples.  Hence:

    * ``shuffle=False`` on every source: aligned, no shuffle.  Shuffle
      cannot then be recovered with a sampler, because an ``IterableDataset``
      takes none; it has to come from the source config.
    * ``shuffle=True`` on every source with identical ``shuffle_seed``,
      ``shuffle_algo``, ``num_canonical_nodes`` and ``batch_size``, **and**
      shard boundaries that coincide: aligned, full locality-aware shuffle.
      ``Datatab.slice_writers(..., flush_every=N)`` is what makes the
      boundaries coincide.
    * ``shuffle_algo='naive'`` is the exception that needs no aligned
      boundaries -- it permutes ``sum(shard_sizes)`` and so depends only on
      the total -- but it is a full global permutation with no locality at
      all, which is the thing iterating was meant to avoid.
    * Per-source sampling (``proportion``/``repeat``/``choose``) resamples
      *per shard* and must stay unset.

    ``__init__`` checks the shard boundaries of any shuffling source and
    refuses to build a zip that cannot be aligned; pass
    ``check_alignment=False`` to skip it.  That check is static, so the
    running safety net is still ``shared={'sample_id', ...}`` with
    ``validate_shared=True`` -- cheap for scalar keys, and the only thing
    that catches the drift described below.

    Sharp edges
    -----------
    * **Barriers must be entered in the same order by every worker.**  Each
      source's ``__iter__`` waits on a shared barrier across the node's
      workers.  ``zip()`` advances the sources in a fixed order, which is
      what makes that safe -- iterating a subset of the sources, or in a
      different order, in some worker deadlocks the rest.
    * **Epoch counters are per source.**  The shuffle seed is
      ``shuffle_seed + epoch``, and ``next_epoch`` is incremented by each
      source's own ``__iter__``.  Iterate one slice out of band -- a
      ``stats()`` pass, a debugging loop -- and that slice alone advances an
      epoch, changing its permutation and silently misaligning the zip.
      Nothing but *validate_shared* will notice.
    * **No random access.**  No ``__getitem__``, no sampler, no ``Subset``.
      ``__len__`` is each source's per-rank length, so ``len(loader)``
      works.
    * **Every source needs ``batch_size=``.**  ``StreamingDataset``
      partitions in whole batches and refuses to iterate without it.  It
      must be the per-device batch size the ``DataLoader`` uses, and the
      same on every source.
    * **Threads.**  Two per source per worker, plus an executor.

    Parameters
    ----------
    check_alignment : bool
        Verify at construction that all shuffling sources shard identically.
        On by default.  Sources that are not ``StreamingDataset``\\ s, or are
        not shuffling, carry no shard metadata to check and are skipped.
    *datasets, names, nested, columns, shared, validate_shared, skip_none, zip_validator
        As ``ZipBase``.
    """

    def __init__(self, *datasets, check_alignment=True, **kwargs):
        super().__init__(*datasets, **kwargs)
        if check_alignment:
            self._check_shard_alignment_()

    def __iter__(self):
        # strict=True: sources of equal length can still yield partitions of
        # unequal length if their world or batch_size configuration differs,
        # and a plain zip() would truncate to the shortest and call that
        # success.  The order sources are advanced in is fixed by zip(),
        # which is what keeps their startup barriers deadlock-free.
        iterators = [iter(ds) for ds in self.datasets]
        for idx, samples in enumerate(zip(*iterators, strict=True)):
            yield self._merge_(idx, samples)

    def _check_shard_alignment_(self):
        """Refuse a zip whose shuffling sources shard differently.

        Static and cheap: ``samples_per_shard`` is read off the index each
        source already loaded, so this costs no downloads.  It cannot catch
        epoch drift or a mid-run reconfiguration, which is what
        *validate_shared* is for.
        """
        sharded = [
            (pos, [int(n) for n in ds.samples_per_shard])
            for pos, ds in enumerate(self.datasets)
            if getattr(ds, 'samples_per_shard', None) is not None
            and getattr(ds, 'shuffle', False)
            # 'naive' permutes the total, not the boundaries, so differing
            # boundaries are harmless there -- and it is the one algo for
            # which this check would be a false alarm.
            and getattr(ds, 'shuffle_algo', None) != 'naive'
        ]
        if len(sharded) < 2:
            return
        (first_pos, first), *rest = sharded
        for pos, sizes in rest:
            if sizes == first:
                continue
            raise ValueError(
                f"{self.__class__.__name__}: source {pos} splits its samples "
                f"into {len(sizes)} shards and source {first_pos} into "
                f"{len(first)}, and both are shuffling.  The shuffle "
                f"permutation is derived from the per-shard sample counts, so "
                f"differently-sharded sources yield different orders and "
                f"zipping them pairs unrelated samples.  Write the slices with "
                f"slice_writers(..., flush_every=N) so their shard boundaries "
                f"coincide, or open them with shuffle=False, or pass "
                f"check_alignment=False if you have another reason to believe "
                f"the orders agree."
            )


# The class zips *several* datasets, so the plural reads truer -- but the
# singular is the name that is already imported elsewhere, so it stays the
# canonical one and this is an alias rather than a rename.
ZippedStreamingDatasets = ZipStreamingDataset


# ═══════════════════════════════════════════════════════════════════════
#  Shard-locality-aware sampling
# ═══════════════════════════════════════════════════════════════════════

def shuffled_chunk_order(num_chunks: int, seed: int) -> list:
    """A random permutation of ``range(num_chunks)`` for the given seed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_chunks, generator=generator).tolist()


class ChunkShuffleSampler(Sampler):
    """Shuffles contiguous chunks of an index space, and within each chunk --
    instead of shuffling the whole range at once.

    For anything backed by shard-organised storage, which is what an MDS
    table is: consecutive sample indices live in the same shard. A full
    global shuffle (``torch.randperm`` over the whole dataset) scatters
    every access across the whole table. Shuffling in chunks keeps the
    working set down to a handful of shards, small enough to stay
    cache-resident, while both chunk order and within-chunk order are
    still randomised every epoch.

    Parameters
    ----------
    n : int
        Length of the index space (e.g. ``len(dataset)``).
    chunk_size : int
        Consecutive indices per chunk.
    seed : int
        Base seed, combined with the epoch so each epoch shuffles
        independently but reproducibly.
    fixed_epoch : bool
        Ignore :meth:`set_epoch` and stay on epoch 0, so the order never
        changes for the life of the sampler.
    """

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, n: int, chunk_size: int = None, seed: int = 0,
                 fixed_epoch: bool = False):
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if chunk_size is None or chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.n = n
        self.chunk_size = chunk_size
        self.seed = seed
        self.epoch = 0
        self._fixed_epoch = fixed_epoch
        self._consumed = 0

    def __len__(self):
        return self.n

    def __iter__(self):
        order = self._full_order_()
        start = self._consumed if self._consumed < len(order) else 0
        for i in range(start, len(order)):
            self._consumed = i + 1
            yield order[i]
        self._consumed = 0

    # 2. Declared API ------------------------------------------------------

    def set_epoch(self, epoch: int):
        if not self._fixed_epoch and epoch != self.epoch:
            self.epoch = epoch
            self._consumed = 0

    def state_dict(self):
        return {'epoch': self.epoch, 'consumed': self._consumed}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self._consumed = state_dict['consumed']

    # 4. Helpers -----------------------------------------------------------

    def _full_order_(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        num_chunks = (self.n + self.chunk_size - 1) // self.chunk_size
        order = []
        for chunk in torch.randperm(num_chunks, generator=generator).tolist():
            start = chunk * self.chunk_size
            end = min(start + self.chunk_size, self.n)
            order.extend(
                start + p
                for p in torch.randperm(end - start, generator=generator).tolist()
            )
        return order


class ResumableDataLoader(DataLoader):
    """A ``DataLoader`` that forwards ``state_dict``/``load_state_dict`` to its
    sampler.

    Trainers that support mid-epoch resume check the *loader* for those
    methods, not the sampler, so a :class:`ChunkShuffleSampler`'s resume
    state is inert unless the loader surfaces it.  Only meaningful with a
    sampler that implements them.
    """

    def state_dict(self):
        sampler = self._stateful_sampler_()
        return {} if sampler is None else sampler.state_dict()

    def load_state_dict(self, state_dict):
        sampler = self._stateful_sampler_()
        if sampler is not None and state_dict:
            sampler.load_state_dict(state_dict)

    def _stateful_sampler_(self):
        """The sampler carrying resume state, through whatever wraps it.

        Not ``self.sampler`` directly: under DDP, Lightning replaces the
        sampler with a ``DistributedSamplerWrapper`` so each rank draws a
        disjoint subset -- correct, since :class:`ChunkShuffleSampler` knows
        nothing about ranks -- and that wrapper has no ``state_dict`` of its
        own.  Asking it for one raised, which is how a run reaches
        ``on_save_checkpoint`` and dies at the first checkpoint rather than at
        the first batch::

            AttributeError: 'DistributedSamplerWrapper' object has no
            attribute 'state_dict'

        The wrapper keeps the original at ``.dataset._sampler``, so the walk
        below finds it.  ``None`` when nothing in the chain is stateful, which
        is the honest answer for a plain sampler: an empty state dict rather
        than an exception.

        What this gives back under DDP is ``consumed=0``: the wrapper
        materialises the whole order once per epoch (``reset()`` calls
        ``list(sampler)``), which runs the inner sampler to exhaustion and
        leaves its counter at zero.  So a resume restarts the epoch instead of
        landing mid-way through it -- which is what the state says, not a
        position that would be wrong.
        """
        sampler = self.sampler
        seen = set()
        while sampler is not None and id(sampler) not in seen:
            seen.add(id(sampler))
            if hasattr(sampler, 'state_dict'):
                return sampler
            sampler = getattr(getattr(sampler, 'dataset', None), '_sampler', None)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Block-granular splitting and loader sizing
# ═══════════════════════════════════════════════════════════════════════
#
# The split counterpart of dbx.datastreams.ChunkShuffleSampler, and here for
# the same reason: consecutive sample indices in a shard-backed table live in
# the same shard, so a per-sample-random train/val split scatters both halves
# across the whole table and defeats the local shard cache just as thoroughly
# as no split at all. Splitting whole contiguous chunks leaves each half a
# union of contiguous runs.

def chunk_split_ranges(n: int, chunk_size: int, fractions, seed: int = 0) -> list:
    """Partition ``range(n)`` into groups at chunk granularity, as ranges.

    Each group comes back as ascending, maximally-merged ``(start, end)``
    pairs: adjacent selected chunks coalesce, so three consecutive 2048-chunks
    are one ``(s, s + 6144)`` range rather than three.

    See :func:`chunk_split_indices` for the ``fractions`` semantics.
    """
    fractions = list(fractions)
    if any(f is None for f in fractions[:-1]):
        raise ValueError(f"only the last entry in `fractions` may be None: {fractions}")
    total = sum(f for f in fractions if f is not None)
    if total > 1.0:
        raise ValueError(f"fractions sum to {total}, which exceeds 1.0: {fractions}")

    num_chunks = (n + chunk_size - 1) // chunk_size
    chunk_order = shuffled_chunk_order(num_chunks, seed)

    groups = []
    start_pos = 0
    for frac in fractions:
        if frac is None:
            chunks = sorted(chunk_order[start_pos:])
            start_pos = num_chunks
        else:
            count = int(frac * num_chunks)
            chunks = sorted(chunk_order[start_pos:start_pos + count])
            start_pos += count
        ranges = []
        for c in chunks:
            s, e = c * chunk_size, min(c * chunk_size + chunk_size, n)
            if ranges and ranges[-1][1] == s:
                ranges[-1] = (ranges[-1][0], e)
            else:
                ranges.append((s, e))
        groups.append(ranges)
    return groups


def chunk_split_indices(n: int, chunk_size: int, fractions, seed: int = 0) -> list:
    """Partition ``range(n)`` into ``len(fractions)`` groups at chunk granularity.

    Whole chunks, not individual samples, are assigned to each group, so a
    group stays a union of contiguous index ranges.  Each group's indices come
    back ascending -- pair with
    :class:`~dbx.datastreams.ChunkShuffleSampler` for a shuffled iteration
    order on top.

    Parameters
    ----------
    n : int
        Length of the index space to split (e.g. ``len(dataset)``).
    chunk_size : int
        Consecutive indices per chunk.
    fractions : Sequence[float | None]
        Fraction of the *chunks* per group, in order.  At most the **last**
        entry may be ``None``, meaning "every remaining chunk" -- which is
        what a plain train/val split wants (``[0.8, None]``), since two
        independently-rounded fractions summing to 1.0 can leave a chunk or
        two assigned to neither.  Give every entry a float when the leftover
        should be deliberately unused (``[0.05, 0.02]`` -- a 7% smoke test).
    seed : int
        Seed for the chunk-order shuffle.
    """
    return [
        [i for s, e in ranges for i in range(s, e)]
        for ranges in chunk_split_ranges(n, chunk_size, fractions, seed=seed)
    ]


def val_loader_workers(num_workers, prefetch_factor, val_max_batches):
    """Right-size a validation loader that only ever consumes a few batches.

    A DataLoader prefetches ``num_workers * prefetch_factor`` batches.  With
    ``val_max_batches=1`` that is 12 workers x 4 = 48 batches fetched to
    consume **one**, and because each worker walks its own slice of the index
    space those 48 batches touch ~48x as many shards as the validation reads.
    Under a cache limit that evicts, the discarded prefetches push the training
    working set out of the cache, so every validation is followed by a burst of
    re-downloads.

    Returns ``(num_workers, prefetch_factor)`` capped so the loader fetches at
    most one batch beyond what will be read.  ``val_max_batches=None`` ("read
    the whole split") passes through untouched.
    """
    if not val_max_batches or num_workers <= 0:
        return num_workers, prefetch_factor
    workers = max(1, min(num_workers, val_max_batches))
    prefetch = max(1, min(prefetch_factor, -(-val_max_batches // workers) + 1))
    return workers, prefetch


# ═══════════════════════════════════════════════════════════════════════
#  Lockstep shard boundaries
# ═══════════════════════════════════════════════════════════════════════

class _CountingWriter_:
    """An ``MDSWriter`` that counts its writes and tells a ``ShardSync``.

    A proxy rather than a subclass so that ``slice_writers(flush_every=N)``
    is the only change a ``__build__()`` needs -- every other attribute and
    method, ``finish()`` included, passes straight through, so the writers
    a tab is handed behave exactly as before.
    """

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, name, writer, sync, structures=None):
        self.name = name
        self.n_written = 0
        self._writer = writer
        self._sync = sync
        #: ``{column: schema}`` for the columns this slice declares by the
        #: structure of the dict they hold -- ``DATASLICE(a=dict(k='int'))``.
        self._structures = dict(structures or {})

    def __getattr__(self, name):
        # Reached only for attributes not found normally.  Going through
        # __dict__ rather than self._writer keeps a lookup that happens
        # before __init__ has run from recursing forever.
        try:
            writer = self.__dict__['_writer']
        except KeyError:  # pragma: no cover
            raise AttributeError(name) from None
        return getattr(writer, name)

    # 2. Declared API ------------------------------------------------------

    def write(self, sample):
        for column, schema in self._structures.items():
            if column in sample:
                check_structure(sample[column], schema,
                                where=f"slice {self.name!r} row {self.n_written} column {column!r}")
        # MDSWriter.write() starts a new shard by itself when the byte budget
        # (size_limit) would be exceeded.  That boundary is at a sample count
        # nothing else shares, so it defeats the whole point of flush_every --
        # and it is invisible afterwards, since a ragged index looks exactly
        # like a correctly built one.  Catch it here, where the cause is still
        # nameable, rather than letting the shuffle silently misalign.
        #
        # Only when there are shard boundaries to hold: without flush_every this
        # proxy is counting and nothing more, and where MDSWriter chooses to end
        # a shard is its own business.
        holding = self._sync.every is not None
        n_shards = len(self._writer.shards) if holding else 0
        self._writer.write(sample)
        if holding and len(self._writer.shards) > n_shards and self.n_written % self._sync.every:
            raise ValueError(
                f"slice {self.name!r} hit its size_limit after "
                f"{self.n_written} samples and started a new shard there, "
                f"which is not a multiple of flush_every="
                f"{self._sync.every}: the slices no longer share shard "
                f"boundaries.  Raise size_limit (or pass size_limit=None) so "
                f"that flush_every is what ends a shard, or lower "
                f"flush_every below what fits in one."
            )
        self.n_written += 1
        self._sync.wrote(self)

    def flush_shard(self):
        """End the current shard here, if it has anything in it.

        Both calls, in this order, are what ``MDSWriter.write()`` itself does
        when the byte budget is hit and what ``finish()`` does for the
        remainder: ``flush_shard()`` emits the pending samples as a shard and
        ``_reset_cache()`` starts the next one.  Flushing without resetting
        would write those samples again.

        The emptiness guard matters at the boundary: ``finish()`` flushes
        only ``if self.new_samples``, so a sample count that is an exact
        multiple of *flush_every* must not leave a zero-sample shard behind
        for the merged index to name.
        """
        if self._writer.new_samples:
            self._writer.flush_shard()
            self._writer._reset_cache()


class ShardSync:
    """Counts a group of writers, and breaks them onto a new shard together.

    Two jobs, and only the second is optional.  Counting says whether the tab
    wrote every slice once per item -- the invariant the slices exist to keep,
    since sample i of one must describe what sample i of another does -- and
    costs an increment per write, so it is always on.  Breaking them onto a
    shared shard boundary is what ``flush_every`` asks for.

    The condition is deliberately "every writer has written the same number
    of samples, and that number is a multiple of *every*" rather than
    "*every* writes have happened on this one": slices are written one
    sample each per item but in whatever order the tab's loop body
    happens to use, so the sync has to fire on the last of them, not the
    first.  A tab that does not write its slices in lockstep never satisfies
    the condition, which is what ``check_lockstep()`` reports at the end.
    """

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, every: 'int | None' = None):
        #: None counts without ever rotating: the lockstep COUNT is what says
        #: whether a tab wrote its slices one sample per item, and that question
        #: is worth answering on every build.  Sharing shard boundaries is a
        #: separate thing to want, and only that one costs anything.
        self.every = every
        self.writers = []

    # 2. Declared API ------------------------------------------------------

    def track(self, name, writer, structures=None) -> _CountingWriter_:
        counting = _CountingWriter_(name, writer, self, structures)
        self.writers.append(counting)
        return counting

    def wrote(self, writer: _CountingWriter_):
        if self.every is None:
            return  # counting only: there is no boundary to break onto
        if writer.n_written % self.every:
            return
        if any(w.n_written != writer.n_written for w in self.writers):
            return  # the other slices have not caught up to this item yet
        for w in self.writers:
            w.flush_shard()

    def check_lockstep(self, what: str):
        counts = {w.name: w.n_written for w in self.writers}
        if len(set(counts.values())) <= 1:
            return
        raise ValueError(
            f"{what}.slice_writers: slices were not written in lockstep -- "
            f"{counts}.  Every slice must receive exactly one sample per "
            f"item, or sample i of one slice does not describe the same thing "
            f"as sample i of another and the slices cannot be zipped."
        )


# ═══════════════════════════════════════════════════════════════════════
#  Collation helpers
# ═══════════════════════════════════════════════════════════════════════

def _sanitize_(obj):
    """Recursively replace ``None`` with ``{}`` in nested dicts/lists."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: _sanitize_(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_(v) for v in obj]
    return obj


def sanitize_collate(batch):
    """Collate a batch of sample dicts, tolerating missing or ``None`` values.

    MDS JSON columns can deserialise to ``None`` (e.g. bags without a
    valid case-id have no annotations), or be absent entirely from some
    samples.  PyTorch's ``default_collate`` rejects both cases, so we:

    1. Union all keys across the batch (some shards may omit optional
       columns entirely).
    2. Fill missing keys with ``None``.
    3. Recursively replace ``None`` with ``{}`` via ``_sanitize_()``.
    4. Delegate to ``default_collate``.
    """
    from torch.utils.data._utils.collate import default_collate
    all_keys = set().union(*(s.keys() for s in batch))
    aligned = [{k: s.get(k, None) for k in all_keys} for s in batch]
    return default_collate([_sanitize_(sample) for sample in aligned])


# ═══════════════════════════════════════════════════════════════════════
#  Azure path conversion
# ═══════════════════════════════════════════════════════════════════════

def abfs_to_mds_azure(path: str) -> str:
    """Convert an ``abfs(s)://`` URL to the ``azure-dl://`` scheme used by MosaicML.

    MosaicML StreamingDataset does not recognise ``abfs://`` or ``abfss://``.
    Its ``AzureDataLakeDownloader`` expects::

        azure-dl://container/blob_path

    with the account name provided via the ``AZURE_ACCOUNT_NAME`` env var
    (set automatically by this function if not already present).

    Supported input formats::

        abfss://container@account.dfs.core.windows.net/path   (full form)
        abfs://container/path                                  (short form)

    The short form requires ``AZURE_ACCOUNT_NAME`` in the environment.

    Non-abfs paths (local, etc.) are returned unchanged.
    """
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme not in ('abfs', 'abfss'):
        return path

    if '@' in parsed.netloc:
        # Full form: abfss://container@account.dfs.core.windows.net/path
        container, host = parsed.netloc.split('@', 1)
        account = host.split('.')[0]
        # Ensure AZURE_ACCOUNT_NAME is set for MDS's AzureDataLakeDownloader
        os.environ.setdefault('AZURE_ACCOUNT_NAME', account)
    else:
        # Short form: abfs://container/path  (netloc = container)
        container = parsed.netloc
        if not os.environ.get('AZURE_ACCOUNT_NAME'):
            raise ValueError(
                f"Cannot convert {path!r} to azure-dl:// — "
                f"URL has no @account and AZURE_ACCOUNT_NAME is not set"
            )

    blob_path = parsed.path.lstrip('/')
    return f"azure-dl://{container}/{blob_path}"


# ═══════════════════════════════════════════════════════════════════════
#  Opening a datastream over an MDS index directory
# ═══════════════════════════════════════════════════════════════════════

def open_datastream(index_dir, *, local=None, cache_dir=None, cache=None,
                    cache_limit=None, shuffle=False, allow_unsafe_types=True,
                    **kwargs):
    """Open a ``streaming.StreamingDataset`` over an MDS index directory.

    *index_dir* is the directory holding ``index.json``; ``StreamingDataset``
    resolves every shard in it as ``os.path.join(index_dir, split, basename)``,
    so it must be an **ancestor** of the shards it names.

    Three things this does that a bare ``StreamingDataset(...)`` does not:

    1. A local *index_dir* is passed as ``local=`` with no ``remote=``, so
       nothing is copied; a remote one is passed as ``remote=`` with a local
       cache directory alongside.
    2. ``abfs://``/``abfss://`` is translated to the ``azure-dl://`` scheme
       MosaicML actually understands (see ``abfs_to_mds_azure()``).
    3. ``Reused local directory`` -- which is what a previous process that
       died without releasing its shared-memory bookkeeping looks like -- is
       retried once after ``clean_stale_shared_memory()``.

    Parameters
    ----------
    index_dir : str
        Directory containing ``index.json``.  Local or remote.
    local : str, optional
        Explicit local cache directory.  Ignored when *index_dir* is itself
        local (it is then its own cache).
    cache_dir : str, optional
        Basename of the cache directory to create under *cache* when *local*
        is not given.  Required for a remote *index_dir*.  It must be unique
        per open dataset: two ``StreamingDataset`` objects alive at once over
        one cache directory is exactly the collision retried above.
    cache : str, optional
        Parent of *cache_dir*.  Defaults to the system temporary directory.
    cache_limit, shuffle, allow_unsafe_types, **kwargs
        Passed through to ``StreamingDataset``.
    """
    scheme = urllib.parse.urlparse(index_dir).scheme
    if scheme in ('', 'file'):
        remote = None
        local = index_dir.removeprefix('file://')
    else:
        remote = abfs_to_mds_azure(index_dir) if scheme in ('abfs', 'abfss') else index_dir
        if local is None:
            if cache_dir is None:
                raise ValueError(
                    f"open_datastream({index_dir!r}) is remote, so it needs a "
                    f"local cache: pass local= or cache_dir="
                )
            local = os.path.join(cache or tempfile.gettempdir(), cache_dir)
    if isinstance(cache_limit, str) and cache_limit.lower() in ('none', 'null'):
        cache_limit = None
    if isinstance(cache, str) and cache.lower() in ('none', 'null'):
        cache = None

    streaming_kwargs = dict(
        remote=remote,
        local=local,
        shuffle=shuffle,
        allow_unsafe_types=allow_unsafe_types,
        cache_limit=cache_limit,
        **kwargs,
    )
    SharedMemoryManager.enable_pid_prefixes()
    try:
        return release_shared_memory_when_collected(StreamingDataset(**streaming_kwargs))
    except ValueError as exc:
        if 'Reused local directory' not in str(exc):
            raise
        SharedMemoryManager.clean_process_shared_memory()
        return release_shared_memory_when_collected(StreamingDataset(**streaming_kwargs))


# ═══════════════════════════════════════════════════════════════════════
#  MDS shard reading
# ═══════════════════════════════════════════════════════════════════════

def read_mds_shard(shard_dir, fs, cache_limit='2gb', tmpdir=None):
    """Read all samples from an MDS shard directory.

    Uses the streaming library's low-level ``reader_from_json`` / ``Reader``
    API rather than ``StreamingDataset``.  ``StreamingDataset`` allocates
    shared-memory segments backed by ``mmap`` on every call; in a long
    single-process loop those mappings accumulate and eventually exhaust the
    kernel's ``vm.max_map_count`` limit, producing ``OSError: [Errno 12]
    Cannot allocate memory``.  The low-level reader reads MDS files directly
    from disk with no shared memory.

    Remote shards (abfs/abfss or other schemes) are downloaded to a temporary
    local directory first via the supplied *fs* filesystem, then read
    in-place.

    Parameters
    ----------
    shard_dir : str
        Path (local or remote) to the shard directory containing
        ``index.json`` and ``.mds`` / ``.mds.zstd`` files.
    fs : fsspec filesystem
        Filesystem for *shard_dir*.  Used to download files when the path is
        remote.
    cache_limit : str
        Ignored (kept for API compatibility).
    tmpdir : str, optional
        Base directory for temporary files (decompression scratch space and
        remote-shard download staging).  Defaults to the system temporary
        directory (usually ``/tmp``).  The directory must already exist.

    Returns
    -------
    list[dict]
        Decoded samples, or an empty list if the shard has 0 samples.
    """
    scheme = urllib.parse.urlparse(shard_dir).scheme
    is_local = scheme in ('', 'file')

    if is_local:
        local_dir = shard_dir.removeprefix('file://')
        cleanup = None
    else:
        local_dir = tempfile.mkdtemp(prefix='mds_read_', dir=tmpdir)
        cleanup = local_dir

    try:
        if not is_local:
            # Download every file in the shard directory flat into local_dir.
            # We list + download individually rather than using fs.get(recursive=True)
            # because some fsspec backends reproduce the remote directory name as a
            # subdirectory inside the target, which would misplace the shard files.
            # This must stay inside the try/finally: a failed fs.ls/fs.get (network
            # error, throttling, missing file) must still trigger cleanup below, or
            # local_dir leaks permanently.
            for remote_file in fs.ls(shard_dir, detail=False):
                fname = os.path.basename(remote_file)
                fs.get(remote_file, os.path.join(local_dir, fname))

        index_path = os.path.join(local_dir, 'index.json')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"MDS shard index not found: {index_path} (shard_dir={shard_dir})")

        with open(index_path) as f:
            index = json.load(f)

        shards_meta = index.get('shards', [])
        if not shards_meta:
            return []

        samples = []
        for shard_meta in shards_meta:
            # reader_from_json(dirname, split, obj) — split=None means no subdir.
            reader = reader_from_json(local_dir, None, shard_meta)

            # MDSReader.get_sample_data unconditionally opens the raw (.mds) file;
            # it has no decompression fallback.  If the shard is stored compressed
            # (.mds.zstd), we must decompress first.
            #
            # For remote shards the files are already in a temp local_dir, so we
            # decompress in-place there (the whole dir is cleaned up in `finally`).
            #
            # For local shards we decompress into a separate staging dir under
            # tmpdir so we never write into the source shard directory.
            decomp_cleanup = None
            if reader.compression:
                if is_local:
                    # Stage decompressed files in a throw-away temp dir
                    staging = tempfile.mkdtemp(prefix='mds_decomp_', dir=tmpdir)
                    decomp_cleanup = staging
                    # Copy index.json so the reader can be re-built from staging
                    shutil.copy2(index_path, os.path.join(staging, 'index.json'))
                    split_src = os.path.join(local_dir, reader.split)
                    split_dst = staging  # split='' → files go directly in staging
                else:
                    staging = local_dir
                    split_src = os.path.join(local_dir, reader.split)
                    split_dst = split_src

                for raw_info, zip_info in reader.file_pairs:
                    if zip_info is None:
                        continue
                    zip_path = os.path.join(split_src, zip_info.basename)
                    raw_path = os.path.join(split_dst, raw_info.basename)
                    if not os.path.exists(raw_path) and os.path.exists(zip_path):
                        with open(zip_path, 'rb') as zfp:
                            compressed = zfp.read()
                        raw_bytes = mds_decompress(reader.compression, compressed)
                        # basename may include subdirectories (e.g. a
                        # merged index whose shard basenames are rewritten
                        # to be relative to some ancestor directory, not
                        # just a flat filename) -- the staging dir only
                        # has its own root created by mkdtemp.
                        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                        with open(raw_path, 'wb') as rfp:
                            rfp.write(raw_bytes)

                if is_local:
                    # Re-build reader pointing at the staging dir so get_item
                    # finds the decompressed .mds files there.
                    reader = reader_from_json(staging, None, shard_meta)

            try:
                for idx in range(reader.size):
                    samples.append(reader.get_item(idx))
            finally:
                if decomp_cleanup is not None:
                    shutil.rmtree(decomp_cleanup, ignore_errors=True)

        return samples
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def concat_data(result, dtype=None):
    """Concatenate a list of tensors/ndarrays or a list of dicts.

    If *result* is a list of tensors or list of numpy arrays, stack them along axis 0.
    If *result* is a list of dicts, process each key: if its values across dicts are
    tensors/ndarrays, stack them along axis 0; otherwise keep them as a list inside the dict.
    """
    if not isinstance(result, list) or not result:
        return result

    first = result[0]

    if isinstance(first, dict):
        keys = first.keys()
        out = {}
        for k in keys:
            val_list = [d[k] for d in result]
            out[k] = concat_data(val_list, dtype=dtype)
        return out

    if dtype is not None:
        dtype_str = str(dtype).lower()
        if dtype_str in ('object', "<class 'object'>"):
            if isinstance(first, torch.Tensor) and all(isinstance(x, torch.Tensor) for x in result):
                return torch.stack(result, dim=0)
            if isinstance(first, np.ndarray) and all(isinstance(x, np.ndarray) for x in result):
                return np.stack(result, axis=0)
            return result
        if dtype_str in ('tensor', 'torch') or dtype_str.startswith('tensor:'):
            if all(isinstance(x, torch.Tensor) for x in result):
                return torch.stack(result, dim=0)
            return torch.stack([torch.as_tensor(x) for x in result], dim=0)
        if dtype_str in ('ndarray', 'numpy') or dtype_str.startswith('ndarray:'):
            if all(isinstance(x, np.ndarray) for x in result):
                return np.stack(result, axis=0)
            return np.array(result)
        try:
            return np.array(result, dtype=dtype)
        except Exception:
            pass

    if isinstance(first, torch.Tensor) and all(isinstance(x, torch.Tensor) for x in result):
        return torch.stack(result, dim=0)

    if isinstance(first, np.ndarray) and all(isinstance(x, np.ndarray) for x in result):
        return np.stack(result, axis=0)

    if isinstance(first, np.generic) and all(isinstance(x, np.generic) for x in result):
        return np.array(result)

    return result

