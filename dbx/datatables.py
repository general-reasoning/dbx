"""datapoints — Datatablet / Datatable blocks over MDS slices."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datatables requires PyTorch.  "
        "Install it with:  pip install datablocks[torch]"
    ) from exc

try:
    from streaming import MDSWriter, Stream, StreamingDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datatables requires mosaicml-streaming.  "
        "Install it with:  pip install datablocks[streaming]"
    ) from exc

from .datablocks import (
    DATADICT,
    SAME,
    DATADIR,
    DATAFILE,
    DIRTOPIC,
    Datablock,
    DataJournalFrame,
    Datastack,
    TopicMarkerMeta,
    _NotedMarkerMeta_,
    forward_property,
    forming_with_journal,
    is_topicmarker,
)
from .datastreams import (
    ChunkShuffleSampler,
    SharedMemoryManager,
    release_shared_memory_when_collected,
    ZipIterableStreamingDatasets,
    ZipStreamingDataset,
    ShardSync,
    abfs_to_mds_azure,
    column_spec,
    concat_data,
    key_path,
    merge_column_specs,
    open_datastream,
    conform_structure,
    project_column,
    slice_spec,
    read_mds_shard,
    reader_from_json,
)

#: Topic marker indicating an MDS slice stream directory inside a block.
SLICETOPIC = 'SLICETOPIC'


def sample_rows(n: int, fraction: float, seed: int = 0) -> np.ndarray:
    """The rows a *fraction* of *n* samples, sorted: the same for the same *n*, *fraction* and *seed*.

    ``round(n * fraction)`` of them, at least one of a non-empty *n*, drawn
    without replacement. What `_Database_.data` reads with ``sample_fraction``:
    every slice of one block has its *n* rows, so one draw picks the same rows
    of each, and a row's slices stay together.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"sample_rows: fraction must be in (0, 1], got {fraction!r}")
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    k = min(n, max(1, round(n * fraction)))
    if k == n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


class _DataSliceMeta_(_NotedMarkerMeta_):
    """Makes ``DATASLICE(idx='int')`` a marker carrying those columns.

    A call returns a SUBCLASS rather than an instance, so everything a TOPICS
    declaration holds is a class and one test -- :func:`is_topicmarker` --
    recognises the lot of them.

    Derived from `DATADIR`'s metaclass because `DATASLICE` is a `DATADIR`. A
    slice is declared by its columns and not by a note, so the call and the
    rendering are the columns', as `TopicMarkerMeta` renders them.
    """

    __repr__ = TopicMarkerMeta.__repr__
    __str__ = TopicMarkerMeta.__repr__

    def __call__(cls, *mapping, **typed):
        if mapping and (typed or len(mapping) > 1 or not isinstance(mapping[0], dict)):
            raise TypeError(
                f"{cls.__name__} takes its columns as keywords -- "
                f"{cls.__name__}(idx='int', image='jpeg') -- or as a single mapping "
                f"when a column name is not an identifier"
            )
        columns = dict(mapping[0]) if mapping else dict(typed)
        for name, coltype in columns.items():
            cls._check_column_(name, coltype)
        return _DataSliceMeta_(cls.__name__, (cls,), {'columns': columns})


class DATASLICE(DATADIR, metaclass=_DataSliceMeta_):
    """One independently-readable MDS stream directory.  ``SLICETOPIC`` as a marker.

    A :class:`~dbx.datablocks.DATADIR`, because a slice IS a directory -- so every
    test that asks whether a topic is one answers for a slice without knowing
    what a slice is.

    Declared with its columns and their MDS types::

        TOPICS = {'frames': DATASLICE(idx='int', image='jpeg')}

    which is the point of it.  The sentinel named a slice and said nothing about
    its shape, so the columns lived in whatever dict ``__build__`` happened to
    hand the writers, reached no hash, and could change under artifacts that
    went on claiming to be the same block.  The marker renders as
    ``topic:frames=DATASLICE(idx='int', image='jpeg')``, so adding, dropping,
    retyping or REORDERING a column re-keys the block.  Order counts because MDS
    rows are read back in the order they were written, and a projection that
    permuted an axis rather than failing is the bug this forecloses.

    The declaration is what `Datatablet.slice_writers` writes, so it takes no
    argument when every slice declares its columns.  A bare ``DATASLICE`` declares
    none, and is the sentinel's behaviour under the marker's spelling: the
    columns are then the writer's to supply.

    A column that holds a dict may be declared by the dict's structure instead
    of by an MDS type, as a ``DATADICT`` declares its keys::

        DATASLICE(annotations=dict(cohort='str', recurrence='int'), idx='int')

    It is written as MDS ``'json'`` -- what a dict column is -- and the
    structure is what renders, so it is in the hash, and what a
    ``(slice, column, key)`` read names the keys of. `declared_columns` gives
    the writer's view, `declared_schema` the declaration.
    """

    #: Empty on the bare marker, and set by the call that parameterises it.
    columns = {}

    #: The MDS type a column declared by its dict structure is written as.
    DICT_COLUMN_TYPE = 'json'

    @classmethod
    def writer_columns(cls, columns) -> dict:
        """*columns* as an MDS writer takes them: a dict-structured column as ``'json'``."""
        return {name: cls.DICT_COLUMN_TYPE if isinstance(coltype, dict) else coltype
                for name, coltype in columns.items()}

    @staticmethod
    def _check_column_(name, coltype):
        """Refuse a column that would render into an ambiguous type string."""
        if isinstance(coltype, dict):
            DATASLICE._check_column_(name, 'dict')
            # The same rules a DATADICT schema follows, since it renders the same way.
            DATADICT._check_schema_(coltype, (name,))
            return
        for text, what in ((name, 'column name'), (coltype, 'column type')):
            if not isinstance(text, str):
                raise TypeError(f"DATASLICE {what} must be a string, got {text!r}")
            if '/' in text:
                raise ValueError(
                    f"DATASLICE {what} {text!r} may not contain '/': the marker is "
                    f"rendered into the type string, whose segments are '/'-joined, "
                    f"so a '/' would let two declarations render alike and collide "
                    f"onto one hash"
                )


class _Database_(Datablock):
    """Base class for sliced datapoint blocks (Datatablet and Datatable).

    A **slice** is one independently-readable MDS stream directory inside a block.
    Slices are declared via topics marked with `SLICETOPIC`.

    A subclass declaring TOPICS constructs its TOPICS dictionary explicitly if extending
    its base class's topics::

        class BaseTablet(Datatablet):
            TOPICS = {'samples': SLICETOPIC, 'meta': 'meta.json'}

        class SubTablet(BaseTablet):
            TOPICS = {'report': 'report.json', **BaseTablet.TOPICS}
    """

    TOPICS = {}

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, *args, cache_limit=None, shared_slice_columns=None, **kwargs):
        super().__init__(
            *args,
            cache_limit=cache_limit,
            shared_slice_columns=shared_slice_columns,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        # Read back off self, so a block reconstructed through __setstate__ is
        # normalised too -- and normalised at once, because a bare str is
        # iterable: left as one, 'pool' would be read as ('p','o','o','l') and
        # four columns nothing has would be compared.
        self.shared_slice_columns = self._norm_shared_columns_(
            getattr(self, 'shared_slice_columns', None)
        )

    # 2. Declared API ------------------------------------------------------

    def valid_slice(self, slice) -> bool:
        """True when *slice* has an `index.json` on disk."""
        try:
            return self.fs.exists(self.slice_index_path(slice))
        except Exception:
            return False

    def valid_topic(self, *topicpath):
        """As `Datablock.valid_topic()`, but slices go through `valid_slice()`."""
        topicpath = self._normtopic_(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices():
                return self.valid_slice(topic_str)
        return super().valid_topic(*topicpath)

    def UNSAFE_copy_from(self, anchorkeypath, *, OVERRIDE: bool = False, overwrite: bool = False, topicpaths=None, validate: bool = True, always_copy_whole_dirpath: bool = False, show_progress: bool = True, **kwargs):
        result = super().UNSAFE_copy_from(anchorkeypath, OVERRIDE=OVERRIDE, overwrite=overwrite, topicpaths=topicpaths, validate=validate, always_copy_whole_dirpath=always_copy_whole_dirpath, show_progress=show_progress, **kwargs)
        if validate:
            self.verify_slice_row_counts_match()
        return result

    def slices(self, recursive: bool = False):
        """The names of this block's slice topics, in declaration order.

        *recursive* adds the slices of the blocks upstream of this one, for a
        block that has them (`DataslicesUpstream`); a block reading only its
        own has nothing to add.

        A method rather than a property, mirroring :meth:`topics`, which it is
        the slice-only filter of.

        Derived on each call rather than frozen at class creation, so a TOPICS
        assigned or amended after the class body still reports its slices, and
        an instance overriding TOPICS -- as :class:`DatatablePart` does -- is
        read through.
        """
        return _Database_._find_slice_topics_(getattr(self, 'TOPICS', None))

    def declared_columns(self, slice) -> 'dict | None':
        """The columns *slice* declares, as ``{column: mds_type}``, or None.

        None when nothing was declared -- the :data:`SLICETOPIC` sentinel, or a
        bare :class:`DATASLICE` -- so the columns are still the writer's to supply.
        A declaration, unlike what a writer is handed, is in the block's hash.
        A column declared by its dict structure is its MDS type here, ``'json'``;
        :meth:`declared_schema` has the structure.
        """
        schema = self.declared_schema(slice)
        return DATASLICE.writer_columns(schema) if schema else None

    def declared_schema(self, slice) -> 'dict | None':
        """The columns *slice* declares, as declared: an MDS type, or a dict column's structure."""
        node = self._topicnode_(*slice.split('/'))
        columns = getattr(node, 'columns', None) if is_topicmarker(node, DATASLICE) else None
        return dict(columns) if columns else None

    def data(self, *slice_columns, nested=True, concat: bool = False,
             columns=None, sample_fraction: float | None = None, sample_seed: int = 0, **kwargs):
        """Every row of the named slices -- or, with *sample_fraction*, a sample of them -- keyed exactly as `dataset()` keys one row.

        The mirror of `dataset()`: same ``*slice_columns`` spec, same
        ``nested`` keying, and the same guarantee that no two slices can
        collide. Where `dataset()` gives one row at a time with a scalar at
        each leaf, this gives the whole slice at once with every row's value
        stacked at that leaf::

            dataset(nested=True)[i]  ->  {slice: {column: value}}
            data(nested=True)        ->  {slice: {column: [value, ...]}}

        so a caller that can address one can address the other unchanged.

        Column-major throughout, and with no special case for a single
        slice. Both of those were previously otherwise: one slice returned a
        bare list rather than a mapping, so ``data(s)`` and ``data(s, t)``
        had different shapes and every consumer had to guess which it held.

        Parameters
        ----------
        *slice_columns
            As `dataset()`: a slice name, a ``(slice, column)`` pair, or a
            ``(slice, [columns])`` pair. No arguments means every slice.
        nested : bool
            True (the default) keys as ``{slice: {column: ...}}``; False
            keys as ``{(slice, column): ...}``.
        concat : bool, optional
            Stack each column's values into one array along a new leading
            axis.  Left False, a leaf is the plain list of per-row values.
        sample_fraction : float, optional
            Only that fraction of the rows, in their order: the rows
            `sample_rows` picks with *sample_seed* -- the same rows of every
            slice, so a row's slices stay together, and the same rows of
            another block of as many, as an upstream block's are. A slice is
            still read whole; what comes back is the sample.
        **kwargs
            Passed to `_read_slice_()`.
        """
        names, per_slice_columns = self._parse_slice_columns_(slice_columns, columns)

        out = {}
        for pos, name in enumerate(names):
            rows = self._read_slice_(name, **kwargs)
            if sample_fraction is not None:
                rows = [rows[i] for i in sample_rows(len(rows), sample_fraction, sample_seed)]
            cols = per_slice_columns[pos] if per_slice_columns else None
            if cols is None:
                specs = [(c, None) for c in (rows[0] if rows else [])]
            else:
                specs = [column_spec(c) for c in cols]
                missing = [c for c, _ in specs if rows and c not in rows[0]]
                if missing:
                    raise KeyError(
                        f"{self.__class__.__name__}.data: slice {name!r} has no "
                        f"column(s) {missing}; it provides {sorted(rows[0])}"
                    )
            out[name] = {}
            structures = self._dict_columns_(name)
            for c, keys in specs:
                where = f"{self.__class__.__name__}.data: slice {name!r} column {c!r}"
                if c in structures:
                    vals = [project_column(conform_structure(r[c], structures[c]), keys, where=where) for r in rows]
                else:
                    vals = [project_column(r[c], keys, where=where) for r in rows]
                out[name][c] = concat_data(vals) if concat else vals

        if nested:
            return out
        return {(name, c): vals
                for name, cols in out.items()
                for c, vals in cols.items()}

    def datastream(self, slice, **kwargs) -> StreamingDataset:
        """One slice as a live `StreamingDataset`."""
        self.slice_names((slice,)) # check slice existence/name correctnless
        cache_limit = kwargs.pop('cache_limit', getattr(self, 'cache_limit', None))
        kwargs.setdefault('cache_dir', f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        kwargs.setdefault('cache', self.cacheroot)
        kwargs.setdefault('cache_limit', cache_limit)
        return open_datastream(self.path(*slice.split('/')), **kwargs)

    def dataset(
        self,
        *slice_columns,
        mode='map',
        nested=True,
        columns=None,
        shared=None,
        validate_shared=None,
        skip_none=True,
        zip_validator=None,
        cache_limit=None,
        **kwargs,
    ):
        """The named slices (with optional per-slice column filtering), zipped into one `Dataset`.

        Parameters
        ----------
        *slice_columns : str | tuple[str, str | list[str] | tuple[str, ...]]
            Positional arguments where each item is either:
            - `str`: slice name for all columns (e.g. `"features"`)
            - `(slice, column)` tuple: slice name and specific column (e.g. `("features", "col1")`)
            - `(slice, [col1, col2])` tuple: slice name and list of columns.
            - `(slice, column, key, ...)`: part of a column holding a dict --
              TUPLES go deeper and LISTS are several side by side, at every
              level: ``(s, 'ann', 'site', 'code')`` (or ``(s, 'ann', ('site',
              'code'))``) is ``value['site']['code']``; ``(s, 'ann', ['label',
              ('site', 'code')])`` the dict pruned to those; ``(s, [('ann',
              'label'), 'idx'])`` two columns, one narrowed. A tuple of columns
              is therefore a path, not several columns -- several is a list.
              Taken as each row is assembled, so nothing else of the dict is
              passed on.
            Passing multiple `(slice, col)` tuples for the same slice accumulates their columns;
            the same column asked for whole and in part is read whole.
            If no positional arguments are passed, defaults to all slices with all columns.
        mode : {'map', 'iter'}
            How the slices are read.
        nested : bool
            Row keying, as `ZipBase`. True (the default) gives
            ``{slice: {column: value}}``; False gives ``{(slice, column): value}``.
            Both keep every column of every slice, including a column two
            slices happen to share.
        columns : legacy keyword parameter for backwards compatibility.
        shared : sequence, optional
            Columns two or more of the slices read here are expected to hold
            the same value of, row for row. Defaults to this block's
            ``shared_slice_columns``, which is where it belongs: which columns
            the slices share is a fact about how they were WRITTEN, and the
            caller is the wrong party to know it. A key that only one of the
            sources read carries is refused rather than passed vacuously.
        validate_shared : bool, optional
            Whether to compare them. None -- the default -- means yes when
            *shared* came from the block's own declaration, and no when a caller
            passed *shared* itself, which is how it behaved before.

            Redundant under ``mode='map'``, where the zip pairs physical index
            *i* with index *i* and cannot drift. It is the only runtime guard
            under ``mode='iter'``: there the shuffle seed is
            ``shuffle_seed + epoch`` and ``next_epoch`` is per slice, so reading
            one slice out of band advances that slice alone into a different
            permutation, and every row after it pairs unrelated samples.
        cache_limit : float or str, optional
            Limit on cache size for streaming downloads.
        """
        if mode not in ('map', 'iter'):
            raise ValueError(
                f"{self.__class__.__name__}.dataset: mode must be 'map' or 'iter', got {mode!r}"
            )
        if mode == 'iter' and not isinstance(kwargs.get('batch_size'), int):
            raise ValueError(
                f"{self.__class__.__name__}.dataset(mode='iter') needs batch_size=: "
                f"iterating partitions each slice over ranks and workers in whole batches."
            )

        names, per_slice_columns = self._parse_slice_columns_(slice_columns, columns)
        shared, validate_shared = self._shared_defaults_(shared, validate_shared)

        if cache_limit is not None:
            kwargs['cache_limit'] = cache_limit

        datasets = [self.datastream(name, **kwargs) for name in names]
        zip_cls = ZipStreamingDataset if mode == 'map' else ZipIterableStreamingDatasets
        return zip_cls(
            *datasets,
            names=names,
            nested=nested,
            columns=per_slice_columns,
            shared=shared,
            validate_shared=validate_shared,
            skip_none=skip_none,
            zip_validator=zip_validator,
            structures=[self._dict_columns_(name) for name in names],
        )

    def stats(self, *slices, **kwargs):
        """User-defined summary of the named slices."""
        names = self.slice_names(slices)
        if len(names) == 1 and slices:
            return self.__stats__(names[0], **kwargs)
        return {name: self.__stats__(name, **kwargs) for name in names}

    def n_rows(self, slice: str) -> int:
        """Total dataset rows in the specified slice."""
        if slice is None:
            raise TypeError(f"{self.__class__.__name__}.n_rows requires an explicit slice argument")
        return sum(self.shard_sizes(slice))

    def shard_sizes(self, slice: str) -> list[int]:
        """Row counts per shard for the specified slice."""
        if slice is None:
            raise TypeError(f"{self.__class__.__name__}.shard_sizes requires an explicit slice argument")
        slice = self.slice_names((slice,))[0]
        sizes = []
        if hasattr(self, 'n_tablets'):
            for idx in range(self.n_tablets):
                tablet = self.tablet(idx)
                try:
                    with tablet.fs.open(tablet.slice_index_path(slice), 'r') as f:
                        index = json.load(f)
                    sizes.extend(
                        reader_from_json('.', None, meta).size
                        for meta in index.get('shards', [])
                    )
                except Exception:
                    pass
        else:
            try:
                with self.fs.open(self.slice_index_path(slice), 'r') as f:
                    index = json.load(f)
                sizes.extend(
                    reader_from_json('.', None, meta).size
                    for meta in index.get('shards', [])
                )
            except Exception:
                pass
        return sizes

    def max_rows_per_shard(self, slice: str) -> int:
        """The largest shard's row count for the specified slice."""
        if slice is None:
            raise TypeError(f"{self.__class__.__name__}.max_rows_per_shard requires an explicit slice argument")
        sizes = self.shard_sizes(slice)
        if not sizes:
            raise ValueError(
                f"{self.__class__.__name__}: no shards in slice {slice!r}; is it built?"
            )
        return max(sizes)

    def chunk_shuffle_sampler(
        self,
        slice: str,
        *,
        chunk_size: int | None = None,
        seed: int = 0,
        fixed_epoch: bool = False,
    ) -> ChunkShuffleSampler:
        """Return a `ChunkShuffleSampler` over the rows of the specified slice's datastream.

        Shuffles dataset rows by chunk-shuffling (permuting index chunks) and then
        shuffling within each chunk. `chunk_size` defaults to `max_rows_per_shard(slice)`.

        Parameters
        ----------
        slice : str
            Slice whose row count and shard capacity determine sampler parameters.
        chunk_size : int, optional
            Number of consecutive indices per chunk. Defaults to `max_rows_per_shard(slice)`.
        seed : int, optional
            Base seed for shuffling.
        fixed_epoch : bool, optional
            If True, keeps epoch 0 for fixed validation order.
        """
        if slice is None:
            raise TypeError(f"{self.__class__.__name__}.chunk_shuffle_sampler requires an explicit slice argument")
        if chunk_size is None:
            chunk_size = self.max_rows_per_shard(slice)
        return ChunkShuffleSampler(
            self.n_rows(slice),
            chunk_size,
            seed=seed,
            fixed_epoch=fixed_epoch,
        )

    def verify_slice_row_counts_match(self) -> dict[str, int]:
        """Check that the total number of dataset rows is identical across all declared slices.

        Returns
        -------
        dict[str, int]
            Mapping from slice name to total row count.
        """
        counts = {}
        for s in self.slices():
            counts[s] = self.n_rows(s)
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"{self.__class__.__name__}: slices are not in lockstep row counts: {counts}"
            )
        return counts

    def slice_index_path(self, slice) -> str:
        """Path of the `index.json` for *slice*'s shards."""
        return os.path.join(self.path(*slice.split('/')), 'index.json')

    def slice_names(self, slices) -> tuple:
        """Normalize a `*slices` varargs tuple; empty means *all* slices."""
        return self._slicenames_(slices)

    # 3. Accessors ---------------------------------------------------------

    @property
    def cacheroot(self) -> str:
        """Local scratch root for everything streaming: read caches, staged writes.

        The block's ``cache=``; else ``$DBX_CACHE``; else ``cache/`` under
        :attr:`localroot` -- which is ``DBX_LOCAL`` (or ``local=``) for a block
        on remote storage, and the storage root itself for one already local.
        Resolved on every access, never stored, so no machine's path reaches a
        handle or a journal.
        """
        return (getattr(self, 'cache', None) or os.environ.get('DBX_CACHE')
                or os.path.join(self.localroot, 'cache'))

    # 4. Helpers -----------------------------------------------------------

    def _parse_slice_columns_(self, slice_columns, columns=None):
        """Normalize a ``*slice_columns`` spec into ``(names, per_slice_columns)``.

        Shared by `dataset()` and `data()` so the two accept exactly the same
        spec: a bare slice name, a ``(slice, column)`` pair, a
        ``(slice, [columns])`` pair, or a ``(slice, column, key | [keys])``
        triple for part of a dict column, in any mixture. The two differ in
        what they do with a slice, never in how a caller names one.

        *names* is in the order the slices were asked for -- position decides
        which source is zipped where, so it is derived from the caller's
        sequence and never from a set.
        """
        items = list(slice_columns)
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            first = items[0]
            if isinstance(first, list) or not (isinstance(first[0], str) and first[0] in self.slices()):
                items = list(first)

        if columns is not None:
            if isinstance(columns, dict):
                for s_name, cols in columns.items():
                    if isinstance(cols, (list, tuple)):
                        for c in cols:
                            items.append((s_name, c))
                    else:
                        items.append((s_name, cols))
            elif isinstance(columns, (list, tuple)):
                items.extend(columns)

        if not items:
            names = self.slice_names(())
            per_slice_columns = None
        else:
            slice_order = []
            slice_cols_map = {}
            has_column_filter = False

            for item in items:
                s_name, cols = slice_spec(item)
                has_column_filter = has_column_filter or cols is not None

                if s_name not in slice_cols_map:
                    slice_order.append(s_name)
                    slice_cols_map[s_name] = list(cols) if cols is not None else None
                else:
                    if slice_cols_map[s_name] is not None:
                        if cols is None:
                            slice_cols_map[s_name] = None
                        else:
                            for c in cols:
                                if c not in slice_cols_map[s_name]:
                                    slice_cols_map[s_name].append(c)

            names = self.slice_names(slice_order)

            if has_column_filter:
                per_slice_columns = [None if slice_cols_map[name] is None
                                     else merge_column_specs(slice_cols_map[name])
                                     for name in names]
                for name, cols in zip(names, per_slice_columns):
                    self._check_column_keys_(name, cols)
                if all(c is None for c in per_slice_columns):
                    per_slice_columns = None
            else:
                per_slice_columns = None

        return names, per_slice_columns

    def _check_column_keys_(self, slice, cols):
        """Refuse a ``(slice, column, key)`` the slice's declaration says cannot be read.

        Only where the slice declares the column's structure: an undeclared
        slice, or a column declared plain ``'json'``, has nothing to check
        against, and its keys are found or not when the data is.
        """
        keyed = [column_spec(spec) for spec in cols or ()]
        keyed = [(column, keys) for column, keys in keyed if keys is not None]
        if not keyed:
            return
        schema = self._slice_declaration_(slice)
        for column, keys in keyed:
            if column not in schema:
                continue
            declared = schema[column]
            if not isinstance(declared, dict):
                if declared == DATASLICE.DICT_COLUMN_TYPE:
                    continue
                raise TypeError(
                    f"{self.__class__.__name__}: slice {slice!r} declares column {column!r} "
                    f"as {declared!r}, which has no keys to take {keys!r} of"
                )
            for key in (keys if isinstance(keys, list) else [keys]):
                path, node = key_path(key), declared
                for depth, k in enumerate(path):
                    if not isinstance(node, dict):
                        raise TypeError(
                            f"{self.__class__.__name__}: slice {slice!r} column {column!r} declares "
                            f"{'.'.join(path[:depth])!r} as {node!r}, which has no key {k!r}"
                        )
                    if k not in node:
                        under = '.'.join(path[:depth])
                        raise KeyError(
                            f"{self.__class__.__name__}: slice {slice!r} column {column!r} declares "
                            f"keys {list(node)}{' under ' + repr(under) if under else ''}, not [{k!r}]"
                        )
                    node = node[k]

    def readable_slices(self) -> dict:
        """``{slice: {column: type or structure} | None}``: every slice this block reads, its own and upstream, as its owner declares it.

        What a journal entry records as ``slices``. A table's slices are its
        tablets' topics, not its own -- not in its TOPICS, nor its hash, nor its
        paths -- and this is where one is told what can be read from it. A
        slice's columns are its owner's declaration (`_route_`, then
        `_slice_declaration_`, which asks a table's first tablet) as declared --
        a dict column by its structure, its keys and their types, not the
        ``'json'`` a writer is handed -- None when
        none is declared, or when the owner cannot be asked yet -- a table
        whose tablets are not dealt, say, journaled before its tree is built.
        Never raises for that: a journal entry must not break the build it records.
        """
        out = {}
        for s in self.slices(recursive=True):
            try:
                out[s] = self.slice_declaration(s) or None
            except Exception:
                out[s] = None
        return out

    def _dict_columns_(self, slice) -> dict:
        """``{column: structure}``: *slice*'s columns declared by their dict structure, ``{}`` when it cannot be told."""
        try:
            declared = self.slice_declaration(slice)
        except Exception:
            return {}
        return {c: d for c, d in declared.items() if isinstance(d, dict)}

    def slice_declaration(self, slice) -> dict:
        """*slice*'s columns as its owner declares them -- this block's own, or the upstream one it reads it from -- else ``{}``."""
        # A block reading nothing upstream owns every slice it has; one that does says whose each is.
        route = getattr(self, '_route_', None)
        owner = route([slice])[0][0] if route is not None else self
        return dict(owner._slice_declaration_(slice) or {})

    def _slice_declaration_(self, slice) -> dict:
        """*slice*'s declared schema wherever it is declared -- here, or on a table's tablets -- else ``{}``.

        A table's slices are its tablets' topics, not its own, and every tablet of
        one table is one TABLET class, so the first tablet speaks for them all.
        """
        try:
            return self.declared_schema(slice) or {}
        except KeyError:
            pass
        n_tablets = getattr(self, 'n_tablets', 0) or 0
        if n_tablets and hasattr(self, 'tablet'):
            try:
                return self.tablet(0).declared_schema(slice) or {}
            except KeyError:
                pass
        return {}

    @staticmethod
    def _node_is_sentinel_(node):
        """True when node is a sentinel the markers replace, :data:`SLICETOPIC` included.

        Which is what stops a :class:`DATASLICE` and a ``SLICETOPIC`` sharing one
        declaration: they are the same topic said two ways, and the two ways
        render differently.
        """
        return Datablock._node_is_sentinel_(node) or node == SLICETOPIC

    @staticmethod
    def _node_is_dirtopic_(node):
        """True when node is a directory topic, :data:`SLICETOPIC` included.

        A :class:`DATASLICE` is a :class:`~dbx.datablocks.DATADIR`, so the base test
        already covers the marker.  What this adds is the sentinel, which is a
        string and would otherwise read as a file named ``SLICETOPIC``.
        """
        return Datablock._node_is_dirtopic_(node) or node == SLICETOPIC

    def _is_dir_topic_(self, *topicpath):
        """True when the topic resolves to a directory rather than a file."""
        topicpath = self._normtopic_(topicpath)
        if not topicpath or topicpath[0] is None:
            return False
        node = self._topicnode_(*topicpath)
        return self._node_is_dirtopic_(node)

    def _slicenames_(self, slices) -> tuple:
        """Normalize a `*slices` varargs tuple; empty means *all* slices."""
        if len(slices) == 1 and isinstance(slices[0], (tuple, list)):
            slices = tuple(slices[0])
        if not slices:
            return self.slices()
        unknown = [s for s in slices if s not in self.slices()]
        if unknown:
            raise KeyError(
                f"{self.__class__.__name__}: unknown slice(s) {unknown}; "
                f"available are {list(self.slices())}"
            )
        return tuple(slices)

    @staticmethod
    def _norm_shared_columns_(shared):
        """A ``shared`` declaration as a tuple, or None.  A bare str is one name."""
        if shared is None:
            return None
        if isinstance(shared, str):
            return (shared,)
        return tuple(str(c) for c in shared)

    def _shared_defaults_(self, shared, validate_shared):
        """``(shared, validate_shared)`` with this block's declaration applied.

        The declaration only fills in for a caller that said nothing. A caller
        who passes ``shared=`` gets what it asked for, and gets it unvalidated
        unless it says otherwise -- which is what ``validate_shared=False``
        meant before this defaulting existed.
        """
        declared = getattr(self, 'shared_slice_columns', None)
        if shared is None and declared is not None:
            shared = declared
            if validate_shared is None:
                validate_shared = True
        return shared, bool(validate_shared)

    def _ensure_cacheroot_(self, cache=None) -> str:
        cacheroot = cache or self.cacheroot
        os.makedirs(cacheroot, exist_ok=True)
        return cacheroot

    def _read_slice_(self, slice, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _read_slice_(slice)"
        )

    def _tablet_stream_(self, tablet, slice, local: str | None = None) -> Stream:
        """One tablet's slice as a `Stream`, cached under *local* when it is remote.

        A local tablet is its own cache and *local* is ignored, as in
        `open_datastream()`. A remote one MUST be given a cache directory of its
        own: left without, `Stream` invents `{tmpdir}/{blake2s(remote)}` and
        REFUSES to reuse it, so the second open of that tablet -- a second process,
        a second run, a retry after a crash -- dies with "Could not create a
        temporary local directory ... already exists". Hence the error rather
        than a default here: the caller knows which cache this belongs in, and
        every path that omitted one was a collision waiting to happen.
        """
        index_dir = tablet.path(*slice.split('/'))
        scheme = urllib.parse.urlparse(index_dir).scheme
        if scheme in ('', 'file'):
            return Stream(local=index_dir.removeprefix('file://'))
        remote = abfs_to_mds_azure(index_dir) if scheme in ('abfs', 'abfss') else index_dir
        if local is None:
            raise ValueError(
                f"{self.__class__.__name__}._tablet_stream_: {index_dir} is remote, so it "
                f"needs a local cache directory of its own; pass local="
            )
        os.makedirs(local, exist_ok=True)
        return Stream(remote=remote, local=local)

    def _tablet_streams_(self, slice, local: str):
        """One `Stream` per tablet, each cached in its own subdirectory of *local*.

        Subdivided by tablet, because `StreamingDataset` takes either `streams=` or
        `local=` and never both -- so the cache directory this class computes
        cannot be handed to the dataset, only to the streams under it -- and
        because two streams sharing one directory is itself the collision.
        Named by the tablet's hash: unique per tablet, and the same across runs, so a
        cache is reused rather than rebuilt.
        """
        return [self._tablet_stream_(tablet, slice, local=os.path.join(local, tablet.hash[:12]))
                for tablet in (self.tablet(idx) for idx in range(self.n_tablets))]

    @staticmethod
    def _find_slice_topics_(topics_dict, prefix=()):
        """Every topic path in `topics_dict` marked with `SLICETOPIC`, as a tuple.

        A tuple because a block's slices are settled once its class is: nothing
        may append to them behind the block's back, and the hash they feed
        would be a lie if anything did.
        """
        slice_topics = []
        if not isinstance(topics_dict, dict):
            return ()
        for key, val in topics_dict.items():
            current = prefix + (key,)
            if is_topicmarker(val, DATASLICE) or val == SLICETOPIC:
                slice_topics.append('/'.join(current) if len(current) > 1 else key)
            elif isinstance(val, dict):
                slice_topics.extend(_Database_._find_slice_topics_(val, current))
        return tuple(slice_topics)


class Datatablet(_Database_):
    """One tablet of a `Datatable`: a Datablock writing MDS slices."""

    #: A tablet's tag is in its path, as it always was: thousands of built
    #: tablets per table are keyed so, and a table's tablets are found where
    #: they were built only if the table at its default does not re-key them.
    KEYBY = 'tag_version_shorthash'

    @dataclass
    class VAR(Datablock.VAR):
        datapoints_per_row: int = 1

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, *args, cache=None, cache_limit=None, **kwargs):
        super().__init__(*args, cache=cache, cache_limit=cache_limit, **kwargs)

    def __build__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __build__(): write every "
            f"slice in {list(self.slices()) or ['...']} in lockstep via self.slice_writers(slices)"
        )

    def __validate__(self, **kwargs):
        """As `Datablock.__validate__`, and the slices must agree on their length.

        The write-time count in `slice_writers` catches a tablet that wrote its
        slices unequally. This reads the finished ``index.json`` files instead
        of trusting the write path, so it also answers for a tablet that arrived by
        copy or redirection, or whose upload stopped half way -- and `TabletMaker`
        calls it after every build and refuses the tablet if it says no.

        On the tablet and not on `_Database_`: a table's `shard_sizes()` opens
        the index of every tablet, so the same override on the table would turn one
        validate() into a read per tablet per slice. It needs no such thing -- a
        table is valid only when every tablet of it validated.
        """
        if not super().__validate__(**kwargs):
            return False
        self.verify_slice_row_counts_match()
        return True

    def __stats__(self, slice, **kwargs) -> dict:
        return {'n_rows': len(self._read_slice_(slice))}

    def __read__(self, *topicpath):
        topicpath = self._normtopic_(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices():
                return self.data(topic_str)
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ override to read {'/'.join(topicpath)!r}"
        )

    # 2. Declared API ------------------------------------------------------

    @contextlib.contextmanager
    def slice_writers(self, slices=None, *, only=None, stage: bool = None, cache=None,
                      flush_every: int = None, **writer_kwargs):
        """One `MDSWriter` per slice, as `{slice: writer}`.

        Parameters
        ----------
        slices : dict or sequence of str, optional
            `{slice_name: {column: mds_type}}`, or sequence of slice names to write.
            Omit it when every slice declares its own columns -- `DATASLICE(idx='int')`
            -- and the declaration is written. Passing columns that disagree with
            a declaration is refused: the declared ones are in this block's hash.
        only : sequence of str, optional
            If provided, restrict the writers to only the named slices. Existing
            directories for undeclared or unselected slices are preserved.
        """
        if isinstance(slices, (list, tuple, set)):
            if only is None:
                only = list(slices)
            slices = None

        names = [n for n in self.slices() if n in only] if only is not None else self.slices()
        slices = self._writable_columns_(slices, names)

        if stage is None:
            stage = not self.is_local_fs
        targets = {name: self.path(*name.split('/'), ensure_dirpath=True) for name in names}

        staging = None
        if stage:
            staging = tempfile.mkdtemp(
                prefix=f"{self.__class__.__name__}_",
                dir=self._ensure_cacheroot_(cache),
            )
            outdirs = {name: os.path.join(staging, name.replace('/', '_')) for name in names}
            for outdir in outdirs.values():
                os.makedirs(outdir, exist_ok=True)
        else:
            outdirs = targets
            for outdir in outdirs.values():
                if self.fs.exists(outdir):
                    self.fs.rm(outdir, recursive=True)
                self.fs.makedirs(outdir, exist_ok=True)

        if flush_every is not None and flush_every <= 0:
            raise ValueError(
                f"{self.__class__.__name__}.slice_writers: flush_every must be "
                f"positive, got {flush_every}"
            )

        writers = {}
        # Always, not only when flush_every asks for shared shard boundaries.
        # The count is what catches a tablet that wrote one slice and skipped
        # another for some item -- an early continue, a swallowed per-item
        # exception, a branch that writes two of three -- and left the slices
        # mis-zipped. Nothing downstream can see that: a map-mode zip pairs
        # index i with index i whatever they mean, so every row pairs one item's
        # image with another item's pose and no read ever raises. Tying the
        # check to flush_every made the guarantee a side effect of wanting
        # row-sized shards, which nothing said and nobody would guess.
        sync = ShardSync(flush_every)
        try:
            for name in names:
                writer = MDSWriter(
                    out=outdirs[name], columns=slices[name], **writer_kwargs,
                )
                # A column declared by its dict structure is checked row by
                # row against it, here, where the row that is wrong is known.
                structures = {c: t for c, t in (self.declared_schema(name) or {}).items()
                              if isinstance(t, dict)}
                writers[name] = sync.track(name, writer, structures)
            yield writers
            sync.check_lockstep(self.__class__.__name__)
            for writer in writers.values():
                writer.finish()
            if staging is not None:
                for name in names:
                    self._upload_slice_(outdirs[name], targets[name])
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    # 4. Helpers -----------------------------------------------------------

    def _writable_columns_(self, slices, names):
        """The `{slice: {column: type}}` to write, declaration and argument agreed.

        The declaration wins wherever there is one, because it is what the hash
        was taken over -- an argument may restate it and no more. Where there is
        none, the argument is all there is, as it always was.

        Order is compared along with names and types: MDS reads a row back in
        the order it was written, and the declaration renders in its own order
        into the hash, so a permutation is a different slice by both accounts.
        """
        declared = {name: self.declared_columns(name) for name in names}
        if slices is None:
            undeclared = [name for name, columns in declared.items() if not columns]
            if undeclared:
                raise ValueError(
                    f"{self.__class__.__name__}.slice_writers: no columns for "
                    f"slice(s) {undeclared}; declare them -- DATASLICE(idx='int') -- "
                    f"or pass them"
                )
            return declared
        missing = [name for name in names if name not in slices]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}.slice_writers: no columns for "
                f"slice(s) {missing}; every declared slice must be written"
            )
        for name, columns in declared.items():
            if columns and list(slices[name].items()) != list(columns.items()):
                raise ValueError(
                    f"{self.__class__.__name__}.slice_writers: slice {name!r} is "
                    f"declared {self._topicnode_(*name.split('/'))} but would be "
                    f"written {dict(slices[name])!r}; the declared columns are in "
                    f"this block's hash, so the two may not differ"
                )
        return slices

    def _read_slice_(self, slice, **kwargs):
        try:
            return read_mds_shard(
                self.path(*slice.split('/')), self.fs,
                tmpdir=kwargs.pop('cache', None) or self._ensure_cacheroot_(), **kwargs,
            )
        except FileNotFoundError as e:
            # A missing shard says where, not why: say why, when it is that this tablet is not valid.
            why = self.why_invalid()
            if why is None:
                raise
            raise FileNotFoundError(
                f"{self.anchorkeypath}: cannot read slice {slice!r}: this tablet is not valid: {why}"
            ) from e

    def _upload_slice_(self, local_dir, target_dir):
        names = sorted(os.listdir(local_dir))
        for name in [n for n in names if n != 'index.json'] + \
                    [n for n in names if n == 'index.json']:
            local_path = os.path.join(local_dir, name)
            target_path = os.path.join(target_dir, name)
            self.fs.makedirs(os.path.dirname(target_path), exist_ok=True)
            self.fs.put_file(local_path, target_path)


def DatatableTablet(table, idx, tag=None, **spec):
    if isinstance(table, str) and (table.startswith('$') or table.startswith('@') or table.startswith('#')):
        from .databits import eval as dbx_eval
        table = dbx_eval(table)
    return table(idx, tag=tag, **spec)


#: The name `DatatableTablet` had until 2026-10-10. Speclines are recorded as written -- in
#: identities, in journals -- and one written then, ``$dbx.datatables.DatatableTab(...)``, must
#: still evaluate. For reading those only; write DatatableTablet.
DatatableTab = DatatableTablet


#: What a table declared before tabs were tablets, on 2026-10-10: old name -> new.
_RETIRED_TAB_NAMES_ = {
    'TAB': 'TABLET', '__tab__': '__tablet__', 'n_tabs': 'n_tablets',
    'TAB_SPECIALIZATIONS': 'TABLET_SPECIALIZATIONS', 'TAB_SPECIALIZATION': 'TABLET_SPECIALIZATION',
}


class Datatable(_Database_, Datastack):
    """A table of Datatablets, sliced the same way as its tablets.

    A table's TOPICS only contains what the table itself owns: the structural
    topic ``done`` and any extra file topics the subclass
    declares (such as ``bag_lens``). The tablet's slice topics are NOT merged into
    the table's TOPICS -- they belong to the tablet, not the table.

    The table's `slices()` is derived from ``TABLET``'s slice topics rather than
    from ``TOPICS``, so slice routing (``data()``, ``dataset()``,
    ``valid_slice()``) continues to work without polluting ``TOPICS``.

    The tablet's ordinary (non-slice) topics are written into each tablet under the
    tablet's own key; the table has nothing at those paths.

    The TABLET is part of the table's identity: its type names it, ``TABLET=<fqcn>``,
    after the spec -- so ``TABLET =`` another class is another table. See
    `Datastack._type_entries_`; ``with_block=False`` gives the type under the
    rules from before, as a ``Datatable.Specialization``'s ``TABLET=None`` does.

    KNOWN GAP -- THE TABLET IS NAMED BY ITS FQCN, NOT ITS CONTENTS
    -----------------------------------------------------------
    Edit the TABLET class IN PLACE -- a topic added or respelled, a VERSION bump,
    a VAR field with a default -- and every tablet re-keys while the table does
    not:

    * the table's old ``done`` marker still answers, so ``valid()`` is True;
    * ``build()`` therefore skips the table -- after adopting, block by block,
      whatever the TABLET's SPECIALIZATIONS resolve to (see `Datastack.build`) --
      and builds none of the topics the re-keyed tablets still owe;
    * a tablet without a specialization back to its old identity reads as
      unbuilt, and one with a partial specialization owes what it did not
      cover, while the table reports itself built.

    After such an edit, rebuild the table's tablets explicitly -- build each tablet
    (``table.tablet(i).build()``), or clear the table's ``done`` and build it.
    The ways out, not yet chosen: `valid()` requiring valid tablets (a check per
    tablet), or a stack's build running whenever a block owes topics.
    """

    TABLET = None

    #: The topics the table machinery itself writes and reads: the sentinels
    #: recording which tablets are built, and the marker that says the stack
    #: completed. Subclasses extending TOPICS explicitly include
    #: Datatable.TOPICS if desired.
    #:
    #: A ``tablets`` directory was declared here too, and nothing ever wrote into
    #: it: a tablet is config-addressed on the table's own url and lands beside the
    #: table rather than inside it. A subclass that wants its tablets nested still
    #: declares ``tablets`` itself and roots them there -- which is what it always
    #: was, an addressable location rather than something the machinery used.
    TOPICS = {'done': DATAFILE('done')}

    @dataclass(frozen=True, repr=False, eq=False)   # the base's repr (note last) and equality
    class Specialization(Datablock.Specialization):
        """A Datablock's Specialization, and the TABLET the narrower table's type names.

        A table's BLOCK is its TABLET, so this is `Datastack.Specialization` with
        the field named for what it is. *TABLET*: SAME for this table's own; a
        string for another fqcn; None for a table built before a table's type
        named its TABLET at all.
        """
        TABLET: str | SAME | None = SAME

        __hash__ = Datablock.Specialization.__hash__

        def __post_init__(self):
            super().__post_init__()
            if not (self.TABLET is SAME or self.TABLET is None or (isinstance(self.TABLET, str) and self.TABLET)):
                raise TypeError(f"Specialization TABLET= is SAME, None or a TABLET fqcn, got {self.TABLET!r}")

        @property
        def _block_(self):
            return self.TABLET

    #: None: a table's own topics are markers over its tablets, cheaply written
    #: again; what is worth reaching are its tablets, each by the TABLET's own
    #: specializations.
    SPECIALIZATIONS = []

    Tablet = staticmethod(DatatableTablet)


    @dataclass
    class VAR(Datastack.VAR):
        datapoints_per_row: int = 1

    validate_tablet = Datastack.validate_block
    validate_block = Datastack.validate_block

    UNSAFE_clear_tablet = Datastack.UNSAFE_clear_block
    UNSAFE_clear_tablets = Datastack.UNSAFE_clear_blocks

    class TabletMaker(Datastack.BlockMaker):
        """Lightweight callable that forms and optionally builds a tablet."""

        def __init__(self, table=None, tablet_idx: int | None = None, **kwargs):
            if isinstance(table, int) and tablet_idx is None:
                tablet_idx = table
                table = None
            super().__init__(tablet_idx)
            self.table = table
            self.tablet_idx = tablet_idx
            self.kwargs = kwargs

        def __call__(self, table=None, *, build=True, journal=None):
            tbl = table if table is not None else self.table
            # Formed -- by __block__, then again by _adopt_'s .set() -- against the
            # journal the table read once and the executor handed this worker.
            with forming_with_journal(journal), tbl._block_specializations_in_force_():
                if tbl is not None and tbl.valid_tablet(self.idx):
                    return {'tablet_idx': self.idx, 'skipped': True}
                tablet = tbl.__block__(self.idx, **self.kwargs)
                tablet = tbl._adopt_(tablet, keyby=True)
            skipped = tablet.valid()
            if build and not skipped:
                tablet.build()
                if tbl is not None and hasattr(tbl, 'validate_tablet'):
                    validated = tbl.validate_tablet(self.idx)
                else:
                    validated = tablet.validate()
                if not validated:
                    raise ValueError(f"Tablet {self.idx} of {tbl} failed to validate")
            result = {'tablet_idx': self.idx, 'skipped': skipped}
            del tablet
            gc.collect()
            return result

    # 1. Protocol and hooks ------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        """A table's TABLET is its BLOCK: declaring one declares the other.

        A table may still name a BLOCK of its own, and then its TABLET must be one.
        """
        super().__init_subclass__(**kwargs)
        retired = sorted(set(cls.__dict__) & set(_RETIRED_TAB_NAMES_))
        if retired:
            # Renamed on 2026-10-10: nothing reads the old names, so a table declaring one would
            # be ignored in silence -- its TABLET, its __tablet__, its n_tablets -- and build something else.
            raise TypeError(
                f"{cls.__qualname__} defines {retired}: a table's tablets are tablets now -- "
                + ", ".join(f"{old} is {_RETIRED_TAB_NAMES_[old]}" for old in retired))
        tablet = cls.__dict__.get('TABLET')
        if tablet is None:
            return
        if isinstance(tablet, forward_property):
            # Declared forward, its TABLET is each instance's -- and so is its BLOCK.
            if 'BLOCK' not in cls.__dict__:
                cls.BLOCK = tablet.named('BLOCK')
            return
        if 'BLOCK' in cls.__dict__ and cls.BLOCK is not None:
            if not (isinstance(tablet, type) and issubclass(tablet, cls.BLOCK)):
                raise TypeError(
                    f"{cls.__qualname__}.TABLET = {getattr(tablet, '__name__', tablet)!r} is not a "
                    f"{cls.BLOCK.__name__}, the BLOCK it declares"
                )
        else:
            cls.BLOCK = tablet

    def _type_entries_(self, specialization=None, *, with_block: bool = True) -> dict:
        """A stack's entries, its BLOCK named for what a table's is: ``TABLET=<fqcn>``."""
        entries = super()._type_entries_(specialization, with_block=with_block)
        return {('TAB' if k == 'BLOCK' else k): v for k, v in entries.items()}

    def __init__(self, *args, cache=None, cache_limit=None, filter_built_tablets: bool = False,
                 use_tablet_specializations: 'bool | str | None' = None, **kwargs):
        # A table's blocks are its tablets, so use_tablet_specializations IS
        # use_block_specializations -- either may be given, both if they agree.
        block = kwargs.get('use_block_specializations')
        if use_tablet_specializations is not None:
            if block is not None and block != use_tablet_specializations:
                raise ValueError(
                    f"{type(self).__name__}: use_tablet_specializations={use_tablet_specializations!r} "
                    f"and use_block_specializations={block!r} disagree; they are one setting"
                )
            kwargs['use_block_specializations'] = use_tablet_specializations
        super().__init__(*args, cache=cache, cache_limit=cache_limit, filter_built_tablets=filter_built_tablets, **kwargs)

    def __tablet__(self, idx: int, *, tag=None, **spec) -> Datatablet:
        if self.TABLET is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set TABLET = <Datatablet subclass>"
            )
        # No journal handed down: forming a tablet resolves nothing -- a tablet's
        # specializations are installed by its build(), and a table's build
        # hands its tablet-building callables the one journal it read.
        tablet_specs = getattr(self, 'TABLET_SPECIALIZATIONS', None) or getattr(self, 'TABLET_SPECIALIZATION', None) or getattr(self, 'BLOCK_SPECIALIZATIONS', None) or getattr(self, 'BLOCK_SPECIALIZATION', None)
        return self.TABLET(
            # The table's own url, RAW -- the specline it was given, not what
            # that resolved to -- so a relocatable table stays relocatable tablet
            # by tablet. Without it a tablet fell back to DBX_ROOT, and a table built
            # anywhere else (a test's tmp_path, a second lake) wrote its tablets to
            # an unrelated root, where they were then looked for in vain.
            datalake=self._datalake_,
            storage_options=self.storage_options,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=spec,
            SPECIALIZATIONS=tablet_specs,
            tag=tag if tag is not None else f"tab_{idx:06d}",
        )

    def __block__(self, idx: int) -> Datatablet:
        return self.__tablet__(idx)

    def __split__(self, *args, **kwargs):
        # A table that declares `tab_paths` in its own TOPICS -- the table
        # topic from before the manifest -- gets the directory, empty: a topic declared and never
        # there would read, to anything resolving a specialization of this
        # table later, as a build that has been cleared. Only when it is THIS
        # table's: under a redirection covering it, `path(ensure_dirpath=True)`
        # refuses to create a directory inside another block's data.
        if 'tab_paths' in self.topics() and 'tab_paths' in self.ownedtopics():
            self.path('tab_paths', ensure_dirpath=True)
        n = self.n_tablets
        self.log.info(
            "%s: %d tablets x %d slices %s",
            self.__class__.__name__, n, len(self.slices()), list(self.slices()),
        )
        devices = getattr(self, '_devices', None) or getattr(self, 'devices', None)
        device_mapping = kwargs.get('device_mapping', None)
        if device_mapping is not None and isinstance(device_mapping, dict):
            block_device = device_mapping.get('block_device', None)
            makers = [
                self.TabletMaker(self, idx, device=block_device[idx])
                for idx in range(n)
            ]
        elif devices:
            n_workers = len(devices)
            chunk_boundaries = np.array_split(range(n), n_workers)
            block_device = {}
            for worker_idx, chunk in enumerate(chunk_boundaries):
                dev = devices[worker_idx % len(devices)]
                for idx in chunk:
                    block_device[idx] = dev
            makers = [
                self.TabletMaker(self, idx, device=block_device[idx])
                for idx in range(n)
            ]
        else:
            makers = [self.TabletMaker(self, idx) for idx in range(n)]
        return makers, dict(build=True)

    def __build__(self, *args, **kwargs):
        callables, callable_kwargs = self.__split__(*args, **kwargs)
        if not callables:
            return self.__stack__([])

        filter_built = getattr(self, 'filter_built_tablets', False)
        if filter_built:
            work_stealing_state = getattr(self, 'work_stealing', False)
            self.log.info(
                f"Building {self.__class__.__name__}: filtering {len(callables)} tablets using "
                f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}, work_stealing={work_stealing_state}"
            )
            validity = self.valid_tablets()

            to_build_callables = []
            callable_results = []

            for i, (c, is_valid) in enumerate(zip(callables, validity)):
                idx = getattr(c, 'tablet_idx', getattr(c, 'idx', i))
                if is_valid:
                    callable_results.append({'tablet_idx': idx, 'skipped': True})
                else:
                    to_build_callables.append(c)

            self.log.info(
                f"{self.__class__.__name__}: {len(callables) - len(to_build_callables)}/{len(callables)} tablets already valid, "
                f"building {len(to_build_callables)} tablets"
            )
        else:
            to_build_callables = callables
            callable_results = []

        if to_build_callables:
            build_executor = self._build_executor_(self._build_tag_(len(to_build_callables)))
            callable_kwargs = self._with_build_journal_(to_build_callables, callable_kwargs)
            built_results = build_executor.exec_callables(to_build_callables, self, **callable_kwargs)
            callable_results.extend(built_results)

        callable_results.sort(key=lambda r: r.get('tablet_idx', 0))
        result = self.__stack__(callable_results)
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return result

    def __stack__(self, results=None):
        n_tablets = n_skipped = 0
        for result in (results or []):
            if result is None:
                continue
            n_tablets += 1
            n_skipped += bool(result.get('skipped'))
        self.log.info(
            "%s.__stack__: %d tablets (%d already built)",
            self.__class__.__name__, n_tablets, n_skipped,
        )

        if self.valid_topic('done'):
            self.log.info("%s.__stack__: done marker already present",
                          self.__class__.__name__)
        else:
            with self.fs.open(self.path('done', ensure_dirpath=True), 'wb'):
                pass
            self.log.info("%s.__stack__: done marker written", self.__class__.__name__)
        return self

    def __read__(self, *topicpath):
        topicpath = self._normtopic_(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices():
                return self.data(topic_str)
        if topicpath == ('tab_paths',) and 'tab_paths' in self.topics():
            return self.path('tab_paths')
        if topicpath == ('done',):
            return self.valid()
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ answers only slices, 'tab_paths' and 'done'; "
            f"override it to read {'/'.join(topicpath)!r}"
        )

    def valid(self):
        return self.valid_topic('done')

    def __stats__(self, slice, **kwargs) -> dict:
        return super().__stats__(slice, **kwargs)

    # 2. Declared API ------------------------------------------------------

    #: Slices come from TABLET, not from this table's own TOPICS.
    def slices(self, recursive: bool = False):
        """The TABLET's slices: a table declares none of its own.

        Slice topics belong to the tablet, so a table reads them off ``TABLET``
        rather than out of its own TOPICS -- which is what keeps them out of
        the table's TOPICS while leaving slice routing (`data()`, `dataset()`,
        `valid_slice()`) working.

        Falls back to its own TOPICS when TABLET is unset or is not a
        `_Database_` -- as for :class:`DatatablePart`, which overrides
        TOPICS per instance and computes its TABLET dynamically.
        """
        tablet = getattr(self, 'TABLET', None)
        if isinstance(tablet, type) and issubclass(tablet, _Database_):
            return _Database_._find_slice_topics_(getattr(tablet, 'TOPICS', None))
        return _Database_._find_slice_topics_(getattr(self, 'TOPICS', None))

    def read(self, *topicpath):
        """As `Datablock.read()`, but slice names bypass the TOPICS guard.

        Slice topics are not in this table's TOPICS (they belong to the tablet),
        so the base `read()` would reject them with a KeyError.  Slice reads
        are valid, they just skip the guard and fall through to `__read__`.
        """
        topicpath = self._normtopic_(topicpath)
        topic_str = '/'.join(topicpath)
        if topic_str in self.slices():
            # Bypass _topicnode_: slices are not in TOPICS but are valid reads.
            return self.__read__(*topicpath)
        return super().read(*topicpath)

    def valid_tablet(self, i: int, validation: str | None = None) -> bool:
        """Whether tablet *i* is valid: `Datastack.valid_block`, by tablet."""
        return Datastack.valid_block(self, i, validation=validation)

    valid_block = valid_tablet

    def redirected_tablet(self, i: int) -> bool:
        """Return whether the tablet at index *i* is redirected."""
        return self.tablet(i).redirected()

    redirected_block = redirected_tablet

    def valid_tablets(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, validation: str | None = None, **kwargs) -> pd.Series:
        """Return a pandas Series of booleans, one per tablet, indicating validity (parallelized): `Datastack.valid_blocks`, by tablet."""
        return self.valid_blocks(parallelization=parallelization, n_workers=n_workers, false_only=false_only, true_only=true_only, validation=validation, **kwargs)

    def tablets_redirected(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, **kwargs) -> pd.Series:
        """Whether each tablet is redirected: `Datastack.blocks_redirected`, by tablet."""
        return self.blocks_redirected(parallelization=parallelization, n_workers=n_workers, false_only=false_only, true_only=true_only, **kwargs)

    #: The name `tablets_redirected` had first.
    redirected_tablets = tablets_redirected

    def get_tablet_redirections(self, parallelization: str | None = None, n_workers: int | None = None,
                             journal=None, redirected_only: bool = False, **kwargs) -> pd.Series:
        """Each tablet's `Redirection`, or None: `Datastack.get_block_redirections`, by tablet."""
        return self.get_block_redirections(parallelization=parallelization, n_workers=n_workers,
                                           journal=journal, redirected_only=redirected_only, **kwargs)

    def find_tablet_specializations(self, parallelization: str | None = None, n_workers: int | None = None,
                                 journal=None, found_only: bool = False, **kwargs) -> pd.Series:
        """`Datastack.find_block_specializations`, by tablet."""
        return self.find_block_specializations(parallelization=parallelization, n_workers=n_workers,
                                               journal=journal, found_only=found_only, **kwargs)

    def specialize_tablets(self, parallelization: str | None = None, n_workers: int | None = None,
                        journal=None, **kwargs) -> pd.Series | None:
        """`Datastack.specialize_blocks`, by tablet."""
        return self.specialize_blocks(parallelization=parallelization, n_workers=n_workers,
                                      journal=journal, **kwargs)

    def UNSAFE_clear_tablet_redirections(self, *, OVERRIDE: bool = False, parallelization: str | None = None,
                                      n_workers: int | None = None, **kwargs) -> pd.Series:
        """`Datastack.UNSAFE_clear_block_redirections`, by tablet."""
        return self.UNSAFE_clear_block_redirections(OVERRIDE=OVERRIDE, parallelization=parallelization,
                                                    n_workers=n_workers, **kwargs)

    def validate_tablets(
        self,
        parallelization: str | None = None,
        n_workers: int | None = None,
        work_stealing: bool | None = None,
        false_only: bool = False,
        true_only: bool = False,
        **kwargs,
    ) -> pd.Series:
        """Return a pandas Series of booleans, one per tablet, indicating validation result (parallelized)."""
        return self.validate_blocks(
            parallelization=parallelization,
            n_workers=n_workers,
            work_stealing=work_stealing,
            false_only=false_only,
            true_only=true_only,
            **kwargs,
        )

    def find_tablets(self, signature=None, *patterns, tag=None, path=None, parallelization: str | None = None, n_workers: int | None = None, work_stealing: bool | None = None, **kwargs) -> list[int]:
        """Return a list of indices of all tablets matching the given signature, tag, and/or path pattern(s) (parallelized)."""
        return self.find_blocks(signature, *patterns, tag=tag, path=path, parallelization=parallelization, n_workers=n_workers, work_stealing=work_stealing, **kwargs)

    def tablet_datajournal(self, **kwargs) -> DataJournalFrame | None:
        """Return the DataJournalFrame for child tablets, or None if no tablets exist or journal fails to load."""
        return self.block_datajournal(**kwargs)

    def valid_slice(self, slice) -> bool:
        return all(
            self.tablet(idx).valid_slice(slice) for idx in range(self.n_tablets)
        )

    def tablet(self, idx: int) -> Datatablet:
        return self.block(idx)

    def tablets(self) -> list:
        return self.blocks()

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
    def tablet_redirections(self) -> pd.Series:
        """`Datastack.block_redirections`, by tablet."""
        return self.block_redirections

    @property
    def use_tablet_specializations(self):
        """`use_block_specializations`, by its table name."""
        return getattr(self, 'use_block_specializations', None)

    @property
    def n_tablets(self) -> int:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement n_tablets"
        )

    @property
    def n_blocks(self) -> int:
        return self.n_tablets

    # 4. Helpers -----------------------------------------------------------


    def _topics_signature_(self, topics=None, *, declared=None):
        """Own TOPICS segments -- plus, in a sentinel declaration, the TABLET's slices.

        A table's slices are the TABLET's declaration and not the table's, and a
        block has no business carrying another's topics in its own identity: the
        TABLET is already part of this table, so a differently-sliced TABLET is already
        a different table. A table that declares its own topics with the markers
        carries only its own.

        The accumulating form is kept for every table spelled the older way, and
        byte-identically (slice topics rendered as ``topic:<name>=SLICETOPIC``),
        so no existing hash moves.
        """
        # Own (non-slice) topics come first, in declaration order. *topics*
        # restricts THOSE -- a specialization names this table's topics, not
        # the TABLET's slices, which are the TABLET's declaration and accumulate here
        # whatever subset of its own a table is being rendered under.
        own = super()._topics_signature_(topics, declared=declared)
        # The era of the declaration being RENDERED: a table reconstructed from a
        # sentinel-era declaration accumulated its TABLET's slices then, whatever
        # this class is spelled in now.
        if self._modern_topics_(self.TOPICS if declared is None else declared):
            return own
        # Then the TABLET's slice topics, in the same format Datastack uses.
        tablet = self.TABLET
        if isinstance(tablet, type) and issubclass(tablet, _Database_):
            # TABLET is a class here, and slices() is an instance method as
            # topics() is, so the shared helper does the work rather than an
            # unbound call.
            slice_segments = tuple(
                f"topic:{name}=SLICETOPIC"
                for name in _Database_._find_slice_topics_(getattr(tablet, 'TOPICS', None))
            )
        else:
            slice_segments = ()
        return own + slice_segments

    def _block_class_(self):
        return getattr(self, 'BLOCK', None) or getattr(self, 'TABLET', None)

    def _read_slice_(self, slice, *, tablets=None, **kwargs):
        indices = range(self.n_tablets) if tablets is None else tablets
        datapoints = []
        for idx in indices:
            datapoints.extend(self.tablet(idx)._read_slice_(slice, **kwargs))
        return datapoints


class Datacollator(Datablock):
    """Callable Datablock naming the columns a consumer reads from a table, by role, and collating them.

    `Datacollator` has no `TOPICS` (it does not build or persist files).

    ``columns`` maps a ROLE to the ``(slice, column)`` pairs that play it:
    ``'signals'`` and ``'labels'`` for a model or a probe, ``'stratifyby'`` and
    ``'partitionby'`` for a `DatatablePartition` -- whatever its consumer reads.
    A role's value is a list of pairs, or one pair as a bare tuple of strings.

    When invoked as `collator(datapoints)`, it extracts the ``signals`` pairs --
    and the ``labels`` pairs, when declared -- from each datapoint dict,
    stacking signal tensors along a new dimension 1 for each datapoint, and
    concatenating datapoints along dimension 0 (batch dimension). A collator
    with no ``labels`` returns ``(signals,)``.

    A pair may go deeper, into a column that holds a dict, read as `dataset()`
    reads a request -- TUPLES are depth, LISTS are several side by side:
    ``('annotations', 'annotations', 'label')`` is ``value['label']``,
    ``('annotations', 'annotations', 'site', 'code')`` -- or
    ``(..., ('site', 'code'))`` -- is ``value['site']['code']``, and a list of
    keys, paths or columns is one pair per item. The entry is taken where the
    value is picked, so it works on a row and on a stacked batch alike.

    ``recursive`` says whether the slices may be the table's UPSTREAM ones --
    a `Featuretablet`'s samples, say, rather than its ``features``. Off unless
    turned on: `slices(table)` refuses a slice the table only borrows, so
    reading across blocks is something a collator declares, not something it
    drifts into.
    """

    TOPICS = {}

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={'recursive': recursive}, topics=SAME,
            redirect_vars={'columns.signals': 'signals', 'columns.labels': 'labels'},
            note=f"signals and labels were fields of their own, before columns held them by role; "
                 f"slices routed upstream unasked -- as recursive={recursive} reads them")
        for recursive in (False, True)
    ]

    @dataclass
    class VAR(Datablock.VAR):
        columns: dict[str, list]
        recursive: bool = False
        length: int | None = None
        skip_missing: bool = False

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        columns = self.var.columns
        if not isinstance(columns, dict) or not all(isinstance(r, str) for r in columns):
            raise TypeError(f"{type(self).__name__}: columns must be a {{role: pairs}} dict, got {columns!r}")

    def __call__(self, datapoints, *, signal_only: bool = False):
        """Collate a batch of datapoint rows into signal (and label) arrays.

        Parameters
        ----------
        datapoints : list[dict] | dict
            Either one row per sample -- what a `DataLoader` over
            `dataset()` yields -- or one already-stacked mapping per slice,
            which is what `data(*collator.slices(), concat=True)` hands back.
            Both are keyed ``{slice: {column: ...}}``; they are told apart by
            whether the outer object is a list, not by a flag, because the
            callers that pass a whole slice at a time (a feature build, a
            probe fit) are the same ones that pass batches elsewhere.
        signal_only : bool
            Return the signal array bare, rather than a tuple.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, ...]
            The signal array alone when *signal_only*; otherwise
            ``(signals, labels)``, or ``(signals,)`` when no labels are
            declared.
        """
        if not self.signal_pairs:
            raise KeyError(f"{type(self).__name__}: no 'signals' to collate; its roles are {list(self.var.columns)}")
        sig_arr = self._collate_pairs_(datapoints, self.signal_pairs)

        length = self.var.length
        if length is not None and getattr(sig_arr, 'ndim', 0) >= 1 and sig_arr.shape[-1] > length:
            sig_arr = sig_arr[..., :length]

        if signal_only:
            return sig_arr

        if not self.label_pairs:
            return (sig_arr,)

        lbl_arr = self._collate_pairs_(datapoints, self.label_pairs)
        if length is not None and getattr(lbl_arr, 'ndim', 0) >= 1 and lbl_arr.shape[-1] > length:
            lbl_arr = lbl_arr[..., :length]
        return (sig_arr, lbl_arr)

    # 2. Declared API ------------------------------------------------------

    def slices(self, table=None) -> list[str]:
        """The slices these pairs name, deduplicated, in declaration order.

        Order-preserving rather than ``set``-derived: this is splatted into
        ``dataset(*collator.slices())`` and ``data(*collator.slices())``, where
        position decides the order sources are zipped in.

        Given a *table*, each slice must be one it holds --
        ``table.slices(recursive=self.var.recursive)`` -- or this raises.
        """
        seen = {}
        for role in self.var.columns:
            for pair in self.pairs(role):
                seen[pair[0]] = None
        names = list(seen)
        if table is not None:
            held = table.slices(recursive=self.var.recursive)
            missing = [s for s in names if s not in held]
            if missing:
                upstream = [s for s in missing if s in table.slices(recursive=True)]
                hint = (f"; {upstream} are upstream of it -- a collator reads those only with recursive=True"
                        if upstream and not self.var.recursive else "")
                raise KeyError(f"{type(self).__name__}: {table.__class__.__name__} {getattr(table, 'tag', None)!r} "
                               f"holds no slice {missing}; it holds {list(held)}{hint}")
        return names

    def pairs(self, role: str) -> tuple[tuple[str, ...], ...]:
        """The pairs playing *role*, each in full form; empty when the collator declares no such role.

        A pair may be declared as a bare name or a one-element sequence, both
        of which mean the column of the same name. A role's value may be one
        pair as a bare tuple of strings -- a partition's ``stratifyby``, say.
        """
        value = self.var.columns.get(role) or ()
        if isinstance(value, tuple) and value and all(isinstance(v, str) for v in value):
            value = (value,)
        return self._norm_pairs_(value)

    # 3. Accessors ---------------------------------------------------------

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self.var.columns)

    @property
    def signal_pairs(self) -> tuple[tuple[str, ...], ...]:
        """The ``'signals'`` pairs, as :meth:`pairs` gives them."""
        return self.pairs('signals')

    @property
    def label_pairs(self) -> tuple[tuple[str, ...], ...]:
        """The ``'labels'`` pairs, as :meth:`pairs` gives them; empty when there are none."""
        return self.pairs('labels')

    # 4. Helpers -----------------------------------------------------------

    @classmethod
    def _norm_pairs_(cls, pairs) -> tuple[tuple[str, ...], ...]:
        """Every pair normalized, a triple naming several keys expanded to one per key."""
        out = []
        for pair in pairs:
            deeper = isinstance(pair, (list, tuple)) and (
                len(pair) > 2 or (len(pair) == 2 and isinstance(pair[1], (list, tuple))))
            if not deeper:
                out.append(cls._norm_pair_(pair))
                continue
            # Lists side by side, tuples deeper -- as dataset() reads a request.
            s_name, specs = slice_spec(tuple(pair))
            for column, keys in specs:
                if keys is None:
                    out.append((s_name, column))
                    continue
                for path in (keys if isinstance(keys, list) else [keys]):
                    out.append((s_name, column, path[0] if len(path) == 1 else path))
        return tuple(out)

    @staticmethod
    def _norm_pair_(pair: Any) -> tuple[str, str]:
        if isinstance(pair, (list, tuple)):
            if len(pair) >= 2:
                return str(pair[0]), str(pair[1])
            elif len(pair) == 1:
                return str(pair[0]), str(pair[0])
        return str(pair), str(pair)

    @staticmethod
    def _as_array_(val):
        if val is None:
            return None
        if torch is not None and isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        arr = np.array(val)
        if hasattr(arr, 'flags') and not arr.flags.writeable:
            arr = np.copy(arr)
        return arr

    @classmethod
    def _pick_pair_(cls, row, pair, what, allow_none: bool = False):
        """The value a normalized pair -- or triple -- names in *row*."""
        return cls._pick_(row, pair[0], pair[1], what, key=pair[2] if len(pair) > 2 else None, allow_none=allow_none)

    @staticmethod
    def _pick_(row, s_name, c_name, what, key=None, allow_none: bool = False):
        """The value at ``(s_name, c_name)`` in one nested row, or a clear error.

        Exact, with no fallbacks. The previous version walked the row with
        ``next(iter(val.values()))`` when a name did not match, which meant a
        misspelled column, a renamed slice, or a row that had lost its slice
        level all silently produced *some* array -- an arbitrary one -- and
        the build wrote it as if it were the requested feature.
        """
        try:
            slice_row = row[s_name]
        except (KeyError, TypeError):
            if allow_none:
                return None
            raise KeyError(
                f"{what}: row has no slice {s_name!r}; it provides "
                f"{sorted(row) if isinstance(row, dict) else type(row).__name__}"
            ) from None
        try:
            value = slice_row[c_name]
        except (KeyError, TypeError):
            if allow_none:
                return None
            raise KeyError(
                f"{what}: slice {s_name!r} has no column {c_name!r}; it provides "
                f"{sorted(slice_row) if isinstance(slice_row, dict) else type(slice_row).__name__}"
            ) from None
        return project_column(value, key, where=f"{what}: slice {s_name!r} column {c_name!r}", allow_none=allow_none)

    def _collate_batch_(self, batch: dict, norm_pairs) -> np.ndarray:
        """Collate a ``{slice: {column: values}}`` mapping already stacked over rows.

        This is what ``data(*collator.slices(), concat=True)`` hands back, as
        opposed to the list of per-row dicts a DataLoader yields.

        One pair passes its array through untouched, so a single-signal
        collation keeps the shape the slice was written with -- which is what
        a model is then fed. Several are stacked along a new axis 1,
        mirroring the signals axis of the per-sample form.
        """
        what = f"{self.__class__.__name__}._collate_batch_"
        skip_missing = self.var.skip_missing
        arrays = [self._as_array_(self._pick_pair_(batch, pair, what, allow_none=skip_missing)) for pair in norm_pairs]
        if skip_missing and any(a is None for a in arrays):
            if all(a is None for a in arrays):
                return None
        if len(arrays) == 1:
            return arrays[0]
        return np.stack(arrays, axis=1)

    def _collate_pairs_(self, datapoints, pairs) -> np.ndarray:
        if not pairs:
            return np.array([])

        norm_pairs = self._norm_pairs_(pairs)

        if isinstance(datapoints, dict):
            return self._collate_batch_(datapoints, norm_pairs)

        what = f"{self.__class__.__name__}._collate_pairs_"
        skip_missing = self.var.skip_missing
        batch_items = []

        for dp in datapoints:
            dp_signals = [self._as_array_(self._pick_pair_(dp, pair, what, allow_none=skip_missing)) for pair in norm_pairs]
            if skip_missing and any(s is None for s in dp_signals):
                continue

            norm_signals = []
            for sig in dp_signals:
                if sig.ndim == 0:
                    norm_signals.append(sig.reshape(1, 1))
                elif sig.ndim == 1:
                    norm_signals.append(sig.reshape(1, -1))
                else:
                    norm_signals.append(sig)

            if len(norm_signals) == 1 and norm_signals[0].ndim >= 3:
                dp_tensor = norm_signals[0]
            elif norm_signals[0].ndim == 2 and all(x.ndim == 2 for x in norm_signals):
                try:
                    dp_tensor = np.stack(norm_signals, axis=1)
                except ValueError:
                    dp_tensor = np.concatenate(norm_signals, axis=1)
            else:
                dp_tensor = np.stack(norm_signals, axis=1)

            batch_items.append(dp_tensor)

        try:
            return np.stack(batch_items, axis=0)
        except ValueError:
            return np.concatenate(batch_items, axis=0)


class DataslicesUpstream:
    """Slice routing for a tablet or table that reads its own slices and an upstream one's.

    A mixin, ahead of `Datatablet` or `Datatable` in the bases. `Featuretablet` owns
    ``features`` and borrows the sample slices of the `Datatablet` it was built
    from; `Featuretable` does the same over a `Datatable`. Both answer
    `dataset()` and `data()` for either, so both need the same three things:
    work out which block owns a requested slice, keep the caller's order, and
    refuse a name that two blocks both claim.

    The upstream is the VAR field ``upstream``: a block of the same kind --
    a tablet's a tablet, a table's a table -- that this one was built from.
    """

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        # As _Database_ does for shared_slice_columns, and for the same
        # reason: a bare str is iterable, so 'idx' left unnormalised would be
        # read as four one-letter columns.
        self.shared_upstream_column = self._norm_shared_columns_(
            getattr(self, 'shared_upstream_column', None)
        )

    # 2. Declared API ------------------------------------------------------

    def dataset(self, *slices, upstream: list | None = None, mode='map',
                nested=True, columns=None, shared=None, validate_shared=None,
                skip_none=True, zip_validator=None, **kwargs):
        """The requested slices -- this block's and the upstream block's -- zipped.

        Keyed exactly as `_Database_.dataset()`, so a row is
        ``{'features': {layer: value}, sample_slice: {column: value}, ...}``.

        *shared* defaults to this block's ``shared_upstream_column``, and
        *validate_shared* to True when it does -- so whether the features are
        still paired with the samples they were computed from is settled by
        whoever built the block, once, rather than by every consumer of it. A
        column only one of the sources read carries is refused: it would be
        compared against nothing and read as checked.
        """
        if mode not in ('map', 'iter'):
            raise ValueError(
                f"{self.__class__.__name__}.dataset: mode must be 'map' or 'iter', got {mode!r}"
            )
        routed = self._route_(slices, upstream)
        shared, validate_shared = self._shared_defaults_(shared, validate_shared)
        datasets = [owner.datastream(s_name, **kwargs) for owner, s_name, _ in routed]
        names = [s_name for _, s_name, _ in routed]
        per_slice_columns = [cols for _, _, cols in routed]
        if all(c is None for c in per_slice_columns):
            per_slice_columns = columns

        zip_cls = ZipStreamingDataset if mode == 'map' else ZipIterableStreamingDatasets
        return zip_cls(
            *datasets,
            names=names,
            nested=nested,
            columns=per_slice_columns,
            shared=shared,
            validate_shared=validate_shared,
            skip_none=skip_none,
            zip_validator=zip_validator,
            structures=[owner._dict_columns_(s_name) for owner, s_name, _ in routed],
        )

    def data(self, *slices, upstream: list | None = None, nested=True,
             concat=True, **kwargs):
        """The requested slices read whole, keyed as `dataset()` keys one row.

        A table reads its own slice through `Datatable._read_slice_`,
        which already runs over every tablet, so nothing here concatenates tablets
        by hand.
        """
        routed = self._route_(slices, upstream)
        out = {}
        for owner, s_name, cols in routed:
            spec = (s_name, cols) if cols else s_name
            out[s_name] = _Database_.data(owner, spec, concat=concat, **kwargs)[s_name]
        if nested:
            return out
        return {(s_name, c): vals
                for s_name, cols in out.items()
                for c, vals in cols.items()}

    # 4. Helpers -----------------------------------------------------------

    def _shared_defaults_(self, shared, validate_shared):
        """As `_Database_._shared_defaults_`, from `shared_upstream_column` first.

        This block's alignment question spans two blocks -- is feature row *i*
        the features OF sample row *i*? -- so the column that answers it is one
        the upstream slice holds and this block carried through when it was
        built. That is a different declaration from the columns this block's own
        slices share, and it takes precedence over it, since a zip that reaches
        across the two blocks is the one where drift is possible.
        """
        declared = getattr(self, 'shared_upstream_column', None)
        if shared is None and declared is not None:
            shared = declared
            if validate_shared is None:
                validate_shared = True
        return super()._shared_defaults_(shared, validate_shared)

    def _upstream_block_(self):
        return self.var.upstream

    def slices(self, recursive: bool = False):
        """This block's own slices; *recursive*, those of its upstream chain after them."""
        own = super().slices()
        return own if not recursive else tuple(own) + tuple(self._upstream_slices_())

    def _upstream_slices_(self) -> dict:
        """Map of slice_name -> owner_block for all available slices across the upstream chain."""
        owners = {}
        curr = self._upstream_block_()
        while curr is not None:
            for s in curr.slices():
                if s not in owners:
                    owners[s] = curr
            if hasattr(curr, '_upstream_block_'):
                curr = curr._upstream_block_()
            else:
                break
        return owners

    @staticmethod
    def _norm_items_(slice_columns):
        """``*slice_columns`` as an ordered ``[(slice, columns | None)]`` list, as `slice_spec` reads each."""
        items = list(slice_columns)
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            first = items[0]
            # One tuple is one request -- (slice, ...) -- and one list several.
            if isinstance(first, list) or not (len(first) >= 2 and isinstance(first[0], str)):
                items = list(first)
        return [slice_spec(item) for item in items]

    def _route_(self, slice_columns, upstream=None):
        """Resolve a request into an ordered ``[(block, slice, columns)]`` list.

        Raises when an upstream block declares a slice this block also owns.
        Rows are keyed by slice name, so two blocks claiming one name have no
        way to both appear in a row -- and silently preferring either one is
        how a caller ends up reading features while believing it asked for
        samples.
        """
        what = self.__class__.__name__
        own = tuple(self.slices())
        up_owners = self._upstream_slices_()
        up = tuple(up_owners.keys())

        clash = sorted(set(own) & set(up))
        if clash:
            raise KeyError(
                f"{what}: upstream declares slice(s) "
                f"{clash}, which this block also owns ({list(own)}). A row is "
                f"keyed by slice name and cannot hold both -- rename the "
                f"upstream slice."
            )

        items = self._norm_items_(slice_columns) or [(s, None) for s in own]
        if upstream:
            asked = {s for s, _ in items}
            items = items + [(str(s), None) for s in upstream if str(s) not in asked]

        routed, seen = [], {}
        for s_name, cols in items:
            if s_name in own:
                owner = self
            elif s_name in up_owners:
                owner = up_owners[s_name]
            else:
                raise KeyError(
                    f"{what}: unknown slice {s_name!r}; "
                    f"available slices are {list(own) + list(up)}"
                )
            if s_name in seen:
                pos = seen[s_name]
                _, _, prev = routed[pos]
                merged = None if (prev is None or cols is None) else \
                    merge_column_specs(prev + cols)
                routed[pos] = (owner, s_name, merged)
            else:
                seen[s_name] = len(routed)
                routed.append((owner, s_name, None if cols is None else merge_column_specs(cols)))
        for owner, s_name, cols in routed:
            owner._check_column_keys_(s_name, cols)
        return routed
