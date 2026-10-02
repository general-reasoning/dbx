"""datapoints — Datatab / Datatable blocks over MDS slices."""

from __future__ import annotations

import contextlib
import functools
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
    DatajournalFrame,
    Datastack,
    TopicMarkerMeta,
    _NotedMarkerMeta_,
    forward_property,
    forming_with_journal,
    is_topicmarker,
)
from .dataparts import callable_executor
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
    project_column,
    slice_spec,
    read_mds_shard,
    reader_from_json,
)

#: Topic marker indicating an MDS slice stream directory inside a block.
SLICETOPIC = 'SLICETOPIC'


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

    The declaration is what `Datatab.slice_writers` writes, so it takes no
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
    """Base class for sliced datapoint blocks (Datatab and Datatable).

    A **slice** is one independently-readable MDS stream directory inside a block.
    Slices are declared via topics marked with `SLICETOPIC`.

    A subclass declaring TOPICS constructs its TOPICS dictionary explicitly if extending
    its base class's topics::

        class BaseTab(Datatab):
            TOPICS = {'samples': SLICETOPIC, 'meta': 'meta.json'}

        class SubTab(BaseTab):
            TOPICS = {'report': 'report.json', **BaseTab.TOPICS}
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

    def slices(self):
        """The names of this block's slice topics, in declaration order.

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
             columns=None, **kwargs):
        """Every row of the named slices, keyed exactly as `dataset()` keys one row.

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
        **kwargs
            Passed to `_read_slice_()`.
        """
        names, per_slice_columns = self._parse_slice_columns_(slice_columns, columns)

        out = {}
        for pos, name in enumerate(names):
            rows = self._read_slice_(name, **kwargs)
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
            for c, keys in specs:
                where = f"{self.__class__.__name__}.data: slice {name!r} column {c!r}"
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
        if hasattr(self, 'n_tabs'):
            for idx in range(self.n_tabs):
                tab = self.tab(idx)
                try:
                    with tab.fs.open(tab.slice_index_path(slice), 'r') as f:
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

    def _slice_declaration_(self, slice) -> dict:
        """*slice*'s declared schema wherever it is declared -- here, or on a table's tabs -- else ``{}``.

        A table's slices are its tabs' topics, not its own, and every tab of
        one table is one TAB class, so the first tab speaks for them all.
        """
        try:
            return self.declared_schema(slice) or {}
        except KeyError:
            pass
        n_tabs = getattr(self, 'n_tabs', 0) or 0
        if n_tabs and hasattr(self, 'tab'):
            try:
                return self.tab(0).declared_schema(slice) or {}
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

    def _tab_stream_(self, tab, slice, local: str | None = None) -> Stream:
        """One tab's slice as a `Stream`, cached under *local* when it is remote.

        A local tab is its own cache and *local* is ignored, as in
        `open_datastream()`. A remote one MUST be given a cache directory of its
        own: left without, `Stream` invents `{tmpdir}/{blake2s(remote)}` and
        REFUSES to reuse it, so the second open of that tab -- a second process,
        a second run, a retry after a crash -- dies with "Could not create a
        temporary local directory ... already exists". Hence the error rather
        than a default here: the caller knows which cache this belongs in, and
        every path that omitted one was a collision waiting to happen.
        """
        index_dir = tab.path(*slice.split('/'))
        scheme = urllib.parse.urlparse(index_dir).scheme
        if scheme in ('', 'file'):
            return Stream(local=index_dir.removeprefix('file://'))
        remote = abfs_to_mds_azure(index_dir) if scheme in ('abfs', 'abfss') else index_dir
        if local is None:
            raise ValueError(
                f"{self.__class__.__name__}._tab_stream_: {index_dir} is remote, so it "
                f"needs a local cache directory of its own; pass local="
            )
        os.makedirs(local, exist_ok=True)
        return Stream(remote=remote, local=local)

    def _tab_streams_(self, slice, local: str):
        """One `Stream` per tab, each cached in its own subdirectory of *local*.

        Subdivided by tab, because `StreamingDataset` takes either `streams=` or
        `local=` and never both -- so the cache directory this class computes
        cannot be handed to the dataset, only to the streams under it -- and
        because two streams sharing one directory is itself the collision.
        Named by the tab's hash: unique per tab, and the same across runs, so a
        cache is reused rather than rebuilt.
        """
        return [self._tab_stream_(tab, slice, local=os.path.join(local, tab.hash[:12]))
                for tab in (self.tab(idx) for idx in range(self.n_tabs))]

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


class Datatab(_Database_):
    """One tab of a `Datatable`: a Datablock writing MDS slices."""

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

        The write-time count in `slice_writers` catches a tab that wrote its
        slices unequally. This reads the finished ``index.json`` files instead
        of trusting the write path, so it also answers for a tab that arrived by
        copy or redirection, or whose upload stopped half way -- and `TabMaker`
        calls it after every build and refuses the tab if it says no.

        On the tab and not on `_Database_`: a table's `shard_sizes()` opens
        the index of every tab, so the same override on the table would turn one
        validate() into a read per tab per slice. It needs no such thing -- a
        table is valid only when every tab of it validated.
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
        # The count is what catches a tab that wrote one slice and skipped
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
            # A missing shard says where, not why: say why, when it is that this tab is not valid.
            why = self.why_invalid()
            if why is None:
                raise
            raise FileNotFoundError(
                f"{self.anchorkeypath}: cannot read slice {slice!r}: this tab is not valid: {why}"
            ) from e

    def _upload_slice_(self, local_dir, target_dir):
        names = sorted(os.listdir(local_dir))
        for name in [n for n in names if n != 'index.json'] + \
                    [n for n in names if n == 'index.json']:
            local_path = os.path.join(local_dir, name)
            target_path = os.path.join(target_dir, name)
            self.fs.makedirs(os.path.dirname(target_path), exist_ok=True)
            self.fs.put_file(local_path, target_path)


def DatatableTab(table, idx, tag=None, **spec):
    if isinstance(table, str) and (table.startswith('$') or table.startswith('@') or table.startswith('#')):
        from .dataparts import eval as dbx_eval
        table = dbx_eval(table)
    return table(idx, tag=tag, **spec)


class Datatable(_Database_, Datastack):
    """A table of Datatabs, sliced the same way as its tabs.

    A table's TOPICS only contains what the table itself owns: the structural
    topic ``done`` and any extra file topics the subclass
    declares (such as ``bag_lens``). The tab's slice topics are NOT merged into
    the table's TOPICS -- they belong to the tab, not the table.

    The table's `slices()` is derived from ``TAB``'s slice topics rather than
    from ``TOPICS``, so slice routing (``data()``, ``dataset()``,
    ``valid_slice()``) continues to work without polluting ``TOPICS``.

    The tab's ordinary (non-slice) topics are written into each tab under the
    tab's own key; the table has nothing at those paths.

    The TAB is part of the table's identity: its type names it, ``TAB=<fqcn>``,
    after the spec -- so ``TAB =`` another class is another table. See
    `Datastack._type_entries_`; ``with_block=False`` gives the type under the
    rules from before, as a ``Datatable.Specialization``'s ``TAB=None`` does.

    KNOWN GAP -- THE TAB IS NAMED BY ITS FQCN, NOT ITS CONTENTS
    -----------------------------------------------------------
    Edit the TAB class IN PLACE -- a topic added or respelled, a VERSION bump,
    a VAR field with a default -- and every tab re-keys while the table does
    not:

    * the table's old ``done`` marker still answers, so ``valid()`` is True;
    * ``build()`` therefore skips the table -- after adopting, block by block,
      whatever the TAB's SPECIALIZATIONS resolve to (see `Datastack.build`) --
      and builds none of the topics the re-keyed tabs still owe;
    * a tab without a specialization back to its old identity reads as
      unbuilt, and one with a partial specialization owes what it did not
      cover, while the table reports itself built.

    After such an edit, rebuild the table's tabs explicitly -- build each tab
    (``table.tab(i).build()``), or clear the table's ``done`` and build it.
    The ways out, not yet chosen: `valid()` requiring valid tabs (a check per
    tab), or a stack's build running whenever a block owes topics.
    """

    TAB = None

    #: The topics the table machinery itself writes and reads: the sentinels
    #: recording which tabs are built, and the marker that says the stack
    #: completed. Subclasses extending TOPICS explicitly include
    #: Datatable.TOPICS if desired.
    #:
    #: A ``tabs`` directory was declared here too, and nothing ever wrote into
    #: it: a tab is config-addressed on the table's own url and lands beside the
    #: table rather than inside it. A subclass that wants its tabs nested still
    #: declares ``tabs`` itself and roots them there -- which is what it always
    #: was, an addressable location rather than something the machinery used.
    TOPICS = {'done': DATAFILE('done')}

    @dataclass(frozen=True, repr=False, eq=False)   # the base's repr (note last) and equality
    class Specialization(Datablock.Specialization):
        """A Datablock's Specialization, and the TAB the narrower table's type names.

        A table's BLOCK is its TAB, so this is `Datastack.Specialization` with
        the field named for what it is. *TAB*: SAME for this table's own; a
        string for another fqcn; None for a table built before a table's type
        named its TAB at all.
        """
        TAB: str | SAME | None = SAME

        __hash__ = Datablock.Specialization.__hash__

        def __post_init__(self):
            super().__post_init__()
            if not (self.TAB is SAME or self.TAB is None or (isinstance(self.TAB, str) and self.TAB)):
                raise TypeError(f"Specialization TAB= is SAME, None or a TAB fqcn, got {self.TAB!r}")

        @property
        def _block_(self):
            return self.TAB

    #: None: a table's own topics are markers over its tabs, cheaply written
    #: again; what is worth reaching are its tabs, each by the TAB's own
    #: specializations.
    SPECIALIZATIONS = []

    Tab = staticmethod(DatatableTab)


    @dataclass
    class VAR(Datastack.VAR):
        datapoints_per_row: int = 1

    validate_tab = Datastack.validate_block
    validate_block = Datastack.validate_block

    UNSAFE_clear_tab = Datastack.UNSAFE_clear_block
    UNSAFE_clear_tabs = Datastack.UNSAFE_clear_blocks

    class TabMaker(Datastack.BlockMaker):
        """Lightweight callable that forms and optionally builds a tab."""

        def __init__(self, table=None, tab_idx: int | None = None, **kwargs):
            if isinstance(table, int) and tab_idx is None:
                tab_idx = table
                table = None
            super().__init__(tab_idx)
            self.table = table
            self.tab_idx = tab_idx
            self.kwargs = kwargs

        def __call__(self, table=None, *, build=True, journal=None):
            tbl = table if table is not None else self.table
            # Formed -- by __block__, then again by _adopt_'s .set() -- against the
            # journal the table read once and the executor handed this worker.
            with forming_with_journal(journal), tbl._block_specializations_in_force_():
                if tbl is not None and tbl.valid_tab(self.idx):
                    return {'tab_idx': self.idx, 'skipped': True}
                tab = tbl.__block__(self.idx, **self.kwargs)
                tab = tbl._adopt_(tab, keyby=True)
            skipped = tab.valid()
            if build and not skipped:
                tab.build()
                if tbl is not None and hasattr(tbl, 'validate_tab'):
                    validated = tbl.validate_tab(self.idx)
                else:
                    validated = tab.validate()
                if not validated:
                    raise ValueError(f"Tab {self.idx} of {tbl} failed to validate")
            result = {'tab_idx': self.idx, 'skipped': skipped}
            del tab
            gc.collect()
            return result

    # 1. Protocol and hooks ------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        """A table's TAB is its BLOCK: declaring one declares the other.

        A table may still name a BLOCK of its own, and then its TAB must be one.
        """
        super().__init_subclass__(**kwargs)
        tab = cls.__dict__.get('TAB')
        if tab is None:
            return
        if isinstance(tab, forward_property):
            # Declared forward, its TAB is each instance's -- and so is its BLOCK.
            if 'BLOCK' not in cls.__dict__:
                cls.BLOCK = tab.named('BLOCK')
            return
        if 'BLOCK' in cls.__dict__ and cls.BLOCK is not None:
            if not (isinstance(tab, type) and issubclass(tab, cls.BLOCK)):
                raise TypeError(
                    f"{cls.__qualname__}.TAB = {getattr(tab, '__name__', tab)!r} is not a "
                    f"{cls.BLOCK.__name__}, the BLOCK it declares"
                )
        else:
            cls.BLOCK = tab

    def _type_entries_(self, specialization=None, *, with_block: bool = True) -> dict:
        """A stack's entries, its BLOCK named for what a table's is: ``TAB=<fqcn>``."""
        entries = super()._type_entries_(specialization, with_block=with_block)
        return {('TAB' if k == 'BLOCK' else k): v for k, v in entries.items()}

    def __init__(self, *args, cache=None, cache_limit=None, filter_built_tabs: bool = False,
                 use_tab_specializations: 'bool | str | None' = None, **kwargs):
        # A table's blocks are its tabs, so use_tab_specializations IS
        # use_block_specializations -- either may be given, both if they agree.
        block = kwargs.get('use_block_specializations')
        if use_tab_specializations is not None:
            if block is not None and block != use_tab_specializations:
                raise ValueError(
                    f"{type(self).__name__}: use_tab_specializations={use_tab_specializations!r} "
                    f"and use_block_specializations={block!r} disagree; they are one setting"
                )
            kwargs['use_block_specializations'] = use_tab_specializations
        super().__init__(*args, cache=cache, cache_limit=cache_limit, filter_built_tabs=filter_built_tabs, **kwargs)

    def __tab__(self, idx: int, *, tag=None, **spec) -> Datatab:
        if self.TAB is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set TAB = <Datatab subclass>"
            )
        # No journal handed down: forming a tab resolves nothing -- a tab's
        # specializations are installed by its build(), and a table's build
        # hands its tab-building callables the one journal it read.
        tab_specs = getattr(self, 'TAB_SPECIALIZATIONS', None) or getattr(self, 'TAB_SPECIALIZATION', None) or getattr(self, 'BLOCK_SPECIALIZATIONS', None) or getattr(self, 'BLOCK_SPECIALIZATION', None)
        return self.TAB(
            # The table's own url, RAW -- the specline it was given, not what
            # that resolved to -- so a relocatable table stays relocatable tab
            # by tab. Without it a tab fell back to DBX_ROOT, and a table built
            # anywhere else (a test's tmp_path, a second lake) wrote its tabs to
            # an unrelated root, where they were then looked for in vain.
            datalake=self._datalake_,
            storage_options=self.storage_options,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=spec,
            SPECIALIZATIONS=tab_specs,
            tag=tag if tag is not None else f"tab_{idx:06d}",
        )

    def __block__(self, idx: int) -> Datatab:
        return self.__tab__(idx)

    def __split__(self, *args, **kwargs):
        # A table that declares `tab_paths` in its own TOPICS -- the table
        # topic from before the manifest -- gets the directory, empty: a topic declared and never
        # there would read, to anything resolving a specialization of this
        # table later, as a build that has been cleared. Only when it is THIS
        # table's: under a redirection covering it, `path(ensure_dirpath=True)`
        # refuses to create a directory inside another block's data.
        if 'tab_paths' in self.topics() and 'tab_paths' in self.ownedtopics():
            self.path('tab_paths', ensure_dirpath=True)
        n = self.n_tabs
        self.log.info(
            "%s: %d tabs x %d slices %s",
            self.__class__.__name__, n, len(self.slices()), list(self.slices()),
        )
        devices = getattr(self, '_devices', None) or getattr(self, 'devices', None)
        device_mapping = kwargs.get('device_mapping', None)
        if device_mapping is not None and isinstance(device_mapping, dict):
            block_device = device_mapping.get('block_device', None)
            makers = [
                self.TabMaker(self, idx, device=block_device[idx])
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
                self.TabMaker(self, idx, device=block_device[idx])
                for idx in range(n)
            ]
        else:
            makers = [self.TabMaker(self, idx) for idx in range(n)]
        return makers, dict(build=True)

    def __build__(self, *args, **kwargs):
        callables, callable_kwargs = self.__split__(*args, **kwargs)
        if not callables:
            return self.__stack__([])

        filter_built = getattr(self, 'filter_built_tabs', False)
        if filter_built:
            work_stealing_state = getattr(self, 'work_stealing', False)
            self.log.info(
                f"Building {self.__class__.__name__}: filtering {len(callables)} tabs using "
                f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}, work_stealing={work_stealing_state}"
            )
            validity = self.valid_tabs()

            to_build_callables = []
            callable_results = []

            for i, (c, is_valid) in enumerate(zip(callables, validity)):
                idx = getattr(c, 'tab_idx', getattr(c, 'idx', i))
                if is_valid:
                    callable_results.append({'tab_idx': idx, 'skipped': True})
                else:
                    to_build_callables.append(c)

            self.log.info(
                f"{self.__class__.__name__}: {len(callables) - len(to_build_callables)}/{len(callables)} tabs already valid, "
                f"building {len(to_build_callables)} tabs"
            )
        else:
            to_build_callables = callables
            callable_results = []

        if to_build_callables:
            build_exec_kwargs = self._executor_kwargs_(
                tag=f"EXECUTING {len(to_build_callables)} callables [{self.__class__.__name__}]"
            )
            build_executor = self.executor_cls(**build_exec_kwargs)
            callable_kwargs = self._with_build_journal_(to_build_callables, callable_kwargs)
            built_results = build_executor.exec_callables(to_build_callables, self, **callable_kwargs)
            callable_results.extend(built_results)

        callable_results.sort(key=lambda r: r.get('tab_idx', 0))
        result = self.__stack__(callable_results)
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return result

    def __stack__(self, results=None):
        n_tabs = n_skipped = 0
        for result in (results or []):
            if result is None:
                continue
            n_tabs += 1
            n_skipped += bool(result.get('skipped'))
        self.log.info(
            "%s.__stack__: %d tabs (%d already built)",
            self.__class__.__name__, n_tabs, n_skipped,
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

    #: Slices come from TAB, not from this table's own TOPICS.
    def slices(self):
        """The TAB's slices: a table declares none of its own.

        Slice topics belong to the tab, so a table reads them off ``TAB``
        rather than out of its own TOPICS -- which is what keeps them out of
        the table's TOPICS while leaving slice routing (`data()`, `dataset()`,
        `valid_slice()`) working.

        Falls back to its own TOPICS when TAB is unset or is not a
        `_Database_` -- as for :class:`DatatablePart`, which overrides
        TOPICS per instance and computes its TAB dynamically.
        """
        tab = getattr(self, 'TAB', None)
        if isinstance(tab, type) and issubclass(tab, _Database_):
            return _Database_._find_slice_topics_(getattr(tab, 'TOPICS', None))
        return _Database_._find_slice_topics_(getattr(self, 'TOPICS', None))

    def read(self, *topicpath):
        """As `Datablock.read()`, but slice names bypass the TOPICS guard.

        Slice topics are not in this table's TOPICS (they belong to the tab),
        so the base `read()` would reject them with a KeyError.  Slice reads
        are valid, they just skip the guard and fall through to `__read__`.
        """
        topicpath = self._normtopic_(topicpath)
        topic_str = '/'.join(topicpath)
        if topic_str in self.slices():
            # Bypass _topicnode_: slices are not in TOPICS but are valid reads.
            return self.__read__(*topicpath)
        return super().read(*topicpath)

    def valid_tab(self, i: int, validation: str | None = None) -> bool:
        """Whether tab *i* is valid: `Datastack.valid_block`, by tab."""
        return Datastack.valid_block(self, i, validation=validation)

    valid_block = valid_tab

    def redirected_tab(self, i: int) -> bool:
        """Return whether the tab at index *i* is redirected."""
        return self.tab(i).redirected()

    redirected_block = redirected_tab

    def valid_tabs(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, validation: str | None = None, **kwargs) -> pd.Series:
        """Return a pandas Series of booleans, one per tab, indicating validity (parallelized): `Datastack.valid_blocks`, by tab."""
        return self.valid_blocks(parallelization=parallelization, n_workers=n_workers, false_only=false_only, true_only=true_only, validation=validation, **kwargs)

    def tabs_redirected(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, **kwargs) -> pd.Series:
        """Whether each tab is redirected: `Datastack.blocks_redirected`, by tab."""
        return self.blocks_redirected(parallelization=parallelization, n_workers=n_workers, false_only=false_only, true_only=true_only, **kwargs)

    #: The name `tabs_redirected` had first.
    redirected_tabs = tabs_redirected

    def get_tab_redirections(self, parallelization: str | None = None, n_workers: int | None = None,
                             journal=None, redirected_only: bool = False, **kwargs) -> pd.Series:
        """Each tab's `Redirection`, or None: `Datastack.get_block_redirections`, by tab."""
        return self.get_block_redirections(parallelization=parallelization, n_workers=n_workers,
                                           journal=journal, redirected_only=redirected_only, **kwargs)

    def find_tab_specializations(self, parallelization: str | None = None, n_workers: int | None = None,
                                 journal=None, found_only: bool = False, **kwargs) -> pd.Series:
        """`Datastack.find_block_specializations`, by tab."""
        return self.find_block_specializations(parallelization=parallelization, n_workers=n_workers,
                                               journal=journal, found_only=found_only, **kwargs)

    def specialize_tabs(self, parallelization: str | None = None, n_workers: int | None = None,
                        journal=None, **kwargs) -> pd.Series | None:
        """`Datastack.specialize_blocks`, by tab."""
        return self.specialize_blocks(parallelization=parallelization, n_workers=n_workers,
                                      journal=journal, **kwargs)

    def UNSAFE_clear_tab_redirections(self, *, OVERRIDE: bool = False, parallelization: str | None = None,
                                      n_workers: int | None = None, **kwargs) -> pd.Series:
        """`Datastack.UNSAFE_clear_block_redirections`, by tab."""
        return self.UNSAFE_clear_block_redirections(OVERRIDE=OVERRIDE, parallelization=parallelization,
                                                    n_workers=n_workers, **kwargs)

    def validate_tabs(
        self,
        parallelization: str | None = None,
        n_workers: int | None = None,
        work_stealing: bool | None = None,
        false_only: bool = False,
        true_only: bool = False,
        **kwargs,
    ) -> pd.Series:
        """Return a pandas Series of booleans, one per tab, indicating validation result (parallelized)."""
        return self.validate_blocks(
            parallelization=parallelization,
            n_workers=n_workers,
            work_stealing=work_stealing,
            false_only=false_only,
            true_only=true_only,
            **kwargs,
        )

    def find_tabs(self, signature=None, *patterns, tag=None, path=None, parallelization: str | None = None, n_workers: int | None = None, work_stealing: bool | None = None, **kwargs) -> list[int]:
        """Return a list of indices of all tabs matching the given signature, tag, and/or path pattern(s) (parallelized)."""
        return self.find_blocks(signature, *patterns, tag=tag, path=path, parallelization=parallelization, n_workers=n_workers, work_stealing=work_stealing, **kwargs)

    def tab_datajournal(self, **kwargs) -> DatajournalFrame | None:
        """Return the DatajournalFrame for child tabs, or None if no tabs exist or journal fails to load."""
        return self.block_datajournal(**kwargs)

    def valid_slice(self, slice) -> bool:
        return all(
            self.tab(idx).valid_slice(slice) for idx in range(self.n_tabs)
        )

    def tab(self, idx: int) -> Datatab:
        return self.block(idx)

    def tabs(self) -> list:
        return self.blocks()

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

    # 3. Accessors ---------------------------------------------------------

    @property
    def tab_redirections(self) -> pd.Series:
        """`Datastack.block_redirections`, by tab."""
        return self.block_redirections

    @property
    def use_tab_specializations(self):
        """`use_block_specializations`, by its table name."""
        return getattr(self, 'use_block_specializations', None)

    @property
    def n_tabs(self) -> int:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement n_tabs"
        )

    @property
    def n_blocks(self) -> int:
        return self.n_tabs

    # 4. Helpers -----------------------------------------------------------


    def _topics_signature_(self, topics=None, *, declared=None):
        """Own TOPICS segments -- plus, in a sentinel declaration, the TAB's slices.

        A table's slices are the TAB's declaration and not the table's, and a
        block has no business carrying another's topics in its own identity: the
        TAB is already part of this table, so a differently-sliced TAB is already
        a different table. A table that declares its own topics with the markers
        carries only its own.

        The accumulating form is kept for every table spelled the older way, and
        byte-identically (slice topics rendered as ``topic:<name>=SLICETOPIC``),
        so no existing hash moves.
        """
        # Own (non-slice) topics come first, in declaration order. *topics*
        # restricts THOSE -- a specialization names this table's topics, not
        # the TAB's slices, which are the TAB's declaration and accumulate here
        # whatever subset of its own a table is being rendered under.
        own = super()._topics_signature_(topics, declared=declared)
        # The era of the declaration being RENDERED: a table reconstructed from a
        # sentinel-era declaration accumulated its TAB's slices then, whatever
        # this class is spelled in now.
        if self._modern_topics_(self.TOPICS if declared is None else declared):
            return own
        # Then the TAB's slice topics, in the same format Datastack uses.
        tab = self.TAB
        if isinstance(tab, type) and issubclass(tab, _Database_):
            # TAB is a class here, and slices() is an instance method as
            # topics() is, so the shared helper does the work rather than an
            # unbound call.
            slice_segments = tuple(
                f"topic:{name}=SLICETOPIC"
                for name in _Database_._find_slice_topics_(getattr(tab, 'TOPICS', None))
            )
        else:
            slice_segments = ()
        return own + slice_segments

    def _block_class_(self):
        return getattr(self, 'BLOCK', None) or getattr(self, 'TAB', None)

    def _read_slice_(self, slice, *, tabs=None, **kwargs):
        indices = range(self.n_tabs) if tabs is None else tabs
        datapoints = []
        for idx in indices:
            datapoints.extend(self.tab(idx)._read_slice_(slice, **kwargs))
        return datapoints


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


class TabPartitionScanCallable:
    """Worker callable: one tab's row count and the single value each partition column holds in it."""

    def __init__(self, partition, idx: int):
        self.partition = partition
        self.idx = idx

    def __call__(self):
        part = self.partition
        tab = part.var.datapoint_table.tab(self.idx)
        out = {'idx': self.idx, 'tag': tab.tag, 'rows': None, 'values': {}}
        if part.var.balance == 'rows' or part.var.method == 'largest_first':
            out['rows'] = int(tab.n_rows(part._partition_slice_name_))
        for name in ('groupby', 'stratifyby'):
            spec = getattr(part.var, name)
            if spec is None:
                continue
            distinct = {}
            for value in _column_values_(tab, tuple(spec)):
                distinct.setdefault(_value_key_(value), value)
            out['values'][name] = list(distinct.values())
        return out


class DatatablePartition(Datablock):
    """Partitions a `Datatable`'s tabs into folds according to target *fractions*.

    Whole tabs are dealt to folds; nothing is repacked, and a fold
    (`DatatablePart`) is a view of the table's own tabs. Four choices decide
    how, each answering one question:

    *groupby* -- which tabs must land in the SAME fold? A column spec,
    ``(slice, column[, key, ...])``: tabs sharing its value (a patient's
    slides, say) are dealt as one unit. Default: each tab alone.

    *stratifyby* -- within which categories must the fractions hold? A column
    spec: the fractions are met within each of its values (each cancer type,
    say), not only overall. Default: overall only.

    *balance* -- fractions of what? ``'rows'`` (the default): of rows, read
    from *partition_slice*. ``'tabs'``: of tabs.

    *seed* -- the order units are dealt in. Within each stratum, the units are
    shuffled with it, then each goes to the fold furthest below its target.

    A *groupby* or *stratifyby* column must hold one value throughout each tab:
    the partition reads it whole, and a tab holding several raises
    `NotImplementedError` -- dealing a tab's rows to different folds is a
    capability for the future, repacking them. A tab holding none is skipped:
    in no fold, said loudly, and recorded in ``summary``. A group must lie in
    one stratum.

    *method* ``'largest_first'`` is the partition from before these choices:
    tabs in descending order of rows, each to the fold with the largest deficit
    -- no groups, no strata, no seed. Its builds are reached by a
    specialization, and it takes none of the four.

    Topics: ``tabs`` -- one list of tab indices per fold, which the folds read;
    ``summary`` -- per fold its tags and counts (tabs, rows, per stratum), and
    the tabs skipped and why.
    """

    VERSION = 1

    TOPICS = {
        'tabs': DATAFILE('tabs.json', 'one list of tab indices per fold'),
        'summary': DATAFILE('summary.json', 'per fold its tags and counts, and the tabs skipped'),
    }
    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, topics=SAME, anchor='dbx.datapoints.DatatablePartition',
            note="this very block, stored under its class's old module name, dbx.datapoints, until 2026-10-02"),
        Datablock.Specialization(
            spec={'method': 'largest_first', 'seed': 0, 'groupby': None, 'stratifyby': None, 'balance': 'rows'},
            topics={'tabs': DATAFILE('tabs.json', 'one list of tab indices per fold')},
            note="the partition from before method/seed/groupby/stratifyby/balance: largest_first, by rows"),
        Datablock.Specialization(
            spec={'method': 'largest_first', 'seed': 0, 'groupby': None, 'stratifyby': None, 'balance': 'rows'},
            topics={'tabs': 'tabs.json'},
            note="... and from before its topic was respelled DATAFILE"),
    ]

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_table: Datatable
        fractions: list[float]
        partition_slice: int | str
        method: str = 'random'
        seed: int = 0
        groupby: tuple | list | None = None
        stratifyby: tuple | list | None = None
        balance: str = 'rows'

    # 1. Protocol and hooks ------------------------------------------------

    def __post_init__(self):
        super().__post_init__()
        var = self.var
        if var.method not in PARTITION_METHODS:
            raise ValueError(f"{type(self).__name__}: unknown method {var.method!r}; expected one of {PARTITION_METHODS}")
        if var.balance not in ('rows', 'tabs'):
            raise ValueError(f"{type(self).__name__}: balance must be 'rows' or 'tabs', got {var.balance!r}")
        for name in ('groupby', 'stratifyby'):
            spec = getattr(var, name)
            if spec is not None and not (isinstance(spec, (tuple, list)) and len(spec) >= 2
                                         and all(isinstance(p, str) for p in spec)):
                raise ValueError(f"{type(self).__name__}: {name} must be a column spec "
                                 f"(slice, column[, key, ...]) of strings, got {spec!r}")
        if var.method == 'largest_first' and (var.groupby is not None or var.stratifyby is not None
                                              or var.balance != 'rows' or var.seed != 0):
            raise ValueError(f"{type(self).__name__}: method='largest_first' takes no groupby, stratifyby, "
                             f"balance or seed -- it is the partition from before them; use method='random'")

    def __build__(self):
        table = self.var.datapoint_table
        if table is None:
            raise ValueError(f"{self.__class__.__name__}: VAR.datapoint_table is required")
        if not self.var.fractions:
            raise ValueError(f"{self.__class__.__name__}: VAR.fractions is required")
        scans = self._scan_()
        if self.valid_topic('tabs'):
            # Adopted -- a build from before the summary existed: its folds stand.
            folds_tabs, skipped = json.loads(self.fs.cat(self.path('tabs'))), []
        else:
            folds_tabs, skipped = ((self._largest_first_(scans), []) if self.var.method == 'largest_first'
                                   else self._deal_(scans))
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

    def tabs_indices(self, fold: int | str) -> list[int]:
        data = json.loads(self.fs.cat(self.path('tabs')))
        return data[int(fold)]

    def tabs(self, fold: int | str) -> list[Datatab]:
        indices = self.tabs_indices(fold)
        table = self.var.datapoint_table
        return [table.tab(i) for i in indices]

    def fold(self, fold: int | str) -> DatatablePart:
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

    # 3. Accessors ---------------------------------------------------------

    @property
    def datapoint_table(self) -> Datatable:
        return self.var.datapoint_table

    @property
    def _partition_slice_name_(self) -> str:
        p_slice = self.var.partition_slice
        return self.var.datapoint_table.slices()[p_slice] if isinstance(p_slice, int) else p_slice

    # 4. Helpers -----------------------------------------------------------

    def _scan_(self) -> list[dict]:
        """Each tab's rows and partition-column values, read in parallel; raise for a tab holding several."""
        table = self.var.datapoint_table
        executor = callable_executor(
            getattr(self, 'parallelization', None) or 'inline',
            n_workers=getattr(self, 'n_workers', 1) or 1,
            tag=f"SCANNING {table.n_tabs} tabs [{type(self).__name__}]")
        scans = executor.exec_callables([TabPartitionScanCallable(self, i) for i in range(table.n_tabs)])
        mixed = [(s['tag'], name, vals) for s in scans for name, vals in s['values'].items() if len(vals) > 1]
        if mixed:
            shown = "\n".join(f"  {tag}: {name} holds {len(vals)} values, e.g. {vals[:3]!r}"
                              for tag, name, vals in mixed[:5])
            raise NotImplementedError(
                f"{type(self).__name__}: {len(mixed)} tab(s) hold more than one value of a partition column:\n"
                f"{shown}\n"
                f"A tab is dealt to one fold whole, so its groupby and stratifyby values must be one per tab. "
                f"Dealing a tab's rows to different folds -- repacking them -- is not implemented yet.")
        return scans

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

    def _deal_(self, scans) -> tuple[list[list[int]], list[dict]]:
        """Units -- groups of tabs -- dealt within each stratum, in seeded order, to the fold furthest below target."""
        fractions = [f / sum(self.var.fractions) for f in self.var.fractions]
        skipped, units = [], {}
        for s in scans:
            missing = [name for name in ('groupby', 'stratifyby')
                       if getattr(self.var, name) is not None and s['values'][name] in ([], [None])]
            if missing:
                skipped.append({'idx': s['idx'], 'tag': s['tag'], 'why': f"no {' or '.join(missing)} value"})
                continue
            group = _value_key_(s['values']['groupby'][0]) if self.var.groupby is not None else f"tab:{s['idx']}"
            stratum = _value_key_(s['values']['stratifyby'][0]) if self.var.stratifyby is not None else ''
            unit = units.setdefault(group, {'tabs': [], 'strata': set(), 'weight': 0})
            unit['tabs'].append(s['idx'])
            unit['strata'].add(stratum)
            unit['weight'] += s['rows'] if self.var.balance == 'rows' else 1
        if skipped:
            self.log.warning(
                f"{type(self).__name__}: skipping {len(skipped)} of {len(scans)} tabs, which hold no "
                f"{'/'.join(n for n in ('groupby', 'stratifyby') if getattr(self.var, n) is not None)} "
                f"value -- they are in no fold: {[s['tag'] for s in skipped[:10]]}"
                + (" ..." if len(skipped) > 10 else ""))
        straddling = {g: u['strata'] for g, u in units.items() if len(u['strata']) > 1}
        if straddling:
            g, strata = next(iter(straddling.items()))
            raise ValueError(f"{type(self).__name__}: {len(straddling)} group(s) span more than one stratum, "
                             f"e.g. group {g} in strata {sorted(strata)}: a group is dealt whole, so it must "
                             f"lie in one stratum")
        by_stratum = {}
        for g in sorted(units):
            by_stratum.setdefault(next(iter(units[g]['strata'])), []).append(g)
        rng = np.random.default_rng(self.var.seed)
        folds_tabs = [[] for _ in fractions]
        for stratum in sorted(by_stratum):
            groups = by_stratum[stratum]
            total = sum(units[g]['weight'] for g in groups)
            target = [total * f for f in fractions]
            have = [0.0] * len(fractions)
            for i in rng.permutation(len(groups)):
                unit = units[groups[i]]
                k = max(range(len(fractions)),
                        key=lambda k: (target[k] - have[k]) / target[k] if target[k] > 0 else -np.inf)
                folds_tabs[k].extend(unit['tabs'])
                have[k] += unit['weight']
        return [sorted(f) for f in folds_tabs], skipped

    def _summary_(self, scans, folds_tabs, skipped) -> dict:
        """What each fold holds -- tags, tabs, rows, per stratum -- and what was left out."""
        by_idx = {s['idx']: s for s in scans}

        def stratum(s):
            vals = s['values'].get('stratifyby')
            return None if not vals else vals[0]

        folds = []
        for k, idxs in enumerate(folds_tabs):
            strata = {}
            for i in idxs:
                st = strata.setdefault(str(stratum(by_idx[i])), {'tabs': 0, 'rows': 0})
                st['tabs'] += 1
                st['rows'] += by_idx[i]['rows'] or 0
            folds.append({
                'fold': k,
                'fraction': self.var.fractions[k],
                'tabs': len(idxs),
                'rows': sum(by_idx[i]['rows'] or 0 for i in idxs) if self.var.balance == 'rows' or self.var.method == 'largest_first' else None,
                'strata': dict(sorted(strata.items())) if self.var.stratifyby is not None else None,
                'tags': [by_idx[i]['tag'] for i in idxs],
            })
        return {'method': self.var.method, 'balance': self.var.balance, 'seed': self.var.seed,
                'groupby': self.var.groupby, 'stratifyby': self.var.stratifyby,
                'n_tabs': len(scans), 'folds': folds, 'skipped': skipped}


class DatatablePart(Datatable):
    """A subset of a `Datatable` defined by tab_indices for a fold."""

    SPECIALIZATIONS = [Datatable.Specialization(
        spec={}, topics=SAME, anchor='dbx.datapoints.DatatablePart',
        note="this very block, stored under its class's old module name, dbx.datapoints, until 2026-10-02")]

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

    def slices(self):
        return self.var.partition.datapoint_table.slices()

    def tab(self, idx: int) -> Datatab:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table.tab(real_idx)

    def valid_tab(self, idx: int, validation: str | None = None) -> bool:
        """Whether tab *idx* is valid: this part's manifest, or its table's tab -- see `Datastack.valid_blocks`."""
        validation = self._validation_(validation)
        if validation == 'cross_check' and self._blocks_cross_checked_():
            return True
        real_idx = self.tab_indices[idx]
        full = 'validate' if validation == 'validate' else 'valid'
        return self.var.partition.datapoint_table.valid_tab(real_idx, validation=full)

    valid_block = valid_tab

    def redirected_tab(self, idx: int) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table.redirected_tab(real_idx)

    redirected_block = redirected_tab

    def validate_tab(self, idx: int, **kwargs) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table.validate_tab(real_idx, **kwargs)

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

    # 3. Accessors ---------------------------------------------------------

    @functools.cached_property
    def tab_indices(self) -> list[int]:
        return self.var.partition.tabs_indices(self.var.fold)

    @property
    def datapoint_table(self) -> Datatable:
        return self.var.partition.datapoint_table

    @property
    def datapoints_per_row(self) -> int:
        return getattr(self.var.partition.datapoint_table.var, 'datapoints_per_row')

    @forward_property(Datatab)
    def TAB(self):
        """The partitioned table's TAB: on the class, `Datatab` -- what every part's tabs are."""
        return getattr(self.var.partition.datapoint_table, 'TAB', None)

    @forward_property({})
    def TOPICS(self):
        """The partitioned table's TOPICS: a part is a view of its tabs, and declares nothing of its own.

        On the class, with no table to ask, ``{}``.
        """
        return self.var.partition.datapoint_table.TOPICS

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
        return self.var.partition.datapoint_table._blocks_datalake_()

    def _block_class_(self):
        """A part's tabs are its table's, and so is its BLOCK."""
        table = getattr(getattr(self, 'var', None), 'partition', None)
        table = getattr(table, 'datapoint_table', None)
        return table._block_class_() if table is not None else None

class DataslicesUpstream:
    """Slice routing for a tab or table that reads its own slices and an upstream one's.

    A mixin, ahead of `Datatab` or `Datatable` in the bases. `Featuretab` owns
    ``features`` and borrows the sample slices of the `Datatab` it was built
    from; `Featuretable` does the same over a `Datatable`. Both answer
    `dataset()` and `data()` for either, so both need the same three things:
    work out which block owns a requested slice, keep the caller's order, and
    refuse a name that two blocks both claim.

    The upstream is found in the VAR field ``UPSTREAM_TABS`` names -- a tab or
    a table, whichever the block was built from.
    """

    #: The VAR field(s) that may hold the upstream tab or table, most specific
    #: first; the first one set is the upstream.
    UPSTREAM_TABS: tuple[str, ...] = ()

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
        )

    def data(self, *slices, upstream: list | None = None, nested=True,
             concat=True, **kwargs):
        """The requested slices read whole, keyed as `dataset()` keys one row.

        A table reads its own slice through `Datatable._read_slice_`,
        which already runs over every tab, so nothing here concatenates tabs
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
        for attr in self.UPSTREAM_TABS:
            block = getattr(self.var, attr, None)
            if block is None:
                block = getattr(self, attr, None)
            if block is not None:
                return block
        return None

    def available_slices(self) -> tuple[str, ...]:
        """All slice names available on this block and its upstream chain."""
        return tuple(self.slices()) + tuple(self._upstream_slices_().keys())

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
