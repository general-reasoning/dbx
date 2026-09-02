"""datapoints — DatapointTab / DatapointTable blocks over MDS slices."""

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
        "dbx.datapoints requires PyTorch.  "
        "Install it with:  pip install datablocks[torch]"
    ) from exc

try:
    from streaming import MDSWriter, Stream, StreamingDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datapoints requires mosaicml-streaming.  "
        "Install it with:  pip install datablocks[streaming]"
    ) from exc

from .datablocks import DIRTOPIC, Datablock, Datastack, Datajournal
from .datastreams import (
    ChunkShuffleSampler,
    SharedMemoryManager,
    ZipIterableStreamingDatasets,
    ZipStreamingDataset,
    _ShardSync,
    abfs_to_mds_azure,
    concat_data,
    open_datastream,
    read_mds_shard,
    reader_from_json,
)

#: Topic marker indicating an MDS slice stream directory inside a block.
SLICETOPIC = 'SLICETOPIC'


class DatapointBase(Datablock):
    """Base class for sliced datapoint blocks (DatapointTab and DatapointTable).

    A **slice** is one independently-readable MDS stream directory inside a block.
    Slices are declared via topics marked with `SLICETOPIC`.

    A subclass declaring TOPICS constructs its TOPICS dictionary explicitly if extending
    its base class's topics::

        class BaseTab(DatapointTab):
            TOPICS = {'samples': SLICETOPIC, 'meta': 'meta.json'}

        class SubTab(BaseTab):
            TOPICS = {'report': 'report.json', **BaseTab.TOPICS}
    """

    TOPICS = {}

    # 1. Datablock Protocol Methods ─────────────────────────────────

    def __init__(self, *args, cache_limit=None, **kwargs):
        super().__init__(*args, cache_limit=cache_limit, **kwargs)

    def valid_slice(self, slice) -> bool:
        """True when *slice* has an `index.json` on disk."""
        try:
            return self.fs.exists(self.slice_index_path(slice))
        except Exception:
            return False

    def valid_topic(self, *topicpath):
        """As `Datablock.valid_topic()`, but slices go through `valid_slice()`."""
        topicpath = self._normtopic(topicpath)
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

    # 2. Properties and Accessors ───────────────────────────────────

    def slices(self):
        """The names of this block's slice topics, in declaration order.

        A method rather than a property, mirroring :meth:`topics`, which it is
        the slice-only filter of.

        Derived on each call rather than frozen at class creation, so a TOPICS
        assigned or amended after the class body still reports its slices, and
        an instance overriding TOPICS -- as :class:`DatapointFold` does -- is
        read through.
        """
        return DatapointBase._find_slice_topics(getattr(self, 'TOPICS', None))

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
            Passed to `_read_slice()`.
        """
        names, per_slice_columns = self._parse_slice_columns(slice_columns, columns)

        out = {}
        for pos, name in enumerate(names):
            rows = self._read_slice(name, **kwargs)
            cols = per_slice_columns[pos] if per_slice_columns else None
            if cols is None:
                cols = list(rows[0]) if rows else []
            else:
                missing = [c for c in cols if rows and c not in rows[0]]
                if missing:
                    raise KeyError(
                        f"{self.__class__.__name__}.data: slice {name!r} has no "
                        f"column(s) {missing}; it provides {sorted(rows[0])}"
                    )
            out[name] = {c: concat_data([r[c] for r in rows]) if concat
                         else [r[c] for r in rows]
                         for c in cols}

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

    def _parse_slice_columns(self, slice_columns, columns=None):
        """Normalize a ``*slice_columns`` spec into ``(names, per_slice_columns)``.

        Shared by `dataset()` and `data()` so the two accept exactly the same
        spec: a bare slice name, a ``(slice, column)`` pair, or a
        ``(slice, [columns])`` pair, in any mixture. The two differ in what
        they do with a slice, never in how a caller names one.

        *names* is in the order the slices were asked for -- position decides
        which source is zipped where, so it is derived from the caller's
        sequence and never from a set.
        """
        items = list(slice_columns)
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            first = items[0]
            if isinstance(first, list) or not (len(first) == 2 and isinstance(first[0], str) and first[0] in self.slices()):
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
                if isinstance(item, str):
                    s_name, cols = item, None
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    s_name = str(item[0])
                    c_val = item[1]
                    if isinstance(c_val, (list, tuple)):
                        cols = [str(c) for c in c_val]
                    else:
                        cols = [str(c_val)]
                    has_column_filter = True
                else:
                    raise ValueError(
                        f"{self.__class__.__name__}.dataset: each slice_column entry must be a slice name (str) "
                        f"or a (slice, column) pair, got {item!r}"
                    )

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
                per_slice_columns = [slice_cols_map[name] for name in names]
                if all(c is None for c in per_slice_columns):
                    per_slice_columns = None
            else:
                per_slice_columns = None

        return names, per_slice_columns

    def dataset(
        self,
        *slice_columns,
        mode='map',
        nested=True,
        columns=None,
        shared=None,
        validate_shared=False,
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
            Passing multiple `(slice, col)` tuples for the same slice accumulates their columns.
            If no positional arguments are passed, defaults to all slices with all columns.
        mode : {'map', 'iter'}
            How the slices are read.
        nested : bool
            Row keying, as `ZipBase`. True (the default) gives
            ``{slice: {column: value}}``; False gives ``{(slice, column): value}``.
            Both keep every column of every slice, including a column two
            slices happen to share.
        columns : legacy keyword parameter for backwards compatibility.
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

        names, per_slice_columns = self._parse_slice_columns(slice_columns, columns)

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

    @property
    def cacheroot(self) -> str:
        """Local scratch root for everything streaming: read caches, staged writes."""
        return getattr(self, 'cache', None) or os.path.join(
            self.localroot, 'streaming',
        )

    # 3. Utility and Private Methods ────────────────────────────────

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

    def block_shuffle_sampler(
        self,
        slice: str,
        *,
        block_size: int | None = None,
        chunk_size: int | None = None,
        seed: int = 0,
        fixed_epoch: bool = False,
    ) -> ChunkShuffleSampler:
        """Alias for `chunk_shuffle_sampler` with `block_size` support."""
        size = chunk_size if chunk_size is not None else block_size
        return self.chunk_shuffle_sampler(
            slice,
            chunk_size=size,
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

    @staticmethod
    def _node_is_dirtopic(node):
        """True when node is DIRTOPIC or SLICETOPIC."""
        return node is DIRTOPIC or node == SLICETOPIC or node is SLICETOPIC

    def _is_dir_topic(self, *topicpath):
        """True when the topic resolves to a directory rather than a file."""
        topicpath = self._normtopic(topicpath)
        if not topicpath or topicpath[0] is None:
            return False
        node = self._topicnode(*topicpath)
        return self._node_is_dirtopic(node)

    def _slicenames(self, slices) -> tuple:
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

    def slice_names(self, slices) -> tuple:
        """Normalize a `*slices` varargs tuple; empty means *all* slices."""
        return self._slicenames(slices)

    def _ensure_cacheroot(self, cache=None) -> str:
        cacheroot = cache or self.cacheroot
        os.makedirs(cacheroot, exist_ok=True)
        return cacheroot

    def _read_slice(self, slice, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _read_slice(slice)"
        )

    def _tab_stream(self, tab, slice, local: str | None = None) -> Stream:
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
                f"{self.__class__.__name__}._tab_stream: {index_dir} is remote, so it "
                f"needs a local cache directory of its own; pass local="
            )
        os.makedirs(local, exist_ok=True)
        return Stream(remote=remote, local=local)

    def _tab_streams(self, slice, local: str):
        """One `Stream` per tab, each cached in its own subdirectory of *local*.

        Subdivided by tab, because `StreamingDataset` takes either `streams=` or
        `local=` and never both -- so the cache directory this class computes
        cannot be handed to the dataset, only to the streams under it -- and
        because two streams sharing one directory is itself the collision.
        Named by the tab's hash: unique per tab, and the same across runs, so a
        cache is reused rather than rebuilt.
        """
        return [self._tab_stream(tab, slice, local=os.path.join(local, tab.hash[:12]))
                for tab in (self.tab(idx) for idx in range(self.n_tabs))]

    @staticmethod
    def _find_slice_topics(topics_dict, prefix=()):
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
            if val == SLICETOPIC or val is SLICETOPIC:
                slice_topics.append('/'.join(current) if len(current) > 1 else key)
            elif isinstance(val, dict):
                slice_topics.extend(DatapointBase._find_slice_topics(val, current))
        return tuple(slice_topics)


class DatapointTab(DatapointBase):
    """One tab of a `DatapointTable`: a Datablock writing MDS slices."""

    @dataclass
    class VAR(Datablock.VAR):
        datapoints_per_row: int = 1

    # 1. Datablock Protocol Methods ─────────────────────────────────

    def __init__(self, *args, cache=None, cache_limit=None, **kwargs):
        super().__init__(*args, cache=cache, cache_limit=cache_limit, **kwargs)

    def __build__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __build__(): write every "
            f"slice in {list(self.slices()) or ['...']} in lockstep via self.slice_writers(slices)"
        )

    def __stats__(self, slice, **kwargs) -> dict:
        return {'n_rows': len(self._read_slice(slice))}

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices():
                return self.data(topic_str)
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ override to read {'/'.join(topicpath)!r}"
        )

    # 2. Properties and Accessors ───────────────────────────────────

    # 3. Utility and Private Methods ────────────────────────────────

    @contextlib.contextmanager
    def slice_writers(self, slices, *, stage: bool = None, cache=None,
                      flush_every: int = None, **writer_kwargs):
        """One `MDSWriter` per slice, as `{slice: writer}`.

        Parameters
        ----------
        slices : dict
            `{slice_name: {column: mds_type}}`, one entry per declared slice.
        """
        names = self.slices()
        missing = [name for name in names if name not in slices]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}.slice_writers: no columns for "
                f"slice(s) {missing}; every declared slice must be written"
            )

        if stage is None:
            stage = not self.is_local_fs
        targets = {name: self.path(*name.split('/'), ensure_dirpath=True) for name in names}

        staging = None
        if stage:
            staging = tempfile.mkdtemp(
                prefix=f"{self.__class__.__name__}_",
                dir=self._ensure_cacheroot(cache),
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
        sync = _ShardSync(flush_every) if flush_every else None
        try:
            for name in names:
                writer = MDSWriter(
                    out=outdirs[name], columns=slices[name], **writer_kwargs,
                )
                writers[name] = sync.track(name, writer) if sync else writer
            yield writers
            if sync is not None:
                sync.check_lockstep(self.__class__.__name__)
            for writer in writers.values():
                writer.finish()
            if staging is not None:
                for name in names:
                    self._upload_slice(outdirs[name], targets[name])
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def _read_slice(self, slice, **kwargs):
        return read_mds_shard(
            self.path(*slice.split('/')), self.fs,
            tmpdir=kwargs.pop('cache', None) or self._ensure_cacheroot(), **kwargs,
        )

    def _upload_slice(self, local_dir, target_dir):
        names = sorted(os.listdir(local_dir))
        for name in [n for n in names if n != 'index.json'] + \
                    [n for n in names if n == 'index.json']:
            local_path = os.path.join(local_dir, name)
            target_path = os.path.join(target_dir, name)
            self.fs.makedirs(os.path.dirname(target_path), exist_ok=True)
            self.fs.put_file(local_path, target_path)


def DatapointTableTab(table, idx, tag=None, **spec):
    if isinstance(table, str) and (table.startswith('$') or table.startswith('@') or table.startswith('#')):
        from .dataparts import eval as dbx_eval
        table = dbx_eval(table)
    return table(idx, tag=tag, **spec)


class DatapointTable(DatapointBase, Datastack):
    """A table of DatapointTabs, sliced the same way as its tabs.

    A table's TOPICS only contains what the table itself owns: the structural
    topics (``tabs``, ``done``) and any extra file topics the subclass declares
    (such as ``bag_lens``). The tab's slice topics are NOT merged into the
    table's TOPICS -- they belong to the tab, not the table.

    The table's `slices()` is derived from ``TAB``'s slice topics rather than
    from ``TOPICS``, so slice routing (``data()``, ``dataset()``,
    ``valid_slice()``) continues to work without polluting ``TOPICS``. Pointing
    a table at a differently-sliced TAB still rekeys the table because the TAB
    class is part of :attr:`signature`.

    The tab's ordinary (non-slice) topics are written into each tab under the
    tab's own key; the table has nothing at those paths.
    """

    TAB = None

    #: The topics the table machinery itself writes and reads: the directory
    #: the tabs live in, and the marker that says the stack completed. Subclasses
    #: extending TOPICS explicitly include DatapointTable.TOPICS if desired.
    TOPICS = {'tabs': DIRTOPIC, 'tab_paths': DIRTOPIC, 'done': 'done'}

    #: Slices come from TAB, not from this table's own TOPICS.
    def slices(self):
        """The TAB's slices: a table declares none of its own.

        Slice topics belong to the tab, so a table reads them off ``TAB``
        rather than out of its own TOPICS -- which is what keeps them out of
        the table's TOPICS while leaving slice routing (`data()`, `dataset()`,
        `valid_slice()`) working.

        Falls back to its own TOPICS when TAB is unset or is not a
        `DatapointBase` -- as for :class:`DatapointFold`, which overrides
        TOPICS per instance and computes its TAB dynamically.
        """
        tab = getattr(self, 'TAB', None)
        if isinstance(tab, type) and issubclass(tab, DatapointBase):
            return DatapointBase._find_slice_topics(getattr(tab, 'TOPICS', None))
        return DatapointBase._find_slice_topics(getattr(self, 'TOPICS', None))

    Tab = staticmethod(DatapointTableTab)

    @dataclass
    class VAR(Datastack.VAR):
        datapoints_per_row: int = 1

    # 1. Datastack / Table Protocol Methods ─────────────────────────

    def __init__(self, *args, cache=None, cache_limit=None, filter_built_tabs: bool = False, **kwargs):
        super().__init__(*args, cache=cache, cache_limit=cache_limit, filter_built_tabs=filter_built_tabs, **kwargs)

    def __tab__(self, idx: int, *, tag=None, **spec) -> DatapointTab:
        if self.TAB is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set TAB = <DatapointTab subclass>"
            )
        return self.TAB(
            # The table's own url, RAW -- the specline it was given, not what
            # that resolved to -- so a relocatable table stays relocatable tab
            # by tab. Without it a tab fell back to DBX_ROOT, and a table built
            # anywhere else (a test's tmp_path, a second lake) wrote its tabs to
            # an unrelated root, where they were then looked for in vain.
            url=self.url,
            storage_options=self.storage_options,
            capture_output=self.capture_output,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=spec,
            tag=tag if tag is not None else f"tab_{idx:06d}",
        )

    def __block__(self, idx: int) -> DatapointTab:
        return self.__tab__(idx)

    def _tab_paths_topic(self) -> str | None:
        topics = self.topics()
        if 'tab_paths' in topics:
            return 'tab_paths'
        if 'built_tabs' in topics:
            return 'built_tabs'
        return None

    def __split__(self, *args, **kwargs):
        self.path('tabs', ensure_dirpath=True)
        topic_name = self._tab_paths_topic()
        if topic_name:
            self.path(topic_name, ensure_dirpath=True)
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
            build_exec_kwargs = self._executor_kwargs(
                tag=f"EXECUTING {len(to_build_callables)} callables [{self.__class__.__name__}]"
            )
            build_executor = self.executor_cls(**build_exec_kwargs)
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

    def read(self, *topicpath):
        """As `Datablock.read()`, but slice names bypass the TOPICS guard.

        Slice topics are not in this table's TOPICS (they belong to the tab),
        so the base `read()` would reject them with a KeyError.  Slice reads
        are valid, they just skip the guard and fall through to `__read__`.
        """
        topicpath = self._normtopic(topicpath)
        topic_str = '/'.join(topicpath)
        if topic_str in self.slices():
            # Bypass _topicnode: slices are not in TOPICS but are valid reads.
            return self.__read__(*topicpath)
        return super().read(*topicpath)

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices():
                return self.data(topic_str)
        topic_name = self._tab_paths_topic()
        if topic_name and topicpath == (topic_name,):
            return self.path(topic_name)
        if topicpath == ('tabs',):
            return self.path('tabs')
        if topicpath == ('done',):
            return self.valid()
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ answers only slices, 'tabs', 'built_tabs', 'tab_paths' and 'done'; "
            f"override it to read {'/'.join(topicpath)!r}"
        )

    def valid(self):
        return self.valid_topic('done')

    def _write_tab_path(self, i: int):
        topic_name = self._tab_paths_topic()
        if not topic_name:
            return
        tab_dir = self.path(topic_name, ensure_dirpath=True)
        sentinel_path = os.path.join(tab_dir, f"tab_{i}.path")
        anchorkeypath = self.tab(i).anchorkeypath
        with self.fs.open(sentinel_path, 'w') as f:
            f.write(anchorkeypath)
        if hasattr(self, '_built_tab_set_cache'):
            self._built_tab_set_cache.add(i)

    def _built_tab_set(self) -> set[int]:
        if not hasattr(self, '_built_tab_set_cache'):
            topic_name = self._tab_paths_topic()
            if not topic_name:
                self._built_tab_set_cache = set()
            else:
                try:
                    tab_dir = self.path(topic_name)
                    if not self.fs.exists(tab_dir):
                        self._built_tab_set_cache = set()
                    else:
                        files = self.fs.ls(tab_dir, detail=False)
                        indices = set()
                        for f in files:
                            fname = os.path.basename(f)
                            if fname.startswith('tab_') and fname.endswith('.path'):
                                try:
                                    idx = int(fname.removeprefix('tab_').removesuffix('.path'))
                                    indices.add(idx)
                                except ValueError:
                                    pass
                        self._built_tab_set_cache = indices
                except Exception:
                    self._built_tab_set_cache = set()
        return self._built_tab_set_cache

    def _check_tab_path(self, i: int) -> bool:
        topic_name = self._tab_paths_topic()
        if not topic_name:
            return False
        if i in self._built_tab_set():
            return True
        try:
            tab_dir = self.path(topic_name)
            sentinel_path = os.path.join(tab_dir, f"tab_{i}.path")
            return self.fs.exists(sentinel_path)
        except Exception:
            return False

    def _remove_tab_path(self, i: int):
        topic_name = self._tab_paths_topic()
        if not topic_name:
            return
        try:
            tab_dir = self.path(topic_name)
            for prefix in ('tab_', 'block_'):
                sentinel_path = os.path.join(tab_dir, f"{prefix}{i}.path")
                if self.fs.exists(sentinel_path):
                    self.fs.rm(sentinel_path)
        except Exception:
            pass
        if hasattr(self, '_built_tab_set_cache') and self._built_tab_set_cache is not None:
            self._built_tab_set_cache.discard(i)

    _write_tab_built = _write_tab_path
    _check_tab_built = _check_tab_path
    _remove_tab_built = _remove_tab_path
    _write_block_path = _write_tab_path
    _remove_block_path = _remove_tab_path

    def valid_tab(self, i: int) -> bool:
        if self._tab_paths_topic():
            if self._check_tab_path(i):
                return True
            return self.tab(i).valid()
        return self.tab(i).valid()

    valid_block = valid_tab

    def redirected_tab(self, i: int) -> bool:
        """Return whether the tab at index *i* is redirected."""
        return self.tab(i).redirected()

    redirected_block = redirected_tab

    def valid_tabs(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, **kwargs) -> pd.Series:
        """Return a pandas Series of booleans, one per tab, indicating validity (parallelized)."""
        return self.valid_blocks(parallelization=parallelization, n_workers=n_workers, false_only=false_only, true_only=true_only, **kwargs)

    def redirected_tabs(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, **kwargs) -> pd.Series:
        """Return a pandas Series of booleans, one per tab, indicating redirection (parallelized)."""
        return self.redirected_blocks(parallelization=parallelization, n_workers=n_workers, false_only=false_only, true_only=true_only, **kwargs)

    validate_tab = Datastack.validate_block
    validate_block = Datastack.validate_block

    UNSAFE_clear_tab = Datastack.UNSAFE_clear_block
    UNSAFE_clear_tabs = Datastack.UNSAFE_clear_blocks

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

    def tab_journal(self, **kwargs) -> Datajournal | None:
        """Return the Datajournal for child tabs, or None if no tabs exist or journal fails to load."""
        return self.block_journal(**kwargs)

    def valid_slice(self, slice) -> bool:
        return all(
            self.tab(idx).valid_slice(slice) for idx in range(self.n_tabs)
        )

    def __stats__(self, slice, **kwargs) -> dict:
        return super().__stats__(slice, **kwargs)

    def signature_topics(self):
        """Own TOPICS segments plus the TAB's slice-topic segments.

        Slice topics no longer live in this table's TOPICS, but they must still
        appear in the signature so that pointing a table at a differently-sliced
        TAB changes its hash. The output is byte-identical to what the old
        accumulating behaviour produced (slice topics rendered as
        ``topic:<name>=SLICETOPIC``), so no existing hashes are invalidated.
        """
        # Own (non-slice) topics come first, in declaration order.
        own = super().signature_topics()
        # Then the TAB's slice topics, in the same format Datastack uses.
        tab = self.TAB
        if isinstance(tab, type) and issubclass(tab, DatapointBase):
            # TAB is a class here, and slices() is an instance method as
            # topics() is, so the shared helper does the work rather than an
            # unbound call.
            slice_segments = tuple(
                f"topic:{name}=SLICETOPIC"
                for name in DatapointBase._find_slice_topics(getattr(tab, 'TOPICS', None))
            )
        else:
            slice_segments = ()
        return own + slice_segments


    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def n_tabs(self) -> int:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement n_tabs"
        )

    @property
    def n_blocks(self) -> int:
        return self.n_tabs

    def tab(self, idx: int) -> DatapointTab:
        return self.block(idx)

    def tabs(self) -> list:
        return self.blocks()

    def datastream(self, slice, **kwargs) -> StreamingDataset:
        self.slice_names((slice,))
        cacheroot = self._ensure_cacheroot(kwargs.pop('cache', None))
        cache_dir = kwargs.pop('cache_dir',
                               f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        local = os.path.join(cacheroot, cache_dir)
        os.makedirs(local, exist_ok=True)
        streams = self._tab_streams(slice, local)
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
            return StreamingDataset(**streaming_kwargs)
        except (ValueError, TypeError) as exc:
            SharedMemoryManager.clean_process_shared_memory()
            streaming_kwargs['streams'] = self._tab_streams(slice, local)
            return StreamingDataset(**streaming_kwargs)

    # 3. Private and Utility Methods ────────────────────────────────

    def _read_slice(self, slice, *, tabs=None, **kwargs):
        indices = range(self.n_tabs) if tabs is None else tabs
        datapoints = []
        for idx in indices:
            datapoints.extend(self.tab(idx)._read_slice(slice, **kwargs))
        return datapoints

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

        def __call__(self, table=None, *, build=True):
            tbl = table if table is not None else self.table
            if tbl is not None and tbl.valid_tab(self.idx):
                return {'tab_idx': self.idx, 'skipped': True}

            tab = tbl.__block__(self.idx, **self.kwargs)
            keyby_val = getattr(tbl, 'keyby', None)
            if keyby_val is not None:
                tab = tab.set(keyby=keyby_val)
            skipped = tab.valid()
            if build and not skipped:
                tab.build()
                if tbl is not None and hasattr(tbl, 'validate_tab'):
                    validated = tbl.validate_tab(self.idx)
                else:
                    validated = tab.validate()
                if not validated:
                    raise ValueError(f"Tab {self.idx} of {tbl} failed to validate")
                if hasattr(tbl, '_write_tab_path'):
                    tbl._write_tab_path(self.idx)
                elif hasattr(tbl, '_write_tab_built'):
                    tbl._write_tab_built(self.idx)
            result = {'tab_idx': self.idx, 'skipped': skipped}
            del tab
            gc.collect()
            return result


class DatapointPartition(Datablock):
    """Partitions a `DatapointTable`'s tabs into folds according to target fractions.

    Uses the Longest Processing Time First (LPT) / Worst-Fit Decreasing (WFD)
    greedy heuristic for multiway number partitioning (an NP-complete problem).
    Tabs are sorted in descending order of row count and greedily assigned to the
    fold with the largest remaining capacity deficit.
    """

    TOPICS = {'tabs': 'tabs.json'}

    @dataclass
    class VAR(Datablock.VAR):
        datapoint_table: DatapointTable
        fractions: list[float]
        partition_slice: int | str

    def __build__(self):
        table = self.var.datapoint_table
        if table is None:
            raise ValueError(f"{self.__class__.__name__}: VAR.datapoint_table is required")
        fractions = self.var.fractions
        if not fractions:
            raise ValueError(f"{self.__class__.__name__}: VAR.fractions is required")

        p_slice = self.var.partition_slice
        if isinstance(p_slice, int):
            slice_name = table.slices()[p_slice]
        else:
            slice_name = p_slice

        n_tabs = table.n_tabs
        tab_rows = [table.tab(i).n_rows(slice_name) for i in range(n_tabs)]
        total_rows = sum(tab_rows)

        sum_frac = sum(fractions)
        norm_fracs = [f / sum_frac for f in fractions]
        target_rows = [total_rows * f for f in norm_fracs]
        fold_rows = [0.0] * len(fractions)
        folds_tabs = [[] for _ in range(len(fractions))]

        # Longest Processing Time First (LPT) / Worst-Fit Decreasing heuristic:
        # Sort tabs in descending order of row count to place largest tabs first,
        # avoiding allocation bottlenecks later when remaining capacity is tight.
        indexed_tabs = sorted(range(n_tabs), key=lambda i: tab_rows[i], reverse=True)
        for t_idx in indexed_tabs:
            t_rows = tab_rows[t_idx]
            # Assign tab to the fold with the largest remaining target deficit (Worst-Fit)
            deficits = [target_rows[k] - fold_rows[k] for k in range(len(fractions))]
            best_fold = max(range(len(fractions)), key=lambda k: deficits[k])
            folds_tabs[best_fold].append(t_idx)
            fold_rows[best_fold] += t_rows

        for k in range(len(folds_tabs)):
            folds_tabs[k].sort()

        with self.fs.open(self.path('tabs', ensure_dirpath=True), 'w') as f:
            json.dump(folds_tabs, f)

    def n_folds(self) -> int:
        return len(self.var.fractions)

    def tabs_indices(self, fold: int | str) -> list[int]:
        data = json.loads(self.fs.cat(self.path('tabs')))
        return data[int(fold)]

    def tabs(self, fold: int | str) -> list[DatapointTab]:
        indices = self.tabs_indices(fold)
        table = self.var.datapoint_table
        return [table.tab(i) for i in indices]

    @property
    def datapoint_table(self) -> DatapointTable:
        return self.var.datapoint_table

    def fold(self, fold: int | str) -> DatapointFold:
        return DatapointFold(
            # As a table gives its tabs its url: a fold of a partition belongs
            # where the partition does, not wherever DBX_ROOT happens to point
            # in the process that asks for it.
            url=self.url,
            storage_options=self.storage_options,
            spec=dict(
                partition=self,
                fold=fold,
            )
        )

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath == ('tabs',):
            return json.loads(self.fs.cat(self.path('tabs')))
        return super().__read__(*topicpath)


class DatapointFold(DatapointTable):
    """A subset of a `DatapointTable` defined by tab_indices for a fold."""

    @dataclass
    class VAR(Datablock.VAR):
        partition: DatapointPartition
        fold: int

    @functools.cached_property
    def tab_indices(self) -> list[int]:
        return self.var.partition.tabs_indices(self.var.fold)

    @property
    def datapoint_table(self) -> DatapointTable:
        return self.var.partition.datapoint_table

    @property
    def datapoints_per_row(self) -> int:
        return getattr(self.var.partition.datapoint_table.var, 'datapoints_per_row')

    @property
    def TAB(self):
        return getattr(self.var.partition.datapoint_table, 'TAB', None)

    def slices(self):
        return self.var.partition.datapoint_table.slices()

    @property
    def TOPICS(self):
        return self.var.partition.datapoint_table.TOPICS

    @property
    def n_tabs(self) -> int:
        return len(self.tab_indices)

    def tab(self, idx: int) -> DatapointTab:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table.tab(real_idx)

    def valid_tab(self, idx: int) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table.valid_tab(real_idx)

    valid_block = valid_tab

    def redirected_tab(self, idx: int) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table.redirected_tab(real_idx)

    redirected_block = redirected_tab

    def validate_tab(self, idx: int, **kwargs) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table.validate_tab(real_idx, **kwargs)

    validate_block = validate_tab

    def _write_tab_path(self, idx: int):
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table._write_tab_path(real_idx)

    def _check_tab_path(self, idx: int) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table._check_tab_path(real_idx)

    def _remove_tab_path(self, idx: int):
        real_idx = self.tab_indices[idx]
        return self.var.partition.datapoint_table._remove_tab_path(real_idx)

    _write_tab_built = _write_tab_path
    _check_tab_built = _check_tab_path
    _remove_tab_built = _remove_tab_path
    _write_block_path = _write_tab_path
    _remove_block_path = _remove_tab_path

    def __tab__(self, idx: int, *, tag=None, **spec) -> DatapointTab:
        return self.tab(idx)

    def __block__(self, idx: int) -> DatapointTab:
        return self.tab(idx)

    def _read_slice(self, slice, *, tabs=None, **kwargs):
        if tabs is None:
            indices = range(self.n_tabs)
        else:
            indices = tabs
        datapoints = []
        for idx in indices:
            datapoints.extend(self.tab(idx)._read_slice(slice, **kwargs))
        return datapoints

    def datastream(self, slice, **kwargs) -> StreamingDataset:
        self.slice_names((slice,))
        cacheroot = self._ensure_cacheroot(kwargs.pop('cache', None))
        cache_dir = kwargs.pop('cache_dir',
                               f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        local = os.path.join(cacheroot, cache_dir)
        os.makedirs(local, exist_ok=True)
        streams = self._tab_streams(slice, local)
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
            return StreamingDataset(**streaming_kwargs)
        except (ValueError, TypeError) as exc:
            SharedMemoryManager.clean_process_shared_memory()
            streaming_kwargs['streams'] = self._tab_streams(slice, local)
            return StreamingDataset(**streaming_kwargs)
