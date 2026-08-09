"""datapoints — DatapointTab / DatapointTable blocks over MDS slices."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass, field

import numpy as np

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

from .datablocks import DIRTOPIC, Datablock, Datastack
from .datastreams import (
    BlockShuffleSampler,
    ChunkShuffleSampler,
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


class SlicesProperty:
    """Descriptor enabling `.slices` to work on both class and instance."""

    def __get__(self, obj, cls=None):
        target = obj if obj is not None else cls
        topics = getattr(target, 'TOPICS', {})
        return tuple(SlicedTopics._find_slice_topics(topics))


class SlicedTopics:
    """Mixin giving a block a topic group of parallel MDS slice topics marked with `SLICETOPIC`.

    A **slice** is one independently-readable MDS stream. Every slice of a
    `DatapointTab` is written in lockstep from one pass over that tab's
    input, so row i of every slice describes the same item.

    Slices are declared via topics marked with `SLICETOPIC`::

        class MyTab(DatapointTab):
            TOPICS = {
                'frames': SLICETOPIC,
                'annotations': SLICETOPIC,
                'stats': 'stats.json',
            }

    Two things are readable off a built slice:

    * `data()` -- the rows themselves, decoded eagerly into a list.
    * `dataset()` -- a live `StreamingDataset` per slice, zipped into
      one `torch.utils.data.Dataset`.

    Note
    ----
    Methods operating on tab shards (e.g. `shard_sizes`, `max_rows_per_shard`,
    `n_rows`) presume this mixin is combined with a block implementing
    `.tab(idx)` (such as `DatapointTable` or `DatapointFold`), and cannot be
    used on their own without `.tab(idx)`.
    """

    TOPICS = {}
    slices = SlicesProperty()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__resolve_topics__()

    @staticmethod
    def _find_slice_topics(topics_dict, prefix=()):
        """Find all topic paths in `topics_dict` marked with `SLICETOPIC`."""
        slice_topics = []
        if not isinstance(topics_dict, dict):
            return slice_topics
        for key, val in topics_dict.items():
            current = prefix + (key,)
            if val == SLICETOPIC or val is SLICETOPIC:
                slice_topics.append('/'.join(current) if len(current) > 1 else key)
            elif isinstance(val, dict):
                slice_topics.extend(SlicedTopics._find_slice_topics(val, current))
        return slice_topics

    @classmethod
    def __resolve_topics__(cls):
        """Rebuild `TOPICS` as inherited + own.

        TOPICS **accumulates** down the hierarchy: a subclass declaring topics
        adds to what it inherits instead of replacing it.
        """
        own = cls.__dict__.get('TOPICS')
        if isinstance(own, property) or own is None:
            own = {}
        elif not isinstance(own, dict):
            raise TypeError(
                f"{cls.__name__}: TOPICS must be a dict for a sliced block, "
                f"got {own!r}"
            )
        else:
            own = dict(own)
        own_slices = getattr(cls, 'SLICES', None) or cls.__dict__.get('SLICES')
        if own_slices:
            for s in own_slices:
                if isinstance(s, (tuple, list)):
                    s = s[0]
                own[str(s)] = SLICETOPIC

        topics = {}
        for klass in reversed(cls.__mro__[1:]):
            inherited = klass.__dict__.get('TOPICS')
            if isinstance(inherited, dict):
                topics.update(inherited)
        topics.update(own)
        cls.TOPICS = topics

    # 1. Datablock Protocol Methods ─────────────────────────────────

    def valid_slice(self, slice) -> bool:
        """True when *slice* has a **non-empty** `index.json`."""
        try:
            index_path = self.slice_index_path(slice)
            if not self.fs.exists(index_path):
                return False
            return bool(json.loads(self.fs.cat(index_path)).get('shards'))
        except Exception:
            return False

    def valid_topic(self, *topicpath):
        """As `Datablock.valid_topic()`, but slices go through `valid_slice()`."""
        topicpath = self._normtopic(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices:
                return self.valid_slice(topic_str)
        return super().valid_topic(*topicpath)

    def validtopic(self, *topicpath):
        """Deprecated alias for valid_topic."""
        return self.valid_topic(*topicpath)

    # 2. Properties and Accessors ───────────────────────────────────

    @property
    def cacheroot(self) -> str:
        """Local scratch root for everything streaming: read caches, staged writes."""
        return getattr(self, 'cache', None) or os.path.join(
            self.localroot, 'streaming',
        )

    def data(self, *slices, concat: bool = False, **kwargs):
        """Every row of the named slices, decoded into a list (or concatenated).

        Parameters
        ----------
        *slices : str
            Slice names to read.
        concat : bool, optional
            If True, concatenate/stack tensors/ndarrays along a new first dimension.
        **kwargs
            Passed to `_read_slice()`.
        """
        names = self._slicenames(slices)
        if len(names) == 1 and slices:
            res = self._read_slice(names[0], **kwargs)
            return concat_data(res) if concat else res
        res = {name: self._read_slice(name, **kwargs) for name in names}
        if concat:
            return {name: concat_data(val) for name, val in res.items()}
        return res

    def datastream(self, slice, **kwargs) -> StreamingDataset:
        """One slice as a live `StreamingDataset`."""
        self._slicenames((slice,))
        cache_limit_gb = kwargs.pop('cache_limit_gb', kwargs.pop('cache_limit', getattr(self, 'cache_limit', None)))
        kwargs.setdefault('cache_dir', f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        kwargs.setdefault('cache', self.cacheroot)
        kwargs.setdefault('cache_limit', cache_limit_gb)
        return open_datastream(self.path(*slice.split('/')), **kwargs)

    def dataset(
        self,
        *slices,
        mode='map',
        columns=None,
        shared=None,
        validate_shared=False,
        on_conflict='last',
        skip_none=True,
        zip_validator=None,
        cache_limit_gb=None,
        **kwargs,
    ):
        """The named slices, zipped into one `Dataset`.

        Parameters
        ----------
        mode : {'map', 'iter'}
            How the slices are read.
        columns : list[tuple[str, str]] | dict | None
            Specified as a list of `(slice, column)` tuples.
        cache_limit_gb : float or str, optional
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
        names = self._slicenames(slices)

        if columns is not None:
            if isinstance(columns, dict):
                col_list = []
                for s_name, cols in columns.items():
                    if isinstance(cols, (list, tuple)):
                        for c in cols:
                            col_list.append((s_name, c))
                    else:
                        col_list.append((s_name, cols))
                columns = col_list

            if not isinstance(columns, (list, tuple)):
                raise TypeError(
                    f"{self.__class__.__name__}.dataset: columns must be a list of (slice, column) tuples, got {columns!r}"
                )
            for col_item in columns:
                if not (isinstance(col_item, (tuple, list)) and len(col_item) == 2):
                    raise ValueError(
                        f"{self.__class__.__name__}.dataset: each column entry must be a (slice, column) tuple, got {col_item!r}"
                    )
                s_name = str(col_item[0])
                if s_name not in names:
                    raise KeyError(
                        f"{self.__class__.__name__}.dataset: columns names slice {s_name!r}, which is not among opened slices {list(names)}"
                    )

            per_slice_columns = []
            for name in names:
                cols = [str(c) for (s, c) in columns if str(s) == name]
                per_slice_columns.append(cols if cols else None)
        else:
            per_slice_columns = None

        if cache_limit_gb is not None:
            kwargs['cache_limit_gb'] = cache_limit_gb

        datasets = [self.datastream(name, **kwargs) for name in names]
        zip_cls = ZipStreamingDataset if mode == 'map' else ZipIterableStreamingDatasets
        return zip_cls(
            *datasets,
            columns=per_slice_columns,
            shared=shared,
            validate_shared=validate_shared,
            on_conflict=on_conflict,
            skip_none=skip_none,
            zip_validator=zip_validator,
        )

    def stats(self, *slices, **kwargs):
        """User-defined summary of the named slices."""
        names = self._slicenames(slices)
        if len(names) == 1 and slices:
            return self.__stats__(names[0], **kwargs)
        return {name: self.__stats__(name, **kwargs) for name in names}

    # 3. Private and Utility Methods ────────────────────────────────

    def _slicenames(self, slices) -> tuple:
        """Normalize a `*slices` varargs tuple; empty means *all* slices."""
        if len(slices) == 1 and isinstance(slices[0], (tuple, list)):
            slices = tuple(slices[0])
        if not slices:
            return self.slices
        unknown = [s for s in slices if s not in self.slices]
        if unknown:
            raise KeyError(
                f"{self.__class__.__name__}: unknown slice(s) {unknown}; "
                f"available are {list(self.slices)}"
            )
        return tuple(slices)

    def slice_index_path(self, slice) -> str:
        """Path of the `index.json` for *slice*'s shards."""
        return os.path.join(self.path(*slice.split('/')), 'index.json')

    def _ensure_cacheroot(self, cache=None) -> str:
        cacheroot = cache or self.cacheroot
        os.makedirs(cacheroot, exist_ok=True)
        return cacheroot

    def _read_slice(self, slice, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _read_slice(slice)"
        )

    def _tab_stream(self, tab, slice) -> Stream:
        index_dir = tab.path(*slice.split('/'))
        scheme = urllib.parse.urlparse(index_dir).scheme
        if scheme in ('', 'file'):
            return Stream(local=index_dir.removeprefix('file://'))
        remote = abfs_to_mds_azure(index_dir) if scheme in ('abfs', 'abfss') else index_dir
        return Stream(remote=remote)

    def shard_sizes(self, slice: str) -> list[int]:
        """Row counts per shard for the specified slice."""
        if slice is None:
            raise TypeError(f"{self.__class__.__name__}.shard_sizes requires an explicit slice argument")
        slice = self._slicenames((slice,))[0]
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

    def n_rows(self, slice: str) -> int:
        """Total dataset rows in the specified slice."""
        if slice is None:
            raise TypeError(f"{self.__class__.__name__}.n_rows requires an explicit slice argument")
        return sum(self.shard_sizes(slice))

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

    def block_shuffle_sampler(self, slice: str = None, *, block_size=None, seed: int = 0, fixed_epoch: bool = False, **kwargs) -> ChunkShuffleSampler:
        """Deprecated alias for chunk_shuffle_sampler."""
        s = slice or kwargs.pop('principal_slice', None)
        if s is None:
            raise TypeError(f"{self.__class__.__name__}.block_shuffle_sampler requires an explicit slice argument")
        return self.chunk_shuffle_sampler(
            s,
            chunk_size=block_size,
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
        for s in self.slices:
            counts[s] = self.n_rows(s)
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"{self.__class__.__name__}: slices are not in lockstep row counts: {counts}"
            )
        return counts

    def check_lockstep_rows(self) -> dict[str, int]:
        """Deprecated alias for verify_slice_row_counts_match."""
        return self.verify_slice_row_counts_match()


class DatapointTab(SlicedTopics, Datablock):
    """One tab of a `DatapointTable`: a Datablock writing MDS slices."""

    @dataclass
    class VAR(Datablock.VAR):
        datapoints_per_row: int = 1

    # 1. Datablock Protocol Methods ─────────────────────────────────

    def __init__(self, *args, cache=None, cache_limit_gb=None, cache_limit=None, **kwargs):
        limit = cache_limit_gb if cache_limit_gb is not None else cache_limit
        super().__init__(*args, cache=cache, cache_limit=limit, **kwargs)

    def __build__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __build__(): write every "
            f"slice in {list(self.slices) or ['...']} in lockstep via self.slice_writers(slices)"
        )

    def __stats__(self, slice, **kwargs) -> dict:
        return {'n_rows': len(self.data(slice))}

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices:
                return self.data(topic_str)
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ override to read {'/'.join(topicpath)!r}"
        )

    # 2. Properties and Accessors ───────────────────────────────────

    # 3. Private and Utility Methods ────────────────────────────────

    @contextlib.contextmanager
    def slice_writers(self, slices, *, stage: bool = None, cache=None,
                      cache_limit_gb=None, flush_every: int = None, **writer_kwargs):
        """One `MDSWriter` per slice, as `{slice: writer}`.

        Parameters
        ----------
        slices : dict
            `{slice_name: {column: mds_type}}`, one entry per declared slice.
        """
        names = self.slices
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


class DatapointTable(SlicedTopics, Datastack):
    """A table of DatapointTabs, sliced the same way as its tabs."""

    TAB = None
    TOPICS = {'tabs': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datastack.VAR):
        datapoints_per_row: int = 1

    # 1. Datastack / Table Protocol Methods ─────────────────────────

    def __init__(self, *args, cache=None, cache_limit_gb=None, cache_limit=None, **kwargs):
        limit = cache_limit_gb if cache_limit_gb is not None else cache_limit
        super().__init__(*args, cache=cache, cache_limit=limit, **kwargs)

    @classmethod
    def __resolve_topics__(cls):
        raw_own = cls.__dict__.get('TOPICS')
        own = dict(raw_own) if isinstance(raw_own, dict) else {}
        if cls.TAB is not None and not isinstance(cls.TAB, property):
            tab_topics = getattr(cls.TAB, 'TOPICS', {})
            tab_slices = SlicedTopics._find_slice_topics(tab_topics)
            if not tab_slices:
                tab_slices = getattr(cls.TAB, 'SLICES', ())
            for s in tab_slices:
                if isinstance(s, (tuple, list)):
                    s = s[0]
                own[str(s)] = SLICETOPIC
        cls.TOPICS = own
        super().__resolve_topics__()

    def __tab__(self, idx: int, *, tag=None, **spec) -> DatapointTab:
        if self.TAB is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set TAB = <DatapointTab subclass>"
            )
        return self.TAB(
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

    def __split__(self, *args, **kwargs):
        self.path('tabs', ensure_dirpath=True)
        n = self.n_tabs
        self.log.info(
            "%s: %d tabs x %d slices %s",
            self.__class__.__name__, n, len(self.slices), list(self.slices),
        )
        return [self.TabMaker(idx) for idx in range(n)], dict(build=True)

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
        topicpath = self._normtopic(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices:
                return self.data(topic_str)
        if topicpath == ('tabs',):
            return self.path('tabs')
        if topicpath == ('done',):
            return self.valid()
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ answers only slices, 'tabs' and 'done'; "
            f"override it to read {'/'.join(topicpath)!r}"
        )

    def valid(self):
        return self.valid_topic('done')

    def valid_slice(self, slice) -> bool:
        return all(
            self.tab(idx).valid_slice(slice) for idx in range(self.n_tabs)
        )

    def __stats__(self, slice, **kwargs) -> dict:
        return super().__stats__(slice, **kwargs)

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
        self._slicenames((slice,))
        streams = [self._tab_stream(self.tab(idx), slice)
                   for idx in range(self.n_tabs)]
        cacheroot = self._ensure_cacheroot(kwargs.pop('cache', None))
        cache_dir = kwargs.pop('cache_dir',
                               f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        local = os.path.join(cacheroot, cache_dir)
        os.makedirs(local, exist_ok=True)
        cache_limit_gb = kwargs.pop('cache_limit_gb', kwargs.pop('cache_limit', getattr(self, 'cache_limit', None)))
        shuffle = kwargs.pop('shuffle', False)
        allow_unsafe_types = kwargs.pop('allow_unsafe_types', True)
        streaming_kwargs = dict(
            streams=streams,
            shuffle=shuffle,
            allow_unsafe_types=allow_unsafe_types,
            cache_limit=cache_limit_gb,
            **kwargs,
        )
        try:
            return StreamingDataset(**streaming_kwargs)
        except ValueError as exc:
            if 'Reused local directory' not in str(exc):
                raise
            from streaming.base.util import clean_stale_shared_memory
            clean_stale_shared_memory()
            return StreamingDataset(**streaming_kwargs)

    # 3. Private and Utility Methods ────────────────────────────────

    def _read_slice(self, slice, *, tabs=None, **kwargs):
        indices = range(self.n_tabs) if tabs is None else tabs
        datapoints = []
        for idx in indices:
            datapoints.extend(self.tab(idx).data(slice, **kwargs))
        return datapoints

    class TabMaker:
        def __init__(self, tab_idx: int):
            self.tab_idx = tab_idx

        def __call__(self, table, *, build=True):
            tab = table.__tab__(self.tab_idx)
            tab.keyby = table.keyby
            skipped = tab.valid()
            if build:
                tab.build()
            result = {'tab_idx': self.tab_idx, 'tag': tab.tag, 'skipped': skipped}
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
        datapoint_table: DatapointTable = None
        fractions: list[float] = field(default_factory=list)
        partition_slice: int | str = 0

    def __build__(self):
        table = self.var.datapoint_table
        if table is None:
            raise ValueError(f"{self.__class__.__name__}: VAR.datapoint_table is required")
        fractions = self.var.fractions
        if not fractions:
            raise ValueError(f"{self.__class__.__name__}: VAR.fractions is required")

        p_slice = self.var.partition_slice
        if isinstance(p_slice, int):
            slice_name = table.slices[p_slice]
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

    def tabs_indices(self, fold: int) -> list[int]:
        data = json.loads(self.fs.cat(self.path('tabs')))
        return data[fold]

    def tabs(self, fold: int) -> list[DatapointTab]:
        indices = self.tabs_indices(fold)
        table = self.var.datapoint_table
        return [table.tab(i) for i in indices]

    def fold(self, fold: int) -> DatapointFold:
        return DatapointFold(
            spec=dict(
                datapoint_table=self.var.datapoint_table,
                tab_indices=self.tabs_indices(fold),
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
    class VAR(DatapointTable.VAR):
        datapoint_table: DatapointTable = None
        tab_indices: list[int] = field(default_factory=list)

    @property
    def TAB(self):
        return getattr(self.var.datapoint_table, 'TAB', None)

    @property
    def slices(self):
        if self.var.datapoint_table is not None:
            return self.var.datapoint_table.slices
        return ()

    @property
    def TOPICS(self):
        if self.var.datapoint_table is not None:
            return self.var.datapoint_table.TOPICS
        return {}

    @property
    def n_tabs(self) -> int:
        return len(self.var.tab_indices)

    def tab(self, idx: int) -> DatapointTab:
        real_idx = self.var.tab_indices[idx]
        return self.var.datapoint_table.tab(real_idx)

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
            datapoints.extend(self.tab(idx).data(slice, **kwargs))
        return datapoints

    def datastream(self, slice, **kwargs) -> StreamingDataset:
        self._slicenames((slice,))
        streams = [self._tab_stream(self.tab(idx), slice)
                   for idx in range(self.n_tabs)]
        cacheroot = self._ensure_cacheroot(kwargs.pop('cache', None))
        cache_dir = kwargs.pop('cache_dir',
                               f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        local = os.path.join(cacheroot, cache_dir)
        os.makedirs(local, exist_ok=True)
        cache_limit_gb = kwargs.pop('cache_limit_gb', kwargs.pop('cache_limit', getattr(self, 'cache_limit', None)))
        shuffle = kwargs.pop('shuffle', False)
        allow_unsafe_types = kwargs.pop('allow_unsafe_types', True)
        streaming_kwargs = dict(
            streams=streams,
            shuffle=shuffle,
            allow_unsafe_types=allow_unsafe_types,
            cache_limit=cache_limit_gb,
            **kwargs,
        )
        try:
            return StreamingDataset(**streaming_kwargs)
        except ValueError as exc:
            if 'Reused local directory' not in str(exc):
                raise
            from streaming.base.util import clean_stale_shared_memory
            clean_stale_shared_memory()
            return StreamingDataset(**streaming_kwargs)
