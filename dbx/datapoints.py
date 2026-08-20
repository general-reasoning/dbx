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


class _slices_descriptor:
    """``slices`` read off the class as readily as off an instance.

    A subclass's slices are a property of how it was DECLARED, so they are
    asked for at class level -- ``LetterTable.slices``, ``'depth' in
    Debuggable.slices`` -- as often as they are asked of a block. A plain
    ``@property`` answers only the second, handing back the descriptor itself
    for the first.

    Derived on each access rather than frozen at class creation, so a TOPICS
    assigned or amended after the class body still reports its slices, and an
    instance overriding TOPICS (as :class:`DatapointFold` does) is read through.
    """

    def __get__(self, obj, owner=None):
        target = owner if obj is None else obj
        return DatapointBase._find_slice_topics(getattr(target, 'TOPICS', None))


class _table_slices_descriptor:
    """``slices`` for a `DatapointTable`, derived from TAB's slices.

    A table does not declare slice topics in its own TOPICS (they belong to the
    tab). Its slices are the TAB's slices -- the same set, but accessed via the
    TAB class rather than by inspecting the table's TOPICS.

    Falls back to reading the instance/class TOPICS if TAB is not set or is not
    a proper DatapointBase subclass (e.g. DatapointFold, which overrides TOPICS
    at instance level and computes its TAB dynamically).
    """

    def __get__(self, obj, owner=None):
        target = owner if obj is None else obj
        tab = getattr(target, 'TAB', None)
        if isinstance(tab, type) and issubclass(tab, DatapointBase):
            return tab.slices
        # Fallback: read from own TOPICS (covers DatapointFold and intermediate
        # bases that have no TAB yet).
        return DatapointBase._find_slice_topics(getattr(target, 'TOPICS', None))


class DatapointBase(Datablock):
    """Base class for sliced datapoint blocks (DatapointTab and DatapointTable).

    A **slice** is one independently-readable MDS stream directory inside a block.
    Slices are declared via topics marked with `SLICETOPIC`.

    A subclass declaring TOPICS keeps its bases' SLICE topics and drops their
    other ones::

        class BaseTab(DatapointTab):
            TOPICS = {'samples': SLICETOPIC, 'meta': 'meta.json'}

        class SubTab(BaseTab):
            TOPICS = {'report': 'report.json'}
            # TOPICS == {'samples': SLICETOPIC, 'report': 'report.json'}

    Slices are what makes such a block the kind of block it is -- a subclass
    that redeclares TOPICS is describing what it adds, not renouncing the
    streams its base reads and writes -- whereas ordinary topics belong to the
    class that declared them and do not accumulate down the hierarchy.
    """

    TOPICS = {}

    #: SLICES is retired: a slice is a TOPICS entry valued SLICETOPIC, and
    #: nothing reads a SLICES attribute any more. A class still carrying one
    #: would come out with no slices at all -- valid, buildable, and empty --
    #: which is worth an error at construction rather than a puzzle later.
    RETIRED_ATTRS = {**Datablock.RETIRED_ATTRS,
                     'SLICES': 'TOPICS entries valued SLICETOPIC'}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._synthesize_topics()

    @classmethod
    def _synthesize_topics(cls):
        """Merge this class's declared TOPICS with its bases' slice topics."""
        declared = cls.__dict__.get('TOPICS')
        if not isinstance(declared, dict):
            # Declares none (inherits its base's TOPICS whole), or overrides the
            # name with something computed -- DatapointFold makes it a property.
            return
        cls.TOPICS = {**cls._inherited_slice_topics(), **declared}

    @classmethod
    def _inherited_slice_topics(cls):
        """The slice topics of the nearest base that declares TOPICS."""
        for base in cls.__mro__[1:]:
            topics = base.__dict__.get('TOPICS')
            if isinstance(topics, dict):
                return cls._slice_topics_only(topics)
        return {}

    @classmethod
    def _slice_topics_only(cls, topics):
        """*topics* with everything that is not a slice pruned out.

        Recurses into groups, and drops a group that holds no slice at all, so
        a group of ordinary topics does not come down as an empty husk.
        """
        kept = {}
        for name, node in topics.items():
            if node == SLICETOPIC or node is SLICETOPIC:
                kept[name] = node
            elif isinstance(node, dict):
                nested = cls._slice_topics_only(node)
                if nested:
                    kept[name] = nested
        return kept

    # 1. Datablock Protocol Methods ─────────────────────────────────

    def __init__(self, *args, cache_limit=None, **kwargs):
        super().__init__(*args, cache_limit=cache_limit, **kwargs)

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

    def UNSAFE_copy_from(self, anchorkeypath, *, OVERRIDE: bool = False, overwrite: bool = False, topicpaths=None, validate: bool = True, always_copy_whole_dirpath: bool = False, show_progress: bool = True, **kwargs):
        result = super().UNSAFE_copy_from(anchorkeypath, OVERRIDE=OVERRIDE, overwrite=overwrite, topicpaths=topicpaths, validate=validate, always_copy_whole_dirpath=always_copy_whole_dirpath, show_progress=show_progress, **kwargs)
        if validate:
            self.verify_slice_row_counts_match()
        return result

    # 2. Properties and Accessors ───────────────────────────────────

    slices = _slices_descriptor()

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
        cache_limit = kwargs.pop('cache_limit', getattr(self, 'cache_limit', None))
        kwargs.setdefault('cache_dir', f"{self.fqcn}-{self.hash[:12]}-{slice.replace('/', '_')}")
        kwargs.setdefault('cache', self.cacheroot)
        kwargs.setdefault('cache_limit', cache_limit)
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
        cache_limit=None,
        **kwargs,
    ):
        """The named slices, zipped into one `Dataset`.

        Parameters
        ----------
        mode : {'map', 'iter'}
            How the slices are read.
        columns : list[tuple[str, str]] | dict | None
            Specified as a list of `(slice, column)` tuples.
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

        if cache_limit is not None:
            kwargs['cache_limit'] = cache_limit

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

    def n_rows(self, slice: str) -> int:
        """Total dataset rows in the specified slice."""
        if slice is None:
            raise TypeError(f"{self.__class__.__name__}.n_rows requires an explicit slice argument")
        return sum(self.shard_sizes(slice))

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
        seed: int = 0,
        fixed_epoch: bool = False,
    ) -> ChunkShuffleSampler:
        """Deprecated alias of :meth:`chunk_shuffle_sampler`; *block_size* is *chunk_size*.

        A chunk was called a block before the name moved to what it describes
        -- consecutive indices, which are a shard's worth of rows, not a block.
        The sampler itself still answers to :class:`BlockShuffleSampler` and to
        ``block_size``, and so does this.
        """
        return self.chunk_shuffle_sampler(
            slice, chunk_size=block_size, seed=seed, fixed_epoch=fixed_epoch,
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
            return self.slices
        unknown = [s for s in slices if s not in self.slices]
        if unknown:
            raise KeyError(
                f"{self.__class__.__name__}: unknown slice(s) {unknown}; "
                f"available are {list(self.slices)}"
            )
        return tuple(slices)

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


class DatapointTable(DatapointBase, Datastack):
    """A table of DatapointTabs, sliced the same way as its tabs.

    A table's TOPICS only contains what the table itself owns: the structural
    topics (``tabs``, ``done``) and any extra file topics the subclass declares
    (such as ``bag_lens``). The tab's slice topics are NOT merged into the
    table's TOPICS -- they belong to the tab, not the table.

    The table's `slices` attribute is derived from ``TAB.slices`` rather than
    from ``TOPICS``, so slice routing (``data()``, ``dataset()``,
    ``valid_slice()``) continues to work without polluting ``TOPICS``. Pointing
    a table at a differently-sliced TAB still rekeys the table because the TAB
    class is part of :attr:`signature`.

    The tab's ordinary (non-slice) topics are written into each tab under the
    tab's own key; the table has nothing at those paths.
    """

    TAB = None

    #: The topics the table machinery itself writes and reads: the directory
    #: the tabs live in, and the marker that says the stack completed. Kept
    #: whatever a subclass declares -- a table missing them cannot be built or
    #: asked whether it is valid, and a subclass redeclaring TOPICS is naming
    #: what it adds, not opting out of being a table.
    STRUCTURAL_TOPICS = {'tabs': DIRTOPIC, 'built_tabs': DIRTOPIC, 'done': 'done'}
    TOPICS = dict(STRUCTURAL_TOPICS)

    #: Slices come from TAB, not from this table's own TOPICS.
    slices = _table_slices_descriptor()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._synthesize_table_topics()

    @classmethod
    def _synthesize_topics(cls):
        super()._synthesize_topics()
        if not isinstance(cls.__dict__.get('TOPICS'), dict):
            return
        missing = {name: node for name, node in cls.STRUCTURAL_TOPICS.items()
                   if name not in cls.TOPICS}
        if missing:
            cls.TOPICS = {**missing, **cls.TOPICS}

    @classmethod
    def _synthesize_table_topics(cls):
        tab = cls.__dict__.get('TAB', cls.TAB)
        if not (isinstance(tab, type) and issubclass(tab, DatapointBase)):
            # No TAB yet (an intermediate base), or one computed per instance --
            # DatapointFold reads its TAB off the table it wraps.
            return
        topics = cls.TOPICS if isinstance(cls.TOPICS, dict) else {}
        # Strip any slice topics that may have crept in (e.g. via inheritance
        # from an older base that still used the accumulating behaviour).  Only
        # table-owned, non-slice topics belong in TOPICS.
        own = {name: node for name, node in topics.items()
               if not cls._slice_topics_only({name: node})}
        cls.TOPICS = own

    @dataclass
    class VAR(Datastack.VAR):
        datapoints_per_row: int = 1

    # 1. Datastack / Table Protocol Methods ─────────────────────────

    def __init__(self, *args, cache=None, cache_limit=None, **kwargs):
        super().__init__(*args, cache=cache, cache_limit=cache_limit, **kwargs)

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

    def __split__(self, *args, **kwargs):
        self.path('tabs', ensure_dirpath=True)
        if 'built_tabs' in self.topics():
            self.path('built_tabs', ensure_dirpath=True)
        n = self.n_tabs
        self.log.info(
            "%s: %d tabs x %d slices %s",
            self.__class__.__name__, n, len(self.slices), list(self.slices),
        )
        return [self.TabMaker(idx) for idx in range(n)], dict(build=True)

    def __build__(self, *args, **kwargs):
        callables, callable_kwargs = self.__split__(*args, **kwargs)
        if not callables:
            return self.__stack__([])

        work_stealing_state = getattr(self, 'work_stealing', False)
        self.log.info(
            f"Building {self.__class__.__name__}: filtering {len(callables)} tabs using "
            f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}, work_stealing={work_stealing_state}"
        )

        filter_exec_kwargs = self._executor_kwargs(
            tag=f"FILTERING {len(callables)} tabs [{self.__class__.__name__}]"
        )
        filter_executor = self.executor_cls(**filter_exec_kwargs)
        checkers = [
            self.TabValidChecker(getattr(c, 'tab_idx', getattr(c, 'idx', i)))
            for i, c in enumerate(callables)
        ]
        validity = filter_executor.exec_callables(checkers, self)

        to_build_callables = []
        callable_results = []

        for i, (c, is_valid) in enumerate(zip(callables, validity)):
            idx = getattr(c, 'tab_idx', getattr(c, 'idx', i))
            tag = getattr(c, 'tag', f"tab_{idx:06d}")
            if is_valid:
                callable_results.append({'tab_idx': idx, 'tag': tag, 'skipped': True})
            else:
                to_build_callables.append(c)

        self.log.info(
            f"{self.__class__.__name__}: {len(callables) - len(to_build_callables)}/{len(callables)} tabs already valid, "
            f"building {len(to_build_callables)} tabs"
        )

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
        if topic_str in self.slices:
            # Bypass _topicnode: slices are not in TOPICS but are valid reads.
            return self.__read__(*topicpath)
        return super().read(*topicpath)

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath:
            topic_str = '/'.join(topicpath)
            if topic_str in self.slices:
                return self.data(topic_str)
        if topicpath == ('tabs',):
            return self.path('tabs')
        if topicpath == ('built_tabs',):
            return self.path('built_tabs')
        if topicpath == ('done',):
            return self.valid()
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ answers only slices, 'tabs', 'built_tabs' and 'done'; "
            f"override it to read {'/'.join(topicpath)!r}"
        )

    def valid(self):
        return self.valid_topic('done')

    def _write_tab_built(self, i: int):
        if 'built_tabs' not in self.topics():
            return
        built_dir = self.path('built_tabs', ensure_dirpath=True)
        sentinel_path = os.path.join(built_dir, f"tab_{i}.built")
        with self.fs.open(sentinel_path, 'wb'):
            pass

    def _check_tab_built(self, i: int) -> bool:
        if 'built_tabs' not in self.topics():
            return False
        try:
            built_dir = self.path('built_tabs')
            sentinel_path = os.path.join(built_dir, f"tab_{i}.built")
            return self.fs.exists(sentinel_path)
        except Exception:
            return False

    def valid_tab(self, i: int) -> bool:
        if 'built_tabs' in self.topics():
            if self._check_tab_built(i):
                return True
            return self.tab(i).valid()
        return self.tab(i).valid()

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
            slice_segments = tuple(
                f"topic:{name}=SLICETOPIC"
                for name in tab.slices
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
        self._slicenames((slice,))
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

    class TabValidChecker:
        def __init__(self, tab_idx: int):
            self.tab_idx = tab_idx

        def __call__(self, table):
            return table.valid_tab(self.tab_idx)

    class TabMaker:
        def __init__(self, tab_idx: int):
            self.tab_idx = tab_idx

        def __call__(self, table, *, build=True):
            tab = table.__tab__(self.tab_idx)
            tab.keyby = table.keyby
            skipped = tab.valid()
            if build:
                tab.build()
                if hasattr(table, '_write_tab_built'):
                    table._write_tab_built(self.tab_idx)
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

    def tabs_indices(self, fold: int | str) -> list[int]:
        data = json.loads(self.fs.cat(self.path('tabs')))
        return data[int(fold)]

    def tabs(self, fold: int | str) -> list[DatapointTab]:
        indices = self.tabs_indices(fold)
        table = self.var.datapoint_table
        return [table.tab(i) for i in indices]

    def fold(self, fold: int | str) -> DatapointFold:
        return DatapointFold(
            # As a table gives its tabs its url: a fold of a partition belongs
            # where the partition does, not wherever DBX_ROOT happens to point
            # in the process that asks for it.
            url=self.url,
            storage_options=self.storage_options,
            spec=dict(
                partition=self,
                datapoint_table=self.var.datapoint_table,
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
        datapoint_table: DatapointTable
        fold: int
        datapoints_per_row: int = 1

    @functools.cached_property
    def tab_indices(self) -> list[int]:
        return self.var.partition.tabs_indices(self.var.fold)

    @property
    def TAB(self):
        return getattr(self.var.datapoint_table, 'TAB', None)

    @property
    def slices(self):
        return self.var.datapoint_table.slices

    @property
    def TOPICS(self):
        return self.var.datapoint_table.TOPICS

    @property
    def n_tabs(self) -> int:
        return len(self.tab_indices)

    def tab(self, idx: int) -> DatapointTab:
        real_idx = self.tab_indices[idx]
        return self.var.partition.var.datapoint_table.tab(real_idx)

    def valid_tab(self, idx: int) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.var.datapoint_table.valid_tab(real_idx)

    def _write_tab_built(self, idx: int):
        real_idx = self.tab_indices[idx]
        return self.var.partition.var.datapoint_table._write_tab_built(real_idx)

    def _check_tab_built(self, idx: int) -> bool:
        real_idx = self.tab_indices[idx]
        return self.var.partition.var.datapoint_table._check_tab_built(real_idx)

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
        except ValueError as exc:
            if 'Reused local directory' not in str(exc):
                raise
            from streaming.base.util import clean_stale_shared_memory
            clean_stale_shared_memory()
            return StreamingDataset(**streaming_kwargs)
