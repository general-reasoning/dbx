"""datapoints — DatapointTab / DatapointTable blocks over MDS slices."""

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
    ZipIterableStreamingDatasets,
    ZipStreamingDataset,
    _ShardSync,
    _parse_slice_entries,
    abfs_to_mds_azure,
    concat_data,
    open_datastream,
    read_mds_shard,
    reader_from_json,
)


class SlicedTopics:
    """Mixin giving a block a `data` topic group of parallel MDS slices.

    A **slice** is one independently-readable MDS stream.  Every slice of a
    `DatapointTab` is written in lockstep from one pass over that tab's
    input, so sample i of every slice describes the same thing: that
    alignment is the whole contract, and it is what makes zipping the slices
    back together by index meaningful.

    Slices are declared once, as `SLICES`; the `data` group of
    `TOPICS` is synthesized from them::

        class MyTab(DatapointTab):
            SLICES = ('frames', 'annotations')
            TOPICS = {'stats': 'stats.json'}       # optional extra topics

        MyTab.TOPICS
        # {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC},
        #  'stats': 'stats.json'}

    Declaring the `data` group explicitly works too and then it defines
    `SLICES`.  Either way `TOPICS` ends up in the shape above, so
    the slices are covered by the block's signature and hence by its hash:
    adding, removing or renaming one re-keys the block rather than quietly
    reusing another shape's artifacts.

    Two things are readable off a built slice, and they are not the same
    thing:

    * `data()` -- the samples themselves, decoded eagerly into a list.
      Lumpy, materialised, and per slice.  Use it to inspect or aggregate.
    * `dataset()` -- a live `StreamingDataset` per slice, zipped into
      one `torch.utils.data.Dataset`.  Lazy, index-aligned, and the thing
      you hand a `DataLoader`.

    `stats()` is the third reader and the one this cannot implement:
    what a useful summary of a slice is depends entirely on what is in it,
    so `__stats__()` is a hook.
    """

    # Name of the topic group holding the slices.  Overridable, but every
    # method below addresses through it, so renaming it renames it everywhere.
    DATA = 'data'

    # Slice names, in the order they are zipped.  Declared by the subclass.
    SLICES = ()
    SLICE_DTYPES = {}

    TOPICS = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__resolve_topics__()

    @classmethod
    def __resolve_topics__(cls):
        """Rebuild `TOPICS` as `data` group + inherited + own.

        TOPICS **accumulates** down the hierarchy here, rather than shadowing
        as a plain class attribute would: a subclass declaring
        `TOPICS = {'note': 'note.txt'}` adds to what it inherits instead of
        replacing it, and its own entries win on a collision.  That is the
        whole mechanism -- it is what lets `DatapointTable` declare
        `tabs` and `done` once, in the ordinary way, and still have them
        after a subclass declares topics of its own.

        The `data` group is exempt and always rebuilt, from a `data` group
        the class declares **itself** (which then also defines `SLICES`)
        or else from `SLICES`.  Own-declaration is read off
        `cls.__dict__` rather than the attribute, because a subclass that
        declares only new `SLICES` inherits its parent's already-resolved
        `data` group -- and that stale group must lose to the new slices,
        not silently override them.
        """
        own = cls.__dict__.get('TOPICS') or {}
        if not isinstance(own, dict):
            raise TypeError(
                f"{cls.__name__}: TOPICS must be a dict for a sliced block, "
                f"got {own!r}"
            )

        declared = own.get(cls.DATA)
        if isinstance(declared, dict) and declared:
            slices, slice_dtypes = _parse_slice_entries(declared)
            data_group = {name: DIRTOPIC for name in slices}
            own_slices = tuple(cls.__dict__.get('SLICES') or ())
            if own_slices and own_slices != slices:
                raise ValueError(
                    f"{cls.__name__} declares both SLICES {list(own_slices)} and a "
                    f"{cls.DATA!r} topic group {list(slices)}, and they disagree; "
                    f"declare one or the other"
                )
        else:
            raw_slices = cls.__dict__.get('SLICES')
            if raw_slices is None:
                raw_slices = cls.SLICES
            slices, slice_dtypes = _parse_slice_entries(raw_slices)
            data_group = {name: DIRTOPIC for name in slices}

        topics = {cls.DATA: data_group} if data_group else {}
        # Base-first, so a subclass's own entry overwrites what it inherits.
        inherited_dtypes = {}
        for klass in reversed(cls.__mro__[1:]):
            inherited = klass.__dict__.get('TOPICS')
            if isinstance(inherited, dict):
                topics.update({name: node for name, node in inherited.items()
                               if name != cls.DATA})
            if hasattr(klass, 'SLICE_DTYPES'):
                inherited_dtypes.update(klass.SLICE_DTYPES)
        topics.update({name: node for name, node in own.items()
                       if name != cls.DATA})
        inherited_dtypes.update(slice_dtypes)

        cls.SLICES = slices
        cls.SLICE_DTYPES = inherited_dtypes
        cls.TOPICS = topics

    # ------------------------------------------------------------------ #
    # Slice addressing
    # ------------------------------------------------------------------ #

    @property
    def slices(self) -> tuple:
        """This block's slice names, in zip order."""
        if not self.SLICES:
            raise NotImplementedError(
                f"{self.__class__.__name__} declares no slices: set "
                f"SLICES = ('name', ...) or TOPICS = {{{self.DATA!r}: {{...}}}}"
            )
        return tuple(self.SLICES)

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

    def slice_index_path(self, slice_name) -> str:
        """Path of the `index.json` for *slice_name*'s shards."""
        return os.path.join(self.path(self.DATA, slice_name), 'index.json')

    # ------------------------------------------------------------------ #
    # Validity
    # ------------------------------------------------------------------ #

    def valid_slice(self, slice_name) -> bool:
        """True when *slice_name* has a **non-empty** `index.json`."""
        try:
            index_path = self.slice_index_path(slice_name)
            if not self.fs.exists(index_path):
                return False
            return bool(json.loads(self.fs.cat(index_path)).get('shards'))
        except Exception:
            return False

    def validtopic(self, *topicpath):
        """As `Datablock.validtopic()`, but slices go through
        `valid_slice()` rather than a bare existence check."""
        topicpath = self._normtopic(topicpath)
        if self.SLICES and topicpath and topicpath[0] == self.DATA:
            names = topicpath[1:] or self.slices
            return all(self.valid_slice(name) for name in names)
        return super().validtopic(*topicpath)

    # ------------------------------------------------------------------ #
    # Reading: samples, datasets, stats
    # ------------------------------------------------------------------ #

    @property
    def cacheroot(self) -> str:
        """Local scratch root for everything streaming: read caches, staged
        writes, decompression scratch.

        The resolved form of the `cache=` kwarg, and named for it the way
        `Datablock.localroot` is named for `local=` -- `cache` is
        what was asked for (usually nothing), `cacheroot` is where it
        actually is.

        Defaults to `<localroot>/streaming`: under the block's local
        staging root (`local=`, `DBX_LOCAL`, or the url itself when that
        is already local), **not** the system temporary directory, so that a
        cache big enough to matter lands on the disk the deployment chose
        for it rather than filling `/tmp`.

        Nothing under it is data.  It is shard downloads (bounded by
        `cache_limit`), staged writes on their way to remote storage, and
        decompression scratch; it can be deleted at any time.
        """
        return getattr(self, 'cache', None) or os.path.join(
            self.localroot, 'streaming',
        )

    def _ensure_cacheroot(self, cache=None) -> str:
        """`cacheroot` (or *cache*), created if it does not exist.

        `tempfile.mkdtemp(dir=...)` and the MDS readers both require the
        parent to exist already, and the default sits under a local staging
        root that nothing else necessarily creates.
        """
        cacheroot = cache or self.cacheroot
        os.makedirs(cacheroot, exist_ok=True)
        return cacheroot

    def data(self, *slices, concat: bool = False, **kwargs):
        """Every sample of the named slices, decoded into a list (or concatenated).

        One slice gives `list[dict]` (or concatenated dict if *concat=True*);
        several (or none, meaning all) give `{slice: list[dict]}` (or
        `{slice: concat_dict}` if *concat=True*).  Materialises the whole slice
        in memory -- for anything training-shaped use `dataset()` instead.

        Parameters
        ----------
        *slices : str
            Slice names to read.
        concat : bool, optional
            If True and the result is a list of tensors/ndarrays, stack them along
            a new first dimension. If the result is a list of dicts, concat each
            dict value separately (tensors/ndarrays are stacked, non-tensors remain
            a list inside the dict). Defaults to False.
        **kwargs
            Passed to `_read_slice()`.
        """
        names = self._slicenames(slices)
        slice_dtypes = getattr(self, 'SLICE_DTYPES', {})
        if len(names) == 1 and slices:
            res = self._read_slice(names[0], **kwargs)
            return concat_data(res, dtype=slice_dtypes.get(names[0])) if concat else res
        res = {name: self._read_slice(name, **kwargs) for name in names}
        if concat:
            return {name: concat_data(val, dtype=slice_dtypes.get(name)) for name, val in res.items()}
        return res

    def _read_slice(self, slice_name, **kwargs):
        """Decode one slice.  Internal: `data()` is the override point."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _read_slice(slice_name)"
        )

    def datastream(self, slice_name, **kwargs) -> StreamingDataset:
        """One slice as a live `StreamingDataset`.

        The singular of `dataset()`: one slice, unzipped, exactly as
        `streaming` hands it over.

        The local cache directory is qualified by class, hash **and** slice
        name: the slices of one block are several `StreamingDataset`
        objects alive in one process, and sharing a cache directory between
        them is exactly the `Reused local directory` collision that
        qualification exists to prevent.
        """
        self._slicenames((slice_name,))
        kwargs.setdefault('cache_dir', f"{self.fqcn}-{self.hash[:12]}-{slice_name}")
        kwargs.setdefault('cache', self.cacheroot)
        kwargs.setdefault('cache_limit', getattr(self, 'cache_limit', None))
        return open_datastream(self.path(self.DATA, slice_name), **kwargs)

    def _tab_stream(self, tab, slice_name) -> Stream:
        """One `Stream` for *tab*'s *slice_name* directory.

        Used by `DatapointTable.datastream()` to compose a
        multi-stream `StreamingDataset` without a consolidated
        `index.json`.
        """
        index_dir = tab.path(self.DATA, slice_name)
        scheme = urllib.parse.urlparse(index_dir).scheme
        if scheme in ('', 'file'):
            return Stream(local=index_dir.removeprefix('file://'))
        remote = abfs_to_mds_azure(index_dir) if scheme in ('abfs', 'abfss') else index_dir
        return Stream(remote=remote)

    def dataset(self, *slices, mode='map', columns=None, shared=None,
                validate_shared=False, on_conflict='last', skip_none=True,
                zip_validator=None, **kwargs):
        """The named slices, zipped into one `Dataset`.

        `dataset()` opens every slice; `dataset('frames')` opens one;
        `dataset('frames', 'annotations')` opens those two.  A consumer
        pays only for what it opens -- skipping the largest slice genuinely
        does not fetch its shards.

        Always a zip, even for a single slice, so the sample a caller gets
        back has the same merged-dict shape however many slices it asked
        for.

        The slices are opened with whatever *kwargs* say, identically -- and
        identically is the operative word, because it is what keeps them
        aligned.  Zipping pairs sample i of one slice with sample i of
        another, so any per-slice difference in ordering pairs unrelated
        samples.

        Parameters
        ----------
        mode : {'map', 'iter'}
            How the slices are read, which is the throughput decision:

            `'map'` (default) zips by *physical index*
            (`ZipStreamingDataset`).  Random access, so any sampler works
            -- shuffle with `sampler=table.sampler()` rather than
            `DataLoader(shuffle=True)`, which is the full permutation that
            defeats the shard cache.  But indexing reads through
            `StreamingDataset.get_item()`, which downloads a missing shard
            inline on the calling thread; there is no download-ahead.

            `'iter'` zips by *iteration order*
            (`ZipIterableStreamingDatasets`).  Each slice keeps its own
            prefetch thread, partitioning and resumption, which on remote
            storage is normally the larger win by far.  It gives up random
            access, it requires `batch_size=`, and it constrains
            shuffling: read that class's *Alignment* section before passing
            `shuffle=True`.
        columns : dict | None
            `{slice: [column, ...]}` -- project a slice down to some of its
            columns.  Keyed by slice name rather than position, since that is
            how the caller named the slices in the first place; slices absent
            from the dict are taken whole.

            Projection happens after the row is decoded, so it saves
            collation and the worker-to-parent handoff, not I/O.  Not
            opening a slice at all is what saves I/O.
        shared, validate_shared, on_conflict, skip_none, zip_validator
            Merge policy, passed to the zip.  Slices written in lockstep
            normally share bookkeeping keys, so `shared={'sample_id', ...}`
            with `validate_shared=True` is the setting that turns a
            mis-zipped table into an error instead of a silent misalignment.
            Optional under `'map'`, which is aligned by construction;
            effectively required under `'iter'`, which is not.
        **kwargs
            Passed to `datastream()` for every slice opened -- and to every
            slice the same, which is what `'iter'` mode needs of the
            shuffle configuration.
        """
        if mode not in ('map', 'iter'):
            raise ValueError(
                f"{self.__class__.__name__}.dataset: mode must be 'map' or "
                f"'iter', got {mode!r}"
            )
        if mode == 'iter' and not isinstance(kwargs.get('batch_size'), int):
            # StreamingDataset insists on this the first time it is iterated,
            # because its partitioning is batch-aware.  Left to it, the
            # complaint arrives on the first batch, inside a DataLoader
            # worker, naming a class the caller never mentioned -- so ask
            # here, where the mode that requires it was chosen.
            raise ValueError(
                f"{self.__class__.__name__}.dataset(mode='iter') needs "
                f"batch_size=: iterating partitions each slice over ranks and "
                f"workers in whole batches, so it has to know the size of "
                f"one.  Pass the same per-device batch size you give the "
                f"DataLoader.  (mode='map' does not partition and does not "
                f"need it.)"
            )
        names = self._slicenames(slices)
        if columns is not None:
            unknown = [s for s in columns if s not in names]
            if unknown:
                raise KeyError(
                    f"{self.__class__.__name__}.dataset: columns names slice(s) "
                    f"{unknown}, which are not among the slices being opened "
                    f"{list(names)}"
                )
            columns = [columns.get(name) for name in names]
        datasets = [self.datastream(name, **kwargs) for name in names]
        zip_cls = (ZipStreamingDataset if mode == 'map'
                   else ZipIterableStreamingDatasets)
        return zip_cls(
            *datasets, columns=columns, shared=shared,
            validate_shared=validate_shared, on_conflict=on_conflict,
            skip_none=skip_none, zip_validator=zip_validator,
        )

    def shard_sizes(self, slice_name=None) -> list:
        """Samples per shard, in tab then shard order, for one slice."""
        slice_name = self._slicenames((slice_name,) if slice_name else ())[0]
        sizes = []
        for idx in range(self.n_tabs):
            tab = self.tab(idx)
            try:
                with tab.fs.open(tab.slice_index_path(slice_name), 'r') as f:
                    index = json.load(f)
                sizes.extend(
                    reader_from_json('.', None, meta).size
                    for meta in index.get('shards', [])
                )
            except Exception:
                pass
        return sizes

    def samples_per_shard(self, slice_name=None) -> int:
        """The largest shard's sample count -- this table's shard capacity.

        The natural block size for `sampler`: a block that size spans
        one shard, or straddles two where blocks and shard boundaries do not
        line up, which is what keeps the working set O(1) shards instead of
        O(table).  The largest rather than the mean because every tab's last
        shard is short, so the mean would understate the capacity that
        actually governs locality.
        """
        sizes = self.shard_sizes(slice_name)
        if not sizes:
            raise ValueError(
                f"{self.__class__.__name__}: no shards in "
                f"{slice_name or self.slices[0]!r}; is it built?"
            )
        return max(sizes)

    def n_samples(self, slice_name=None) -> int:
        """Total samples in a slice -- and so in any of them, since the
        slices are written in lockstep and are equal by contract."""
        return sum(self.shard_sizes(slice_name))

    def sampler(self, *, block_size=None, seed: int = 0,
                fixed_epoch: bool = False, slice_name=None) -> BlockShuffleSampler:
        """A `BlockShuffleSampler` sized to this table's own shards.

        The point of putting it here rather than leaving it to the caller:
        *block_size* wants to be the shard capacity, and this is the only
        place that knows it.  A caller passing a constant is guessing at a
        number the storage already determines -- and guessing high scatters
        reads across shards, guessing low shrinks the shuffle for no gain.

        ::

            loader = DataLoader(table.dataset(), sampler=table.sampler(),
                                batch_size=32)

        Use `fixed_epoch=True` and a distinct *seed* for a validation
        loader; pass an explicit *block_size* to override the default.
        """
        if block_size is None:
            block_size = self.samples_per_shard(slice_name)
        return BlockShuffleSampler(
            self.n_samples(slice_name), block_size,
            seed=seed, fixed_epoch=fixed_epoch,
        )

    def stats(self, *slices, **kwargs):
        """User-defined summary of the named slices.

        Shared by `DatapointTab` and `DatapointTable`: both have it,
        each dispatching to its own `__stats__()`.

        The calling sequence is one `__stats__(slice)` per named slice,
        and the return shape follows how it was called -- a bare dict for
        one *named* slice, a dict of dicts otherwise::

            tab.stats('frames')                 -> {'n_samples': 1024}
            tab.stats('frames', 'annotations')  -> {'frames': {...},
                                                    'annotations': {...}}
            tab.stats()                         -> every slice, as above
            table.stats('frames')               -> the table's own summary

        Per slice because a slice is the unit that has its own shards, its
        own index and its own length -- there is no single number that
        describes "the tab" when its slices can be read independently.  A
        summary that genuinely does not vary by slice is an `__stats__()`
        that ignores its argument, or a method of your own next to it.

        A table's `__stats__()` is not derived from its tabs' -- summing,
        averaging or taking extrema over
        `[self.tab(i).stats(slice) for i in range(self.n_tabs)]` is a
        choice only the statistic itself can make.
        """
        names = self._slicenames(slices)
        if len(names) == 1 and slices:
            return self.__stats__(names[0], **kwargs)
        return {name: self.__stats__(name, **kwargs) for name in names}

    def __stats__(self, slice_name, **kwargs) -> dict:
        """Summarise one slice.  Overridden by the subclass; see `stats()`."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __stats__(slice_name) "
            f"to support .stats()"
        )


class DatapointTab(SlicedTopics, Datablock):
    """One tab of a `DatapointTable`: a Datablock writing MDS slices.

    Fill out three things:

    * `SLICES` -- the parallel streams this tab writes.
    * `VAR` -- whatever addresses this tab's input, on top of the
      `table` and `tab_idx` already declared here.
    * `__build__()` -- write every slice, in lockstep, via
      `slice_writers()`.

    Everything else -- where the shards land, what counts as built, how the
    slices read back -- is preimplemented.

    SLICES and TOPICS
    -----------------
    `SLICES` is the only declaration a tab needs; the `data` group of
    `TOPICS` is built from it, and anything else declared in `TOPICS` is
    kept alongside::

        class FrameTab(DatapointTab):
            SLICES = ('frames', 'annotations')
            TOPICS = {'note': 'note.txt'}

        FrameTab.TOPICS
        # {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC},
        #  'note': 'note.txt'}

    So `SLICES` and `TOPICS` are not two ways of saying one thing: the
    first names the MDS streams, the second is the ordinary Datablock
    declaration, and the first *becomes* one group of the second.  Declaring
    the group directly instead works and then it defines `SLICES`; doing
    both and disagreeing is an error::

        class FrameTab(DatapointTab):
            TOPICS = {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC}}

        FrameTab.SLICES        # ('frames', 'annotations')

    `TOPICS` accumulates down the hierarchy -- a subclass adds to what it
    inherits rather than replacing it, its own entries winning a collision --
    while the `data` group is always rebuilt from `SLICES`::

        class DebuggableFrameTab(FrameTab):
            TOPICS = {'debug': {'plots': DIRTOPIC}}     # 'note' is still there
            SLICES = ('frames', 'annotations', 'depth')  # data group rebuilt

    Non-slice topics behave exactly as on any Datablock: they stay under the
    tab's own key, count towards `valid()`, and are the subclass's to
    answer in `__read__()`.  Only the slices are redirected into the
    table's per-slice roots, and only they get the non-empty-`index.json`
    rule instead of a plain existence check.

    Storage layout
    --------------
    A tab's shards do **not** live under its own `anchorkeypath`.  They
    live under the table's, in the per-slice root::

        <table anchorkeypath>/data/<slice>/index.json     <- table's merged index
        <table anchorkeypath>/data/<slice>/<tabdir>/    <- this tab's shards
        <table anchorkeypath>/tabs/<fqcn>/<key>/        <- this tab's other topics

    That is forced by `StreamingDataset`, which resolves a shard as
    `os.path.join(root, split, basename)`: a slice's merged index must sit
    at an *ancestor* of that slice's shards.  Several slices therefore cannot
    share one directory, and `'../'` is not an option because Azure Data
    Lake's REST API does not resolve it.  `dirpath()` implements the
    redirect; non-slice topics are untouched and stay under the tab's own
    key.

    Example
    -------
    ::

        class AnnotatedFrameTab(DatapointTab):
            SLICES = ('frames', 'annotations')

            @dataclass
            class VAR(DatapointTab.VAR):
                episode: str = None

            def __build__(self):
                columns = {
                    'frames':      {'idx': 'int', 'image': 'jpeg'},
                    'annotations': {'idx': 'int', 'label': 'json'},
                }
                with self.slice_writers(columns) as writers:
                    for i, (image, label) in enumerate(load(self.var.episode)):
                        writers['frames'].write({'idx': i, 'image': image})
                        writers['annotations'].write({'idx': i, 'label': label})

            def __stats__(self, slice_name):
                return {'n_samples': len(self.data(slice_name))}
    """

    @dataclass
    class VAR(Datablock.VAR):
        pass

    # ------------------------------------------------------------------ #
    # Datablock protocol
    # ------------------------------------------------------------------ #

    def __init__(self, *args, cache=None, cache_limit=None, **kwargs):
        super().__init__(*args, cache=cache, cache_limit=cache_limit, **kwargs)

    def __build__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __build__(): write every "
            f"slice in {list(self.SLICES) or ['...']} in lockstep, one sample per "
            f"slice per item, via self.slice_writers(columns)"
        )

    def __stats__(self, slice_name, **kwargs) -> dict:
        """Summarise one of this tab's slices.  Optional.

        Declared here, rather than left on the shared mixin, because this is
        where a tab author looks for it.  Reached through `stats()`, once
        per named slice::

            def __stats__(self, slice_name):
                return {'n_samples': len(self.data(slice_name))}

        A tab's slices are written in lockstep, so their counts agree by
        construction; what differs between them is what is *in* a sample,
        which is exactly what only the subclass knows.
        """
        return super().__stats__(slice_name, **kwargs)

    def __read__(self, *topicpath):
        """`read('data', slice)` is `data()`; `read('data')` is every
        slice.  Other topics are the subclass's to answer."""
        topicpath = self._normtopic(topicpath)
        if topicpath and topicpath[0] == self.DATA:
            return self.data(*topicpath[1:])
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ answers only {self.DATA!r} topics; "
            f"override it to read {'/'.join(topicpath)!r}"
        )

    # ------------------------------------------------------------------ #
    # Writing a tab's slices
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def slice_writers(self, columns, *, stage: bool = None, cache=None,
                      flush_every: int = None, **writer_kwargs):
        """One `MDSWriter` per slice, as `{slice: writer}`.

        Yields the writers, finishes them on a clean exit, and -- when
        staging -- uploads each slice's files to its target directory with
        `index.json` **last**, so a partial upload never looks complete to
        `valid_slice()`.

        On an exception nothing is finished and nothing is uploaded: the
        slice keeps no `index.json`, so the tab reports unbuilt and gets
        redone rather than half-read.  This is the same all-or-nothing rule
        the lockstep contract needs -- a tab whose first slice landed and
        whose second did not is exactly the misalignment zipping forbids.

        Parameters
        ----------
        columns : dict
            `{slice: {column: mds_type}}`, one entry per declared slice.
        stage : bool, optional
            Write to a temporary local directory and upload on success.
            Defaults to *True* for remote storage (where it is the only way
            `MDSWriter` can write at all) and *False* for local storage
            (where it would mean copying every shard twice).  A non-staged
            target directory is cleared first, so a previous failed attempt's
            orphaned shards do not end up in this attempt's index.
        cache : str, optional
            Parent of the staging directory.  Defaults to `cacheroot`.
        flush_every : int, optional
            Break every slice onto a new shard every *flush_every* samples,
            so that all slices carry **identical shard boundaries**.

            `MDSWriter` otherwise starts a new shard on a byte budget
            (`size_limit`, 64 MiB by default), so a frames slice and an
            annotations slice holding the same samples split at completely
            different places.  That is invisible to
            `ZipStreamingDataset`, which addresses by index -- but
            `ZipIterableStreamingDatasets` cannot shuffle across it, since
            the shuffle permutation is derived from the per-shard sample
            counts.  This is the knob that makes shuffled iterator-mode
            zipping possible; see that class.

            Shards come out uniform in samples and ragged in bytes, so size
            it by the *largest* slice: `flush_every` samples of that slice
            is what one of its shards will weigh, and tens of megabytes is
            the range to aim at.

            It does not override `size_limit` -- whichever comes first
            ends the shard -- so a `size_limit` that fires first would put
            a boundary in one slice and not the others.  That is detected
            and raised rather than left to surface as a misaligned shuffle;
            raise `size_limit`, or pass `size_limit=None` to let
            *flush_every* be the only thing that ends a shard.

            Also turns the lockstep contract into a checked one: writes are
            counted per slice, and a tab that has not written every slice
            the same number of times by the end raises rather than
            producing a table that cannot be zipped.
        **writer_kwargs
            Passed to every `MDSWriter` (`compression`, `size_limit`, …).
        """
        names = self.slices
        missing = [name for name in names if name not in columns]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}.slice_writers: no columns for "
                f"slice(s) {missing}; every declared slice must be written"
            )

        if stage is None:
            stage = not self.is_local_fs
        targets = {name: self.path(self.DATA, name, ensure_dirpath=True) for name in names}

        staging = None
        if stage:
            staging = tempfile.mkdtemp(
                prefix=f"{self.__class__.__name__}_",
                dir=self._ensure_cacheroot(cache),
            )
            outdirs = {name: os.path.join(staging, name) for name in names}
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
                    out=outdirs[name], columns=columns[name], **writer_kwargs,
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

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _is_slicepath(self, topicpath) -> bool:
        """True for a `(DATA, slice)` leaf -- the one shape that redirects."""
        return (len(topicpath) == 2 and topicpath[0] == self.DATA
                and topicpath[1] in self.SLICES)

    def _read_slice(self, slice_name, **kwargs):
        """Decode every sample of one slice out of this tab's shards."""
        return read_mds_shard(
            self.path(self.DATA, slice_name), self.fs,
            tmpdir=kwargs.pop('cache', None) or self._ensure_cacheroot(), **kwargs,
        )

    def _upload_slice(self, local_dir, target_dir):
        """Copy one finished slice's files up, `index.json` last."""
        names = sorted(os.listdir(local_dir))
        for name in [n for n in names if n != 'index.json'] + \
                    [n for n in names if n == 'index.json']:
            self.fs.put_file(os.path.join(local_dir, name),
                             os.path.join(target_dir, name))


class DatapointTable(SlicedTopics, Datastack):
    """A table of `DatapointTab`s, sliced the same way as its tabs.

    Fill out three things:

    * `TAB` -- the `DatapointTab` subclass.  `SLICES` is
      taken from it unless declared here.
    * `n_tabs` -- how many.
    * `__tab__()` -- tab idx's own VAR fields, on top of the
      placement `super()` fills in.  Only when a tab needs any.

    The rest of the `Datastack` protocol is preimplemented:
    `__split__()` creates the slice roots and fans one tab build out
    per index; `__stack__()` merges every tab's per-slice `index.json`
    into one index per slice and then writes the `done` marker.  Override
    either and call `super()` if a table needs more.

    Reading mirrors `DatapointTab`, over the whole table: `data()`
    concatenates the tabs' samples for a slice, `dataset()` opens the
    merged per-slice indexes and zips them, `stats()` reaches
    `__stats__()`, which is yours.

    SLICES and TOPICS
    -----------------
    A table does not declare `SLICES`: it *takes* its tab's, exactly, and
    declaring them here too is an error unless they agree.  That is not the
    accumulation `TOPICS` does -- there is one set of slice roots and the
    tabs write into them, so a slice only one of the two knows about is one
    the tab writes and the table never merges::

        class FrameTable(DatapointTable):
            TAB = FrameTab                 # SLICES = ('frames', 'annotations')

        FrameTable.SLICES                  # ('frames', 'annotations')
        FrameTable.TOPICS
        # {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC},
        #  'tabs': DIRTOPIC, 'done': 'done'}

    `tabs` and `done` are declared as ordinary `TOPICS` on this class,
    and reach a subclass by the accumulation rule rather than by any
    registry of required names.  So a table that declares topics of its own
    keeps them::

        class ReportedFrameTable(FrameTable):
            TOPICS = {'report': 'report.json'}

        ReportedFrameTable.TOPICS          # data, tabs, done, report

    Redeclaring `done` (a different filename, say) is harmless.
    Redeclaring `tabs` as a file topic is not: it is the `url=` every tab
    is formed under.

    Example
    -------
    ::

        class AnnotatedFrameTable(DatapointTable):
            TAB = AnnotatedFrameTab

            @dataclass
            class VAR(DatapointTable.VAR):
                episodes: list = None

            @property
            def n_tabs(self):
                return len(self.var.episodes)

            def __tab__(self, idx):
                return super().__tab__(idx, episode=self.var.episodes[idx])

        table = AnnotatedFrameTable(url=dbx.env('DATA_ROOT'), spec=dict(episodes=[...]),
                         parallelization='multiprocessing', n_workers=8)
        table.build()

        table.data('annotations')                  # every annotation, concatenated
        table.dataset()                            # frames+annotations, zipped
        table.dataset('annotations')               # annotations only: no image bytes
    """

    # The DatapointTab subclass this table is made of.
    TAB = None

    TOPICS = {'tabs': DIRTOPIC, 'done': 'done'}

    # ------------------------------------------------------------------ #
    # Datastack / table protocol
    # ------------------------------------------------------------------ #

    def __init__(self, *args, cache=None, cache_limit=None, **kwargs):
        super().__init__(*args, cache=cache, cache_limit=cache_limit, **kwargs)

    @classmethod
    def __resolve_topics__(cls):
        tab_slices = tuple(getattr(cls.TAB, 'SLICES', ()) or ())
        own_slices = tuple(cls.__dict__.get('SLICES') or ())
        if tab_slices:
            if not own_slices:
                cls.SLICES = tab_slices
            elif own_slices != tab_slices:
                raise ValueError(
                    f"{cls.__name__}.SLICES {list(own_slices)} disagrees with "
                    f"{cls.TAB.__name__}.SLICES {list(tab_slices)}: a table and "
                    f"its tabs share one set of slice roots, so declare the "
                    f"slices on the tab and let the table inherit them"
                )
        super().__resolve_topics__()

    @property
    def n_tabs(self) -> int:
        """Number of tabs.  Subclasses **must** override this."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement n_tabs"
        )

    @property
    def n_blocks(self) -> int:
        return self.n_tabs

    def __tab__(self, idx: int, *, tag=None, **spec) -> DatapointTab:
        """Form tab idx.  `Datastack.__block__()` for tables."""
        if self.TAB is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set TAB = <DatapointTab subclass> "
                f"(or override __tab__(idx) outright)"
            )
        return self.TAB(
            url=self.path('tabs'),
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

    def tab(self, idx: int) -> DatapointTab:
        """Tab idx, cached on this table instance."""
        return self.block(idx)

    def tabs(self) -> list:
        return self.blocks()

    def __split__(self, *args, **kwargs):
        """Create the tab root, then one maker per tab."""
        self.path('tabs', ensure_dirpath=True)
        n = self.n_tabs
        self.log.info(
            "%s: %d tabs x %d slices %s",
            self.__class__.__name__, n, len(self.slices), list(self.slices),
        )
        return [self.TabMaker(idx) for idx in range(n)], dict(build=True)

    def __stack__(self, results=None):
        """Write the done marker once all tabs are built."""
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

        if self.validtopic('done'):
            self.log.info("%s.__stack__: done marker already present",
                          self.__class__.__name__)
        else:
            with self.fs.open(self.path('done', ensure_dirpath=True), 'wb'):
                pass
            self.log.info("%s.__stack__: done marker written", self.__class__.__name__)
        return self

    def valid(self):
        return self.validtopic('done')

    def valid_slice(self, slice_name) -> bool:
        """True when every tab has a non-empty `index.json` for *slice_name*."""
        return all(
            self.tab(idx).valid_slice(slice_name) for idx in range(self.n_tabs)
        )

    def __stats__(self, slice_name, **kwargs) -> dict:
        return super().__stats__(slice_name, **kwargs)

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath and topicpath[0] == self.DATA:
            return self.data(*topicpath[1:])
        if topicpath == ('tabs',):
            return self.path('tabs')
        if topicpath == ('done',):
            return self.valid()
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ answers only {self.DATA!r}, 'tabs' "
            f"and 'done'; override it to read {'/'.join(topicpath)!r}"
        )

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def _read_slice(self, slice_name, *, tabs=None, **kwargs):
        """Concatenate the tabs' samples for one slice."""
        indices = range(self.n_tabs) if tabs is None else tabs
        samples = []
        for idx in indices:
            samples.extend(self.tab(idx).data(slice_name, **kwargs))
        return samples

    def datastream(self, slice_name, **kwargs) -> StreamingDataset:
        """One slice as a `StreamingDataset` composed of per-tab `Stream`s."""
        self._slicenames((slice_name,))
        streams = [self._tab_stream(self.tab(idx), slice_name)
                   for idx in range(self.n_tabs)]
        cacheroot = self._ensure_cacheroot(kwargs.pop('cache', None))
        cache_dir = kwargs.pop('cache_dir',
                               f"{self.fqcn}-{self.hash[:12]}-{slice_name}")
        local = os.path.join(cacheroot, cache_dir)
        os.makedirs(local, exist_ok=True)
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

    # ------------------------------------------------------------------ #
    # Worker callable
    # ------------------------------------------------------------------ #

    class TabMaker:
        """Picklable callable that forms and builds one tab in a worker."""

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


# Backward compatibility aliases
DatasampleTab = DatapointTab
DatasampleTable = DatapointTable
