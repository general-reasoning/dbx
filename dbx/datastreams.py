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
    from dbx.datastreams import DatastreamTab, DatastreamTable
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datastreams requires PyTorch.  "
        "Install it with:  pip install datablocks[torch]"
    ) from exc

try:
    from streaming import MDSWriter, StreamingDataset
    from streaming.base.compression import decompress as mds_decompress
    from streaming.base.format import reader_from_json
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datastreams requires mosaicml-streaming.  "
        "Install it with:  pip install datablocks[streaming]"
    ) from exc

from .datablocks import DIRTOPIC, Datablock, Datastack


def _same_value(a, b):
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


class _ZipBase:
    """Merge configuration and merge logic, shared by the two zip datasets.

    ``ZipStreamingDataset`` (map-style, index-addressed) and
    ``ZipIterableStreamingDatasets`` (iterator-style, lockstep) differ only
    in how they *obtain* one sample per source; what they then do with those
    samples is identical.  Keeping that half here means one merge policy,
    one set of error messages, and one place to change either.

    Beyond the plain merge, two things a multi-slice
    ``DatastreamTable`` needs:

    * **Per-source column projection** (*columns*), so a caller can say
      "frames, and only these two annotation columns" without paying to
      materialise the rest of the annotation row.
    * **Collision handling that need not be silent** (*shared*,
      *on_conflict*).  Sources written by one builder legitimately share
      bookkeeping keys, so a merge conflict is normal and worth resolving
      deliberately rather than by source ordering.

    Parameters
    ----------
    *datasets
        One or more ``StreamingDataset`` instances to zip.
    columns : sequence | None
        Per-dataset column projection, positionally parallel to *datasets*.
        Entry *i* is either ``None`` ("every column of dataset *i*") or an
        iterable of keys to keep.  ``None`` means no projection at all.

        Deliberately a separate argument rather than ``(dataset, columns)``
        pairs: a dataset may itself be a tuple or a list subclass, so there
        is no reliable way to tell a pair from a bare list-like dataset, and
        guessing wrong silently reinterprets the caller's data as a
        projection spec.
    shared : iterable[str] | None
        Keys expected in more than one source.  Exempt from *on_conflict*;
        the first source's value wins.
    validate_shared : bool
        Assert that every *shared* key present in more than one source holds
        an equal value at the same index -- which is what makes a mis-zipped
        stream loud instead of silently misaligned.  Off by default: it
        costs a comparison per shared key per item.
    on_conflict : {'last', 'first', 'error'}
        What to do when two sources supply the same key after projection and
        it is not in *shared*.  Defaults to ``'last'``, which is a plain
        dict merge in source order and what this class has always done;
        ``'error'`` raises ``KeyError`` naming the key and the source
        positions.
    skip_none : bool
        Drop ``None`` values while merging, so a source carrying a key as
        ``None`` does not mask another's real value.  On by default, again
        for continuity -- but a projection that means to carry a genuinely
        null column wants ``skip_none=False`` and an explicit *shared* /
        *on_conflict* policy instead.
    zip_validator : callable | None
        Optional callable ``(idx, *samples) → None`` that is invoked
        with the flat index and the individual sample dicts **before**
        they are projected or merged.  It should raise on inconsistency
        (e.g. when ``bag_name`` or ``tile_index`` disagree across the
        datasets).
    """

    def __init__(self, *datasets, columns=None, shared=None,
                 validate_shared=False, on_conflict='last', skip_none=True,
                 zip_validator=None):
        if on_conflict not in ('last', 'first', 'error'):
            raise ValueError(
                f"on_conflict must be 'last', 'first' or 'error', "
                f"got {on_conflict!r}"
            )
        if not datasets:
            raise ValueError("ZipStreamingDataset needs at least one dataset")
        lengths = [len(d) for d in datasets]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"ZipStreamingDataset requires datasets of equal length, "
                f"got {lengths}"
            )
        if columns is None:
            columns = [None] * len(datasets)
        elif len(columns) != len(datasets):
            raise ValueError(
                f"columns must be positionally parallel to datasets: got "
                f"{len(columns)} entries for {len(datasets)} datasets"
            )
        self.datasets = datasets
        self.columns = [None if cols is None else list(cols) for cols in columns]
        self.shared = set(shared or ())
        self.validate_shared = validate_shared
        self.on_conflict = on_conflict
        self.skip_none = skip_none
        self.zip_validator = zip_validator

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        samples = [ds[idx] for ds in self.datasets]
        if self.zip_validator is not None:
            self.zip_validator(idx, *samples)

        merged = {}
        origin = {}
        for pos, (sample, cols) in enumerate(zip(samples, self.columns)):
            if cols is None:
                items = sample.items()
            else:
                missing = [c for c in cols if c not in sample]
                if missing:
                    raise KeyError(
                        f"ZipStreamingDataset source {pos} has no column(s) "
                        f"{missing} at index {idx}; it provides {sorted(sample)}"
                    )
                items = ((c, sample[c]) for c in cols)

            for key, value in items:
                if self.skip_none and value is None:
                    continue
                if key not in merged:
                    merged[key] = value
                    origin[key] = pos
                    continue
                if key in self.shared:
                    if self.validate_shared and not _same_value(merged[key], value):
                        raise ValueError(
                            f"ZipStreamingDataset: shared key {key!r} disagrees "
                            f"between source {origin[key]} and source {pos} at "
                            f"index {idx} -- the streams are not aligned"
                        )
                    continue
                if self.on_conflict == 'error':
                    raise KeyError(
                        f"ZipStreamingDataset: key {key!r} supplied by both "
                        f"source {origin[key]} and source {pos}. Project it "
                        f"away, or pass shared={{{key!r}, ...}} if both are "
                        f"expected to carry it, or set on_conflict='first'/'last'."
                    )
                if self.on_conflict == 'last':
                    merged[key] = value
                    origin[key] = pos
        return merged


# The class zips *several* datasets, so the plural reads truer -- but the
# singular is the name that is already imported elsewhere, so it stays the
# canonical one and this is an alias rather than a rename.
ZippedStreamingDatasets = ZipStreamingDataset


# ═══════════════════════════════════════════════════════════════════════
#  Shard-locality-aware sampling
# ═══════════════════════════════════════════════════════════════════════

def shuffled_block_order(num_blocks: int, seed: int) -> list:
    """A random permutation of ``range(num_blocks)`` for the given seed."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_blocks, generator=generator).tolist()


class BlockShuffleSampler(Sampler):
    """Shuffles contiguous blocks of an index space, and within each block --
    instead of shuffling the whole range at once.

    For anything backed by shard-organised storage, which is what an MDS
    table is: consecutive sample indices live in the same shard.  A full
    global shuffle (``torch.randperm`` over the whole dataset, i.e. what
    ``DataLoader(shuffle=True)`` does) scatters every access across the whole
    table, so almost every batch needs a shard that is not in the local
    cache -- a cache can only help if consecutive accesses tend to land in
    the same few shards.  Shuffling in blocks keeps the *working set* down to
    a handful of shards, small enough to stay cache-resident, while both
    block order and within-block order are still randomised every epoch.

    ``DatastreamTable.sampler()`` builds one of these sized to the table's
    own shards, which is the block size worth using -- see there.

    Re-shuffled per epoch via :meth:`set_epoch`, which trainers call by the
    same convention as ``torch.utils.data.DistributedSampler``.  Pass
    ``fixed_epoch=True`` for a validation sampler: otherwise the held-out
    subset a capped validation run scores against changes every epoch, and
    val loss at one step stops being comparable to val loss at another.

    :meth:`state_dict`/:meth:`load_state_dict` track how far into the epoch
    the sampler has got, so a checkpoint saved mid-epoch can resume near
    where it left off rather than replaying the epoch.  Approximate, not
    exact: with ``num_workers > 0`` the sampler runs ahead of what has
    actually been consumed by up to ``num_workers * prefetch_factor``
    batches, so a few samples at the boundary are skipped rather than
    replayed.  See :class:`ResumableDataLoader` for the wiring.

    Parameters
    ----------
    n : int
        Length of the index space (e.g. ``len(dataset)``).
    block_size : int
        Consecutive indices per block.
    seed : int
        Base seed, combined with the epoch so each epoch shuffles
        independently but reproducibly.
    fixed_epoch : bool
        Ignore :meth:`set_epoch` and stay on epoch 0, so the order never
        changes for the life of the sampler.
    """

    def __init__(self, n: int, block_size: int, seed: int = 0,
                 fixed_epoch: bool = False):
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.n = n
        self.block_size = block_size
        self.seed = seed
        self.epoch = 0
        self._fixed_epoch = fixed_epoch
        self._consumed = 0

    def set_epoch(self, epoch: int):
        if not self._fixed_epoch and epoch != self.epoch:
            self.epoch = epoch
            self._consumed = 0

    def __len__(self):
        return self.n

    def _full_order(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        num_blocks = (self.n + self.block_size - 1) // self.block_size
        order = []
        for block in torch.randperm(num_blocks, generator=generator).tolist():
            start = block * self.block_size
            end = min(start + self.block_size, self.n)
            order.extend(
                start + p
                for p in torch.randperm(end - start, generator=generator).tolist()
            )
        return order

    def __iter__(self):
        order = self._full_order()
        start = self._consumed if self._consumed < len(order) else 0
        for i in range(start, len(order)):
            self._consumed = i + 1
            yield order[i]
        self._consumed = 0

    def state_dict(self):
        return {'epoch': self.epoch, 'consumed': self._consumed}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self._consumed = state_dict['consumed']


class ResumableDataLoader(DataLoader):
    """A ``DataLoader`` that forwards ``state_dict``/``load_state_dict`` to its
    sampler.

    Trainers that support mid-epoch resume check the *loader* for those
    methods, not the sampler, so a :class:`BlockShuffleSampler`'s resume
    state is inert unless the loader surfaces it.  Only meaningful with a
    sampler that implements them.
    """

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        self.sampler.load_state_dict(state_dict)


# ═══════════════════════════════════════════════════════════════════════
#  Collation helpers
# ═══════════════════════════════════════════════════════════════════════

def _sanitize(obj):
    """Recursively replace ``None`` with ``{}`` in nested dicts/lists."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def sanitize_collate(batch):
    """Collate a batch of sample dicts, tolerating missing or ``None`` values.

    MDS JSON columns can deserialise to ``None`` (e.g. bags without a
    valid case-id have no annotations), or be absent entirely from some
    samples.  PyTorch's ``default_collate`` rejects both cases, so we:

    1. Union all keys across the batch (some shards may omit optional
       columns entirely).
    2. Fill missing keys with ``None``.
    3. Recursively replace ``None`` with ``{}`` via ``_sanitize()``.
    4. Delegate to ``default_collate``.
    """
    from torch.utils.data._utils.collate import default_collate
    all_keys = set().union(*(s.keys() for s in batch))
    aligned = [{k: s.get(k, None) for k in all_keys} for s in batch]
    return default_collate([_sanitize(sample) for sample in aligned])


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
        os.makedirs(local, exist_ok=True)

    streaming_kwargs = dict(
        remote=remote,
        local=local,
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
            return []

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


# ═══════════════════════════════════════════════════════════════════════
#  DatastreamTab / DatastreamTable — a Datablock / Datastack pair over MDS slices
# ═══════════════════════════════════════════════════════════════════════

class SlicedTopics:
    """Mixin giving a block a ``data`` topic group of parallel MDS *slices*.

    A **slice** is one independently-readable MDS stream.  Every slice of a
    ``DatastreamTab`` is written in lockstep from one pass over that tab's
    input, so sample *i* of every slice describes the same thing: that
    alignment is the whole contract, and it is what makes zipping the slices
    back together by index meaningful.

    Slices are declared once, as ``SLICES``; the ``data`` group of
    ``TOPICS`` is synthesized from them::

        class MyTab(DatastreamTab):
            SLICES = ('frames', 'annotations')
            TOPICS = {'stats': 'stats.json'}       # optional extra topics

        MyTab.TOPICS
        # {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC},
        #  'stats': 'stats.json'}

    Declaring the ``data`` group explicitly works too and then *it* defines
    ``SLICES``.  Either way ``TOPICS`` ends up in the shape above, so
    the slices are covered by the block's signature and hence by its hash:
    adding, removing or renaming one re-keys the block rather than quietly
    reusing another shape's artifacts.

    Two things are readable off a built slice, and they are not the same
    thing:

    * ``data()`` -- the samples themselves, decoded eagerly into a list.
      Lumpy, materialised, and per slice.  Use it to inspect or aggregate.
    * ``dataset()`` -- a live ``StreamingDataset`` per slice, zipped into
      one ``torch.utils.data.Dataset``.  Lazy, index-aligned, and the thing
      you hand a ``DataLoader``.

    ``stats()`` is the third reader and the one this cannot implement:
    what a useful summary of a slice is depends entirely on what is in it,
    so ``__stats__()`` is a hook.
    """

    # Name of the topic group holding the slices.  Overridable, but every
    # method below addresses through it, so renaming it renames it everywhere.
    DATA = 'data'

    # Slice names, in the order they are zipped.  Declared by the subclass.
    SLICES = ()

    TOPICS = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__resolve_topics__()

    @classmethod
    def __resolve_topics__(cls):
        """Rebuild ``TOPICS`` as ``data`` group + inherited + own.

        TOPICS **accumulates** down the hierarchy here, rather than shadowing
        as a plain class attribute would: a subclass declaring
        ``TOPICS = {'note': 'note.txt'}`` adds to what it inherits instead of
        replacing it, and its own entries win on a collision.  That is the
        whole mechanism -- it is what lets ``DatastreamTable`` declare
        ``tabs`` and ``done`` once, in the ordinary way, and still have them
        after a subclass declares topics of its own.

        The ``data`` group is exempt and always rebuilt, from a ``data`` group
        the class declares **itself** (which then also defines ``SLICES``)
        or else from ``SLICES``.  Own-declaration is read off
        ``cls.__dict__`` rather than the attribute, because a subclass that
        declares only new ``SLICES`` inherits its parent's already-resolved
        ``data`` group -- and that stale group must lose to the new slices,
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
            slices = tuple(declared)
            data_group = dict(declared)
            own_slices = tuple(cls.__dict__.get('SLICES') or ())
            if own_slices and own_slices != slices:
                raise ValueError(
                    f"{cls.__name__} declares both SLICES {list(own_slices)} and a "
                    f"{cls.DATA!r} topic group {list(slices)}, and they disagree; "
                    f"declare one or the other"
                )
        else:
            slices = tuple(dict.fromkeys(cls.SLICES))
            data_group = {name: DIRTOPIC for name in slices}

        topics = {cls.DATA: data_group} if data_group else {}
        # Base-first, so a subclass's own entry overwrites what it inherits.
        for klass in reversed(cls.__mro__[1:]):
            inherited = klass.__dict__.get('TOPICS')
            if isinstance(inherited, dict):
                topics.update({name: node for name, node in inherited.items()
                               if name != cls.DATA})
        topics.update({name: node for name, node in own.items()
                       if name != cls.DATA})

        cls.SLICES = slices
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
        """Normalize a ``*slices`` varargs tuple; empty means *all* slices."""
        if len(slices) == 1 and isinstance(slices[0], (tuple, list)):
            slices = tuple(slices[0])
        if not slices:
            return self.slices
        unknown = [s for s in slices if s not in self.slices]
        if unknown:
            raise KeyError(
                f"{self.__class__.__name__}: unknown slice(s) {unknown}; "
                f"declared slices are {list(self.slices)}"
            )
        return tuple(slices)

    def slice_index_path(self, slice_name) -> str:
        """Path of the ``index.json`` addressing *slice_name*'s shards."""
        return os.path.join(self.path(self.DATA, slice_name), 'index.json')

    # ------------------------------------------------------------------ #
    # Validity
    # ------------------------------------------------------------------ #

    def valid_slice(self, slice_name) -> bool:
        """True when *slice_name* has a **non-empty** ``index.json``.

        Existence of the directory is not enough, and neither is existence of
        the index: ``MDSWriter.finish()`` writes an ``index.json`` even when
        nothing was ever written through it, so a tab whose every input was
        unreadable would otherwise report built and contribute an empty
        stream to the table's merged index -- which surfaces much later, and
        much less clearly, as ``Stream contains no samples``.
        """
        try:
            index_path = self.slice_index_path(slice_name)
            if not self.fs.exists(index_path):
                return False
            return bool(json.loads(self.fs.cat(index_path)).get('shards'))
        except Exception:
            return False

    def validtopic(self, *topicpath):
        """As ``Datablock.validtopic()``, but slices go through
        ``valid_slice()`` rather than a bare existence check."""
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

        The resolved form of the ``cache=`` kwarg, and named for it the way
        ``Datablock.localroot`` is named for ``local=`` -- ``cache`` is
        what was asked for (usually nothing), ``cacheroot`` is where it
        actually is.

        Defaults to ``<localroot>/streaming``: under the block's local
        staging root (``local=``, ``DBX_LOCAL``, or the url itself when that
        is already local), **not** the system temporary directory, so that a
        cache big enough to matter lands on the disk the deployment chose
        for it rather than filling ``/tmp``.

        Nothing under it is data.  It is shard downloads (bounded by
        ``cache_limit``), staged writes on their way to remote storage, and
        decompression scratch; it can be deleted at any time.
        """
        return getattr(self, 'cache', None) or os.path.join(
            self.localroot, 'streaming',
        )

    def _ensure_cacheroot(self, cache=None) -> str:
        """``cacheroot`` (or *cache*), created if it does not exist.

        ``tempfile.mkdtemp(dir=...)`` and the MDS readers both require the
        parent to exist already, and the default sits under a local staging
        root that nothing else necessarily creates.
        """
        cacheroot = cache or self.cacheroot
        os.makedirs(cacheroot, exist_ok=True)
        return cacheroot

    def data(self, *slices, **kwargs):
        """Every sample of the named slices, decoded into a list.

        One slice gives ``list[dict]``; several (or none, meaning all) give
        ``{slice: list[dict]}``.  Materialises the whole slice in memory --
        for anything training-shaped use ``dataset()`` instead.
        """
        names = self._slicenames(slices)
        if len(names) == 1 and slices:
            return self._read_slice(names[0], **kwargs)
        return {name: self._read_slice(name, **kwargs) for name in names}

    def _read_slice(self, slice_name, **kwargs):
        """Decode one slice.  Internal: ``data()`` is the override point."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _read_slice(slice_name)"
        )

    def datastream(self, slice_name, **kwargs) -> StreamingDataset:
        """One slice as a live ``StreamingDataset``.

        The singular of ``dataset()``: one slice, unzipped, exactly as
        ``streaming`` hands it over.

        The local cache directory is qualified by class, hash **and** slice
        name: the slices of one block are several ``StreamingDataset``
        objects alive in one process, and sharing a cache directory between
        them is exactly the ``Reused local directory`` collision that
        qualification exists to prevent.
        """
        self._slicenames((slice_name,))
        kwargs.setdefault('cache_dir', f"{self.fqcn}-{self.hash[:12]}-{slice_name}")
        kwargs.setdefault('cache', self.cacheroot)
        kwargs.setdefault('cache_limit', getattr(self, 'cache_limit', None))
        return open_datastream(self.path(self.DATA, slice_name), **kwargs)

    def dataset(self, *slices, columns=None, shared=None, validate_shared=False,
                on_conflict='last', skip_none=True, zip_validator=None, **kwargs):
        """The named slices, zipped by index into one ``Dataset``.

        ``dataset()`` opens every slice; ``dataset('frames')`` opens one;
        ``dataset('frames', 'annotations')`` opens those two.  A consumer
        pays only for what it opens -- skipping the largest slice genuinely
        does not fetch its shards.

        Always a ``ZipStreamingDataset``, even for a single slice, so
        the sample a caller gets back has the same merged-dict shape however
        many slices it asked for.

        The slices are opened unshuffled and zipped by *physical* index,
        which is what keeps them aligned: shuffling the slices separately
        would pair sample *i* of one with an unrelated sample *i* of
        another.  Shuffle over the zip instead -- but with
        ``sampler=table.sampler()`` rather than ``DataLoader(shuffle=True)``,
        which is the full permutation that defeats the shard cache.  See
        :meth:`sampler` and :class:`BlockShuffleSampler`.

        Parameters
        ----------
        columns : dict | None
            ``{slice: [column, ...]}`` -- project a slice down to some of its
            columns.  Keyed by slice name rather than position, since that is
            how the caller named the slices in the first place; slices absent
            from the dict are taken whole.
        shared, validate_shared, on_conflict, skip_none, zip_validator
            Merge policy, passed to ``ZipStreamingDataset``.  Slices
            written in lockstep normally share bookkeeping keys, so
            ``shared={'sample_id', ...}`` with ``validate_shared=True`` is
            the setting that turns a mis-zipped table into an error instead
            of a silent misalignment.
        **kwargs
            Passed to ``datastream()`` for every slice opened.
        """
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
        return ZipStreamingDataset(
            *datasets, columns=columns, shared=shared,
            validate_shared=validate_shared, on_conflict=on_conflict,
            skip_none=skip_none, zip_validator=zip_validator,
        )

    def shard_sizes(self, slice_name=None) -> list:
        """Samples per shard, in index order, for one slice.

        Read off the merged ``index.json`` rather than by opening the slice:
        the shard metadata carries the count, so this costs one small read
        and no downloads.
        """
        slice_name = self._slicenames((slice_name,) if slice_name else ())[0]
        with self.fs.open(self.slice_index_path(slice_name), 'r') as f:
            index = json.load(f)
        # dirname is irrelevant -- the reader is built for its metadata only.
        return [reader_from_json('.', None, meta).size
                for meta in index.get('shards', [])]

    def samples_per_shard(self, slice_name=None) -> int:
        """The largest shard's sample count -- this table's shard capacity.

        The natural block size for :meth:`sampler`: a block that size spans
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
        """A :class:`BlockShuffleSampler` sized to this table's own shards.

        The point of putting it here rather than leaving it to the caller:
        *block_size* wants to be the shard capacity, and this is the only
        place that knows it.  A caller passing a constant is guessing at a
        number the storage already determines -- and guessing high scatters
        reads across shards, guessing low shrinks the shuffle for no gain.

        ::

            loader = DataLoader(table.dataset(), sampler=table.sampler(),
                                batch_size=32)

        Use ``fixed_epoch=True`` and a distinct *seed* for a validation
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

        Shared by ``DatastreamTab`` and ``DatastreamTable``: both have it,
        each dispatching to its own ``__stats__()``.

        The calling sequence is one ``__stats__(slice)`` per named slice,
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
        summary that genuinely does not vary by slice is an ``__stats__()``
        that ignores its argument, or a method of your own next to it.

        A table's ``__stats__()`` is not derived from its tabs' -- summing,
        averaging or taking extrema over
        ``[self.tab(i).stats(slice) for i in range(self.n_tabs)]`` is a
        choice only the statistic itself can make.
        """
        names = self._slicenames(slices)
        if len(names) == 1 and slices:
            return self.__stats__(names[0], **kwargs)
        return {name: self.__stats__(name, **kwargs) for name in names}

    def __stats__(self, slice_name, **kwargs) -> dict:
        """Summarise one slice.  Overridden by the subclass; see ``stats()``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __stats__(slice_name) "
            f"to support .stats()"
        )


class DatastreamTab(SlicedTopics, Datablock):
    """One tab of a ``DatastreamTable``: a Datablock writing MDS slices.

    Fill out three things:

    * ``SLICES`` -- the parallel streams this tab writes.
    * ``VAR`` -- whatever addresses this tab's input, on top of the
      ``table`` and ``tab_idx`` already declared here.
    * ``__build__()`` -- write every slice, in lockstep, via
      ``slice_writers()``.

    Everything else -- where the shards land, what counts as built, how the
    slices read back -- is preimplemented.

    SLICES and TOPICS
    -----------------
    ``SLICES`` is the only declaration a tab needs; the ``data`` group of
    ``TOPICS`` is built from it, and anything else declared in ``TOPICS`` is
    kept alongside::

        class FrameTab(DatastreamTab):
            SLICES = ('frames', 'annotations')
            TOPICS = {'note': 'note.txt'}

        FrameTab.TOPICS
        # {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC},
        #  'note': 'note.txt'}

    So ``SLICES`` and ``TOPICS`` are not two ways of saying one thing: the
    first names the MDS streams, the second is the ordinary Datablock
    declaration, and the first *becomes* one group of the second.  Declaring
    the group directly instead works and then it defines ``SLICES``; doing
    both and disagreeing is an error::

        class FrameTab(DatastreamTab):
            TOPICS = {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC}}

        FrameTab.SLICES        # ('frames', 'annotations')

    ``TOPICS`` accumulates down the hierarchy -- a subclass adds to what it
    inherits rather than replacing it, its own entries winning a collision --
    while the ``data`` group is always rebuilt from ``SLICES``::

        class DebuggableFrameTab(FrameTab):
            TOPICS = {'debug': {'plots': DIRTOPIC}}     # 'note' is still there
            SLICES = ('frames', 'annotations', 'depth')  # data group rebuilt

    Non-slice topics behave exactly as on any Datablock: they stay under the
    tab's own key, count towards ``valid()``, and are the subclass's to
    answer in ``__read__()``.  Only the slices are redirected into the
    table's per-slice roots, and only they get the non-empty-``index.json``
    rule instead of a plain existence check.

    Storage layout
    --------------
    A tab's shards do **not** live under its own ``anchorkeypath``.  They
    live under the table's, in the per-slice root::

        <table anchorkeypath>/data/<slice>/index.json     <- table's merged index
        <table anchorkeypath>/data/<slice>/<tabdir>/    <- this tab's shards
        <table anchorkeypath>/tabs/<fqcn>/<key>/        <- this tab's other topics

    That is forced by ``StreamingDataset``, which resolves a shard as
    ``os.path.join(root, split, basename)``: a slice's merged index must sit
    at an *ancestor* of that slice's shards.  Several slices therefore cannot
    share one directory, and ``'../'`` is not an option because Azure Data
    Lake's REST API does not resolve it.  ``dirpath()`` implements the
    redirect; non-slice topics are untouched and stay under the tab's own
    key.

    Example
    -------
    ::

        class AnnotatedFrameTab(DatastreamTab):
            SLICES = ('frames', 'annotations')

            @dataclass
            class VAR(DatastreamTab.VAR):
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

    # The table is the tab's parent, so descending into it while validating
    # the tab's upstream would walk straight back into the tab.
    TREE_SKIP_VALIDATION = {'table'}

    @dataclass
    class VAR(Datablock.VAR):
        # Both required.  A tab's slices live in its table's per-slice roots,
        # so a tab without a table is one whose shards no merged index will
        # ever name; tab_idx is what orders them within it.
        #
        # Requiring them also leaves a subclass free to default its own
        # fields: a dataclass forbids a non-default field *after* a defaulted
        # one, so it is defaulting these that would force every subclass
        # field to carry a default, not leaving them required.
        table: DatastreamTable
        tab_idx: int

    # ------------------------------------------------------------------ #
    # Datablock protocol
    # ------------------------------------------------------------------ #

    def __init__(self, *args, cache=None, cache_limit=None, **kwargs):
        # Operational only: where local scratch lives and how much of it is
        # kept.  Neither changes a byte of what is written or read, so
        # neither belongs in VAR, where it would move the hash.  Left raw
        # here -- ``cacheroot`` resolves the default -- so that an
        # unset cache stays unset in the handle rather than baking one
        # machine's absolute path into it.
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
        where a tab author looks for it.  Reached through ``stats()``, once
        per named slice::

            def __stats__(self, slice_name):
                return {'n_samples': len(self.data(slice_name))}

        A tab's slices are written in lockstep, so their counts agree by
        construction; what differs between them is what is *in* a sample,
        which is exactly what only the subclass knows.
        """
        return super().__stats__(slice_name, **kwargs)

    def __read__(self, *topicpath):
        """``read('data', slice)`` is ``data()``; ``read('data')`` is every
        slice.  Other topics are the subclass's to answer."""
        topicpath = self._normtopic(topicpath)
        if topicpath and topicpath[0] == self.DATA:
            return self.data(*topicpath[1:])
        raise NotImplementedError(
            f"{self.__class__.__name__}.__read__ answers only {self.DATA!r} topics; "
            f"override it to read {'/'.join(topicpath)!r}"
        )

    def path(self, *topicpath, ensure_dirpath: bool = False, bare: bool = False,
             local: bool = False):
        """As ``Datablock.path()``, with a slice leaf redirected.

        A slice is a ``DIRTOPIC``, so the base would already answer it
        with ``dirpath()`` -- but only after resolving the node through
        TOPICS.  Short-circuiting keeps the redirect in one place, and the
        two agree on the details: the base likewise returns a DIRTOPIC leaf's
        directory without applying *bare*, and its ``ensure_dirpath`` is
        ``ensure_path()``, which is the ``fs.makedirs(..., exist_ok=True)``
        that ``dirpath(ensure=True)`` does here.

        Everything else -- a file topic, a nested group, a SYNTOPIC, the
        ``tabs``/``done`` topics of a table -- falls through to
        ``Datablock.path()`` unchanged, and is stored under this tab's own
        ``anchorkeypath`` like any Datablock's.  ``Datablock.path`` needed no
        modification for any of this.

        Groups need no case of their own.  ``Datablock.path`` describes a
        group by recursing through ``self.path`` for each member, and
        ``self`` is this class -- so ``path('data')`` comes back as
        ``{slice: redirected path}`` without the override ever seeing the
        group, and a non-slice group like ``path('debug')`` comes back
        wholly unredirected.  The same virtual dispatch is what makes
        ``validtopic()``, ``ls()``, ``list()`` and ``size()`` follow
        the redirect for free.
        """
        topicpath = self._normtopic(topicpath)
        if self._is_slicepath(topicpath):
            return self.dirpath(*topicpath, ensure=ensure_dirpath, local=local)
        return super().path(*topicpath, ensure_dirpath=ensure_dirpath, bare=bare,
                            local=local)

    def dirpath(self, *topicpath, ensure: bool = False, list: bool = False,
                local: bool = False):
        """As ``Datablock.dirpath()``, except that a *slice* leaf redirects
        into the table's per-slice root (see the class docstring)."""
        topicpath = self._normtopic(topicpath)
        if not self._is_slicepath(topicpath):
            return super().dirpath(*topicpath, ensure=ensure, list=list, local=local)

        # self.table only after the path is known to be a slice: a tab with no
        # table can still address its other topics, and only fails on the ones
        # that genuinely need one.
        dirpath = os.path.join(
            self.table.dirpath(self.DATA, topicpath[1], local=local), self.tabdir,
        )
        fs = self.localfs if local else self.fs
        if ensure:
            fs.makedirs(dirpath, exist_ok=True)
        if list:
            _lspath = dirpath if dirpath.endswith('/') else dirpath + '/'
            return fs.ls(_lspath)
        return dirpath

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def table(self):
        """The ``DatastreamTable`` this tab belongs to."""
        table = self.var.table
        if table is None:
            raise ValueError(
                f"{self.__class__.__name__} has no table, so its slices have "
                f"nowhere to live: a tab addresses its shards through its "
                f"table's per-slice roots.  Form tabs with "
                f"DatastreamTable.tab(idx), or pass "
                f"spec=dict(table=..., tab_idx=...)"
            )
        return table

    @property
    def tabdir(self) -> str:
        """This tab's directory under a slice root: its own ``key``.

        Not a naming scheme of its own.  The key is what dbx already derives
        from the tab's identity -- and the table already controls it, through
        the ``tag=`` its ``__tab__()`` supplies and the
        ``keyby`` it propagates.  So ``<table>/data/<slice>/<key>/`` and
        ``<table>/tabs/<fqcn>/<key>/`` carry the same key, and a tab whose
        spec changed writes beside the old shards rather than into them,
        for the same reason its own topics do.

        Which is also why placement cannot go through ``anchor=`` instead.
        A tab has one ``anchorkeypath``, so its slices are always siblings
        under it -- and then no directory is an ancestor of one slice
        without being an ancestor of the others too, which is precisely
        what a per-slice merged index needs.  A single-slice table (compare
        soundworld's ``PoseGridReel``) can and does hoist its one index up
        to the table instead; several slices (``PoseAggReel``) cannot.
        """
        return self.key

    # ------------------------------------------------------------------ #
    # Writing a tab's slices
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def slice_writers(self, columns, *, stage: bool = None, cache=None, **writer_kwargs):
        """One ``MDSWriter`` per slice, as ``{slice: writer}``.

        Yields the writers, finishes them on a clean exit, and -- when
        staging -- uploads each slice's files to its target directory with
        ``index.json`` **last**, so a partial upload never looks complete to
        ``valid_slice()``.

        On an exception nothing is finished and nothing is uploaded: the
        slice keeps no ``index.json``, so the tab reports unbuilt and gets
        redone rather than half-read.  This is the same all-or-nothing rule
        the lockstep contract needs -- a tab whose first slice landed and
        whose second did not is exactly the misalignment zipping forbids.

        Parameters
        ----------
        columns : dict
            ``{slice: {column: mds_type}}``, one entry per declared slice.
        stage : bool, optional
            Write to a temporary local directory and upload on success.
            Defaults to *True* for remote storage (where it is the only way
            ``MDSWriter`` can write at all) and *False* for local storage
            (where it would mean copying every shard twice).  A non-staged
            target directory is cleared first, so a previous failed attempt's
            orphaned shards do not end up in this attempt's index.
        cache : str, optional
            Parent of the staging directory.  Defaults to ``cacheroot``.
        **writer_kwargs
            Passed to every ``MDSWriter`` (``compression``, ``size_limit``, …).
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
                # MDSWriter refuses a non-empty output directory, and shards
                # left by a failed attempt would otherwise be indexed by this
                # one.  Only reached when the tab is invalid -- build()
                # skips a valid block -- so nothing complete is discarded.
                if self.fs.exists(outdir):
                    self.fs.rm(outdir, recursive=True)
                self.fs.makedirs(outdir, exist_ok=True)

        writers = {}
        try:
            for name in names:
                writers[name] = MDSWriter(
                    out=outdirs[name], columns=columns[name], **writer_kwargs,
                )
            yield writers
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
        """True for a ``(DATA, slice)`` leaf -- the one shape that redirects."""
        return (len(topicpath) == 2 and topicpath[0] == self.DATA
                and topicpath[1] in self.SLICES)

    def _read_slice(self, slice_name, **kwargs):
        """Decode every sample of one slice out of this tab's shards."""
        return read_mds_shard(
            self.path(self.DATA, slice_name), self.fs,
            tmpdir=kwargs.pop('cache', None) or self._ensure_cacheroot(), **kwargs,
        )

    def _upload_slice(self, local_dir, target_dir):
        """Copy one finished slice's files up, ``index.json`` last.

        Not ``Datablock.push()``, in either of its forms, and the reasons
        differ:

        * ``push(local_dir, target_dir)`` -- the directory form -- is one
          recursive ``fs.put``, which uploads the index at an arbitrary point
          among the shards.  ``valid_slice()`` reads a non-empty
          ``index.json`` as "this slice is built", so an interrupted
          recursive upload could leave a slice that reports built while
          shards its index names are still missing.  Sending the index last
          makes an interrupted upload look like what it is: unbuilt, and
          retried.  Ordering is the whole job here, and the directory form
          cannot express it.
        * ``push(file, file)`` in this loop -- the file form -- would add
          three redundant calls per file: an ``exists`` and an ``isdir`` on a
          path just listed, and a ``makedirs`` of a target directory
          ``slice_writers()`` already ensured once.  Against thousands of
          shards on an object store that is real traffic bought for nothing,
          since the src/dest no-op and the progress callback do not apply
          here either.

        So ``fs.put_file`` directly, and the ordering in plain sight.
        """
        names = sorted(os.listdir(local_dir))
        for name in [n for n in names if n != 'index.json'] + \
                    [n for n in names if n == 'index.json']:
            self.fs.put_file(os.path.join(local_dir, name),
                             os.path.join(target_dir, name))



class DatastreamTable(SlicedTopics, Datastack):
    """A table of ``DatastreamTab``\\ s, sliced the same way as its tabs.

    Fill out three things:

    * ``TAB`` -- the ``DatastreamTab`` subclass.  ``SLICES`` is
      taken from it unless declared here.
    * ``n_tabs`` -- how many.
    * ``__tab__()`` -- tab *idx*'s own VAR fields, on top of the
      placement ``super()`` fills in.  Only when a tab needs any.

    The rest of the ``Datastack`` protocol is preimplemented:
    ``__split__()`` creates the slice roots and fans one tab build out
    per index; ``__stack__()`` merges every tab's per-slice ``index.json``
    into one index per slice and then writes the ``done`` marker.  Override
    either and call ``super()`` if a table needs more.

    Reading mirrors ``DatastreamTab``, over the whole table: ``data()``
    concatenates the tabs' samples for a slice, ``dataset()`` opens the
    merged per-slice indexes and zips them, ``stats()`` reaches
    ``__stats__()``, which is yours.

    SLICES and TOPICS
    -----------------
    A table does not declare ``SLICES``: it *takes* its tab's, exactly, and
    declaring them here too is an error unless they agree.  That is not the
    accumulation ``TOPICS`` does -- there is one set of slice roots and the
    tabs write into them, so a slice only one of the two knows about is one
    the tab writes and the table never merges::

        class FrameTable(DatastreamTable):
            TAB = FrameTab                 # SLICES = ('frames', 'annotations')

        FrameTable.SLICES                  # ('frames', 'annotations')
        FrameTable.TOPICS
        # {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC},
        #  'tabs': DIRTOPIC, 'done': 'done'}

    ``tabs`` and ``done`` are declared as ordinary ``TOPICS`` on this class,
    and reach a subclass by the accumulation rule rather than by any
    registry of required names.  So a table that declares topics of its own
    keeps them::

        class ReportedFrameTable(FrameTable):
            TOPICS = {'report': 'report.json'}

        ReportedFrameTable.TOPICS          # data, tabs, done, report

    Redeclaring ``done`` (a different filename, say) is harmless.
    Redeclaring ``tabs`` as a file topic is not: it is the ``url=`` every tab
    is formed under.

    Example
    -------
    ::

        class AnnotatedFrameTable(DatastreamTable):
            TAB = AnnotatedFrameTab

            @dataclass
            class VAR(DatastreamTable.VAR):
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

    # The DatastreamTab subclass this table is made of.
    TAB = None

    # 'tabs' holds each tab's own (non-slice) topics; 'done' is the marker
    # that says every slice index has been merged.  The slice roots come from
    # SLICES.  Declared as ordinary TOPICS: a subclass declaring topics of its
    # own accumulates onto these rather than replacing them (see
    # SlicedTopics.__resolve_topics__), so they survive without a second
    # class attribute to remember.
    TOPICS = {'tabs': DIRTOPIC, 'done': 'done'}

    # ------------------------------------------------------------------ #
    # Datastack / table protocol
    # ------------------------------------------------------------------ #

    def __init__(self, *args, cache=None, cache_limit=None, **kwargs):
        super().__init__(*args, cache=cache, cache_limit=cache_limit, **kwargs)

    @classmethod
    def __resolve_topics__(cls):
        # A table is sliced exactly like its tabs, so it takes SLICES from
        # TAB rather than restating them.  Restating them is allowed --
        # a table may set TAB dynamically in __tab__ -- but a *disagreement*
        # is not: a tab writes its shards into the table's slice root, so a
        # slice only one of the two knows about is one the tab writes and
        # the table never merges, which surfaces as silently missing samples.
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

    def __tab__(self, idx: int, *, tag=None, **spec) -> DatastreamTab:
        """Form tab *idx*.  ``Datastack.__block__()`` for tables.

        Nothing about a tab is special: this is the same hook, at the same
        point, and it runs inside the worker for the same reason.  What the
        base implementation adds is only the placement a tab cannot be
        correct without -- the url under ``tabs``, ``table``/``tab_idx`` in
        the spec, the storage and cache settings inherited from the table --
        so a subclass supplies the tab's *own* VAR fields and leaves the rest
        to ``super()``::

            def __tab__(self, idx):
                return super().__tab__(idx, episode=self.var.episodes[idx])

        Override it outright instead if a tab needs something else entirely;
        then only ``TAB`` goes unused.  Either way it runs inside the
        worker, so it must not depend on anything the table picked up after
        being pickled -- read off ``self.var``, or off something
        ``__split__()`` wrote to storage.

        Parameters
        ----------
        idx : int
        tag : str, optional
            Tab *idx*'s tag -- its readable name in paths and logs, and by
            way of ``key`` its directory under each slice root.  Defaults to
            ``tab_<idx>``.
        **spec
            The tab's own VAR fields.
        """
        if self.TAB is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set TAB = <DatastreamTab subclass> "
                f"(or override __tab__(idx) outright)"
            )
        return self.TAB(
            url=self.path('tabs'),
            storage_options=self.storage_options,
            capture_output=self.capture_output,
            cache=getattr(self, 'cache', None),
            cache_limit=getattr(self, 'cache_limit', None),
            verbose=False,
            spec=dict(table=self, tab_idx=idx, **spec),
            tag=tag if tag is not None else f"tab_{idx:06d}",
        )

    def __block__(self, idx: int) -> DatastreamTab:
        return self.__tab__(idx)

    def tab(self, idx: int) -> DatastreamTab:
        """Tab *idx*, cached on this table instance."""
        return self.block(idx)

    def tabs(self) -> list:
        return self.blocks()

    def __split__(self, *args, **kwargs):
        """Create the tab root and the slice roots, then one maker per tab.

        The slice roots must exist before any tab writes: a tab addresses
        its shards *through* them, and on a local filesystem a missing parent
        is a hard error rather than the no-op it is on an object store.
        """
        self.path('tabs', ensure_dirpath=True)
        for name in self.slices:
            self.path(self.DATA, name, ensure_dirpath=True)
        n = self.n_tabs
        self.log.info(
            "%s: %d tabs x %d slices %s",
            self.__class__.__name__, n, len(self.slices), list(self.slices),
        )
        return [self.TabMaker(idx) for idx in range(n)], dict(build=True)

    def __stack__(self, results=None):
        """Merge each slice's per-tab indexes, then write the done marker."""
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

        # Every index before the marker: the marker is what valid() reads.
        for name in self.slices:
            if self.fs.exists(self.slice_index_path(name)):
                self.log.info("%s.__stack__: %s index present, skipping",
                              self.__class__.__name__, name)
            else:
                self.build_index(name)

        if self.validtopic('done'):
            self.log.info("%s.__stack__: done marker already present",
                          self.__class__.__name__)
        else:
            # ensure_dirpath: 'done' is a file inside its own topic directory
            # and nothing else creates that directory.  Invisible on an object
            # store, which has no real directories; a hard failure locally.
            with self.fs.open(self.path('done', ensure_dirpath=True), 'wb'):
                pass
            self.log.info("%s.__stack__: done marker written", self.__class__.__name__)
        return self

    def valid(self):
        # The done marker, not the topics: it is written last, after every
        # slice index is merged, so it is the only thing that means "this
        # table is readable" rather than "some tab got that far".  One
        # exists() rather than one per slice, which matters on object stores.
        return self.validtopic('done')

    def __stats__(self, slice_name, **kwargs) -> dict:
        """Summarise one of this table's slices, across every tab.  Optional.

        Reached through ``stats()``, once per named slice.  The per-tab
        material is one comprehension, and how to combine it is the part
        only the statistic itself can decide::

            def __stats__(self, slice_name):
                per_tab = [self.tab(i).stats(slice_name)
                           for i in range(self.n_tabs)]
                return {'n_samples': sum(s['n_samples'] for s in per_tab)}

        Forming every tab is not free, so a table with many of them is
        usually better served by a figure ``__stack__()`` already persisted.
        """
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
        """Concatenate the tabs' samples for one slice.

        Read tab by tab rather than through the merged index: the merged
        index names shards in per-tab subdirectories, and ``read_mds_shard()``
        stages a remote shard directory flat, which would collide those names.
        """
        indices = range(self.n_tabs) if tabs is None else tabs
        samples = []
        for idx in indices:
            samples.extend(self.tab(idx).data(slice_name, **kwargs))
        return samples

    # ------------------------------------------------------------------ #
    # Index merging and worker callables
    # ------------------------------------------------------------------ #

    class TabMaker:
        """Picklable callable that forms and builds one tab in a worker.

        Carries only the index; the tab itself is constructed inside
        ``__call__()`` so the main process never instantiates N tabs.
        The table arrives as ``ctx_args[0]``, shared per worker.
        """

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

    class TabIndexFetcher:
        """Reads one tab's ``index.json`` for one slice, rebasing its shards.

        Each shard's ``basename`` is rewritten from tab-relative to
        slice-root-relative, because the slice root is the single directory
        the consuming ``StreamingDataset`` is constructed with.  Nested in the
        table for namespace locality; pickles via ``__qualname__``.
        """

        def __init__(self, tab_idx: int, slice_name: str):
            self.tab_idx = tab_idx
            self.slice_name = slice_name

        def __call__(self, table):
            tab = table.__tab__(self.tab_idx)
            tab_dir = tab.path(table.DATA, self.slice_name)
            slice_root = table.path(table.DATA, self.slice_name)
            fs = table.fs
            index_path = os.path.join(tab_dir, 'index.json')
            if not fs.exists(index_path):
                return []
            with fs.open(index_path, 'r') as f:
                index = json.load(f)
            prefix = os.path.relpath(
                fs._strip_protocol(tab_dir), fs._strip_protocol(slice_root),
            )
            shards = []
            for shard in index.get('shards', []):
                shard = dict(shard)
                for key in ('raw_data', 'zip_data'):
                    if shard.get(key):
                        entry = dict(shard[key])
                        entry['basename'] = os.path.join(prefix, entry['basename'])
                        shard[key] = entry
                shards.append(shard)
            return shards

    def build_index(self, slice_name):
        """Merge every tab's ``index.json`` for one slice into one index.

        Shards are concatenated in tab-index order, and the same order is
        used for every slice -- which is what carries the tabs' per-sample
        lockstep up to the table, so that sample *i* of one merged slice and
        sample *i* of another still describe the same thing.
        """
        slice_root = self.path(self.DATA, slice_name, ensure_dirpath=True)
        fetchers = [self.TabIndexFetcher(idx, slice_name)
                    for idx in range(self.n_tabs)]
        self.log.info("%s.build_index[%s]: merging %d tabs",
                      self.__class__.__name__, slice_name, len(fetchers))

        executor_kwargs = dict(
            n_workers=self.n_workers,
            tag=f"{self.__class__.__name__}.build_index[{slice_name}] x{len(fetchers)}",
        )
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = True
        # exec_callables places results by item index, so this stays in tab
        # order however the workers happened to interleave.
        results = self.executor_cls(**executor_kwargs).exec_callables(fetchers, self)

        shards = [shard for result in (results or []) for shard in (result or [])]
        index_path = os.path.join(slice_root, 'index.json')
        with self.fs.open(index_path, 'w') as f:
            json.dump({'version': 2, 'shards': shards}, f)
        self.log.info("%s.build_index[%s]: %d shards merged -> %s",
                      self.__class__.__name__, slice_name, len(shards), index_path)



