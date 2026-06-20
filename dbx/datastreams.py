"""datastreams — Utilities for composing streaming datasets.

This module requires ``torch`` at *import time* (specifically
``torch.utils.data.Dataset``).  It is intentionally **not** re-exported
from ``dbx.__init__``, so the rest of ``dbx`` never forces a dependency
on PyTorch or any streaming library.

Usage::

    from dbx.datastreams import ZipStreamingDataset
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.parse

try:
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datastreams requires PyTorch.  "
        "Install it with:  pip install datablocks[torch]"
    ) from exc

from streaming.base.compression import decompress as mds_decompress
from streaming.base.format import reader_from_json


class ZipStreamingDataset(Dataset):
    """Pairs multiple :class:`StreamingDataset` objects by index.

    All datasets must have the same length.  ``__getitem__`` merges
    the sample dicts from all datasets into a single dict.

    This avoids opening per-bag tile datasets inside DataLoader
    workers (which breaks DDP barriers) by creating multiple
    rank-coordinated ``StreamingDataset`` objects at the top level.

    Parameters
    ----------
    *datasets
        One or more ``StreamingDataset`` instances to zip.
    zip_validator : callable | None
        Optional callable ``(idx, *samples) → None`` that is invoked
        with the flat index and the individual sample dicts **before**
        they are merged.  It should raise on inconsistency (e.g. when
        ``bag_name`` or ``tile_index`` disagree across the datasets).
    """

    def __init__(self, *datasets, zip_validator=None):
        lengths = [len(d) for d in datasets]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"ZipStreamingDataset requires datasets of equal length, "
                f"got {lengths}"
            )
        self.datasets = datasets
        self.zip_validator = zip_validator

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        samples = [ds[idx] for ds in self.datasets]
        if self.zip_validator is not None:
            self.zip_validator(idx, *samples)
        merged = {}
        for sample in samples:
            for k, v in sample.items():
                if v is not None:
                    merged[k] = v
        return merged


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
    3. Recursively replace ``None`` with ``{}`` via :func:`_sanitize`.
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
        # Download every file in the shard directory flat into local_dir.
        # We list + download individually rather than using fs.get(recursive=True)
        # because some fsspec backends reproduce the remote directory name as a
        # subdirectory inside the target, which would misplace the shard files.
        local_dir = tempfile.mkdtemp(prefix='mds_read_', dir=tmpdir)
        cleanup = local_dir
        for remote_file in fs.ls(shard_dir, detail=False):
            fname = os.path.basename(remote_file)
            fs.get(remote_file, os.path.join(local_dir, fname))

    try:
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

