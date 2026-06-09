"""datastreams — Utilities for composing streaming datasets.

This module requires ``torch`` at *import time* (specifically
``torch.utils.data.Dataset``).  It is intentionally **not** re-exported
from ``dbx.__init__``, so the rest of ``dbx`` never forces a dependency
on PyTorch or any streaming library.

Usage::

    from dbx.datastreams import ZipStreamingDataset
"""

from __future__ import annotations

try:
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dbx.datastreams requires PyTorch.  "
        "Install it with:  pip install datablocks[torch]"
    ) from exc


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
