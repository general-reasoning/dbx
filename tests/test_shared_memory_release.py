"""
A StreamingDataset's shared memory is released when the dataset is.

mosaicml-streaming registers each SharedMemory it creates with ``atexit``, so
the registry kept every segment -- and its file descriptors -- until the process
exited. A process opening a dataset per tab ran out of descriptors: the suite
itself died of it, in Metal failing to load its shader library ("Too many open
files").
"""
import gc
import os
import sys

import pytest

pytest.importorskip("streaming")

sys.path.insert(0, os.path.dirname(__file__))
from test_datafeaturetab import DummySampleTab  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


def _open_fds():
    gc.collect()
    return len(os.listdir('/dev/fd'))


def test_opening_datasets_does_not_accumulate_descriptors(tmp_path):
    tab = DummySampleTab(datalake=str(tmp_path), tag='s').build()
    ds = tab.dataset('samples')          # warm: imports, caches, the first segments
    ds[0]
    del ds
    before = _open_fds()
    for _ in range(10):
        ds = tab.dataset('samples')
        ds[0]
        del ds
    grown = _open_fds() - before
    assert grown < 10, f"{grown} descriptors left open by 10 datasets opened and dropped"
