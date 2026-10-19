"""A partition's layout is read once per process, shared by every instance, and memory-mapped.

Each core tablet evaluates partition instances of its own, so a layout cached
on the instance was read again by every one of them, and every copy kept: ten
tablets of a taps clip held 34 copies of one 12.9 MB layout.
"""
import os

import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

import dbx.dataparts as dataparts
from dbx.databits import eval as dbx_eval

from test_core_partition import core, points  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.fixture
def loads(monkeypatch):
    """The paths `np.load` reads, as it reads them."""
    seen, load = [], np.load

    def counting(file, *args, **kwargs):
        seen.append(file)
        return load(file, *args, **kwargs)

    monkeypatch.setattr(np, 'load', counting)
    return seen


def test_every_instance_shares_one_read_of_the_layout(tmp_path, loads):
    p = core(tmp_path, points(tmp_path))
    again = dbx_eval(p.quote())
    assert again is not p and again.hash == p.hash
    loads.clear()
    first = p._layouts_(0)
    assert again._layouts_(0) is first and p._layouts_(0) is first
    assert len(loads) == 1, loads


def test_the_layout_is_memory_mapped_and_read_only(tmp_path):
    p = core(tmp_path, points(tmp_path))
    layout = p._layouts_(0)
    assert isinstance(layout, np.memmap)
    assert not layout.flags.writeable
    np.testing.assert_array_equal(layout, p.layout(0))


def test_lots_are_as_the_layout_has_them(tmp_path):
    p = core(tmp_path, points(tmp_path))
    layout = p.layout(0)
    for idx, (start, end) in enumerate(p._assignments_[0]):
        tabs = layout[start:end, 0]
        expected = [int(t) for t in tabs[np.r_[0, np.flatnonzero(np.diff(tabs)) + 1]]] if len(tabs) else []
        assert p.lot_tablets(0, idx) == expected


def test_a_layout_written_again_is_read_again(tmp_path):
    p = core(tmp_path, points(tmp_path))
    before = p._layouts_(0)
    path = os.path.join(p.path('layout'), '0.npy')
    np.save(path, np.concatenate([np.asarray(before), np.asarray(before[:1])]))
    after = dbx_eval(p.quote())._layouts_(0)
    assert after is not before and len(after) == len(before) + 1


def test_a_remote_layout_is_staged_locally_and_mapped(tmp_path, loads):
    p = core(tmp_path, points(tmp_path))
    remote = os.path.join(p.path('layout'), '0.npy')
    # The same files, through a filesystem that is not local: what a lake on blob storage is.
    p.fs = DirFileSystem(path='/', fs=LocalFileSystem())
    staging = tmp_path / 'staging'
    p.localfs, p.localroot, p.local = LocalFileSystem(), str(staging), str(staging)
    layout = p._layouts_(0)
    assert isinstance(layout, np.memmap)
    assert str(loads[-1]).startswith(str(staging)), loads
    np.testing.assert_array_equal(layout, np.load(remote))
