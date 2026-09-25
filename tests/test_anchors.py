"""
Tests for dbx.anchors(): the top-level directories of a datalake root.
"""
import os

import pytest

import dbx


def _lake(tmp_path):
    root = tmp_path / 'lake'
    for d in ('pkg.mod.Block', 'custom_anchor/k1', '.journal/exec', '.dbx'):
        (root / d).mkdir(parents=True)
    (root / 'stray.txt').write_text('not an anchor')
    return root


def test_anchors_lists_top_level_dirs_only(tmp_path):
    assert dbx.anchors(str(_lake(tmp_path))) == ['custom_anchor', 'pkg.mod.Block']


def test_anchors_defaults_to_dbx_root_ahead_of_dbx_url(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_ROOT', str(_lake(tmp_path)))
    monkeypatch.setenv('DBX_URL', str(tmp_path / 'elsewhere'))
    assert dbx.anchors() == ['custom_anchor', 'pkg.mod.Block']


def test_anchors_falls_back_to_dbx_url(tmp_path, monkeypatch):
    monkeypatch.delenv('DBX_ROOT', raising=False)
    monkeypatch.setenv('DBX_URL', str(_lake(tmp_path)))
    assert dbx.anchors() == ['custom_anchor', 'pkg.mod.Block']


def test_anchors_resolves_a_specline_url(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_TEST_LAKE', str(_lake(tmp_path)))
    assert dbx.anchors("$dbx.getenv('DBX_TEST_LAKE')") == ['custom_anchor', 'pkg.mod.Block']


def test_anchors_of_missing_root_is_empty(tmp_path):
    assert dbx.anchors(str(tmp_path / 'nope')) == []


def test_anchors_without_any_url_raises(monkeypatch):
    monkeypatch.delenv('DBX_ROOT', raising=False)
    monkeypatch.delenv('DBX_URL', raising=False)
    with pytest.raises(ValueError, match='DBX_ROOT'):
        dbx.anchors()


def test_anchors_on_memory_fs():
    import fsspec
    fs = fsspec.filesystem('memory')
    fs.makedirs('/anchorslake/a.B/key', exist_ok=True)
    fs.makedirs('/anchorslake/.journal', exist_ok=True)
    try:
        assert dbx.anchors('memory:///anchorslake') == ['a.B']
    finally:
        fs.rm('/anchorslake', recursive=True)
