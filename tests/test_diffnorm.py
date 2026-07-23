"""
Tests for Datablock.diffnorm(other_norm=None, *, journal=None).

The ``journal`` selector dict must carry exactly one of ``entry_path``,
``iloc``, or ``loc``; the matching journal entry supplies the "other" norm.

Verifies:
1. A raw other_norm still diffs key-by-key (no journal).
2. journal={'iloc': -1} reads the last entry's norm.
3. journal={'loc': <label>} reads that entry's norm.
4. journal={'entry_path': <parquet path>} reads that file's norm.
5. Comparing a block to its own just-built entry yields no diff.
6. Supplying none / more than one selector raises ValueError.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class NormBlock(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        x: int = 1

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write('data')


def _built_pair(tmp_path):
    """Build ``a`` (x=1) which writes the journal; return (a, b) where ``b``
    (x=2) shares the anchor's journal but is not built."""
    a = NormBlock(url=str(tmp_path), spec={'x': 1})
    a.build()
    b = NormBlock(url=str(tmp_path), spec={'x': 2})
    return a, b


class TestRawOtherNorm:

    def test_diffs_against_string(self, tmp_path):
        a, b = _built_pair(tmp_path)
        diff = b.diffnorm(a.norm())
        assert diff == {'spec': ("{'x': '2'}", "{'x': '1'}")}

    def test_identical_norm_no_diff(self, tmp_path):
        a, b = _built_pair(tmp_path)
        assert a.diffnorm(a.norm()) == {}


class TestJournalSelectors:

    def test_iloc(self, tmp_path):
        a, b = _built_pair(tmp_path)
        assert b.diffnorm(journal={'iloc': -1}) == {'spec': ("{'x': '2'}", "{'x': '1'}")}

    def test_loc(self, tmp_path):
        a, b = _built_pair(tmp_path)
        loc = a.journal().index[-1]
        assert b.diffnorm(journal={'loc': loc}) == {'spec': ("{'x': '2'}", "{'x': '1'}")}

    def test_entry_path(self, tmp_path):
        a, b = _built_pair(tmp_path)
        entry_path = a.journal()['entry_path'].iloc[-1]
        assert b.diffnorm(journal={'entry_path': entry_path}) == {'spec': ("{'x': '2'}", "{'x': '1'}")}

    def test_self_against_own_entry(self, tmp_path):
        a, b = _built_pair(tmp_path)
        assert a.diffnorm(journal={'iloc': -1}) == {}


class TestSelectorValidation:

    def test_empty_raises(self, tmp_path):
        a, b = _built_pair(tmp_path)
        with pytest.raises(ValueError):
            b.diffnorm(journal={})

    def test_multiple_raises(self, tmp_path):
        a, b = _built_pair(tmp_path)
        with pytest.raises(ValueError):
            b.diffnorm(journal={'iloc': 0, 'loc': 0})
