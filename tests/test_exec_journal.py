"""
Tests for dbx.exec() writing a journal entry when the evaluated result is a Datablock.

Covers:
    r = __eval__(s, globals(), cxt)
    if isinstance(r, Datablock):
        r.write_journal_entry(event="dbx:exec", note=s, inline_note=True)
    return r
"""

from dataclasses import dataclass
import pytest

import dbx
from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class ExecSampleBlock(Datablock):
    """Minimal Datablock for dbx.exec testing."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        val: int = 42

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write(str(self.var.val))


def test_exec_writes_journal_entry_for_datablock(tmp_path):
    """dbx.exec(s) must write event='dbx:exec' to the journal when s evaluates to a Datablock."""
    url = str(tmp_path / "exec_block")
    expr = f"ExecSampleBlock(url='{url}')"
    res = dbx.exec(expr, ExecSampleBlock=ExecSampleBlock)

    assert isinstance(res, ExecSampleBlock)
    journal = res.journal()
    assert len(journal) >= 1
    exec_entries = journal[journal['event'] == 'dbx:exec']
    assert len(exec_entries) == 1
    entry = exec_entries.iloc[0]
    assert entry['message'] == expr


def test_exec_returns_non_datablock_without_journal_error():
    """dbx.exec(s) returning a non-Datablock (int, str, dict) works without error."""
    assert dbx.exec("1 + 1") == 2
    assert dbx.exec("'hello'") == "hello"
    assert dbx.exec("dict(a=1, b=2)") == {"a": 1, "b": 2}


def test_exec_with_kwargs_and_datablock(tmp_path):
    """dbx.exec(s, **kwargs) passes kwargs into context and journals the event."""
    url = str(tmp_path / "exec_kw_block")
    expr = f"ExecSampleBlock(url='{url}', spec=dict(val=val_param))"
    res = dbx.exec(expr, ExecSampleBlock=ExecSampleBlock, val_param=99)

    assert isinstance(res, ExecSampleBlock)
    assert res.var.val == 99

    journal = res.journal()
    exec_entries = journal[journal['event'] == 'dbx:exec']
    assert len(exec_entries) == 1
    assert exec_entries.iloc[0]['message'] == expr
