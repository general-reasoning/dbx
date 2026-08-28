"""
Tests for dbx.exec() writing a journal entry when the evaluated result is a Datablock.

Covers:
    r = __eval__(s, globals(), cxt)
    if isinstance(r, Datablock):
        r.write_journal_entry(event="dbx:exec", note=s, inline_note=True)
    return r
"""

from dataclasses import dataclass
import pandas as pd
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


def test_exec_writes_journal_entry_for_datablock(tmp_path, monkeypatch):
    """dbx.exec(s) records to $DBX_URL eval journal and NOT to the Datablock journal."""
    dbx_url = str(tmp_path / "exec_root")
    monkeypatch.setenv('DBX_URL', dbx_url)

    url = str(tmp_path / "exec_block")
    expr = f"ExecSampleBlock(url='{url}')"
    res = dbx.exec(expr, ExecSampleBlock=ExecSampleBlock)

    assert isinstance(res, ExecSampleBlock)
    res.build()
    # Datablock journal should NOT contain note string
    j_block = res.journal()
    if not j_block.empty and 'event' in j_block.columns:
        assert 'dbx:exec' not in j_block['event'].values

    # $DBX_URL eval journal MUST contain the expression
    df_eval = dbx.journal()
    assert isinstance(df_eval, pd.DataFrame)
    assert not df_eval.empty
    assert expr in df_eval['exec'].values


def test_exec_returns_non_datablock_without_journal_error():
    """dbx.exec(s) returning a non-Datablock (int, str, dict) works without error."""
    assert dbx.exec("1 + 1") == 2
    assert dbx.exec("'hello'") == "hello"
    assert dbx.exec("dict(a=1, b=2)") == {"a": 1, "b": 2}


def test_exec_with_kwargs_and_datablock(tmp_path, monkeypatch):
    """dbx.exec(s, **kwargs) passes kwargs into context and records in eval journal."""
    dbx_url = str(tmp_path / "exec_root_kw")
    monkeypatch.setenv('DBX_URL', dbx_url)

    url = str(tmp_path / "exec_kw_block")
    expr = f"ExecSampleBlock(url='{url}', spec=dict(val=val_param))"
    res = dbx.exec(expr, ExecSampleBlock=ExecSampleBlock, val_param=99)

    assert isinstance(res, ExecSampleBlock)
    assert res.var.val == 99

    df_eval = dbx.journal()
    assert expr in df_eval['exec'].values
