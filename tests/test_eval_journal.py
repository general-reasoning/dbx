from dataclasses import dataclass
import os
import pandas as pd
import pytest

import dbx
from dbx.datablocks import Datablock, DatajournalEntry, journal
from dbx.dataparts import write_exec_journal, read_exec_journal


class Built(Datablock):
    TOPICS = ['output']

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        self._write_str('output', 'hello')


class TestDatajournalEntryMethods:

    def test_datajournal_entry_methods(self):
        series = pd.Series({
            'signature': 'test.Anchor(spec={})',
            'type': 'test.Anchor(spec={})/version=1.0',
            'hash': '12345',
            'anchor': 'test.Anchor'
        })
        entry = DatajournalEntry(series)

        # Property access
        assert entry.signature == 'test.Anchor(spec={})'
        assert entry.type == 'test.Anchor(spec={})/version=1.0'

        # Method access
        assert entry.signature() == 'test.Anchor(spec={})'
        assert entry.sig() == 'test.Anchor(spec={})'

        assert entry.type() == 'test.Anchor(spec={})/version=1.0'
        assert entry.tp() == 'test.Anchor(spec={})/version=1.0'

    def test_datajournal_entry_methods_none(self):
        series = pd.Series({'hash': '12345', 'anchor': 'test.Anchor'})
        entry = DatajournalEntry(series)

        assert entry.signature is None
        assert entry.sig() is None
        assert entry.type is None
        assert entry.tp() is None


class TestEvalJournal:

    def test_exec_records_to_eval_journal(self, tmp_path, monkeypatch):
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        block = Built(url=dbx_url, spec={'x': 1})
        block.build()

        # Execute dbx expression string
        expr = f"dbx.datablocks.Datablock(url={dbx_url!r}, spec={{'x': 1}})"
        res = dbx.exec(expr)
        assert isinstance(res, Datablock)

        # Verify no note eval string injection in block journal
        j_block = block.journal()
        if 'note' in j_block.columns:
            notes = j_block['note'].dropna().tolist()
            assert expr not in notes

        # Verify eval journal recorded entry
        df_journal = dbx.journal()
        assert isinstance(df_journal, pd.DataFrame)
        assert not df_journal.empty
        assert 'exec' in df_journal.columns
        assert 'datetime' in df_journal.columns
        assert 'id' in df_journal.columns
        assert expr in df_journal['exec'].tolist()

    def test_write_exec_journal_before_eval_failure(self, tmp_path, monkeypatch):
        """write_exec_journal is called before __eval__ so failing expressions are recorded."""
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        bad_expr = "1 / 0"
        with pytest.raises(ZeroDivisionError):
            dbx.exec(bad_expr)

        j = dbx.journal()
        assert bad_expr in j['exec'].tolist()

    def test_read_exec_journal_options(self, tmp_path, monkeypatch):
        """read_exec_journal and dbx.journal support loc, iloc, filter, index, n_workers, and log."""
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        write_exec_journal("expr1", url=dbx_url)
        write_exec_journal("expr2", url=dbx_url)
        write_exec_journal("expr3", url=dbx_url)

        # Full journal
        j = read_exec_journal(url=dbx_url)
        assert len(j) == 3

        # iloc / loc access (most recent first)
        entry_0 = read_exec_journal(url=dbx_url, iloc=0)
        assert isinstance(entry_0, pd.Series)
        assert entry_0['exec'] == 'expr3'

        entry_last = dbx.journal(iloc=2)
        assert isinstance(entry_last, pd.Series)
        assert entry_last['exec'] == 'expr1'

        # Filter
        j_filtered = dbx.journal(exec='expr2')
        assert len(j_filtered) == 1
        assert j_filtered.iloc[0]['exec'] == 'expr2'

        # Index
        j_indexed = read_exec_journal(url=dbx_url, index='exec')
        assert 'expr2' in j_indexed.index
        assert isinstance(j_indexed.loc['expr2'], pd.Series)

