from dataclasses import dataclass
import os
import pandas as pd
import pytest

import dbx
from dbx.datablocks import Datablock, DatajournalEntry, journal, record_exec_journal, read_exec_journal


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
