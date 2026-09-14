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

        # The ENTRY: properties over the row's own columns.
        assert entry.block.signature() == 'test.Anchor(spec={})'
        assert entry.block.type() == 'test.Anchor(spec={})/version=1.0'
        assert entry.block.signature() == 'test.Anchor(spec={})'
        assert entry.block.sig() == "{'spec': {}}"
        assert entry.block.type() == 'test.Anchor(spec={})/version=1.0'
        assert entry.block.tp() == "{'paths': None, 'signature': {'spec': {}}, 'topics': (), 'version': '1.0'}"

        # The BLOCK: Datablock-shaped, so these are calls, not properties.
        assert entry.block.signature() == 'test.Anchor(spec={})'
        assert entry.block.type() == 'test.Anchor(spec={})/version=1.0'
        assert entry.block.sig() == "{'spec': {}}"

    def test_datajournal_entry_methods_none(self):
        series = pd.Series({'hash': '12345', 'anchor': 'test.Anchor'})
        entry = DatajournalEntry(series)

        assert entry.block.signature() is None
        assert entry.block.sig() is None
        assert entry.block.type() is None
        assert entry.block.tp() is None
        assert entry.block.signature() is None
        assert entry.block.type() is None


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

    def test_read_exec_journal_prefix_matching(self, tmp_path, monkeypatch):
        """Prefix matching on exec (e.g. exec='autopath.') should match starting strings."""
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        write_exec_journal("autopath.pipeline.Run(a=1)", url=dbx_url)
        write_exec_journal("autopath.model.Train(b=2)", url=dbx_url)
        write_exec_journal("dbx.Datablock(x=3)", url=dbx_url)

        # Single prefix match
        j_auto = dbx.journal(exec='autopath.')
        assert len(j_auto) == 2
        assert all(s.startswith('autopath.') for s in j_auto['exec'])

        # Multiple prefixes match
        j_multi = dbx.journal(exec=['autopath.pipeline', 'dbx.'])
        assert len(j_multi) == 2




class TestExecStatements:

    def test_statements_share_a_namespace_and_the_last_is_the_value(self, tmp_path, monkeypatch):
        """`a; b; c` runs in order, in one namespace, and `c` is what comes back."""
        monkeypatch.setenv('DBX_URL', str(tmp_path / 'dbx_root'))

        assert dbx.exec("x = 2; y = x * 3; y + 1") == 7
        assert dbx.exec("a = 1\nb = 2\na + b") == 3

    def test_last_statement_binding_returns_none(self, tmp_path, monkeypatch):
        """Nothing is evaluated last, so there is no value to return."""
        monkeypatch.setenv('DBX_URL', str(tmp_path / 'dbx_root'))

        assert dbx.exec("x = 5; y = x + 1") is None

    def test_later_statement_sees_earlier_binding_inside_a_comprehension(self, tmp_path, monkeypatch):
        """The namespace is one dict, not globals-plus-locals.

        Split, the comprehension's own scope could not reach `n` -- the rule
        that makes a class body unable to see its own names -- and the idiom
        sequencing exists for would break.
        """
        monkeypatch.setenv('DBX_URL', str(tmp_path / 'dbx_root'))

        assert dbx.exec("n = 3; ns = [1, 2]; [i * n for i in ns]") == [3, 6]

    def test_dotted_names_are_imported_for_every_statement(self, tmp_path, monkeypatch):
        """Not just the one before the first `(`: any statement may name a module."""
        monkeypatch.setenv('DBX_URL', str(tmp_path / 'dbx_root'))

        assert dbx.exec("e = xml.etree.ElementTree.Element('a'); e.tag") == 'a'

    def test_trailing_comment_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_URL', str(tmp_path / 'dbx_root'))

        assert dbx.exec("1 + 1  # a note about why") == 2
        assert dbx.exec("x = 2; x * 3; # trailing semicolon then comment") == 6

    def test_nothing_to_execute_raises(self, tmp_path, monkeypatch):
        """A string that is all comment ran nothing; saying so beats returning None."""
        monkeypatch.setenv('DBX_URL', str(tmp_path / 'dbx_root'))

        with pytest.raises(ValueError, match="No statement to execute"):
            dbx.exec("# just a comment")

    def test_comment_is_journaled_in_its_own_column(self, tmp_path, monkeypatch):
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        dbx.exec("1 + 1  # why this ran")

        j = dbx.journal()
        assert j.iloc[0]['comment'] == 'why this ran'

    def test_no_comment_journals_as_null(self, tmp_path, monkeypatch):
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        dbx.exec("1 + 1")

        j = dbx.journal()
        assert 'comment' in j.columns
        assert pd.isna(j.iloc[0]['comment'])

    def test_comment_is_filterable(self, tmp_path, monkeypatch):
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        write_exec_journal("expr1  # nightly refresh", url=dbx_url)
        write_exec_journal("expr2  # one-off", url=dbx_url)

        j = dbx.journal(comment='nightly')
        assert len(j) == 1
        assert j.iloc[0]['exec'] == 'expr1  # nightly refresh'

    def test_exec_comment_reads_the_trailing_comment(self):
        from dbx.dataparts import exec_comment

        assert exec_comment("a = 1  # note") == 'note'
        assert exec_comment("a = 1") is None
        assert exec_comment("x = '''unterminated") is None


@pytest.mark.pinned
class TestExecJournalRecordsWhatWasTyped:
    """The exec journal holds the string as it was typed.

    It is the only record of what was run, and it is read to find out what
    produced a build -- long after the person who typed it could be asked.
    A row that held a normalised, re-joined or comment-stripped rendering
    would still look like a command and no longer be the one that ran.
    """

    def test_original_string_is_recorded_verbatim(self, tmp_path, monkeypatch):
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        expr = "x = 2;  y = x * 3;   y + 1;  # comment to be ignored"
        dbx.exec(expr)

        assert dbx.journal().iloc[0]['exec'] == expr

    def test_hash_inside_a_string_literal_is_not_a_comment(self, tmp_path, monkeypatch):
        """Cutting at the first `#` would silently change what runs.

        A URL fragment, a colour, a `#` in a format: the expression means what
        Python says it means, and only a comment token is a comment.
        """
        dbx_url = str(tmp_path / 'dbx_root')
        monkeypatch.setenv('DBX_URL', dbx_url)

        assert dbx.exec("'https://host/p#frag'") == 'https://host/p#frag'

        entry = dbx.journal().iloc[0]
        assert entry['exec'] == "'https://host/p#frag'"
        assert pd.isna(entry['comment'])
