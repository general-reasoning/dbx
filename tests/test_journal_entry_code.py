"""``write_journal_entry()`` stamps every entry with its own ``entry_code``.

Nothing else on an entry identifies a *row*: ``hash`` and ``key`` are shared by
every entry of a block, ``uuid`` by every entry of one live instance, and
``datetime`` only to its resolution. ``entry_code`` is a fresh uuid per call and
is returned, so a caller can address exactly the row it wrote.

The one sharp edge is not about the code but about where entries live: a journal
file is per instance, so a second write from the same instance overwrites the
first, and the overwritten code stops resolving.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, DatajournalEntry


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Built(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


def block(tmp_path, x=1, **kwargs):
    return Built(url=str(tmp_path), spec={'x': x}, **kwargs)


class TestEveryCallGetsItsOwnCode:

    def test_write_returns_a_code(self, tmp_path):
        code = block(tmp_path).write_journal_entry(event='note')
        assert isinstance(code, str) and code

    def test_successive_calls_return_different_codes(self, tmp_path):
        b = block(tmp_path)
        codes = [b.write_journal_entry(event=f'note:{i}') for i in range(4)]
        assert len(set(codes)) == 4

    def test_separate_instances_get_different_codes(self, tmp_path):
        first = block(tmp_path).write_journal_entry(event='note')
        second = block(tmp_path).write_journal_entry(event='note')
        assert first != second

    def test_a_build_stamps_its_entry(self, tmp_path):
        b = block(tmp_path)
        b.build()
        entry = b.journal(loc=0)
        assert entry.block.id
        assert not hasattr(entry, 'entry_code')
        assert not hasattr(entry, 'subhash')
        assert not hasattr(entry, 'subsignature')


class TestTheCodeIsRecorded:

    def test_column_is_written(self, tmp_path):
        b = block(tmp_path)
        b.write_journal_entry(event='note')
        cols = block(tmp_path).journal().columns
        assert 'id' in cols
        assert 'code' in cols
        assert 'entry_code' not in cols
        assert 'subhash' not in cols
        assert 'subsignature' not in cols

    def test_the_recorded_code_is_the_returned_one(self, tmp_path):
        code = block(tmp_path).write_journal_entry(event='note')
        assert block(tmp_path).journal(loc=0).id == code

    def test_the_journal_can_be_filtered_by_it(self, tmp_path):
        block(tmp_path).write_journal_entry(event='wanted')
        code = block(tmp_path).write_journal_entry(event='also-wanted')
        entry = block(tmp_path).journal(id=code, loc=0)
        assert isinstance(entry, DatajournalEntry)
        assert entry.get('event') == 'also-wanted'

    def test_codes_are_unique_across_the_whole_journal(self, tmp_path):
        for x in (1, 2, 3):
            block(tmp_path, x=x).build()
        codes = list(block(tmp_path).journal()['id'])
        assert len(codes) == 3
        assert len(set(codes)) == 3


class TestDistinctFromTheSession:
    """``session`` identifies the run, ``id`` the entry."""

    def test_they_are_not_the_same_value(self, tmp_path):
        b = block(tmp_path)
        code = b.write_journal_entry(event='note')
        assert code != b.session

    def test_one_instance_two_entries_share_a_session_but_not_a_code(self, tmp_path):
        """Distinct journal_prefix values, so both entries survive -- an
        instance otherwise overwrites its own file."""
        b = block(tmp_path)
        first = b.write_journal_entry(event='one', journal_prefix='a-')
        second = b.write_journal_entry(event='two', journal_prefix='b-')
        journal = block(tmp_path).journal()
        assert len(journal) == 2
        assert set(journal['id']) == {first, second}
        assert journal['session'].nunique() == 1

    def test_session_accessor_reads_the_column(self, tmp_path):
        b = block(tmp_path)
        b.write_journal_entry(event='note')
        assert block(tmp_path).journal(loc=0).block.session == b.session

    def test_a_session_can_be_given(self, tmp_path):
        b = block(tmp_path, session='RUN-7')
        b.write_journal_entry(event='note')
        assert block(tmp_path).journal(loc=0).block.session == 'RUN-7'

    def test_a_session_does_not_change_identity(self, tmp_path):
        """Which run built a block cannot change what the block IS."""
        assert block(tmp_path, session='A').hash == block(tmp_path, session='B').hash


class TestOverwriteWithinOneInstance:
    """A journal file is per instance, so rewriting replaces the row."""

    def test_a_second_write_replaces_the_first(self, tmp_path):
        b = block(tmp_path)
        first = b.write_journal_entry(event='one')
        second = b.write_journal_entry(event='two')
        journal = block(tmp_path).journal()
        assert len(journal) == 1
        assert list(journal['id']) == [second]
        assert first not in set(journal['id'])

    def test_build_leaves_the_end_entry_not_the_start(self, tmp_path):
        """build() writes build:start then build:end from one instance, so the
        start is overwritten -- which is why a code resolves only until its
        instance writes again."""
        b = block(tmp_path)
        b.build()
        assert list(block(tmp_path).journal()['event']) == ['build:end']


class TestLegacyJournals:

    def test_absent_column_reads_as_none(self, tmp_path):
        """An entry from a journal written before the field existed."""
        b = block(tmp_path)
        b.write_journal_entry(event='note')
        entry = b.journal(loc=0)
        assert DatajournalEntry(entry.drop(['entry_code', 'id'], errors='ignore')).block.id is None


class TestUuid16:

    def test_short_form_is_honoured(self, tmp_path):
        """The code follows uuid16, so the two identifiers in one entry are
        the same shape."""
        b = block(tmp_path, uuid16=True)
        code = b.write_journal_entry(event='note')
        assert len(code) == 16
        assert len(b.session) == 16

    def test_long_form_by_default(self, tmp_path):
        assert len(block(tmp_path).write_journal_entry(event='note')) == 36
