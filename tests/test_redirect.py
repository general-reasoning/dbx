"""A failed read can be sent to another journal entry's data.

``UNSAFE_redirect()`` records, in the journal and nowhere else, where reads of
this block should go when its own data is gone: an ``entry_code`` naming one
entry, or a filter naming whichever entries match. ``read()`` consults that
only after ``__read__`` has already raised, and hands the redirected-to entry's
recorded path to ``__read__`` -- so a block with no redirection, or one whose
redirection leads nowhere, fails exactly as it did before.
"""
import pandas as pd
import pytest
from dataclasses import dataclass
from unittest.mock import patch

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
            f.write(f"data-{self.var.x}")

    def __read__(self, topic, path=None):
        with open(path if path is not None else self.path(topic)) as f:
            return f.read()


class Deaf(Built):
    """Its ``__read__`` predates the *path* argument."""

    def __read__(self, topic):
        with open(self.path(topic)) as f:
            return f.read()


def block(tmp_path, x=1, **kwargs):
    return Built(url=str(tmp_path), spec={'x': x}, **kwargs)


@pytest.fixture
def source(tmp_path):
    """A built block, and the entry_code of the entry recording that build."""
    b = block(tmp_path, x=1)
    b.build()
    return b, b.journal(loc=0).entry_code


@pytest.fixture
def broken(tmp_path):
    """A block of its own hash whose data was never written."""
    return block(tmp_path, x=2)


class TestRedirectionIsRecordedInTheEntry:

    def test_an_entry_code_round_trips(self, tmp_path):
        block(tmp_path).write_journal_entry(event='note', redirection='abc-123')
        assert block(tmp_path).journal(loc=0).redirection == 'abc-123'

    def test_a_filter_round_trips_as_a_dict(self, tmp_path):
        block(tmp_path).write_journal_entry(event='note', redirection={'hash': 'deadbeef', 'event': 'build:end'})
        assert block(tmp_path).journal(loc=0).redirection == {'hash': 'deadbeef', 'event': 'build:end'}

    def test_it_lives_in_the_journal_not_in_a_file(self, tmp_path):
        """Not a path to a .txt, the way message/quote/norm are: the value itself."""
        block(tmp_path).write_journal_entry(event='note', redirection='abc-123')
        assert block(tmp_path).journal()['redirection'].iloc[0] == 'abc-123'

    def test_an_ordinary_entry_records_none(self, tmp_path):
        block(tmp_path).write_journal_entry(event='note')
        assert block(tmp_path).journal(loc=0).redirection is None

    def test_absent_column_reads_as_none(self, tmp_path):
        """An entry from a journal written before the column existed."""
        b = block(tmp_path)
        b.write_journal_entry(event='note', redirection='abc-123')
        entry = b.journal(loc=0)
        assert DatajournalEntry(entry.drop('redirection')).redirection is None

    def test_a_non_str_non_dict_is_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            block(tmp_path).write_journal_entry(event='note', redirection=['a', 'b'])


class TestUNSAFERedirectWritesIt:

    def test_entry_code_is_recorded(self, broken):
        broken.UNSAFE_redirect(entry_code='abc-123', OVERRIDE=True)
        assert broken.journal(loc=0).redirection == 'abc-123'

    def test_filter_is_recorded(self, broken):
        broken.UNSAFE_redirect(filter={'event': 'build:end'}, OVERRIDE=True)
        assert broken.journal(loc=0).redirection == {'event': 'build:end'}

    def test_the_event_names_itself(self, broken):
        broken.UNSAFE_redirect(entry_code='abc-123', OVERRIDE=True)
        assert broken.journal(loc=0).get('event') == 'UNSAFE_redirect'

    def test_it_returns_its_own_entry_code(self, broken):
        code = broken.UNSAFE_redirect(entry_code='abc-123', OVERRIDE=True)
        assert broken.journal(loc=0).entry_code == code

    def test_it_does_not_overwrite_an_earlier_entry_of_the_same_instance(self, broken):
        """Its own journal file, so the build entry it corrects survives it."""
        broken.write_journal_entry(event='build:end')
        broken.UNSAFE_redirect(entry_code='abc-123', OVERRIDE=True)
        assert set(broken.journal()['event']) == {'build:end', 'UNSAFE_redirect'}

    def test_exactly_one_of_the_two_is_required(self, broken):
        with pytest.raises(ValueError):
            broken.UNSAFE_redirect(OVERRIDE=True)
        with pytest.raises(ValueError):
            broken.UNSAFE_redirect(entry_code='abc-123', filter={'event': 'build:end'}, OVERRIDE=True)

    def test_an_empty_filter_is_rejected(self, broken):
        """It would match every entry, and so redirect to whatever was last written."""
        with pytest.raises(ValueError):
            broken.UNSAFE_redirect(filter={}, OVERRIDE=True)

    def test_it_asks_before_writing(self, broken):
        broken.write_journal_entry(event='note')     # so there is a journal to inspect
        with patch('builtins.input', return_value='n') as ask:
            assert broken.UNSAFE_redirect(entry_code='abc-123') is None
        ask.assert_called()
        assert list(broken.journal()['event']) == ['note']

    def test_override_does_not_ask(self, broken):
        with patch('builtins.input') as ask:
            broken.UNSAFE_redirect(entry_code='abc-123', OVERRIDE=True)
        ask.assert_not_called()


class TestAFailedReadFollowsIt:

    def test_by_entry_code(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        assert broken.read('output') == 'data-1'

    def test_by_filter(self, source, broken):
        src, _ = source
        broken.UNSAFE_redirect(filter={'hash': src.hash, 'event': 'build:end'}, OVERRIDE=True)
        assert broken.read('output') == 'data-1'

    def test_a_working_read_is_left_alone(self, source, tmp_path):
        """The redirection is never consulted while __read__ still succeeds."""
        src, code = source
        other = block(tmp_path, x=3)
        other.build()
        other.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        assert other.read('output') == 'data-3'

    def test_it_is_announced(self, source, broken, capsys):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        capsys.readouterr()
        broken.read('output')
        out = capsys.readouterr().out
        assert 'REDIRECTING' in out and 'REDIRECTED' in out
        assert 'INFO' in out

    def test_the_topic_is_still_validated_first(self, source, broken):
        """A mistyped topic is a mistake, not something to redirect around."""
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        with pytest.raises(KeyError):
            broken.read('nosuchtopic')


class TestTheLatestOneWins:

    def test_of_several_redirections(self, tmp_path):
        """Redirections are corrections: the newest is the one still meant."""
        first = block(tmp_path, x=1)
        first.build()
        second = block(tmp_path, x=3)
        second.build()

        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(entry_code=first.journal(hash=first.hash, loc=0).entry_code, OVERRIDE=True)
        block(tmp_path, x=2).UNSAFE_redirect(entry_code=second.journal(hash=second.hash, loc=0).entry_code,
                                             OVERRIDE=True)
        assert block(tmp_path, x=2).read('output') == 'data-3'

    def test_of_several_entries_matching_one_filter(self, tmp_path):
        """A filter names a set on purpose; the newest member answers."""
        block(tmp_path, x=1).build()
        block(tmp_path, x=3).build()
        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(filter={'event': 'build:end'}, OVERRIDE=True)
        assert broken.read('output') == 'data-3'

    def test_a_filter_follows_a_rebuild(self, tmp_path):
        """Unlike an entry_code, which is pinned to the build it was returned for."""
        src = block(tmp_path, x=1)
        src.build()
        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(filter={'hash': src.hash, 'event': 'build:end'}, OVERRIDE=True)
        assert broken.read('output') == 'data-1'

        rebuilt = Built(url=str(tmp_path), spec={'x': 1})
        with open(rebuilt.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data-1-rebuilt')
        rebuilt.write_journal_entry(event='build:end')
        assert block(tmp_path, x=2).read('output') == 'data-1-rebuilt'


class TestWhenTheRedirectionLeadsNowhere:
    """The read fails as it would have without one -- with its own exception."""

    def test_no_redirection_recorded(self, broken):
        with pytest.raises(FileNotFoundError):
            broken.read('output')

    def test_no_journal_at_all(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            block(tmp_path).read('output')

    def test_an_entry_code_that_matches_nothing(self, source, broken):
        broken.UNSAFE_redirect(entry_code='no-such-code', OVERRIDE=True)
        with pytest.raises(FileNotFoundError):
            broken.read('output')

    def test_a_filter_on_a_column_that_does_not_exist(self, source, broken):
        broken.UNSAFE_redirect(filter={'nosuchcolumn': 'x'}, OVERRIDE=True)
        with pytest.raises(FileNotFoundError):
            broken.read('output')

    def test_an_entry_recording_no_such_topic(self, tmp_path, broken):
        """The redirected-to entry is real, but has nothing for this topic."""
        class Other(Built):
            TOPICS = {'elsewhere': 'elsewhere.txt'}

        other = Other(url=str(tmp_path), spec={'x': 1})
        code = other.write_journal_entry(event='note')
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        with pytest.raises(FileNotFoundError):
            broken.read('output')

    def test_the_redirected_read_fails_on_its_own_terms(self, tmp_path, broken):
        """Redirected to an entry whose recorded data is itself gone."""
        src = block(tmp_path, x=1)
        code = src.write_journal_entry(event='build:end')   # never built
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        with pytest.raises(FileNotFoundError):
            broken.read('output')


class TestTheRedirectionIsResolvedOnceAndCached:
    """Resolving one means reading the whole journal; a block pays for it once."""

    def test_it_is_the_entry_the_redirection_names(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        assert isinstance(broken.redirection, DatajournalEntry)
        assert broken.redirection.entry_code == code

    def test_none_when_nothing_is_recorded(self, source, broken):
        assert broken.redirection is None

    def test_none_when_redirect_is_off(self, source, tmp_path):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        assert block(tmp_path, x=2, redirect=False).redirection is None

    def test_repeated_reads_consult_the_journal_once(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        with patch.object(broken, 'journal', wraps=broken.journal) as reads:
            broken.read('output')
            resolving = reads.call_count
            broken.read('output')
            broken.read('output')
        assert resolving > 0
        assert reads.call_count == resolving

    def test_the_journal_is_not_kept_alive_by_the_cache(self, source, broken):
        """Only the entry is cached, detached from the frame it came out of."""
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        broken.read('output')
        assert not any(isinstance(v, pd.DataFrame) for v in vars(broken).values())

    def test_a_later_redirection_is_not_seen_by_an_instance_that_looked(self, tmp_path):
        first = block(tmp_path, x=1)
        first.build()
        second = block(tmp_path, x=3)
        second.build()

        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(entry_code=first.journal(hash=first.hash, loc=0).entry_code, OVERRIDE=True)
        assert broken.read('output') == 'data-1'

        block(tmp_path, x=2).UNSAFE_redirect(entry_code=second.journal(hash=second.hash, loc=0).entry_code,
                                             OVERRIDE=True)
        assert broken.read('output') == 'data-1'                 # cached
        assert block(tmp_path, x=2).read('output') == 'data-3'    # a fresh instance looks again

    def test_the_cache_can_be_dropped(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        assert broken.redirection is not None
        del broken.redirection
        assert broken.redirection.entry_code == code


class TestARedirectedBlockDeclinesToBuild:
    """Its reads answer from elsewhere, so nothing would read what it built."""

    def test_it_does_not_build(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        broken.build()
        assert not broken.valid()

    def test_it_returns_self(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        assert broken.build() is broken

    def test_it_is_announced(self, source, broken, capsys):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        capsys.readouterr()
        broken.build()
        out = capsys.readouterr().out
        assert 'BUILD DECLINED' in out
        assert 'INFO' in out

    def test_it_journals_no_build(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        broken.build()
        assert set(broken.journal(hash=broken.hash)['event']) == {'UNSAFE_redirect'}

    def test_it_still_reads(self, source, broken):
        """Declining to build is the point: the data it answers with is elsewhere."""
        _, code = source
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        broken.build()
        assert broken.read('output') == 'data-1'

    def test_redirect_off_builds_anyway(self, source, tmp_path):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        b = block(tmp_path, x=2, redirect=False)
        b.build()
        assert b.valid()
        assert b.read('output') == 'data-2'

    def test_an_unredirected_block_builds(self, tmp_path):
        b = block(tmp_path, x=5)
        b.build()
        assert b.valid()
        assert b.read('output') == 'data-5'

    def test_build_tree_does_not_rebuild_a_redirected_block(self, source, tmp_path):
        """What a sweep would otherwise quietly undo."""
        _, code = source
        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        broken.build_tree()
        assert not block(tmp_path, x=2).valid()


class TestTheRedirectParameter:

    def test_it_defaults_to_on(self, tmp_path):
        assert block(tmp_path).redirect is True

    def test_off_re_raises_instead_of_redirecting(self, source, tmp_path):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        b = block(tmp_path, x=2, redirect=False)
        with pytest.raises(FileNotFoundError):
            b.read('output')

    def test_off_says_nothing(self, source, tmp_path, capsys):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        b = block(tmp_path, x=2, redirect=False)
        capsys.readouterr()
        with pytest.raises(FileNotFoundError):
            b.read('output')
        assert 'REDIRECT' not in capsys.readouterr().out

    def test_it_survives_a_round_trip_through_dfn(self, tmp_path):
        b = block(tmp_path, redirect=False)
        assert b.dfn['redirect'] is False
        assert Built(**b.dfn).redirect is False

    def test_it_does_not_move_the_hash(self, tmp_path):
        """Identity is norm-borne: how a block reads is not what it is."""
        assert block(tmp_path, redirect=False).hash == block(tmp_path, redirect=True).hash

    def test_a_block_from_before_the_parameter_redirects(self, tmp_path):
        """State pickled without the key -- the default applies to it too."""
        b = block(tmp_path)
        state = b.__getstate__()
        del state['redirect']
        assert Built(**state).redirect is True


class TestAnOverrideThatDoesNotTakePath:
    """It fails at the redirection, rather than silently reading its own path."""

    def test_the_unredirected_read_is_unaffected(self, tmp_path):
        deaf = Deaf(url=str(tmp_path), spec={'x': 1})
        deaf.build()
        assert deaf.read('output') == 'data-1'

    def test_the_redirected_read_raises_typeerror(self, tmp_path):
        src = Deaf(url=str(tmp_path), spec={'x': 1})
        src.build()
        code = src.journal(loc=0).entry_code
        broken = Deaf(url=str(tmp_path), spec={'x': 2})
        broken.UNSAFE_redirect(entry_code=code, OVERRIDE=True)
        with pytest.raises(TypeError):
            broken.read('output')
