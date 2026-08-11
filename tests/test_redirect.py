"""A redirected block reads somewhere else, through its own ``path()``.

``UNSAFE_redirect()`` records, in the journal and nowhere else, where this
block's topics live instead: a ``filter`` selecting a journal entry whose paths
become this block's (optionally re-keyed by a ``topic_map``), or ``paths``
naming them outright. :attr:`Datablock.path` then answers with those, so
everything that resolves through it — ``read``, ``valid``, ``ls``, ``list``,
``size`` — describes the data actually being read, with no knowledge of
redirection anywhere in the way.
"""
import os

import pandas as pd
import pytest
from dataclasses import dataclass
from unittest.mock import patch

from dbx.datablocks import DIRTOPIC, Datablock, DatajournalEntry


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

    def __read__(self, topic):
        with open(self.path(topic)) as f:
            return f.read()


class TwoTopics(Built):
    TOPICS = {'output': 'output.txt', 'notes': 'notes.txt'}

    def __build__(self):
        super().__build__()
        with open(self.path('notes', ensure_dirpath=True), 'w') as f:
            f.write(f"notes-{self.var.x}")


class Renamed(Built):
    """The same data under a different topic name."""
    TOPICS = {'result': 'output.txt'}

    def __build__(self):
        with open(self.path('result', ensure_dirpath=True), 'w') as f:
            f.write(f"data-{self.var.x}")


def block(tmp_path, x=1, cls=Built, **kwargs):
    return cls(url=str(tmp_path), spec={'x': x}, **kwargs)


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


class TestTheRecordedRedirection:

    def test_a_filter_is_recorded_as_a_dict(self, broken):
        broken.UNSAFE_redirect(filter={'event': 'build:end'}, OVERRIDE=True)
        assert broken.journal(loc=0).redirection == {'filter': {'event': 'build:end'}}

    def test_a_topic_map_travels_with_it(self, broken):
        broken.UNSAFE_redirect(filter={'event': 'build:end'},
                               topic_map={'output': 'result'}, OVERRIDE=True)
        assert broken.journal(loc=0).redirection == {
            'filter': {'event': 'build:end'}, 'topic_map': {'output': 'result'},
        }

    def test_paths_are_recorded_as_a_dict(self, broken):
        broken.UNSAFE_redirect(paths={'output': '/data/out.txt'}, OVERRIDE=True)
        assert broken.journal(loc=0).redirection == {'paths': {'output': '/data/out.txt'}}

    def test_it_lives_in_the_journal_not_in_a_file(self, broken):
        broken.UNSAFE_redirect(paths={'output': '/data/out.txt'}, OVERRIDE=True)
        assert broken.journal()['redirection'].iloc[0] == str({'paths': {'output': '/data/out.txt'}})

    def test_an_ordinary_entry_records_none(self, tmp_path):
        block(tmp_path).write_journal_entry(event='note')
        assert block(tmp_path).journal(loc=0).redirection is None

    def test_the_event_names_itself(self, broken):
        broken.UNSAFE_redirect(paths={'output': '/x'}, OVERRIDE=True)
        assert broken.journal(loc=0).get('event') == 'UNSAFE_redirect'

    def test_it_returns_its_own_entry_code(self, broken):
        code = broken.UNSAFE_redirect(paths={'output': '/x'}, OVERRIDE=True)
        assert broken.journal(loc=0).entry_code == code

    def test_it_does_not_overwrite_an_earlier_entry_of_the_same_instance(self, broken):
        broken.write_journal_entry(event='build:end')
        broken.UNSAFE_redirect(paths={'output': '/x'}, OVERRIDE=True)
        assert set(broken.journal()['event']) == {'build:end', 'UNSAFE_redirect'}


class TestTheArguments:

    def test_exactly_one_of_filter_and_paths(self, broken):
        with pytest.raises(ValueError):
            broken.UNSAFE_redirect(OVERRIDE=True)
        with pytest.raises(ValueError):
            broken.UNSAFE_redirect(filter={'event': 'build:end'},
                                   paths={'output': '/x'}, OVERRIDE=True)

    def test_an_empty_filter_is_rejected(self, broken):
        """It would match every entry, and so redirect to whatever was last written."""
        with pytest.raises(ValueError):
            broken.UNSAFE_redirect(filter={}, OVERRIDE=True)

    def test_empty_paths_are_rejected(self, broken):
        with pytest.raises(ValueError):
            broken.UNSAFE_redirect(paths={}, OVERRIDE=True)

    def test_a_topic_map_beside_paths_is_ignored(self, broken, capsys):
        """There is nothing to re-key when the paths are given outright."""
        broken.UNSAFE_redirect(paths={'output': '/x'}, topic_map={'a': 'b'}, OVERRIDE=True)
        assert broken.journal(loc=0).redirection == {'paths': {'output': '/x'}}
        assert 'ignoring it' in capsys.readouterr().out

    def test_it_asks_before_writing(self, broken):
        broken.write_journal_entry(event='note')     # so there is a journal to inspect
        with patch('builtins.input', return_value='n') as ask:
            assert broken.UNSAFE_redirect(paths={'output': '/x'}) is None
        ask.assert_called()
        assert list(broken.journal()['event']) == ['note']

    def test_override_does_not_ask(self, broken):
        with patch('builtins.input') as ask:
            broken.UNSAFE_redirect(paths={'output': '/x'}, OVERRIDE=True)
        ask.assert_not_called()


class TestPathIsWhatRedirects:
    """Everything else follows from this one."""

    def test_path_answers_with_the_redirected_location(self, source, broken):
        src, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.path('output') == src.path('output')

    def test_paths_given_outright_are_answered_verbatim(self, broken, tmp_path):
        target = tmp_path / 'elsewhere' / 'thing.txt'
        broken.UNSAFE_redirect(paths={'output': str(target)}, OVERRIDE=True)
        assert broken.path('output') == str(target)

    def test_an_unredirected_block_answers_with_its_own(self, tmp_path):
        b = block(tmp_path)
        assert b.path('output').startswith(b.anchorkeypath)

    def test_the_local_path_is_never_redirected(self, source, broken):
        """The local cache is this block's own, wherever it reads from."""
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.path('output', local=True).startswith(broken.localanchorkeypath)

    def test_a_redirected_path_is_not_ensured(self, broken, tmp_path):
        """Creating directories inside another block's data is not this block's business."""
        target = tmp_path / 'elsewhere' / 'thing.txt'
        broken.UNSAFE_redirect(paths={'output': str(target)}, OVERRIDE=True)
        broken.path('output', ensure_dirpath=True)
        assert not target.parent.exists()

    def test_dirpath_follows_a_file_topic_to_its_parent(self, source, broken):
        src, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.dirpath('output') == os.path.dirname(src.path('output'))

    def test_dirpath_of_a_directory_topic_is_the_path_itself(self, tmp_path):
        class Dir(Built):
            TOPICS = {'output': DIRTOPIC}

        target = tmp_path / 'elsewhere'
        b = block(tmp_path, x=2, cls=Dir)
        b.UNSAFE_redirect(paths={'output': str(target)}, OVERRIDE=True)
        assert b.dirpath('output') == str(target)

    def test_an_unmapped_topic_keeps_its_own_path(self, tmp_path):
        """A partial redirection redirects only what it names."""
        b = block(tmp_path, x=2, cls=TwoTopics)
        b.UNSAFE_redirect(paths={'output': '/elsewhere/out.txt'}, OVERRIDE=True)
        assert b.path('output') == '/elsewhere/out.txt'
        assert b.path('notes').startswith(b.anchorkeypath)


class TestReadingFollows:

    def test_by_filter(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.read('output') == 'data-1'

    def test_by_a_filter_matching_many_entries(self, tmp_path):
        """A filter names a set on purpose; the newest member answers."""
        block(tmp_path, x=1).build()
        block(tmp_path, x=3).build()
        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(filter={'event': 'build:end'}, OVERRIDE=True)
        assert broken.read('output') == 'data-3'

    def test_by_paths(self, source, broken):
        src, _ = source
        broken.UNSAFE_redirect(paths={'output': src.path('output')}, OVERRIDE=True)
        assert broken.read('output') == 'data-1'

    def test_a_filter_follows_a_rebuild_where_an_entry_code_does_not(self, tmp_path):
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

    def test_the_reader_needs_no_knowledge_of_it(self, source, broken):
        """__read__ resolves self.path(topic), which is where this happens."""
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert type(broken).__read__ is Built.__read__
        assert broken.read('output') == 'data-1'

    def test_the_topic_is_still_validated_first(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        with pytest.raises(KeyError):
            broken.read('nosuchtopic')


class TestTopicMap:
    """``{'mine': 'theirs'}``: my topic is that entry's."""

    @pytest.fixture
    def renamed_source(self, tmp_path):
        """Under Built's anchor: a filter selects from the journal of the block
        doing the redirecting, and a journal is per anchor."""
        src = Renamed(url=str(tmp_path), spec={'x': 1}, anchor=Built(url=str(tmp_path)).anchor)
        src.build()
        return src, src.journal(loc=0).entry_code

    def test_a_topic_reads_from_the_mapped_one(self, renamed_source, broken):
        src, code = renamed_source
        broken.UNSAFE_redirect(filter={'entry_code': code},
                               topic_map={'output': 'result'}, OVERRIDE=True)
        assert broken.path('output') == src.path('result')
        assert broken.read('output') == 'data-1'

    def test_without_it_the_name_does_not_line_up(self, renamed_source, broken):
        _, code = renamed_source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.path('output').startswith(broken.anchorkeypath)

    def test_an_unmentioned_topic_keeps_its_name(self, tmp_path):
        """Topics line up by name to begin with; the map only adds to that."""
        src = TwoTopics(url=str(tmp_path), spec={'x': 1})
        src.build()
        code = src.journal(loc=0).entry_code
        b = block(tmp_path, x=2, cls=TwoTopics)
        b.UNSAFE_redirect(filter={'entry_code': code},
                          topic_map={'output': 'notes'}, OVERRIDE=True)
        assert b.read('output') == 'notes-1'    # mapped
        assert b.read('notes') == 'notes-1'     # its own name, still the entry's

    def test_a_mapping_the_other_side_lacks_leaves_its_topic_alone(self, source, broken, capsys):
        """Asked for `nosuchtopic` and given `output` instead is the one answer
        that is certainly wrong, so the topic is left unredirected."""
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code},
                               topic_map={'output': 'nosuchtopic'}, OVERRIDE=True)
        capsys.readouterr()
        assert broken.path('output').startswith(broken.anchorkeypath)
        assert 'left unredirected' in capsys.readouterr().out


class TestValidity:

    def test_a_redirected_block_is_valid_when_the_data_it_reads_is(self, source, broken):
        _, code = source
        assert not broken.valid()
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.valid()

    def test_it_is_invalid_when_the_redirected_to_data_is_gone(self, broken, tmp_path):
        broken.UNSAFE_redirect(paths={'output': str(tmp_path / 'nothing.txt')}, OVERRIDE=True)
        assert not broken.valid()

    def test_valid_goes_through_the_hook(self, tmp_path):
        class Fussy(Built):
            def __valid__(self, path=None):
                return False

        assert not Fussy(url=str(tmp_path), spec={'x': 1}).build().valid()

    def test_the_hook_is_given_no_path_without_a_redirection(self, tmp_path):
        seen = []

        class Watching(Built):
            def __valid__(self, path=None):
                seen.append(path)
                return super().__valid__(path=path)

        Watching(url=str(tmp_path), spec={'x': 1}).valid()
        assert seen == [None]

    def test_the_hook_is_given_the_redirected_to_block_dir(self, source, tmp_path):
        src, code = source
        seen = []

        class Watching(Built):
            def __valid__(self, path=None):
                seen.append(path)
                return super().__valid__(path=path)

        b = Watching(url=str(tmp_path), spec={'x': 2}, anchor=src.anchor)
        b.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        b.valid()
        assert seen == [src.anchorkeypath]

    def test_paths_name_no_block_dir_so_the_hook_gets_none(self, broken, tmp_path):
        seen = []

        class Watching(Built):
            def __valid__(self, path=None):
                seen.append(path)
                return super().__valid__(path=path)

        b = Watching(url=str(tmp_path), spec={'x': 2})
        b.UNSAFE_redirect(paths={'output': str(tmp_path / 'x.txt')}, OVERRIDE=True)
        b.valid()
        assert seen == [None]


class TestListingsFollow:

    def test_ls_lists_the_redirected_to_directory(self, source, broken):
        src, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.ls('output') == src.ls('output')

    def test_size_measures_the_data_it_reads(self, source, broken):
        src, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.size('output') == src.size('output') > 0

    def test_list_describes_the_data_it_reads(self, source, broken):
        src, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert [e['name'] for e in broken.list('output')] == [e['name'] for e in src.list('output')]


class TestBuildDeclines:
    """Its reads answer from elsewhere, so nothing would read what it built."""

    def test_it_does_not_build(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        broken.build()
        assert not os.path.exists(os.path.join(broken.anchorkeypath, 'output', 'output.txt'))

    def test_it_is_announced(self, source, broken, capsys):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        capsys.readouterr()
        broken.build()
        out = capsys.readouterr().out
        assert 'BUILD DECLINED' in out and 'INFO' in out

    def test_it_journals_no_build(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        broken.build()
        assert set(broken.journal(hash=broken.hash)['event']) == {'UNSAFE_redirect'}

    def test_build_tree_does_not_rebuild_it(self, source, tmp_path):
        _, code = source
        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        broken.build_tree()
        assert block(tmp_path, x=2).read('output') == 'data-1'

    def test_redirect_off_builds_and_reads_its_own(self, source, tmp_path):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        b = block(tmp_path, x=2, redirect=False)
        b.build()
        assert b.read('output') == 'data-2'

    def test_an_unredirected_block_builds(self, tmp_path):
        b = block(tmp_path, x=5)
        b.build()
        assert b.valid() and b.read('output') == 'data-5'


class TestResolution:

    def test_the_latest_redirection_wins(self, tmp_path):
        """A redirection is a correction: the newest is the one still meant."""
        block(tmp_path, x=1).build()
        block(tmp_path, x=3).build()
        first = block(tmp_path, x=1).journal(hash=block(tmp_path, x=1).hash, loc=0).entry_code
        second = block(tmp_path, x=3).journal(hash=block(tmp_path, x=3).hash, loc=0).entry_code
        block(tmp_path, x=2).UNSAFE_redirect(filter={'entry_code': first}, OVERRIDE=True)
        block(tmp_path, x=2).UNSAFE_redirect(filter={'entry_code': second}, OVERRIDE=True)
        assert block(tmp_path, x=2).read('output') == 'data-3'

    def test_it_is_resolved_once(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        with patch.object(broken, 'journal', wraps=broken.journal) as reads:
            broken.path('output')
            resolving = reads.call_count
            broken.path('output')
            broken.read('output')
        assert resolving > 0 and reads.call_count == resolving

    def test_an_unredirected_block_does_not_read_the_anchors_journal(self, source, tmp_path):
        """path() asks on first use, and a table of a thousand tabs asks a
        thousand times: the question has to be answerable from this block's own
        journal directory, not by globbing every entry under the anchor."""
        b = block(tmp_path, x=7)
        with patch.object(b, 'journal', wraps=b.journal) as whole_journal:
            b.path('output')
            assert b.redirection is None
        whole_journal.assert_not_called()

    def test_a_redirected_block_reads_it_once_to_resolve_the_filter(self, source, tmp_path):
        _, code = source
        b = block(tmp_path, x=2)
        b.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        fresh = block(tmp_path, x=2)
        with patch.object(fresh, 'journal', wraps=fresh.journal) as whole_journal:
            fresh.path('output')
            fresh.path('output')
        assert whole_journal.call_count == 1

    def test_the_journal_is_not_kept_alive_by_the_cache(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        broken.read('output')
        assert not any(isinstance(v, pd.DataFrame) for v in vars(broken).values())

    def test_it_is_announced_when_it_resolves(self, source, broken, capsys):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        capsys.readouterr()
        broken.read('output')
        out = capsys.readouterr().out
        assert 'REDIRECTION' in out and 'INFO' in out

    def test_a_later_redirection_is_not_seen_by_an_instance_that_looked(self, tmp_path):
        block(tmp_path, x=1).build()
        block(tmp_path, x=3).build()
        first = block(tmp_path, x=1).journal(hash=block(tmp_path, x=1).hash, loc=0).entry_code
        second = block(tmp_path, x=3).journal(hash=block(tmp_path, x=3).hash, loc=0).entry_code

        broken = block(tmp_path, x=2)
        broken.UNSAFE_redirect(filter={'entry_code': first}, OVERRIDE=True)
        assert broken.read('output') == 'data-1'
        block(tmp_path, x=2).UNSAFE_redirect(filter={'entry_code': second}, OVERRIDE=True)
        assert broken.read('output') == 'data-1'                 # cached
        assert block(tmp_path, x=2).read('output') == 'data-3'    # a fresh instance looks again

    def test_the_cache_can_be_dropped(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert broken.redirection is not None
        del broken.redirection
        assert broken.redirection.entry.entry_code == code

    def test_the_resolution_names_its_parts(self, source, broken):
        _, code = source
        broken.UNSAFE_redirect(filter={'entry_code': code},
                               topic_map={'output': 'output'}, OVERRIDE=True)
        r = broken.redirection
        assert r.filter == {'entry_code': code}
        assert r.topic_map == {'output': 'output'}
        assert r.entry.entry_code == code
        assert 'output' in r.paths

    def test_paths_resolve_without_an_entry(self, broken):
        broken.UNSAFE_redirect(paths={'output': '/x/y.txt'}, OVERRIDE=True)
        r = broken.redirection
        assert r.entry is None and r.paths == {'output': '/x/y.txt'}

    def test_none_when_redirect_is_off(self, source, tmp_path):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        assert block(tmp_path, x=2, redirect=False).redirection is None

    def test_none_when_nothing_is_recorded(self, source, broken):
        assert broken.redirection is None

    def test_none_when_the_filter_matches_nothing(self, source, broken):
        broken.UNSAFE_redirect(filter={'entry_code': 'no-such-code'}, OVERRIDE=True)
        assert broken.redirection is None
        assert broken.path('output').startswith(broken.anchorkeypath)

    def test_none_when_the_filter_names_no_such_column(self, source, broken):
        broken.UNSAFE_redirect(filter={'nosuchcolumn': 'x'}, OVERRIDE=True)
        assert broken.redirection is None

    def test_no_journal_at_all(self, tmp_path):
        assert block(tmp_path).redirection is None


class TestLegacyRedirections:
    """A bare entry_code, from before a redirection was a dict."""

    def test_it_reads_as_a_filter_on_the_code(self, source, broken):
        _, code = source
        broken.write_journal_entry(event='UNSAFE_redirect', redirection=code,
                                   journal_prefix='redirect-')
        fresh = Built(url=broken._url_, spec={'x': 2})
        assert fresh.redirection.entry.entry_code == code
        assert fresh.read('output') == 'data-1'


class TestTheRedirectParameter:

    def test_it_defaults_to_on(self, tmp_path):
        assert block(tmp_path).redirect is True

    def test_off_reads_its_own_path(self, source, tmp_path):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        b = block(tmp_path, x=2, redirect=False)
        assert b.path('output').startswith(b.anchorkeypath)

    def test_off_says_nothing(self, source, tmp_path, capsys):
        _, code = source
        block(tmp_path, x=2).UNSAFE_redirect(filter={'entry_code': code}, OVERRIDE=True)
        b = block(tmp_path, x=2, redirect=False)
        capsys.readouterr()
        b.path('output')
        assert 'REDIRECT' not in capsys.readouterr().out

    def test_it_survives_a_round_trip_through_dfn(self, tmp_path):
        b = block(tmp_path, redirect=False)
        assert b.dfn['redirect'] is False
        assert Built(**b.dfn).redirect is False

    def test_it_does_not_move_the_hash(self, tmp_path):
        """Identity is norm-borne: where a block reads is not what it is."""
        assert block(tmp_path, redirect=False).hash == block(tmp_path, redirect=True).hash

    def test_a_block_from_before_the_parameter_redirects(self, tmp_path):
        b = block(tmp_path)
        state = b.__getstate__()
        del state['redirect']
        assert Built(**state).redirect is True


class TestDeprecatedValidAliases:
    """``valid_paths`` and friends; the old spellings still answer."""

    def test_valid_topics(self, tmp_path):
        b = block(tmp_path)
        b.build()
        assert b.valid_topics(reduce=True) is b.validtopics(reduce=True) is True

    def test_valid_paths(self, tmp_path):
        b = block(tmp_path)
        b.build()
        assert b.valid_paths(reduce=True) is b.validpaths(reduce=True) is True

    def test_valid_path(self, tmp_path):
        b = block(tmp_path)
        b.build()
        assert b.valid_path(b.path('output')) is b.validpath(b.path('output')) is True

    def test_valid_topic(self, tmp_path):
        b = block(tmp_path)
        b.build()
        assert b.valid_topic('output') is b.validtopic('output') is True
