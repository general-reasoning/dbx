"""A filtered Datajournal must be renumbered 0..N-1, not keep its old labels.

``Datablock.Journal`` sorts the concatenated journal newest-first and resets the
index, so a label is also a position. But ``Datajournal.__init__`` applied its
``filter_kwargs`` *after* that reset without renumbering, so a filtered journal
kept the labels its rows had in the full journal -- while ``loc=`` and
:meth:`Datajournal.get` still index by label.

The result: ``lastbuilt()`` (``journal(event='build:end').get(0)``) raised
``KeyError: 0`` for any block whose newest journal entry was some *other* event.
That is the normal state for a block whose artifact was copied in rather than
built -- its newest entry is an ``UNSAFE_copy_from:END``.

The failure was always loud (label 0 is label 0, so no wrong row was ever
returned silently) but it made ``lastbuilt()`` unusable exactly when it matters.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, Datajournal, DatajournalEntry


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Built(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        x: int = 1

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


@pytest.fixture
def newest_is_another_event(tmp_path):
    """Build, then write a NEWER entry of a different event.

    One journal file per instance, so the second entry has to come from a second
    instance -- writing again from the first would overwrite its build:end.
    """
    block = Built(url=str(tmp_path), spec={'x': 1})
    block.build()
    Built(url=str(tmp_path), spec={'x': 1}).write_journal_entry(
        event='UNSAFE_copy_from:END')
    return block


class TestFilteredIndexIsRenumbered:

    def test_unfiltered_journal_is_still_zero_based(self, newest_is_another_event):
        j = newest_is_another_event.journal()
        assert list(j.index) == list(range(len(j)))

    def test_filtered_journal_is_renumbered(self, newest_is_another_event):
        j = newest_is_another_event.journal(event='build:end')
        assert len(j) == 1
        assert list(j.index) == [0], "filtered journal kept its old labels"

    def test_filtered_journal_is_still_newest_first(self, tmp_path):
        # Distinct specs: build() is idempotent, so rebuilding the same identity
        # returns early and writes no second build:end.
        for x in (1, 2, 3):
            Built(url=str(tmp_path), spec={'x': x}).build()
        j = Built(url=str(tmp_path), spec={'x': 1}).journal(event='build:end')
        assert len(j) == 3
        times = list(j['datetime'])
        assert times == sorted(times, reverse=True)

    def test_get_and_loc_agree_with_position(self, newest_is_another_event):
        j = newest_is_another_event.journal(event='build:end')
        assert j.get(0).get('event') == 'build:end'
        assert j.iloc[0]['event'] == 'build:end'


class TestLastbuilt:

    def test_returns_the_build_end_when_a_newer_event_exists(self, newest_is_another_event):
        """The repro: this raised KeyError: 0."""
        entry = newest_is_another_event.lastbuilt()
        assert isinstance(entry, DatajournalEntry)
        assert entry.get('event') == 'build:end'

    def test_returns_the_newest_of_several_builds(self, tmp_path):
        for x in (1, 2, 3):
            Built(url=str(tmp_path), spec={'x': x}).build()
        block = Built(url=str(tmp_path), spec={'x': 1})
        entry = block.lastbuilt()
        newest = block.journal(event='build:end')['datetime'].max()
        assert entry.get('datetime') == newest

    def test_returns_none_when_nothing_was_built(self, tmp_path):
        Built(url=str(tmp_path), spec={'x': 1}).write_journal_entry(event='note')
        assert Built(url=str(tmp_path), spec={'x': 1}).lastbuilt() is None


class TestLocSelectorWithFilters:
    """``journal(event=..., loc=0)`` was broken the same way ``lastbuilt`` was."""

    def test_loc_zero_selects_the_first_filtered_row(self, newest_is_another_event):
        entry = newest_is_another_event.journal(event='build:end', loc=0)
        assert entry.get('event') == 'build:end'

    def test_loc_and_iloc_agree_on_a_filtered_journal(self, newest_is_another_event):
        block = newest_is_another_event
        by_loc = block.journal(event='build:end', loc=0)
        by_iloc = block.journal(event='build:end', iloc=0)
        assert by_loc.get('datetime') == by_iloc.get('datetime')


class TestUserSlicingKeepsPandasSemantics:
    """The renumbering is guarded on filter_kwargs, so it must not leak.

    ``running()`` slices with boolean masks and then indexes with ``.iloc``; if
    renumbering fired on every construction, a caller relating a slice back to
    the full journal by label would silently get the wrong row.
    """

    def test_boolean_slice_keeps_original_labels(self, newest_is_another_event):
        j = newest_is_another_event.journal()
        sliced = j[j['event'] == 'build:end']
        assert list(sliced.index) == [1], "a user slice was renumbered"

    def test_running_still_works(self, newest_is_another_event):
        # No build:start outstanding here, so None is the correct answer -- the
        # point is that it resolves rather than raising on the sliced frame.
        assert newest_is_another_event.running() is None

    def test_datajournal_built_from_a_frame_without_filters_is_untouched(
            self, newest_is_another_event):
        j = newest_is_another_event.journal()
        raw = j[j['event'] == 'build:end']
        assert list(Datajournal(raw).index) == [1]
