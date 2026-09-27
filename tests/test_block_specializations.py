"""
Finding a specialization without installing it; a stack deciding its blocks'
use_specializations; clearing a redirection.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_specializations import RowTab, TestOneJournalReadForAWholeTable, adopted, v1table  # noqa: E402

from dbx.datablocks import Datablock  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.fixture
def grown(tmp_path, monkeypatch):
    """A table of 3 tabs built, then its TAB grown a topic with a specialization back to it."""
    v1table(tmp_path, spec={'n': 3}).build()
    TestOneJournalReadForAWholeTable._grow_the_tab(monkeypatch)
    return tmp_path


def _redirected(table):
    return [table._form_block_(i, use_specializations=False).redirected() for i in range(table.n_tabs)]


class TestFindSpecialization:

    def test_found_not_installed(self, grown):
        table = v1table(grown, spec={'n': 3}, use_tab_specializations=False)
        tab = table.tab(0)
        assert tab.use_specializations is False
        assert not tab.redirected()
        sp = tab.find_specialization()
        assert isinstance(sp, Datablock.Specialization) and list(sp.topics) == ['rows']
        assert not tab.redirected(), "finding it installs nothing"

    def test_nothing_to_find_when_nothing_narrower_was_built(self, tmp_path, monkeypatch):
        TestOneJournalReadForAWholeTable._grow_the_tab(monkeypatch)
        table = v1table(tmp_path, spec={'n': 2}, use_tab_specializations=False)
        assert table.tab(0).find_specialization() is None

    def test_a_table_finds_its_tabs_and_installs_nothing(self, grown):
        table = v1table(grown, spec={'n': 3}, use_tab_specializations=False)
        found = table.find_tab_specializations(parallelization='inline')
        assert found.index.tolist() == [0, 1, 2]
        assert all(list(sp.topics) == ['rows'] for sp in found)
        assert table.find_block_specializations(found_only=True).index.tolist() == [0, 1, 2]
        assert _redirected(table) == [False, False, False]

    def test_forming_installs_nothing_and_adopting_installs_what_was_found(self, grown):
        table = v1table(grown, spec={'n': 3})
        found = table.find_tab_specializations(parallelization='inline')
        assert _redirected(table) == [False, False, False]      # finding is a query
        assert [table.tab(i).specialization for i in range(3)] == [None] * 3   # forming, too
        assert [adopted(table.tab(i)).specialization for i in range(3)] == list(found)
        assert _redirected(table) == [True, True, True]


class TestUseBlockSpecializations:

    def test_the_stack_sets_its_blocks(self, grown):
        assert v1table(grown, spec={'n': 3}, use_block_specializations=False).tab(1).use_specializations is False
        assert v1table(grown, spec={'n': 3}, use_tab_specializations='memory').tab(1).use_specializations == 'memory'

    def test_left_unset_it_is_the_blocks_own(self, grown):
        assert v1table(grown, spec={'n': 3}).tab(1).use_specializations == RowTab.USE_SPECIALIZATIONS

    def test_the_two_names_are_one_setting(self, grown):
        t = v1table(grown, spec={'n': 3}, use_tab_specializations=False, use_block_specializations=False)
        assert t.use_tab_specializations is False and t.use_block_specializations is False
        with pytest.raises(ValueError, match="disagree"):
            v1table(grown, spec={'n': 3}, use_tab_specializations=False, use_block_specializations=True)

    def test_it_is_not_in_the_quote_unless_given(self, grown):
        assert 'use_block_specializations' not in v1table(grown, spec={'n': 3}).quote()
        assert 'use_block_specializations=False' in v1table(
            grown, spec={'n': 3}, use_block_specializations=False).quote()

    def test_nor_in_the_blocks_quote(self, grown):
        quiet = v1table(grown, spec={'n': 3}, use_block_specializations=False).tab(0)
        assert quiet.quote() == v1table(grown, spec={'n': 3}, use_block_specializations=False).tab(0).quote()
        assert 'use_specializations=False' not in quiet.quote()


class TestClearRedirection:

    def test_a_block_reads_its_own_again(self, grown):
        table = v1table(grown, spec={'n': 3})
        tab = adopted(table.tab(0))
        assert tab.redirected() and tab.get_redirection() is not None
        assert tab.UNSAFE_clear_redirection(OVERRIDE=True) is True
        assert not tab.redirected() and tab.get_redirection() is None
        # Recorded: a fresh instance, specializations off, is not redirected either.
        again = v1table(grown, spec={'n': 3}, use_tab_specializations=False).tab(0)
        assert not again.redirected() and again.get_redirection() is None
        assert again.journal(event='^UNSAFE_clear_redirection$', index=None)['hash'].tolist() == [tab.hash]
        # And nothing that was built is gone.
        assert v1table(grown, spec={'n': 3}, use_tab_specializations=False).valid()

    def test_nothing_to_clear(self, tmp_path):
        tab = v1table(tmp_path, spec={'n': 1}).tab(0)
        assert tab.UNSAFE_clear_redirection(OVERRIDE=True) is False

    def test_a_table_clears_its_tabs(self, grown):
        table = v1table(grown, spec={'n': 3})
        assert [adopted(table.tab(i)).redirected() for i in range(3)] == [True, True, True]
        cleared = table.UNSAFE_clear_tab_redirections(OVERRIDE=True, parallelization='inline')
        assert cleared.tolist() == [True, True, True]
        assert _redirected(v1table(grown, spec={'n': 3}, use_tab_specializations=False)) == [False, False, False]
        assert v1table(grown, spec={'n': 3}).block_journal(event='^UNSAFE_clear_redirection$', index=None) is not None

    def test_without_override_it_does_nothing(self, grown, monkeypatch):
        monkeypatch.setattr('builtins.input', lambda *_: 'n')
        tab = adopted(v1table(grown, spec={'n': 3}).tab(0))
        assert tab.UNSAFE_clear_redirection() is False
        assert tab.redirected()


def test_a_query_does_not_reach_the_blocks_a_stack_caches(grown):
    """Finding forms tabs with specializations off; the tabs the table caches must not inherit that."""
    table = v1table(grown, spec={'n': 3})
    table.find_tab_specializations(parallelization='inline')
    assert all(table.tab(i).use_specializations == RowTab.USE_SPECIALIZATIONS for i in range(3))
    assert all(adopted(table.tab(i)).specialization is not None for i in range(3))
