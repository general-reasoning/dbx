"""
Finding a specialization without installing it; a stack deciding its blocks'
use_specializations; clearing a redirection.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_specializations import (  # noqa: E402
    TABLE_ANCHOR, RowTab, RowTableV1, TestOneJournalReadForAWholeTable, adopted, v1, v1table,
)

from dbx.datablocks import Datablock, InvalidBlocksError  # noqa: E402


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


class TestSpecializeMethod:

    def test_find_specialization_returns_joined_row_with_paths(self, grown):
        table = v1table(grown, spec={'n': 3}, use_tab_specializations=False)
        tab = table.tab(0)
        row = tab.find_specialization()
        assert row is not None
        assert isinstance(row, Datablock.Specialization)
        assert isinstance(row, dict)
        assert row.resolved is True
        assert 'rows' in row.paths
        assert row.paths['rows'].endswith('rows.txt')
        rendered = str(row)
        assert 'RESOLVED to journal entry' in rendered
        assert 'paths' in rendered
        assert 'rows:' in rendered
        assert 'specialization Specialization(' in rendered

    def test_specialize_installs_specialization(self, grown):
        tab = v1table(grown, spec={'n': 3}, tag='spec_only').tab(0)
        assert not tab.redirected()
        assert tab.find_specialization() is not None
        res = tab.specialize()
        assert res is tab
        assert tab.redirected()
        assert tab.validtopic('rows')

    def test_find_specialization_on_redirected_block_resolves(self, grown):
        table = v1table(grown, spec={'n': 3}, tag='spec_redirected')
        tab = table.tab(0)
        tab.specialize()
        assert tab.redirected()
        row = tab.find_specialization()
        assert row is not None
        assert row.resolved is True
        assert 'rows' in row.paths

    def test_valid_block_not_re_specialized(self, grown):
        table = v1table(grown, spec={'n': 3}, tag='valid_tab', use_tab_specializations=False)
        tab = table.tab(0)
        tab.build()
        assert tab.valid()
        assert not tab.redirected()

        # When already valid, specialize() must not install a redirection
        tab_to_spec = v1table(grown, spec={'n': 3}, tag='valid_tab').tab(0)
        assert not tab_to_spec.redirected()
        res = tab_to_spec.specialize()
        assert res is tab_to_spec
        assert not tab_to_spec.redirected()
        with tab_to_spec.fs.open(tab_to_spec.path('rows'), 'r') as f:
            assert f.read() == "rows-0\n"

    def test_datastack_specialize_with_blocks_journal_does_not_crash(self, grown):
        from dbx.datablocks import BlocksJournal
        table = v1table(grown, spec={'n': 3}, tag='bj_test')
        bj = BlocksJournal(journal=None, anchor='nonexistent', datalake=str(grown))
        table.specialize(journal=bj)

    def test_valid_stack_skips_block_specializations(self, grown):
        table = v1table(grown, spec={'n': 3}, tag='all_valid')
        table.build()
        assert table.valid()
        res = table._install_block_specializations_()
        assert res is not None
        assert all(v is None for v in res)

    def test_stack_with_own_specializations_and_blocks_journal_does_not_crash(self, grown):
        from dbx.datablocks import BlocksJournal
        from test_specializations import v2table

        table = v2table(grown, tag='stack_spec_test')
        bj = BlocksJournal(journal=None, anchor=RowTab.anchor, datalake=str(grown))
        # Must not raise AttributeError: 'BlocksJournal' object has no attribute 'empty'
        table.specialize(journal=bj)

    def test_install_block_specializations_skips_valid_and_specializes_invalid(self, grown):
        # Build tab 0 directly without specializations
        tab0 = v1table(grown, spec={'n': 3}, tag='mixed_valid', use_tab_specializations=False).tab(0)
        tab0.build()
        assert tab0.valid()
        assert not tab0.redirected()

        # Table with use_tab_specializations=True has tab 0 valid, tabs 1 and 2 unbuilt
        table = v1table(grown, spec={'n': 3}, tag='mixed_valid')
        res = table._install_block_specializations_()
        assert res is not None
        # tab 0 was already valid, so not specialized
        assert res[0] is None
        # tabs 1 and 2 adopted specializations
        assert res[1] is not None
        assert res[2] is not None

        # Verify tab 0 stayed unredirected with its direct data
        t0 = table.tab(0)
        assert not t0.redirected()
        with t0.fs.open(t0.path('rows'), 'r') as f:
            assert f.read() == "rows-0\n"

        # Verify tab 1 was redirected
        t1 = table.tab(1)
        assert t1.redirected()

    def test_valid_does_not_install_specialization(self, grown):
        table = v1table(grown, spec={'n': 3}, tag='valid_no_spec')
        # table is not built, tabs are not built
        # calling valid() must NOT install specializations
        assert not table.valid()
        assert not table.tab(0).valid()
        assert not table.tab(0).redirected()

        # calling valid_tabs() must NOT install specializations
        v_series = table.valid_tabs()
        assert not v_series.any()
        assert not table.tab(0).redirected()
        assert not table.tab(1).redirected()
        assert not table.tab(2).redirected()

    def test_valid_tabs_with_all_sentinels_reads_no_journal(self, grown, monkeypatch):
        # Build the table so all tab_paths sentinels exist
        table = v1table(grown, spec={'n': 3}, tag='sentinels_built')
        table.build()
        assert table.valid()

        # Re-instantiate table: sentinels exist on disk
        t2 = v1table(grown, spec={'n': 3}, tag='sentinels_built')
        # Monkeypatch _build_journal_ to fail if called
        def fail_build_journal(*args, **kwargs):
            raise AssertionError("_build_journal_ should not be called by valid_tabs!")
        monkeypatch.setattr(t2, '_build_journal_', fail_build_journal)

        # valid_tabs must succeed and return all True without calling _build_journal_
        valid_res = t2.valid_tabs()
        assert len(valid_res) == 3
        assert valid_res.all()

    def test_tab_specializations_from_string_representation(self, grown):
        spec_str = (
            "[Specialization(spec={}, topics={'rows': 'SLICETOPIC'}, "
            "anchor='old.RowTab', legacy=['all'], note='Legacy build')]"
        )
        table = v1table(grown, spec={'n': 3}, TAB_SPECIALIZATIONS=spec_str)
        specs = table._block_specializations_()
        assert len(specs) == 1
        assert isinstance(specs[0], Datablock.Specialization)
        assert specs[0].anchor == 'old.RowTab'

        # Also test that quote() produces evaluable TAB_SPECIALIZATIONS records
        q = table.quote()
        assert "TAB_SPECIALIZATIONS=" in q

    def test_stack_already_valid_skips_install_block_specializations(self, grown, monkeypatch):
        table = v1table(grown, spec={'n': 3}, tag='already_valid_test')
        table.build()
        assert table.valid()

        # Reconstructed table that is already valid
        t2 = v1table(grown, spec={'n': 3}, tag='already_valid_test')
        assert t2.valid()

        def fail_blocks_journal(*args, **kwargs):
            raise AssertionError("_blocks_journal_ should not be called when stack is already valid!")
        monkeypatch.setattr(t2, '_blocks_journal_', fail_blocks_journal)

        res = t2._install_block_specializations_()
        assert res is not None
        assert len(res) == 3
        assert all(x is None for x in res)

    def test_chased_redirections(self, grown):
        # Create block C with actual data
        b_c = v1(grown, tag='block_c').build()
        assert b_c.valid()
        c_path = b_c.path('spectra')

        # Block B redirects to Block C
        b_b = v1(grown, tag='block_b')
        assert b_b.UNSAFE_redirect(paths={'spectra': c_path}, OVERRIDE=True)
        assert b_b._redirected_paths_['spectra'] == c_path

        # Block A redirects to Block B (which is itself redirected)
        b_a = v1(grown, tag='block_a')
        b_intermediate = os.path.join(b_b.anchorkeypath, 'spectra')
        assert b_a.UNSAFE_redirect(paths={'spectra': b_intermediate}, OVERRIDE=True)

        # b_a must have chased through b_b to point directly to c_path!
        assert b_a._redirected_paths_['spectra'] == c_path
        with b_a.fs.open(b_a.path('spectra'), 'r') as f:
            assert f.read() == "spectra-16000"

    def test_read_mds_shard_raises_on_missing_index(self, tmp_path):
        import fsspec
        from dbx.datastreams import read_mds_shard
        import pytest

        fs = fsspec.filesystem('file')
        nonexistent = str(tmp_path / "nonexistent_shard")
        with pytest.raises(FileNotFoundError, match="MDS shard index not found"):
            read_mds_shard(nonexistent, fs)


class TestCombineSpecializations:

    def test_combining_multiple_specializations(self, tmp_path):
        from dbx.datablocks import Datablock, DATAFILE

        class Source1(Datablock):
            VERSION = 1
            TOPICS = {'t1': DATAFILE('t1.txt')}

            def __build__(self):
                p = self.path('t1', ensure_dirpath=True)
                with self.fs.open(p, 'w') as f:
                    f.write("content_t1")

        class Source2(Datablock):
            VERSION = 1
            TOPICS = {'t2': DATAFILE('t2.txt')}

            def __build__(self):
                p = self.path('t2', ensure_dirpath=True)
                with self.fs.open(p, 'w') as f:
                    f.write("content_t2")

        lake = str(tmp_path / "lake")
        s1 = Source1(datalake=lake, tag="common")
        s1.build()
        assert s1.valid()

        s2 = Source2(datalake=lake, tag="common")
        s2.build()
        assert s2.valid()

        class Composite(Datablock):
            VERSION = 1
            TOPICS = {'t1': DATAFILE('t1.txt'), 't2': DATAFILE('t2.txt')}
            SPECIALIZATIONS = [
                Datablock.Specialization(
                    spec={},
                    topics={'t1': DATAFILE('t1.txt')},
                    redirect_topics=['t1'],
                    anchor=Source1.anchor,
                    note='t1 from Source1',
                ),
                Datablock.Specialization(
                    spec={},
                    topics={'t2': DATAFILE('t2.txt')},
                    redirect_topics=['t2'],
                    anchor=Source2.anchor,
                    note='t2 from Source2',
                ),
            ]

        comp = Composite(datalake=lake, tag="common", use_specializations=False)
        assert not comp.valid()

        comp_specialized = Composite(datalake=lake, tag="common", use_specializations=True)
        res = comp_specialized._install_specialization_()
        assert res is not None
        assert comp_specialized.valid()
        assert comp_specialized._redirected_paths_['t1'] == s1.path('t1')
        assert comp_specialized._redirected_paths_['t2'] == s2.path('t2')
        with comp_specialized.fs.open(comp_specialized.path('t1'), 'r') as f:
            assert f.read() == "content_t1"
        with comp_specialized.fs.open(comp_specialized.path('t2'), 'r') as f:
            assert f.read() == "content_t2"

    def test_polymorphic_journal_dict_and_list(self, tmp_path):
        from dbx.datablocks import Datablock, DATAFILE

        class SourceA(Datablock):
            VERSION = 1
            TOPICS = {'ta': DATAFILE('ta.txt')}

            def __build__(self):
                p = self.path('ta', ensure_dirpath=True)
                with self.fs.open(p, 'w') as f:
                    f.write("from_A")

        class SourceB(Datablock):
            VERSION = 1
            TOPICS = {'tb': DATAFILE('tb.txt')}

            def __build__(self):
                p = self.path('tb', ensure_dirpath=True)
                with self.fs.open(p, 'w') as f:
                    f.write("from_B")

        lake = str(tmp_path / "lake")
        sa = SourceA(datalake=lake, tag="multi_j")
        sa.build()
        sb = SourceB(datalake=lake, tag="multi_j")
        sb.build()

        class Target(Datablock):
            VERSION = 1
            TOPICS = {'ta': DATAFILE('ta.txt'), 'tb': DATAFILE('tb.txt')}
            SPECIALIZATIONS = [
                Datablock.Specialization(
                    spec={},
                    topics={'ta': DATAFILE('ta.txt')},
                    redirect_topics=['ta'],
                    anchor=SourceA.anchor,
                ),
                Datablock.Specialization(
                    spec={},
                    topics={'tb': DATAFILE('tb.txt')},
                    redirect_topics=['tb'],
                    anchor=SourceB.anchor,
                ),
            ]

        # Test passing journal as dict
        j_dict = {
            SourceA.anchor: sa.journal(),
            SourceB.anchor: sb.journal(),
        }
        target_dict = Target(datalake=lake, tag="multi_j", use_specializations=True)
        res = target_dict._install_specialization_(journal=j_dict)
        assert res is not None
        assert target_dict.valid()
        assert target_dict._redirected_paths_['ta'] == sa.path('ta')
        assert target_dict._redirected_paths_['tb'] == sb.path('tb')

        # Test passing journal as list (corresponding to SPECIALIZATIONS order)
        j_list = [sa.journal(), sb.journal()]
        target_list = Target(datalake=lake, tag="multi_j_list", use_specializations=True)
        res_list = target_list._install_specialization_(journal=j_list)
        assert res_list is not None
        assert target_list.valid()
        assert target_list._redirected_paths_['ta'] == sa.path('ta')
        assert target_list._redirected_paths_['tb'] == sb.path('tb')


class RowTableNoMarkers(RowTableV1):
    """RowTableV1 without `tab_paths`: which of its tabs are built is known only by asking them."""

    TOPICS = {'summary': 'summary.txt', 'done': 'done'}


def nomarkers(url, **kw):
    spec = kw.pop('spec', {'n': 3})
    return RowTableNoMarkers(datalake=str(url), anchor=TABLE_ANCHOR, spec=spec, **kw)


@pytest.fixture
def rekeyed(tmp_path, monkeypatch):
    """A table without `tab_paths`, built; then its TAB grown a topic, re-keying every tab.

    The table's own identity does not move, so it is still valid over tabs that are not.
    """
    nomarkers(tmp_path).build()
    TestOneJournalReadForAWholeTable._grow_the_tab(monkeypatch)
    table = nomarkers(tmp_path)
    assert table.valid()
    assert not table.valid_tabs(parallelization='inline').any()
    return tmp_path


class TestAValidStackSpecializesItsTabs:
    """A stack's own validity says nothing about tabs that moved to identities of their own."""

    def test_build_adopts_the_tabs_of_a_valid_stack(self, rekeyed):
        # Adopted, the tabs still owe `extra`: a valid stack over them is not done, and says so.
        with pytest.raises(InvalidBlocksError, match=r"(?s)3 of 3 blocks are not valid.*owes \['extra'\].*build_blocks\(\)"):
            nomarkers(rekeyed).build()
        table = nomarkers(rekeyed)
        assert table.tabs_redirected(parallelization='inline').all()
        assert table.tab(1).read('rows') == "rows-1\n"

    def test_build_blocks_builds_what_the_adopted_tabs_owe_and_vouches_for_them(self, rekeyed):
        nomarkers(rekeyed).build_blocks()
        table = nomarkers(rekeyed)
        assert table.valid_tabs(parallelization='inline', validation='valid').all()
        assert table._blocks_cross_checked_()
        nomarkers(rekeyed).build()      # done, and quiet about it

    def test_a_rekeyed_tab_fails_the_cross_check(self, rekeyed, monkeypatch):
        nomarkers(rekeyed).build_blocks()
        assert nomarkers(rekeyed)._blocks_cross_checked_()
        monkeypatch.setattr(RowTab, 'VERSION', 2)
        assert not nomarkers(rekeyed)._blocks_cross_checked_()

    def test_specialize_tabs_adopts_them_and_builds_nothing(self, rekeyed):
        found = nomarkers(rekeyed).specialize_tabs(parallelization='inline')
        assert [sp is not None for sp in found] == [True, True, True]
        table = nomarkers(rekeyed)
        assert table.tabs_redirected(parallelization='inline').all()
        assert not any(table.tab(i).validtopic('extra') for i in range(3)), "adopting builds nothing"

    def test_again_it_finds_nothing_to_do(self, rekeyed):
        nomarkers(rekeyed).specialize_tabs(parallelization='inline')
        again = nomarkers(rekeyed).specialize_tabs(parallelization='inline')
        assert [sp is None for sp in again] == [True, True, True]
