"""
Finding a specialization without installing it; a stack deciding its blocks'
use_specializations; clearing a redirection.
"""
import os

import dbx
import sys

import pytest
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
from test_specializations import (  # noqa: E402
    TABLE_ANCHOR, RowTablet, RowTableV1, TestOneJournalReadForAWholeTable, adopted, v1, v1table,
)

from dbx.datablocks import DATAFILE, Datablock, InvalidBlocksError  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.fixture
def grown(tmp_path, monkeypatch):
    """A table of 3 tablets built, then its TABLET grown a topic with a specialization back to it."""
    v1table(tmp_path, spec={'n': 3}).build()
    TestOneJournalReadForAWholeTable._grow_the_tablet(monkeypatch)
    return tmp_path


def _redirected(table):
    return [table._form_block_(i, use_specializations=False).redirected() for i in range(table.n_tablets)]


class TestFindSpecialization:

    def test_found_not_installed(self, grown):
        table = v1table(grown, spec={'n': 3}, use_tablet_specializations=False)
        tablet = table.tablet(0)
        assert tablet.use_specializations is False
        assert not tablet.redirected()
        sp = tablet.find_specialization()
        assert isinstance(sp, Datablock.Specialization) and list(sp.topics) == ['rows']
        assert not tablet.redirected(), "finding it installs nothing"

    def test_nothing_to_find_when_nothing_narrower_was_built(self, tmp_path, monkeypatch):
        TestOneJournalReadForAWholeTable._grow_the_tablet(monkeypatch)
        table = v1table(tmp_path, spec={'n': 2}, use_tablet_specializations=False)
        assert table.tablet(0).find_specialization() is None

    def test_a_table_finds_its_tablets_and_installs_nothing(self, grown):
        table = v1table(grown, spec={'n': 3}, use_tablet_specializations=False)
        found = table.find_tablet_specializations(parallelization='inline')
        assert found.index.tolist() == [0, 1, 2]
        assert all(list(sp.topics) == ['rows'] for sp in found)
        assert table.find_block_specializations(found_only=True).index.tolist() == [0, 1, 2]
        assert _redirected(table) == [False, False, False]

    def test_forming_installs_nothing_and_adopting_installs_what_was_found(self, grown):
        table = v1table(grown, spec={'n': 3})
        found = table.find_tablet_specializations(parallelization='inline')
        assert _redirected(table) == [False, False, False]      # finding is a query
        assert [table.tablet(i).specialization for i in range(3)] == [None] * 3   # forming, too
        assert [adopted(table.tablet(i)).specialization for i in range(3)] == list(found)
        assert _redirected(table) == [True, True, True]


class TestUseBlockSpecializations:

    def test_the_stack_sets_its_blocks(self, grown):
        assert v1table(grown, spec={'n': 3}, use_block_specializations=False).tablet(1).use_specializations is False
        assert v1table(grown, spec={'n': 3}, use_tablet_specializations='memory').tablet(1).use_specializations == 'memory'

    def test_left_unset_it_is_the_blocks_own(self, grown):
        assert v1table(grown, spec={'n': 3}).tablet(1).use_specializations == RowTablet.USE_SPECIALIZATIONS

    def test_the_two_names_are_one_setting(self, grown):
        t = v1table(grown, spec={'n': 3}, use_tablet_specializations=False, use_block_specializations=False)
        assert t.use_tablet_specializations is False and t.use_block_specializations is False
        with pytest.raises(ValueError, match="disagree"):
            v1table(grown, spec={'n': 3}, use_tablet_specializations=False, use_block_specializations=True)

    def test_it_is_not_in_the_quote_unless_given(self, grown):
        assert 'use_block_specializations' not in v1table(grown, spec={'n': 3}).quote()
        assert 'use_block_specializations=False' in v1table(
            grown, spec={'n': 3}, use_block_specializations=False).quote()

    def test_nor_in_the_blocks_quote(self, grown):
        quiet = v1table(grown, spec={'n': 3}, use_block_specializations=False).tablet(0)
        assert quiet.quote() == v1table(grown, spec={'n': 3}, use_block_specializations=False).tablet(0).quote()
        assert 'use_specializations=False' not in quiet.quote()


class TestClearRedirection:

    def test_a_block_reads_its_own_again(self, grown):
        table = v1table(grown, spec={'n': 3})
        tablet = adopted(table.tablet(0))
        assert tablet.redirected() and tablet.get_redirection() is not None
        assert tablet.UNSAFE_clear_redirection(OVERRIDE=True) is True
        assert not tablet.redirected() and tablet.get_redirection() is None
        # Recorded: a fresh instance, specializations off, is not redirected either.
        again = v1table(grown, spec={'n': 3}, use_tablet_specializations=False).tablet(0)
        assert not again.redirected() and again.get_redirection() is None
        assert again.buildjournal(event='^UNSAFE_clear_redirection$', index=None)['hash'].tolist() == [tablet.hash]
        # And nothing that was built is gone.
        assert v1table(grown, spec={'n': 3}, use_tablet_specializations=False).valid()

    def test_nothing_to_clear(self, tmp_path):
        tablet = v1table(tmp_path, spec={'n': 1}).tablet(0)
        assert tablet.UNSAFE_clear_redirection(OVERRIDE=True) is False

    def test_a_table_clears_its_tablets(self, grown):
        table = v1table(grown, spec={'n': 3})
        assert [adopted(table.tablet(i)).redirected() for i in range(3)] == [True, True, True]
        cleared = table.UNSAFE_clear_tablet_redirections(OVERRIDE=True, parallelization='inline')
        assert cleared.tolist() == [True, True, True]
        assert _redirected(v1table(grown, spec={'n': 3}, use_tablet_specializations=False)) == [False, False, False]
        assert v1table(grown, spec={'n': 3}).block_buildjournal(event='^UNSAFE_clear_redirection$', index=None) is not None

    def test_without_override_it_does_nothing(self, grown, monkeypatch):
        monkeypatch.setattr('builtins.input', lambda *_: 'n')
        tablet = adopted(v1table(grown, spec={'n': 3}).tablet(0))
        assert tablet.UNSAFE_clear_redirection() is False
        assert tablet.redirected()


def test_a_query_does_not_reach_the_blocks_a_stack_caches(grown):
    """Finding forms tablets with specializations off; the tablets the table caches must not inherit that."""
    table = v1table(grown, spec={'n': 3})
    table.find_tablet_specializations(parallelization='inline')
    assert all(table.tablet(i).use_specializations == RowTablet.USE_SPECIALIZATIONS for i in range(3))
    assert all(adopted(table.tablet(i)).specialization is not None for i in range(3))


class TestSpecializeMethod:

    def test_find_specialization_returns_joined_row_with_paths(self, grown):
        table = v1table(grown, spec={'n': 3}, use_tablet_specializations=False)
        tablet = table.tablet(0)
        row = tablet.find_specialization()
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
        tablet = v1table(grown, spec={'n': 3}, tag='spec_only').tablet(0)
        assert not tablet.redirected()
        assert tablet.find_specialization() is not None
        res = tablet.specialize()
        assert res is tablet
        assert tablet.redirected()
        assert tablet.validtopic('rows')

    def test_find_specialization_on_redirected_block_resolves(self, grown):
        table = v1table(grown, spec={'n': 3}, tag='spec_redirected')
        tablet = table.tablet(0)
        tablet.specialize()
        assert tablet.redirected()
        row = tablet.find_specialization()
        assert row is not None
        assert row.resolved is True
        assert 'rows' in row.paths

    def test_valid_block_not_re_specialized(self, grown):
        table = v1table(grown, spec={'n': 3}, tag='valid_tablet', use_tablet_specializations=False)
        tablet = table.tablet(0)
        tablet.build()
        assert tablet.valid()
        assert not tablet.redirected()

        # When already valid, specialize() must not install a redirection
        tablet_to_spec = v1table(grown, spec={'n': 3}, tag='valid_tablet').tablet(0)
        assert not tablet_to_spec.redirected()
        res = tablet_to_spec.specialize()
        assert res is tablet_to_spec
        assert not tablet_to_spec.redirected()
        with tablet_to_spec.fs.open(tablet_to_spec.path('rows'), 'r') as f:
            assert f.read() == "rows-0\n"

    def test_datastack_specialize_with_a_journal_for_its_blocks_does_not_crash(self, grown):
        from dbx.journals import BuildJournalFrame
        table = v1table(grown, spec={'n': 3}, tag='bj_test')
        bj = BuildJournalFrame(None, read_anchor='nonexistent', read_datalake=str(grown))
        table.specialize(journal=bj)

    def test_valid_stack_skips_block_specializations(self, grown):
        table = v1table(grown, spec={'n': 3}, tag='all_valid')
        table.build()
        assert table.valid()
        res = table._install_block_specializations_()
        assert res is not None
        assert all(v is None for v in res)

    def test_stack_with_own_specializations_and_a_journal_for_its_blocks_does_not_crash(self, grown):
        from dbx.journals import BuildJournalFrame
        from test_specializations import v2table

        table = v2table(grown, tag='stack_spec_test')
        bj = BuildJournalFrame(None, read_anchor=RowTablet.anchor, read_datalake=str(grown))
        # The blocks' journal is the blocks', not the stack's own: it must not reach the stack's specialize.
        table.specialize(journal=bj)

    def test_install_block_specializations_skips_valid_and_specializes_invalid(self, grown):
        # Build tablet 0 directly without specializations
        tablet0 = v1table(grown, spec={'n': 3}, tag='mixed_valid', use_tablet_specializations=False).tablet(0)
        tablet0.build()
        assert tablet0.valid()
        assert not tablet0.redirected()

        # Table with use_tablet_specializations=True has tablet 0 valid, tablets 1 and 2 unbuilt
        table = v1table(grown, spec={'n': 3}, tag='mixed_valid')
        res = table._install_block_specializations_()
        assert res is not None
        # tablet 0 was already valid, so not specialized
        assert res[0] is None
        # tablets 1 and 2 adopted specializations
        assert res[1] is not None
        assert res[2] is not None

        # Verify tablet 0 stayed unredirected with its direct data
        t0 = table.tablet(0)
        assert not t0.redirected()
        with t0.fs.open(t0.path('rows'), 'r') as f:
            assert f.read() == "rows-0\n"

        # Verify tablet 1 was redirected
        t1 = table.tablet(1)
        assert t1.redirected()

    def test_valid_does_not_install_specialization(self, grown):
        # Tag-keyed, so it is not the table `grown` built.
        table = v1table(grown, spec={'n': 3}, tag='valid_no_spec', keyby='tag_version_shorthash')
        # table is not built, tablets are not built
        # calling valid() must NOT install specializations
        assert not table.valid()
        assert not table.tablet(0).valid()
        assert not table.tablet(0).redirected()

        # calling valid_tablets() must NOT install specializations
        v_series = table.valid_tablets()
        assert not v_series.any()
        assert not table.tablet(0).redirected()
        assert not table.tablet(1).redirected()
        assert not table.tablet(2).redirected()

    def test_valid_tablets_with_all_sentinels_reads_no_journal(self, grown, monkeypatch):
        # Build the table so all tab_paths sentinels exist
        table = v1table(grown, spec={'n': 3}, tag='sentinels_built')
        table.build()
        assert table.valid()

        # Re-instantiate table: sentinels exist on disk
        t2 = v1table(grown, spec={'n': 3}, tag='sentinels_built')
        # Monkeypatch _build_journal_ to fail if called
        def fail_build_journal(*args, **kwargs):
            raise AssertionError("_build_journal_ should not be called by valid_tablets!")
        monkeypatch.setattr(t2, '_build_journal_', fail_build_journal)

        # valid_tablets must succeed and return all True without calling _build_journal_
        valid_res = t2.valid_tablets()
        assert len(valid_res) == 3
        assert valid_res.all()

    def test_tablet_specializations_from_string_representation(self, grown):
        spec_str = (
            "[Specialization(spec={}, topics={'rows': 'SLICETOPIC'}, "
            "anchor='old.RowTab', legacy=['all'], note='Legacy build')]"
        )
        table = v1table(grown, spec={'n': 3}, TABLET_SPECIALIZATIONS=spec_str)
        specs = table._block_specializations_()
        assert len(specs) == 1
        assert isinstance(specs[0], Datablock.Specialization)
        assert specs[0].anchor == 'old.RowTab'

        # Also test that quote() produces evaluable TABLET_SPECIALIZATIONS records
        q = table.quote()
        assert "TABLET_SPECIALIZATIONS=" in q

    def test_stack_already_valid_skips_install_block_specializations(self, grown, monkeypatch):
        table = v1table(grown, spec={'n': 3}, tag='already_valid_test')
        table.build()
        assert table.valid()

        # Reconstructed table that is already valid
        t2 = v1table(grown, spec={'n': 3}, tag='already_valid_test')
        assert t2.valid()

        def fail_blocks_journal(*args, **kwargs):
            raise AssertionError("_blocks_buildjournal_ should not be called when stack is already valid!")
        monkeypatch.setattr(t2, '_blocks_buildjournal_', fail_blocks_journal)

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
            SourceA.anchor: sa.buildjournal(),
            SourceB.anchor: sb.buildjournal(),
        }
        # Tag-keyed, so the two are two blocks, each to install its own.
        target_dict = Target(datalake=lake, tag="multi_j", keyby='tag_version_shorthash', use_specializations=True)
        res = target_dict._install_specialization_(journal=j_dict)
        assert res is not None
        assert target_dict.valid()
        assert target_dict._redirected_paths_['ta'] == sa.path('ta')
        assert target_dict._redirected_paths_['tb'] == sb.path('tb')

        # Test passing journal as list (corresponding to SPECIALIZATIONS order)
        j_list = [sa.buildjournal(), sb.buildjournal()]
        target_list = Target(datalake=lake, tag="multi_j_list", keyby='tag_version_shorthash', use_specializations=True)
        res_list = target_list._install_specialization_(journal=j_list)
        assert res_list is not None
        assert target_list.valid()
        assert target_list._redirected_paths_['ta'] == sa.path('ta')
        assert target_list._redirected_paths_['tb'] == sb.path('tb')


class RowTableNoMarkers(RowTableV1):
    """RowTableV1 without `tab_paths`: which of its tablets are built is known only by asking them."""

    TOPICS = {'summary': 'summary.txt', 'done': 'done'}


def nomarkers(url, **kw):
    spec = kw.pop('spec', {'n': 3})
    return RowTableNoMarkers(datalake=str(url), anchor=TABLE_ANCHOR, spec=spec, **kw)


@pytest.fixture
def rekeyed(tmp_path, monkeypatch):
    """A table without `tab_paths`, built; then its TABLET grown a topic, re-keying every tablet.

    The table's own identity does not move, so it is still valid over tablets that are not.
    """
    nomarkers(tmp_path).build()
    TestOneJournalReadForAWholeTable._grow_the_tablet(monkeypatch)
    table = nomarkers(tmp_path)
    assert table.valid()
    assert not table.valid_tablets(parallelization='inline').any()
    return tmp_path


class TestAValidStackSpecializesItsTablets:
    """A stack's own validity says nothing about tablets that moved to identities of their own."""

    def test_build_adopts_the_tablets_of_a_valid_stack_and_builds_what_they_owe(self, rekeyed):
        # Adopted, the tablets still owe `extra`: the stack is done, its tablets are not -- so its build builds them.
        nomarkers(rekeyed).build()
        table = nomarkers(rekeyed)
        assert table.tablets_redirected(parallelization='inline').all()
        assert table.tablet(1).read('rows') == "rows-1\n", "adopted"
        assert table.tablet(1).read('extra') == "extra-1\n", "built"
        assert table.valid_tablets(parallelization='inline', validation='valid').all()
        assert table._blocks_cross_checked_()

    def test_a_tablet_its_build_leaves_invalid_is_an_error(self, rekeyed, monkeypatch):
        # Said by the tablet's own build, where it failed -- not by the stack after the fact.
        monkeypatch.setattr(RowTablet, '__build__', lambda self: None)      # writes nothing
        with pytest.raises(ValueError, match=r"Tablet 0 of .*failed to validate"):
            nomarkers(rekeyed).build()

    def test_build_blocks_builds_what_the_adopted_tablets_owe_and_vouches_for_them(self, rekeyed):
        nomarkers(rekeyed).build_blocks()
        table = nomarkers(rekeyed)
        assert table.valid_tablets(parallelization='inline', validation='valid').all()
        assert table._blocks_cross_checked_()
        nomarkers(rekeyed).build()      # done, and quiet about it

    def test_a_rekeyed_tablet_fails_the_cross_check(self, rekeyed, monkeypatch):
        nomarkers(rekeyed).build_blocks()
        assert nomarkers(rekeyed)._blocks_cross_checked_()
        monkeypatch.setattr(RowTablet, 'VERSION', 2)
        assert not nomarkers(rekeyed)._blocks_cross_checked_()

    def test_specialize_tablets_adopts_them_and_builds_nothing(self, rekeyed):
        found = nomarkers(rekeyed).specialize_tablets(parallelization='inline')
        assert [sp is not None for sp in found] == [True, True, True]
        table = nomarkers(rekeyed)
        assert table.tablets_redirected(parallelization='inline').all()
        assert not any(table.tablet(i).validtopic('extra') for i in range(3)), "adopting builds nothing"

    def test_again_it_finds_nothing_to_do(self, rekeyed):
        nomarkers(rekeyed).specialize_tablets(parallelization='inline')
        again = nomarkers(rekeyed).specialize_tablets(parallelization='inline')
        assert [sp is None for sp in again] == [True, True, True]


class Over(Datablock):
    """A block over the table -- as a clip's fold is over its bags' table: built on it, its hash not the tablets'."""

    TOPICS = {'over': DATAFILE('over.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        table: RowTableNoMarkers = None

    def __build__(self):
        with open(self.path('over', ensure_dirpath=True), 'w') as f:
            f.write('over')


def over(url):
    return Over(datalake=str(url), spec={'table': nomarkers(url)})


class TestBuildTreeSpecializesUnderAValidChild:
    """build_tree(deep='try_specialize_children'): a valid child vouches for itself, not for the tablets that moved under it."""

    @pytest.fixture
    def built(self, rekeyed):
        over(rekeyed).build_tree()
        assert over(rekeyed).valid() and nomarkers(rekeyed).valid()
        assert not nomarkers(rekeyed).tablets_redirected(parallelization='inline').any()
        return rekeyed

    def test_shallow_leaves_the_tablets_unadopted(self, built):
        over(built).build_tree()
        assert not nomarkers(built).tablets_redirected(parallelization='inline').any()

    def test_try_specialize_children_adopts_them_and_builds_nothing(self, built, monkeypatch):
        monkeypatch.setattr(Over, '__build__', lambda self: pytest.fail("a valid block is not built"))
        over(built).build_tree(deep='try_specialize_children')
        table = nomarkers(built)
        assert table.tablets_redirected(parallelization='inline').all()
        assert table.tablet(1).read('rows') == "rows-1\n", "adopted"
        assert not any(table.tablet(i).validtopic('extra') for i in range(3)), "adopting builds nothing"

    def test_try_specialize_tree_adopts_them_under_a_valid_stack(self, built):
        over(built).try_specialize_tree()
        assert nomarkers(built).tablets_redirected(parallelization='inline').all()

    def test_a_deep_it_does_not_know_is_refused(self, built):
        with pytest.raises(ValueError, match="'try_specialize_children'"):
            over(built).build_tree(deep='adopt')


class TestTheBlocksBuildJournalIsAFrame:
    """What a stack hands the workers forming its blocks is a BuildJournalFrame, its read_anchor theirs."""

    @staticmethod
    def frame(**kwargs):
        import pandas as pd
        from dbx.journals import BuildJournalFrame
        return BuildJournalFrame(pd.DataFrame({'hash': ['h'], 'event': ['build:end']}), unnormalized=True, **kwargs)

    def test_it_survives_dbx_eval_as_a_frame(self):
        frame = self.frame(read_anchor='pkg.A', read_datalake='/lake')
        back = dbx.databits.eval(frame)
        assert back is frame and (back.read_anchor, back.read_datalake, back.by_hash) == ('pkg.A', '/lake', False)

    def test_it_crosses_a_process_with_what_it_was_read_for(self):
        import pickle
        from dbx.journals import BuildJournalFrame
        frame = pickle.loads(pickle.dumps(BuildJournalFrame(None, read_anchor='pkg.A', read_datalake='/lake',
                                                            by_hash=True)))
        assert (frame.read_anchor, frame.read_datalake, frame.by_hash) == ('pkg.A', '/lake', True)

    def test_only_its_own_blocks_resolve_against_it(self):
        from dbx.datablocks import BY_HASH, _forming_journal_, forming_with_journal
        from dbx.journals import BuildJournalFrame
        with forming_with_journal(BuildJournalFrame(None, read_anchor='pkg.A', read_datalake='/lake', by_hash=True)):
            assert _forming_journal_('pkg.A', '/lake') is BY_HASH
            assert _forming_journal_('pkg.B', '/lake') is None
            assert _forming_journal_('pkg.A', '/elsewhere') is None
        assert _forming_journal_('pkg.A', '/lake') is None

    def test_a_block_gets_it_as_a_journal_given(self):
        from dbx.datablocks import _forming_journal_, forming_with_journal
        from dbx.journals import BuildJournalFrame
        with forming_with_journal(self.frame(read_anchor='pkg.A', read_datalake='/lake')):
            got = _forming_journal_('pkg.A', '/lake')
        assert isinstance(got, BuildJournalFrame) and got.read_anchor is None and list(got['hash']) == ['h']

    def test_a_journal_given_is_not_one_read_for_blocks(self):
        from dbx.datablocks import forming_with_journal, _forming_journal_
        with forming_with_journal(self.frame()):
            assert _forming_journal_('pkg.A', '/lake') is None
