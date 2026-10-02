"""
Failing loudly, clearly and early: at the cause, not far from it.

A manifest recorded under other identities vouches for nothing; a stack's
validation is its user's to choose; a block says why it is not valid; a
probe refuses, before any worker starts, to read tabs that are not valid;
and a VAR field that fails to evaluate says which field of which block.
"""
import os
import shutil
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_block_specializations import grown, nomarkers, rekeyed  # noqa: E402,F401
from test_specializations import RowTab  # noqa: E402
from test_probes import DummyModelEvaluatorFactory, DummySampleTable  # noqa: E402
from test_specializations import v1table  # noqa: E402

from dbx import FeatureAffineLogisticProbe  # noqa: E402
from dbx.datablocks import Datablock, InvalidBlocksError  # noqa: E402
from dbx.featuretables import Datacollator  # noqa: E402
from dbx import Featuretable  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class TestTheManifest:

    def test_tabs_that_moved_are_not_vouched_for(self, grown):
        # Built, with its manifest; then every tab re-keyed.
        table = v1table(grown, spec={'n': 3})
        assert table._read_blocks_manifest_() is not None
        assert not table._blocks_cross_checked_()
        assert not table.valid_tabs(parallelization='inline').any()

    def test_a_manifest_under_other_identities_vouches_for_nothing(self, rekeyed):
        table = nomarkers(rekeyed)
        assert os.path.exists(table._blocks_manifest_path_())
        assert not table._blocks_cross_checked_()

    def test_clearing_blocks_forgets_the_manifest(self, rekeyed):
        nomarkers(rekeyed).build_blocks()
        table = nomarkers(rekeyed)
        assert table._blocks_cross_checked_()
        table.UNSAFE_clear_blocks(indices=[1], clear_done=False, OVERRIDE=True)
        assert not nomarkers(rekeyed)._blocks_cross_checked_()
        assert nomarkers(rekeyed).valid_tabs(parallelization='inline').tolist() == [True, False, True]

    def test_the_sample_is_block_0_and_the_rest_at_random(self, rekeyed, monkeypatch):
        nomarkers(rekeyed).build_blocks()
        formed = []
        table = nomarkers(rekeyed, cross_check_blocks=2, cross_check_seed=7)
        orig = type(table).block
        monkeypatch.setattr(type(table), 'block', lambda self, i: formed.append(i) or orig(self, i))
        assert table._blocks_cross_checked_()
        assert formed[0] == 0 and len(formed) == 2 and formed[1] in (1, 2)

    def test_a_mismatch_anywhere_in_the_sample_is_caught(self, rekeyed):
        nomarkers(rekeyed).build_blocks()
        table = nomarkers(rekeyed, cross_check_blocks=3)
        path = table._blocks_manifest_path_()
        lines = open(path).read().splitlines()
        lines[3] = lines[3] + '-moved'          # block 2, as the manifest records it
        open(path, 'w').write('\n'.join(lines) + '\n')
        assert not table._blocks_cross_checked_()


class TestValidation:

    def test_it_is_the_stacks_and_not_its_identity(self, rekeyed):
        assert nomarkers(rekeyed, validation='validate').hash == nomarkers(rekeyed).hash
        with pytest.raises(ValueError, match="validation must be one of"):
            nomarkers(rekeyed, validation='thorough')

    def test_valid_asks_every_block_behind_the_manifest(self, rekeyed):
        nomarkers(rekeyed).build_blocks()
        os.remove(nomarkers(rekeyed).tab(1).path('rows'))    # behind the stack's back
        assert nomarkers(rekeyed).valid_tabs(parallelization='inline').all()          # cross_check cannot see it
        assert nomarkers(rekeyed, validation='valid').valid_tabs(parallelization='inline').tolist() == [True, False, True]

    def test_validate_fails_the_build_and_names_the_remedy(self, rekeyed, monkeypatch):
        nomarkers(rekeyed).build_blocks()
        monkeypatch.setattr(RowTab, '__validate__', lambda self, **kw: self.var.tab_idx != 1)
        with pytest.raises(InvalidBlocksError, match=r"(?s)1 of 3 blocks.*fails validate\(\).*UNSAFE_clear_blocks\(indices=\[1\]"):
            nomarkers(rekeyed, validation='validate').build()
        with pytest.raises(InvalidBlocksError, match="fails validate"):
            nomarkers(rekeyed, validation='validate').specialize_tabs(parallelization='inline')

    def test_validate_undoes_an_adoption_that_fails_it(self, rekeyed, monkeypatch):
        # The tabs owe `extra`, so let them owe nothing: adopted, they are valid -- and then fail validate().
        monkeypatch.setattr(RowTab, 'TOPICS', {'rows': 'rows.txt'})
        monkeypatch.setattr(RowTab, 'VERSION', 2)
        monkeypatch.setattr(RowTab, 'SPECIALIZATIONS', [
            Datablock.Specialization(spec={}, topics={'rows': 'rows.txt'}, version=1, note='v1')])
        monkeypatch.setattr(RowTab, '__validate__', lambda self, **kw: False)
        with pytest.raises(InvalidBlocksError,
                           match=r"(?s)resolves to fails validate\(\), and was not adopted.*without their specializations"):
            nomarkers(rekeyed, validation='validate').specialize_tabs(parallelization='inline')
        assert not nomarkers(rekeyed).tabs_redirected(parallelization='inline').any()
        # The remedy it names: built, not adopted -- and so passing validate(), were it not always False.
        monkeypatch.setattr(RowTab, '__validate__', lambda self, **kw: self.valid())
        nomarkers(rekeyed, use_block_specializations=False).build_blocks([0, 1, 2])
        table = nomarkers(rekeyed)
        assert not table.tabs_redirected(parallelization='inline').any()
        assert table.valid_tabs(parallelization='inline', validation='validate').all()


class TestWhyInvalid:

    def test_an_unadopted_tab(self, rekeyed):
        why = nomarkers(rekeyed).tab(0).why_invalid()
        assert "a specialization resolves for it and was never adopted" in why

    def test_an_adopted_tab_that_owes(self, rekeyed):
        nomarkers(rekeyed).specialize_tabs(parallelization='inline')
        assert "owes ['extra']" in nomarkers(rekeyed).tab(0).why_invalid()

    def test_a_tab_never_built(self, tmp_path):
        assert "no specialization resolves" in nomarkers(tmp_path).tab(0).why_invalid()

    def test_a_valid_tab(self, rekeyed):
        nomarkers(rekeyed).build_blocks()
        assert nomarkers(rekeyed).tab(0).why_invalid() is None


def _probe(url):
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5), tag="sample_table").build()
    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(signals=[("samples", "samples")], labels=[("labels", "labels")])),
        ),
        devices=["cpu"],
        tag="feature_table",
    ).build()
    from dbx.datatables import DatatablePartition
    split = DatatablePartition(datalake=url, tag="split", spec=dict(
        datapoint_table=featuretable, fractions=[0.5, 0.5], partition_slice='features', balance='tabs')).build()
    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            fit_table=split.fold(0).build(),
            eval_table=split.fold(1).build(),
            collator=Datacollator(spec=dict(signals=[("features", "final")], labels=[("labels", "labels")])),
        ),
        tag="log_probe",
    )
    return sampletable, probe


class TestAProbeChecksWhatItWillRead:

    def test_before_any_worker_starts(self, tmp_path):
        sampletable, probe = _probe(str(tmp_path))
        sampletable.UNSAFE_clear_blocks('labels', indices=[1], OVERRIDE=True)
        with pytest.raises(InvalidBlocksError, match=r"(?s)reading \['labels'\] from its tabs.*1 of 2 blocks"):
            probe.build()
        assert not probe.valid()

    def test_a_read_that_misses_says_why(self, tmp_path):
        # Removed behind the stack's back, so what vouches for its tabs cannot know:
        # the read fails -- and says it is the tab that is not valid, and why.
        sampletable, probe = _probe(str(tmp_path))
        tab = sampletable.tab(1)
        shutil.rmtree(tab.path('labels'))
        with pytest.raises(FileNotFoundError, match=r"cannot read slice 'labels': this tab is not valid"):
            probe.build()


class Failing(Datablock):
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = "'x'"


def unknown_fold(name):
    raise ValueError(f"Unknown fold: {name}")


def test_a_field_that_fails_to_evaluate_says_which(tmp_path):
    # Raised by construction, which computes the identity -- in the traceback that
    # prompted this, from inside a log line's repr() of VAR.
    with pytest.raises(ValueError, match="Unknown fold: X") as info:
        Failing(datalake=str(tmp_path), spec={'label': "@test_fail_early.unknown_fold('X')"})
    assert any("while evaluating Failing.VAR.label = \"@test_fail_early.unknown_fold('X')\"" in n
               for n in getattr(info.value, '__notes__', []))
