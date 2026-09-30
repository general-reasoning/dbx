"""
Failing loudly, clearly and early: at the cause, not far from it.

Stale proxies vouch for nothing; a block says why it is not valid; a probe
refuses, before any worker starts, to read tabs that are not valid; and a
VAR field that fails to evaluate says which field of which block it was.
"""
import os
import shutil
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_block_specializations import grown, nomarkers, rekeyed  # noqa: E402,F401
from test_probes import DummyModelEvaluatorFactory, DummySampleTable  # noqa: E402
from test_specializations import v1table  # noqa: E402

from dbx import FeatureAffineLogisticProbe  # noqa: E402
from dbx.datablocks import Datablock, InvalidBlocksError  # noqa: E402
from dbx.datafeatures import Datacollator  # noqa: E402
from dbx import DatafeatureTable  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class TestStaleProxies:

    def test_markers_of_tabs_that_moved_vouch_for_nothing(self, grown):
        # Built with markers; then every tab re-keyed. The markers name the old tabs.
        table = v1table(grown, spec={'n': 3})
        assert table._built_tab_set_() == set()
        assert not table.valid_tabs(parallelization='inline').any()

    def test_a_record_under_other_identities_vouches_for_nothing(self, rekeyed):
        # Built, and so vouched for, before the tabs were re-keyed.
        table = nomarkers(rekeyed)
        assert os.path.exists(table._blocks_fingerprint_path_())
        assert not table._blocks_vouched_()

    def test_clearing_blocks_forgets_the_record(self, rekeyed):
        nomarkers(rekeyed).build_blocks()
        table = nomarkers(rekeyed)
        assert table._blocks_vouched_()
        table.UNSAFE_clear_blocks(indices=[1], clear_done=False, OVERRIDE=True)
        assert not nomarkers(rekeyed)._blocks_vouched_()
        assert nomarkers(rekeyed).valid_tabs(parallelization='inline').tolist() == [True, False, True]


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
    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(signals=[("samples", "samples")], labels=[("labels", "labels")])),
        ),
        devices=["cpu"],
        tag="feature_table",
    ).build()
    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(signals=[("features", "final")], labels=[("labels", "labels")])),
            training_fraction=0.8,
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
