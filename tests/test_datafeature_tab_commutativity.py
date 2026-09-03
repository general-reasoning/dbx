"""A DatapointTab reached two ways from a DatafeatureTable must be one block.

Every test here is `@pytest.mark.pinned`: each states an invariant rather than
the shape the code currently has. If one fails, the code is wrong -- changing
the test to agree with the new behaviour would erase the only record that the
invariant was ever meant to hold.

A DatafeatureTable holds the DatapointTable it was built over, and each of its
DatafeatureTabs holds the DatapointTab it was built over. So the same upstream
tab is reachable by two routes::

    table.var.datapoint_table.tab(i)      # down the table, then to the tab
    table.tab(i).var.datapoint_tab        # to the tab, then to what it holds

They are separately constructed objects, and nothing makes them agree except
that both are built from the same configuration. If the two ever diverge, a
feature block reads its inputs from one place and the table believes they came
from another -- silently, because both are valid blocks.

`DatafeatureTable.validate_tab` checks the signature at build time. These tests
pin the rest of the chain, because signature parity does not imply identity
parity: ``type()`` adds the version and the topic list, and ``hash`` is
``sha256(type())``. Two tabs can agree on their spec and still address
different storage.
"""
import pytest
import numpy as np
import torch.nn as nn
from dataclasses import dataclass

from dbx import (
    SLICETOPIC,
    DatapointTab,
    DatapointTable,
    DatamodelEvaluatorFactory,
    DatafeatureTable,
)
from dbx.datafeatures import Datacollator


#: A source path assembled at read time from the environment -- the shape the
#: real pipelines use, and the reason these tests exist.
def tab_specline(idx):
    return f"$os.path.join(dbx.getenv('COMMUTATIVITY_LAKE'), 'tab{idx}.tfrecords')"


def sample_collator(**spec):
    return Datacollator(spec=dict(
        signals=[("samples", "samples")],
        labels=[("labels", "labels")],
        **spec,
    ))


class DummySampleTab(DatapointTab):
    TOPICS = {"samples": SLICETOPIC, "labels": SLICETOPIC}

    @dataclass
    class VAR(DatapointTab.VAR):
        n_samples: int = 10
        #: Carries a SPECLINE in the real pipelines -- a source path built from
        #: the environment. var.source is the resolved value and spec['source']
        #: the unexpanded text, so a route that rebuilds from one rather than
        #: the other changes the block's identity without changing the block.
        source: str = None

    def __build__(self):
        specs = {
            "samples": {"samples": "ndarray:float32"},
            "labels": {"labels": "int64"},
        }
        with self.slice_writers(specs) as writers:
            for i in range(self.var.n_samples):
                writers["samples"].write({"samples": np.arange(4, dtype=np.float32) + i})
                writers["labels"].write({"labels": np.int64(i % 2)})
        return self


class DummySampleTable(DatapointTable):
    TAB = DummySampleTab

    @dataclass
    class VAR(DatapointTable.VAR):
        samples_per_tab: int = 10

    @property
    def n_tabs(self):
        return 2

    def __tab__(self, idx: int) -> DummySampleTab:
        # The tabs differ in SPEC, not only in tag. A tag is not part of the
        # signature, so tabs differing only by tag share one hash -- and a
        # crossed index would then go unnoticed by a parity test.
        return self.TAB(url=self.url,
                        spec=dict(n_samples=self.var.samples_per_tab + idx,
                                  source=tab_specline(idx)),
                        tag=f"tab_{idx}")


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


class DummyModelEvaluatorFactory(DatamodelEvaluatorFactory):
    @property
    def model(self):
        return DummyModel()


@pytest.fixture(autouse=True)
def lake(monkeypatch):
    """What the tabs' speclines resolve against."""
    monkeypatch.setenv('COMMUTATIVITY_LAKE', '/lake')


@pytest.fixture
def featuretable(tmp_path):
    url = str(tmp_path)
    return DatafeatureTable(
        url=url,
        spec=dict(
            datapoint_table=DummySampleTable(url=url,
                                             spec=dict(samples_per_tab=5),
                                             tag="sample_table"),
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=sample_collator(),
        ),
        devices=["cpu"],
        tag="feature_table",
    )


def _both_ways(featuretable, idx=0):
    """The same upstream tab, reached down each route."""
    return (featuretable.var.datapoint_table.tab(idx),
            featuretable.tab(idx).var.datapoint_tab)


@pytest.mark.pinned
class TestIdentityCommutes:

    def test_signature_agrees(self, featuretable):
        via_table, via_tab = _both_ways(featuretable)
        assert via_table.signature() == via_tab.signature()

    def test_type_agrees(self, featuretable):
        """The stronger of the two: type() carries the version and topics that
        signature() does not, and it is what hash is computed from."""
        via_table, via_tab = _both_ways(featuretable)
        assert via_table.type() == via_tab.type()

    def test_hash_agrees(self, featuretable):
        via_table, via_tab = _both_ways(featuretable)
        assert via_table.hash == via_tab.hash

    def test_key_and_path_agree(self, featuretable):
        """What identity is FOR: the two must read and write the same place."""
        via_table, via_tab = _both_ways(featuretable)
        assert via_table.key == via_tab.key
        assert via_table.anchorkeypath == via_tab.anchorkeypath

    def test_every_tab_not_only_the_first(self, featuretable):
        for idx in range(featuretable.var.datapoint_table.n_tabs):
            via_table, via_tab = _both_ways(featuretable, idx)
            assert via_table.type() == via_tab.type(), f"tab {idx}"

    def test_distinct_tabs_stay_distinct(self, featuretable):
        """Parity must not come from every tab collapsing onto one identity."""
        first, second = _both_ways(featuretable, 0)[0], _both_ways(featuretable, 1)[0]
        assert first.hash != second.hash
        assert first.key != second.key
        assert first.anchorkeypath != second.anchorkeypath


@pytest.mark.pinned
class TestSessionIsNotIdentity:

    def test_session_may_differ_without_disturbing_identity(self, featuretable):
        """Which run reached a block is not part of what the block IS.

        The two routes form their tabs independently, so they carry different
        sessions -- and must still be the same block.
        """
        via_table, via_tab = _both_ways(featuretable)
        assert via_table.session != via_tab.session
        assert via_table.hash == via_tab.hash


@pytest.mark.pinned
class TestSpeclinesCommute:
    """The case most likely to break: a spec value that is not its own value.

    ``var.source`` is the RESOLVED path and ``spec['source']`` the unexpanded
    text. A route that rebuilds a tab from the resolved value rather than the
    specline produces a block that is configured identically and identified
    differently -- and every check that compares configuration would pass.
    """

    def test_the_tab_really_carries_one(self, featuretable):
        via_table, via_tab = _both_ways(featuretable)
        for tab in (via_table, via_tab):
            assert tab.spec['source'].startswith('$'), "the fixture lost its specline"
            assert tab.var.source == '/lake/tab0.tfrecords', "specline did not resolve"

    def test_identity_commutes_across_the_specline(self, featuretable):
        via_table, via_tab = _both_ways(featuretable)
        assert via_table.signature() == via_tab.signature()
        assert via_table.type() == via_tab.type()
        assert via_table.hash == via_tab.hash

    def test_the_specline_is_what_lands_in_the_identity(self, featuretable):
        """Unexpanded, on both routes.

        This is what makes a signature relocatable: the same configuration
        keeps one identity across environments that resolve it differently.
        Rendering the resolved path instead would pin the identity to whatever
        the environment happened to say at build time.
        """
        for tab in _both_ways(featuretable):
            assert "dbx.getenv('COMMUTATIVITY_LAKE')" in tab.signature()
            assert '/lake/tab0.tfrecords' not in tab.signature()

    def test_resolving_it_would_change_the_block(self, featuretable, tmp_path):
        """The guard that gives the tests above their teeth.

        If a resolved source and a specline resolving TO it were the same
        block, none of this could detect a route that resolved too early.
        """
        via_table, _ = _both_ways(featuretable)
        resolved = DummySampleTab(url=str(tmp_path),
                                  spec=dict(n_samples=via_table.var.n_samples,
                                            source='/lake/tab0.tfrecords'),
                                  tag='tab_0')
        assert resolved.var.source == via_table.var.source
        assert resolved.hash != via_table.hash
