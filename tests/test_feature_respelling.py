"""
The feature blocks were respelled with the markers; what was built before is rescued.

`Featuretab` declared its slice with the SLICETOPIC sentinel, and `Featuretable`
inherited `Datatable`'s sentinel TOPICS -- which also added the TAB's slices to
the table's identity.
Respelled, all four are new identities, and each declares a Specialization that
reconstructs the old one. These tests build under the OLD spelling -- the class
respelled back in place, since a tab's anchor is its fqcn and a differently
named class would build somewhere else -- and then read it under the new one.
"""
import os
import sys

import pytest

pytest.importorskip("torch")

from dbx import DIRTOPIC, SLICETOPIC
from dbx.featuretables import (
    BipolarFeaturetab, BipolarFeaturetable, Featuretab, Featuretable)

sys.path.insert(0, os.path.dirname(__file__))
from test_datafeaturetab import (  # noqa: E402
    DummyModelEvaluatorFactory, DummySampleTab, DummySampleTable, sample_collator)


class SeededEvaluatorFactory(DummyModelEvaluatorFactory):
    """The dummy model, the same weights every time it is made.

    One hash is one computation: a specialization resolves to ANY build of the
    narrower hash, whatever its tag. A randomly initialised model breaks that
    premise -- two tabs of one hash would hold different features -- so the
    model here is what a real, trained one is: the same whenever it is loaded.
    """

    @property
    def model(self):
        import torch
        from test_datafeaturetab import DummyModel
        torch.manual_seed(0)
        return DummyModel()


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


def _the_old_name(monkeypatch):
    """Featuretab and Featuretable as they were before feature_namemap was renamed feature_for_column."""
    from dataclasses import dataclass
    from dbx import Datablock, Datacollator
    from dbx.backbones import ModelEvaluatorBuilder
    from dbx.datatables import Datatab, Datatable
    from dbx.featuretables import feature_map

    @dataclass
    class TabVAR(Datablock.VAR):
        datapoint_tab: Datatab
        evaluator_factory: ModelEvaluatorBuilder
        collator: Datacollator
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26

    @dataclass
    class TableVAR(Datablock.VAR):
        datapoint_table: Datatable
        evaluator_factory: ModelEvaluatorBuilder
        collator: Datacollator
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26

    for cls, VAR in ((Featuretab, TabVAR), (Featuretable, TableVAR)):
        monkeypatch.setattr(cls, 'VAR', VAR)
        monkeypatch.setattr(cls, 'feature_for_column', property(
            lambda self: feature_map(self.var.feature_namemap, self.var.evaluator_factory)))
        monkeypatch.setattr(cls, 'SPECIALIZATIONS', [])
    # A table hands its tabs the field under its new name.
    init = Featuretab.__init__

    def old_init(self, *args, spec=None, **kwargs):
        if isinstance(spec, dict) and 'feature_for_column' in spec:
            spec = {('feature_namemap' if k == 'feature_for_column' else k): v for k, v in spec.items()}
        init(self, *args, spec=spec, **kwargs)

    monkeypatch.setattr(Featuretab, '__init__', old_init)


def _the_old_spelling(monkeypatch):
    """Every feature class as it was declared before the respelling -- and the rename."""
    _the_old_name(monkeypatch)
    old_table_topics = {'tab_paths': DIRTOPIC, 'done': 'done'}
    monkeypatch.setattr(Featuretab, 'TOPICS', {'features': SLICETOPIC})
    for cls in (Featuretab, Featuretable):
        monkeypatch.setattr(cls, 'SPECIALIZATIONS', [])
    monkeypatch.setattr(Featuretable, 'TOPICS', old_table_topics)
    # Featuretab declares its columns per instance; the old one declared none.
    post_init = Featuretab.__post_init__

    def sentinel_post_init(self):
        post_init(self)
        self.TOPICS = {'features': SLICETOPIC}

    monkeypatch.setattr(Featuretab, '__post_init__', sentinel_post_init)
    # ... and the old builds handed the writers the columns nothing declared.
    old_columns = {
        Featuretab: lambda self: {'features': {c: 'ndarray:float32' for c in self.feature_for_column}},
    }
    for cls, columns in old_columns.items():
        def slice_writers(self, slices=None, *, _writers=cls.slice_writers, _columns=columns, **kwargs):
            return _writers(self, _columns(self) if slices is None else slices, **kwargs)
        monkeypatch.setattr(cls, 'slice_writers', slice_writers)


def _features(url, evaluator):
    sampletab = DummySampleTab(datalake=url, tag='samples')
    return Featuretab(datalake=url, tag='features', device='cpu', spec=dict(
        datapoint_tab=sampletab, evaluator_factory=evaluator, collator=sample_collator()))


def test_the_new_identities_reconstruct_the_old(tmp_path, monkeypatch):
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    new = _features(url, ef)
    with monkeypatch.context() as m:
        _the_old_spelling(m)
        old = _features(url, ef)
        old_hash = old.hash
    assert new.hash != old_hash, "the respelling is a new identity"
    assert old_hash in new.specialization_hashes()


def test_a_feature_tab_built_before_the_rename_is_read_after_it(tmp_path, monkeypatch):
    """topics=SAME: the rename alone, under the columns this very tab declares."""
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    DummySampleTab(datalake=url, tag='samples').build()
    with monkeypatch.context() as m:
        _the_old_name(m)
        old = _features(url, ef).build()
        old_hash = old.hash
    tab = _features(url, ef)
    assert tab.hash != old_hash
    assert tab.specialization_hashes()[0] == old_hash
    tab.build()                                 # adopts: nothing left to build
    assert tab.redirected_topics() == ['features']


def test_a_feature_tab_built_under_the_sentinel_is_read_under_the_marker(tmp_path, monkeypatch):
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    DummySampleTab(datalake=url, tag='samples').build()
    with monkeypatch.context() as m:
        _the_old_spelling(m)
        old = _features(url, ef).build()
        old_final = old.data(('features', 'final'))['features']['final']

    tab = _features(url, ef)
    assert tab.redirected_topics() == [], "constructing it adopts nothing"
    tab.build()                                 # adopts: nothing left to build
    assert tab.redirected_topics() == ['features']
    assert tab.valid()
    assert (tab.data(('features', 'final'))['features']['final'] == old_final).all()
    assert tab.declared_columns('features') == {'final': 'ndarray:float32'}


def test_the_bipolar_blocks_reach_nothing_built_before():
    """Their builds thresholded each tab against its OWN median, which erases
    what sets one tab apart from another; nothing is to adopt them."""
    assert not BipolarFeaturetab.SPECIALIZATIONS
    assert not BipolarFeaturetable.SPECIALIZATIONS


def test_a_feature_table_built_under_the_sentinel_is_read_under_the_marker(tmp_path, monkeypatch):
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    samples = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5), tag='samples')
    samples.build()

    def table():
        return Featuretable(datalake=url, tag='features', devices=['cpu'], spec=dict(
            datapoint_table=samples, evaluator_factory=ef, collator=sample_collator()))

    with monkeypatch.context() as m:
        _the_old_spelling(m)
        old = table().build()
        old_final = old.data(('features', 'final'), concat=True)['features']['final']

    new = table()
    assert new.hash != old.hash
    assert not new.valid(), "constructing it adopts nothing"
    # Adopts its own old build -- and, first, each tab's: the tabs moved too.
    new.build()
    assert new.valid()
    assert all(new.tab(i).redirected_topics() == ['features'] for i in range(new.n_tabs)), \
        "adopted, not rebuilt"
    got = new.data(('features', 'final'), concat=True)['features']['final']
    assert got.shape == old_final.shape == (10, 8)
    assert (got == old_final).all()
