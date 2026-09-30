"""
The feature blocks were respelled with the markers; what was built before is rescued.

`Featuretab` and `BipolarFeaturetab` declared their slices with the SLICETOPIC
sentinel, and `Featuretable` / `BipolarFeaturetable` inherited `Datatable`'s
sentinel TOPICS -- which also added the TAB's slices to the table's identity.
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


def _the_old_spelling(monkeypatch):
    """Every feature class as it was declared before the respelling."""
    old_table_topics = {'tab_paths': DIRTOPIC, 'done': 'done'}
    monkeypatch.setattr(Featuretab, 'TOPICS', {'features': SLICETOPIC})
    monkeypatch.setattr(BipolarFeaturetab, 'TOPICS',
                        {'bipolar_features': SLICETOPIC, 'tab_bipolar_features': SLICETOPIC})
    for cls in (Featuretab, BipolarFeaturetab, Featuretable, BipolarFeaturetable):
        monkeypatch.setattr(cls, 'SPECIALIZATIONS', [])
    for cls in (Featuretable, BipolarFeaturetable):
        monkeypatch.setattr(cls, 'TOPICS', old_table_topics)
    # Featuretab declares its columns per instance; the old one declared none.
    post_init = Featuretab.__post_init__

    def sentinel_post_init(self):
        post_init(self)
        self.TOPICS = {'features': SLICETOPIC}

    monkeypatch.setattr(Featuretab, '__post_init__', sentinel_post_init)
    # ... and the old builds handed the writers the columns nothing declared.
    old_columns = {
        Featuretab: lambda self: {'features': {c: 'ndarray:float32' for c in self._feature_map}},
        BipolarFeaturetab: lambda self: {'bipolar_features': {'bipolar_features': 'ndarray:int8'},
                                         'tab_bipolar_features': {'tab_bipolar_features': 'ndarray:int8'}},
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
    assert new.specialization_hashes() == [old_hash]


def test_a_feature_tab_built_under_the_sentinel_is_read_under_the_marker(tmp_path, monkeypatch):
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    DummySampleTab(datalake=url, tag='samples').build()
    with monkeypatch.context() as m:
        _the_old_spelling(m)
        old = _features(url, ef).build()
        old_final = old.data(('features', 'final'))['features']['final']
        bipolar_old = BipolarFeaturetab(datalake=url, tag='bipolar', spec=dict(
            featuretab=old, feature='final', threshold=0.3)).build()
        old_bipolar = bipolar_old.data('bipolar_features')['bipolar_features']['bipolar_features']

    tab = _features(url, ef)
    assert tab.redirected_topics() == [], "constructing it adopts nothing"
    tab.build()                                 # adopts: nothing left to build
    assert tab.redirected_topics() == ['features']
    assert tab.valid()
    assert (tab.data(('features', 'final'))['features']['final'] == old_final).all()
    assert tab.declared_columns('features') == {'final': 'ndarray:float32'}

    bipolar = BipolarFeaturetab(datalake=url, tag='bipolar', spec=dict(
        featuretab=tab, feature='final', threshold=0.3)).build()
    assert sorted(bipolar.redirected_topics()) == ['bipolar_features', 'tab_bipolar_features']
    assert bipolar.valid()
    got = bipolar.data('bipolar_features')['bipolar_features']['bipolar_features']
    assert (got == old_bipolar).all()


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
    got = new.data(('features', 'final'), concat=True)['features']['final']
    assert got.shape == old_final.shape == (10, 8)
    assert (got == old_final).all()
