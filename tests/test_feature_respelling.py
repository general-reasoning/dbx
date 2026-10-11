"""
The feature blocks were respelled with the markers; what was built before is rescued.

`Featuretablet` declared its slice with the SLICETOPIC sentinel, and `Featuretable`
inherited `Datatable`'s sentinel TOPICS -- which also added the TABLET's slices to
the table's identity.
Respelled, all four are new identities, and each declares a Specialization that
reconstructs the old one. These tests build under the OLD spelling -- the class
respelled back in place, since a tablet's anchor is its fqcn and a differently
named class would build somewhere else -- and then read it under the new one.
"""
import os
import sys

import pytest

pytest.importorskip("torch")

from dbx import DIRTOPIC, SLICETOPIC
from dbx.featuretables import (
    BipolarFeaturetablet, BipolarFeaturetable, Featuretablet, Featuretable)

sys.path.insert(0, os.path.dirname(__file__))
from test_featuretablet import (  # noqa: E402
    DummyModelEvaluatorFactory, DummySampleTablet, DummySampleTable, sample_collator)


class SeededEvaluatorFactory(DummyModelEvaluatorFactory):
    """The dummy model, the same weights every time it is made.

    One hash is one computation: a specialization resolves to ANY build of the
    narrower hash, whatever its tag. A randomly initialised model breaks that
    premise -- two tablets of one hash would hold different features -- so the
    model here is what a real, trained one is: the same whenever it is loaded.
    """

    @property
    def model(self):
        import torch
        from test_featuretablet import DummyModel
        torch.manual_seed(0)
        return DummyModel()


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


def _the_old_name(monkeypatch):
    """Featuretablet and Featuretable as they were before feature_namemap was renamed feature_for_column.

    And before their upstream field was renamed ``upstream``: the code reads
    ``var.upstream``, so the old VARs answer it under the old field.
    """
    from dataclasses import dataclass
    from dbx import Datablock, Datacollator
    from dbx.backbones import ModelEvaluatorBuilder
    from dbx.datatables import Datatablet, Datatable
    from dbx.featuretables import feature_map

    @dataclass
    class TabletVAR(Datablock.VAR):
        datapoint_tab: Datatablet
        evaluator_factory: ModelEvaluatorBuilder
        collator: Datacollator
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26

        @property
        def upstream(self):
            return self.datapoint_tab

    @dataclass
    class TableVAR(Datablock.VAR):
        datapoint_table: Datatable
        evaluator_factory: ModelEvaluatorBuilder
        collator: Datacollator
        feature_namemap: dict | None = None
        shard_size_limit_bytes: int = 1 << 26

        @property
        def upstream(self):
            return self.datapoint_table

    for cls, VAR in ((Featuretablet, TabletVAR), (Featuretable, TableVAR)):
        monkeypatch.setattr(cls, 'VAR', VAR)
        monkeypatch.setattr(cls, 'feature_for_column', property(
            lambda self: feature_map(self.var.feature_namemap, self.var.evaluator_factory)))
        monkeypatch.setattr(cls, 'SPECIALIZATIONS', [])
    # Everything hands them the fields under their new names.
    for cls, upstream in ((Featuretablet, 'datapoint_tab'), (Featuretable, 'datapoint_table')):
        old_name = {'feature_for_column': 'feature_namemap', 'upstream': upstream}

        def old_init(self, *args, spec=None, _init=cls.__init__, _old_name=old_name, **kwargs):
            if isinstance(spec, dict):
                spec = {_old_name.get(k, k): v for k, v in spec.items()}
            _init(self, *args, spec=spec, **kwargs)

        monkeypatch.setattr(cls, '__init__', old_init)


def _the_old_spelling(monkeypatch):
    """Every feature class as it was declared before the respelling -- and the rename."""
    _the_old_name(monkeypatch)
    old_table_topics = {'tab_paths': DIRTOPIC, 'done': 'done'}
    monkeypatch.setattr(Featuretablet, 'TOPICS', {'features': SLICETOPIC})
    for cls in (Featuretablet, Featuretable):
        monkeypatch.setattr(cls, 'SPECIALIZATIONS', [])
    monkeypatch.setattr(Featuretable, 'TOPICS', old_table_topics)
    # Featuretablet declares its columns per instance; the old one declared none.
    post_init = Featuretablet.__post_init__

    def sentinel_post_init(self):
        post_init(self)
        self.TOPICS = {'features': SLICETOPIC}

    monkeypatch.setattr(Featuretablet, '__post_init__', sentinel_post_init)
    # ... and the old builds handed the writers the columns nothing declared.
    old_columns = {
        Featuretablet: lambda self: {'features': {c: 'ndarray:float32' for c in self.feature_for_column}},
    }
    for cls, columns in old_columns.items():
        def slice_writers(self, slices=None, *, _writers=cls.slice_writers, _columns=columns, **kwargs):
            return _writers(self, _columns(self) if slices is None else slices, **kwargs)
        monkeypatch.setattr(cls, 'slice_writers', slice_writers)


def _features(url, evaluator):
    sampletablet = DummySampleTablet(datalake=url, tag='samples')
    return Featuretablet(datalake=url, tag='features', device='cpu', spec=dict(
        upstream=sampletablet, evaluator_factory=evaluator, collator=sample_collator()))


def test_the_new_identities_reconstruct_the_old(tmp_path, monkeypatch):
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    new = _features(url, ef)
    with monkeypatch.context() as m:
        _the_old_spelling(m)
        old = _features(url, ef)
        old_hash = old.hash
    assert new.hash != old_hash, "the respelling is a new identity"
    assert old_hash in new.specialization_hashes()


    """topics=SAME: the renames alone, under the columns this very tablet declares."""
    """topics=SAME: the rename alone, under the columns this very tablet declares."""
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    DummySampleTablet(datalake=url, tag='samples').build()
    with monkeypatch.context() as m:
        _the_old_name(m)
        old = _features(url, ef).build()
        old_hash = old.hash
    tablet = _features(url, ef)
    assert tablet.hash != old_hash
    # Both renames at once -- datapoint_tab and feature_namemap -- the second declared past.
    assert tablet.specialization_hashes()[1] == old_hash
    tablet.build()                                 # adopts: nothing left to build
    assert tablet.redirected_topics() == ['features']


def test_a_feature_tablet_built_under_the_sentinel_is_read_under_the_marker(tmp_path, monkeypatch):
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    DummySampleTablet(datalake=url, tag='samples').build()
    with monkeypatch.context() as m:
        _the_old_spelling(m)
        old = _features(url, ef).build()
        old_final = old.data(('features', 'final'))['features']['final']

    tablet = _features(url, ef)
    assert tablet.redirected_topics() == [], "constructing it adopts nothing"
    tablet.build()                                 # adopts: nothing left to build
    assert tablet.redirected_topics() == ['features']
    assert tablet.valid()
    assert (tablet.data(('features', 'final'))['features']['final'] == old_final).all()
    assert tablet.declared_columns('features') == {'final': 'ndarray:float32'}


def test_the_bipolar_blocks_reach_nothing_built_before_version_2():
    """Version 1 thresholded each tablet against its OWN median, which erases what
    sets one tablet apart from another; nothing is to adopt it. Their pasts are the
    field rename alone -- and, since 2026-10-10, the class's old name, BipolarFeaturetab."""
    from dbx.datablocks import ABSENT, SAME
    for cls in (BipolarFeaturetablet, BipolarFeaturetable):
        assert all(sp.spec == {} and sp.topics is SAME and sp.version is ABSENT
                   and set(sp.redirect_vars or ()) <= {'upstream'} for sp in cls.SPECIALIZATIONS)
        assert [sp for sp in cls.SPECIALIZATIONS if sp.redirect_vars]


def test_a_feature_table_built_under_the_sentinel_is_read_under_the_marker(tmp_path, monkeypatch):
    url, ef = str(tmp_path), SeededEvaluatorFactory(spec=dict(capture_final=True))
    samples = DummySampleTable(datalake=url, spec=dict(samples_per_tablet=5), tag='samples')
    samples.build()

    def table():
        return Featuretable(datalake=url, tag='features', devices=['cpu'], spec=dict(
            upstream=samples, evaluator_factory=ef, collator=sample_collator()))

    with monkeypatch.context() as m:
        _the_old_spelling(m)
        old = table().build()
        old_final = old.data(('features', 'final'), concat=True)['features']['final']

    new = table()
    assert new.hash != old.hash
    assert not new.valid(), "constructing it adopts nothing"
    # Adopts its own old build -- and, first, each tablet's: the tablets moved too.
    new.build()
    assert new.valid()
    assert all(new.tablet(i).redirected_topics() == ['features'] for i in range(new.n_tablets)), \
        "adopted, not rebuilt"
    got = new.data(('features', 'final'), concat=True)['features']['final']
    assert got.shape == old_final.shape == (10, 8)
    assert (got == old_final).all()
