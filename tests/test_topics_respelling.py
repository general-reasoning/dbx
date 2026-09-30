"""
Every block respelled with the topic markers reconstructs the identity it had.

`DATAFILE('x')` names the same file as ``'x'``, `DATADICT` the same file with its
keys declared -- but they render differently, so each respelling is a new hash,
and each respelled class declares a Specialization back to its old spelling.
Here each class is put back into its old spelling IN PLACE -- a block's anchor
is its fqcn, so a stand-in class would be a different block -- and the hash that
gives is what the current class's specialization must reconstruct.
"""
import os
import sys

import pytest

from dbx.datablocks import DIR, DIRTOPIC

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


def _reconstructs(make, cls, old_topics, monkeypatch, *, per_instance=False):
    new = make()
    with monkeypatch.context() as m:
        m.setattr(cls, 'SPECIALIZATIONS', [])
        m.setattr(cls, 'TOPICS', old_topics(None) if per_instance else old_topics)
        if per_instance:
            post_init = cls.__post_init__

            def old_post_init(self):
                post_init(self)
                self.TOPICS = old_topics(self)
                self.SPECIALIZATIONS = []

            m.setattr(cls, '__post_init__', old_post_init)
        old = make()
        assert old.specialization_hashes() == []
        old_hash = old.hash
    assert new.hash != old_hash, "the respelling is a new identity"
    assert old_hash in new.specialization_hashes()


def test_weights(tmp_path, monkeypatch):
    pytest.importorskip("lightning")
    from dbx.stills import Weights
    _reconstructs(lambda: Weights(datalake=str(tmp_path), tag='w', spec=dict(ckpt=None)),
                  Weights, {'weights': 'weights.pt'}, monkeypatch)


def test_still(tmp_path, monkeypatch):
    pytest.importorskip("lightning")
    from dbx.stills import Still
    from test_stills import _toy
    _reconstructs(lambda: _toy(tmp_path), Still,
                  {'ckpts': DIR, 'logs': DIR, 'done': 'done'}, monkeypatch)


def test_a_partition(tmp_path, monkeypatch):
    from dbx.datatables import DatatablePartition
    from test_build_journal_reads import Table
    table = Table(datalake=str(tmp_path / 't'), spec={'n': 4})
    _reconstructs(lambda: DatatablePartition(datalake=str(tmp_path / 'p'), spec=dict(
        datapoint_table=table, fractions=[0.5, 0.5], partition_slice=0)),
        DatatablePartition, {'tabs': 'tabs.json'}, monkeypatch)


def _feature_table(url):
    pytest.importorskip("torch")
    from dbx import Datacollator, Featuretable
    from test_datafeaturetab import DummyModelEvaluatorFactory, DummySampleTable
    samples = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5), tag='samples')
    return Featuretable(datalake=url, tag='features', devices=['cpu'], spec=dict(
        datapoint_table=samples,
        evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
        collator=Datacollator(spec=dict(signals=[("samples", "samples")],
                                        labels=[("labels", "labels")]))))


def test_the_affine_logistic_probe(tmp_path, monkeypatch):
    from dbx import Datacollator
    from dbx.probes import FeatureAffineLogisticProbe
    table = _feature_table(str(tmp_path))
    _reconstructs(lambda: FeatureAffineLogisticProbe(datalake=str(tmp_path), spec=dict(
        feature_table=table, collator=Datacollator(spec=dict(
            signals=[("features", "final")], labels=[("labels", "labels")])))),
        FeatureAffineLogisticProbe, {
            'labels': 'labels.npz', 'features': 'features.npy', 'columns': 'columns.pkl',
            'evaluation_report': 'evaluation_report.pkl', 'coef': 'coef.npy',
            'intercept': 'intercept.npy', 'classes': 'classes.npz'}, monkeypatch)


def test_topics_computed_per_instance_are_in_the_hash(tmp_path):
    """TOPICS set in __post_init__ are identity like any others.

    The logger's name used to take the hash, and cache it, in __setstate__ --
    before __post_init__ ran -- so a block computing its TOPICS there had a
    `hash` that was not the sha256 of its own `typestr()`.
    """
    import hashlib
    from dbx.datablocks import DATAFILE, Datablock

    class PerInstance(Datablock):
        TOPICS = {'count': DATAFILE('count.npz')}

        def __post_init__(self):
            super().__post_init__()
            self.TOPICS = {**type(self).TOPICS, 'extra': DATAFILE('extra.npz')}

    block = PerInstance(datalake=str(tmp_path))
    assert 'topic:extra' in block.typestr()
    assert block.hash == hashlib.sha256(block.typestr().encode()).hexdigest()
    assert block.hash[:8] in block.log.name


def test_the_stats_probe_hashes_its_per_column_topics(tmp_path):
    import hashlib
    from dbx import Datacollator
    from dbx.probes import FeatureStatsProbe
    table = _feature_table(str(tmp_path))
    probe = FeatureStatsProbe(datalake=str(tmp_path), spec=dict(
        feature_table=table, collator=Datacollator(spec=dict(
            signals=[("features", "final")], labels=[("samples", "samples")]))))
    assert "DATADICT('stat.npz'" in probe.typestr()
    assert probe.hash == hashlib.sha256(probe.typestr().encode()).hexdigest()


def test_a_table_on_the_base_topics_built_before_the_respelling_is_found(tmp_path, monkeypatch):
    """A table on TAB_PATHS_TOPICS -- the base's, while they included tab_paths -- moved with their
    respelling; TAB_PATHS_SPECIALIZATIONS reach back."""
    from dbx.datatables import Datatable
    from test_datapointtable import LetterTable

    class TabPathsLetterTable(LetterTable):
        TOPICS = Datatable.TAB_PATHS_TOPICS
        SPECIALIZATIONS = Datatable.TAB_PATHS_SPECIALIZATIONS

    def table():
        return TabPathsLetterTable(datalake=str(tmp_path), spec=dict(n_tabs_=2, per_tab=2))

    from dbx.datablocks import Datastack
    with monkeypatch.context() as m:
        m.setattr(TabPathsLetterTable, 'TOPICS', {'tab_paths': DIRTOPIC, 'done': 'done'})
        m.setattr(TabPathsLetterTable, 'SPECIALIZATIONS', [])
        # ... and before a table's type named its TAB.
        m.setattr(Datastack, '_type_entries_', lambda self, specialization=None, **kw: {})
        old = table().build()
        old_hash = old.hash
    new = table()
    assert new.hash != old_hash
    assert old_hash in new.specialization_hashes()
    assert not new.valid(), "constructing it adopts nothing"
    new.build()                                 # adopts both topics: nothing left to build
    assert sorted(new.redirected_topics()) == ['done', 'tab_paths']
    assert new.valid()


def test_a_table_declaring_its_own_topics_does_not_inherit_the_bases_past(tmp_path):
    from dbx.datablocks import DATADIR, DATAFILE
    from dbx.datatables import Datatable

    class TabPaths(Datatable):
        TOPICS = Datatable.TAB_PATHS_TOPICS
        SPECIALIZATIONS = Datatable.TAB_PATHS_SPECIALIZATIONS

    class Own(TabPaths):
        TOPICS = {'tabs': DATADIR, 'tab_paths': DATADIR, 'done': DATAFILE('done')}

    class Declared(TabPaths):
        SPECIALIZATIONS = [*Datatable.TAB_PATHS_SPECIALIZATIONS]

    assert Own.SPECIALIZATIONS == [] and TabPaths.SPECIALIZATIONS
    assert Declared.SPECIALIZATIONS == Datatable.TAB_PATHS_SPECIALIZATIONS
    # The base's TOPICS are `done` alone, which nothing was built under before: no past to reach.
    assert Datatable.SPECIALIZATIONS == []


def test_leaving_out_the_bases_specializations_warns():
    from dbx.datablocks import Datablock
    from dbx.datatables import Datatable
    with pytest.warns(UserWarning, match=r"\[\*Datatable.TAB_PATHS_SPECIALIZATIONS"):
        class Forgot(Datatable):
            TOPICS = Datatable.TAB_PATHS_TOPICS
            SPECIALIZATIONS = [Datablock.Specialization(spec={}, topics={'done': 'done'})]


def test_a_legacy_spelling_warns_and_a_canonical_one_does_not():
    import warnings
    from dbx.datablocks import (DATADICT, DATADIR, DATAFILE, SYNTHETIC, Datablock,
                                LegacyTopicsWarning)
    from dbx.datatables import DATASLICE
    with pytest.warns(LegacyTopicsWarning, match=r"a, b, c/d, e"):
        class Old(Datablock):
            TOPICS = {'a': 'a.txt', 'b': DIR, 'c': {'d': DIRTOPIC}, 'e': ()}
    with warnings.catch_warnings():
        warnings.simplefilter('error', LegacyTopicsWarning)

        class Canonical(Datablock):
            TOPICS = {'f': DATAFILE('f'), 'g': DATADICT('g.json', k='int'), 'h': DATADIR,
                      'i': {'j': DATASLICE(x='int')}, 'k': SYNTHETIC}
