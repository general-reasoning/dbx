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
    """The partition from before method/seed/groupby/stratifyby/balance -- and before its topic was
    respelled -- is reached by a partition built with method='largest_first', the one it computed."""
    from dataclasses import dataclass
    from dbx.datablocks import Datablock
    from dbx.datatables import Datatable, DatatablePartition
    from test_build_journal_reads import Table
    table = Table(datalake=str(tmp_path / 't'), spec={'n': 4})

    def partition(field='datatable', **spec):
        return DatatablePartition(datalake=str(tmp_path / 'p'), spec=dict(
            {field: table}, fractions=[0.5, 0.5], partition_slice=0, **spec))

    @dataclass
    class OldVAR(Datablock.VAR):
        datapoint_table: Datatable
        fractions: list
        partition_slice: int | str

    with monkeypatch.context() as m:
        m.setattr(DatatablePartition, 'VAR', OldVAR)
        m.setattr(DatatablePartition, 'TOPICS', {'tabs': 'tabs.json'})
        m.setattr(DatatablePartition, 'SPECIALIZATIONS', [])
        m.setattr(DatatablePartition, '__post_init__', Datablock.__post_init__)
        old_hash = partition('datapoint_table').hash
    assert old_hash in partition(method='largest_first').specialization_hashes()
    assert old_hash != partition().hash, "method='random' is another computation"


def _feature_table(url):
    pytest.importorskip("torch")
    from dbx import Datacollator, Featuretable
    from test_featuretab import DummyModelEvaluatorFactory, DummySampleTable
    samples = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5), tag='samples')
    return Featuretable(datalake=url, tag='features', devices=['cpu'], spec=dict(
        upstream=samples,
        evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
        collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]}))))


def test_the_affine_logistic_probe_reaches_nothing_built_before():
    """Version 3 fits one table and scores another; the ones before split one table inside --
    another computation, which no specialization reaches."""
    from dbx.probes import FeatureAffineLogisticProbe
    assert not FeatureAffineLogisticProbe.SPECIALIZATIONS


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
        feature_table=table, collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("samples", "samples")]}))))
    # The per-feature DATADICT, statistics spelled out, under each column's path.
    assert ("DATADICT('stats.npz', mean='ndarray', std='ndarray', median='ndarray', "
            "min='ndarray', max='ndarray')") in probe.typestr()
    assert probe.hash == hashlib.sha256(probe.typestr().encode()).hexdigest()


def test_a_table_reaches_its_tabs_past_not_its_own():
    """A table's own topics are markers over its tabs, written again in a moment; its tabs are
    what is worth reaching, and each is, by its TAB's specializations. A table declares only
    its VAR renames -- what a block holding it, a partition say, renders it as."""
    from dbx.datablocks import ABSENT, SAME
    from dbx.datatables import Datatable
    from dbx.featuretables import BipolarFeaturetable, Featuretable
    assert Datatable.SPECIALIZATIONS == []
    for cls in (Featuretable, BipolarFeaturetable):
        assert cls.SPECIALIZATIONS and all(sp.spec == {} and sp.topics is SAME and sp.version is ABSENT
                                           and sp.redirect_vars for sp in cls.SPECIALIZATIONS)
    assert Featuretable.TOPICS is Datatable.TOPICS
    assert not hasattr(Datatable, 'TAB_PATHS_TOPICS') and not hasattr(Datatable, 'TAB_PATHS_SPECIALIZATIONS')


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
