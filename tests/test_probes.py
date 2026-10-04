"""test_probes.py — Unit tests for FeatureAffineLogisticProbe and FeatureStatsProbe."""

from dataclasses import dataclass
import numpy as np
import pytest
import torch
import torch.nn as nn

import dbx
from dbx.datatables import Datacollator
from dbx import (
    DATADIR,
    DIRTOPIC,
    SLICETOPIC,
    Featuretab,
    Featuretable,
    Datatab,
    Datatable,
    ModelEvaluator,
    ModelEvaluatorBuilder,
    FeatureAffineLogisticProbe,
    FeatureStatsProbe,
    normalize_features,
)


class DummySampleTab(Datatab):
    TOPICS = {"samples": SLICETOPIC, "labels": SLICETOPIC}

    @dataclass
    class VAR(Datatab.VAR):
        samples_per_tab: int = 5

    def __build__(self):
        samples_per_tab = self.var.samples_per_tab
        spec = {
            "samples": {"samples": "ndarray:float32"},
            "labels": {"labels": "int64"},
        }
        with self.slice_writers(spec) as writers:
            for i in range(samples_per_tab):
                x = np.random.randn(4).astype(np.float32)
                y = int(i % 2)
                writers["samples"].write({"samples": x})
                writers["labels"].write({"labels": y})
        return self


class DummySampleTable(Datatable):
    TAB = DummySampleTab

    #: This table roots its tabs inside itself -- see __tab__ -- so it declares
    #: the directory to root them in. The base no longer does.
    TOPICS = {'tabs': DATADIR, **Datatable.TOPICS}

    @dataclass
    class VAR(Datatable.VAR):
        samples_per_tab: int = 5

    @property
    def n_tabs(self) -> int:
        return 2

    def __tab__(self, idx: int, tag=None) -> DummySampleTab:
        return self.TAB(
            datalake=self.path('tabs'),
            spec=dict(samples_per_tab=self.var.samples_per_tab),
            tag=tag or f"tab_{idx}",
        )

    def __split__(self, *args, **kwargs):
        return [self.TabMaker(idx) for idx in range(2)], dict(build=True)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


class DummyModelEvaluatorFactory(ModelEvaluatorBuilder):
    @property
    def model(self):
        return DummyModel()



def _fit_eval_(url, featuretable, tag='split'):
    """Two folds of the feature table -- one tab each -- as a probe's fit_table and eval_table."""
    from dbx.datatables import DatatablePartition
    split = DatatablePartition(datalake=url, tag=tag, spec=dict(
        datatable=featuretable, fractions=[0.5, 0.5], partition_slice='features', balance='tabs')).build()
    return dict(fit_table=split.fold(0).build(), eval_table=split.fold(1).build())


def test_normalize_features():
    x_numpy = np.array([[3.0, 4.0], [-1.0, 1.0]])
    x_torch = torch.tensor(x_numpy, dtype=torch.float32)

    # L2 norm
    l2_np = normalize_features(x_numpy, "l2")
    l2_th = normalize_features(x_torch, "l2")
    assert np.allclose(np.linalg.norm(l2_np, axis=1), 1.0)
    assert np.allclose(l2_np, l2_th.numpy())

    # Corner L1/L2 (sign)
    sgn_np = normalize_features(x_numpy, "corner-l1")
    sgn_th = normalize_features(x_torch, "corner-l1")
    assert np.array_equal(sgn_np, np.array([[1.0, 1.0], [-1.0, 1.0]]))
    assert np.array_equal(sgn_np, sgn_th.numpy())


def test_feature_affine_logistic_probe(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="feature_table",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            **_fit_eval_(url, featuretable),
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("labels", "labels")]}, recursive=True)),
        ),
        tag="log_probe",
    ).build()

    assert probe.valid()
    # Fitted on one tab, scored on the other: 5 rows each.
    assert probe.read('fit', 'features').shape == (5, 8)
    assert probe.read(['eval', 'features']).shape == (5, 8)
    assert len(probe.read('fit', 'labels')) == 5
    assert set(probe.read('eval')) == {'labels', 'features'}
    assert probe.read('coef').shape[1] == 8
    assert isinstance(probe.read('evaluation_report'), str)

    # The column layout is stored, so a coefficient can be attributed to the
    # pair it belongs to -- the whole point of pinning the signal order.
    assert probe.read('columns') == [('features', 'final', 8)]
    assert probe.feature_columns()[0] == ('features', 'final', 0)
    assert len(probe.feature_columns()) == probe.read('coef').shape[1]

    asph = probe.asphericity()
    assert isinstance(asph, dict)
    assert len(asph) > 0


def test_affine_logistic_probe_concatenates_several_signals(tmp_path):
    """Several signal pairs are laid end to end into one vector per sample,
    in declaration order, and the layout records the widths."""
    url = str(tmp_path)

    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5),
                                   tag="sample_table_multi").build()
    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="feature_table_multi",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            **_fit_eval_(url, featuretable),
            # 'final' is 8 wide, the upstream 'samples' column is 4.
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final"), ("samples", "samples")], 'labels': [("labels", "labels")]}, recursive=True)),
        ),
        tag="log_probe_multi",
    ).build()

    assert probe.read('columns') == [('features', 'final', 8), ('samples', 'samples', 4)]
    assert probe.read('fit', 'features').shape == (5, 12)
    assert probe.read('coef').shape[1] == 12
    # Column 8 is the first of the second pair.
    assert probe.feature_columns()[8] == ('samples', 'samples', 0)


def test_affine_logistic_probe_refuses_a_missing_signal_column(tmp_path):
    """No silent fallback to an arbitrary column."""
    url = str(tmp_path)
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5),
                                   tag="sample_table_bad").build()
    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="feature_table_bad",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            **_fit_eval_(url, featuretable),
            collator=Datacollator(spec=dict(columns={'signals': [("features", "nope")], 'labels': [("labels", "labels")]}, recursive=True)),
        ),
        tag="log_probe_bad",
    )
    with pytest.raises(KeyError, match="has no column 'nope'"):
        probe.build()


def test_feature_stats_probe(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_stats",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="feature_table_stats",
    ).build()

    stats_probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("samples", "samples")]}, recursive=True)),
        ),
        tag="stats_probe",
    ).build()

    assert stats_probe.valid()

    # Each column lives under its own path, the pair's names one level each,
    # exactly as dataset()/data() address the data being described; the file
    # there holds every statistic, as PER_FEATURE_DATADICT declares.
    stats = stats_probe.read('stats', 'features', 'final')
    assert set(stats) == set(FeatureStatsProbe.PER_FEATURE_DATADICT.schema) \
        == {'mean', 'std', 'median', 'min', 'max'}
    assert all(v.shape == (8,) for v in stats.values())
    assert stats_probe.read(('stats', 'features', 'final'))['median'].shape == (8,)

    every = stats_probe.read('stats')
    assert set(every) == {('features', 'final'), ('samples', 'samples')}
    assert every[('samples', 'samples')]['mean'].shape == (4,)

    for name in ('mean', 'std', 'median', 'min', 'max'):
        assert stats_probe.stat(name, ('features', 'final')).shape == (8,)
    np.testing.assert_array_equal(stats_probe.stat('median', ('features', 'final')), stats['median'])
    assert stats_probe.stat('norm', ('features', 'final')).shape == (10,)

    # Per-tab counterparts stack over the 2 tabs.
    assert stats_probe.stat('mean', ('features', 'final'), per_tab=True).shape == (2, 8)
    assert stats_probe.stat('max', ('samples', 'samples'), per_tab=True).shape == (2, 4)
    assert set(stats_probe.read('tab_stats', 'features', 'final')) == {'mean', 'std', 'median', 'min', 'max'}

    assert stats_probe.count == 10
    assert stats_probe.columns == [('features', 'final'), ('samples', 'samples')]

    # TOPICS names the columns, so an unknown one is refused before any file is opened.
    with pytest.raises(KeyError, match='nope'):
        stats_probe.read('stats', 'features', 'nope')
    with pytest.raises(KeyError, match='nope'):
        stats_probe.stat('median', ('features', 'nope'))
    with pytest.raises(KeyError, match='no statistic'):
        stats_probe.stat('mode', ('features', 'final'))


def test_stats_probe_describes_every_declared_pair(tmp_path):
    """Not just the first signal: keying by pair leaves room for all of them,
    which is what the old per-tab 'describes the first pair only' warning was
    apologising for."""
    url = str(tmp_path)
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5),
                                   tag="sample_table_all").build()
    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="feature_table_all",
    ).build()

    probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final"), ("samples", "samples")], 'labels': [("labels", "labels")]}, recursive=True)),
        ),
        tag="stats_probe_all",
    ).build()

    assert set(probe.read('stats')) == {
        ('features', 'final'), ('samples', 'samples'), ('labels', 'labels'),
    }
    assert probe.stat('mean', ('labels', 'labels')).shape == (1,)


def test_feature_stats_probe_parallel(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_stats_par",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="feature_table_stats_par",
    ).build()

    stats_probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("samples", "samples")]}, recursive=True)),
        ),
        parallelization='multithreading',
        n_workers=2,
        work_stealing=True,
        tag="stats_probe_par",
    ).build()

    assert stats_probe.valid()
    assert stats_probe.stat('mean', ('features', 'final')).shape == (8,)
    assert stats_probe.stat('mean', ('features', 'final'), per_tab=True).shape == (2, 8)


def test_feature_affine_logistic_probe_parallel(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_log_par",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="feature_table_log_par",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            **_fit_eval_(url, featuretable),
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("labels", "labels")]}, recursive=True)),
        ),
        parallelization='multithreading',
        n_workers=2,
        work_stealing=True,
        tag="probe_log_par",
    ).build()

    assert probe.valid()
    assert probe.read('coef').shape[1] == 8


if __name__ == "__main__":
    import sys
    dbx.dataparts.gitwrkreposetup = lambda *a, **k: None
    sys.exit(pytest.main([__file__]))


@pytest.mark.parametrize('band_bytes', [8, 10**9])
def test_table_stats_by_band_equal_the_stats_of_the_concatenation(band_bytes, monkeypatch):
    """Banding is a memory decision, not a numerical one.

    The whole-table statistics used to be `column_stats` of every tab's rows
    concatenated -- a full copy of the table, and another inside np.median.
    Taken a band of columns at a time they must come out the same, whether
    the band is one column (8 bytes forces it) or the whole width.
    """
    import dbx.probes as probes
    monkeypatch.setattr(probes, 'TABLE_STATS_BAND_BYTES', band_bytes)
    rng = np.random.default_rng(0)
    tabs = [rng.normal(size=(n, 7)) for n in (5, 0, 11, 3)]
    results = [{'columns': {('s', 'c'): t}, 'stats': {('s', 'c'): probes.column_stats(t) if len(t) else
                                                     {'norm': np.zeros(0)}}} for t in tabs]

    class Probe:
        column_paths = [('s', 'c')]
        _table_stats_ = probes.FeatureStatsProbe._table_stats_

    got = Probe()._table_stats_(results)[('s', 'c')]
    want = probes.column_stats(np.concatenate(tabs, axis=0))
    assert set(got) == set(want)
    for name in want:
        np.testing.assert_allclose(got[name], want[name], rtol=1e-12, atol=0, err_msg=name)


class UnevenSampleTable(DummySampleTable):
    """Tabs of 3 and 7 samples: real slides do not have one size."""

    def __tab__(self, idx: int, tag=None) -> DummySampleTab:
        return self.TAB(
            datalake=self.path('tabs'),
            spec=dict(samples_per_tab=3 + 4 * idx),
            tag=tag or f"tab_{idx}",
        )


def test_feature_stats_probe_over_tabs_of_different_sizes(tmp_path):
    """``tab_norm`` is one value per ROW, so uneven tabs do not stack.

    The build died on exactly that -- ``np.stack`` over per-tab norms of 3 and
    7 -- after every tab's stats and the whole-table ones had been computed. A
    table whose tabs all have one size, as the fixtures above do, never
    reaches it.
    """
    url = str(tmp_path)
    sampletable = UnevenSampleTable(datalake=url, tag="uneven_samples").build()
    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="uneven_features",
    ).build()
    probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(feature_table=featuretable,
                  collator=Datacollator(spec=dict(columns={'signals': [("features", "final")]}))),
        tag="uneven_stats",
    ).build()

    assert probe.count == 10
    norms = probe.stat('norm', ('features', 'final'), per_tab=True)
    assert [len(n) for n in norms] == [3, 7]
    np.testing.assert_allclose(np.concatenate(norms), probe.stat('norm', ('features', 'final')))
    # The reductions still stack: one row per tab.
    assert probe.stat('mean', ('features', 'final'), per_tab=True).shape == (2, 8)


def test_tab_aggregation_refuses_a_tab_whose_rows_disagree(tmp_path):
    url = str(tmp_path)
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5), tag="agg_samples").build()
    featuretable = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag="agg_features",
    ).build()
    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            **_fit_eval_(url, featuretable),
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("labels", "labels")]}, recursive=True)),
            tab_aggregation='mean',
        ),
        tag="agg_probe",
    )
    # The dummy tabs label their rows 0, 1, 0, ...: one tab, two labels.
    with pytest.raises(ValueError, match="can carry only one label"):
        probe.build()


def _feature_table_(url, tag, n=4):
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=n), tag=f"{tag}_samples").build()
    return Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})),
        ),
        devices=["cpu"],
        tag=tag,
    ).build()


def test_a_probe_refuses_fit_and_eval_tables_that_share_a_tab(tmp_path):
    url = str(tmp_path)
    featuretable = _feature_table_(url, "shared_features")
    probe = FeatureAffineLogisticProbe(datalake=url, tag="shared_probe", spec=dict(
        fit_table=featuretable, eval_table=_fit_eval_(url, featuretable)['eval_table'],
        collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("labels", "labels")]}, recursive=True))))
    with pytest.raises(ValueError, match="share 1 tab"):
        probe.build()


def test_max_rows_per_tab_samples_each_tab_the_same_way_every_time(tmp_path):
    url = str(tmp_path)
    featuretable = _feature_table_(url, "rows_features", n=10)
    def probe(tag):
        return FeatureAffineLogisticProbe(datalake=url, tag=tag, spec=dict(
            **_fit_eval_(url, featuretable), max_rows_per_tab=4,
            collator=Datacollator(spec=dict(columns={'signals': [("features", "final")], 'labels': [("labels", "labels")]}, recursive=True)))).build()
    one = probe("rows_a").read('fit', 'features')
    assert one.shape == (4, 8)
    np.testing.assert_array_equal(one, probe("rows_b").read('fit', 'features'))
