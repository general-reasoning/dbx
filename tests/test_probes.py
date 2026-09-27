"""test_probes.py — Unit tests for FeatureAffineLogisticProbe and FeatureStatsProbe."""

from dataclasses import dataclass
import numpy as np
import pytest
import torch
import torch.nn as nn

import dbx
from dbx.datafeatures import Datacollator
from dbx import (
    DATADIR,
    DIRTOPIC,
    SLICETOPIC,
    DatafeatureTab,
    DatafeatureTable,
    DatapointTab,
    DatapointTable,
    ModelEvaluator,
    ModelEvaluatorBuilder,
    FeatureAffineLogisticProbe,
    FeatureAffineLogisticProber,
    FeatureStatsProbe,
    normalize_features,
)


class DummySampleTab(DatapointTab):
    TOPICS = {"samples": SLICETOPIC, "labels": SLICETOPIC}

    @dataclass
    class VAR(DatapointTab.VAR):
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


class DummySampleTable(DatapointTable):
    TAB = DummySampleTab

    #: This table roots its tabs inside itself -- see __tab__ -- so it declares
    #: the directory to root them in. The base no longer does.
    TOPICS = {'tabs': DATADIR, **DatapointTable.TOPICS}

    @dataclass
    class VAR(DatapointTable.VAR):
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


def test_datafeature_affine_logistic_probe(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(
                signals=[("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        devices=["cpu"],
        tag="feature_table",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(
                signals=[("features", "final")],
                labels=[("labels", "labels")],
            )),
            training_fraction=0.8,
        ),
        tag="log_probe",
    ).build()

    assert probe.valid()
    assert probe.read('features').shape == (10, 8)
    assert probe.read(['features']).shape == (10, 8)
    assert len(probe.read('labels')) == 10
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
    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(
                signals=[("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        devices=["cpu"],
        tag="feature_table_multi",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            # 'final' is 8 wide, the upstream 'samples' column is 4.
            collator=Datacollator(spec=dict(
                signals=[("features", "final"), ("samples", "samples")],
                labels=[("labels", "labels")],
            )),
            training_fraction=0.8,
        ),
        tag="log_probe_multi",
    ).build()

    assert probe.read('columns') == [('features', 'final', 8), ('samples', 'samples', 4)]
    assert probe.read('features').shape == (10, 12)
    assert probe.read('coef').shape[1] == 12
    # Column 8 is the first of the second pair.
    assert probe.feature_columns()[8] == ('samples', 'samples', 0)


def test_affine_logistic_probe_refuses_a_missing_signal_column(tmp_path):
    """No silent fallback to an arbitrary column."""
    url = str(tmp_path)
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5),
                                   tag="sample_table_bad").build()
    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(
                signals=[("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        devices=["cpu"],
        tag="feature_table_bad",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(
                signals=[("features", "nope")],
                labels=[("labels", "labels")],
            )),
            training_fraction=0.8,
        ),
        tag="log_probe_bad",
    )
    with pytest.raises(KeyError, match="has no column 'nope'"):
        probe.build()


def test_datafeature_stats_probe(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_stats",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(
                signals=[("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        devices=["cpu"],
        tag="feature_table_stats",
    ).build()

    stats_probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(
                signals=[("features", "final")],
                labels=[("samples", "samples")],
            )),
        ),
        tag="stats_probe",
    ).build()

    assert stats_probe.valid()

    # A statistic is a topic; the columns live underneath it, keyed by pair
    # exactly as dataset()/data() key the data being described.
    means = stats_probe.read('mean')
    assert isinstance(means, dict)
    assert set(means) == {'features.final', 'samples.samples'}
    assert means['features.final'].shape == (8,)
    assert means['samples.samples'].shape == (4,)

    assert stats_probe.read('mean', 'features.final').shape == (8,)
    assert stats_probe.read(['mean', 'features.final']).shape == (8,)
    assert stats_probe.read(('mean', 'features.final')).shape == (8,)

    for name in ('std', 'median', 'min', 'max'):
        assert stats_probe.read(name, 'features.final').shape == (8,)
    assert stats_probe.read('norm', 'features.final').shape == (10,)

    # Per-tab counterparts stack over the 2 tabs.
    assert stats_probe.read('tab_mean', 'features.final').shape == (2, 8)
    assert stats_probe.read('tab_max', 'samples.samples').shape == (2, 4)

    assert stats_probe.count == 10
    assert stats_probe.columns == ['features.final', 'samples.samples']

    # TOPICS names the columns, so an unknown one is refused by topic-path
    # validation before any file is opened.
    with pytest.raises(KeyError, match='features.nope'):
        stats_probe.read('mean', 'features.nope')


def test_stats_probe_describes_every_declared_pair(tmp_path):
    """Not just the first signal: keying by pair leaves room for all of them,
    which is what the old per-tab 'describes the first pair only' warning was
    apologising for."""
    url = str(tmp_path)
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5),
                                   tag="sample_table_all").build()
    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(
                signals=[("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        devices=["cpu"],
        tag="feature_table_all",
    ).build()

    probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(
                signals=[("features", "final"), ("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        tag="stats_probe_all",
    ).build()

    assert set(probe.read('mean')) == {
        'features.final', 'samples.samples', 'labels.labels',
    }
    assert probe.read('mean', 'labels.labels').shape == (1,)


def test_datafeature_stats_probe_parallel(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_stats_par",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(
                signals=[("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        devices=["cpu"],
        tag="feature_table_stats_par",
    ).build()

    stats_probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(
                signals=[("features", "final")],
                labels=[("samples", "samples")],
            )),
        ),
        parallelization='multithreading',
        n_workers=2,
        work_stealing=True,
        tag="stats_probe_par",
    ).build()

    assert stats_probe.valid()
    assert stats_probe.read('mean', 'features.final').shape == (8,)
    assert stats_probe.read('tab_mean', 'features.final').shape == (2, 8)


def test_datafeature_affine_logistic_probe_parallel(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_log_par",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
            collator=Datacollator(spec=dict(
                signals=[("samples", "samples")],
                labels=[("labels", "labels")],
            )),
        ),
        devices=["cpu"],
        tag="feature_table_log_par",
    ).build()

    probe = FeatureAffineLogisticProbe(
        datalake=url,
        spec=dict(
            feature_table=featuretable,
            collator=Datacollator(spec=dict(
                signals=[("features", "final")],
                labels=[("labels", "labels")],
            )),
            training_fraction=0.8,
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
    results = [{'columns': {'s.c': t}, 'stats': {'s.c': probes.column_stats(t) if len(t) else
                                                 {'norm': np.zeros(0)}}} for t in tabs]

    class Probe:
        column_keys = ['s.c']
        _table_stats_ = probes.FeatureStatsProbe._table_stats_

    got = Probe()._table_stats_(results)['s.c']
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


def test_datafeature_stats_probe_over_tabs_of_different_sizes(tmp_path):
    """``tab_norm`` is one value per ROW, so uneven tabs do not stack.

    The build died on exactly that -- ``np.stack`` over per-tab norms of 3 and
    7 -- after every tab's stats and the whole-table ones had been computed. A
    table whose tabs all have one size, as the fixtures above do, never
    reaches it.
    """
    url = str(tmp_path)
    sampletable = UnevenSampleTable(datalake=url, tag="uneven_samples").build()
    featuretable = DatafeatureTable(
        datalake=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(signals=[("samples", "samples")],
                                            labels=[("labels", "labels")])),
        ),
        devices=["cpu"],
        tag="uneven_features",
    ).build()
    probe = FeatureStatsProbe(
        datalake=url,
        spec=dict(feature_table=featuretable,
                  collator=Datacollator(spec=dict(signals=[("features", "final")]))),
        tag="uneven_stats",
    ).build()

    assert probe.count == 10
    norms = probe.read('tab_norm', 'features.final')
    assert [len(n) for n in norms] == [3, 7]
    np.testing.assert_allclose(np.concatenate(norms), probe.read('norm', 'features.final'))
    # The reductions still stack: one row per tab.
    assert probe.read('tab_mean', 'features.final').shape == (2, 8)


def test_per_tab_arrays_round_trip(tmp_path):
    from dbx.dataparts import write_npz
    even = [np.arange(3.0), np.arange(3.0) + 1]
    uneven = [np.arange(3.0), np.arange(5.0), np.zeros(0)]
    for per_tab in (even, uneven):
        path = str(tmp_path / f"stat{len(per_tab)}.npz")
        write_npz(path, **FeatureStatsProbe._per_tab_arrays_(per_tab))
        got = FeatureStatsProbe._read_stat_(path)
        if per_tab is even:
            assert isinstance(got, np.ndarray) and got.shape == (2, 3)
        else:
            assert [a.tolist() for a in got] == [a.tolist() for a in per_tab]
