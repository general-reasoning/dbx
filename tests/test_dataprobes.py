"""test_dataprobes.py — Unit tests for DatafeatureAffineLogisticProbe and DatafeatureStatsProbe."""

from dataclasses import dataclass
import numpy as np
import pytest
import torch
import torch.nn as nn

import dbx
from dbx import (
    DatafeatureTab,
    DatafeatureTable,
    DatasampleTab,
    DatasampleTable,
    DatamodelEvaluator,
    DatamodelEvaluatorFactory,
    DatafeatureAffineLogisticProbe,
    DatafeatureAffineLogisticProber,
    DatafeatureStatsProbe,
    normalize_features,
)


class DummySampleTab(DatasampleTab):
    SLICES = ("samples", "labels")

    @dataclass
    class VAR(DatasampleTab.VAR):
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


class DummySampleTable(DatasampleTable):
    TAB = DummySampleTab

    @dataclass
    class VAR(DatasampleTable.VAR):
        samples_per_tab: int = 5

    @property
    def n_tabs(self) -> int:
        return 2

    def __tab__(self, idx: int, tag=None) -> DummySampleTab:
        return self.TAB(
            url=self.path('tabs'),
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
        url=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table",
    ).build()

    eval_factory = DatamodelEvaluatorFactory(model="$test_dataprobes.DummyModel()", capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretable = DatafeatureTable(
        url=url,
        spec=dict(
            sampletable=sampletable,
            evaluator_factory=eval_factory,
        ),
        devices=["cpu"],
        tag="feature_table",
    ).build()

    probe = DatafeatureAffineLogisticProbe(
        url=url,
        spec=dict(
            featuretable=featuretable,
            regressor=("features_final", "features_final"),
            label=("labels", "labels"),
            evaluation_fraction=0.8,
        ),
        tag="log_probe",
    ).build()

    assert probe.valid()
    assert probe.read('features').shape == (10, 8)
    assert len(probe.read('labels')) == 10
    assert probe.read('coef').shape[1] == 8
    assert isinstance(probe.read('evaluation_report'), str)

    asph = probe.asphericity()
    assert isinstance(asph, dict)
    assert len(asph) > 0


def test_datafeature_stats_probe(tmp_path):
    url = str(tmp_path)

    sampletable = DummySampleTable(
        url=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_stats",
    ).build()

    eval_factory = DatamodelEvaluatorFactory(model="$test_dataprobes.DummyModel()", capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretable = DatafeatureTable(
        url=url,
        spec=dict(
            sampletable=sampletable,
            evaluator_factory=eval_factory,
        ),
        devices=["cpu"],
        tag="feature_table_stats",
    ).build()

    stats_probe = DatafeatureStatsProbe(
        url=url,
        spec=dict(
            featuretable=featuretable,
            feature="features_final",
        ),
        tag="stats_probe",
    ).build()

    assert stats_probe.valid()
    assert stats_probe.tile_count == 10
    assert stats_probe.tile_feature_mean.shape == (8,)
    assert stats_probe.tile_feature_std.shape == (8,)
    assert stats_probe.tile_feature_median.shape == (8,)
    assert stats_probe.tile_feature_min.shape == (8,)
    assert stats_probe.tile_feature_max.shape == (8,)
    assert stats_probe.tile_feature_norms.shape == (10,)
    assert stats_probe.read('distinct_tile_count') > 0


if __name__ == "__main__":
    import sys
    dbx.dataparts.gitwrkreposetup = lambda *a, **k: None
    sys.exit(pytest.main([__file__]))
