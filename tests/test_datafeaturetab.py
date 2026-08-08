"""test_datafeaturetab.py — Unit tests for DatafeatureTab/Table and BipolarDatafeatureTab/Table."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from dbx import (
    DatastreamTab,
    DatastreamTable,
    DatamodelEvaluator,
    DatamodelEvaluatorFactory,
    DatafeatureTab,
    DatafeatureTable,
    BipolarDatafeatureTab,
    BipolarDatafeatureTable,
)


class DummySampleTab(DatastreamTab):
    SLICES = ("samples", "labels")

    @dataclass
    class VAR(DatastreamTab.VAR):
        n_samples: int = 10

    def __build__(self):
        specs = {
            "samples": {"samples": "ndarray:float32"},
            "labels": {"labels": "int64"},
        }
        with self.slice_writers(specs) as writers:
            for i in range(self.var.n_samples):
                vec = np.arange(4, dtype=np.float32) + i
                label = np.int64(i % 2)
                writers["samples"].write({"samples": vec})
                writers["labels"].write({"labels": label})
        return self


class DummySampleTable(DatastreamTable):
    TAB = DummySampleTab

    @dataclass
    class VAR(DatastreamTable.VAR):
        samples_per_tab: int = 10

    @property
    def n_tabs(self):
        return 2

    def __tab__(self, idx: int) -> DummySampleTab:
        return self.TAB(
            url=self.url,
            spec=dict(n_samples=self.var.samples_per_tab),
            tag=f"tab_{idx}",
        )


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


def test_datafeature_tab_build_and_slice_inheritance(tmp_path):
    url = str(tmp_path)

    # 1. Build upstream sample tab
    sampletab = DummySampleTab(url=url, tag="samples_0").build()
    assert sampletab.slices == ("samples", "labels")
    assert sampletab.valid()

    # 2. Build feature tab
    eval_factory = DatamodelEvaluatorFactory(capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            sampletab=sampletab,
            evaluator_factory=eval_factory,
        ),
        device="cpu",
        tag="features_0",
    ).build()

    assert featuretab.valid()
    assert "features_final" in featuretab.slices
    assert set(featuretab.available_slices) == {"features_final", "samples", "labels"}

    # 3. Read data combining feature slice and inherited sample slices
    res = featuretab.data("features_final", "samples", "labels")
    assert "features_final" in res
    assert "samples" in res
    assert "labels" in res

    assert res["features_final"].shape == (10, 8)
    assert res["samples"].shape == (10, 4)
    assert len(res["labels"]) == 10

    # 4. Map-style dataset zipping feature slice and sample slice
    ds = featuretab.dataset("features_final", "labels", mode="map")
    sample_0 = ds[0]
    assert "features_final" in sample_0
    assert "labels" in sample_0
    assert sample_0["features_final"].shape == (8,)


def test_bipolar_datafeature_tab_build_and_slice_inheritance(tmp_path):
    url = str(tmp_path)

    sampletab = DummySampleTab(url=url, tag="samples_1").build()
    eval_factory = DatamodelEvaluatorFactory(capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            sampletab=sampletab,
            evaluator_factory=eval_factory,
        ),
        device="cpu",
        tag="features_1",
    ).build()

    bipolar_tab = BipolarDatafeatureTab(
        url=url,
        spec=dict(
            featuretab=featuretab,
            layer="final",
            threshold=0.3,
        ),
        tag="bipolar_1",
    ).build()

    assert bipolar_tab.valid()
    assert set(bipolar_tab.slices) == {"bipolar_features", "tab_bipolar_features"}
    assert set(bipolar_tab.available_slices) == {
        "bipolar_features",
        "tab_bipolar_features",
        "features_final",
        "samples",
        "labels",
    }

    # Test reading data across bipolar, raw features, and original sample labels
    b_data = bipolar_tab.data("bipolar_features", "features_final", "labels")
    assert b_data["bipolar_features"].shape == (10, 8)
    assert set(np.unique(b_data["bipolar_features"])).issubset({-1, 1})
    assert b_data["features_final"].shape == (10, 8)
    assert len(b_data["labels"]) == 10


def test_datafeature_table_and_bipolar_table(tmp_path):
    url = str(tmp_path)

    # 1. Build sample table with 2 tabs
    sampletable = DummySampleTable(
        url=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table",
    ).build()

    eval_factory = DatamodelEvaluatorFactory(capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    # 2. Build feature table
    featuretable = DatafeatureTable(
        url=url,
        spec=dict(
            sampletable=sampletable,
            evaluator_factory=eval_factory,
        ),
        devices=["cpu"],
        tag="feature_table",
    ).build()

    assert featuretable.n_tabs == 2
    assert set(featuretable.available_slices) == {"features_final", "samples", "labels"}

    # Test reading combined data across table
    tbl_data = featuretable.data("features_final", "labels")
    assert tbl_data["features_final"].shape == (10, 8)
    assert len(tbl_data["labels"]) == 10

    # 3. Build bipolar feature table
    bipolar_table = BipolarDatafeatureTable(
        url=url,
        spec=dict(
            featuretable=featuretable,
            layer="final",
        ),
        tag="bipolar_table",
    ).build()

    assert bipolar_table.n_tabs == 2
    assert set(bipolar_table.available_slices) == {
        "bipolar_features",
        "tab_bipolar_features",
        "features_final",
        "samples",
        "labels",
    }

    b_tbl_data = bipolar_table.data("bipolar_features", "labels")
    assert b_tbl_data["bipolar_features"].shape == (10, 8)
    assert len(b_tbl_data["labels"]) == 10


if __name__ == "__main__":
    import sys
    dbx.dataparts.gitwrkreposetup = lambda *a, **k: None
    sys.exit(pytest.main([__file__]))
