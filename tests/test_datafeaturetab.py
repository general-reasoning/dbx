"""test_datafeaturetab.py — Unit tests for DatafeatureTab/Table and BipolarDatafeatureTab/Table."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from dbx import (
    DatapointTab,
    DatapointTable,
    DatamodelEvaluator,
    DatamodelEvaluatorFactory,
    DatafeatureTab,
    DatafeatureTable,
    BipolarDatafeatureTab,
    BipolarDatafeatureTable,
)


class DummySampleTab(DatapointTab):
    SLICES = ("samples", "labels")

    @dataclass
    class VAR(DatapointTab.VAR):
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


class DummySampleTable(DatapointTable):
    TAB = DummySampleTab

    @dataclass
    class VAR(DatapointTable.VAR):
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
    eval_factory = DatamodelEvaluatorFactory(model="$test_datafeaturetab.DummyModel()", capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
        ),
        device="cpu",
        tag="features_0",
    ).build()

    assert featuretab.valid()
    assert set(featuretab.slices) == {"features"}

    # 3. Read data combining feature slice and inherited sample slices
    res = featuretab.data(("features", "final"), "samples", "labels")
    assert "features" in res
    assert "samples" in res
    assert "labels" in res

    assert res["features"]["final"].shape == (10, 8)
    assert res["samples"].shape == (10, 4)
    assert len(res["labels"]) == 10

    # 4. Map-style dataset zipping feature slice and sample slice
    ds = featuretab.dataset("features", "labels", mode="map")
    sample_0 = ds[0]
    assert "final" in sample_0
    assert "labels" in sample_0
    assert sample_0["final"].shape == (8,)


def test_bipolar_datafeature_tab_build_and_slice_inheritance(tmp_path):
    url = str(tmp_path)

    sampletab = DummySampleTab(url=url, tag="samples_1").build()
    eval_factory = DatamodelEvaluatorFactory(model="$test_datafeaturetab.DummyModel()", capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
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
        "features",
    }

    # Test reading data across bipolar, raw features, and original sample labels
    b_data = bipolar_tab.data("bipolar_features", ("features", "final"))
    assert b_data["bipolar_features"].shape == (10, 8)
    assert set(np.unique(b_data["bipolar_features"])).issubset({-1, 1})
    assert b_data["features"]["final"].shape == (10, 8)


def test_datafeature_table_and_bipolar_table(tmp_path):
    url = str(tmp_path)

    # 1. Build sample table with 2 tabs
    sampletable = DummySampleTable(
        url=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table",
    ).build()

    eval_factory = DatamodelEvaluatorFactory(model="$test_datafeaturetab.DummyModel()", capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    # 2. Build feature table
    featuretable = DatafeatureTable(
        url=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
        ),
        devices=["cpu"],
        tag="feature_table",
    ).build()

    assert featuretable.n_tabs == 2
    assert set(featuretable.slices) == {"features"}

    # Test reading combined data across table
    feat_data = featuretable.data(("features", "final"), concat=True)
    assert feat_data["features"]["final"].shape == (10, 8)
    label_data = featuretable.data("labels", concat=True)
    assert len(label_data["labels"]) == 10

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
        "features",
    }

    b_tbl_data = bipolar_table.data("bipolar_features", ("features", "final"))
    assert b_tbl_data["bipolar_features"].shape == (10, 8)
    assert b_tbl_data["features"]["final"].shape == (10, 8)


def test_custom_features_mapping(tmp_path):
    url = str(tmp_path)
    sampletab = DummySampleTab(url=url, tag="samples_cust").build()
    eval_factory = DatamodelEvaluatorFactory(model="$test_datafeaturetab.DummyModel()", capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            feature_namemap={"custom_output": "final"},
        ),
        device="cpu",
        tag="features_cust",
    ).build()

    assert featuretab.slices == ("features",)
    res = featuretab.data(("features", "custom_output"))
    assert res["features"]["custom_output"].shape == (10, 8)


def test_signal_selection(tmp_path):
    url = str(tmp_path)
    sampletab = DummySampleTab(url=url, tag="samples_sig").build()
    eval_factory = DatamodelEvaluatorFactory(model="$test_datafeaturetab.DummyModel()", capture_final=True)
    eval_factory.Evaluator = lambda model=None, **kwargs: DatamodelEvaluator(
        model=DummyModel(), **{k: v for k, v in kwargs.items() if k != 'device'}, device="cpu"
    )

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            collator=("samples", "samples"),
        ),
        device="cpu",
        tag="features_sig",
    ).build()

    assert featuretab.valid()
    res = featuretab.data("features_final")
    assert res["features_final"].shape == (10, 8)


def test_datacollator():
    from dbx.datafeatures import Datacollator

    collator = Datacollator(
        spec=dict(
            signals=[("samples", "samples"), ("extra", "extra")],
            labels=[("labels", "labels")],
        )
    )

    batch_datapoints = [
        {
            "samples": {"samples": np.ones((5, 4), dtype=np.float32)},
            "extra": {"extra": np.zeros((5, 4), dtype=np.float32)},
            "labels": {"labels": np.int64(1)},
        },
        {
            "samples": {"samples": np.ones((5, 4), dtype=np.float32) * 2},
            "extra": {"extra": np.zeros((5, 4), dtype=np.float32) * 2},
            "labels": {"labels": np.int64(0)},
        },
    ]

    out = collator(batch_datapoints)
    assert "signal" in out
    assert "label" in out
    assert out["signal"].shape == (2, 5, 2, 4)  # batch=2, tokens=5, signals=2, d=4
    assert out["label"].shape == (2, 1, 1, 1)

    # Test length trimming
    c_len = Datacollator(
        spec=dict(
            signals=[("samples", "samples")],
            labels=[("labels", "labels")],
            length=2,
        )
    )
    out_len = c_len(batch_datapoints)
    assert out_len["signal"].shape == (2, 5, 1, 2)  # last dim trimmed to 2

    # Test strip_keys
    c_strip = Datacollator(
        spec=dict(
            signals=[("samples", "samples")],
            labels=[("labels", "labels")],
            strip_keys=True,
        )
    )
    out_strip = c_strip(batch_datapoints)
    assert isinstance(out_strip, tuple)
    assert len(out_strip) == 2

    # Test signal_only (strip_keys=False) -> dict with 'signal' key only
    c_sig_dict = Datacollator(
        spec=dict(
            signals=[("samples", "samples")],
            labels=[("labels", "labels")],
            signal_only=True,
            strip_keys=False,
        )
    )
    out_sig_dict = c_sig_dict(batch_datapoints)
    assert isinstance(out_sig_dict, dict)
    assert list(out_sig_dict.keys()) == ["signal"]
    assert out_sig_dict["signal"].shape == (2, 5, 1, 4)

    # Test signal_only (strip_keys=True) -> signal value directly
    c_sig_val = Datacollator(
        spec=dict(
            signals=[("samples", "samples")],
            labels=[("labels", "labels")],
            signal_only=True,
            strip_keys=True,
        )
    )
    out_sig_val = c_sig_val(batch_datapoints)
    assert isinstance(out_sig_val, np.ndarray)
    assert out_sig_val.shape == (2, 5, 1, 4)


if __name__ == "__main__":
    import sys
    dbx.dataparts.gitwrkreposetup = lambda *a, **k: None
    sys.exit(pytest.main([__file__]))
