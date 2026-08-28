"""test_datafeaturetab.py — Unit tests for DatafeatureTab/Table and BipolarDatafeatureTab/Table."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from dbx import (
    SLICETOPIC,
    DatapointTab,
    DatapointTable,
    DatamodelEvaluator,
    DatamodelEvaluatorFactory,
    DatafeatureTab,
    DatafeatureTable,
    BipolarDatafeatureTab,
    BipolarDatafeatureTable,
)
from dbx.datafeatures import Datacollator


def sample_collator(**spec):
    """What feeds DummyModel: the 'samples' slice's own column, labelled by 'labels'.

    A feature block takes its collator as a required VAR -- which pairs of
    (slice, column) are the signal is a decision about the block's identity,
    not something to be defaulted behind the author's back.
    """
    return Datacollator(spec=dict(
        signals=[("samples", "samples")],
        labels=[("labels", "labels")],
        **spec,
    ))


class DummySampleTab(DatapointTab):
    TOPICS = {"samples": SLICETOPIC, "labels": SLICETOPIC}

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


class DummyModelEvaluatorFactory(DatamodelEvaluatorFactory):
    @property
    def model(self):
        return DummyModel()


def test_datafeature_tab_build_and_slice_inheritance(tmp_path):
    url = str(tmp_path)

    # 1. Build upstream sample tab
    sampletab = DummySampleTab(url=url, tag="samples_0").build()
    assert sampletab.slices == ("samples", "labels")
    assert sampletab.valid()

    # 2. Build feature tab
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
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
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
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

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    # 2. Build feature table
    featuretable = DatafeatureTable(
        url=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
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
        devices=["cpu"],
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
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            feature_namemap={"custom_output": "final"},
            collator=sample_collator(),
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
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
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

    # strip_keys and signal_only are how a CALLER wants the output shaped, not
    # part of what the collator is, so they are call arguments rather than spec.
    c_one = Datacollator(
        spec=dict(
            signals=[("samples", "samples")],
            labels=[("labels", "labels")],
        )
    )

    # Test strip_keys
    out_strip = c_one(batch_datapoints, strip_keys=True)
    assert isinstance(out_strip, tuple)
    assert len(out_strip) == 2

    # Test signal_only (strip_keys=False) -> dict with 'signal' key only
    out_sig_dict = c_one(batch_datapoints, signal_only=True)
    assert isinstance(out_sig_dict, dict)
    assert list(out_sig_dict.keys()) == ["signal"]
    assert out_sig_dict["signal"].shape == (2, 5, 1, 4)

    # Test signal_only (strip_keys=True) -> signal value directly
    out_sig_val = c_one(batch_datapoints, signal_only=True, strip_keys=True)
    assert isinstance(out_sig_val, np.ndarray)
    assert out_sig_val.shape == (2, 5, 1, 4)


def test_datafeature_tab_streaming(tmp_path):
    url = str(tmp_path)
    sampletab = DummySampleTab(url=url, tag="samples_str").build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab_bulk = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        device="cpu",
        streaming=False,
        tag="features_bulk",
    ).build()

    featuretab_stream = DatafeatureTab(
        url=url,
        spec=dict(
            datapoint_tab=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        device="cpu",
        streaming=True,
        dataloader_kwargs={"num_workers": 0},
        tag="features_stream",
    ).build()

    assert featuretab_stream.streaming is True
    assert featuretab_stream.dataloader_kwargs == {"num_workers": 0}
    assert featuretab_stream.valid()

    data_bulk = featuretab_bulk.data(("features", "final"))["features"]["final"]
    data_stream = featuretab_stream.data(("features", "final"))["features"]["final"]
    np.testing.assert_allclose(np.squeeze(data_stream), np.squeeze(data_bulk))


def test_datafeature_table_streaming(tmp_path):
    url = str(tmp_path)
    sampletable = DummySampleTable(
        url=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_str",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable_stream = DatafeatureTable(
        url=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        devices=["cpu"],
        streaming=True,
        dataloader_kwargs={"num_workers": 0},
        tag="feature_table_str",
    ).build()

    assert featuretable_stream.streaming is True
    assert featuretable_stream.dataloader_kwargs == {"num_workers": 0}
    tab0 = featuretable_stream.tab(0)
    assert tab0.streaming is True
    assert tab0.dataloader_kwargs == {"num_workers": 0}

    feat_data = featuretable_stream.data(("features", "final"), concat=True)
    assert np.squeeze(feat_data["features"]["final"]).shape == (10, 8)


def test_datapoint_table_tab_signature_commutativity(tmp_path):
    """Verify commutativity between table.var.datapoint_table.tab(idx).signature() and table.tab(idx).var.datapoint_tab.signature()."""
    url = str(tmp_path)
    sampletable = DummySampleTable(
        url=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table",
    )
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))
    featuretable = DatafeatureTable(
        url=url,
        spec=dict(
            datapoint_table=sampletable,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        devices=["cpu"],
        tag="feature_table",
    )

    sig_via_table = featuretable.var.datapoint_table.tab(0).signature()
    sig_via_tab = featuretable.tab(0).var.datapoint_tab.signature()
    assert sig_via_table == sig_via_tab


if __name__ == "__main__":
    import sys
    dbx.dataparts.gitwrkreposetup = lambda *a, **k: None
    sys.exit(pytest.main([__file__]))
