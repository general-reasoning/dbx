"""test_featuretab.py — Unit tests for Featuretab/Table and BipolarFeaturetab/Table."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from dbx import (
    SLICETOPIC,
    Datatab,
    Datatable,
    ModelEvaluator,
    ModelEvaluatorBuilder,
    Featuretab,
    Featuretable,
    BipolarFeaturetab,
    BipolarFeaturetable,
)
from dbx.datatables import Datacollator


def sample_collator(**spec):
    """What feeds DummyModel: the 'samples' slice's own column, labelled by 'labels'.

    A feature block takes its collator as a required VAR -- which pairs of
    (slice, column) are the signal is a decision about the block's identity,
    not something to be defaulted behind the author's back.
    """
    return Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]}, **spec))


class DummySampleTab(Datatab):
    TOPICS = {"samples": SLICETOPIC, "labels": SLICETOPIC}

    @dataclass
    class VAR(Datatab.VAR):
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


class DummySampleTable(Datatable):
    TAB = DummySampleTab

    @dataclass
    class VAR(Datatable.VAR):
        samples_per_tab: int = 10

    @property
    def n_tabs(self):
        return 2

    def __tab__(self, idx: int) -> DummySampleTab:
        return self.TAB(
            datalake=self.url,
            spec=dict(n_samples=self.var.samples_per_tab),
            tag=f"tab_{idx}",
        )


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


def test_featuretab_build_and_slice_inheritance(tmp_path):
    url = str(tmp_path)

    # 1. Build upstream sample tab
    sampletab = DummySampleTab(datalake=url, tag="samples_0").build()
    assert sampletab.slices() == ("samples", "labels")
    assert sampletab.valid()

    # 2. Build feature tab
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = Featuretab(
        datalake=url,
        spec=dict(
            upstream=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        device="cpu",
        tag="features_0",
    ).build()

    assert featuretab.valid()
    assert set(featuretab.slices()) == {"features"}

    # 3. Read data combining feature slice and inherited sample slices
    res = featuretab.data(("features", "final"), "samples", "labels")
    assert "features" in res
    assert "samples" in res
    assert "labels" in res

    assert res["features"]["final"].shape == (10, 8)
    assert res["samples"]["samples"].shape == (10, 4)
    assert len(res["labels"]["labels"]) == 10

    # 4. Map-style dataset zipping feature slice and sample slice
    ds = featuretab.dataset("features", "labels", mode="map")
    sample_0 = ds[0]
    assert "final" in sample_0["features"]
    assert "labels" in sample_0
    assert sample_0["features"]["final"].shape == (8,)


def _stats_probe(url, feature_block, tag, *, signals=(("features", "final"),), normalization=None):
    """The calibration a bipolar block thresholds against: a stats probe's medians."""
    from dbx.probes import FeatureStatsProbe
    return FeatureStatsProbe(
        datalake=url,
        spec=dict(
            feature_table=feature_block,
            collator=Datacollator(spec=dict(columns={'signals': list(signals)})),
            normalization=normalization,
        ),
        tag=tag,
    )


def _feature_table(url, tag, **spec):
    sampletable = DummySampleTable(datalake=url, spec=dict(samples_per_tab=5), tag=f"{tag}_samples").build()
    return Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=sample_collator(),
            **spec,
        ),
        devices=["cpu"],
        tag=tag,
    ).build()


def test_bipolar_featuretab_build_and_slice_inheritance(tmp_path):
    url = str(tmp_path)

    sampletab = DummySampleTab(datalake=url, tag="samples_1").build()
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = Featuretab(
        datalake=url,
        spec=dict(
            upstream=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        device="cpu",
        tag="features_1",
    ).build()
    stats = _stats_probe(url, featuretab, "stats_1").build()

    bipolar_tab = BipolarFeaturetab(
        datalake=url,
        spec=dict(upstream=featuretab, stats_probe=stats),
        tag="bipolar_1",
    ).build()

    assert bipolar_tab.valid()
    assert set(bipolar_tab.slices()) == {"bipolar"}
    assert bipolar_tab.declared_columns("bipolar") == {"final": "ndarray:int8"}
    assert set(bipolar_tab.slices(recursive=True)) == {"bipolar", "features", "samples", "labels"}

    # Reading across the encoding, the raw features, and the original labels.
    b_data = bipolar_tab.data(("bipolar", "final"), ("features", "final"), "labels")
    bipolar = b_data["bipolar"]["final"]
    features = b_data["features"]["final"]
    assert bipolar.shape == (10, 8)
    assert set(np.unique(bipolar)).issubset({-1, 1})
    median = stats.stat("median", ("features", "final"))
    np.testing.assert_array_equal(bipolar, np.where(features >= median, 1, -1))
    assert len(b_data["labels"]["labels"]) == 10


def test_featuretable_and_bipolar_table(tmp_path):
    url = str(tmp_path)
    featuretable = _feature_table(url, "feature_table")

    assert featuretable.n_tabs == 2
    assert set(featuretable.slices()) == {"features"}

    # Test reading combined data across table
    feat_data = featuretable.data(("features", "final"), concat=True)
    assert feat_data["features"]["final"].shape == (10, 8)
    label_data = featuretable.data("labels", concat=True)
    assert len(label_data["labels"]["labels"]) == 10

    stats = _stats_probe(url, featuretable, "stats_table").build()
    bipolar_table = BipolarFeaturetable(
        datalake=url,
        spec=dict(upstream=featuretable, stats_probe=stats),
        devices=["cpu"],
        tag="bipolar_table",
    ).build()

    assert bipolar_table.n_tabs == 2
    assert set(bipolar_table.slices(recursive=True)) == {"bipolar", "features", "samples", "labels"}

    b_tbl_data = bipolar_table.data("bipolar", ("features", "final"), "labels")
    assert b_tbl_data["bipolar"]["final"].shape == (10, 8)
    assert b_tbl_data["features"]["final"].shape == (10, 8)
    assert len(b_tbl_data["labels"]["labels"]) == 10


def test_bipolar_thresholds_against_the_calibration_not_the_tabs_own_median(tmp_path):
    """Every tab is encoded against the one table-wide median.

    The samples grow with their index, so the table's two tabs sit on either
    side of the table median. Against its own median each tab would come out
    exactly half +1 in every column -- the difference between the tabs gone.
    """
    url = str(tmp_path)
    featuretable = _feature_table(url, "calib_features")
    stats = _stats_probe(url, featuretable, "calib_stats").build()
    bipolar_table = BipolarFeaturetable(
        datalake=url, spec=dict(upstream=featuretable, stats_probe=stats), tag="calib_bipolar",
    ).build()

    median = stats.stat("median", ("features", "final"))
    tab_means = []
    for i in range(2):
        tab = bipolar_table.tab(i)
        d = tab.data(("bipolar", "final"), ("features", "final"))
        np.testing.assert_array_equal(d["bipolar"]["final"], np.where(d["features"]["final"] >= median, 1, -1))
        tab_means.append(d["bipolar"]["final"].mean(axis=0))
    assert not np.allclose(tab_means[0], tab_means[1]), "the tabs' encodings still tell them apart"


def test_bipolar_normalizes_as_its_stats_probe_did(tmp_path):
    from dbx.probes import normalize_features
    url = str(tmp_path)
    featuretable = _feature_table(url, "norm_features")
    stats = _stats_probe(url, featuretable, "norm_stats", normalization="l2").build()
    bipolar_table = BipolarFeaturetable(
        datalake=url, spec=dict(upstream=featuretable, stats_probe=stats), tag="norm_bipolar",
    ).build()

    d = bipolar_table.data(("bipolar", "final"), ("features", "final"), concat=True)
    x = normalize_features(d["features"]["final"].astype(np.float64), "l2")
    median = stats.stat("median", ("features", "final"))
    np.testing.assert_array_equal(d["bipolar"]["final"], np.where(x >= median, 1, -1))


def test_bipolar_encodes_every_feature_column_or_the_ones_named(tmp_path):
    url = str(tmp_path)
    featuretable = _feature_table(url, "two_features", feature_for_column={"a": "final", "b": "final"})
    stats = _stats_probe(url, featuretable, "two_stats",
                         signals=(("features", "a"), ("features", "b"))).build()

    every = BipolarFeaturetable(
        datalake=url, spec=dict(upstream=featuretable, stats_probe=stats), tag="every")
    assert every.tab(0).declared_columns("bipolar") == {"a": "ndarray:int8", "b": "ndarray:int8"}

    only_b = BipolarFeaturetable(
        datalake=url, spec=dict(upstream=featuretable, stats_probe=stats, features=["b"]),
        tag="only_b").build()
    assert only_b.tab(0).declared_columns("bipolar") == {"b": "ndarray:int8"}
    assert only_b.data(("bipolar", "b"), concat=True)["bipolar"]["b"].shape == (10, 8)
    assert only_b.hash != every.hash

    with pytest.raises(ValueError, match="not columns"):
        BipolarFeaturetable(
            datalake=url, spec=dict(upstream=featuretable, stats_probe=stats, features=["c"]),
            tag="nope").tab(0)


def test_bipolar_refuses_a_stats_probe_without_the_median_it_needs(tmp_path):
    url = str(tmp_path)
    featuretable = _feature_table(url, "short_features")
    stats = _stats_probe(url, featuretable, "short_stats", signals=(("samples", "samples"),))
    with pytest.raises(ValueError, match="has no median for feature column"):
        BipolarFeaturetable(
            datalake=url, spec=dict(upstream=featuretable, stats_probe=stats), tag="short").tab(0)


def test_bipolar_refuses_to_build_against_an_unbuilt_stats_probe(tmp_path):
    url = str(tmp_path)
    featuretable = _feature_table(url, "unbuilt_features")
    stats = _stats_probe(url, featuretable, "unbuilt_stats")
    tab = BipolarFeaturetable(
        datalake=url, spec=dict(upstream=featuretable, stats_probe=stats), tag="unbuilt").tab(0)
    # build() asks its VAR blocks before building, and names the one that is not.
    with pytest.raises(ValueError, match="'stats_probe': False"):
        tab.build()


def test_custom_features_mapping(tmp_path):
    url = str(tmp_path)
    sampletab = DummySampleTab(datalake=url, tag="samples_cust").build()
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = Featuretab(
        datalake=url,
        spec=dict(
            upstream=sampletab,
            evaluator_factory=eval_factory,
            feature_for_column={"custom_output": "final"},
            collator=sample_collator(),
        ),
        device="cpu",
        tag="features_cust",
    ).build()

    assert featuretab.slices() == ("features",)
    res = featuretab.data(("features", "custom_output"))
    assert res["features"]["custom_output"].shape == (10, 8)


def test_signal_selection(tmp_path):
    url = str(tmp_path)
    sampletab = DummySampleTab(datalake=url, tag="samples_sig").build()
    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab = Featuretab(
        datalake=url,
        spec=dict(
            upstream=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        device="cpu",
        tag="features_sig",
    ).build()

    assert featuretab.valid()
    res = featuretab.data(("features", "final"))
    assert res["features"]["final"].shape == (10, 8)


def test_datacollator():
    from dbx.datatables import Datacollator

    collator = Datacollator(
        spec=dict(columns={'signals': [("samples", "samples"), ("extra", "extra")], 'labels': [("labels", "labels")]})
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

    signals, labels = collator(batch_datapoints)
    assert signals.shape == (2, 5, 2, 4)  # batch=2, tokens=5, signals=2, d=4
    assert labels.shape == (2, 1, 1, 1)

    # Test length trimming
    c_len = Datacollator(
        spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]}, length=2)
    )
    len_signals, _ = c_len(batch_datapoints)
    assert len_signals.shape == (2, 5, 1, 2)  # last dim trimmed to 2

    # signal_only is how a CALLER wants the output shaped, not part of what the
    # collator is, so it is a call argument rather than spec.
    c_one = Datacollator(
        spec=dict(columns={'signals': [("samples", "samples")], 'labels': [("labels", "labels")]})
    )

    # A tuple, always -- never a dict keyed by names three files had to agree on.
    out = c_one(batch_datapoints)
    assert isinstance(out, tuple) and len(out) == 2

    # signal_only hands back the one array bare, not a tuple of one.
    out_sig = c_one(batch_datapoints, signal_only=True)
    assert isinstance(out_sig, np.ndarray)
    assert out_sig.shape == (2, 5, 1, 4)

    # With no labels declared the tuple is still a tuple, of length one.
    c_nolabel = Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': []}))
    assert len(c_nolabel(batch_datapoints)) == 1


def test_datacollator_without_labels():
    """No ``labels`` role, or one that is None: a collator of signals alone reads no label slice."""
    from dbx.datatables import Datacollator

    rows = [{"samples": {"samples": np.ones((5, 4), dtype=np.float32) * i}} for i in range(2)]
    for c in (Datacollator(spec=dict(columns={'signals': [("samples", "samples")]})),
              Datacollator(spec=dict(columns={'signals': [("samples", "samples")], 'labels': None}))):
        assert not c.var.columns.get('labels') and c.label_pairs == ()
        assert c.slices() == ["samples"]
        out = c(rows)
        assert isinstance(out, tuple) and len(out) == 1 and out[0].shape == (2, 5, 1, 4)
        assert c(rows, signal_only=True).shape == (2, 5, 1, 4)
        batch = {"samples": {"samples": np.stack([r["samples"]["samples"] for r in rows])}}
        assert c(batch)[0].shape == (2, 5, 4)


def test_a_labelless_collator_is_refused_by_a_classifier():
    """A probe that fits labels says so, rather than fitting nothing."""
    from dbx.datatables import Datacollator
    from dbx.probes import label_vector

    c = Datacollator(spec=dict(columns={'signals': [("samples", "samples")]}))
    with pytest.raises(ValueError, match="one label column"):
        label_vector(c, {"samples": {"samples": np.zeros((2, 4))}})


def test_datacollator_slices_are_deterministic():
    """`slices` is splatted into dataset()/data(), where position decides the
    zip order, so it must not come from a set: str hashing is seeded per
    process and the order would differ between workers and between reruns."""
    from dbx.datatables import Datacollator

    collator = Datacollator(
        spec=dict(columns={'signals': [("zeta", "a"), ("alpha", "b"), ("zeta", "c")], 'labels': [("mu", "y")]})
    )
    assert collator.slices() == ["zeta", "alpha", "mu"]


def test_datacollator_refuses_a_missing_column():
    """No silent fallback: a wrong column name used to select an arbitrary
    array via next(iter(...)) and the build wrote it as the real feature."""
    from dbx.datatables import Datacollator

    collator = Datacollator(spec=dict(columns={'signals': [("samples", "nope")], 'labels': []}))
    rows = [{"samples": {"samples": np.ones((2, 2), dtype=np.float32)}}]
    with pytest.raises(KeyError, match="has no column 'nope'"):
        collator(rows)


def test_featuretab_streaming(tmp_path):
    url = str(tmp_path)
    sampletab = DummySampleTab(datalake=url, tag="samples_str").build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretab_bulk = Featuretab(
        datalake=url,
        spec=dict(
            upstream=sampletab,
            evaluator_factory=eval_factory,
            collator=sample_collator(),
        ),
        device="cpu",
        streaming=False,
        tag="features_bulk",
    ).build()

    featuretab_stream = Featuretab(
        datalake=url,
        spec=dict(
            upstream=sampletab,
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


def test_featuretable_streaming(tmp_path):
    url = str(tmp_path)
    sampletable = DummySampleTable(
        datalake=url,
        spec=dict(samples_per_tab=5),
        tag="sample_table_str",
    ).build()

    eval_factory = DummyModelEvaluatorFactory(spec=dict(capture_final=True))

    featuretable_stream = Featuretable(
        datalake=url,
        spec=dict(
            upstream=sampletable,
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


class FeaturesNamedSampleTab(DummySampleTab):
    """A sample tab whose own slice is called 'features' -- the one name a
    Featuretab cannot borrow, because it owns that name itself."""

    TOPICS = {"features": SLICETOPIC, "labels": SLICETOPIC}

    def __build__(self):
        specs = {
            "features": {"samples": "ndarray:float32"},
            "labels": {"labels": "int64"},
        }
        with self.slice_writers(specs) as writers:
            for i in range(self.var.n_samples):
                writers["features"].write({"samples": np.arange(4, dtype=np.float32) + i})
                writers["labels"].write({"labels": np.int64(i % 2)})
        return self


def test_an_upstream_slice_named_features_is_refused(tmp_path):
    """A row is keyed by slice name, so the tab's own 'features' and an
    upstream 'features' have no way to both appear in one.  Silently
    preferring either is how a caller reads features believing it asked for
    samples, so this is an error rather than a precedence rule."""
    url = str(tmp_path)
    sampletab = FeaturesNamedSampleTab(datalake=url, tag="clash_samples").build()

    featuretab = Featuretab(
        datalake=url,
        spec=dict(
            upstream=sampletab,
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=Datacollator(spec=dict(columns={'signals': [("features", "samples")], 'labels': [("labels", "labels")]})),
            feature_for_column={"final": "final"},
        ),
        device="cpu",
        tag="clash_features",
    )

    with pytest.raises(KeyError, match="which this block also owns"):
        featuretab.dataset()
    with pytest.raises(KeyError, match="which this block also owns"):
        featuretab.data("labels")


if __name__ == "__main__":
    import sys
    dbx.dataparts.gitwrkreposetup = lambda *a, **k: None
    sys.exit(pytest.main([__file__]))


def test_a_collator_quotes_as_a_block_not_as_a_function():
    """`dbx.quote(collator)` asked callable() first, and a Datacollator is callable.

    It then tried to render the INSTANCE as a function, by a __qualname__ it
    does not have -- which is how a pipeline passing ``collator=dbx.quote(c)``
    died before building anything.
    """
    import dbx
    from dbx.datatables import Datacollator

    c = Datacollator(spec=dict(columns={'signals': [("features", "feature_final")]}))
    assert callable(c)
    assert dbx.quote(c) == c.quote()
    assert dbx.eval(dbx.quote(c)).hash == c.hash
    # ... and as an argument to a quoted call, which goes through quote() too.
    assert c.quote() in dbx.quotefn("some.pipeline", collator=c).replace('\\"', '"')
    # A CLASS still renders as the call it names.
    assert dbx.quote(Datacollator, spec={}).startswith("$dbx.datatables.Datacollator(")


def test_a_callable_instance_that_cannot_quote_itself_is_refused_by_name():
    """No __qualname__ and no .quote(): nothing could rebuild it, and the error says so."""
    import dbx

    class Evaluator:
        def __call__(self, x):
            return x

    for attempt in (lambda: dbx.quote(Evaluator()),
                    lambda: dbx.quotefn(Evaluator()),
                    lambda: dbx.quotefn("some.fn", Evaluator())):
        with pytest.raises(TypeError, match="cannot quote a Evaluator instance"):
            attempt()
    # A function and a class still render as the calls they name.
    assert dbx.quote(len, 3) == "$builtins.len(3)"
    assert dbx.quote(Evaluator).endswith("Evaluator()")
