"""test_datamodels.py — Tests for DatamodelEvaluator/Factory and DataformerEvaluator/Factory."""

import pytest
import torch
import torch.nn as nn

from dbx import (
    DatamodelEvaluator,
    DatamodelEvaluatorFactory,
    DataformerEvaluator,
    DataformerEvaluatorFactory,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 16)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(8, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 8),
        ])

    def forward(self, x):
        # x shape: (B, N, d) e.g. (2, 5, 8)
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class DummyModelEvaluatorFactory(DatamodelEvaluatorFactory):
    @property
    def model(self):
        return DummyModel()


class DummyTransformerEvaluatorFactory(DataformerEvaluatorFactory):
    @property
    def model(self):
        return DummyTransformer()


def test_datamodel_evaluator_basic():
    model = DummyModel()
    evaluator = DatamodelEvaluator(
        model=model,
        capture_layers=["layer1"],
        capture_final=True,
        device="cpu",
    )
    assert evaluator.layer_names == ["layer1", "final"]

    x = torch.randn(2, 4)
    res = evaluator(x)

    assert "layer1" in res
    assert "final" in res
    assert res["layer1"].shape == (2, 8)
    assert res["final"].shape == (2, 16)
    assert evaluator.layer_features["layer1"].shape == (2, 8)

    evaluator.clear()
    assert len(evaluator.layer_features) == 0


def test_datamodel_evaluator_factory():
    factory = DummyModelEvaluatorFactory(
        spec=dict(capture_layers=["layer1"], capture_final=True)
    )
    ev1 = factory.evaluator(device="cpu")
    ev2 = factory.evaluator(device="cpu")
    assert ev1 is ev2
    assert isinstance(ev1, DatamodelEvaluator)
    assert ev1.layer_names == ["layer1", "final"]


def test_dataformer_evaluator_blocks_and_cls_only():
    model = DummyTransformer()
    evaluator = DataformerEvaluator(
        model=model,
        capture_blocks=[0, -1],
        capture_final=True,
        cls_token_only=True,
        device="cpu",
    )
    assert evaluator.layer_names == ["block.0", "block.2", "final"]

    x = torch.randn(2, 5, 8)  # B=2, N=5 (CLS token at index 0), d=8
    res = evaluator(x)

    assert "block.0" in res
    assert "block.2" in res
    assert "final" in res
    # With cls_token_only=True, 3D tensor (2, 5, 8) is sliced to (2, 8)
    assert res["block.0"].shape == (2, 8)
    assert res["block.2"].shape == (2, 8)
    assert res["final"].shape == (2, 8)


def test_dataformer_evaluator_all_blocks():
    model = DummyTransformer()
    evaluator = DataformerEvaluator(
        model=model,
        capture_blocks="all",
        capture_final=False,
        cls_token_only=False,
        device="cpu",
    )
    assert evaluator.layer_names == ["block.0", "block.1", "block.2"]

    x = torch.randn(2, 5, 8)
    res = evaluator(x)

    assert len(res) == 3
    assert "final" not in res
    assert res["block.0"].shape == (2, 5, 8)


def test_dataformer_evaluator_factory():
    factory = DummyTransformerEvaluatorFactory(
        spec=dict(capture_blocks=[0, 1], cls_token_only=True, capture_final=True)
    )
    assert factory.layer_names == ["block.0", "block.1", "final"]
    ev1 = factory.evaluator(device="cpu")
    ev2 = factory.evaluator(device="cpu")
    assert ev1 is ev2
    assert isinstance(ev1, DataformerEvaluator)
    assert ev1.layer_names == ["block.0", "block.1", "final"]
    factory_all = DummyTransformerEvaluatorFactory(
        spec=dict(capture_blocks="all", capture_final=True)
    )
    assert factory_all.layer_names == ["block.0", "block.1", "block.2", "final"]


class TransformedModelEvaluatorFactory(DatamodelEvaluatorFactory):
    @property
    def model(self):
        return DummyModel()

    @property
    def transform(self):
        return lambda x: x * 2.0


class TransformedDataformerEvaluatorFactory(DataformerEvaluatorFactory):
    @property
    def model(self):
        return DummyTransformer()

    @property
    def transform(self):
        return lambda x: x * 3.0


def test_datamodel_evaluator_factory_transform():
    factory_default = DummyModelEvaluatorFactory()
    ev_default = factory_default.evaluator(device="cpu")
    if torch is not None:
        assert isinstance(ev_default.transform, torch.nn.Identity)

    factory_custom = TransformedModelEvaluatorFactory(spec=dict(capture_final=True))
    ev_custom = factory_custom.evaluator(device="cpu")
    x = torch.ones(2, 4)
    res_custom = ev_custom(x)

    model_direct = DummyModel()
    model_direct.load_state_dict(ev_custom.model.state_dict())
    expected_out = model_direct(x * 2.0)
    torch.testing.assert_close(res_custom["final"], expected_out)


def test_dataformer_evaluator_factory_transform():
    factory_custom = TransformedDataformerEvaluatorFactory(
        spec=dict(capture_blocks=[0], capture_final=True)
    )
    ev_custom = factory_custom.evaluator(device="cpu")
    x = torch.ones(2, 5, 8)
    res_custom = ev_custom(x)

    model_direct = DummyTransformer()
    model_direct.load_state_dict(ev_custom.model.state_dict())
    expected_out = model_direct(x * 3.0)
    torch.testing.assert_close(res_custom["final"], expected_out)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))


