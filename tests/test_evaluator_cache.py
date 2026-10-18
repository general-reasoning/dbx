"""A process loads each evaluator's model once, however many tablets evaluate it.

Every tablet of a Featuretable holds its own `ModelEvaluatorBuilder`, evaluated
from the table's spec. `ModelEvaluatorBuilder.evaluator` caches per process, by
``(type(self), self.hash, device)``, so those builders share one evaluator --
and one model -- per device, as the batches within a tablet do.
"""
import os
import uuid
from dataclasses import dataclass

import numpy as np
import pytest
import torch.nn as nn

from dbx import (
    SLICETOPIC,
    Datatablet,
    Datatable,
    ModelEvaluatorBuilder,
    Featuretable,
)
from dbx.datatables import Datacollator

N_TABLETS = 6

#: Where `CountingEvaluatorFactory.model` records each construction: one file per
#: model, named for the process that built it -- so worker processes are counted too.
BUILDS_DIR = 'EVALUATOR_CACHE_BUILDS_DIR'


def builds() -> dict[int, int]:
    """Models built so far, by process id."""
    counts = {}
    for name in os.listdir(os.environ[BUILDS_DIR]):
        pid = int(name.split('-', 1)[0])
        counts[pid] = counts.get(pid, 0) + 1
    return counts


def sample_collator(**spec):
    return Datacollator(spec=dict(columns={'signals': [("samples", "samples")]}, **spec))


class SampleTablet(Datatablet):
    TOPICS = {"samples": SLICETOPIC}

    @dataclass
    class VAR(Datatablet.VAR):
        n_samples: int = 4

    def __build__(self):
        with self.slice_writers({"samples": {"samples": "ndarray:float32"}}) as writers:
            for i in range(self.var.n_samples):
                writers["samples"].write({"samples": np.arange(4, dtype=np.float32) + i})
        return self


class SampleTable(Datatable):
    TABLET = SampleTablet

    @property
    def n_tablets(self):
        return N_TABLETS

    def __tablet__(self, idx: int) -> SampleTablet:
        return self.TABLET(datalake=self.url, spec=dict(n_samples=4 + idx), tag=f"tab_{idx}")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


class CountingEvaluatorFactory(ModelEvaluatorBuilder):
    """Records every model it constructs (`builds`)."""

    @property
    def model(self):
        open(os.path.join(os.environ[BUILDS_DIR], f"{os.getpid()}-{uuid.uuid4().hex}"), 'w').close()
        return Model()


class OtherCountingEvaluatorFactory(CountingEvaluatorFactory):
    """`CountingEvaluatorFactory` by another name: the same spec, so the same hash."""


@pytest.fixture(autouse=True)
def builds_dir(tmp_path, monkeypatch):
    path = tmp_path / 'builds'
    path.mkdir()
    monkeypatch.setenv(BUILDS_DIR, str(path))
    return path


def feature_table(url, **kwargs):
    return Featuretable(
        datalake=url,
        spec=dict(
            upstream=SampleTable(datalake=url, tag="samples").build(),
            evaluator_factory=CountingEvaluatorFactory(spec=dict(capture_final=True)),
            collator=sample_collator(),
        ),
        tag="features",
        **kwargs,
    )


def test_inline_table_builds_its_model_once(tmp_path):
    table = feature_table(str(tmp_path / 'lake'), devices=["cpu"], parallelization='inline').build()
    assert table.valid()
    assert builds() == {os.getpid(): 1}


def test_each_worker_builds_its_model_once(tmp_path):
    table = feature_table(str(tmp_path / 'lake'), devices=["cpu", "cpu"],
                          device_parallelization='multiprocessing').build()
    assert table.valid()
    counts = builds()
    assert os.getpid() not in counts
    assert counts and set(counts.values()) == {1}, f"models built per worker: {counts}"


def test_same_class_and_spec_share_an_evaluator():
    a = CountingEvaluatorFactory(spec=dict(capture_final=True))
    b = CountingEvaluatorFactory(spec=dict(capture_final=True))
    assert a is not b and a.hash == b.hash
    assert a.evaluator(device="cpu") is b.evaluator(device="cpu")
    assert sum(builds().values()) == 1


def test_another_class_with_the_same_spec_gets_its_own_evaluator():
    a, a2 = (CountingEvaluatorFactory(spec=dict(capture_final=True)) for _ in range(2))
    b, b2 = (OtherCountingEvaluatorFactory(spec=dict(capture_final=True)) for _ in range(2))
    assert a.hash == b.hash
    assert a.evaluator(device="cpu") is a2.evaluator(device="cpu")
    assert b.evaluator(device="cpu") is b2.evaluator(device="cpu")
    assert a.evaluator(device="cpu") is not b.evaluator(device="cpu")
    assert sum(builds().values()) == 2


def test_another_hash_gets_its_own_evaluator():
    a, a2 = (CountingEvaluatorFactory(spec=dict(capture_final=True)) for _ in range(2))
    b, b2 = (CountingEvaluatorFactory(spec=dict(capture_final=False)) for _ in range(2))
    assert a.hash != b.hash
    assert a.evaluator(device="cpu") is a2.evaluator(device="cpu")
    assert b.evaluator(device="cpu") is b2.evaluator(device="cpu")
    assert a.evaluator(device="cpu") is not b.evaluator(device="cpu")
    assert sum(builds().values()) == 2


def test_clearing_the_cache_builds_again():
    factory = CountingEvaluatorFactory(spec=dict(capture_final=True))
    first = factory.evaluator(device="cpu")
    ModelEvaluatorBuilder.evaluator_cache_clear()
    assert factory.evaluator(device="cpu") is not first
    assert sum(builds().values()) == 2
