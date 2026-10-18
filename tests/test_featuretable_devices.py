"""A Featuretable checks its devices before any worker starts: GPUs it can see, and a CPU for each."""
import os

import pytest

from test_featuretablet import DummySampleTable, DummyModelEvaluatorFactory, sample_collator  # noqa: E402

from dbx import Featuretable
import dbx.featuretables as featuretables


@pytest.fixture
def three_gpus(monkeypatch):
    """This process sees cuda:0..cuda:2."""
    monkeypatch.setattr(featuretables.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(featuretables.torch.cuda, 'device_count', lambda: 3)


def feature_table(url, devices):
    return Featuretable(
        datalake=url,
        spec=dict(
            upstream=DummySampleTable(datalake=url, tag="samples"),
            evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
            collator=sample_collator(),
        ),
        devices=devices,
        tag="features",
    )


def test_gpus_with_fewer_cpus_than_them_raise(tmp_path, three_gpus, monkeypatch):
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {16})
    table = feature_table(str(tmp_path), ['cuda:0', 'cuda:1', 'cuda:2'])
    with pytest.raises(RuntimeError, match=r"3 GPU workers.*1 CPU\(s\).*one CPU per GPU"):
        table._check_devices_()


def test_a_cpu_per_gpu_passes(tmp_path, three_gpus, monkeypatch):
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0, 1, 2})
    feature_table(str(tmp_path), ['cuda:0', 'cuda:1', 'cuda:2'])._check_devices_()


def test_cpu_devices_need_no_cpu_each(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: {0})
    feature_table(str(tmp_path), ['cpu', 'cpu'])._check_devices_()
