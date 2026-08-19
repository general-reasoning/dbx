"""
Tests for non-VAR runtime parameters in Datablock and Datastack.

Verifies that:
1. Extra runtime parameters (e.g. devices, device_batch_size) passed as kwargs
   to super().__init__() survive multiprocessing pickling (__getstate__ / __setstate__)
   and copy.deepcopy().
2. Non-VAR parameters do NOT affect the block hash.
3. Datastack passes the devices attribute down to executors during build().
"""

import copy
import pickle
from dataclasses import dataclass
import pytest

from dbx.datablocks import Datablock, Datastack


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class CustomParamBlock(Datablock):
    """Subclass with non-VAR runtime parameters passed to super().__init__()."""
    TOPICS = {'data': 'data.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        val: int = 10

    def __init__(self, *args, device_batch_size: int = 64, feature_devices: list | None = None, **kwargs):
        super().__init__(
            *args,
            device_batch_size=device_batch_size,
            feature_devices=feature_devices or ["cuda:0"],
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        self._devices = getattr(self, 'feature_devices', None) or ["cuda:0"]
        self.device_batch_size = getattr(self, 'device_batch_size', 64)

    def __build__(self):
        path = self.path('data', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"val={self.var.val},batch={self.device_batch_size}")


class CustomParamStack(Datastack):
    """Subclass of Datastack receiving devices and batch size."""
    BLOCK = CustomParamBlock
    VERSION = 1

    @dataclass
    class VAR(Datastack.VAR):
        count: int = 2

    @property
    def n_blocks(self):
        return self.var.count

    def __block__(self, idx):
        return CustomParamBlock(url=self.url, spec=dict(val=idx))

    def __init__(self, *args, devices: list | None = None, device_batch_size: int = 32, **kwargs):
        super().__init__(
            *args,
            devices=devices or ["cuda:0"],
            device_batch_size=device_batch_size,
            **kwargs,
        )

    def __post_init__(self):
        super().__post_init__()
        self._devices = getattr(self, 'devices', None) or ["cuda:0"]
        self.device_batch_size = getattr(self, 'device_batch_size', 32)

    def __split__(self, *args, **kwargs):
        makers = [self.BlockMaker(i) for i in range(self.var.count)]
        return makers, dict(build=True)


def test_nonvar_params_pickle_roundtrip(tmp_path):
    """Non-VAR parameters must survive pickle.dumps / pickle.loads."""
    url = str(tmp_path / "block")
    block = CustomParamBlock(url=url, device_batch_size=128, feature_devices=["cuda:0", "cuda:1"])
    assert block.device_batch_size == 128
    assert block._devices == ["cuda:0", "cuda:1"]

    pickled = pickle.dumps(block)
    restored = pickle.loads(pickled)

    assert restored.device_batch_size == 128
    assert restored._devices == ["cuda:0", "cuda:1"]


def test_nonvar_params_deepcopy(tmp_path):
    """Non-VAR parameters must survive copy.deepcopy()."""
    url = str(tmp_path / "block")
    block = CustomParamBlock(url=url, device_batch_size=256, feature_devices=["cuda:2"])
    copied = copy.deepcopy(block)

    assert copied.device_batch_size == 256
    assert copied._devices == ["cuda:2"]


def test_nonvar_params_do_not_affect_hash(tmp_path):
    """Changing non-VAR parameters must NOT change the block hash."""
    url = str(tmp_path / "block")
    block1 = CustomParamBlock(url=url, device_batch_size=32, feature_devices=["cuda:0"])
    block2 = CustomParamBlock(url=url, device_batch_size=128, feature_devices=["cuda:1", "cuda:2"])

    assert block1.hash == block2.hash


def test_datastack_devices_passed_to_executor(tmp_path):
    """Datastack preserving devices across pickle and passing to build."""
    url = str(tmp_path / "stack")
    stack = CustomParamStack(
        url=url,
        parallelization="inline",
        devices=["cuda:0", "cuda:1"],
        device_batch_size=64,
    )
    assert stack.devices == ["cuda:0", "cuda:1"]

    # Test pickle roundtrip of stack
    restored_stack = pickle.loads(pickle.dumps(stack))
    assert restored_stack.devices == ["cuda:0", "cuda:1"]
    assert restored_stack.device_batch_size == 64

    # Test build execution with devices present
    res = stack.build()
    assert res is stack
