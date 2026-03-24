"""
Tests for TorchMultithreadingCallableExecutor, TorchMultiprocessingCallableExecutor,
TorchMultithreadingDatablocksBuilder, and TorchMultiprocessingDatablocksBuilder.

Verifies:
1. _validate_callable: callables with .to() pass, without raise TypeError.
2. TorchMultithreadingCallableExecutor: executes callables with device management.
3. TorchMultiprocessingCallableExecutor: rejects callables without .to().
4. TorchMultithreadingDatablocksBuilder: builds Datablocks with device management.
5. TorchMultiprocessingDatablocksBuilder: rejects blocks without .to().
"""
import os
import pytest
from dataclasses import dataclass

from dbx.databolts import (
    TorchMultithreadingCallableExecutor,
    TorchMultiprocessingCallableExecutor,
)
from dbx.datablocks import (
    Datablock,
    TorchMultithreadingDatablocksBuilder,
    TorchMultiprocessingDatablocksBuilder,
)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_torch_builders')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample callables (for executor tests)
# ---------------------------------------------------------------------------

class CallableWithTo:
    """A callable that implements .to(device) — usable with TorchXXX executors."""
    def __init__(self, value):
        self.value = value
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        return self

    def __call__(self, *args, **kwargs):
        return f"result:{self.value}:device={self.device}"


class CallableWithoutTo:
    """A callable that does NOT implement .to(device)."""
    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return f"result:{self.value}"


# ---------------------------------------------------------------------------
# Sample Datablock subclasses (for builder tests)
# ---------------------------------------------------------------------------

class BlockWithTo(Datablock):
    """Datablock that implements .to(device) as TorchXXX builders require."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def to(self, device):
        self.device = device
        return self

    def __build__(self, *args, **kwargs):
        path = self.path()
        self.dirpath(ensure=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}:device={getattr(self, 'device', 'none')}")
        return self


class BlockWithoutTo(Datablock):
    """Datablock that does NOT implement .to(device)."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'world'"

    def __build__(self, *args, **kwargs):
        path = self.path()
        self.dirpath(ensure=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")
        return self


# ===========================================================================
# 1. _validate_callable (static method on the mixin)
# ===========================================================================

class TestValidateCallable:

    def test_callable_with_to_passes(self):
        c = CallableWithTo(42)
        result = TorchMultithreadingCallableExecutor._validate_callable(c)
        assert result is c

    def test_callable_without_to_raises(self):
        c = CallableWithoutTo(42)
        with pytest.raises(TypeError, match="does not implement .to"):
            TorchMultithreadingCallableExecutor._validate_callable(c)

    def test_returns_callable_for_chaining(self):
        """_validate_callable returns the callable so it can be chained with .to()."""
        c = CallableWithTo(99)
        result = TorchMultithreadingCallableExecutor._validate_callable(c).to('cuda')
        assert result is c
        assert c.device == 'cuda'


# ===========================================================================
# 2. TorchMultithreadingCallableExecutor
# ===========================================================================

class TestTorchMultithreadingCallableExecutor:

    def test_exec_callables_with_to(self):
        """Callables with .to() should execute successfully."""
        callables = [CallableWithTo(i) for i in range(3)]
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables(callables)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert f"result:{i}" in r

    def test_exec_callables_without_to_raises(self):
        """Callables without .to() should fail with TypeError."""
        callables = [CallableWithoutTo(0)]
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        with pytest.raises(TypeError, match="does not implement .to"):
            executor.exec_callables(callables)

    def test_empty_callables_is_noop(self):
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables([])
        assert results == [None] * 0  # empty list

    def test_to_is_called_with_device(self):
        """After execution, callables should have been moved to device then to cpu."""
        callables = [CallableWithTo(0)]
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        executor.exec_callables(callables)
        # After run_items, the callable is moved back to 'cpu'
        assert callables[0].device == 'cpu'

    def test_multiple_devices(self):
        """Callables should be split across devices."""
        callables = [CallableWithTo(i) for i in range(4)]
        executor = TorchMultithreadingCallableExecutor(devices=['cpu', 'cpu'])
        results = executor.exec_callables(callables)
        assert len(results) == 4


# ===========================================================================
# 3. TorchMultiprocessingCallableExecutor
# ===========================================================================

class TestTorchMultiprocessingCallableExecutor:

    def test_exec_callables_without_to_raises(self):
        """Callables without .to() should fail with TypeError."""
        callables = [CallableWithoutTo(0)]
        executor = TorchMultiprocessingCallableExecutor(devices=['cpu'])
        with pytest.raises(TypeError, match="does not implement .to"):
            executor.exec_callables(callables)

    def test_empty_callables_is_noop(self):
        executor = TorchMultiprocessingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables([])
        assert results == []


# ===========================================================================
# 4. TorchMultithreadingDatablocksBuilder (delegates to executor)
# ===========================================================================

class TestTorchMultithreadingBuilder:

    def test_build_blocks_with_to(self, tmp_path):
        """Blocks with .to() should build successfully."""
        blocks = [BlockWithTo(root=str(tmp_path), spec=dict(label=f"item{i}")) for i in range(3)]
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        result = builder.build_blocks(blocks)
        assert result is blocks
        for block in blocks:
            assert block.valid(), f"Block {block.cfg.label} should be valid after build"

    def test_build_blocks_without_to_raises(self, tmp_path):
        """Blocks without .to() should fail with TypeError."""
        blocks = [BlockWithoutTo(root=str(tmp_path))]
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        with pytest.raises(TypeError, match="does not implement .to"):
            builder.build_blocks(blocks)

    def test_empty_blocks_is_noop(self):
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        result = builder.build_blocks([])
        assert result == []

    def test_to_is_called_with_device(self, tmp_path):
        """After build, blocks should have been moved to cpu."""
        blocks = [BlockWithTo(root=str(tmp_path), spec=dict(label='test'))]
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        builder.build_blocks(blocks)
        assert blocks[0].device == 'cpu'


# ===========================================================================
# 5. TorchMultiprocessingDatablocksBuilder (delegates to executor)
# ===========================================================================

class TestTorchMultiprocessingBuilder:

    def test_build_blocks_without_to_raises(self, tmp_path):
        """Blocks without .to() should fail with TypeError."""
        blocks = [BlockWithoutTo(root=str(tmp_path))]
        builder = TorchMultiprocessingDatablocksBuilder(devices=['cpu'])
        with pytest.raises(TypeError, match="does not implement .to"):
            builder.build_blocks(blocks)

    def test_empty_blocks_is_noop(self):
        builder = TorchMultiprocessingDatablocksBuilder(devices=['cpu'])
        result = builder.build_blocks([])
        assert result == []
