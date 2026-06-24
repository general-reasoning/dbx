"""
Tests for TorchMultithreadingCallableExecutor, TorchMultiprocessingCallableExecutor,
TorchMultithreadingDatablocksBuilder, and TorchMultiprocessingDatablocksBuilder.

Verifies:
1. _maybe_to_device: callables with .to() are moved, without .to() pass through.
2. TorchMultithreadingCallableExecutor: executes callables with device management.
3. Callables without .to() execute successfully (permissive mode).
4. n_workers + round-robin device assignment.
5. Work-stealing mode with device management.
6. TorchMultithreadingDatablocksBuilder: builds Datablocks with device management.
"""
import os
import functools
import pytest
from dataclasses import dataclass

torch = pytest.importorskip("torch", reason="torch is required for TorchMulti* tests")

from dbx.dataparts import (
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
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def to(self, device):
        self.device = device
        return self

    def __build__(self, *args, **kwargs):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}:device={getattr(self, 'device', 'none')}")
        return self


class BlockWithoutTo(Datablock):
    """Datablock that does NOT implement .to(device)."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'world'"

    def __build__(self, *args, **kwargs):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")
        return self


# ===========================================================================
# 1. _maybe_to_device (static method on the mixin)
# ===========================================================================

class TestMaybeToDevice:

    def test_callable_with_to_is_moved(self):
        c = CallableWithTo(42)
        result = TorchMultithreadingCallableExecutor._maybe_to_device(c, 'cuda')
        assert result is c
        assert c.device == 'cuda'

    def test_callable_without_to_passes_through(self):
        """Callables without .to() should pass through unchanged (no error)."""
        c = CallableWithoutTo(42)
        result = TorchMultithreadingCallableExecutor._maybe_to_device(c, 'cuda')
        assert result is c



# ===========================================================================
# 2. _device_for_worker (round-robin assignment)
# ===========================================================================

class TestDeviceForWorker:

    def test_single_device(self):
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        assert executor._device_for_worker(0) == 'cpu'
        assert executor._device_for_worker(1) == 'cpu'
        assert executor._device_for_worker(99) == 'cpu'

    def test_multiple_devices_1to1(self):
        executor = TorchMultithreadingCallableExecutor(devices=['cpu', 'cuda:0', 'cuda:1'])
        assert executor._device_for_worker(0) == 'cpu'
        assert executor._device_for_worker(1) == 'cuda:0'
        assert executor._device_for_worker(2) == 'cuda:1'

    def test_round_robin(self):
        """When n_workers > len(devices), devices wrap around."""
        executor = TorchMultithreadingCallableExecutor(
            devices=['cuda:0', 'cuda:1'], n_workers=6
        )
        expected = ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']
        for i, dev in enumerate(expected):
            assert executor._device_for_worker(i) == dev


# ===========================================================================
# 3. n_workers parameter
# ===========================================================================

class TestNWorkersParameter:

    def test_default_n_workers_equals_len_devices(self):
        """When n_workers is not given, it defaults to len(devices)."""
        executor = TorchMultithreadingCallableExecutor(devices=['cpu', 'cpu'])
        assert executor.n_workers == 2

    def test_n_workers_overrides_len_devices(self):
        """n_workers can exceed len(devices)."""
        executor = TorchMultithreadingCallableExecutor(
            devices=['cpu'], n_workers=4
        )
        assert executor.n_workers == 4
        assert len(executor.devices) == 1

    def test_string_device_is_normalised(self):
        """A bare string device should be wrapped in a list."""
        executor = TorchMultithreadingCallableExecutor(devices='cpu', n_workers=3)
        assert executor.devices == ['cpu']
        assert executor.n_workers == 3


# ===========================================================================
# 4. TorchMultithreadingCallableExecutor
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

    def test_exec_callables_without_to_succeeds(self):
        """Callables without .to() should execute successfully (permissive)."""
        callables = [CallableWithoutTo(i) for i in range(3)]
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables(callables)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r == f"result:{i}"

    def test_mixed_callables(self):
        """Mix of callables with and without .to() should all execute."""
        callables = [CallableWithTo(0), CallableWithoutTo(1), CallableWithTo(2)]
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables(callables)
        assert len(results) == 3
        assert "result:0" in results[0]
        assert results[1] == "result:1"
        assert "result:2" in results[2]

    def test_empty_callables_is_noop(self):
        executor = TorchMultithreadingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables([])
        assert results == []

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

    def test_n_workers_greater_than_devices(self):
        """More workers than devices should work via round-robin."""
        callables = [CallableWithTo(i) for i in range(12)]
        executor = TorchMultithreadingCallableExecutor(
            devices=['cpu'], n_workers=4
        )
        results = executor.exec_callables(callables)
        assert len(results) == 12
        for i, r in enumerate(results):
            assert f"result:{i}" in r


# ===========================================================================
# 5. Work-stealing with device management
# ===========================================================================

class TestTorchWorkStealing:

    def test_work_stealing_with_to(self):
        """Work-stealing mode should work with callables that have .to()."""
        callables = [CallableWithTo(i) for i in range(10)]
        executor = TorchMultithreadingCallableExecutor(
            devices=['cpu'], n_workers=3, work_stealing=True
        )
        results = executor.exec_callables(callables)
        assert len(results) == 10
        for i, r in enumerate(results):
            assert f"result:{i}" in r

    def test_work_stealing_without_to(self):
        """Work-stealing mode should work with callables without .to() (permissive)."""
        callables = [CallableWithoutTo(i) for i in range(10)]
        executor = TorchMultithreadingCallableExecutor(
            devices=['cpu'], n_workers=3, work_stealing=True
        )
        results = executor.exec_callables(callables)
        assert len(results) == 10
        for i, r in enumerate(results):
            assert r == f"result:{i}"

    def test_work_stealing_round_robin(self):
        """Work-stealing with n_workers > len(devices) should work."""
        callables = [CallableWithTo(i) for i in range(20)]
        executor = TorchMultithreadingCallableExecutor(
            devices=['cpu'], n_workers=5, work_stealing=True
        )
        results = executor.exec_callables(callables)
        assert len(results) == 20

    def test_work_stealing_streaming(self):
        """Streaming mode should work with work-stealing + devices."""
        callables = [CallableWithTo(i) for i in range(10)]
        executor = TorchMultithreadingCallableExecutor(
            devices=['cpu'], n_workers=3, work_stealing=True
        )
        results = list(executor.exec_callables_streaming(callables))
        assert len(results) == 10

    def test_work_stealing_error_propagation(self):
        """Errors in work-stealing mode should propagate."""
        class FailingCallable:
            def to(self, device): return self
            def __call__(self): raise ValueError("deliberate failure")

        callables = [FailingCallable() for _ in range(5)]
        executor = TorchMultithreadingCallableExecutor(
            devices=['cpu'], n_workers=2, work_stealing=True
        )
        with pytest.raises(ValueError, match="deliberate failure"):
            executor.exec_callables(callables)


# ===========================================================================
# 6. TorchMultiprocessingCallableExecutor
# ===========================================================================

class TestTorchMultiprocessingCallableExecutor:

    def test_exec_callables_without_to_succeeds(self):
        """Callables without .to() should execute successfully (permissive)."""
        callables = [CallableWithoutTo(i) for i in range(3)]
        executor = TorchMultiprocessingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables(callables)
        assert len(results) == 3

    def test_empty_callables_is_noop(self):
        executor = TorchMultiprocessingCallableExecutor(devices=['cpu'])
        results = executor.exec_callables([])
        assert results == []

    def test_n_workers_parameter(self):
        """n_workers should be accepted and override len(devices)."""
        executor = TorchMultiprocessingCallableExecutor(
            devices=['cpu'], n_workers=3
        )
        assert executor.n_workers == 3


# ===========================================================================
# 7. TorchMultithreadingDatablocksBuilder (delegates to executor)
# ===========================================================================

class TestTorchMultithreadingBuilder:

    def test_build_blocks_with_to(self, tmp_path):
        """Blocks with .to() should build successfully."""
        blocks = [BlockWithTo(url=str(tmp_path), spec=dict(label=f"item{i}")) for i in range(3)]
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        result = builder.build_blocks(blocks)
        assert result is blocks
        for block in blocks:
            assert block.valid(), f"Block {block.cfg.label} should be valid after build"

    def test_build_blocks_without_to_raises(self, tmp_path):
        """Blocks without .to() should fail — _TorchBlockCallable_ validates."""
        blocks = [BlockWithoutTo(url=str(tmp_path)) for _ in range(2)]
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        with pytest.raises(TypeError, match="does not implement .to"):
            builder.build_blocks(blocks)

    def test_empty_blocks_is_noop(self):
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        result = builder.build_blocks([])
        assert result == []

    def test_to_is_called_with_device(self, tmp_path):
        """After build, blocks should have been moved to cpu."""
        blocks = [BlockWithTo(url=str(tmp_path), spec=dict(label='test'))]
        builder = TorchMultithreadingDatablocksBuilder(devices=['cpu'])
        builder.build_blocks(blocks)
        assert blocks[0].device == 'cpu'


# ===========================================================================
# 8. TorchMultiprocessingDatablocksBuilder (delegates to executor)
# ===========================================================================

class TestTorchMultiprocessingBuilder:

    def test_build_blocks_without_to_raises(self, tmp_path):
        """Blocks without .to() should fail — _TorchBlockCallable_ validates."""
        blocks = [BlockWithoutTo(url=str(tmp_path))]
        builder = TorchMultiprocessingDatablocksBuilder(devices=['cpu'])
        with pytest.raises(TypeError, match="does not implement .to"):
            builder.build_blocks(blocks)

    def test_empty_blocks_is_noop(self):
        builder = TorchMultiprocessingDatablocksBuilder(devices=['cpu'])
        result = builder.build_blocks([])
        assert result == []
