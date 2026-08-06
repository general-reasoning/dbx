"""
Tests for select_executor (dataparts).

Verifies:
1. Each valid parallelization string returns the correct class.
2. Case-insensitivity works.
3. None maps to inline.
4. Unknown strings raise ValueError.
"""
import os
import pytest

from dbx.dataparts import (
    select_executor,
    callable_executor,
    InlineCallableExecutor,
    MultithreadingCallableExecutor,
    MultiprocessingCallableExecutor,
    TorchMultithreadingCallableExecutor,
    TorchMultiprocessingCallableExecutor,
)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ===========================================================================
# select_executor
# ===========================================================================

class TestSelectExecutor:

    @pytest.mark.parametrize("key,expected", [
        ("inline",                InlineCallableExecutor),
        ("multithreading",        MultithreadingCallableExecutor),
        ("multiprocessing",       MultiprocessingCallableExecutor),
        ("torch_multithreading",  TorchMultithreadingCallableExecutor),
        ("torch_multiprocessing", TorchMultiprocessingCallableExecutor),
    ])
    def test_valid_keys(self, key, expected):
        assert select_executor(key) is expected

    def test_none_returns_inline(self):
        assert select_executor(None) is InlineCallableExecutor

    def test_default_returns_inline(self):
        assert select_executor() is InlineCallableExecutor

    @pytest.mark.parametrize("key", [
        "Inline", "MULTITHREADING", "Torch_Multithreading",
        "TORCH_MULTIPROCESSING", "MultiProcessing",
    ])
    def test_case_insensitive(self, key):
        cls = select_executor(key)
        assert cls is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown parallelization"):
            select_executor("nonexistent")

    def test_callable_executor_creates_instance(self):
        """callable_executor should return an instance, not a class."""
        executor = callable_executor("inline", n_workers=1)
        assert isinstance(executor, InlineCallableExecutor)
