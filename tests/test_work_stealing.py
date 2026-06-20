"""Tests for the work_stealing mode in callable executors.

Verifies:
1. Results are correct and in input order.
2. All callables are executed exactly once.
3. work_stealing parameter is accepted by all executor constructors.
4. Error handling works correctly in work-stealing mode.
5. Streaming mode works with work_stealing.
"""
import time
import pytest

from dbx.dataparts import (
    MultithreadingCallableExecutor,
    MultiprocessingCallableExecutor,
    InlineCallableExecutor,
    callable_executor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity(x):
    """Return the argument unchanged.  Used as a trivial callable."""
    return x


def _square(x):
    """Return x squared."""
    return x * x


def _slow(x):
    """Simulate a slow callable for load-balancing tests."""
    time.sleep(x)
    return x


def _fail(x):
    """Always raises an exception."""
    raise ValueError(f"deliberate failure: {x}")


# ---------------------------------------------------------------------------
# MultithreadingCallableExecutor with work_stealing
# ---------------------------------------------------------------------------

class TestMultithreadingWorkStealing:

    def test_basic_results(self):
        """Results should be correct and in input order."""
        import functools
        executor = MultithreadingCallableExecutor(
            n_workers=3, work_stealing=True
        )
        callables = [functools.partial(_identity, i) for i in range(20)]
        results = executor.exec_callables(callables)
        assert results == list(range(20))

    def test_all_callables_executed(self):
        """Every callable should be executed exactly once."""
        import functools
        executor = MultithreadingCallableExecutor(
            n_workers=4, work_stealing=True
        )
        callables = [functools.partial(_square, i) for i in range(50)]
        results = executor.exec_callables(callables)
        assert results == [i * i for i in range(50)]

    def test_empty_callables(self):
        """Empty callable list should return empty results."""
        executor = MultithreadingCallableExecutor(
            n_workers=2, work_stealing=True
        )
        results = executor.exec_callables([])
        assert results == []

    def test_single_callable(self):
        """Single callable should work correctly."""
        import functools
        executor = MultithreadingCallableExecutor(
            n_workers=3, work_stealing=True
        )
        results = executor.exec_callables([functools.partial(_identity, 42)])
        assert results == [42]

    def test_more_workers_than_callables(self):
        """Should work when n_workers > len(callables)."""
        import functools
        executor = MultithreadingCallableExecutor(
            n_workers=10, work_stealing=True
        )
        callables = [functools.partial(_identity, i) for i in range(3)]
        results = executor.exec_callables(callables)
        assert results == [0, 1, 2]

    def test_with_batch_size(self):
        """batch_size should work together with work_stealing."""
        import functools
        executor = MultithreadingCallableExecutor(
            n_workers=3, batch_size=5, work_stealing=True
        )
        callables = [functools.partial(_identity, i) for i in range(20)]
        results = executor.exec_callables(callables)
        assert results == list(range(20))

    def test_error_propagation(self):
        """Exceptions from callables should propagate correctly."""
        import functools
        executor = MultithreadingCallableExecutor(
            n_workers=2, work_stealing=True
        )
        callables = [functools.partial(_fail, i) for i in range(5)]
        with pytest.raises(ValueError, match="deliberate failure"):
            executor.exec_callables(callables)

    def test_streaming_results(self):
        """Streaming mode should yield results in input order."""
        import functools
        executor = MultithreadingCallableExecutor(
            n_workers=3, work_stealing=True
        )
        callables = [functools.partial(_identity, i) for i in range(15)]
        results = list(executor.exec_callables_streaming(callables))
        assert results == list(range(15))


# ---------------------------------------------------------------------------
# Compare work_stealing=False (default) vs work_stealing=True
# ---------------------------------------------------------------------------

class TestWorkStealingVsPartitioned:

    def test_same_results(self):
        """Both modes should produce the same results."""
        import functools
        callables = [functools.partial(_square, i) for i in range(30)]

        executor_partitioned = MultithreadingCallableExecutor(
            n_workers=3, work_stealing=False
        )
        executor_stealing = MultithreadingCallableExecutor(
            n_workers=3, work_stealing=True
        )
        r1 = executor_partitioned.exec_callables(callables)
        r2 = executor_stealing.exec_callables(callables)
        assert r1 == r2

    def test_default_is_partitioned(self):
        """work_stealing should default to False."""
        executor = MultithreadingCallableExecutor(n_workers=2)
        assert executor.work_stealing is False


# ---------------------------------------------------------------------------
# Constructor acceptance
# ---------------------------------------------------------------------------

class TestWorkStealingConstructor:

    def test_multithreading_accepts(self):
        executor = MultithreadingCallableExecutor(
            n_workers=2, work_stealing=True
        )
        assert executor.work_stealing is True

    def test_multithreading_default(self):
        executor = MultithreadingCallableExecutor(n_workers=2)
        assert executor.work_stealing is False

    def test_inline_accepts(self):
        """InlineCallableExecutor should accept work_stealing without error."""
        executor = InlineCallableExecutor(work_stealing=True)
        assert executor.work_stealing is True

    def test_callable_executor_factory(self):
        """callable_executor should forward work_stealing."""
        executor = callable_executor('multithreading', n_workers=2,
                                     work_stealing=True)
        assert executor.work_stealing is True


# ---------------------------------------------------------------------------
# Heterogeneous workloads
# ---------------------------------------------------------------------------

class TestWorkStealingLoadBalance:

    def test_heterogeneous_workload(self):
        """With heterogeneous callables, work-stealing should complete
        and produce correct results.  We don't assert timing here,
        just correctness."""
        import functools
        # Create callables with varying duration: mostly fast, a few slow
        durations = [0.0] * 18 + [0.05, 0.05]  # 18 instant + 2 slow
        callables = [functools.partial(_slow, d) for d in durations]
        executor = MultithreadingCallableExecutor(
            n_workers=4, work_stealing=True
        )
        results = executor.exec_callables(callables)
        assert results == durations
