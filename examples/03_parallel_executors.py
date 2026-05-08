"""03_parallel_executors.py — Callable executor demonstration.

Demonstrates:
- InlineCallableExecutor (sequential baseline)
- MultithreadingCallableExecutor (threaded parallelism)
- MultiprocessingCallableExecutor (process-based parallelism)
- Streaming mode (exec_callables_streaming)
"""

import functools
import time

from dbx import (
    callable_executor,
    InlineCallableExecutor,
    MultithreadingCallableExecutor,
    MultiprocessingCallableExecutor,
)


def cpu_work(n: int) -> int:
    """Simulate a CPU-bound task."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def io_work(seconds: float) -> float:
    """Simulate an I/O-bound task."""
    time.sleep(seconds)
    return seconds


def demo_executor(name: str, executor, callables):
    """Run callables on the given executor and report timing."""
    start = time.time()
    results = executor.execute(callables)
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.3f}s  ({len(results)} results)")
    return elapsed


def main():
    n_tasks = 12
    io_duration = 0.2  # seconds per task

    print(f"=== I/O-bound tasks ({n_tasks} × {io_duration}s sleep) ===\n")
    callables = [functools.partial(io_work, io_duration) for _ in range(n_tasks)]

    t_inline = demo_executor(
        "Inline (1 worker)",
        callable_executor('inline'),
        callables,
    )
    t_threaded = demo_executor(
        "Threaded (4 workers)",
        callable_executor('multithreading', n_workers=4),
        callables,
    )

    if t_inline > 0:
        print(f"\n  Speedup: {t_inline / t_threaded:.1f}x")

    # --- Streaming mode ---
    print(f"\n=== Streaming mode (threaded, 4 workers) ===\n")
    executor = MultithreadingCallableExecutor(n_workers=4)
    print("  Results as they arrive:")
    for i, result in enumerate(executor.exec_callables_streaming(callables)):
        print(f"    [{i}] {result}")

    print("\nDone.")


if __name__ == '__main__':
    main()
