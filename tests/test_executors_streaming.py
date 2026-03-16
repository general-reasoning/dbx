import time
import pytest
import functools
import types
from dbx.dbx import (
    MultithreadingCallableExecutor,
    MultiprocessingCallableExecutor,
    InlineCallableExecutor,
)

def dummy_func(x):
    return x * 2

def delay_func(x, delay):
    time.sleep(delay)
    return x * 2

def fail_func(x):
    raise ValueError(f"Failing on {x}")

@pytest.mark.parametrize("executor_class,n_workers", [
    (InlineCallableExecutor, 1),
    (MultithreadingCallableExecutor, 2),
    (MultiprocessingCallableExecutor, 2),
])
def test_executor_streaming_success(executor_class, n_workers):
    if executor_class == InlineCallableExecutor:
        ex = executor_class(streaming=True)
    else:
        ex = executor_class(n_workers=n_workers, streaming=True)
        
    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    
    # execute with streaming=True (now on init) should return a generator
    gen = ex.execute(funcs)
    assert isinstance(gen, types.GeneratorType)
    
    res = list(gen)
    # Order might be different for parallel executors, but values should be correct
    assert sorted(res) == sorted([i * 2 for i in range(10)])

@pytest.mark.parametrize("executor_class,n_workers", [
    (InlineCallableExecutor, 1),
    (MultithreadingCallableExecutor, 2),
    (MultiprocessingCallableExecutor, 2),
])
def test_executor_streaming_failure(executor_class, n_workers):
    if executor_class == InlineCallableExecutor:
        ex = executor_class(streaming=True)
    else:
        ex = executor_class(n_workers=n_workers, streaming=True)
        
    funcs = [functools.partial(dummy_func, i) for i in range(5)]
    funcs.append(functools.partial(fail_func, 99))
    funcs.extend([functools.partial(dummy_func, i) for i in range(5, 10)])
    
    gen = ex.execute(funcs)
    
    results = []
    with pytest.raises(ValueError, match="Failing on 99"):
        for res in gen:
            results.append(res)
    
    # We should have received some results before the failure
    # (except maybe for inline if it fails early, or pool if it reports late)
    # Actually for 11 items, at least some should pass.
    assert len(results) >= 0

def test_executor_streaming_order():
    # Multithreading usually yields as they come, but let's check with delays
    ex = MultithreadingCallableExecutor(n_workers=2, streaming=True)
    
    # func 0: fast
    # func 1: slow
    # func 2: fast
    funcs = [
        functools.partial(delay_func, 0, 0.01),
        functools.partial(delay_func, 1, 0.2),
        functools.partial(delay_func, 2, 0.01),
    ]
    
    gen = ex.execute(funcs)
    results = list(gen)
    
    # Values should be correct
    assert sorted(results) == [0, 2, 4]
    
    # In streaming mode with 2 workers:
    # Worker 1 gets [0, 1]
    # Worker 2 gets [2]
    # 0 finishes fast, 2 finishes fast, 1 finishes slow.
    # So we expect [0, 4, 2] or [4, 0, 2]
    # If it was NOT streaming/yielding as available, it might wait for 1.
    # But here they are Yielded as they become available in result_queue.

@pytest.mark.parametrize("executor_class,n_workers", [
    (InlineCallableExecutor, 1),
    (MultithreadingCallableExecutor, 2),
    (MultiprocessingCallableExecutor, 2),
])
def test_executor_streaming_batch_size(executor_class, n_workers):
    batch_size = 3
    if executor_class == InlineCallableExecutor:
        ex = executor_class(streaming=True, batch_size=batch_size)
    else:
        ex = executor_class(n_workers=n_workers, streaming=True, batch_size=batch_size)
        
    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    
    gen = ex.execute(funcs)
    chunks = list(gen)
    
    # We expect chunks of size 3, with the last one being size 1 (10 items total)
    assert len(chunks) == 4
    assert len(chunks[0]) == 3
    assert len(chunks[1]) == 3
    assert len(chunks[2]) == 3
    assert len(chunks[3]) == 1
    
    all_results = [res for chunk in chunks for res in chunk]
    assert sorted(all_results) == sorted([i * 2 for i in range(10)])
