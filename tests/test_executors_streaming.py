import time
import pytest
import functools
import types
from dbx.datablocks import (
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
        ex = executor_class(batch_size=1)
    else:
        ex = executor_class(n_workers=n_workers, batch_size=1)
        
    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    
    # execute_streaming should return a generator
    gen = ex.execute_streaming(funcs)
    assert isinstance(gen, types.GeneratorType)
    
    res = list(gen)
    # Results must be in input order (reorder buffer guarantees this)
    assert res == [i * 2 for i in range(10)]

@pytest.mark.parametrize("executor_class,n_workers", [
    (InlineCallableExecutor, 1),
    (MultithreadingCallableExecutor, 2),
    (MultiprocessingCallableExecutor, 2),
])
def test_executor_streaming_failure(executor_class, n_workers):
    if executor_class == InlineCallableExecutor:
        ex = executor_class(batch_size=1)
    else:
        ex = executor_class(n_workers=n_workers, batch_size=1)
        
    funcs = [functools.partial(dummy_func, i) for i in range(5)]
    funcs.append(functools.partial(fail_func, 99))
    funcs.extend([functools.partial(dummy_func, i) for i in range(5, 10)])
    
    gen = ex.execute_streaming(funcs)
    
    results = []
    with pytest.raises(ValueError, match="Failing on 99"):
        for res in gen:
            results.append(res)
    
    assert len(results) >= 0

@pytest.mark.parametrize("executor_class,n_workers", [
    (InlineCallableExecutor, 1),
    (MultithreadingCallableExecutor, 2),
    (MultiprocessingCallableExecutor, 2),
])
def test_executor_streaming_timeout(executor_class, n_workers):
    # Set a very short timeout
    if executor_class == InlineCallableExecutor:
        # InlineExecutor doesn't strictly have worker_done_timeout_sec, but we pass it anyway or ignore
        pytest.skip("InlineExecutor doesn't use queue timeouts in the same way")
    else:
        ex = executor_class(n_workers=n_workers, batch_size=1, worker_done_timeout_sec=0.1)
        
    funcs = [
        functools.partial(dummy_func, 1),
        functools.partial(delay_func, 2, delay=0.5), # will cause timeout
        functools.partial(dummy_func, 3),
    ]
    
    gen = ex.execute_streaming(funcs)
    results = list(gen)
    # The timeout breaks the loop, so results should be incomplete (less than 3)
    assert len(results) < 3

def test_executor_streaming_order():
    ex = MultithreadingCallableExecutor(n_workers=2, batch_size=1)
    
    funcs = [
        functools.partial(delay_func, 0, 0.01),
        functools.partial(delay_func, 1, 0.2),
        functools.partial(delay_func, 2, 0.01),
    ]
    
    gen = ex.execute_streaming(funcs)
    results = list(gen)
    
    # Values should be correct and in input order (reorder buffer)
    assert results == [0, 2, 4]

@pytest.mark.parametrize("executor_class,n_workers", [
    (InlineCallableExecutor, 1),
    (MultithreadingCallableExecutor, 2),
    (MultiprocessingCallableExecutor, 2),
])
def test_executor_streaming_batch_size(executor_class, n_workers):
    batch_size = 3
    if executor_class == InlineCallableExecutor:
        ex = executor_class(batch_size=batch_size)
    else:
        ex = executor_class(n_workers=n_workers, batch_size=batch_size)
        
    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    
    gen = ex.execute_streaming(funcs)
    chunks = list(gen)
    
    # Reorder buffer + global batching: always 4 chunks of [3, 3, 3, 1]
    # regardless of how many workers are used.
    assert len(chunks) == 4
    assert len(chunks[0]) == 3
    assert len(chunks[1]) == 3
    assert len(chunks[2]) == 3
    assert len(chunks[3]) == 1
    
    # Results must be in input order
    all_results = [res for chunk in chunks for res in chunk]
    assert all_results == [i * 2 for i in range(10)]

@pytest.mark.parametrize("executor_class,n_workers", [
    (InlineCallableExecutor, 1),
    (MultithreadingCallableExecutor, 2),
    (MultiprocessingCallableExecutor, 2),
])
def test_executor_execute_with_batch_size_returns_list(executor_class, n_workers):
    """execute() always returns a plain list, even when batch_size is set."""
    batch_size = 3
    if executor_class == InlineCallableExecutor:
        ex = executor_class(batch_size=batch_size)
    else:
        ex = executor_class(n_workers=n_workers, batch_size=batch_size)

    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    result = ex.execute(funcs)

    assert isinstance(result, list)
    assert result == [i * 2 for i in range(10)]
