import time
import pytest
import functools
from dbx.datablocks import (
    MultithreadingCallableExecutor,
    MultiprocessingCallableExecutor,
    InlineCallableExecutor,
)

# ---------------------------------------------------------
# Dummy callables for CallableExecutors
# ---------------------------------------------------------
def dummy_func(x):
    return x * 2

def delay_func(x, delay):
    time.sleep(delay)
    return x * 2

def fail_func(x):
    raise ValueError(f"Failing on {x}")

# ---------------------------------------------------------
# MultithreadingCallableExecutor Tests
# ---------------------------------------------------------
def test_multithreading_executor_success():
    ex = MultithreadingCallableExecutor(n_workers=2)
    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    res = ex.execute(funcs)
    assert res == [i * 2 for i in range(10)]

def test_multithreading_executor_failure():
    ex = MultithreadingCallableExecutor(n_workers=2)
    funcs = [functools.partial(dummy_func, i) for i in range(5)]
    funcs.append(functools.partial(fail_func, 99))
    with pytest.raises(ValueError, match="Failing on 99"):
        ex.execute(funcs)

def test_multithreading_executor_args():
    ex = MultithreadingCallableExecutor(n_workers=2)
    # Using execute with args/kwargs
    funcs = [functools.partial(delay_func, delay=0.01) for _ in range(5)]
    res = ex.execute(funcs, 5)
    assert res == [10] * 5

def test_multithreading_executor_timeout():
    # Set a very short timeout to simulate deadlock
    ex = MultithreadingCallableExecutor(n_workers=2, worker_done_timeout_sec=0.1)
    funcs = [
        functools.partial(dummy_func, 1),
        functools.partial(delay_func, 2, delay=0.5), # will cause timeout
        functools.partial(dummy_func, 3),
    ]
    res = ex.execute(funcs)
    # Because of the timeout, at least one payload should be None
    assert None in res
    assert len(res) == 3

# ---------------------------------------------------------
# MultiprocessingCallableExecutor Tests
# ---------------------------------------------------------
def test_multiprocessing_executor_success():
    ex = MultiprocessingCallableExecutor(n_workers=2)
    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    res = ex.execute(funcs)
    assert res == [i * 2 for i in range(10)]

def test_multiprocessing_executor_failure():
    ex = MultiprocessingCallableExecutor(n_workers=2)
    funcs = [functools.partial(dummy_func, i) for i in range(5)]
    funcs.append(functools.partial(fail_func, 99))
    # Multiprocessing executor wraps the exception but re-raises
    with pytest.raises(ValueError, match="Failing on 99"):
        ex.execute(funcs)

def test_multiprocessing_executor_args():
    ex = MultiprocessingCallableExecutor(n_workers=2)
    funcs = [functools.partial(delay_func, delay=0.01) for _ in range(5)]
    res = ex.execute(funcs, 5)
    assert res == [10] * 5

def test_multiprocessing_executor_timeout():
    # Set a very short timeout to simulate deadlock
    ex = MultiprocessingCallableExecutor(n_workers=2, worker_done_timeout_sec=0.1)
    funcs = [
        functools.partial(dummy_func, 1),
        functools.partial(delay_func, 2, delay=0.5), # will cause timeout
        functools.partial(dummy_func, 3),
    ]
    res = ex.execute(funcs)
    # Because of the timeout, at least one payload should be None
    assert None in res
    assert len(res) == 3

# ---------------------------------------------------------
# InlineCallableExecutor Tests
# ---------------------------------------------------------
def test_inline_executor_success():
    ex = InlineCallableExecutor()
    funcs = [functools.partial(dummy_func, i) for i in range(10)]
    res = ex.execute(funcs)
    assert res == [i * 2 for i in range(10)]

def test_inline_executor_failure():
    ex = InlineCallableExecutor()
    funcs = [functools.partial(dummy_func, i) for i in range(5)]
    funcs.append(functools.partial(fail_func, 99))
    with pytest.raises(ValueError, match="Failing on 99"):
        ex.execute(funcs)

def test_inline_executor_args():
    ex = InlineCallableExecutor()
    funcs = [functools.partial(delay_func, delay=0.01) for _ in range(5)]
    res = ex.execute(funcs, 5)
    assert res == [10] * 5

# ---------------------------------------------------------
# Shuffle Callables Tests
# ---------------------------------------------------------
def test_multithreading_executor_shuffle():
    ex = MultithreadingCallableExecutor(n_workers=2, shuffle_callables=True)
    # 50 items to ensure shuffling actually happens and we can unshuffle it properly
    funcs = [functools.partial(dummy_func, i) for i in range(50)]
    res = ex.execute(funcs)
    assert res == [i * 2 for i in range(50)]

def test_multiprocessing_executor_shuffle():
    ex = MultiprocessingCallableExecutor(n_workers=2, shuffle_callables=True)
    funcs = [functools.partial(dummy_func, i) for i in range(50)]
    res = ex.execute(funcs)
    assert res == [i * 2 for i in range(50)]

# ---------------------------------------------------------
# Devices Parameter Tests
# ---------------------------------------------------------
def test_executor_devices_parameter():
    ex_inline = InlineCallableExecutor(devices=["cuda:0"])
    assert ex_inline.devices == ["cuda:0"]

    ex_thread = MultithreadingCallableExecutor(n_workers=2, devices=["cuda:0", "cuda:1"])
    assert ex_thread.devices == ["cuda:0", "cuda:1"]

    ex_mp = MultiprocessingCallableExecutor(n_workers=2, devices=["cuda:0", "cuda:1"])
    assert ex_mp.devices == ["cuda:0", "cuda:1"]

