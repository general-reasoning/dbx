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

@pytest.mark.pinned
def test_multithreading_executor_idle_timeout_keeps_late_results():
    """A result a worker produced may not be discarded.

    `result_idle_timeout_sec` is the gap allowed between consecutive results,
    not a limit on the run, so a callable slower than the gap trips it on a
    perfectly healthy run. It used to `break` and then drain the queue into the
    void: the workers kept running, kept producing, and every payload that
    arrived was thrown away. The results are recoverable, so losing them is a
    bug however the timeout is configured.
    """
    ex = MultithreadingCallableExecutor(n_workers=2, result_idle_timeout_sec=0.1)
    funcs = [
        functools.partial(dummy_func, 1),
        functools.partial(delay_func, 2, delay=0.5),  # outlives the idle timeout
        functools.partial(dummy_func, 3),
    ]
    assert ex.execute(funcs) == [2, 4, 6]


def test_idle_timeout_says_what_it_is_and_names_its_knob(capsys):
    """The visible symptom is a stalled progress bar and a process that looks
    hung, so the one line explaining it has to be at a level that is seen."""
    ex = MultithreadingCallableExecutor(n_workers=2, result_idle_timeout_sec=0.1)
    ex.execute([functools.partial(delay_func, 1, delay=0.5)])
    out = capsys.readouterr().out
    assert 'WARNING' in out
    assert 'INTER-RESULT' in out
    assert 'result_idle_timeout_sec' in out


def test_the_worker_done_knob_no_longer_drives_the_result_loop():
    """The two were one knob, and a value chosen for the worker's wait on its
    stop sentinel was being applied to the main loop's wait for results -- in
    the direction that lost them."""
    ex = MultithreadingCallableExecutor(n_workers=2, worker_done_timeout_sec=0.1)
    assert ex._result_idle_timeout() == ex.RESULT_IDLE_TIMEOUT_SEC
    assert ex.RESULT_IDLE_TIMEOUT_SEC >= 3600

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

@pytest.mark.pinned
def test_multiprocessing_executor_idle_timeout_keeps_late_results():
    """As the threading case: a produced result is not thrown away.

    Across processes as well, where the drain also has to keep reading so a
    worker blocked writing into a full pipe can finish and be joined.
    """
    ex = MultiprocessingCallableExecutor(n_workers=2, result_idle_timeout_sec=0.1)
    funcs = [
        functools.partial(dummy_func, 1),
        functools.partial(delay_func, 2, delay=0.5),  # outlives the idle timeout
        functools.partial(dummy_func, 3),
    ]
    assert ex.execute(funcs) == [2, 4, 6]

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

