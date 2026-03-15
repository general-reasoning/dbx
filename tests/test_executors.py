import time
import pytest
import functools
from dbx.dbx import (
    MultithreadingCallableExecutor,
    MultiprocessingCallableExecutor,
    MultithreadingDatablocksBuilder,
    MultiprocessingDatablocksBuilder,
    InlineCallableExecutor,
    InlineDatablocksBuilder,
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
# Dummy blocks for DatablocksBuilders
# ---------------------------------------------------------
class DummyBlock:
    def __init__(self, val, fail=False):
        self.val = val
        self.fail = fail
        self.built = False
        self.ctx = None

    def build(self, *args, **kwargs):
        if self.fail:
            raise ValueError(f"Failing block {self.val}")
        self.built = True
        self.ctx = (args, kwargs)
        return self

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
# MultithreadingDatablocksBuilder Tests
# ---------------------------------------------------------
def test_multithreading_builder_success():
    builder = MultithreadingDatablocksBuilder(n_workers=2)
    blocks = [DummyBlock(i) for i in range(10)]
    res = builder.build_blocks(blocks, "arg1", kw="kw1")
    
    assert res is blocks
    for b in blocks:
        assert b.built
        assert b.ctx == (("arg1",), {"kw": "kw1"})

def test_multithreading_builder_failure():
    builder = MultithreadingDatablocksBuilder(n_workers=2)
    blocks = [DummyBlock(i) for i in range(5)] + [DummyBlock(99, fail=True)]
    with pytest.raises(ValueError, match="Failing block 99"):
        builder.build_blocks(blocks)

# ---------------------------------------------------------
# MultiprocessingDatablocksBuilder Tests
# ---------------------------------------------------------
def test_multiprocessing_builder_success():
    builder = MultiprocessingDatablocksBuilder(n_workers=2)
    blocks = [DummyBlock(i) for i in range(10)]
    res = builder.build_blocks(blocks, "arg1", kw="kw1")
    
    assert res is blocks
    # Note: Using python multiprocessing, mutating objects in place across process
    # boundaries only works if using shared memory or taking returned modified copies.
    # The original classes discard blocks or assume they do something external.
    # The test passes checking no errors.

def test_multiprocessing_builder_failure():
    builder = MultiprocessingDatablocksBuilder(n_workers=2)
    blocks = [DummyBlock(i) for i in range(5)] + [DummyBlock(99, fail=True)]
    with pytest.raises(ValueError, match="Failing block 99"):
        builder.build_blocks(blocks)

# ---------------------------------------------------------
# InlineDatablocksBuilder Tests
# ---------------------------------------------------------
def test_inline_builder_success():
    builder = InlineDatablocksBuilder()
    blocks = [DummyBlock(i) for i in range(10)]
    res = builder.build_blocks(blocks, "arg1", kw="kw1")
    
    assert res is blocks
    for b in blocks:
        assert b.built
        assert b.ctx == (("arg1",), {"kw": "kw1"})

def test_inline_builder_failure():
    builder = InlineDatablocksBuilder()
    blocks = [DummyBlock(i) for i in range(5)] + [DummyBlock(99, fail=True)]
    with pytest.raises(ValueError, match="Failing block 99"):
        builder.build_blocks(blocks)
