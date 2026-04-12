"""
Tests for Datastack — the abstract shard-orchestrating Datablock.

Coverage
--------
1. Subclass must implement shards().
2. Default __build__ invokes the correct builder on the shards.
3. All four parallelization strategies are accepted.
4. Invalid parallelization string is rejected.
5. Inline build actually builds every shard.
6. Multithreading build actually builds every shard.
7. n_workers is forwarded to the builder.
8. Datastack itself is a proper Datablock (has hash, root, etc.).
"""
import os
import math
import tempfile
import unittest
from dataclasses import dataclass

from dbx.datablocks import Datablock, Datastack


# ---------------------------------------------------------------------------
# Minimal concrete shard
# ---------------------------------------------------------------------------
class CounterShard(Datablock):
    """Trivial shard that records that it was built."""

    @dataclass
    class CONFIG(Datablock.CONFIG):
        idx: int = None

    TOPICFILE = "shard.txt"

    def __build__(self, *args, **kwargs):
        # Write a small marker file so we can verify the build happened
        path = self.path(ensure_dirpath=True)
        fs, _ = __import__('fsspec').url_to_fs(path)
        with fs.open(path, "w") as f:
            f.write(f"built:{self.cfg.idx}")
        return self

    def __read__(self, topic=None):
        path = self.path()
        fs, _ = __import__('fsspec').url_to_fs(path)
        with fs.open(path, "r") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Concrete Datastack for testing
# ---------------------------------------------------------------------------
class SimpleStack(Datastack):
    """A stack that produces N shards based on total_items / shard_size."""

    @dataclass
    class CONFIG(Datablock.CONFIG):
        total_items: int = 10
        shard_size: int = 3

    TOPICFILE = "stack_meta.txt"

    @property
    def n_shards(self):
        return math.ceil(self.cfg.total_items / self.cfg.shard_size)

    def __shard__(self, idx):
        return CounterShard(
            root=self.root,
            spec=dict(idx=idx),
        )

    def shards(self):
        return [self.__shard__(i) for i in range(self.n_shards)]

    def __read__(self, topic=None):
        return f"stack with {len(self.shards())} shards"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDatastackAbstract(unittest.TestCase):
    """Verify the abstract contract."""

    def test_shards_not_implemented(self):
        """Direct Datastack subclass without shards() should raise."""
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        class BadStack(Datastack):
            TOPICFILE = "bad.txt"

        with tempfile.TemporaryDirectory() as tmp:
            stack = BadStack(root=tmp)
            with self.assertRaises(NotImplementedError):
                stack.shards()

    def test_invalid_parallelization_rejected(self):
        """Unknown parallelization string raises ValueError."""
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                SimpleStack(root=tmp, parallelization='quantum')


class TestDatastackIsDatablock(unittest.TestCase):
    """Verify Datastack instances are valid Datablocks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_isinstance(self):
        stack = SimpleStack(root=self.tmpdir)
        self.assertIsInstance(stack, Datablock)

    def test_has_hash(self):
        stack = SimpleStack(root=self.tmpdir, spec=dict(total_items=10, shard_size=3))
        self.assertIsNotNone(stack.hash)

    def test_has_cfg(self):
        stack = SimpleStack(root=self.tmpdir, spec=dict(total_items=6, shard_size=2))
        self.assertEqual(stack.cfg.total_items, 6)
        self.assertEqual(stack.cfg.shard_size, 2)


class TestDatastackShards(unittest.TestCase):
    """Verify shards() returns the correct child Datablocks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_shard_count(self):
        stack = SimpleStack(root=self.tmpdir, spec=dict(total_items=10, shard_size=3))
        shards = stack.shards()
        self.assertEqual(len(shards), 4)  # ceil(10/3) = 4

    def test_shard_count_exact(self):
        stack = SimpleStack(root=self.tmpdir, spec=dict(total_items=9, shard_size=3))
        shards = stack.shards()
        self.assertEqual(len(shards), 3)  # 9/3 = 3

    def test_shards_are_datablocks(self):
        stack = SimpleStack(root=self.tmpdir, spec=dict(total_items=4, shard_size=2))
        for shard in stack.shards():
            self.assertIsInstance(shard, Datablock)
            self.assertIsInstance(shard, CounterShard)

    def test_shard_configs(self):
        stack = SimpleStack(root=self.tmpdir, spec=dict(total_items=6, shard_size=3))
        shards = stack.shards()
        indices = [s.cfg.idx for s in shards]
        self.assertEqual(indices, [0, 1])


class TestDatastackBuild(unittest.TestCase):
    """Verify __build__ orchestrates shard building correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_inline_build(self):
        """Default (inline) build should build all shards."""
        stack = SimpleStack(
            root=self.tmpdir,
            spec=dict(total_items=6, shard_size=2),
        )
        stack.build()
        # All 3 shards should have been built
        shards = stack.shards()
        for shard in shards:
            self.assertTrue(shard.valid(), f"Shard {shard.cfg.idx} was not built")
            content = shard.read()
            self.assertEqual(content, f"built:{shard.cfg.idx}")

    def test_multithreading_build(self):
        """Multithreading build should build all shards."""
        stack = SimpleStack(
            root=self.tmpdir,
            spec=dict(total_items=6, shard_size=2),
            parallelization='multithreading',
            n_workers=2,
        )
        stack.build()
        shards = stack.shards()
        for shard in shards:
            self.assertTrue(shard.valid(), f"Shard {shard.cfg.idx} was not built")
            content = shard.read()
            self.assertEqual(content, f"built:{shard.cfg.idx}")

    def test_multiprocessing_build(self):
        """Multiprocessing build should build all shards (no cross-process state)."""
        stack = SimpleStack(
            root=self.tmpdir,
            spec=dict(total_items=4, shard_size=2),
            parallelization='multiprocessing',
            n_workers=2,
        )
        stack.build()
        # Verify shards were built by checking files exist
        shards = stack.shards()
        for shard in shards:
            self.assertTrue(shard.valid(), f"Shard {shard.cfg.idx} was not built")

    def test_build_returns_self(self):
        """build() should return the stack itself."""
        stack = SimpleStack(
            root=self.tmpdir,
            spec=dict(total_items=3, shard_size=3),
        )
        result = stack.build()
        self.assertIs(result, stack)


class TestDatastackParallelization(unittest.TestCase):
    """Verify parallelization parameter handling."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_default_is_inline(self):
        stack = SimpleStack(root=self.tmpdir)
        from dbx.datablocks import InlineDatablocksBuilder
        self.assertIs(stack.builder_cls, InlineDatablocksBuilder)

    def test_explicit_inline(self):
        stack = SimpleStack(root=self.tmpdir, parallelization='inline')
        from dbx.datablocks import InlineDatablocksBuilder
        self.assertIs(stack.builder_cls, InlineDatablocksBuilder)

    def test_multithreading(self):
        stack = SimpleStack(root=self.tmpdir, parallelization='multithreading')
        from dbx.datablocks import MultithreadingDatablocksBuilder
        self.assertIs(stack.builder_cls, MultithreadingDatablocksBuilder)

    def test_multiprocessing(self):
        stack = SimpleStack(root=self.tmpdir, parallelization='multiprocessing')
        from dbx.datablocks import MultiprocessingDatablocksBuilder
        self.assertIs(stack.builder_cls, MultiprocessingDatablocksBuilder)

    def test_ray(self):
        stack = SimpleStack(root=self.tmpdir, parallelization='ray')
        from dbx.datablocks import RayDatablocksBuilder
        self.assertIs(stack.builder_cls, RayDatablocksBuilder)

    def test_case_insensitive(self):
        stack = SimpleStack(root=self.tmpdir, parallelization='Multithreading')
        from dbx.datablocks import MultithreadingDatablocksBuilder
        self.assertIs(stack.builder_cls, MultithreadingDatablocksBuilder)

    def test_n_workers_stored(self):
        stack = SimpleStack(root=self.tmpdir, n_workers=8)
        self.assertEqual(stack.n_workers, 8)


class TestDatastackClearShards(unittest.TestCase):
    """Verify UNSAFE_clear_shards() removes shard data correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def _build_stack(self, **kwargs):
        stack = SimpleStack(
            root=self.tmpdir,
            spec=dict(total_items=6, shard_size=2),
            **kwargs,
        )
        stack.build()
        return stack

    def test_inline_clear(self):
        """UNSAFE_clear_shards with inline parallelization removes shard files."""
        stack = self._build_stack()
        # All shards valid after build
        for shard in stack.shards():
            self.assertTrue(shard.valid())
        # Clear
        stack.UNSAFE_clear_shards(OVERRIDE=True)
        # All shards invalid after clear
        for shard in stack.shards():
            self.assertFalse(shard.valid())

    def test_multithreading_clear(self):
        """UNSAFE_clear_shards with multithreading removes shard files."""
        stack = self._build_stack(parallelization='multithreading', n_workers=2)
        for shard in stack.shards():
            self.assertTrue(shard.valid())
        stack.UNSAFE_clear_shards(OVERRIDE=True)
        for shard in stack.shards():
            self.assertFalse(shard.valid())

    def test_rebuild_after_clear(self):
        """Shards can be rebuilt after clearing."""
        stack = self._build_stack()
        stack.UNSAFE_clear_shards(OVERRIDE=True)
        for shard in stack.shards():
            self.assertFalse(shard.valid())
        # Rebuild
        for shard in stack.shards():
            shard.build()
        for shard in stack.shards():
            self.assertTrue(shard.valid())
            content = shard.read()
            self.assertEqual(content, f"built:{shard.cfg.idx}")

    def test_returns_self(self):
        """UNSAFE_clear_shards should return the stack itself."""
        stack = self._build_stack()
        result = stack.UNSAFE_clear_shards(OVERRIDE=True)
        self.assertIs(result, stack)


class TestCallableExecutorFactory(unittest.TestCase):
    """Verify the callable_executor() factory from dataparts."""

    def test_default_is_inline(self):
        from dbx.dataparts import callable_executor, InlineCallableExecutor
        executor = callable_executor(n_workers=1)
        self.assertIsInstance(executor, InlineCallableExecutor)

    def test_explicit_inline(self):
        from dbx.dataparts import callable_executor, InlineCallableExecutor
        executor = callable_executor('inline', n_workers=1)
        self.assertIsInstance(executor, InlineCallableExecutor)

    def test_multithreading(self):
        from dbx.dataparts import callable_executor, MultithreadingCallableExecutor
        executor = callable_executor('multithreading', n_workers=2)
        self.assertIsInstance(executor, MultithreadingCallableExecutor)

    def test_multiprocessing(self):
        from dbx.dataparts import callable_executor, MultiprocessingCallableExecutor
        executor = callable_executor('multiprocessing', n_workers=2)
        self.assertIsInstance(executor, MultiprocessingCallableExecutor)

    def test_unknown_raises(self):
        from dbx.dataparts import callable_executor
        with self.assertRaises(ValueError):
            callable_executor('quantum', n_workers=1)

    def test_case_insensitive(self):
        from dbx.dataparts import callable_executor, MultithreadingCallableExecutor
        executor = callable_executor('Multithreading', n_workers=1)
        self.assertIsInstance(executor, MultithreadingCallableExecutor)


class TestDatastackPreStack(unittest.TestCase):
    """Verify __pre_stack__() hook is called before shards are built."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_pre_stack_called_during_build(self):
        """__pre_stack__() should be called when build() runs."""
        class TrackedStack(Datastack):
            pre_stack_called = False

            @dataclass
            class CONFIG(Datablock.CONFIG):
                total_items: int = 3
                shard_size: int = 3

            TOPICFILE = "tracked_meta.txt"

            @property
            def n_shards(self):
                return math.ceil(self.cfg.total_items / self.cfg.shard_size)

            def __shard__(self, idx):
                return CounterShard(root=self.root, spec=dict(idx=idx))

            def shards(self):
                return [self.__shard__(i) for i in range(self.n_shards)]

            def __pre_stack__(self):
                TrackedStack.pre_stack_called = True
                return self

        TrackedStack.pre_stack_called = False
        stack = TrackedStack(root=self.tmpdir, spec=dict(total_items=3, shard_size=3))
        stack.build()
        self.assertTrue(TrackedStack.pre_stack_called)

    def test_pre_stack_called_before_shards(self):
        """__pre_stack__() should be called before any shard is built."""
        call_order = []

        class OrderedShard(Datablock):
            @dataclass
            class CONFIG(Datablock.CONFIG):
                idx: int = None

            TOPICFILE = "shard.txt"

            def __build__(self, *args, **kwargs):
                call_order.append(f"shard:{self.cfg.idx}")
                path = self.path(ensure_dirpath=True)
                fs, _ = __import__('fsspec').url_to_fs(path)
                with fs.open(path, "w") as f:
                    f.write(f"built:{self.cfg.idx}")
                return self

            def __read__(self, topic=None):
                return "x"

        class OrderedStack(Datastack):
            @dataclass
            class CONFIG(Datablock.CONFIG):
                n: int = 2

            TOPICFILE = "ordered_meta.txt"

            @property
            def n_shards(self):
                return self.cfg.n

            def __shard__(self, idx):
                return OrderedShard(root=self.root, spec=dict(idx=idx))

            def shards(self):
                return [self.__shard__(i) for i in range(self.n_shards)]

            def __pre_stack__(self):
                call_order.append("pre_stack")
                return self

            def __stack__(self):
                call_order.append("stack")
                return self

        call_order.clear()
        stack = OrderedStack(root=self.tmpdir, spec=dict(n=2))
        stack.build()
        # pre_stack must come first, then shards, then stack
        self.assertEqual(call_order[0], "pre_stack")
        self.assertIn("shard:0", call_order)
        self.assertIn("shard:1", call_order)
        self.assertEqual(call_order[-1], "stack")
        pre_idx = call_order.index("pre_stack")
        stack_idx = call_order.index("stack")
        self.assertLess(pre_idx, stack_idx)

    def test_default_pre_stack_returns_self(self):
        """Default __pre_stack__() should return self."""
        stack = SimpleStack(root=self.tmpdir)
        result = stack.__pre_stack__()
        self.assertIs(result, stack)


if __name__ == "__main__":
    unittest.main()
