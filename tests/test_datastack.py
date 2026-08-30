"""
Tests for Datastack — the abstract block-orchestrating Datablock.

Coverage
--------
1. Subclass must implement blocks().
2. Default __build__ invokes the correct builder on the blocks.
3. All four parallelization strategies are accepted.
4. Invalid parallelization string is rejected.
5. Inline build actually builds every block.
6. Multithreading build actually builds every block.
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
# Minimal concrete block
# ---------------------------------------------------------------------------
class CounterBlock(Datablock):
    """Trivial block that records that it was built."""

    @dataclass
    class VAR(Datablock.VAR):
        idx: int = None

    TOPICS = {'block': 'block.txt'}

    def __build__(self, *args, **kwargs):
        # Write a small marker file so we can verify the build happened
        path = self.path('block', ensure_dirpath=True)
        fs, _ = __import__('fsspec').url_to_fs(path)
        with fs.open(path, "w") as f:
            f.write(f"built:{self.var.idx}")
        return self

    def __read__(self, topic):
        path = self.path('block')
        fs, _ = __import__('fsspec').url_to_fs(path)
        with fs.open(path, "r") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Concrete Datastack for testing
# ---------------------------------------------------------------------------
class SimpleStack(Datastack):
    """A stack that produces N blocks based on total_items / block_size."""

    @dataclass
    class VAR(Datablock.VAR):
        total_items: int = 10
        block_size: int = 3

    TOPICS = {'stack_meta': 'stack_meta.txt'}

    @property
    def n_blocks(self):
        return math.ceil(self.var.total_items / self.var.block_size)

    def __block__(self, idx):
        return CounterBlock(
            url=self.url,
            spec=dict(idx=idx),
        )

    def blocks(self):
        return [self.__block__(i) for i in range(self.n_blocks)]

    def __read__(self, topic):
        return f"stack with {len(self.blocks())} blocks"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDatastackAbstract(unittest.TestCase):
    """Verify the abstract contract."""

    def test_blocks_not_implemented(self):
        """Direct Datastack subclass without blocks() should raise."""
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        class BadStack(Datastack):
            TOPICS = {'bad': 'bad.txt'}

        with tempfile.TemporaryDirectory() as tmp:
            stack = BadStack(url=tmp)
            with self.assertRaises(NotImplementedError):
                stack.blocks()

    def test_invalid_parallelization_rejected(self):
        """Unknown parallelization string raises ValueError."""
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                SimpleStack(url=tmp, parallelization='quantum')


class TestDatastackIsDatablock(unittest.TestCase):
    """Verify Datastack instances are valid Datablocks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_isinstance(self):
        stack = SimpleStack(url=self.tmpdir)
        self.assertIsInstance(stack, Datablock)

    def test_has_hash(self):
        stack = SimpleStack(url=self.tmpdir, spec=dict(total_items=10, block_size=3))
        self.assertIsNotNone(stack.hash)

    def test_has_var(self):
        stack = SimpleStack(url=self.tmpdir, spec=dict(total_items=6, block_size=2))
        self.assertEqual(stack.var.total_items, 6)
        self.assertEqual(stack.var.block_size, 2)


class TestDatastackBlocks(unittest.TestCase):
    """Verify blocks() returns the correct child Datablocks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_block_count(self):
        stack = SimpleStack(url=self.tmpdir, spec=dict(total_items=10, block_size=3))
        blocks = stack.blocks()
        self.assertEqual(len(blocks), 4)  # ceil(10/3) = 4

    def test_block_count_exact(self):
        stack = SimpleStack(url=self.tmpdir, spec=dict(total_items=9, block_size=3))
        blocks = stack.blocks()
        self.assertEqual(len(blocks), 3)  # 9/3 = 3

    def test_blocks_are_datablocks(self):
        stack = SimpleStack(url=self.tmpdir, spec=dict(total_items=4, block_size=2))
        for blk in stack.blocks():
            self.assertIsInstance(blk, Datablock)
            self.assertIsInstance(blk, CounterBlock)

    def test_block_configs(self):
        stack = SimpleStack(url=self.tmpdir, spec=dict(total_items=6, block_size=3))
        blocks = stack.blocks()
        indices = [s.var.idx for s in blocks]
        self.assertEqual(indices, [0, 1])


class TestDatastackBuild(unittest.TestCase):
    """Verify __build__ orchestrates block building correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_inline_build(self):
        """Default (inline) build should build all blocks."""
        stack = SimpleStack(
            url=self.tmpdir,
            spec=dict(total_items=6, block_size=2),
        )
        stack.build()
        # All 3 blocks should have been built
        blocks = stack.blocks()
        for blk in blocks:
            self.assertTrue(blk.valid(), f"Block {blk.var.idx} was not built")
            content = blk.read('block')
            self.assertEqual(content, f"built:{blk.var.idx}")

    def test_multithreading_build(self):
        """Multithreading build should build all blocks."""
        stack = SimpleStack(
            url=self.tmpdir,
            spec=dict(total_items=6, block_size=2),
            parallelization='multithreading',
            n_workers=2,
        )
        stack.build()
        blocks = stack.blocks()
        for blk in blocks:
            self.assertTrue(blk.valid(), f"Block {blk.var.idx} was not built")
            content = blk.read('block')
            self.assertEqual(content, f"built:{blk.var.idx}")

    def test_multiprocessing_build(self):
        """Multiprocessing build should build all blocks (no cross-process state)."""
        stack = SimpleStack(
            url=self.tmpdir,
            spec=dict(total_items=4, block_size=2),
            parallelization='multiprocessing',
            n_workers=2,
        )
        stack.build()
        blocks = stack.blocks()
        for blk in blocks:
            self.assertTrue(blk.valid(), f"Block {blk.var.idx} was not built")


    def test_build_returns_self(self):
        """build() should return the stack itself."""
        stack = SimpleStack(
            url=self.tmpdir,
            spec=dict(total_items=3, block_size=3),
        )
        result = stack.build()
        self.assertIs(result, stack)


class TestDatastackParallelization(unittest.TestCase):
    """Verify parallelization parameter handling."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_default_is_inline(self):
        stack = SimpleStack(url=self.tmpdir)
        from dbx.dataparts import InlineCallableExecutor
        self.assertIs(stack.executor_cls, InlineCallableExecutor)

    def test_explicit_inline(self):
        stack = SimpleStack(url=self.tmpdir, parallelization='inline')
        from dbx.dataparts import InlineCallableExecutor
        self.assertIs(stack.executor_cls, InlineCallableExecutor)

    def test_multithreading(self):
        stack = SimpleStack(url=self.tmpdir, parallelization='multithreading')
        from dbx.dataparts import MultithreadingCallableExecutor
        self.assertIs(stack.executor_cls, MultithreadingCallableExecutor)

    def test_multiprocessing(self):
        stack = SimpleStack(url=self.tmpdir, parallelization='multiprocessing')
        from dbx.dataparts import MultiprocessingCallableExecutor
        self.assertIs(stack.executor_cls, MultiprocessingCallableExecutor)

    def test_ray(self):
        stack = SimpleStack(url=self.tmpdir, parallelization='ray')
        from dbx.dataparts import RayCallableExecutor
        self.assertIs(stack.executor_cls, RayCallableExecutor)

    def test_case_insensitive(self):
        stack = SimpleStack(url=self.tmpdir, parallelization='Multithreading')
        from dbx.dataparts import MultithreadingCallableExecutor
        self.assertIs(stack.executor_cls, MultithreadingCallableExecutor)

    def test_n_workers_stored(self):
        stack = SimpleStack(url=self.tmpdir, n_workers=8)
        self.assertEqual(stack.n_workers, 8)


class TestDatastackClearBlocks(unittest.TestCase):
    """Verify UNSAFE_clear_blocks() removes block data correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def _build_stack(self, **kwargs):
        stack = SimpleStack(
            url=self.tmpdir,
            spec=dict(total_items=6, block_size=2),
            **kwargs,
        )
        stack.build()
        return stack

    def test_inline_clear(self):
        """UNSAFE_clear_blocks with inline parallelization removes block files."""
        stack = self._build_stack()
        # All blocks valid after build
        for blk in stack.blocks():
            self.assertTrue(blk.valid())
        # Clear
        stack.UNSAFE_clear_blocks(OVERRIDE=True)
        # All blocks invalid after clear
        for blk in stack.blocks():
            self.assertFalse(blk.valid())

    def test_multithreading_clear(self):
        """UNSAFE_clear_blocks with multithreading removes block files."""
        stack = self._build_stack(parallelization='multithreading', n_workers=2)
        for blk in stack.blocks():
            self.assertTrue(blk.valid())
        stack.UNSAFE_clear_blocks(OVERRIDE=True)
        for blk in stack.blocks():
            self.assertFalse(blk.valid())

    def test_rebuild_after_clear(self):
        """Blocks can be rebuilt after clearing."""
        stack = self._build_stack()
        stack.UNSAFE_clear_blocks(OVERRIDE=True)
        for blk in stack.blocks():
            self.assertFalse(blk.valid())
        # Rebuild
        for blk in stack.blocks():
            blk.build()
        for blk in stack.blocks():
            self.assertTrue(blk.valid())
            content = blk.read('block')
            self.assertEqual(content, f"built:{blk.var.idx}")

    def test_returns_self(self):
        """UNSAFE_clear_blocks should return the stack itself."""
        stack = self._build_stack()
        result = stack.UNSAFE_clear_blocks(OVERRIDE=True)
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
    """Verify __split__() hook is called before blocks are built."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_pre_stack_called_during_build(self):
        """__split__() should be called when build() runs."""
        class TrackedStack(Datastack):
            pre_stack_called = False

            @dataclass
            class VAR(Datablock.VAR):
                total_items: int = 3
                shard_size: int = 3

            TOPICS = {'tracked_meta': 'tracked_meta.txt'}

            @property
            def n_blocks(self):
                return math.ceil(self.var.total_items / self.var.shard_size)

            def __block__(self, idx):
                return CounterBlock(url=self.url, spec=dict(idx=idx))

            def blocks(self):
                return [self.__block__(i) for i in range(self.n_blocks)]

            def __split__(self):
                TrackedStack.pre_stack_called = True
                return super().__split__()

        TrackedStack.pre_stack_called = False
        stack = TrackedStack(url=self.tmpdir, spec=dict(total_items=3, shard_size=3))
        stack.build()
        self.assertTrue(TrackedStack.pre_stack_called)

    def test_pre_stack_called_before_blocks(self):
        """__split__() should be called before any block is built."""
        call_order = []

        class OrderedBlock(Datablock):
            @dataclass
            class VAR(Datablock.VAR):
                idx: int = None

            TOPICS = {'block': 'block.txt'}

            def __build__(self, *args, **kwargs):
                call_order.append(f"block:{self.var.idx}")
                path = self.path('block', ensure_dirpath=True)
                fs, _ = __import__('fsspec').url_to_fs(path)
                with fs.open(path, "w") as f:
                    f.write(f"built:{self.var.idx}")
                return self

            def __read__(self, topic):
                return "x"

        class OrderedStack(Datastack):
            @dataclass
            class VAR(Datablock.VAR):
                n: int = 2

            TOPICS = {'ordered_meta': 'ordered_meta.txt'}

            @property
            def n_blocks(self):
                return self.var.n

            def __block__(self, idx):
                return OrderedBlock(url=self.url, spec=dict(idx=idx))

            def blocks(self):
                return [self.__block__(i) for i in range(self.n_blocks)]

            def __split__(self, *args, **kwargs):
                call_order.append("pre_stack")
                return super().__split__(*args, **kwargs)

            def __stack__(self, results=None):
                call_order.append("stack")
                return self

        call_order.clear()
        stack = OrderedStack(url=self.tmpdir, spec=dict(n=2))
        stack.build()
        # pre_stack must come first, then blocks, then stack
        self.assertEqual(call_order[0], "pre_stack")
        self.assertIn("block:0", call_order)
        self.assertIn("block:1", call_order)
        self.assertEqual(call_order[-1], "stack")
        pre_idx = call_order.index("pre_stack")
        stack_idx = call_order.index("stack")
        self.assertLess(pre_idx, stack_idx)

    def test_default_split_returns_callables(self):
        """Default __split__() should return (callables, kwargs) tuple."""
        stack = SimpleStack(url=self.tmpdir)
        callables, kwargs = stack.__split__()
        self.assertIsInstance(callables, list)
        self.assertIsInstance(kwargs, dict)


class TestValidateAndCustomCallable(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_validate_default_calls_valid(self):
        block = CounterBlock(url=self.tmpdir, spec=dict(idx=0))
        block.build()
        self.assertTrue(block.validate())

    def test_validate_override_used_by_copy_from(self):
        validated = []
        class CustomBlock(CounterBlock):
            def validate(self):
                validated.append(True)
                return self.valid()

        b1 = CustomBlock(url=os.path.join(self.tmpdir, 'src'), spec=dict(idx=0)).build()
        b2 = CustomBlock(url=os.path.join(self.tmpdir, 'dst'), spec=dict(idx=0))
        b2.UNSAFE_copy_from(b1.anchorkeypath, OVERRIDE=True, validate=True)
        self.assertTrue(validated)

    def test_custom_callable_clear_and_copy_blocks(self):
        cleared_custom = []
        copied_custom = []

        from dbx.datablocks import UNSAFE_clear_block_from_callable, UNSAFE_copy_block_from_callable

        def custom_clear(block, topics, clear_dirpath):
            cleared_custom.append(block.var.idx)
            return UNSAFE_clear_block_from_callable(block, topics, clear_dirpath)

        def custom_copy(block, anchorkeypath, overwrite=False, topicpaths=None, validate=True, always_copy_whole_dirpath=False):
            copied_custom.append(block.var.idx)
            return UNSAFE_copy_block_from_callable(block, anchorkeypath, overwrite, topicpaths, validate, always_copy_whole_dirpath)

        src_stack = SimpleStack(url=os.path.join(self.tmpdir, 'src_stack'), spec=dict(total_items=4, block_size=2)).build()
        dst_stack = SimpleStack(url=os.path.join(self.tmpdir, 'dst_stack'), spec=dict(total_items=4, block_size=2))

        dst_stack.UNSAFE_copy_blocks_from(lambda blk: src_stack.blocks()[blk.var.idx].anchorkeypath, OVERRIDE=True, callable=custom_copy)
        self.assertEqual(len(copied_custom), 2)

        src_stack.UNSAFE_clear_blocks(OVERRIDE=True, callable=custom_clear)
        self.assertEqual(len(cleared_custom), 2)


class TestValidAndRedirectedBlocks(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ.setdefault('DBX_ROOT', self.tmpdir)
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_valid_blocks_and_valid_tabs(self):
        import pandas as pd
        stack = SimpleStack(
            url=os.path.join(self.tmpdir, 'valid_stack'),
            spec=dict(total_items=12, block_size=3),  # 4 blocks: 0, 1, 2, 3
            parallelization='multithreading',
            n_workers=2,
        )
        self.assertIsInstance(stack.valid_blocks(), pd.Series)
        self.assertIsInstance(stack.valid_tabs(), pd.Series)
        self.assertEqual(stack.valid_blocks().tolist(), [False, False, False, False])
        self.assertEqual(stack.valid_tabs().tolist(), [False, False, False, False])

        # Test overrides
        self.assertEqual(stack.valid_blocks(parallelization='inline').tolist(), [False, False, False, False])
        self.assertEqual(stack.valid_tabs(n_workers=1).tolist(), [False, False, False, False])

        # Build blocks 0 and 2
        stack.block(0).build()
        stack.block(2).build()

        self.assertEqual(stack.valid_blocks().tolist(), [True, False, True, False])
        self.assertEqual(stack.valid_tabs().tolist(), [True, False, True, False])
        self.assertEqual(stack.valid_blocks(parallelization='inline').tolist(), [True, False, True, False])

        # Test false_only and true_only
        self.assertEqual(stack.valid_blocks(false_only=True).index.tolist(), [1, 3])
        self.assertEqual(stack.valid_blocks(true_only=True).index.tolist(), [0, 2])
        with self.assertRaises(ValueError):
            stack.valid_blocks(false_only=True, true_only=True)

        self.assertTrue(stack.valid_block(0))
        self.assertFalse(stack.valid_block(1))
        self.assertTrue(stack.valid_tab(2))
        self.assertFalse(stack.valid_tab(3))

        stack.build()
        self.assertEqual(stack.valid_blocks().tolist(), [True, True, True, True])
        self.assertEqual(stack.valid_tabs().tolist(), [True, True, True, True])
        self.assertTrue(stack.valid_blocks(false_only=True).empty)
        self.assertEqual(stack.valid_blocks(true_only=True).index.tolist(), [0, 1, 2, 3])

    def test_redirected_blocks_and_redirected_tabs(self):
        import pandas as pd
        src_stack = SimpleStack(
            url=os.path.join(self.tmpdir, 'red_src_stack'),
            spec=dict(total_items=9, block_size=3),  # 3 blocks: 0, 1, 2
        ).build()

        dst_stack = SimpleStack(
            url=os.path.join(self.tmpdir, 'red_dst_stack'),
            spec=dict(total_items=9, block_size=3),
            parallelization='multithreading',
            n_workers=2,
        )
        self.assertIsInstance(dst_stack.redirected_blocks(), pd.Series)
        self.assertIsInstance(dst_stack.redirected_tabs(), pd.Series)
        self.assertEqual(dst_stack.redirected_blocks().tolist(), [False, False, False])
        self.assertEqual(dst_stack.redirected_tabs().tolist(), [False, False, False])

        # Test overrides
        self.assertEqual(dst_stack.redirected_blocks(parallelization='inline').tolist(), [False, False, False])

        dst_stack.block(1).UNSAFE_redirect(paths=src_stack.block(1).paths(), OVERRIDE=True)

        self.assertTrue(dst_stack.block(1).redirected())
        self.assertFalse(dst_stack.block(0).redirected())
        self.assertTrue(dst_stack.redirected_block(1))
        self.assertFalse(dst_stack.redirected_block(0))
        self.assertTrue(dst_stack.redirected_tab(1))
        self.assertFalse(dst_stack.redirected_tab(0))
        self.assertEqual(dst_stack.redirected_blocks().tolist(), [False, True, False])
        self.assertEqual(dst_stack.redirected_tabs().tolist(), [False, True, False])
        self.assertEqual(dst_stack.redirected_blocks(parallelization='inline').tolist(), [False, True, False])

        # Test false_only and true_only
        self.assertEqual(dst_stack.redirected_blocks(false_only=True).index.tolist(), [0, 2])
        self.assertEqual(dst_stack.redirected_blocks(true_only=True).index.tolist(), [1])
        with self.assertRaises(ValueError):
            dst_stack.redirected_blocks(false_only=True, true_only=True)


if __name__ == "__main__":
    unittest.main()
