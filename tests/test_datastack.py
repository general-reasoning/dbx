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


if __name__ == "__main__":
    unittest.main()
