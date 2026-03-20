"""
Test suite for the DBX Remote functionality.

This module contains unit tests for the Ray-based RPC and distributed execution system
in dbx. The tests verify:

1. test_remote_instantiation: Basic connectivity and remote actor creation via `remote()`.
2. test_remote_apply: Execution of arbitrary callables on remote actors using `run()`.
3. test_remote_callable_executor: Parallel task execution across multiple workers using `RayCallableExecutor`.
4. test_nested_proxying: Handling of objects returned by remote actors (proxies within proxies).
5. test_remote_exception_handling: Correct propagation and reraising of exceptions from remote tasks.
6. test_remote_datablocks_builder: Distributed building of Datablocks using `RayDatablocksBuilder`.

Note: These tests require a clean git repository if DBX_GIT_REPO is set.
"""

import os
import ray
import unittest
import numpy as np
import threading
import queue
import tqdm
import functools
from dbx import datablocks
from dbx.datablocks import remote, RayCallableExecutor, Datablock, RayDatablocksBuilder

class TestRemote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Ray once for all tests in this class
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        os.environ.setdefault('DBX_ROOT', '/tmp/dbx_test')
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_remote_instantiation(self):
        """Verify that remote() returns a valid Remote proxy and can access attributes."""
        r = remote()
        self.assertIsNotNone(r)
        # Verify we can access an attribute from the remote dbx module
        self.assertIsNotNone(r.Logger)

    def test_remote_apply(self):
        """Verify executing a local function on a remote actor via r.run()."""
        r = remote()
        
        def add(a, b):
            return a + b
        
        result = r.run(add, 10, 20)
        self.assertEqual(result, 30)

    def test_remote_callable_executor(self):
        """Verify parallel execution of multiple tasks using RayCallableExecutor."""
        n_workers = 2
        workers = [remote() for _ in range(n_workers)]
        executor = RayCallableExecutor(workers=workers)
        
        def multiply(x, y):
            return x * y
        
        def task(i):
            return multiply(i, 2)
            
        callables = [functools.partial(task, i) for i in range(5)]
        results = executor.execute(callables)
        
        # RayCallableExecutor now returns a flat list, consistent with other executors
        expected = [i * 2 for i in range(5)]
        self.assertEqual(results, expected)

    def test_remote_run_batch(self):
        """Verify executing multiple callables on a remote actor in one round-trip."""
        r = remote()
        def add(a, b): return a + b
        
        batch = [
            (add, (1, 2), {}),
            (add, (10, 20), {}),
            (len, ([1, 2, 3],), {})
        ]
        results = r.run_batch(batch)
        self.assertEqual(results, [3, 30, 3])

    def test_remote_callable_executor_streaming(self):
        """Verify streaming results from RayCallableExecutor."""
        executor = RayCallableExecutor(workers=[remote() for _ in range(2)], batch_size=1)
        def task(i): return i * 2
        
        callables = [functools.partial(task, i) for i in range(10)]
        gen = executor.execute(callables)
        
        results = list(gen)
        self.assertEqual(sorted(results), sorted([i * 2 for i in range(10)]))

    def test_nested_proxying(self):
        """Verify that returning a Datablock (or other dbx objects) from a remote call returns a proxy."""
        r = remote()
        # Datablock is a class in dbx. Calling it remotely should return a Remote handle to the instance.
        db = r.Datablock()
        self.assertTrue(hasattr(db, "_handle"))
        # Verify we can call methods/properties on the nested proxy
        self.assertIsNotNone(db.hash)

    def test_remote_exception_handling(self):
        """Verify that exceptions raised in remote workers are correctly propagated to the client."""
        executor = RayCallableExecutor(workers=[remote()])
        
        def fail():
            raise ValueError("Intentional failure")
        
        with self.assertRaisesRegex(ValueError, "Intentional failure"):
            executor.execute([fail])

    def test_remote_datablocks_builder(self):
        """Verify that RayDatablocksBuilder can build multiple Datablock remotely."""
        class TestBlock(Datablock):
            def __init__(self, **kwargs):
                # Pass built=False to super to ensure it's tracked in parameters
                kwargs.setdefault('built', False)
                super().__init__(**kwargs)

            def valid(self):
                # Always return False to force build() to call __build__()
                return False

            def __build__(self, *args, **kwargs):
                self.built = True

        # Use a small number of threads/workers
        builder = RayDatablocksBuilder(n_workers=2)
        
        # Create a few TestBlocks
        blocks = [TestBlock() for _ in range(3)]
        
        # Initially they should not be marked as built
        for b in blocks:
            self.assertFalse(b.built)
            
        # Build them remotely
        builder.build_blocks(blocks)
        
        # After building, they should be marked as built (state synchronized from remote)
        for b in blocks:
            self.assertTrue(b.built)

    def test_remote_callable_executor_streaming_batch_size(self):
        """Verify streaming results in chunks from RayCallableExecutor."""
        batch_size = 4
        executor = RayCallableExecutor(workers=[remote() for _ in range(2)], batch_size=batch_size)
        def task(i): return i * 2
        
        callables = [functools.partial(task, i) for i in range(10)]
        gen = executor.execute(callables)
        
        chunks = list(gen)
        # 10 items, batch size 4 -> chunks of [4, 4, 2]
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 4)
        self.assertEqual(len(chunks[1]), 4)
        self.assertEqual(len(chunks[2]), 2)
        
        all_results = [res for chunk in chunks for res in chunk]
        self.assertEqual(sorted(all_results), sorted([i * 2 for i in range(10)]))

if __name__ == "__main__":
    unittest.main()
