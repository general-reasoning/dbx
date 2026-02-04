"""
Test suite for the DBX Remote functionality.

This module contains unit tests for the Ray-based RPC and distributed execution system
in dbx. The tests verify:

1. test_remote_instantiation: Basic connectivity and remote actor creation via `remote()`.
2. test_remote_apply: Execution of arbitrary callables on remote actors using `run()`.
3. test_remote_callable_executor: Parallel task execution across multiple workers using `RemoteCallableExecutor`.
4. test_nested_proxying: Handling of objects returned by remote actors (proxies within proxies).
5. test_remote_exception_handling: Correct propagation and reraising of exceptions from remote tasks.
6. test_remote_datablocks_builder: Distributed building of Datablocks using `RemoteDatablocksBuilder`.

Note: These tests require a clean git repository if DBXGITREPO is set.
"""

import os
import ray
import unittest
import numpy as np
import threading
import queue
import tqdm
import functools
from dbx import dbx
from dbx.dbx import remote, RemoteCallableExecutor, Datablock, RemoteDatablocksBuilder

class TestRemote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Ray once for all tests in this class
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
        """Verify parallel execution of multiple tasks using RemoteCallableExecutor."""
        n_threads = 2
        executor = RemoteCallableExecutor(n_threads=n_threads)
        
        def multiply(x, y):
            return x * y
        
        def task(i):
            return multiply(i, 2)
            
        callables = [functools.partial(task, i) for i in range(5)]
        results = executor.execute(callables)
        
        # RemoteCallableExecutor returns a list of lists: [[res1], [res2], ...]
        expected = [[i * 2] for i in range(5)]
        self.assertEqual(results, expected)

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
        executor = RemoteCallableExecutor(n_threads=1)
        
        def fail():
            raise ValueError("Intentional failure")
        
        with self.assertRaisesRegex(ValueError, "Intentional failure"):
            executor.execute([fail])

    def test_remote_datablocks_builder(self):
        """Verify that RemoteDatablocksBuilder can build multiple Datablocks remotely."""
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
        builder = RemoteDatablocksBuilder(n_threads=2)
        
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

if __name__ == "__main__":
    unittest.main()
