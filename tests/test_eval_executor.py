"""
Tests for worker-side dbx.eval() resolution of ctx_args/ctx_kwargs
in all CallableExecutor variants.

Verifies that specline expressions (prefixed with @, $, #) passed as
ctx_args or ctx_kwargs are resolved inside the worker before callables
are invoked.
"""

import os
import functools
import unittest

from dbx.dataparts import (
    eval as dbx_eval,
    InlineCallableExecutor,
    MultithreadingCallableExecutor,
    MultiprocessingCallableExecutor,
)


class TestDbxEvalDict(unittest.TestCase):
    """Verify that dbx.eval() now handles dicts recursively."""

    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        os.environ['EVAL_EXEC_VAL'] = '/resolved/path'

    def test_eval_dict_resolves_specline_values(self):
        d = {'key': "$dbx.getenv('EVAL_EXEC_VAL')", 'plain': 'hello'}
        result = dbx_eval(d)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['key'], '/resolved/path')
        self.assertEqual(result['plain'], 'hello')

    def test_eval_dict_passthrough_non_string_values(self):
        d = {'num': 42, 'lst': [1, 2, 3]}
        result = dbx_eval(d)
        self.assertEqual(result['num'], 42)
        self.assertEqual(result['lst'], [1, 2, 3])

    def test_eval_nested_dict(self):
        d = {'outer': {'inner': "$dbx.getenv('EVAL_EXEC_VAL')"}}
        result = dbx_eval(d)
        self.assertEqual(result['outer']['inner'], '/resolved/path')


def _capture_args_callable(*args, **kwargs):
    """A simple callable that returns its received args and kwargs."""
    return (args, kwargs)


class TestEvalCtxArgsInline(unittest.TestCase):
    """Test dbx.eval() of ctx_args/ctx_kwargs in InlineCallableExecutor."""

    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        os.environ['EVAL_EXEC_A'] = '/data/a'
        os.environ['EVAL_EXEC_B'] = '/data/b'

    def test_specline_ctx_arg_resolved(self):
        executor = InlineCallableExecutor(n_workers=1)
        callables = [_capture_args_callable]
        results = executor.exec_callables(
            callables,
            "$dbx.getenv('EVAL_EXEC_A')",
        )
        args, kwargs = results[0]
        self.assertEqual(args[0], '/data/a')

    def test_specline_ctx_kwarg_resolved(self):
        executor = InlineCallableExecutor(n_workers=1)
        callables = [_capture_args_callable]
        results = executor.exec_callables(
            callables,
            root="$dbx.getenv('EVAL_EXEC_B')",
        )
        args, kwargs = results[0]
        self.assertEqual(kwargs['root'], '/data/b')

    def test_plain_args_unchanged(self):
        executor = InlineCallableExecutor(n_workers=1)
        callables = [_capture_args_callable]
        results = executor.exec_callables(callables, 'plain', num=42)
        args, kwargs = results[0]
        self.assertEqual(args[0], 'plain')
        self.assertEqual(kwargs['num'], 42)

    def test_streaming_specline_resolved(self):
        executor = InlineCallableExecutor(n_workers=1)
        callables = [_capture_args_callable]
        results = list(executor.exec_callables_streaming(
            callables,
            "$dbx.getenv('EVAL_EXEC_A')",
        ))
        args, kwargs = results[0]
        self.assertEqual(args[0], '/data/a')


class TestEvalCtxArgsMultithreading(unittest.TestCase):
    """Test dbx.eval() of ctx_args/ctx_kwargs in MultithreadingCallableExecutor."""

    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        os.environ['EVAL_EXEC_MT'] = '/mt/resolved'

    def test_specline_ctx_arg_resolved(self):
        executor = MultithreadingCallableExecutor(n_workers=2)
        callables = [_capture_args_callable, _capture_args_callable]
        results = executor.exec_callables(
            callables,
            "$dbx.getenv('EVAL_EXEC_MT')",
        )
        for args, kwargs in results:
            self.assertEqual(args[0], '/mt/resolved')

    def test_specline_ctx_kwarg_resolved(self):
        executor = MultithreadingCallableExecutor(n_workers=1)
        callables = [_capture_args_callable]
        results = executor.exec_callables(
            callables,
            path="$dbx.getenv('EVAL_EXEC_MT')",
        )
        args, kwargs = results[0]
        self.assertEqual(kwargs['path'], '/mt/resolved')

    def test_plain_passthrough(self):
        executor = MultithreadingCallableExecutor(n_workers=1)
        callables = [_capture_args_callable]
        results = executor.exec_callables(callables, 'hello', x=99)
        args, kwargs = results[0]
        self.assertEqual(args[0], 'hello')
        self.assertEqual(kwargs['x'], 99)


class TestEvalCtxArgsMultiprocessing(unittest.TestCase):
    """Test dbx.eval() of ctx_args/ctx_kwargs in MultiprocessingCallableExecutor."""

    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        os.environ['EVAL_EXEC_MP'] = '/mp/resolved'

    def test_specline_ctx_arg_resolved(self):
        executor = MultiprocessingCallableExecutor(n_workers=1, start_method='fork')
        callables = [_capture_args_callable]
        results = executor.exec_callables(
            callables,
            "$dbx.getenv('EVAL_EXEC_MP')",
        )
        args, kwargs = results[0]
        self.assertEqual(args[0], '/mp/resolved')

    def test_plain_passthrough(self):
        executor = MultiprocessingCallableExecutor(n_workers=1, start_method='fork')
        callables = [_capture_args_callable]
        results = executor.exec_callables(callables, 'hello', x=99)
        args, kwargs = results[0]
        self.assertEqual(args[0], 'hello')
        self.assertEqual(kwargs['x'], 99)


if __name__ == "__main__":
    unittest.main()
