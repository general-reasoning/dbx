"""Tests for the Env class and env() factory for relocatable root references."""
import os
import unittest

from dbx.datablocks import Datablock, Env, env


class TestEnvClass(unittest.TestCase):
    """Unit tests for the Env helper class itself."""

    def test_repr_is_symbolic(self):
        e = Env('MY_VAR')
        self.assertEqual(repr(e), "env('MY_VAR')")

    def test_str_equals_repr(self):
        e = Env('MY_VAR')
        self.assertEqual(str(e), repr(e))

    def test_fspath_resolves_env_var(self):
        os.environ['_DBX_TEST_FSPATH_'] = '/resolved/path'
        try:
            e = Env('_DBX_TEST_FSPATH_')
            self.assertEqual(os.fspath(e), '/resolved/path')
        finally:
            del os.environ['_DBX_TEST_FSPATH_']

    def test_resolve_returns_env_value(self):
        os.environ['_DBX_TEST_RESOLVE_'] = '/some/root'
        try:
            e = Env('_DBX_TEST_RESOLVE_')
            self.assertEqual(e.resolve(), '/some/root')
        finally:
            del os.environ['_DBX_TEST_RESOLVE_']

    def test_resolve_raises_on_missing_var(self):
        e = Env('_DBX_TEST_MISSING_VAR_')
        # Ensure it's not set
        os.environ.pop('_DBX_TEST_MISSING_VAR_', None)
        with self.assertRaises(KeyError):
            e.resolve()

    def test_equality(self):
        a = Env('MY_VAR')
        b = Env('MY_VAR')
        c = Env('OTHER_VAR')
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, 'MY_VAR')

    def test_hash_stable(self):
        a = Env('MY_VAR')
        b = Env('MY_VAR')
        self.assertEqual(hash(a), hash(b))
        # Can be used as dict key / in sets
        d = {a: 1}
        self.assertEqual(d[b], 1)

    def test_env_factory(self):
        e = env('KEY')
        self.assertIsInstance(e, Env)
        self.assertEqual(e.key, 'KEY')


class TestEnvInDatablock(unittest.TestCase):
    """Integration tests: Env passed as root= to a Datablock."""

    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')
        os.environ['_DBX_TEST_ROOT_'] = '/tmp/dbx_env_test'

    def tearDown(self):
        os.environ.pop('_DBX_TEST_ROOT_', None)

    def test_root_resolves_to_env_value(self):
        """self.root should be the resolved filesystem path."""
        class EnvBlock(Datablock):
            pass

        block = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        self.assertEqual(block.root, '/tmp/dbx_env_test')

    def test_root_private_preserves_env_object(self):
        """self._root_ should remain the original Env object."""
        class EnvBlock(Datablock):
            pass

        block = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        self.assertIsInstance(block._root_, Env)
        self.assertEqual(block._root_.key, '_DBX_TEST_ROOT_')

    def test_handle_contains_symbolic_env(self):
        """handle() should contain env('_DBX_TEST_ROOT_'), NOT the resolved path."""
        class EnvBlock(Datablock):
            pass

        block = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        h = block.handle()
        self.assertIn("env('_DBX_TEST_ROOT_')", h)
        self.assertNotIn('/tmp/dbx_env_test', h)

    def test_hashstr_contains_symbolic_env(self):
        """hashstr should contain env('_DBX_TEST_ROOT_'), NOT the resolved path."""
        class EnvBlock(Datablock):
            pass

        block = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        hs = block.hashstr
        self.assertIn("env('_DBX_TEST_ROOT_')", hs)
        self.assertNotIn('/tmp/dbx_env_test', hs)

    def test_hash_stable_across_env_value_changes(self):
        """
        Two blocks with the same Env key but different resolved values
        should produce the same hash (BID stability).
        """
        class EnvBlock(Datablock):
            pass

        os.environ['_DBX_TEST_ROOT_'] = '/machine_A/data'
        block_a = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        hash_a = block_a.hash

        os.environ['_DBX_TEST_ROOT_'] = '/machine_B/data'
        block_b = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        hash_b = block_b.hash

        self.assertEqual(hash_a, hash_b)

    def test_hash_differs_from_literal_root(self):
        """
        A block with root=env('X') and root='/literal' should NOT share
        the same hash (different handle representation).
        """
        class EnvBlock(Datablock):
            pass

        block_env = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        block_lit = EnvBlock(root='/tmp/dbx_env_test')
        self.assertNotEqual(block_env.hash, block_lit.hash)

    def test_getstate_preserves_env(self):
        """__getstate__ should return the Env object for round-tripping."""
        class EnvBlock(Datablock):
            pass

        block = EnvBlock(root=env('_DBX_TEST_ROOT_'))
        state = block.__getstate__()
        self.assertIsInstance(state['root'], Env)
        self.assertEqual(state['root'].key, '_DBX_TEST_ROOT_')

    def test_pickle_roundtrip(self):
        """A pickled-then-unpickled block should preserve Env semantics."""
        import pickle

        block = _PickleEnvBlock(root=env('_DBX_TEST_ROOT_'))
        original_hash = block.hash

        data = pickle.dumps(block)
        restored = pickle.loads(data)

        self.assertEqual(restored.hash, original_hash)
        self.assertIsInstance(restored._root_, Env)
        self.assertEqual(restored.root, '/tmp/dbx_env_test')


# Module-level class so pickle can find it
class _PickleEnvBlock(Datablock):
    pass


if __name__ == '__main__':
    unittest.main()
