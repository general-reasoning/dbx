"""
Tests for the Env class and env() factory function.

Verifies:
1. Env basics: repr, str, resolve, fspath, equality, hashing.
2. Datablock with root=env('X'): handle/hashstr contain symbolic env('X').
3. Relocatability: changing the env var does not change the hash.
4. Filesystem resolution: self.root resolves to the actual path.
5. Quote round-trip: eval(quote) reconstructs the Datablock correctly.
6. eval/exec context: bare `env(...)` is available in the eval namespace.
"""
import os
import pytest
from dataclasses import dataclass

from dbx.dataparts import Env, env, eval as dbx_eval, exec as dbx_exec
from dbx.datablocks import Datablock
from dbx.datawraps import datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_env')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.setenv('TEST_ROOT', '/tmp/test_root_value')


# ---------------------------------------------------------------------------
# Sample Datablock subclass
# ---------------------------------------------------------------------------

class EnvBlock(Datablock):
    """Minimal block for testing Env."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write(f"built:{self.cfg.label}")


# ---------------------------------------------------------------------------
# 1. Env basics
# ---------------------------------------------------------------------------

class TestEnvBasics:

    def test_repr(self):
        e = env('MY_VAR')
        assert repr(e) == "env('MY_VAR')"

    def test_str_is_repr(self):
        """str() should return the symbolic form, not the resolved value."""
        e = env('TEST_ROOT')
        assert str(e) == "env('TEST_ROOT')"

    def test_resolve(self):
        e = env('TEST_ROOT')
        assert e.resolve() == '/tmp/test_root_value'

    def test_fspath(self):
        e = env('TEST_ROOT')
        assert os.fspath(e) == '/tmp/test_root_value'

    def test_resolve_missing_key_raises(self):
        e = env('NONEXISTENT_VAR_12345')
        with pytest.raises(KeyError):
            e.resolve()

    def test_equality(self):
        assert env('A') == env('A')
        assert env('A') != env('B')

    def test_hash(self):
        assert hash(env('A')) == hash(env('A'))
        s = {env('A'), env('A'), env('B')}
        assert len(s) == 2

    def test_isinstance(self):
        assert isinstance(env('X'), Env)

    def test_os_path_join(self):
        """os.path.join should resolve Env via __fspath__."""
        e = env('TEST_ROOT')
        result = os.path.join(e, 'subdir', 'file.txt')
        assert result == '/tmp/test_root_value/subdir/file.txt'

    def test_idempotent(self):
        """env(env('X')) should return the inner Env unchanged."""
        inner = env('MY_VAR')
        outer = env(inner)
        assert outer is inner

    def test_idempotent_triple(self):
        """Triple wrapping should still return the original Env."""
        e = env(env(env('MY_VAR')))
        assert isinstance(e, Env)
        assert e.key == 'MY_VAR'


# ---------------------------------------------------------------------------
# 2. Datablock with root=env('X')
# ---------------------------------------------------------------------------

class TestEnvInDatablock:

    def test_root_resolves_to_actual_path(self):
        block = EnvBlock(root=env('TEST_ROOT'))
        assert block.root == '/tmp/test_root_value'

    def test_root_underscore_is_env(self):
        """_root_ should store the Env object, not the resolved path."""
        block = EnvBlock(root=env('TEST_ROOT'))
        assert isinstance(block._root_, Env)
        assert block._root_.key == 'TEST_ROOT'

    def test_handle_contains_env(self):
        """handle() should contain env('TEST_ROOT'), not the resolved path."""
        block = EnvBlock(root=env('TEST_ROOT'))
        handle = block.handle()
        assert "env('TEST_ROOT')" in handle
        assert '/tmp/test_root_value' not in handle

    def test_hashstr_contains_env(self):
        """hashstr should contain env('TEST_ROOT')."""
        block = EnvBlock(root=env('TEST_ROOT'))
        assert "env('TEST_ROOT')" in block.hashstr
        assert '/tmp/test_root_value' not in block.hashstr

    def test_quote_contains_env(self):
        """quote() should contain env('TEST_ROOT')."""
        block = EnvBlock(root=env('TEST_ROOT'))
        quote = block.quote()
        assert "env('TEST_ROOT')" in quote

    def test_anchorkeypath_uses_resolved_path(self):
        """anchorkeypath should use the actual resolved path for I/O."""
        block = EnvBlock(root=env('TEST_ROOT'))
        assert block.anchorkeypath.startswith('/tmp/test_root_value')

    def test_build_with_env_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv('TEST_BUILD_ROOT', str(tmp_path))
        block = EnvBlock(root=env('TEST_BUILD_ROOT'))
        assert block.valid() is False
        block.build()
        assert block.valid() is True


# ---------------------------------------------------------------------------
# 3. Relocatability: hash stable across env changes
# ---------------------------------------------------------------------------

class TestEnvRelocatability:

    def test_hash_stable_across_env_values(self, monkeypatch):
        """Changing the env var value should NOT change the hash."""
        monkeypatch.setenv('RELOCATE_ROOT', '/path/a')
        block_a = EnvBlock(root=env('RELOCATE_ROOT'))
        hash_a = block_a.hash

        monkeypatch.setenv('RELOCATE_ROOT', '/path/b')
        block_b = EnvBlock(root=env('RELOCATE_ROOT'))
        hash_b = block_b.hash

        assert hash_a == hash_b

    def test_hash_differs_for_different_env_keys(self, monkeypatch):
        """Different env var keys should produce different hashes."""
        monkeypatch.setenv('ROOT_X', '/same/path')
        monkeypatch.setenv('ROOT_Y', '/same/path')
        block_x = EnvBlock(root=env('ROOT_X'))
        block_y = EnvBlock(root=env('ROOT_Y'))
        assert block_x.hash != block_y.hash

    def test_hash_differs_from_literal_root(self):
        """env('TEST_ROOT') should hash differently than the literal path."""
        block_env = EnvBlock(root=env('TEST_ROOT'))
        block_lit = EnvBlock(root='/tmp/test_root_value')
        assert block_env.hash != block_lit.hash


# ---------------------------------------------------------------------------
# 4. eval/exec context: env() available in eval namespace
# ---------------------------------------------------------------------------

class TestEnvEvalContext:

    def test_env_available_in_dbx_eval(self):
        """env('TEST_ROOT') should be evaluable via dbx's eval mechanism."""
        result = dbx_eval("@dbx.datablocks.env('TEST_ROOT')")
        assert isinstance(result, Env)
        assert result.key == 'TEST_ROOT'

    def test_env_in_datablock_quote_roundtrip(self, monkeypatch):
        """A Datablock's quote should be instantiable via eval."""
        monkeypatch.setenv('RT_ROOT', '/tmp/roundtrip')
        block = EnvBlock(root=env('RT_ROOT'))
        quote = block.quote()
        # The quote starts with $ — strip it for eval
        assert quote.startswith('$')
        restored = dbx_eval(quote)
        assert isinstance(restored, EnvBlock)
        assert isinstance(restored._root_, Env)
        assert restored._root_.key == 'RT_ROOT'
        assert restored.root == '/tmp/roundtrip'
        assert restored.hash == block.hash


# ---------------------------------------------------------------------------
# 5. Env in spec fields
# ---------------------------------------------------------------------------

class EnvSpecBlock(Datablock):
    """Block with an Env-valued spec field."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        data_path: str = "'/default'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write(f"path:{self.cfg.data_path}")


class TestEnvInSpec:

    def test_cfg_resolves_env_to_real_path(self, monkeypatch):
        """cfg.data_path should be the resolved path, not the Env object."""
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(root='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        assert block.cfg.data_path == '/data/resolved'
        assert isinstance(block.cfg.data_path, str)

    def test_spec_stores_env_object(self, monkeypatch):
        """self.spec should retain the Env object."""
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(root='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        assert isinstance(block.spec['data_path'], Env)

    def test_handle_contains_symbolic_env(self, monkeypatch):
        """handle() should contain env('SPEC_PATH'), not the resolved path."""
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(root='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        handle = block.handle()
        assert "env('SPEC_PATH')" in handle
        assert '/data/resolved' not in handle

    def test_hashstr_contains_symbolic_env(self, monkeypatch):
        """hashstr should contain env('SPEC_PATH')."""
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(root='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        assert "env('SPEC_PATH')" in block.hashstr
        assert '/data/resolved' not in block.hashstr

    def test_quote_contains_symbolic_env(self, monkeypatch):
        """quote() should contain env('SPEC_PATH')."""
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(root='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        assert "env('SPEC_PATH')" in block.quote()

    def test_hash_stable_across_env_values(self, monkeypatch):
        """Changing the env var should NOT change the hash."""
        monkeypatch.setenv('SPEC_PATH', '/path/a')
        block_a = EnvSpecBlock(root='/tmp/dbx_test_env',
                               spec=dict(data_path=env('SPEC_PATH')))
        hash_a = block_a.hash

        monkeypatch.setenv('SPEC_PATH', '/path/b')
        block_b = EnvSpecBlock(root='/tmp/dbx_test_env',
                               spec=dict(data_path=env('SPEC_PATH')))
        hash_b = block_b.hash

        assert hash_a == hash_b

    def test_hash_differs_from_literal_spec(self, monkeypatch):
        """env('SPEC_PATH') in spec should hash differently than a literal."""
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block_env = EnvSpecBlock(root='/tmp/dbx_test_env',
                                 spec=dict(data_path=env('SPEC_PATH')))
        block_lit = EnvSpecBlock(root='/tmp/dbx_test_env',
                                 spec=dict(data_path='/data/resolved'))
        assert block_env.hash != block_lit.hash

    def test_build_with_env_spec(self, tmp_path, monkeypatch):
        """Build should use the resolved path in cfg."""
        monkeypatch.setenv('SPEC_PATH', '/data/build_test')
        block = EnvSpecBlock(root=str(tmp_path),
                             spec=dict(data_path=env('SPEC_PATH')))
        block.build()
        assert block.valid() is True
        with open(block.path(), 'r') as f:
            assert f.read() == 'path:/data/build_test'

    def test_combined_env_root_and_spec(self, monkeypatch):
        """Both root and spec can use Env simultaneously."""
        monkeypatch.setenv('MY_ROOT', '/tmp/combined')
        monkeypatch.setenv('MY_DATA', '/data/combined')
        block = EnvSpecBlock(root=env('MY_ROOT'),
                             spec=dict(data_path=env('MY_DATA')))
        assert block.root == '/tmp/combined'
        assert block.cfg.data_path == '/data/combined'
        assert "env('MY_ROOT')" in block.handle()
        assert "env('MY_DATA')" in block.handle()
