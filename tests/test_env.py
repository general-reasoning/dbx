"""
Tests for the env() specline factory and getenv() function.

Verifies:
1. env() basics: returns a specline string, idempotent.
2. getenv() resolves environment variables.
3. Datablock with url=env('X'): norm/hashstr contain specline.
4. Relocatability: changing the env var does not change the hash.
5. Spec fields with env(): specline kept in norm, resolved in cfg.
6. Quote round-trip via eval.
"""
import os
import pytest
from dataclasses import dataclass

from dbx.dataparts import env, getenv, eval as dbx_eval
from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_env')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.setenv('TEST_ROOT', '/tmp/test_root_value')


# ---------------------------------------------------------------------------
# Sample Datablock subclasses
# ---------------------------------------------------------------------------

class EnvBlock(Datablock):
    """Minimal block for testing env in root."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write(f"built:{self.cfg.label}")


class EnvSpecBlock(Datablock):
    """Block with an env-valued spec field."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        data_path: str = "'/default'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write(f"path:{self.cfg.data_path}")


# ---------------------------------------------------------------------------
# 1. env() and getenv() basics
# ---------------------------------------------------------------------------

class TestEnvBasics:

    def test_env_returns_specline(self):
        result = env('MY_VAR')
        assert result == "$dbx.getenv('MY_VAR')"

    def test_env_starts_with_dollar(self):
        assert env('X').startswith('$')

    def test_env_idempotent(self):
        """env(env('X')) should return the same specline."""
        inner = env('MY_VAR')
        outer = env(inner)
        assert outer == inner

    def test_env_idempotent_triple(self):
        e = env(env(env('MY_VAR')))
        assert e == "$dbx.getenv('MY_VAR')"

    def test_getenv_resolves(self):
        assert getenv('TEST_ROOT') == '/tmp/test_root_value'

    def test_getenv_missing_raises(self):
        with pytest.raises(EnvironmentError, match="Required environment variable 'NONEXISTENT_VAR_12345' is not set"):
            getenv('NONEXISTENT_VAR_12345')

    def test_dbx_eval_resolves_specline(self):
        """dbx.eval should resolve the specline to the env var value."""
        result = dbx_eval(env('TEST_ROOT'))
        assert result == '/tmp/test_root_value'

    def test_dbx_eval_passes_plain_string(self):
        """dbx.eval should return a plain string as-is."""
        assert dbx_eval('/some/path') == '/some/path'


# ---------------------------------------------------------------------------
# 2. Datablock with url=env('X')
# ---------------------------------------------------------------------------

class TestEnvInRoot:

    def test_root_resolves_to_actual_path(self):
        block = EnvBlock(url=env('TEST_ROOT'))
        assert block.root == '/tmp/test_root_value'

    def test_root_underscore_is_specline(self):
        block = EnvBlock(url=env('TEST_ROOT'))
        assert block.url == "$dbx.getenv('TEST_ROOT')"

    def test_handle_contains_specline(self):
        block = EnvBlock(url=env('TEST_ROOT'))
        handle = block.norm()
        assert "$dbx.getenv('TEST_ROOT')" in handle
        assert '/tmp/test_root_value' not in handle

    def test_hashstr_contains_specline(self):
        block = EnvBlock(url=env('TEST_ROOT'))
        assert "$dbx.getenv('TEST_ROOT')" in block.hashstr
        assert '/tmp/test_root_value' not in block.hashstr

    def test_anchorkeypath_uses_resolved_path(self):
        block = EnvBlock(url=env('TEST_ROOT'))
        assert block.anchorkeypath.startswith('/tmp/test_root_value')

    def test_build_with_env_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv('TEST_BUILD_ROOT', str(tmp_path))
        block = EnvBlock(url=env('TEST_BUILD_ROOT'))
        assert block.valid() is False
        block.build()
        assert block.valid() is True


# ---------------------------------------------------------------------------
# 3. Relocatability: hash stable across env changes
# ---------------------------------------------------------------------------

class TestEnvRelocatability:

    def test_hash_stable_across_env_values(self, monkeypatch):
        monkeypatch.setenv('RELOCATE_ROOT', '/path/a')
        block_a = EnvBlock(url=env('RELOCATE_ROOT'))
        hash_a = block_a.hash

        monkeypatch.setenv('RELOCATE_ROOT', '/path/b')
        block_b = EnvBlock(url=env('RELOCATE_ROOT'))
        hash_b = block_b.hash

        assert hash_a == hash_b

    def test_hash_differs_for_different_env_keys(self, monkeypatch):
        monkeypatch.setenv('ROOT_X', '/same/path')
        monkeypatch.setenv('ROOT_Y', '/same/path')
        block_x = EnvBlock(url=env('ROOT_X'))
        block_y = EnvBlock(url=env('ROOT_Y'))
        assert block_x.hash != block_y.hash

    def test_hash_differs_from_literal_root(self):
        block_env = EnvBlock(url=env('TEST_ROOT'))
        block_lit = EnvBlock(url='/tmp/test_root_value')
        assert block_env.hash != block_lit.hash


# ---------------------------------------------------------------------------
# 4. Env in spec fields
# ---------------------------------------------------------------------------

class TestEnvInSpec:

    def test_cfg_resolves_to_real_path(self, monkeypatch):
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(url='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        assert block.cfg.data_path == '/data/resolved'

    def test_spec_stores_specline(self, monkeypatch):
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(url='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        assert block.spec['data_path'] == "$dbx.getenv('SPEC_PATH')"

    def test_handle_contains_specline(self, monkeypatch):
        monkeypatch.setenv('SPEC_PATH', '/data/resolved')
        block = EnvSpecBlock(url='/tmp/dbx_test_env',
                             spec=dict(data_path=env('SPEC_PATH')))
        handle = block.norm()
        assert "$dbx.getenv('SPEC_PATH')" in handle
        assert '/data/resolved' not in handle

    def test_hash_stable_across_env_values(self, monkeypatch):
        monkeypatch.setenv('SPEC_PATH', '/path/a')
        block_a = EnvSpecBlock(url='/tmp/dbx_test_env',
                               spec=dict(data_path=env('SPEC_PATH')))
        hash_a = block_a.hash

        monkeypatch.setenv('SPEC_PATH', '/path/b')
        block_b = EnvSpecBlock(url='/tmp/dbx_test_env',
                               spec=dict(data_path=env('SPEC_PATH')))
        assert block_a.hash == block_b.hash

    def test_build_with_env_spec(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SPEC_PATH', '/data/build_test')
        block = EnvSpecBlock(url=str(tmp_path),
                             spec=dict(data_path=env('SPEC_PATH')))
        block.build()
        assert block.valid() is True
        with open(block.path(), 'r') as f:
            assert f.read() == 'path:/data/build_test'

    def test_combined_env_root_and_spec(self, monkeypatch):
        monkeypatch.setenv('MY_ROOT', '/tmp/combined')
        monkeypatch.setenv('MY_DATA', '/data/combined')
        block = EnvSpecBlock(url=env('MY_ROOT'),
                             spec=dict(data_path=env('MY_DATA')))
        assert block.root == '/tmp/combined'
        assert block.cfg.data_path == '/data/combined'
        assert "$dbx.getenv('MY_ROOT')" in block.norm()
        assert "$dbx.getenv('MY_DATA')" in block.norm()


# ---------------------------------------------------------------------------
# 5. Quote round-trip
# ---------------------------------------------------------------------------

class TestEnvQuoteRoundtrip:

    def test_quote_roundtrip_with_env_root(self, monkeypatch):
        monkeypatch.setenv('RT_ROOT', '/tmp/roundtrip')
        block = EnvBlock(url=env('RT_ROOT'))
        quote = block.quote()
        assert quote.startswith('$')
        restored = dbx_eval(quote)
        assert isinstance(restored, EnvBlock)
        assert restored.url == "$dbx.getenv('RT_ROOT')"
        assert restored.root == '/tmp/roundtrip'
        assert restored.hash == block.hash
