"""
Tests for the getenv EnvironmentError and the url/._url_ attribute swap.

Covers:
1. getenv() raises EnvironmentError (not KeyError) with a descriptive message.
2. After the swap: self.url = raw specline, self._url_ = resolved path.
3. Serialization roundtrips preserve self.url (specline), not self._url_.
4. The EnvironmentError propagates clearly through dbx.eval().
"""
import copy
import os
import pickle
import pytest
from dataclasses import dataclass

from dbx.dataparts import env, getenv, eval as dbx_eval
from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_getenv')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class MinBlock(Datablock):
    """Minimal Datablock for testing."""
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'x'"

    def __build__(self):
        path = self.path('out', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")


# ---------------------------------------------------------------------------
# 1. getenv EnvironmentError
# ---------------------------------------------------------------------------

class TestGetenvError:

    def test_raises_environment_error(self):
        """getenv() must raise EnvironmentError, not KeyError."""
        with pytest.raises(EnvironmentError):
            getenv('NONEXISTENT_XYZ_12345')

    def test_error_message_contains_variable_name(self):
        """The error message should name the missing variable."""
        with pytest.raises(EnvironmentError, match='MISSING_VAR_ABC'):
            getenv('MISSING_VAR_ABC')

    def test_error_message_contains_export_hint(self):
        """The error message should suggest how to fix it."""
        with pytest.raises(EnvironmentError, match='export MISSING_VAR_ABC'):
            getenv('MISSING_VAR_ABC')

    def test_no_chained_key_error(self):
        """The 'from None' should suppress the original KeyError chain."""
        try:
            getenv('NONEXISTENT_XYZ_12345')
        except EnvironmentError as e:
            assert e.__cause__ is None

    def test_getenv_succeeds_when_set(self, monkeypatch):
        """getenv() still works fine when the variable exists."""
        monkeypatch.setenv('MY_TEST_VAR', 'hello')
        assert getenv('MY_TEST_VAR') == 'hello'


# ---------------------------------------------------------------------------
# 2. EnvironmentError propagation through eval
# ---------------------------------------------------------------------------

class TestGetenvErrorPropagation:

    def test_eval_propagates_environment_error(self):
        """dbx.eval of a specline referencing a missing env var should
        propagate the EnvironmentError, not a raw KeyError."""
        specline = env('TOTALLY_MISSING_VAR')
        with pytest.raises(EnvironmentError, match='TOTALLY_MISSING_VAR'):
            dbx_eval(specline)

    def test_block_init_with_missing_env_var(self):
        """Constructing a Datablock with url=env('MISSING') should raise
        EnvironmentError with a meaningful message."""
        with pytest.raises(EnvironmentError, match='MISSING_ROOT_VAR'):
            MinBlock(url=env('MISSING_ROOT_VAR'))


# ---------------------------------------------------------------------------
# 3. url / _url_ swap semantics
# ---------------------------------------------------------------------------

class TestUrlSwapSemantics:

    def test_url_holds_raw_specline(self, monkeypatch):
        """self.url should be the raw specline string."""
        monkeypatch.setenv('SWAP_ROOT', '/tmp/swap')
        block = MinBlock(url=env('SWAP_ROOT'))
        assert block.url == "$dbx.getenv('SWAP_ROOT')"

    def test_url_underscore_holds_resolved_path(self, monkeypatch):
        """self._url_ should be the resolved filesystem path."""
        monkeypatch.setenv('SWAP_ROOT', '/tmp/swap')
        block = MinBlock(url=env('SWAP_ROOT'))
        assert block._url_ == '/tmp/swap'

    def test_root_is_resolved(self, monkeypatch):
        """self.root should match the resolved path."""
        monkeypatch.setenv('SWAP_ROOT', '/tmp/swap')
        block = MinBlock(url=env('SWAP_ROOT'))
        assert block.root == '/tmp/swap'

    def test_literal_url_stored_in_url(self):
        """When url is a plain string (not a specline), self.url = literal."""
        block = MinBlock(url='/tmp/literal')
        assert block.url == '/tmp/literal'

    def test_literal_url_resolved_same(self):
        """For a literal url, self._url_ should equal self.url."""
        block = MinBlock(url='/tmp/literal')
        assert block._url_ == '/tmp/literal'


# ---------------------------------------------------------------------------
# 4. Serialization preserves specline (not resolved path)
# ---------------------------------------------------------------------------

class TestUrlSerializationRoundtrip:

    def test_getstate_serializes_specline(self, monkeypatch):
        """__getstate__['url'] should be the raw specline, not resolved."""
        monkeypatch.setenv('SER_ROOT', '/tmp/ser_a')
        block = MinBlock(url=env('SER_ROOT'))
        state = block.__getstate__()
        assert state['url'] == "$dbx.getenv('SER_ROOT')"
        assert state['url'] != '/tmp/ser_a'

    def test_pickle_preserves_specline(self, monkeypatch):
        """Pickle roundtrip must preserve the raw specline."""
        monkeypatch.setenv('SER_ROOT', '/tmp/ser_b')
        block = MinBlock(url=env('SER_ROOT'))
        restored = pickle.loads(pickle.dumps(block))
        assert restored.url == "$dbx.getenv('SER_ROOT')"
        assert restored._url_ == '/tmp/ser_b'
        assert restored.hash == block.hash

    def test_deepcopy_preserves_specline(self, monkeypatch):
        """deepcopy must preserve the raw specline."""
        monkeypatch.setenv('SER_ROOT', '/tmp/ser_c')
        block = MinBlock(url=env('SER_ROOT'))
        restored = copy.deepcopy(block)
        assert restored.url == "$dbx.getenv('SER_ROOT')"
        assert restored._url_ == '/tmp/ser_c'
        assert restored.hash == block.hash

    def test_setstate_roundtrip_preserves_specline(self, monkeypatch):
        """Manual __getstate__/__setstate__ preserves the specline."""
        monkeypatch.setenv('SER_ROOT', '/tmp/ser_d')
        block = MinBlock(url=env('SER_ROOT'))
        state = block.__getstate__()
        restored = MinBlock.__new__(MinBlock)
        restored.__setstate__(state)
        assert restored.url == "$dbx.getenv('SER_ROOT')"
        assert restored._url_ == '/tmp/ser_d'
        assert restored.hash == block.hash

    def test_hash_stable_after_roundtrip(self, monkeypatch):
        """Hash must be the same before and after serialization."""
        monkeypatch.setenv('SER_ROOT', '/tmp/ser_e')
        block = MinBlock(url=env('SER_ROOT'))
        original_hash = block.hash

        # Change the env value — hash should still match because
        # the specline (not the resolved path) is what's hashed.
        monkeypatch.setenv('SER_ROOT', '/tmp/different')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.hash == original_hash

    def test_literal_url_survives_roundtrip(self):
        """Literal (non-specline) URLs should also survive roundtrips."""
        block = MinBlock(url='/tmp/literal_rt')
        state = block.__getstate__()
        assert state['url'] == '/tmp/literal_rt'
        restored = pickle.loads(pickle.dumps(block))
        assert restored.url == '/tmp/literal_rt'
        assert restored.hash == block.hash


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
