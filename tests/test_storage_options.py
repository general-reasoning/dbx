"""Tests for _parse_storage_options and DBX_STORAGE_OPTIONS env-var fallback."""
import os
import pytest
from dbx.datablocks import Datablock
from dbx.dataparts import parse_storage_options


# ---------------------------------------------------------------------------
# _parse_storage_options unit tests
# ---------------------------------------------------------------------------

class TestParseStorageOptions:

    def test_none_returns_empty(self):
        assert parse_storage_options(None) == {}

    def test_empty_string_returns_empty(self):
        assert parse_storage_options("") == {}

    def test_single_pair(self):
        assert parse_storage_options("account_name=myacct") == {
            'account_name': 'myacct',
        }

    def test_multiple_pairs(self):
        result = parse_storage_options("account_name=myacct;account_key=secret")
        assert result == {'account_name': 'myacct', 'account_key': 'secret'}

    def test_whitespace_is_stripped(self):
        result = parse_storage_options("  k1 = v1 ; k2=v2  ")
        assert result == {'k1': 'v1', 'k2': 'v2'}

    def test_trailing_semicolon_ignored(self):
        result = parse_storage_options("a=b;")
        assert result == {'a': 'b'}

    def test_value_with_equals(self):
        """Values may contain '=' (e.g. base64 keys)."""
        result = parse_storage_options("key=abc=def==")
        assert result == {'key': 'abc=def=='}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="expected key=value"):
            parse_storage_options("no_equals_here")


# ---------------------------------------------------------------------------
# Datablock env-var fallback integration
# ---------------------------------------------------------------------------

class SimpleBlock(Datablock):
    def __build__(self):
        pass


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class TestStorageOptionsFallback:

    def test_explicit_storage_options_used(self):
        """Explicit storage_options= should take precedence over env var."""
        block = SimpleBlock(storage_options={'x': '1'})
        assert block.storage_options == {'x': '1'}

    def test_env_var_fallback(self, monkeypatch):
        """When storage_options is None, fall back to DBX_STORAGE_OPTIONS."""
        monkeypatch.setenv('DBX_STORAGE_OPTIONS', 'account_name=acct;sas_token=tok')
        block = SimpleBlock()
        assert block.storage_options == {
            'account_name': 'acct',
            'sas_token': 'tok',
        }

    def test_no_env_var_defaults_to_empty(self, monkeypatch):
        """Without env var or explicit arg, storage_options should be {}."""
        monkeypatch.delenv('DBX_STORAGE_OPTIONS', raising=False)
        block = SimpleBlock()
        assert block.storage_options == {}

    def test_explicit_empty_dict_skips_env(self, monkeypatch):
        """Passing storage_options={} explicitly should NOT fall back to env."""
        monkeypatch.setenv('DBX_STORAGE_OPTIONS', 'k=v')
        block = SimpleBlock(storage_options={})
        assert block.storage_options == {}
