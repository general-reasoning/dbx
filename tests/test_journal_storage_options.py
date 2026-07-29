"""Tests for storage_options threading through DatajournalEntry and Datajournal."""
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import fsspec

from dbx.datablocks import (
    DatajournalEntry,
    Datajournal,
    Datablock,
    journal,
    parse_storage_options,
    default_storage_options,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.delenv('DBX_STORAGE_OPTIONS', raising=False)


# ---------------------------------------------------------------------------
# parse_storage_options / default_storage_options
# ---------------------------------------------------------------------------

class TestDefaultStorageOptions:

    def test_returns_empty_when_env_unset(self):
        assert default_storage_options() == {}

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv('DBX_STORAGE_OPTIONS', 'account_name=acct;key=val')
        assert default_storage_options() == {'account_name': 'acct', 'key': 'val'}


# ---------------------------------------------------------------------------
# DatajournalEntry storage_options
# ---------------------------------------------------------------------------

def _make_entry(url='/tmp/dbx', anchor='mod.Block', hash_='abc123',
                storage_options=None):
    """Helper to construct a DatajournalEntry with minimal fields."""
    data = {
        'anchor': anchor,
        'hash': hash_,
        'url': url,
        'keyby': 'taghash',
        'tag': None,
    }
    return DatajournalEntry(pd.Series(data), storage_options=storage_options)


class TestDatajournalEntryStorageOptions:

    def test_default_storage_options_empty(self):
        entry = _make_entry()
        assert entry.storage_options == {}

    def test_explicit_storage_options_stored(self):
        so = {'account_name': 'myacct'}
        entry = _make_entry(storage_options=so)
        assert entry.storage_options == so

    def test_root_uses_storage_options(self):
        """storage_options should be passed to fsspec.url_to_fs in root property."""
        so = {'account_name': 'test'}
        entry = _make_entry(url='memory://bucket/data', storage_options=so)
        # Should not raise; memory:// doesn't care about account_name
        root = entry.root
        assert 'bucket/data' in root

    def test_anchorkeypath_local(self):
        entry = _make_entry(url='/tmp/dbx', anchor='mod.Block', hash_='abc123')
        # local paths: no protocol prefix
        path = entry.anchorkeypath
        assert '/tmp/dbx/mod.Block' in path

    def test_anchorkeypath_memory_fs(self):
        entry = _make_entry(url='memory://bucket/data', anchor='mod.Block',
                            hash_='abc123')
        path = entry.anchorkeypath
        # Should include protocol for non-local fs
        assert path.startswith('memory://')
        assert 'mod.Block' in path

    def test_read_passes_storage_options(self):
        """read() should forward storage_options to read_str."""
        so = {'k': 'v'}
        entry = _make_entry(storage_options=so)
        # Simulate a journal entry with a 'spec' field pointing to a .yaml file
        entry['spec'] = '/tmp/fake_spec.yaml'
        with patch('dbx.datablocks.read_yaml', return_value={'x': 1}) as mock_read:
            entry.read('spec')
        mock_read.assert_called_once()
        assert mock_read.call_args.kwargs.get('storage_options') == so


# ---------------------------------------------------------------------------
# Datajournal storage_options propagation
# ---------------------------------------------------------------------------

class TestDatajournalStorageOptions:

    def test_default_storage_options_empty(self):
        df = pd.DataFrame({'hash': ['a'], 'datetime': ['2026-01-01T00-00-00.000000']})
        frame = Datajournal(df)
        assert frame.storage_options == {}

    def test_explicit_storage_options(self):
        so = {'key': 'val'}
        df = pd.DataFrame({'hash': ['a'], 'datetime': ['2026-01-01T00-00-00.000000']})
        frame = Datajournal(df, storage_options=so)
        assert frame.storage_options == so

    def test_get_propagates_storage_options(self):
        """Datajournal.get() should create DatajournalEntry with storage_options."""
        so = {'key': 'val'}
        df = pd.DataFrame({
            'hash': ['abc'],
            'datetime': ['2026-01-01T00-00-00.000000'],
            'anchor': ['mod.Block'],
            'url': ['/tmp/dbx'],
        })
        frame = Datajournal(df, storage_options=so)
        entry = frame.get(0)
        assert isinstance(entry, DatajournalEntry)
        assert entry.storage_options == so

    def test_call_propagates_storage_options(self):
        """Datajournal.__call__() should also propagate storage_options."""
        so = {'key': 'val'}
        df = pd.DataFrame({
            'hash': ['abc'],
            'datetime': ['2026-01-01T00-00-00.000000'],
            'anchor': ['mod.Block'],
            'url': ['/tmp/dbx'],
        })
        frame = Datajournal(df, storage_options=so)
        entry = frame(0)
        assert isinstance(entry, DatajournalEntry)
        assert entry.storage_options == so


# ---------------------------------------------------------------------------
# def journal() storage_options
# ---------------------------------------------------------------------------

class TestJournalFunctionStorageOptions:

    def test_dataframe_path_propagates(self):
        """journal(df, storage_options=...) should pass to Datajournal."""
        df = pd.DataFrame({'hash': ['a'], 'datetime': ['2026-01-01T00-00-00.000000']})
        so = {'k': 'v'}
        result = journal(df, storage_options=so)
        assert isinstance(result, Datajournal)
        assert result.storage_options == so
