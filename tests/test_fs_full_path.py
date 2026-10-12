"""Tests for fs_full_path: protocol restoration including Azure account/credentials."""
import pytest
from unittest.mock import MagicMock

import fsspec

from dbx.databits import fs_full_path


# ---------------------------------------------------------------------------
# Local filesystem — bare path returned unchanged
# ---------------------------------------------------------------------------

class TestFsFullPathLocal:

    def test_local_returns_bare_path(self):
        fs = fsspec.filesystem('file')
        assert fs_full_path(fs, '/tmp/data/file.parquet') == '/tmp/data/file.parquet'

    def test_local_with_relative_path(self):
        fs = fsspec.filesystem('file')
        assert fs_full_path(fs, 'relative/path') == 'relative/path'


# ---------------------------------------------------------------------------
# memory:// filesystem — protocol restored via unstrip_protocol
# ---------------------------------------------------------------------------

class TestFsFullPathMemory:

    def test_memory_restores_protocol(self):
        fs = fsspec.filesystem('memory')
        result = fs_full_path(fs, '/some/path')
        assert result.startswith('memory://')
        assert '/some/path' in result


# ---------------------------------------------------------------------------
# Azure (abfs/abfss) — account name must be restored
# ---------------------------------------------------------------------------

class TestFsFullPathAzure:

    @staticmethod
    def _make_azure_fs(*, account_name='myaccount', protocol='abfs'):
        """Create a mock Azure filesystem with account_name."""
        fs = MagicMock()
        fs.protocol = protocol
        fs.account_name = account_name
        return fs

    def test_abfs_restores_account(self):
        """abfss://container@account.dfs.core.windows.net/path must be reconstructed."""
        fs = self._make_azure_fs(account_name='storageacct')
        result = fs_full_path(fs, 'mycontainer/subdir/file.parquet')
        assert result == 'abfss://mycontainer@storageacct.dfs.core.windows.net/subdir/file.parquet'

    def test_abfs_container_only(self):
        """Path with no subdirectory should still work."""
        fs = self._make_azure_fs(account_name='acct')
        result = fs_full_path(fs, 'container')
        assert result == 'abfss://container@acct.dfs.core.windows.net/'

    def test_az_protocol_also_restores_account(self):
        """'az' is an alias for abfs — same reconstruction logic."""
        fs = self._make_azure_fs(account_name='acct', protocol='az')
        result = fs_full_path(fs, 'cont/path/to/data')
        assert result == 'abfss://cont@acct.dfs.core.windows.net/path/to/data'

    def test_abfs_without_account_name_falls_back(self):
        """If account_name is empty, fall back to unstrip_protocol."""
        fs = self._make_azure_fs(account_name='')
        fs.unstrip_protocol.return_value = 'abfs://cont/data'
        result = fs_full_path(fs, 'cont/data')
        assert result == 'abfs://cont/data'
        fs.unstrip_protocol.assert_called_once_with('cont/data')

    def test_abfs_no_account_name_attr_falls_back(self):
        """If the fs object doesn't have account_name, fall back."""
        fs = MagicMock(spec=[])  # no attributes
        fs.protocol = 'abfs'
        fs.unstrip_protocol = MagicMock(return_value='abfs://cont/data')
        result = fs_full_path(fs, 'cont/data')
        assert result == 'abfs://cont/data'

    def test_deep_nested_path_preserves_structure(self):
        """Deeply nested paths should be fully preserved after the container."""
        fs = self._make_azure_fs(account_name='stor')
        result = fs_full_path(fs, 'mycontainer/a/b/c/d/file.csv')
        assert result == 'abfss://mycontainer@stor.dfs.core.windows.net/a/b/c/d/file.csv'


# ---------------------------------------------------------------------------
# Tuple protocol (some fsspec implementations expose protocol as a tuple)
# ---------------------------------------------------------------------------

class TestFsFullPathTupleProtocol:

    def test_tuple_protocol_local(self):
        """fs.protocol as tuple ('file', 'local') should be treated as local."""
        fs = MagicMock()
        fs.protocol = ('file', 'local')
        result = fs_full_path(fs, '/some/path')
        assert result == '/some/path'

    def test_tuple_protocol_remote(self):
        """fs.protocol as tuple ('gcs', 'gs') should restore protocol."""
        fs = MagicMock()
        fs.protocol = ('gcs', 'gs')
        fs.unstrip_protocol.return_value = 'gcs://bucket/key'
        result = fs_full_path(fs, 'bucket/key')
        assert result == 'gcs://bucket/key'


# ---------------------------------------------------------------------------
# ssh / sftp — the host must be restored
# ---------------------------------------------------------------------------

class _SSHFS_:
    """What fs_full_path sees of an sshfs filesystem: no `host` attribute.

    As `fsspec.url_to_fs('ssh://host/...')` makes one, the host is a keyword
    argument, in storage_options; constructed by hand, it may be positional.
    """

    def __init__(self, protocol=('ssh', 'sftp'), host='wheelbarrow', positional=False, **options):
        self.protocol = protocol
        self.storage_args = (host,) if positional else ()
        self.storage_options = options if positional else {'host': host, **options}

    def unstrip_protocol(self, path):
        return f"ssh://{path}"


class TestFsFullPathSSH:

    def test_sshfs_restores_host(self):
        """unstrip_protocol() alone gives ssh:///home/... -- a url naming no machine."""
        assert fs_full_path(_SSHFS_(), '/home/u/lake/x.parquet') == 'ssh://wheelbarrow/home/u/lake/x.parquet'

    def test_sshfs_positional_host(self):
        assert fs_full_path(_SSHFS_(positional=True), '/data/x') == 'ssh://wheelbarrow/data/x'

    def test_sftp_host_attribute(self):
        """fsspec's own sftp filesystem keeps the host as fs.host."""
        fs = _SSHFS_(protocol='sftp', host=None)
        fs.host = 'cilantro'
        assert fs_full_path(fs, '/data/x') == 'sftp://cilantro/data/x'

    def test_user_and_port(self):
        fs = _SSHFS_(username='me', port=2222)
        assert fs_full_path(fs, '/data/x') == 'ssh://me@wheelbarrow:2222/data/x'

    def test_round_trip_through_url_to_fs(self):
        """The url names the same host and path fsspec parsed it from."""
        from fsspec.utils import infer_storage_options
        url = fs_full_path(_SSHFS_(username='me'), '/home/u/lake')
        opts = infer_storage_options(url)
        assert (opts['host'], opts['username'], opts['path']) == ('wheelbarrow', 'me', '/home/u/lake')
