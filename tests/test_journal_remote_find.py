"""Listing a journal on ssh: one `find` on the host, not a round trip per directory.

`BuildJournal._journal_files_` finds the same entry files either way; on an
ssh filesystem whose client runs commands, it asks the host once.
"""
import io
import subprocess
from types import SimpleNamespace

import pytest
from fsspec.implementations.local import LocalFileSystem

from dbx.datablocks import BuildJournal
from dbx.journals import Datalog


class _Channel:
    def __init__(self, status):
        self.status = status

    def recv_exit_status(self):
        return self.status


class _Out(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.channel = _Channel(status)


class _Client:
    """A paramiko client's `exec_command`, run here."""

    def __init__(self):
        self.commands = []

    def exec_command(self, command):
        self.commands.append(command)
        p = subprocess.run(command, shell=True, capture_output=True)
        return None, _Out(p.stdout, p.returncode), io.BytesIO(p.stderr)


class _SSHFS(LocalFileSystem):
    protocol = ('ssh', 'sftp')
    cachable = False

    def __init__(self, client=None, **kw):
        super().__init__(**kw)
        self.client = client
        self.listed = 0

    def ls(self, path, detail=True, **kw):
        self.listed += 1
        return super().ls(path, detail=detail, **kw)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')


@pytest.fixture
def anchor(tmp_path):
    root = tmp_path / '.pipelines' / 'a.B'
    for k in range(3):
        touch(root / f'k{k}' / '.journal' / 'a.B' / 'journal' / f'h{k}' / f'a.B-journal-h{k}-dt.parquet')
        touch(root / f'k{k}' / '.journal' / 'a.B' / 'spec' / f'h{k}' / f'a.B-spec-h{k}-dt.parquet')
        (root / f'k{k}' / 'empty' / 'dirs').mkdir(parents=True)
    touch(root / '.journal' / 'a.B' / 'journal' / 'a.B-journal-compacted.parquet')
    return root


def files(fs, root):
    return BuildJournal._journal_files_(fs, str(root), n_workers=1, log=Datalog())


class TestRemoteFind:

    def test_it_finds_what_the_walk_finds(self, anchor):
        walked = files(_SSHFS(), anchor)
        assert len(walked) == 4
        assert files(_SSHFS(_Client()), anchor) == walked

    def test_it_is_one_command_and_no_listing(self, anchor):
        fs = _SSHFS(_Client())
        files(fs, anchor)
        assert len(fs.client.commands) == 1
        assert fs.listed == 0

    def test_a_missing_directory_has_none(self, tmp_path):
        fs = _SSHFS(_Client())
        assert files(fs, tmp_path / 'nowhere') == []

    def test_a_failing_command_walks(self, anchor):
        class Failing(_Client):
            def exec_command(self, command):
                return super().exec_command('exit 2')
        fs = _SSHFS(Failing())
        assert files(fs, anchor) == files(_SSHFS(), anchor)
        assert fs.listed > 0

    def test_a_path_with_spaces(self, tmp_path):
        root = tmp_path / 'a lake' / '.pipelines'
        touch(root / '.journal' / 'a.B' / 'journal' / 'h' / 'x.parquet')
        assert files(_SSHFS(_Client()), root) == files(_SSHFS(), root)

    def test_sshfs_execute(self, anchor):
        """sshfs -- asyncssh, the `ssh://` datalake's filesystem -- runs it with `execute`."""
        commands = []

        class SSHFS(_SSHFS):
            def execute(self, command, check=True):
                commands.append(command)
                p = subprocess.run(command, shell=True, capture_output=True, text=True)
                return SimpleNamespace(stdout=p.stdout, stderr=p.stderr, exit_status=p.returncode)

        fs = SSHFS()
        assert files(fs, anchor) == files(_SSHFS(), anchor)
        assert len(commands) == 1 and fs.listed == 0

    def test_not_ssh_walks(self, anchor):
        class Local(_SSHFS):
            protocol = ('file',)
        fs = Local(_Client())
        files(fs, anchor)
        assert fs.client.commands == []
