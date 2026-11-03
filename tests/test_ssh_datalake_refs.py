"""A reference to an entry on an ssh datalake, written on one side of ssh and read on the other.

An ssh datalake -- an Anyscale workspace's, say -- is a bare path on its host
and `ssh://host/path` everywhere else, and a reference records the datalake
as its writer saw it. Written on the host and read over ssh, or written over
ssh and read on the host, it must still be found. A datalake with a protocol
of its own, `gs://` or `abfss://`, is the same to every reader.

The host here is a fake: `ssh://host/P` is `{tmp}/host/P` on the local disk.
"""
import json
import os
import re
import sys
from dataclasses import dataclass

import fsspec
import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem

import dbx
from dbx.datablocks import Datablock, BuildJournal
from dbx.journals import _datalakes_seen_


class Built(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.delenv('DBX_STORAGE_OPTIONS', raising=False)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """The fake host's `/`: `ssh://host/P` and `sftp://host/P` are this + P."""
    root = str(tmp_path / 'host')

    class HostFS(LocalFileSystem):
        protocol = ('ssh', 'sftp')
        cachable = False

        @classmethod
        def _strip_protocol(cls, path):
            m = re.match(r'^(?:ssh|sftp)://[^/]+(/.*)$', str(path))
            return super()._strip_protocol(root + m[1] if m else path)

    monkeypatch.setitem(sys.modules['fsspec.registry']._registry, 'ssh', HostFS)
    monkeypatch.setitem(sys.modules['fsspec.registry']._registry, 'sftp', HostFS)
    return root


@pytest.fixture
def unreachable(monkeypatch):
    """No ssh host answers: reading over ssh raises, as it would for a host not there."""

    class Unreachable(LocalFileSystem):
        protocol = ('ssh', 'sftp')
        cachable = False

        def __init__(self, *a, **k):
            raise ConnectionError("no such host")

    monkeypatch.setitem(sys.modules['fsspec.registry']._registry, 'ssh', Unreachable)


def built_on(lake, x=1):
    """A block built on the bare-path datalake *lake*, and the reference to its last entry."""
    b = Built(datalake=lake, spec={'x': x})
    with BuildJournal() as journal:
        b.build()
    [*_, ref] = journal.written_entries()
    return b, ref


def recorded_as(ref, datalake):
    """*ref*, as a writer that saw its datalake as *datalake* would have recorded it."""
    return json.dumps({**json.loads(ref), 'datalake': datalake}, sort_keys=True)


class TestDatalakesSeen:

    def test_written_on_the_host_read_over_ssh_is_on_the_host(self):
        assert _datalakes_seen_('/mnt/lake', 'ssh://ws/mnt/lake') == ['ssh://ws/mnt/lake']

    def test_on_the_host_whatever_datalake_the_reader_has_there(self):
        assert _datalakes_seen_('/mnt/other', 'ssh://ws/mnt/lake') == ['ssh://ws/mnt/other']

    def test_written_over_ssh_read_locally_is_its_path_first(self):
        assert _datalakes_seen_('ssh://ws/mnt/lake', '/mnt/lake') == ['/mnt/lake', 'ssh://ws/mnt/lake']

    def test_written_over_ssh_read_over_ssh_is_as_written(self):
        assert _datalakes_seen_('ssh://ws/mnt/lake', 'ssh://ws/mnt/lake') == ['ssh://ws/mnt/lake']

    @pytest.mark.parametrize('lake', ['gs://bucket/lake', 'abfss://c@acct.dfs.core.windows.net/lake'])
    @pytest.mark.parametrize('read_from', [None, '/mnt/lake', 'ssh://ws/mnt/lake', 'gs://bucket/lake'])
    def test_a_global_url_is_the_same_to_every_reader(self, lake, read_from):
        assert _datalakes_seen_(lake, read_from) == [lake]

    def test_a_bare_path_read_locally_is_itself(self):
        assert _datalakes_seen_('/mnt/lake', '/mnt/lake') == ['/mnt/lake']
        assert _datalakes_seen_('/mnt/lake', None) == ['/mnt/lake']


class TestReadRefs:

    @pytest.mark.pinned
    def test_written_on_the_host_is_found_over_ssh(self, host):
        lake = host + '/mnt/lake'
        b, ref = built_on(lake)
        on_host = recorded_as(ref, '/mnt/lake')     # as the host saw it: not a path here
        frame = BuildJournal.read_refs([on_host], read_from='ssh://ws/mnt/lake')
        assert list(frame['id']) == [json.loads(ref)['id']]
        assert list(frame['hash']) == [b.hash]

    @pytest.mark.pinned
    def test_written_over_ssh_is_found_on_the_host(self, tmp_path, unreachable):
        lake = str(tmp_path / 'lake')
        _, ref = built_on(lake)
        over_ssh = recorded_as(ref, f'ssh://ws{lake}')
        frame = BuildJournal.read_refs([over_ssh], read_from=lake)
        assert list(frame['id']) == [json.loads(ref)['id']]

    def test_without_where_it_was_read_from_a_host_path_is_not_found(self, host):
        _, ref = built_on(host + '/mnt/lake')
        frame = BuildJournal.read_refs([recorded_as(ref, '/mnt/lake')])
        assert len(frame) == 0


class TestExecJournalOverSsh:

    @pytest.mark.pinned
    def test_a_command_run_on_the_host_reads_its_entries_over_ssh(self, host, monkeypatch):
        """What `dbx.execjournal(id=...).buildjournal()` does from a laptop with an Anyscale DBX_DATALAKE."""
        lake = host + '/mnt/lake'
        monkeypatch.setenv('DBX_DATALAKE', lake)
        b = dbx.exec("b = Built(datalake=lake, spec={'x': 1}); b.build(); b", Built=Built, lake=lake)
        exec_dir = os.path.join(lake, '.journal', 'exec')
        [f] = os.listdir(exec_dir)
        row = pd.read_parquet(os.path.join(exec_dir, f))
        # As the host wrote it: its datalake a bare path, and no session index to fall back on.
        row['buildjournal_entries'] = [[recorded_as(r, '/mnt/lake') for r in row['buildjournal_entries'].iloc[0]]]
        row['session'] = None
        row.to_parquet(os.path.join(exec_dir, f))

        monkeypatch.setenv('DBX_DATALAKE', 'ssh://ws/mnt/lake')
        frame = dbx.execjournal(iloc=0).buildjournal()
        assert set(frame['hash']) == {b.hash}
        assert len(frame) == len(row['buildjournal_entries'].iloc[0])
