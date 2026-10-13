"""trueinst() from another machine: a Ray head on the datalake's host (SshRay), and what remote() and an entry need for it.

The end-to-end path -- a head started over ssh, a block instantiated in a
pinned worker there, a proxy to it here -- needs a host to ssh to, and is not
run here. These pin down the pieces it stands on.
"""
import sys
import types

import pandas as pd
import pytest

import dbx
from dbx import databits as databits_mod
from dbx import journals as journals_mod
from dbx.journals import DataJournalEntry, _on_fs_of_


@pytest.fixture
def stub_ray(monkeypatch):
    """A ray stub capturing what remote() initializes it with."""
    captured = {}
    ray = types.ModuleType('ray')
    ray.__version__ = '9.9.9-stub'
    ray.is_initialized = lambda: False
    ray.init = lambda **kw: captured.update(init=kw)
    ray.remote = lambda cls: types.SimpleNamespace(remote=lambda **kw: object())
    ray.actor = types.SimpleNamespace(ActorHandle=type('ActorHandle', (), {}))
    ray.runtime_env = types.SimpleNamespace(RuntimeEnv=type('RuntimeEnv', (), {'known_fields': {'env_vars'}}))
    monkeypatch.setitem(sys.modules, 'ray', ray)
    return captured


class TestRemoteEnv:

    def test_env_sets_and_unsets_the_workers_variables(self, stub_ray, monkeypatch):
        monkeypatch.setenv('DBX_DATALAKE', 'ssh://host/lake')
        monkeypatch.setenv('DBX_LOCAL', '/laptop/local')
        databits_mod.remote(env={'DBX_DATALAKE': '/lake', 'DBX_LOCAL': None, 'OTHER': 3})
        env = stub_ray['init']['runtime_env']['env_vars']
        assert env['DBX_DATALAKE'] == '/lake'
        assert 'DBX_LOCAL' not in env
        assert env['OTHER'] == '3'

    def test_without_env_this_process_dbx_variables_go(self, stub_ray, monkeypatch):
        monkeypatch.setenv('DBX_DATALAKE', 'ssh://host/lake')
        databits_mod.remote()
        assert stub_ray['init']['runtime_env']['env_vars']['DBX_DATALAKE'] == 'ssh://host/lake'

    def test_an_offhost_cluster_takes_gitrepo_without_a_repo_here(self, stub_ray, monkeypatch):
        """From a machine with no DBX_GIT_REPO of its own: the host's repo paths are given instead."""
        monkeypatch.setattr(databits_mod, '_DBX_GIT_REPO_', None)
        databits_mod.remote(address='ray://localhost:1', gitrepo='/home/me/dbx:/home/me/proj')
        assert stub_ray['init']['address'] == 'ray://localhost:1'
        assert stub_ray['init']['runtime_env']['env_vars']['DBX_GIT_REPO'] == '/home/me/dbx:/home/me/proj'

    def test_an_offhost_cluster_without_any_repo_path_is_still_refused(self, stub_ray, monkeypatch):
        monkeypatch.setattr(databits_mod, '_DBX_GIT_REPO_', None)
        with pytest.raises(ValueError, match='shared_repo'):
            databits_mod.remote(address='ray://localhost:1')


class TestRinstDatalake:

    @pytest.fixture
    def captured(self, monkeypatch):
        calls = []

        class _Handle:
            def run(self, func):
                return 'PROXY'

        monkeypatch.setattr(journals_mod, 'remote', lambda **kw: calls.append(kw) or _Handle())
        monkeypatch.setattr(DataJournalEntry, 'read', lambda self, *a, **k: '$pkg.mod.Block()')
        return calls

    def entry(self, **columns):
        return DataJournalEntry(pd.Series({'anchor': 'pkg.mod.Block', 'hash': 'a' * 64,
                                           'revision': 'dbxsha:projsha', **columns}))

    def test_the_worker_gets_the_datalake_the_entry_recorded(self, captured):
        self.entry(datalake='/home/me/lake').trueinst(address='ray://localhost:1')
        assert captured[0]['env'] == {'DBX_DATALAKE': '/home/me/lake'}

    def test_a_callers_env_is_laid_over_it(self, captured):
        self.entry(datalake='/home/me/lake').rinst(env={'DBX_DATALAKE': '/elsewhere', 'X': '1'})
        assert captured[0]['env'] == {'DBX_DATALAKE': '/elsewhere', 'X': '1'}

    def test_an_entry_recording_no_datalake_sets_none(self, captured):
        self.entry().rinst()
        assert captured[0]['env'] == {}


class TestOnFsOf:

    SSH = 'ssh://host/data/lake/.journal/e.parquet'

    def test_a_hosts_bare_path_goes_on_its_ssh_filesystem(self):
        assert _on_fs_of_('/data/lake/x.txt', self.SSH) == 'ssh://host/data/lake/x.txt'
        assert _on_fs_of_('/data/x', 'sftp://me@host:22/data/e.parquet') == 'sftp://me@host:22/data/x'

    @pytest.mark.parametrize('path', ['abfss://c@acct.dfs.core.windows.net/lake/x.txt',
                                      'gs://bucket/lake/x.txt', 's3://bucket/x', 'ssh://other/x'])
    def test_a_global_path_is_never_touched(self, path):
        assert _on_fs_of_(path, self.SSH) == path

    @pytest.mark.parametrize('url', ['abfss://c@acct.dfs.core.windows.net/lake/e.parquet',
                                     'gs://bucket/lake/e.parquet', 'memory://lake/e.parquet',
                                     'file:///data/e.parquet', '/data/lake/e.parquet', None])
    def test_a_bare_path_is_left_alone_off_a_host(self, url):
        assert _on_fs_of_('/data/lake/x.txt', url) == '/data/lake/x.txt'

    def test_a_relative_path_is_left_alone(self):
        assert _on_fs_of_('lake/x.txt', self.SSH) == 'lake/x.txt'

    def test_an_entry_reads_its_side_files_where_it_was_read_from(self, monkeypatch):
        seen = []
        monkeypatch.setattr(journals_mod, 'read_str', lambda path, **kw: seen.append(path) or 'Q')
        entry = DataJournalEntry(pd.Series({'anchor': 'a', 'hash': 'h', 'quote': '/data/lake/q.txt',
                                            'entry_path': self.SSH}))
        assert entry.read('quote', raw=True) == 'Q'
        assert seen == ['ssh://host/data/lake/q.txt']


def test_a_ray_client_actor_handle_is_wrapped():
    pytest.importorskip('ray')
    from ray.util.client.common import ClientActorHandle

    handle = object.__new__(ClientActorHandle)
    assert databits_mod.Remote._is_actor_handle_(handle)
    assert not databits_mod.Remote._is_actor_handle_('a str')


class TestSshRaySession:

    @pytest.fixture
    def started(self, monkeypatch):
        calls = []

        class _Proc:
            def poll(self):
                return None

        def start(self, timeout):
            calls.append((self.host, self.python))
            self._head = self._tunnel = _Proc()
            self._address = f'ray://localhost:{1000 + len(calls)}'

        monkeypatch.setattr(dbx.SshRay, '_start_', start)
        monkeypatch.setattr(dbx.SshRay, '_SESSIONS_', {})
        monkeypatch.setattr(dbx.SshRay, 'stop', lambda self: None)
        return calls

    def test_one_head_per_host_and_python(self, started):
        a = dbx.SshRay.session('host', python='/env/a/bin/python')
        assert dbx.SshRay.session('host', python='/env/a/bin/python') is a
        b = dbx.SshRay.session('host', python='/env/b/bin/python')
        assert b is not a and started == [('host', '/env/a/bin/python'), ('host', '/env/b/bin/python')]
        assert a.address == 'ray://localhost:1001'

    def test_a_head_that_died_is_started_again(self, started):
        a = dbx.SshRay.session('host', python='p')
        a._head = types.SimpleNamespace(poll=lambda: 1)
        assert dbx.SshRay.session('host', python='p') is not a and len(started) == 2


def test_shquote_survives_single_quotes():
    import subprocess

    text = "echo 'it''s' \"$HOME\" && echo done"
    out = subprocess.run(['bash', '-c', f"printf %s {databits_mod._shquote_(text)}"], capture_output=True, text=True).stdout
    assert out == text


def test_the_metrics_agent_complaints_are_dropped_and_nothing_else(monkeypatch, capsys):
    pytest.importorskip('ray')
    from ray.util.client.logsclient import LogstreamClient

    monkeypatch.setattr(LogstreamClient, 'stdstream', LogstreamClient.stdstream)
    dbx.SshRay._quiet_()
    dbx.SshRay._quiet_()  # installed once, however often asked
    msg = ("(RemoteDBX pid=1) [E core_worker_process.cc:842] Failed to establish connection to the metrics exporter agent. x\n"
           "(RemoteDBX pid=1) building the probe\n"
           "(gcs_server) Failed to establish connection to the event+metrics exporter agent. Events and metrics will not be exported.\n")
    LogstreamClient.stdstream(object.__new__(LogstreamClient), -2, msg)
    assert capsys.readouterr().err == "(RemoteDBX pid=1) building the probe\n"
