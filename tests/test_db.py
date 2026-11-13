"""`dbx.db`: datablock locks in the Postgres database `DBX_DB` names.

Without `DBX_DB`, every lock is a no-op. With it -- a server of the test's own, started by
`dbx.db.up` in a temporary directory -- an exclusive lock excludes another process, shared locks
do not, a lock dies with its process or is forced free by `unlock`, and two processes building
one datablock build it once: the second waits, then finds it valid.

The server needs PostgreSQL's programs, `pg_ctl` and `initdb`: on `PATH`, or in
`DBX_TEST_PG_BIN_DIR`. They are not a Python package, so they cannot be declared in
`pyproject.toml`; the tests that need a server skip, saying so, when they are missing.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

from dbx import db


# A datablock whose build is slow enough to overlap and counts its builds in a file beside the lake.
_DATABLOCK_MODULE_ = '''
import os, time
from dataclasses import dataclass
from dbx.datablocks import Datablock

class SlowDatablock(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = "'slow'"

    def __build__(self):
        with open(os.environ['BUILDS_FILE'], 'a') as f:
            f.write(f"{os.getpid()}\\n")
        time.sleep(float(os.environ.get('BUILD_S', '2')))
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('built')
'''


def _bin_dir_():
    found = os.environ.get('DBX_TEST_PG_BIN_DIR') or (os.path.dirname(shutil.which('pg_ctl') or '') or None)
    return found if found and os.path.exists(os.path.join(found, 'pg_ctl')) else None


def _free_port_():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def server(tmp_path_factory):
    """`DBX_DB` of a server of this module's own, up for its tests and down after."""
    bin_dir = _bin_dir_()
    if bin_dir is None:
        pytest.skip("PostgreSQL's pg_ctl is neither on PATH nor in DBX_TEST_PG_BIN_DIR")
    # Short: a Unix socket's path is limited to about 100 characters.
    data_dir = os.path.join('/tmp', f"dbx-db-test-{os.getpid()}-{_free_port_()}")
    url = f"postgresql://dbxtest:pw-{os.getpid()}@127.0.0.1:{_free_port_()}/dbxtest"
    old = os.environ.get('DBX_DB')
    os.environ['DBX_DB'] = url
    try:
        db.up(data_dir=data_dir, bin_dir=bin_dir)
        yield url
    finally:
        db.down(data_dir=data_dir, bin_dir=bin_dir)
        shutil.rmtree(data_dir, ignore_errors=True)
        if old is None:
            os.environ.pop('DBX_DB', None)
        else:
            os.environ['DBX_DB'] = old


@pytest.fixture
def with_db(server, monkeypatch):
    monkeypatch.setenv('DBX_DB', server)
    yield server
    db.unlock(all=True)


def _holder_(path, mode, *, hold_s=30, env=None):
    """A process that takes *mode* on *path*, says `held`, and holds it for *hold_s*."""
    code = textwrap.dedent(f"""
        import sys, time
        from dbx import db
        with db.lock({path!r}, {mode!r}):
            print('held', flush=True)
            time.sleep({hold_s})
    """)
    p = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, **(env or {})})
    assert p.stdout.readline().strip() == 'held'
    return p


def _took_s_(path, mode):
    t = time.monotonic()
    with db.lock(path, mode):
        pass
    return time.monotonic() - t


# --------------------------------------------------------------------------------------------- #
# Without DBX_DB
# --------------------------------------------------------------------------------------------- #

def test_without_dbx_db_a_lock_is_a_noop(monkeypatch):
    monkeypatch.delenv('DBX_DB', raising=False)
    with db.lock('/nowhere/at/all', 'exclusive'):
        pass
    assert getattr(db._LOCAL_, 'conn', None) is None


def test_an_unknown_mode_is_refused(monkeypatch):
    monkeypatch.delenv('DBX_DB', raising=False)
    with pytest.raises(ValueError, match='mode'):
        with db.lock('/x', 'write'):
            pass


def test_an_unreachable_dbx_db_fails_fast(monkeypatch):
    monkeypatch.setenv('DBX_DB', f"postgresql://u:secret@127.0.0.1:{_free_port_()}/d")
    monkeypatch.setattr(db, '_LOCAL_', type(db._LOCAL_)())
    with pytest.raises(RuntimeError, match=r'cannot connect.*unset DBX_DB') as e:
        with db.lock('/x', 'shared'):
            pass
    assert 'secret' not in str(e.value)


# --------------------------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------------------------- #

def test_storage_key_names_an_object_store_by_its_url():
    assert db._storage_key_('gs://bucket/lake/a/b') == 'gs://bucket/lake/a/b'


def test_storage_key_names_a_local_path_by_its_host(tmp_path):
    """Unless the test's temporary directory is itself on a network filesystem."""
    key = db._storage_key_(str(tmp_path))
    assert key.endswith(str(tmp_path.resolve()))
    assert key == f"{socket.gethostname()}:{tmp_path.resolve()}" or not key.startswith(socket.gethostname())


def test_storage_key_names_a_network_path_by_the_mount_source(monkeypatch, tmp_path):
    mounts = tmp_path / 'mounts'
    mounts.write_text("overlay / overlay rw 0 0\n"
                      "server:/export/teams/x /mnt/team-x nfs rw,vers=3 0 0\n")
    real_open = open
    monkeypatch.setattr('builtins.open', lambda f, *a, **k: real_open(mounts if f == '/proc/mounts' else f, *a, **k))
    monkeypatch.setattr(db.os.path, 'realpath', lambda p: p)
    assert db._storage_key_('/mnt/team-x/lake/a') == 'server:/export/teams/x/lake/a'
    assert db._storage_key_('/home/me/lake/a') == f"{socket.gethostname()}:/home/me/lake/a"


# --------------------------------------------------------------------------------------------- #
# With a server
# --------------------------------------------------------------------------------------------- #

def test_an_exclusive_lock_waits_for_another_process(with_db, tmp_path):
    p = _holder_(str(tmp_path), 'exclusive', hold_s=2)
    try:
        assert _took_s_(str(tmp_path), 'shared') > 1
    finally:
        p.wait()


def test_shared_locks_do_not_wait_for_each_other(with_db, tmp_path):
    p = _holder_(str(tmp_path), 'shared')
    try:
        assert _took_s_(str(tmp_path), 'shared') < 1
    finally:
        p.kill(), p.wait()


def test_locks_on_other_datablocks_do_not_wait(with_db, tmp_path):
    p = _holder_(str(tmp_path / 'a'), 'exclusive')
    try:
        assert _took_s_(str(tmp_path / 'b'), 'exclusive') < 1
    finally:
        p.kill(), p.wait()


def test_a_lock_is_reentrant_within_a_thread(with_db, tmp_path):
    with db.lock(str(tmp_path), 'exclusive'):
        with db.lock(str(tmp_path), 'shared'):
            with db.lock(str(tmp_path), 'exclusive'):
                pass
    p = _holder_(str(tmp_path), 'exclusive', hold_s=0)
    p.wait()


def test_locks_says_who_holds_what(with_db, tmp_path):
    p = _holder_(str(tmp_path), 'exclusive', env={'DBX_RUN': 'run-123', 'DBX_CLUSTER': 'here'})
    try:
        df = db.locks()
        mine = df[df['path'] == str(tmp_path)]
        assert list(mine['state']) == ['held'] and list(mine['mode']) == ['exclusive']
        assert list(mine['run']) == ['run-123'] and list(mine['pid']) == [p.pid]
    finally:
        p.kill(), p.wait()


def test_a_lock_dies_with_its_process(with_db, tmp_path):
    p = _holder_(str(tmp_path), 'exclusive')
    p.send_signal(signal.SIGKILL), p.wait()
    assert _took_s_(str(tmp_path), 'exclusive') < 5
    assert str(tmp_path) not in set(db.locks()['path'])


def test_unlock_forces_a_run_s_locks_free(with_db, tmp_path):
    p = _holder_(str(tmp_path), 'exclusive', env={'DBX_RUN': 'run-unlock'})
    try:
        freed = db.unlock('run-unlock')
        assert list(freed['path']) == [str(tmp_path)]
        assert _took_s_(str(tmp_path), 'exclusive') < 5
    finally:
        p.kill(), p.wait()


def test_unlock_names_what_to_free(with_db):
    with pytest.raises(ValueError, match='all=True'):
        db.unlock()


def test_two_processes_build_one_datablock_once(with_db, tmp_path):
    """The second waits for the first, finds the datablock valid, and skips it."""
    (tmp_path / 'slowdatablock.py').write_text(_DATABLOCK_MODULE_)
    builds = tmp_path / 'builds'
    env = {**os.environ, 'PYTHONPATH': f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
           'DBX_DATALAKE': str(tmp_path / 'lake'), 'BUILDS_FILE': str(builds), 'BUILD_S': '2',
           'DBX_LOG_VERBOSE': '1'}
    code = "import slowdatablock; d = slowdatablock.SlowDatablock(); d.build(); print('valid', d.valid())"
    ps = [subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=120)[0] for p in ps]
    assert all(p.returncode == 0 for p in ps), outs
    assert len(builds.read_text().split()) == 1, outs
    assert all('valid True' in out for out in outs), outs
    for out in outs:
        assert '>>>--- LOCK exclusive: ' in out and '<<<--- LOCK exclusive: ' in out, out
    assert sum('waiting -- held by' in out for out in outs) == 1, outs


# --------------------------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------------------------- #

def test_init_writes_dbx_db_with_a_new_password_for_this_user_alone(tmp_path):
    f = tmp_path / 'dbx-db'
    shown = db.init(str(f), host='dbx-db.ns.svc.cluster.local', port=6543, user='u', database='d')
    line = f.read_text()
    assert line.startswith('export DBX_DB=postgresql://u:') and line.endswith('@dbx-db.ns.svc.cluster.local:6543/d\n')
    assert oct(f.stat().st_mode & 0o777) == '0o600'
    password = line.split('postgresql://u:')[1].split('@')[0]
    assert len(password) >= 20 and password not in shown and ':***@' in shown


def test_init_refuses_to_replace_a_password_unless_forced(tmp_path):
    f = tmp_path / 'dbx-db'
    db.init(str(f), host='h')
    first = f.read_text()
    with pytest.raises(FileExistsError, match='--force'):
        db.init(str(f), host='h')
    assert f.read_text() == first
    db.init(str(f), host='h', force=True)
    assert f.read_text() != first and oct(f.stat().st_mode & 0o777) == '0o600'


def test_a_file_init_wrote_is_what_up_reads(tmp_path):
    f = tmp_path / 'dbx-db'
    db.init(str(f), host='127.0.0.1', port=_free_port_())
    out = subprocess.run(['bash', '-c', f"source {f} && echo $DBX_DB"], capture_output=True, text=True).stdout.strip()
    assert out == f.read_text().split('=', 1)[1].strip()
