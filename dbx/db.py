"""dbx.db -- datablock locks, held in a Postgres database that `DBX_DB` names.

A datablock's `build()` takes an exclusive lock on it, and its `read()`, `ls()` and `paths()` a shared
one, released when the call returns. Two processes building the same datablock -- two Jobs whose
trees share a subtree -- take turns: the second waits for the first, then finds the datablock valid
and skips it, rather than building into the same directory at the same time. With `DBX_DB` unset,
every lock is a no-op.

    export DBX_DB=postgresql://dbx:<password>@dbx-db.kueue-team-x.svc.cluster.local:5432/dbx
    dbx.db up --expose crusoe:pod lambda:tailnet      # on the machine that serves it
    dbx.db locks                                      # who holds what, and who waits
    dbx.db unlock dmitry-scqaeval-build-tree-479c1    # force: a run, host, path or backend pid
    dbx.db down

The locks are Postgres advisory locks, held by the process's connection -- one per thread, so
threads exclude each other as processes do -- and released when it closes: at exit, or when the
process dies and the server notices its connection is gone (TCP keepalives, a minute or two). A
table, `dbx_locks`, says which datablock each one is, and who holds or waits for it; `locks`
reads it against the server's own lock table, so a row whose holder is gone is never shown.

A lock is keyed by the storage the datablock is on and its path there: the source of the network
filesystem mounted at that path (`/proc/mounts`), or the host, for a local one, or the URL, for
an object store. Two clusters mounting different shares at the same path do not contend; a
workspace and the Jobs that mount its share do.

The server is local to the machine that runs `up`: its data directory is on local disk (`/tmp`),
so a reboot of that machine clears every lock, and `up` starts a fresh server. `up` takes the
role, password, port and database from `DBX_DB`; `--expose` makes the URL's host reach it from
each cluster -- through the pod network where this machine is a pod of that cluster (`pod`), or
through the cluster's Tailscale egress proxy (`tailnet`).

A process that holds an exclusive lock on a datablock and waits on another process that reads it
-- a Ray worker reading the datablock its driver is building -- waits forever: neither the server
nor dbx can see that wait. The work a build hands to workers reads and writes plain paths, as
dbx's skills ask, and takes no lock.
"""
import argparse
import atexit
import contextlib
import getpass
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import threading
import urllib.parse

import pandas as pd

from .journals import SESSION_ENV, log_section


__all__ = ['lock', 'locks', 'unlock', 'up', 'down', 'status', 'main']

# What a datablock can be locked as.
MODES = ('shared', 'exclusive')
# The table that says which datablock each advisory lock is, and who holds or waits for it.
TABLE = 'dbx_locks'


# --------------------------------------------------------------------------------------------- #
# Locks
# --------------------------------------------------------------------------------------------- #

@contextlib.contextmanager
def lock(path: str, mode: str = 'shared', *, log=None):
    """Hold a *mode* lock on the datablock at *path* for the body; a no-op when `DBX_DB` is unset.

    Waits while another process holds a conflicting one, saying who at VERBOSE first. The time it
    is held is a `log_section` of *log*, `LOCK <mode>: <path>`, at VERBOSE. Reentrant within a
    thread: a lock this thread holds is granted again, in either mode.
    """
    if mode not in MODES:
        raise ValueError(f"lock(mode={mode!r}): mode is one of {MODES}")
    if not os.environ.get('DBX_DB'):
        yield
        return
    conn = _connection_()
    key = _storage_key_(path)
    lock_id = _lock_id_(key)
    suffix = '_shared' if mode == 'shared' else ''
    # One round trip to take an uncontended lock, and one to release it: a cluster may be far away.
    row, state = conn.execute(
        f"INSERT INTO {TABLE} (lock_id, mode, key, path, host, pid, thread, cluster, run, session, state) "
        f"SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
        f"CASE WHEN pg_try_advisory_lock{suffix}(%s) THEN 'held' ELSE 'waiting' END RETURNING id, state",
        (lock_id, mode, key, path, socket.gethostname(), os.getpid(), threading.get_ident(),
         os.environ.get('DBX_CLUSTER'), os.environ.get('DBX_RUN'), os.environ.get(SESSION_ENV), lock_id)).fetchone()
    try:
        if state == 'waiting':
            if log is not None:
                log.verbose(f"LOCK {mode}: {path}: waiting -- {_holders_text_(conn, lock_id)}")
            conn.execute(f"SELECT pg_advisory_lock{suffix}(%s)", (lock_id,))
            state = 'held'
            conn.execute(f"UPDATE {TABLE} SET state = 'held', since = now() WHERE id = %s", (row,))
            # What the holder wrote, another host may not see yet: an NFS client trusts its cached
            # lookups -- "not there" included -- for up to a minute. Reading the directories
            # refreshes them (READDIRPLUS), so a build waited for is found valid, not built again.
            if '://' not in path:
                with contextlib.suppress(OSError):
                    for _ in os.walk(path):
                        pass
        with (log_section(log, f"LOCK {mode}: {path}") if log is not None else contextlib.nullcontext()):
            yield
    finally:
        # A connection the server dropped -- `unlock` forced -- holds nothing left to release.
        if not conn.closed:
            with contextlib.suppress(Exception):
                if state == 'held':
                    conn.execute(f"WITH gone AS (DELETE FROM {TABLE} WHERE id = %s) "
                                 f"SELECT pg_advisory_unlock{suffix}(%s)", (row, lock_id))
                else:
                    conn.execute(f"DELETE FROM {TABLE} WHERE id = %s", (row,))


def locks(url: str | None = None) -> pd.DataFrame:
    """Every lock held or waited for, oldest first: who, on which datablock, in which mode, since when.

    Rows whose holder's connection is gone are deleted first: their locks went with it.
    """
    with _admin_connection_(url) as conn:
        _prune_(conn)
        cur = conn.execute(f"SELECT state, mode, path, host, pid, cluster, run, session, since, backend_pid, key "
                           f"FROM {TABLE} ORDER BY since, id")
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def unlock(*targets: str, all: bool = False, url: str | None = None) -> pd.DataFrame:
    """Force the locks matching *targets* free, by ending the connections that hold them; the locks it freed.

    A target matches a lock's run, host, session or backend pid exactly, or is a substring of its
    path. Ending a connection frees every lock it holds, not only the matching ones: they are
    all one process's, and it can no longer be trusted to hold any. That process is not stopped:
    it goes on, unprotected, until its next lock fails to release. *all* frees every lock.
    """
    if not targets and not all:
        raise ValueError("unlock: name a run, host, session, backend pid or part of a path -- or all=True")
    with _admin_connection_(url) as conn:
        _prune_(conn)
        df = locks(url)
        if not all:
            match = (df[['run', 'host', 'session']].isin(targets).any(axis=1)
                     | df['backend_pid'].astype(str).isin(targets)
                     | df['path'].apply(lambda p: any(t in p for t in targets)))
            df = df[match]
        pids = sorted(set(df['backend_pid']))
        for pid in pids:
            conn.execute("SELECT pg_terminate_backend(%s)", (int(pid),))
        _prune_(conn)
        return df


# --------------------------------------------------------------------------------------------- #
# The server
# --------------------------------------------------------------------------------------------- #

def up(*, data_dir: str = '/tmp/dbx-db', bin_dir: str | None = None, expose: list | None = None) -> str:
    """Start the database `DBX_DB` names on this machine, its data in *data_dir*; `DBX_DB`, password hidden.

    Initializes *data_dir* when it is new: the role, password, port and database of `DBX_DB`,
    reachable over TCP with that password, and over a socket in *data_dir* by this user alone.
    Started already, it is left as it is. *bin_dir* holds `pg_ctl` and `initdb` (default: on
    `PATH`). *expose* is `['cluster:via', ...]`, via `pod` or `tailnet`: see `expose`.
    """
    url = _url_()
    pg_ctl = _program_('pg_ctl', bin_dir)
    if not os.path.exists(os.path.join(data_dir, 'PG_VERSION')):
        _initdb_(data_dir, url, bin_dir)
    if subprocess.run([pg_ctl, '-D', data_dir, 'status'], capture_output=True).returncode != 0:
        subprocess.run([pg_ctl, '-D', data_dir, '-l', os.path.join(data_dir, 'server.log'), '-w', 'start'],
                       check=True, capture_output=True)
    _bootstrap_(data_dir, url)
    for spec in expose or ():
        name, _, via = spec.partition(':')
        expose_to(name, via or 'tailnet')
    return _redacted_(url)


def down(*, data_dir: str = '/tmp/dbx-db', bin_dir: str | None = None) -> None:
    """Stop the database in *data_dir*: every lock it holds goes with it. Its data stays, for the next `up`."""
    subprocess.run([_program_('pg_ctl', bin_dir), '-D', data_dir, '-m', 'fast', '-w', 'stop'],
                   check=True, capture_output=True)


def status(url: str | None = None) -> dict:
    """Whether the database `DBX_DB` names answers, and how many locks are held and waited for."""
    try:
        with _admin_connection_(url) as conn:
            _prune_(conn)
            counts = dict(conn.execute(f"SELECT state, count(*) FROM {TABLE} GROUP BY state").fetchall())
            version = conn.execute("SHOW server_version").fetchone()[0]
    except Exception as e:
        return {'db': _redacted_(url or _url_()), 'up': False, 'error': str(e).strip()}
    return {'db': _redacted_(url or _url_()), 'up': True, 'version': version,
            'held': counts.get('held', 0), 'waiting': counts.get('waiting', 0)}


def expose_to(cluster: str, via: str = 'tailnet') -> str:
    """Make `DBX_DB`'s host reach this machine from the pods of *cluster*, a `dbx.kube` cluster; the Service's target.

    The host is `<service>.<namespace>.svc.cluster.local`, and the Service is replaced -- deleted
    and created, as a Kueue namespace allows no patch -- with an ExternalName one: *via* `pod`
    names this machine's pod IP, for a cluster this machine is a pod of; `tailnet`, this
    machine's Tailscale name, which the cluster's Tailscale operator proxies to (the tailnet's
    ACL must let the cluster's tag reach this machine on the port).
    """
    from . import kube  # Circular: dbx.kube imports dbx.clusters, which imports the datablocks that import this.
    c = kube.cluster(cluster)
    parts = urllib.parse.urlsplit(_url_())
    service = parts.hostname.split('.')[0]
    port = parts.port or 5432
    if via == 'pod':
        target, annotations = f"{_pod_ip_().replace('.', '-')}.{c.namespace}.pod.cluster.local", {}
    elif via == 'tailnet':
        target, annotations = 'placeholder', {'tailscale.com/tailnet-fqdn': _tailnet_name_()}
    else:
        raise ValueError(f"expose_to(via={via!r}): via is 'pod' or 'tailnet'")
    manifest = {'apiVersion': 'v1', 'kind': 'Service',
                'metadata': {'name': service, 'namespace': c.namespace, 'annotations': annotations,
                             'labels': {'dbx/role': 'db'}},
                'spec': {'type': 'ExternalName', 'externalName': target,
                         'ports': [{'name': 'postgres', 'port': port, 'protocol': 'TCP'}]}}
    kube._kubectl_(c.context, '-n', c.namespace, 'delete', 'service', service, '--ignore-not-found')
    kube._kubectl_(c.context, 'apply', '-f', '-', input=json.dumps(manifest))
    return annotations.get('tailscale.com/tailnet-fqdn', target)


def main(argv: list | None = None) -> None:
    """`dbx.db up|down|status|locks|unlock`: the database of datablock locks that `DBX_DB` names."""
    p = argparse.ArgumentParser(prog='dbx.db', description=main.__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('up', 'down'):
        s = sub.add_parser(name, help=(up if name == 'up' else down).__doc__.split('\n')[0])
        s.add_argument('--data-dir', default='/tmp/dbx-db', help="the server's data directory, on local disk")
        s.add_argument('--bin-dir', help="where pg_ctl and initdb are (default: on PATH)")
        if name == 'up':
            s.add_argument('--expose', nargs='*', default=[], metavar='CLUSTER[:pod|tailnet]',
                           help="make DBX_DB's host reach this machine from these clusters' pods (default via: tailnet)")
    sub.add_parser('status', help=status.__doc__.split('\n')[0])
    sub.add_parser('locks', help=locks.__doc__.split('\n')[0])
    s = sub.add_parser('unlock', help=unlock.__doc__.split('\n')[0])
    s.add_argument('targets', nargs='*', help="a run, host, session or backend pid, or part of a datablock's path")
    s.add_argument('--all', action='store_true', help="every lock")
    a = p.parse_args(argv)
    if a.command == 'up':
        print(up(data_dir=a.data_dir, bin_dir=a.bin_dir, expose=a.expose))
    elif a.command == 'down':
        down(data_dir=a.data_dir, bin_dir=a.bin_dir)
    elif a.command == 'status':
        print(json.dumps(status(), indent=2, default=str))
    else:
        df = locks() if a.command == 'locks' else unlock(*a.targets, all=a.all)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 250,
                               'display.max_colwidth', 120):
            print(df.drop(columns=['key']).to_string(index=False) if len(df) else "no locks")


# --------------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------------- #

# This thread's connection, and the process that opened it: a forked child opens its own.
_LOCAL_ = threading.local()
# Every connection this process opened, for `_release_all_` at exit.
_CONNECTIONS_ = []
_CONNECTIONS_LOCK_ = threading.Lock()
# Filesystems a mount's source names storage that other machines mount too.
_NETWORK_FSTYPES_ = ('nfs', 'nfs4', 'cifs', 'smb3', 'ceph', 'glusterfs', 'lustre', 'beegfs')


def _url_() -> str:
    url = os.environ.get('DBX_DB')
    if not url:
        raise RuntimeError("DBX_DB is not set: it is the database's URL, postgresql://user:password@host:port/db")
    return url


def _redacted_(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.password is None:
        return url
    return parts._replace(netloc=parts.netloc.replace(f":{parts.password}@", ':***@')).geturl()


def _connect_(url: str):
    """A new autocommit connection to *url*, with keepalives so the server soon sees it die."""
    import psycopg  # Only a process that locks needs it.
    try:
        return psycopg.connect(url, autocommit=True, connect_timeout=10, keepalives=1, keepalives_idle=30,
                               keepalives_interval=10, keepalives_count=6, application_name='dbx')
    except psycopg.OperationalError as e:
        raise RuntimeError(f"DBX_DB={_redacted_(url)}: cannot connect ({' '.join(str(e).split())}). Bring it up -- "
                           f"dbx.db up -- or unset DBX_DB to take no locks") from e


def _connection_():
    """This thread's connection, opened on first use; registered to release everything at exit."""
    conn = getattr(_LOCAL_, 'conn', None)
    if conn is None or conn.closed or _LOCAL_.pid != os.getpid():
        conn = _connect_(_url_())
        conn.execute(f"SET application_name = 'dbx {socket.gethostname()}:{os.getpid()}'")
        _LOCAL_.conn, _LOCAL_.pid = conn, os.getpid()
        with _CONNECTIONS_LOCK_:
            if not _CONNECTIONS_:
                atexit.register(_release_all_)
            _CONNECTIONS_.append((os.getpid(), conn))
    return conn


def _release_all_() -> None:
    """At exit: release every lock this process's connections hold, drop their rows and close them."""
    with _CONNECTIONS_LOCK_:
        mine = [conn for pid, conn in _CONNECTIONS_ if pid == os.getpid()]
        _CONNECTIONS_.clear()
    for conn in mine:
        if conn.closed:
            continue
        with contextlib.suppress(Exception):
            conn.execute("SELECT pg_advisory_unlock_all()")
            conn.execute(f"DELETE FROM {TABLE} WHERE backend_pid = pg_backend_pid()")
        with contextlib.suppress(Exception):
            conn.close()


@contextlib.contextmanager
def _admin_connection_(url: str | None):
    conn = _connect_(url or _url_())
    try:
        yield conn
    finally:
        conn.close()


def _prune_(conn) -> None:
    """Delete the rows of connections that are gone: the server released their locks with them."""
    conn.execute(f"DELETE FROM {TABLE} WHERE backend_pid NOT IN (SELECT pid FROM pg_stat_activity)")


def _holders_text_(conn, lock_id: int) -> str:
    rows = conn.execute(f"SELECT mode, host, pid, run FROM {TABLE} WHERE lock_id = %s AND state = 'held' "
                        f"AND backend_pid <> pg_backend_pid() ORDER BY since", (lock_id,)).fetchall()
    if not rows:
        return "held by a process that has not said who it is"
    return "held by " + ', '.join(f"{host}:{pid}" + (f" (run {run})" if run else '') + f" [{mode}]"
                                  for mode, host, pid, run in rows)


def _storage_key_(path: str) -> str:
    """What names *path*'s storage everywhere.

    The network filesystem's source and the path on it, `host:path` for a local filesystem, or the
    URL itself for an object store.
    """
    if '://' in path and not path.startswith('file://'):
        return path
    path = os.path.realpath(path.removeprefix('file://'))
    best = ('', '', '')
    with contextlib.suppress(OSError), open('/proc/mounts') as f:
        for line in f:
            source, mountpoint, fstype = line.split()[:3]
            mountpoint = mountpoint.replace('\\040', ' ')
            inside = path == mountpoint or path.startswith(mountpoint.rstrip('/') + '/')
            if inside and len(mountpoint) > len(best[1]):
                best = (source, mountpoint, fstype)
    source, mountpoint, fstype = best
    if fstype in _NETWORK_FSTYPES_ or fstype.startswith('fuse'):
        return f"{source}{path[len(mountpoint.rstrip('/')):]}"
    return f"{socket.gethostname()}:{path}"


def _lock_id_(key: str) -> int:
    """The advisory lock's bigint: 8 bytes of *key*'s BLAKE2b."""
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'big', signed=True)


def _program_(name: str, bin_dir: str | None) -> str:
    found = os.path.join(bin_dir, name) if bin_dir else shutil.which(name)
    if not found or not os.path.exists(found):
        raise RuntimeError(f"{name} not found {'in ' + bin_dir if bin_dir else 'on PATH'}: pass --bin-dir, "
                           f"the bin directory of a PostgreSQL install")
    return found


def _initdb_(data_dir: str, url: str, bin_dir: str | None) -> None:
    """A new cluster in *data_dir*: TCP with a password on `DBX_DB`'s port; a socket in *data_dir*, where this
    OS user is the superuser (peer authentication), and the superuser has a random password no one keeps."""
    parts = urllib.parse.urlsplit(url)
    pwfile = os.path.join(os.path.dirname(data_dir.rstrip('/')) or '.', f".dbx-db-pw-{secrets.token_hex(4)}")
    with open(os.open(pwfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        f.write(secrets.token_urlsafe(24))
    try:
        subprocess.run([_program_('initdb', bin_dir), '-D', data_dir, '-U', getpass.getuser(), f"--pwfile={pwfile}",
                        '--auth-local=peer', '--auth-host=scram-sha-256', '--encoding=UTF8'],
                       check=True, capture_output=True)
    finally:
        os.remove(pwfile)
    with open(os.path.join(data_dir, 'postgresql.conf'), 'a') as f:
        f.write(f"\n# dbx.db\nlisten_addresses = '*'\nport = {parts.port or 5432}\n"
                f"unix_socket_directories = '{data_dir}'\nmax_connections = 500\n"
                f"tcp_keepalives_idle = 30\ntcp_keepalives_interval = 10\ntcp_keepalives_count = 6\n"
                f"client_connection_check_interval = '10s'\n")
    with open(os.path.join(data_dir, 'pg_hba.conf'), 'a') as f:
        f.write("\n# dbx.db\nhost all all 0.0.0.0/0 scram-sha-256\nhost all all ::/0 scram-sha-256\n")


def _bootstrap_(data_dir: str, url: str) -> None:
    """`DBX_DB`'s role, with its password, its database and the lock table: whichever are missing."""
    import psycopg  # Only `up` needs it here.
    from psycopg import sql
    parts = urllib.parse.urlsplit(url)
    role, password, dbname = parts.username, parts.password, parts.path.lstrip('/')
    socket_kwargs = dict(host=data_dir, port=parts.port or 5432, user=getpass.getuser(), autocommit=True)
    with psycopg.connect(dbname='postgres', **socket_kwargs) as conn:
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone():
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(role)))
        conn.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
        # Ending another role's connection -- `unlock` -- takes this.
        conn.execute(sql.SQL("GRANT pg_signal_backend TO {}").format(sql.Identifier(role)))
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(dbname), sql.Identifier(role)))
    with psycopg.connect(dbname=dbname, **socket_kwargs) as conn:
        conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id bigserial PRIMARY KEY,
                lock_id bigint NOT NULL,
                mode text NOT NULL,
                key text NOT NULL,
                path text NOT NULL,
                host text, pid integer, thread bigint, cluster text, run text, session text,
                state text NOT NULL,
                since timestamptz NOT NULL DEFAULT now(),
                backend_pid integer NOT NULL DEFAULT pg_backend_pid())""")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_lock_id ON {TABLE} (lock_id)")


def _pod_ip_() -> str:
    """This machine's address on its cluster's pod network."""
    return socket.gethostbyname(socket.gethostname())


def _tailnet_name_() -> str:
    """This machine's Tailscale name, from `tailscale status`.

    Through `~/.tailscale/tailscaled.sock` when tailscaled runs in userspace there, as in an
    Anyscale workspace.
    """
    sock = os.path.expanduser('~/.tailscale/tailscaled.sock')
    cmd = ['tailscale', *(['--socket', sock] if os.path.exists(sock) else []), 'status', '--self', '--json']
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)['Self']['DNSName'].rstrip('.')
