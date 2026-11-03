"""A running Anyscale workspace as a datalake: fsspec over ssh, the way `anyscale workspace_v2 ssh` connects.

From the shell:

    export DBX_STORAGE_OPTIONS=`dbx.anyscale_storage_options --name my-workspace`
    export DBX_DATALAKE=`dbx.anyscale_datalake --name my-workspace --root /mnt/cluster_storage/lake`
    hyp --journal

or from Python:

    fs, root = fsspec.url_to_fs(datalake('/mnt/cluster_storage/lake', name='my-workspace'),
                                **storage_options(name='my-workspace'))

A workspace is named by `--id`, or by `--name` with `--cloud` and `--project`
to disambiguate. Each falls back on the environment: `ANYSCALE_WORKSPACE_ID`
(else `ANYSCALE_EXPERIMENTAL_WORKSPACE_ID`, which a workspace sets for itself),
`ANYSCALE_WORKSPACE_NAME`, `ANYSCALE_CLOUD`, `ANYSCALE_PROJECT`. The Anyscale
API is reached as `anyscale login` (or `ANYSCALE_CLI_TOKEN`) authorizes.

The connection is ssh as `ray`, tunnelled through a WebSocket to the
workspace's public hostname, keyed by its cluster's ssh key and authorized by a
short-lived cluster access token. Everything ssh needs for that goes in an
OpenSSH config written under `~/.dbx/anyscale/<workspace id>`, with one Host,
the workspace's id:

- the datalake is `ssh://<id>/<root>`: sshfs, the `ssh://` driver, connects
  through asyncssh;
- the storage options are `config=<that file>`, which sshfs hands to asyncssh
  in place of `~/.ssh/config`.

The config's ProxyCommand is this module, `python -m dbx.anyscale proxy <id>`,
which looks up the workspace's hostname, key and a fresh token on every
connection and then becomes Anyscale's own WebSocket proxy. So the two
variables stay good across the token's expiry.

The workspace end needs two things the Anyscale CLI does not: a host key to
verify, and an sshd that serves sftp, which a workspace's own -- run as `ray`
from `/tmp/ray/ssh/sshd_config` -- does not. `ssh_config()` connects once to
pin the host key in `known_hosts` beside the config and, if sftp is refused,
adds `Subsystem sftp internal-sftp` to that sshd's config and reloads it. Both
last as long as the workspace's cluster: after a restart, run
`dbx.anyscale_storage_options` again -- it prints the same thing, so nothing
needs re-exporting.

`Include ~/.dbx/anyscale/*/config` at the top of `~/.ssh/config` makes `ssh
<id>` reach the workspace too, which is how a remote python (`hyp
--remote-python`, `SshRay`) is started on the datalake's host.

The user's guide is CLUSTERS.md, at the root of the repo.
"""
import argparse
import asyncio
import contextlib
import importlib.resources
import os
import posixpath
import shlex
import sys

import anyscale
import asyncssh
from anyscale.workspace._private.workspace_sdk import WSS_PATH, PrivateWorkspaceSDK
from anyscale.workspace.models import WorkspaceState


HOME = os.path.expanduser('~/.dbx/anyscale')

# The workspace's coordinates, each falling back on these variables, in order.
ID_ENV_VARS = ('ANYSCALE_WORKSPACE_ID', 'ANYSCALE_EXPERIMENTAL_WORKSPACE_ID')
NAME_ENV_VAR = 'ANYSCALE_WORKSPACE_NAME'
CLOUD_ENV_VAR = 'ANYSCALE_CLOUD'
PROJECT_ENV_VAR = 'ANYSCALE_PROJECT'

# A Host block for asyncssh and OpenSSH alike. The host key is the one
# `ssh_config()` pins in known_hosts; asyncssh holds to it, and OpenSSH, told
# StrictHostKeyChecking=no as `anyscale workspace_v2 ssh` tells it, only warns
# when a restarted cluster's differs.
SSH_CONFIG = """\
Host {host}
    User ray
    Port 443
    IdentityFile {key}
    IdentitiesOnly yes
    PreferredAuthentications publickey
    UserKnownHostsFile {known_hosts}
    StrictHostKeyChecking no
    ServerAliveInterval 30
    ProxyCommand {proxy}
"""

# Run on the workspace when it refuses sftp: the sshd of this user's started
# with a config of its own (-f) gets an sftp subsystem, and rereads its config
# on SIGHUP. Connections already open keep the config they started with.
ENABLE_SFTP = r"""
set -e
pid=$(pgrep -o -u "$(id -u)" -f 'sshd.* -f ') || { echo "no sshd of $(id -un)'s run with -f <config>" >&2; exit 1; }
conf=$(ps -o args= -p "$pid" | sed -n 's/.* -f \([^ ]*\).*/\1/p')
grep -qi '^[[:space:]]*Subsystem[[:space:]][[:space:]]*sftp' "$conf" ||
    printf '\n# dbx.anyscale: sftp, for fsspec over ssh\nSubsystem sftp internal-sftp\n' >> "$conf"
kill -HUP "$pid"
"""


def storage_options(*, id=None, name=None, cloud=None, project=None) -> dict:
    """The fsspec storage options for a `datalake()` url on the workspace: the ssh config `ssh_config()` writes."""
    return dict(config=ssh_config(id=id, name=name, cloud=cloud, project=project))


def datalake(root: str, *, id=None, name=None, cloud=None, project=None) -> str:
    """*root*, an absolute path on the workspace, as the `ssh://` url that `storage_options()` reaches."""
    if not posixpath.isabs(root):
        raise ValueError(f"A workspace's datalake root is an absolute path on it, not {root!r}")
    return f"ssh://{_host_(workspace_id(id=id, name=name, cloud=cloud, project=project))}{posixpath.normpath(root)}"


def ssh_config(*, id=None, name=None, cloud=None, project=None) -> str:
    """The path of the workspace's ssh config, written -- with its key -- under `~/.dbx/anyscale/<id>`."""
    sdk = PrivateWorkspaceSDK()
    model = _running_workspace_(sdk, **_workspace_args_(id, name, cloud, project))
    _refresh_(sdk, model)
    proxy = shlex.join([sys.executable, '-m', 'dbx.anyscale', 'proxy', model.id])
    path = os.path.join(workspace_dir(model.id), 'config')
    _write_private_(path, SSH_CONFIG.format(host=_host_(model.id), key=_key_path_(model.id),
                                            known_hosts=_known_hosts_path_(model.id), proxy=proxy))
    asyncio.run(_prepare_(model.id, path))
    return path


def workspace_id(*, id=None, name=None, cloud=None, project=None) -> str:
    """The workspace's id: *id*, or the id of the workspace *name* names -- a lookup."""
    args = _workspace_args_(id, name, cloud, project)
    if args['id'] is not None:
        return args['id']
    return anyscale.workspace.get(**args).id


def workspace_dir(workspace_id: str) -> str:
    """Where the workspace's ssh config, key and token are kept: `~/.dbx/anyscale/<id>`."""
    return os.path.join(HOME, workspace_id)


def proxy(workspace_id: str):
    """Become the WebSocket proxy to the workspace, with a fresh token: an ssh ProxyCommand, so it never returns.

    stdin and stdout are ssh's channel from the start, so whatever the Anyscale
    SDK prints while the workspace is looked up goes to stderr.
    """
    with contextlib.redirect_stdout(sys.stderr):
        sdk = PrivateWorkspaceSDK()
        hostname = _refresh_(sdk, _running_workspace_(sdk, id=workspace_id))
    script = importlib.resources.files('anyscale.utils') / 'ssh_websocket_proxy.py'
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, str(script), f"wss://{hostname}{WSS_PATH}",
                              '--token-file', _token_path_(workspace_id)])


def storage_options_main(argv=None):
    """`dbx.anyscale_storage_options`: print the workspace's storage options, as `DBX_STORAGE_OPTIONS` reads them."""
    args = _workspace_parser_("Print DBX_STORAGE_OPTIONS for a datalake on a running Anyscale workspace.").parse_args(argv)
    with _cli_errors_('dbx.anyscale_storage_options'):
        options = storage_options(id=args.id, name=args.name, cloud=args.cloud, project=args.project)
    print(';'.join(f"{k}={v}" for k, v in options.items()))


def datalake_main(argv=None):
    """`dbx.anyscale_datalake`: print `--root` on the workspace as a datalake url, as `DBX_DATALAKE` reads it."""
    parser = _workspace_parser_("Print DBX_DATALAKE for a datalake on a running Anyscale workspace.")
    parser.add_argument('--root', required=True, help="the datalake's absolute path on the workspace")
    args = parser.parse_args(argv)
    with _cli_errors_('dbx.anyscale_datalake'):
        print(datalake(args.root, id=args.id, name=args.name, cloud=args.cloud, project=args.project))


def main(argv=None):
    """`python -m dbx.anyscale proxy <id>`: the ProxyCommand `ssh_config()` writes."""
    parser = argparse.ArgumentParser(prog='python -m dbx.anyscale')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('proxy', help="tunnel ssh's stdin and stdout to the workspace").add_argument('id')
    args = parser.parse_args(argv)
    with _cli_errors_(f"dbx.anyscale proxy {args.id}"):
        proxy(args.id)


@contextlib.contextmanager
def _cli_errors_(prog: str):
    """An error as one line on stderr and exit status 1, not a traceback: what a backquoted command should leave."""
    try:
        yield
    except Exception as e:
        sys.exit(f"{prog}: {type(e).__name__}: {e}")


def _workspace_args_(id, name, cloud, project) -> dict:
    """The workspace's coordinates, each given or else from the environment; cloud and project only with a name."""
    if id is None and name is None:
        id = next((os.environ[v] for v in ID_ENV_VARS if os.environ.get(v)), None)
        if id is None:
            name = os.environ.get(NAME_ENV_VAR) or None
    if id is None and name is None:
        raise ValueError(f"No Anyscale workspace: pass --id or --name, or set {' or '.join(ID_ENV_VARS)} or {NAME_ENV_VAR}")
    if id is None:
        cloud = cloud or os.environ.get(CLOUD_ENV_VAR) or None
        project = project or os.environ.get(PROJECT_ENV_VAR) or None
    return dict(id=id, name=name, cloud=cloud, project=project)


def _running_workspace_(sdk, **args):
    """The workspace's model, which must be running: a stopped one has no cluster to reach."""
    model = sdk._resolve_to_workspace_model(**args)
    state = sdk._get_workspace_status(model)
    if state != WorkspaceState.RUNNING:
        raise RuntimeError(f"Anyscale workspace {model.name} ({model.id}) is {state}, not RUNNING")
    return model


def _refresh_(sdk, model) -> str:
    """Write the workspace's current key and a fresh token where its ssh config reads them; return its public hostname."""
    target = sdk._resolve_ssh_target(model)
    if not target.hostname or not target.token:
        raise RuntimeError(f"Anyscale workspace {model.name} ({model.id}) has no ssh tunnel: no public hostname or token")
    _write_private_(_key_path_(model.id), target.private_key)
    _write_private_(_token_path_(model.id), target.token)
    return target.hostname


async def _prepare_(workspace_id: str, config: str):
    """Pin the workspace's host key in its known_hosts, and make its sshd serve sftp if it does not."""
    host = _host_(workspace_id)
    async with asyncssh.connect(host, config=config, known_hosts=None) as conn:
        key = conn.get_server_host_key().export_public_key('openssh').decode().split()
        _write_private_(_known_hosts_path_(workspace_id), f"[{host}]:443 {key[0]} {key[1]}\n")
        if await _serves_sftp_(conn):
            return
        await conn.run(ENABLE_SFTP, check=True)
    # The sshd this connection was forked from read its config before the
    # change: only a new connection sees the subsystem, once sshd has re-exec'd.
    for _ in range(10):
        await asyncio.sleep(1)
        async with asyncssh.connect(host, config=config) as conn:
            if await _serves_sftp_(conn):
                return
    raise RuntimeError(f"Anyscale workspace {workspace_id} still refuses sftp after its sshd was given a subsystem")


async def _serves_sftp_(conn) -> bool:
    try:
        async with conn.start_sftp_client():
            return True
    except asyncssh.ChannelOpenError:
        return False


def _workspace_parser_(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--id', help=f"the workspace's id (default: ${ID_ENV_VARS[0]}, else ${ID_ENV_VARS[1]})")
    parser.add_argument('--name', help=f"the workspace's name, when no id is given (default: ${NAME_ENV_VAR})")
    parser.add_argument('--cloud', help=f"with --name: the workspace's cloud (default: ${CLOUD_ENV_VAR})")
    parser.add_argument('--project', help=f"with --name: the workspace's project (default: ${PROJECT_ENV_VAR})")
    return parser


def _host_(workspace_id: str) -> str:
    """The ssh Host the workspace is reached as: its id, lowercased, since a url's host is."""
    return workspace_id.lower()


def _key_path_(workspace_id: str) -> str:
    return os.path.join(workspace_dir(workspace_id), 'key.pem')


def _token_path_(workspace_id: str) -> str:
    return os.path.join(workspace_dir(workspace_id), 'token')


def _known_hosts_path_(workspace_id: str) -> str:
    return os.path.join(workspace_dir(workspace_id), 'known_hosts')


def _write_private_(path: str, content: str):
    """Write *content* to *path*, readable only by this user, even if it was not before."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    with open(path, 'w', opener=lambda p, f: os.open(p, f, 0o600)) as f:
        os.chmod(path, 0o600)
        f.write(content)


if __name__ == '__main__':
    main()
