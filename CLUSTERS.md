# Clusters and remote datalakes

dbx separates where the data are from where the code runs:

- **Where the data are** is the datalake: `DBX_DATALAKE`, read with
  `DBX_STORAGE_OPTIONS`. It can be a local directory, a cloud bucket, a
  directory on another host over ssh, or an
  [Anyscale workspace](#an-anyscale-workspace-as-a-datalake).
- **Where the code runs** is this process by default. A block can be
  instantiated on [a Ray head on the datalake's host](#a-ray-head-on-the-datalakes-host-sshray)
  or run on [a Slurm job](#slurm).

Contents:

- [The datalake: DBX_DATALAKE and DBX_STORAGE_OPTIONS](#the-datalake-dbx_datalake-and-dbx_storage_options)
- [A datalake on another host, over ssh](#a-datalake-on-another-host-over-ssh)
- [An Anyscale workspace as a datalake](#an-anyscale-workspace-as-a-datalake)
- [A Ray head on the datalake's host: SshRay](#a-ray-head-on-the-datalakes-host-sshray)
- [Slurm](#slurm)

## The datalake: DBX_DATALAKE and DBX_STORAGE_OPTIONS

`DBX_DATALAKE` is the datalake a block is stored in when it is not given
`datalake=`. It is a local path or an fsspec URL: `/data/lake`,
`gs://bucket/lake`, `abfss://...`, `ssh://host/path/to/lake`.

`DBX_ROOT` and `DBX_URL` are its names from before the rename. They are
deprecated but still read, and only when `DBX_DATALAKE` is unset. A plain local
path left in `DBX_ROOT` therefore does not matter while `DBX_DATALAKE` is set.
If `DBX_DATALAKE` is unset, that path is read as a local directory on this
machine.

`DBX_STORAGE_OPTIONS` is passed to fsspec with every datalake path, as
`key=val;key=val`. It holds what the URL cannot say, such as credentials or an
ssh config file. If it holds a `protocol`, that protocol overrides the path's.

To see what a shell will use:

```bash
echo "$DBX_DATALAKE | $DBX_ROOT | $DBX_URL | $DBX_STORAGE_OPTIONS"
```

A journal entry records the environment it was built in, `env`, with the
values of secrets withheld. A `DBX_STORAGE_OPTIONS` there shows as
`<withheld>`, because it may hold credentials.

## A datalake on another host, over ssh

An `ssh://` datalake is read through fsspec's sshfs, on asyncssh, which applies
`~/.ssh/config`. `ssh://host/path` names the host as `ssh host` does, user and
all. Put the host and the path into `DBX_DATALAKE` as one URL, and leave
`DBX_STORAGE_OPTIONS` unset:

```bash
export DBX_DATALAKE=ssh://wheelbarrow/home/t-9dkarp/datalake
unset DBX_STORAGE_OPTIONS
hyp --journal
```

`wheelbarrow` is a `Host` entry in `~/.ssh/config`. Its `HostName`, `User`,
`IdentityFile` and `ProxyJump` apply as they do to `ssh wheelbarrow`, so if
`ssh wheelbarrow` logs in, dbx and hyp can read the datalake. The path is the
datalake's directory on that host, the one holding `.journal`.

The URL is for reaching the datalake from another machine. On the datalake's
own host, set `DBX_DATALAKE` to the plain path:
`export DBX_DATALAKE=/home/t-9dkarp/datalake`. With the `ssh://` URL there,
dbx logs into its own host over ssh, and fails with `PermissionDenied` unless
that host accepts a key from itself.

An entry written on the datalake's host records its paths as that host's
local paths. Read from elsewhere over ssh, they are put on the filesystem of
the entry's `entry_path`, so they are read from the host they were written on.

When it fails:

- **`OSError: [Errno 45] Operation not supported: '/home/...'`**: a path meant
  for the datalake's host is being read on this machine. Set `DBX_DATALAKE` to
  its `ssh://` URL.
- **`OSError: [Errno 65] No route to host`, though `ssh wheelbarrow` works**:
  the failing connection is the one to the `ProxyJump` host. macOS may be
  refusing Python access to local-network addresses. Allow the terminal app
  under System Settings → Privacy & Security → Local Network, then restart it.
- **`gaierror: [Errno 8] nodename nor servname provided, or not known`**: the
  host's name does not resolve from here. Check the VPN, and that the host is
  a `Host` in `~/.ssh/config`.

## An Anyscale workspace as a datalake

`dbx.anyscale` lets dbx and hyp read and write a datalake that lives on a
running Anyscale workspace, from your laptop, over ssh. It connects the same way
`anyscale workspace_v2 ssh` does. The connection goes over HTTPS port 443, so it
needs no VPN and no open ports.

### Setup, once

1. Install dbx with the `anyscale` extra. The `dbx` conda env (`dbx.yml`)
   already has it through `[all]`.

   ```bash
   pip install -e '.[anyscale]'
   ```

2. Log in to Anyscale. Setting `ANYSCALE_CLI_TOKEN` also works.

   ```bash
   anyscale login
   ```

3. Start the workspace. These commands only reach a workspace that is RUNNING.

### Each shell

```bash
export DBX_STORAGE_OPTIONS=`dbx.anyscale_storage_options --name my-workspace`
export DBX_DATALAKE=`dbx.anyscale_datalake --name my-workspace --root /mnt/cluster_storage/datalake`
hyp --journal
```

- `dbx.anyscale_storage_options` prints `config=~/.dbx/anyscale/<workspace id>/config`.
  That is the ssh config for reaching the workspace, which is what an
  Anyscale datalake needs in `DBX_STORAGE_OPTIONS`. Unlike a
  [plain ssh host](#a-datalake-on-another-host-over-ssh), the workspace is not
  in `~/.ssh/config` unless you [include it](#plain-ssh-and-remote-python).
- `dbx.anyscale_datalake` prints `ssh://<workspace id>/mnt/cluster_storage/datalake`.
  `--root` is an absolute path on the workspace.

Anything that reads `DBX_DATALAKE` and `DBX_STORAGE_OPTIONS` now works against
the workspace, for example `hyp --journal`, `hyp --journal ANCHOR`,
`dbx.anchors()` and `dbx.pprint "..."`.

#### Naming the workspace

Name it with `--id expwrk_...`, or with `--name ...`. When several workspaces
share a name, add `--cloud ...` and `--project ...` to pick one. A flag you
leave out falls back on the environment:

| flag        | falls back on                                                  |
|-------------|----------------------------------------------------------------|
| `--id`      | `ANYSCALE_WORKSPACE_ID`, then `ANYSCALE_EXPERIMENTAL_WORKSPACE_ID` |
| `--name`    | `ANYSCALE_WORKSPACE_NAME` (read only when no id is set)        |
| `--cloud`   | `ANYSCALE_CLOUD` (used only with a name)                       |
| `--project` | `ANYSCALE_PROJECT` (used only with a name)                     |

Set the variables once and the commands need only `--root`:

```bash
export ANYSCALE_WORKSPACE_NAME=my-workspace
export DBX_STORAGE_OPTIONS=`dbx.anyscale_storage_options`
export DBX_DATALAKE=`dbx.anyscale_datalake --root /mnt/cluster_storage/datalake`
```

`--id` is faster for `dbx.anyscale_datalake`: with an id it makes no Anyscale
API call, and with a name it has to look the id up.

### From Python

```python
import fsspec
from dbx import anyscale

ws = dict(name='my-workspace')
fs, root = fsspec.url_to_fs(anyscale.datalake('/mnt/cluster_storage/datalake', **ws),
                            **anyscale.storage_options(**ws))
fs.ls(root)
```

### Plain `ssh`, and remote python

To make `ssh <workspace id>` work from a terminal, add this line at the top of
`~/.ssh/config`:

```
Include ~/.dbx/anyscale/*/config
```

This also makes the workspace reachable by anything that starts a process on
the datalake's host through plain `ssh`, such as `hyp --remote-python` and
[`SshRay`](#a-ray-head-on-the-datalakes-host-sshray). Run
`dbx.anyscale_storage_options` first, because it writes the config.

### After the workspace restarts

Run `dbx.anyscale_storage_options` again. It prints the same thing, so you
don't need to re-export anything. It's needed because a restart gives the
workspace a new cluster, with a new host key and a new sshd config (see the
next section). The access token is not a reason to rerun it: a fresh token is
fetched on every connection.

### What it sets up

Everything is kept in `~/.dbx/anyscale/<workspace id>/`. Every file there is
readable only by you.

| file          | what it is                                                       |
|---------------|------------------------------------------------------------------|
| `config`      | an OpenSSH config with one `Host`, the workspace id; asyncssh (through sshfs) and OpenSSH both read it |
| `key.pem`     | the cluster's ssh key                                            |
| `token`       | the latest cluster access token                                  |
| `known_hosts` | the workspace's host key, saved on first connection              |

The config's `ProxyCommand` is `python -m dbx.anyscale proxy <id>`. On every
connection it looks up the workspace's current hostname, key and a fresh token,
then hands the connection to Anyscale's own WebSocket proxy script.

**On the workspace:** the workspace's own sshd runs as `ray` from
`/tmp/ray/ssh/sshd_config`, and that config serves no sftp, which fsspec needs.
If sftp is refused, `dbx.anyscale_storage_options` adds
`Subsystem sftp internal-sftp` to that file and sends the sshd a HUP to reread
it. Sessions that are already open are not affected. It changes nothing else,
and the change goes away when the workspace restarts.

### When it fails

- **`No Anyscale workspace`**: pass `--id` or `--name`, or set one of the
  variables above.
- **`... is TERMINATED, not RUNNING`**: start the workspace.
- **Auth errors from the Anyscale API**: run `anyscale login` again.
- **`Can't create any SFTP connections!`** from sshfs: the workspace's sshd is
  not serving sftp. This is usually because the workspace restarted. Run
  `dbx.anyscale_storage_options` again.
- **To debug the tunnel**: `ANYSCALE_SSH_DEBUG=1 ssh -v -F ~/.dbx/anyscale/<id>/config <id>`.

## A Ray head on the datalake's host: SshRay

Reading a block's topics means instantiating the block, and the instance that
reads the data recorded in a journal entry is the one pinned to that entry's
own revisions of dbx and the project (`trueinst`). `dbx.SshRay` starts a Ray
head on the datalake's host over ssh, so that this instance runs where the
data are and is interrogated from here:

```python
python = dbx.BuildJournalEntry.column(entry, 'python')     # the interpreter the build ran in
cluster = dbx.SshRay.session('host', python=python)
block = entry.trueinst(address=cluster.address,
                       gitrepo=dbx.BuildJournalEntry.column(entry, 'gitrepo'))
block.hash, block.read('classes')                          # forwarded to the pinned worker there
```

- **`python`** is the interpreter on the host whose `ray` starts the head. Its
  Ray must be the same version as this one's, which Ray's client requires.
- **The head** runs in a temporary directory of its own, on ports picked here,
  with a `--temp-dir` of its own. Nothing of another Ray on the host is
  touched, and `ray stop` is never needed.
- **The client port** is reached through `ssh -L`, so the host's ports need not
  be open. `cluster.address` is `ray://localhost:<port>`.
- **Lifetime:** `session` keeps one head per (host, python) for as long as the
  process runs, and stops it at exit. A watchdog on the host stops the head and
  removes its directories once the ssh shell is gone, whether it was told to
  stop, hung up on, or killed.

hyp's `read` menu does this for you, run by `--remote-python`. See
[HYP.md](HYP.md). For an Anyscale workspace, `SshRay` needs the workspace in
`~/.ssh/config`: see [Plain ssh, and remote python](#plain-ssh-and-remote-python).

## Slurm

`dbx.slurm.exec` runs a dbx expression on a Ray cluster that it starts inside
a Slurm job, and cancels the job when the expression is done.
`dbx.slurm.pprint` (or `dbx.slurm.print`) also prints the value, as
`dbx.pprint` does.

`dbx.slurm.eval` is installed, but does not work yet: the function it names,
`dbx.slurm_eval`, does not exist. There is a TODO for it in `pyproject.toml`.

```bash
dbx.slurm.exec 'module.function()' gpus=1 mem=32G cpus=8 time=04:00:00 partition=gpu
```

| argument    | default    | what it sets                                   |
|-------------|------------|------------------------------------------------|
| `gpus`      | `0`        | `--gres=gpu:N`, and Ray's `--num-gpus`, per node |
| `mem`       | `8G`       | `--mem`                                        |
| `cpus`      | `1`        | `--cpus-per-task`, and Ray's `--num-cpus`, per node |
| `nodes`     | `1`        | `--nodes`: the head on the first, a Ray worker on each other |
| `partition` | none       | `--partition`                                  |
| `nodelist`  | none       | `--nodelist`                                   |
| `time`      | `01:00:00` | `--time`                                       |
| `revision`  | none       | the dbx and project revisions to pin the worker to, `dbxsha:projsha` |
| `conda`     | none       | the conda env the worker runs in: Ray's `runtime_env` `conda` |

Any other `key=value` argument is passed to the expression's namespace,
evaluated as Python where it parses and as a string where it does not.

The job's script and its Slurm output go to a directory of its own under
`~/.dbx/slurm/`. From Python, `dbx.slurm_remote(...)` takes the same
arguments and returns the `Remote` handle. The job is left running for you to
use: nothing cancels it but its time limit or `r._slurm.cancel()`, through the
handle's private `_slurm` until it has a public way.
