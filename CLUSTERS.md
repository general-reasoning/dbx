# Clusters and remote datalakes

dbx separates where the data are from where the code runs:

- **Where the data are** is the datalake: `DBX_DATALAKE`, read with
  `DBX_STORAGE_OPTIONS`. It can be a local directory, a cloud bucket, a
  directory on another host over ssh, or an
  [Anyscale workspace](#an-anyscale-workspace-as-a-datalake).
- **Where the code runs** is this process by default. A block can be
  instantiated on [a Ray head on the datalake's host](#a-ray-head-on-the-datalakes-host-sshray)
  or run on [a Slurm job](#slurm). A pipeline's call, or a shell command, can be
  [submitted to a cluster](#submitting-a-pipelines-call-to-a-cluster-and-watching-it):
  a Kubernetes Job, journaled and followed from any process.

Contents:

- [The datalake: DBX_DATALAKE and DBX_STORAGE_OPTIONS](#the-datalake-dbx_datalake-and-dbx_storage_options)
- [A datalake on another host, over ssh](#a-datalake-on-another-host-over-ssh)
- [An Anyscale workspace as a datalake](#an-anyscale-workspace-as-a-datalake)
- [A Ray head on the datalake's host: SshRay](#a-ray-head-on-the-datalakes-host-sshray)
- [Slurm](#slurm)
- [Submitting a pipeline's call to a cluster, and watching it](#submitting-a-pipelines-call-to-a-cluster-and-watching-it)

## The datalake: DBX_DATALAKE and DBX_STORAGE_OPTIONS

`DBX_DATALAKE` is the datalake a block is stored in when it is not given
`datalake=`. It is a local path or an fsspec URL: `/data/lake`,
`gs://bucket/lake`, `abfss://...`, `ssh://host/path/to/lake`.

`DBX_STORAGE_OPTIONS` is passed to fsspec with every datalake path, as
`key=val;key=val`. It holds what the URL cannot say, such as credentials or an
ssh config file. If it holds a `protocol`, that protocol overrides the path's.

To see what a shell will use:

```bash
echo "$DBX_DATALAKE | $DBX_STORAGE_OPTIONS"
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
dbx.slurm.exec 'module.function()' n_gpus=1 mem=32G n_cpus=8 time=04:00:00 partition=gpu
```

| argument    | default    | what it sets                                   |
|-------------|------------|------------------------------------------------|
| `n_gpus`    | `0`        | `--gres=gpu:N`, and Ray's `--num-gpus`, per node |
| `mem`       | `8G`       | `--mem`, in Slurm's own format                 |
| `n_cpus`    | `1`        | `--cpus-per-task`, and Ray's `--num-cpus`, per node |
| `n_nodes`   | `1`        | `--nodes`: the head on the first, a Ray worker on each other |
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

## Submitting a pipeline's call to a cluster, and watching it

A pipeline's call -- `build_tree()`, `try_specialize_tree(...)`, any method of its entrypoint -- or a
plain shell command can be run on another machine: today a Kubernetes cluster with Kueue
(`dbx.kube`), later a Slurm partition. The submission, the run's identity and its output are
journaled, so a run is found and followed again from any process, any later session.

```bash
dbx.pprint "(p := myproject.pipelines.train('preset', n_gpus=8)).runner('lambda').build_tree().wait() and p.entry.valid()"
dbx.pprint "myproject.pipelines.train('preset').runner().runs()"           # its runs on every cluster, as a dash
dbx.pprint "myproject.pipelines.train('preset').runner().runs()[-1].cancel()"   # cancel the dash's last row
dbx.pprint "dbx.runner('myproject.blocks.Train').runs()"               # the runs of every Train, any hash
dbx.kubejobs                       # every user's running Jobs on every cluster (-m: mine; -s all: every status)
dbx.kubequeues -q myteam           # the clusters' Kueue queues: quota, use, pending
```

Modules: `dbx/cluster.py` (clusters, `Runner`, `Run`, the journal), `dbx/kube.py` (Kubernetes),
`dbx/image.py` (an image of a venv's code), `dbx/gpu.py` (a function on a node's GPUs, through Ray),
`dbx/anyscale.py` (what a Job or an image needs on Anyscale; also a workspace as a datalake, above). `EXECUTORS.md` is the earlier proposal.

### 1. Clusters: values, passed where they are used

A run goes to a **cluster**: a `dbx.kube.KubeCluster`, and later a `SlurmCluster`, each kind a
subclass of `dbx.cluster.Cluster`. dbx defines none, and keeps none: a cluster is a value, defined
in the project and passed at call time. Nothing is registered.

```python
# myproject/clusters.py
from dbx import kube
LAMBDA = kube.KubeCluster(
    'lambda',                                   # the cluster's name; kubectl's context by default
    namespace='kueue-team-x', queue='x-queue', cpu_queue='cpu-only-queue', priority='normal-priority',
    mounts={'/mnt/team-x': 'default-pvc-team-x'},  # path in the Job -> volume claim mounted there
    readonly=(),                                # mount paths mounted read-only
    n_cpus_per_gpu=20, mem_gib_per_gpu=187,     # a Job's CPUs and memory per GPU, by default
    image='registry/image:tag',                 # the Job's image, unless given
    storage={'/mnt/team-x': 'sftp://host/data'},  # the cluster's own filesystem, by URL (see 3)
    storage_options='myproject.clusters:sftp_options',  # fsspec's options for those URLs (see 3)
    user='me',                                  # prefixes Job names; $USER by default
    options=dict(venv='/mnt/team-x/me/venv', ray=True),  # job options always given here
)
CLUSTERS = {c.name: c for c in (LAMBDA,)}
```

```python
my_pipeline('preset').runner(LAMBDA).build_tree()
```

**Names**, for where a value cannot be passed -- a `dbx.pprint` one-liner, `dbx.kubejobs`, a new
shell. The project names its clusters as an **entry point** of group `dbx.clusters` in its packaging:
an object that is a Cluster, or a dict or list of them.

```toml
[project.entry-points."dbx.clusters"]
myproject = "myproject.clusters:CLUSTERS"
```

`dbx.cluster.cluster('lambda')` -- and so `runner('lambda')`, `dbx.kubejobs`, `dbx.kubequeues` --
loads every `dbx.clusters` entry point and looks the name up in what they name. It does that on
each call and keeps nothing: what a name means is a function of the installed packages, not of
what this process happened to import. `dbx.cluster.clusters()` lists them. (sctx: `sctx/clusters.py`,
`LAMBDA` and `CRUSOE`; entry point `sctx = "sctx.clusters:CLUSTERS"`.)

**A journaled run carries its cluster.** `job:submit`'s record holds the cluster itself as
`cluster_spec` -- its class and its fields -- so a `Runner`'s `runs()` rebuilds each run's cluster
from its record, with no lookup: a run submitted to a cluster defined in a notebook is still
followed from another process, and a later change to a project's `LAMBDA` does not rewrite where an
earlier run went.

The site's facts -- namespace, queues, claims, shares -- live in the project, not in dbx; so does the
choice of image and venv.

### 2. Submitting

`Pipeline.runner(cluster, *, mode=None, force=False, **options)` hands out the pipeline's runs on
*cluster*: a `Runner` of the cluster's kind (`cluster.RUNNER`: `dbx.kube.KubeRunner` for a KubeCluster). It
launches -- `.build_tree()`, `.build()`, `.launch('<method>', ...)` for any other method of the
entrypoint -- and lists, tables and cancels the runs (section 5). A call runs there as:

```
dbx.pprint --capture-output "dbx.eval('<the pipeline's quote>').entry.<method>(<args>)"
```

with `DBX_DATALAKE` the run's datalake. The quote rebuilds the same pipeline -- the same hashes,
the same operational kwargs -- so the run builds the very blocks the call would build here, in
that datalake. It returns a `Run` (section 5).

**The run's datalake**, its `DBX_DATALAKE`, is the most specific one given: the pipeline's own,
when it was given `datalake=` -- its quote carries it, so it builds there wherever it runs -- else
`DBX_DATALAKE` in the runner's `env=`, else in the cluster's `options['env']` (the cluster knows
where its datalake is), and only else the one the pipeline resolved here. A pipeline not given
one resolves `DBX_DATALAKE` where its quote is evaluated, so on the cluster it picks up the run's;
here it picked up this machine's, which may mean nothing there. A `build`/`build_tree` is skipped
as already valid only when the run's datalake is the pipeline's here: elsewhere, validity here
says nothing.

- **Arguments** must be literals (`ast.literal_eval(repr(x)) == x`): they travel as text.
- **Already built:** `build` and `build_tree` of an entrypoint valid here launch nothing -- a Run
  `skipped` -- unless `force=True`.
- **`mode`** ('kube'), when given, must be the cluster's kind: a check, not a choice.
- **Options** are the kind's, and are refused at `runner(...)` when it does not take them -- a
  misspelling fails before anything is sent. `n_gpus` defaults to the pipeline's `n_gpus`, or 0.
- **A shell command:** `LAMBDA.submit(command, **options)`, the same Job without a
  pipeline; journaled only when given `block=`. `LAMBDA.job(command, **options)` is its manifest.

#### Kube options (`KubeCluster.job`)

| option | |
|---|---|
| `n_gpus`, `n_cpus`, `mem_gib` | GPUs; CPUs and memory default to the cluster's per-GPU share times `n_gpus` (8 / 32 GiB without GPUs). Kueue admits on all three: lower `n_cpus`/`mem_gib` when the queue waits on those. |
| `image` | the Job's image; else `KubeCluster.image`; else, with `anyscale=True`, `$ANYSCALE_RAY_IMAGE_URI` |
| `venv` | a venv to `source` first -- one built for the Job's image |
| `revision`, `pin_source` | git pins, as `dbx.remote(revision=...)`: `'dbx_rev:project_rev'`, or True for both HEADs (refused when dirty). Cloned in the Job from `pin_source` (default: the repos' own paths -- reachable where the cluster mounts them; give git URLs otherwise) and put first on `PYTHONPATH`. |
| `workdirs` | `{src: target}`: trees the run needs that are not in git (section 3) |
| `pack` | send each workdir as one archive, unpacked by the Job (section 3) |
| `env`, `setup`, `cwd` | environment variables; shell lines run first; the working directory |
| `ray`, `ray_bin` | start a single-node Ray head first, for code that calls `ray.init(address='auto')` |
| `anyscale` | False by default. True adds what an Anyscale image needs (`dbx.anyscale.job_options`): its `ray` user (1000:100), its Ray (`/home/ray/venv/bin/ray`), `ANYSCALE_WORKING_DIR`, and its image by default. |
| `run_as`, `priority`, `deadline_hours`, `ttl_days`, `name` | the pod's uid/gid; Kueue priority class; deadline (24 h); how long a finished Job is kept (7 days); a name the Job's is made from |
| `dry_run` | section 4 |

`KubeCluster.options` are defaults for all of them; `env` and `setup` add to the cluster's.

### 3. Code and data on the cluster

What the run imports and reads must be there, at the paths the expression names:

- **A cluster that mounts the paths seen here** (no `storage`; e.g. a claim that is the workspace's
  own storage) runs the live tree: a venv, editable installs, the datalake are simply there. An edit
  made here while it runs is seen by it -- pin with `revision` when that matters.
- **A cluster with its own filesystem** (`storage={local_prefix: url_prefix}`) has the same paths
  under another root, reached from here by URL: `KubeCluster.url(path)`, opened with
  `KubeCluster.fsspec_options()`. Those are `storage_options`: a dict, or `'module:function'` returning
  one when they are not data -- an SSH socket, an auth strategy -- and so cannot be journaled with the
  cluster. `workdirs={src: target}` are synced there before the Job is created -- file by file,
  skipping those whose size and mtime match, keeping modes (console scripts stay executable) and
  symlinks, deleting nothing (`KubeCluster.sync(src, target)`), with a progress bar per tree on a terminal.
  **`pack=True`** (a job option, or in `KubeCluster.options` for every run there) sends each tree as one
  archive instead, for a link where every file costs a round trip: the tree's digest -- each file's
  path, size, mtime, mode and link target, `__pycache__` and `*.pyc` left out -- names a gzip
  tarball in `.dbx-packs/` beside the target, uploaded (with a bar) only when the cluster does not
  have it. The Job unpacks it first, under a lock, into a sibling swapped in for the target, unless
  the target's `.dbx-pack` names that digest already: the target BECOMES the tree, so a target that
  is a mount point, or holds one or the datalake, is refused. Over Lambda's tailnet SFTP (~0.12 s a
  file, ~1.6 MB/s), an 8,000-file, 160 MB venv is ~20 min file by file, ~2 min packed, and seconds
  when unchanged. A block built there is copied back with
  `UNSAFE_copy_from(cluster.url(block.anchorkeypath))` inside `with cluster.fsspec_defaults():` --
  `UNSAFE_copy_from` opens its source with the block's own options, so the cluster's are set as
  fsspec's defaults for its protocols, and only while the block copies. Its anchorkeypath is the
  same path on both sides.
- **The code itself:** `revision` pins the dbx and project repos to commits, cloned in the Job.
- **The code and the venv, as an image:** `image=dbx.image.build(venv, repo=..., anyscale=True)` --
  the venv's editable installs as they are on disk, the packages it adds over the base image, and any
  `extra` trees, appended to the base image as one layer (`crane`, no Docker build). The tag is the
  layer's sha256, so the same code is the same image, an unchanged rebuild pushes nothing, and the
  manifest a run journals names the code it ran. Nothing to sync, and no venv to keep on the cluster.

Within the run, `dbx.gpu.run(fn, calls, ...)` farms `fn` out over the node's GPUs, one Ray task per
call, each with its own log; on a Kube Job that needs `ray=True`, the Job's own Ray head.

### 4. Dry runs

`dry_run` is an option of every kind:

| `dry_run` | what happens | journaled |
|---|---|---|
| `True` | the record and the Job's manifest are made (`Run.record['manifest']`); nothing is contacted | no |
| `'server'` | kube: the API server validates the Job (`kubectl create --dry-run=server`), creating nothing | no |
| `'cluster'` | a small real run on the cluster (kube: no GPUs, 2 CPUs, 8 GiB, no Ray) that only evaluates the pipeline's quote there and `dbx.cluster.verify`s the entry: every import, the venv, the revision, the workdirs, the datalake, and the same hash as here. It builds nothing; the call it stands for is in its env as `DBX_SUBMIT_CALL`. `run.wait(); run.output()` shows the entry's anchorkeypath, whether it is valid there, and which dbx it imported. A hash that differs there is a ValueError: the code or its configuration is not the same. | yes |

### 5. The journal, the run, and its output

A run is journaled by both sides, each in its own journal, on the **pipeline**: in the
`PipeJournal` (`{datalake}/.pipelines`) of the datalake it writes in -- the pipeline launched it:
its quote, its operational kwargs, its specializations. Each entry has the pipeline's `entrypath`
and the run's `cluster` and `run` (the Job's name).

**The submitter**, in the datalake it submitted from:

| event | note |
|---|---|
| `job:submit` | before the run is launched; a file holding the whole record as JSON: `executor`, `cluster` (its name), `cluster_spec` (the cluster itself; `target` and `target_spec` in records from before 2026-11-09, read as these), `id` (the Job), `method`, `command`, `expression`, `session` (the submitting command's, which the run joins), `pipeline_anchor`, `datalake`, `submitted`, and the kind's own -- kube: `cluster`, `context`, `namespace`, `image`, `revision`, `workdirs` (what was synced or packed), `manifest` (the Job itself: `kubectl create -f` it to run the same again) |
| `job:complete` | it saw the run complete: `Run.wait`, or a `Runs.dash` that asked the cluster |
| `job:fail` | launching failed, or it saw the run fail (with its log tail) |
| `job:cancel` | `Run.cancel` |

**The run**, in the datalake it runs in -- what its Job evaluates is `dbx.cluster.run(pipeline,
'<method>', ...)`:

| event | note |
|---|---|
| `job:start` | before the call |
| `job:end` | after it: `ok`, or what it raised |

The two journals are one where the cluster mounts the submitter's datalake (Crusoe), and two where
the cluster has a filesystem of its own (Lambda): there, the same path is the cluster's copy, and
`Runs.dash` reads the run's events from it by URL. Neither side decides for the other: each says what
it saw, and `dash` shows both. They agree on `(session, run)`. The run joins the submitting
command's session (`DBX_SESSION`), and its Job sets `DBX_CLUSTER` and `DBX_RUN`, so every entry it
writes -- its `job:start`/`job:end`, each datablock's `build:start`, `build:end`, its exec-journal
row -- names the run. Which run built a datablock is in its own build entries. Entries written from
no run, and those written before the columns existed, have None. Runs journaled before 2026-11-04
were `submit:BEGIN`, `submit:END`, `submit:ERROR`, `submit:CANCEL` -- some on the entrypoint, in its
own build journal -- and are read as these.

`PipeJournal.collect` (`dbx.pipejournal`) reads only `create` entries, and `UNSAFE_compact` keeps
every run's entry, an unjournaled class's too. A `Runner` reads the runs for you, from another
process, a later session:

| | whose runs |
|---|---|
| `p.runner(cluster)`, `p.runner()` | the pipeline's (its hash), on *cluster* or on every cluster |
| `dbx.runner(dblk)` | those whose Job built the datablock (its build entries' `run`), and those of the pipelines it is the entry of |
| `dbx.runner(DatablockClass)`, `dbx.runner('<anchor>')` | the same, for every datablock of the class, whatever its hash; `datalake=`, or `DBX_DATALAKE` |
| `dbx.runner(PipelineClass)` | every pipeline's of the class, whatever its hash |

- `runs(open=None, adopted=False)`: each submission as its `Run`, oldest first, in a `Runs` -- a
  list that prints as their dash, each row numbered as its index, so `runs()[-1].cancel()` cancels
  the last row printed. A run is closed when the submitter's `job:complete`, `job:fail` or
  `job:cancel` is journaled (`run.closing` is that event and its time). *adopted* adds the runs
  of the builds the datablock adopted through a specialization: those its redirection reads from.
  The printout's columns are `run, cluster, method, status, submitted, elapsed`; it asks the
  cluster for each open run's status, and journals nothing.
- `runs().dash(live=True, pod=True, close=True)`: the whole table, row `i` being `runs()[i]` -- `run,
  cluster, entry (the entry's key), method, status, submitted, started, ended, observed, elapsed,
  outcome, event`. The submitter's: `submitted`, and for a closed run its `event` and when it was
  `observed`. The run's own (*pod*, read where it ran): `started`, `ended` and `outcome` (`ok`, or
  what it raised). An open run's status is asked of its cluster (`gone` once the Job is deleted),
  and a run found over is closed then (*close*): its `job:complete` or `job:fail` journaled, as
  `wait` would.
- `cancel()`: cancels the runs still pending or running, each journaled `job:cancel` where its
  `job:submit` is -- its pipeline, even when found by class (the pipeline's quote in its record is
  evaluated again for it). `runs().cancel()` and `runs()[i].cancel()` do the same for those listed.
- A `Run`'s own repr is what is journaled -- `KubeRun(lambda, <job>, build_tree, submitted
  2026-11-04 23:15, open)` -- without asking the cluster.

`dbx.cluster.runs(what, cluster)` is the same, as a function.

**The run's own record is in the datalake it ran in** (`DBX_DATALAKE` there). It runs
`dbx.pprint --capture-output` under the submitting command's session (`DBX_SESSION`, recorded as
the record's `session`). So under that session, and that `run`, there:

- its exec-journal row: `Run.command()`, which is
  `dbx.execjournal(session='^<session>$', run='^<run>$', iloc=0)` in the record's datalake -- at
  `KubeCluster.url` with its `fsspec_options` on a cluster with `storage` -- or the `datalake=` given;
  the `run` tells it from the submitting command's own row;
- every build-journal entry it wrote, each with its `log` -- the capture it was written under;
- its captured stdout and stderr, which the row lists as `output_captures`: the master's (the
  command's process) and a worker's per worker process, each written when its process ends.

The journal is the one record of the output, and the one way to it: `run.output()`, which is
`run.command().output()`, wherever the cluster keeps the datalake.

```python
run.output()                                                   # once the run has ended; run.logs() while it runs
run.output(basename='worker-')                                 # a worker process's capture
```

```bash
# on the cluster's own machines, with DBX_DATALAKE set to its datalake
dbx.pprint "dbx.execjournal(session='^<session>$', run='^<run>$', iloc=0).output()"
```

The row is written when the run ends. Until then -- while a Job runs -- `Run.logs()` is its pod's
log, through kubectl, until the Job is deleted (`ttl_days`). Over an sftp:// URL the journal readers
work serially: one SFTP connection cannot serve several threads at once.

### 6. Watching

| | |
|---|---|
| `run.status()` | 'pending' (not admitted, or no pod yet), 'running', 'complete', 'failed'; 'skipped', 'dry-run' |
| `run.reason()` | kube: why a pending Job waits -- Kueue's message on its Workload, e.g. a quota |
| `run.logs(tail=-1, follow=False)` | the pod's log; `follow=True` streams it until the pod ends |
| `run.wait(poll=30, timeout=None)` | until complete (True) or *timeout* (False); raises with the log tail on failure; journals END / ERROR |
| `run.cancel()` | deletes the Job and its pod; journals CANCEL |
| `run.command()` | the run's exec-journal row; `.output()` prints what it printed (section 5) |
| `dbx.cluster.runs(block)` | the Runs of every submission of a block, from its journal |
| `LAMBDA.jobs(user=, mine_only=, status=)` | the cluster's Jobs, every user's and running by default (`user=` one user's, `mine_only=True` mine, over any `user=`; `status='all'`, 'pending', 'complete', 'failed'), newest first: status, GPUs (all its pods'), pods, times, the expression, and Kueue's reason for a pending one |
| `LAMBDA.queues(resource=, queues=, all=)` | the Kueue queues the cluster sees: `n_gpus_quota, _used, _free, _borrowed, _cohort` (`n_cpus_...` for `cpu`, `mem_gib_...` for `memory`), and `n_workloads_pending, n_workloads_admitted` (one workload per Job). `borrowed` is borrowed *by* the queue, beyond its quota, so `used` = `quota + borrowed` when over it; `mine` marks the queues your namespace's LocalQueues feed; `_cohort` is the cohort's quota, what borrowing can take the queue to |
| `dbx.kubejobs [CLUSTER ...] [-s STATUS\|all] [-u USER \| -m]` | `jobs` of the `dbx.clusters` entry points' clusters (or those named), as one table; `dbx.kube.jobs(...)` returns it |
| `dbx.kubequeues [CLUSTER ...] [-r RESOURCE] [-q REGEX] [-a]` | `queues` of those clusters, as one table, a kubectl context shared by two queried once; `dbx.kube.queues(...)` returns it |

kubectl itself, for what these do not cover: `kubectl --context C -n NS describe job/<id>`,
`get workload`, `describe workload <name>`, `get events --sort-by=.lastTimestamp`.

### 7. Adding a kind

A kind subclasses three classes of `dbx.cluster`:

- `Cluster`, with `MODE` (its name, e.g. 'slurm'), `RUNNER` (its Runner class) and a `name`;
- `Runner`, implementing `_prepare_(command, **options)` (the record, its `id` among it),
  `_create_(record)` (launch it), and as needed `_check_(options)` (refuse options it does not take),
  `_stage_(record, **options)` (get files in place), `_dry_run_(record, dry_run)` and
  `_check_options_()` (the least a `dry_run='cluster'` run takes); `RUN` is its Run class;
- `Run`, implementing `_status_()`, `_logs_(tail)`, `_cancel_()`.

The base classes do the rest -- the launched expression, the literal check, the skip, the sessions,
the journal, `wait`, `runs`, `command` -- the same for every kind.

### 8. Not done

- Slurm as a kind (`SlurmCluster`, `SlurmRunner`, `SlurmRun`), reusing `SlurmRayCluster`.
- Resources declared by the blocks (`EXECUTORS.md` section 3): today `n_gpus` comes from the pipeline's `n_gpus`.
- Fan-out: one run per tree.
- `UNSAFE_copy_from` given the source's storage options, rather than `fsspec_defaults`.

## Appendix: cheatsheet

`dbx`, `dbx.exec` and `dbx.pprint` evaluate their argument as Python statements and return the last one's
value (`dbx.pprint` prints it). A dotted name is imported first, so `dbx.cluster.clusters()` works, but a
bare name must already be defined: `dbx clusters` fails with `NameError: name 'clusters' is not defined`.
`dbx.clusters` is the entry-point group clusters are registered under (section 1), not a command.

Set `P` once to a pipeline's expression; the examples below expand it inside their double quotes:

```bash
P="myproject.pipelines.train('preset')"
```

**Clusters** (`dbx.cluster`: every kind; `dbx.kube`: the Kube ones only).

```bash
dbx.pprint "dbx.cluster.clusters()"              # every cluster the dbx.clusters entry points name, by name
dbx.pprint "dbx.cluster.clusters('kube')"        # of one kind only; = dbx.kube.clusters()
dbx.pprint "dbx.cluster.cluster('lambda')"       # one Cluster
```

**Running a pipeline** (a pipeline's `.runner(cluster, **options)` is a `Runner`; what it launches is a `Run`, a
`KubeRun` on Kube).

```bash
dbx.pprint "$P.runner('lambda', n_gpus=8).build_tree().wait() and $P.entry.valid()"
dbx.pprint "$P.runner('lambda', dry_run='cluster').build_tree()"    # CPU-only check: imports, code hash
dbx.pprint "$P.runner('lambda', n_gpus=0, n_cpus=30).launch('<method>', <args>)"   # any entry method
dbx.pprint "$P.runner('lambda', workdirs={'/mnt/team-x/code': '/mnt/team-x/code'}, venv='/mnt/team-x/venv').build()"
```

Common options: `n_gpus`, `n_cpus`, `mem_gib`, `venv`, `workdirs`, `env`, `image`, `name`, and `dry_run`
(`True`: the record only; `'cluster'`: the check above). Sections 2-4.

**Watching and cancelling** (section 6). `runs()` prints as a table whose row `i` is `runs()[i]`.

```bash
dbx.pprint "$P.runner().runs()"                   # its runs on every cluster, oldest first
dbx.pprint "$P.runner('lambda').runs(open=True)"  # only those not complete, failed or cancelled
dbx.pprint "$P.runner().runs().dash()"            # every column, with the pod and what the run journaled
dbx.pprint "$P.runner().runs()[-1].status()"      # also .logs(tail=200), .reason(), .wait(), .cancel()
dbx.pprint "$P.runner().runs()[-1].output()"     # what it printed, once it ended (section 5)
dbx.pprint "$P.runner('lambda').cancel()"         # every open run there
dbx.pprint "dbx.runner('myproject.blocks.Train').runs()"   # every Train's, whatever its hash
dbx.pprint "dbx.cluster.runs($P, 'lambda', open=True)"      # = $P.runner('lambda').runs(open=True)
dbx.pprint "$P.runner('lambda').cluster.jobs()"             # the running Jobs there, anyone's, any pipeline: = dbx.kubejobs lambda
```

**What is on a cluster, and running a command there** (methods of `dbx.kube.KubeCluster`; `C` is one).

```bash
C="dbx.cluster.cluster('lambda')"
dbx.pprint "$C.jobs()"                           # every user's running Jobs there
dbx.pprint "$C.jobs(mine_only=True, status='all')"    # mine, every status; why a pending one waits
dbx.pprint "$C.user"                             # who "my" is: the cluster's user (else $USER), my Jobs' name prefix
dbx.pprint "$C.jobs(user='hank', status='failed')"   # one user's
dbx.pprint "$C.queues(queues='myteam', resource='memory')"   # its Kueue queues; resource cpu|memory|nvidia.com/gpu
dbx.pprint "$C.submit('nvidia-smi', n_gpus=1, name='smi')"   # a shell command as a Job: a KubeRun
dbx.pprint "$C.job('nvidia-smi', n_gpus=1)"                  # its manifest, not submitted
dbx.pprint "$C.sync('/mnt/team-x/code')"         # a tree onto the cluster's storage; workdirs= does this for a run
```

`submit` journals nothing unless given `block=`: it is a shell command, not a pipeline run.

**Over several clusters** (`dbx.kube`; consoles too).

```bash
dbx.kubejobs                                     # every user's running Jobs on every cluster
dbx.kubejobs lambda -s failed                    # one cluster; -s running|pending|complete|failed|all
dbx.kubejobs lambda -m -s all                    # mine, every status (-u NAME: one user's)
dbx.kubequeues                                   # every queue's GPU quota and use
dbx.kubequeues lambda -q myteam -r memory        # queues matching a regex; -r cpu|memory; -a every queue
dbx.pprint "dbx.kube.jobs(['lambda'], status='running')"     # dbx.kubejobs as a DataFrame
dbx.pprint "dbx.kube.queues(resource='cpu')"                 # dbx.kubequeues as a DataFrame
```

**The raw data behind them.** `jobs()` and `queues()` are kubectl calls, parsed; these show all of it. `C` and `NS` are
the cluster's `context` and `namespace` (`dbx.cluster.cluster('lambda').context`, `.namespace`).

```bash
kubectl --context $C -n $NS get job -o json                  # jobs(): every Job of the namespace, everyone's
kubectl --context $C -n $NS get job <name> -o yaml           # one Job, whole
kubectl --context $C -n $NS get workloads -o json            # jobs()' reason: Kueue's Workload per Job, its conditions
kubectl --context $C -n $NS get pods -l job-name=<name> -o wide   # a Job's pods, and their nodes
kubectl --context $C get clusterqueue -o json                # queues(): quotas, .status.flavorsUsage, workload counts
kubectl --context $C -n $NS get localqueue -o json           # queues()' mine: the ClusterQueue each LocalQueue feeds
# what Kueue reserved per admitted workload: pods x GPUs, by flavor
kubectl --context $C -n $NS get workloads -o json | python -c "import json, sys
for w in json.load(sys.stdin)['items']:
    a = w.get('status', {}).get('admission')
    if a: print(w['metadata']['name'], [(p['count'], p.get('resourceUsage', {}).get('nvidia.com/gpu'), p.get('flavors', {}).get('nvidia.com/gpu')) for p in a['podSetAssignments']])"
```

What `jobs()` takes from a Job: `metadata.name`; its status from `status.conditions`, `active`, `succeeded`, `failed`;
`n_gpus`, the first container's `resources.requests["nvidia.com/gpu"]` times `spec.parallelism` (`n_pods`); the times
`metadata.creationTimestamp`, `status.startTime`, `status.completionTime`; the command from the container's
`DBX_SUBMIT_EXPRESSION`, or the `dbx/command` annotation; and whether it is yours from the `dbx/user` label or the name's
`<user>-` prefix.
