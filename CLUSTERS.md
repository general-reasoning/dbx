# Clusters: submitting a pipeline's call, and watching it

A pipeline's call -- `build_tree()`, `try_specialize_tree(...)`, any method of its entrypoint -- or a
plain shell command can be run on another machine: today a Kubernetes cluster with Kueue
(`dbx.kube`), later a Slurm partition. The submission, the run's identity and its output are
journaled, so a run is found and followed again from any process, any later session.

```bash
dbx.pprint "myproject.pipelines.train('preset', n_gpus=8).submit('lambda').build_tree()"
dbx.kubejobs                       # my Jobs on every cluster: status, and why a pending one waits
dbx.kubedash -q myteam             # the clusters' Kueue queues: quota, use, pending
```

Modules: `dbx/cluster.py` (targets, `Submit`, `Run`, the journal), `dbx/kube.py` (Kubernetes),
`dbx/anyscale.py` (what a Job on an Anyscale image needs). `EXECUTORS.md` is the earlier proposal.

## 1. Targets: values, passed where they are used

A **target** is where a run goes: a `dbx.kube.Cluster`, and later a Slurm one. dbx defines none,
and keeps none: a target is a value, defined in the project and passed at call time. Nothing is
registered.

```python
# myproject/clusters.py
from dbx import kube
LAMBDA = kube.Cluster(
    'lambda',                                   # the target's name; kubectl's context by default
    namespace='kueue-team-x', queue='x-queue', cpu_queue='cpu-only-queue', priority='normal-priority',
    mounts={'/mnt/team-x': 'default-pvc-team-x'},  # path in the Job -> volume claim mounted there
    readonly=(),                                # mount paths mounted read-only
    cpu_per_gpu=20, gib_per_gpu=187,            # a Job's CPU and memory per GPU, by default
    image='registry/image:tag',                 # the Job's image, unless given
    storage={'/mnt/team-x': 'sftp://host/data'},  # the cluster's own filesystem, by URL (see 3)
    user='me',                                  # prefixes Job names; $USER by default
    options=dict(venv='/mnt/team-x/me/venv', ray=True),  # job options always given here
)
CLUSTERS = {c.name: c for c in (LAMBDA,)}
```

```python
my_pipeline('preset').submit(LAMBDA).build_tree()
```

**Names**, for where a value cannot be passed -- a `dbx.pprint` one-liner, `dbx.kubejobs`, a new
shell. The project names its targets as an **entry point** of group `dbx.targets` in its packaging:
an object that is a Target, or a dict or list of them.

```toml
[project.entry-points."dbx.targets"]
myproject = "myproject.clusters:CLUSTERS"
```

`dbx.cluster.target('lambda')` -- and so `submit('lambda')`, `dbx.kubejobs`, `dbx.kubedash` --
loads every `dbx.targets` entry point and looks the name up in what they name. It does that on
each call and keeps nothing: what a name means is a function of the installed packages, not of
what this process happened to import. `dbx.cluster.targets()` lists them. (sctx: `sctx/kube.py`,
`LAMBDA` and `CRUSOE`; entry point `sctx = "sctx.kube:CLUSTERS"`.)

**A journaled run carries its target.** `submit:BEGIN`'s record holds the target itself as
`target_spec` -- its class and its fields -- so `dbx.cluster.runs(block)` rebuilds each run's target
from its record, with no lookup: a run submitted to a target defined in a notebook is still
followed from another process, and a later change to a project's `LAMBDA` does not rewrite where an
earlier run went.

The site's facts -- namespace, queues, claims, shares -- live in the project, not in dbx; so does the
choice of image and venv.

## 2. Submitting

`Pipeline.submit(target, *, mode=None, force=False, **options)` hands out the pipeline's entrypoint
as a `Submit` of the target's kind (`target.SUBMIT`: `dbx.kube.KubeSubmit` for a Cluster). A method
called on it -- `.build_tree()`, `.entry.build_tree()`, `.try_specialize_tree(...)` -- runs there:

```
dbx.pprint --capture-output "dbx.eval('<the pipeline's quote>').entry.<method>(<args>)"
```

with the entry's datalake as `DBX_DATALAKE`. The quote rebuilds the same pipeline -- the same hashes,
the same operational kwargs -- so the run builds the very blocks the call would build here. It
returns a `Run` (section 5).

- **Arguments** must be literals (`ast.literal_eval(repr(x)) == x`): they travel as text.
- **Already built:** `build` and `build_tree` of an entrypoint valid here launch nothing -- a Run
  `skipped` -- unless `force=True`.
- **`mode`** ('kube'), when given, must be the target's kind: a check, not a choice.
- **Options** are the kind's, and are refused at `submit(...)` when it does not take them -- a
  misspelling fails before anything is sent. `gpus` defaults to the pipeline's `n_gpus`, or 0.
- **A shell command:** `dbx.kube.submit(command, LAMBDA, **options)`, the same Job without a
  pipeline; journaled only when given `block=`.

### Kube options (`dbx.kube.job`)

| option | |
|---|---|
| `gpus`, `cpus`, `gib` | GPUs; CPU and memory default to the cluster's per-GPU share times `gpus` (8 / 32 GiB without GPUs). Kueue admits on all three: lower `cpus`/`gib` when the queue waits on those. |
| `image` | the Job's image; else `Cluster.image`; else, with `anyscale=True`, `$ANYSCALE_RAY_IMAGE_URI` |
| `venv` | a venv to `source` first -- one built for the Job's image |
| `revision`, `pin_source` | git pins, as `dbx.remote(revision=...)`: `'dbx_rev:project_rev'`, or True for both HEADs (refused when dirty). Cloned in the Job from `pin_source` (default: the repos' own paths -- reachable where the cluster mounts them; give git URLs otherwise) and put first on `PYTHONPATH`. |
| `workdirs` | `{src: target}`: trees the run needs that are not in git (section 3) |
| `env`, `setup`, `cwd` | environment variables; shell lines run first; the working directory |
| `ray`, `ray_bin` | start a single-node Ray head first, for code that calls `ray.init(address='auto')` |
| `anyscale` | False by default. True adds what an Anyscale image needs (`dbx.anyscale.job_options`): its `ray` user (1000:100), its Ray (`/home/ray/venv/bin/ray`), `ANYSCALE_WORKING_DIR`, and its image by default. |
| `run_as`, `priority`, `deadline_hours`, `ttl_days`, `name` | the pod's uid/gid; Kueue priority class; deadline (24 h); how long a finished Job is kept (7 days); a name the Job's is made from |
| `dry_run` | section 4 |

`Cluster.options` are defaults for all of them; `env` and `setup` add to the cluster's.

## 3. Code and data on the cluster

What the run imports and reads must be there, at the paths the expression names:

- **A cluster that mounts the paths seen here** (no `storage`; e.g. a claim that is the workspace's
  own storage) runs the live tree: a venv, editable installs, the datalake are simply there. An edit
  made here while it runs is seen by it -- pin with `revision` when that matters.
- **A cluster with its own filesystem** (`storage={local_prefix: url_prefix}`) has the same paths
  under another root, reached from here by URL: `Cluster.url(path)`. `workdirs={src: target}` are
  synced there before the Job is created -- file by file, skipping those whose size and mtime match,
  keeping modes (console scripts stay executable) and symlinks, deleting nothing (`dbx.kube.sync`).
  A block built there is copied back with `UNSAFE_copy_from(cluster.url(block.anchorkeypath))`:
  its anchorkeypath is the same path on both sides.
- **The code itself:** `revision` pins the dbx and project repos to commits, cloned in the Job. A
  venv, an image -- what is not in git -- are the project's to provide.

## 4. Dry runs

`dry_run` is an option of every kind:

| `dry_run` | what happens | journaled |
|---|---|---|
| `True` | the record and the Job's manifest are made (`Run.record['manifest']`); nothing is contacted | no |
| `'server'` | kube: the API server validates the Job (`kubectl create --dry-run=server`), creating nothing | no |
| `'cluster'` | a small real run on the target (kube: no GPUs, 2 CPUs, 8 GiB, no Ray) that only evaluates the pipeline's quote there and `dbx.cluster.verify`s the entry: every import, the venv, the revision, the workdirs, the datalake, and the same hash as here. It builds nothing; the call it stands for is in its env as `DBX_SUBMIT_CALL`. `run.wait(); run.command().output()` shows the entry's anchorkeypath, whether it is valid there, and which dbx it imported. A hash that differs there is a ValueError: the code or its configuration is not the same. | yes |

## 5. The journal, the run, and its output

A submission is journaled on the entrypoint, in the **submitting** datalake:

| event | note |
|---|---|
| `submit:BEGIN` | before the run is launched; a file holding the whole record as JSON: `executor`, `target` (its name), `target_spec` (the target itself), `id` (the Job), `command`, `expression`, `session` (the submitting one), `run_session`, `datalake`, `submitted`, and the kind's own -- kube: `cluster`, `context`, `namespace`, `image`, `revision`, `workdirs` (what was synced), `manifest` (the Job itself: `kubectl create -f` it to run the same again) |
| `submit:END` | `Run.wait` saw it complete |
| `submit:ERROR` | launching failed, or `Run.wait` saw it fail (with its log tail) |
| `submit:CANCEL` | `Run.cancel` |

`entry.buildjournal(event='^submit')` lists them. `dbx.cluster.runs(block_or_pipeline)` hands each
submission back as its `Run`, oldest first -- from another process, a later session -- closed when
its END, ERROR or CANCEL is journaled.

**The run's own record is in the target's datalake** (the entry's `DBX_DATALAKE` there). It runs
`dbx.pprint --capture-output` under a session chosen at submission (`DBX_JOURNAL_SESSION`,
recorded as `run_session`; the submitting one is passed as `DBX_PARENT_SESSION`). So under that one
session, there:

- its exec-journal row: `Run.command()`, which is `dbx.execjournal(session='^<run_session>$', iloc=0)`
  in the record's datalake (or the `datalake=` given);
- every build-journal entry it wrote, each with its `log` -- the capture it was written under;
- its captured stdout and stderr, which the row lists as `output_captures`: the master's (the
  command's process) and a worker's per worker process, each written when its process ends.

The journal is the one record of the output, and the one way to it: `ExecJournalEntry.output()`.

```python
run.command().output()                                         # a cluster that mounts the datalake
run.command(datalake=LAMBDA.url(run.record['datalake'])).output()   # by URL
```

```bash
# on the cluster's own machines, with DBX_DATALAKE set to its datalake
dbx.pprint "dbx.execjournal(session='^<run_session>$', iloc=0).output()"
```

The row is written when the run ends. Until then -- while a Job runs -- `Run.logs()` is its pod's
log, through kubectl, until the Job is deleted (`ttl_days`). Over an sftp:// URL the journal readers
work serially: one SFTP connection cannot serve several threads at once.

## 6. Watching

| | |
|---|---|
| `run.status()` | 'pending' (not admitted, or no pod yet), 'running', 'complete', 'failed'; 'skipped', 'dry-run' |
| `run.reason()` | kube: why a pending Job waits -- Kueue's message on its Workload, e.g. a quota |
| `run.logs(tail=-1, follow=False)` | the pod's log; `follow=True` streams it until the pod ends |
| `run.wait(poll=30, timeout=None)` | until complete (True) or *timeout* (False); raises with the log tail on failure; journals END / ERROR |
| `run.cancel()` | deletes the Job and its pod; journals CANCEL |
| `run.command()` | the run's exec-journal row; `.output()` prints what it printed (section 5) |
| `dbx.cluster.runs(block)` | the Runs of every submission of a block, from its journal |
| `dbx.kubejobs [CLUSTER ...] [-s STATUS] [-u USER \| -A]` | my Jobs (`-A`: everyone's) on the `dbx.targets` entry points' clusters (or those named), newest first: status, GPUs, times, the expression, and Kueue's reason for a pending one. `dbx.kube.jobs(...)` returns the DataFrame. |
| `dbx.kubedash [CONTEXT ...] [-r RESOURCE] [-q REGEX] [-a]` | each cluster's Kueue queues: quota, used, free, borrowed, pending, admitted; `-r cpu` / `-r memory` (GiB). Contexts default to the `dbx.targets` entry points' clusters', else kubectl's, one per cluster. `dbx.kube.kubedash(...)` returns the DataFrame. |

kubectl itself, for what these do not cover: `kubectl --context C -n NS describe job/<id>`,
`get workload`, `describe workload <name>`, `get events --sort-by=.lastTimestamp`.

## 7. Adding a kind

A kind subclasses three classes of `dbx.cluster`:

- `Target`, with `MODE` (its name, e.g. 'slurm'), `SUBMIT` (its Submit class) and a `name`;
- `Submit`, implementing `_prepare_(command, **options)` (the record, its `id` among it),
  `_create_(record)` (launch it), and as needed `_check_(options)` (refuse options it does not take),
  `_stage_(record, **options)` (get files in place), `_dry_run_(record, dry_run)` and
  `_check_options_()` (the least a `dry_run='cluster'` run takes); `RUN` is its Run class;
- `Run`, implementing `_status_()`, `_logs_(tail)`, `_cancel_()`.

The base classes do the rest -- the shipped expression, the literal check, the skip, the sessions,
the journal, `wait`, `runs`, `command` -- the same for every kind.

## 8. Not done

- Slurm as a kind (`SlurmCluster`, `SlurmSubmit`, `SlurmRun`), reusing `SlurmRayCluster`.
- Resources declared by the blocks (`EXECUTORS.md` section 3): today `gpus` comes from `n_gpus`.
- Fan-out: one run per tree.
- Code that is not in git, shipped as an image layer (sctx's `image.py` prototype).
