# Executors (proposal)

**Status: superseded for Kubernetes by `CLUSTERS.md`** (2026-11-03), which is what was built: not
an `executor` operational kwarg, but `Pipeline.submit(target, **options)`, handing out the
entrypoint as a `Submit` of the target's kind; a method called on it runs there and returns a `Run`
(`dbx/cluster.py`, `dbx/kube.py`). Sections 1, 2, 4 and 5 are done that way, section 6 by
`Cluster.url`; Slurm, the per-block resources (section 3) and fan-out are not. The proposal is kept
as written.

## The problem

`entry.build_tree()` runs in the calling process. Building somewhere else (a Slurm allocation, a
Kubernetes Job) today means either `dbx.remote(slurm=...)`, a live interpreter on a Ray cluster that
this process connects to, or project code that writes a job script by hand. The second kind ties a
pipeline call to a file (a job YAML naming the preset and its ops), which has to be edited, and
committed, whenever the call changes.

Not every backend can offer the first kind. A Kueue Job on a cluster reached only through its
Kubernetes API (no network path from the submitting host to the pod) cannot be connected to; it can
only be submitted, waited on and read back. That is the common ground of every backend.

## What dbx already has

- **The quote contract:** `dbx.eval(x.quote()).hash == x.hash`, wherever it is evaluated, with the
  operational kwargs included (`Datablock.quote`). A pipeline is reproduced exactly by its own quote,
  and that is what the `PipeJournal` records. *Checked for `sc_representation('nuscenes_08-14-2026_stage3_val', n_gpus=8,
  n_viz_workers=16)`: the pipeline hash, the entry hash, the ops and the re-quote all survive
  `dbx.eval(pipeline.quote())`.*
- **Operational kwargs:** a block's undeclared kwargs stay out of its hash, are shown in its
  `quote()`, and reach `__build__` as `getattr(self, name, default)`. The pipeline hands them on with
  `**self.operational`. `n_gpus` is one.
- **Journals:** the exec journal records each `dbx.exec` / `dbx.pprint` command, under a session that
  every build-journal entry written during it shares (JOURNALS.md).
- **Slurm:** `databits.SlurmRayCluster` (an sbatch script that brings up Ray), `dbx.remote(slurm=...)`
  and the `dbx.slurm.*` commands.

## Proposal

### 1. `executor` as an operational kwarg

```python
sc_representation('nuscenes_08-14-2026_stage3_val', n_gpus=8, executor=dbx.kube('lambda')).entry.build_tree()
```

`executor` is an op like `n_gpus`: it says where the tree is built, never what it is, so it stays
out of every hash. When a block whose `executor` is set is asked to build, it does not build here.
It hands the executor the expression that builds it elsewhere, and returns a **run handle**:

```
dbx.eval('<block.quote() without executor>').build_tree()
```

Dropping `executor` from the shipped quote is what stops the remote side from submitting again. The
alternative, a marker in the remote environment (`DBX_EXECUTOR_RUN=1`) that makes `executor` a no-op,
keeps the quote untouched, at the cost of a hidden switch. That choice is open (below).

### 2. One executor interface: submit, wait, read

```python
class Executor:
    def submit(self, expression: str, resources: Resources, *, name: str) -> Run: ...

class Run:
    id: str                   # e.g. 'job.batch/<name>' or a Slurm job id
    def status(self) -> str   # pending | running | complete | failed
    def wait(self, timeout=None) -> bool   # True when complete; raises with the log tail on failure
    def logs(self, tail=-1) -> str
    def cancel(self) -> None
```

- **`LocalExecutor`**: a subprocess running `dbx.pprint`. For tests, and the reference semantics.
- **`SlurmExecutor`**: an sbatch script running `dbx.pprint "<expression>"`, reusing what
  `SlurmRayCluster` already does to bring up Ray inside the allocation.
- **`KubeExecutor`**: a `batch/v1` Job created through `kubectl --context <ctx> create -f -`. Its
  configuration is the project's, not dbx's: context, namespace, Kueue queue and priority, image,
  volume claims and where they mount, the CPU and memory per GPU, and the environment (`DBX_DATALAKE`
  among it). The prototype's table of these per cluster is `sctx.kube.CLUSTERS`.

`dbx.remote` stays as it is: a live interpreter is an extra that a backend with a reachable Ray
address (Slurm, a cluster on the same network) can offer, and the executors do not need.

### 3. Resources declared by the block

A block knows what its build needs from its own ops. A hook returns it:

```python
def __resources__(self) -> dict:      # SCRepresentation
    n = getattr(self, 'n_gpus', 2)
    return dict(gpus=n, cpus=n * getattr(self, 'cpus_per_gpu', 4) + getattr(self, 'n_viz_workers', 2) + 4)
```

The default is no GPU and a few CPUs. For a tree, the run needs the maximum over the blocks built in
it, since `build_tree` builds them one after another in one process. The executor turns these into
its own terms: Kueue requests and limits, `--gres` and `--cpus-per-task`. Explicit
`executor=dbx.kube('lambda', cpus=128)` arguments override them. *Prototype finding: Kueue admits on
every resource, not GPUs alone; an 8-GPU Job waited on the team's CPU quota until its CPU request
was lowered.*

### 4. Recording the submission

`submit` writes an exec-journal row in the submitting datalake: the executor kind and target
(cluster, context), the run id, the image or revision, and the shipped expression. The remote side's
own `dbx.pprint` writes its row and its build journal where it runs, in the datalake its environment
names. Passing the submitting session to the remote (`DBX_PARENT_SESSION`) links the two.

### 5. Code identity

What the remote side runs is part of what a result depends on:
- **Kube:** the image, best a content-addressed one: the base image plus one layer of the project's
  code, tagged by the layer's hash (the prototype is `sctx/image.py`, which uses `crane append`).
- **Slurm:** the `revision` pinning `dbx.remote` already does.

The executor records it with the run, and the remote's build journal could record it too.

### 6. Reading results back

Where the remote datalake is the submitting one (Slurm on shared storage, Kube on a cluster whose
volume claim is the workspace's storage), `entry.read()` works as soon as the run completes. Where it
is not (a cluster with its own filesystem), the executor could name a `datalake` URL reachable from
here, e.g. the cluster filesystem's S3 endpoint through fsspec `storage_options`, so the same
`entry.read()` resolves there. Not tried.

## Open questions

- **How the remote side avoids resubmitting:** drop `executor` from the shipped quote, or a
  remote-environment marker. The first keeps the shipped text honest about what ran, but makes
  `quote()` aware of one op.
- **`dbx.pprint` and a leading `$`:** the CLI rejects a bare specline (`dbx.pprint "$X(...).build_tree()"`
  is a SyntaxError), so the prototype wraps it as `dbx.eval('<quote>')`. Accepting a leading `$` in
  `dbx.exec` would make the shipped command the quote itself.
- **`__repr__`:** a block's `repr()` today shows spec values quoted twice (`'split': "'val'"`), and
  `eval(repr(x))` builds a different block (a different hash). Making `__repr__` return `quote()`
  would make what is printed the reusable form. The `quote(specline=False)` variant is plain-Python
  evaluable only for flat specs: nested datablocks are quoted as `"$X(...)"` strings, which only
  `dbx.eval` turns back into blocks.
- **Granularity:** one run per tree is what the prototype does. A `Datastack` could fan out one run
  per child block; that needs the per-block resources above, and the journal to tie the runs together.
- **Multi-node:** out of scope until a pipeline needs more GPUs than one node has.

## Prototype notes (sctx, Lambda, 2026-11-01)

- `kube.submit(pipeline, cluster='lambda')` ships `dbx.eval(pipeline.quote()).entry.build_tree()` and
  sizes the Job from `n_gpus`. `kube.wait` and `kube.logs` complete it.
- It runs the workspace's image with the project's code on the cluster's team volume, mounted at the
  workspace's path so every absolute path resolves unchanged. On a cluster with its own filesystem
  the code is uploaded first, and an upload that drops execute bits breaks the venv's console
  scripts: an image with the code baked in removes that step.
- The smoke preset built end to end, and a block's hash came out the same on both clusters, so a
  block built on one is the same block on the other.
