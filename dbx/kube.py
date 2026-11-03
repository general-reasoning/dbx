"""A shell command, or a pipeline's build, run as a Kueue Job on a Kubernetes cluster.

A project registers its clusters -- what kubectl calls them, their namespace, Kueue queues, volume
claims and where they mount, and the job options it always wants there -- and submits by name:

    dbx.kube.register(dbx.kube.Cluster('lambda', namespace='kueue-team-x', queue='x-queue',
                                       mounts={'/mnt/team-x': 'default-pvc-team-x'}, image=...))
    run = dbx.kube.submit('python -m my.thing', 'lambda', gpus=8)
    run.wait()                      # True when the Job completes; raises with its log tail if it fails

A pipeline submits its own calls (`Pipeline.submit`, `dbx.submits`), as a `KubeSubmit` on a Cluster:

    my_pipeline('preset').submit('lambda').build_tree()      # a KubeRun

Registering puts a Cluster among `dbx.submits`' targets; the module that does it is best named as a
`dbx.targets` entry point of its package, so a target is found by name with nothing imported first.

The Job is generated here, so a different command is a different call, never a file to edit. It
runs *command* with bash, after:

- *setup*: shell lines of the caller's;
- *venv*: `source <venv>/bin/activate` -- a venv built for the Job's image;
- *revision*: `'dbx_rev:project_rev'` (or True: both repos' HEADs) cloned from *pin_source* (default
  the repos' own paths, which a cluster mounting them reaches; give git URLs otherwise) and put
  first on `PYTHONPATH`, as `dbx.remote(revision=...)` pins a Ray worker -- so the code it ran is
  the code its record names;
- *ray*: a single-node Ray head, for code that farms out tasks with `ray.init(address='auto')`.

Where the cluster gets code that is not in git: *workdirs* `{src: target}`. A cluster with no
`storage` mounts the paths this process sees, so a workdir is already there (src == target); one
with `storage` -- its own filesystem, reached by URL -- is synced to first, file by file, skipping
those whose size and mtime match, keeping modes (console scripts stay executable).

`anyscale=True` adds what an Anyscale image needs (`dbx.anyscale.job_options`): its image, its
`ray` user, its Ray, its Ray working directory. Off Anyscale, give *image* (or `Cluster.image`).

A pipeline's submission is journaled on its entrypoint, in the submitting datalake (`dbx.submits`):
`submit:BEGIN`, its note the whole record -- cluster, Job, expression, image, revision, workdirs,
session, and the Job's manifest itself; `submit:END` / `submit:ERROR` when `Run.wait` sees it
finish; `submit:CANCEL`.

Watching: `Run.status()`, `Run.logs(follow=True)`, `KubeRun.reason()` (why Kueue has not
admitted it), `Run.wait()`, `Run.cancel()` -- and, from another process, `dbx.submits.runs(block)`
for the Runs journaled on a block. `jobs` (console `dbx.kubejobs`) is the Jobs of one or more
clusters as a table, a pending one with Kueue's reason; `kubedash` (console `dbx.kubedash`) is
their Kueue queues.
"""
import argparse
import dataclasses
import datetime
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import uuid

import pandas as pd

from . import submits
from .submits import Run, Submit, Target


__all__ = ['Cluster', 'KubeRun', 'KubeSubmit', 'register', 'cluster', 'clusters', 'job', 'submit', 'sync', 'jobs',
           'kubedash']


@dataclasses.dataclass
class Cluster(Target):
    """A Kubernetes cluster with Kueue, as a project submits to it.

    *mounts* maps a path in the Job to the volume claim mounted there, *readonly* names the ones
    mounted read-only. *storage* maps a path prefix here to the URL of the same files on the
    cluster's own filesystem -- empty when the cluster mounts what this process sees. *options* are
    `job` options always given on this cluster (`anyscale`, `venv`, `ray`, ...).
    """
    name: str
    context: str | None = None                  # kubectl's name for it; *name* by default
    namespace: str = 'default'
    queue: str | None = None                    # Kueue LocalQueue of Jobs with GPUs
    cpu_queue: str | None = None                # ... and without; *queue* by default
    priority: str | None = None                 # Kueue WorkloadPriorityClass
    mounts: dict = dataclasses.field(default_factory=dict)
    readonly: tuple = ()
    cpu_per_gpu: int = 8
    gib_per_gpu: int = 64
    image: str | None = None
    storage: dict = dataclasses.field(default_factory=dict)
    user: str | None = None                     # prefixes Job names; $USER by default
    options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.context = self.context or self.name

    def url(self, path: str) -> str:
        """Where *path* is on the cluster: its URL under `storage`, or *path* itself when the cluster mounts it."""
        if not self.storage:
            return path
        path = os.path.abspath(path)
        for local, remote in self.storage.items():
            rel = os.path.relpath(path, local)
            if not rel.startswith('..'):
                return f"{remote.rstrip('/')}/{rel}" if rel != '.' else remote
        raise ValueError(f"{path} is under none of cluster {self.name!r}'s storage: {sorted(self.storage)}")


def register(c: Cluster) -> Cluster:
    """Make *c* a target by its name (`dbx.submits.register`)."""
    return submits.register(c)


def cluster(c: 'str | Cluster') -> Cluster:
    """The registered cluster named *c*; *c* itself when a Cluster."""
    return submits.target(c, 'kube')


def clusters() -> dict:
    """The registered clusters, by name."""
    return submits.targets('kube')


# --------------------------------------------------------------------------------------------- #
# The Job
# --------------------------------------------------------------------------------------------- #

def _slug_(s: str, n: int = 40) -> str:
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')[:n].strip('-')


def _pins_(revision, pin_source) -> list:
    """`[(source, sha), ...]` to clone, dbx first, from *revision* -- `'dbx_rev:project_rev'`, `(dbx_rev, project_rev)`, or True for both HEADs."""
    from .databits import dbx_repos, dbx_revisions
    repos = dbx_repos()
    sources = pin_source.split() if isinstance(pin_source, str) else list(pin_source or repos)
    if revision is True:
        revs = []
        for repo in repos:
            if repo is None:
                revs.append(None)
                continue
            dirty = subprocess.run(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'],
                                   capture_output=True, text=True, check=True).stdout.strip()
            if dirty:
                raise ValueError(f"revision=True pins {repo}'s HEAD, which is not what is checked out there: "
                                 f"commit first, or pass the revision")
            revs.append('HEAD')
    else:
        revs = list(dbx_revisions(revision))
    pins = []
    for source, repo, rev in zip(sources, repos, revs):
        if rev is None:
            continue
        if source is None:
            raise ValueError(f"revision {revision!r} pins a repo with no source: set DBX_GIT_REPO or pass pin_source")
        sha = rev
        if repo is not None:  # Resolved where it can be, so the record names a commit, not a branch.
            sha = subprocess.run(['git', '-C', repo, 'rev-parse', rev], capture_output=True, text=True,
                                 check=True).stdout.strip()
        pins.append((source, sha))
    return pins


def job(command: str, c: 'str | Cluster', **options) -> dict:
    """The Job running *command* on cluster *c*: a `batch/v1` manifest, its name set.

    *options* (see the module doc and `_manifest_`) are over the cluster's own `options`; `env`
    and `setup` add to the cluster's.
    """
    c = cluster(c)
    merged = {**c.options, **options}
    for k in ('env', 'setup'):
        if k in c.options and k in options:
            merged[k] = ({**c.options[k], **options[k]} if k == 'env' else [*c.options[k], *options[k]])
    return _manifest_(command, c, **merged)


def _manifest_(command: str, c: Cluster, *, name: str = 'dbx', gpus: int = 0, cpus: int | None = None,
        gib: int | None = None, image: str | None = None, venv: str | None = None, cwd: str | None = None,
        revision=None, pin_source=None, env: dict | None = None, setup: list | None = None,
        anyscale: bool = False, ray: bool = False, ray_bin: str = 'ray', run_as: tuple | None = None,
        priority: str | None = None, deadline_hours: float = 24, ttl_days: float = 7,
        workdirs: dict | None = None) -> dict:
    """`job`'s manifest, from all its options.

    *cpus* and *gib* (memory) default to the cluster's per-GPU share times *gpus*; lower them when the
    queue's CPU or memory quota, not its GPUs, is what Kueue is waiting on. *workdirs* are recorded
    on the Job; `submit` syncs them.
    """
    env, setup = dict(env or {}), list(setup or [])
    if anyscale:
        from . import anyscale as _anyscale
        a = _anyscale.job_options()
        image = image or a['image']
        run_as = run_as or a['run_as']
        ray_bin = a['ray_bin'] if ray_bin == 'ray' else ray_bin
        env = {**a['env'], **env}
        setup = a['setup'] + setup
    image = image or c.image
    if image is None:
        raise ValueError(f"No image for the Job on {c.name!r}: pass image=, set Cluster.image, or anyscale=True "
                         f"in an Anyscale workspace")
    cpu = cpus or (c.cpu_per_gpu * gpus if gpus else 8)
    mem = gib or (c.gib_per_gpu * gpus if gpus else 32)
    user = _slug_(c.user or os.environ.get('USER') or 'dbx', 16)
    jobname = f"{user}-{_slug_(name, 40)}-{uuid.uuid4().hex[:5]}"
    pins = _pins_(revision, pin_source) if revision is not None else []

    script = ['set -euo pipefail', *setup]
    if venv:
        script.append(f"source {shlex.quote(os.path.join(venv, 'bin', 'activate'))}")
    for source, sha in reversed(pins):  # dbx first on PYTHONPATH: prepended last.
        d = f"/tmp/dbx-pins/{_slug_(os.path.basename(source.rstrip('/')).removesuffix('.git'), 40)}-{sha[:12]}"
        script += [f"git clone --quiet {shlex.quote(source)} {d}", f"git -C {d} checkout --quiet {sha}",
                   f'export PYTHONPATH={d}${{PYTHONPATH:+:$PYTHONPATH}}']
    if ray:
        script.append(f"{ray_bin} start --head --num-gpus={gpus} --num-cpus={cpu} --disable-usage-stats")
    script.append('exec bash -c "$DBX_KUBE_COMMAND"')

    resources = {'cpu': str(cpu), 'memory': f"{mem}Gi", 'ephemeral-storage': '100Gi', 'nvidia.com/gpu': str(gpus)}
    queue = c.queue if gpus else (c.cpu_queue or c.queue)
    labels = {'dbx/user': user}
    if queue:
        labels['kueue.x-k8s.io/queue-name'] = queue
    if priority or c.priority:
        labels['kueue.x-k8s.io/priority-class'] = priority or c.priority
    annotations = {'dbx/command': command, 'dbx/image': image}
    if pins:
        annotations['dbx/revision'] = ' '.join(f"{s}@{sha}" for s, sha in pins)
    if workdirs:
        annotations['dbx/workdirs'] = json.dumps(workdirs)
    mounts = [{'name': f"v{i}", 'mountPath': path, **({'readOnly': True} if path in c.readonly else {})}
              for i, path in enumerate(c.mounts)]
    volumes = [{'name': f"v{i}", 'persistentVolumeClaim': {'claimName': claim,
                                                           **({'readOnly': True} if path in c.readonly else {})}}
               for i, (path, claim) in enumerate(c.mounts.items())]
    mounts.append({'name': 'dshm', 'mountPath': '/dev/shm'})
    volumes.append({'name': 'dshm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': f"{max(1, min(64, mem // 4))}Gi"}})
    container = {
        'name': 'dbx', 'image': image, 'command': ['bash', '-c', '\n'.join(script)],
        'env': [{'name': 'DBX_KUBE_COMMAND', 'value': command}]
               + [{'name': k, 'value': str(v)} for k, v in env.items()],
        'resources': {'requests': resources, 'limits': resources},
        'volumeMounts': mounts,
    }
    if cwd:
        container['workingDir'] = cwd
    pod = {'restartPolicy': 'Never', 'containers': [container], 'volumes': volumes}
    if run_as:
        pod['securityContext'] = {'runAsUser': run_as[0], 'runAsGroup': run_as[1]}
    return {
        'apiVersion': 'batch/v1', 'kind': 'Job',
        'metadata': {'name': jobname, 'namespace': c.namespace, 'labels': labels, 'annotations': annotations},
        'spec': {
            # Created suspended: Kueue unsuspends it when its queue admits it.
            'suspend': bool(queue), 'backoffLimit': 0, 'activeDeadlineSeconds': int(deadline_hours * 3600),
            'ttlSecondsAfterFinished': int(ttl_days * 86400),
            'template': {'spec': pod},
        },
    }


def _kubectl_(context: str, *args, input: str | None = None) -> str:
    """kubectl's output for *args* in *context*; raises with its stderr when it fails."""
    p = subprocess.run(['kubectl', '--context', context, *args], input=input, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"kubectl --context {context} {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


# --------------------------------------------------------------------------------------------- #
# Workdirs
# --------------------------------------------------------------------------------------------- #

def _mtime_(info) -> float | None:
    t = info.get('mtime', info.get('LastModified', info.get('modified')))
    if isinstance(t, datetime.datetime):
        return t.timestamp()
    return float(t) if t is not None else None


def sync(src: str, url: str, *, storage_options: dict | None = None) -> dict:
    """Copy the tree *src* to *url*, skipping files whose size and mtime (to the second) match; `{'files', 'bytes', 'skipped'}`.

    Modes and mtimes are kept where the filesystem lets them be set (local, sftp), and symlinks are
    made as symlinks there. Nothing at *url* is deleted.
    """
    import fsspec
    import fsspec.implementations.local
    fs, root = fsspec.core.url_to_fs(url, **(storage_options or {}))
    try:
        there = {p: i for p, i in fs.find(root, detail=True, withdirs=False).items()}
    except FileNotFoundError:
        there = {}
    sftp = getattr(fs, 'ftp', None)
    local = isinstance(fs, fsspec.implementations.local.LocalFileSystem)
    stats = dict(files=0, bytes=0, skipped=0)
    made = set()
    for dirpath, dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        rdir = root if rel == '.' else f"{root.rstrip('/')}/{rel}"
        for n in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            path, rpath = os.path.join(dirpath, n), f"{rdir.rstrip('/')}/{n}"
            st = os.lstat(path)
            if os.path.islink(path):
                if rpath not in there and (sftp or local):
                    if rdir not in made:
                        fs.makedirs(rdir, exist_ok=True)
                        made.add(rdir)
                    try:
                        (sftp.symlink if sftp else os.symlink)(os.readlink(path), rpath)
                    except OSError:
                        pass  # Already there, as a link the listing does not show.
                continue
            info = there.get(rpath) or there.get(rpath.lstrip('/'))
            if info and info.get('size') == st.st_size and _mtime_(info) is not None \
                    and int(_mtime_(info)) == int(st.st_mtime):
                stats['skipped'] += 1
                continue
            if rdir not in made:
                fs.makedirs(rdir, exist_ok=True)
                made.add(rdir)
            fs.put_file(path, rpath)
            if sftp:
                sftp.chmod(rpath, st.st_mode & 0o7777)
                sftp.utime(rpath, (st.st_atime, st.st_mtime))
            elif local:
                os.chmod(rpath, st.st_mode & 0o7777)
                os.utime(rpath, (st.st_atime, st.st_mtime))
            stats['files'] += 1
            stats['bytes'] += st.st_size
    return stats


def _sync_workdirs_(c: Cluster, workdirs: dict) -> dict:
    """Make each workdir `{src: target}` the cluster's: synced where it has `storage`, checked where it mounts ours."""
    done = {}
    for src, target in workdirs.items():
        src = os.path.abspath(src)
        if not c.storage:
            if os.path.abspath(target) != src:
                raise ValueError(f"Cluster {c.name!r} mounts the paths this process sees, so workdir {src} is "
                                 f"there already, at {src}, not {target}")
            done[src] = dict(target=target, files=0, bytes=0, skipped=None)
            continue
        done[src] = dict(target=target, **sync(src, c.url(target)))
    return done


# --------------------------------------------------------------------------------------------- #
# Submitting, and the run
# --------------------------------------------------------------------------------------------- #

class KubeRun(Run):
    """A submitted Job; `id` is its name."""

    @property
    def job(self) -> str | None:
        return self.id

    def _kubectl_(self, *args) -> str:
        return _kubectl_(self.target.context, '-n', self.target.namespace, *args)

    def _status_(self) -> str:
        return _job_status_(json.loads(self._kubectl_('get', f"job/{self.id}", '-o', 'json')))

    def _logs_(self, tail: int) -> str:
        return self._kubectl_('logs', f"job/{self.id}", f"--tail={tail}")

    def logs(self, tail: int = -1, *, follow: bool = False) -> str:
        """The Job's pod log: the last *tail* lines, all when -1; with *follow*, printed as it grows, until the pod ends."""
        if not follow or self.id is None:
            return super().logs(tail)
        subprocess.run(['kubectl', '--context', self.target.context, '-n', self.target.namespace, 'logs', '-f',
                        f"job/{self.id}", f"--tail={tail}"])
        return ''

    def reason(self) -> str | None:
        """Why the Job is not running yet: Kueue's message on its Workload (e.g. a quota it waits on); None once admitted."""
        if self.id is None:
            return None
        return _workload_reasons_(self.target).get(self.id)

    def _cancel_(self) -> None:
        self._kubectl_('delete', f"job/{self.id}", '--cascade=background')

    def _where_(self) -> str:
        return f"kube:{self.target.name} job/{self.id}"


class KubeSubmit(Submit):
    """A Submit whose runs are Kueue Jobs on a `Cluster`; its options are `job`'s, and `dry_run`.

    `dry_run=True` contacts nothing; `dry_run='server'` has the API server validate the Job and
    create nothing. *workdirs* are synced in `_stage_`, after a dry run would have stopped.
    """
    RUN = KubeRun

    def _check_(self, options: dict) -> None:
        takes = set(inspect.signature(_manifest_).parameters) - {'command', 'c'} | {'dry_run'}
        bad = sorted(set(options) - takes)
        if bad:
            raise TypeError(f"{type(self).__name__}: no option {', '.join(bad)}; it takes {', '.join(sorted(takes))}")

    def _prepare_(self, command: str, **options) -> dict:
        c = self._target_
        manifest = job(command, c, **options)
        container = manifest['spec']['template']['spec']['containers'][0]
        return dict(id=manifest['metadata']['name'], job=manifest['metadata']['name'], cluster=c.name,
                    context=c.context, namespace=c.namespace, image=container['image'],
                    revision=manifest['metadata']['annotations'].get('dbx/revision'),
                    datalake={e['name']: e['value'] for e in container['env']}.get('DBX_DATALAKE'),
                    manifest=manifest)

    def _dry_run_(self, record: dict, dry_run) -> None:
        if dry_run == 'server':
            _kubectl_(self._target_.context, 'create', '-f', '-', '--dry-run=server', '-o', 'name',
                      input=json.dumps(record['manifest']))

    def _stage_(self, record: dict, **options) -> dict:
        workdirs = {**self._target_.options.get('workdirs', {}), **(options.get('workdirs') or {})}
        return dict(workdirs=_sync_workdirs_(self._target_, workdirs) if workdirs else {})

    def _create_(self, record: dict) -> None:
        _kubectl_(self._target_.context, 'create', '-f', '-', '-o', 'name', input=json.dumps(record['manifest']))

    def _check_options_(self) -> dict:
        return dict(gpus=0, cpus=2, gib=8, ray=False)


Cluster.MODE, Cluster.SUBMIT = 'kube', KubeSubmit


def submit(command: str, c: 'str | Cluster', *, block=None, expression: str | None = None, **options) -> KubeRun:
    """Create the Job running *command* on cluster *c* (`job` for *options*, and `dry_run`); its `KubeRun`.

    *workdirs* are synced first. With *block*, the submission is journaled on it (`dbx.submits`):
    `submit:BEGIN` before the Job is created, its note the whole record, the manifest among it, and
    `submit:ERROR` if creating it fails. *expression* is recorded as what *command* evaluates.
    *dry_run*: True submits nothing and contacts nothing; `'server'` has the API server validate the
    Job, and creates nothing. Neither syncs or journals.
    """
    return KubeSubmit(None, c, **options)._launch_(command, block=block, expression=expression)


# --------------------------------------------------------------------------------------------- #
# kubedash
# --------------------------------------------------------------------------------------------- #

_BYTES_ = {'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'Ki': 2 ** 10, 'Mi': 2 ** 20, 'Gi': 2 ** 30,
           'Ti': 2 ** 40, 'Pi': 2 ** 50}


def _workload_reasons_(c: Cluster) -> dict:
    """`{job name: Kueue's message}` for each Job of the cluster's namespace whose Workload is not admitted."""
    items = json.loads(_kubectl_(c.context, '-n', c.namespace, 'get', 'workload', '-o', 'json'))['items']
    out = {}
    for w in items:
        owners = [o['name'] for o in w['metadata'].get('ownerReferences', []) if o.get('kind') == 'Job']
        conds = {x['type']: x for x in w.get('status', {}).get('conditions', [])}
        if owners and conds.get('Admitted', {}).get('status') != 'True':
            q = conds.get('QuotaReserved') or conds.get('Admitted') or {}
            out[owners[0]] = q.get('message') or q.get('reason') or 'not admitted'
    return out


def _job_status_(j: dict) -> str:
    done = {c['type'] for c in j.get('status', {}).get('conditions', []) if c['status'] == 'True'}
    if 'Complete' in done:
        return 'complete'
    if 'Failed' in done:
        return 'failed'
    if j.get('spec', {}).get('suspend') or not j.get('status', {}).get('active'):
        return 'pending'
    return 'running'


def jobs(clusters_: list | None = None, *, user: str | None = None, all_users: bool = False,
         status: str | None = None) -> pd.DataFrame:
    """The Jobs in each cluster's namespace -- of *clusters_* (registered names; default all) -- as a table, newest first.

    One row per Job: `cluster, job, status, gpus, created, started, ended, command`, and `reason`,
    Kueue's message for a pending one. Only *user*'s (default each cluster's `user`, else $USER),
    unless *all_users*; only those of *status* ('pending', 'running', 'complete', 'failed') when given.
    """
    cs = [cluster(n) for n in clusters_] if clusters_ else list(clusters().values())
    rows = []
    for c in cs:
        who = _slug_(user or c.user or os.environ.get('USER') or 'dbx', 16)
        try:
            items = json.loads(_kubectl_(c.context, '-n', c.namespace, 'get', 'job', '-o', 'json'))['items']
            reasons = _workload_reasons_(c)
        except Exception as e:
            rows.append(dict(cluster=c.name, error=str(e).splitlines()[0][:200]))
            continue
        for j in items:
            md, st = j['metadata'], j.get('status', {})
            if not all_users and not (md.get('labels', {}).get('dbx/user') == who or md['name'].startswith(f"{who}-")):
                continue
            s = _job_status_(j)
            if status and s != status:
                continue
            cont = j['spec']['template']['spec']['containers'][0]
            command = md.get('annotations', {}).get('dbx/command') or md.get('annotations', {}).get('sctx/command') or ''
            expression = {e.get('name'): e.get('value') for e in cont.get('env', [])}.get('DBX_SUBMIT_EXPRESSION')
            rows.append(dict(cluster=c.name, job=md['name'], status=s,
                             gpus=cont.get('resources', {}).get('requests', {}).get('nvidia.com/gpu', '0'),
                             created=md.get('creationTimestamp'), started=st.get('startTime'),
                             ended=st.get('completionTime'), command=(expression or command)[:120],
                             reason=reasons.get(md['name']) if s == 'pending' else None))
    cols = ['cluster', 'job', 'status', 'gpus', 'created', 'started', 'ended', 'reason', 'command']
    df = pd.DataFrame(rows, columns=cols + (['error'] if any('error' in r for r in rows) else []))
    return df.sort_values('created', ascending=False, na_position='first').reset_index(drop=True)


def jobs_main(argv: list | None = None) -> None:
    """`dbx.kubejobs [CLUSTER ...] [-s STATUS] [-u USER | -A]`: print `jobs`' table (`jobs` returns it)."""
    p = argparse.ArgumentParser(prog='dbx.kubejobs', description=jobs.__doc__.split('\n')[0])
    p.add_argument('clusters', nargs='*', help="registered clusters to list (default: all)")
    p.add_argument('-s', '--status', choices=['pending', 'running', 'complete', 'failed'])
    p.add_argument('-u', '--user', help="whose Jobs (default: the cluster's user, else $USER)")
    p.add_argument('-A', '--all-users', action='store_true', help="every user's Jobs in the namespace")
    a = p.parse_args(argv)
    df = jobs(a.clusters, user=a.user, all_users=a.all_users, status=a.status)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 250,
                           'display.max_colwidth', 120):
        print(df.to_string(index=False))


def _quantity_(q, resource: str = '') -> float:
    """A Kubernetes quantity as a number: '24' -> 24, '500m' -> 0.5; a size ('187Gi', '200G') in GiB."""
    if q is None:
        return 0.0
    m = re.fullmatch(r'([0-9.eE+-]+)([A-Za-z]*)', str(q))
    if not m:
        return float('nan')
    n, unit = float(m.group(1)), m.group(2)
    if unit in _BYTES_:
        return n * _BYTES_[unit] / 2 ** 30
    if unit == 'm':
        return n / 1000
    return n / 2 ** 30 if resource in ('memory', 'ephemeral-storage') else n


def _contexts_() -> list:
    """kubectl's contexts, one per cluster -- the first by name of those naming the same one."""
    cfg = json.loads(subprocess.run(['kubectl', 'config', 'view', '-o', 'json'], capture_output=True, text=True,
                                    check=True).stdout)
    first = {}
    for ctx in sorted(cfg.get('contexts') or [], key=lambda x: x['name']):
        first.setdefault(ctx['context'].get('cluster'), ctx['name'])
    return sorted(first.values())


def kubedash(contexts: list | None = None, *, resource: str = 'nvidia.com/gpu', queues: str | None = None,
             all: bool = False) -> pd.DataFrame:
    """The Kueue ClusterQueues of each kubectl context in *contexts*: their *resource* quota, use and workloads.

    One row per queue: `context, queue, cohort, quota, used, free, borrowed, pending, admitted`, the
    quantities *resource*'s summed over the queue's flavors (memory in GiB). *contexts* default to
    the registered clusters', else kubectl's, one per cluster. *queues* is a regex the queue name must
    match; queues with no quota, no use and no workloads are left out unless *all*. A context that
    cannot be reached is a row with its error.
    """
    if not contexts:
        contexts = sorted({c.context for c in clusters().values()}) or _contexts_()
    rows = []
    for ctx in contexts:
        try:
            items = json.loads(_kubectl_(ctx, 'get', 'clusterqueue', '-o', 'json'))['items']
        except Exception as e:
            rows.append(dict(context=ctx, queue=None, error=str(e).splitlines()[0][:200]))
            continue
        for q in items:
            name = q['metadata']['name']
            if queues and not re.search(queues, name):
                continue
            spec, status = q.get('spec', {}), q.get('status', {})
            quota = sum(_quantity_(r.get('nominalQuota'), resource) for g in spec.get('resourceGroups', [])
                        for f in g.get('flavors', []) for r in f.get('resources', []) if r['name'] == resource)
            used = sum(_quantity_(r.get('total'), resource) for f in status.get('flavorsUsage', [])
                       for r in f.get('resources', []) if r['name'] == resource)
            borrowed = sum(_quantity_(r.get('borrowed'), resource) for f in status.get('flavorsUsage', [])
                           for r in f.get('resources', []) if r['name'] == resource)
            pending, admitted = status.get('pendingWorkloads', 0), status.get('admittedWorkloads', 0)
            if not all and not (quota or used or pending or admitted):
                continue
            rows.append(dict(context=ctx, queue=name, cohort=spec.get('cohortName', spec.get('cohort')),
                             quota=quota, used=used, free=max(quota - used, 0), borrowed=borrowed,
                             pending=pending, admitted=admitted))
    df = pd.DataFrame(rows, columns=['context', 'queue', 'cohort', 'quota', 'used', 'free', 'borrowed',
                                     'pending', 'admitted'] + (['error'] if any('error' in r for r in rows) else []))
    for col in ('quota', 'used', 'free', 'borrowed'):
        if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
            df[col] = df[col].astype('Int64')
    return df


def kubedash_main(argv: list | None = None) -> None:
    """`dbx.kubedash [CONTEXT ...] [-r RESOURCE] [-q REGEX] [-a]`: print `kubedash`'s table (`kubedash` returns it)."""
    p = argparse.ArgumentParser(prog='dbx.kubedash', description=kubedash.__doc__.split('\n')[0])
    p.add_argument('contexts', nargs='*', help="kubectl contexts to query (default: kubectl's, one per cluster)")
    p.add_argument('-r', '--resource', default='nvidia.com/gpu',
                   help="the resource to tabulate: nvidia.com/gpu (default), cpu, memory, ...")
    p.add_argument('-q', '--queues', help="only the queues whose name matches this regex")
    p.add_argument('-a', '--all', action='store_true', help="include queues with no quota, use or workloads")
    a = p.parse_args(argv)
    df = kubedash(a.contexts, resource=a.resource, queues=a.queues, all=a.all)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(df.to_string(index=False))


