"""Python functions on GPUs through Ray, one task per call: `run(fn, calls, ...)`.

    results = dbx.gpu.run(build_shard, [dict(shard=i, num_shards=8) for i in range(8)],
                          log_dir=logs, n_gpus=1, n_cpus=20, env=myproject.env.exported())

calls `fn(**kwargs, log=...)` for each `kwargs` of *calls* as a Ray task on a node with `n_gpus`
GPUs free (a MIG slice is one), and returns their results in order. An exception raised by `fn` is
raised here, as the same exception type. Each task's output, and every line it logs, goes to
`log_dir/task<i>.log`.

It joins the Ray cluster at *address* -- 'auto': the one this machine is part of, an Anyscale
workspace's or a Kube Job's own head (`dbx.kube`'s `ray=True`). The workers run this process's
interpreter, so they import what it imports, and get *env* as their environment variables: the
locations a project resolved here, say, for workers that would resolve them differently. On
Anyscale, *working_dir* defaults to `dbx.anyscale.working_dir()`: without one, its runtime-env hook
snapshots the current directory for every task.

This is the fan-out within one process's work -- a block's build across a node's GPUs -- where
`dbx.clusters` ships a whole call to another machine.
"""
import contextlib
import os
import sys

from . import anyscale


__all__ = ['run']


def run(fn, calls: list[dict], *, log_dir: str, n_gpus: float = 1, n_cpus: float = 4,
        env: dict | None = None, working_dir: str | None = None, address: str | None = 'auto') -> list:
    """Call *fn* once per kwargs in *calls*, each as a Ray task with *n_gpus* and *n_cpus*; their results, in order."""
    import ray  # Heavy, and only this path needs it.
    runtime_env = {'py_executable': sys.executable, 'env_vars': dict(env or {})}
    working_dir = working_dir or (anyscale.working_dir() if anyscale.present() else None)
    if working_dir:
        runtime_env['working_dir'] = working_dir
    ray.init(address=address, ignore_reinit_error=True, runtime_env=runtime_env)
    task = ray.remote(num_gpus=n_gpus, num_cpus=n_cpus)(_call_logged_)
    return ray.get([task.remote(fn, kwargs, os.path.join(log_dir, f"task{i}.log")) for i, kwargs in enumerate(calls)])


def _call_logged_(fn, kwargs: dict, log_path: str):
    """`fn(**kwargs)` with its output and its `log` lines in *log_path*."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        return fn(**kwargs, log=lambda msg: print(msg, flush=True))
