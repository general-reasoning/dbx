---
name: naming-counts
description: How to name a count, a cap, a per-unit count and a quantity with a unit in dbx and the projects built on it - a count is n_<things> (n_gpus, n_cpus, n_workers, n_viz_workers, n_shards, n_frames), never the bare plural (gpus, cpus, workers) and never num_<things>; a per-unit count is n_<things>_per_<unit> (n_cpus_per_gpu); a cap is max_<things>; a quantity carries its unit as a suffix (mem_gib, window_s, distance_m). Third-party names (Ray's num_gpus, a pool's max_workers, a DataLoader's num_workers) stay only at the call into that API. Load when naming or reviewing a parameter, an operational kwarg, a VAR field, a job option or a config key that counts or measures something.
---

# Naming counts and quantities

One spelling for each kind of number, so a name says what it holds without
reading the code.

| kind | spelling | examples |
|---|---|---|
| a count of things | `n_<things>`, plural | `n_gpus`, `n_cpus`, `n_workers`, `n_viz_workers`, `n_shards`, `n_frames` |
| a count per unit of something else | `n_<things>_per_<unit>` | `n_cpus_per_gpu`, `n_frames_per_window` |
| an upper bound | `max_<things>` | `max_scenes`, `max_windows`, `max_new_tokens` |
| a quantity with a unit | `<what>_<unit>` | `mem_gib`, `window_s`, `stride_s`, `distance_m`, `speed_limit_mph` |
| an index into a count | `<thing>` (`i`, `shard`) | `shard` of `n_shards` |

**Never** the bare plural (`gpus=`, `cpus=`, `workers=`) or `num_<things>`
(`num_cpus=`, `num_shards=`), and never a count without its noun (`nproc=`, `j=`).
The bare plural reads as the things themselves -- a list of GPUs, the worker
objects -- and `num_` is a second spelling of what `n_` already says.

A quantity names its unit, whatever the unit: `gib=` alone says the unit and
not what it measures (`mem_gib`); `timeout=` says neither (`timeout_s`).

## Third-party names stay at the boundary

Ray's `@ray.remote(num_gpus=, num_cpus=)`, `ThreadPoolExecutor(max_workers=)`,
torch's `DataLoader(num_workers=)`, Lightning's `seed_everything(workers=True)`,
`kubectl`'s resource names: those are the other framework's API, forwarded as-is
exactly where it is called -- we do not rename them, nor wrap them only to rename them. Our own
signature around the call takes the `n_` name and translates:

```python
def run(fn, calls, *, n_gpus=1, n_cpus=4): ...
    ray.remote(num_gpus=n_gpus, num_cpus=n_cpus)(fn)
```

## Renaming one

- **A parameter, operational kwarg, job option or CLI flag** is outside a
  datablock's identity: rename it everywhere in one commit -- code, tests,
  examples, docstrings and docs -- and keep no alias, shim or deprecation warning
  for the old name: they only clog the code. Grep for the old name before calling
  it done (`complete every instruction`). A run record or config written before
  keeps the old name; whatever reads one ignores what it no longer knows.
- **A VAR field** is IN the identity: a rename moves the hash. Declare it with
  `redirect_vars` in a `Specialization` so the builds under the old name are
  adopted (see defining-pipelines).

## Renamed (2026-11-04)

- dbx: the runner's and `dbx.kube.job`'s `gpus`, `cpus`, `gib` → `n_gpus`, `n_cpus`,
  `mem_gib`; `kube.Cluster`'s `cpu_per_gpu`, `gib_per_gpu` → `n_cpus_per_gpu`,
  `mem_gib_per_gpu`; `dbx.gpu.run`'s `num_gpus`, `num_cpus` → `n_gpus`, `n_cpus`;
  `dbx.slurm`'s `gpus`, `cpus`, `nodes` → `n_gpus`, `n_cpus`, `n_nodes` (`mem` stays:
  its value is Slurm's string, unit included).
- sctx: the operational `cpus_per_gpu`, `loader_workers` → `n_cpus_per_gpu`,
  `n_loader_workers`; `num_shards`, `workers` and `gpus` parameters and CLI flags →
  `n_shards`, `n_workers`, `n_gpus`.

Not renamed, and not to be: `databits`' `workers` -- lists of worker processes, not
a count -- and every framework's own name, passed through as-is at the call.
