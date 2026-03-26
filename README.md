# DBX: Data Experiment Management Hub

DBX is a powerful tool for managing data experiments, providing a flexible interface for data handling, remote execution, and parallel processing.

## Installation

To install the package in development mode, run:

```bash
pip install -e .
```

Ensure you have the following dependencies installed (as listed in `requirements.txt`):
- `ray`
- `numpy`
- `tqdm`
- `gitpython`
- `fsspec`
- `pandas`
- `torch`
- `scipy`
- `pyyaml`

## Running Tests

The test suite uses the standard Python `unittest` framework.

### Pre-requisites for Remote Tests
Remote tests rely on **Ray**. If you are running tests in an environment with a git repository, the tests will fail if the repository has uncommitted changes (unless `DBXGITREPO` is unset).

### Execute all tests

To run the full suite of tests from the package root:

```bash
python -m unittest discover tests
```

### Execute specific tests

To run only the remote functionality tests:

```bash
python -m unittest tests/test_remote.py
```

## Features

- **Remote execution**: Use the `remote()` function to instantiate a remote dbx interpreter via Ray.
- **Parallel processing**: Use `RayCallableExecutor` to execute tasks in parallel across distributed workers.
- **Data handling**: Structured datablocks for tracking experiments and results.
- **Nested Proxying**: Transparently interact with remote objects as if they were local.

## Datablockable/Datastackable: Wrapper Method Mapping Reference

How `datablock(Datablockable)` and `datastack(Datastackable)` map user-defined methods onto the framework.

### `datablock(Datablockable)`

#### Shielded (framework version wins over MRO)

| Datablockable defines | Wrapper gets | How |
|---|---|---|
| `__init__(…)` | `Datablock.__init__` | `class_attrs['__init__']` — framework handles root, spec, hash, journaling. Datablockable's `__init__` is never called. |
| `build(…)` | `Datablock.build` | `class_attrs['build']` — framework build (validity check → journal → `__build__` → journal). The raw `build()` is remapped to `__build__` (see below). |
| `read(topic)` | `Datablock.read` | `class_attrs['read']` — framework read (topic validation → `__read__`). The raw `read()` is remapped to `__read__` (see below). |

#### Remapped to dunders (user implementation → framework hook)

| Datablockable defines | Becomes on wrapper | How |
|---|---|---|
| `build(…)` | `__build__(…)` | `class_attrs['__build__']` calls `cls.build(self, …)` explicitly — bypasses MRO. |
| `read(topic)` | `__read__(topic)` | `class_attrs['__read__']` calls `cls.read(self, topic)` explicitly — bypasses MRO. |

#### Lifted (copied to wrapper class level)

| Datablockable defines | Wrapper gets | Notes |
|---|---|---|
| `TOPICFILES` | `class_attrs['TOPICFILES']` | Copied as-is. |
| `TOPICFILE` | `class_attrs['TOPICFILE']` | Copied as-is. |
| `VERSION` | `class_attrs['VERSION']` | Copied as-is. |
| `CONFIG` | `class_attrs['CONFIG']` | Rebased onto `Datablock.CONFIG` for LazyLoader support (or synthesized empty). |

#### Unshielded (cls wins MRO if defined, else Datablock default)

| Datablockable may define | Falls back to |
|---|---|
| `__post_init__()` | `Datablock.__post_init__` (no-op) |
| `path(topic)` | `Datablock.path` |
| `valid(topic)` | `Datablock.valid` |
| `anchor` (property) | `Datablock.anchor` → `self.fqcn` |
| Any custom method / property | Not on Datablock — only accessible if cls defines it |

---

### `datastack(Datastackable)`

#### Shielded

| Datastackable defines | Wrapper gets | How |
|---|---|---|
| `__init__(…)` | `Datastack.__init__` | `class_attrs['__init__']` — framework init + parallelization setup. |
| `build(…)` (if any) | `Datablock.build` | `class_attrs['build']` — same framework build as Datablock. Remapped to `__build__` if defined (see below). |
| `shard(idx)` | `Datastack.shard` | `class_attrs['shard']` — framework shard: lazy-inits `_shards_`, calls `__shard__(idx)`, sets `keyby`. |

#### Remapped to dunders

| Datastackable defines | Becomes on wrapper | How |
|---|---|---|
| `shard(idx)` | `__shard__(idx)` | `class_attrs['__shard__']` calls `cls.shard(self, idx)`, then wraps result via `ShardBlock.from_datablockable()`. |
| `build(…)` *(optional)* | `__build__(…)` | Only if cls has `build()`. Calls `cls.build(self, …)`. Falls back to `Datastack.__build__` (shard-based). |
| `read(topic)` *(optional)* | `__read__(topic)` | Only if cls has `read()`. Calls `cls.read(self, topic)`. |
| `stack()` *(optional)* | `__stack__()` | Only if cls has `stack()`. Calls `cls.stack(self)`. Falls back to `Datastack.__stack__` (no-op). |

#### Lifted

Same as `datablock()` (`TOPICFILES`, `TOPICFILE`, `VERSION`, `CONFIG`), plus:

| Datastackable defines | Wrapper gets |
|---|---|
| `SHARD` | Used to create `_ShardBlock_ = datablock(cls.SHARD)` stored in `class_attrs`. |

#### Unshielded

| Datastackable may define | Falls back to |
|---|---|
| `n_shards` (property) | `Datastack.n_shards` (raises `NotImplementedError`) |
| `__post_init__()` | `Datablock.__post_init__` (no-op) |
| `path(topic)` | `Datablock.path` |
| `anchor` (property) | `Datablock.anchor` → `self.fqcn` |

---

### Build Flow

```
datablock wrapper                          datastack wrapper
═══════════════                            ═══════════════
wrapper.build()                            wrapper.build()
  │ Datablock.build (shielded)               │ Datablock.build (shielded)
  ├─ valid() → cls or Datablock              ├─ valid() → cls or Datablock
  ├─ __pre_build__() → journal               ├─ __pre_build__() → journal
  ├─ __build__()                             ├─ __build__()
  │   └─ cls.build(self, …)                  │   │ cls.build (if defined) OR
  ├─ __post_build__() → journal              │   │ Datastack.__build__ (default):
  └─ return self                             │   ├─ shards() → shard(0..n)
                                             │   │   └─ shard(idx)
                                             │   │       │ Datastack.shard (shielded)
                                             │   │       └─ __shard__(idx)
                                             │   │           ├─ cls.shard(self, idx)
                                             │   │           └─ ShardBlock.from_datablockable()
                                             │   ├─ builder.build_blocks(shards)
                                             │   └─ __stack__() → cls.stack or no-op
                                             ├─ __post_build__() → journal
                                             └─ return self
```
