# datablocks

**Config-addressed, journaled data-experiment management for Python.**

[![PyPI](https://img.shields.io/pypi/v/datablocks)](https://pypi.org/project/datablocks/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

---

## Overview

`datablocks` (imported as `dbx`) is a framework for building reproducible data pipelines where every computation step is:

- **Config-addressed** — identified by a deterministic SHA-256 hash of its class name, configuration, and version.
- **Journaled** — every build writes a Parquet journal entry recording the full provenance (timestamp, git revision, config, hash).
- **Idempotent** — calling `build()` on an already-valid block is a no-op.
- **Storage-agnostic** — paths are resolved via [fsspec](https://filesystem-spec.readthedocs.io/), supporting local, S3, GCS, ADLS, and any other backend.

## Key Features

| Feature | Description |
|---|---|
| `Datablock` | Config-addressed unit of computation with topic-based output paths |
| `Datastack` | Orchestrates parallel builds of child Datablocks (blocks) |
| `VAR` | Dataclass-based configuration schema with lazy evaluation and specline support |
| Journaling | Automatic Parquet-based provenance tracking for every build |
| `env()` | Relocatable environment variable references that stay symbolic in handles |
| Callable Executors | Pluggable parallelism: inline, threading, multiprocessing, Ray |
| Torch Executors | Device-aware executors that automatically move callables between CPU/GPU |
| Remote Execution | Ray-based remote `dbx` interpreter with transparent proxying |
| Slurm Integration | Launch Ray clusters on Slurm and execute pipelines remotely |
| `@tagged` | Decorator that auto-generates human-readable pipeline tags |

## Installation

```bash
pip install datablocks
```

With Ray support (for remote execution and `RayCallableExecutor`):

```bash
pip install datablocks[ray]
```

For development:

```bash
git clone https://github.com/dmitry-karpeyev/datablocks.git
cd datablocks
pip install -e ".[dev]"
```

## Quickstart

```python
import os, tempfile
from dataclasses import dataclass
import pandas as pd
from dbx import Datablock, write_frame, read_frame

class SampleBlock(Datablock):
    """A minimal Datablock that generates a DataFrame."""

    TOPICFILES = {'output': 'data.parquet'}

    @dataclass
    class VAR(Datablock.VAR):
        n_rows: int = 10

    def __build__(self):
        df = pd.DataFrame({'x': range(self.var.n_rows)})
        write_frame(df, self.path('output', ensure_dirpath=True))
        return self

    def __read__(self, topic='output'):
        return read_frame(self.path(topic))

# Build and read
root = tempfile.mkdtemp()
os.environ['DBXGITREPO'] = ''  # skip git check for this example

block = SampleBlock(url=root, spec=dict(n_rows=5))
block.build()
df = block.read('output')
print(df)
#    x
# 0  0
# 1  1
# 2  2
# 3  3
# 4  4
```

## Core Concepts

### Datablock

A `Datablock` is the fundamental unit. Subclass it and implement:

- **`__build__(self)`** — produce your outputs (write to `self.path(topic)`).
- **`__read__(self, topic)`** — load and return the output for a topic.
- **`TOPICFILES`** — dict mapping topic names to filenames.
- **`VAR`** — dataclass defining your configuration schema.

```python
class MyBlock(Datablock):
    TOPICFILES = {'features': 'features.parquet', 'model': 'model.pt'}

    @dataclass
    class VAR(Datablock.VAR):
        input_path: str = None
        n_components: int = 64

    def __build__(self):
        # self.var.input_path, self.var.n_components are available
        # Write outputs to self.path('features'), self.path('model')
        ...
```

### Path Hierarchy

Every Datablock maps to a deterministic directory tree:

```
{root}/{anchor}/{key}/{topic}/{TOPICFILE}
```

- **root** — storage root (local path or fsspec URL)
- **anchor** — defaults to the fully-qualified class name
- **key** — derived from the content hash (configurable via `keyby`)
- **topic** — named output channel
- **TOPICFILE** — the filename within the topic directory

### Datastack

A `Datastack` manages a collection of child Datablocks (blocks) and builds them in parallel:

```python
class MyStack(Datastack):
    @dataclass
    class VAR(Datablock.VAR):
        n_items: int = 1000
        block_size: int = 100

    @property
    def n_blocks(self):
        return math.ceil(self.var.n_items / self.var.block_size)

    def __block__(self, idx):
        return MyBlock(url=self.root, spec=dict(idx=idx, ...))

stack = MyStack(url='/data', spec=dict(n_items=1000),
                parallelization='multithreading', n_workers=4)
stack.build()
```

### Configuration & `env()`

Use `env()` to reference environment variables symbolically — the handle stays portable across machines:

```python
from dbx import env

block = MyBlock(url=env('DATA_ROOT'), spec=dict(input_path=env('INPUT_DIR')))
# The handle contains "$dbx.getenv('DATA_ROOT')" instead of the resolved path
```

### Callable Executors

Pluggable parallelism for any sequence of callables:

```python
from dbx import callable_executor

executor = callable_executor('multithreading', n_workers=8, tag='my-job')
results = executor.execute(list_of_callables)

# Available strategies: 'inline', 'multithreading', 'multiprocessing',
#   'ray', 'torch_multithreading', 'torch_multiprocessing'
```

### Journaling

Every `build()` writes journal entries (Parquet) recording the timestamp, git revision, config, and hash. Query them later:

```python
j = block.journal()            # Datajournal (DataFrame subclass)
entry = j.get(0)               # DatajournalEntry (Series subclass)
print(entry.hash, entry.anchor, entry.revision)
```

## CLI

The package installs several console entry points:

```bash
dbx 'module.function(arg1, arg2)'          # evaluate a dbx expression
dbx.exec 'module.function(arg1, arg2)'     # same as dbx
dbx.pprint 'module.function(arg1, arg2)'   # evaluate and pretty-print
dbx.slurm.exec 'module.function()'         # execute on a Slurm Ray cluster
```

## Running Tests

```bash
python -m pytest tests/ -x -q
```

### Pre-requisites for Remote Tests

Remote tests rely on **Ray**. If you are running tests in an environment with a git repository, the tests will fail if the repository has uncommitted changes (unless `DBXGITREPO` is unset).

## Environment Variables

| Variable | Description |
|---|---|
| `DBX_ROOT` | Default storage root when `root` is not specified |
| `DBX_GIT_REPO` | Path to the git repository for revision tracking |
| `DBX_DIRTY_REPO_OK` | Set to skip the dirty-repo check |
| `DBX_LOG_INFO`, `DBX_LOG_DEBUG`, etc. | Control per-level log output |
| `DBX_LOG_SELECTION` | Comma-separated FQN list for `selected()` log filtering |

## License

[MIT](LICENSE)
