---
name: defining-pipelines
description: How a project defines its dbx pipelines and implements their Datablocks - one XXXPipeline(dbx.Pipeline) class plus one xxx(name, **ops) function per pipeline, presets keyed by name feeding a flat XXXPipeline.VAR, @dbx.pipeline on the entrypoint function, one qualified tag module.qualname:name shared by the pipeline and its blocks (dbx.entrypoint_tag), preset names prefixed with what they are built over, the tree's VARs built from the pipeline VAR only, **ops reserved for runtime behavior and handed on with self.operational, imports from dbx only, machine-specific defaults in the project's env module, no datalake handling in code, __build__ resuming topic by topic (every topic checked before it is built), and __read__ returning the data of one explicitly named topic (no default, no fallthrough, never a path, None when nothing sensible to read). Load when writing or reviewing a project's pipelines or Datablocks, adding a preset, or deciding whether a setting is identity (VAR) or operational (ops).
---

# Defining pipelines

A pipeline names a tree of Datablocks and the configuration it is built with.
Each pipeline is **two definitions**, side by side in the project's
`pipelines.py`:

1. **`XXXPipeline(dbx.Pipeline)`**: its `VAR` is the whole configuration of
   the tree, and its `__create__` builds the tree from that VAR and returns
   the root, which is the *entrypoint*.
2. **`xxx(name, **ops)`**: a function that holds the named **presets** and
   returns an `XXXPipeline`. For example, `sc_representation()` returns an `SCRepresentationPipeline`.

The Datablock classes the pipelines build live in the project's
`datablocks.py`, which `pipelines.py` imports from. `pipelines.py` holds only
the pipeline classes, their functions and preset helpers.
- A block's lake directory is named by its class's `module.Name`
  (`sctx.datablocks.SCScenery/<hash>`). So moving or renaming a Datablock
  class moves its blocks, even when the hash stays the same. Don't move the
  lake's directories, and **don't touch anchoring**: no `ANCHOR` on a class,
  no `anchor=` on a block. Reach the old builds with a specialization pointing
  at the old anchor, declared in `pipelines.py` and passed as
  `SPECIALIZATIONS=` where `__create__` constructs the block:

  ```python
  SCENERY_SPECIALIZATIONS = [
      Datablock.Specialization(spec={}, topics=SAME, anchor='sctx.pipelines.SCScenery',
                               note="SCScenery moved from sctx.pipelines to sctx.datablocks unchanged"),
  ]
  ```

  The full chain -- specialize, fix dbx for a generic miss, redirect manually
  for a corner case, and who decides -- is the skill `reaching-existing-builds`.

```python
from dataclasses import dataclass

from dbx import Pipeline

from sctx import env
from sctx.datablocks import SCRepresentation, SCScenery


class SCRepresentationPipeline(Pipeline):
    """querious-v2 SR over a scenery: an SCRepresentation rooted on the SCScenery it takes as input."""

    JOURNAL = True

    @dataclass
    class VAR(Pipeline.VAR):
        # SCScenery
        dataroot: str = env.SCENES_DATAROOT
        splits: tuple = ('val',)
        # SCRepresentation
        split: str = 'val'
        checkpoint: str = None

    def __create__(self):
        v = self.var
        scenery = SCScenery(spec={'dataroot': v.dataroot, 'splits': v.splits}, tag=self.tag)
        return SCRepresentation(spec={'scenery': scenery, 'split': v.split, 'checkpoint': v.checkpoint},
                     tag=self.tag, **self.operational)


"""
dbx.pprint "sctx.pipelines.sc_representation('nuscenes_08-14-2026_stage3_val')"
dbx.pprint "sctx.pipelines.sc_representation('nuscenes_08-14-2026_stage3_val').entry.cite()"
dbx.pprint "sctx.pipelines.sc_representation('nuscenes_08-14-2026_stage3_val_smoke').entry.build_tree()"
dbx.pprint "sctx.pipelines.sc_representation('nuscenes_08-14-2026_stage3_val', n_gpus=2).entry.build_tree()"
"""
def sc_representation(name, **ops) -> SCRepresentationPipeline:
    val_scenery = dict(dataroot=env.SCENES_DATAROOT, splits=('val',))
    stage3 = dict(checkpoint='checkpoints/stage3.pt')
    presets = {
        'nuscenes_08-14-2026_stage3_val': dict(**val_scenery, **stage3, split='val'),
        'nuscenes_08-14-2026_stage3_val_smoke': dict(**val_scenery, **stage3, split='val', max_scenes=2),
    }
    return SCRepresentationPipeline(spec=presets[name], tag=f"{__name__}.sc_representation:{name}", **ops)
```

## The rules

- **`name` is a short, unique tag for a set of VARs.** The VARs define the
  pipeline, the pipeline defines the Datablock tree, and `.entry` is the
  tree's root. So `xxx(name)` alone identifies the whole tree;
  `xxx(name, **ops)` adds how to build it.
- **Usage examples go in one `"""` block directly above each pipeline
  function.** The block holds several `dbx.pprint "pkg.pipelines.xxx(name, ...)"`
  lines covering the presets and the common calls: the pipeline itself,
  `.entry.cite()`, `.entry.build_tree()` (with the ops you'd actually use),
  and `.entry.read(topic)`. Each line should run as written from a shell.
- **The pipeline VAR, and only it, defines the tree's VARs.** Every value that
  reaches a block's `spec` in `__create__` comes from `self.var`. Don't read
  the environment, module globals, files or `ops` there. A pipeline is then
  reproduced exactly by its own quote, which is what the `PipeJournal` records.
- **The pipeline VAR is flat: one field per VAR of the tree.** Each preset of
  `sc_representation` is one set of VARs. Some define the upstream `SCScenery.VAR`, the
  rest the `SCRepresentation.VAR`, and `SCRepresentation.VAR` takes the upstream `SCScenery` as an
  input VAR. A value that two blocks share is one field, passed to both. Group
  the fields by the block they feed, with a comment per group.
  - Don't hold a preset *name*, or another pipeline, in a VAR. The VAR holds
    the values themselves.
- **Presets live inside the function**, as a dict keyed by `name`; each value
  is a dict that becomes `XXXPipeline.VAR`.
  - Presets that share a block's VARs build them from one local dict
    (`val_scenery` above) and splat it into each preset.
  - An unknown name should fail by listing the presets that exist.
- **`**ops` is runtime behavior only**: parallelization, `n_workers`,
  `n_devices`/`n_gpus`, batch sizes that don't change results, verbosity.
  - These settings change how the tree is built, never what it is. Changing
    one must not change any block's hash.
  - The function passes `ops` to the pipeline, and `__create__` hands them on
    with `**self.operational`, only to the blocks that use them.
    `dbx.Pipeline.operational` is the pipeline's undeclared keyword
    arguments.
  - Every keyword argument a Datablock doesn't declare stays out of its hash,
    kept as a parameter and shown in its `quote()`. The block reads it in
    `__build__` as `getattr(self, 'n_gpus', <default>)`, and passes it on to
    whatever does the work, e.g. the code that makes the Ray call. That code
    takes these as arguments and has no defaults of its own hidden elsewhere.
- **Identity or operational?** If changing the setting could change any
  output byte, apart from nondeterministic numerics, it belongs in the VAR.
  Otherwise it's an op. When unsure, put it in the VAR: a hash that moves too
  often costs a rebuild, while a hash that should have moved but didn't
  silently serves stale data.
  - Identify inputs by content where their path could be reused for
    different content: e.g. a checkpoint's `checkpoint_sha256` as a VAR,
    checked in `__build__`.
  - Give a block a `VERSION` and bump it when the code that writes its output
    changes what it writes.
- **Tags: one qualified tag per preset, never the bare preset `name`.**
  Several projects write to one journal, and a bare
  `'nuscenes_08-14-2026_stage3_val'` says neither which project nor which
  pipeline. The pipeline and its blocks carry the same tag,
  `<module>.<function>:<name>`, e.g.
  `sctx.pipelines.sc_representation:nuscenes_08-14-2026_stage3_val`. A tag of
  the function alone would conflate all of its presets.
  - `@dbx.pipeline` tags the pipeline so when the function's first argument
    is `name` (see below).
  - The function tags the blocks it constructs, the root and those upstream
    of it, with `dbx.entrypoint_tag(fn, name)`, the same string.
- **Preset names say what they are built over.** Prefix a preset with the
  presets or the dataset upstream of it (`nuscenes_08-14-2026_stage3_val`, not
  `stage3_val`), so that the same model over another scenery doesn't take the
  same name. Renaming a preset moves nothing: the name only selects values, and
  blocks are keyed by class, version and hash.

  A tag filter matches by containment, so `tag='nuscenes_08-14-2026_stage3_val'` still finds
  them. A tag is only a label: tagging a block that is already built doesn't
  move it or change its hash.
- **`JOURNAL = True`** on pipelines that are run and looked up again later.
  Leave it off for helpers that other pipelines only nest.

## Imports, environment and the datalake

- **Import from `dbx`, the single point of import:**
  `from dbx import DATADIR, DATAFILE, Datablock, Pipeline`, not
  `from dbx.datablocks import ...`. `dbx` re-exports everything, and project
  code shouldn't depend on which dbx module defines a name.
- **Machine-specific defaults live in the project's `env` module** (e.g.
  `sctx/env.py`): module-level names, each read from the environment
  variable of the same name, with a default when it isn't set:

  ```python
  SCENES_DATAROOT = os.environ.get('SCENES_DATAROOT', '/mnt/.../nuscenes')
  ```

  Presets and VAR defaults name `env.SCENES_DATAROOT`, never a literal path.
  Processes started elsewhere (Ray workers) don't inherit this environment,
  so the launcher forwards the `env` module's resolved values to them, once
  (e.g. as Ray `runtime_env` `env_vars`). Never prefix individual commands
  with `VAR=...`.
  No other module computes paths from `__file__`, the home directory or
  another root. That keeps the code relocatable: moving it, or running on
  another machine, means setting an environment variable, not editing code.
- **Don't handle the datalake in code.** Every block resolves `DBX_DATALAKE`
  itself; don't pass `datalake=` or read the variable. It is the user's to
  set, in their shell profile; never set, export or override it for them
  (not in a venv's `activate`, not in a script). If it isn't set, dbx
  files exec journals under `./dbx` in the current directory, and that
  directory then shadows the `dbx` package for `python -c` run from there.
- **A dependency that isn't a package** (a research checkout that imports its
  own modules by bare name) is isolated in one module of the project that
  puts it on `sys.path` (e.g. `sctx.qv2`). Its location comes from the
  `env` module, and everything that uses the dependency imports that module
  first. Import it inside `__build__` when it is heavy, so building a tree
  doesn't import it.
- **The project is a package**, with a `pyproject.toml` whose dependencies
  come from its `requirements.txt`, installed with `pip install -e .` into
  the env that runs it. Commands run as `python -m pkg.module`, not by script
  path.

## Implementing `__build__`: resume topic by topic

dbx skips a whole block that is already valid, but not the topics inside one.
A build that died part way leaves some topics written and others not, so
`__build__` must handle that itself:

- **Build the topics in dependency order**: a topic after the topics it reads.
- **Check each topic's validity before building it, and skip it when it is
  valid.** Every topic, with no exceptions. Use `self.valid_topic(t)` for a
  file topic. A directory topic exists as soon as its first item is written,
  so `valid_topic` alone says nothing there: check that every item it should
  hold is present (every expected file, one per split or per scene).
- **Use `self.path(topic)` for every topic.** It returns the file for a
  `DATAFILE` and the directory for a `DATADIR`, and `ensure_dirpath=True`
  creates what is needed for writing. Don't switch to `dirpath()` by topic
  type.
- **Call Python steps as Python, never through a shell.** A step written in
  Python is a function the block calls, on remote workers too (a Ray task
  calling the function in the project's interpreter, through
  `runtime_env={'py_executable': sys.executable}`). Running it as a
  subprocess command loses its exceptions, tracebacks and return values,
  and leaves only an exit code. A script's CLI is a thin `main()` around the
  function.
- **Write each item atomically** (to `x.tmp`, then `os.replace`), so an item
  that exists is complete and the check above can trust it.
- **Write a summary topic last** (e.g. `config`, with the resolved settings and
  counts). Then the block is valid only once everything before it is.
- **Bracket each long step with `self.log.section(...)`**, so the log shows
  which step a failed build died in.
- **Imports go at module level, all of them.** A project's own modules, and
  the packages it depends on (torch, ray, a module that puts a checkout on
  `sys.path`), are imported at the top, even when only `__build__` uses them.
  The python-style exceptions (a circular import, a genuinely optional
  dependency) are rare in project code. "Only the build needs it" is not one
  of them.

## Implementing `__read__`: the data, one topic at a time

```python
def __read__(self, topic):
    if topic == 'config':
        return dbx.read_yaml(self.path('config'))
    if topic == 'sr':
        return {p.stem: dict(np.load(p)) for p in self._scene_files_()}
    if topic == 'viz':
        return None
    raise ValueError(f"{type(self).__name__} has no topic {topic!r}")
```

- **No default topic.** `__read__(self, topic)`: a reader always names the
  topic.
- **One explicit branch per topic, and no fallthrough.** A final `return`
  shared by several topics makes it hard to tell which topic a call reads.
  End with a `raise` for a name that isn't a topic.
- **Return the data, never its path.** `read('cache')` returns what the cache
  holds, not the directory. Paths are `path(topic)` and `dirpath(topic)`.
- **Return `None` when there is nothing to read, or reading makes no sense**
  (videos meant to be watched, a marker file).
- **Read it even when it is large.** Whether something is too big to load is
  the caller's call. Say so in the topic's note, e.g.
  `DATADIR(note='... ~7 MB per scene, and read() loads every scene')`. Mind
  that a note renders into the block's identity, so changing it re-keys the
  block.

## Use `@dbx.pipeline` for the entrypoint function

`dbx.pipeline` turns the preset function into the pipeline: it builds the
`<FnInCamelCase>Pipeline` class, tags each call `module.qualname:<name>`,
and journals it with `journal=True`. Use it, and tag the blocks with
`dbx.entrypoint_tag`, which gives the same string:

```python
@dbx.pipeline(journal=True)
def sc_representation(name, **ops):
    presets = {...}                      # as above
    v = _preset_(presets, name)
    tag = dbx.entrypoint_tag(sc_representation, name)
    scenery = SCScenery(spec=..., tag=tag)
    return SCRepresentation(spec=dict(scenery=scenery, ...), tag=tag, **ops)
```

- The function's leading arguments become the pipeline's VAR, so here the
  VAR is `{'name': 'nuscenes_08-14-2026_stage3_val'}`. That is the exception to "no preset name
  in a VAR": the pipeline is identified by the preset, and the blocks by the
  values the preset holds. Their hashes don't depend on the name.
- `**ops` are operational: passed to the function, out of every hash. Hand
  them to the blocks that use them as `**ops`.
- Don't declare a `tag` parameter: the decorator passes one to the pipeline
  class, and a second would arrive in `**ops`.
- Calls are cached per process by all their arguments, so a pipeline other
  pipelines nest is created, and journaled, once.

The explicit form above, a hand-written `XXXPipeline` plus a function
returning it, is equivalent, but its `__create__` doesn't see the preset
name. It sets the qualified tag on the pipeline by hand
(`tag=f"{__name__}.sc_representation:{name}"`), and `__create__` passes
`self.tag` to its blocks. Use one form or the other within a project, not both.

## Running a pipeline on a cluster

Don't write job scripts or YAML for a pipeline. A pipeline runs itself:
`p.runner('<target>').build_tree()` runs `dbx.eval(<p.quote()>).entry.build_tree()`
there and journals the submission on the pipeline, in the PipeJournal (`CLUSTERS.md`); the same
`Runner` lists (`runs`, which prints as their dash, row `i` being `runs()[i]`) and cancels
(`cancel`, or `runs()[i].cancel()`) its runs, and
`dbx.runner(<Datablock class or anchor>)` those of every datablock of a class.
In a one-liner, wait and check:

```bash
dbx.pprint "(p := <pipeline>).runner('<target>', ...).build_tree().wait() and p.entry.valid()"
```

That puts two requirements on the definition:

- **The quote must rebuild the pipeline there.** Keep everything that decides
  the tree in its VAR and its operational kwargs, and read paths through the
  project's env module (above), not from the submitting shell's state.
  `runner(..., dry_run='cluster')` checks exactly this: same hash there as here.
- **The project defines its targets, not the pipeline.** Clusters are values
  (`dbx.kube.KubeCluster`) in a module of their own, passed at call time
  (`p.runner(LAMBDA)`); nothing is registered. For names in one-liners, the
  module's dict of them is a `dbx.clusters` entry point in the project's
  packaging. A pipeline module never defines or names a cluster.

*This skill is meant to grow: record new conventions here as the pipelines
that use it settle them.*
