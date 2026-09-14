# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **`dbx.exec` takes a sequence of statements, and a comment.** The CLI
  argument was a single expression, evaluated with `eval` -- so anything that
  needed a name to be bound first needed a script, and a one-liner could not
  build a block and then read it. It is parsed with `ast.parse` now and runs
  every statement the string holds, separated by `;` or by newlines, in one
  namespace, returning the value of the last:

  ```bash
  dbx.pprint "b = my.Block(spec={'x': 1}); b.build(); b.read()  # nightly"
  ```

  One dict serves as both globals and locals, so the statements run the way
  module-level code does: split, a comprehension in a later statement could
  not see a name an earlier one bound -- the rule that keeps a class body from
  seeing its own names -- which would break exactly the idiom sequencing
  exists for. The dotted names are collected from every statement's AST rather
  than from the text before the first `(`, so any statement may name a module
  that nothing has imported yet -- and a name an earlier statement bound, which
  is no module path at all, resolves to nothing rather than failing.

  A trailing `#` comment is ignored when the command runs -- Python's parser
  drops it -- and recorded in the exec journal's new `comment` column. The
  `exec` column still holds the string VERBATIM, comment and all, because a
  journal row is the only record of what was run and has to stay re-runnable;
  the separate column is there because the expression says what a command did
  and only its comment says what it was for, and the journal is read long
  after the person who typed it could be asked. The comment is tokenized, not
  split on `#`, so a `#` inside a string literal stays part of the expression.

- **`Datablock.SPECIALIZATIONS`: reading a narrower block's data instead of
  rebuilding it.** A class grows a `VAR` field and a topic; every block of it
  re-keys, and the topics that did not change are rebuilt for nothing. A
  `Datablock.Specialization(spec=..., topics=..., note=...)` says that when the
  new fields hold the values in its `spec`, the topics it names ARE the topics
  of the block this class used to be — whose identity is this one's with those
  fields dropped and those topics alone.

  That identity is reconstructible from here, because a signature is built from
  the spec and the topics and nothing else: `type(specialization=...)` renders
  it and `get_hash(specialization=...)` hashes it, so the narrower block's
  build is found in the journal by its own hash, with no access to the older
  class and no record that it ever existed. An unbuilt block with a matching
  specialization reads through it; `build()` then produces only the topics the
  specialization does not cover (`buildtopics()`, the complement of
  `redirected_topics()`).

  A pin is matched against the RENDERED spec, not the `spec` dict a caller
  passed: a field left at its default is absent from that dict, and a field
  left at its default is exactly the case this exists for. Specializations are
  tried in declaration order and the first that both matches *and resolves*
  wins, so one whose build has been cleared does not shadow the next. A miss is
  never silent: `specializations()` reports every declared one with the hash it
  looked for and why it did not apply.

  Installing one is recorded, through `UNSAFE_redirect`: an installed
  specialization is a claim about where a block's data came from, and the
  journal entry is the only record that this block read another's build and
  which specialization said it could. Recording also costs less than not
  recording — it writes the hidden `.redirection` topic, which every later
  construction reads instead of scanning the journal again, so the write
  happens once per block rather than the scan happening once per construction.
  Within a construction, the resolution travels with the block through pickle,
  deepcopy and `set()`. `use_specializations='memory'` installs without
  recording; `UNSAFE_specialize()` is the explicit form.

- **Partial redirection.** `UNSAFE_redirect(topics=[...])` redirects only the
  topics it names; every other topic reads as it would unredirected. `build()`
  no longer declines wholesale when a block is redirected — it declines only
  when the redirection is total, and otherwise builds the rest.

### Changed
- **A redirection is no longer part of a block's identity.** `type()` used to
  append `_redirected_paths_=...` once a redirection was installed, so `hash`
  depended on *when* it was first called — before the redirection or after —
  and `anchorkeypath`, the journal directory and the redirection lookup moved
  with it. Masked, until `keyby` stopped naming the hash, by `__setstate__`
  building the logger name out of `self.key` and caching `_hash` on the way
  past. A block does not become a different block by being read from somewhere
  else.

- **`set()`/`replace()` refuse `spec=`.** `set()` amends a block's non-identity
  parameters; a new spec is a new block, and handing one back through a method
  that reads as an amendment of this one also carried this block's resolved
  state over to an identity it was never resolved for. Construct it instead:
  `type(b)(**{**b.dfn, 'spec': {...}})`.

- **Writing to a redirected topic is refused.** `path(topic, ensure_dirpath=True)`
  on a redirected topic raises rather than quietly not creating the directory:
  it is how this codebase asks for somewhere to write, and not creating the
  directory does not stop the write when the other block's directory is already
  there.
- **`dbx.stills`: `Still`, `LightningBuilder`, `Weights`.** One training run
  as a config-addressed Datablock — topics `ckpts`/`logs`, the `_COMPLETE` marker,
  the checkpoint save/upload/free/resume dance, the TensorBoard log symlink, the
  atexit sync, the `UNSAFE_*` helpers, the `Trainer` assembly and a generic
  `dataloaders()`. Every training pipeline in soundworld had grown its own copy of
  all of it; `IJEPAsaurUSStill` shed ~930 lines becoming a subclass, with its hash,
  key, signature, `quote()` and `cite()` byte-identical (its `VAR` is inherited
  whole, so the field *order* a `LEGACY_NORM` block hashes can only be the base's).

  Two ways to say what to train. **Explicit mode** puts a `LightningBuilder` in
  `VAR.lightning`, which is what you want when the module's construction has
  upstream artifacts of its own. **cfg mode** names two ordinary classes — `Model`
  and `Lightning`, importing nothing from dbx — and mirrors their `cfg_`-prefixed
  keyword arguments into the still's own `VAR`; a name declared on both is one VAR
  field passed to both, which is the right reading for a knob like `crop_size` that
  two objects must agree about. `Still.export()` then writes a standalone module
  carrying that configuration as literals, with no dbx in it.

  The mirrored `VAR` is checked-in source, generated once by `scaffold_still()`,
  **not** reflected at import time: `_typed_specdict` walks every `VAR` field into
  the hash, so a `cfg_` knob added in a file that does not mention dbx would
  silently re-key every run of that model. As source, that movement is a diff.
  `__check_cfg__` raises on construction if the two have drifted apart, and refuses
  a `cfg_` name that collides with one of `Still`'s own fields (`cfg_ckpt` is
  initial weights; `VAR.ckpt` is the run to resume from — one field cannot be both).

  `VAR.ckpt` holding another *block* is a **warm start**: its latest checkpoint is
  loaded weights-only, at step 0, with a fresh optimizer, and only until this run
  has a checkpoint of its own. Carrying a different run's optimizer moments and
  epoch counter over is not a resume, it is a way to lose a training run quietly —
  a source that already reached its own `max_epochs` leaves the new one with
  nothing to do, so `fit()` returns at once and the `_COMPLETE` marker lands over
  an untrained model. (The stills in soundworld took the old reading, which is why
  `IJEPAsaurUSPoseStill`'s phase-1 chains passed `var_reset_optimizer_state=True`
  or ran short.) A bare *path* in `VAR.ckpt` still resumes in full — that is how
  you point a run at its own checkpoint that has moved, and the escape hatch from
  the new default. A warm start that matches none of the model's parameter names
  now raises rather than training from random init behind a banner that says
  otherwise.

  Not imported by `dbx/__init__.py`: it needs lightning at module scope, so import
  it by name, exactly as for `dbx.datastreams`. New `[lightning]` extra.
- **`dbx.datastreams`: `block_split_indices`, `block_split_ranges`,
  `val_loader_workers`.** The split counterpart of `ChunkShuffleSampler`, and here
  for the same reason: a per-sample-random train/val split scatters both halves
  across the whole table and defeats the local shard cache as thoroughly as no
  split at all. `val_loader_workers` caps a validation loader that reads
  `val_max_batches` batches so it stops prefetching (and discarding) an order of
  magnitude more shards than it reads, evicting the training working set every time
  it validates. In `datastreams` rather than `stills` so that torch-only code
  can use them without pulling in lightning.

### Breaking Changes
- **The `Data` prefix comes off three modules and their classes.**
  `dbx/databackbones.py` → **`dbx/backbones.py`**, `dbx/dataprobes.py` →
  **`dbx/probes.py`**, `dbx/datastills.py` → **`dbx/stills.py`**; and with them
  `DatamodelEvaluator` → **`ModelEvaluator`**, `DatamodelEvaluatorFactory` →
  **`ModelEvaluatorBuilder`**, `DataformerEvaluator` → **`TransformerEvaluator`**,
  `DataformerEvaluatorFactory` → **`TransformerEvaluatorBuilder`**,
  `DatafeatureAffineLogisticProbe(r)` → **`FeatureAffineLogisticProbe(r)`**,
  `DatafeatureStatsProbe` → **`FeatureStatsProbe`**, `Datastill` → **`Still`**,
  `Datalightning` → **`LightningBuilder`**, `Dataweights` → **`Weights`**. Every old
  name still resolves — the class names as plain aliases, and `dbx.databackbones`,
  `dbx.datamodels` and `dbx.dataprobes` as the SAME module object under the old
  path, the way `dbx.datapoints` and `dbx.datafeatures` already are.

  Unlike the `datapoints`/`datafeatures` rename, the class names moved too, so
  `fqcn` moved with them and no amount of pinning `__module__` could have held
  it. That is allowed here only because nothing was ever *built as* one of these:
  every backbone builder, probe and still downstream is a subclass, which reports
  its own module and its own name, and there is no `dbx.*` anchor directory in any
  root. Renaming a base costs nothing; renaming the subclass is what would cost.
  Checked, not assumed — `IJEPAsaurUSStill`'s hash, key and `type()` are
  byte-identical across the rename.

  `dbx.datastills` gets no module alias at all: it is a week old, was never
  released, and an alias exists to keep a *recorded* string resolving. There is no
  such string. `Datalightning` becomes `LightningBuilder` rather than plain
  `Lightning`: the bare word is already `Still.Lightning`, the `LightningModule`
  subclass cfg mode dispatches to, and that is the one a still author writes.
  The suffix also matches `ModelEvaluatorBuilder` beside it.
- **Renamed `Datablock.CONFIG` → `Datablock.VAR` and `.cfg` / `.config` → `.var`.**
  Both old names are kept as aliases, so existing subclasses keep working unchanged:
  `Datablock.CONFIG` still resolves (it is an alias of `Datablock.VAR`), and a subclass
  that declares `class CONFIG(Datablock.CONFIG)` has it bound to `self.VAR` in
  `__setstate__`, so it survives construction, `deepcopy` and unpickling. `.cfg` and
  `.config` are now read-only aliases of `.var`. `VERBOSE_CONFIG` is likewise renamed to
  `VERBOSE_VAR`, with the old spelling still honored. Identity is unaffected — spec,
  norm, hash and key are byte-identical either way.
- **Renamed `validate_cfg=` → `validate_vars=` and `valid_cfg()` → `valid_var()`.**
  `valid_cfg()` is gone, with no alias. `validate_cfg=` survives only as a deprecated
  `__init__` parameter and state key, so a block reconstructed from a `dfn` recorded
  before the rename still works. It is kept *explicit* rather than left to `**kwargs`
  because a stray dynamic kwarg would be silently ignored — validation would stay on
  for a block whose `dfn` says `validate_cfg=False` — and would then persist as a dead
  parameter in `dfn`/`quote()`/`cite()`. It is never re-serialized: `__getstate__` emits
  only `validate_vars`. Neither flag is part of a block's identity — `norm()` is built
  from `url`/`anchor`/`hash` and `spec` alone.
- **Retired `VALIDATE_CFG_EXEMPTIONS`** in favour of `TREE_SKIP_VALIDATION`, which
  already did the same job. A subclass still declaring the old name now raises
  `AttributeError` at construction naming the replacement, rather than being ignored —
  silently dropping it would re-enable the validation it existed to suppress, surfacing
  as a confusing "Not all upstream Datablocks in var are valid" at build time.
- **Renamed `hashstr` → `signature` and `superhashstr` → `supersignature`**, in the
  `Datablock` properties, the `Bid` fields, the journal columns, and the recorded
  files (`signature.txt` / `supersignature.txt`). No aliases on `Datablock` or `Bid`.
  The string itself is unchanged, so `hash`, `superhash` and `key` are byte-identical —
  this renames the name, not the identity. Journals written before the change recorded
  the columns as `hashstr` / `superhashstr`: `DatajournalEntry.signature` and
  `.supersignature` fall back to them, treating a NaN as absent so that a journal
  spanning the rename (both columns present, NaN-filled per row) reads correctly on
  either side.
- **Removed the whole `*DatablocksBuilder` family, `select_builder()` and `_DATABLOCKS_BUILDERS`**
  — `Inline`, `Multithreading`, `Multiprocessing`, `Ray`, `TorchMultithreading` and
  `TorchMultiprocessing`, plus the private `_build_block`, `_build_block_with_to` and
  `_TorchBlockCallable_` helpers. Nothing constructed them: `Datastack` resolves
  `executor_cls` to the `*CallableExecutor` family directly, and `select_builder` was
  reachable only from its own tests. Use the executors (`select_executor`,
  `callable_executor`, `parallelization='...'` on a `Datastack`) instead. The one
  behaviour with no executor equivalent was `RayDatablocksBuilder` copying a remote
  block's `__getstate__()` back onto the local object after a build; that is a no-op
  unless a subclass registers a build-time value as a serializable parameter.
- **Renamed `FDCapture` → `OutputTee`, and removed `Tee`.** `Tee` wrote to several
  Python file objects at once, for installing as `sys.stdout` — but that only sees writes
  going through the Python stream object, so it missed C extensions writing to fd 1/2,
  subprocesses (which inherit descriptors, not `sys.stdout`), and any code holding the
  real stream from before the swap. `OutputTee` redirects the descriptors themselves
  with `dup2`, so all three are caught. It was already unused; nothing referenced it.
  `capture_output=True` still **mirrors rather than silences** — output continues to
  reach the terminal, which is what `Tee` was meant to provide.
- **Removed the `@tagged` decorator** (and its `_make_tag` / `_TAGGED_SKIP_DEFAULTS`
  helpers), which auto-generated a call-string tag for pipeline functions. Unused.
  Pass `tag=` explicitly.
- **Renamed `JournalFrame` → `Datajournal` and `JournalEntry` → `DatajournalEntry`.**
  No aliases are kept; update any `from dbx.datablocks import JournalEntry`. The
  `journal()` function, `Datablock.journal()` / `Datablock.Journal()`, and every journal
  column name are unchanged. (Entries in earlier releases below still use the old names,
  which is what those releases shipped.)
- **`norm()` now quotes strings and reprs spec values once**, which **changes the
  hash of every Datablock** that does not opt out. The old rendering was ambiguous:
  top-level string kwargs were bare (`url=abfss://…`), and non-string spec values were
  repr'd twice (int `5` → `'5'`) while strings were repr'd once, so `n=5` and `n='5'`
  produced the *same* hash. Set **`LEGACY_NORM = True`** on any class whose artifacts
  are already built and keyed, to keep the exact bytes its hashes were computed from.
  New classes should leave it alone (default `False`).

### Added
- **`DatastreamTab` / `DatastreamTable`** in `dbx.datastreams` — an abstract `Datablock` /
  `Datastack` pair over parallel MDS *slices*, documented in
  [DATAPOINTS.md](DATAPOINTS.md). A tab declares `SLICES = ('frames', 'annotations')`
  and writes them in lockstep inside `__build__`; the `data` group of its `TOPICS`
  is synthesized from them, so the slices are covered by the block's hash. A table
  inherits `SLICES` from its `TAB` and needs only `n_tabs`, with `__tab__(idx)`
  — `Datastack.__block__` for tables, whose `super()` fills in placement — implemented
  only when a tab needs a spec of its own. Block placement, `__split__`, and a `__stack__` that merges
  every tab's per-slice `index.json` into one index per slice are preimplemented.
  A tab's shards land in the table's per-slice root (`<table>/data/<slice>/<tabdir>/`)
  rather than under the tab's own key, because `StreamingDataset` resolves a shard
  relative to the directory holding the index that names it. Both classes read back
  as `data(slice)` (lumped samples), `datastream(slice)` (one live
  `StreamingDataset`), `dataset(*slices)` (the zip of those) and `stats(slice)`
  (a `__stats__` hook). Local scratch — shard cache, write staging, decompression —
  goes under `cacheroot`, which defaults to `<localroot>/streaming` rather than `/tmp`
  and is overridden with the `cache=` kwarg.
- **`BlockShuffleSampler` / `ResumableDataLoader` / `shuffled_block_order()`** in
  `dbx.datastreams` — shuffle contiguous blocks of the index space, and within each
  block, instead of permuting the whole range. Consecutive sample indices share an
  MDS shard, so `DataLoader(shuffle=True)` scatters every access and defeats the
  shard cache; block shuffling keeps the working set down to a few shards while still
  randomising both orders each epoch. `fixed_epoch=True` pins a validation sampler's
  order so capped val runs stay comparable; `state_dict()`/`load_state_dict()` (plus
  `ResumableDataLoader`, which surfaces them where trainers look) allow approximate
  mid-epoch resume. `DatastreamTable.sampler()` builds one whose `block_size` defaults
  to the table's **own** shard capacity — read off the merged index by the new
  `shard_sizes()` / `samples_per_shard()` / `n_samples()`, without downloading a shard
  — rather than leaving every caller to guess a constant.
- **`ZipIterableStreamingDatasets` and `dataset(mode='iter')`** in `dbx.datastreams` —
  zip the slices by *iteration order* instead of by physical index. Map-style indexing
  reads through `StreamingDataset.get_item()`, which downloads a missing shard inline
  on the calling thread; the download-ahead thread, the rank/worker partitioning,
  `num_canonical_nodes`, the shard-locality shuffle and mid-epoch resumption all live
  in `__iter__` and so never run. Iterating each slice puts every slice back in
  possession of them, which on remote storage is normally the largest single factor in
  throughput. The merge is unchanged — both classes now share it through `ZipBase` —
  and so are the defaults: `mode='map'` remains what `dataset()` does.
  Iterator-mode zipping is only correct while every slice yields the same sequence,
  which `shuffle=False` gives for free (the partition reads sample *counts*, never
  shard structure) but `shuffle=True` does not: the permutation is derived from the
  per-shard sample counts, so differently-sharded slices shuffle differently and pair
  unrelated samples. `__init__` checks that and refuses; `shared=` +
  `validate_shared=True` remains the running check, and is effectively required here.
  `mode='iter'` also demands `batch_size=`, which `StreamingDataset` would otherwise
  only complain about on the first batch from inside a `DataLoader` worker.
- **`slice_writers(..., flush_every=N)`** in `DatastreamTab` — break every slice onto a
  new shard every *N* samples, so all slices carry identical shard boundaries.
  `MDSWriter` otherwise starts a shard on a byte budget, so slices of different
  per-sample size split at unrelated places; that is invisible to index-addressed
  zipping but is exactly what stops `mode='iter'` from shuffling. Whichever of
  `flush_every` and `size_limit` comes first still ends the shard, so a `size_limit`
  that fires first is detected and raised rather than left to surface as a misaligned
  shuffle. Writes are counted per slice, which also turns the lockstep contract into a
  checked one: a tab that has not written every slice the same number of times raises
  instead of producing a table that cannot be zipped.
- **`open_datastream()`** in `dbx.datastreams` — opens a `StreamingDataset` over an
  MDS index directory, local or remote, translating `abfs(s)://` to `azure-dl://` and
  retrying once past a stale `Reused local directory` shared-memory registration.
- **`ZipStreamingDataset` gained a merge policy** — `columns` (per-source column
  projection), `shared` + `validate_shared` (keys expected in several sources, and
  an equality check that makes a mis-zipped set of streams loud rather than silently
  misaligned), `on_conflict` (`'last'`/`'first'`/`'error'`) and `skip_none`. Defaults
  reproduce the previous plain last-wins merge exactly, so existing callers are
  unaffected. This is what a multi-slice `DatastreamTable` needs to be readable
  column-by-column, and it generalises `soundworld.databits.ZipDataset`.
- **`ZippedStreamingDatasets`** — alias of `ZipStreamingDataset`. The singular stays
  the canonical name; nothing is renamed.
- **Hierarchical `TOPICS`** — a dict value may itself be a dict, nesting topics:
  ```python
  TOPICS = {'data': {'frames': DIRTOPIC, 'annotations': SYNTOPIC,
                     'index': 'index.csv'},
            'model': 'model.pt'}
  ```
  Every topic-addressing method takes one name per level: `path('data', 'frames')`,
  `read('data', 'annotations')`, `ls`, `list`, `size`, `dirpath`, `validtopic`,
  `UNSAFE_clear`. The nesting is mirrored on disk under the block's key. A *group*
  is addressable in its own right — `dirpath('data')` is the parent directory,
  `path('data')` is the dict of its members' paths, and `validtopic('data')` is the
  conjunction over the leaves beneath it (`validpath` already recursed into dicts).
  New `leaftopics()` enumerates leaves as name tuples, and `is_topicgroup()` tests a
  path. Journal entries record the declared shape, and `DatajournalEntry.ls`/`list`/
  `size`/`_is_dir_topic`/`_is_syntopic` take the same per-level arguments.
  **Fully backward compatible**: flat dict-TOPICS, list-TOPICS and no-TOPICS blocks
  produce byte-identical signatures — hence the same hash, key and storage paths —
  verified against the previous commit. A nested leaf is rendered `topic:data/frames=...`,
  so a topic name may no longer contain `/`, which would make that ambiguous.
- **`DIRTOPIC`** — the filename of a directory topic in a dict-valued `TOPICS`:
  `TOPICS = {'images': 'images.csv', 'masks': DIRTOPIC}`. It *is* `None`, the value the
  topic machinery has always tested for, so `{'masks': None}` stays valid and produces
  an identical `signature`, `hash` and `key`; the constant only says out loud what a
  bare `None` left the reader to infer.
- **`SYNTOPIC`** — a *synthetic* topic, one the block presents but never stores, so it
  has no location: `TOPICS = {'data': 'data.parquet', 'cache': SYNTOPIC}`. `path()` and
  `dirpath()` are both `None`, nothing is created, listed, copied or cleared for it, and
  it is vacuously valid — a topic that was never going to be written cannot be missing,
  so it must not hold the block back. Distinct from `DIRTOPIC`, which *is* a location —
  a real directory that merely has no filename inside it. `SYNTOPIC` is `()` rather than
  another `None`-alike exactly so the two cannot collide; it is still declared, so it
  appears in the `signature` (as `topic:cache=()`) and is part of the block's identity.
- **`Datablock.cite()` recorded alongside `quote()`**: new `Bid.cite` field, a `cite.txt`
  written by `write_journal_entry()`, and a `JournalEntry.cite` property (returns `None`
  on journals written before the column existed, rather than raising).
- **`entry_code` on every journal entry** — a fresh uuid per `write_journal_entry()` call,
  recorded as a column and returned by the call, plus `DatajournalEntry.entry_code` and a
  matching `.uuid` accessor. It is the only field that identifies a *row*: `hash` and
  `key` are shared by every entry of a block, `uuid` by every entry of one live instance,
  and `datetime` only to its resolution — two entries written in the same microsecond, or
  by two processes at once, collide. So `journal(entry_code=code, loc=0)` addresses
  exactly the row a caller wrote. Follows `uuid16`, so the two identifiers in one entry
  are the same shape. Journals written before the column read as `None` rather than
  raising. A code resolves only until its *instance* writes again: a journal file is keyed
  by `self.dt`, so a second call from one instance overwrites the first — which is why
  `build()` leaves a `build:end` and no `build:start`. Pass a distinct `journal_prefix` to
  keep both.
- **`UNSAFE_redirect()` — read this block's topics from somewhere else.** Takes exactly
  one of `filter=` ({column: value} pairs selecting a journal entry, as `journal()`
  filters, the first — newest — match winning, so `{'hash': other.hash, 'event':
  'build:end'}` follows that block as it is rebuilt where `{'entry_code': code}` is pinned
  to the one build that code was returned for) and `paths=` ({topic: path}, naming the
  locations outright, consulting no journal at all). `topic_map=` re-keys a filtered
  entry's topics, reading mine → theirs: `{'out': 'output'}` says this block's `out` is
  that entry's `output`. Topics line up by name to begin with and the map only adds to
  that; a mapping whose target the other side lacks leaves its topic *unredirected* and
  says so, since asking for theirs and silently getting mine is the one answer that is
  certainly wrong. The non-None parts are packaged into a `redirection` dict and written
  to a fresh journal entry. Nothing is copied, moved or validated — a redirection is a
  note in the journal. It travels in the entry rather than in a file of its own, unlike
  `message` and `quote`/`norm`/`spec`, because a stand-in for missing data must not itself
  depend on a second file still being there; and it is written under a `redirect-` prefix
  so an instance that has already journalled does not overwrite that entry. A filter
  selects from this block's journal, which is per anchor, so it reaches another build of
  this class but not another class's unless the two share an `anchor=`.
- **`path()` and `dirpath()` are where a redirection takes effect**, so everything that
  resolves through them — `read()`, `valid()`, `ls()`, `list()`, `size()` — describes the
  data actually being read, and no reader needs to know: an override reads
  `self.path(topic)` exactly as it always did. A topic the redirection does not name keeps
  its own path, so a partial redirection redirects only what it names. `local=True` is
  never redirected (the local cache is this block's own) and a redirected path is never
  `ensure`d (creating directories inside another block's data is not this block's
  business). UNSAFE because every path the block reports then names data it did not
  produce and whose hash does not describe it, which nothing downstream can detect — so
  the redirection is announced at INFO when it resolves. `build()` refuses a redirected
  block: nothing would read what it wrote, and a `build_tree()` sweeping past would
  otherwise quietly rebuild the very block someone redirected away from.
  `Datablock.redirection` is the resolved `Redirection(paths, entry, filter, topic_map)`,
  cached — including the `None`, so a redirection recorded after a block has looked is
  seen by the next instance rather than that one — and detached from the frame its entry
  came out of. Whether a block HAS one is answered from its own journal directory rather
  than by globbing the anchor's, because `path()` asks on first use and a table of a
  thousand tabs asks a thousand times. `redirect=False` at construction opts out of all
  of it.
- **`valid()` goes through a `__valid__(path=None)` hook.** The default validates this
  block's topics, which for a redirected block are the redirected-to paths already, so it
  needs no knowledge of redirection either; *path* is the redirected-to block directory,
  for an override that wants to validate the location by more than the presence of its
  topics, and is `None` when the block is not redirected or was given paths outright.
- **`valid_topics()`, `valid_paths()`, `valid_path()`** are the canonical spellings,
  joining `valid_topic()`; `validtopics()`, `validpaths()` and `validpath()` remain as
  deprecated aliases.
- **A table's slices are synthesized from its `TAB`**, and a subclass declaring `TOPICS`
  keeps its bases' *slice* topics while dropping their ordinary ones — slices are what
  makes such a block the kind of block it is, whereas an ordinary topic belongs to the
  class that declared it. A table keeps `tabs`/`done` whatever it declares, and `slices`
  now reads off the class (`LetterTable.slices`) as readily as off an instance, as a
  tuple: what a block is sliced by is settled once its class is, and it feeds the hash.
- **`block_shuffle_sampler()`** on a datapoint block, the deprecated alias of
  `chunk_shuffle_sampler()` that `BlockShuffleSampler` and `block_size` already had.
- **`datablocks[azure]`** — `adlfs`, the fsspec driver for `abfs://` and `abfss://`, as a
  named extra rather than an assumption. Nothing imports it (fsspec resolves it from the
  url), so its absence surfaced only as `ValueError: Protocol not known: abfss` from a
  block that happened to be given one.
- **`diff()`, `difftopics()`, `diffversion()` — the rest of what a hash is made of.**
  A `signature` is a norm, a version and the topics, joined, so those three diffs between
  them account for every way two blocks can hash differently; `diff()` returns them as a
  `Diff` triple (also reachable as `.norm`, `.topics`, `.version`), and `any(a.diff(b))`
  is "is this a different block". `difftopics()` compares `signature_topics()` — the very
  segments the signature is built from, now rendered in one place instead of twice — so
  the diff and the hash cannot drift: it is non-empty exactly when the topics contribute
  to a difference in signature. It reports a sparse `{topic path: (self, other)}` dict,
  with `ABSENT` for a path one side does not declare, and a difference belonging to no
  single path — a reordering, or `TOPICS = {}` against no TOPICS at all, both of which
  move the hash — under the `SIGNATURE_TOPICS` sentinel key. `diffversion()` returns
  `(self, other)` or `None`, comparing as the signature renders (`1` and `'1'` are the
  same version because they are the same hash) while reporting both values as they are.
  Each takes its other side as a block, a journal entry, a declaration, or a `journal=`
  selector, as `diffnorm()` does.
- **`diffnorm()` descends recursively** into nested blocks and spec dicts, returning a
  *sparse* nested dict so a changed leaf appears at the end of a short path instead of
  as two multi-kilobyte strings. New options: `recursive=False` (previous flat
  behaviour), `deslash=True` (strip escapes from the reported values), `report=True`
  (flat `path` + self/other text), `maxlen` (truncation, report only).
- **`diffnorm()` reports TYPED leaves.** A norm is flat text, but the text records the
  type, so a non-`LEGACY_NORM` block's `ori_extent=15.0` comes back as the float `15.0`
  while a legacy block's comes back as the string `'15.0'` — making a pair like
  `(15.0, '15.0')` legible as "different `LEGACY_NORM` settings", not "the value changed".
  Detection still compares the raw text, so `n=1` vs `n=1.0` is reported even though
  `1 == 1.0` in Python; and where evaluation would erase the difference (bare vs quoted
  `url=`) the bytes are shown instead. `raw=True` returns the source text.
- **`norm(legacy=...)` / `supernorm(legacy=...)` / `diffnorm(legacy=...)`** — temporarily
  override `LEGACY_NORM` for one call, propagating to nested blocks so the whole subtree
  renders the same way. `None` (default) means every block uses its own flag and is
  byte-identical to before; `signature` never passes an override, so `hash` is unaffected.
  `a.diffnorm(b.norm(legacy=False), legacy=False)` gives typed leaves even for classes
  that still carry the marker.
- **`ABSENT`** — a key present on only one side of a `diffnorm` now carries this marker
  (`<absent>`) instead of `None`, which is no longer distinguishable from a value that
  genuinely *is* `None` now that leaves are typed.
- **`Datablock.format_diffnorm(diff)`** — renders a `diffnorm` dict as text.

### Fixed
- **`entry.inst()` raised `AttributeError` on any entry journaled without a git
  repo.** An entry is a Series built with `dropna()`, so a column recorded null
  loses its LABEL, and attribute access — which a Series answers out of its index —
  raises where every reader expects `None`. `revision` and `gitrepo` are null
  whenever neither `DBX_GIT_REPO` nor `DBX_USE_WORK_REPO` is set (a pip-installed
  dbx, a container without the checkout), and `inst()` defaults to
  `revision='journal_entry'` — so it failed on exactly the entries that recorded no
  revision, taking `UNSAFE_redirect()` with it. `instantiate()` and `rinst()` now
  read both through `DatajournalEntry.column()`, and `None` means what
  `instantiate()`'s own default already means: the current environment. Latent
  since the `dropna()` arrived in `loc=`/`iloc=`; invisible wherever a repo is
  configured, since then the revision is a real sha.
- **A remote tab was cached where mosaic guessed, not where the table said.**
  `DatapointTable.datastream()` (and `DatapointFold`'s) computed a cache directory,
  created it, and then never passed it on, so every `Stream` over a remote tab was left
  without a `local=`. Mosaic then derives one itself — `{tmpdir}/{blake2s(remote)}`, the
  same path for every process on the box — and REFUSES to reuse it, so the second open of
  that tab (a second process, a second run, a retry after a crash) died with `Could not
  create a temporary local directory ... already exists`. It cannot simply be handed to
  `StreamingDataset`, which takes `streams=` or `remote`/`local` and never both: it
  belongs on each stream, one subdirectory per tab, named by the tab's hash so it is
  unique per tab and the same across runs. `_tab_stream()` now raises rather than letting
  a remote stream go without one. Local tabs are unaffected — a local slice is its own
  cache, and nothing is copied for it.
- **A journal on any non-local filesystem read back empty.** `Journal()` globbed its
  parquet files through the block's filesystem — which names them protocol-stripped — and
  then handed those paths to pandas, which looked for them on the LOCAL disk. Every entry
  was logged as "unreadable" and skipped, so a `memory://` or remote journal came back
  empty rather than failing. Entry files are now opened through that filesystem.
- **A table's tabs escaped to `DBX_ROOT`.** `DatapointTable.__tab__()` constructed its
  `TAB` without a url, so a table built anywhere other than the ambient root wrote its
  tabs to an unrelated one, where they were then looked for in vain under the table. A
  tab now inherits the table's url — raw, so a relocatable table stays relocatable tab by
  tab — and a `DatapointFold` likewise takes the partition's.
- **`Datacollator` could not consume what `data()` returns.** Every caller — a feature
  build, both probes — passes `data(*collator.slices, concat=True)`, a `{slice: data}`
  mapping in which the batch is already stacked; iterating that yields its KEYS, so the
  collation was over strings. Such a mapping is now collated as the batch it is: one pair
  passes its array through untouched, several are stacked along a new axis 1. New
  `signal_pairs`/`label_pairs` accessors give the normalized `(slice, column)` pairs.
- **`DatafeatureTab.__post_init__` and `FeatureStatsProbe.__build__` raised
  `NameError`** on names left behind by the move to `Datacollator` (`factory`,
  `feat_slice`/`sig_slice`): neither block could be constructed or built. The probe's
  per-tab breakdowns now follow the collator's first signal/label pair, and say so when
  it names more than one.
- **`SLICES` is rejected rather than ignored.** It is retired — a slice is a `TOPICS`
  entry valued `SLICETOPIC` — and a class still carrying one came out with no slices at
  all: valid, buildable, and empty. It now raises at construction, naming the
  replacement, like every other retired attribute.
- **`leave_breadcrumbs()` raised `IsADirectoryError` on any directory topic.** It passed
  `path(topic)` to `leave_breadcrumbs_at_path()`, which opened it for writing — but for a
  directory topic (list-TOPICS, or dict-TOPICS with `DIRTOPIC`) `path()` *is* the directory, so
  the call blew up and the method was unusable on such a block.
  `leave_breadcrumbs_at_path(path, crumbs=None)` now always takes a **directory** path:
  with `crumbs` the breadcrumb is `{path}/{crumbs}`, without it `{path}.crumbs` alongside.
  A file topic passes its own filename, so its breadcrumb is still its own empty file and
  the block still reads as valid; a directory topic gets the sibling marker rather than a
  stray entry inside a listing of itself. Breadcrumbs are now touched only when nothing is
  there, so they never clobber a real artifact. `SYNTOPIC` topics are skipped.
- **A filtered `Datajournal` kept the row labels of the unfiltered journal**, so
  `loc=`/`Datajournal.get()` — which index by label — raised `KeyError` for positions
  whose rows the filter had removed. `lastbuilt()` is `journal(event='build:end').get(0)`,
  so it failed for any block whose newest journal entry was some other event — the normal
  state for a block whose artifact was copied in (`UNSAFE_copy_from:END`). Filtered
  journals are now renumbered 0..N-1; frames you slice yourself keep pandas' label
  semantics.
- **`diffnorm(journal=...)` silently dropped extra selector keys**: `dict(event='build:end',
  iloc=0)` ignored `event` and returned the newest entry of *any* event. Extra keys are now
  forwarded to `journal()` as column filters; combining them with `entry_path` raises.

## [0.0.3] — 2026-05-12

### Breaking Changes
- **`journal()` API**: The `entry` parameter is renamed to `loc`; a new positional `iloc` parameter selects rows by integer position. Passing both raises `ValueError`.
- **`.dbx` metadata paths**: `fqcn` is now always included in the `.dbx/` directory hierarchy (`.dbx/{fqcn}/journal/`, `.dbx/{fqcn}/log/`, etc.), even when `anchor == fqcn`. Previously the `fqcn` segment was omitted in that case.
- **`parse_storage_options`**: Moved from a private helper in `datablocks.py` to the public `dataparts` module. Import as `from dbx.dataparts import parse_storage_options`.

### Added
- **`Datablock.lastbuilt()`** — returns the most recent `build:end` journal entry, or `None`.
- **`Datablock.running()`** — returns the latest `build:start` entry with no matching `build:end` (i.e. an in-progress build), or `None`.
- **`JournalEntry.bid`** property — reconstructs a `Datablock.Bid` namedtuple from a journal row.
- **`JournalFrame` hash prefix-match filter**: `journal(hash="ab3c")` now matches any hash starting with `"ab3c"`, so short hashes work.
- **`BUILD_TREE_EXEMPTIONS`**: Class-level set of spec keys to skip during `build_tree()`, allowing dependency graphs to be built without re-training checkpoint-based subtrees.
- **`dataparts.default_storage_options()`** — convenience wrapper that reads `DBX_STORAGE_OPTIONS` from the environment.
- **`storage_options` threading**: `JournalFrame` and `JournalEntry` now carry `storage_options` through the full read chain, fixing remote journal access.
- Serialization roundtrip tests for `Datablock` and `Datastack` (`deepcopy`, `pickle`, `__getstate__`/`__setstate__`).

### Fixed
- **`Datastack.executor_cls`**: Refactored from an `__init__`-set attribute to a property, so objects reconstructed via `deepcopy` or `pickle` (which bypass `__init__`) still resolve the correct executor.
- **`capture_output`**: Build logs are now written to a local temp file first, then uploaded to the (possibly remote) `logpath` on completion. Fixes failures when `fs.open()` on remote backends did not support streaming writes.
- **`fs_full_path` Azure fix**: Manually reconstructs the full `abfss://container@account.dfs.core.windows.net/path` form, because `adlfs`'s `unstrip_protocol()` drops the account portion.
- **`keyby='tag'` guard**: Raises `ValueError` immediately when `keyby='tag'` is specified without providing a `tag=` argument, preventing the infinite recursion that previously occurred.
- **TOPICS-only `path()`/`dirpath()`**: When a `Datablock` defines `TOPICS` without `TOPICFILES`, the topic is now correctly treated as a directory name under `anchorkeypath`, not derived from `path()`.
- **Dirty-repo check timing**: The uncommitted-changes check now runs in `gitwrkreposetup()` against the original repositories *before* cloning, so uncommitted work is never silently discarded.

## [0.0.2] — 2026-05-08

### Fixed
- **Remote storage path resolution**: `anchorkeypath`, `anchorpath()`, `dirpath()`, and `path()` now return correct protocol-prefixed URLs for remote storage backends (`abfs://`, `gcs://`, `memory://`, etc.). Previously, these methods returned protocol-free paths that silently resolved to the local filesystem.
- **`JournalEntry.anchorkeypath` / `anchorhashpath`**: Now derive root from the stored `url` field instead of referencing a non-existent `root` field. Legacy journal entries with an explicit `root` field are still supported.
- **`validpath()`**: Replaced inconsistent `os.path.exists()` / hardcoded `gcs` branches with a single `self.fs.exists()` call that works for any fsspec backend.
- Eliminated 14 redundant `_url_to_fs()` calls in favour of the already-initialised `self.fs`.

### Added
- `dataparts.fs_full_path(fs, path)` — utility that re-attaches the protocol prefix for remote filesystems while keeping local paths bare (compatible with Python's built-in `open()`).
- 29 new tests using the `memory://` filesystem to verify path correctness, build/valid lifecycle, journal round-trips, and `JournalEntry` path derivation on non-local storage.

## [0.0.1] — 2026-05-07

### Added
- Initial release on PyPI.
- `Datablock` base class with config-addressed storage, journaling, and fsspec-based IO.
- `Datastack` for parallel block builds (inline, multithreading, multiprocessing, Ray, Torch).
- `JournalEntry` / `JournalFrame` for structured build-event history.
- Environment-variable configuration (`DBX_ROOT`, `DBX_STORAGE_OPTIONS`, `DBX_DIRTY_REPO_OK`).
