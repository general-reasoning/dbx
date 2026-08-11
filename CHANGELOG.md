# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Breaking Changes
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
  [DATASTREAMTABLE.md](DATASTREAMTABLE.md). A tab declares `SLICES = ('frames', 'annotations')`
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
- **`UNSAFE_redirect()` — send a failed read to another entry's data**, which is what
  `entry_code` exists to make addressable. Takes exactly one of `entry_code=` (one entry,
  the value `write_journal_entry()` returned for it) or `filter=` (whichever entries match
  its `{column: value}` pairs, latest match winning), and records it verbatim in a fresh
  entry's new `redirection` column. So `{'hash': other.hash, 'event': 'build:end'}`
  follows that block as it is rebuilt, where an `entry_code` is pinned to the one build it
  was returned for. Nothing is copied, moved or validated: a redirection is a note in the
  journal, consulted by `read()` only *after* a read has already failed. The redirection
  travels in the entry rather than in a file of its own — unlike `message` and
  `quote`/`norm`/`spec` — because a fallback for missing data must not itself depend on a
  second file still being there; and it is written under a `redirect-` prefix so an
  instance that has already journalled does not overwrite that entry. UNSAFE because the
  read then answers with data this block did not produce and whose hash does not describe
  it, which nothing downstream can detect — so every followed redirection is announced at
  INFO.
- **`read()` follows a redirection, and `build()` declines to run under one.** A failed
  `__read__` is retried against the path the redirected-to entry recorded for that topic,
  announced at INFO both times; the original exception is re-raised unchanged whenever no
  usable redirection turns up, so a block with none behaves exactly as before. `__read__`
  takes the path as an optional argument — an override that does not accept it fails at
  the redirection rather than silently reading its own path. `Datablock.redirection` is
  the resolved entry, cached (a resolution reads the whole journal, so a block reading ten
  topics pays once) and detached from the frame it came out of; the `None` is cached too,
  so a redirection recorded after a block has looked is seen by the next instance rather
  than that one. `build()` refuses a redirected block: nothing would read what it wrote,
  and a `build_tree()` sweeping past would otherwise quietly rebuild the very block
  someone redirected away from. `redirect=False` at construction opts out of all of it.
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
- **`DatafeatureTab.__post_init__` and `DatafeatureStatsProbe.__build__` raised
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
