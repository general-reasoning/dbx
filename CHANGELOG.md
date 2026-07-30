# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Breaking Changes
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
- **`Datablock.cite()` recorded alongside `quote()`**: new `Bid.cite` field, a `cite.txt`
  written by `write_journal_entry()`, and a `JournalEntry.cite` property (returns `None`
  on journals written before the column existed, rather than raising).
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
  byte-identical to before; `hashstr` never passes an override, so `hash` is unaffected.
  `a.diffnorm(b.norm(legacy=False), legacy=False)` gives typed leaves even for classes
  that still carry the marker.
- **`ABSENT`** — a key present on only one side of a `diffnorm` now carries this marker
  (`<absent>`) instead of `None`, which is no longer distinguishable from a value that
  genuinely *is* `None` now that leaves are typed.
- **`Datablock.format_diffnorm(diff)`** — renders a `diffnorm` dict as text.

### Fixed
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
