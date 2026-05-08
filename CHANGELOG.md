# Changelog

All notable changes to this project will be documented in this file.

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
- `Datastack` for sharded parallel builds (inline, multithreading, multiprocessing, Ray, Torch).
- `JournalEntry` / `JournalFrame` for structured build-event history.
- Environment-variable configuration (`DBX_ROOT`, `DBX_STORAGE_OPTIONS`, `DBX_DIRTY_REPO_OK`).
