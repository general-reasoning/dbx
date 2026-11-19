"""`dbx.env`: the environment variables dbx reads, in one place.

Each is an attribute, read from `os.environ` when it is accessed -- never
once at import -- so a variable set, or a test's `monkeypatch.setenv`, after
`import dbx` is seen: `dbx.env.DBX_DATALAKE` is `$DBX_DATALAKE` now, or
None when it is unset or empty. Values are the strings as set; a reader
decides what a flag's string means.

The module is callable, as `dbx.env` was before it was a module:
`dbx.env('KEY')` is the specline `$dbx.getenv('KEY')`, as
`dbx.databits.env` makes it.

Writing one stays `os.environ[...] = ...`: what dbx sets there is meant for
its child processes and workers as much as for itself. `dbx._pinshim_` reads
its own three (`DBX_PIN_*`) directly, as it must not import dbx.
"""
import os
import sys
import types

# Every variable dbx reads, and what it says.
VARS = {
    # Where blocks are.
    'DBX_DATALAKE': "The datalake: the url blocks, journals and pipelines are under when none is passed.",
    'DBX_STORAGE_OPTIONS': "fsspec storage options for the datalake, as `parse_storage_options` decodes them.",
    'DBX_LOCAL': "The local staging root of a block whose url is not local; `/tmp/dbx` by default.",
    'DBX_CACHE': "A datatable's cache directory, where its block passes no `cache=`.",
    # Which code runs.
    'DBX_GIT_REPO': "The dbx and project repos, `dbx_repo:project_repo`; else the repo the cwd is in.",
    'DBX_USE_WORK_REPO': "'True': run from a clean clone of the repos, not the working tree.",
    'DBX_DIRTY': "'True' or '1': run the working tree as it is -- DBX_USE_WORK_REPO=False, DBX_DIRTY_REPO_OK=True.",
    'DBX_DIRTY_REPO_OK': "Set: do not refuse a repo with uncommitted changes.",
    'DBX_WORK_ROOT': "The work repo clone, as `gitwrkreposetup` set it up -- for child processes.",
    'DBX_PINNED_REVISION': "The revision this interpreter was pinned to before it started; it refuses another.",
    'DBX_PIN_SOURCE': "The pin shim's clone sources, space-separated, dbx first (read by `dbx._pinshim_`).",
    'DBX_PIN_REVISION': "The pin shim's revisions, `dbx_rev:project_rev` (read by `dbx._pinshim_`).",
    'DBX_PIN_NODE_ROOT': "Where the pin shim caches its clones on a node; `<tmp>/dbx-pins` by default.",
    # Where it runs.
    'DBX_RAY': "The Ray cluster `remote` connects to when no `address=` is passed: 'auto', or an address. "
               "Unset: whatever `ray.init()` finds or starts.",
    'DBX_CLUSTER': "The cluster a run was submitted to, journaled with what it does.",
    'DBX_RUN': "The run, journaled with what it does.",
    'DBX_SESSION': "The submitting command's session, which a run joins.",
    'DBX_DB': "The Postgres url of the block-lock database; unset, blocks are not locked.",
    'DBX_LOG_SELECTION': "Comma-separated names: `Datalog.selected()` is on for them alone.",
}

# Families of variables, by prefix: any name under one is read as a listed one is.
FAMILIES = {
    'DBX_LOG_': "`DBX_LOG_<LEVEL>` -- `DBX_LOG_INFO`, `DBX_LOG_DEBUG`, ...: a `Datalog` level on or off, "
                "where neither the log nor its scope says.",
}


def __getattr__(name):
    """`$name` as it is now, or None when it is unset or empty -- for a name in `VARS` or under a `FAMILIES` prefix."""
    if name in VARS or name.startswith(tuple(FAMILIES)):
        return os.environ.get(name) or None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}: not a dbx environment variable")


def __dir__():
    return sorted([*globals(), *VARS])


# DBX_DIRTY=True: run the working tree as it is -- DBX_USE_WORK_REPO=False and DBX_DIRTY_REPO_OK=True together, set
# here once, before anything reads them, so every reader and every child process sees the pair. It overrides a
# DBX_USE_WORK_REPO=True a shell profile exports.
if (os.environ.get('DBX_DIRTY') or '').lower() in ('true', '1'):
    os.environ['DBX_USE_WORK_REPO'] = 'False'
    os.environ['DBX_DIRTY_REPO_OK'] = 'True'


class _Callable_(types.ModuleType):
    """This module, callable: `dbx.env('KEY', default)` is `dbx.databits.env`'s specline for *KEY*."""

    def __call__(self, key, default=None):
        from .databits import env
        return env(key, default)


sys.modules[__name__].__class__ = _Callable_
