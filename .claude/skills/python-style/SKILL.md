---
name: python-style
description: The Python style rules in this repo that are not about class layout - imports go at module level, except to break a circular import or to defer a heavy or questionable dependency like torch or ray; strings are formatted with f-strings, not % or .format(), except for a named template filled elsewhere; a section of computation is bracketed in log lines with Datalog.section(), not hand-written BEGIN/END lines; and dbx commands are run through dbx.exec or dbx.pprint, even from Python, so that the exec journal records them. Load when writing or reviewing any Python in dbx or hyp, adding an import, formatting a string, logging the start and end of a step, or running a dbx command. For member order and private naming, load class-layout.
---

# Python style

Four rules, and the few exceptions each one allows. Member order and naming
inside a class are covered in class-layout. This skill covers everything else
in a Python file.

## Imports go at module level

Every import goes at the top of the module, not inside a function or
method. That way a reader can see everything a module depends on in one
place, and a missing package fails when the module is imported, not halfway
through a build.

An import moves out of the header only for a specific reason, and the reason
has to apply to that import:

1. **A circular import.** Two dbx modules need each other, and importing one
   at module level would run before the other has finished defining its
   names. Import inside the function that needs it, and add a comment saying
   which module causes the cycle.
2. **A heavy or questionable dependency**, such as torch, ray or
   google.cloud: one that is slow to import, not always installed, or that
   has side effects on import. The repo handles these in two ways:

   - A **guarded module-level import** when the module uses the package in
     many places and can run without it:

     ```python
     try:
         import torch
         _TORCH_AVAILABLE = True
     except ImportError:
         _TORCH_AVAILABLE = False
     ```

     Code that uses the package checks the flag. Do not import the package
     again inside the function.
   - An **import inside the function** when only one or two paths need the
     package (`import ray` in the ray executor, `from google.cloud import
     storage` on the GCS path). A user who never takes that path never pays
     for the import.

Neither exception covers the standard library or packages the repo always
installs: `pprint`, `json`, `argparse`, `yaml`, `numpy`, `pandas`, `fsspec`.
Those go at the top. Some existing code imports them inside functions. That
is not a precedent: when you edit a function like that, move the import to
the top of the module.

**Watch the names a module-level import adds.** `dbx/__init__` star-imports
`databits`, `datablocks`, `journals` and the other modules that have no
`__all__`, so every name imported at the top of one of them becomes a `dbx`
name too. A later module can replace an earlier one's name. `import pprint`
in `datablocks` replaced `dbx.pprint`, the function from `databits`. `from
tqdm import tqdm` replaced `dbx.tqdm`, the module, with the function. Import
in a form that adds no clashing name, such as `from pprint import pformat`,
`import tqdm` called as `tqdm.tqdm(...)`, or `databits.eval` through `from .
import databits`, and compare `dir(dbx)` before and after. A private alias
also works, because a star import skips names that start with an underscore:
`databits` imports `pprint as _pprint_`.

Moving an import to the top also changes what a test's `monkeypatch` sees:
`from .store import Store` binds the class when the module is imported, so a
test that patches `hyp.store.Store` no longer reaches it. Import the module
and call `store.Store(...)`, rather than editing the test.

A test module follows the same rule. A missing optional dependency is
handled as described in running-tests, not by importing inside the test.

## Format strings with f-strings

Build strings with f-strings, not `%` and not `.format()`:

```python
raise ValueError(f"{self.__class__.__name__}: no entry for {key!r}")
```

The value is written where it appears in the output, so a reader does not
have to match up positions in an argument list.

This includes log messages: write `self.log.info(f"Saved checkpoint:
{filename}")`, not `self.log.info("Saved checkpoint: %s", filename)`.

One exception:

- **A named template filled in somewhere else.** If a format string is a
  constant defined once and filled at several call sites, it cannot be an
  f-string, because an f-string is evaluated where it is written. Name the
  fields and use `.format()` with keywords:
  `cls.COMPACTED.format(anchor=anchor)`, `SYSTEM.format(root=..., tools=...)`.
  Never use positional `{}` in a template like that.

Format codes that are not string formatting are not affected:
`time.strftime('%Y-%m-%dT%H:%M:%S')`, `squeue -o %t`, and similar arguments
passed to other programs stay as they are.

## Bracket a section of computation with `Datalog.section`

To log where a step starts and ends, use `section` on the log, not a
hand-written pair of BEGIN and END lines:

```python
with self.log.section(f"Copying files from {anchorkeypath}"):
    ...
```

It writes `>>>--- text` before the body and `<<<--- text` after it, at
VERBOSE by default. Pass the level as the second argument, for example
`self.log.section("Forming var from spec", 'detailed')`. The closing line is
written however the body ends, and reads `<<<--- text: raised ValueError`
when it raises. A hand-written END line is skipped by an exception, and a log
where a step opens and never closes cannot be read for where it failed.

For a `log=` argument, which may be any object with the level methods, use
`log_section(log, text, level)` from `dbx.journals` instead.
`Datalog.section` calls it.

`section` lines are ordinary lines at their level, under the view's name. The
build of a subtree has its own banners, `Datalog.subtree`, at the TREE level.
Use those only for that.

## Run dbx commands through `dbx.exec` or `dbx.pprint`

Run a dbx command through `dbx.exec` or `dbx.pprint`, from the shell or from
Python, not by calling the blocks directly. `exec` writes one row to the exec
journal for each command, when the command finishes or fails. The row holds
the command's text, its trailing `# comment`, and the build session every
block it builds writes under. `pprint` runs through `exec` and also prints
the result. A build called directly from Python leaves no row, so the
journal cannot say what produced the block or why.

```bash
dbx.pprint "b = pkg.Block(spec={'x': 1}); b.build(); b.read()  # why it was run"
```

```python
dbx.exec("b = pkg.Block(spec={'x': 1}); b.build()  # why it was run")
dbx.pprint("pkg.Block(spec={'x': 1}).read()")
```

Add `--capture-output` (or `capture_output=True`) to keep what the command
printed, too. `dbx.eval` only resolves a `@`, `$` or `#` term and writes
nothing to the journal, so run a term inside `exec` or `pprint` when the run
has to be recorded.
