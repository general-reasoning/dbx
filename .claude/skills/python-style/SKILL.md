---
name: python-style
description: The Python style rules in this repo that are not about class layout - imports go at module level, except to break a circular import or to defer a heavy or questionable dependency like torch or ray, and strings are formatted with f-strings, not % or .format(), except for a named template filled elsewhere. Load when writing or reviewing any Python in dbx or hyp, adding an import, or formatting a string. For member order and private naming, load class-layout.
---

# Python style

Two rules, and the few exceptions each one allows. Member order and naming
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
