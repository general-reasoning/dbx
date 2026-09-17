---
name: running-tests
description: How to run this repo's test suite - the conda env that has torch, lightning and tensorboard, and why importorskip must not be used to paper over a missing one. Load before running pytest here, or when a test module reports as skipped rather than passed.
---

# Running the tests

The suite needs the **`dbx` conda env**. The system python has none of the
optional dependencies, so `python -m pytest` outside this env silently skips
the modules that matter.

```bash
DBX_USE_WORK_REPO=False DBX_DIRTY_REPO_OK=True \
    /home/dmitry/miniconda3/envs/dbx/bin/python -m pytest -q             # linux
DBX_USE_WORK_REPO=False DBX_DIRTY_REPO_OK=True \
    /opt/homebrew/Caskroom/miniconda/base/envs/dbx/bin/python -m pytest -q   # macOS
```

Call the interpreter by absolute path, and check which of the two exists here
before using one. `conda activate dbx` does not survive between tool calls —
each Bash invocation is a fresh shell — and `conda` on its own fails out of
the shell snapshot on this machine.

**Both environment variables, every time.** The same two apply to soundworld's
suite.

`DBX_USE_WORK_REPO=False` is the important one. `~/.bashrc` exports it as
`True`, which makes `gitwrkreposetup` clone dbx and the project at **HEAD**
into /tmp and prepend both to `sys.path` — so a run tests the last commit and
not the working tree. That is right for a reproducible experiment and wrong
for a test run: the change under your hands is invisible, the suite passes,
and it passed on code you did not write. The tell is an entrypoint rejecting a
parameter that is plainly in its signature, or a test that cannot fail no
matter what you break.

`DBX_DIRTY_REPO_OK=True` silences the guard that refuses to clone a dirty
repo. With the work repo off there is no clone to make, so it is belt and
braces — harmless, and it keeps the command working whichever way the first
variable is set.

A full run takes **7–8 minutes**. A single module is seconds; scope with a
path (`... -m pytest tests/test_stills.py -q`) while iterating and run the
whole suite once at the end.

`-p no:randomly` makes a failure reproducible while you work on it.

## Skipped is not passed

`pytest.importorskip` is right for a genuinely optional dependency —
`mosaicml-streaming`, `ray` — and wrong for one this package declares. The
whole of `test_stills.py` went unrun for a long time behind
`importorskip('lightning')` against an env that predated the `lightning`
extra: the suite reported green, and 15 real failures were waiting underneath.

So: when a test module reports as **skipped**, find out why before believing
the run. If the missing package is in `pyproject.toml`, the env is stale, not
the test — install it rather than skipping around it.

## Keeping the env current

`dbx.yml` deliberately declares no packages of its own. It installs
`-e .[all]`, so `pyproject.toml` is the single source of truth and the comment
in `dbx.yml` says why adding anything there is a mistake. To pick up a newly
declared extra:

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/dbx/bin/python -m pip install -e '.[all]'
```

`tensorboard` is needed by `Still`'s `TensorBoardLogger` at run time, and
lightning does not declare it -- it raises "Neither `tensorboard` nor
`tensorboardX` is available" from the Trainer instead. It is now declared in
the `lightning` extra, so a reinstall picks it up; an env that predates that
needs the reinstall above rather than a one-off `pip install tensorboard`.
