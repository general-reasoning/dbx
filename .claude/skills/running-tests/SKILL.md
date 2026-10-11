---
name: running-tests
description: How to run this repo's test suite - always in parallel (pytest -n 16), in the conda env that has torch, lightning and tensorboard, and why importorskip must not be used to paper over a missing one. Load before running pytest here, or when a test module reports as skipped rather than passed.
---

# Running the tests

The suite needs the **`dbx` conda env**. The system python has none of the
optional dependencies, so `python -m pytest` outside this env silently skips
the modules that matter.

```bash
DBX_USE_WORK_REPO=False DBX_DIRTY_REPO_OK=True \
    ~/miniconda3/envs/dbx/bin/python -m pytest -q -n 16                          # linux
DBX_USE_WORK_REPO=False DBX_DIRTY_REPO_OK=True \
    /opt/homebrew/Caskroom/miniconda/base/envs/dbx/bin/python -m pytest -q -n 8   # macOS
```

Call the interpreter by absolute path, and check which exists here before
using one: `~/miniconda3/envs/dbx` on the linux machines (the home directory
differs from one to the next -- `/home/dmitry`, `/home/t-9dkarp`), the
Homebrew path on macOS. `conda activate dbx` does not survive between tool calls —
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

## Always in parallel

**Run the suite with `-n 16`** -- `pytest-xdist`, declared in the `dev` extra.
Serially the full suite takes about **28 minutes** on the 32-core linux
machine; with `-n 16` it takes **under 5** (2510 tests, 2026-10-11). Nothing in
it needs to run alone: every test writes under its own `tmp_path`, and
streaming's shared memory is named per process. Use about half the cores
(`nproc`): `-n auto` takes them all, and the tests that build in worker pools
of their own then compete for them.

Scope with a path (`... -m pytest tests/test_stills.py -q`) while iterating --
a single module is seconds, and needs no `-n` -- and run the whole suite, in
parallel, once at the end.

A failure seen only under `-n` is a test sharing state it should not --
a fixed path, an environment variable, a module-level cache -- and is a bug in
that test: fix it rather than dropping `-n`. To reproduce one, run its module
alone with `-p no:randomly`.

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
