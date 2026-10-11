---
name: committing
description: How to commit in this repo - directly on main, never on a local feature branch, with main brought up to date with origin/main first. Load before committing, creating a branch, merging, or pulling.
---

# Committing

**Commit directly on `main`. Do not create local feature branches** -- not to
"keep main clean", not because general guidance says to branch before
committing on the default branch. This repo's rule overrides that.

## Why

A local branch only puts off integration. Work committed on one has to be
merged into `main` later, and by then `origin/main` has moved on. On
2026-10-11 two days of work on a branch met a commit that had landed on
`origin/main` meanwhile, renaming `Datajournal` to `DataJournal` across the
code and tests, and the merge conflicted in eleven files, most of them import
lines both sides had renamed. Found when the work began, that commit would
have cost one small merge.

## How

1. **Before starting anything sizeable**, bring `main` up to date:

   ```bash
   git fetch origin
   git log --oneline main..origin/main      # what has landed meanwhile
   git merge origin/main                    # on main, resolve, test
   ```

   A rename, a module move, or any change touching many files is sizeable: it
   is the kind most likely to collide with someone else's.

2. **Commit on `main`**, when asked to commit, in commits that each say what
   they do. End the message with the attribution line the session gives.

3. **If `main` has diverged** from `origin/main`, merge `origin/main` into
   `main`, resolve the conflicts, run the suite (in parallel -- see
   `running-tests`), and commit the merge. Never branch to work around it.

4. **Push only when asked.** Committing is local; pushing publishes.
