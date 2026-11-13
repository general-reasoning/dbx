---
name: reaching-existing-builds
description: How a datablock reaches data that was built (or copied in) under another identity or anchor -- a class moved or renamed, a VAR or topic renamed, a build brought from another lake. The chain is specialize, then fix dbx if the miss is a generic pattern, then manual redirection; never anchoring. Load before moving or renaming a Datablock class, VAR field or topic, before copying builds between lakes, and whenever a datablock that should find old data rebuilds or reads as invalid instead.
---

# Reaching existing builds

dbx controls where builds live and how they are found until a candidate is
shipped. A datablock reaches data stored elsewhere through **redirection**:
recorded in the datablock's journal, undone with `UNSAFE_clear_redirection`. A
**specialization** is automated redirection: the class declares how its old
self differed, dbx works out the old hash, finds that build in the journal and
redirects to it.

**Never touch anchoring** to get there: no `ANCHOR` on a class, no `anchor=`
on a datablock, nothing that moves lake directories so a lookup happens to
succeed. That hides the move from the journal.

## The chain

1. **Specialize.** Declare a `Datablock.Specialization` saying how the old
   datablock differed: `spec`, `topics`, `version`, `anchor`, `redirect_vars`,
   `redirect_topics`. For a moved class that is `spec={}`, `topics=SAME`,
   `anchor=<old anchor>`. Check it against the lake before building; none of
   these write anything:
   - `dblk.specializations()`: one row per specialization, with *why* it did
     or didn't resolve;
   - `dblk.find_specialization()`: what `build()` would adopt;
   - `dblk.UNSAFE_specialize(dry_run=True, dry_validate=True, OVERRIDE=True)`:
     the proposed redirection, and whether each topic's data is there.

   `entry.try_specialize_tree()` then adopts every datablock of the tree that
   resolves, building nothing; `build_tree()` adopts them before building the
   rest. A specialization adopts a hash whose journal under the target anchor
   has an entry in `SPECIALIZATION_EVENTS` (`build:end`,
   `UNSAFE_copy_from:END`, `UNSAFE_redirect`) recording paths, for all the
   topics it names, that are still there.

2. **When it misses, read the `why` and ask whether the miss is a generic
   pattern dbx should handle.** Example (2026-11-12): builds copied in with
   `UNSAFE_copy_from` were skipped because `SPECIALIZATION_EVENTS` lacked
   `UNSAFE_copy_from:END`. Every copy would hit that, so the fix belonged in
   dbx: one event, plus tests that a copy is adopted, including across an
   anchor move. Judge each case separately. A one-off corner case does not
   justify a change to dbx. Propose the change, with a test that fails without
   it, to the owner. Don't change dbx without their go-ahead.

3. **Otherwise, redirect manually.** For a corner case that isn't worth
   automating, or a dbx fix that can't or shouldn't be made, name the topic
   paths directly:

   ```python
   dblk.UNSAFE_redirect(paths={'sr': '<lake>/<anchor>/<key>/sr'}, topics=['sr'],
                         dry_run=True, dry_validate=True, OVERRIDE=True)   # look first
   dblk.UNSAFE_redirect(paths={'sr': ...}, topics=['sr'], validate=True, OVERRIDE=True)
   ```

   `paths=` is taken as given: the journal is not consulted to find or check
   the target. It only follows any further redirection on those paths. The
   redirection is still recorded (`.redirection/paths.yaml` and an
   `UNSAFE_redirect` journal entry), so the datablock stays traceable.

## Who decides

Steps 2 and 3 are the owner's call. Bring them each case as a list: class,
hash, the journal events, the specialization's `why`, and the proposal (the
dbx change, or the `paths=` to redirect to). Never redirect on your own.

## Copies

Data from another lake (Lambda's, over `sftp://`, say) comes in with
`UNSAFE_copy_from`, never with `cp` or `rsync`. It lands in the dbx layout,
under the datablock's own anchor and key, journaled `UNSAFE_copy_from:BEGIN` and
`:END`. It is that datablock's data, reachable and specializable like a build.
