---
name: moving-data
description: All data movement between machines, clusters and lakes goes through Datablocks - data a run reads is a topic of a datablock, moved by shipping that datablock (UNSAFE_ship, out) or copying it in (UNSAFE_copy_from), and a project's wrappers only bind a cluster's URL and options (sctx's UNSAFE_ship_to_lambda, UNSAFE_copy_from_lambda); when a run needs part of a topic, the part becomes a datablock's own topic, behind a VAR whose default the earlier builds adopt by specialization; workdirs carry code and environments, never data; a standard path that fails is fixed in dbx, not worked around. Load before copying data to or from another cluster (Lambda) or lake, planning a run where the storage is not shared, or writing anything that stages, syncs, uploads or downloads files.
---

# Moving data

Data moves between machines, clusters and lakes **as datablocks**, so that every
copy is journaled, lands in the dbx layout, and can be found, validated and
specialized like a build. No ad hoc staging directories, `cp`/`rsync`/`scp` of
data, or anonymous trees uploaded beside a run.

## The paths

| direction | how | what it records |
|---|---|---|
| out, to another lake or cluster | `dblk.UNSAFE_ship({topic: target})`, per topic | `UNSAFE_ship:BEGIN/END` with the targets, on the shipped datablock |
| in, from another lake or cluster | `dblk.UNSAFE_copy_from(anchorkeypath)`, every topic, validated | `UNSAFE_copy_from:BEGIN/END`, which specializations adopt |
| code and environments | a run's `workdirs={src: target}` (packed on a slow link) | the run's `job:submit` record |

A project may wrap the first two to bind a cluster -- its URL and fsspec
options, a `with` block a `dbx.pprint` one-liner cannot hold -- and nothing
more: sctx's `SCPipeline.UNSAFE_ship_to_lambda(*dblks, topics=...)` and
`SCPipeline.UNSAFE_copy_from_lambda()`.

- **Ship to the datablock's own path** (`dblk.path(topic, local=True)` under the
  other lake), not to where a redirection here points: the other lake has no
  journal to redirect by, and a datablock is valid there once its own topics are.
- **Workdirs are for code and environments** -- venvs, checkouts, `.git`
  directories for the pinned clone -- never for data a datablock should own.
- **Bring results back with `UNSAFE_copy_from`**, and finish a build where its
  inputs are: a build that cannot complete remotely stops cleanly (an
  operational kwarg, e.g. sctx's `n_viz_workers=0` before videos the remote side
  cannot render) and is finished by an ordinary build here.

## A run needs part of a topic

Do not cut the part out by hand. Make it a **datablock's own topic**: the
datablock that already says what is needed gets a VAR that switches the topic
on, and shipping that datablock ships the part.

- The VAR's default is the old behaviour (the topic off), and a
  `Specialization(spec={field: default}, topics=<the old TOPICS>)` lets every
  earlier build adopt (see reaching-existing-builds).
- Within one filesystem the topic can be hard links into the source datablock's
  files: no extra disk, and the bytes are real when shipped.
- Example (2026-11-18): QA evals on Lambda read a few percent of a nuPlan
  representation's `images`. A one-off `stage_asked_images` hard-link tree, sent
  as a workdir, was the wrong shape: anonymous, unjournaled, a special case per
  eval. The windows already say which frames and cameras matter, so `SCWindows`
  gets the images of its windows as a topic (`image_hz`), shipped with it.

## When a standard path fails

Fix it in dbx -- a generic bug, a missing case -- with a test that fails
without the fix, and the owner's go-ahead, as reaching-existing-builds says for
specializations. Do not route around it in the project. (2026-11-18:
`UNSAFE_ship` of a directory topic to SFTP handed the directory to paramiko as
a file.)
