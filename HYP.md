# hyp

`hyp` is an app, shipped with dbx as a package of its own: a text space on
the Poincaré disk. Texts are embedded (nomic-embed-text, through Ollama, by
default), laid out with UMAP's hyperbolic embedding
(`output_metric="hyperboloid"`), and drawn in a pygame window navigated
hyperbolically. Pointed at a datalake, its texts are the datalake's pipelines.

Nothing in dbx imports `hyp`; `hyp` imports dbx only to read a datalake.

## Setup

`hyp` shares dbx's env: `dbx.yml` installs `-e .[all]`, and `[all]` includes
the `[hyp]` extra. Into another env, `pip install -e '.[hyp]'`.

```bash
brew install ollama            # hyp starts `ollama serve` itself if it is not running
hyp                            # or: python -m hyp
```

On first run it pulls `nomic-embed-text` (embeddings) and `deepseek-r1:1.5b`
(one-line summaries) through Ollama. The seed corpus is kept under
`~/.hyp/seeds`; a datalake's pipelines, in the datalake itself.

## A datalake's pipelines

```bash
DBX_DATALAKE=ssh://wheelbarrow/home/t-9dkarp/datalake hyp --pipejournal
```

runs `HypDbxPipelines`, the specialization of the app over a datalake's
pipelines. The datalake is dbx's own -- `DBX_DATALAKE`, then `DBX_ROOT`,
`DBX_URL` -- as every block resolves it; hyp names none. Its corpus --
embeddings, layout, points -- is kept in the datalake, under
`{DBX_DATALAKE}/.hyp`, and `--reset` clears it there. `dbx.pipejournal()` reads the pipeline journal and aggregates it --
one row per pipeline, every run of it, with every entrypoint it was created
to -- and each row is a point:

- embedded from the pipeline's fqcn (`anchor`), spec, and signature (or type)
- labeled with the last component of its `tag`, and below it its `VAR.name`,
  where its spec has one, `v{version}`, where it has one, and its datetime
- on right click, its row's columns and its side files' contents -- spec, dfn
  and kwargs as the dicts they hold, `topics` and `paths` one key per line --
  under a title -- the tag, `VAR.name` where its spec has one, and the
  datetime -- that has the top row to itself, with a row of menus beneath:
  - the title itself: every journal entry with that tag, by datetime and
    `VAR.name`, to switch to
  - constructed entrypoints, from `dbx.pipejournal(construct_only=True)`:
    picking one opens that block's own journal over the pipeline's -- below
    its header, which stays in view -- its title a menu of the pipeline's
    other constructed blocks, and beside it, *session blocks*: every block
    journaled under the session of the entry shown, from the index a
    `dbx.exec` command keeps (`DataJournal.session_entries`), to show in its
    place. A block built outside `dbx.exec` has no index to list.
  - *read* and *ls*, beside them, list the shown block's recorded topics.
    *ls* lists what is at a topic's recorded path, over ssh, with nothing
    instantiated. *read* calls the entry's `trueinst()` here, against a Ray
    head that `dbx.SshRay` starts on the datalake's host for the session --
    run by `--remote-python`, reached through an ssh tunnel, and stopped when
    hyp exits -- so the block is a proxy to an instance pinned to the entry's
    own revision of dbx and the project, where its data are. The topic is
    read there and its repr shown. The first read of a block takes about a
    minute; the block is kept, and reading it again takes seconds. Ray here
    must be the version Ray there is.
- in either window, an `output` button, where the entry shown had its output
  captured (dbx records the capture as `log`): it opens the process's
  captured stdout and stderr -- its last 1 MB, escape codes dropped, a
  progress bar as last drawn. A capture is uploaded when its command ends,
  so one still running has none to show yet.

The menus are read in the background once the points are up. Each launch adds
new pipelines and re-embeds the ones whose spec changed.

An `ssh://` datalake is read through fsspec's sshfs, on asyncssh, which applies
`~/.ssh/config`: `ssh://host/path` names the host as `ssh host` does, user and
all.

## Controls

| Input | Effect |
|---|---|
| left click | slide the clicked point to the center along a geodesic; time ∝ distance |
| left click on a pipeline (`--pipejournal`) | its window, as a right click opens a point otherwise |
| right click on a pipeline (`--pipejournal`) | its constructed entrypoints, in a menu at the mouse: picking one opens that block's window directly |
| left drag | move the plane rigidly (a hyperbolic translation); the grabbed point stays under the cursor |
| scroll | hyperbolic zoom: a point at distance *d* is drawn at *k·d* |
| right click on a point | its stored text |
| right click elsewhere | type a text; Enter embeds, summarizes, stores and places it |
| hover | the point's label, in full, and geodesics to the 5 nearest neighbors in embedding space |
| Home, or double click off the disk | back to the original view: the corpus center, zoom 1 |
| (lower left) | a spinning wheel while data are being loaded or prepared |
| settings (top right) | names and versions (`--pipejournal`) and datetimes, each on and off: under labels and in tooltips; line numbers, in text windows (off by default) |
| in a text window | scroll / Up Down / PgUp PgDn move; Left / Right step through its title menu; × or Esc closes it |
| in a text window: the triangle by a line | a line whose next line is indented deeper folds the deeper lines after it into `… N lines`, and unfolds them. Spec, dfn and kwargs are shown one key per line, a level per four spaces, so they fold by nesting |
| in a text window: *fold* (top right), or 1–9, 0 | fold every such line at that many levels of indentation or deeper; *unfold all*, or 0, unfolds them all |
| in a text window: drag, Cmd+A, Cmd+C | select text, select all; copy the selection -- or, with none, all of it |

## Options

```
hyp --data DIR               # the corpus directory, local or an fsspec url
                             # (default: ~/.hyp/seeds; with --pipejournal, {DBX_DATALAKE}/.hyp)
hyp --reset                  # discard the corpus and start over
hyp --pipejournal            # the pipelines of DBX_DATALAKE (HypDbxPipelines)
hyp --n-workers N            # threads reading the datalake (default 16)
hyp --remote-python PATH     # the python on the datalake's host whose Ray read's trueinst() runs in
                             # (default ~/miniconda3/envs/autopath/bin/python)
hyp --embedder deepseek      # DeepSeek-R1-Distill-Qwen-1.5B's pooled hidden states,
                             # through transformers: `pip install -e '.[hyp-deepseek]'`
hyp --embed-model M          # another Ollama embedding model
hyp --summary-model M        # another Ollama model for the summaries
```

Ollama will not embed with a generative model such as DeepSeek-R1, hence the
transformers path for `--embedder deepseek`. Switching embedders re-embeds the
stored corpus.

## Specializing

`hyp.app.Hyp` is the app; a specialization overrides its hooks, and only
those: `__source__` and `__default_data__` (where the corpus comes from),
`__open_point__` (a right click on a point), and `__point_label__`,
`__point_datetime__`, `__point_tooltip__` (how a point is labeled).
`hyp.dbxpipelines.HypDbxPipelines` is the example.

## Tests

`tests/test_hyp_*.py`, run with the rest of the suite (see Running Tests in the
README).
