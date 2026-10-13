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
DBX_DATALAKE=ssh://host/path/to/datalake hyp --pipejournal
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
  - constructed entry blocks, from `dbx.pipejournal(construct_only=True)`:
    picking one opens that block's own journal over the pipeline's -- below
    its header, which stays in view -- its title a menu of the pipeline's
    other constructed blocks, and beside it, *session blocks*: every block
    journaled under the session of the entry shown, from the index a
    `dbx.exec` command keeps (`DataJournal.session_entries`), to show in its
    place. A block built outside `dbx.exec` has no index to list.
  - session blocks: every block journaled under the pipeline's own session,
    as the entry blocks' windows list theirs: picking one opens its journal,
    as a constructed entry block's opens
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
  - *data*, *labels*, *rows*, *tablets* and *workers*, where the block's
    entry records `slices`: the columns its `DATASLICE(...)` declarations
    record -- a dict column key by key, as declared -- one as the data, and
    another, or none, as the labels. *hyp* samples them in the pinned worker
    on the datalake's host: of a table, *tablets* of its tablets, each read
    whole -- its rows together, as its shards hold them -- in one of *workers*
    processes forked there, *rows* of each kept. Auto is 64 tablets and 4,000
    rows split among them. *tablets* and *workers* are dimmed while the
    table's tablets are counted there -- begun as the block's window opens --
    and gone for a table of one. The sample is laid out with hyperbolic UMAP here,
    on a disk in a window of its own, colored by label. A slice that declares
    no columns -- a bare `DATASLICE` -- cannot be viewed.
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

## A directory's documents

```bash
hyp --docs                          # the current directory
hyp --docs --dir ssh://host/notes   # or any directory, local or an fsspec url
```

Every `.md`, `.markdown` and `.pdf` under the directory -- hidden
directories, `node_modules`, `__pycache__`, `site-packages`, `venv`, `env`,
`build` and `dist` aside, and at most 5,000 -- is a point: embedded by its name
and the start of its text (a pdf's first 30 pages), labeled by its name, its
modification time beneath. A document whose time and size are unchanged is
not read again. The corpus is kept under `~/.hyp/docs/<hash of the
directory>`: nothing is written to the directory scanned. A document removed
from it stays on the disk, for now, until `--reset`.

- **left click** opens a document in the center, filling the window;
- **right click** places it instead: a quarter (top left, top right, bottom
  left, bottom right), a half (left, right, top, bottom), or the center. A
  window placed replaces the windows it would overlap.

A markdown is rendered -- headings, lists, quotes, code, tables, links -- and
its ` ```mermaid ` diagrams are drawn by the mermaid CLI, `mmdc` (or the one
`HYP_MMDC` names), on this machine; without it a diagram is shown as its
source. `npm install -g @mermaid-js/mermaid-cli` installs it. A pdf is shown
page under page, as wide as its window at 100%, rendered by pdfium.

In a document's window: the wheel, or the arrows and Page Up/Down, scroll; a
sideways swipe, or Shift and the wheel, scroll across; a drag pans; + and -
zoom, 0 back to 100%; Home and End go to its ends; × or Esc closes it. The
keys go to the window under the mouse, else to the one last clicked.

**Layers.** The disk, the placed windows ("tiles") and the center window are
three layers. What is under the mouse in the front-most of them gets the
clicks and the wheel -- the disk, inside its circle -- so the disk can be
navigated with windows open: bring it to the front with the **layers** menu
(top left), which brings any layer to the front or sends it to the back, or
with Tab, which brings the back layer to the front. The disk is see-through
over the layers behind it. Opening a window brings its layer to the front.

## Controls

| Input | Effect |
|---|---|
| left click | slide the clicked point to the center along a geodesic; time ∝ distance |
| left click on a pipeline (`--pipejournal`) | its window, as a right click opens a point otherwise |
| right click on a pipeline (`--pipejournal`) | its constructed entry blocks, in a menu at the mouse: picking one opens that block's window directly |
| left drag | move the plane rigidly (a hyperbolic translation); the grabbed point stays under the cursor |
| scroll | hyperbolic zoom: a point at distance *d* is drawn at *k·d* |
| right click on a point | its stored text |
| right click elsewhere | type a text; Enter embeds, summarizes, stores and places it |
| hover | the point's label, in full, and geodesics to the 5 nearest neighbors in embedding space |
| Home (fn+← on a Mac), or double click off the disk | back to the original view: the corpus center, zoom 1 -- on a slice disk too |
| (lower left) | a spinning wheel while data are being loaded or prepared; an error, in red, for 15 s or until the next read -- its full traceback in the terminal |
| points (top right) | every point, by its label and second line in full, alphabetically (the first 2,000): picking one slides it to the center |
| settings (top right) | names and versions (`--pipejournal`) and datetimes, each on and off: under labels and in tooltips; line numbers, in text windows (off by default); title colors -- a point's window titled in the point's color, and its entry blocks' windows in the same |
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
hyp --pipejournal --ece ckpt # --extra-construction-events: events that also make an entry block count as
                             # constructed, each alone or followed by = or : -- here, a still that checkpointed
hyp --docs [--dir DIR]       # the markdowns and pdfs under DIR, local or an fsspec url (HypDocs; default: the current directory)
hyp --n-workers N            # threads reading the datalake (default 16)
hyp --remote-python PATH     # the python on the datalake's host whose Ray read's and hyp's trueinst() run in,
                             # with dbx, ray (the version here) and the project; a path there, quoted so ~
                             # expands there (default $HYP_REMOTE_PYTHON, else the python each entry records
                             # its build ran in, else -- for entries older than that -- python3)
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
