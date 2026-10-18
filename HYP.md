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
hyp                            # or: dbx.hyp, or python -m hyp
```

On first run it pulls `nomic-embed-text` (embeddings) and `deepseek-r1:1.5b`
(one-line summaries) through Ollama. The seed corpus is kept under
`~/.hyp/seeds`; a datalake's pipelines, in the datalake itself.

## PIPELINES

```bash
DBX_DATALAKE=ssh://host/path/to/datalake hyp --journal     # or: --journal pipelines
```

runs `HypDbxJournal`, the specialization of the app over a datalake's
pipelines. The datalake is dbx's own -- `DBX_DATALAKE`, then the deprecated
`DBX_ROOT` and `DBX_URL` -- as every block resolves it; hyp names none. Its corpus --
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

### Pointing hyp at a remote datalake

Put the host and the path into `DBX_DATALAKE` as one `ssh://` URL, and leave
`DBX_STORAGE_OPTIONS` unset:

```bash
export DBX_DATALAKE=ssh://wheelbarrow/home/t-9dkarp/datalake
unset DBX_STORAGE_OPTIONS
hyp --journal
```

`wheelbarrow` is a `Host` entry in `~/.ssh/config`. Its `HostName`, `User`,
`IdentityFile` and `ProxyJump` apply as they do to `ssh wheelbarrow`, so if
`ssh wheelbarrow` logs in, hyp can read the datalake. The path is the
datalake's directory on that host, the one holding `.journal`.

The URL is for reaching the datalake from another machine. On the datalake's
own host, set `DBX_DATALAKE` to the plain path:
`export DBX_DATALAKE=/home/t-9dkarp/datalake`. With the `ssh://` URL there,
hyp logs into its own host over ssh, and fails with `PermissionDenied` unless
that host accepts a key from itself.

`DBX_DATALAKE` wins over `DBX_ROOT` and `DBX_URL`, its names from before the
rename, deprecated and still read. So a plain local path left
in `DBX_ROOT` (`/home/...`) does not matter while `DBX_DATALAKE` is set. If
`DBX_DATALAKE` is unset, hyp falls back to that path and reads it as a local
directory, on this machine.

`DBX_STORAGE_OPTIONS` (`key=val;key=val`) is passed to fsspec with every
datalake path. An ssh datalake needs none of it, because the URL and
`~/.ssh/config` say everything. If it holds a `protocol`, that protocol
overrides the path's.

When the corpus cannot be opened, hyp says which one, and where its path came
from, and stops:

```
hyp: cannot open the corpus /home/t-9dkarp/datalake/.hyp (from DBX_ROOT, deprecated: set DBX_DATALAKE instead; read as a local path):
     OSError: [Errno 45] Operation not supported: '/home/t-9dkarp'
```

Here a path meant for the datalake's host is being read on this machine:
set `DBX_DATALAKE` to its `ssh://` URL. If `ssh wheelbarrow` works but hyp
fails with `OSError: [Errno 65] No route to host`, the failing connection
is the one to the `ProxyJump` host. macOS may be refusing Python access to
local-network addresses: allow the terminal app under System Settings →
Privacy & Security → Local Network, then restart it.

To see what a shell will use: `echo "$DBX_DATALAKE | $DBX_ROOT | $DBX_URL |
$DBX_STORAGE_OPTIONS"`.

### How a pipeline is embedded

A pipeline's embedding text has its label -- the last component of its `tag`
-- on the first line by itself. The anchor, the spec and the signature (or
type) follow it (`dbxpipelines.entry`). `HypDbxJournal.__embedder__` wraps
the embedder `--embedder` names in a `LabelFirstEmbedder`, and that wrapper
splits each text at its first line.

- **The label** becomes a unit vector of its character trigrams: each one is
  hashed (crc32) to one of 256 dimensions and given a sign. One label always
  gives the same vector. Labels that share words, such as `gigapath_probe` and
  `gigapath_probe_partition`, share trigrams and lie near each other.
- **The rest** is embedded by the base embedder (nomic-embed-text by default)
  and normalized to a unit vector.

The point's vector is the label's vector, scaled by a weight *w* (3 by
default), followed by the rest's vector. UMAP lays the points out under the
cosine metric. Take two pipelines with the same label. The label parts give
a dot product of *w²* and the rest parts give at least −1, while each vector
has a squared norm of *w² + 1*. So their cosine similarity is at least
(*w²* − 1)/(*w²* + 1), which is 0.8 at *w* = 3, however much their specs
differ. Pipelines with one label therefore form one cluster, and their
specs, signatures and `VAR.name` (a key of the spec) place them within it.
Pipelines with different labels are held apart by up to the full weight of
their label vectors.

The embedder's name, `label-first(3):<base>`, is stored with the corpus.
Changing the weight or the base embedder therefore re-embeds the corpus, as
switching embedders does.

## DATABLOCKS

```bash
DBX_DATALAKE=ssh://host/path/to/datalake hyp --journal dbx.dataparts.DatatableCorePart
DBX_DATALAKE=ssh://host/path/to/datalake hyp --journal pkg.Model --filter 'tag=TRAIN, event=build:end|note'
```

shows one point per block of the anchor, from its block journal -- each
block's newest entry, embedded and labeled as a pipeline is; pipeline
entries, filed in the pipeline journal, are not among them. Its corpus is
kept under `{DBX_DATALAKE}/.hyp/anchors/<anchor>`.

**Switching.** The **journal** menu in the top bar lists the pipelines and
every anchor of the datalake (`dbx.anchors()`); picking one shows it in
place, read, embedded and laid out in the background, each in its own corpus.
With an anchor shown, the **filter** field beside it takes `key=pattern,
key=pattern, ...` -- each *key* a journal column, each *pattern* matched
against it, as `DataJournal.read`'s filters are (a comma splits only before
another `key=`, so a pattern may hold one; `|` alternates) -- and Enter
applies it; Esc puts it back. `--filter` gives it at the start. Blocks a
filter leaves out keep their embeddings, so widening it again embeds
nothing; those shown are laid out by themselves.

- **left click** opens the block's window -- the one a constructed entry
  block opens in under a pipeline: its journal's events and newest entry,
  with *session blocks*, *read*, *ls*, the slice menus and *output* -- its
  title a menu of the anchor's blocks, to step through;
- **right click** lists the blocks of its `dbx.exec` session, at the mouse,
  each opening its window.

## A directory's documents

```bash
hyp --markdowns                          # the current directory
hyp --markdowns --dir ssh://host/notes   # or any directory, local or an fsspec url
```

Every `.md`, `.markdown` and `.pdf` under the directory -- hidden
directories, `node_modules`, `__pycache__`, `site-packages`, `venv`, `env`,
`build` and `dist` aside, and at most 5,000 -- is a point: embedded by its name
and the start of its text (a pdf's first 30 pages), labeled by its name, its
modification time beneath. A document whose time and size are unchanged is
not read again. The corpus is kept under `~/.hyp/docs/<hash of the
directory>`. Nothing is written to the directory scanned, except the Claude
sessions saved there (below). A document removed from it leaves the disk at
the next start or reload. A markdown's YAML front matter is not drawn.

- **left click** opens a document in the center, filling the window;
- **right click** places it instead, picked from a 3×3 grid of little
  windows, each showing the part of the window it takes -- the quarters in
  its corners, the halves at its edges, the center in its middle; the one
  under the mouse is named beneath. A window placed replaces the windows it
  would overlap. A right click on a document in the **points** menu offers
  the same.

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
with Tab, which brings the back layer to the front -- or by clicking it
wherever it shows. The disk is see-through
over the layers behind it. Opening a window brings its layer to the front.

### Claude sessions

```bash
hyp --markdowns                          # sessions saved in the current directory
hyp --markdowns --claude-dir claude      # or in ./claude, under it
```

The **Claude** button (top right) opens a conversation with Claude in a
window a third of the window's area, in its middle (`hyp/claude.py`). The
conversation is above and what is being typed below: Enter sends,
Shift+Enter starts a new line, Cmd+V pastes. The reply streams in as it is
written. PageUp/PageDown, Home/End and the wheel scroll the conversation;
Esc or × closes the window. A turn that fails says why beneath, and puts
what was sent back in the box, to send again.

**Claude Code, under your own login.** Each turn runs the Claude Code CLI
once -- `claude -p`, streaming -- so a Claude subscription covers it; no API
key or credits are needed. The CLI is `$HYP_CLAUDE`, else `claude` on the
PATH, else the one VS Code's Claude Code extension carries. It must have
been logged in once (`claude`, then `/login`). Claude Code keeps the
conversation itself: the first turn starts a session with an id of hyp's
(`--session-id`), and every turn after resumes it (`--resume`). Claude runs
in `--dir` with read-only tools -- `Read`, `Glob`, `Grep`, and no others,
none asked about -- so it can read, list and search the documents on the
disk, and change nothing. Each tool it uses is noted in the reply. A
`--dir` that is a url is no directory Claude Code can work in: there it
runs in your home directory, with no tools, and sees only what is pasted
in.

**Saved and embedded at every turn.** As each reply comes in whole, the
session is saved into the directory scanned (`--dir`), or into
`--claude-dir`: a directory relative to it that the scan reaches, so not
absolute, not above `--dir`, and not a hidden or skipped one. It is one
markdown, `claude-session-<started>.md`, a `## You` / `## Claude` section a
turn. Its front matter says what it is: `hyp: claude-session`, the session's
id, the model, and the first line as its `title`. The disk is brought up to
date with it at once, in the background, the window usable meanwhile: the
directory is read again, the session embedded, and the layout nudged from
where it was rather than laid out afresh. So the session's point,
`Claude · <title>`, moves among the documents as the conversation goes, and
its arcs and arrows relate it to them as they do any document. Closing the
window saves nothing more, unless a turn is not yet saved.

**Embedded whole.** A document is embedded by its name and its first 4,000
characters; a session by all of its transcript, up to 256,000 characters.
Hence `ChunkedEmbedder` (`hyp/docs.py`): a text longer than that is
embedded a 4,000-character piece at a time -- cut at paragraph breaks where
it can be, the name before each piece -- and the pieces' unit vectors
averaged, so a long session lies where all of it points, not where its
first turns do. A text of one piece gets the embedder's own vector, as
before. No markdown's front matter is embedded, a session's or any other's.

**Reopened by its point.** A left click on a session's point opens it
again: its transcript is shown, and the next turn resumes it by its id. Closed,
it is saved over the same file, the new turns after the old. A right click
places its transcript, as any document's. Claude Code keeps a session on the
machine, and for the directory, it ran in. A session saved on one machine
and reopened on another is shown, but Claude there has no record of it: the
window says so.

## Arrows

Each arc a hover draws is an arrow, from the hovered point to a neighbor:
the difference of their embeddings, `ê_there − ê_here`, each unit length
first. Hold Shift over the point, and each arrow is labeled with a short
line saying what moving along it means -- `Moves from test-split
evaluation to the 602020 training split`. An embedding has no decoder: no
text comes back out of a vector. So the line is made in two steps
(`hyp/arrows.py`, `Engine.arrow_notes`).

1. **The arrow is read against the corpus's phrases.** Every stored point's
   embed text is cut into phrases: a line each, a line longer than 120
   characters at its sentences, quotes, brackets and backslashes stripped.
   A phrase in every text is dropped, as it points nowhere, and the 4,000
   in the most texts are kept. Each is embedded with the corpus's own
   embedder -- through `LabelFirstEmbedder`, for a journal -- so it lies in
   the points' space. Every embedding shares a large common direction, so
   each phrase's unit vector is taken less the mean of them all, and the
   arrow's cosine with each is its score. The 10 highest are where moving
   there leads, the 10 lowest what it leaves behind. The phrases are
   embedded the first time Shift is held over a corpus, and kept in the
   store (`phrases.json`, `phrases.npy`) with the embedder's name.
2. **Claude says what that means.** The phrases and their scores, with the
   two points' labels, are given to Claude Code (`claude -p`, under your
   login, with no tools, as the Claude sessions run), which answers in one
   line of at most 12 words an arrow. All of a hover's arrows not said
   before go in one question, about 5 seconds. "…" stands on each arc until
   the lines come.

Each line is kept in the store (`notes.json`), by the two points' embed
texts, so an arrow is said once; a point's text changing asks again, and a
change of embedder discards them all. Two neighbors with one embedding -- a
pipeline run again -- are not asked about: their arc says `the same
embedding`. A reading that fails -- no Claude Code, not logged in -- says
why on the first arc, and is tried again after 30 seconds of Shift.

Claude is given the phrases, not the texts: it says what the embedding
weighs, which the phrases are the embedding's own reading of. Giving it
the two texts as well would let it say more of what they are, at the cost
of long prompts -- specs, transcripts -- and of describing the texts rather
than the vector.

## Controls

| Input | Effect |
|---|---|
| left click | slide the clicked point to the center along a geodesic; time ∝ distance |
| reload (top left) | read the corpus's source again, re-embed what changed, and lay it out afresh |
| left click on a pipeline (`--journal`) | its window, as a right click opens a point otherwise |
| right click on a pipeline (`--journal`) | its constructed entry blocks, in a menu at the mouse: picking one opens that block's window directly |
| left drag | move the plane rigidly (a hyperbolic translation); the grabbed point stays under the cursor |
| scroll | hyperbolic zoom: a point at distance *d* is drawn at *k·d* |
| right click on a point | its stored text |
| right click elsewhere | type a text; Enter embeds, summarizes, stores and places it |
| hover | the point's label, in full, and geodesics to the 5 nearest neighbors in embedding space -- by cosine over the whole vectors, so an arc may reach a point far off on the disk |
| Shift, while hovering | each of those arcs labeled with a line, by Claude, saying what moving along it means. See [Arrows](#arrows) |
| Home (fn+← on a Mac), or double click off the disk | back to the original view: the corpus center, zoom 1 -- on a slice disk too |
| (lower left) | a spinning wheel while data are being loaded or prepared; an error, in red, for 15 s or until the next read -- its full traceback in the terminal |
| while a corpus loads | the window is dimmed and takes no input. A card in its middle shows the stage under way, with a bar, the files done out of the total, the time taken and the time left. An anchor's journal is listed first, as a stage of its own (`dbx.DataJournal.list`). Until the listing is done there is no total, so the bar slides and the card counts the files found so far, after every directory. Then the files are read (`dbx.DataJournal.read_frame`), counted against that total |
| Esc, while a corpus loads | interrupt it at its next step and drop anything queued behind it. A reload that is interrupted leaves the disk as it was; a switch of journal leaves it empty until the next reload. Laying out the disk (UMAP) cannot be cut short, so an Esc during it waits for it to finish |
| points (top right) | every point, by its label and second line in full, alphabetically (the first 2,000): picking one slides it to the center; a right click on one does what a right click on its point does |
| settings (top right) | names and versions (`--journal`) and datetimes, each on and off: under labels and in tooltips; line numbers, in text windows (off by default); title colors -- a point's window titled in the point's color, and its entry blocks' windows in the same |
| in a text window | scroll / Up Down / PgUp PgDn move; Left / Right step through its title menu; × or Esc closes it |
| in a text window: the triangle by a line | a line whose next line is indented deeper folds the deeper lines after it into `… N lines`, and unfolds them. Spec, dfn and kwargs are shown one key per line, a level per four spaces, so they fold by nesting |
| in a text window: *fold* (top right), or 1–9, 0 | fold every such line at that many levels of indentation or deeper; *unfold all*, or 0, unfolds them all |
| in a text window: drag, Cmd+A, Cmd+C | select text, select all; copy the selection -- or, with none, all of it |

## Options

```
hyp --data DIR               # the corpus directory, local or an fsspec url
                             # (default: ~/.hyp/seeds; with --journal, {DBX_DATALAKE}/.hyp, or .hyp/anchors/ANCHOR)
hyp --reset                  # discard the corpus and start over
hyp --journal                # the pipelines of DBX_DATALAKE (HypDbxJournal); also: --journal pipelines
hyp --journal ANCHOR         # the blocks of ANCHOR, from its block journal; the journal menu switches either
hyp --journal ANCHOR --filter 'key=pattern, ...'   # its blocks filtered; the filter field changes it
hyp --journal --ece ckpt     # --extra-construction-events: events that also make an entry block count as
                             # constructed, each alone or followed by = or : -- here, a still that checkpointed
hyp --markdowns [--dir DIR]  # the markdowns and pdfs under DIR, local or an fsspec url (HypDocs; default: the current directory)
hyp --markdowns --claude-dir REL   # Claude sessions saved under DIR/REL, not DIR itself (REL relative, and scanned)
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
those: `__source__`, `__default_data__` and `__data_origin__` (where the
corpus comes from, and what to name when it cannot be opened),
`__open_point__` (a right click on a point), and `__point_label__`,
`__point_datetime__`, `__point_tooltip__` (how a point is labeled).
`hyp.dbxpipelines.HypDbxJournal` is the example.

## Tests

`tests/test_hyp_*.py`, run with the rest of the suite (see Running Tests in the
README).
