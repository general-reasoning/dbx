# hyp

`hyp` is an app, shipped with dbx as a package of its own: a text space on
the Poincaré disk. Texts are embedded (nomic-embed-text, through Ollama, by
default), laid out with UMAP's hyperbolic embedding
(`output_metric="hyperboloid"`), and drawn in a pygame window navigated
hyperbolically. Fewer than five texts are too few for UMAP, which scales
each point's distances by its nearest neighbor's and so draws three points
equidistant whatever they mean: they are placed by their cosine distances
instead (classical multidimensional scaling), the farthest at radius 0.3.
A character the window's font has no glyph for -- `⊙`, `∧`, `⟨⟩` in
Helvetica Neue -- is drawn from a fallback that has it (`hyp/fonts.py`:
STIX Two Math, STIX General, Apple Symbols, Menlo), not as a box. Pointed at a datalake, its texts are the datalake's pipelines.

Nothing in dbx imports `hyp`; `hyp` imports dbx only to read a datalake.

## Setup

`hyp` shares dbx's env: `dbx.yml` installs `-e .[all]`, and `[all]` includes
the `[hyp]` extra. Into another env, `pip install -e '.[hyp]'`.

```bash
brew install ollama            # hyp starts `ollama serve` itself if it is not running
hyp                            # or: dbx.hyp, or python -m hyp
```

Run from a terminal, hyp gives it back once its window is up: it goes on in
the background, in a session of its own, its output in
`~/.hyp/logs/hyp-<started>-<pid>.log` (the newest 20 kept), and is ended by
closing its window -- not by Ctrl+C there, nor by the terminal closing. What
goes wrong before the window is up is said in the terminal all the same, as
the log's end, with the run's exit status. `--fg` keeps the terminal,
as before; a hyp whose output is no terminal -- a pipe, a script -- keeps
running where it is.

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
    `dbx.exec` command keeps (`BuildJournal.session_entries`), to show in its
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
- in either window, `quote` and `cite` buttons, where the entry shown
  recorded its block's quote: each opens a bash line, to copy (Cmd+C), that
  makes the block again from that quote and prints it --
  `dbx.pprint "dbx.eval(\"\$pkg.Block(spec={...}, ...)\")"` -- or, for
  `cite`, prints its cite, the same with `.cite()` after it. The quote is
  evaluated by `dbx.eval`, as recorded: its nested blocks are written bare,
  and only `dbx.eval` reads them back as the speclines they were -- run as
  Python, they would be made into blocks, and the block made would not be
  the one quoted. The cite is made from the quote: the quote is the faithful
  form, operational kwargs and all, and cites recorded before dbx kept
  speclines in a cite name this machine's values. The window's title switches
  between the
  two; beside it, `[ ] --revision=` pins the line to the entry's recorded
  revision, `dbx.pprint --revision=<dbxsha>:<projsha> "..."`, so it runs as
  the block was built. Off, as it starts, the line runs at today's checkout.

The menus are read in the background once the points are up. Each launch adds
new pipelines and re-embeds the ones whose spec changed.

### Pointing hyp at a remote datalake

hyp reads the datalake `DBX_DATALAKE` and `DBX_STORAGE_OPTIONS` name, as dbx
does. How to set them for a datalake on another host over ssh, or on an
Anyscale workspace, and what to do when the connection fails, is in
[CLUSTERS.md](CLUSTERS.md):

```bash
export DBX_DATALAKE=ssh://wheelbarrow/home/t-9dkarp/datalake    # a Host in ~/.ssh/config
unset DBX_STORAGE_OPTIONS
hyp --journal
```

When the corpus cannot be opened, hyp says which one, and where its path came
from, and stops:

```
hyp: cannot open the corpus /home/t-9dkarp/datalake/.hyp (from DBX_ROOT, deprecated: set DBX_DATALAKE instead; read as a local path):
     OSError: [Errno 45] Operation not supported: '/home/t-9dkarp'
```

Here a path meant for the datalake's host is being read on this machine: set
`DBX_DATALAKE` to its `ssh://` URL. The other failures are listed in
[CLUSTERS.md](CLUSTERS.md#a-datalake-on-another-host-over-ssh).

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

shows one point per block of the anchor, from its build journal -- each
block's newest entry, embedded and labeled as a pipeline is; pipeline
entries, filed in the pipeline journal, are not among them. Beneath a
block's label, its full `tag`, then its datetime -- on the same line where
both fit in 48 characters, else on a third. Its corpus is
kept under `{DBX_DATALAKE}/.hyp/anchors/<anchor>`.

**Switching.** The **journal** menu in the top bar lists the pipelines and
every anchor of the datalake (`dbx.anchors()`); picking one shows it in
place, read, embedded and laid out in the background, each in its own corpus.
With an anchor shown, the **filter** field beside it takes `key=pattern,
key=pattern, ...` -- each *key* a journal column, each *pattern* matched
against it, as `BuildJournal.read`'s filters are (a comma splits only before
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
hyp --docs                               # the current directory
hyp --docs --dir ssh://host/notes        # or any directory, local or an fsspec url
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
  would overlap. Beneath the grid, **ask Claude about it** opens a new Claude
  session instead, its box holding `About <path>: ` for the question to
  follow. A right click on a document in the **points** menu offers the same;
- **right click elsewhere** on the disk offers **Claude** or **Markdown**,
  then a box to type in. Enter, for Claude, asks it as the first turn of a
  new session; for Markdown, makes it the first line of a new markdown --
  saved at once in the directory scanned as `<first line>.md`, never over a
  file there -- and opens it in the middle, the size of a Claude session's
  window, to go on writing (`hyp/notes.py`). There Enter starts a new line,
  the text edits as the box does, the wheel scrolls, Cmd+S saves, and Esc or
  × close it, saving what is not saved; each save embeds it. Its title is its
  file's name, edited in place as a Claude session's title is: kept, the file
  is renamed -- never over another, which is refused, said in the window.
  **preview**, at the title row's right, splits it: the markdown rendered
  above, mermaid diagrams too, as it is written below; the bar between them
  is dragged to share the window, and the wheel scrolls the pane under the
  mouse. (Outside
  `--docs`, the box embeds its text as a point of its own; here every point
  is a file.)

A markdown is rendered -- headings, lists, quotes, code, tables, links -- and
its ` ```mermaid ` diagrams are drawn by the mermaid CLI, `mmdc` (or the one
`HYP_MMDC` names), on this machine; without it a diagram is shown as its
source. `npm install -g @mermaid-js/mermaid-cli` installs it. A pdf is shown
page under page, as wide as its window at 100%, rendered by pdfium.

In a document's window: the wheel, or the arrows and Page Up/Down, scroll; a
sideways swipe, or Shift and the wheel, scroll across; + and - zoom, 0 back
to 100%, with Cmd or without -- in a Claude session's window, with it, as
+ and - alone are typed into its box; Home and End go to its ends; × or Esc
closes it. A right click on its title bar is the place picker for it: a
slot moves it there, closing what it would cover, and the middle expands it
to fill the window, a Claude session's too. A drag that starts
on text selects it -- the window scrolls, dragged past its edge -- Shift+click
extends the selection, a double click selects a word, Cmd+A all of it, and
Cmd+C copies it, as it reads: a paragraph's wrapped lines joined, code's lines
kept. A drag that starts anywhere else pans. A Claude session's transcript is
selected the same way, and Cmd+C copies it even while its box is typed in: to
paste it there (Cmd+V) and feed it back. The keys go to the window under the
mouse, else to the one last clicked.

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
hyp --docs                               # sessions saved in the current directory
hyp --docs --claude-dir claude           # or in ./claude, under it
```

A right click opens a conversation with Claude -- on empty disk, asking what
is typed in its box; on a document, about it (above) -- in a window a third
of the window's area, in its middle (`hyp/claude.py`). The
conversation is above and what is being typed below: Enter sends,
Shift+Enter starts a new line. The box edits as the title does (below) --
a cursor, placed by a click; a selection, by a drag, Shift with the arrows
or a click, a double click or Cmd+A; Up and Down a line -- and Cmd+Z or
Ctrl+Z undoes, a run of typing at once, Cmd+Shift+Z or Ctrl+Y redoes.
Cmd+V pastes, its lines kept, in place of what is selected; Cmd+C copies the
box's selection, or else the transcript's. The reply streams in as it is
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

**Web access** (settings, off by default) gives Claude `WebSearch` and
`WebFetch` besides, from the next turn on -- the sessions open, and those
handed to the phone from then -- and its system prompt says so; off, it is
told it cannot reach the web. The window says `web access on`, after the
account. It is kept with the corpus, in its `settings.json`, so it stays as
it was left. It is off by default because a session that both reads your
documents and fetches pages could be led -- by a page, or a document --
to put what it read into a url it fetches.

**The model, and the usage.** At the right of the box, `model: default ▾`:
a click switches it -- Claude Code's default, `opus`, `sonnet`, `haiku` --
from the next turn on, which resumes the session on that model
(`--model`). Beneath the window's title, two lines. First, the account
Claude Code runs as -- `Claude Code as <email> · <organization> · <plan>`,
from `claude auth status`, read in the background as the window opens, after
each turn and as the session is handed off -- then the session's file. That
is the account the session runs on, and the one whose claude.ai a session
handed off goes to.

**Its own login.** The Claude Code hyp runs has a configuration directory --
its login, and its record of every conversation -- of its own:
`--claude-config-dir DIR`, else `$HYP_CLAUDE_CONFIG_DIR`, else, with `--docs` and a
local `--dir`, that directory's `.claude` (made if missing), else Claude Code's
own `~/.claude`. hyp runs every Claude Code with `CLAUDE_CONFIG_DIR` set to it,
and nothing else on the machine is changed: the directory's sessions, and the
phone they are handed to, can be on another account than Claude Code
elsewhere. Log in there once, before the first session:

```bash
hyp --docs --claude-login                  # in the directory scanned, or with --dir
hyp --docs --claude-login me@example.com   # the email filled in on the login page
```

It runs `claude auth login --claudeai` with that `CLAUDE_CONFIG_DIR` -- a
claude.ai account, never the Console's API billing, which has no Remote
Control -- shows the login it made, then offers to answer Claude Code's
one-time questions for remote handoffs -- a text style, trusting the
directory, accepting Remote Control -- in a session it opens (answer them,
then `/exit`), and exits.

`ant auth login` -- the Claude Developer Platform's CLI -- logs in nothing of
Claude Code's, and Remote Control needs a claude.ai login. A session begun
under another directory's login is carried over -- its record copied in --
the first time it is resumed under this one. On macOS the login itself is in
the Keychain; what the directory holds -- transcripts, history -- travels with
it, synced where the directory is. Then the session's
usage: the model the last turn ran on, the tokens in (cached or not) and
out over every turn, what they would cost at API rates -- a subscription is
not billed them -- and the quota: what is left of the plan's 5-hour and
7-day windows and when each resets (`quota: 5 h 96% left (resets 12:40) ·
7 d 59% left (resets Fri 05:00)`), and `on overage` past them. A turn
reports the quota, so until one has it says it is known after a turn. The choice and
the totals are kept in the front matter (`choice`, `tokens_in`,
`tokens_out`, `cost_usd`), so a session reopened goes on from them.

**Handing a conversation to the cloud.** By default -- the setting **hand
off to the cloud**, kept with the corpus -- **remote** hands a conversation
to a cloud session of Claude Code's, on claude.ai: it goes on there, in the
Claude app or at claude.ai/code, the Mac free to sleep, and comes back when
it is opened here again (`claude.CloudHandoff`).

```bash
cd ~/THINK
echo dkarpeyev/hyp-cloud-think > .hyp-github     # the GitHub repository its notes are pushed to
echo .claude/ >> .gitignore                      # hyp's own Claude Code login and records: never in the repository
git init && git remote add origin git@github.com:dkarpeyev/hyp-cloud-think.git
git add -A && git commit -m notes && git push -u origin main
```

The directory scanned (`--dir`) must be a git repository with a remote at
the GitHub repository its `.hyp-github` names -- `owner/name` -- with the
Claude GitHub app installed on it, and Claude Code's own directory, where it
is inside, ignored: the window says what is missing. The cloud can be given
files, not a conversation, so a handoff commits the conversation's files --
its own, and its parents' -- pushes them, and begins a cloud session on the
repository (`claude --cloud`), its first message the conversation itself:
every turn written out as it was said, numbered, who said what -- so that
the phone shows the conversation whole, not Claude's summary of it -- and
Claude told to take it as said and reply only "Ready." (CLOUD_OPENING; a
conversation past 200,000 characters is pointed to its files instead, to
read). Its url is shown as a QR code, to scan with the
phone, and kept in the conversation's front matter (`cloud`), so that a
restart of hyp does not lose it. Opening it here takes it back:

- what is not committed is set aside (`git stash`), and Claude Code's
  `--teleport` brings the cloud session here, checking out the branch it
  pushed its changes to, if it changed any;
- a line is typed at it, so that Claude Code records the conversation here,
  and it is ended;
- the branch the notes were on is checked out again, the cloud's merged into
  it and pushed -- a conflict is left for you to merge, and said -- and what
  was set aside is put back;
- the turns made in the cloud are added to the conversation's file, and it
  goes on here from the conversation teleported. A message sent while Claude
  was still answering counts as a turn of its own.

hyp's own messages -- the cloud session's first, and the take-back's line --
begin `[hyp]`, and are no turns: neither shown nor counted. The turns before
the cloud are held by the conversation it was handed off from, and the
file says so (`start`, `earlier`): a branch from one of them forks that
conversation, and one from a later turn the conversation taken back. In
the cloud, Claude has the earlier turns as it read them, not as it said
them: it does not recall its own tool calls.

**One cloud session a branch, kept.** Cloud sessions are not taken back out
of claude.ai -- `--teleport` copies one here, and it stays there -- and no
command archives one: so a branch keeps its cloud session, after it is
taken back, to go on in again (`cloud`, with `away` while it is there, and
how far each side has got: `cloud_seen`, the cloud's turns here;
`cloud_upto`, the branch's turns the cloud has). Opened here, or handed off
again, what was said there since it was last here comes home first
(`CloudHandoff.sync`, a teleport):

- **only the cloud went on**: its turns are the branch's, and the same
  cloud session goes on;
- **only the Mac went on**: the cloud session is given the Mac's turns,
  written out as they were said, in one message -- hyp's, `[hyp] …`
  (CATCH_UP_TURNS; past 200,000 characters, where they are in the pushed
  files) -- and goes on; the phone shows it as a message of yours;
- **both went on**: the conversation forked where they last met. The
  cloud's turns come home as a branch going on from there, which keeps the
  cloud session, and the window says so; the Mac's line is given a cloud
  session of its own.

A handoff is named as the branch is: its first turn's label, then `→` and
its last turn's name (`QM: MEAS › Wigner's friend → GRW collapse`), so that
the phone's list says which branch of which conversation each is. With the
setting off, **remote** hands the conversation to the phone by Remote
Control, below.

**all to the cloud**, in the top bar, hands off every branch on the disk
at once -- every conversation's file, whole, a cloud session each, told to
read it and its parents up to where it goes on from -- so that any branch
can be gone on with on the phone; one with a cloud session of its own goes
on in it. The files are pushed once, first, and the sessions begun or gone
on in one at a time, as an engine job (its card, its bar, Esc to stop it);
branches in the cloud already, or open here, are left as they are. Every
turn's tooltip says when its branch is in the cloud. On the Mac, a branch is
still shown turn by turn, every turn a point.

**Handing a session to a phone.** A session goes on on a phone -- the Claude
app, or claude.ai/code -- through Claude Code's Remote Control, and comes back.
hyp runs a turn at a time (`claude -p`), and Remote Control needs a session
that stays, so a session handed off is resumed interactively --
`claude --resume <id>` -- in a terminal of hyp's own that no one sees
(`claude.Handoff`), and `/remote-control hyp · <title>` typed at its prompt
once its screen has settled: typed in, not passed as `--remote-control`, which
shows the phone only what comes after, so that the phone is given the
conversation whole. (Not `--fork-session`: a fork's claude.ai session showed
the phone nothing of what came before.) Its screen is
read for the session's url, and Remote Control's own consent, where it is
asked, accepted: it is what was asked for. It is saved to its markdown first, there and then. Handed off before -- or
run under another account's organization -- it goes as a fresh copy of its
conversation, under a new id: its record of a claude.ai session left out, and
what keeps its history from one (`history-suppression`; `credential_org`,
given this account's organization), each link of its chain past what is left
out mended, so that the phone is given all of it; the original is removed, in
hyp's own configuration directory only, and the markdown follows the copy.

- **remote**, at the window's title row, hands it off: its QR code covers the
  transcript, to scan, and nothing is sent from the box while it is away.
  **take back** brings it back: Claude Code there is ended, and the turns made
  on the phone -- read from Claude Code's own record of the conversation,
  `~/.claude/projects/<directory>/<id>.jsonl` (`$CLAUDE_CONFIG_DIR` where set)
  -- are added, and the session saved and embedded.
- **Every session is on remote** but the one open here, with the setting
  **sessions on remote** (on by default): each with a turn to hand off is
  handed off as its window closes -- to the cloud, or by Remote Control -- and,
  by Remote Control only, as hyp starts, one at a time (to the cloud, that
  would be a cloud session each, at every start); it is in the phone's list
  of sessions as `hyp · <title>`. **Opening** one here
  takes it back. Each on remote is a Claude Code of its own, about 250 MB. A
  session on remote goes on only while hyp runs; quit, it goes offline, and reopened later, what
  was said anywhere since it was saved -- the phone, a terminal -- comes in all
  the same.
- Claude Code asks some things once, interactively: a text style, whether to
  trust the directory. Asked in the terminal no one sees, they are not
  answered: the window says to run
  `claude --remote-control` once, in a terminal, in the directory, and answer
  them. A login for API usage billing -- the Claude Developer Platform's,
  which Claude Code takes from `ant` where it has none of its own -- has no
  Remote Control: the window says to log in with a claude.ai account. One
  that does not connect within 30 s is given up, saying what Claude Code's
  screen said.

**Titling a session.** The window's title is the session's -- `Claude ·
<title>`, its first line until it is given one. A click on it edits it, the
cursor where it was clicked. Left and Right move the cursor, Option with
them a word, Cmd with them -- or Home and End -- to either end; Shift with
any of them selects, as a drag does, Shift+click extends the selection, a
double click selects a word, and Cmd+A all of it. Typing or pasting replaces
the selection; Backspace and Delete remove it, or else before and after the
cursor, Option+Backspace a word; Cmd+C copies it, Cmd+X cuts it. Enter keeps the
title, and so does a click anywhere off it -- into the box below, say, which
then takes what is typed; Esc puts it back, and a blank one puts back the
first line. A session with no title yet starts the edit empty: its start
time, shown until then, is no title.
A title kept is written into the front matter (`title:`) and the
transcript's heading at once, and the point relabeled -- or, before the
first turn, with it, as there is no file to write until then. The point is
labeled `Claude · <title>`, and beneath it is dated by the session's last
write, its front matter's `updated`, not by the file's.

**Renaming its file.** The file can be renamed, or moved anywhere the scan
reaches, while its window is closed: what resumes it is the `session` id in
its front matter, not its name. Keep the `.md`, and reload. A file renamed
while its window is open is written again under its old name at the next
turn.

**Saved at every turn.** As each reply comes in whole, the session is saved
into the directory scanned (`--dir`), or into `--claude-dir`: a directory
relative to it that the scan reaches, so not absolute, not above `--dir`,
and not a hidden or skipped one. It is one markdown,
`claude-session-<started>-<id>.md`, a `## You` / `## Claude` section a turn.
Its front matter says what it is: `hyp: claude-session`, the session's id,
the model, and the first line as its `title`. The disk is brought up to date
with it at once, in the background, the window usable meanwhile: the
directory is read again, the new turn embedded, and the layout nudged from
where it was rather than laid out afresh. Closing the window saves nothing
more, unless a turn is not yet saved.

**Every turn, a point.** A conversation is shown turn by turn: each turn is
a point of its own, keyed `<file>#<n>`, embedded by what was asked and
answered in it -- not its file's name, nor its title, nor the transcript's
markup (the `# Claude ·` heading, the `## You` and `## Claude` lines), which
are every turn's and would draw them all together. A long answer is
embedded a 4,000-character piece at a time, its question's first line
before each piece, and the pieces' unit vectors averaged (`ChunkedEmbedder`,
`hyp/docs.py`). No markdown's front matter is embedded, a session's or any
other's.

A turn is labeled `<title> › <name>`: its conversation's title -- the
root conversation's, for a branch's turns too -- cut to 22 characters at a
word, then a few words naming what the turn is about, up to 48 characters in
all; its tooltip says its conversation and which turn of it it is, and how
the question begins. The names are Claude's, of what was asked and the
start of what was answered, both: a question alone often says little of
what the turn came to. The turns not named yet, as the points come in, are
asked of it in the background, 40 at a time (`claude -p`, no tools), and
kept in the corpus's `words.json`, by question and answer, so a turn is
named once; till then a turn is named by its question's first words. A turn
Claude could not name -- not logged in, say -- is not asked again till hyp
starts again, and the status line says why. A branch's last turn is dated by
its file's last write; the others are not dated. A handoff -- to the cloud,
or by Remote Control -- is named as its turn is labeled, its conversation's
last for a conversation (`claude -n`): so the phone's list says what each
session is. A turn cut short -- `[Request interrupted by user]`, as Claude
Code records it -- is no turn of its own, but its turn's, which goes on
after it, in the file and in Claude Code's record alike.

**Joined, turn to turn.** A faint geodesic joins each turn to the one before
it, and a branch's first turn to the turn it goes on from (below): a
conversation is drawn as the tree it is, a small arrowhead two thirds of the
way along each line pointing from a turn to the one after it, root to leaf
(`Hyp.arrowhead`; none on a line drawn too short to hold one). Joined, turns are laid out near
each other, whatever they go on to: a link adds proximity to the metric the
layout measures by, not to the embeddings, which still say only what the
texts say -- so a point's nearest, as the hover's arcs and the arrows have
them, are as they were. The cosine distance between the two is shortened by
`LINK_PULL` (70%), each is among the other's neighbors in the graph UMAP is
given, and a new turn starts beside, and slides in from, the one it follows.
One neighbor among fifteen does not hold a turn in place, so one laid out
farther from the turn it follows than `LINK_REACH` -- 1.5 times the layout's
median distance from a point to its nearest -- is moved in along the
geodesic between them, in the order the conversation grew
(`hyp/layout.py`). A link made or gone lays the disk out again, as a
changed embedding does.

**Going on from a turn.** A left click on a turn opens its conversation up
to that turn, and the box beneath it: what is sent from there goes on after
it. The turn its branch's last, that is the branch, going on; any other,
it is a new branch, and the first thing sent from it makes the branch --
closed before then, it leaves nothing. A branch's turns before its own are
shown above a line saying where it goes on from. Claude Code keeps a
conversation on the machine, and for the directory, it ran in: one saved on
one machine and reopened on another is shown, but Claude there has no
record of it, and the window says so. A right click on a turn places its
conversation's file, at the turn, as any document's.

A branch is a markdown of its own, holding only its own turns, its front
matter naming its parent's file -- relative to its own -- and `fork`, the
number of the turns of its parent's whole conversation it keeps; the turns
before its own are assembled from its parents as it is opened, and its
first turn is joined to its parent's turn `fork`. Its parent is the branch
whose own turns hold the turn clicked, so the tree is kept as it grew. On
Claude Code's side, its first turn forks the conversation it was made from,
at the last message of that turn (`claude -p --resume <id>
--resume-session-at <message> --fork-session --session-id <new id>`): the
branch has the turns before it, and not those after, and the conversation
it came from is as it was. A branch whose parent's file is gone shows its
own turns and says so.

## Arrows

Each arc a hover draws is an arrow, from the hovered point to a neighbor:
the difference of their embeddings, `ê_there − ê_here`, each unit length
first. Hold Shift over the point, and each arc is labeled with what its
direction means -- `Test split toward train split; probe outputs toward raw
data-pipeline stages` -- and beneath it, what that rests on: `the same move
in 5 other pairs · cos up to 0.97`.

An arrow's two endpoints say only what differs between them, which their
labels say already. What the arrow means is its direction, and a direction
is the same anywhere in the space, so it is read where else it runs in the
corpus (`hyp/arrows.py`, `Engine.arrow_notes`):

- **The same move elsewhere.** Every pair of neighbors in the corpus is an
  arrow too. Those whose unit difference points most nearly the same way --
  the cosine of the two directions, 0.5 or more -- are the same move made
  elsewhere: up to 5 of them, a pair the corpus holds again (a pipeline run
  again) counted once. Those at 0.7 or more are what the line beneath
  counts.
- **Its axis.** The whole corpus projected on the direction: the 5 points
  furthest toward each end are what it runs from, and toward, whoever its
  endpoints are.

That evidence, by the points' labels -- a pipeline's with its `VAR.name` --
is what Claude Code is given (`claude -p`, under your login, with no
tools), and asked to name the direction itself, in one line of at most 12
words, as all of it has in common: not the two endpoints, and not a list of
the evidence. All of a hover's arrows not named before go in one question,
a few seconds; "…" stands on each arc until the lines come.

Each is kept in the store (`notes.json`), by the two points' embed texts and
the corpus's size -- the evidence is the whole corpus's, so a corpus that
grows asks again -- and a change of embedder discards them all. Two
neighbors with one embedding -- a pipeline run again -- are not asked
about: their arc says `the same embedding`. A reading that fails -- no
Claude Code, not logged in -- says why on the first arc, and is tried again
after 30 seconds of Shift.

What a direction can mean is what the texts hold. A pipeline's text is its
configuration, so in a journal most directions are a split's or a variant's
change, and the evidence shows how consistently the embedding makes it; in
a directory of markdowns, they are directions of content.

## Controls

| Input | Effect |
|---|---|
| left click | slide the clicked point to the center along a geodesic; time ∝ distance |
| reload (top left) | read the corpus's source again, re-embed what changed, and lay it out afresh |
| re-embed (beside it) | embed the whole corpus afresh, from its source, and lay it out anew; points the source no longer gives -- a text typed in, where every point is a file -- are dropped |
| left click on a pipeline (`--journal`) | its window, as a right click opens a point otherwise |
| right click on a pipeline (`--journal`) | its constructed entry blocks, in a menu at the mouse: picking one opens that block's window directly |
| left drag | move the plane rigidly (a hyperbolic translation); the grabbed point stays under the cursor |
| scroll | hyperbolic zoom: a point at distance *d* is drawn at *k·d* |
| right click on a point | its stored text |
| left click on a Claude turn (`--docs`) | its conversation up to that turn, the box beneath: what is sent goes on after it -- a new branch, unless it is its branch's last. See [Claude sessions](#claude-sessions) |
| right click elsewhere | type a text; Enter embeds, summarizes, stores and places it -- in `--docs`, asks it of Claude in a new session, or, while sessions' turns are shown, folds them back |
| hover | the point's label, in full, and geodesics to the 5 nearest neighbors in embedding space -- by cosine over the whole vectors, so an arc may reach a point far off on the disk |
| Shift, while hovering | each of those arcs labeled with what its direction means, named by Claude from where else it runs in the corpus, and what that rests on. See [Arrows](#arrows) |
| Home (fn+← on a Mac), or double click off the disk | back to the original view: the corpus center, zoom 1 -- on a slice disk too |
| (lower left) | a spinning wheel while data are being loaded or prepared; an error, in red, for 15 s or until the next read -- its full traceback in the terminal |
| while a corpus loads | the window is dimmed and takes no input. A card in its middle shows the stage under way, with a bar, the files done out of the total, the time taken and the time left. An anchor's journal is listed first, as a stage of its own (`dbx.BuildJournal.list`). Until the listing is done there is no total, so the bar slides and the card counts the files found so far, after every directory. Then the files are read (`dbx.BuildJournal.read_frame`), counted against that total |
| Esc, while a corpus loads | interrupt it at its next step and drop anything queued behind it. A reload that is interrupted leaves the disk as it was; a switch of journal leaves it empty until the next reload. Laying out the disk (UMAP) cannot be cut short, so an Esc during it waits for it to finish |
| points (top right) | every point, by its label and second line in full, alphabetically (the first 2,000): picking one slides it to the center; a right click on one does what a right click on its point does |
| settings (top right) | tags (an anchor's blocks), names and versions (`--journal`) and datetimes, each on and off: under labels and in tooltips; line numbers, in text windows (off by default); title colors -- a point's window titled in the point's color, and its entry blocks' windows in the same; in `--docs`, sessions on remote, and web access -- Claude may search the web and fetch pages, kept with the corpus (see [Claude sessions](#claude-sessions)) |
| in a text window | scroll / Up Down / PgUp PgDn move; Left / Right step through its title menu; × or Esc closes it |
| in a text window: the triangle by a line | a line whose next line is indented deeper folds the deeper lines after it into `… N lines`, and unfolds them. Spec, dfn and kwargs are shown one key per line, a level per four spaces, so they fold by nesting |
| in a text window: *fold* (top right), or 1–9, 0 | fold every such line at that many levels of indentation or deeper; *unfold all*, or 0, unfolds them all |
| in a text window: drag, Cmd+A, Cmd+C | select text, select all; copy the selection -- or, with none, all of it |

## Options

```
hyp --data DIR               # the corpus directory, local or an fsspec url
                             # (default: ~/.hyp/seeds; with --journal, {DBX_DATALAKE}/.hyp, or .hyp/anchors/ANCHOR)
hyp --reset                  # discard the corpus and start over
hyp --fg                     # keep the terminal: the output there, Ctrl+C stopping it
hyp --journal                # the pipelines of DBX_DATALAKE (HypDbxJournal); also: --journal pipelines
hyp --journal ANCHOR         # the blocks of ANCHOR, from its build journal; the journal menu switches either
hyp --journal ANCHOR --filter 'key=pattern, ...'   # its blocks filtered; the filter field changes it
hyp --journal --ece ckpt     # --extra-construction-events: events that also make an entry block count as
                             # constructed, each alone or followed by = or : -- here, a still that checkpointed
hyp --docs [--dir DIR]       # the markdowns and pdfs under DIR, local or an fsspec url (HypDocs; default: the current directory)
hyp --docs --claude-dir REL        # Claude sessions saved under DIR/REL, not DIR itself (REL relative, and scanned)
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
`__open_point__` (a right click on a point), `__submit_text__` and
`BOX_HINT` (what a text typed on empty disk does), `__wants_text__`
(whether anything is being typed into: text input is on exactly then), and
`__point_label__`,
`__point_datetime__`, `__point_tooltip__` (how a point is labeled).
`hyp.dbxpipelines.HypDbxJournal` is the example.

## Tests

`tests/test_hyp_*.py`, run with the rest of the suite (see Running Tests in the
README).
