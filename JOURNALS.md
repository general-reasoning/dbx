# Journals

dbx keeps two journals, both as Parquet files under the datalake:

| | what one row is | read with | returns |
|---|---|---|---|
| **exec journal** | one `dbx.exec` / `dbx.pprint` command | `dbx.execjournal()` | `ExecjournalFrame` / `ExecjournalEntry` |
| **data journal** | one journal entry of one block (`build:end`, a note, …) | `dbx.datajournal(anchor)` | `DatajournalFrame` / `DatajournalEntry` |

They are linked by the command's `session`. Every journal entry written while
a command runs is written under its session, and the session keeps an index of
those entries in storage, under `<datalake>/.journal/sessions/<session>/`,
next to the exec journal. The methods below follow that link.

All the frames are pandas DataFrames, and all the entries are pandas Series,
so ordinary pandas works on every one of them.

## Filters

Both readers take column filters as keyword arguments, `column=pattern`, and a
row is kept when its value matches:

| pattern | matches |
|---|---|
| `'a6'` | substring anywhere |
| `'^a6'`, `'a6.*'` | regex (`re.search`) |
| `'a6*'`, `'*a6*'`, `'*e3'` | glob, anchored to the whole value: has `*`/`?` and no regex metacharacter |
| `'x=1'`, `'x:1'` | `key=value` / `key: value` pair inside the text, e.g. a spec |
| `re.compile(...)` | `.search` |
| callable | `f(value)` is truthy |
| `[p, q]` (list) | any of them (OR) |
| `(p, q)` (tuple) | all of them (AND) |

`date=` takes a date, datetime or string, or a list of them, and matches the
rows whose `datetime` falls on that day. `datetime=` matches exactly.

A filter on a column the frame doesn't have returns an **empty** frame. It does
not raise.

## `dbx.execjournal()`

```python
dbx.execjournal(loc=None, *, iloc=None, datalake=None, index=..., **filters)
```

This reads `<datalake>/.journal/exec/`. `datalake` defaults to
`DBX_DATALAKE`, then `DBX_ROOT`, then `DBX_URL`. That is where `dbx.exec`
writes every command, whatever datalake the command's blocks use.

The result is an `ExecjournalFrame`, **newest first**, indexed by the
command's `id` (pass `index=None` to number the rows 0..N-1 instead). With
`loc=` (an id, or a prefix of exactly one id) or `iloc=` (a position), it
returns that one row as an `ExecjournalEntry`.

Columns, in the order the frame holds them:

| column | holds |
|---|---|
| `datetime`, `exec:start:datetime` | when it started (a string, `2026-09-28T22-22-09.418165`) |
| `exec` | command **verbatim**, trailing `# comment` included |
| `exec:end:datetime` | when it finished, including when it raised |
| `exception` | what it raised, as `'<Type>: <message>'`; None if it returned |
| `traceback` | where it raised, as Python prints it; None if it returned |
| `id` | row's uuid |
| `session` | `Datajournal` session every block in the command wrote under |
| `datajournal_entries` | paths of the block journal entries the command's own process wrote or got back from its workers, in the order first written (the session index is the complete list) |
| `output_captures` | paths of the command's captured stdout/stderr, the master capture first; empty unless it ran with `capture_output=True` / `--capture-output` (see [Captured output](#captured-output)) |
| `comment` | trailing `# comment` alone: what the command was *for* |
| `success` | whether it returned rather than raised |

```python
dbx.exec("a = Apple(spec={'n': 1}); b = Apple(spec={'n': 2}); a.build(); b.build()  # two apples")
dbx.exec("Apple(spec={'n': 1}).build()  # again")

ej = dbx.execjournal()
ej[['exec', 'comment']]
#                                       exec                                     comment
# id
# 976f41ea-…  Apple(spec={'n': 1}).build()  # again                            again
# f20351c5-…  a = Apple(spec={'n': 1}); b = Apple(spec={'n': 2}); …            two apples

ej.get('976f41ea')                                  # the displayed id is enough
e = dbx.execjournal(comment='two apples', iloc=0)   # ExecjournalEntry
e['exec'], e['session'], len(e['datajournal_entries'])   # (..., '692e946a-…', 2)
```

### `ExecjournalFrame`

| call | returns |
|---|---|
| `ej.get(id)` | `ExecjournalEntry` at label `id`: the full id, or a prefix no other id shares, such as the short id the frame displays |
| `ej(id)` | same, with null columns dropped |
| `ej.show(width=, max_colwidth=, max_rows=, full=False)` | prints the frame wider than its repr does, or with `full=True` every column and value exactly as held |
| `ej.dataentries()` | Python **list** of `DatajournalEntry`, one per block journal entry that any command in the frame wrote: the first command's entries in written order, then the next command's |
| `ej.datajournal()` | `DatajournalFrame` of the **same records**, one row each: row *i* is `ej.dataentries()[i]` |
| `ej.anchors()` | sorted list of the anchors those entries belong to |
| `ej.constructed(anchor=None, **filters)` | as on an entry (below), over every command in the frame |

`dataentries()` and `datajournal()` hold the same records. Use the list to work
with one entry at a time (`e.dataentries()[0].block.paths()`), and the frame to
work with columns (`e.datajournal()[['anchor', 'event', 'hash']]`,
`e.datajournal()['hash'].tolist()`).

**What a frame displays is not what it holds.** The repr shows:
- ids and sessions as their first 8 characters, `976f41ea…`;
- timestamps to the second;
- `datajournal_entries` and `output_captures` as counts, `[2]`.

It leaves out:
- the `id` column, when it only repeats the index;
- `exec:start:datetime`, when it equals `datetime`;
- `traceback`, whose last line is `exception` anyway.

The data keeps every value in full, so indexing, filtering and `.loc` see
full ids. The repr takes the terminal's width even if pandas'
`display.width` is narrower, and it wraps columns onto further blocks rather
than hiding any. For a wider line, a longer `exec`, or everything exactly as
held, use `ej.show(width=250, max_colwidth=120)` or `ej.show(full=True)`.

**A frame derived from one is still an `ExecjournalFrame`.** A column
selection, a boolean filter, a sort or `head()` all come back as
`ExecjournalFrame`s with the same datalake, so they display the same way and
keep `.get()` and `.datajournal()`. The same holds for a `DatajournalFrame`.

To pick commands with a filter and then look at what they wrote:

```python
dbx.execjournal(comment='nightly').datajournal()      # every entry every nightly run wrote
dbx.execjournal(date='2026-09-28').constructed()      # {anchor: DatajournalFrame} built that day
```

### `ExecjournalEntry`

| call | returns |
|---|---|
| `e.dataentries()` | Python **list** of `DatajournalEntry`, one per entry written under the command's session, in the order written |
| `e.datajournal()` | `DatajournalFrame` of the **same records**, one row each: row *i* is `e.dataentries()[i]`, numbered 0..N-1, plus an `entry_path` column naming the file each row was read from |
| `e.anchors()` | sorted list of the anchors the command wrote entries for |
| `e.constructed()` | `{anchor: DatajournalFrame}` of what the command **constructed**, for every anchor that has any |
| `e.constructed(anchor)` | that one anchor's `DatajournalFrame` |
| `e.constructed(anchor, **filters)` | same, further filtered, e.g. `hash='^3c29'` |
| `e.constructed(event=None)` | drops the default event filter, which leaves every entry the command wrote, grouped by anchor |
| `e.rerun(**kwargs)` | runs `exec` again through `dbx.exec` and returns its value (below) |
| `e.output(idx=0)` | prints `output_captures[idx]` to stdout; 0 is the master capture |
| `e.output(basename='…')` | prints the one capture whose file name the regex matches at its start (below) |

"Constructed" means an entry whose `event` is one of `CONSTRUCTED_EVENTS`:
`build:end`, `UNSAFE_redirect` or `UNSAFE_copy_from:END`, matched exactly. The
default `event` filter is applied **in addition** to any other filter you pass.
Passing `event=` yourself replaces it.

```python
e.dataentries()                    # [DatajournalEntry, DatajournalEntry]
e.datajournal()[['anchor', 'event', 'hash']]
#               anchor      event        hash
# 0  demo_blocks.Apple  build:end  3c293846…
# 1  demo_blocks.Apple  build:end  2e3ec8ff…
e.anchors()                        # ['demo_blocks.Apple']
e.constructed()                    # {'demo_blocks.Apple': DatajournalFrame (2 rows)}
e.dataentries()[0].block.paths()   # {'x': '…/demo_blocks.Apple/3c293846/x/x.txt'}

dbx.execjournal(comment='again', iloc=0).dataentries()   # []
```

The second command wrote **nothing**. `build()` on a block that is already
built returns without journaling, so a rebuild of a valid block is not among
the command's entries or its `constructed()`. That holds with `deep=True`
too: `deep` makes `build_tree()` go into subtrees that are already valid, but
each valid block's own `build()` still skips without writing.

**Reads are live.** The entries are read from the session index at call
time, so `.dataentries()` shows what is in those files *now*, and a path that has
since been cleared is skipped with a warning. A row written before the index
existed falls back to its `datajournal_entries`.

**One entry per block instance.** A block instance rewrites its one entry
file. So a `build()` appears as its final event: `build:end`,
`build:exception` if it raised, or `build:start` if it never finished. A
failed build is in `datajournal()` and not in `constructed()`. Two instances
of the same block give two entries. The same goes for `build_tree()`: it
writes `build_tree:<var>:begin` and `build_tree:<var>:end` for each var into
that one file, so after a `build_tree()` that skipped its own build, the entry
holds only the last var's `end`.

**A command that raises is still recorded**, together with the entries it had
written before the exception.

**Other threads and processes are included.** Whatever writes an entry also
writes its index marker, so the command's rows list it however it got there:
- **A dbx executor's workers** (`multithreading`, `multiprocessing`,
  `torch_multiprocessing`, `ray`): the executor sends the command's session
  out with the work, and brings the written paths back with the results.
- **A process started while the command runs** (a subprocess, a Slurm job):
  `dbx.exec` exports the session in `DBX_JOURNAL_SESSION` and
  `DBX_JOURNAL_DATALAKE`, and a process that starts with them set writes under
  it. A `dbx.exec` in that process joins the session, so its rows and the
  parent's share their entries.
- **A worker that dies before returning:** its entries are listed anyway,
  because it wrote their markers itself.

What still isn't listed:
- **A remote process outside dbx's executors** that doesn't inherit the
  environment, such as a Ray actor you create and call yourself.
- **Entries written inside a `with Datajournal()` you open yourself** within
  the command, which is a session of its own.

A worker on another host must also be able to write to the command's datalake
for its markers to land.

`rerun()` first prints the shell line that would run the same command,
`dbx.pprint "…"`, and then runs it as a **new** command, which gets its own
exec-journal row and session. It runs against the code as it is **now**: it
does not check out the revision the original ran under. Pass the names the
original was given as keyword arguments, e.g. `e.rerun(Apple=Apple)`.

### Captured output

A command run with `capture_output=True`, or `--capture-output` on the
command line, tees its stdout and stderr to files under its session:

```bash
dbx.pprint --capture-output "my.Stack(spec={'n': 8}).build()  # nightly"
```

```python
dbx.exec("my.Stack(spec={'n': 8}).build()", capture_output=True)
```

`dbx`, `dbx.exec`, `dbx.print` and `dbx.pprint` all take the flag. It is a
flag and not a `capture_output=True` argument because every `k=v` argument
binds a name for the statements. `dbx.pprint` opens the capture itself, so the
printed result is captured too. The capture works on file descriptors 1 and 2,
so it holds what C extensions and subprocesses write as well as Python's
output, and everything still reaches the terminal.

- **The master capture**, `output_captures[0]`, is the process that ran the
  command. It opens when the command starts and closes when it finishes.
  Worker **threads** write here, because every thread in a process shares
  its file descriptors.
- **A worker process** that a dbx executor sends work to (`multiprocessing`,
  `torch_multiprocessing`, `ray`) runs each callable inside a capture of its
  own. The executor brings its path back with the result, including when
  the callable raised, and it is appended to `output_captures`. A spawned or
  forked worker inherits its parent's descriptors, so its output is in the
  master capture too. A Ray worker's output isn't.
- **A nested `dbx.exec`** joins the capture that is already open, and its row
  lists the same master.

The files live in `<datalake>/.journal/sessions/<session>/output/` and are
named `master-<host>-<pid>-<datetime>.log` or
`worker-<host>-<pid>-<datetime>.log`.

```python
e = dbx.execjournal(comment='nightly', iloc=0)
e.output()                              # the master capture
e.output(2)                             # output_captures[2]
e.output(basename='worker-node3')       # a prefix of the file name…
e.output(basename=r'worker-.*-4242-')   # …or a regex, matched at its start
```

`basename=` must match exactly one capture, or it raises `LookupError` and
lists the names. It searches every capture in the session directory, not only
the ones the row lists, so it also finds the capture of a worker that died
before it could send its path back.

A block doesn't capture anything itself. Its journal entry's `log` column is
the path of the capture open in the process that built it. That file is
shared with everything else the process printed, and it is only written once
the capture closes. `Datablock(capture_output=...)` is still accepted, so
that recorded dfns still reconstruct, but it is ignored and not recorded.

## `dbx.datajournal()`

```python
dbx.datajournal(what, loc=None, *, iloc=None, datalake=None, index=..., unnormalized=False, **filters)
```

`what` can be any of these:

- an anchor string such as `'demo_blocks.Apple'`
- a `Datablock` class
- a block instance, whose `datalake`, `storage_options` and `log` then become the defaults
- a DataFrame to wrap as a `DatajournalFrame`, which reads nothing

This reads every entry under `<datalake>/<anchor>/`, for every hash and tag.
The result is a `DatajournalFrame`, **newest first**, indexed by the entry's
`id`. With `loc=` (an id, or a prefix of exactly one id) or `iloc=` (a
position), it returns one `DatajournalEntry`. A prefix that matches no id or
several ids raises `KeyError`, listing the matches. An anchor with no journal
directory raises `FileNotFoundError`.

`block.datajournal(**filters)` reads the same journal through the block's own
`Datajournal`, but its frame is numbered rather than indexed by id, so its
`loc=` is a row number. Pass `index='id'` to look an entry up by id or id
prefix instead. `frame.get(...)` takes the same labels, prefixes included.

```python
dbx.datajournal(Apple)                                  # every Apple entry
dbx.datajournal(Apple, event='build:end', iloc=0)       # the latest build
dbx.datajournal(a)                                      # a's anchor, a's datalake
dbx.datajournal('demo_blocks.Apple', session=e['session'])   # what one command wrote for this anchor
dbx.datajournal(Apple, hash='^3c29', date='2026-09-28')
```

The main columns:

- `anchor`, `hash`, `key`, `tag`, `version`, `revision`, `gitrepo`, `datalake`: the block and the code it came from
- `event`: `build:end`, `build:exception`, `UNSAFE_redirect`, `note:…`, …
- `datetime`: parsed to a real datetime here, unlike in the exec journal
- `session`, `tree`: the command, and the build tree within it
- `id`, `code`: identify the row
- `topics`, `paths`: the recorded `{topic: filename}` and `{topic: path}`
- `log`: path of the output capture open while the block was built, or None (see [Captured output](#captured-output))
- `spec`, `quote`, `cite`, `repr`, `signature`, `type`, `note`: **paths** to the files holding each rendering

Rows from older eras have their `type` and `signature` columns resolved per
row, so a filter on them means the same thing on every row. Pass
`unnormalized=True` to get the columns exactly as recorded.

### `DatajournalFrame`

| call | returns |
|---|---|
| `dj.get(label)` / `dj(label)` | `DatajournalEntry` at label `label` (`.loc`; `dj()` drops null columns) |
| `dj.list(column, take='last'\|'first'\|'all', sortby=None)` | a frame of `column` read for each hash: one row per hash by default. `'spec'` expands to a column per spec field |

```python
dbx.datajournal(Apple).list('spec')
#    n       hash                   datetime
# 0  2  2e3ec8ff…  2026-09-28 22:21:57.770803
# 1  1  3c293846…  2026-09-28 22:21:57.735728
```

### `DatajournalEntry`

A row. Plain column values read as `entry['event']` or `entry.anchor`.
`DatajournalEntry.column(entry, name)` reads a column that may be missing from
the row, or that has been renamed since, and returns None where that column is
absent.

| call | returns |
|---|---|
| `entry.read('spec')`, `entry.read('quote', 'note')` | **contents** behind a path column: YAML parsed, `.txt`/`.log` as text; several names give a dict |
| `entry.block` | `Block`: the block **as recorded**, with `Datablock`'s API shape |
| `entry.inst()` | re-creates the block by evaluating its recorded `quote` in this interpreter |
| `entry.rinst()` / `entry.inst(remote=True)` / `entry.trueinst()` | same, but on a Ray worker pinned to the entry's `revision`, so the hash matches the recorded one; returns a proxy |

`inst()` can check out the project at the recorded revision, but not `dbx`,
which is already imported. So if `dbx`'s rendering has changed since the
entry was written, the local instance comes back with a different hash and
paths that hold nothing. Use `rinst()` when that matters.

`entry.block` is read **off the row**, not recomputed:

```python
blk = e.dataentries()[0].block       # Block(demo_blocks.Apple/3c2938466890…)
blk.hash, blk.anchor, blk.key, blk.revision, blk.session, blk.tree   # properties
blk.paths()        # {'x': '…/3c293846/x/x.txt'}: the paths actually written
blk.topics()       # ['x']
blk.quote()        # "$demo_blocks.Apple(spec={'n': 1}, …)": the text, not the path
blk.signature(), blk.type(), blk.typestr(), blk.cite(), blk.repr(), blk.note()
blk.ls('x'), blk.list('x'), blk.size('x')      # storage at the recorded paths
```

## Across anchors

`dbx.constructed(anchor=None, *, event=..., datalake=None, **filters)` works
on the data journal **without** an exec row:

- given an anchor, it returns that anchor's constructed entries as one `DatajournalFrame`
- with no anchor, it reads every anchor in the datalake into one frame, newest first
- `event=None` drops the event filter

```python
dbx.constructed(Apple)                  # every Apple built, redirected or copied in
dbx.constructed(date='2026-09-28')      # everything constructed that day, all anchors
```

`dbx.journal(what=None, …)` is the older single entry point. With `what` it
calls `datajournal`, and without it `execjournal`. New code should call the one
it means.
