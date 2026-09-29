# Journals

dbx keeps two journals, both as Parquet files under the datalake:

| | what one row is | read with | returns |
|---|---|---|---|
| **exec journal** | one `dbx.exec` / `dbx.pprint` command | `dbx.execjournal()` | `ExecjournalFrame` / `ExecjournalEntry` |
| **data journal** | one journal entry of one block (`build:end`, a note, …) | `dbx.datajournal(anchor)` | `DatajournalFrame` / `DatajournalEntry` |

They are linked: every block a command writes a journal entry for does so
under the command's `session`, and the command's exec-journal row lists those
entries' paths in `datajournal_entries`. The methods below follow that link.

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
`loc=` (an id) or `iloc=` (a position), it returns that one row as an
`ExecjournalEntry`.

Columns:

| column | holds |
|---|---|
| `exec` | command **verbatim**, trailing `# comment` included |
| `datetime`, `exec:start:datetime` | when it started (a string, `2026-09-28T22-22-09.418165`) |
| `exec:end:datetime` | when it finished, including when it raised |
| `id` | row's uuid |
| `session` | `Datajournal` session every block in the command wrote under |
| `datajournal_entries` | paths of the block journal entries the command wrote, in the order first written |
| `comment` | trailing `# comment` alone: what the command was *for* |

```python
dbx.exec("a = Apple(spec={'n': 1}); b = Apple(spec={'n': 2}); a.build(); b.build()  # two apples")
dbx.exec("Apple(spec={'n': 1}).build()  # again")

ej = dbx.execjournal()
ej[['exec', 'comment']]
#                                       exec                                     comment
# id
# 976f41ea-…  Apple(spec={'n': 1}).build()  # again                            again
# f20351c5-…  a = Apple(spec={'n': 1}); b = Apple(spec={'n': 2}); …            two apples

e = dbx.execjournal(comment='two apples', iloc=0)   # ExecjournalEntry
e['exec'], e['session'], len(e['datajournal_entries'])   # (..., '692e946a-…', 2)
```

### `ExecjournalFrame`

| call | returns |
|---|---|
| `ej.get(id)` | `ExecjournalEntry` at label `id` (`.loc`) |
| `ej(id)` | same, with null columns dropped |
| `ej.dataentries()` | Python **list** of `DatajournalEntry`, one per block journal entry that any command in the frame wrote: the first command's entries in written order, then the next command's |
| `ej.datajournal()` | `DatajournalFrame` of the **same records**, one row each: row *i* is `ej.dataentries()[i]` |
| `ej.anchors()` | sorted list of the anchors those entries belong to |
| `ej.constructed(anchor=None, **filters)` | as on an entry (below), over every command in the frame |

`dataentries()` and `datajournal()` hold the same records. Use the list to work
with one entry at a time (`e.dataentries()[0].block.paths()`), and the frame to
work with columns (`e.datajournal()[['anchor', 'event', 'hash']]`,
`e.datajournal()['hash'].tolist()`).

To pick commands with a filter and then look at what they wrote:

```python
dbx.execjournal(comment='nightly').datajournal()      # every entry every nightly run wrote
dbx.execjournal(date='2026-09-28').constructed()      # {anchor: DatajournalFrame} built that day
```

### `ExecjournalEntry`

| call | returns |
|---|---|
| `e.dataentries()` | Python **list** of `DatajournalEntry`, one per path in `datajournal_entries`, in the order written |
| `e.datajournal()` | `DatajournalFrame` of the **same records**, one row each: row *i* is `e.dataentries()[i]`, numbered 0..N-1, plus an `entry_path` column naming the file each row was read from |
| `e.anchors()` | sorted list of the anchors the command wrote entries for |
| `e.constructed()` | `{anchor: DatajournalFrame}` of what the command **constructed**, for every anchor that has any |
| `e.constructed(anchor)` | that one anchor's `DatajournalFrame` |
| `e.constructed(anchor, **filters)` | same, further filtered, e.g. `hash='^3c29'` |
| `e.constructed(event=None)` | drops the default event filter, which leaves every entry the command wrote, grouped by anchor |
| `e.rerun(**kwargs)` | runs `exec` again through `dbx.exec` and returns its value (below) |

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

**Reads are live.** The entries are read from `datajournal_entries` at call
time, so `.dataentries()` shows what is in those files *now*, and a path that has
since been cleared is skipped with a warning.

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

**Worker processes are included.** A block that a stack builds on a
`multiprocessing`, `torch_multiprocessing` or `ray` executor writes under the
command's session, and its entry is listed here. The executor sends the
command's `Datajournal` handle out with the work and brings the written paths
back with the results. There are two exceptions:
- **A process you start yourself** (outside a dbx executor) writes under its
  own session and isn't listed.
- **A block given `datajournal=` explicitly** writes to that handle rather
  than the command's.

`rerun()` first prints the shell line that would run the same command,
`dbx.pprint "…"`, and then runs it as a **new** command, which gets its own
exec-journal row and session. It runs against the code as it is **now**: it
does not check out the revision the original ran under. Pass the names the
original was given as keyword arguments, e.g. `e.rerun(Apple=Apple)`.

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
`id`. With `loc=` (an id) or `iloc=` (a position), it returns one
`DatajournalEntry`. An anchor with no journal directory raises
`FileNotFoundError`.

`block.journal(**filters)` reads the same journal through the block's own
`Datajournal`, but its frame is numbered rather than indexed by id.

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
