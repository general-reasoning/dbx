# DatastreamTab / DatastreamTable

`DatastreamTable` is a `Datastack` whose blocks — `DatastreamTab`s — each write several
**parallel MDS streams**, called *slices*, that can be zipped back into one
`torch.utils.data.Dataset` on demand.

Both live in `dbx.datastreams`, which needs `torch` and `mosaicml-streaming`
at import time:

```python
from dbx.datastreams import DatastreamTab, DatastreamTable
```

## Why slices

A slice is one independently-readable MDS stream.  Splitting a tab's payload
across several of them means a consumer pays only for what it opens: images
alone for pixel pretraining, annotations alone for a label probe, or any
subset zipped together for training.  Interleaving everything into one stream
would force every consumer to fetch the image bytes.

The contract that makes this work is **lockstep**: every slice of a tab is
written from one pass over that tab's input, so sample *i* of every slice
describes the same item.  A tab must never write to one slice without
writing the matching entry to the others.

Slices are not the same thing as the lumps `data()` returns.  `data(slice)`
materialises a whole slice as a list; `dataset(*slices)` opens live
`StreamingDataset`s and zips them lazily.

## Implementing a Tab

Declare `SLICES`, add whatever VAR addresses the tab's input, and write the
slices in lockstep inside `__build__`:

```python
class FrameTab(DatastreamTab):
    VERSION = 1
    SLICES = ('frames', 'annotations')
    TOPICS = {'note': 'note.txt'}          # optional extra topics

    @dataclass
    class VAR(DatastreamTab.VAR):
        episode: str = None                # on top of table and tab_idx

    COLUMNS = {
        'frames':      {'idx': 'int', 'image': 'jpeg'},
        'annotations': {'idx': 'int', 'label': 'json'},
    }

    def __build__(self):
        with self.slice_writers(self.COLUMNS, compression='zstd') as writers:
            for i, (image, label) in enumerate(load(self.var.episode)):
                writers['frames'].write({'idx': i, 'image': image})
                writers['annotations'].write({'idx': i, 'label': label})

    def __stats__(self, slice_name):
        return {'n_samples': len(self.data(slice_name))}
```

`TOPICS` is synthesized from `SLICES`:

```python
FrameTab.TOPICS
# {'data': {'frames': DIRTOPIC, 'annotations': DIRTOPIC}, 'note': 'note.txt'}
```

so the slices are covered by the block's signature and hence by its hash:
adding, removing or renaming one re-keys the block rather than quietly reusing
another shape's artifacts.  Declaring the `data` group explicitly works too,
and then *it* defines `SLICES`; declaring both and having them disagree is an
error.

`slice_writers` opens one `MDSWriter` per slice, stages to a temporary
directory when storage is remote, and uploads each slice with `index.json`
**last**.  On an exception nothing is finished and nothing is uploaded, so a
failed tab reports unbuilt and is redone rather than half-read.

### Topics besides the slices

Anything else in `TOPICS` is an ordinary Datablock topic and behaves like one.
The synthesized `data` group is merged with what the class declares, so a tab
or table keeps its own files, dicts and `DIRTOPIC`s — nested ones included:

```python
class FrameTab(DatastreamTab):
    SLICES = ('frames', 'annotations')
    TOPICS = {'note': 'note.txt', 'debug': {'plots': DIRTOPIC}}
```

They differ from slices in three ways:

- **Location.** Only the slices are redirected into the table's per-slice
  roots.  Extra topics stay under the tab's own `anchorkeypath`, i.e.
  `<table>/tabs/<fqcn>/<key>/note/note.txt`.
- **Validity.** They count: `valid()` is the conjunction over every top-level
  topic, so a tab whose `note` is missing is unbuilt even if both slices
  landed.  Only the slices get the non-empty-`index.json` rule.
- **Reading.** `__read__` answers `data` (and, on a table, `tabs` and `done`);
  anything else raises `NotImplementedError` naming the topic, for the
  subclass to override — the same contract as a plain Datablock.

`TOPICS` **accumulates** down the hierarchy here, rather than shadowing as a
plain class attribute would: a subclass declaring `TOPICS = {'note': ...}`
adds to what it inherits, and its own entries win on a collision.  That is the
whole mechanism by which a table keeps the `tabs` and `done` topics
`DatastreamTable` declares — no second class attribute, no registry.  The
`data` group is the one exemption: it is always rebuilt from `SLICES`.

Redeclaring `done` (say, as a different filename) is harmless.  Redeclaring
`tabs` as a file topic is not: it is the `url=` every tab is formed under.

## Implementing a Table

```python
class FrameTable(DatastreamTable):
    VERSION = 1
    TAB = FrameTab                     # SLICES are inherited from it

    @dataclass
    class VAR(DatastreamTable.VAR):
        episodes: list = None

    @property
    def n_tabs(self):                    # REQUIRED
        return len(self.var.episodes)

    def __tab__(self, idx):              # only if a tab needs a spec
        return super().__tab__(idx, episode=self.var.episodes[idx])
```

**Required:** `TAB`, `n_tabs`.
**Optional:** `__tab__(idx)`, `__stats__(slice)` — and `__split__` /
`__stack__`, which are preimplemented and should be extended via `super()`
rather than replaced.

`DatastreamTab.VAR` declares `table` and `tab_idx`, both **required**: a tab's
slices live in its table's per-slice roots, so a tab without a table is one
whose shards no merged index will ever name.  Form tabs with `table.tab(idx)`
rather than constructing them directly.  (Requiring them rather than defaulting
them is also what leaves a subclass free to default *its* fields — a dataclass
forbids a non-default field after a defaulted one.)

`__tab__` is `Datastack.__block__` for tables — the same hook, at the same
point, running inside the worker for the same reason.  Nothing about a tab is
special; what `super().__tab__` adds is only the placement a tab cannot be
correct without: the url under `tabs`, `table`/`tab_idx` in the spec, and the
storage and cache settings inherited from the table.  A subclass passes the
tab's *own* VAR fields and leaves the rest alone.  Because it runs in the
worker it must read off `self.var`, or off something `__split__` wrote to
storage — see [DATASTACK.md](DATASTACK.md) for what survives pickling.

A table whose tabs need no spec of their own does not implement it at all.

Declaring `SLICES` on the table as well as on the tab is an error unless they
agree: the two share one set of slice roots, so a slice only one of them knows
about is one the tab writes and the table never merges.

## Storage layout

A tab's shards do **not** live under its own `anchorkeypath`:

```
<table anchorkeypath>/data/<slice>/index.json    <- merged index, one per slice
<table anchorkeypath>/data/<slice>/<key>/        <- one tab's shards for it
<table anchorkeypath>/tabs/<fqcn>/<key>/         <- that tab's other topics
<table anchorkeypath>/done                       <- written last
```

`StreamingDataset` resolves a shard as `os.path.join(root, split, basename)`,
so a slice's merged index must sit at an **ancestor** of that slice's shards.
Several slices therefore cannot share one directory, and `'../'` is not an
option because Azure Data Lake's REST API does not resolve it.
`DatastreamTab.dirpath` implements the redirect; non-slice topics are untouched
and stay under the tab's own key.

Both `<key>` above are the same string — `tabdir` is just the tab's `key`, so
its shards and its own topics are addressed by one identity, and the table
controls both through the `tag=` it passes to `__tab__`.

### Why not `anchor=`

Placement cannot go through the tab's anchor instead.  A tab has **one**
`anchorkeypath`, so its slices are always siblings under it — and then no
directory is an ancestor of one slice without being an ancestor of the others
too, which is exactly what a per-slice merged index needs.

The two soundworld reels show both halves of this.  `PoseGridReel` writes one
stream per cell, so it has one index, and it hoists that index *up* to the
reel's own `anchorkeypath` (a `dirpath('index')` override) where it is an
ancestor of every cell — no per-cell override needed.  `PoseAggReel` writes
three, so three indexes cannot share one directory, and it pushes each cell's
shards *down* into `<reel>/<stream>/cell_<id>/` (a per-cell `dirpath`
override).  This scaffolding is the general case, so it does what `PoseAggReel`
does; a single-slice table simply has one slice root instead of several.

`__stack__` merges every tab's per-slice `index.json` into one index per
slice, rebasing each shard's `basename` onto the slice root, and then writes
the `done` marker.  Shards are concatenated in tab-index order, and the same
order is used for every slice — which is what carries the tabs' per-sample
lockstep up to the table.

## Validity

A slice is built only when its `index.json` exists **and names at least one
shard**.  `MDSWriter.finish()` writes an `index.json` even when nothing was
written through it, so existence alone would let a tab whose every input was
unreadable report built and contribute an empty stream — which surfaces much
later, and much less clearly, as `Stream contains no samples`.

`table.valid()` is the `done` marker alone: it is written last, after every
slice index is merged, so it is the only thing that means *this table is
readable* rather than *some tab got that far*.

## Reading

```python
table.build()

table.data('annotations')             # list[dict], every tab concatenated
table.data()                          # {slice: list[dict]}
table.read('data', 'annotations')     # same, through the topic protocol

table.dataset()                       # every slice, zipped by index
table.dataset('frames')               # frames only — no annotation bytes fetched
table.dataset('frames', 'annotations')

table.stats('frames')                 # your __stats__ over the tabs' stats

tab = table.tab(3)
tab.data('frames')                 # just this tab
tab.dataset()                      # just this tab, zipped
```

`data()` reads tab by tab rather than through the merged index: the merged
index names shards in per-tab subdirectories, and `read_mds_shard` stages a
remote shard directory flat, which would collide those names.

`dataset()` always returns a `ZipStreamingDataset`, even for a single slice,
so the sample shape does not depend on how many slices were asked for.  The
slices are opened unshuffled and zipped by *physical* index, which is what
keeps them aligned — shuffle at the `DataLoader`, over the zip.

### Merge policy

Slices written in lockstep normally carry the same bookkeeping keys, so
merging them is a real decision rather than a dict update.  `dataset()` takes
the policy and hands it to `ZipStreamingDataset`:

```python
table.dataset(                                  # frames + only two annotation columns
    columns={'annotations': ['label', 'case_id']},
    shared={'idx'}, validate_shared=True,       # loud if the slices drift apart
    on_conflict='error',                        # any other collision is a bug
)
```

- `columns={slice: [...]}` projects a slice down, keyed by slice name rather
  than position.  A caller pays for the annotation columns it names and no
  others.
- `shared` names the keys that are *expected* in more than one slice; the
  first slice's value wins, and `validate_shared=True` additionally asserts
  they agree, which is what turns a mis-zipped table into an error instead of
  a silent misalignment.
- `on_conflict` (`'last'`, `'first'`, `'error'`) governs everything else.

The defaults — `on_conflict='last'`, `skip_none=True`, no projection — are the
plain merge `ZipStreamingDataset` has always done, so nothing already using it
changes behaviour.

## Plug-in points

Everything a subclass is meant to touch, and nothing else.  The `__dunder__`
methods are the override points, matching the `Datablock`/`Datastack`
convention (`__build__`, `__read__`, `__block__`); a leading underscore means
internal, and a plain name means it is for calling, not overriding.

### `DatastreamTab`

| | | |
|---|---|---|
| `SLICES` | **required** | the parallel streams this tab writes |
| `VAR` | | on top of the declared `table` and `tab_idx` |
| `TOPICS`, `VERSION` | | as on any `Datablock` |
| `__build__()` | **required** | write every slice in lockstep, via `slice_writers` |
| `__stats__(slice)` | | needed only if you call `stats()` |
| `__read__(*topicpath)` | | needed only for topics besides `data` |
| `tabdir` | | directory under a slice root; defaults to the tab's `key` |

### `DatastreamTable`

| | | |
|---|---|---|
| `TAB` | **required** | the `DatastreamTab` subclass; `SLICES` follows it |
| `VAR`, `VERSION` | | as on any `Datastack` |
| `n_tabs` | **required** | how many tabs |
| `__tab__(idx)` | | tab *idx*'s own VAR fields; `super()` fills in placement |
| `__stats__(slice)` | | over `[self.tab(i).stats(slice) for i in ...]` |
| `__read__(*topicpath)` | | needed only for topics besides `data`/`tabs`/`done` |
| `__split__()`, `__stack__()` | | extend via `super()`; both are implemented |

### Call, don't override

`data()`, `dataset()`, `datastream()`, `stats()`, `slices`, `valid()`,
`valid_slice()`, `slice_index_path()`, `cacheroot`, `tab()`, `tabs()`,
`build_index()`, `TabMaker`, `TabIndexFetcher`.

`slice_writers()` is the one method meant to be *called* from a subclass's
`__build__` rather than by a consumer — public in the same sense
`Datablock.path()` is.

## Operational parameters

`cache` and `cache_limit` are `__init__` kwargs on both classes, not VAR
fields: they say where local scratch lives and how much of it is kept, and
neither changes a byte of what is written or read, so neither belongs in the
hash.  A table passes both down to every tab it forms.

`cacheroot` is the resolved form of `cache`, named for it the way dbx's own
`localroot` is named for `local`.  It defaults to `<localroot>/streaming` —
under the block's local staging root (`local=`, `DBX_LOCAL`, or the url itself
when that is already local), **not** `/tmp`, so a cache big enough to matter
lands on the disk the deployment chose for it.  Under it go shard downloads
(bounded by `cache_limit`), staged writes on their way to remote storage, and
decompression scratch — no data, and safe to delete at any time.

`cache` is stored raw and resolved lazily, so leaving it unset keeps it unset
in the block's handle and journal instead of baking one machine's absolute
path into them.

```python
table = FrameTable(
    url=dbx.env('DATA_ROOT'),
    spec=dict(episodes=[...]),
    parallelization='multiprocessing', n_workers=8,
    cache='/scratch/streaming', cache_limit='10gb',
)
```
