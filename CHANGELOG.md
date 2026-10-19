# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **`Datablock.update_tree(note)` and `touch_tree(note)`** run `update` or
  `touch` on every block of the tree, bottom up, each block once. They skip
  the subtrees `build_tree` skips (`TREE_SKIP_BUILDING`, unless `skips=`
  says otherwise), and `exclude_self=True` leaves the top block out. Every
  entry carries the top block's `tree` id, so they are found together.
- **`Datablock.walk_tree(skips, exclude_self=, adopt=, skip_callback=)`:**
  the reusable walk behind them. It yields `(path, block)` for every block of
  the tree, bottom up and each once, lazily, so a block is dealt with before
  its parent is reached. `try_specialize_tree` now walks with it.
- **`Datablock.touch()`: a `note` under the event `touch`**, as `update()`
  is under `update`. `touch` is a construction event, in `CONSTRUCTED_EVENTS`
  and `PipeJournal.ENTRYPOINT_EVENTS`, so a touched block counts as
  constructed.

### Changed
- **Journal files are read without Arrow's thread pool, and a compacted journal once per
  process.** A specialization pass hung for good in one worker: reading an anchor's
  compacted journal with a hash filter, through an open file, Arrow waited on a task no
  thread of its pool held. Every journal `read_parquet` now passes `use_threads=False`
  (journal files are KB to MB; threads gain nothing). `BuildJournal.read_hash` reads a
  compacted journal whole, once per process -- by its file: path, size, modification
  time, so one compacted again is read again; the 16 most recent are kept -- and filters
  by hash in memory, rather than reading the file again for every past of every tablet.
  The rows are those the filtered read gave.
- **A core partition's layout is read once per process, shared, and memory-mapped.**
  `DatatableClusterCorePartition._layouts_` -- what `lot` and `lot_tablets` read --
  cached the layout on the instance, and each core tablet evaluates instances of its
  own: forming ten tablets of a taps clip read one 12.9 MB layout 50 times and kept 34
  copies, 439 MB. It is now shared by every instance, by its file (path, size,
  modification time, so a layout written again is read again), read-only, and
  memory-mapped: from the lake when it is local, from a copy pulled into local staging
  when it is not. Ten tablets now read 3 MB and add 62 MB, not 478. `layout(part)` still
  reads the whole array, as before.
- **A quote writes its nested blocks bare: nothing in it is escaped, at any depth.** A
  spec value that is a `$` specline -- a nested block's quote, or `$dbx.getenv('X')` --
  is written `'upstream': fqcn(...)`, not as a string literal, in `quote()` and
  `repr()` (`databits.bare_specline`). As a string, it was escaped once more at every
  level of nesting, every backslash doubled: a feature tablet twelve levels deep had a
  5 MB quote, 98% backslashes, now 113 KB with none. `dbx.eval` reads a bare value
  back as the specline string it was (`databits.quote_spec`), and parses a datablock
  constructor's arguments once per process, by their text. Forming a tablet of such a
  clip went from 8 s to 1 s. No identity changes: a hash is computed from signatures,
  which are unchanged. Quotes recorded the old way evaluate as before; one recorded
  now does not evaluate under an earlier dbx (`inst(revision=...)` of an older one).
  `cite_specline` lays out bare nested blocks as it does quoted ones.
- **`slice_columns(slice)` and `slice_schema(slice)`, renamed from `declared_columns` and
  `declared_schema`.** The columns a slice declares, as the MDS writer takes them (a
  dict column as `'json'`), and as declared (a dict column's structure). The old names
  remain as deprecated aliases.
- **A Featuretable with fewer CPUs than GPUs raises before its workers start.** Its
  device workers -- one per GPU -- need a CPU each to feed their GPUs; sharing fewer
  (a Slurm job given `--gres=gpu:3` and no `--cpus-per-task` has one) they take turns,
  and the GPUs run one after another. `_check_devices_` now raises a `RuntimeError`
  naming the CPUs `os.sched_getaffinity` allows and the fix.
- **`ModelEvaluatorBuilder.evaluator()` caches per process, not per builder.** The key is
  `(type(builder), builder.hash, device)`, so the builders of a feature table's tablets --
  one each, evaluated from the table's spec -- share one evaluator, and a process loads
  its model once per device instead of once per tablet. The class is in the key because
  an identity names none: two builder classes with the same spec have the same hash.
  The cache holds each model for the life of the process;
  `ModelEvaluatorBuilder.evaluator_cache_clear()` empties it (a subclass's, its own
  entries). The `Evaluator` is constructed with `model` moved to *device*, `transform`
  and the new `evaluator_kwargs` property, which `TransformerEvaluatorBuilder` extends
  instead of overriding `evaluator()`. No identity changes.
- **A failed build records why.** Its `build:exception` (or `build:keyboard_interrupt`)
  entry holds `exception` -- `'<Type>: <message>'`, as the exec journal has it -- and
  `traceback`, written in the process the build failed in, a worker's too: the
  cause is in the journal whether or not output was captured.
- **`BuildJournalEntry.block` is `.pretty`, and `Block` is `Pretty`.** The view of an
  entry as the block it records; `Datastack.block(i)` is unchanged. `Pretty.quote()`
  lays the recorded quote out as `cite()` does -- a keyword a line, nested blocks
  indented, nothing escaped (`cite_specline`); `quote(pretty=False)` is the
  evaluable text as recorded, which `to_dict()` exports.
- **No entry records a `dfn`.** It said nothing the quote, the spec and the kwargs do
  not; entries from before keep theirs, and PipeJournal compaction their files.
- **`use_block_specializations=False` looks for no past.** A stack with it off no
  longer enters its "ATTEMPTING SPECIALIZATION" pass -- no journal handed to its
  blocks, no lookup per block -- but classifies its invalid blocks and builds them:
  what a run whose blocks are known to have no earlier build of themselves asks for.
- **`BlocksJournal` is gone: a stack hands its blocks a `BuildJournalFrame`.** The
  build journal a stack reads once for its blocks -- or `BY_HASH` -- travels to the
  workers forming them as the frame itself, its new `read_anchor` and
  `read_datalake` saying whose it is (a block of another anchor, a tablet's
  upstream, reads its own) and `by_hash` standing for `BY_HASH`.
  `Datastack._blocks_buildjournal_(journal=None)` is the one way to get it, read or
  given; `_shared_blocks_journal_` is gone, and so is its `(journal, anchor, url)`
  tuple. `dbx.eval` returns a DataFrame or Series as it is: iterated, it rebuilt one
  from its column names -- which is why the frame had been wrapped. A stack's block
  methods refuse a `dict` or `list` journal, which no single frame can hold.
- **`DataJournal` is `BuildJournal`: the journal of what happens to a block's data.**
  Builds, clears, copies, redirects and adoptions, notes: entries keyed by
  `(anchor, hash)` and filed under the block. A clean break, with no aliases:
  `DataJournal` → `BuildJournal`, `DataJournalFrame` → `BuildJournalFrame`,
  `DataJournalEntry` → `BuildJournalEntry`, `DEFAULT_DATAJOURNAL` →
  `DEFAULT_BUILDJOURNAL` (and the other `*DATAJOURNAL*` names), `datajournal()` →
  `buildjournal()` -- the module function, `Datablock.buildjournal`,
  `ExecJournalEntry`/`ExecJournalFrame.buildjournal`, `Datablock.BuildJournal` --
  `Datastack.block_datajournal` → `block_buildjournal`,
  `child_specialization_datajournal` → `child_specialization_buildjournal`,
  `Datatable.tablet_datajournal` → `tablet_buildjournal`, and every other
  `*datajournal*` helper to match (`JournaledResult.buildjournal_entries`,
  `CARRIES_BUILDJOURNAL`, ...). A subclass defining a builder under an old name is
  refused, as before. On disk only the exec journal's column changes:
  `datajournal_entries` is `buildjournal_entries`, and a row written under the old
  name -- or `written_entries`, before it -- reads under the new one. Entry files,
  `.journal/` directories and session markers are named for no class, and do not
  change. An exec row recorded verbatim calling `.datajournal(...)` reads as it is,
  but no longer runs again.
- **A block the tree never builds is adopted, when a past of it resolves.**
  `build_tree` specializes a `TREE_SKIP_BUILDING` child that is not valid -- whatever
  `deep` -- and every such child below a child it skips as valid (`_adopt_exempt_`): an
  adoption computes nothing, and a still an evaluator reads, moved to a new identity,
  is valid again in every process that reads it, rather than in none.
  `UNSAFE_specialize` installs what `find_specialization` finds and `build()` would
  install -- a past composed with its nested blocks' pasts too -- not only a declared
  specialization.
- **A journal entry is found by what it is -- its `id` -- never by where it is filed.**
  Every `DataJournal.write` keeps its entry: a file of its own, named by its `id`, so
  a build leaves its `build:start` and its `build:end`, and the `journal_prefix`es
  dbx passed to keep one write from overwriting another are gone. A session records
  `DataJournal.entry_ref`s -- `anchor`, `hash`, `id`, and the `version`, `tag`,
  `keyby` and `datalake` that finding a block's entries by hash takes -- in
  `written_entries()`, its index's markers (named by `id`) and an exec-journal row's
  `datajournal_entries`; `DataJournal.read_refs` reads them back from an entry's own
  file or, compacted since, its anchor's compacted journal. An exec row from before
  names paths: one whose file is gone is its block's entries under the row's
  session. `journal={'id': ...}` selects an entry where `{'entry_path': ...}` did.
  `UNSAFE_compact` compacts redirection entries too; `PipeJournal.UNSAFE_compact`
  keeps the side files its kept entries name -- in their columns -- not the ones
  whose file names share their instance's `dt`.
- **`read_hash` reads a compacted journal too.** A block whose anchor was compacted
  was "never recorded" to `_past_recorded_`, and no past nested in it composed.
- **A pullback of whole tablets builds no map.** Every tablet taken whole, no lot is
  read: each tablet is its table's own, from the partition's entries. Its map is not
  built, nor asked for by its validity (`TREE_SKIP_BUILDING`/`TREE_SKIP_VALIDATION`,
  per instance); another table laid out alike is pulled back along it all the same.
- **Subtree banners name whose field they are:** `BUILDING SUBTREE at upstream of
  <anchorkey> [<tag>]`.
- **A pipeline is created once, journaled only when it says so, and cached by its call.**
  `Pipeline.create()` -- and so `entrypoint`, `entry`, `entrypath` and
  `read('entrypoint')` -- arranges the tree once per instance, not afresh on every
  access. `Pipeline.JOURNAL`, False by default, says whether that creation is written
  to the `PipeJournal`; `@pipeline(journal=True)` sets it. An `@pipeline` call is cached
  by ALL its arguments (dicts, lists and blocks by value; an omitted `datalake=` by
  `DBX_DATALAKE` at the call), so the pipelines several others nest are one instance,
  created and journaled once per process; `f.cache_clear()` empties it. Until now every
  `.entry` on every fresh instance wrote a 12-file entry: hundreds per command for one
  nested collator. Pass `journal=True` to the pipelines whose entries you want.
- **`PipeJournal.UNSAFE_compact()`: one entry per pipeline and entrypath, the rest removed.**
  The newest entry of each distinct `entrypath` is kept, with its side files; `collect`
  reads the same pipelines and entrypaths after. `unjournaled=True` also removes the
  journals of pipelines whose class says `JOURNAL = False`, but those in `spare=`;
  `dry_run=True` says what would go. Asks first, as every `UNSAFE_` does (`OVERRIDE=True`).
- **`DataJournal.UNSAFE_compact(anchors)` and `ExecJournal.UNSAFE_compact()`: a journal
  as one parquet file.** Every entry file of an anchor -- `dbx.anchors()` by default --
  is read and written back, row for row, to `<anchor>/.journal/<anchor>/journal/
  <anchor>-journal-compacted.parquet`, where `read` already finds entry files; the
  exec journal to `.journal/exec/exec-compacted.parquet`. Nothing is deduplicated.
  Anchors go to a `parallelization` executor (`'multithreading'` by default) of
  `n_workers`; `dry_run=True` touches nothing. A compacted entry's `entry_path` is the
  compacted file; a session finds it by reference all the same (above). `ExecJournal`
  is new: the exec journal's directory and its compaction.
- **Every tablet is filed by its hash, and tagged by its table: no tag in any path.**
  `Datatablet.KEYBY` is `version_shorthash`, as every block's. A tablet filed under its
  tag until 2026-10-17 -- hundreds of thousands, in tables valid as they stand -- is
  read where it was filed, its own redirection there followed: a tablet not built where
  it is filed now, nor redirected, looks where its tag filed it (`_tags_filed_under_`:
  its tag; a mapped tablet's, its partition's `tag_of`; a map tablet's,
  `map<hash>_<i>`). Nothing is built again. A map tablet and a mapped tablet are tagged
  `<table's tag>:<i>`; a part's map `<partition's tag>:part<k>`; a pullback layer not
  tagged otherwise `<its table's tag> along <its map's tag>`.
- **Two blocks of a stack at one path are refused before either is built.** Filed by
  hash, tablets that differ only in their tags -- nothing in their identity telling
  them apart -- would be built and read as one; a stack's build raises, naming them.
  Give each what tells it apart, its index say, in its spec. An error naming invalid
  blocks names each by its tag.
- **`Specialization(tag=...)`: a past filed under a tag** -- a block keyed by tag is
  looked for under its tag, and this names the tag its past was filed under, as
  `anchor` does a renamed class.
- **Log names show a tag without its entrypoint's module** (`shown_tag`): `f:NAME`
  for `pkg.mod.f:NAME`. The tag itself, the journals and the paths keep it whole.
- **`BUILD ELIDED` is a VERBOSE line.**
- **`TREE`, a log level of its own: the `BUILDING`/`SKIPPING SUBTREE` banners.**
  `DBX_LOG_TREE=1`, `Datalog(tree=True)` or a `with Datalog(tree=...)` scope shows
  them without every VERBOSE line; set nowhere, they follow `VERBOSE`, as before.
- **Log lines: `main:12:------ <time>`.** A worker's name is dashed out to the
  timestamp, not spaced, and its parts are joined by `:` -- a process by its bare
  number (an executor's `p12` is `12`), a thread as `t<i>`: `main:12`, `main:t0`,
  `main:12:t1`.
- **A part's map and a pullback's layers carry tags, and log lines name tags.** A
  map is tagged `<partition's tag>:part<k>`; a pullback layer not tagged otherwise,
  `<its table's tag> along <its map's tag>` -- the top layer takes a tag given to
  `pullback`. A tag is in no identity, and no tablet's tag moves. `PARTITION`, `MAP`
  and `PULLBACK` lines name the blocks they mention with their tags.
- **Devices are checked as features are evaluated, not as `build` starts:** a
  feature table that adopts an earlier build needs no GPU.
- **`DatatableCorePartition` is `DatatableClusterCorePartition`.** A core partition is
  named for how it chooses its rows; another way would take other arguments, and be a
  class of its own. Its specialization reaches what was built at the old class's path,
  and its hash is pinned. `DatatablePartition._DealVAR_` is `_PartitionVAR_`.
- **Fixed: the core partition's hash, moved by a reworded topic description.** The
  ``layout`` topic's description -- part of its identity -- is restored as built.
- **A part is its map; `DatatablePart`, `FeaturetablePart` and `fold` are gone.**
  `partition.part(k)` is part *k*'s `PartitionDatamapTable`, VAR `{partition, part}`;
  `part(k).pullback(table, collator)` is a table read along it, a `MappedDatatable`
  or `MappedFeaturetable`, whose TOPICS are `Datatable`'s own, not its table's.
  `fold(k)`, `map(k)` and `tablets()` are gone, `n_folds()` is `n_parts()`, and
  every `fold` parameter is `part`. What a `DatatablePart` or `FeaturetablePart`
  built is reached through `PART_PAST`: the pullback rendered as the part's VAR
  `{partition, fold}`, its upstream `determined` by its map, table and collator.
- **No more "deal": a partition is partitioned, and assigns.** `dealt()` is
  `partitioned()`; `_dealt_tablets_` is `_assignments_`; `_deal_`,
  `_cluster_and_deal_` and `_deal_rows_` are `_assign_`, `_cluster_and_assign_` and
  `_assign_rows_`; the log says `PARTITION <partition>: partitioned`.
- **A block whose fields a past moves out still composes the pasts nested in
  them.** A dotted move -- `PART_PAST`'s `datamap.partition` -- renders that field's
  own fields as they are now, and its nested blocks may still render as they were:
  its pasts that rename and pin nothing. A `PartitionDatamapTable`'s past that
  changes only how its partition renders is real when the partition's is, though no
  map was built before 2026-10-16. A part built on a partition whose table changed
  identity finds its past again.
- **The most recent past wins.** `find_specialization` and `build` install the
  most recently recorded of the pasts that resolve, ties in declaration order: a
  later adoption of a build, redirecting to it, over the build itself.
- **Each layer of a pullback says what it copies, as it is built.** In build
  order, bottom up: `PULLBACK <table> along fold k of <deal>: n copies of
  [slices]`, next to its `COPY: n tablets of <table>` bar -- the layers pulled back
  independently along one map, a skipped layer saying nothing. The map says the
  deal and itself once, as it builds: `DEAL <deal>: dealt; MAP <map>: fold k, n
  lots`. The single summary with its arrows between layers is gone.
- **`partition.map(k)` and `fold(k)`; a mapped feature table holds its upstream.**
  `map(k)` is fold *k*'s map -- `DatamapTable` and `DatamapTablet` are
  `PartitionDatamapTable` and `PartitionDatamapTablet` -- and `fold(k)` its own
  table pulled back, the identity every fold built hashes through; `part(k)` and
  `datamap(k)` are gone. `map.pullback(table, collator)` builds the stack from the
  bottom: a `MappedFeaturetable`'s VAR ``upstream`` is the pullback of the nearest
  layer below it copies, so `map.pullback(ft).upstream == map.pullback(ft.upstream)`
  and so for every copy, and `build_tree` builds it bottom up, each layer in
  parallel. A copy no longer builds its upstream copy: its table builds the layer
  below first. A part of whole tablets builds no upstream. Every
  `MappedFeaturetable` moves hash, the CPTAC 602020 cores' among them.
- **Every layered block answers `.upstream`, and the mixin is `Layered`.** The
  block a table or tablet reads upstream of is `.upstream`: its VAR's
  ``upstream`` for a `Featuretable(t)` or `MappedFeaturetablet`, and, for a
  `MappedFeaturetable` or a `FeaturetablePart` -- whose map and source already
  fix it, so it is in no identity -- its source's upstream pulled back along the
  same map. `upstream_mapped` is gone; `.upstream` replaces it. The mixin
  `DataslicesUpstream` is `Layered`; it is no block's class, so no anchor or
  hash moves.
- **A partition's stages say what they are.** A deal's build opens with a
  `DEAL` line naming its stages; a mapped table's with a `PULLBACK` summary --
  the deal, dealt and kept or not; the map, built or to build; how many tablets
  are copies, and each layer's slices -- and the progress bars read `MAP:
  resolving n lots ...` and `COPY: n tablets, layer by layer`. A stack names its
  stage by overriding `Datastack._build_tag_`. A table of copies no longer
  searches for earlier builds of them: its block class's specializations are its
  source tablets', and no copy is one. DATAPARTITION.md opens with every class at
  a glance.
- **A map is a real table.** `DatamapTable` builds its `DatamapTablet`s, in
  parallel with its deal's workers, each storing its lot -- resolved once from the
  deal -- as its `lot` slice; `DatamapTablet.lot` reads it, and raises while it is
  not built. A copy builds its map tablet first; a mapped table naming a map
  builds it first. The deal answers `lot_tablets(fold, i)` -- the tablets a lot
  draws on -- without resolving a row, for what a copy needs before anything is
  built. A part of whole tablets builds no map.
- **One map for every deal, and a fold tablet's rows are its lot.** A deal answers
  for its lots -- `DatatablePartition.lot(fold, i)`, `whole`, `tag_of`, `dealt` --
  in the form it wrote them, a core deal's a run of its layout; `DatamapTable`
  only asks, and `DatamapTablet.lot` is the resolved `[(tablet, rows | None)]`
  (`record` until now). A pullback to another table is a `MappedDatatable` whose
  VAR names it, its copies checking it is laid out alike. `DatamapCoreTable`,
  `DatamapCarbonTable`, `DatatableCorePart` and `FeaturetableCorePart` are gone;
  `DatatableCorePartition.part(k)` is `datamap(k).pullback(table, collator)`. A
  mapped table has `source_index()`.
- **A core part is its table pulled back along the fold's map.** A core fold's
  map is a `DatamapCoreTable`, each core tablet a run of the layout; a
  `DatatableCorePart` -- a `FeaturetableCorePart` for a feature table, keeping
  its layers -- copies what the collator names, along it. A carbon copy is
  `core.datamap(k).pullback(table, collator)`, a `DatamapCarbonTable` over a
  table laid out alike. `DatatableCoreTablet`, `DatatableCarbonCorePartition`,
  `core_tablet()` and `carried_declarations()` are gone. Core partitions keep
  their hashes and layouts; core parts are rebuilt. A mapped table's `slices()`
  are what it holds: its collator's, and `source` when every tablet is a copy.
- **A part of a feature table is a `FeaturetablePart`**, a `DataslicesUpstream`
  whose upstream is the feature table's upstream pulled back along the fold's
  map, so the part routes an upstream slice -- an annotation -- as the feature
  table does. Generally, a table that reads upstream pulls back to a
  `MappedFeaturetable`. Its hash is a `DatatablePart`'s, as it was; its path is
  its class's, and a part built as a `DatatablePart` is adopted where it is by
  `FeaturetablePart`'s specialization.
- **A feature table is cut and packed, layer by layer.** A piece or packed
  tablet of a `Featuretable` is a `MappedFeaturetablet`: a copy of its own
  slices' rows, whose `upstream` is the same rows of the upstream table, pulled
  back along a `DatamapCarbonTable` -- a fold's map over a table laid out alike,
  checked record by record. `source` is written once, at the bottom. The
  `NotImplementedError`s for a mixed feature tablet and for `coalesce='pack'` of
  a feature table are gone. A partition column read as a numpy scalar is
  recorded as the Python value it is.
- **A part is its table pulled back along its fold's map.** New in
  `dbx.dataparts`: `DatamapTable`, a fold's layout, a `DatamapTablet` per tablet
  of the fold, each with its `record`, `[(tablet, rows | None)]`; and
  `MappedDatatable`, a table pulled back along a map, whose tablets are the
  mapped table's own where the record takes them whole, and otherwise
  `MappedDatatablet`s copying the rows. `DatatablePart` is a `MappedDatatable`
  whose map is derived from its VAR `{partition, fold}`, so every part, and
  everything built on one, keeps its hash. `DatatabletPiece` and
  `DatatablePartitionTablet` are gone, both now `MappedDatatablet`s:
  pieces and packed tablets are rebuilt. `DatatablePartition.tablet(entry)` is
  gone; `tablets(fold)` reads its part's. See DATAPARTITION.md, "Parts are
  tables pulled back along maps".
- **A partition's parts build with its workers.** `part()` of a
  `DatatablePartition` and of a `DatatableCorePartition` (a carbon one too)
  gives the part the partition's `parallelization` and `n_workers`, so its
  pieces, packed tablets or core tablets are built in parallel, as the partition
  scanned. Each part is built by its own `build()`, one after another. Workers
  are no identity: no hash moves.
- **`DatatableCorePartition`'s `layout` is retired** for
  `DatatableCarbonCorePartition`, and refused when given. It stays in the VAR,
  always None, so that no core partition built moves its hash.
- **hyp switches journals in place, and filters an anchor's.** A **journal**
  menu in the top bar lists the pipelines and every anchor of the datalake.
  A **filter** field, shown for an anchor, takes `key=pattern, ...` and passes
  it to `DataJournal.read`; `--filter` sets it at the start. Each journal
  keeps its own corpus. Points a filter hides keep their embeddings and are
  laid out without them, and a deleted `--docs` file leaves the disk the same
  way. **reload** moves to the top left. `HypDbxPipelines` is now
  `HypDbxJournal`, one class for both journals.
- **`hyp --journal [ANCHOR]` replaces `hyp --pipejournal`.** With no anchor,
  or `pipelines`, it is the pipelines, as before. With an anchor, it is that
  anchor's blocks from its block journal, one point each (`HypDbxJournal`).
  A click opens the block's window, and a right click lists its session's
  blocks. Its corpus is kept under `{DBX_DATALAKE}/.hyp/anchors/<anchor>`.
  `--pipejournal` is gone, with no alias.
- **`update` is a construction event**, in `CONSTRUCTED_EVENTS` and
  `PipeJournal.ENTRYPOINT_EVENTS`, and **`UNSAFE_copy_from:END` counts for a
  pipeline's entry blocks**, as it did for `constructed()`: the two lists
  name the same events.

### Added
- **`Specialization(determined={field: (determining field, ...)})`**: VAR fields
  the narrower block never had, left out of the past rendering whatever they
  hold, because the fields named fix them -- which the past rendering keeps, or
  pins. A pin leaves a field out only at the value it names; a field holding a
  block that varies with the rest of the VAR -- a mapped feature table's
  `upstream`, its source's upstream pulled back along its map -- has no one value
  to name. The declarer states the dependency: no field both pinned and
  determined, none determined by a field itself left out, every one named this
  class's VAR field (raising otherwise, as a pin does). Recorded and keyed only
  when set: no specialization from before keys differently. Blocks built on one
  find their pasts through it, as through any nested past.
- **`DatatableCarbonCorePartition`: a core partition's layout, over a table
  laid out as its table is, other columns copied.** A feature table and the
  table it was computed from, say. Its VAR is the `datatable`, the `template`
  (any core partition, a carbon one too), a `collator` naming only `carry`, and
  `rows_per_tablet`, the template's by default. Nothing is clustered or dealt;
  the tables are checked to be laid out alike, tablet for tablet and row for row.
  See DATAPARTITION.md, "A carbon copy".
- **`build_tree(deep='try_specialize_children')`: a valid child vouches for
  itself, not for its children.** Its children -- its VAR subtrees and, a
  stack's, its blocks -- are tried for specializations, as
  `try_specialize_tree()` tries them, and nothing is built. A child that moved
  to a new hash under a parent that kept its own adopts its earlier build,
  rather than being found unbuilt by whatever reads it. See DATASTACK.md, "A
  Valid Stack and Its Children". Any other `deep` but `False` or `True` is
  refused.
- **`pipejournal(construct_only=True, extra_construction_events=[...])`, and
  `hyp --journal --extra-construction-events ...` (`--ece`).** Events
  that also make an entry block count as constructed, besides `build:end`,
  `UNSAFE_redirect` and `note`. Each matches an event alone, or followed by `=`
  or `:` and the rest, so `ckpt` takes a still that has checkpointed
  (`ckpt=<tag>`) but not finished building.
- **`hyp --docs [--dir DIR]`: a directory's markdowns and pdfs on the disk.**
  Local or any fsspec url; each document labeled by its name and modification
  time. A left click opens it in the center; a right click places it in a
  quarter or a half. Markdown is rendered with its tables, code and mermaid
  diagrams (through `mmdc`, when installed); pdfs page by page, by pdfium.
  Windows scroll both ways and zoom with + and -. The disk, the placed windows
  and the center window are layers, rearranged by the layers menu or Tab, so
  the disk stays navigable with windows open. New `[hyp]` dependencies:
  `markdown-it-py`, `pypdfium2`.
- **A build entry records `python` and `host`: the interpreter its build ran
  in, and where.** Revision pins a block's code; this names its environment,
  so that `trueinst()` from another machine can run it again in the same one.
  `hyp` starts its `SshRay` head in an entry's recorded `python`;
  `HYP_REMOTE_PYTHON` and `--remote-python` now only override it, and
  `python3` stands in for entries written before this.
- **`data(sample_fraction=, sample_seed=)`: a sample of a slice's rows.** The
  same rows of every slice -- `datatables.sample_rows` picks them -- so a
  row's slices stay together, and an upstream block's routed `data()` takes
  them too. A slice is still read whole; what comes back is the sample.
- **`dbx.SshRay`: a block's true instance, from another machine.** It starts
  a Ray head on a host over ssh for the session -- its own ports, its own
  temp dir, in a directory of its own -- and tunnels to its client port, so
  that `entry.trueinst(address=cluster.address, gitrepo=<the host's repo
  paths>)` returns a proxy here to the block, instantiated where its data
  are and pinned to the entry's revision. `SshRay.session(host, python=)`
  keeps one per host for the process. The head stops, and leaves nothing
  behind, on `stop()`, at exit, and when its connection drops. Ray here must
  be the version Ray there is.

### Added
- **`hyp`, an app.** A package of its own in this distribution, with the
  `hyp` console script and the `[hyp]` extra (in `[all]`, so `dbx.yml`
  installs it): texts embedded through Ollama and laid out on the Poincaré
  disk with hyperbolic UMAP, navigated by geodesic slides and hyperbolic zoom.
  `hyp --journal` makes the texts the pipelines of dbx's datalake
  (`DBX_DATALAKE`), from `dbx.pipejournal()`, each openable to its journal
  row, its other runs, and the journals of the entrypoints it constructed;
  their embeddings are kept in the datalake, under `{DBX_DATALAKE}/.hyp`. See
  [HYP.md](HYP.md).

### Fixed
- **`try_specialize_tree()` specializes the blocks of a valid stack too.** It
  asked only blocks that were not valid, so a valid stack's blocks were never
  reached, however many had moved under it.
- **A pinned interpreter makes no work repo, asked for a revision or not.**
  `gitwrkreposetup` returned early for `DBX_PINNED_REVISION` only when given a
  revision, so the import of `dbx.datablocks` in a pinned process went on to
  check its source repos for uncommitted changes -- and refused, though its
  code is the revision's. A pinned job's environment also takes
  `DBX_DIRTY_REPO_OK`, for a cluster whose own dbx predates the fix.
- **`SshRay`'s head imports dbx without making a work repo**, and one that
  cannot start -- no such `ray` -- is reported at once, with what it printed
  and the `ray` it tried, rather than waited out.
- **`trueinst()` gives the worker the datalake its entry recorded.**
  `remote()` passes this process's `DBX_*` variables to its workers, so a
  block whose quote leaves its datalake to the environment took the
  caller's -- from another machine, an `ssh://` url its own host could not
  reach. `rinst()` now sets the worker's `DBX_DATALAKE` to the entry's
  `datalake`; `remote(env=)` sets, overrides or (with `None`) unsets any.
- **An entry read over ssh reads its side files there.** Its quote, spec and
  the rest are recorded as its host's local paths, and were opened on the
  local disk -- so `trueinst()` of such an entry found no quote. A bare
  absolute path is now read on the `ssh://`/`sftp://` filesystem the entry's
  `entry_path` names, as the block view lists its paths; a global path
  (`abfss://`, `gs://`, ...) is never touched.
- **`remote()` from a machine with no `DBX_GIT_REPO`,** given the host's repo
  paths as `gitrepo=`, no longer refuses an off-host cluster.
- **Proxies through Ray's client.** Over `ray://`, an actor's handle is a
  `ClientActorHandle`, which `Remote` did not recognise: `trueinst()` returned
  the bare handle instead of a proxy to the block.
- **`import dbx` works without torch and mosaicml-streaming** again, as the
  package docstring promised: the data modules (`datatables`, `dataparts`,
  `backbones`, `featuretables`, `probes`) are imported only when both are
  installed. Pinned by `test_import_core`.
- **Journal files are listed a directory level at a time, in parallel.**
  `DataJournal.read` globbed `**`, one listing per round trip: over ssh, a
  pipeline journal of 233 entries took 77 s to list, and `dbx.pipejournal()`
  about 90 s; now about 8 s with `n_workers=16`. The siblings of `journal/`
  under `.journal/{fqcn}/` -- `spec/`, `dfn/`, `kwargs/` -- are no longer
  entered.
- **`fs_full_path` keeps an ssh host.** It gave `ssh:///home/...` for an
  sshfs path; now `ssh://[user@]host[:port]/home/...`, so `entry_path` names
  the machine.
- **`pipejournal(construct_only=True)` over a remote datalake.** Entrypaths
  are recorded as the datalake's host names them, and were looked for on the
  local disk -- so every one was dropped. They are looked for on the
  datalake's filesystem.

### Changed
- **A noted block counts as constructed.** `CONSTRUCTED_EVENTS` -- what
  `dbx.constructed()` and `ExecJournalEntry.constructed()` select -- and
  `PipeJournal.ENTRYPOINT_EVENTS` -- what `pipejournal(construct_only=True)`
  keeps an entrypath for -- both take `note`, the event `Datablock.note()`
  writes by default. Matched exactly: `note:checked` and the like still are
  not constructions.
- **`quote()` and `cite()` render a set `tag` before `spec`.** The tag names
  the block where the spec, often dozens of entries, says how it was made, so
  it is read first rather than found under the spec. An unset tag stays after
  the spec. Only the rendering moves: a quote still evaluates to the same
  block, and no hash depends on the order.
- **A block's tag is a label, not a location: `keyby='version_shorthash'` is
  the default.** It keys a block `version=<V>/<hash8>`, or `<hash8>` with no
  VERSION -- for an untagged block exactly the key `tag_version_shorthash`
  made, so no untagged build moves. A tagged block is keyed as if untagged,
  so tagging one already built finds it built. The default is the class's
  `KEYBY`, and `Datatablet.KEYBY` stays `'tag_version_shorthash'`: every
  tablet -- a table's, a piece, a packed or core tablet, an autopath
  `__tablet__` override's -- keeps its tag in its path, as built. A stack
  hands its keyby to its blocks only when it is not the stack's `KEYBY`, so
  a table at its default builds its tablets where `tablet(i)` reads them; an
  explicit keyby on a table still reaches them. Every quote spells its
  keyby, so a block reconstructed from a journalled quote keeps the path it
  was built at. **Moves:** a non-tablet block built with a tag under the old
  default is no longer found by the same constructor without
  `keyby='tag_version_shorthash'`. `DataJournal.read_hash` lists a tagged,
  untag-keyed block's directory both beside the untagged and under its tag,
  so a specialization still finds a narrower build of either layout by its
  directory. A pipeline's `PipeJournal` entries move the same way;
  `pipejournal()` scans, and finds both.
- **Tabs are tablets, in every name code uses.** `Datatab` is `Datatablet`,
  `Featuretab` (and `BipolarFeaturetab`, `TernaryFeaturetab`) is
  `Featuretablet` (and `BipolarFeaturetablet`, `TernaryFeaturetablet`),
  `DatatabPiece` is `DatatabletPiece`, `DatatableCoreTab` is
  `DatatableCoreTablet`, and `DatatablePackTab` is `DatatablePartitionTablet`.
  A table's `TAB`, `__tab__`, `n_tabs`, `tab()`, `valid_tab()`,
  `tab_indices`, `TAB_SPECIALIZATIONS`, `filter_built_tabs` -- every method,
  attribute and parameter -- say `tablet` now; a table class still defining
  `TAB`, `__tab__`, `n_tabs` or `TAB_SPECIALIZATION(S)` is refused when it is
  defined, rather than ignored. VAR fields: the piece's `tab` and `tab_index`
  are `tablet` and `tablet_index`, and `min_tab_rows`, `max_tab_rows`,
  `rows_per_tab` and `max_rows_per_tab` are `min_tablet_rows`,
  `max_tablet_rows`, `rows_per_tablet` and `max_rows_per_tablet`. The
  partitions' topic `tabs` is `tablets`, and `balance='tabs'` is
  `balance='tablets'`. **Not renamed, being on disk or in identities:** the
  topic's file `tabs.json` and its records' `"tab"` key, the `source` slice's
  `tab` column, a tablet's default tag `tab_000003`, the `tab_paths` topic, and
  the `TAB=` a table's identity renders its tablet class under -- so no
  table's hash moved for the word. The feature tables and their tablets reach
  what was built under the old class names, and a piece what was built with
  its old VAR names, through specializations; the partitions are rebuilt. A
  constructor state saved with `TAB_SPECIALIZATIONS` is still read. One old
  name stays, for reading only: `dbx.datatables.DatatableTab`, which speclines
  recorded before the rename name and must still evaluate; write
  `DatatableTablet`. The test modules `test_featuretab*.py`,
  `test_tab_stream_cache.py` and `test_datatable_tab_specline.py` are
  `test_featuretablet*.py`, `test_tablet_stream_cache.py` and
  `test_datatable_tablet_specline.py`.
- **A partition's roles are renamed: `groupby` is `stratifyby`, and what was
  `stratifyby` is `partitionby`.** `stratifyby` names the column whose values,
  strata, must not cross the parts' boundaries -- a patient's rows go to one
  part. `partitionby` names the column whose values, partitions, are each
  divided between the parts by the fractions -- every part takes its share of
  every cohort. The same for `DatatableCorePartition`, in `tabs.json` records
  (`{"tab": i, "stratifyby": v, "partitionby": v}`), and in `summary`, whose
  per-value counts are under `partitions` (were `strata`) and a core part's
  under `strata` (were `groups`). **A collator still naming `stratifyby` in
  the old sense now keeps its values together, silently**: rename it to
  `partitionby`. A collator naming `groupby` is refused, as any unknown role
  is. `DatatablePartition` and `DatatableCorePartition` are VERSION 2, so a
  partition built before -- whose collator may spell the same and mean the
  opposite -- is rebuilt, never taken for one built now; only `largest_first`
  partitions, with no collator, are reached. Pieces are kept: a piece is the
  rows of a tab holding a value, whichever role read it.
- **`dbx.dataparts` is now the partitions' module; the utilities it held are
  `dbx.databits`.** The standalone helpers -- I/O, callable executors, the term
  evaluator, git-revision pinning -- moved from `dbx.dataparts` to
  `dbx.databits` unchanged: `from dbx.databits import callable_executor`. Then
  `DatatablePartition`, `DatatablePart`, `DatatabPiece`, the new
  `DatatablePackTab`, and `DatatableCorePartition`, `DatatableCorePart` and
  `DatatableCoreTab` moved from `dbx.datatables` to `dbx.dataparts`:
  `from dbx.dataparts import DatatablePartition`. `Datacollator` stays in
  `dbx.datatables`. No hash moved -- an identity names no module -- and a
  piece's specialization anchored at `dbx.datatables` reaches the pieces built
  there; partitions are rebuilt since the roles were renamed (above). `dbx.*`
  re-exports all of them as before.
- **`FeatureAffineLogisticProbe`: `aggregation` is `signal_reduction`, and
  `tab_aggregation` is gone.** The old `VAR.aggregation` averaged within a row
  -- a `(T, D)` token feature over `T` -- and is now `VAR.signal_reduction`
  (`reduce_signal_np`, was `aggregate_features_np`; `signal_matrix(reduction=)`).
  `VAR.aggregation` now averages across rows, by `aggregateby` (below).
  `tab_aggregation='mean'`, one sample per tab, is removed: a tab is no longer
  one slide once a partition rejoins or packs tabs; aggregate by a column
  naming the slide instead. VERSION 4. A specialization cannot restore a
  removed field, so probes built before are not reached, and rebuild.
- **`DatatablePartition` gained `coalesce`, `min_tab_rows` and `max_tab_rows`,
  and so every partition's hash moved.** `DatatableCorePartition` takes no
  `coalesce`.
- **Breaking: the journal classes are spelled `DataJournal`, `ExecJournal…`.**
  `Datajournal` -> `DataJournal`, `DatajournalFrame` -> `DataJournalFrame`,
  `DatajournalEntry` -> `DataJournalEntry`, `ExecjournalFrame` ->
  `ExecJournalFrame`, `ExecjournalEntry` -> `ExecJournalEntry`, and the
  `Datablock.Datajournal()` reader -> `Datablock.DataJournal()`. There are no
  aliases. A subclass that still defines `Datajournal` is refused at
  definition, since nothing calls it any more. The readers `dbx.datajournal()`
  and `dbx.execjournal()` keep their names.
- **One name for the block a feature block was built from: `upstream`.**
  `Featuretab.VAR.datapoint_tab`, `Featuretable.VAR.datapoint_table`,
  `BipolarFeaturetab.VAR.featuretab` and `BipolarFeaturetable.VAR.featuretable`
  are all `VAR.upstream`, a block of the same kind. `UPSTREAM_TABS` and
  `available_slices()` are gone; `slices(recursive=True)` gives a block's own
  slices, then its upstream chain's. Each class declares its rename as a
  specialization, so what was built before is adopted.
- **`Datacollator` names columns by role, and reads upstream only when told.**
  `VAR.signals` and `VAR.labels` are roles of `VAR.columns`, a
  `{role: pairs}` dict, which a `DatatablePartition` reads as `groupby` and
  `stratifyby`. `VAR.recursive` (default False) decides whether a column may
  come from the table's upstream; `slices(table)` refuses one that does unless
  it is set. `Datacollator` is in `dbx.datatables`.
- **`DatatablePartition` takes its table as `datatable` and its columns from a
  collator.** `VAR.datapoint_table`, `VAR.groupby` and `VAR.stratifyby` are
  gone: `VAR.collator` holds `groupby` and `stratifyby`, with `recursive=True`
  for columns upstream of the table (a feature table's annotations). Its
  specializations reach the partitions built before.

### Added
- **`Pipeline.entry`: `entrypoint`, for short.** Like `entrypoint`, it creates
  the tree afresh and records the creation in the `PipeJournal`.

- **`DatatablePartition(coalesce=...)`: what a part's tabs are, once dealt.**
  `None` (the default): whole tabs and pieces, as dealt. `'whole'`: a cut tab
  whose pieces all went to one part, none skipped, is that
  part's whole tab again -- read where it is, not copied. `'pack'`: as
  `'whole'`, then each part's tabs and pieces of fewer than `min_tab_rows`
  rows are pooled, in source order, into `DatatablePackTab`s of `max_tab_rows`
  rows (twice `min_tab_rows` by default), recorded in `tabs` as
  `{"fold": k, "pack": j, "segments": [[entry, offset, length], ...]}` and
  built by the part. A rejoined or packed tab may hold rows of several groups
  and strata. No mode moves a row to another part. `part.source_index()`
  gives every row of a part as the `(tab, row)` of the partitioned table.
- **`FeatureAffineLogisticProbe(aggregation='mean')`: rows aggregated by a
  column.** The collator's new `aggregateby` role names the rows to aggregate
  (a slide's id); the probe averages the rows sharing an `aggregateby` value
  and a label, from whichever tabs they lie in -- each worker sums its tab's
  share, the probe divides. With an `aggregateby` column the fit and eval
  tables must share no value of it; and two parts of different partitions of
  one table must share no row (by `source_index()`).
- **`Pipeline`: a Datablock tree defined flat.** A subclass declares the VAR
  of the whole tree, and its `__create__()` arranges the blocks out of it and
  returns the root, the entrypoint. `create()`, `entrypoint` and
  `read('entrypoint')`, its one SYNTHETIC topic, each record the creation in
  the `PipeJournal`. That is a `DataJournal` laid out as a block's under
  `{datalake}/.pipelines`, with an `entrypath` column.
  `dbx.pipejournal()` (also `dbx.journal('pipelines')`) groups its entries by
  pipeline, with every `entrypaths` each was created to; `construct_only=True`
  keeps those whose entrypoints were built or redirected. See `JOURNALS.md`.
- **`DatatablePartition` splits a tab holding several values of a partition
  column.** Such a tab -- several patients in one bag, tiles annotated one by
  one -- used to raise `NotImplementedError`. It is now split into pieces, one
  per combination of its groupby/stratifyby values, each a `DatatabPiece`: the
  rows of that tab holding those values, every slice written with the same
  rows. A piece is dealt as a tab is, with its group and within its stratum;
  in `tabs` it is a record, `{"tab": i, "groupby": v, "stratifyby": v}`, beside
  the indices of the tabs dealt whole, and its tag is `<tab tag>#<value>...`.
  The part builds its pieces, in parallel: `partition.part(k).build()`. Each
  piece's `source` slice records its source tab and rows, in order. That is
  what a table built from the source row by row (a feature table) can take
  the same piece of itself by. Rows of a mixed tab holding no value are a piece
  that is skipped, and recorded in `summary` with their count. Only mixed tabs
  are split, so a partition of constant tabs -- its hash, `tabs`, folds and
  summary -- is unchanged. Not for a table reading upstream (a `Featuretable`):
  a mixed tab of one still raises `NotImplementedError`.
- **`DatatableCorePartition`: coresets.** Rows are clustered by the columns
  a collator names `coreby` -- flattened per row, k-means in the master
  (`clustering='master'`, `MiniBatchKMeans`) or across the workers
  (`'distributed'`, Lloyd passes over per-tab sums) -- and dealt nearest the
  center first, honouring `groupby` and `stratifyby`. Fractions are of the
  rows: summing to 1 deals every row; less takes the rows nearest each center.
  Each fold is a `DatatableCorePart` of `DatatableCoreTab`s, built in
  parallel, carrying only the columns the collator names (`carry` for the
  rest), upstream ones copied flat. The layout -- `(tab, row, cluster)` per
  row per fold -- is a topic, and `layout=` lets a table laid out alike take
  it without clustering. `stratifyby` targets accumulate over a stratum's
  clusters, so each stratum lands within half a unit of its share.
- **Coverage of a core partition: the `coverage` topic.** How tightly each
  fold fills out the table, in the clustered space: the distance from
  sampled table rows to the nearest fold row (quantiles, against a random
  subset of the same size and the table's own nearest-neighbour spacing),
  the separation between fold rows, and the clusters' mix per fold. Sampled
  (`coverage_sample`) and compared within each row's nearest clusters
  (`coverage_clusters`), tab by tab in the workers -- no O(N^2).
- **A `source` slice in every repartitioned tab.** `DATASLICE(tab='int',
  row='int')`: row i says which tab and row of the partitioned table row i of
  the other slices came from. A piece's replaces its `source_rows` file.
- **`DatatablePartition.part(k)`**: fold *k* as a table, a `DatatablePart`.
  `fold(k)` is the name it had first, and still works.
- **DATAPARTITION.md**: partitions, parts and pieces -- what decides a
  partition, the scan/deal/build flow, and the way to downstream tables.
- **Probes read parts holding pieces.** `check_probe_inputs` asks a part about
  its own blocks, pieces included, and says to build the part when a piece is
  missing; `_check_disjoint_` tells pieces apart by tag. Stratifying a
  partition by the label column turns a tab whose rows carry two labels into
  pieces carrying one each, which `tab_aggregation='mean'` takes as samples.
- **`TernaryFeaturetab` / `TernaryFeaturetable`: a bipolar tab's rows, kept where
  their bag is sure of the sign.** Per dimension, the bag's mean bipolar value
  `m = 2p - 1` is the estimate; its sign is decided where `|m| >= bag_threshold`
  and 0 otherwise. A row keeps its value where the sign is decided and it
  agrees, else 0. Topics: a `ternary` slice, and `bag/<column>` (`mean`,
  `sign`, `n_rows`).
- **`dbx.stills.read_checkpoint`** reads a checkpoint whose pickled extras
  name a class since renamed, standing in for it with a warning; warm starts
  read through it.
- **Specializations compose through nested blocks.** A nested block renders in
  its container's identity, so a rename inside it used to strand every build
  of the container. Each block now also looks for its past with its nested
  blocks rendered as THEIR own specializations say they were, recursively; a
  rename is declared once, by the class whose field it is. Only pasts the
  journal holds a build or adoption of are composed, which keeps the
  combinations few.
- **Dotted specialization fields.** `redirect_vars={'columns.signals':
  'signals'}` moves a dict entry out to the field it used to be (an absent one
  as None), and a dotted pin, `spec={'collator.recursive': False}`, matches
  and drops a nested block's field that did not exist then.

### Removed
- **`FeatureAffineLogisticProber`.** Its `evaluate_features(2)` split rows at
  random, unseeded, and with no regard for patients; nothing called it, and
  `FeatureAffineLogisticProbe` only constructed one it never used. A fit/eval
  split is two folds of a `DatatablePartition` -- now for mixed tabs too.
- **The old module and class names.** `dbx.datapoints`, `dbx.datafeatures`,
  `dbx.datamodels`, `dbx.dataprobes` and `dbx.databackbones` no longer import,
  and the aliases `DatapointTab`, `DatapointTable`, `DatapointPartition`,
  `DatapointFold`, `DatatabBase`, `UpstreamTabSlices`, `DatafeatureTab(le)`,
  `FeatureTab(le)`, `BipolarDatafeatureTab(le)`, `Datafeature*Probe(r)`,
  `Datamodel*`, `Dataformer*`, `Datastill`, `Datalightning` and `Dataweights`
  are gone; `DatapointTableTab` is `DatatableTab`. Classes record their real
  module (`dbx.datatables.DatatablePartition`), which moves the anchor of a
  block of a dbx class used directly; hashes do not move. `DatatablePartition`
  and `DatatablePart` declare specializations to their `dbx.datapoints.*`
  anchors, so what was built there is adopted.

### Added
- **`loc=` takes an id prefix in `datajournal()` too.** `dbx.datajournal(...,
  loc=...)` and `DatajournalFrame.get()` accept any prefix of exactly one id,
  as `execjournal()` and `ExecjournalFrame.get()` already did, and raise
  `KeyError` if no id or several ids match. The prefix is matched against the
  index, so a numbered frame, such as a block's `datajournal()` by default,
  still takes row numbers only.
- **`ExecjournalFrame` displays compactly and keeps every value.** Ids and
  sessions show as 8 characters, timestamps to the second, and path lists as
  counts. The `id` column is hidden when it only repeats the index,
  `exec:start:datetime` when it equals `datetime`, and `traceback` always. The
  repr uses the terminal's width and wraps columns rather than hiding them.
  `ej.show(width=, max_colwidth=, max_rows=, full=)` prints it wider, or
  exactly as held. `ej.get()`, `ej(...)` and `execjournal(loc=...)` accept any
  prefix of exactly one id, including the displayed `976f41ea…`. `.loc` stays
  exact.
- **A frame derived from a journal frame keeps its class.** Column
  selections, filters, sorts and `head()` of an `ExecjournalFrame` or a
  `DatajournalFrame` used to come back as plain DataFrames, which lost
  `.get()`, `.datajournal()` and the datalake. They are wrapped as they are:
  re-running `__init__` would renormalize a block journal selection, adding
  back the `type` and `signature` columns it had left out.
- **Output capture belongs to the command: `dbx.exec(..., capture_output=True)`,
  and `--capture-output` for `dbx`, `dbx.exec`, `dbx.print` and `dbx.pprint`.**
  The process that runs the command tees fd 1 and 2 to a master
  `OutputCapture` from start to finish. Every worker process a dbx executor
  sends work to runs each callable inside a capture of its own, and its path
  comes back with the result, the same way the journal entries it wrote do.
  Worker threads share the master, because descriptors belong to the process.
  The exec journal records the paths in a new `output_captures` column, placed
  after `datajournal_entries`, with the master always at index 0.
  `ExecjournalEntry.output(idx)` prints one to stdout, and
  `.output(basename=...)` finds one by a prefix or regex matched at the start
  of its file name. `dbx.pprint` opens the capture around its own printing,
  and the pinned-revision runner does the same, so the printed result is
  captured too.

- **`dbx.exec` takes a sequence of statements, and a comment.** The CLI
  argument was a single expression, evaluated with `eval` -- so anything that
  needed a name to be bound first needed a script, and a one-liner could not
  build a block and then read it. It is parsed with `ast.parse` now and runs
  every statement the string holds, separated by `;` or by newlines, in one
  namespace, returning the value of the last:

  ```bash
  dbx.pprint "b = my.Block(spec={'x': 1}); b.build(); b.read()  # nightly"
  ```

  One dict serves as both globals and locals, so the statements run the way
  module-level code does: split, a comprehension in a later statement could
  not see a name an earlier one bound -- the rule that keeps a class body from
  seeing its own names -- which would break exactly the idiom sequencing
  exists for. The dotted names are collected from every statement's AST rather
  than from the text before the first `(`, so any statement may name a module
  that nothing has imported yet -- and a name an earlier statement bound, which
  is no module path at all, resolves to nothing rather than failing.

  A trailing `#` comment is ignored when the command runs -- Python's parser
  drops it -- and recorded in the exec journal's new `comment` column. The
  `exec` column still holds the string VERBATIM, comment and all, because a
  journal row is the only record of what was run and has to stay re-runnable;
  the separate column is there because the expression says what a command did
  and only its comment says what it was for, and the journal is read long
  after the person who typed it could be asked. The comment is tokenized, not
  split on `#`, so a `#` inside a string literal stays part of the expression.

- **`Datablock.SPECIALIZATIONS`: reading a narrower block's data instead of
  rebuilding it.** A class grows a `VAR` field and a topic; every block of it
  re-keys, and the topics that did not change are rebuilt for nothing. A
  `Datablock.Specialization(spec=..., topics=..., note=...)` says that when the
  new fields hold the values in its `spec`, the topics it names ARE the topics
  of the block this class used to be — whose identity is this one's with those
  fields dropped and those topics alone.

  That identity is reconstructible from here, because a signature is built from
  the spec and the topics and nothing else: `type(specialization=...)` renders
  it and `get_hash(specialization=...)` hashes it, so the narrower block's
  build is found in the journal by its own hash, with no access to the older
  class and no record that it ever existed. An unbuilt block with a matching
  specialization reads through it; `build()` then produces only the topics the
  specialization does not cover (`ownedtopics()`, the complement of
  `redirected_topics()`).

  A pin is matched against the RENDERED spec, not the `spec` dict a caller
  passed: a field left at its default is absent from that dict, and a field
  left at its default is exactly the case this exists for. Specializations are
  tried in declaration order and the first that both matches *and resolves*
  wins, so one whose build has been cleared does not shadow the next. A miss is
  never silent: `specializations()` reports every declared one with the hash it
  looked for and why it did not apply.

  Installing one is recorded, through `UNSAFE_redirect`: an installed
  specialization is a claim about where a block's data came from, and the
  journal entry is the only record that this block read another's build and
  which specialization said it could. Recording also costs less than not
  recording — it writes the hidden `.redirection` topic, which every later
  construction reads instead of scanning the journal again, so the write
  happens once per block rather than the scan happening once per construction.
  Within a construction, the resolution travels with the block through pickle,
  deepcopy and `set()`. `use_specializations='memory'` installs without
  recording; `UNSAFE_specialize()` is the explicit form.

  It covers a `DatapointTable`, which took two things a plain `Datablock` had
  not needed. A table's `valid()` is the `done` MARKER, and `done` is one of
  the topics a specialization redirects -- so the table reported itself built,
  off another build's marker, before the topic it was grown for existed, and
  `build()` (which asked `valid()` and nothing else) skipped. Silently, and
  however many times you rebuilt. `build()` now also asks **`owedtopics()`** --
  the ownedtopics that are not there, empty for any block that is not
  partially redirected -- and logs `BUILD OWED` when the two disagree. It is
  deliberately not folded into `valid()`: what validity means belongs to the
  class, and a marker is a real answer; whether a block has produced what a
  redirection left to it is a different question, and the only one `build()`
  has business asking.

  And `DatapointTable.__split__` ensured the `tab_paths` directory before
  anything else, which `path(ensure_dirpath=True)` refuses for a redirected
  topic -- correctly, since it would be creating a directory inside another
  block's data. So the tab machinery *raised* for a specialized table rather
  than merely being unnecessary to it. That call and `_write_tab_path` are
  guarded by `ownedtopics()` now; the sentinels already in the other block's
  `tab_paths` name the same tabs, because a redirection that covers
  `tab_paths` is by construction one that did not re-key the TAB.

  **`Specialization(version=...)`** completes the reconstruction. `type()` is
  built from the spec, the version and the topics and nothing else, and a
  specialization used to name only two of the three: the version came from the
  CLASS, so bumping `VERSION` moved the reconstructed hash onto an identity
  nobody had ever built, and the only way to keep a specialization working was
  never to bump. Naming all three describes the narrower block completely, and
  makes two real cases expressible -- a bump that changed some topics and not
  others, and one that turned out to change none. The field defaults to
  `ABSENT` rather than `None`, because `None` is what `version` reads as for a
  class that declares no `VERSION` at all: a real value a specialization has to
  be able to name, and the one a class names the day it starts versioning. It
  is in `Specialization.key` (two specializations alike but for the version are
  two identities, and `get_hash` caches per key) and in the journal record.

  **`buildtopics()` is `ownedtopics()`**, with the old name kept as an alias.
  "build" said what to do with the topics rather than which ones they were, so
  a reader had to guess whether it meant "must produce" or "has yet to produce"
  -- which are now two methods: `ownedtopics()` is what this block is
  responsible for, `owedtopics()` the subset of those that are not there yet.
  Ownership does not change when a build runs; owing does. The two are one
  letter apart, so each docstring names the other.

  **One journal read for a whole stack, not one per child.** A block resolves
  its specializations in `__setstate__` -- at CONSTRUCTION, before anything
  calls `__build__` -- and resolving means reading the journal. A stack
  constructs every child, so a stack whose children are specialized paid one
  read PER CHILD: invisible against a local directory, a glob over
  `**/*.parquet` plus a parquet read per child against the object storage these
  stacks actually live on. Measured on a 24-tab table: 22 reads, 0.92 per tab.

  `Datablock.__init__` takes `specialization_journal=`, a journal already read
  for `_install_specialization` to resolve against instead of reading one --
  the `journal=` argument that `_install_specialization`, `UNSAFE_redirect` and
  `_specialization_paths` have always accepted and that nothing ever passed.
  `Datastack.child_specialization_journal()` reads the children's journal once
  and keeps it; `DatapointTable.__tab__` hands it to every tab, but only when
  the TAB declares SPECIALIZATIONS, so a table whose tabs have none reads
  nothing. The same table now: 2 reads. A Tab class opts in by declaring
  SPECIALIZATIONS and nothing else.

  It could NOT have been a `journal=` kwarg on `__build__`, which is the
  obvious place to reach for: by then every read has already happened.

  Nothing crosses to a worker. `__getstate__` already carries a resolved
  `__redirected_paths__` through a pickle, so a child that resolved in the
  parent arrives with its answer and reads nothing -- better than shipping a
  journal frame per worker. And `specialization_journal` is in the new
  `TRANSIENT_PARAMS`, so it is exempt from `__explicit_params__()` and never
  reaches `parameters`, `dfn`, `quote()`, `cite()` or a pickle: a block
  reconstructed from any of those must come back the same block, and one
  carrying a stale journal would not.

  The shared journal is a SNAPSHOT: a child resolving against it cannot see an
  entry a sibling wrote during the same build. That is right for a
  specialization, which resolves to a build that predates this one.

  **`specializations()` answers with a `SpecializationRow`**, which explains
  itself. A dict, like `Validation`, so every reader written against the keys
  goes on working -- and a type, so the answer renders as a report instead of
  each caller assembling one out of `dict(matches=..., why=..., ...)`. Truthy
  exactly when it RESOLVED, which is the question being asked, so `if not row:`
  reads correctly. Two new keys the caller used to compute: `topics` (what the
  specialization names) and `builds` (this block's topics that it does not --
  what a build would still have to produce). `builds` is deliberately not
  `ownedtopics()`: that answers for the redirection a block HAS, and a row is
  about one it may not have.

  **And `why` says which of four things went wrong**, because they call for
  four different responses: the narrower block was never built here, it was
  built and its data has since been cleared, an entry records only some of the
  topics (a specialization is all of them or none), or a pin does not fit this
  block. The first two used to read identically -- "no entry with hash ...
  records data that is still there" -- and they are the pair most worth telling
  apart. A pin mismatch went from `window='hamming', pinned 'hann'` to naming
  both sides and saying that it therefore describes a different block, which is
  what a pin is FOR and not something to fix.

- **`UNSAFE_redirect(dry_run=True)`**, and `UNSAFE_specialize(dry_run=True)`.
  Resolves the whole redirection and reports what it would record — the record,
  the specialization, the source journal entry, the resulting paths, and the
  topics that would still be built — installing nothing and writing nothing.
  The return type says which happened: a `Redirection` is a proposal, `True` is
  a redirection that is now in place. To inspect without writing but with the
  paths actually in use, construct with `use_specializations='memory'` and read
  `.specialization` / `.redirection`; `UNSAFE_specialize()` then promotes it.

- **Partial redirection.** `UNSAFE_redirect(topics=[...])` redirects only the
  topics it names; every other topic reads as it would unredirected. `build()`
  no longer declines wholesale when a block is redirected — it declines only
  when the redirection is total, and otherwise builds the rest.

### Changed
- **Breaking: what builds a `Datajournal` or `DatajournalFrame` is named for it.**
  `Datablock.journal()` is now `Datablock.datajournal()`, `Datablock.Journal()`
  is `Datablock.Datajournal()`, `Datastack.block_journal()` is
  `block_datajournal()`, `Datatable.tab_journal()` is `tab_datajournal()`, and
  `Datastack.child_specialization_journal()` is
  `child_specialization_datajournal()`. No aliases are kept. A subclass that
  still defines one of the old names raises `TypeError` at class definition,
  because nothing calls the old names and an override would be ignored.
  `dbx.journal()`, which dispatches to either journal, is unchanged.
- **Exec journal columns: `datetime` and `exec` first, `comment` and `success`
  last.** `exception` and `traceback` follow `exec:end:datetime`; columns the
  list doesn't name go before `comment`.
- **A block no longer captures output itself.** Its journal entry's `log` is
  the path of the capture open around its build, taken from
  `OutputCapture.current()`, and None when there is none.
  `Datablock(capture_output=...)` is still accepted, because every recorded
  dfn spells it, but it is ignored: it logs a warning when True, and it is no
  longer part of `dfn`, `quote()` or a pickle.
- **A redirection is no longer part of a block's identity.** `type()` used to
  append `_redirected_paths_=...` once a redirection was installed, so `hash`
  depended on *when* it was first called — before the redirection or after —
  and `anchorkeypath`, the journal directory and the redirection lookup moved
  with it. Masked, until `keyby` stopped naming the hash, by `__setstate__`
  building the logger name out of `self.key` and caching `_hash` on the way
  past. A block does not become a different block by being read from somewhere
  else.

- **`set()`/`replace()` refuse `spec=`.** `set()` amends a block's non-identity
  parameters; a new spec is a new block, and handing one back through a method
  that reads as an amendment of this one also carried this block's resolved
  state over to an identity it was never resolved for. Construct it instead:
  `type(b)(**{**b.dfn, 'spec': {...}})`.

- **Writing to a redirected topic is refused.** `path(topic, ensure_dirpath=True)`
  on a redirected topic raises rather than quietly not creating the directory:
  it is how this codebase asks for somewhere to write, and not creating the
  directory does not stop the write when the other block's directory is already
  there.
- **`dbx.stills`: `Still`, `LightningBuilder`, `Weights`.** One training run
  as a config-addressed Datablock — topics `ckpts`/`logs`, the `_COMPLETE` marker,
  the checkpoint save/upload/free/resume dance, the TensorBoard log symlink, the
  atexit sync, the `UNSAFE_*` helpers, the `Trainer` assembly and a generic
  `dataloaders()`. Every training pipeline in soundworld had grown its own copy of
  all of it; `IJEPAsaurUSStill` shed ~930 lines becoming a subclass, with its hash,
  key, signature, `quote()` and `cite()` byte-identical (its `VAR` is inherited
  whole, so the field *order* a `LEGACY_NORM` block hashes can only be the base's).

  Two ways to say what to train. **Explicit mode** puts a `LightningBuilder` in
  `VAR.lightning`, which is what you want when the module's construction has
  upstream artifacts of its own. **cfg mode** names two ordinary classes — `Model`
  and `Lightning`, importing nothing from dbx — and mirrors their `cfg_`-prefixed
  keyword arguments into the still's own `VAR`; a name declared on both is one VAR
  field passed to both, which is the right reading for a knob like `crop_size` that
  two objects must agree about. `Still.export()` then writes a standalone module
  carrying that configuration as literals, with no dbx in it.

  The mirrored `VAR` is checked-in source, generated once by `scaffold_still()`,
  **not** reflected at import time: `_typed_specdict` walks every `VAR` field into
  the hash, so a `cfg_` knob added in a file that does not mention dbx would
  silently re-key every run of that model. As source, that movement is a diff.
  `__check_cfg__` raises on construction if the two have drifted apart, and refuses
  a `cfg_` name that collides with one of `Still`'s own fields (`cfg_ckpt` is
  initial weights; `VAR.ckpt` is the run to resume from — one field cannot be both).

  `VAR.ckpt` holding another *block* is a **warm start**: its latest checkpoint is
  loaded weights-only, at step 0, with a fresh optimizer, and only until this run
  has a checkpoint of its own. Carrying a different run's optimizer moments and
  epoch counter over is not a resume, it is a way to lose a training run quietly —
  a source that already reached its own `max_epochs` leaves the new one with
  nothing to do, so `fit()` returns at once and the `_COMPLETE` marker lands over
  an untrained model. (The stills in soundworld took the old reading, which is why
  `IJEPAsaurUSPoseStill`'s phase-1 chains passed `var_reset_optimizer_state=True`
  or ran short.) A bare *path* in `VAR.ckpt` still resumes in full — that is how
  you point a run at its own checkpoint that has moved, and the escape hatch from
  the new default. A warm start that matches none of the model's parameter names
  now raises rather than training from random init behind a banner that says
  otherwise.

  Not imported by `dbx/__init__.py`: it needs lightning at module scope, so import
  it by name, exactly as for `dbx.datastreams`. New `[lightning]` extra.
- **`dbx.datastreams`: `block_split_indices`, `block_split_ranges`,
  `val_loader_workers`.** The split counterpart of `ChunkShuffleSampler`, and here
  for the same reason: a per-sample-random train/val split scatters both halves
  across the whole table and defeats the local shard cache as thoroughly as no
  split at all. `val_loader_workers` caps a validation loader that reads
  `val_max_batches` batches so it stops prefetching (and discarding) an order of
  magnitude more shards than it reads, evicting the training working set every time
  it validates. In `datastreams` rather than `stills` so that torch-only code
  can use them without pulling in lightning.

### Breaking Changes
- **The `Data` prefix comes off three modules and their classes.**
  `dbx/databackbones.py` → **`dbx/backbones.py`**, `dbx/dataprobes.py` →
  **`dbx/probes.py`**, `dbx/datastills.py` → **`dbx/stills.py`**; and with them
  `DatamodelEvaluator` → **`ModelEvaluator`**, `DatamodelEvaluatorFactory` →
  **`ModelEvaluatorBuilder`**, `DataformerEvaluator` → **`TransformerEvaluator`**,
  `DataformerEvaluatorFactory` → **`TransformerEvaluatorBuilder`**,
  `DatafeatureAffineLogisticProbe(r)` → **`FeatureAffineLogisticProbe(r)`**,
  `DatafeatureStatsProbe` → **`FeatureStatsProbe`**, `Datastill` → **`Still`**,
  `Datalightning` → **`LightningBuilder`**, `Dataweights` → **`Weights`**. Every old
  name still resolves — the class names as plain aliases, and `dbx.databackbones`,
  `dbx.datamodels` and `dbx.dataprobes` as the SAME module object under the old
  path, the way `dbx.datapoints` and `dbx.datafeatures` already are.

  Unlike the `datapoints`/`datafeatures` rename, the class names moved too, so
  `fqcn` moved with them and no amount of pinning `__module__` could have held
  it. That is allowed here only because nothing was ever *built as* one of these:
  every backbone builder, probe and still downstream is a subclass, which reports
  its own module and its own name, and there is no `dbx.*` anchor directory in any
  root. Renaming a base costs nothing; renaming the subclass is what would cost.
  Checked, not assumed — `IJEPAsaurUSStill`'s hash, key and `type()` are
  byte-identical across the rename.

  `dbx.datastills` gets no module alias at all: it is a week old, was never
  released, and an alias exists to keep a *recorded* string resolving. There is no
  such string. `Datalightning` becomes `LightningBuilder` rather than plain
  `Lightning`: the bare word is already `Still.Lightning`, the `LightningModule`
  subclass cfg mode dispatches to, and that is the one a still author writes.
  The suffix also matches `ModelEvaluatorBuilder` beside it.
- **Renamed `Datablock.CONFIG` → `Datablock.VAR` and `.cfg` / `.config` → `.var`.**
  Both old names are kept as aliases, so existing subclasses keep working unchanged:
  `Datablock.CONFIG` still resolves (it is an alias of `Datablock.VAR`), and a subclass
  that declares `class CONFIG(Datablock.CONFIG)` has it bound to `self.VAR` in
  `__setstate__`, so it survives construction, `deepcopy` and unpickling. `.cfg` and
  `.config` are now read-only aliases of `.var`. `VERBOSE_CONFIG` is likewise renamed to
  `VERBOSE_VAR`, with the old spelling still honored. Identity is unaffected — spec,
  norm, hash and key are byte-identical either way.
- **Renamed `validate_cfg=` → `validate_vars=` and `valid_cfg()` → `valid_var()`.**
  `valid_cfg()` is gone, with no alias. `validate_cfg=` survives only as a deprecated
  `__init__` parameter and state key, so a block reconstructed from a `dfn` recorded
  before the rename still works. It is kept *explicit* rather than left to `**kwargs`
  because a stray dynamic kwarg would be silently ignored — validation would stay on
  for a block whose `dfn` says `validate_cfg=False` — and would then persist as a dead
  parameter in `dfn`/`quote()`/`cite()`. It is never re-serialized: `__getstate__` emits
  only `validate_vars`. Neither flag is part of a block's identity — `norm()` is built
  from `url`/`anchor`/`hash` and `spec` alone.
- **Retired `VALIDATE_CFG_EXEMPTIONS`** in favour of `TREE_SKIP_VALIDATION`, which
  already did the same job. A subclass still declaring the old name now raises
  `AttributeError` at construction naming the replacement, rather than being ignored —
  silently dropping it would re-enable the validation it existed to suppress, surfacing
  as a confusing "Not all upstream Datablocks in var are valid" at build time.
- **Renamed `hashstr` → `signature` and `superhashstr` → `supersignature`**, in the
  `Datablock` properties, the `Bid` fields, the journal columns, and the recorded
  files (`signature.txt` / `supersignature.txt`). No aliases on `Datablock` or `Bid`.
  The string itself is unchanged, so `hash`, `superhash` and `key` are byte-identical —
  this renames the name, not the identity. Journals written before the change recorded
  the columns as `hashstr` / `superhashstr`: `DatajournalEntry.signature` and
  `.supersignature` fall back to them, treating a NaN as absent so that a journal
  spanning the rename (both columns present, NaN-filled per row) reads correctly on
  either side.
- **Removed the whole `*DatablocksBuilder` family, `select_builder()` and `_DATABLOCKS_BUILDERS`**
  — `Inline`, `Multithreading`, `Multiprocessing`, `Ray`, `TorchMultithreading` and
  `TorchMultiprocessing`, plus the private `_build_block`, `_build_block_with_to` and
  `_TorchBlockCallable_` helpers. Nothing constructed them: `Datastack` resolves
  `executor_cls` to the `*CallableExecutor` family directly, and `select_builder` was
  reachable only from its own tests. Use the executors (`select_executor`,
  `callable_executor`, `parallelization='...'` on a `Datastack`) instead. The one
  behaviour with no executor equivalent was `RayDatablocksBuilder` copying a remote
  block's `__getstate__()` back onto the local object after a build; that is a no-op
  unless a subclass registers a build-time value as a serializable parameter.
- **Renamed `FDCapture` → `OutputTee`, and removed `Tee`.** `Tee` wrote to several
  Python file objects at once, for installing as `sys.stdout` — but that only sees writes
  going through the Python stream object, so it missed C extensions writing to fd 1/2,
  subprocesses (which inherit descriptors, not `sys.stdout`), and any code holding the
  real stream from before the swap. `OutputTee` redirects the descriptors themselves
  with `dup2`, so all three are caught. It was already unused; nothing referenced it.
  `capture_output=True` still **mirrors rather than silences** — output continues to
  reach the terminal, which is what `Tee` was meant to provide.
- **Removed the `@tagged` decorator** (and its `_make_tag` / `_TAGGED_SKIP_DEFAULTS`
  helpers), which auto-generated a call-string tag for pipeline functions. Unused.
  Pass `tag=` explicitly.
- **Renamed `JournalFrame` → `Datajournal` and `JournalEntry` → `DatajournalEntry`.**
  No aliases are kept; update any `from dbx.datablocks import JournalEntry`. The
  `journal()` function, `Datablock.journal()` / `Datablock.Journal()`, and every journal
  column name are unchanged. (Entries in earlier releases below still use the old names,
  which is what those releases shipped.)
- **`norm()` now quotes strings and reprs spec values once**, which **changes the
  hash of every Datablock** that does not opt out. The old rendering was ambiguous:
  top-level string kwargs were bare (`url=abfss://…`), and non-string spec values were
  repr'd twice (int `5` → `'5'`) while strings were repr'd once, so `n=5` and `n='5'`
  produced the *same* hash. Set **`LEGACY_NORM = True`** on any class whose artifacts
  are already built and keyed, to keep the exact bytes its hashes were computed from.
  New classes should leave it alone (default `False`).

### Added
- **`DatastreamTab` / `DatastreamTable`** in `dbx.datastreams` — an abstract `Datablock` /
  `Datastack` pair over parallel MDS *slices*, documented in
  [DATATABLES.md](DATATABLES.md). A tab declares `SLICES = ('frames', 'annotations')`
  and writes them in lockstep inside `__build__`; the `data` group of its `TOPICS`
  is synthesized from them, so the slices are covered by the block's hash. A table
  inherits `SLICES` from its `TAB` and needs only `n_tabs`, with `__tab__(idx)`
  — `Datastack.__block__` for tables, whose `super()` fills in placement — implemented
  only when a tab needs a spec of its own. Block placement, `__split__`, and a `__stack__` that merges
  every tab's per-slice `index.json` into one index per slice are preimplemented.
  A tab's shards land in the table's per-slice root (`<table>/data/<slice>/<tabdir>/`)
  rather than under the tab's own key, because `StreamingDataset` resolves a shard
  relative to the directory holding the index that names it. Both classes read back
  as `data(slice)` (lumped samples), `datastream(slice)` (one live
  `StreamingDataset`), `dataset(*slices)` (the zip of those) and `stats(slice)`
  (a `__stats__` hook). Local scratch — shard cache, write staging, decompression —
  goes under `cacheroot`, which defaults to `<localroot>/streaming` rather than `/tmp`
  and is overridden with the `cache=` kwarg.
- **`BlockShuffleSampler` / `ResumableDataLoader` / `shuffled_block_order()`** in
  `dbx.datastreams` — shuffle contiguous blocks of the index space, and within each
  block, instead of permuting the whole range. Consecutive sample indices share an
  MDS shard, so `DataLoader(shuffle=True)` scatters every access and defeats the
  shard cache; block shuffling keeps the working set down to a few shards while still
  randomising both orders each epoch. `fixed_epoch=True` pins a validation sampler's
  order so capped val runs stay comparable; `state_dict()`/`load_state_dict()` (plus
  `ResumableDataLoader`, which surfaces them where trainers look) allow approximate
  mid-epoch resume. `DatastreamTable.sampler()` builds one whose `block_size` defaults
  to the table's **own** shard capacity — read off the merged index by the new
  `shard_sizes()` / `samples_per_shard()` / `n_samples()`, without downloading a shard
  — rather than leaving every caller to guess a constant.
- **`ZipIterableStreamingDatasets` and `dataset(mode='iter')`** in `dbx.datastreams` —
  zip the slices by *iteration order* instead of by physical index. Map-style indexing
  reads through `StreamingDataset.get_item()`, which downloads a missing shard inline
  on the calling thread; the download-ahead thread, the rank/worker partitioning,
  `num_canonical_nodes`, the shard-locality shuffle and mid-epoch resumption all live
  in `__iter__` and so never run. Iterating each slice puts every slice back in
  possession of them, which on remote storage is normally the largest single factor in
  throughput. The merge is unchanged — both classes now share it through `ZipBase` —
  and so are the defaults: `mode='map'` remains what `dataset()` does.
  Iterator-mode zipping is only correct while every slice yields the same sequence,
  which `shuffle=False` gives for free (the partition reads sample *counts*, never
  shard structure) but `shuffle=True` does not: the permutation is derived from the
  per-shard sample counts, so differently-sharded slices shuffle differently and pair
  unrelated samples. `__init__` checks that and refuses; `shared=` +
  `validate_shared=True` remains the running check, and is effectively required here.
  `mode='iter'` also demands `batch_size=`, which `StreamingDataset` would otherwise
  only complain about on the first batch from inside a `DataLoader` worker.
- **`slice_writers(..., flush_every=N)`** in `DatastreamTab` — break every slice onto a
  new shard every *N* samples, so all slices carry identical shard boundaries.
  `MDSWriter` otherwise starts a shard on a byte budget, so slices of different
  per-sample size split at unrelated places; that is invisible to index-addressed
  zipping but is exactly what stops `mode='iter'` from shuffling. Whichever of
  `flush_every` and `size_limit` comes first still ends the shard, so a `size_limit`
  that fires first is detected and raised rather than left to surface as a misaligned
  shuffle. Writes are counted per slice, which also turns the lockstep contract into a
  checked one: a tab that has not written every slice the same number of times raises
  instead of producing a table that cannot be zipped.
- **`open_datastream()`** in `dbx.datastreams` — opens a `StreamingDataset` over an
  MDS index directory, local or remote, translating `abfs(s)://` to `azure-dl://` and
  retrying once past a stale `Reused local directory` shared-memory registration.
- **`ZipStreamingDataset` gained a merge policy** — `columns` (per-source column
  projection), `shared` + `validate_shared` (keys expected in several sources, and
  an equality check that makes a mis-zipped set of streams loud rather than silently
  misaligned), `on_conflict` (`'last'`/`'first'`/`'error'`) and `skip_none`. Defaults
  reproduce the previous plain last-wins merge exactly, so existing callers are
  unaffected. This is what a multi-slice `DatastreamTable` needs to be readable
  column-by-column, and it generalises `soundworld.databits.ZipDataset`.
- **`ZippedStreamingDatasets`** — alias of `ZipStreamingDataset`. The singular stays
  the canonical name; nothing is renamed.
- **Hierarchical `TOPICS`** — a dict value may itself be a dict, nesting topics:
  ```python
  TOPICS = {'data': {'frames': DIRTOPIC, 'annotations': SYNTOPIC,
                     'index': 'index.csv'},
            'model': 'model.pt'}
  ```
  Every topic-addressing method takes one name per level: `path('data', 'frames')`,
  `read('data', 'annotations')`, `ls`, `list`, `size`, `dirpath`, `validtopic`,
  `UNSAFE_clear`. The nesting is mirrored on disk under the block's key. A *group*
  is addressable in its own right — `dirpath('data')` is the parent directory,
  `path('data')` is the dict of its members' paths, and `validtopic('data')` is the
  conjunction over the leaves beneath it (`validpath` already recursed into dicts).
  New `leaftopics()` enumerates leaves as name tuples, and `is_topicgroup()` tests a
  path. Journal entries record the declared shape, and `DatajournalEntry.ls`/`list`/
  `size`/`_is_dir_topic`/`_is_syntopic` take the same per-level arguments.
  **Fully backward compatible**: flat dict-TOPICS, list-TOPICS and no-TOPICS blocks
  produce byte-identical signatures — hence the same hash, key and storage paths —
  verified against the previous commit. A nested leaf is rendered `topic:data/frames=...`,
  so a topic name may no longer contain `/`, which would make that ambiguous.
- **`DIRTOPIC`** — the filename of a directory topic in a dict-valued `TOPICS`:
  `TOPICS = {'images': 'images.csv', 'masks': DIRTOPIC}`. It *is* `None`, the value the
  topic machinery has always tested for, so `{'masks': None}` stays valid and produces
  an identical `signature`, `hash` and `key`; the constant only says out loud what a
  bare `None` left the reader to infer.
- **`SYNTOPIC`** — a *synthetic* topic, one the block presents but never stores, so it
  has no location: `TOPICS = {'data': 'data.parquet', 'cache': SYNTOPIC}`. `path()` and
  `dirpath()` are both `None`, nothing is created, listed, copied or cleared for it, and
  it is vacuously valid — a topic that was never going to be written cannot be missing,
  so it must not hold the block back. Distinct from `DIRTOPIC`, which *is* a location —
  a real directory that merely has no filename inside it. `SYNTOPIC` is `()` rather than
  another `None`-alike exactly so the two cannot collide; it is still declared, so it
  appears in the `signature` (as `topic:cache=()`) and is part of the block's identity.
- **`Datablock.cite()` recorded alongside `quote()`**: new `Bid.cite` field, a `cite.txt`
  written by `write_journal_entry()`, and a `JournalEntry.cite` property (returns `None`
  on journals written before the column existed, rather than raising).
- **`entry_code` on every journal entry** — a fresh uuid per `write_journal_entry()` call,
  recorded as a column and returned by the call, plus `DatajournalEntry.entry_code` and a
  matching `.uuid` accessor. It is the only field that identifies a *row*: `hash` and
  `key` are shared by every entry of a block, `uuid` by every entry of one live instance,
  and `datetime` only to its resolution — two entries written in the same microsecond, or
  by two processes at once, collide. So `journal(entry_code=code, loc=0)` addresses
  exactly the row a caller wrote. Follows `uuid16`, so the two identifiers in one entry
  are the same shape. Journals written before the column read as `None` rather than
  raising. A code resolves only until its *instance* writes again: a journal file is keyed
  by `self.dt`, so a second call from one instance overwrites the first — which is why
  `build()` leaves a `build:end` and no `build:start`. Pass a distinct `journal_prefix` to
  keep both.
- **`UNSAFE_redirect()` — read this block's topics from somewhere else.** Takes exactly
  one of `filter=` ({column: value} pairs selecting a journal entry, as `journal()`
  filters, the first — newest — match winning, so `{'hash': other.hash, 'event':
  'build:end'}` follows that block as it is rebuilt where `{'entry_code': code}` is pinned
  to the one build that code was returned for) and `paths=` ({topic: path}, naming the
  locations outright, consulting no journal at all). `topic_map=` re-keys a filtered
  entry's topics, reading mine → theirs: `{'out': 'output'}` says this block's `out` is
  that entry's `output`. Topics line up by name to begin with and the map only adds to
  that; a mapping whose target the other side lacks leaves its topic *unredirected* and
  says so, since asking for theirs and silently getting mine is the one answer that is
  certainly wrong. The non-None parts are packaged into a `redirection` dict and written
  to a fresh journal entry. Nothing is copied, moved or validated — a redirection is a
  note in the journal. It travels in the entry rather than in a file of its own, unlike
  `message` and `quote`/`norm`/`spec`, because a stand-in for missing data must not itself
  depend on a second file still being there; and it is written under a `redirect-` prefix
  so an instance that has already journalled does not overwrite that entry. A filter
  selects from this block's journal, which is per anchor, so it reaches another build of
  this class but not another class's unless the two share an `anchor=`.
- **`path()` and `dirpath()` are where a redirection takes effect**, so everything that
  resolves through them — `read()`, `valid()`, `ls()`, `list()`, `size()` — describes the
  data actually being read, and no reader needs to know: an override reads
  `self.path(topic)` exactly as it always did. A topic the redirection does not name keeps
  its own path, so a partial redirection redirects only what it names. `local=True` is
  never redirected (the local cache is this block's own) and a redirected path is never
  `ensure`d (creating directories inside another block's data is not this block's
  business). UNSAFE because every path the block reports then names data it did not
  produce and whose hash does not describe it, which nothing downstream can detect — so
  the redirection is announced at INFO when it resolves. `build()` refuses a redirected
  block: nothing would read what it wrote, and a `build_tree()` sweeping past would
  otherwise quietly rebuild the very block someone redirected away from.
  `Datablock.redirection` is the resolved `Redirection(paths, entry, filter, topic_map)`,
  cached — including the `None`, so a redirection recorded after a block has looked is
  seen by the next instance rather than that one — and detached from the frame its entry
  came out of. Whether a block HAS one is answered from its own journal directory rather
  than by globbing the anchor's, because `path()` asks on first use and a table of a
  thousand tabs asks a thousand times. `redirect=False` at construction opts out of all
  of it.
- **`valid()` goes through a `__valid__(path=None)` hook.** The default validates this
  block's topics, which for a redirected block are the redirected-to paths already, so it
  needs no knowledge of redirection either; *path* is the redirected-to block directory,
  for an override that wants to validate the location by more than the presence of its
  topics, and is `None` when the block is not redirected or was given paths outright.
- **`valid_topics()`, `valid_paths()`, `valid_path()`** are the canonical spellings,
  joining `valid_topic()`; `validtopics()`, `validpaths()` and `validpath()` remain as
  deprecated aliases.
- **A table's slices are synthesized from its `TAB`**, and a subclass declaring `TOPICS`
  keeps its bases' *slice* topics while dropping their ordinary ones — slices are what
  makes such a block the kind of block it is, whereas an ordinary topic belongs to the
  class that declared it. A table keeps `tabs`/`done` whatever it declares, and `slices`
  now reads off the class (`LetterTable.slices`) as readily as off an instance, as a
  tuple: what a block is sliced by is settled once its class is, and it feeds the hash.
- **`block_shuffle_sampler()`** on a datapoint block, the deprecated alias of
  `chunk_shuffle_sampler()` that `BlockShuffleSampler` and `block_size` already had.
- **`datablocks[azure]`** — `adlfs`, the fsspec driver for `abfs://` and `abfss://`, as a
  named extra rather than an assumption. Nothing imports it (fsspec resolves it from the
  url), so its absence surfaced only as `ValueError: Protocol not known: abfss` from a
  block that happened to be given one.
- **`diff()`, `difftopics()`, `diffversion()` — the rest of what a hash is made of.**
  A `signature` is a norm, a version and the topics, joined, so those three diffs between
  them account for every way two blocks can hash differently; `diff()` returns them as a
  `Diff` triple (also reachable as `.norm`, `.topics`, `.version`), and `any(a.diff(b))`
  is "is this a different block". `difftopics()` compares `signature_topics()` — the very
  segments the signature is built from, now rendered in one place instead of twice — so
  the diff and the hash cannot drift: it is non-empty exactly when the topics contribute
  to a difference in signature. It reports a sparse `{topic path: (self, other)}` dict,
  with `ABSENT` for a path one side does not declare, and a difference belonging to no
  single path — a reordering, or `TOPICS = {}` against no TOPICS at all, both of which
  move the hash — under the `SIGNATURE_TOPICS` sentinel key. `diffversion()` returns
  `(self, other)` or `None`, comparing as the signature renders (`1` and `'1'` are the
  same version because they are the same hash) while reporting both values as they are.
  Each takes its other side as a block, a journal entry, a declaration, or a `journal=`
  selector, as `diffnorm()` does.
- **`diffnorm()` descends recursively** into nested blocks and spec dicts, returning a
  *sparse* nested dict so a changed leaf appears at the end of a short path instead of
  as two multi-kilobyte strings. New options: `recursive=False` (previous flat
  behaviour), `deslash=True` (strip escapes from the reported values), `report=True`
  (flat `path` + self/other text), `maxlen` (truncation, report only).
- **`diffnorm()` reports TYPED leaves.** A norm is flat text, but the text records the
  type, so a non-`LEGACY_NORM` block's `ori_extent=15.0` comes back as the float `15.0`
  while a legacy block's comes back as the string `'15.0'` — making a pair like
  `(15.0, '15.0')` legible as "different `LEGACY_NORM` settings", not "the value changed".
  Detection still compares the raw text, so `n=1` vs `n=1.0` is reported even though
  `1 == 1.0` in Python; and where evaluation would erase the difference (bare vs quoted
  `url=`) the bytes are shown instead. `raw=True` returns the source text.
- **`norm(legacy=...)` / `supernorm(legacy=...)` / `diffnorm(legacy=...)`** — temporarily
  override `LEGACY_NORM` for one call, propagating to nested blocks so the whole subtree
  renders the same way. `None` (default) means every block uses its own flag and is
  byte-identical to before; `signature` never passes an override, so `hash` is unaffected.
  `a.diffnorm(b.norm(legacy=False), legacy=False)` gives typed leaves even for classes
  that still carry the marker.
- **`ABSENT`** — a key present on only one side of a `diffnorm` now carries this marker
  (`<absent>`) instead of `None`, which is no longer distinguishable from a value that
  genuinely *is* `None` now that leaves are typed.
- **`Datablock.format_diffnorm(diff)`** — renders a `diffnorm` dict as text.

### Fixed
- **`dbx.datajournal(frame, loc=..., iloc=...)` ignored `loc` and `iloc`**
  when handed a DataFrame to wrap, and returned the whole frame. It now returns
  the one entry, as it does for an anchor.
- **`OutputTee.close()` no longer hangs on a process that outlives it.** A
  process started while the tee was open, such as multiprocessing's resource
  tracker or a Ray raylet, inherited the pipe, and `close()` waited for an
  end-of-file that never came. `close()` now logs what is already in the pipe
  and stops logging. Anything such a process writes afterwards still reaches
  the terminal.
- **`entry.inst()` raised `AttributeError` on any entry journaled without a git
  repo.** An entry is a Series built with `dropna()`, so a column recorded null
  loses its LABEL, and attribute access — which a Series answers out of its index —
  raises where every reader expects `None`. `revision` and `gitrepo` are null
  whenever neither `DBX_GIT_REPO` nor `DBX_USE_WORK_REPO` is set (a pip-installed
  dbx, a container without the checkout), and `inst()` defaults to
  `revision='journal_entry'` — so it failed on exactly the entries that recorded no
  revision, taking `UNSAFE_redirect()` with it. `instantiate()` and `rinst()` now
  read both through `DatajournalEntry.column()`, and `None` means what
  `instantiate()`'s own default already means: the current environment. Latent
  since the `dropna()` arrived in `loc=`/`iloc=`; invisible wherever a repo is
  configured, since then the revision is a real sha.
- **A remote tab was cached where mosaic guessed, not where the table said.**
  `DatapointTable.datastream()` (and `DatapointFold`'s) computed a cache directory,
  created it, and then never passed it on, so every `Stream` over a remote tab was left
  without a `local=`. Mosaic then derives one itself — `{tmpdir}/{blake2s(remote)}`, the
  same path for every process on the box — and REFUSES to reuse it, so the second open of
  that tab (a second process, a second run, a retry after a crash) died with `Could not
  create a temporary local directory ... already exists`. It cannot simply be handed to
  `StreamingDataset`, which takes `streams=` or `remote`/`local` and never both: it
  belongs on each stream, one subdirectory per tab, named by the tab's hash so it is
  unique per tab and the same across runs. `_tab_stream()` now raises rather than letting
  a remote stream go without one. Local tabs are unaffected — a local slice is its own
  cache, and nothing is copied for it.
- **A journal on any non-local filesystem read back empty.** `Journal()` globbed its
  parquet files through the block's filesystem — which names them protocol-stripped — and
  then handed those paths to pandas, which looked for them on the LOCAL disk. Every entry
  was logged as "unreadable" and skipped, so a `memory://` or remote journal came back
  empty rather than failing. Entry files are now opened through that filesystem.
- **A table's tabs escaped to `DBX_ROOT`.** `DatapointTable.__tab__()` constructed its
  `TAB` without a url, so a table built anywhere other than the ambient root wrote its
  tabs to an unrelated one, where they were then looked for in vain under the table. A
  tab now inherits the table's url — raw, so a relocatable table stays relocatable tab by
  tab — and a `DatapointFold` likewise takes the partition's.
- **`Datacollator` could not consume what `data()` returns.** Every caller — a feature
  build, both probes — passes `data(*collator.slices, concat=True)`, a `{slice: data}`
  mapping in which the batch is already stacked; iterating that yields its KEYS, so the
  collation was over strings. Such a mapping is now collated as the batch it is: one pair
  passes its array through untouched, several are stacked along a new axis 1. New
  `signal_pairs`/`label_pairs` accessors give the normalized `(slice, column)` pairs.
- **`DatafeatureTab.__post_init__` and `FeatureStatsProbe.__build__` raised
  `NameError`** on names left behind by the move to `Datacollator` (`factory`,
  `feat_slice`/`sig_slice`): neither block could be constructed or built. The probe's
  per-tab breakdowns now follow the collator's first signal/label pair, and say so when
  it names more than one.
- **`SLICES` is rejected rather than ignored.** It is retired — a slice is a `TOPICS`
  entry valued `SLICETOPIC` — and a class still carrying one came out with no slices at
  all: valid, buildable, and empty. It now raises at construction, naming the
  replacement, like every other retired attribute.
- **`leave_breadcrumbs()` raised `IsADirectoryError` on any directory topic.** It passed
  `path(topic)` to `leave_breadcrumbs_at_path()`, which opened it for writing — but for a
  directory topic (list-TOPICS, or dict-TOPICS with `DIRTOPIC`) `path()` *is* the directory, so
  the call blew up and the method was unusable on such a block.
  `leave_breadcrumbs_at_path(path, crumbs=None)` now always takes a **directory** path:
  with `crumbs` the breadcrumb is `{path}/{crumbs}`, without it `{path}.crumbs` alongside.
  A file topic passes its own filename, so its breadcrumb is still its own empty file and
  the block still reads as valid; a directory topic gets the sibling marker rather than a
  stray entry inside a listing of itself. Breadcrumbs are now touched only when nothing is
  there, so they never clobber a real artifact. `SYNTOPIC` topics are skipped.
- **A filtered `Datajournal` kept the row labels of the unfiltered journal**, so
  `loc=`/`Datajournal.get()` — which index by label — raised `KeyError` for positions
  whose rows the filter had removed. `lastbuilt()` is `journal(event='build:end').get(0)`,
  so it failed for any block whose newest journal entry was some other event — the normal
  state for a block whose artifact was copied in (`UNSAFE_copy_from:END`). Filtered
  journals are now renumbered 0..N-1; frames you slice yourself keep pandas' label
  semantics.
- **`diffnorm(journal=...)` silently dropped extra selector keys**: `dict(event='build:end',
  iloc=0)` ignored `event` and returned the newest entry of *any* event. Extra keys are now
  forwarded to `journal()` as column filters; combining them with `entry_path` raises.

## [0.0.3] — 2026-05-12

### Breaking Changes
- **`journal()` API**: The `entry` parameter is renamed to `loc`; a new positional `iloc` parameter selects rows by integer position. Passing both raises `ValueError`.
- **`.dbx` metadata paths**: `fqcn` is now always included in the `.dbx/` directory hierarchy (`.dbx/{fqcn}/journal/`, `.dbx/{fqcn}/log/`, etc.), even when `anchor == fqcn`. Previously the `fqcn` segment was omitted in that case.
- **`parse_storage_options`**: Moved from a private helper in `datablocks.py` to the public `dataparts` module. Import as `from dbx.dataparts import parse_storage_options`.

### Added
- **`Datablock.lastbuilt()`** — returns the most recent `build:end` journal entry, or `None`.
- **`Datablock.running()`** — returns the latest `build:start` entry with no matching `build:end` (i.e. an in-progress build), or `None`.
- **`JournalEntry.bid`** property — reconstructs a `Datablock.Bid` namedtuple from a journal row.
- **`JournalFrame` hash prefix-match filter**: `journal(hash="ab3c")` now matches any hash starting with `"ab3c"`, so short hashes work.
- **`BUILD_TREE_EXEMPTIONS`**: Class-level set of spec keys to skip during `build_tree()`, allowing dependency graphs to be built without re-training checkpoint-based subtrees.
- **`dataparts.default_storage_options()`** — convenience wrapper that reads `DBX_STORAGE_OPTIONS` from the environment.
- **`storage_options` threading**: `JournalFrame` and `JournalEntry` now carry `storage_options` through the full read chain, fixing remote journal access.
- Serialization roundtrip tests for `Datablock` and `Datastack` (`deepcopy`, `pickle`, `__getstate__`/`__setstate__`).

### Fixed
- **`Datastack.executor_cls`**: Refactored from an `__init__`-set attribute to a property, so objects reconstructed via `deepcopy` or `pickle` (which bypass `__init__`) still resolve the correct executor.
- **`capture_output`**: Build logs are now written to a local temp file first, then uploaded to the (possibly remote) `logpath` on completion. Fixes failures when `fs.open()` on remote backends did not support streaming writes.
- **`fs_full_path` Azure fix**: Manually reconstructs the full `abfss://container@account.dfs.core.windows.net/path` form, because `adlfs`'s `unstrip_protocol()` drops the account portion.
- **`keyby='tag'` guard**: Raises `ValueError` immediately when `keyby='tag'` is specified without providing a `tag=` argument, preventing the infinite recursion that previously occurred.
- **TOPICS-only `path()`/`dirpath()`**: When a `Datablock` defines `TOPICS` without `TOPICFILES`, the topic is now correctly treated as a directory name under `anchorkeypath`, not derived from `path()`.
- **Dirty-repo check timing**: The uncommitted-changes check now runs in `gitwrkreposetup()` against the original repositories *before* cloning, so uncommitted work is never silently discarded.

## [0.0.2] — 2026-05-08

### Fixed
- **Remote storage path resolution**: `anchorkeypath`, `anchorpath()`, `dirpath()`, and `path()` now return correct protocol-prefixed URLs for remote storage backends (`abfs://`, `gcs://`, `memory://`, etc.). Previously, these methods returned protocol-free paths that silently resolved to the local filesystem.
- **`JournalEntry.anchorkeypath` / `anchorhashpath`**: Now derive root from the stored `url` field instead of referencing a non-existent `root` field. Legacy journal entries with an explicit `root` field are still supported.
- **`validpath()`**: Replaced inconsistent `os.path.exists()` / hardcoded `gcs` branches with a single `self.fs.exists()` call that works for any fsspec backend.
- Eliminated 14 redundant `_url_to_fs()` calls in favour of the already-initialised `self.fs`.

### Added
- `dataparts.fs_full_path(fs, path)` — utility that re-attaches the protocol prefix for remote filesystems while keeping local paths bare (compatible with Python's built-in `open()`).
- 29 new tests using the `memory://` filesystem to verify path correctness, build/valid lifecycle, journal round-trips, and `JournalEntry` path derivation on non-local storage.

## [0.0.1] — 2026-05-07

### Added
- Initial release on PyPI.
- `Datablock` base class with config-addressed storage, journaling, and fsspec-based IO.
- `Datastack` for parallel block builds (inline, multithreading, multiprocessing, Ray, Torch).
- `JournalEntry` / `JournalFrame` for structured build-event history.
- Environment-variable configuration (`DBX_ROOT`, `DBX_STORAGE_OPTIONS`, `DBX_DIRTY_REPO_OK`).
