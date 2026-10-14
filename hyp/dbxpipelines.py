"""Hyp over a journal of a dbx datalake (`HypDbxJournal`): its pipelines, ``hyp --journal``, or an anchor's blocks, ``hyp --journal ANCHOR``.

`dbx.pipejournal()` reads the datalake's pipeline journal and aggregates it:
one row per pipeline, its newest entry, with ``entrypaths`` -- every entrypoint
it was created to. Each row is a point, embedded from the pipeline's class
(``anchor``, its fqcn) and its spec and signature (or type) side files.

Right click opens the row. Its header has two menus: the pipeline journal's
other entries with the same tag, by datetime, to switch to; and the
entrypoints dbx found constructed (`pipejournal(construct_only=True)`), each
opening its own journal in a window over this one. Both are read in the
background once the points are up.

The datalake is dbx's -- ``DBX_DATALAKE`` -- and the corpus is kept in it,
under ``{datalake}/.hyp``. It may be any fsspec url; over ``ssh://host/path``,
sshfs connects through asyncssh, which resolves ``host`` from
``~/.ssh/config``. Side file paths are recorded as the host wrote them,
without a protocol, and are read through the protocol and host of the row's
``entry_path``.
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .app import Button, DiskViewer, Dropdown, Hyp, TextViewer

# Columns shown first, in this order; the rest follow as the frame has them,
# then the side files, then TAIL_COLUMNS.
HEAD_COLUMNS = ("anchor", "tag", "key", "event", "datetime", "entrypaths")
TAIL_COLUMNS = ("gitrepo", "entry_path", "wrkrepo")
# Columns holding paths to side files: their contents are shown, in this
# order, in place of the paths -- cite first, right below the table, and dfn
# right after it. The note is shown in the table, right after what can be
# read from the block: NOTE_AFTER.
SIDE_FILES = ("cite", "dfn", "spec", "kwargs", "signature", "type", "repr", "quote")
# The note follows the first of these the row has: as dbx records it.
NOTE_AFTER = ("slices", "paths", "topics")
# Side files shown deslashed: a dfn's speclines are dumped with their quotes escaped.
DESLASHED = ("dfn",)
# Side files that are yaml dumps of dicts: shown as the dicts they hold.
DICT_FILES = ("spec", "dfn", "kwargs")
# Path columns shown relative to the datalake, which is shown once.
LAKE_PATHS = ("anchorkeypath", "entrypaths")
# Columns holding a dict's repr: shown one key per line.
DICT_COLUMNS = ("topics", "paths", "slices")
SIDE_FILE_LIMIT = 16_000
# A captured output longer than this is shown by its end: what a run came to.
OUTPUT_LIMIT = 1_000_000
# The repr of what a topic reads is cut short past this.
READ_LIMIT = 200_000
# A slice's disk samples about this many rows unless told otherwise: "auto".
AUTO_ROWS = 4000
# Of a table's tablets, "auto" reads this many: each one costs a construction.
AUTO_TABLETS = 64
# The fractions of each tablet's rows, and of a table's tablets, the menus offer; None is auto.
ROW_FRACTIONS = (None, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
TABLET_FRACTIONS = (None, 0.01, 0.1, 0.25, 0.5, 1.0)
# Processes the pinned worker forks to read tablets in parallel; 1 reads them in it.
WORKERS = (1, 4, 8, 16, 32)
DEFAULT_WORKERS = 16
# The interpreter on the datalake's host whose Ray -- started there, for the
# session, by dbx.SshRay -- trueinst() instantiates blocks in: it needs dbx,
# ray (the version this one is) and the project's package. An entry records
# the one its build ran in, ``python``, and that one is used; HYP_REMOTE_PYTHON
# -- or --remote-python, for one launch -- overrides it, and FALLBACK_PYTHON
# stands in where an entry, written before entries recorded it, does not say.
REMOTE_PYTHON = __import__("os").environ.get("HYP_REMOTE_PYTHON") or None
FALLBACK_PYTHON = "python3"
EMBED_LIMIT = 4_000


class LabelFirstEmbedder:
    """An embedder for texts whose first line is a label: texts sharing a label are made to cluster.

    Each vector is two unit vectors side by side: the label's own, *weight*
    times over, and *base*'s embedding of the rest of the text. Under the
    cosine metric the layout uses, two texts with one label are then at
    least ``(w**2 - 1) / (w**2 + 1)`` alike -- 0.8 for the default 3 --
    however their rests differ, so they lie together; their rests place
    them within that. A label's vector is its character trigrams, hashed
    into `DIM` dimensions: one label is one vector, and labels sharing
    words -- ``gigapath_probe``, ``gigapath_probe_partition`` -- lie near.
    """

    DIM = 256

    def __init__(self, base, weight=3.0):
        self.base = base
        self.weight = weight
        self.name = f"label-first({weight:g}):{base.name}"

    def prepare(self, progress=print):
        self.base.prepare(progress)

    def __call__(self, texts):
        import numpy as np

        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        parts = [(t.split("\n", 1) + [""])[:2] for t in texts]
        rest = np.asarray(self.base([r or l for l, r in parts]), dtype=np.float32)
        rest = rest / np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
        labels = np.stack([self.label_vector(l) for l, _ in parts])
        return np.concatenate([self.weight * labels, rest], axis=1).astype(np.float32)

    @classmethod
    def label_vector(cls, label):
        """*label* as a unit vector: its character trigrams, each hashed to a dimension and a sign."""
        import zlib

        import numpy as np

        v = np.zeros(cls.DIM, dtype=np.float32)
        padded = f"_{label.strip().lower()}_"
        for i in range(max(1, len(padded) - 2)):
            h = zlib.crc32(padded[i : i + 3].encode())
            v[h % cls.DIM] += 1.0 if (h >> 16) & 1 else -1.0
        return v / max(float(np.linalg.norm(v)), 1e-12)


@dataclass(frozen=True)
class Entry:
    key: str  # the pipeline's anchorkeypath: what a re-sync matches by
    text: str  # every column and side file, shown on right click
    embed: str  # the label, on a line of its own; then anchor, spec, and signature or type
    summary: str  # the tag's last component and the datetime: the tooltip
    extra: dict = field(default_factory=dict)  # stored with the point


def rebase(path, url):
    """*path*, as the datalake's host recorded it, as a url readable from here.

    *url* is any url on the datalake's filesystem: dbx's ``entry_path``, the
    journal file a row was read from, names it in full.
    """
    m = re.match(r"^([a-z0-9+]+://[^/]+)/", str(url))
    if m and isinstance(path, str) and path.startswith("/"):
        return m[1] + path
    return path


def when(dt, seconds=False):
    """A journal datetime -- a Timestamp, or dbx's path-safe 2026-10-12T10-45-20 -- as 2026-10-12 10:45[:20]."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2})[-:](\d{2})[-:](\d{2})", str(dt))
    if not m:
        return str(dt)
    return f"{m[1]} {m[2]}:{m[3]}" + (f":{m[4]}" if seconds else "")


def dict_lines(text, indent="    "):
    """A dict's repr, one key per line at every level, as lines.

    ``topics`` and ``paths`` are recorded as ``str(dict)`` -- with
    ``DATAFILE(...)`` in them, so not literals to parse back -- and are laid
    out by their brackets instead: a dict opens a level, its commas end a
    line; tuples, lists and calls stay on one.
    """
    out, stack, quote = [], [], None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == quote and text[i - 1] != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "{" and text[i + 1 : i + 2] == "}":
            out.append("{}")
            i += 1
        elif ch == "{":
            stack.append(ch)
            out.append("{\n" + indent * len(stack))
            while text[i + 1 : i + 2] == " ":
                i += 1
        elif ch == "}" and stack:
            stack.pop()
            out.append("\n" + indent * len(stack) + "}")
        elif ch in "([":
            stack.append(ch)
            out.append(ch)
        elif ch in ")]" and stack:
            stack.pop()
            out.append(ch)
        elif ch == "," and stack and stack[-1] == "{":
            out.append(",\n" + indent * len(stack))
            while text[i + 1 : i + 2] == " ":
                i += 1
        else:
            out.append(ch)
        i += 1
    return "".join(out).split("\n")


def tail(tag):
    return str(tag).rsplit(".", 1)[-1]


def _is_blank_(v):
    import pandas as pd

    if isinstance(v, str):
        return v in ("", "None")  # dbx writes a missing value as "None"
    try:
        return bool(pd.isna(v))  # None, NaN, NaT
    except (TypeError, ValueError):  # array-likes: pd.isna answers elementwise
        return False


class _Tagged_:
    """A python object a yaml dump records -- ``!!python/object:...`` and kin -- as its tag and contents: never made."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name if self.value is None else f"{self.name}({self.value!r})"


def _inert_loader_():
    """A yaml loader making only data of a dbx dump: tuples as tuples, any other python tag as a `_Tagged_` -- importing and running nothing.

    The unsafe loader, which dbx reads its dumps back with, makes the objects
    a dump names -- a block's dfn holds its upstream blocks, and making them
    instantiates each, here, with whatever of their code this process has.
    """
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def tagged(loader, suffix, node):
        if suffix == "tuple":
            return tuple(loader.construct_sequence(node, deep=True))
        name = suffix.split(":", 1)[-1]
        if isinstance(node, yaml.MappingNode):
            return _Tagged_(name, loader.construct_mapping(node, deep=True))
        if isinstance(node, yaml.SequenceNode):
            return _Tagged_(name, loader.construct_sequence(node, deep=True))
        value = loader.construct_scalar(node)
        return _Tagged_(name, value or None)

    Loader.add_multi_constructor("tag:yaml.org,2002:python/", tagged)
    return Loader


def read_dump(path):
    """The yaml dump at *path* -- a spec, dfn or kwargs side file -- as data, with `_inert_loader_`."""
    import yaml
    from dbx.databits import read_str

    return yaml.load(read_str(path), Loader=_inert_loader_())


def _side_file_(column, path, url):
    """The side file *path* names, as ``(shown, compact)`` text; None where it cannot be read.

    For DICT_FILES, *shown* is the dict one key per line, a level per four
    spaces -- so that a viewer folds it by level -- and *compact* the dict
    pretty-printed, as points have always been embedded from: a change to
    how it is shown must not move every point. Otherwise both are the file.
    """
    import pprint

    import yaml
    from dbx.databits import read_str

    if not isinstance(path, str) or not re.search(r"\.(txt|yaml|yml|json)$", path):
        return None
    path = rebase(path, url)
    try:
        shown = compact = None
        if column in DICT_FILES and path.endswith((".yaml", ".yml")):
            try:
                # As data: the dumps hold python tuples, which the safe
                # loader does not make, and blocks, which must not be made.
                data = read_dump(path)
                shown = "\n".join(dict_lines(repr(data)))
                compact = pprint.pformat(data, width=100, sort_dicts=False)
            except yaml.YAMLError:
                pass  # shown as the yaml it is
        if shown is None:
            shown = compact = read_str(path)
    except (FileNotFoundError, OSError):
        return None

    def limited(text):
        return text if len(text) <= SIDE_FILE_LIMIT else text[:SIDE_FILE_LIMIT] + "\n…"

    return limited(shown), limited(compact)


def var_name(row):
    """The pipeline's ``VAR.name``, from the spec side file *row* names; None where it has none, or it cannot be read."""
    import yaml

    path = row.get("spec")
    if not isinstance(path, str) or not path.endswith((".yaml", ".yml")):
        return None
    try:
        spec = read_dump(rebase(path, row.get("entry_path", "")))
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return None
    name = spec.get("name") if isinstance(spec, dict) else None
    return None if name is None or str(name) == "" else str(name)


def entry(row):
    """One `dbx.pipejournal()` row, or any journal row, with its side files read, as an Entry."""
    row = {k: v for k, v in dict(row).items() if not _is_blank_(v)}
    url = row.get("entry_path", "")
    read = {c: t for c in SIDE_FILES if c in row and (t := _side_file_(c, row[c], url))}
    sides = {c: shown.replace("\\", "") if c in DESLASHED else shown for c, (shown, _) in read.items()}
    # The note, inline or in a side file of its own: shown as its text.
    note = row.get("note")
    if isinstance(note, str) and (t := _side_file_("note", note, url)):
        note = t[0]
    compact = {c: text for c, (_, text) in read.items()}
    lake = str(row.get("datalake", ""))

    def shown(c):
        v = row[c]
        if c in LAKE_PATHS:
            paths = [v] if isinstance(v, str) else list(v)
            return [p[len(lake) :].lstrip("/") if lake and str(p).startswith(lake) else str(p) for p in paths]
        if c in DICT_COLUMNS:
            return dict_lines(str(v))
        if c == "note":
            return str(note).rstrip().split("\n")
        return [str(v)]

    cols = [c for c in HEAD_COLUMNS if c in row] + [c for c in row if c not in HEAD_COLUMNS + TAIL_COLUMNS]
    cols = [c for c in cols if c not in sides and c != "note"]
    if note is not None:
        after = next((c for c in NOTE_AFTER if c in cols), None)
        cols.insert(cols.index(after) + 1 if after else len(cols), "note")
    tails = [c for c in TAIL_COLUMNS if c in row]
    width = max((len(c) for c in cols + tails), default=0)

    def table(columns):
        return [f"{(c if i == 0 else '').ljust(width)}  {v}" for c in columns for i, v in enumerate(shown(c))]

    lines = table(cols)
    for c, t in sides.items():
        lines += ["", f"── {c} ──", t.rstrip()]
    if tails:
        lines += [""] + table(tails)

    key = str(row.get("anchorkeypath") or f"{row.get('anchor')}/{row.get('hash')}")
    label = tail(row.get("tag", row.get("anchor", "?")))
    # The label first, on a line of its own: what LabelFirstEmbedder clusters by.
    embed = [label, str(row.get("anchor", ""))]
    embed += [compact[c] for c in ("spec",) if c in compact]
    embed += [compact[c] for c in ("signature", "type") if c in compact][:1]
    name = var_name(row)
    return Entry(
        key=key,
        text="\n".join(lines),
        embed="\n".join(embed)[:EMBED_LIMIT],
        summary=f"{label} · {when(row['datetime'])}" if "datetime" in row else label,
        extra={"anchor": str(row.get("anchor", "")), "label": label, "name": name, "row": {k: str(v) for k, v in row.items()}},
    )


def _dbx_():
    # Imported when a corpus is read, not with hyp: the seed corpus needs no
    # dbx, and importing it takes seconds.
    import dbx

    return dbx


def datalake():
    """dbx's datalake -- ``DBX_DATALAKE``, then ``DBX_ROOT``, ``DBX_URL`` -- where hyp reads the pipelines and keeps its corpus."""
    lake = _dbx_().default_datalake()
    if not lake:
        raise SystemExit("hyp --journal: no datalake; set DBX_DATALAKE")
    return lake


def _counted_(progress, msg):
    """A progress callback for dbx's journal reads -- ``(desc, done, total)`` -- telling *progress* *msg*, with the counts.

    `DataJournal.read` lists the journal's files before it reads them: while
    it does, there is no total, and *done* is the files found so far.
    """
    return lambda desc, done, total: progress(msg if total is not None else f"{msg}: listing the journal files", done, total)


def _entries_(df, progress, msg, n_workers):
    """`entry` of each row of *df*, in order, on *n_workers* threads: *msg*, and how many are done, told to *progress* as each is.

    An exception *progress* raises -- the user interrupting -- cancels the
    rows not yet begun, and is raised on.
    """
    out = [None] * len(df)
    progress(msg, 0, len(df))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(entry, df.iloc[i]): i for i in range(len(df))}
        try:
            for done, future in enumerate(as_completed(futures), 1):
                out[futures[future]] = future.result()
                progress(msg, done, len(df))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return out


def pipelines(progress=print, n_workers=16):
    """Every pipeline in dbx's datalake, from `dbx.pipejournal()`, newest first.

    *n_workers* threads list and read the journal -- `dbx.pipejournal()`
    parallelizes with threads only, which suits it: the work is round trips
    to the datalake -- and then read the side files. *progress* is told
    each stage and, where they are counted, how many files of it are read.
    """
    progress("dbx.pipejournal()")
    df = _dbx_().pipejournal(n_workers=n_workers, progress=_counted_(progress, "dbx.pipejournal()"))
    return _entries_(df, progress, f"reading the side files of {len(df)} pipelines", n_workers)


def blocks(anchor, progress=print, n_workers=16, filters=None):
    """Every block of *anchor* in dbx's datalake, from its block journal: one each, its newest entry, newest first.

    *filters* -- ``{column: pattern}``, as `parse_filters` makes them -- are
    the journal's own: `DataJournal.read` applies them, to the entries, before
    a block's newest is taken. A pipeline's entries -- filed under the pipeline
    journal, ``.pipelines`` -- are not among them.
    """
    import pandas as pd

    filters = dict(filters or {})
    msg = f"dbx.DataJournal.read({anchor!r}{''.join(f', {k}={v!r}' for k, v in filters.items())})"
    progress(msg)
    try:
        df = pd.DataFrame(_dbx_().DataJournal.read(anchor, datalake=datalake(), n_workers=n_workers, index=None,
                                                   progress=_counted_(progress, msg), **filters))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"no journal of {anchor} in {datalake()}: {e}") from e
    if not len(df):
        return []
    if "entry_path" in df.columns:
        df = df[~df["entry_path"].astype(str).str.contains("/.pipelines/", regex=False)]
    if "datetime" in df.columns:
        df = df.sort_values("datetime", ascending=False, kind="stable")
    if "anchorkeypath" in df.columns:
        df = df.drop_duplicates("anchorkeypath", keep="first")
    df = df.reset_index(drop=True)
    return _entries_(df, progress, f"reading the side files of {len(df)} blocks of {anchor}", n_workers)


#: What the journal menu calls the pipeline journal: no anchor can be named so, starting with a dot.
PIPELINES = ".pipelines"


def anchor_of(journal):
    """The anchor ``--journal`` names, or None for the pipelines: none named, ``pipelines`` or ``.pipelines``."""
    return None if journal in (None, "pipelines", PIPELINES) else journal


def corpus_dir(anchor=None):
    """Where a journal's corpus is kept: the pipelines', ``{datalake}/.hyp``; an anchor's, ``{datalake}/.hyp/anchors/{anchor}``."""
    lake = datalake().rstrip("/")
    return f"{lake}/.hyp" if anchor is None else f"{lake}/.hyp/anchors/{anchor}"


def journal_source(anchor=None, filters=None, n_workers=16):
    """The corpus source of a journal: the pipelines, or *anchor*'s blocks with *filters*."""
    if anchor is None:
        return lambda progress: pipelines(progress, n_workers=n_workers)
    return lambda progress: blocks(anchor, progress, n_workers=n_workers, filters=filters)


def parse_filters(text):
    """``key=value, key=value, ...`` as ``{key: value}``: each value a pattern, as the journal's filters take them.

    A comma splits only before another ``key=``, so a value may hold one --
    ``tag=a{1,3}`` -- and ``|`` alternates, as in any pattern: ``event=build:end|note``.
    """
    out = {}
    for part in re.split(r",\s*(?=[\w.:-]+\s*=)", (text or "").strip()):
        part = part.strip().rstrip(",").strip()
        if not part:
            continue
        key, eq, value = part.partition("=")
        if not eq or not key.strip():
            raise ValueError(f"{part!r} is not key=value")
        out[key.strip()] = value.strip()
    return out


def format_filters(filters):
    return ", ".join(f"{k}={v}" for k, v in (filters or {}).items())


def journal_entries(n_workers=16):
    """Every entry of the pipeline journal, unaggregated, newest first, as a DataFrame.

    With a ``name`` column: each entry's `var_name`, read from its spec side
    file on *n_workers* threads -- what tells apart the runs the tag menu lists.
    """
    import pandas as pd

    dbx = _dbx_()
    df = dbx.PipeJournal.read(dbx.PipeJournal.ANCHOR, datalake=datalake(), n_workers=n_workers, index=None)
    df = pd.DataFrame(df).reset_index(drop=True)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        df["name"] = list(ex.map(lambda i: var_name(df.iloc[i]), range(len(df))))
    return df


def constructed_entrypaths(n_workers=16, extra_events=()):
    """``{anchorkeypath: [entrypath]}``: each pipeline's entry blocks that were built, redirected or noted -- or have an event of *extra_events*."""
    df = _dbx_().pipejournal(construct_only=True, extra_construction_events=extra_events, n_workers=n_workers)
    if not len(df):
        return {}
    return {str(r["anchorkeypath"]): [str(p) for p in r["entrypaths"]] for _, r in df.iterrows()}


def has_output(row):
    """Whether the process that wrote *row*'s entry had its output captured: dbx records where, as ``log``."""
    return bool(row) and not _is_blank_(row.get("log", None))


def output_text(row, limit=OUTPUT_LIMIT):
    """The captured stdout and stderr of the process that wrote *row*'s entry, as text: its end, past *limit* bytes.

    ``log`` is recorded as the datalake's host wrote it and is read, as the
    side files are, through the protocol and host of the row's
    ``entry_path``. A capture is uploaded when its command ends, so one not
    there yet may be of a command still running. Escape codes are dropped,
    and a line redrawn with carriage returns -- a progress bar -- is shown as
    it was last drawn.
    """
    import fsspec

    path = rebase(row["log"], row.get("entry_path", ""))
    fs, p = fsspec.url_to_fs(path, **_dbx_().default_storage_options())
    try:
        size = fs.size(p)
    except FileNotFoundError:
        return f"{path}\n\nNot there. A capture is uploaded when its command ends: it may still be running."
    with fs.open(p, "rb") as f:
        if size > limit:
            f.seek(size - limit)
        data = f.read()
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", data.decode("utf-8", errors="replace"))
    lines = [ln.rstrip("\r").split("\r")[-1] for ln in text.split("\n")]
    head = [path] + ([f"… its first {size - limit:,} of {size:,} bytes left out"] if size > limit else [])
    return "\n".join(head + [""] + lines)


def session_blocks(row, n_workers=16):
    """The blocks journaled under *row*'s session: ``[(anchorkeypath, last event)]``, in the order first written.

    From the index a ``dbx.exec`` command keeps of every entry written under
    its session, by any process -- `DataJournal.session_entries` -- under
    ``{datalake}/.journal/sessions/{session}/``. The index and the entries it
    lists are recorded as the datalake's host wrote them, and are read, as the
    side files are, through the row's ``entry_path``. A session no command
    ran -- a block built outside ``dbx.exec`` -- has no index, and no blocks.
    A pipeline's entries are left out: they are its creations, filed in the
    pipeline journal, and its block directory holds none of them.
    """
    import pandas as pd

    dbx = _dbx_()
    session, lake, url = row.get("session"), str(row.get("datalake", "")), row.get("entry_path", "")
    if _is_blank_(session) or not lake:
        return []
    options = dbx.default_storage_options()
    paths = dbx.DataJournal.session_entries(session, datalake=rebase(lake, url), storage_options=options)
    if not paths:
        return []
    df = pd.DataFrame(dbx.DataJournal.read_frame([rebase(p, url) for p in paths], storage_options=options, n_workers=n_workers))
    blocks = {}
    for _, r in df.iterrows():
        if not _is_blank_(r.get("anchorkeypath")) and "/.pipelines/" not in str(r.get("entry_path", "")):
            blocks[str(r["anchorkeypath"])] = str(r.get("event", ""))
    return list(blocks.items())


def _entry_(row):
    """*row* -- a block's journal row, as a window keeps one -- as a `DataJournalEntry` whose filesystem is the datalake's, over ssh if it is remote."""
    import pandas as pd

    dbx = _dbx_()
    lake = rebase(str(row.get("datalake", "")), row.get("entry_path", ""))
    return dbx.DataJournalEntry(pd.Series({**row, "datalake": lake}), storage_options=dbx.default_storage_options())


def topics(row):
    """The leaf topics *row*'s block recorded, ``[(topicpath, stored)]``: *stored* False for a synthetic one, which has no location."""
    from dbx.datablocks import SYNTHETIC, SYNTOPIC, is_topicmarker

    if _is_blank_(row.get("topics") if row else None):
        return []
    out = []

    def walk(node, path):
        if isinstance(node, dict):
            for name, child in node.items():
                walk(child, path + (name,))
        else:
            out.append((path, not (node == SYNTOPIC or is_topicmarker(node, SYNTHETIC))))

    try:
        walk(_entry_(row).block.TOPICS, ())
    except Exception:
        return []
    return out


def ls_text(row, topic):
    """What is at the path *row*'s block recorded for *topic*: one line per entry, its size and its name.

    The block's own `ls`, over its recorded paths -- nothing is instantiated.
    """
    entry = _entry_(row)
    listing = entry.block.ls(*topic, detail=True)
    head = [f"ls {'/'.join(topic)} of {row.get('anchorkeypath', '?')}", ""]
    if not listing:
        return "\n".join(head + ["nothing there"])
    lines = []
    for e in sorted(listing, key=lambda e: str(e.get("name"))):
        kind = e.get("type")
        size = "dir" if kind == "directory" else f"{e.get('size') or 0:,}"
        lines.append(f"{size:>16}  {e.get('name')}")
    return "\n".join(head + lines)


# The block read_text last instantiated, and what it is of: kept, so that
# reading it again does not start another worker. One at a time: each
# trueinst() connects Ray afresh, which the one before does not survive.
_REMOTE_ = {"key": None, "block": None, "quote": None}
_REMOTE_LOCK_ = threading.Lock()


def _where_(row):
    """The host *row*'s entry is on -- as its entry_path names it -- or None for this one."""
    m = re.match(r"^(?:ssh|sftp)://([^/:]+)(?::\d+)?/", str(row.get("entry_path", "")))
    return m[1] if m else None


def remote_python(entry, python=None):
    """The interpreter to run *entry*'s block in: *python* if given, else the one its build recorded, else FALLBACK_PYTHON."""
    if python:
        return python
    recorded = _dbx_().DataJournalEntry.column(entry, "python")
    return FALLBACK_PYTHON if _is_blank_(recorded) else str(recorded)


def _remote_block_(row, python):
    """*row*'s block, by ``trueinst()`` where its data are: kept for the next call. Called holding `_REMOTE_LOCK_`."""
    dbx = _dbx_()
    url = str(row.get("entry_path", ""))
    key = (url, str(row.get("id")))
    if _REMOTE_["key"] != key:
        entry = dbx.DataJournal.read_entries([url])[0]
        kwargs = {}
        host = _where_(row)
        if host:
            cluster = dbx.SshRay.session(host, python=remote_python(entry, python))
            kwargs = {"address": cluster.address, "gitrepo": dbx.DataJournalEntry.column(entry, "gitrepo")}
        _REMOTE_.update(key=None, block=None, quote=None)
        _REMOTE_["block"] = entry.trueinst(**kwargs)
        _REMOTE_["quote"] = entry.read("quote", raw=True)
        _REMOTE_["key"] = key
    return _REMOTE_["block"]


def recorded_slices(row):
    """The slices *row*'s entry records its block can be read by -- its ``slices`` -- by name; none where it records none."""
    import ast

    text = row.get("slices") if row else None
    if _is_blank_(text):
        return []
    try:
        slices = ast.literal_eval(str(text))
    except (ValueError, SyntaxError):
        return []
    return list(slices) if isinstance(slices, dict) else []


def declared_columns(row):
    """``{slice: {column: type}}``: the columns *row*'s recorded slices declare -- their ``DATASLICE(...)`` -- by slice.

    What can be viewed of a block, and how: a slice that declares none -- a
    bare ``DATASLICE`` -- says nothing of what its rows hold, and is left out.
    """
    import ast

    text = row.get("slices") if row else None
    if _is_blank_(text):
        return {}
    try:
        slices = ast.literal_eval(str(text))
    except (ValueError, SyntaxError):
        return {}
    return {s: dict(cols) for s, cols in slices.items() if isinstance(cols, dict) and cols} if isinstance(slices, dict) else {}


def column_choices(row):
    """``[(slice, column, path, type)]``: what can be viewed of *row*'s block, a column or -- a dict column -- each key in it.

    A column the journal records by its declared structure, a dict, is
    offered key by key, down to its leaves -- *path* the keys -- and read as
    ``data((slice, [(column, *path)]))`` reads one; a column recorded by its
    type alone is offered whole, *path* empty.
    """
    out = []

    def leaves(node, path):
        if isinstance(node, dict) and node:
            for k, v in node.items():
                yield from leaves(v, path + (k,))
        else:
            yield path, node

    for s, cols in declared_columns(row).items():
        for c, declared in cols.items():
            for path, kind in leaves(declared, ()):
                out.append((s, c, path, kind if isinstance(kind, str) else type(kind).__name__))
    return out


def _selection_(pick):
    """What ``data()`` reads for a *pick*, ``(slice, column, path)``: the column, or ``(column, *path)``."""
    _, c, path = pick
    return (c, *path) if path else c


def _remote_fns_():
    """``(describe, sample)``: what runs in the pinned worker, defined here so that it is sent by value.

    The worker runs the entry's revision of dbx and the project, and no hyp:
    a function of this module's would be sent by reference, and not found
    there. Nested, each is pickled whole. Both rebuild the block from its
    quote -- the instance trueinst() made, made again where it is -- and send
    back only plain values: a streaming dataset, or anything else bound to
    the process it was made in, cannot travel.
    """

    def describe(quote, slices):
        """``{"tablets": n, "rows0": {slice: rows}}``: a table's tablets, and each slice's rows in its first -- what a sample is planned by."""
        import dbx

        b = dbx.eval(quote)
        n = getattr(b, "n_tablets", None)
        first = b.tablet(0) if n else b
        rows0 = {}
        for s in slices:
            try:
                rows0[s] = int(first.n_rows(s))
            except Exception:
                rows0[s] = None
        return {"tablets": n or 1, "rows0": rows0}

    def sample(quote, specs, tablets, fraction, per_tablet, seed, workers, data, labels):
        """A *fraction* of the rows of each of *tablets* -- or *per_tablet* of them -- read *workers* at a time: ``{"X", "labels", "rows", "errors"}``.

        Each tablet is constructed and read whole -- its rows together, as its
        shards hold them -- in a process forked from this one, which has the
        pinned code already; then its sample is taken, the rows `sample_rows`
        picks with *seed* + its index. *rows* are ``(tablet, row)``.
        """
        import multiprocessing
        import queue as queues

        import dbx
        import numpy as np

        b = dbx.eval(quote)
        tabled = bool(getattr(b, "n_tablets", None))
        data = (*data, ()) if len(data) == 2 else tuple(data)
        labels = None if labels is None else ((*labels, ()) if len(labels) == 2 else tuple(labels))

        def picked(n, i):
            if n <= 0:
                return []
            k = min(n, per_tablet) if per_tablet else min(n, max(1, round(n * fraction)))
            if k == n:
                return list(range(n))
            return sorted(np.random.default_rng(seed + i).choice(n, size=k, replace=False).tolist())

        def at(value, path):
            """*value* at *path*, where data() handed back more of it -- a dict pruned to two keys of one column."""
            for k in path:
                if not (isinstance(value, dict) and k in value):
                    break
                value = value[k]
            return value

        def scalar(v):
            return v.item() if hasattr(v, "item") and getattr(v, "ndim", 0) == 0 else v

        def one(i):
            got = (b.tablet(i) if tabled else b).data(*specs)
            values = got[data[0]][data[1]]
            rows = picked(len(values), i)
            X = np.stack([np.asarray(at(values[r], data[2]), dtype=np.float32).reshape(-1) for r in rows]) if rows else None
            L = None
            if labels is not None:
                lv = got[labels[0]][labels[1]]
                L = [str(scalar(at(lv[r], labels[2]))) for r in rows]
            return i, X, L, rows

        results, errors = [], []
        if workers <= 1 or len(tablets) <= 1:
            for i in tablets:
                try:
                    results.append(one(i))
                except Exception as e:
                    errors.append(f"tablet {i}: {type(e).__name__}: {e}"[:400])
        else:
            ctx = multiprocessing.get_context("fork")
            q = ctx.Queue()

            def run(chunk):
                for i in chunk:
                    try:
                        q.put(("ok", one(i)))
                    except Exception as e:
                        q.put(("error", f"tablet {i}: {type(e).__name__}: {e}"[:400]))

            procs = [ctx.Process(target=run, args=(tablets[k::workers],), daemon=True) for k in range(min(workers, len(tablets)))]
            for p in procs:
                p.start()
            while len(results) + len(errors) < len(tablets):
                try:
                    status, payload = q.get(timeout=10)
                except queues.Empty:
                    if not any(p.is_alive() for p in procs):
                        errors.append(f"{len(tablets) - len(results) - len(errors)} tablets never came back: a worker died")
                        break
                    continue
                (results if status == "ok" else errors).append(payload)
            for p in procs:
                p.join(timeout=5)
        results = [r for r in sorted(results, key=lambda r: r[0]) if r[1] is not None]
        if not results:
            raise RuntimeError("no rows read: " + ("; ".join(errors[:3]) or "no tablets"))
        return {
            "X": np.concatenate([r[1] for r in results]),
            "labels": [l for r in results for l in r[2]] if labels is not None else None,
            "rows": [(r[0], row) for r in results for row in r[3]],
            "errors": errors,
        }

    return describe, sample


def _quote_and_handle_(row, python):
    """The quote of *row*'s block and the pinned worker its ``trueinst()`` runs in. Called holding `_REMOTE_LOCK_`."""
    block = _remote_block_(row, python)
    return _REMOTE_["quote"], block._origin


def slice_plan(row, python=REMOTE_PYTHON):
    """What a sample of *row*'s block is planned by: ``{"tablets": n, "rows0": {slice: rows}}``.

    A table's tablets, and the rows of each declared slice in its first --
    a table has about *tablets* times as many. Read in the worker
    ``trueinst()`` pins to the entry's revision, where the data are.
    """
    describe, _ = _remote_fns_()
    with _REMOTE_LOCK_:
        quote, handle = _quote_and_handle_(row, python)
        return handle.run(describe, quote, list(declared_columns(row)))


def slice_sample(row, data, labels=None, fraction=1.0, tablets=None, workers=DEFAULT_WORKERS, python=REMOTE_PYTHON, seed=0,
                 per_tablet=None):
    """A sample of *row*'s block -- *fraction* of the rows of each of *tablets*, or *per_tablet* of them -- as ``(X, labels, rows, errors)``.

    *data* and *labels* are ``(slice, column, path)`` -- *path* the keys of a
    dict column, empty for all of it -- labels None for none. Read
    in the pinned worker on the datalake's host, *workers* tablets at a time;
    *X* is the data column, a row a vector. *tablets* are indices, all of a
    table's when None.
    """
    _, sample = _remote_fns_()
    data = (*data, ()) if len(data) == 2 else tuple(data)
    labels = None if labels is None else ((*labels, ()) if len(labels) == 2 else tuple(labels))
    specs = {}
    for pick in [data] + ([labels] if labels is not None else []):
        selections = specs.setdefault(pick[0], [])
        if _selection_(pick) not in selections:
            selections.append(_selection_(pick))
    specs = list(specs.items())
    with _REMOTE_LOCK_:
        quote, handle = _quote_and_handle_(row, python)
        got = handle.run(sample, quote, specs, list(tablets) if tablets is not None else [0], fraction, per_tablet, seed, workers,
                         data, labels)
    return got["X"], got["labels"], got["rows"], got["errors"]


def _sample_indices_(n, fraction, seed=0):
    """The rows dbx's ``sample_rows`` picks of *n*, sorted: as it does, here."""
    import numpy as np

    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    k = min(n, max(1, round(n * fraction)))
    if k == n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def pick_tablets(n, fraction, seed=0):
    """Which of a table's *n* tablets a *fraction* of them is, sorted."""
    return [int(i) for i in _sample_indices_(n, fraction, seed)]


def read_text(row, topic, python=REMOTE_PYTHON, limit=READ_LIMIT):
    """``read(*topic)`` of *row*'s block, instantiated with ``trueinst()`` where its data are, as text.

    trueinst() pins a Ray worker to the entry's own revision of dbx and of
    the project, so the block comes back with the hash it was built under,
    reading the data that hash names. For a datalake reached over ssh, the
    worker is on its host, in the Ray head `dbx.SshRay` keeps there for the
    session, run by *python*; the block, a proxy to it here. The block is
    kept for the next read of it. What comes back is the value's repr -- taken
    there, when the value stays there behind a proxy -- cut short past *limit*;
    or the traceback.
    """
    import traceback

    where = _where_(row) or "here"
    head = f"read {'/'.join(topic)} of {row.get('anchorkeypath', '?')}  -- trueinst() on {where}"
    try:
        with _REMOTE_LOCK_:
            value = _remote_block_(row, python).read(*topic)
            if type(value).__name__ == "Remote":
                # Left there, behind a proxy: it forwards a dunder only when
                # asked for by name -- repr() would describe the proxy.
                text, kind = value.__getattr__("__repr__")(), f"(a proxy to the value on {where})"
            else:
                text, kind = repr(value), f"{type(value).__module__}.{type(value).__qualname__}"
    except Exception:
        return "\n".join([head, "", traceback.format_exc()])
    cut = f"   (its first {limit:,} of {len(text):,} characters)" if len(text) > limit else ""
    return "\n".join([head, f"{kind}{cut}", "", text[:limit]])


def entrypoint_text(entrypath, row, n_workers=16):
    """The journal of the block at *entrypath*: its events, then its newest entry as `entry` shows one.

    *row* is a pipeline's journal row: its ``datalake`` and ``entry_path`` say
    where *entrypath*, recorded as the host wrote it, is read from. Returns
    ``(text, newest)``: *newest* the newest entry's row, as `entry` keeps
    one, or None where there is no entry.
    """
    import pandas as pd

    dbx = _dbx_()
    lake = str(row.get("datalake", ""))
    if not lake or not entrypath.startswith(lake):
        raise ValueError(f"{entrypath} is not under the datalake {lake!r}")
    # The block's directory as an anchor of its own: its journal, and nothing
    # else in the anchor, is read -- as PipeJournal reads .pipelines.
    under = entrypath[len(lake) :].strip("/")
    df = pd.DataFrame(dbx.DataJournal.read(under, datalake=rebase(lake, row.get("entry_path", "")), n_workers=n_workers, index=None))
    if "anchorkeypath" in df.columns:
        df = df[df["anchorkeypath"].astype(str) == entrypath]
    if not len(df):
        return f"No journal entries for {entrypath}", None
    events = [f"{when(r.get('datetime'), seconds=True)}  {r.get('event', '')}" for _, r in df.iterrows()]
    newest = entry(df.iloc[0])
    return "\n".join(["── events ──", *events, "", newest.text]), newest.extra["row"]


class HypDbxJournal(Hyp):
    """A journal of a dbx datalake on the disk: its pipelines, one point each, or an anchor's blocks.

    Which, the journal menu in the top bar says -- ``--journal [ANCHOR]`` at
    the start -- and switches, each journal a corpus of its own. An anchor's
    blocks may be filtered, by the filter field beside it: ``key=value, ...``,
    each a pattern on the journal's column *key*.

    A pipeline opens its window, its entry blocks and session blocks listed;
    a block opens the window an entry block does. A right click lists a
    pipeline's entry blocks, or a block's session's blocks.
    """

    title = "hyp: dbx pipelines"
    HINTS = {
        "pipelines": "click: open · right-click: entry blocks · drag: move · scroll: zoom · Home or double-click outside: reset",
        "anchor": "click: open · right-click: session blocks · drag: move · scroll: zoom · Home or double-click outside: reset",
    }
    HINT = HINTS["pipelines"]

    @classmethod
    def __default_data__(cls, args):
        return corpus_dir(anchor_of(getattr(args, "journal", None)))

    @classmethod
    def __embedder__(cls, embedder):
        """Pipelines sharing a label cluster: LabelFirstEmbedder, over the embedder --embedder names for the rest."""
        return LabelFirstEmbedder(embedder)

    @classmethod
    def __storage_options__(cls, args):
        return _dbx_().default_storage_options()

    @classmethod
    def __source__(cls, args):
        return journal_source(anchor_of(getattr(args, "journal", None)), parse_filters(getattr(args, "filter", None)), args.n_workers)

    def __init__(self, engine, worker, args=None):
        from .app import TextField

        self.anchor = anchor_of(getattr(args, "journal", None))
        self.filters = parse_filters(getattr(args, "filter", None))
        super().__init__(engine, worker, args)
        self.anchors = None  # every anchor of the datalake, for the journal menu; None until read
        self.journal_menu = Dropdown(self._journal_label_(), self._journal_items_, self._pick_journal_, empty="no anchors")
        self.filter_field = TextField("filter", self._apply_filter_, placeholder="key=pattern, key=pattern, ...",
                                      visible=lambda: self.anchor is not None, text=format_filters(self.filters))
        self.controls = [self.journal_menu, self.filter_field]  # right to left from points: the journal, then its filter
        self._titled_()
        # A pipeline's VAR.name and version, under its label before the datetime: listed first.
        self.settings = {"names": True, "versions": True, **self.settings}
        self.n_workers = getattr(args, "n_workers", 16)
        # Events that make an entry block count as constructed, besides dbx's own: --extra-construction-events.
        self.extra_events = tuple(getattr(args, "extra_construction_events", None) or ())
        self.remote_python = getattr(args, "remote_python", None) or REMOTE_PYTHON
        self.journal = None  # every pipeline journal entry, newest first; None until read
        self.constructed = None  # {anchorkeypath: [entrypath]}; None until read
        self.sessions = {}  # {session: [(anchorkeypath, last event)]}; None while read
        self.slice_info = {}  # {(entry_path, id): {slice: (rows, {column: kind})}}; None while read
        self._read_journals_()
        self.run_async(lambda: _dbx_().anchors(datalake()), lambda a: setattr(self, "anchors", a))

    def __reload__(self):
        """The journal, the constructed entry blocks, the session indexes and the anchors, read again with the points."""
        self.journal = self.constructed = None
        self.sessions = {}
        self.slice_info = {}
        self._read_journals_()
        self.run_async(lambda: _dbx_().anchors(datalake()), lambda a: setattr(self, "anchors", a))

    # The journal shown, and switching it ------------------------------------------

    def _titled_(self):
        self.title = "hyp: dbx pipelines" if self.anchor is None else f"hyp: dbx {self.anchor}"
        self.HINT = self.HINTS["pipelines" if self.anchor is None else "anchor"]
        try:
            import pygame

            pygame.display.set_caption(self.title)
        except Exception:
            pass

    def _journal_label_(self):
        name = "pipelines" if self.anchor is None else self.anchor
        if len(name) > 44:
            name = "…" + name[-43:]
        return f"journal: {name}"

    def _journal_items_(self):
        """The journal menu: the pipelines, then every anchor of the datalake; None while the anchors are read."""
        if self.anchors is None:
            return None
        return [("pipelines", PIPELINES)] + [(a, a) for a in self.anchors]

    def _pick_journal_(self, value):
        anchor = anchor_of(value)
        self.switch(anchor, self.filters if anchor == self.anchor else {})

    def _apply_filter_(self, text):
        try:
            filters = parse_filters(text)
        except ValueError as e:
            self.error, self.error_at = f"filter: {e}", __import__("time").monotonic()
            return
        self.switch(self.anchor, filters)

    def switch(self, anchor, filters=None):
        """Show another journal -- the pipelines, for *anchor* None, or *anchor*'s blocks with *filters* -- read, embedded and laid out in the background."""
        from .store import Store

        self.anchor, self.filters = anchor, dict(filters or {})
        self.filter_field.text = self.filter_field.committed = format_filters(self.filters)
        self.journal_menu.label = self._journal_label_()
        self._titled_()
        self.viewers, self.popup, self.snap, self.tween = [], None, None, None
        self.journal = self.constructed = None
        self.sessions, self.slice_info = {}, {}
        self._read_journals_()
        data, source = corpus_dir(anchor), journal_source(anchor, self.filters, self.n_workers)
        options = _dbx_().default_storage_options()
        engine, events = self.engine, self.worker.events

        def job(progress):
            progress(f"opening the corpus at {data}")
            engine.switch(Store(data, options), source)
            engine.start(progress)
            events.put(("snapshot", (engine.snapshot(), None)))
            progress("ready")

        self.worker.submit(job)

    def _read_journals_(self):
        """What the windows list besides the points, read in the background: every entry of the pipeline journal, and the entry blocks.

        An anchor's blocks need neither: they are the points, and a block's window reads its own journal.
        """
        if self.anchor is not None:
            self.journal, self.constructed = None, {}
            return
        self.run_async(lambda: journal_entries(self.n_workers), lambda df: setattr(self, "journal", df))
        self.run_async(lambda: constructed_entrypaths(self.n_workers, self.extra_events), lambda d: setattr(self, "constructed", d))

    def __point_label__(self, i):
        return self.snap.points[i].get("label") or self.snap.summaries[i]

    def __point_name__(self, i):
        return self.snap.points[i].get("name")

    def __point_version__(self, i):
        version = self.snap.points[i].get("row", {}).get("version")
        return None if _is_blank_(version) else str(version)

    def __point_datetime__(self, i):
        dt = self.snap.points[i].get("row", {}).get("datetime")
        return when(dt) if dt else None

    def __click_point__(self, i, now):
        """Left click on a pipeline, or a block: its window."""
        self.__open_point__(i)

    def __context_point__(self, i, pos):
        """Right click on a pipeline: its constructed entry blocks, at the mouse; picking one opens its window, over the disk.

        On a block: the blocks of its session, likewise.
        """
        row = self.snap.points[i].get("row", {})
        if self.anchor is not None:
            return self._block_context_(i, row, pos)
        self.open_popup(lambda: self._entrypoints_(row), lambda p: self._open_entrypoint_(row, p, self.title_color(i)), pos,
                        empty="none constructed", **self._copies_anchorkey_(row))

    def __open_point__(self, i):
        row = self.snap.points[i].get("row", {})
        if self.anchor is not None:
            # A block: the window an entry block opens in, its title a menu of the anchor's blocks.
            self.viewers = []
            return self._open_entrypoint_(row, str(row.get("anchorkeypath", "")), self.title_color(i))
        viewer = TextViewer(self.snap.summaries[i], self.snap.texts[i])
        viewer.row = row
        viewer.title_color = self.title_color(i)
        times = Dropdown(
            self._title_(row, self.snap.points[i].get("name")),
            lambda: self._datetimes_(row.get("tag")),
            lambda k: self._show_entry_(viewer, k),
            empty="no entries",
            match=lambda k: self._is_entry_(k, viewer.row),
        )
        roots = Dropdown("constructed entry blocks", lambda: self._entrypoints_(viewer.row), lambda p: self._open_entrypoint_(viewer.row, p, viewer.title_color),
                         empty="none constructed", **self._copies_anchorkey_(viewer.row))
        # The blocks journaled under the pipeline's session -- as an entry block's window lists them -- each opened as one.
        session = Dropdown(
            "session blocks",
            lambda: self._session_blocks_(viewer, viewer.row),
            lambda p: self._open_entrypoint_(viewer.row, p, viewer.title_color),
            empty="none indexed: the session was no dbx.exec command's",
            visible=lambda: bool(viewer.row) and not _is_blank_(viewer.row.get("session")),
            **self._copies_anchorkey_(viewer.row),
        )
        viewer.menus = [times, roots, session, self._output_button_(viewer)]
        self.viewers = [viewer]

    # Helpers -----------------------------------------------------------------

    @staticmethod
    def _title_(row, name=None):
        """A pipeline window's title: its tag's last component, its ``VAR.name`` where it has one, and its datetime."""
        parts = [tail(row.get("tag", row.get("anchor", "?"))), name, when(row.get("datetime", ""), seconds=True)]
        return " · ".join(p for p in parts if p)

    def _datetimes_(self, tag):
        """The journal's entries with *tag*, newest first, as dropdown items; None until read."""
        if self.journal is None:
            return None
        df = self.journal
        if "tag" not in df.columns:
            return []
        same = df[df["tag"].astype(str) == str(tag)]

        def text(r):
            name = r.get("name")
            named = f"{name}   " if isinstance(name, str) and name else ""
            return f"{when(r.get('datetime'), seconds=True)}   {named}{str(r.get('hash', ''))[:8]}   {r.get('event', '')}"

        return [(text(r), k) for k, r in same.iterrows()]

    def _is_entry_(self, k, row):
        """Whether the journal's entry *k* is the one *row* -- a stored, stringified row -- records."""
        e = self.journal.loc[k]
        return str(e.get("id")) == str(row.get("id")) and when(e.get("datetime"), True) == when(row.get("datetime"), True)

    def _show_entry_(self, viewer, k):
        """Show the journal's entry *k* in *viewer*, its side files read off the UI thread."""
        row = self.journal.loc[k]
        self.status = f"reading the entry of {when(row.get('datetime'), seconds=True)}"

        def shown(e):
            if viewer.menus[0].value != k:
                return  # stepped on past it meanwhile
            viewer.set_text(e.text)
            viewer.row = e.extra["row"]
            viewer.menus[0].label = self._title_(e.extra["row"], e.extra.get("name"))
            self.status = "ready"

        self.run_async(lambda: entry(row), shown)

    def _entrypoints_(self, row):
        """The title menu of an entry block's window: its pipeline's constructed entry blocks -- or, of an anchor's block, the anchor's blocks."""
        if self.anchor is not None:
            return self._anchor_blocks_(row)
        if self.constructed is None:
            return None
        lake = str(row.get("datalake", ""))
        paths = self.constructed.get(str(row.get("anchorkeypath")), [])
        return [(p[len(lake) :].lstrip("/") if lake and p.startswith(lake) else p, p) for p in paths]

    def _open_entrypoint_(self, row, entrypath, title_color=None):
        """Open a window over the pipeline's, its title a menu of the pipeline's constructed blocks -- in *title_color*, the pipeline's."""
        window = TextViewer(None, "")
        if title_color is not None:
            window.title_color = title_color
        window.row = None  # the newest entry of the block shown, once read
        window.menus = [
            Dropdown(
                self._relative_(row, entrypath),
                lambda: self._entrypoints_(row),
                lambda p: self._show_entrypoint_(window, row, p),
                empty="none constructed",
                value=entrypath,
                **self._copies_anchorkey_(row),
            ),
            Dropdown(
                "session blocks",
                lambda: self._session_blocks_(window, row),
                lambda p: self._show_session_block_(window, row, p),
                empty="none indexed: the session was no dbx.exec command's",
                visible=lambda: bool(window.row) and not _is_blank_(window.row.get("session")),
                **self._copies_anchorkey_(row),
            ),
            Dropdown(
                "read",
                lambda: self._topic_items_(window, stored_only=False),
                lambda t: self._open_topic_(window.row, t, "read"),
                empty="no topics",
                visible=lambda: bool(window.row) and not _is_blank_(window.row.get("topics")),
            ),
            Dropdown(
                "ls",
                lambda: self._topic_items_(window, stored_only=True),
                lambda t: self._open_topic_(window.row, t, "ls"),
                empty="no stored topics",
                visible=lambda: bool(window.row) and not _is_blank_(window.row.get("topics")),
            ),
            *self._slice_menus_(window),
            self._output_button_(window),
        ]
        self.push_viewer(window)
        self._show_entrypoint_(window, row, entrypath)

    def _show_entrypoint_(self, window, row, entrypath):
        """Show the journal of the block at *entrypath* in *window*, read off the UI thread."""
        menu = window.menus[0]
        menu.label = self._relative_(row, entrypath)
        window.row = None
        window.set_text(f"reading the journal of {menu.label} …")
        self._reset_slice_menus_(window)

        def shown(result):
            if menu.value == entrypath:  # not stepped on past it meanwhile
                text, window.row = result
                window.set_text(text)
                if declared_columns(window.row):
                    self._slice_info_(window)  # start planning a sample: tablets and workers wait on it

        self.run_async(lambda: entrypoint_text(entrypath, row, self.n_workers), shown)

    def _session_blocks_(self, window, row):
        """The blocks of the session of the entry *window* shows, as dropdown items; None while read, the first time asked."""
        shown = window.row
        session = str(shown.get("session"))
        if session not in self.sessions:
            self.sessions[session] = None

            def read():
                try:
                    return session_blocks(shown, self.n_workers)
                except Exception:
                    self.sessions[session] = []  # not read: shown as none, the error in the status line
                    raise

            self.run_async(read, lambda blocks: self.sessions.__setitem__(session, blocks))
        blocks = self.sessions[session]
        return None if blocks is None else [(f"{self._relative_(row, p)}   {event}", p) for p, event in blocks]

    def _show_session_block_(self, window, row, path):
        """Show the journal of the session's block at *path* in *window*, as picking a constructed entrypoint does."""
        window.menus[0].value = path
        self._show_entrypoint_(window, row, path)

    @staticmethod
    def _topic_items_(window, stored_only):
        """The topics of the block *window* shows, as dropdown items: only those with a location, for ls."""
        return [("/".join(t), t) for t, stored in topics(window.row) if stored or not stored_only]

    def _open_topic_(self, row, topic, how):
        """Open ``ls`` or ``read`` of *row*'s block's *topic* in a window over the one it was asked from, run off the UI thread."""
        name = "/".join(topic)
        if how == "ls":
            window = TextViewer(f"ls {name} · {self._relative_(row, str(row.get('anchorkeypath', '')))}", f"listing {name} …")
            fn = lambda: ls_text(row, topic)  # noqa: E731
        else:
            window = TextViewer(f"read {name} · {self._relative_(row, str(row.get('anchorkeypath', '')))}",
                                f"reading {name} through trueinst(): the first read of a block starts a Ray worker pinned to its revision, which takes a while …")
            fn = lambda: read_text(row, topic, self.remote_python)  # noqa: E731
        self.push_viewer(window)
        self.run_async(fn, window.set_text)

    # -- a block's slices, on a disk of their own -------------------------------

    MENU_LABELS = {"data": "data", "labels": "labels: none", "rows": "rows: auto", "tablets": "tablets: auto",
                   "workers": f"workers: {DEFAULT_WORKERS}"}

    def _slice_menus_(self, window):
        """A block window's ``data``, ``labels``, ``rows``, ``tablets`` and ``workers`` menus, and its ``hyp`` button: where the block records slices."""
        shown = lambda: bool(recorded_slices(window.row))  # noqa: E731
        declared = lambda: bool(declared_columns(window.row))  # noqa: E731
        empty = "no declared columns: its DATASLICEs declare none, so nothing of it can be viewed"
        # Only a plan already read: asking for one here would read it at every
        # draw of every block with slices. Until it is, they are shown dimmed;
        # once it is, only a table of more than one tablet has them.
        tabled = lambda: declared() and (self._known_plan_(window) or {"tablets": 2}).get("tablets", 1) > 1  # noqa: E731
        planned = lambda: self._known_plan_(window) is not None  # noqa: E731
        menu = lambda which, items, **kw: Dropdown(self.MENU_LABELS[which], items, lambda v: self._pick_slice_(window, which, v), **kw)  # noqa: E731
        window.slice_menus = {
            "data": menu("data", lambda: self._slice_items_(window), empty=empty, visible=shown),
            "labels": menu("labels", lambda: self._slice_items_(window, none=True), empty=empty, visible=shown),
            "rows": menu("rows", lambda: self._fraction_items_(window, "rows"), visible=declared),
            "tablets": menu("tablets", lambda: self._fraction_items_(window, "tablets"), visible=tabled, enabled=planned),
            "workers": menu("workers", [(str(n), n) for n in WORKERS], visible=tabled, enabled=planned),
        }
        self._reset_slice_menus_(window)
        button = Button("hyp", lambda: self._open_slice_disk_(window), visible=lambda: declared() and window.slice_pick.get("data") is not None)
        return [*window.slice_menus.values(), button]

    def _reset_slice_menus_(self, window):
        window.slice_pick = {"data": None, "labels": None, "rows": None, "tablets": None, "workers": DEFAULT_WORKERS}
        for name, m in getattr(window, "slice_menus", {}).items():
            m.label = self.MENU_LABELS[name]

    def _known_plan_(self, window):
        """The plan of the block *window* shows, if read -- reading none."""
        row = window.row or {}
        return self.slice_info.get((str(row.get("entry_path")), str(row.get("id"))))

    def _slice_info_(self, window):
        """What a sample of the block *window* shows is planned by -- `slice_plan` -- or None while read, the first time asked."""
        row = window.row
        key = (str(row.get("entry_path")), str(row.get("id")))
        if key not in self.slice_info:
            self.slice_info[key] = None

            def read():
                try:
                    return slice_plan(row, self.remote_python)
                except Exception:
                    self.slice_info[key] = {"tablets": 1, "rows0": {}}  # not read: planned without, the error in the status line
                    raise

            self.run_async(read, lambda info: self.slice_info.__setitem__(key, info))
        return self.slice_info[key]

    def _slice_items_(self, window, none=False):
        """The columns -- and dict columns' keys -- the block *window* shows declares, as dropdown items: from its journal row, at once."""
        items = [(f"{'/'.join((s, c, *path))}   {kind}", (s, c, path)) for s, c, path, kind in column_choices(window.row)]
        if items:
            self._slice_info_(window)  # start planning the sample: the tablets, and their rows
        return ([("none", None)] if none and items else []) + items

    def _plan_(self, window):
        """``(tablets, fraction, per tablet, expected rows)`` the picks come to.

        Auto: AUTO_TABLETS of the tablets, and AUTO_ROWS rows split among
        them -- a number per tablet, since tablets differ in size. *expected*
        is by the first tablet's rows, so a guess; None before they are read.
        """
        info = self._slice_info_(window) or {}
        n = info.get("tablets", 1)
        data = window.slice_pick.get("data")
        rows0 = info.get("rows0", {}).get(data[0]) if data else None
        tf = window.slice_pick.get("tablets")
        count = max(1, round(n * tf)) if tf is not None else min(n, AUTO_TABLETS)
        rf = window.slice_pick.get("rows")
        per = None if rf is not None else max(1, -(-AUTO_ROWS // count))
        expect = (round(rows0 * count * rf) if rf is not None else min(rows0, per) * count) if rows0 else None
        return count, rf if rf is not None else 1.0, per, expect

    def _fraction_items_(self, window, which):
        info = self._slice_info_(window) or {}
        n = info.get("tablets", 1)
        items = []
        for f in (ROW_FRACTIONS if which == "rows" else TABLET_FRACTIONS):
            name = "auto" if f is None else f"{f:.0%}" if f >= 0.01 else f"{f:.1%}"
            if which == "tablets":
                count = min(n, AUTO_TABLETS) if f is None else max(1, round(n * f))
                name += f"   {count:,} of {n:,}"
            items.append((name, f))
        return items

    def _pick_slice_(self, window, which, value):
        window.slice_pick[which] = value
        m = window.slice_menus[which]
        if which in ("data", "labels"):
            m.label = f"{which}: {'/'.join((value[0], value[1], *value[2]))}" if value is not None else self.MENU_LABELS[which]
        elif which == "workers":
            m.label = f"workers: {value}"
        else:
            m.label = f"{which}: auto" if value is None else f"{which}: {value:.0%}" if value >= 0.01 else f"{which}: {value:.1%}"

    def _open_slice_disk_(self, window):
        """Sample the slice columns picked in the pinned worker, lay the sample out with hyperbolic UMAP, and show it on a disk over *window*."""
        from . import layout

        row, pick = window.row, dict(window.slice_pick)
        data, labels, workers = pick["data"], pick["labels"], pick["workers"]
        info = self._slice_info_(window) or {}
        n = info.get("tablets", 1)
        count, fraction, per, expect = self._plan_(window)
        tablets = pick_tablets(n, count / n) if n > 1 else None
        named = lambda pick: "/".join((pick[0], pick[1], *pick[2]))  # noqa: E731
        title = f"{self._relative_(row, str(row.get('anchorkeypath', '')))} · {named(data)}" + (f" by {named(labels)}" if labels else "")
        each = f"{per:,} rows of each" if per else f"{fraction:.1%} of each"
        where = f"{count:,} of {n:,} tablets, {each}" if n > 1 else (f"{per:,} rows" if per else f"{fraction:.1%} of the rows")
        about = f"≈ {expect:,} rows -- " if expect else ""
        disk = DiskViewer(title, note=f"reading {about}{where} -- on {workers} workers …")
        self.push_viewer(disk)

        def work():
            try:
                X, L, rows, errors = slice_sample(row, data, labels, fraction, tablets=tablets, workers=workers,
                                                  python=self.remote_python, per_tablet=per)
                disk.set_note(f"laying out {len(X):,} points with hyperbolic UMAP …")
                positions = layout.compute(X, seed=0)
            except Exception as e:
                disk.set_note(f"could not: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
                raise
            tips = [" · ".join(p for p in ((L[i] if L else None), f"tablet {t:,} row {r:,}" if n > 1 else f"row {r:,}") if p)
                    for i, (t, r) in enumerate(rows)]
            if errors:
                self.status = f"{len(errors)} tablets could not be read: {errors[0]}"
            return positions, L, tips

        self.run_async(work, lambda result: disk.set_points(*result))

    def _output_button_(self, viewer):
        """The ``output`` button of a journal entry's window: shown while the entry it shows -- ``viewer.row`` -- had its output captured."""
        return Button("output", lambda: self._open_output_(viewer.row), visible=lambda: has_output(getattr(viewer, "row", None)))

    def _open_output_(self, row):
        """Open the captured output of the entry *row* in a window over the one it was asked from, read off the UI thread."""
        window = TextViewer(f"output · {when(row.get('datetime', ''), seconds=True)} · {str(row.get('log')).rsplit('/', 1)[-1]}", "reading the captured output …")
        self.push_viewer(window)
        self.run_async(lambda: output_text(row), window.set_text)

    def _anchor_blocks_(self, row):
        """The anchor's blocks -- the points -- as dropdown items, newest first."""
        if self.snap is None:
            return []
        lake = str(row.get("datalake", ""))
        out = []
        for p in self.snap.points:
            r = p.get("row", {})
            path = str(r.get("anchorkeypath", ""))
            if path:
                shown = path[len(lake) :].lstrip("/") if lake and path.startswith(lake) else path
                out.append((f"{shown}   {when(r.get('datetime', ''), seconds=True)}", path))
        return out

    def _block_context_(self, i, row, pos):
        """Right click on a block: the blocks journaled under its session, at the mouse; picking one opens its window."""
        from types import SimpleNamespace

        if _is_blank_(row.get("session")):
            self.status = "no session: the block was built outside dbx.exec"
            return
        held = SimpleNamespace(row=row)

        def opened(path):
            self.viewers = []
            self._open_entrypoint_(row, path, self.title_color(i))

        self.open_popup(lambda: self._session_blocks_(held, row), opened, pos, empty="none indexed: the session was no dbx.exec command's",
                        **self._copies_anchorkey_(row))

    def _copies_anchorkey_(self, row):
        """A block menu's Dropdown kwargs: right-click copies an item's anchorkey -- its path less *row*'s lake -- for a journal filter."""
        return dict(copy=lambda path: self._relative_(row, path), copy_hint="right-click: copy anchorkey")

    @staticmethod
    def _relative_(row, path):
        lake = str(row.get("datalake", ""))
        return path[len(lake) :].lstrip("/") if lake and path.startswith(lake) else path

