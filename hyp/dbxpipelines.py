"""Hyp over the pipelines of a dbx datalake.

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

import pprint
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .app import Dropdown, Hyp, TextViewer

# Columns shown first, in this order; the rest follow as the frame has them,
# then the side files, then TAIL_COLUMNS.
HEAD_COLUMNS = ("anchor", "tag", "key", "event", "datetime", "entrypaths")
TAIL_COLUMNS = ("gitrepo", "entry_path", "wrkrepo")
# Columns holding paths to side files: their contents are shown, in this
# order, in place of the paths -- cite first, right below the table.
SIDE_FILES = ("cite", "spec", "dfn", "kwargs", "signature", "type", "repr", "quote", "note")
# Side files that are yaml dumps of dicts: shown as the dicts they hold.
DICT_FILES = ("spec", "dfn", "kwargs")
# Path columns shown relative to the datalake, which is shown once.
LAKE_PATHS = ("anchorkeypath", "entrypaths")
# Columns holding a dict's repr: shown one key per line.
DICT_COLUMNS = ("topics", "paths")
SIDE_FILE_LIMIT = 16_000
EMBED_LIMIT = 4_000


@dataclass(frozen=True)
class Entry:
    key: str  # the pipeline's anchorkeypath: what a re-sync matches by
    text: str  # every column and side file, shown on right click
    embed: str  # anchor, spec, and signature or type
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


def _side_file_(column, path, url):
    """The side file *path* names, as text -- a dict pretty-printed, for DICT_FILES."""
    import yaml
    from dbx.databits import read_str, read_yaml

    if not isinstance(path, str) or not re.search(r"\.(txt|yaml|yml|json)$", path):
        return None
    path = rebase(path, url)
    try:
        text = None
        if column in DICT_FILES and path.endswith((".yaml", ".yml")):
            try:
                # dbx's own reader, as it reads them back: the dumps hold
                # python tuples, which only the unsafe loader makes.
                text = pprint.pformat(read_yaml(path, safe=False), width=100, sort_dicts=False)
            except yaml.YAMLError:
                pass  # shown as the yaml it is
        if text is None:
            text = read_str(path)
    except (FileNotFoundError, OSError):
        return None
    return text if len(text) <= SIDE_FILE_LIMIT else text[:SIDE_FILE_LIMIT] + "\n…"


def var_name(row):
    """The pipeline's ``VAR.name``, from the spec side file *row* names; None where it has none, or it cannot be read."""
    import yaml
    from dbx.databits import read_yaml

    path = row.get("spec")
    if not isinstance(path, str) or not path.endswith((".yaml", ".yml")):
        return None
    try:
        spec = read_yaml(rebase(path, row.get("entry_path", "")), safe=False)
    except (FileNotFoundError, OSError, yaml.YAMLError):
        return None
    name = spec.get("name") if isinstance(spec, dict) else None
    return None if name is None or str(name) == "" else str(name)


def entry(row):
    """One `dbx.pipejournal()` row, or any journal row, with its side files read, as an Entry."""
    row = {k: v for k, v in dict(row).items() if not _is_blank_(v)}
    url = row.get("entry_path", "")
    sides = {c: t for c in SIDE_FILES if c in row and (t := _side_file_(c, row[c], url))}
    lake = str(row.get("datalake", ""))

    def shown(c):
        v = row[c]
        if c in LAKE_PATHS:
            paths = [v] if isinstance(v, str) else list(v)
            return [p[len(lake) :].lstrip("/") if lake and str(p).startswith(lake) else str(p) for p in paths]
        if c in DICT_COLUMNS:
            return dict_lines(str(v))
        return [str(v)]

    cols = [c for c in HEAD_COLUMNS if c in row] + [c for c in row if c not in HEAD_COLUMNS + TAIL_COLUMNS]
    cols = [c for c in cols if c not in sides]
    tails = [c for c in TAIL_COLUMNS if c in row]
    width = max((len(c) for c in cols + tails), default=0)

    def table(columns):
        return [f"{(c if i == 0 else '').ljust(width)}  {v}" for c in columns for i, v in enumerate(shown(c))]

    lines = table(cols)
    for c, t in sides.items():
        lines += ["", f"── {c} ──", t.rstrip()]
    if tails:
        lines += [""] + table(tails)

    embed = [str(row.get("anchor", ""))]
    embed += [sides[c] for c in ("spec",) if c in sides]
    embed += [sides[c] for c in ("signature", "type") if c in sides][:1]
    key = str(row.get("anchorkeypath") or f"{row.get('anchor')}/{row.get('hash')}")
    label = tail(row.get("tag", row.get("anchor", "?")))
    return Entry(
        key=key,
        text="\n".join(lines),
        embed="\n".join(embed)[:EMBED_LIMIT],
        summary=f"{label} · {when(row['datetime'])}" if "datetime" in row else label,
        extra={"anchor": str(row.get("anchor", "")), "label": label, "row": {k: str(v) for k, v in row.items()}},
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
        raise SystemExit("hyp --pipejournal: no datalake; set DBX_DATALAKE")
    return lake


def pipelines(progress=print, n_workers=16):
    """Every pipeline in dbx's datalake, from `dbx.pipejournal()`, newest first.

    *n_workers* threads list and read the journal -- `dbx.pipejournal()`
    parallelizes with threads only, which suits it: the work is round trips
    to the datalake -- and then read the side files.
    """
    progress("dbx.pipejournal()")
    df = _dbx_().pipejournal(n_workers=n_workers)
    progress(f"reading the side files of {len(df)} pipelines")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(lambda i: entry(df.iloc[i]), range(len(df))))


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


def constructed_entrypaths(n_workers=16):
    """``{anchorkeypath: [entrypath]}``: each pipeline's entrypoints that were built or redirected."""
    df = _dbx_().pipejournal(construct_only=True, n_workers=n_workers)
    if not len(df):
        return {}
    return {str(r["anchorkeypath"]): [str(p) for p in r["entrypaths"]] for _, r in df.iterrows()}


def entrypoint_text(entrypath, row, n_workers=16):
    """The journal of the block at *entrypath*: its events, then its newest entry as `entry` shows one.

    *row* is a pipeline's journal row: its ``datalake`` and ``entry_path`` say
    where *entrypath*, recorded as the host wrote it, is read from.
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
        return f"No journal entries for {entrypath}"
    events = [f"{when(r.get('datetime'), seconds=True)}  {r.get('event', '')}" for _, r in df.iterrows()]
    return "\n".join(["── events ──", *events, "", entry(df.iloc[0]).text])


class HypDbxPipelines(Hyp):
    """One point per pipeline of a dbx datalake; right click shows its journal row and side files."""

    title = "hyp: dbx pipelines"

    @classmethod
    def __default_data__(cls, args):
        return f"{datalake().rstrip('/')}/.hyp"

    @classmethod
    def __storage_options__(cls, args):
        return _dbx_().default_storage_options()

    @classmethod
    def __source__(cls, args):
        return lambda progress: pipelines(progress, n_workers=args.n_workers)

    def __init__(self, engine, worker, args=None):
        super().__init__(engine, worker, args)
        self.n_workers = getattr(args, "n_workers", 16)
        self.journal = None  # every pipeline journal entry, newest first; None until read
        self.constructed = None  # {anchorkeypath: [entrypath]}; None until read
        self.run_async(lambda: journal_entries(self.n_workers), lambda df: setattr(self, "journal", df))
        self.run_async(lambda: constructed_entrypaths(self.n_workers), lambda d: setattr(self, "constructed", d))

    def __point_label__(self, i):
        return self.snap.points[i].get("label") or self.snap.summaries[i]

    def __point_datetime__(self, i):
        dt = self.snap.points[i].get("row", {}).get("datetime")
        return when(dt) if dt else None

    def __point_tooltip__(self, i):
        dt = self.__point_datetime__(i) if self.settings["datetimes"] else None
        label = self.__point_label__(i)
        return f"{label} · {dt}" if dt else label

    def __open_point__(self, i):
        row = self.snap.points[i].get("row", {})
        viewer = TextViewer(self.snap.summaries[i], self.snap.texts[i])
        viewer.row = row
        times = Dropdown(
            self._title_(row),
            lambda: self._datetimes_(row.get("tag")),
            lambda k: self._show_entry_(viewer, k),
            empty="no entries",
            match=lambda k: self._is_entry_(k, viewer.row),
        )
        roots = Dropdown("constructed entrypoints", lambda: self._entrypoints_(viewer.row), lambda p: self._open_entrypoint_(viewer.row, p), empty="none constructed")
        viewer.menus = [times, roots]
        self.viewers = [viewer]

    # Helpers -----------------------------------------------------------------

    @staticmethod
    def _title_(row):
        return f"{tail(row.get('tag', row.get('anchor', '?')))} · {when(row.get('datetime', ''), seconds=True)}"

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
            viewer.menus[0].label = self._title_(e.extra["row"])
            self.status = "ready"

        self.run_async(lambda: entry(row), shown)

    def _entrypoints_(self, row):
        if self.constructed is None:
            return None
        lake = str(row.get("datalake", ""))
        paths = self.constructed.get(str(row.get("anchorkeypath")), [])
        return [(p[len(lake) :].lstrip("/") if lake and p.startswith(lake) else p, p) for p in paths]

    def _open_entrypoint_(self, row, entrypath):
        """Open a window over the pipeline's, its title a menu of the pipeline's constructed blocks."""
        window = TextViewer(None, "")
        window.menus = [
            Dropdown(
                self._relative_(row, entrypath),
                lambda: self._entrypoints_(row),
                lambda p: self._show_entrypoint_(window, row, p),
                empty="none constructed",
                value=entrypath,
            )
        ]
        self.push_viewer(window)
        self._show_entrypoint_(window, row, entrypath)

    def _show_entrypoint_(self, window, row, entrypath):
        """Show the journal of the block at *entrypath* in *window*, read off the UI thread."""
        menu = window.menus[0]
        menu.label = self._relative_(row, entrypath)
        window.set_text(f"reading the journal of {menu.label} …")

        def shown(text):
            if menu.value == entrypath:  # not stepped on past it meanwhile
                window.set_text(text)

        self.run_async(lambda: entrypoint_text(entrypath, row, self.n_workers), shown)

    @staticmethod
    def _relative_(row, path):
        lake = str(row.get("datalake", ""))
        return path[len(lake) :].lstrip("/") if lake and path.startswith(lake) else path
