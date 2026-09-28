"""Journals: what was built (Datajournal) and what was run (Execjournal).

- :class:`Datajournal` -- the handle a block writes its build records through,
  and reads them back with; :class:`DatajournalFrame` / :class:`DatajournalEntry`
  are those records as a frame and as a row, and :class:`Block` is an entry's
  view of the block it recorded. :func:`datajournal` / :func:`journal` read one.
- :func:`execjournal` / :class:`ExecjournalFrame` / :class:`ExecjournalEntry` --
  the record of each ``dbx.exec()`` run, written by :func:`write_exec_journal`.
- :func:`filter_journal_frame` -- the column filters (regex, glob, dates) both
  journals are queried with.

Imports ``datablocks`` as a module and only uses it at call time: ``datablocks``
imports from here at the top, so the names in it are not there yet when this
module is executed.
"""
import ast
import copy
import datetime
import fnmatch
import functools
import hashlib
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import tqdm

from . import dataparts
from .dataparts import (
    Logger,
    Remote,
    default_datalake,
    default_storage_options,
    eval,
    exec,
    exec_comment,
    fs_full_path,
    gitwrkreposetup,
    list_path,
    ls_path,
    read_str,
    read_yaml,
    remote,
    size,
    write_str,
    write_yaml,
)
from . import datablocks


__eval__ = __builtins__['eval']


#: How dbx writes a timestamp: ``isoformat()`` with ``' '`` and ``':'`` replaced
#: by ``'-'``, so that a timestamp can also be a path component. pandas cannot
#: infer it -- dateutil reads the ``-`` between the hour and the minute as a
#: date separator and raises -- so every parse of a journal ``datetime`` has to
#: name it. That is what `_journal_datetimes_` is for.
JOURNAL_DATETIME_FORMAT = '%Y-%m-%dT%H-%M-%S.%f'


def write_exec_journal(s: str, datalake: str | None = None, storage_options: dict | None = None, *,
                       comment: str | None = None, session: str | None = None,
                       datajournal_entries=None, dt: str | None = None,
                       end_dt: str | None = None, url: str | None = None) -> dict:
    """Record an exec expression string in the <datalake>/.journal/exec/ journal.

    ``exec`` holds *s* VERBATIM -- the string as it was typed, comment and all,
    so that a journal row can be re-run as it stands. ``comment`` holds the
    trailing ``#`` comment on its own, because that is the half that says what
    the command was *for*, and reading it out of the expression again at every
    query is work the journal can do once. Pass *comment* to override what
    :func:`exec_comment` reads off *s*.

    ``session`` is the `Datajournal` session the command ran under, and
    ``datajournal_entries`` the block journal entries it wrote -- the keys from
    this row to what the command did. *dt* is when the command started --
    ``datetime`` and ``exec:start:datetime`` -- and *end_dt* when it finished,
    ``exec:end:datetime``: `exec` records the row once it is over. Returns the
    row as written.
    """
    dbx_url = datalake or url or default_datalake() or './dbx'
    exec_dir = os.path.join(dbx_url, '.journal', 'exec')
    fs, _ = fsspec.url_to_fs(exec_dir, **(storage_options or {}))
    try:
        fs.makedirs(exec_dir, exist_ok=True)
    except Exception:
        pass
    id = str(uuid.uuid4())
    file_path = os.path.join(exec_dir, f"exec_{id}.parquet")
    dt = dt or datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
    entry_data = {
        'exec': str(s),
        'datetime': dt,
        'exec:start:datetime': dt,
        'exec:end:datetime': end_dt,
        'id': id,
        'session': session,
        'datajournal_entries': list(datajournal_entries or []),
        'comment': comment if comment is not None else exec_comment(s),
    }
    df = pd.DataFrame([entry_data])
    with fs.open(file_path, 'wb') as f:
        df.to_parquet(f)
    return entry_data


def _is_glob_(p: str) -> bool:
    """A filter value is a shell-style glob when it has a wildcard and nothing only a regex would use.

    ``'*a6*'`` (a6 anywhere) and ``'a6*'`` (a6 at the start) are globs;
    ``'^a6'`` and ``'a6.*'`` are regexes. Deciding it is what keeps ``'a6*'``
    meaning what it looks like: read as a regex as well, it would also match
    any 'a' at all.
    """
    return any(ch in p for ch in '*?') and not any(ch in p for ch in '^$\\.+()|{}')


def _match_single_journal_val_(x, p) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    if isinstance(p, str) and _is_glob_(p):
        # Anchored, as a glob is: the whole value, not a part of it.
        return fnmatch.fnmatchcase(str(x), p)
    if isinstance(p, str):
        x_str = str(x)
        if p in x_str:
            return True
        if '=' in p:
            k_sub, v_sub = p.split('=', 1)
            k_sub, v_sub = k_sub.strip(), v_sub.strip().strip("'\"")
            pattern_re = rf"['\"]?{re.escape(k_sub)}['\"]?\s*[:=]\s*['\"]?[^,;)\n]*?{re.escape(v_sub)}"
            if re.search(pattern_re, x_str):
                return True
        elif ':' in p:
            k_sub, v_sub = p.split(':', 1)
            k_sub, v_sub = k_sub.strip(), v_sub.strip().strip("'\"")
            pattern_re = rf"['\"]?{re.escape(k_sub)}['\"]?\s*[:=]\s*['\"]?[^,;)\n]*?{re.escape(v_sub)}"
            if re.search(pattern_re, x_str):
                return True
        try:
            if re.search(p, x_str):
                return True
        except re.error:
            pass
        return False
    elif isinstance(p, re.Pattern):
        return bool(p.search(str(x)))
    elif callable(p):
        return bool(p(x))
    else:
        if x == p:
            return True
        return str(p) in str(x)


def _match_journal_filter_(x, spec) -> bool:
    if isinstance(spec, tuple):
        # ANDed
        return all(_match_single_journal_val_(x, p) for p in spec)
    elif isinstance(spec, list):
        # ORed
        return any(_match_journal_filter_(x, item) for item in spec)
    else:
        return _match_single_journal_val_(x, spec)


def _journal_datetimes_(series: pd.Series) -> pd.Series:
    """Coerce a journal ``datetime`` column to real datetimes.

    A block journal arrives already parsed -- `DatajournalFrame` does it on the way
    in -- and is handed back untouched. The exec journal does not: it reaches a
    filter holding the raw `JOURNAL_DATETIME_FORMAT` strings, which pandas
    cannot parse unaided.

    Anything the exact format misses is parsed again rather than left as NaT,
    because a filter that quietly drops the rows it could not read is worse
    than one that reads them: ``isoformat()`` omits ``.%f`` on a whole second,
    and a frame may carry a timestamp that came from somewhere other than dbx.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')
    todo = series.notna()
    for fmt in (JOURNAL_DATETIME_FORMAT, JOURNAL_DATETIME_FORMAT.removesuffix('.%f'), None):
        if not todo.any():
            break
        parsed[todo] = pd.to_datetime(series[todo], format=fmt, errors='coerce')
        todo &= parsed.isna()
    return parsed


def _journal_datetime_value_(v):
    """Parse one datetime a caller filtered by, dbx's own format first."""
    if isinstance(v, str):
        try:
            return datetime.datetime.strptime(v, JOURNAL_DATETIME_FORMAT)
        except ValueError:
            return pd.to_datetime(v)
    return v


def _journal_date_value_(v) -> datetime.date:
    """Parse one date a caller filtered by. A datetime is truncated to its date."""
    if isinstance(v, datetime.datetime):
        return v.date()
    if isinstance(v, datetime.date):
        return v
    return pd.Timestamp(_journal_datetime_value_(v)).date()


def _journal_date_mask_(dt_series: pd.Series, v) -> pd.Series:
    """Rows of *dt_series* falling on date *v*, or on any date in a list of them."""
    if isinstance(v, (list, tuple)):
        return dt_series.dt.date.isin([_journal_date_value_(x) for x in v])
    return dt_series.dt.date == _journal_date_value_(v)


def filter_journal_frame(df: pd.DataFrame, **filter_kwargs) -> pd.DataFrame:
    """Filter a journal DataFrame by column matching, substring/pattern matching anywhere in the string, and date/datetime."""
    if df is None or df.empty or not filter_kwargs:
        return df

    for k, v in filter_kwargs.items():
        if k not in df.columns:
            if k == 'entry_code' and 'id' in df.columns:
                k = 'id'
            elif k == 'subhash' and 'code' in df.columns:
                k = 'code'
            elif k == 'subsignature' and 'signature' in df.columns:
                k = 'signature'
            elif k == 'date' and 'datetime' in df.columns:
                df = df[_journal_date_mask_(_journal_datetimes_(df['datetime']), v)]
                continue
            else:
                return df.iloc[0:0].reset_index(drop=True)

        if k == 'date':
            df = df[_journal_date_mask_(_journal_datetimes_(df['datetime']), v)]
        elif k == 'datetime':
            dt_series = _journal_datetimes_(df[k])
            if isinstance(v, (list, tuple)):
                df = df[dt_series.isin([_journal_datetime_value_(x) for x in v])]
            else:
                df = df[dt_series == _journal_datetime_value_(v)]
        else:
            df = df[df[k].apply(lambda x: _match_journal_filter_(x, v))]

    df = df.reset_index(drop=True)
    return df


#: The events a block journal entry holds when the command that wrote it
#: CONSTRUCTED the block: built it, made it readable by redirection, or copied
#: its data in. Anchored, because a filter string is a pattern and a bare
#: 'UNSAFE_redirect' would also match a stack's 'UNSAFE_redirect_blocks:end'.
#: A block instance rewrites its one entry file, so a build that failed ends
#: at 'build:exception', and one that never finished at 'build:start': neither
#: is here.
CONSTRUCTED_EVENTS = ['^build:end$', '^UNSAFE_redirect$', '^UNSAFE_copy_from:END$']


def _constructed_(frame, anchor, filters):
    """The rows of a DatajournalFrame *filters* select: one anchor's, or ``{anchor: rows}``."""
    filters = dict(filters)
    filters.setdefault('event', CONSTRUCTED_EVENTS)
    if filters['event'] is None:
        del filters['event']
    storage_options = getattr(frame, 'storage_options', None)
    df = filter_journal_frame(pd.DataFrame(frame), **filters) if len(frame) else pd.DataFrame(frame)

    def of(a):
        rows = df[df['anchor'] == a] if 'anchor' in df.columns else df.iloc[0:0]
        return DatajournalFrame(rows.reset_index(drop=True), storage_options=storage_options)

    if anchor is not None:
        return of(anchor)
    present = df['anchor'].dropna().unique() if 'anchor' in df.columns else []
    return {a: of(a) for a in sorted(present)}


def _anchors_(frame) -> list:
    return sorted(frame['anchor'].dropna().unique()) if 'anchor' in getattr(frame, 'columns', ()) else []


def _shell_double_quoted_(text: str) -> str:
    """*text* escaped for the inside of a bash double-quoted string: \\, ", $ and `."""
    for ch in ('\\', '"', '$', '`'):
        text = text.replace(ch, '\\' + ch)
    return text


class ExecjournalEntry(pd.Series):
    """One `dbx.exec` command, as the exec journal recorded it.

    The counterpart of `DatajournalEntry` for the exec journal: ``exec``,
    ``exec``, ``datetime``, ``id``, ``session``, ``datajournal_entries`` and
    ``comment`` are its columns; :meth:`entries` and :meth:`datajournal`
    follow ``datajournal_entries`` to the block journal entries the command
    wrote.
    """
    #: Carried by pandas across operations that rebuild the object -- see
    #: `DatajournalEntry._metadata`.
    _metadata = ['storage_options']

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, series: pd.Series, *, storage_options: dict | None = None):
        super().__init__(series)
        self.storage_options = storage_options or {}

    # 2. Declared API ------------------------------------------------------

    def entries(self, *, n_workers: int | None = None) -> list:
        """The `DatajournalEntry` of every block journal entry this command wrote, in the order written."""
        return Datajournal(storage_options=self.storage_options or None).read_entries(
            ExecjournalEntry._written_paths_(self), n_workers=n_workers)

    def datajournal(self, *, n_workers: int | None = None):
        """The block journal entries this command wrote, as one `DatajournalFrame`.

        Its ``datajournal_entries``, read as a block journal is -- the same
        legacy columns resolved, filterable the same way -- one row per entry,
        in the order written.
        """
        return Datajournal(storage_options=self.storage_options or None).read_frame(
            ExecjournalEntry._written_paths_(self), n_workers=n_workers)

    def anchors(self) -> list:
        """Every anchor this command wrote a block journal entry for, sorted."""
        return _anchors_(self.datajournal())

    def constructed(self, anchor: str | None = None, **filters):
        """The block journal entries of what this command CONSTRUCTED.

        *anchor* given: that anchor's entries, as a `DatajournalFrame`. Not
        given: ``{anchor: DatajournalFrame}`` for every anchor with any.

        *filters* are `DatajournalFrame` filters. ``event`` defaults to
        `CONSTRUCTED_EVENTS` -- a block built, redirected, or copied in -- and
        applies alongside any other filter unless given itself;
        ``event=None`` drops it. A block the command found already built was
        not constructed by it, and wrote no entry: it is not here.
        """
        return _constructed_(self.datajournal(), anchor, filters)

    def rerun(self, **kwargs):
        """Execute this command's ``exec`` string again, through `dbx.exec`, and return its value.

        A new command, recorded as one: its own exec-journal row, session and
        ``written_entries``. It runs against the code as it is NOW -- nothing
        here checks out the revision the original ran under. *kwargs* bind
        names for the statements, as `dbx.exec`'s own do.

        Prints the command first, as the shell line that would run it --
        ``dbx.pprint "..."`` -- so what is being re-run is on screen, and can be
        pasted.
        """
        print(f'dbx.pprint "{_shell_double_quoted_(self["exec"])}"', flush=True)
        return exec(self['exec'], **kwargs)

    # 4. Helpers -----------------------------------------------------------

    @staticmethod
    def _written_paths_(row) -> list:
        """The ``datajournal_entries`` of *row* as a list of paths; empty for a row from before the column."""
        paths = row.get('datajournal_entries')
        if paths is None or (isinstance(paths, float) and pd.isna(paths)):
            return []
        return [str(p) for p in paths]


class ExecjournalFrame(pd.DataFrame):
    """The exec journal: one `dbx.exec` command per row, newest first.

    The counterpart of `DatajournalFrame`. :meth:`get` answers with an
    `ExecjournalEntry`, and :meth:`entries` with the block journal entries
    of every command in the frame -- so a filter narrows to the commands and
    ``.entries()`` goes on to what they wrote::

        dbx.journal(comment='nightly').entries()
    """
    _metadata = ['storage_options']

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, df: pd.DataFrame | None = None, *, storage_options: dict | None = None):
        super().__init__(pd.DataFrame() if df is None else df)
        self.storage_options = storage_options or {}

    def __call__(self, entry):
        return self.get(entry, dropna=True)

    # 2. Declared API ------------------------------------------------------

    def get(self, entry, *, dropna: bool = False) -> ExecjournalEntry:
        """The command at LABEL *entry* (``.loc``). As `DatajournalFrame.get`."""
        row = self.loc[entry]
        if dropna:
            row = row.dropna()
        return ExecjournalEntry(row, storage_options=self.storage_options)

    def entries(self, *, n_workers: int | None = None) -> list:
        """The block journal entries every command here wrote: row by row, each in the order written."""
        return Datajournal(storage_options=self.storage_options or None).read_entries(
            self._written_paths_(), n_workers=n_workers)

    def datajournal(self, *, n_workers: int | None = None):
        """What :meth:`entries` reads, as one `DatajournalFrame`."""
        return Datajournal(storage_options=self.storage_options or None).read_frame(
            self._written_paths_(), n_workers=n_workers)

    def anchors(self) -> list:
        """Every anchor the commands here wrote a block journal entry for, sorted."""
        return _anchors_(self.datajournal())

    def constructed(self, anchor: str | None = None, **filters):
        """As `ExecjournalEntry.constructed`, over every command here."""
        return _constructed_(self.datajournal(), anchor, filters)

    # 4. Helpers -----------------------------------------------------------

    def _written_paths_(self) -> list:
        return [p for _, row in self.iterrows() for p in ExecjournalEntry._written_paths_(row)]


#: The exec journal's columns in the order a frame shows them: the command first
#: and what it was FOR last, with what it did between.
EXEC_JOURNAL_COLUMNS = ['exec', 'datetime', 'exec:start:datetime', 'exec:end:datetime',
                        'id', 'session', 'datajournal_entries', 'comment']


def _exec_journal_columns_(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy column names resolved, and ``exec`` first, ``comment`` last."""
    if 'written_entries' in df.columns:
        # Its name for a day, before `datajournal_entries`: taken per row, so
        # a journal holding rows of both kinds loses neither.
        if 'datajournal_entries' in df.columns:
            df['datajournal_entries'] = df['datajournal_entries'].combine_first(df['written_entries'])
        else:
            df = df.rename(columns={'written_entries': 'datajournal_entries'})
        df = df.drop(columns=['written_entries'], errors='ignore')
    middle = [c for c in df.columns if c not in ('exec', 'comment')]
    return df[[c for c in ('exec',) if c in df.columns] + middle
              + [c for c in ('comment',) if c in df.columns]]


def read_exec_journal(
    datalake: str | None = None,
    loc: int | None = None,
    *,
    iloc: int | None = None,
    filter: dict | None = None,
    storage_options: dict | None = None,
    log: Logger | None = None,
    n_workers: int = 8,
    index: str | None = None,
    url: str | None = None,
    **filter_kwargs,
):
    """Read recorded dbx.exec() entries from the <datalake>/.journal/exec/ journal.

    Returns an `ExecjournalFrame`, or the one `ExecjournalEntry` at *loc* or *iloc*.
    """
    if loc is not None and iloc is not None:
        raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
    if n_workers is None:
        n_workers = 8

    dbx_url = datalake or url or default_datalake() or './dbx'
    exec_dir = os.path.join(dbx_url, '.journal', 'exec')
    fs, _ = fsspec.url_to_fs(exec_dir, **(storage_options or {}))
    try:
        if not fs.exists(exec_dir):
            files = []
        else:
            files = fs.glob(os.path.join(exec_dir, '*.parquet'))
    except Exception:
        files = []

    if not files:
        df = pd.DataFrame(columns=EXEC_JOURNAL_COLUMNS)
    else:
        def read_file(file):
            with fs.open(file, 'rb') as f:
                return pd.read_parquet(f)

        dfs = []
        with ThreadPoolExecutor(max_workers=min(n_workers, max(1, len(files)))) as ex:
            futures = [ex.submit(read_file, file) for file in files]
            for future in as_completed(futures):
                try:
                    dfs.append(future.result())
                except Exception as e:
                    if log:
                        log.warning(f"Skipping unreadable exec journal file: {e}")
                    continue
        if not dfs:
            df = pd.DataFrame(columns=EXEC_JOURNAL_COLUMNS)
        else:
            df = _exec_journal_columns_(pd.concat(dfs, ignore_index=True))
            if 'datetime' in df.columns:
                df = df.sort_values('datetime', ascending=False).reset_index(drop=True)

    all_filters = dict(filter or {})
    all_filters.update(filter_kwargs)

    df = filter_journal_frame(df, **all_filters)

    if index is ...:
        # dbx.journal()'s default: by id, where there is one to index by.
        index = 'id' if 'id' in df.columns else None
    if index is not None:
        if index in df.columns:
            df = df.set_index(index, drop=False)
        else:
            raise KeyError(f"Column {index!r} not found in journal DataFrame")

    frame = ExecjournalFrame(df, storage_options=storage_options)
    if loc is not None:
        return frame.get(loc, dropna=True)
    elif iloc is not None:
        return ExecjournalEntry(frame.iloc[iloc].dropna(), storage_options=storage_options)
    return frame


def execjournal(loc=None, *, iloc=None, datalake=None, storage_options=None, log=None,
                n_workers=8, index: 'str | None' = ..., url=None, **filter_kwargs):
    """Read the exec journal: every `dbx.exec` command, newest first, then filtered.

    An `ExecjournalFrame`, or the `ExecjournalEntry` at *loc* / *iloc*. *index*
    defaults to ``'id'``, so ``loc=`` is a command's id; ``index=None`` numbers
    the rows. Filters are patterns, as on a block journal -- see
    :func:`datajournal`. *datalake* (``url``, its old name) defaults to
    ``DBX_DATALAKE``, then ``DBX_ROOT``, then ``DBX_URL``.
    """
    return read_exec_journal(datalake=datalake or url, loc=loc, iloc=iloc, storage_options=storage_options,
                             log=log, n_workers=n_workers, index=index, **filter_kwargs)


class CallableStr(str):
    def __call__(self, *args, **kwargs):
        return str(self)


def normalize_journal_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical ``type`` and ``signature`` columns, decided per ROW.

    The two names have meant different things at different times::

        <= 2026-08-03          type is 'hashstr',   signature is 'norm'
        2026-08-03 .. 08-27    type is 'signature', signature is 'subsignature'/'norm'
        >= 2026-08-27          type is 'type',      signature is 'signature'

    So a column literally named ``signature`` holds today's TYPE in the middle
    era. A journal frame concatenates rows from every era and a column is
    frame-wide, so no rename can sort that out: ``df['signature']`` would still
    mean one thing on some rows and another on the rest. Deciding it per row
    can, and the discriminator is in the data -- a row carrying a ``type``
    value is a modern one.

    `DatajournalEntry.COLUMN_CHAINS` resolves the same thing lazily, per
    access, which is why the ACCESSORS are right either way. This is for the
    frame itself, so that filtering on ``type`` or reading ``df['signature']``
    means what it says -- most of the point of a journal being a DataFrame.
    """
    if hasattr(df, 'journal'):
        df = df.journal
    if df is None or getattr(df, 'empty', True):
        return df

    def col(name):
        return df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)

    modern = col('type').notna()
    middle = ~modern & col('signature').notna()

    # Vectorised, not row-wise: a journal runs to thousands of rows and this
    # happens on every read.
    type_col = col('type').where(modern, col('signature').where(middle, col('hashstr')))
    sig_col = col('signature').where(
        modern, col('subsignature').fillna(col('norm')).where(middle, col('norm')))

    df = df.copy()
    df['type'] = type_col
    df['signature'] = sig_col
    return df


def datajournal(cls_anchor_or_df, loc=None, *, iloc=None, datalake=None, storage_options=None, log=None, n_workers=8, index: 'str | None' = ..., unnormalized: bool = False, url=None, **filter_kwargs):
    """Read a block journal, or wrap a frame of one.

    Parameters
    ----------
    cls_anchor_or_df : type | str | Datablock | pd.DataFrame
        A Datablock class, an anchor string, a block (whose url, storage
        options and log are then the defaults), or a raw DataFrame to wrap.
    loc : int, optional
        If given, return a single :class:`DatajournalEntry` at this label index.
    iloc : int, optional
        If given, return a single :class:`DatajournalEntry` at this positional index.
        Mutually exclusive with *loc*.
    url : str, optional
        The datalake (``url``, its old name, is accepted). Defaults to
        ``DBX_DATALAKE``, then ``DBX_ROOT``, then ``DBX_URL``.
    storage_options : dict, optional
        Storage options for fsspec.  Defaults to ``default_storage_options()``.
    log : Logger, optional
        Logger instance.
    n_workers : int, default 8
        Number of workers for reading journal files.
    index : str or None, default: ``'id'`` where the journal has one
        Column to index the returned frame by -- so, by default, ``loc=`` is an
        entry's ``id`` and ``iloc=`` its position, newest first. A journal
        with no ``id`` column (written before ids existed) is left numbered,
        rather than refused. ``index=None`` always leaves it numbered 0..N-1,
        and then ``loc=`` is a position too. An explicit column must exist.
        The index column stays a column: an entry keeps its own ``id``.
    unnormalized : bool, default False
        Leave the era-dependent ``type``/``signature`` columns exactly as
        recorded. By default they are resolved per row, so the frame means
        what it says -- see `normalize_journal_frame`.
    **filter_kwargs
        Forwarded to :class:`DatajournalFrame` for filtering.

    Returns
    -------
    DatajournalFrame, or the DatajournalEntry at *loc* / *iloc*. The exec
    journal is :func:`execjournal`.
    """
    if cls_anchor_or_df is None:
        raise TypeError("datajournal() needs an anchor, a Datablock class or block, or a frame; "
                        "the exec journal is execjournal()")
    url = one_datalake(datalake, url, 'datajournal')
    if loc is not None and iloc is not None:
        raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
    if isinstance(cls_anchor_or_df, pd.DataFrame):
        return DatajournalFrame(cls_anchor_or_df, storage_options=storage_options, index=index, unnormalized=unnormalized, **filter_kwargs)
    else:
        if isinstance(cls_anchor_or_df, str):
            anchor = cls_anchor_or_df
        elif isinstance(cls_anchor_or_df, type) and issubclass(cls_anchor_or_df, datablocks.Datablock):
            anchor = cls_anchor_or_df.anchor
        elif isinstance(cls_anchor_or_df, type):
            anchor = cls_anchor_or_df.__module__ + "." + cls_anchor_or_df.__name__
        elif hasattr(cls_anchor_or_df, 'anchor'):
            anchor = cls_anchor_or_df.anchor
            if url is None and hasattr(cls_anchor_or_df, 'datalake'):
                url = cls_anchor_or_df.datalake
            if storage_options is None and hasattr(cls_anchor_or_df, 'storage_options'):
                storage_options = cls_anchor_or_df.storage_options
            if log is None and hasattr(cls_anchor_or_df, 'log'):
                log = cls_anchor_or_df.log
        else:
            anchor = cls_anchor_or_df.__module__ + "." + cls_anchor_or_df.__name__
        return datablocks.Datablock.Journal(anchor, loc=loc, iloc=iloc, datalake=url, storage_options=storage_options, log=log, n_workers=n_workers, index=index, unnormalized=unnormalized, **filter_kwargs)


def journal(cls_anchor_or_df=None, loc=None, **kwargs):
    """:func:`datajournal` for an anchor, class, block or frame; :func:`execjournal` with none.

    The one entry point both journals used to share, kept so code written
    against it goes on working. Say which journal you mean instead.
    """
    if cls_anchor_or_df is None:
        kwargs.pop('unnormalized', None)
        return execjournal(loc, **kwargs)
    return datajournal(cls_anchor_or_df, loc, **kwargs)


def constructed(anchor=None, *, event=..., datalake=None, storage_options=None,
                n_workers=8, url=None, **filters) -> 'DatajournalFrame':
    """The block journal entries of what was CONSTRUCTED: built, redirected, or copied in.

    *anchor* -- a Datablock class, an anchor string or a block -- reads that
    anchor's journal; None reads every anchor in the datalake (`anchors`)
    into one frame, newest first. An anchor with no journal contributes
    nothing.

    *event* ``...``, the default, is `CONSTRUCTED_EVENTS`; None drops the
    event filter, and anything else is the filter, as `DatajournalFrame`
    takes one. *filters* are its other filters.
    """
    if event is ...:
        event = CONSTRUCTED_EVENTS
    if event is not None:
        filters['event'] = event
    datalake = one_datalake(datalake, url, 'constructed')
    if anchor is not None:
        return datajournal(anchor, datalake=datalake, storage_options=storage_options,
                           n_workers=n_workers, **filters)
    frames = []
    for a in dataparts.anchors(datalake, storage_options=storage_options):
        try:
            frames.append(datajournal(a, datalake=datalake, storage_options=storage_options,
                                      n_workers=n_workers, **filters))
        except FileNotFoundError:
            continue
    frames = [f for f in frames if len(f)]
    if not frames:
        return DatajournalFrame(pd.DataFrame(), storage_options=storage_options or {})
    frame = pd.concat(frames)
    if 'datetime' in frame.columns:
        frame = frame.sort_values('datetime', ascending=False, kind='stable')
    return DatajournalFrame(frame, storage_options=storage_options or {})


class CallableSignature(CallableStr):
    """A signature string already rendered and stored, e.g. on a journal row.

    ``legacy=`` selects which rendering to PRODUCE, so it has nothing to act
    on here -- the rendering happened before this string was stored. Passing
    it raises rather than being quietly ignored; ask the block itself
    (``block.signaturestr(legacy=...)``) for the other rendering.
    """

    def __call__(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = False, **kwargs):
        if legacy is not None:
            raise TypeError(
                f"{type(self).__name__}: legacy= chooses how a signature is "
                f"rendered, but this one is already rendered and stored. Call "
                f"signaturestr(legacy={legacy!r}) on the block instead."
            )
        s = str(self)
        # Before parsing, not after: stripping backslashes from the formatted
        # output would eat repr's escapes inside the leaves.
        if deslash:
            s = s.replace('\\', '')
        if pretty:
            import pprint
            parsed = datablocks.Datablock._parse_signature_(s)
            sig_dict = {k: datablocks.Datablock._structure_from_signature_text_(v) for k, v in parsed.items()}
            return pprint.pformat(sig_dict, indent=2, width=120)
        return s


class CallableSig(CallableStr):
    """A signature string already rendered and stored, e.g. on a journal row.

    ``legacy=`` selects which rendering to PRODUCE, so it has nothing to act
    on here -- the rendering happened before this string was stored. Passing
    it raises rather than being quietly ignored; ask the block itself
    (``block.signaturestr(legacy=...)``) for the other rendering.
    """

    def __call__(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = True, **kwargs):
        if legacy is not None:
            raise TypeError(
                f"{type(self).__name__}: legacy= chooses how a signature is "
                f"rendered, but this one is already rendered and stored. Call "
                f"signaturestr(legacy={legacy!r}) on the block instead."
            )
        s = str(self)
        # Before parsing, not after: stripping backslashes from the formatted
        # output would eat repr's escapes inside the leaves.
        if deslash:
            s = s.replace('\\', '')
        if pretty:
            import pprint
            parsed = datablocks.Datablock._parse_signature_(s)
            sig_dict = {k: datablocks.Datablock._structure_from_signature_text_(v) for k, v in parsed.items()}
            return pprint.pformat(sig_dict, indent=2, width=120)
        return s


class CallableType(CallableStr):
    def __new__(cls, val='', block=None):
        instance = super().__new__(cls, val)
        instance._block = block
        return instance

    def __call__(self, *, deslash: bool = False, pretty: bool = False, **kwargs):
        if pretty:
            import pprint
            try:
                t_str = str(self)
                parts = t_str.split(os.sep) if os.sep in t_str else t_str.split('/')
                topics = []
                paths = None
                version = self._block.version if self._block is not None else None
                sig_part = str(self._block.signaturestr()) if self._block is not None else ''
                for p in parts:
                    if p.startswith('topic:'):
                        topics.append(p)
                    elif p.startswith('_paths_='):
                        paths = p[len('_paths_='):]
                    elif p.startswith('version='):
                        ver_str = p[len('version='):]
                        version = None if ver_str == 'None' else ver_str

                sig_dict = datablocks.Datablock._parse_signature_(sig_part)
                sig_dict = {k: datablocks.Datablock._structure_from_signature_text_(v) for k, v in sig_dict.items()}
                d = {
                    'paths': paths,
                    'signature': sig_dict,
                    'topics': tuple(topics),
                    'version': version,
                }
                return pprint.pformat(d, indent=2, width=120)
            except Exception:
                pass
        t = str(self)
        if deslash:
            t = t.replace('\\', '')
        return t


class CallableTp(CallableStr):
    def __new__(cls, val='', block=None):
        instance = super().__new__(cls, val)
        instance._block = block
        return instance

    def __call__(self, *, deslash: bool = False, pretty: bool = True, **kwargs):
        if pretty:
            import pprint
            try:
                t_str = str(self)
                parts = t_str.split(os.sep) if os.sep in t_str else t_str.split('/')
                topics = []
                paths = None
                version = self._block.version if self._block is not None else None
                sig_part = str(self._block.signaturestr()) if self._block is not None else ''
                for p in parts:
                    if p.startswith('topic:'):
                        topics.append(p)
                    elif p.startswith('_paths_='):
                        paths = p[len('_paths_='):]
                    elif p.startswith('version='):
                        ver_str = p[len('version='):]
                        version = None if ver_str == 'None' else ver_str

                sig_dict = datablocks.Datablock._parse_signature_(sig_part)
                sig_dict = {k: datablocks.Datablock._structure_from_signature_text_(v) for k, v in sig_dict.items()}
                d = {
                    'paths': paths,
                    'signature': sig_dict,
                    'topics': tuple(topics),
                    'version': version,
                }
                return pprint.pformat(d, indent=2, width=120)
            except Exception:
                pass
        t = str(self)
        if deslash:
            t = t.replace('\\', '')
        return t


class Block:
    """A `Datablock`-shaped view of one journal entry.

    Everything here is READ OFF THE ROW -- the block as it was when the entry
    was written -- and nothing is recomputed. That is the point: a live block
    recomputes its identity from today's rendering, which is how
    ``inst().paths()`` comes back naming a directory that was never written.
    This answers with what was actually built.

    It mimics the parts of `Datablock` the row DETERMINES, and stops there.
    Not the entry: `ls`, `list`, `size` and `read` are the entry's, they go to
    storage, and `Datablock.read` reads a TOPIC while `DatajournalEntry.read`
    reads a journal COLUMN -- one name with two meanings is worse than no
    name. Nor does it build or validate.

    The API shape is mirrored, not merely the names: what is a property on
    `Datablock` is a property here, what is a method there is a method here.
    So ``paths()``, ``signature()``, ``type()`` and their ``*str()`` renderings
    are calls and ``hash``, ``anchor``, ``key`` are not, and code written
    against a live block reads one of these unchanged. Accessors with no `Datablock` counterpart
    (``gitrepo``, ``url``, ``id``, ``keyby``) are here too, because they
    describe the block rather than the journal.
    """

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, entry: 'DatajournalEntry'):
        self._entry = entry

    def __repr__(self):
        return f"Block({self.anchor}/{self.hash})"

    # 2. Declared API ------------------------------------------------------

    def signaturestr(self, *, deslash: bool = False, **kwargs):
        """The recorded signature TEXT.

        A method, as on `Datablock` -- but with nothing left to render: the row
        holds one rendering, chosen when it was written. ``legacy=`` and its
        kin therefore raise rather than being ignored.
        """
        self._reject_rendering_choice_('signaturestr', kwargs)
        val = self._signature_text_(self._entry)
        return val.replace('\\', '') if (deslash and val) else val

    def sigstr(self, *, deslash: bool = False, pretty: bool = True, **kwargs):
        val = self._signature_text_(self._entry)
        return CallableSig(val)(deslash=deslash, pretty=pretty) if val else None

    def typestr(self, *, deslash: bool = False, **kwargs):
        """The recorded type TEXT."""
        self._reject_rendering_choice_('typestr', kwargs)
        val = self._type_text_(self._entry)
        return val.replace('\\', '') if (deslash and val) else val

    def tpstr(self, *, deslash: bool = False, pretty: bool = True, **kwargs):
        val = self._type_text_(self._entry)
        return CallableTp(val, block=self)(deslash=deslash, pretty=pretty) if val else None

    def signature(self, *, deslash: bool = False, **kwargs) -> dict:
        """As `Datablock.signature`, over the signature this row records."""
        self._reject_rendering_choice_('signature', kwargs)
        text = self._signature_text_(self._entry) or ''
        if deslash:
            text = text.replace('\\', '')
        parsed = datablocks.Datablock._parse_signature_(text)
        return {k: datablocks.Datablock._structure_from_signature_text_(v) for k, v in parsed.items()}

    def sig(self, *, deslash: bool = False, **kwargs) -> dict:
        return self.signature(deslash=deslash, **kwargs)

    def type(self, *, deslash: bool = False, **kwargs) -> dict:
        self._reject_rendering_choice_('type', kwargs)
        version, paths, topics, entries = self.version, None, [], {}
        for part in self._type_parts_(self._type_text_(self._entry) or ''):
            if part.startswith('TAB='):
                entries['TAB'] = part[len('TAB='):]      # see Datatable._type_entries_
            elif part.startswith('topic:'):
                topics.append(part)
            elif part.startswith('_paths_='):
                paths = part[len('_paths_='):]
            elif part.startswith('version='):
                version = self._as_version_(part[len('version='):])
        return {
            'paths': paths,
            'signature': self.signature(deslash=deslash),
            **entries,
            'topics': tuple(topics),
            'version': version,
        }

    def tp(self, *, deslash: bool = False, **kwargs) -> dict:
        return self.type(deslash=deslash, **kwargs)

    def paths(self) -> dict:
        """Recorded ``{topic: path}`` mapping.

        A method, as `Datablock.paths` is -- and the reason this class exists:
        the paths the build actually wrote, not paths derived from an identity
        recomputed under today's rendering.
        """
        return self._dict_column_(self._entry, 'paths')

    def topics(self) -> list:
        """The recorded topic names, as `Datablock.topics` answers.

        Names, not the ``{name: filename}`` mapping the column holds: this
        mirrors the live API, and the mapping is the entry's own business --
        read it off the row when you want it.
        """
        return list(self._dict_column_(self._entry, 'topics', parse=datablocks.literal_topics))

    def quote(self, *, deslash: bool = False):
        """The recorded `Datablock.quote` TEXT -- the evaluable specline -- or None.

        The TEXT, as on `Datablock`; the row's ``quote`` column holds the path
        to the file it was written to.
        """
        return self._recorded_text_('quote', deslash)

    def cite(self, *, deslash: bool = False):
        """The recorded `Datablock.cite` TEXT, or None. The path is the row's ``cite`` column."""
        return self._recorded_text_('cite', deslash)

    def repr(self, *, deslash: bool = False):
        """The recorded `Datablock.repr` TEXT -- every kwarg the block had -- or None.

        Rows written before ``repr()`` existed recorded ``__repr__()`` in this
        column instead, and that is what they answer with.
        """
        return self._recorded_text_('repr', deslash)

    def note(self):
        """Path to or content of this entry's note, or None."""
        return DatajournalEntry.column(self._entry, 'note')

    def is_topicgroup(self, *topicpath):
        """True when the recorded entry for *topicpath* is a group of topics."""
        return isinstance(self._walk_(self.TOPICS,
                                     self._normtopic_(topicpath)), dict)

    def ls(self, *topicpath, detail=False):
        """List what is at the recorded path for a topic.

        As `Datablock.ls`, but resolving the path from what the row RECORDED
        rather than from an identity recomputed today. A group concatenates
        its members' listings.
        """
        topicpath = self._normtopic_(topicpath)
        p = self._topic_path_(*topicpath)
        fs = self._fs_(self._entry)
        if isinstance(p, dict):
            return [e for leaf in self._leaf_paths_(p)
                    for e in ls_path(fs, leaf, False, detail=detail)]
        return ls_path(fs, p, self._is_dir_topic_(*topicpath), detail=detail)

    def list(self, *topicpath):
        """Detailed, recursive listing of every file under the topic path.

        As `Datablock.list`, over the recorded paths.
        """
        topicpath = self._normtopic_(topicpath)
        p = self._topic_path_(*topicpath)
        fs = self._fs_(self._entry)
        if isinstance(p, dict):
            return [e for leaf in self._leaf_paths_(p)
                    for e in list_path(fs, leaf, False)]
        return list_path(fs, p, self._is_dir_topic_(*topicpath))

    def size(self, *topicpath):
        """Total bytes under the topic path. As `Datablock.size`."""
        return size(self.list(*self._normtopic_(topicpath)))

    def to_dict(self, *, deslash: bool = False) -> dict:
        d = {name: getattr(self, name) for name in (
            'hash', 'code', 'version', 'revision', 'gitrepo', 'datalake',
            'anchor', 'tag', 'key', 'keyby', 'tree', 'session', 'id')}
        d['note'] = self.note()
        # The TEXT of each rendering, not the path of the file it was written to.
        d['signature'] = self.signaturestr()
        d['type'] = self.typestr()
        d['quote'] = self.quote()
        d['cite'] = self.cite()
        d['repr'] = self.repr()
        d['paths'] = self.paths()
        d['topics'] = self.topics()
        if deslash:
            d = {k: v.replace('\\', '') if isinstance(v, str) else v for k, v in d.items()}
        return d

    def fields(self) -> dict:
        return self.to_dict()

    def deslash(self, attr):
        a = getattr(self, attr)
        if callable(a):
            a = a()
        return a.replace('\\', '') if isinstance(a, str) else a

    # 3. Accessors ---------------------------------------------------------

    @property
    def TOPICS(self):
        """The recorded ``{topic: filename_or_DIRTOPIC}`` mapping.

        Named for `Datablock.TOPICS`, which is the same thing declared rather
        than recorded -- so `topics()` answers with names on both, and this
        carries what each name maps to.
        """
        return self._dict_column_(self._entry, 'topics', parse=datablocks.literal_topics)

    @property
    def anchor(self):
        return self._entry.get('anchor')

    @property
    def hash(self):
        return self._entry.get('hash')

    @property
    def code(self):
        return self._entry.get('code') or self._entry.get('subhash')

    @property
    def version(self):
        return self._entry.get('version')

    @property
    def tree(self):
        """The build tree that wrote this entry, or None.

        Shared by every entry of that tree, across blocks -- unlike ``id``,
        which is unique per row, and ``hash``, which is per block.
        """
        return DatajournalEntry.column(self._entry, 'tree')

    @property
    def session(self):
        """The `Datajournal` session that wrote this entry, or None.

        A row from before sessions has none -- and on such a row a ``session``
        column, if present, is its TREE under that column's old name.
        """
        if self._entry.get('tree') is None or pd.isna(self._entry.get('tree')):
            return None
        return DatajournalEntry.column(self._entry, 'session')

    @property
    def id(self):
        """This entry's own row id, or None."""
        return self._entry.get('id') or self._entry.get('entry_code')

    @property
    def tag(self):
        return self._entry.get('tag')

    @property
    def keyby(self):
        return self._entry.get('keyby', 'tag_version_shorthash')

    @property
    def revision(self):
        return self._entry.get('revision')

    @property
    def gitrepo(self):
        """The repo(s) the block was built from.

        No live counterpart: a block knows which revision produced it only for
        as long as the journal remembers.
        """
        return self._entry.get('gitrepo')

    @property
    def datalake(self):
        """The datalake the block was stored in -- recorded as ``url`` before the rename."""
        return DatajournalEntry.column(self._entry, 'datalake')

    @property
    def url(self):
        """The name `datalake` had first."""
        return self.datalake

    @property
    def root(self):
        """Protocol-free root derived from ``datalake`` via ``fsspec.url_to_fs``."""
        url = self.datalake
        if url is None:
            return self._entry.get('root')  # legacy fallback
        _, root = fsspec.url_to_fs(url, **self._entry.storage_options)
        return root

    @property
    def key(self):
        """The key, recorded if the row has one and reconstructed if not."""
        recorded = DatajournalEntry.column(self._entry, 'key')
        if recorded is not None:
            return recorded
        return self._key_from_(self.keyby, self.hash, self.tag, self.version,
                              signature=lambda: self.signaturestr())

    @property
    def anchorkey(self):
        key = self.key
        return os.path.join(self.anchor, key) if key else self.anchor

    @property
    def anchorkeypath(self):
        recorded = DatajournalEntry.column(self._entry, 'anchorkeypath')
        if recorded is not None:
            return recorded
        url = self.datalake
        if url is None:
            root = self._entry.get('root')  # legacy: only 'root' available
            return os.path.join(root, self.anchorkey) if root else self.anchorkey
        fs, root = fsspec.url_to_fs(url, **self._entry.storage_options)
        return fs_full_path(fs, os.path.join(root, self.anchorkey))

    @functools.cached_property
    def redirection(self):
        """Where this entry sends a failed read: an ``id``, a filter, or None.

        Unlike ``quote``/``signature``/``note``, whose columns hold the PATH of
        a file carrying the value, this column holds the redirection itself --
        a dict recorded as ``str(dict)``, the way ``paths`` and ``topics`` are.
        There is no file to go missing, so it resolves as long as the journal
        does. None when the row records none, and for every row in a journal
        written before the column existed.
        """
        raw = self._entry.get('redirection')
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None
        if isinstance(raw, dict):
            return raw
        raw = str(raw)
        if not raw:
            return None
        return ast.literal_eval(raw) if raw.startswith('{') else raw

    # 4. Helpers -----------------------------------------------------------

    # Static, and scoped here rather than left as private methods on the
    # entry: they belong to this class, and a pandas Series shares its
    # attribute namespace with every column name in the journal.
    @staticmethod
    def _fs_(entry):
        url = DatajournalEntry.column(entry, 'datalake') or entry.get('root')  # legacy fallback
        fs, _ = fsspec.url_to_fs(url, **entry.storage_options)
        return fs

    @staticmethod
    def _walk_(mapping, topicpath):
        """Descend a recorded mapping one segment per level; None if absent."""
        node = mapping
        for name in topicpath:
            if not isinstance(node, dict) or name not in node:
                return None
            node = node[name]
        return node

    @staticmethod
    def _normtopic_(topicpath):
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            return tuple(topicpath[0])
        return tuple(topicpath)

    @staticmethod
    def _leaf_paths_(node):
        """Every recorded path at or below *node*, flattened."""
        if isinstance(node, dict):
            return [p for child in node.values() for p in Block._leaf_paths_(child)]
        return [node]

    def _topic_path_(self, *topicpath):
        topicpath = self._normtopic_(topicpath)
        paths = self.paths()
        node = self._walk_(paths, topicpath)
        if node is None and self._walk_(self.TOPICS, topicpath) is None:
            raise KeyError(
                f"topic {'/'.join(topicpath)!r} not recorded in this journal entry's "
                f"paths; available topics: {sorted(paths)}"
            )
        return node

    def _is_dir_topic_(self, *topicpath):
        """A directory topic: recorded as :data:`DIRTOPIC` or the :class:`DIR` marker."""
        node = self._walk_(self.TOPICS, self._normtopic_(topicpath))
        return node is datablocks.DIRTOPIC or datablocks.is_topicmarker(node, datablocks.DIR)

    def _is_syntopic_(self, *topicpath):
        """A synthetic topic -- recorded as :data:`SYNTOPIC` or the :class:`SYNTHETIC` marker."""
        node = self._walk_(self.TOPICS, self._normtopic_(topicpath))
        return (isinstance(node, tuple) and len(node) == 0) or datablocks.is_topicmarker(node, datablocks.SYNTHETIC)

    @staticmethod
    def _signature_text_(entry):
        """The signature TEXT, read through the column when it holds a path.

        `Datablock.signature` answers with the signature itself, so this does
        too. The column may hold the text or the path of a file carrying it,
        depending on when the row was written.
        """
        for name in ('signature', 'subsignature', 'norm'):
            val = entry.read(name, safe=True)
            if val:
                return val
        raw = DatajournalEntry.column(entry, 'signature')
        return str(raw) if raw is not None else None

    @staticmethod
    def _type_text_(entry):
        """The type TEXT, read through the column when it holds a path."""
        val = entry.read('type', safe=True)
        if val:
            return val
        raw = DatajournalEntry.column(entry, 'type')
        return str(raw) if raw is not None else None

    @staticmethod
    def _dict_column_(entry, field, parse=None):
        """A journal column recorded as ``str(dict)``, back as a dict.

        *parse* is how the text is read, :func:`ast.literal_eval` by default and
        :func:`literal_topics` for the topics column, whose values may be topic
        markers rather than literals.
        """
        raw = entry.get(field)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return {}
        if isinstance(raw, dict):
            return raw
        return (parse or ast.literal_eval)(raw)

    @staticmethod
    def _type_parts_(text):
        return text.split(os.sep) if os.sep in text else text.split('/')

    @staticmethod
    def _as_version_(text):
        try:
            return int(text)
        except (ValueError, TypeError):
            return None if text == 'None' else text

    @staticmethod
    def _key_from_(keyby, hash, tag, version, signature=None):
        """Reconstruct a key from recorded fields, mirroring `Datablock.key`.

        *signature* is deferred: keying by it is rare, and resolving it may
        read a file the column only names.
        """
        if keyby is None:
            return None
        if keyby == 'hash':
            return hash
        if keyby == 'signature':
            return signature() if signature is not None else None
        if keyby == 'tag':
            return tag
        if keyby in ('taghash', 'tag_hash'):
            return hash if tag is None else f"{tag}/{hash[:8]}"
        if keyby == 'version_hash':
            return f"version={version}/{hash[:8]}" if version is not None else hash
        if keyby in ('tag_version_hash', 'tag_version_shorthash'):
            parts = []
            if tag is not None:
                parts.append(tag)
            if version is not None:
                parts.append(f"version={version}")
            parts.append(hash[:8] if (keyby == 'tag_version_shorthash' or parts) else hash)
            return '/'.join(parts)
        return hash  # fallback

    def _recorded_text_(self, column, deslash):
        """The text of the side file *column* names, or None when the row has none."""
        val = self._entry.read(column, safe=True)
        return val.replace('\\', '') if (deslash and val) else val

    @staticmethod
    def _reject_rendering_choice_(what, kwargs):
        """Refuse ``legacy*=`` on a rendering that was already produced.

        They choose how a signature is PRODUCED, and this one was produced when
        the entry was written. Accepting and ignoring them would answer a
        question about one rendering with the text of another.
        """
        offending = sorted(k for k in kwargs
                           if k.startswith('legacy') and kwargs[k] is not None)
        if offending:
            raise TypeError(
                f"Block.{what}: {offending} choose how a signature is rendered, "
                f"but this row records one already rendered. Ask the block "
                f"itself ({what}(legacy_typing=...)) for another rendering."
            )


class DatajournalEntry(pd.Series):
    """A single row from a Datablock journal, with convenience accessors.

    Inherits from :class:`pandas.Series` so all standard pandas
    operations work.  Named properties expose journal-specific fields
    (``anchor``, ``hash``, ``url``, ``revision``, …).
    """
    #: pandas carries only the attributes named here across operations that
    #: rebuild the object -- pickling among them. Without this, an entry that
    #: crosses a process boundary (a Ray proxy, a multiprocessing executor)
    #: arrives with its data intact but no `logger`, and the next method to log
    #: dies with AttributeError. Mirrors :attr:`DatajournalFrame._metadata`.
    _metadata = ['storage_options', 'logger']

    #: Column-name chains, oldest spelling last. A journal written before a
    #: rename still resolves, and one written after does not pay for the
    #: fallback.
    COLUMN_CHAINS = {
        'subsignature': ('subsignature', 'norm'),
        'note': ('note', 'message'),
        'signature': ('signature', 'subsignature', 'norm'),
        'type': ('type', 'signature', 'hashstr'),
        'tree': ('tree', 'session', 'uuid'),
        'datalake': ('datalake', 'url'),
    }

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, series: pd.Series, *, storage_options: dict = None,
                 logger: Logger = Logger(name="DatajournalEntry")):
        super().__init__(series)
        self.storage_options = storage_options or {}
        self.logger = logger

    def __tag__(self):
        return f"DatajournalEntry:{self.get('anchor')}/{self.get('hash')}"

    # 2. Declared API ------------------------------------------------------

    def read(self, *things, raw: bool = False, deslash: bool = False, safe: bool = False):
        def read_thing(thing):
            target_attr = 'subsignature' if thing in ('subsignature', 'norm') else ('note' if thing in ('note', 'message') else thing)
            # The COLUMN, not the accessor: the Datablock-shaped accessors are
            # methods now, and getattr would hand back a bound method whose
            # str() is its repr rather than the path the column holds.
            val = self.column(self, target_attr)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                path = str(val)
                _, _ext = os.path.splitext(path)
                ext = _ext[1:] if _ext else ''
                try:
                    if raw or ext in ('txt', 'log'):
                        result = read_str(path, storage_options=self.storage_options)
                    elif ext == 'yaml':
                        result = read_yaml(path, safe=safe, storage_options=self.storage_options)
                    else:
                        result = str(val)
                except (FileNotFoundError, OSError):
                    self.logger.warning(f"read: {thing}: file not found, returning None: {path}")
                    result = None
            else:
                result = None
            self.logger.detailed(f"read: {thing}: >>\n{result}")
            return result
        if len(things) == 0:
            result = None
        elif len(things) == 1:
            result = read_thing(things[0])
        else:
            result = {thing: read_thing(thing) for thing in things}
        if deslash:
            if isinstance(result, dict):
                result = {k: v.replace('\\', '') if isinstance(v, str) else v for k, v in result.items()}
            elif isinstance(result, str):
                result = result.replace('\\', '')
        return result

    def eval(self, thing, *, debug: bool = False, context={}, eval: bool = False, deslash: bool = False, gitrepo=None, revision=None):
        exc = None
        thingstr = self.read(thing, raw=True)
        if deslash:
            thingstr = thingstr.replace('\\', '')
        r = None
        # Call this here because a new revision may need to be checked out
        gitwrkreposetup(revision=revision, gitrepo=gitrepo, reason=f"because of evaluating a DatajournalEntry field {thing}")
        try:
            if eval:
                __eval__ = vars(datablocks)['eval']
                r = __eval__(thingstr)
            else:
                r = __eval__(thingstr, vars(datablocks), context)
        except Exception as exc:
            raise exc
        return r

    def instantiate(self, gitrepo=None, revision=None):
        if revision == 'journal_entry':
            revision = self.column(self, 'revision')
            self.logger.verbose(f"Instantiating {self.__tag__()} with revision from journal entry {revision}")
        else:
            self.logger.verbose(f"Instantiating {self.__tag__()} with revision {revision}")
        if gitrepo == 'journal_entry':
            gitrepo = self.column(self, 'gitrepo')
            self.logger.verbose(f"Instantiating {self.__tag__()} with gitrepo from journal entry {gitrepo}")
        else:
            self.logger.verbose(f"Instantiating {self.__tag__()} with gitrepo {gitrepo}")
        return self.eval('quote', eval=True, gitrepo=gitrepo, revision=revision)

    def inst(self, gitrepo=None, revision='journal_entry', *, remote=False, **remote_kwargs):
        """Rebuild this entry's Datablock by re-running its recorded ``quote``.

        The default evaluates the quote in THIS interpreter. That can rewind the
        project repo but never ``dbx``, which is already imported -- so a block
        whose hash depends on ``dbx`` rendering that has since changed comes back
        with a DIFFERENT hash, and therefore different paths, than the entry
        records. :meth:`rinst` (aka :meth:`trueinst`) is the way around that.

        ``remote=True`` instantiates on a Ray worker pinned to this entry's own
        revision and returns a proxy (see :meth:`rinst`). Pass an existing
        :class:`Remote` instead of ``True`` to reuse a worker; any other keyword
        arguments are forwarded to :func:`remote`.
        """
        if remote is not False and remote is not None:
            return self.rinst(gitrepo=gitrepo, revision=revision,
                              handle=remote if isinstance(remote, Remote) else None,
                              **remote_kwargs)
        if gitrepo is None:
            gitrepo = dataparts.DBX_GIT_REPO
        if gitrepo is None:
            gitrepo = 'journal_entry'
        return self.instantiate(gitrepo=gitrepo, revision=revision)

    def rinst(self, gitrepo=None, revision='journal_entry', *, handle=None, **remote_kwargs):
        """Instantiate on a pinned Ray worker; return a proxy to the block THERE.

        Exactly equivalent to :meth:`inst` with ``remote=``; that method does
        nothing but translate ``remote=True`` to ``handle=None`` and
        ``remote=<Remote>`` to ``handle=<Remote>``, then call this. Every other
        argument, including *gitrepo*, means the same thing in both and reaches
        :func:`remote` identically -- there is no behaviour reachable through one
        that is not reachable through the other.

        The block is constructed in a worker whose ``dbx`` and project repo were
        both pinned -- before that interpreter started -- to *revision*, which
        defaults to the one this entry recorded. Nothing but a handle comes back,
        so the block never has to survive a trip into an interpreter running
        different code. That is what makes the hash come out right::

            i = entry.inst(remote=True)
            i.hash        # == entry.hash, unlike the local inst()
            i.subsignaturestr()      # forwarded to the worker, result returned here

        *handle* reuses an existing :func:`remote` worker instead of starting one
        per call; it is the caller's job to ensure it was pinned compatibly.

        The proxy forwards attribute and method access, but it is a
        :class:`Remote`, not an ``IJEPAsaurUSPoseStill``: ``isinstance`` is false,
        and implicit dunder protocols (``repr``, ``len``, ``[]``) are looked up on
        the type by the interpreter and so are not forwarded.
        """
        if handle is not None:
            ignored = sorted(remote_kwargs) + (['gitrepo'] if gitrepo is not None else [])
            if ignored:
                raise ValueError(
                    f"rinst: {ignored} configure a NEW worker and cannot be passed alongside "
                    f"an existing handle, whose pinning is already fixed"
                )
        if revision == 'journal_entry':
            revision = self.column(self, 'revision')

        quote = self.read('quote', raw=True)
        if quote is None:
            raise ValueError(f"{self.__tag__()} records no quote to instantiate from")

        if handle is None:
            handle = remote(revision=revision, gitrepo=gitrepo, **remote_kwargs)

        def _build():
            # Runs in the worker. dbx.eval resolves the leading '$' of the quote,
            # importing the project package -- from the pin, since the pin is on
            # that interpreter's path from the moment it started.
            import dbx
            return dbx.eval(quote)

        proxy = handle.run(_build)
        # Keep the pinned worker alive for as long as the caller holds the block.
        # The block lives in an actor of its own, but its class was imported from
        # the pinned worker's path, and the pin clones are owned by this process.
        if isinstance(proxy, Remote):
            proxy._origin = handle
        return proxy

    def trueinst(self, gitrepo=None, revision='journal_entry', *, handle=None, **remote_kwargs):
        """Alias for :meth:`rinst` -- the instantiation whose hash is the recorded one.

        Named for what distinguishes it from :meth:`inst`: the local one cannot
        rewind ``dbx``, which is already imported, so a block whose identity
        depends on rendering that has since changed comes back under a hash the
        entry never had -- and paths that hold nothing. This one is pinned
        before its interpreter starts, so it comes back as the block that was
        actually built.
        """
        return self.rinst(gitrepo=gitrepo, revision=revision, handle=handle, **remote_kwargs)

    @staticmethod
    def column(entry, name):
        """The first present value along *name*'s rename chain, or None.

        The way to read ANY column, rename chain or not. An entry is a Series
        built with `dropna` (see `Datablock.Journal`), so a column that was
        recorded null is not merely None on the row -- its label is gone, and
        ``entry.name`` raises AttributeError where the reader expected None.
        """
        for candidate in DatajournalEntry.COLUMN_CHAINS.get(name, (name,)):
            value = entry.get(candidate)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return value
        return None

    # 3. Accessors ---------------------------------------------------------

    @property
    def block(self):
        """This entry's `Block`: the block as it was when the entry was written.

        The Datablock-shaped API lives there, and the accessors on this class
        forward to it, so ``entry.paths()`` and ``entry.block.paths()`` are the
        same call. Reach for ``.block`` when you want to hand something a
        block-like object rather than a pandas row.
        """
        return Block(self)

    # 4. Helpers -----------------------------------------------------------


class DatajournalFrame(pd.DataFrame):
    _metadata = ['storage_options', 'logger']

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, df: pd.DataFrame|None, *, storage_options: dict = None,
                 parse_datetimes: bool = True, logger: Logger = Logger(),
                 index: str | None = None, unnormalized: bool = False, **filter_kwargs):
        
        # Guard against an empty journal (no parquet files written yet), or unwrap BlocksJournal.
        if hasattr(df, 'journal'):
            df = df.journal
        if df is None:
            df = pd.DataFrame()

        # Before filtering: a filter on 'type' or 'signature' should mean the
        # same thing on every row, which is what normalising decides.
        if not unnormalized:
            df = normalize_journal_frame(df)

        # Process the dataframe before calling super().__init__()
        if parse_datetimes:
            if 'datetime' in df.columns and len(df) and not isinstance(df['datetime'].iloc[0], datetime.datetime): # TODO: use dtype?
                df['datetime'] = pd.to_datetime(df['datetime'], format=JOURNAL_DATETIME_FORMAT)
        df = filter_journal_frame(df, **filter_kwargs)

        if index is ...:
            index = 'id' if 'id' in df.columns else None
        if index is not None:
            if index in df.columns:
                df = df.set_index(index, drop=False)
            else:
                raise KeyError(f"Column {index!r} not found in journal DataFrame")

        # Initialize the DataFrame first
        super().__init__(df)
        
        # Set custom attributes AFTER super().__init__()
        self.storage_options = storage_options or {}
        self.logger = logger

    def __call__(self, entry:int):
        return self.get(entry, dropna=True)

    # 2. Declared API ------------------------------------------------------

    def get(self, entry:int, *, dropna: bool = False):
        """Return the entry at LABEL *entry* (``.loc``, not ``.iloc``).

        A DatajournalFrame is numbered 0..N-1 newest-first, including one built with
        filter kwargs, so a label is also a position -- but only for a journal
        this class constructed. Index a frame you sliced yourself with ``.iloc``.
        """
        entry = self.loc[entry]
        if dropna:
            entry = entry.dropna()
        return DatajournalEntry(entry, storage_options=self.storage_options)

    def list(self, thing, *, take: str = 'last', sortby: Optional[str] = None, ascending: bool = False, raw: bool = False, safe: bool = False, dropna: bool = False):
        if take == 'last':
            unique_rows = self.groupby('hash').last()
        elif take == 'first':
            unique_rows = self.groupby('hash').first()
        elif take == 'all':
            unique_rows = self.set_index('hash')
        else:
            raise ValueError(f"Unknown take value: {take}")
        hashes = []
        datetimes = []
        entries = []
        for hash, row in unique_rows.iterrows():
            try:
                entry = None
                entry = DatajournalEntry(row, storage_options=self.storage_options)
                th = None
                th = entry.read(thing, raw=raw, safe=safe)
                entries.append(th)
            except Exception as exc: 
                self.logger.silent(f"DatajournalFrame: EXCEPTION when reading {thing}: {row=}, {entry=}, {th=}:\nEXCEPTION: {exc}")
                entries.append(pd.Series())
            datetimes.append(row.datetime if 'datetime' in row.index else None)
            hashes.append(hash)
        if raw:
            thingsframe = pd.DataFrame.from_dict({hash: entry for hash, entry in zip(hashes, entries)}, orient='index')
            thingsframe.columns = [thing]
            thingsframe.index.name = 'hash'
            thingsframe = thingsframe.reset_index()
        else:
            thingsframe = pd.DataFrame.from_records(entries)
        thingsframe['hash'] = hashes
        thingsframe['datetime'] = datetimes
        if dropna:
            thingsframe = thingsframe.dropna()
        if sortby is not None and sortby in thingsframe.columns:
            thingsframe = thingsframe.sort_values(sortby, ascending=ascending).set_index(sortby).reset_index() # force sortby to be the first column
        return thingsframe


def one_datalake(datalake, url, what):
    """*datalake*, or *url* -- its name before the rename -- and never two that disagree."""
    if url is not None and datalake is not None and url != datalake:
        raise ValueError(f"{what}: datalake={datalake!r} and url={url!r} disagree; "
                         f"url is datalake's old name")
    return datalake if datalake is not None else url


#: The Datajournals whose ``with`` blocks are open, innermost last. Process-wide
#: rather than a ContextVar, on purpose: a thread does not inherit context
#: variables, and a pipeline that constructs blocks inside a thread pool would
#: then write them under a different session -- silently.
_ACTIVE_DATAJOURNALS = []


_ACTIVE_DATAJOURNALS_LOCK = threading.Lock()


class Datajournal:
    """Where a block's journal lives: how entries are written to it and read back.

    A handle, not the records -- constructing one touches no storage. The
    records are what :meth:`read` returns (a `DatajournalFrame`, or one
    `DatajournalEntry`), and :meth:`write` is how a block adds one.

    Both halves keep to ONE on-disk layout, which is why they live together:
    an entry of block B is ::

        {B.anchorkeypath}/.journal/{fqcn}/journal/{hash}/{fqcn}-journal-{hash}-{dt}.parquet

    with its side files (spec, dfn, kwargs, quote, cite, repr, signature, type,
    note) beside it under ``.journal/{fqcn}/{x}/{hash}/`` -- see :meth:`path` --
    and :meth:`read` globs for exactly that under ``{url}/{anchor}``.

    A block writes every entry through one, and hands one it was given down
    its build tree as it hands down ``tree``. Which one, decided at each read
    and write: the ``datajournal=`` it was given or inherited; else the
    innermost ``with Datajournal()`` open at that moment -- the next one out
    once an inner one has closed; else the process-wide DEFAULT_DATAJOURNAL.
    So ::

        with Datajournal() as dj:
            run_pipeline()          # every block it constructs, however deep
        dj.written_entries()        # ... wrote here, under dj.session

    which is how ``dbx.exec`` puts one command's blocks under one session.
    The scope is the process, threads included. Another process has none of
    its own: a block pickled into one writes to that process's default unless
    it was given a journal explicitly. It is operational, like ``tree``: not part
    of the signature, and never rendered into ``quote()`` or ``cite()``.

    *url*, *storage_options*, *log* and *n_workers* are the defaults
    :meth:`read` uses when not given its own. A block always passes its own url
    and storage options, so for a block they only matter when read directly::

        Datajournal('abfss://lake@acct.dfs.core.windows.net').read('my.Block', event='build:end')
    """

    # 1. Protocol and hooks ------------------------------------------------

    def __init__(self, datalake: str | None = None, *, storage_options: dict | None = None,
                 log: 'Logger | None' = None, n_workers: int | None = 8, url: str | None = None):
        self.datalake = one_datalake(datalake, url, 'Datajournal')
        self.storage_options = storage_options
        self.log = log
        self.n_workers = n_workers
        self._session = str(uuid.uuid4())
        self._written = {}      # entry path -> None: a set that keeps write order
        self._written_lock = threading.Lock()

    def __repr__(self):
        args = [f"{k}={v!r}" for k, v, default in (
            ('datalake', self.datalake, None),
            ('storage_options', self.storage_options, None),
            ('n_workers', self.n_workers, 8)) if v != default]
        # Qualified, so that a repr() carrying it is evaluable as a specline.
        return f"dbx.Datajournal({', '.join(args)})"

    def __getstate__(self):
        # The configuration and the session, not the logger: a handle crosses
        # process boundaries (a pickled block, a .set() deepcopy) and lands in
        # dfn.yaml. The session goes with it, because a copy made on the way
        # to a worker or a child block is the same journal session, not a new
        # one -- otherwise one build tree would be written under many.
        return {'datalake': self.datalake, 'storage_options': self.storage_options,
                'n_workers': self.n_workers, 'session': self._session}

    def __enter__(self):
        with _ACTIVE_DATAJOURNALS_LOCK:
            _ACTIVE_DATAJOURNALS.append(self)
        return self

    def __exit__(self, *exc):
        with _ACTIVE_DATAJOURNALS_LOCK:
            # The innermost entry of THIS handle: nesting one handle in itself
            # is legal, and exits out of order must not pop someone else's.
            for i in range(len(_ACTIVE_DATAJOURNALS) - 1, -1, -1):
                if _ACTIVE_DATAJOURNALS[i] is self:
                    del _ACTIVE_DATAJOURNALS[i]
                    break
        return False

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # One handle per journal session, shared rather than copied: .set()
        # deep-copies a block's state, and _adopt_() hands the handle to every
        # child that way. A copy would split the session's written_entries()
        # across objects nobody holds.
        return self

    def __setstate__(self, state):
        state = dict(state)
        if 'url' in state:                      # pickled before the rename
            state['datalake'] = state.pop('url')
        session = state.pop('session', None)
        self.__init__(**state)
        if session is not None:
            self._session = session

    # 2. Declared API ------------------------------------------------------

    @staticmethod
    def current() -> 'Datajournal | None':
        """The innermost ``with Datajournal()`` open in this process, or None."""
        with _ACTIVE_DATAJOURNALS_LOCK:
            return _ACTIVE_DATAJOURNALS[-1] if _ACTIVE_DATAJOURNALS else None

    def read(self, anchor, loc: int = None, *, iloc: int = None, datalake=None, storage_options=None,
             log=None, n_workers=None, index=None, unnormalized: bool = False, url=None, desc=None, **filter_kwargs):
        """Read *anchor*'s journal under *url*: every entry, newest first, then filtered.

        Returns a `DatajournalFrame`, or the one `DatajournalEntry` at *loc*
        (a label) or *iloc* (a position). *url*, *storage_options*, *log* and
        *n_workers* default to this handle's, and *datalake* then to
        ``DBX_DATALAKE``, ``DBX_ROOT``, ``DBX_URL``.
        """
        log = log or self.log or Logger()
        n_workers = n_workers or self.n_workers or 8
        url = one_datalake(datalake, url, 'Datajournal.read')
        if url is None:
            url = self.datalake
        if storage_options is None:
            storage_options = self.storage_options
        if loc is not None and iloc is not None:
            raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
        if url is None:
            url = default_datalake()
        # A url may arrive as a specline -- a block's `_url_` is one whenever it
        # was constructed with env(...) -- and it is resolved here as
        # __setstate__ resolves a block's own. Without this, fsspec takes
        # "$dbx.getenv('LAKE')" for a protocol-less relative path and roots the
        # journal at the CWD: a directory that cannot exist, reported as a
        # journal that is merely missing.
        if datablocks.Datablock.is_specline(url):
            resolved = eval(url)
            log.detailed(f"Journal: resolved url specline {url!r} to {resolved!r}")
            url = resolved
        if storage_options is None:
            storage_options = default_storage_options()

        fs, root = fsspec.url_to_fs(url, **(storage_options or {}))

        anchordirpath = fs_full_path(fs, os.path.join(root, anchor))

        glob_patterns = [
            os.path.join(anchordirpath, ".dbx", "*/journal/**/*.parquet"),
            os.path.join(anchordirpath, "**/.journal", "*/journal/**/*.parquet"),
        ]

        log.verbose(f"Retrieving journal files from {anchordirpath=} using globs: {glob_patterns} BEGIN")
        parquet_files = []
        with ThreadPoolExecutor(max_workers=min(n_workers, len(glob_patterns))) as glob_ex:
            glob_futures = [glob_ex.submit(fs.glob, p) for p in glob_patterns]
            for gf in as_completed(glob_futures):
                try:
                    parquet_files.extend(gf.result())
                except Exception as e:
                    log.warning(f"Error globbing journal files: {e}")

        # Deduplicate found files while preserving order
        seen = set()
        unique_parquet_files = []
        for pf in parquet_files:
            if pf not in seen:
                seen.add(pf)
                unique_parquet_files.append(pf)
        parquet_files = unique_parquet_files

        if len(parquet_files) == 0 and not fs.exists(anchordirpath):
            raise FileNotFoundError(
                f"Journal directory not found for {anchor!r}: {anchordirpath}\n"
                f"Check that the class name / anchor and url are correct."
            )

        log.verbose(f"Retrieved {len(parquet_files)} parquet_files")
        log.verbose(f"Retrieving journal files from {anchordirpath=} using globs: {glob_patterns} END")

        log.detailed(f"READING JOURNAL: from {anchordirpath=}, files: {parquet_files}")
        df = None
        desc = desc or f"Reading {anchor} journal files"
        if len(parquet_files) > 0:
            dfs = [d for d in Datajournal._read_files_(fs, parquet_files, n_workers=n_workers, log=log, desc=desc)
                   if d is not None]
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df = Datajournal._normalize_columns_(df)
            else:
                df = None
        frame = DatajournalFrame(df, storage_options=storage_options, index=index,
                              unnormalized=unnormalized, **filter_kwargs)
        if loc is not None:
            result = DatajournalEntry(frame.loc[loc].dropna(), storage_options=storage_options)
        elif iloc is not None:
            result = DatajournalEntry(frame.iloc[iloc].dropna(), storage_options=storage_options)
        else:
            result = frame
        return result

    def read_frame(self, paths, *, storage_options=None, log=None, n_workers=None) -> 'DatajournalFrame':
        """The entries at *paths* -- journal entry files, as :meth:`written_entries` lists them.

        One row per path, numbered in the order given, and read exactly as
        :meth:`read` reads a journal:
        the same legacy columns resolved, the same ``entry_path`` recorded. A
        path that cannot be read -- cleared since, say -- is skipped with a
        warning, as :meth:`read` skips one.
        """
        paths = [str(p) for p in paths]
        if not paths:
            return DatajournalFrame(None)
        log = log or self.log or Logger()
        n_workers = n_workers or self.n_workers or 8
        if storage_options is None:
            storage_options = self.storage_options
        if storage_options is None:
            storage_options = default_storage_options()
        fs, _ = fsspec.url_to_fs(paths[0], **storage_options)
        dfs = []
        for order, (path, d) in enumerate(zip(paths, Datajournal._read_files_(
                fs, paths, n_workers=n_workers, log=log))):
            if d is not None:
                d['entry_path'] = path
                d['__order__'] = order
                dfs.append(d)
        if not dfs:
            return DatajournalFrame(None, storage_options=storage_options)
        df = Datajournal._normalize_columns_(pd.concat(dfs, ignore_index=True))
        df = df.sort_values('__order__').drop(columns=['__order__']).reset_index(drop=True)
        return DatajournalFrame(df, storage_options=storage_options)

    def read_entries(self, paths, *, storage_options=None, log=None, n_workers=None) -> 'list[DatajournalEntry]':
        """What :meth:`read_frame` reads, one `DatajournalEntry` per path, in the order given."""
        frame = self.read_frame(paths, storage_options=storage_options, log=log, n_workers=n_workers)
        return [frame.get(i, dropna=True) for i in range(len(frame))]

    def write(self, block, event: str, *, note: str = None, inline_note: bool = False,
              message: str = None, inline_message: bool = False, journal_prefix: str = '',
              redirection: 'str | dict | None' = None):
        """Write one journal entry for *event*, and return its ``entry_code``.

        ``entry_code`` is a fresh uuid per call, and it is the only field that
        identifies a *row*.  Everything else on an entry describes the block
        or the moment: ``hash`` and ``key`` are shared by every entry of that
        block, ``tree`` by every entry of one build tree, and ``datetime``
        is only as unique as its resolution -- two entries written inside the
        same microsecond, or by two processes at once, collide.  So a caller
        holding an ``entry_code`` can address exactly the row it wrote:

            code = block.write_journal_entry(event='note')
            entry = block.journal(entry_code=code, loc=0)

        With one caveat that is a property of where entries live rather than
        of the code.  A journal *file* is per live instance -- its path is
        built from ``block.dt``, which does not move -- so a second call from
        the same instance **overwrites** the first.  The new code is written;
        the old one is gone from storage, though the call that made it still
        returned it.  A code therefore resolves only until that instance
        writes again, which is why ``build()`` leaves a ``build:end`` and no
        ``build:start``: same instance, same file.  To keep both entries,
        write them from separate instances, or pass distinct
        *journal_prefix* values.

        Journals written before this field have no such column; the
        ``entry_code`` accessor on ``DatajournalEntry`` returns None for them.

        *redirection* -- an ``entry_code`` or a journal filter, normally passed
        by :meth:`UNSAFE_redirect` rather than directly -- is recorded IN the
        entry, in the ``redirection`` column, not written out to a file the way
        *note* and ``quote``/``subsignature`/``spec`` are. A redirection is what
        :meth:`read` falls back to when the data it wanted is gone, so it must
        not itself depend on a second file still being there.
        """
        if note is None and message is not None:
            note = message
        if not inline_note and inline_message:
            inline_note = inline_message

        if redirection is not None and not isinstance(redirection, (str, dict)):
            raise TypeError(
                f"redirection must be an entry_code str or a journal filter dict, "
                f"got {type(redirection).__name__}: {redirection!r}"
            )
        # A dict goes in as str(dict), the way 'paths' and 'topics' do -- one
        # parquet column cannot hold both a string and a mapping.
        redirection_value = redirection if (redirection is None or isinstance(redirection, str)) else str(redirection)
        entry_id = uuid.uuid4().hex[:16] if getattr(block, '_uuid16_', False) else str(uuid.uuid4())
        dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        code_seed = f"{block.hash}:{block.tree}:{dt}:{event}:{entry_id}"
        code = hashlib.sha256(code_seed.encode('utf-8')).hexdigest()[:32]

        self._write_dict_(block, 'spec', block.spec)
        dfn = block.dfn
        if dfn.get('datajournal') is not None:
            # Its repr, not the object: a python/object tag is not something
            # read_yaml(safe=True) can load, and the session is in its own column.
            dfn = {**dfn, 'datajournal': repr(dfn['datajournal'])}
        self._write_dict_(block, 'dfn', dfn)
        self._write_dict_(block, 'kwargs', block.kwargs)
        self._write_text_(block, 'quote', block.quote())
        self._write_text_(block, 'cite', block.cite())
        self._write_text_(block, 'repr', block.repr())
        self._write_text_(block, 'signature', block.signaturestr())
        self._write_text_(block, 'type', block.typestr())
        if note is not None and not inline_note:
            self._write_text_(block, 'note', note)

        spec_path = self.path(block, 'spec', 'yaml')
        dfn_path = self.path(block, 'dfn', 'yaml')
        kwargs_path = self.path(block, 'kwargs', 'yaml')
        quote_path = self.path(block, 'quote', 'txt')
        cite_path = self.path(block, 'cite', 'txt')
        signature_path = self.path(block, 'signature', 'txt')
        repr_path = self.path(block, 'repr', 'txt')
        type_path = self.path(block, 'type', 'txt')
        if note is not None and not inline_note:
            note_path = self.path(block, 'note', 'txt')
            note_val = note_path
        else:
            note_val = note
        #
        logpath = self.path(block, 'log', ensure_dirpath=True)
        if logpath is not None:
            has_log = block.fs.exists(logpath)
        else:
            has_log = False
        #
        _TOPICS = getattr(block, 'TOPICS', None)
        topics_dict = ({name: copy.deepcopy(node) for name, node in _TOPICS.items()}
                       if isinstance(_TOPICS, dict)
                       else {topic: datablocks.DIRTOPIC for topic in block.topics()})
        paths_dict = block.paths()
        #
        journal_path = self.path(block, 'journal', 'parquet', ensure_dirpath=True, filename_prefix=journal_prefix)
        df = pd.DataFrame.from_records([{'datetime': dt,
                                         'build:start:datetime': block._build_start_dt,
                                         'build:end:datetime': block._build_end_dt,
                                         'version': block.version,
                                         'dbx_version': block.dbx_version,
                                         'revision': block.revision, 
                                         'datalake': block.datalake,
                                         'anchor': block.anchor,
                                         'hash': block.hash,
                                         'keyby': block.keyby,
                                         'key': block.key,
                                         'anchorkeypath': block.anchorkeypath,
                                         'code': block.code,
                                         'tree': block.tree,
                                         'session': self.session,
                                         'id': entry_id,
                                         'tag': block.tag,
                                         'topics': str(topics_dict),
                                         'paths': str(paths_dict),
                                         'log': logpath if has_log else None,
                                         'event': event,
                                         'redirection': redirection_value,
                                         'spec': spec_path,
                                         'dfn': dfn_path,
                                         'kwargs': kwargs_path,
                                         'quote': quote_path,
                                         'cite': cite_path,
                                         'signature': signature_path,
                                         'type': type_path,
                                         'repr': repr_path,
                                         'note': note_val,
                                         'gitrepo': dataparts.DBX_GIT_REPO,
                                         'wrkrepo': dataparts.DBX_USE_WORK_REPO,
        }])
        with block.fs.open(journal_path, 'wb') as f:
            df.to_parquet(f)
        with self._written_lock:
            self._written[journal_path] = None
        
        tagstr = f"with tag {repr(block.tag)} " if block.tag is not None else ""
        block.log.debug(f"WROTE JOURNAL entry {entry_id} for event {repr(event)} {tagstr}"
                         f"to journal_path {journal_path}")
        return entry_id

    def written_entries(self):
        """Every journal entry path this handle has written, in the order first written.

        A path appears once however often it was written: an instance
        rewrites its one entry file (see :meth:`write`), so the file holds the
        latest entry and the path is listed where it first appeared. Only this
        object's writes -- a block pickled into another process writes through
        a copy, which keeps the session but collects its own list.
        """
        with self._written_lock:
            return list(self._written)

    def path(self, block, x, ext=None, *, ensure_dirpath: bool = True, filename_prefix: str = ''):
        """The file *block*'s artefact *x* is written to: ``.journal/{fqcn}/{x}/{hash}/``, per instance.

        Named by ``block.dt``, which is fixed per live instance -- so every
        write of *x* from one instance lands on the same file.
        """
        xdir = self.dirpath(block, x)
        if ensure_dirpath:
            block.fs.makedirs(xdir, exist_ok=True)
        if ext is None:
            ext = x
        return os.path.join(xdir, f'{filename_prefix}{block.fqcn}-{x}-{block.hash}-{block.dt}.{ext}')

    def dirpath(self, block, x='journal'):
        """The directory holding *block*'s artefact *x* -- for ``'journal'``, this block's entries and no others."""
        return os.path.join(block.anchorkeypath, ".journal", block.fqcn, x, block.hash)

    # 3. Accessors ---------------------------------------------------------

    @property
    def url(self):
        """The name `datalake` had first."""
        return self.datalake

    @property
    def session(self):
        """This handle's session: generated when it is constructed, and fixed for its lifetime.

        Written to every entry this handle writes, so a journal can be cut by
        the process -- or by whoever built their own handle -- that wrote it,
        across build trees. Survives pickling and copying with the handle.
        """
        return self._session

    # 4. Helpers -----------------------------------------------------------

    def _write_dict_(self, block, name, data, *, add_credentials: bool = False):
        if add_credentials:
            data = copy.deepcopy(data)
            data['hash'] = block.hash
            data['datetime'] = block.dt
        #
        ypath = self.path(block, name, 'yaml')
        write_yaml(data, ypath, storage_options=block.storage_options)
        assert block.fs.exists(ypath), f"path {ypath} does not exist after writing"
        block.log.detailed(f"WROTE: {name.upper()}: yaml: {ypath}")
        #
        pqpath = self.path(block, name, 'parquet')
        df = pd.DataFrame.from_records([{k: repr(v) for k, v in data.items()}])
        with block.fs.open(pqpath, 'wb') as f:
            df.to_parquet(f)
        assert block.fs.exists(pqpath), f"pqpath {pqpath} does not exist after writing"
        block.log.detailed(f"WROTE: {name.upper()}: parquet: {pqpath}")

    def _write_text_(self, block, name, text):
        #
        path = self.path(block, name, 'txt')
        write_str(text, path, storage_options=block.storage_options)
        assert block.fs.exists(path), f"scopepath {path} does not exist after writing"
        block.log.detailed(f"WROTE: {name.upper()}: txt: {path}")

    @staticmethod
    def _read_files_(fs, files, *, n_workers, log, desc=None):
        """One frame per entry file, aligned with *files*; None where one could not be read."""
        def read_entry_file(file):
            # Through `fs`, not by path: a glob returns paths as that
            # filesystem names them -- protocol-stripped -- so handing one to
            # pandas reads it off the LOCAL disk, where a memory:// or remote
            # journal file is not. Every entry then "skipped as unreadable" and
            # the journal came back empty rather than failing.
            with fs.open(file, 'rb') as f:
                return pd.read_parquet(f, engine='pyarrow')

        desc = desc or 'Reading journal files'
        results = [None] * len(files)
        with ThreadPoolExecutor(max_workers=max(1, min(n_workers, len(files)))) as ex:
            futures = {ex.submit(read_entry_file, file): i for i, file in enumerate(files)}
            for future in tqdm.tqdm(as_completed(futures), desc=desc, total=len(files)):
                i = futures[future]
                try:
                    _df = future.result()
                    _df['entry_path'] = fs_full_path(fs, files[i])
                except Exception as e:
                    log.warning(f"Skipping unreadable journal file {files[i]}: {e}")
                    continue
                results[i] = _df
        return results

    @staticmethod
    def _normalize_columns_(df):
        """Legacy column names and the canonical column order, on a frame just read."""
        if 'revision' not in df.columns:
            df = df.rename(columns={'version': 'revision',})
        # Backward compat: rename legacy 'context' column to 'note' and alias 'message'
        if 'context' in df.columns and 'note' not in df.columns:
            df = df.rename(columns={'context': 'note'})
        if 'message' in df.columns:
            # Replaced by 'note'. Old rows are read under the new name;
            # the old one is not carried forward.
            if 'note' not in df.columns:
                df = df.rename(columns={'message': 'note'})
            else:
                df = df.drop(columns=['message'])
        # Backward compat: rename legacy 'build_datetime' to 'build:end:datetime'
        if 'build_datetime' in df.columns and 'build:end:datetime' not in df.columns:
            df = df.rename(columns={'build_datetime': 'build:end:datetime'})
        if 'build_datetime' in df.columns:
            if 'build:end:datetime' not in df.columns:
                df['build:end:datetime'] = df['build_datetime']
            if 'datetime' not in df.columns:
                df['datetime'] = df['build_datetime']
            df = df.drop(columns=['build_datetime'])
        # Renamed columns, each applied only when the new name is
        # absent -- a journal spanning the rename has both, and the
        # new one is the one that was written deliberately.
        for legacy, current in (('entry_code', 'id'),
                                ('subhash', 'code')):
            if legacy in df.columns and current not in df.columns:
                df = df.rename(columns={legacy: current})
            elif legacy in df.columns:
                df = df.drop(columns=[legacy])
        df = Datajournal._legacy_tree_(df)
        if 'url' in df.columns:
            # `datalake` was `url` until it was renamed; a journal spanning the
            # rename has both, each row filled in one.
            if 'datalake' in df.columns:
                df['datalake'] = df['datalake'].where(df['datalake'].notna(), df['url'])
            else:
                df = df.rename(columns={'url': 'datalake'})
            df = df.drop(columns=['url'], errors='ignore')
        columns = [c for c in datablocks.Datablock.JOURNAL_COLUMNS
                   if c in df.columns and c != 'event']
        # Anything unlisted keeps its place at the back, ahead of
        # 'event', so a column added later still shows up.
        columns += [c for c in df.columns if c not in set(columns + ['event'])]
        if 'event' in df.columns:
            columns.append('event')
        df = df.sort_values('datetime', ascending=False)[columns].reset_index(drop=True)
        df = df.rename(columns={'build_log': 'log'})
        return df

    @staticmethod
    def _legacy_tree_(df):
        """The build-tree id, recorded as ``uuid``, then ``session``, now ``tree``.

        Decided per ROW, not per frame, because ``session`` is a column again:
        the Datajournal session, written beside ``tree``. So a row with no
        ``tree`` is from before the rename and its ``session`` IS its tree --
        moved across, leaving no session, which that row never had -- while a
        row with a ``tree`` keeps its ``session`` as written.
        """
        if 'tree' not in df.columns:
            df['tree'] = None
        for legacy in ('session', 'uuid'):
            if legacy not in df.columns:
                continue
            old = df['tree'].isna() & df[legacy].notna()
            df.loc[old, 'tree'] = df.loc[old, legacy]
            if legacy == 'session':
                df.loc[old, 'session'] = None
            else:
                df = df.drop(columns=[legacy])
        if df['tree'].isna().all():
            df = df.drop(columns=['tree'])
        return df


#: The journal a block uses when it is given none: one per process, so its
#: `session` is the process's and `written_entries()` everything it wrote.
DEFAULT_DATAJOURNAL = Datajournal()
