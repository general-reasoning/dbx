"""Core framework classes: Datablock, Datastack, journaling, and remote execution.

This module defines the central abstractions of dbx:

- :class:`Datablock` — a config-addressed, journaled unit of computation.
  Each block is uniquely identified by a SHA-256 hash derived from its
  fully-qualified class name, configuration (``spec``), and version.
  Builds are journaled as Parquet entries for full reproducibility.

- :class:`Datastack` — a Datablock that orchestrates the parallel
  construction of child Datablocks (blocks).

- :class:`Remote` / :func:`remote` — Ray-based remote execution of
  dbx pipelines.

- :class:`SlurmRayCluster` — Slurm integration for launching Ray clusters.
"""
import ast
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass, fields, asdict, replace
import datetime
import functools
import gc
import hashlib
import inspect
import os
import re
import shutil
import sys
import tempfile
from typing import Callable, Optional, Union
import uuid


import tqdm

# Disable tqdm's background TMonitor thread.
# The monitor races with explicit update() calls (causing the bar count to
# visually bounce) and is alive at fork() time, triggering the Python 3.12
# DeprecationWarning "This process is multi-threaded, use of fork() may lead
# to deadlocks".  We drive all updates explicitly so the monitor is unneeded.
tqdm.tqdm.monitor_interval = 0

# numpy stays in this namespace even though nothing here calls it: journal
# quotes are eval'd against these globals, and a numpy scalar in a spec
# repr's as "np.float32(1.5)", so re-instantiating one needs the name.
import numpy as np

import fsspec

import pandas as pd


__eval__ = __builtins__['eval']

from . import dataparts
from .dataparts import (
    InlineCallableExecutor,
    Logger,
    LogVolume,
    MultiprocessingCallableExecutor,
    MultithreadingCallableExecutor,
    OutputTee,
    RayCallableExecutor,
    Remote,
    TorchMultiprocessingCallableExecutor,
    TorchMultithreadingCallableExecutor,
    UNSAFE_allowed,
    callable_executor,
    default_storage_options,
    ensure_path,
    eval,
    fs_full_path,
    gitrevision,
    gitwrkreposetup,
    list_path,
    ls_path,
    read_str,
    read_yaml,
    read_exec_journal,
    write_exec_journal,
    filter_journal_frame,
    remote,
    size,
    write_str,
    write_yaml,
)
__version__ = "0.2.0"

class AbsentKey:
    """Singleton marking a key present on only ONE side of a :meth:`Datablock.diffsig`.

    Needed because diffsubsig reports typed values: a key whose value *is* ``None``
    and a key that is missing entirely would otherwise both come back as
    ``None``, which are very different findings -- "this setting changed to None"
    versus "this setting did not exist when that build ran".
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return '<absent>'

    def __bool__(self):
        return False


ABSENT = AbsentKey()


class SignatureTopicsKey:
    """Singleton key under which :meth:`Datablock.difftopics` reports a whole-rendering
    difference -- one that belongs to no single topic.

    Two of those exist. The topics are joined into the :attr:`Datablock.signature`
    in declaration order, so the same topics declared in a different order are a
    different signature and a different hash, though no one topic changed. And a
    block declaring ``TOPICS = {}`` contributes no segment at all where a block
    declaring none contributes ``topics:None``, which again differs without any
    topic differing.

    Reported under a sentinel rather than a string key so it cannot collide with
    a topic that happens to be named for it.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return '<signature topics>'


SIGNATURE_TOPICS = SignatureTopicsKey()


#: The filename of a directory topic in a dict-valued ``TOPICS``, i.e. no file
#: at all -- the topic IS the directory::
#:
#:     TOPICS = {'images': 'images.csv', 'masks': DIRTOPIC}
#:
#: It is literally ``None``, which is what the topic machinery has always
#: tested for, so ``{'masks': None}`` stays valid and identical. The name only
#: says out loud what a bare ``None`` leaves the reader to infer -- and reads
#: correctly against a topic whose filename is genuinely unset.
DIRTOPIC = None

#: A SYNTHETIC topic: one the block presents but never stores, so it has no
#: location -- neither a file nor a directory::
#:
#:     TOPICS = {'data': 'data.parquet', 'cache': SYNTOPIC}
#:
#: ``path()`` and ``dirpath()`` are both ``None``, nothing on the filesystem is
#: created, listed, copied or cleared for it, and it is vacuously valid -- a
#: topic that was never going to be written cannot be missing, so it must not
#: hold the block back from being built or read as valid.
#:
#: Distinct from :data:`DIRTOPIC`, which IS a location: a real directory that
#: merely has no filename inside it. The empty tuple is used precisely so the
#: two cannot collide -- it is falsy like ``None`` but never equal to it, and
#: in CPython it is interned, so ``is SYNTOPIC`` is an exact test.
SYNTOPIC = ()


class _TopicMarkerMeta(type):
    """Renders a topic marker as the declaration that made it.

    A leaf reaches :meth:`Datablock.signature_topics` as ``str(node)``, and the
    journal records ``str(TOPICS)``, which ``repr``s it -- so a marker has to
    spell itself the same way in both, and that spelling has to be text that
    :func:`literal_topics` reads back into this very class.
    """

    #: ``{name: marker}`` for every marker a declaration may name, so a recorded
    #: ``{'masks': DIR}`` reads back as the class DIR while ``{'masks': 'DIR'}``
    #: stays a topic stored in a file called ``DIR``.
    REGISTRY = {}

    # 1. Declared API ---------------------------------------------------

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        # A parameterised marker -- SLICE(idx='int') -- is a subclass carrying
        # `columns`, and is not a name anything may be declared under: it
        # renders as the call that made it and reads back through that call.
        if not namespace.get('columns'):
            _TopicMarkerMeta.REGISTRY.setdefault(name, cls)

    def __repr__(cls):
        columns = cls.__dict__.get('columns')
        if not columns:
            return cls.__name__
        return f"{cls.__name__}({_render_columns(columns)})"

    __str__ = __repr__


class TOPICMARKER(metaclass=_TopicMarkerMeta):
    """Base of the topic markers: :class:`DIR`, :class:`SYNTHETIC`, ``SLICE``.

    A marker says what a topic IS.  The sentinels say it by what value they
    happen to be -- :data:`DIRTOPIC` is ``None`` and :data:`SYNTOPIC` the empty
    tuple -- which a reader has to know by heart and which a filename can be
    mistaken for.  A marker is a CLASS and never a string, so ``{'masks': DIR}``
    and ``{'masks': 'DIR'}`` are different declarations and stay different
    everywhere: the first is a directory topic, the second a topic stored in a
    file named ``DIR``.  Which is why a filename renders quoted under the flag
    and a marker does not -- see :meth:`Datablock._topictext`.

    A declaration that holds a marker IS a modern one -- there is nothing else
    it could mean, so nothing announces it.  It may hold no sentinel as well:
    one declaration renders one way, and the two spellings render differently
    (``topic:masks=DIR`` against ``topic:masks=None``, and a filename quoted
    against bare), so a mixture is refused rather than left to render half of
    itself each way.

    Which also means a marker re-keys the block that adopts it.  The type string
    is what the hash is taken over, and every leaf of a modern declaration
    renders differently from how it did: adopt them on a new class, or accept
    that the old artifacts are orphaned.
    """


class SYNTHETIC(TOPICMARKER):
    """A synthetic topic: presented, never stored.  :data:`SYNTOPIC` as a marker.

    ``path()`` and ``dirpath()`` are both ``None``, nothing is created, listed,
    copied or cleared for it, and it is vacuously valid.
    """


class DIR(TOPICMARKER):
    """A directory topic: the topic IS the directory.  :data:`DIRTOPIC` as a marker.

    A location, unlike :class:`SYNTHETIC` -- a real directory that merely has no
    filename inside it.
    """


def _is_topicmarker(node, kind=TOPICMARKER):
    """True when *node* is the marker *kind*, or a parameterisation of it.

    Markers are classes, so this is the test that keeps a filename out: a
    string is never a marker however it is spelled.
    """
    return isinstance(node, type) and issubclass(node, kind)


def _render_columns(columns):
    """A marker's ``columns`` as the call arguments that reconstruct it.

    Keyword form when every name is an identifier, which is what a declaration
    almost always looks like, and the mapping form when one is not -- ``SLICE``
    accepts both, so either rendering reads back as the same marker.
    """
    if all(isinstance(name, str) and name.isidentifier() for name in columns):
        return ', '.join(f"{name}={coltype!r}" for name, coltype in columns.items())
    return repr(dict(columns))


def literal_topics(text):
    """A recorded ``str(TOPICS)`` back as a TOPICS declaration.

    :func:`ast.literal_eval` with the markers added to its grammar: a bare name
    is the marker declared under it, and a call is that marker parameterised.
    So the distinction a declaration drew between ``DIR`` and ``'DIR'`` survives
    a round trip through the journal, where both are just text in a column.

    A parse, not an eval -- nothing outside the marker registry is resolved, and
    no expression is executed.
    """
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    return _topic_literal(ast.parse(text, mode='eval').body)


def _topic_literal(node):
    """One :mod:`ast` node of a recorded TOPICS as the value it stands for."""
    if isinstance(node, ast.Dict):
        return {_topic_literal(k): _topic_literal(v)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        return _marker_named(node.id)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"recorded TOPICS: {ast.dump(node.func)} is not a marker")
        marker = _marker_named(node.func.id)
        args = [_topic_literal(arg) for arg in node.args]
        return marker(*args, **{kw.arg: _topic_literal(kw.value) for kw in node.keywords})
    return ast.literal_eval(node)


def _marker_named(name):
    """The marker declared under *name*, or a ValueError naming what is known."""
    marker = _TopicMarkerMeta.REGISTRY.get(name)
    if marker is None:
        raise ValueError(
            f"recorded TOPICS names {name!r}, which is not a topic marker; "
            f"known markers are {sorted(_TopicMarkerMeta.REGISTRY)}"
        )
    return marker


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
    if df is None or df.empty:
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


def journal(cls_anchor_or_df=None, loc=None, *, iloc=None, url=None, storage_options=None, log=None, n_workers=8, index=None, unnormalized: bool = False, **filter_kwargs):
    """Retrieve or wrap a Datablock journal.

    Parameters
    ----------
    cls_anchor_or_df : type | str | pd.DataFrame, optional
        A Datablock class, an anchor string, a raw DataFrame, or None to return the eval journal.
    loc : int, optional
        If given, return a single :class:`DatajournalEntry` at this label index.
    iloc : int, optional
        If given, return a single :class:`DatajournalEntry` at this positional index.
        Mutually exclusive with *loc*.
    url : str, optional
        Storage URL.  Defaults to ``DBX_ROOT`` or its alias ``DBX_URL``.
    storage_options : dict, optional
        Storage options for fsspec.  Defaults to ``default_storage_options()``.
    log : Logger, optional
        Logger instance.
    n_workers : int, default 8
        Number of workers for reading journal files.
    index : str, optional
        Column name to set as DataFrame index on the returned Datajournal.
    unnormalized : bool, default False
        Leave the era-dependent ``type``/``signature`` columns exactly as
        recorded. By default they are resolved per row, so the frame means
        what it says -- see `normalize_journal_frame`.
    **filter_kwargs
        Forwarded to :class:`Datajournal` for filtering.

    Returns
    -------
    Datajournal, DatajournalEntry, or pd.DataFrame
    """
    if cls_anchor_or_df is None:
        return read_exec_journal(
            url=url,
            loc=loc,
            iloc=iloc,
            storage_options=storage_options,
            log=log,
            n_workers=n_workers,
            index=index,
            **filter_kwargs,
        )
    if loc is not None and iloc is not None:
        raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
    if isinstance(cls_anchor_or_df, pd.DataFrame):
        return Datajournal(cls_anchor_or_df, storage_options=storage_options, index=index, **filter_kwargs)
    else:
        if isinstance(cls_anchor_or_df, str):
            anchor = cls_anchor_or_df
        elif isinstance(cls_anchor_or_df, type):
            anchor = cls_anchor_or_df.__module__ + "." + cls_anchor_or_df.__name__
        elif hasattr(cls_anchor_or_df, 'anchor'):
            anchor = cls_anchor_or_df.anchor
            if url is None and hasattr(cls_anchor_or_df, '_url_'):
                url = cls_anchor_or_df._url_
            if storage_options is None and hasattr(cls_anchor_or_df, 'storage_options'):
                storage_options = cls_anchor_or_df.storage_options
            if log is None and hasattr(cls_anchor_or_df, 'log'):
                log = cls_anchor_or_df.log
        else:
            anchor = cls_anchor_or_df.__module__ + "." + cls_anchor_or_df.__name__
        return Datablock.Journal(anchor, loc=loc, iloc=iloc, url=url, storage_options=storage_options, log=log, n_workers=n_workers, index=index, **filter_kwargs)


def valid(*args, n_workers=None, summary=False, url=None, events=None, **kwargs):
    """Check validity of the top matching build/instance for specified events for given anchors.

    Parameters
    ----------
    *args : str | type | list | tuple
        Anchors (anchor strings or Datablock classes) to validate.
    n_workers : int, optional
        Number of workers for reading journal files.
    summary : bool, default False
        If True, return the boolean AND of all validation results.
        If False, return a dict mapping anchor to its validation results.
    url : str, optional
        Storage URL for reading journals.
    events : list[str] | str, optional
        Event name(s) to check validity for. Defaults to ``['build:end']``.
    **kwargs
        Additional keyword arguments forwarded to journal query.

    Returns
    -------
    dict or bool
        A dict mapping each anchor to a boolean (or dict of event->bool if multiple events),
        or a single boolean value if *summary* is True.
    """
    called_from_cli = False
    if not args:
        called_from_cli = True
        dataparts.pintrampoline()
        import argparse
        parser = argparse.ArgumentParser(prog="dbx.valid", description="Validate latest builds for given anchors.")
        parser.add_argument("anchors", nargs="*", help="Anchor keys or Datablock names")
        parser.add_argument("--n-workers", type=int, default=None, help="Number of workers for journal scanning")
        parser.add_argument("--summary", action="store_true", help="Return boolean AND of results")
        parser.add_argument("--url", type=str, default=None, help="Storage URL")
        parser.add_argument("--events", nargs="+", default=None, help="Event names to check validity for (default: build:end)")

        cli_argv = [a for a in sys.argv[1:] if not a.startswith(dataparts.PIN_FLAGS)]
        parsed, unknown = parser.parse_known_args(cli_argv)

        anchors = parsed.anchors
        if parsed.n_workers is not None and n_workers is None:
            n_workers = parsed.n_workers
        if parsed.summary:
            summary = True
        if parsed.url is not None and url is None:
            url = parsed.url
        if parsed.events is not None and events is None:
            events = parsed.events

        for arg in unknown:
            if "=" in arg:
                k, v = arg.split("=", 1)
                try:
                    kwargs[k] = eval(v)
                except Exception:
                    kwargs[k] = v
    else:
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            anchors = list(args[0])
        else:
            anchors = list(args)

    if events is None:
        events = ['build:end']
    elif isinstance(events, str):
        events = [events]
    else:
        events = list(events)

    results = {}
    for anchor in anchors:
        if isinstance(anchor, str):
            anchor_key = anchor
        elif isinstance(anchor, type):
            anchor_key = f"{anchor.__module__}.{anchor.__name__}"
        elif hasattr(anchor, "anchor"):
            anchor_key = anchor.anchor
        else:
            anchor_key = str(anchor)

        j_kwargs = dict(kwargs)
        if n_workers is not None:
            j_kwargs['n_workers'] = n_workers
        if url is not None:
            j_kwargs['url'] = url

        try:
            j = journal(anchor_key, **j_kwargs)
        except Exception:
            j = None

        event_results = {}
        for ev in events:
            is_val = False
            if j is not None and len(j) > 0 and 'event' in j.columns:
                j_ev = j[j['event'] == ev]
                if len(j_ev) > 0:
                    try:
                        row = j_ev.iloc[0]
                        entry = DatajournalEntry(row.dropna(), storage_options=getattr(j, 'storage_options', None))
                        block = entry.instantiate()
                        val_res = block.valid()
                        if isinstance(val_res, dict):
                            is_val = bool(all(val_res.values()))
                        else:
                            is_val = bool(val_res)
                    except Exception:
                        is_val = False

            event_results[ev] = is_val

        if len(events) == 1:
            results[anchor] = event_results[events[0]]
        else:
            results[anchor] = event_results

    def _all_true(obj):
        if isinstance(obj, dict):
            return all(_all_true(v) for v in obj.values())
        return bool(obj)

    if summary:
        final_res = _all_true(results) if results else True
    else:
        final_res = results

    if called_from_cli:
        import pprint
        if isinstance(final_res, bool):
            print(final_res)
        else:
            pprint.pprint(final_res)

    return final_res



class CallableSignature(CallableStr):
    """A signature string already rendered and stored, e.g. on a journal row.

    ``legacy=`` selects which rendering to PRODUCE, so it has nothing to act
    on here -- the rendering happened before this string was stored. Passing
    it raises rather than being quietly ignored; ask the block itself
    (``block.signature(legacy=...)``) for the other rendering.
    """

    def __call__(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = False, **kwargs):
        if legacy is not None:
            raise TypeError(
                f"{type(self).__name__}: legacy= chooses how a signature is "
                f"rendered, but this one is already rendered and stored. Call "
                f"signature(legacy={legacy!r}) on the block instead."
            )
        s = str(self)
        # Before parsing, not after: stripping backslashes from the formatted
        # output would eat repr's escapes inside the leaves.
        if deslash:
            s = s.replace('\\', '')
        if pretty:
            import pprint
            parsed = Datablock._parse_signature(s)
            sig_dict = {k: Datablock._structure_from_signature_text(v) for k, v in parsed.items()}
            return pprint.pformat(sig_dict, indent=2, width=120)
        return s


class CallableSig(CallableStr):
    """A signature string already rendered and stored, e.g. on a journal row.

    ``legacy=`` selects which rendering to PRODUCE, so it has nothing to act
    on here -- the rendering happened before this string was stored. Passing
    it raises rather than being quietly ignored; ask the block itself
    (``block.signature(legacy=...)``) for the other rendering.
    """

    def __call__(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = True, **kwargs):
        if legacy is not None:
            raise TypeError(
                f"{type(self).__name__}: legacy= chooses how a signature is "
                f"rendered, but this one is already rendered and stored. Call "
                f"signature(legacy={legacy!r}) on the block instead."
            )
        s = str(self)
        # Before parsing, not after: stripping backslashes from the formatted
        # output would eat repr's escapes inside the leaves.
        if deslash:
            s = s.replace('\\', '')
        if pretty:
            import pprint
            parsed = Datablock._parse_signature(s)
            sig_dict = {k: Datablock._structure_from_signature_text(v) for k, v in parsed.items()}
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
                sig_part = str(self._block.signature()) if self._block is not None else ''
                for p in parts:
                    if p.startswith('topic:'):
                        topics.append(p)
                    elif p.startswith('_paths_='):
                        paths = p[len('_paths_='):]
                    elif p.startswith('version='):
                        ver_str = p[len('version='):]
                        version = None if ver_str == 'None' else ver_str

                sig_dict = Datablock._parse_signature(sig_part)
                sig_dict = {k: Datablock._structure_from_signature_text(v) for k, v in sig_dict.items()}
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
                sig_part = str(self._block.signature()) if self._block is not None else ''
                for p in parts:
                    if p.startswith('topic:'):
                        topics.append(p)
                    elif p.startswith('_paths_='):
                        paths = p[len('_paths_='):]
                    elif p.startswith('version='):
                        ver_str = p[len('version='):]
                        version = None if ver_str == 'None' else ver_str

                sig_dict = Datablock._parse_signature(sig_part)
                sig_dict = {k: Datablock._structure_from_signature_text(v) for k, v in sig_dict.items()}
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
    So ``paths()``, ``signature()``, ``type()`` are calls and ``hash``,
    ``anchor``, ``key`` are not, and code written against a live block reads
    one of these unchanged. Accessors with no `Datablock` counterpart
    (``gitrepo``, ``url``, ``id``, ``keyby``) are here too, because they
    describe the block rather than the journal.
    """

    # 1. Declared API ---------------------------------------------------

    def __init__(self, entry: 'DatajournalEntry'):
        self._entry = entry

    def __repr__(self):
        return f"Block({self.anchor}/{self.hash})"

    def signature(self, *, deslash: bool = False, **kwargs):
        """The recorded signature TEXT.

        A method, as on `Datablock` -- but with nothing left to render: the row
        holds one rendering, chosen when it was written. ``legacy=`` and its
        kin therefore raise rather than being ignored.
        """
        self._reject_rendering_choice('signature', kwargs)
        val = self._signature_text(self._entry)
        return val.replace('\\', '') if (deslash and val) else val

    def sig(self, *, deslash: bool = False, pretty: bool = True, **kwargs):
        val = self._signature_text(self._entry)
        return CallableSig(val)(deslash=deslash, pretty=pretty) if val else None

    def type(self, *, deslash: bool = False, **kwargs):
        """The recorded type TEXT."""
        self._reject_rendering_choice('type', kwargs)
        val = self._type_text(self._entry)
        return val.replace('\\', '') if (deslash and val) else val

    def tp(self, *, deslash: bool = False, pretty: bool = True, **kwargs):
        val = self._type_text(self._entry)
        return CallableTp(val, block=self)(deslash=deslash, pretty=pretty) if val else None

    def signaturedict(self, *, deslash: bool = False, **kwargs) -> dict:
        """As `Datablock.signaturedict`, over the signature this row records."""
        self._reject_rendering_choice('signaturedict', kwargs)
        text = self._signature_text(self._entry) or ''
        if deslash:
            text = text.replace('\\', '')
        parsed = Datablock._parse_signature(text)
        return {k: Datablock._structure_from_signature_text(v) for k, v in parsed.items()}

    def sigdict(self, *, deslash: bool = False, **kwargs) -> dict:
        return self.signaturedict(deslash=deslash, **kwargs)

    def typedict(self, *, deslash: bool = False, **kwargs) -> dict:
        self._reject_rendering_choice('typedict', kwargs)
        version, paths, topics = self.version, None, []
        for part in self._type_parts(self._type_text(self._entry) or ''):
            if part.startswith('topic:'):
                topics.append(part)
            elif part.startswith('_paths_='):
                paths = part[len('_paths_='):]
            elif part.startswith('version='):
                version = self._as_version(part[len('version='):])
        return {
            'paths': paths,
            'signature': self.signaturedict(deslash=deslash),
            'topics': tuple(topics),
            'version': version,
        }

    def tpdict(self, *, deslash: bool = False, **kwargs) -> dict:
        return self.typedict(deslash=deslash, **kwargs)

    def paths(self) -> dict:
        """Recorded ``{topic: path}`` mapping.

        A method, as `Datablock.paths` is -- and the reason this class exists:
        the paths the build actually wrote, not paths derived from an identity
        recomputed under today's rendering.
        """
        return self._dict_column(self._entry, 'paths')

    def topics(self) -> list:
        """The recorded topic names, as `Datablock.topics` answers.

        Names, not the ``{name: filename}`` mapping the column holds: this
        mirrors the live API, and the mapping is the entry's own business --
        read it off the row when you want it.
        """
        return list(self._dict_column(self._entry, 'topics', parse=literal_topics))

    def cite(self, **kwargs):
        """Path to this entry's ``cite.txt``, or None."""
        return self._entry.get('cite')

    def note(self):
        """Path to or content of this entry's note, or None."""
        return DatajournalEntry.column(self._entry, 'note')

    def is_topicgroup(self, *topicpath):
        """True when the recorded entry for *topicpath* is a group of topics."""
        return isinstance(self._walk(self.TOPICS,
                                     self._normtopic(topicpath)), dict)

    def ls(self, *topicpath, detail=False):
        """List what is at the recorded path for a topic.

        As `Datablock.ls`, but resolving the path from what the row RECORDED
        rather than from an identity recomputed today. A group concatenates
        its members' listings.
        """
        topicpath = self._normtopic(topicpath)
        p = self._topic_path(*topicpath)
        fs = self._fs(self._entry)
        if isinstance(p, dict):
            return [e for leaf in self._leaf_paths(p)
                    for e in ls_path(fs, leaf, False, detail=detail)]
        return ls_path(fs, p, self._is_dir_topic(*topicpath), detail=detail)

    def list(self, *topicpath):
        """Detailed, recursive listing of every file under the topic path.

        As `Datablock.list`, over the recorded paths.
        """
        topicpath = self._normtopic(topicpath)
        p = self._topic_path(*topicpath)
        fs = self._fs(self._entry)
        if isinstance(p, dict):
            return [e for leaf in self._leaf_paths(p)
                    for e in list_path(fs, leaf, False)]
        return list_path(fs, p, self._is_dir_topic(*topicpath))

    def size(self, *topicpath):
        """Total bytes under the topic path. As `Datablock.size`."""
        return size(self.list(*self._normtopic(topicpath)))

    def to_dict(self, *, deslash: bool = False) -> dict:
        d = {name: getattr(self, name) for name in (
            'hash', 'code', 'version', 'revision', 'gitrepo', 'url',
            'anchor', 'tag', 'key', 'keyby', 'session', 'id')}
        d.update({name: getattr(self, name)() for name in
                  ('signature', 'type', 'cite', 'note')})
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

    # 2. Accessors ------------------------------------------------------

    @property
    def TOPICS(self):
        """The recorded ``{topic: filename_or_DIRTOPIC}`` mapping.

        Named for `Datablock.TOPICS`, which is the same thing declared rather
        than recorded -- so `topics()` answers with names on both, and this
        carries what each name maps to.
        """
        return self._dict_column(self._entry, 'topics', parse=literal_topics)

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
    def session(self):
        """The run that wrote this entry, or None.

        Shared by every entry of that run, across blocks -- unlike ``id``,
        which is unique per row, and ``hash``, which is per block.
        """
        return self._entry.get('session')

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
    def url(self):
        return self._entry.get('url')

    @property
    def root(self):
        """Protocol-free root derived from ``url`` via ``fsspec.url_to_fs``."""
        url = self._entry.get('url')
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
        return self._key_from(self.keyby, self.hash, self.tag, self.version,
                              signature=lambda: self.signature())

    @property
    def anchorkey(self):
        key = self.key
        return os.path.join(self.anchor, key) if key else self.anchor

    @property
    def anchorkeypath(self):
        recorded = DatajournalEntry.column(self._entry, 'anchorkeypath')
        if recorded is not None:
            return recorded
        url = self._entry.get('url')
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

    # 3. Helpers --------------------------------------------------------
    # Static, and scoped here rather than left as private methods on the
    # entry: they belong to this class, and a pandas Series shares its
    # attribute namespace with every column name in the journal.

    @staticmethod
    def _fs(entry):
        url = entry.get('url') or entry.get('root')  # legacy fallback
        fs, _ = fsspec.url_to_fs(url, **entry.storage_options)
        return fs

    @staticmethod
    def _walk(mapping, topicpath):
        """Descend a recorded mapping one segment per level; None if absent."""
        node = mapping
        for name in topicpath:
            if not isinstance(node, dict) or name not in node:
                return None
            node = node[name]
        return node

    @staticmethod
    def _normtopic(topicpath):
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            return tuple(topicpath[0])
        return tuple(topicpath)

    @staticmethod
    def _leaf_paths(node):
        """Every recorded path at or below *node*, flattened."""
        if isinstance(node, dict):
            return [p for child in node.values() for p in Block._leaf_paths(child)]
        return [node]

    def _topic_path(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        paths = self.paths()
        node = self._walk(paths, topicpath)
        if node is None and self._walk(self.TOPICS, topicpath) is None:
            raise KeyError(
                f"topic {'/'.join(topicpath)!r} not recorded in this journal entry's "
                f"paths; available topics: {sorted(paths)}"
            )
        return node

    def _is_dir_topic(self, *topicpath):
        """A directory topic: recorded as :data:`DIRTOPIC` or the :class:`DIR` marker."""
        node = self._walk(self.TOPICS, self._normtopic(topicpath))
        return node is DIRTOPIC or _is_topicmarker(node, DIR)

    def _is_syntopic(self, *topicpath):
        """A synthetic topic -- recorded as :data:`SYNTOPIC` or the :class:`SYNTHETIC` marker."""
        node = self._walk(self.TOPICS, self._normtopic(topicpath))
        return (isinstance(node, tuple) and len(node) == 0) or _is_topicmarker(node, SYNTHETIC)

    @staticmethod
    def _signature_text(entry):
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
    def _type_text(entry):
        """The type TEXT, read through the column when it holds a path."""
        val = entry.read('type', safe=True)
        if val:
            return val
        raw = DatajournalEntry.column(entry, 'type')
        return str(raw) if raw is not None else None

    @staticmethod
    def _dict_column(entry, field, parse=None):
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
    def _type_parts(text):
        return text.split(os.sep) if os.sep in text else text.split('/')

    @staticmethod
    def _as_version(text):
        try:
            return int(text)
        except (ValueError, TypeError):
            return None if text == 'None' else text

    @staticmethod
    def _key_from(keyby, hash, tag, version, signature=None):
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

    @staticmethod
    def _reject_rendering_choice(what, kwargs):
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
    #: dies with AttributeError. Mirrors :attr:`Datajournal._metadata`.
    _metadata = ['storage_options', 'logger']

    def __init__(self, series: pd.Series, *, storage_options: dict = None,
                 logger: Logger = Logger(name="DatajournalEntry")):
        super().__init__(series)
        self.storage_options = storage_options or {}
        self.logger = logger

    def __tag__(self):
        return f"DatajournalEntry:{self.get('anchor')}/{self.get('hash')}"

    # 1. Declared API ---------------------------------------------------

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
                __eval__ = globals()['eval']
                r = __eval__(thingstr)
            else:
                r = __eval__(thingstr, globals(), context)
        except Exception as exc:
            raise exc
        return r
    
    def instantiate(self, gitrepo=None, revision=None):
        if revision == 'journal_entry':
            revision = self.revision
            self.logger.verbose(f"Instantiating {self.__tag__()} with revision from journal entry {revision}")
        else:
            self.logger.verbose(f"Instantiating {self.__tag__()} with revision {revision}")
        if gitrepo == 'journal_entry':
            gitrepo = self.gitrepo
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
            i.subsignature()      # forwarded to the worker, result returned here

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
            revision = self.revision

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

    # 2. Accessors ------------------------------------------------------

    @property
    def block(self):
        """This entry's `Block`: the block as it was when the entry was written.

        The Datablock-shaped API lives there, and the accessors on this class
        forward to it, so ``entry.paths()`` and ``entry.block.paths()`` are the
        same call. Reach for ``.block`` when you want to hand something a
        block-like object rather than a pandas row.
        """
        return Block(self)

    # 3. Helpers --------------------------------------------------------

    #: Column-name chains, oldest spelling last. A journal written before a
    #: rename still resolves, and one written after does not pay for the
    #: fallback.
    COLUMN_CHAINS = {
        'subsignature': ('subsignature', 'norm'),
        'note': ('note', 'message'),
        'signature': ('signature', 'subsignature', 'norm'),
        'type': ('type', 'signature', 'hashstr'),
    }

    @staticmethod
    def column(entry, name):
        """The first present value along *name*'s rename chain, or None."""
        for candidate in DatajournalEntry.COLUMN_CHAINS.get(name, (name,)):
            value = entry.get(candidate)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return value
        return None

    def _renamed_column(self, name, legacy):
        """Value of column *name*, falling back to the pre-rename *legacy* column."""
        def absent(v):
            return v is None or (isinstance(v, float) and pd.isna(v))
        value = self.get(name)
        if absent(value):
            value = self.get(legacy)
        return None if absent(value) else value


class Datajournal(pd.DataFrame):
    _metadata = ['storage_options', 'logger']

    def __init__(self, df: pd.DataFrame|None, *, storage_options: dict = None,
                 parse_datetimes: bool = True, logger: Logger = Logger(),
                 index: str | None = None, unnormalized: bool = False, **filter_kwargs):
        
        # Guard against an empty journal (no parquet files written yet).
        if df is None:
            df = pd.DataFrame()

        # Before filtering: a filter on 'type' or 'signature' should mean the
        # same thing on every row, which is what normalising decides.
        if not unnormalized:
            df = normalize_journal_frame(df)

        # Process the dataframe before calling super().__init__()
        if parse_datetimes:
            if 'datetime' in df.columns and not isinstance(df['datetime'].iloc[0], datetime.datetime): # TODO: use dtype?
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H-%M-%S.%f')
        df = filter_journal_frame(df, **filter_kwargs)

        if index is not None:
            if index in df.columns:
                df = df.set_index(index)
            else:
                raise KeyError(f"Column {index!r} not found in journal DataFrame")

        # Initialize the DataFrame first
        super().__init__(df)
        
        # Set custom attributes AFTER super().__init__()
        self.storage_options = storage_options or {}
        self.logger = logger
            

    def get(self, entry:int, *, dropna: bool = False):
        """Return the entry at LABEL *entry* (``.loc``, not ``.iloc``).

        A Datajournal is numbered 0..N-1 newest-first, including one built with
        filter kwargs, so a label is also a position -- but only for a journal
        this class constructed. Index a frame you sliced yourself with ``.iloc``.
        """
        entry = self.loc[entry]
        if dropna:
            entry = entry.dropna()
        return DatajournalEntry(entry, storage_options=self.storage_options)
    
    def __call__(self, entry:int):
        return self.get(entry, dropna=True)

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
                self.logger.silent(f"Datajournal: EXCEPTION when reading {thing}: {row=}, {entry=}, {th=}:\nEXCEPTION: {exc}")
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

    
gitwrkreposetup(reason="datablocks import")


class Datablock:
    """
    Declare topics via TOPICS::

        TOPICS = ['images', 'masks']                             # directory topics
        TOPICS = {'images': 'images.csv', 'masks': DIRTOPIC}     # file and directory
        TOPICS = {'data': 'data.parquet'}                        # single file topic
        TOPICS = {'data': 'data.parquet', 'cache': SYNTOPIC}     # 'cache' is synthetic

    TOPICS must be a list or a dict.  Every topic has a name.  In the dict form
    a filename of :data:`DIRTOPIC` (which is ``None``) marks a directory topic,
    the same thing every entry of the list form is; :data:`SYNTOPIC` marks a
    synthetic topic -- one that is never stored, so ``path()`` and ``dirpath()``
    are ``None``, nothing is created or copied, and validity is vacuous.

    A dict value may itself be a dict, which nests topics::

        TOPICS = {
            'data': {'frames': DIRTOPIC, 'annotations': SYNTOPIC,
                     'index': 'index.csv'},
            'model': 'model.pt',
        }

    Every topic-addressing method then takes one name per level -- ``path('data',
    'frames')``, ``read('data', 'annotations')``, ``ls``, ``size``,
    ``validtopic`` -- and the nesting is mirrored on disk under the block's key.
    A GROUP is addressable in its own right: ``dirpath('data')`` is the parent
    directory of its members, ``path('data')`` is the dict of their paths, and
    ``validtopic('data')`` is true when every leaf beneath it is.
    :meth:`leaftopics` enumerates the leaves as name tuples.  A topic name may
    not contain ``'/'``, which is the separator the signature nests with.

    Storage layout::

        protocol://path --- module/class/ --- topic [--- file]
               url            [anchor]        [topic]   [file]

        url:                'protocol://path/to/root'
        anchorpath:         '{root}/{anchor}'          (root = fsspec-relative path)
        anchorkeypath:      '{root}/{anchor}/{key}'
        dirpath:            '{root}/{anchor}/{key}/topic'
        path:               '{root}/{anchor}/{key}/topic/{TOPICS[topic]}'

    Attributes::

        self.url  = original URL string
        self.fs   = fsspec filesystem object
        self.root = protocol-free path (via fsspec.url_to_fs)
    """
    # Log var formation at .verbose instead of .detailed.
    # VERBOSE_CONFIG is the deprecated spelling and is still honored.
    VERBOSE_VAR = False

    # Set to True on a subclass whose artifacts were already built and are
    # identified by hashes computed BEFORE string kwargs were quoted in subsignature().
    #
    # The unquoted form is ambiguous in two ways that can collide two distinct
    # blocks onto one hash:
    #
    #   * top-level kwargs -- `url=abfss://c@a.net/x, anchor=A` is a flat
    #     string, so a url whose own text contains ', anchor=' is
    #     indistinguishable from a different url plus a different anchor;
    #   * spec values -- a non-string was rendered `repr()`-then-dict-repr'd
    #     (int 5 -> "'5'") while a string was dict-repr'd once ('5' -> "'5'"),
    #     so `n=5` and `n='5'` produced the SAME subsiganture.
    #
    # LEGACY_NORM=False (the default, i.e. every NEW subclass) quotes strings
    # and reprs spec values exactly once, which removes both collisions -- and
    # necessarily changes the hash. Existing subclasses set it to True so their
    # already-computed hashes, keys and storage paths stay valid.
    #: Journal column order: identity first, then when, then where, then what
    #: was recorded, and the event last. Columns not listed here are kept, in
    #: their own order, just ahead of 'event'.
    JOURNAL_COLUMNS = [
        'hash', 'code', 'session', 'id',
        'datetime', 'build:start:datetime', 'build:end:datetime',
        'version', 'dbx_version', 'revision',
        'url', 'anchor', 'keyby', 'key', 'anchorkeypath', 'tag',
        'topics', 'paths',
        'spec', 'dfn', 'kwargs', 'quote', 'cite', 'repr', 'signature', 'type',
        'gitrepo', 'entry_path', 'event',
    ]

    LEGACY_NORM = False
    LEGACY_SIGNATURE = False

    #: Render signature/type the pre-typing way.  LEGACY_SIGNATURE is accepted
    #: as an equivalent spelling, and LEGACY_NORM still implies it, so a
    #: subclass already pinned to the old rendering keeps its hashes without
    #: being touched.
    LEGACY_TYPING = False

    @staticmethod
    def _coerce_to_annotation(value, annotation):
        """*value* as its declared type, when it arrived as text.

        VAR is a dataclass with real annotations, but nothing enforces them, so
        ``shard_size='256'`` reaches a field declared ``int``. Left alone it
        renders quoted and hashes differently from ``256`` -- the same config
        silently becoming two blocks. Coercion is attempted only for a string
        standing in for a non-string field, and only when it is unambiguous:
        anything that does not parse, or parses to the wrong type, is returned
        untouched rather than guessed at.
        """
        if not isinstance(value, str):
            return value
        text = str(annotation)
        # No declared type to coerce toward: `object`/`Any` accept anything, and
        # a str-compatible field may legitimately hold text that looks like a
        # literal. Coercing under `v: object` would re-collide `v=5` with
        # `v='5'`, which is precisely what the non-legacy rendering exists to
        # tell apart.
        if annotation in (object, str, 'object', 'str', None, '') or 'str' in text \
                or 'object' in text or 'Any' in text:
            return value
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return value
        if isinstance(annotation, type):
            # Exact type, not isinstance: bool is an int subclass, so `'True'`
            # on an `int` field would otherwise arrive as True.
            return parsed if type(parsed) is annotation else value
        # A string annotation (`int | None`, a forward ref): accept the literal
        # only when it clearly denoted something other than text.
        return value if isinstance(parsed, str) else parsed

    def _typed_specdict(self, *, legacy: 'bool | None' = None) -> dict:
        """The spec as real Python values -- ints as ints, blocks as sub-dicts.

        Built from ``self.var``, NOT by parsing the rendered signature. The
        rendering is text, and reading it back gives text: that is why
        ``sigdict()`` used to report ``'256'`` for a field holding ``256``,
        with no coercion bug anywhere in sight.

        Speclines stay strings, since a specline IS a string; every other leaf
        is its declared type.
        """
        legacy = self._legacy_typing(legacy)
        fields = self.VAR.__dataclass_fields__
        keys = [f.name for f in fields.values()]
        if not legacy:
            keys = sorted(keys)

        out = {}
        for k in keys:
            value = getattr(self.var, k)
            raw = self.spec[k] if (isinstance(getattr(self, 'spec', None), dict) and k in self.spec) else value
            if isinstance(value, Datablock):
                out[k] = value._typed_specdict(legacy=legacy)
            elif self.is_specline(raw):
                # A specline standing for a block renders as that block, the
                # same as holding the block directly -- which is what keeps
                # quote() -> eval() identity-preserving, since the round trip
                # turns one into the other. Only a specline denoting something
                # that is NOT a block stays text.
                try:
                    evaluated = dataparts.eval(raw)
                except Exception:
                    evaluated = None
                out[k] = (evaluated._typed_specdict(legacy=legacy)
                          if isinstance(evaluated, Datablock) else raw)
            else:
                out[k] = self._coerce_to_annotation(value, fields[k].type)
        return out

    def _legacy_norm(self) -> bool:
        """Whether the pre-LEGACY_NORM rendering applies: root kwargs, str()'d spec.

        Untouched by the typing change, and deliberately SEPARATE from
        :meth:`_legacy_typing`. signature() and hash are relocatable -- free of
        url -- and only this flag has ever decided otherwise. Tying the typing
        opt-out to it would put a url into the identity of every block pinned
        for typing, which is the opposite of preserving it.
        """
        return bool(getattr(self, 'LEGACY_SIGNATURE', False)
                    or getattr(self, 'LEGACY_NORM', False))

    def _legacy_typing(self, legacy: 'bool | None' = None) -> bool:
        """Whether to render the pre-typing way: text leaves, nested blocks as strings.

        Independent of :meth:`_legacy_norm`: this chooses the TYPING, and that
        one chooses whether root kwargs are in the identity. A block pinned
        here keeps exactly the rendering it had, relocatable or not.

        An explicit *legacy* wins, and propagates to nested blocks, so a whole
        subtree renders one way.
        """
        if legacy is not None:
            return bool(legacy)
        return bool(getattr(self, 'LEGACY_TYPING', False)
                    or getattr(self, 'LEGACY_SIGNATURE', False)
                    or getattr(self, 'LEGACY_NORM', False))

    @staticmethod
    def _topictext(node, modern):
        """A topic leaf as the text that follows the ``=`` in its segment.

        In a *modern* declaration a filename is QUOTED and a marker is not,
        which is what keeps the rendering injective now that a leaf can be
        either: bare, a topic stored in a file called ``DIR`` and a :class:`DIR`
        topic would both render ``topic:masks=DIR`` and collide onto one hash
        while meaning different things -- a file in the one and a directory in
        the other.

        A declaration spelled with the sentinels renders every leaf bare, as it
        always has, so no existing hash moves.  Which is the other half of why
        the two spellings may not be mixed: the quotes are themselves a
        re-keying, and one declaration cannot re-key half of itself.
        """
        if modern and isinstance(node, str):
            return repr(node)
        return str(node)

    def _modern_topics(self, topics=ABSENT) -> bool:
        """Whether *topics* is spelled with the markers rather than the sentinels.

        Derived from the declaration rather than announced by a flag: a TOPICS
        holding a marker is a modern one, and there is nothing else it could
        mean.  Derived on each call, so a TOPICS assigned or amended after the
        class body -- or computed per instance, as
        :class:`~dbx.datapoints.DatapointFold`'s is -- is answered as it stands.

        A declaration holding both spellings has no era, and raises rather than
        rendering half of itself each way.
        """
        if topics is ABSENT:
            topics = getattr(self, 'TOPICS', None)
        modern, legacy = self._topic_spellings(topics)
        if modern and legacy:
            raise ValueError(
                f"{self.__class__.__name__}: TOPICS mixes the topic markers with "
                f"the sentinels they replace -- {modern} against {legacy}. The two "
                f"render differently, so one declaration cannot be both: spell "
                f"every topic the one way or the other"
            )
        return bool(modern)

    def _topic_spellings(self, topics, prefix=()):
        """The leaf paths declared as markers, and those declared as sentinels.

        A filename belongs to neither: it is spelled the same either way, and
        only its rendering differs.
        """
        modern, legacy = [], []
        if not isinstance(topics, dict):
            return modern, legacy
        for name, node in topics.items():
            path = prefix + (str(name),)
            if isinstance(node, dict):
                nested = self._topic_spellings(node, path)
                modern.extend(nested[0])
                legacy.extend(nested[1])
            elif _is_topicmarker(node):
                modern.append('/'.join(path))
            elif self._node_is_sentinel(node):
                legacy.append('/'.join(path))
        return modern, legacy

    @staticmethod
    def _node_is_sentinel(node):
        """True when node is one of the sentinels the markers replace."""
        return node is DIRTOPIC or (isinstance(node, tuple) and len(node) == 0)

    @dataclass
    class VAR:
        class LazyLoader:
            def __init__(self, term):
                self.term = term
                self.value = None
            def __call__(self):
                if self.value is None:
                    if isinstance(self.term, str):
                        self.value = dataparts.eval(self.term)
                    else:
                        # from_datablockable passes raw Python objects
                        self.value = self.term
                return self.value

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if isinstance(attr, Datablock.VAR.LazyLoader):
                return attr()
            return attr

    # DEPRECATED ALIAS: subclasses used to declare `class CONFIG(Datablock.CONFIG)`.
    # The name is kept so those declarations still resolve; __setstate__ maps a
    # subclass-declared CONFIG onto self.VAR (see _resolve_legacy_CONFIG).
    CONFIG = VAR

    # Spec keys whose upstream subtree valid_var()/valid_tree() must not descend
    # into. Supersedes the retired VALIDATE_CFG_EXEMPTIONS.
    TREE_SKIP_VALIDATION = {}

    def __init__(
        self,
        *,
        url: str = None,
        spec: Optional[Union[str, dict]] = None,
        anchor: str = None,
        tag: str|None = None,
        info: bool = None,
        verbose: bool = None,
        debug: bool = None,
        detailed: bool = None,
        capture_output: bool = False,
        revision: str = None,
        keyby: str = 'tag_version_shorthash',
        uuid16: bool = False,
        # Shared by every block of one run, and generated when not given. A
        # block's own identity does not depend on it, so it stays out of the
        # signature; it is how the journal groups the entries of a run.
        session: str | None = None,
        # When a read fails, follow a redirection recorded by UNSAFE_redirect()
        # and read from the entry it names instead. See :meth:`read`.
        redirect: bool = True,
        validate_vars: bool = True,
        # DEPRECATED alias of validate_vars. Kept as an explicit parameter so a
        # dfn recorded before the rename still reconstructs faithfully. Left to
        # **kwargs it would be SILENTLY IGNORED -- validation would stay on for
        # a block whose dfn says validate_cfg=False -- and it would additionally
        # persist as a dead dynamic kwarg, drifting quote()/cite() (and hence
        # the journal) from an otherwise identical block. Identity is unaffected
        # either way: subsignature() reads only url/anchor/hash and spec.
        validate_cfg: bool = None,
        storage_options: dict = None,
        local: str|None = None,
        local_must_exist: bool = False,
        **kwargs,
    ):
        # Initialize early logger for __post_init__ if needed, though usually hash is needed
        self.log = Logger(
            f"{self.fqcn}",
            debug=debug,
            verbose=verbose,
            detailed=detailed,
            info=info,
            # stack_depth=2 (default) is correct for both _print (stack[2]) and selected (_getframe(1))
        )
        self._working_params_ = []
        self._uuid16_ = uuid16
        self._uuid = uuid.uuid4().hex[:16] if uuid16 else str(uuid.uuid4())  # unique per live instance, not preserved across serialization
        self.log.detailed(f"__init__: ------------------------------------------------> {tag=}")
        state = {
            'url': url,
            'spec': spec,
            'anchor': anchor,
            'tag': tag,
            'info': info,
            'verbose': verbose,
            'debug': debug,
            'detailed': detailed,
            'capture_output': capture_output,
            'revision': revision,
            'keyby': keyby,
            'uuid16': uuid16,
            'session': session,
            'redirect': redirect,
            'validate_vars': validate_vars if validate_cfg is None else validate_cfg,
            'storage_options': storage_options,
            'local': local,
            'local_must_exist': local_must_exist,
        }
        self.log.detailed(f"__init__: ------------------------------------------------> initial:         {state=}")
        state.update(kwargs)
        self.log.detailed(f"__init__: ------------------------------------------------> updated(kwargs): {state=}")
        self.__setstate__(state)
        self.log.detailed(f"__init__: ------------------------------------------------> __setstate__:    {self._tag_=},{self.tag=}")
        
    def __setstate__(self, state):
        """NB: state keys should match __init__'s keyword arguments, with extra args properly captured in state."""
        # Early logger for unpickling path (__setstate__ is called without __init__)
        if not hasattr(self, 'log'):
            self.log = Logger(
                f"{self.__class__.__name__}",
                debug=state.get('debug', False),
                verbose=state.get('verbose', False),
                detailed=state.get('detailed', False),
                info=state.get('info', True),
            )
        self._working_params_ = []
        self._resolve_legacy_CONFIG()

        # Backward compatibility for legacy pickles or explicit kwargs dict arguments
        old_kwargs = state.pop('kwargs', None)
        old_state = state.pop('state', None)

        if old_kwargs is not None and isinstance(old_kwargs, dict):
            for k, v in old_kwargs.items():
                if k not in state:
                    state[k] = v

        if old_state is not None and isinstance(old_state, dict):
            for k, v in old_state.items():
                if k not in state:
                    state[k] = v

        # `validate_cfg` was renamed `validate_vars`. State pickled before the
        # rename carries only the old key; pop it so it is never re-serialized.
        legacy_validate = state.pop('validate_cfg', None)
        if legacy_validate is not None and state.get('validate_vars') is None:
            state['validate_vars'] = legacy_validate

        def _unquote(v):
            if isinstance(v, str):
                v_strip = v.strip()
                if len(v_strip) >= 2 and ((v_strip[0] == "'" and v_strip[-1] == "'") or (v_strip[0] == '"' and v_strip[-1] == '"')):
                    try:
                        return ast.literal_eval(v_strip)
                    except Exception:
                        pass
            return v

        # Explicit parameters
        self.url = _unquote(state.get('url'))
        # Resolve specline URLs (e.g. "$dbx.getenv('KEY')") to real paths.
        self._url_ = eval(self.url) if self.url is not None else None
        if self._url_ is None:
            self._url_ = os.environ.get('DBX_ROOT') or os.environ.get('DBX_URL')
        if self._url_ is None:
            raise ValueError(f"No url for {self.__class__.__name__}: pass url= or set DBX_ROOT or its alias DBX_URL")

        self.local = _unquote(state.get('local'))
        if self.local == 'None':
            self.local = None
        self.local_must_exist = bool(_unquote(state.get('local_must_exist', False)))

        self.storage_options = _unquote(state.get('storage_options'))
        if isinstance(self.storage_options, str):
            try:
                self.storage_options = ast.literal_eval(self.storage_options)
            except Exception:
                pass
        if self.storage_options is None or not isinstance(self.storage_options, dict):
            self.storage_options = default_storage_options()

        self.fs, self.root = fsspec.url_to_fs(self._url_, **self.storage_options)
        _url_protocol = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
        if _url_protocol in ('file', 'local', ''):
            # url/root is already local storage: local=True and local=False
            # must be identical, so DBX_LOCAL/local= are never consulted.
            self._local_ = self._url_
            self.localfs, self.localroot = self.fs, self.root
        else:
            # Resolve specline LOCALs (e.g. "$dbx.getenv('KEY')") to real paths.
            self._local_ = eval(self.local) if self.local is not None else None
            if self._local_ is None:
                self._local_ = os.environ.get('DBX_LOCAL') or '/tmp/dbx'
            if self._local_ is None:
                raise ValueError(f"No local for {self.__class__.__name__}: pass local= or set DBX_LOCAL")
            if self.local_must_exist and not os.path.isdir(self._local_):
                raise FileNotFoundError(
                    f"local={self._local_!r} for {self.__class__.__name__} does not "
                    f"exist (local_must_exist=True) -- provision/mount it before "
                    f"running (e.g. a dedicated scratch disk that must actually be "
                    f"attached), or construct with local_must_exist=False to let it "
                    f"be auto-created on demand instead."
                )
            self.localfs, self.localroot = fsspec.url_to_fs(self._local_, **self.storage_options)
        self._spec_ = _unquote(state.get('spec'))
        if isinstance(self._spec_, str):
            try:
                parsed_spec = ast.literal_eval(self._spec_)
                if isinstance(parsed_spec, dict):
                    self._spec_ = parsed_spec
            except Exception:
                pass
        if self._spec_ is None:
            self.spec = asdict(self.VAR())
        else:
            self.spec = self._spec_
        self._anchor_ = _unquote(state.get('anchor'))
        if self._anchor_ == 'None':
            self._anchor_ = None
        if state.get('hash') not in (None, 'None'):
            raise TypeError(
                f"{self.__class__.__name__}: hash= is gone. A block's identity "
                f"is sha256(type()) and nothing else, so a pinned hash could "
                f"disagree with the block it was attached to. To read a block "
                f"identified by older code, pin the rendering that produced "
                f"that hash with LEGACY_TYPING / LEGACY_SIGNATURE."
            )
        self._code_ = _unquote(state.get('code') or state.get('subhash'))
        if self._code_ == 'None':
            self._code_ = None
        self._subhash_ = self._code_
        self._tag_ = _unquote(state.get('tag'))
        if self._tag_ == 'None':
            self._tag_ = None
        
        self._revision_ = _unquote(state.get('revision'))
        if self._revision_ == 'None':
            self._revision_ = None
        self.capture_output = bool(_unquote(state.get('capture_output', False)))
        self.keyby = _unquote(state.get('keyby', 'tag_version_shorthash'))
        if self.keyby not in (None, 'hash', 'code', 'subhash', 'superhash', 'norm', 'signature', 'subsignature', 'tag', 'taghash', 'tag_hash', 'version_hash', 'tag_version_hash', 'tag_version_shorthash', 'custom'):
            raise ValueError(f"keyby must be None, 'hash', 'code', 'signature', 'tag', 'taghash', 'tag_hash', 'version_hash', 'tag_version_hash', 'tag_version_shorthash', 'custom', got {self.keyby!r}")
        if self.keyby == 'tag' and self._tag_ is None:
            raise ValueError(
                f"keyby='tag' requires an explicit tag= argument, but none was provided for {self.__class__.__name__}"
            )
        self._uuid16_ = state.get('uuid16', False)
        self._session_ = _unquote(state.get('session'))
        if self._session_ == 'None':
            self._session_ = None
        # Redirection config: dict(code=..., filter=..., paths=...) or legacy bool
        self.redirect = state.get('redirect')
        self.validate_vars = state.get('validate_vars', True)
        self._paths_ = None

        explicit_keys = set(self.__explicit_params__())
        state_params = {k: v for k, v in state.items() if k not in explicit_keys}

        for key in state_params.keys():
            assert key not in explicit_keys | set(self._working_params_), \
                f"Key {key} in state_params conflicts with __explicit_params__() + _working_params_: {explicit_keys | set(self._working_params_)}"
        for k, v in state_params.items():
            setattr(self, k, v)
            
        # self.parameters used for state retrieval
        self.parameters = self.__explicit_params__() + list(state_params.keys())
        
        self.dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        self._build_start_dt = None
        self._build_end_dt = None
        
        # Redefine logger with hash (and tag if present)
        log_name = f"{self.anchor}/{self.key}"
        if self._anchor_ is not None:
            log_name = f"{self.fqcn}: {log_name}"
        if self._tag_ is not None:
            log_name = f"{log_name} [{self._tag_}]"
        self.log = Logger(
            name=log_name,
            debug=state.get('debug', False),
            verbose=state.get('verbose', False),
            detailed=state.get('detailed', False),
            info=state.get('info', True),
            # stack_depth=2 (default) is correct for both _print (stack[2]) and selected (_getframe(1))
        )
        if isinstance(self.redirect, dict):
            self._process_redirect()
        self.__post_init__()
        self.log.detailed(f"======--------------> code: {self.code}")

    def _process_redirect(self):
        if not isinstance(self.redirect, dict):
            return

        code = self.redirect.get('code')
        filter_spec = self.redirect.get('filter')
        paths = self.redirect.get('paths')

        non_nones = [v for v in (code, filter_spec, paths) if v is not None]
        if len(non_nones) != 1:
            raise ValueError(
                f"redirect dict must specify exactly one of 'code', 'filter', or 'paths' as non-None, got {self.redirect!r}"
            )

        code_of_target = None
        resolved_paths = None
        target_entry = None

        if code is not None:
            code_of_target = code
            target_entry = self._find_journal_entry_by_code(code_of_target)
            if target_entry is None:
                raise ValueError(f"redirect failed: no journal entry found for code {code_of_target!r}")
            resolved_paths = target_entry.block.paths()

        elif filter_spec is not None:
            target_entry = self._find_journal_entry_by_filter(filter_spec)
            if target_entry is None:
                raise ValueError(f"redirect failed: filter {filter_spec!r} matches no journal entry")
            code_of_target = target_entry.block.id
            resolved_paths = target_entry.block.paths()

        elif paths is not None:
            code_of_target = None
            resolved_paths = paths

        if target_entry is not None and (resolved_paths is None or not isinstance(resolved_paths, dict)):
            target_block = target_entry.inst()
            resolved_paths = target_block.paths()

        if resolved_paths is None:
            raise ValueError(f"redirect failed: could not resolve paths for {self.redirect!r}")

        self._paths_ = resolved_paths

        if code_of_target is not None and target_entry is not None:
            code_of_source = self.write_journal_entry(event='redirect:target', note=code_of_target)
            target_block = target_entry.inst()
            target_block.write_journal_entry(event='redirection:target', note=code_of_source)

    def _find_journal_entry_by_code(self, code: str):
        try:
            j = self.journal(id=code)
            if len(j) > 0:
                return DatajournalEntry(j.iloc[0].dropna(), storage_options=self.storage_options)
        except Exception:
            pass

        try:
            fs, root = fsspec.url_to_fs(self._url_, **(self.storage_options or {}))
            pattern = os.path.join(fs_full_path(fs, root), "**/journal/**/*.parquet")
            parquet_files = fs.glob(pattern)
            for file in parquet_files:
                try:
                    with fs.open(file, 'rb') as f:
                        df = pd.read_parquet(f, engine='pyarrow')
                    if 'id' in df.columns and code in df['id'].values:
                        row = df[df['id'] == code].iloc[0]
                        return DatajournalEntry(row.dropna(), storage_options=self.storage_options)
                    elif 'entry_code' in df.columns and code in df['entry_code'].values:
                        row = df[df['entry_code'] == code].iloc[0]
                        return DatajournalEntry(row.dropna(), storage_options=self.storage_options)
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def _find_journal_entry_by_filter(self, filter_spec: Union[dict, str]):
        try:
            if isinstance(filter_spec, dict):
                j = self.journal(**filter_spec)
            elif isinstance(filter_spec, str):
                j = self.journal(event=filter_spec)
                if len(j) == 0:
                    j = self.journal(id=filter_spec)
            else:
                return None
            if len(j) > 0:
                return DatajournalEntry(j.iloc[0].dropna(), storage_options=self.storage_options)
        except Exception:
            pass
        return None

    def _resolve_legacy_CONFIG(self):
        """Honor a subclass that still declares ``class CONFIG`` instead of ``VAR``.

        ``Datablock.CONFIG`` is only an alias of ``Datablock.VAR``, so a subclass
        declaring ``CONFIG`` shadows the alias without overriding ``VAR``.  Walk
        the MRO up to ``Datablock`` and take whichever name that subclass chain
        declares first: a ``VAR`` override needs nothing, a ``CONFIG`` override
        is bound to ``self.VAR`` so the rest of the code only ever reads ``VAR``.
        """
        for klass in type(self).__mro__:
            if klass is Datablock:
                break
            if 'VAR' in klass.__dict__:
                break
            if 'CONFIG' in klass.__dict__:
                self.VAR = klass.__dict__['CONFIG']
                break



    def __getstate__(self):
        # Serialization convention for explicit params (url, spec, anchor, …):
        #
        #   _{k}_ = the *original* value the user passed in (or None).
        #           This is what gets serialized so that the block can be
        #           faithfully reconstructed by __setstate__.
        #   {k}   = the *resolved* / post-processed value used at runtime.
        #           For most params the resolution is simple (e.g. eval of
        #           a default expression), but for ``url`` it involves
        #           evaluating speclines like ``$dbx.getenv('KEY')``.
        #
        # The loop below prefers _{k}_ over {k} to capture the original.
        #
        # Exception — ``url``:
        #   After the url/._url_ swap, the naming is inverted:
        #     self.url  = raw specline (what the user passed)
        #     self._url_ = resolved filesystem path
        #   The _{k}_ pattern would pick up the resolved path, losing the
        #   specline and breaking env() relocatability.  We override it
        #   explicitly below.
        _state = {}
        for k in self.__explicit_params__():
            if hasattr(self, f"_{k}_"):
                _state[k] = getattr(self, f"_{k}_")
            elif hasattr(self, k):
                _state[k] = getattr(self, k)
        # Override: serialize the raw specline, not the resolved _url_ and _local_.
        _state['url'] = self.url
        _state['local'] = self.local
        
        #TODO: why does 'log' end up in self.parameters?
        for k in self.parameters:
            if k not in self.__explicit_params__() and k != 'log' and hasattr(self, k):
                _state[k] = getattr(self, k)
        return _state

    @staticmethod
    def __explicit_params__():
        sig = inspect.signature(Datablock.__init__)
        return [
            p.name for p in sig.parameters.values()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and p.name != 'self'
        ]
    
    def set(self, **kw):
        _kw = copy.deepcopy(self.__getstate__())
        _kw.update(kw)     
        return self.__class__(**_kw)
    
    def replace(self, **kw):
        return self.set(**kw)
    
    def __post_init__(self):
        ...

    def _url_to_fs(self, path):
        """Wrapper around ``fsspec.url_to_fs`` that injects ``self.storage_options``."""
        return fsspec.url_to_fs(path, **self.storage_options)

    @property
    def is_local_fs(self):
        """True when this block's storage is on a local filesystem."""
        protocol = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
        return protocol in ('file', 'local', '')

    @property
    def _is_local_fs(self):
        """Deprecated alias for `is_local_fs`."""
        return self.is_local_fs

    def valid_topic(self, *topicpath):
        """Validity of one topic, or of a whole group."""
        topicpath = self._normtopic(topicpath)
        path = self.path(*topicpath)
        valid = self.valid_path(path)
        self.log.detailed(f"{self.anchor}: topic {'/'.join(topicpath)} valid: {valid}")
        return valid

    def validtopic(self, *topicpath):
        """Deprecated alias for valid_topic."""
        return self.valid_topic(*topicpath)

    def validtopics(self, topics=None, *, reduce: bool = False):
        """Deprecated alias for valid_topics."""
        return self.valid_topics(topics, reduce=reduce)

    def valid_topics(self, topics=None, *, reduce: bool = False):
        result = None
        if topics is None:
            topics = self.topics()
        if topics:
            results = {
                topic:
                self.valid_topic(topic) for topic in topics
            }
            if reduce:
                result = all(list(results.values()))
            else:
                result = results
        else:
            result = True  # no topics → always valid
        return result

    def validpath(self, path):
        """Deprecated alias for valid_path."""
        return self.valid_path(path)

    def valid_path(self, path):
        if path is None:
            return True
        elif isinstance(path, dict):
            return all([self.valid_path(p) for p in path.values()])
        elif isinstance(path, list):
            return all([self.valid_path(p) for p in path])
        if path is None or path.endswith("None"): #If topic filename ends with 'None', it is considered to be valid by default
            result = True
        elif isinstance(path, dict):
            result = all([self.valid_path(p) for p in path.values()])
        else:
            result = self.fs.exists(path)
        self.log.detailed(f"{self.anchor}: path {path} valid: {result}") 
        return result
    
    def validpaths(self, topics=None, *, reduce: bool = False):
        """Deprecated alias for valid_paths."""
        return self.valid_paths(topics, reduce=reduce)

    def valid_paths(self, topics=None, *, reduce: bool = False):
        result = None
        if topics is None:
            topics = self.topics()
        results = {
            topic: self.valid_path(self.path(topic))
            for topic in topics
        }
        if reduce:
            result = all(list(results.values()))
        else:
            result = results
        return result

    def valid(self):
        red = self.redirection
        entry = red.entry if red is not None else None
        return self.__valid__(path=entry.block.anchorkeypath if entry is not None else None)

    def validate(self, **kwargs):
        """Validate this block's data. Default implementation calls self.valid().

        Specializations may override it to perform custom validation logic.
        """
        return self.__validate__(**kwargs)

    def __valid__(self, path: str|None = None):
        """Whether this block's data is there; override to decide differently.

        *path* is the directory this block has been redirected to -- the
        anchorkeypath of the entry a ``filter`` matched -- and None when it has
        not been redirected, or when the redirection gave paths outright and so
        names no single block directory. It is for an override that validates a
        redirected-to location by more than the presence of its topics; the
        default needs no such thing, since a redirected block's own
        :meth:`path` already answers with the redirected-to paths.
        """
        return self.valid_topics(reduce=True)

    def __validate__(self, **kwargs):
        """Whether this block's data is there and is correct; override to decide differently.
        """
        return self.valid()
    
    def topics(self):
        """Return the list of TOP-LEVEL topic names.

        For dict-TOPICS, returns the keys.  For list-TOPICS, returns the list.
        Returns an empty list when TOPICS is not defined.  A key naming a
        nested group is returned as itself; :meth:`leaftopics` enumerates
        what is underneath it.
        """
        if not hasattr(self, 'TOPICS'):
            return []
        if isinstance(self.TOPICS, dict):
            return list(self.TOPICS.keys())
        if isinstance(self.TOPICS, list):
            return list(self.TOPICS)
        return []

    def leaftopics(self):
        """Every leaf topic, as a tuple of names, depth-first in declaration order.

        A flat TOPICS yields one-element tuples, in the same order as
        :meth:`topics`, so anything built from this reads identically to the
        pre-hierarchy form -- which is what keeps :attr:`signature` stable.
        """
        def walk(node, prefix):
            if not isinstance(node, dict):
                yield prefix
                return
            for name, child in node.items():
                yield from walk(child, prefix + (self._check_topicname(name),))

        if not self.has_topics():
            return []
        if self._topics_is_list:
            return [(self._check_topicname(name),) for name in self.TOPICS]
        return [tp for name, child in self.TOPICS.items()
                for tp in walk(child, (self._check_topicname(name),))]

    def has_topics(self):
        """True when this block declares named topics (list or dict TOPICS)."""
        return hasattr(self, 'TOPICS') and isinstance(self.TOPICS, (list, dict))

    @property
    def _topics_is_list(self):
        """True when TOPICS is defined as a list (directory-per-topic mode)."""
        return hasattr(self, 'TOPICS') and isinstance(self.TOPICS, list)

    @property
    def _topicfiles(self):
        """The effective topic → filename mapping.

        Returns TOPICS when it is a dict, otherwise None.  For a hierarchical
        TOPICS the values of group keys are themselves such mappings.
        """
        if hasattr(self, 'TOPICS') and isinstance(self.TOPICS, dict):
            return self.TOPICS
        return None

    @staticmethod
    def _normtopic(topicpath):
        """Accept ``('data', 'frames')``, ``(('data', 'frames'),)`` and ``('data',)``.

        The tuple form lets a caller feed a :meth:`leaftopics` entry straight
        back in without unpacking it.
        """
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            return tuple(topicpath[0])
        return tuple(topicpath)

    def _topicnode(self, *topicpath):
        """Resolve a topic path to its TOPICS entry.

        Returns a filename (``str``), :data:`DIRTOPIC`, :data:`SYNTOPIC`, or a
        ``dict`` for a group.  Raises KeyError naming the offending segment
        when the path does not exist, so a typo in a nested name says which
        level it failed at rather than surfacing as a missing file later.
        """
        topicpath = self._normtopic(topicpath)
        if not topicpath:
            # TypeError, not ValueError: before these became varargs this was a
            # missing-positional-argument error, and callers may catch that.
            raise TypeError(
                f"{self.__class__.__name__}: a topic path needs at least one name"
            )
        if not self.has_topics():
            raise KeyError(f"{self.__class__.__name__} declares no TOPICS")
        # For the side effect: a declaration mixing the two spellings has no era
        # and says so here, rather than surfacing as a rendering later.
        self._modern_topics()
        if self._topics_is_list:
            if len(topicpath) > 1:
                raise KeyError(
                    f"list-TOPICS has no groups: {'/'.join(topicpath)} is nested"
                )
            if topicpath[0] not in self.TOPICS:
                raise KeyError(f"topic {topicpath[0]!r} not in {list(self.TOPICS)}")
            return DIRTOPIC

        node = self.TOPICS
        for i, name in enumerate(topicpath):
            self._check_topicname(name)
            if not isinstance(node, dict):
                raise KeyError(
                    f"topic {'/'.join(topicpath[:i])!r} is a leaf, "
                    f"so it has no member {name!r}"
                )

            node = node[name]
            if not (isinstance(node, (dict, str)) or node is DIRTOPIC
                    or self._node_is_syntopic(node) or _is_topicmarker(node)):
                raise TypeError(
                    f"TOPICS entry {'/'.join(topicpath[:i+1])!r} is {node!r}; "
                    f"expected a filename, a topic marker, DIRTOPIC, SYNTOPIC, "
                    f"or a dict of these"
                )
        return node

    @staticmethod
    def _node_is_syntopic(node):
        """True when node is :data:`SYNTOPIC` or the :class:`SYNTHETIC` marker."""
        return (isinstance(node, tuple) and len(node) == 0) or _is_topicmarker(node, SYNTHETIC)

    @staticmethod
    def _check_topicname(name):
        """A topic name may not contain '/'.

        Nesting is rendered into :attr:`signature` as ``topic:data/frames=...``,
        and the signature's own segments are '/'-joined. Allowing a '/' inside
        a name would let two different TOPICS trees render identically and so
        collide onto one hash.
        """
        if not isinstance(name, str):
            raise TypeError(f"topic name must be a string, got {name!r}")
        if '/' in name:
            raise ValueError(
                f"topic name {name!r} may not contain '/': nesting is expressed "
                f"by nesting dicts, and '/' would make the signature ambiguous"
            )
        return name

    def is_topicgroup(self, *topicpath):
        """True when *topicpath* names a group of topics rather than a leaf."""
        return isinstance(self._topicnode(*topicpath), dict)

    def _leaves_under(self, *topicpath):
        """Leaf topic paths at or below *topicpath*, as full tuples from the root."""
        topicpath = self._normtopic(topicpath)
        node = self._topicnode(*topicpath)

        def walk(node, prefix):
            if not isinstance(node, dict):
                yield prefix
                return
            for name, child in node.items():
                yield from walk(child, prefix + (name,))

        return list(walk(node, topicpath))

    @staticmethod
    def _node_is_dirtopic(node):
        """True when node is :data:`DIRTOPIC` or the :class:`DIR` marker.

        A parameterised marker is a subclass of the one it parameterises, so
        ``SLICE(idx='int')`` -- a slice IS a directory -- lands here too.
        """
        return node is DIRTOPIC or _is_topicmarker(node, DIR)

    def _is_dir_topic(self, *topicpath):
        """True when the topic resolves to a directory rather than a file.

        True for list-TOPICS entries and for :data:`DIRTOPIC` leaves.  A
        :data:`SYNTOPIC` is neither, and neither is a group -- a group has a
        directory, but :meth:`path` describes it by its members.
        """
        topicpath = self._normtopic(topicpath)
        if not topicpath or topicpath[0] is None:
            return False
        node = self._topicnode(*topicpath)
        return self._node_is_dirtopic(node)

    def _is_syntopic(self, *topicpath):
        """True when the topic is declared :data:`SYNTOPIC` -- so it has no location.

        Only dict-TOPICS can declare one; every entry of a list-TOPICS is a
        directory.
        """
        topicpath = self._normtopic(topicpath)
        if not topicpath or not self.has_topics() or self._topics_is_list:
            return False
        try:
            return self._node_is_syntopic(self._topicnode(*topicpath))
        except (KeyError, TypeError):
            return False

    def build(self, *args, **kwargs):
        # A redirected block answers its reads out of another entry's data (see
        # :attr:`redirection`), so building it would produce data that nothing
        # would go on to read. Declining is also what makes a redirect stick:
        # a build_tree() sweeping past would otherwise quietly rebuild the very
        # block someone redirected away from. Costs one journal read per
        # instance, which :attr:`redirection` caches.
        if self._redirected_paths_ is not None:
            entry = self.redirection.entry if self.redirection is not None else None
            whither = (f"journal entry {entry.block.id} (hash {entry.block.hash})"
                       if entry is not None else f"the paths {self._redirected_paths_}")
            self.log.info(
                f"BUILD DECLINED: {self.anchorkeypath} is REDIRECTED to {whither}, and reads "
                f"from there instead: nothing would read what a build of it wrote. Undo the "
                f"redirection, or construct with redirect=False, to build it anyway."
            )
            return self
        if self.capture_output:
            logpath = self._dbxanchorhashpathx('log', ext='log', ensure_dirpath=True)
            self.log.verbose(f"-------------------- Capturing stdout/stderr to {logpath} ------------------")

            # Write to a local temp file; upload to remote logpath at the end.
            _local_log = tempfile.NamedTemporaryFile(
                mode='w', suffix='.log', prefix='dbx_capture_', delete=False, encoding='utf-8',
            )
            _output_tee = OutputTee(_local_log)
        _log_uploaded = False
        try:
            if not self.valid():
                self.__pre_build__(*args, **kwargs)
                self.__build__(*args, **kwargs)
                self._build_end_dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
                # Upload the captured log BEFORE __post_build__ writes the
                # journal entry, so that the journal's fs.exists(logpath)
                # check finds the file and records the path.
                if self.capture_output:
                    _output_tee.close()
                    _local_log.close()
                    _log_uploaded = True
                    try:
                        self.fs.put(_local_log.name, logpath)
                        self.log.verbose(f"Captured output uploaded to {logpath}")
                    except Exception as upload_exc:
                        self.log.verbose(f"Failed to upload captured output to {logpath}: {upload_exc}")
                    finally:
                        os.unlink(_local_log.name)
                self.__post_build__(*args, **kwargs)
            else:
                self.log.selected(f"Skipping existing datablock: {self.anchorkeypath}")
        except KeyboardInterrupt as e:
            if self.capture_output and not _log_uploaded:
                _output_tee.close()
                _local_log.close()
                _log_uploaded = True
                try:
                    self.fs.put(_local_log.name, logpath)
                    self.log.verbose(f"Captured output uploaded to {logpath}")
                except Exception as upload_exc:
                    self.log.verbose(f"Failed to upload captured output to {logpath}: {upload_exc}")
                finally:
                    os.unlink(_local_log.name)
            self.__post_build__(*args, event="build:keyboard_interrupt", **kwargs)
            raise(e)
        except Exception as e:
            if self.capture_output and not _log_uploaded:
                _output_tee.close()
                _local_log.close()
                _log_uploaded = True
                try:
                    self.fs.put(_local_log.name, logpath)
                    self.log.verbose(f"Captured output uploaded to {logpath}")
                except Exception as upload_exc:
                    self.log.verbose(f"Failed to upload captured output to {logpath}: {upload_exc}")
                finally:
                    os.unlink(_local_log.name)
            self.__post_build__(*args, event="build:exception", **kwargs)
            raise(e)
        finally:
            if self.capture_output and not _log_uploaded:
                _output_tee.close()
                _local_log.close()
                try:
                    self.fs.put(_local_log.name, logpath)
                    self.log.verbose(f"Captured output uploaded to {logpath}")
                except Exception as upload_exc:
                    self.log.verbose(f"Failed to upload captured output to {logpath}: {upload_exc}")
                finally:
                    os.unlink(_local_log.name)
        return self

    def __pre_build__(self, *args, **kwargs):
        if self.validate_vars:
            valid_var = self.valid_var()
            if not all(list(valid_var.values())):
                for k, v in valid_var.items():
                    if not v:
                        blk = getattr(self.var, k, None)
                        if hasattr(blk, 'valid_topics'):
                            self.log.error(f"Upstream Datablock '{k}' is invalid: valid_topics={blk.valid_topics()} valid_paths={blk.valid_paths()} anchorkeypath={blk.anchorkeypath}")
                raise ValueError(f"Not all upstream Datablocks in var are valid: {valid_var=}")
        self._build_start_dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        self.write_journal_entry(event="build:start",)
        return self

    def __post_build__(self, *args, event="build:end", **kwargs):
        self.write_journal_entry(event=event,)
        return self
    
    def __build__(self, *args, **kwargs):
        return self

    def _transfer_callback(self, desc, *, show_progress: bool):
        """An fsspec transfer callback: a tqdm byte-progress bar when
        *show_progress*, otherwise fsspec's default no-op callback."""
        if not show_progress:
            return fsspec.callbacks.NoOpCallback()
        return fsspec.callbacks.TqdmCallback(tqdm_kwargs=dict(desc=desc, unit='B', unit_scale=True))

    def pull(self, src, dest, *, show_progress: bool = False):
        """Copy *src* (a path on this block's ``fs``) to *dest* (a local path).

        A no-op when *src* and *dest* already refer to the same location
        (e.g. when this block's storage is itself local, so local staging
        aliases the canonical path). Copies files and directories alike.

        show_progress : bool, default False
            If True, show a tqdm byte-progress bar for the transfer —
            useful for large (e.g. multi-GB checkpoint) files.
        """
        if src is None or not self.fs.exists(src):
            self.log.warning(f"pull: source {src!r} does not exist, nothing to download")
            return self
        if src == dest:
            return self
        callback = self._transfer_callback(f"pull {os.path.basename(src.rstrip('/'))}", show_progress=show_progress)
        if self.fs.isdir(src):
            self.localfs.makedirs(dest, exist_ok=True)
            self.fs.get(src, dest, recursive=True, callback=callback)
        else:
            self.localfs.makedirs(os.path.dirname(dest), exist_ok=True)
            self.fs.get(src, dest, callback=callback)
        return self

    def push(self, src, dest, *, free_src: bool = False, show_progress: bool = False):
        """Copy *src* (a local path) to *dest* (a path on this block's ``fs``).

        A no-op when *src* and *dest* already refer to the same location
        (e.g. when this block's storage is itself local, so local staging
        aliases the canonical path). Copies files and directories alike.

        free_src : bool, default False
            If True, remove *src* after a successful upload (skipped
            when *src*/*dest* coincide, since there is nothing to free).
        show_progress : bool, default False
            If True, show a tqdm byte-progress bar for the transfer —
            useful for large (e.g. multi-GB checkpoint) files.
        """
        if src is None or not self.localfs.exists(src):
            self.log.warning(f"push: source {src!r} does not exist, nothing to upload")
            return self
        if src == dest:
            return self
        callback = self._transfer_callback(f"push {os.path.basename(src.rstrip('/'))}", show_progress=show_progress)
        if self.localfs.isdir(src):
            self.fs.makedirs(dest, exist_ok=True)
            self.fs.put(src, dest, recursive=True, callback=callback)
        else:
            self.fs.makedirs(os.path.dirname(dest), exist_ok=True)
            self.fs.put(src, dest, callback=callback)
        if free_src:
            self.localfs.rm(src, recursive=self.localfs.isdir(src))
        return self

    def pulltopics(self, *, path=None, root='.'):
        for topic in self.topics():
            topic_path = None if path is None else os.path.join(path, topic)
            self.pulltopic(topic, path=topic_path, root=root)
        return self

    def pulltopic(self, topic, *, path=None, root='.'):
        """Download topic *topic* to a local destination.

        By default (*path* is ``None``), pulls to local staging
        (``self.path(topic, local=True)``) — the topic's canonical path
        when this block's url is itself local, otherwise the DBX_LOCAL
        staging path. When *path* is given, pulls to
        ``os.path.join(root, path)`` instead.

        Parameters
        ----------
        topic : str
            The topic to download.
        path : str, optional
            Destination path, joined with *root*. When ``None``, pulls
            to local staging instead.
        root : str, default ``'.'``
            Prefix joined with *path* when *path* is given.

        Returns
        -------
        self
        """
        src = self.path(topic)
        if src is None:
            src = self.dirpath(topic)
        if src is None:
            self.log.warning(f"pulltopic: no path for topic={topic!r}, nothing to download")
            return self
        dest = self.path(topic, local=True) if path is None else os.path.join(root, path)
        return self.pull(src, dest)

    def pushtopics(self, *, path=None, root='.'):
        for topic in self.topics():
            topic_path = None if path is None else os.path.join(path, topic)
            self.pushtopic(topic, path=topic_path, root=root)
        return self

    def pushtopic(self, topic, *, path=None, root='.'):
        """Upload topic *topic* from a local source to this block's canonical storage.

        By default (*path* is ``None``), uploads from local staging
        (``self.path(topic, local=True)``) — the counterpart of
        :meth:`pulltopic`'s default. When *path* is given, uploads from
        ``os.path.join(root, path)`` instead.

        Parameters
        ----------
        topic : str
            The topic to upload.
        path : str, optional
            Source path, joined with *root*. When ``None``, uploads
            from local staging instead.
        root : str, default ``'.'``
            Prefix joined with *path* when *path* is given.

        Returns
        -------
        self
        """
        dst = self.path(topic)
        if dst is None:
            dst = self.dirpath(topic)
        if dst is None:
            self.log.warning(f"pushtopic: no path for topic={topic!r}, nothing to upload")
            return self
        src = self.path(topic, local=True) if path is None else os.path.join(root, path)
        return self.push(src, dst)

    def synclocal(self, topic, *, suffix=None, key=None, validate=None, latest: bool = False,
                  show_progress: bool = False):
        """Sync entries of directory-topic *topic* to local staging.

        Lists ``dirpath(topic)``, optionally keeping only entries whose
        name ends with *suffix*, and sorts them ascending by
        ``key(basename)`` (lexical order when *key* is omitted) — e.g.
        a checkpoints topic with filenames like ``ckpt_step_{n}.pt``,
        sorted by the numeric step parsed out of the name. Generalizes
        the find-latest-checkpoint-and-pull-it-if-missing pattern.

        When *latest* is False (default), every matching entry missing
        from local staging is pulled there, and the full list of local
        paths (in sorted order) is returned.

        When *latest* is True, only the single latest (last-sorted)
        entry is synced: pulled to local staging if missing, then, if
        *validate* is given and rejects it (e.g. a truncated/corrupt
        checkpoint), the next-latest entry is tried instead, and so on.
        Returns the local path of the first entry to validate, or
        ``None`` if none did (or there were no matching entries).

        Parameters
        ----------
        topic : str
            The directory topic to sync.
        suffix : str, optional
            Only consider entries whose name ends with *suffix*.
        key : callable, optional
            ``key(basename) -> sortable``. Defaults to lexical order;
            pass one to sort e.g. by an embedded step number instead.
        validate : callable, optional
            ``validate(local_path) -> bool``. Only consulted when
            *latest* is True.
        latest : bool, default False
            See above.
        show_progress : bool, default False
            Forwarded to the underlying :meth:`pull` calls.

        Returns
        -------
        list[str] or str or None
        """
        entries = [(os.path.basename(e.rstrip('/')), e) for e in self.ls(topic)]
        if suffix is not None:
            entries = [(name, path) for name, path in entries if name.endswith(suffix)]
        sort_key = key or (lambda name: name)
        entries.sort(key=lambda item: sort_key(item[0]))

        local_dir = self.dirpath(topic, local=True)

        def sync_one(name, remote_path):
            local_path = os.path.join(local_dir, name)
            if not self.localfs.exists(local_path):
                self.pull(remote_path, local_path, show_progress=show_progress)
            return local_path

        if not latest:
            return [sync_one(name, remote_path) for name, remote_path in entries]

        for name, remote_path in reversed(entries):
            local_path = sync_one(name, remote_path)
            if validate is None or validate(local_path):
                return local_path
        return None

    def note(self, note: str | None = None, event: str = 'note', *, inline: bool = False, message: str | None = None):
        """Write a journal entry with the given *event* and optional *note*.

        The journal parquet file is prepended with ``{event}-`` so it can
        be distinguished from regular journal entries, but it still
        lives under the ``journal/`` directory and therefore is read
        by :meth:`journal`.

        Parameters
        ----------
        note : str, optional
            If provided, recorded in the journal ``note`` field.
        event : str, default 'note'
            The event name recorded in the journal (e.g. ``'keep'``,
            ``'note'``).
        inline : bool, default False
            When ``True`` the *note* string is stored directly in the
            journal record.  When ``False`` the note is written to a
            separate text file and the journal stores the file path.
        message : str, optional
            Legacy alias for *note*.

        Returns
        -------
        self
        """
        if note is None and message is not None:
            note = message
        self.write_journal_entry(
            event=event,
            note=note,
            inline_note=inline,
            journal_prefix=f'{event}-',
        )
        return self

    def leave_breadcrumbs(self):
        """Touch a breadcrumb for every topic that has a location.

        A file topic's breadcrumb IS its file, so the block reads as valid.
        A directory topic has no filename, so it gets ``{dirpath}.crumbs``
        beside it rather than a stray entry inside a listing of it -- writing
        to the directory path itself is what used to raise IsADirectoryError.
        A :data:`SYNTOPIC` topic is synthetic, has no location, and is skipped.
        """
        topics = self.topics()
        if not topics:
            raise NotImplementedError(
                f"{self.__class__.__name__}.leave_breadcrumbs() requires TOPICS"
            )
        for leaf in self.leaftopics():
            if self._is_syntopic(*leaf):
                continue
            dirpath = self.dirpath(*leaf, ensure=True)
            node = self._topicnode(*leaf)
            crumbs = None if self._node_is_dirtopic(node) else node
            self.leave_breadcrumbs_at_path(dirpath, crumbs=crumbs)
        return self

    def _iter_var_blocks(self, exemptions_attr=None, skip_callback=None):
        """Yield (key, Datablock) pairs from self.var that are not in the given exemptions list."""
        exemptions = set(getattr(self, exemptions_attr, ())) if exemptions_attr else set()
        for s in self.spec.keys():
            if s in exemptions:
                if skip_callback:
                    skip_callback(s)
                continue
            c = getattr(self.var, s)
            if isinstance(c, Datablock):
                yield s, c

    def build_tree(self, *args, exclude_self: bool = False, deep: bool = False, **kwargs):
        self.log.verbose(f"Building tree for {self} with roots {self.spec.keys()}")
        def skip_cb(s):
            self.log.verbose(f"------------------------ SKIPPING SUBTREE at {s} (BUILD_TREE_EXEMPTIONS) --------")
        
        for s, c in self._iter_var_blocks('BUILD_TREE_EXEMPTIONS', skip_callback=skip_cb):
            if not deep and c.valid():
                self.log.verbose(f"------------------------ SKIPPING SUBTREE at {s}: already valid --------")
                continue
            self.write_journal_entry(event=f"build_tree:{s}:begin")
            self.log.verbose(f"------------------------ BUILDING SUBTREE at {s}: BEGIN --------------------------------")
            # A child built as part of this tree belongs to this run. VAR is
            # where it was constructed, which is too early to know that.
            self._adopt(c).build_tree(*args, deep=deep, **kwargs)
            self.log.verbose(f"------------------------ BUILDING SUBTREE at {s}: END --------------------------------")
            self.write_journal_entry(event=f"build_tree:{s}:end")
        if not exclude_self:
            self.build(*args, **kwargs)
        return self
    
    def valid_var(self, *, reduce=False):
        if not self.validate_vars:
            return True if reduce else {}
        results = {s: c.valid() for s, c in self._iter_var_blocks('TREE_SKIP_VALIDATION')}
        if reduce:
            return all(list(results.values())) if results else True
        else:
            return results

    def valid_tree(self):
        """Return a nested dictionary mapping var keys to their valid status and the valid status of their subtrees."""
        if not self.validate_vars:
            return {}
        return {
            s: {'valid': c.valid(), 'tree': c.valid_tree()}
            for s, c in self._iter_var_blocks('TREE_SKIP_VALIDATION')
        }
    
    def read(self, *topicpath):
        """Read a topic: ``read('out')``, or ``read('data', 'annotations')``.

        The path is validated against TOPICS first, so a mistyped name fails
        here naming the level it failed at, rather than inside ``__read__``.
        A single name is forwarded to ``__read__`` bare, which keeps every
        existing one-argument override working untouched.

        Nothing here knows about redirection: an override reads
        ``self.path(topic)``, and that is where a :meth:`UNSAFE_redirect` takes
        effect, so a redirected block reads the redirected-to data through the
        same override that reads its own.
        """
        topicpath = self._normtopic(topicpath)
        self._topicnode(*topicpath)      # raises KeyError if it does not exist
        if len(topicpath) == 1:
            return self.__read__(topicpath[0])
        return self.__read__(*topicpath)

    def __read__(self, *topicpath):
        raise NotImplementedError()

    #REDIRECT: BEGIN
    @dataclass
    class Redirection:
        """A resolved redirection: where this block's topics are read from instead,
        and what it was resolved from. `paths` is the nested {topic: path}
        mapping :meth:`path` answers out of; `entry` is the journal entry a
        filter matched, and is None for a redirection given paths directly.
        """
        paths: dict | None = None
        entry: Optional['DatajournalEntry'] = None
        filter: dict | None = None
        topic_map: dict | None = None

    @property
    def _redirected_paths_(self):
        """Active redirection paths for this block, loaded from .redirection/paths.yaml or journal."""
        if not getattr(self, 'redirect', False):
            return None
        if '__redirected_paths__' in self.__dict__:
            return self.__dict__['__redirected_paths__']
        red_dir = os.path.join(self.anchorkeypath, '.redirection')
        red_yaml = os.path.join(red_dir, 'paths.yaml')
        try:
            if self.fs.exists(red_yaml):
                paths = read_yaml(red_yaml, storage_options=self.storage_options)
                self.__dict__['__redirected_paths__'] = paths
                return paths
        except Exception as e:
            self.log.detailed(f"_redirected_paths_: could not read hidden topic .redirection: {e}")

        return None



    @_redirected_paths_.setter
    def _redirected_paths_(self, value):
        if value is None:
            self.__dict__.pop('__redirected_paths__', None)
        else:
            self.__dict__['__redirected_paths__'] = value

    @_redirected_paths_.deleter
    def _redirected_paths_(self):
        self.__dict__.pop('__redirected_paths__', None)

    _paths_ = _redirected_paths_

    @functools.cached_property
    def redirection(self):
        """Where this block reads from instead, as a :attr:`Redirection`, or None.

        An informational property describing how this block is redirected.
        """
        return self._get_redirection()

    def get_redirection(self, journal=None):
        """Where this block reads from instead, as a :attr:`Redirection`, or None.

        If *journal* is provided, use it directly to find matching redirection entries
        instead of reloading journal files from storage.
        """
        return self._get_redirection(journal=journal)

    def _get_redirection(self, journal=None):
        if not getattr(self, 'redirect', False):
            return None

        recorded = self._recorded_redirection(journal=journal)
        if recorded is None:
            if '__redirected_paths__' in self.__dict__ and self.__dict__['__redirected_paths__'] is not None:
                return self.Redirection(paths=self.__dict__['__redirected_paths__'], entry=None, filter=None, topic_map=None)
            red_dir = os.path.join(self.anchorkeypath, '.redirection')
            red_yaml = os.path.join(red_dir, 'paths.yaml')
            try:
                if self.fs.exists(red_yaml):
                    paths = read_yaml(red_yaml, storage_options=self.storage_options)
                    return self.Redirection(paths=paths, entry=None, filter=None, topic_map=None)
            except Exception:
                pass
            return None

        if isinstance(recorded, str):
            recorded = {'filter': {'id': recorded}}

        filter = recorded.get('filter')
        topic_map = recorded.get('topic_map')
        paths = recorded.get('paths')

        if paths is not None:
            self.log.verbose(
                f"REDIRECTION: {self.anchorkeypath} reads from the given paths instead: {paths}"
            )
            self.__dict__['__redirected_paths__'] = paths
            return self.Redirection(paths=paths, entry=None, filter=None, topic_map=None)

        if filter is None and isinstance(recorded, dict) and not ('filter' in recorded or 'paths' in recorded or 'topic_map' in recorded):
            self.log.verbose(
                f"REDIRECTION: {self.anchorkeypath} reads from the given paths instead: {recorded}"
            )
            self.__dict__['__redirected_paths__'] = recorded
            return self.Redirection(paths=recorded, entry=None, filter=None, topic_map=None)

        entry = self._redirect_entry(filter, journal=journal)
        if entry is None:
            self.log.warning(f"redirection: filter {filter!r} matches no journal entry")
            return None

        resolved_paths = self._mapped_paths(entry.block.paths(), topic_map)
        self.__dict__['__redirected_paths__'] = resolved_paths

        self.log.verbose(
            f"REDIRECTION: {self.anchorkeypath} reads from journal entry {entry.block.id} "
            f"instead (hash {entry.block.hash}, event {entry.get('event')!r}, written "
            f"{entry.get('datetime')}), matched by {filter!r}"
            + (f", topics mapped {topic_map!r}" if topic_map else "")
        )
        return self.Redirection(paths=resolved_paths, entry=entry, filter=filter, topic_map=topic_map)

    def redirected(self) -> bool:
        """True if this block is redirected (checks presence of hidden .redirection topic without reading the journal)."""
        if not getattr(self, 'redirect', False):
            return False
        if '__redirected_paths__' in self.__dict__ and self.__dict__['__redirected_paths__'] is not None:
            return True
        red_dir = os.path.join(self.anchorkeypath, '.redirection')
        red_yaml = os.path.join(red_dir, 'paths.yaml')
        try:
            return self.fs.exists(red_yaml) or self.fs.exists(red_dir)
        except Exception as e:
            self.log.detailed(f"redirected: error checking .redirection topic: {e}")
            return False


    def _mapped_paths(self, paths, topic_map):
        """*paths*, re-keyed by *topic_map* -- which reads mine -> theirs.

        Topics line up by name to begin with, as they would with no map at all;
        ``{'out': 'output'}`` then says this block's ``out`` is that entry's
        ``output``, so the entry's ``output`` path also comes back under ``out``.
        Every topic the map does not mention is untouched.

        A mapping whose target the other side does not have leaves its topic
        with NO redirected path, rather than falling back to the name it was
        told not to use: asking for ``theirs`` and silently getting ``mine`` is
        the one answer that is certainly wrong. That topic then reads as it
        would unredirected, and the mapping is reported.
        """
        if not paths:
            return {}
        native_topics = self.topics()
        if not native_topics:
            if not topic_map:
                return dict(paths)
            mapped = dict(paths)
            for mine, theirs in topic_map.items():
                if theirs in paths:
                    mapped[mine] = paths[theirs]
                else:
                    self.log.warning(
                        f"redirection: topic_map sends {mine!r} to {theirs!r}, which the "
                        f"redirected-to entry does not record; {mine!r} is left unredirected"
                    )
                    mapped.pop(mine, None)
            return mapped

        mapped = {}
        for topic in native_topics:
            source_topic = topic_map.get(topic, topic) if topic_map else topic
            if source_topic in paths:
                mapped[topic] = paths[source_topic]
            elif topic_map and topic in topic_map:
                self.log.warning(
                    f"redirection: topic_map sends {topic!r} to {source_topic!r}, which the "
                    f"redirected-to entry does not record; {topic!r} is left unredirected"
                )
        if topic_map:
            for mine, theirs in topic_map.items():
                if mine not in native_topics:
                    self.log.warning(
                        f"redirection: topic_map specifies {mine!r} -> {theirs!r}, but {mine!r} "
                        f"is not a topic of this block; ignoring it"
                    )
        return mapped

    def _redirect_path(self, *topicpath):
        """The path this block's *topicpath* is redirected to, or None.

        None whenever there is no redirection, or it records nothing for this
        topic -- which leaves :meth:`path` to answer with this block's own, so a
        partial redirection redirects only what it names.
        """
        if len(topicpath) > 0 and str(topicpath[0]).startswith('.'):
            return None
        paths = self._redirected_paths_
        if paths is None:
            return None
        if len(topicpath) == 0:
            return paths
        node = paths
        for name in topicpath:
            if not isinstance(node, dict) or name not in node:
                return None
            node = node[name]
        return node

    def _redirect_dirpath(self, *topicpath):
        """The directory of the path *topicpath* is redirected to, or None.

        A directory topic IS its path; a file topic's directory is the parent of
        the file it was redirected to -- which is what makes ``ls``, ``list``
        and ``size`` describe the redirected-to data rather than this block's
        empty one, since all three resolve through here.
        """
        redirected = self._redirect_path(*topicpath)
        if redirected is None:
            return None
        if isinstance(redirected, dict):
            # A group: it has no path of its own, and its members carry theirs.
            return None
        node = self._topicnode(*topicpath)
        if self._node_is_dirtopic(node):
            return redirected
        return os.path.dirname(redirected)

    def _journal_hashdirpath(self):
        """The directory holding THIS block's journal entries, and no others."""
        return os.path.join(self.anchorkeypath, ".journal", self.fqcn, "journal", self.hash)

    def _recorded_redirection(self, journal=None):
        """The latest redirection recorded for this block's hash, or None.

        Latest, because a redirection is a correction and the newest one is the
        one still meant.
        """
        if journal is not None and isinstance(journal, pd.DataFrame) and not journal.empty:
            if 'hash' in journal.columns and 'redirection' in journal.columns:
                sub = journal[journal['hash'] == self.hash]
                if not sub.empty:
                    latest = None
                    for _, row in sub.iterrows():
                        entry = DatajournalEntry(row, storage_options=self.storage_options)
                        when = row.get('datetime')
                        if entry.block.redirection is False or entry.get('event', '').startswith('UNSAFE_clear'):
                            if latest is None or str(when) > str(latest[0]):
                                latest = (when, None)
                        elif entry.block.redirection is not None:
                            if latest is None or str(when) > str(latest[0]):
                                red_val = entry.block.redirection
                                if isinstance(red_val, str) and not red_val.startswith('{'):
                                    latest = (when, {'filter': {'id': red_val}})
                                else:
                                    latest = (when, red_val)
                    if latest is not None:
                        return latest[1]
        try:
            dirpath = self._journal_hashdirpath()
            legacy_dirpath = os.path.join(
                Datablock._dbxanchorpathx(self._url_, self.anchor, 'journal',
                                          fqcn=self.fqcn, storage_options=self.storage_options),
                self.hash,
            )
            files = []
            if self.fs.exists(dirpath):
                files.extend(self.fs.glob(os.path.join(dirpath, '*.parquet')))
            if self.fs.exists(legacy_dirpath):
                files.extend(self.fs.glob(os.path.join(legacy_dirpath, '*.parquet')))
            if not files:
                return None
        except Exception as e:
            self.log.detailed(f"redirection: no journal directory to read: {e}")
            return None
        latest = None
        for file in files:
            try:
                with self.fs.open(file, 'rb') as f:
                    df = pd.read_parquet(f)
            except Exception as e:
                self.log.warning(f"redirection: skipping unreadable journal file {file}: {e}")
                continue
            if 'redirection' not in df.columns:
                continue
            for _, row in df.iterrows():
                entry = DatajournalEntry(row, storage_options=self.storage_options)
                when = row.get('datetime')
                if entry.block.redirection is False or entry.get('event', '').startswith('UNSAFE_clear'):
                    if latest is None or str(when) > str(latest[0]):
                        latest = (when, None)
                elif entry.block.redirection is not None:
                    if latest is None or str(when) > str(latest[0]):
                        red_val = entry.block.redirection
                        if isinstance(red_val, str) and not red_val.startswith('{'):
                            latest = (when, {'filter': {'id': red_val}})
                        else:
                            latest = (when, red_val)
        return latest[1] if latest is not None else None

    def _redirect_entry(self, filter, journal=None):
        """The FIRST journal entry matching *filter*, or None."""
        try:
            if journal is not None and isinstance(journal, pd.DataFrame) and not journal.empty:
                j = journal
                for k, v in filter.items():
                    if k in j.columns:
                        j = j[j[k] == v]
                    elif k == 'entry_code' and 'id' in j.columns:
                        j = j[j['id'] == v]
                    elif k == 'id' and 'entry_code' in j.columns:
                        j = j[j['entry_code'] == v]
                    else:
                        j = pd.DataFrame()
                        break
            else:
                j = self.journal(**dict(filter))
        except (KeyError, FileNotFoundError, TypeError) as e:
            self.log.warning(f"redirection filter {filter!r} is not usable: {e}")
            return None
        if len(j) == 0:
            return None
        if hasattr(j, 'get') and 0 in j.index:
            row = j.get(0, dropna=True)
            return DatajournalEntry(pd.Series(row).copy(deep=True), storage_options=self.storage_options)
        else:
            row = j.iloc[0]
            return DatajournalEntry(row.dropna(), storage_options=self.storage_options)

    def UNSAFE_redirect(self, *, redirector: Callable|None = None, journal: Datajournal|None = None, filter: dict|None = None, topic_map: dict|None = None,
                        paths: dict|None = None, validate: bool = False, remote: bool | Remote = False, OVERRIDE: bool = False):
        """Record that this block's topics are read from somewhere else and the location of this somewhere else."""
        if not UNSAFE_allowed("UNSAFE_redirect", OVERRIDE=OVERRIDE):
            return False

        explicit_journal = journal is not None
        if journal is None:
            try:
                journal = self.journal()
            except Exception:
                journal = None

        if redirector is not None:
            res = redirector(self, journal=journal)
            if isinstance(res, dict):
                filter = res.get('filter', filter)
                topic_map = res.get('topic_map', topic_map)
                paths = res.get('paths', paths)
                validate = res.get('validate', validate)
                remote = res.get('remote', remote)

        if filter is not None:
            if not isinstance(filter, dict) or not filter:
                raise ValueError(f"UNSAFE_redirect: filter must be a non-empty dict, got {filter!r}")
        if paths is not None:
            if not isinstance(paths, dict) or not paths:
                raise ValueError(f"UNSAFE_redirect: paths must be a non-empty dict, got {paths!r}")
        if topic_map is not None and not isinstance(topic_map, dict):
            raise ValueError(f"UNSAFE_redirect: topic_map must be a dict, got {topic_map!r}")

        entry = None
        target_paths = None
        redirect_record = None

        # If the redirector gave us paths directly, use those immediately.
        redirector_paths = paths if (redirector is not None and isinstance(res, dict) and 'paths' in res) else None
        if redirector_paths is not None:
            target_paths = redirector_paths
            redirect_record = redirector_paths

        if target_paths is None and filter is not None and journal is not None:
            try:
                j = Datajournal(journal, storage_options=getattr(journal, 'storage_options', self.storage_options), **dict(filter))
                if len(j) > 0:
                    entry = j.get(0, dropna=True) if hasattr(j, 'get') and 0 in j.index else DatajournalEntry(j.iloc[0].dropna(), storage_options=getattr(j, 'storage_options', self.storage_options))
                    target_paths = entry.block.paths()
                    if target_paths is None or not isinstance(target_paths, dict):
                        try:
                            target_paths = entry.inst(remote=remote).paths()
                        except Exception as e:
                            self.log.detailed(f"UNSAFE_redirect: entry.inst() failed: {e}")
                    redirect_record = entry.block.id
            except Exception as e:
                self.log.warning(f"UNSAFE_redirect: filter {filter!r} failed on journal: {e}")

        if target_paths is None and paths is not None:
            target_paths = paths
            redirect_record = paths

        if target_paths is None and filter is None and explicit_journal:
            try:
                j = journal if isinstance(journal, Datajournal) else Datajournal(journal, storage_options=self.storage_options)
                if len(j) > 0:
                    entry = j.get(0, dropna=True) if hasattr(j, 'get') and 0 in j.index else DatajournalEntry(j.iloc[0].dropna(), storage_options=getattr(j, 'storage_options', self.storage_options))
                    target_paths = entry.block.paths()
                    if target_paths is None or not isinstance(target_paths, dict):
                        try:
                            target_paths = entry.inst(remote=remote).paths()
                        except Exception as e:
                            self.log.detailed(f"UNSAFE_redirect: entry.inst() failed: {e}")
                    redirect_record = entry.block.id
            except Exception as e:
                self.log.warning(f"UNSAFE_redirect: failed reading journal: {e}")

        if target_paths is None:
            self.log.warning(f"UNSAFE_redirect: no redirection for hash {self.hash}")
            return False

        remapped_paths = self._mapped_paths(target_paths, topic_map)

        self._redirected_paths_ = remapped_paths
        self.__dict__.pop('redirection', None)

        try:
            red_dir = self.dirpath('.redirection', ensure=True)
            red_yaml = os.path.join(red_dir, 'paths.yaml')
            write_yaml(self._redirected_paths_, red_yaml, storage_options=self.storage_options)
        except Exception as e:
            self.log.detailed(f"UNSAFE_redirect: could not write hidden topic .redirection: {e}")

        code = self.write_journal_entry(
            event='UNSAFE_redirect',
            redirection=redirect_record,
            journal_prefix='redirect-',
        )

        if validate:
            val_res = self.validate()
            is_val = bool(all(val_res.values())) if isinstance(val_res, dict) else bool(val_res)
            if not is_val:
                self.log.warning(f"UNSAFE_redirect: block {self.hash} remains invalid after redirection")
                return False

        self.log.verbose(f"UNSAFE_redirect: {self.hash} -> {redirect_record!r} (entry {code})")
        return True
    #REDIRECT: END

    def UNSAFE_clear(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
        if not UNSAFE_allowed("UNSAFE_clear", OVERRIDE=OVERRIDE):
            return self
        
        def clear_path(path, *, recursive=False, throw=False):
            if path is None:
                return
            if not self.fs.exists(path):
                return
            self.log.verbose(f"removing {path}")
            try:
                if isinstance(path, str) and path.startswith("gs://"):
                    """
                    Circumvent bugs in fsspec.
                    """
                    from google.cloud import storage

                    client = storage.Client()
                    bits = path.removeprefix("gs://").split("/")
                    bucket_name = bits[0]
                    blob_name = "/".join(bits[1:])
                    bucket = client.get_bucket(bucket_name)
                    if recursive:
                        blobs = bucket.list_blobs(prefix=blob_name)
                        bucket.delete_blobs(blobs)
                    else:
                        blob = bucket.get_blob(blob_name)
                        blob.delete()
                else:
                    self.fs.rm(path, recursive=recursive)
            except (FileNotFoundError, os.error):
                pass
            except Exception as e:
                self.log.warning(f"Error when trying to remove {path}")
                self.log.warning(f"EXCEPTION: {e}")
                if throw:
                    raise (e)

        had_redirection = False
        try:
            red_dir = self.dirpath('.redirection')
            if self.fs.exists(red_dir):
                had_redirection = True
                self.log.info(f"UNSAFE_clear: removing .redirection for {self.anchorkeypath} (the underlying block being redirected to is not affected)")
                clear_path(red_dir, recursive=True)
        except Exception:
            pass

        self._redirected_paths_ = None
        # Invalidate @functools.cached_property cache on this instance after clearing.
        self.__dict__.pop('redirection', None)

        def clear_topic(topicpath):
            # A group names a directory but holds no data of its own; clearing
            # it means clearing what is under it.
            if clear_dirpath:
                clear_path(self.dirpath(*topicpath), recursive=True)
                return
            for leaf in self._leaves_under(*topicpath):
                clear_path(self.__path__(*leaf), recursive=self._is_dir_topic(*leaf))

        if len(topics) == 0:
            for topic in self.topics():
                clear_topic((topic,))
            self.write_journal_entry(event="UNSAFE_clear", redirection={'cleared': True})
        else:
            for topic in topics:
                clear_topic(self._normtopic((topic,)))
            self.write_journal_entry(event=f"UNSAFE_clear:{[topics]}", redirection={'cleared': True})

        msg = f"UNSAFE_clear: cleared block {self.hash}"
        if had_redirection:
            msg += " (redirection removed)"
        self.log.info(msg)
        return self

    def _UNSAFE_copy_fs(self, *, src_path, dst_path, recursive: bool = False):
        """Copy a single file or directory between two fsspec paths.

        fsspec does not implement a generic cross-filesystem ``.copy()``, so
        this dispatches to put/get directly when either side is local, and
        falls back to a temporary local directory when both are remote.
        """
        src_fs, _ = self._url_to_fs(src_path)
        dst_fs, _ = self._url_to_fs(dst_path)

        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            dst_fs.makedirs(dst_dir, exist_ok=True)

        if 'file' in src_fs.protocol or 'file' in dst_fs.protocol:
            # At least one is local filesystem, use put/get directly
            if 'file' in src_fs.protocol:
                # Source is local, destination is remote
                dst_fs.put(src_path, dst_path, recursive=recursive)
            else:
                # Source is remote, destination is local
                src_fs.get(src_path, dst_path, recursive=recursive)
        else:
            # Both are remote, use temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                basename = os.path.basename(src_path.rstrip('/'))
                if not basename:
                    basename = "root"
                tmp_path = os.path.join(tmpdir, basename)
                src_fs.get(src_path, tmp_path, recursive=recursive)
                dst_fs.put(tmp_path, dst_path, recursive=recursive)

    def _UNSAFE_copy_file(self, src_path, dst_path):
        """Copy a single file, preferring a fast server-side blob copy.

        When *src_path* and *dst_path* resolve to the same non-local
        filesystem (e.g. two Azure blob paths in the same account), this
        does a direct blob-to-blob copy with no data transiting the local
        machine. Otherwise it falls back to :meth:`_UNSAFE_copy_fs`
        (get+put, possibly via a local temporary directory), which is the
        only option when a real cross-filesystem or local-disk hop is
        required. Mirrors the ``use_server_side`` branch already used by
        :meth:`_UNSAFE_copy_topic_dir` for whole-directory copies, exposed
        here for single-file copies (e.g. a subclass copying a subset of a
        topic's files instead of the whole directory).
        """
        src_fs, _ = self._url_to_fs(src_path)
        dst_fs, _ = self._url_to_fs(dst_path)
        if src_fs == dst_fs and 'file' not in getattr(src_fs, 'protocol', ()):
            dst_dir = os.path.dirname(dst_path)
            if dst_dir:
                dst_fs.makedirs(dst_dir, exist_ok=True)
            dst_fs.cp_file(src_path, dst_path)
        else:
            self._UNSAFE_copy_fs(src_path=src_path, dst_path=dst_path, recursive=False)

    @staticmethod
    def _topicpaths_lookup(topicpaths, topicpath):
        """Find *topicpath* in a caller-supplied override map.

        Accepts the flat form keyed by name, the tuple-keyed form, and a
        mapping nested to match TOPICS, so callers can express an override at
        whatever depth is convenient.
        """
        if tuple(topicpath) in topicpaths:
            return topicpaths[tuple(topicpath)]
        node = topicpaths
        for name in topicpath:
            if not isinstance(node, dict) or name not in node:
                raise KeyError(
                    f"topicpaths has no entry for {'/'.join(topicpath)!r}"
                )
            node = node[name]
        return node

    def _UNSAFE_copy_topic_file(self, topic, anchorkeypath, *, topicpaths=None):
        """Copy the individual .path(topic) file."""
        topic = self._normtopic((topic,))
        dst_path = self.path(*topic)
        if topicpaths is not None:
            _src_path = self._topicpaths_lookup(topicpaths, topic)
        else:
            if self._topicfiles is None:
                raise ValueError(
                    f"Cannot copy topic file for {'/'.join(topic)!r}: TOPICS is not a dict "
                    f"(no filename mapping). Use always_copy_whole_dirpath=True for list-mode topics."
                )
            _src_path = os.path.join(*topic, self._topicnode(*topic))
        if dst_path is not None:
            src_path = os.path.join(anchorkeypath, _src_path)
            self.log.detailed(f"Copying file {src_path} to {dst_path}")
            self._UNSAFE_copy_fs(src_path=src_path, dst_path=dst_path, recursive=False)

    def _UNSAFE_copy_topic_dir(self, topic, anchorkeypath, *, topicpaths=None):
        """Copy the entire .dirpath(topic) directory."""
        topic = self._normtopic((topic,))
        if topicpaths is not None:
            _src_path = self._topicpaths_lookup(topicpaths, topic)
        else:
            _src_path = os.path.join(*topic)
        src_path = os.path.join(anchorkeypath, _src_path)
        src_fs, _ = self._url_to_fs(src_path)
        dest_fs, _ = self._url_to_fs(self.dirpath(*topic))
        use_server_side = (src_fs == dest_fs
                           and 'file' not in getattr(src_fs, 'protocol', ()))
        # Ensure dst dir pre-exists only for server-side copy (Azure) or list-mode topics.
        # For dict-mode fscopy, pre-creating dst causes fsspec to copy src INTO dst instead of AS dst.
        ensure = use_server_side or (self._topicfiles is None)
        dst_path = self.dirpath(*topic, ensure=ensure)
        if not src_fs.exists(src_path):
            return
        self.log.detailed(f"Copying directory {src_path} to {dst_path}")
        if use_server_side:
            # fsspec's generic recursive _copy() (which .cp(recursive=True)
            # delegates to for remote-to-remote copies) expands the source
            # directory to include the bare directory itself alongside its
            # file contents, then blindly _cp_file()s every entry. Azure's
            # blob-to-blob "copy from URL" API has no notion of copying a
            # directory, so that entry always raises InvalidInput -- even
            # though the real files copy fine. Expand to files only
            # (find(..., withdirs=False)) and cp_file() each one ourselves.
            src_bare = src_fs._strip_protocol(src_path)
            dst_bare = dest_fs._strip_protocol(dst_path)
            for file_path in src_fs.find(src_bare, withdirs=False):
                rel = file_path[len(src_bare):].lstrip('/')
                self.fs.cp_file(file_path, os.path.join(dst_bare, rel))
        elif getattr(self, 'parallelization', None):
            # Parallelize on top-level directory contents; each item is copied independently.
            # _fscopy_item_callable is a module-level function (picklable for multiprocessing).
            items = [src_fs.unstrip_protocol(p)
                     for p in src_fs.ls(src_path, detail=False)]
            if items:
                _storage_options = getattr(self, 'storage_options', {})
                _n_workers = getattr(self, 'n_workers', 1)
                _tag = f"fscopy {len(items)} items [{_src_path}]"
                _executor_kwargs = dict(n_workers=_n_workers, tag=_tag)
                if (hasattr(self, 'multiprocessing_start_method')
                        and self.multiprocessing_start_method is not None
                        and (self.parallelization or '').lower() in ('multiprocessing', 'torch_multiprocessing')):
                    _executor_kwargs['start_method'] = self.multiprocessing_start_method
                _executor = callable_executor(self.parallelization, **_executor_kwargs)
                _callables = [
                    functools.partial(
                        _fscopy_item_callable,
                        src_item,
                        dst_path.rstrip('/') + '/' + src_item.rstrip('/').rsplit('/', 1)[-1],
                        _storage_options,
                    )
                    for src_item in items
                ]
                _executor.exec_callables(_callables)
        else:
            self._UNSAFE_copy_fs(src_path=src_path, dst_path=dst_path, recursive=True)

    def _UNSAFE_copy_topic(self, topic, anchorkeypath, *, topicpaths=None, always_copy_whole_dirpath: bool = False, **kwargs):
        """Copy one topic's data from anchorkeypath into this Datablock.

        Dispatches to :meth:`_UNSAFE_copy_topic_dir` or
        :meth:`_UNSAFE_copy_topic_file` depending on TOPICS shape. Overriding
        this in a subclass is the extension point for customizing how a
        *specific* topic gets copied (e.g. copying only a subset of files
        instead of the whole directory) while leaving the rest of
        :meth:`UNSAFE_copy_from` (overwrite check, journal entries,
        post-copy validation) untouched -- see
        ``IJEPAsaurUSStill._UNSAFE_copy_topic`` in soundworld for an example
        that restricts the ``ckpts`` topic to a subset of checkpoints.
        ``**kwargs`` is accepted (and ignored here) so subclasses can declare
        extra keyword-only parameters on their override without changing
        this base signature; :meth:`UNSAFE_copy_from` forwards its own
        ``**kwargs`` to every topic's call.
        """
        topic = self._normtopic((topic,))
        if self.is_topicgroup(*topic):
            for leaf in self._leaves_under(*topic):
                self._UNSAFE_copy_topic(leaf, anchorkeypath, topicpaths=topicpaths,
                                        always_copy_whole_dirpath=always_copy_whole_dirpath,
                                        **kwargs)
            return
        if self._is_syntopic(*topic):
            # No location on either side -- there is nothing to copy.
            self.log.verbose(f"Skipping SYNTOPIC topic {'/'.join(topic)}: it has no location")
            return
        # Use directory copy when:
        #  - always_copy_whole_dirpath is explicitly requested, OR
        #  - TOPICS is a list (self._topicfiles is None -> every topic IS a dir), OR
        #  - TOPICS is a dict but this topic maps to DIRTOPIC (directory-only topic)
        use_dir = (
            always_copy_whole_dirpath
            or self._topicfiles is None
            or (isinstance(self._topicfiles, dict) and self._is_dir_topic(*topic))
        )
        if use_dir:
            self.log.verbose(f"Using copy_topic_dir for topic {topic}: BEGIN")
            self._UNSAFE_copy_topic_dir(topic, anchorkeypath, topicpaths=topicpaths)
            self.log.verbose(f"Using copy_topic_dir for topic {topic}: END")
        else:
            self.log.verbose(f"Using copy_topic_file for topic {topic}: BEGIN")
            self._UNSAFE_copy_topic_file(topic, anchorkeypath, topicpaths=topicpaths)
            self.log.verbose(f"Using copy_topic_file for topic {topic}: END")

    def UNSAFE_copy_from(self, anchorkeypath, *, OVERRIDE: bool = False, overwrite: bool = False, topicpaths=None, validate: bool = True, always_copy_whole_dirpath: bool = False, show_progress: bool = True, **kwargs):
        """Copy topic data from an external directory into this Datablock.

        Parameters
        ----------
        anchorkeypath : str
            Filesystem path to the source anchor+key directory containing
            the topic subdirectories (e.g. ``ckpts/``, ``logs/``).
        OVERRIDE : bool, default False
            If True, skip the interactive confirmation prompt (see
            :func:`UNSAFE_allowed`) -- same convention as
            :meth:`UNSAFE_clear`/:meth:`UNSAFE_copy_blocks_from`.
        overwrite : bool, default False
            If False (default), asserts that this Datablock is not already
            valid before copying.  Set to True to overwrite existing data.
        topicpaths : dict or str, optional
            Override the default source-relative paths for each topic.
            For dict TOPICS: a ``{topic: relative_path}`` dict.
            For string TOPICS: a single relative path string.
            When None, source paths are derived from the Datablock's own
            TOPICS definitions.
        validate : bool, default True
            If True, asserts that ``self.valid()`` returns True after
            the copy completes.  Set to False to skip post-copy validation.
        always_copy_whole_dirpath : bool, default False
            If False (default), copies individual topic files via
            ``self.path(topic)``.  If True, copies entire topic
            directories via ``self.dirpath(topic)`` recursively.
        show_progress : bool, default True
            If True (default), show a per-topic tqdm progress bar. Set to
            False when a caller (e.g. :meth:`UNSAFE_copy_blocks_from`) is
            already reporting aggregate progress across many blocks, so
            each block's own (typically 1-topic, so always instantly
            "100%") bar doesn't flood the output.
        **kwargs
            Forwarded to :meth:`_UNSAFE_copy_topic` for every topic; ignored
            by the base implementation but available to subclasses that
            override :meth:`_UNSAFE_copy_topic` to accept additional
            per-topic options.
        """
        if not UNSAFE_allowed("UNSAFE_copy_from", OVERRIDE=OVERRIDE):
            return self
        if not overwrite:
            assert not self.valid(), f"Attempting to overwrite a valid Datablock {self}. Missing 'overwrite' argument?"
        fs, _ = self._url_to_fs(anchorkeypath)
        assert fs.isdir(anchorkeypath), f"Nonexistent hashpath {anchorkeypath}"
        self.log.verbose(f"Copying files from {anchorkeypath}: BEGIN")
        self.write_journal_entry(event="UNSAFE_copy_from:BEGIN", note=anchorkeypath, inline_note=True)
        try:
            topics = self.topics()
            if not topics:
                raise NotImplementedError(
                    f"{self.__class__.__name__}.UNSAFE_copy_from() requires TOPICS"
                )
            topics_iter = tqdm.tqdm(topics, desc="UNSAFE_copy_from", unit="topic") if show_progress else topics
            for topic in topics_iter:
                self._UNSAFE_copy_topic(
                    topic, anchorkeypath, topicpaths=topicpaths,
                    always_copy_whole_dirpath=always_copy_whole_dirpath, **kwargs,
                )

            self.log.verbose(f"Copying files from {anchorkeypath}: END")
            self.write_journal_entry(event="UNSAFE_copy_from:END", note=anchorkeypath, inline_note=True)
            if validate:
                assert self.validate(), f"Invalid Datablock after copy: {self}"
        except Exception as e:
            self.log.error(f"UNSAFE_copy_from: Error when trying to copy files from {anchorkeypath}")
            self.log.error(f"EXCEPTION: {e}")
            self.write_journal_entry(event="UNSAFE_copy_from:ERROR", note=anchorkeypath, inline_note=True)
            raise e
        return self

    def UNSAFE_copy_from_journal(self, journal: dict, *, OVERRIDE: bool = False, overwrite: bool = False, topicpaths=None, validate: bool = True, always_copy_whole_dirpath: bool = False, show_progress: bool = True, **kwargs):
        """Copy topic data using the ``anchorkeypath`` recorded in a journal entry.

        Thin wrapper around :meth:`UNSAFE_copy_from`: it extracts a single
        journal entry (via :meth:`journal`) and forwards that entry's
        ``anchorkeypath`` as the copy source.

        Parameters
        ----------
        journal : dict
            Keyword arguments passed to :meth:`journal` to select the entry
            whose ``anchorkeypath`` is used as the copy source, e.g.
            ``{'iloc': 0}``, ``{'loc': 3}``, or filter kwargs like
            ``{'event': 'build:end'}``. Must resolve to a single
            :class:`DatajournalEntry`.
        OVERRIDE : bool, default False
            If True, skip the interactive confirmation prompt. Forwarded to
            :meth:`UNSAFE_copy_from`.

        All remaining keyword arguments (including ``**kwargs``) are
        forwarded to :meth:`UNSAFE_copy_from`.
        """
        entry = self.journal(**journal)
        return self.UNSAFE_copy_from(
            entry.block.anchorkeypath,
            OVERRIDE=OVERRIDE,
            overwrite=overwrite,
            topicpaths=topicpaths,
            validate=validate,
            always_copy_whole_dirpath=always_copy_whole_dirpath,
            show_progress=show_progress,
            **kwargs,
        )

    def _spec_to_var(self, spec):
        var = self.VAR(**spec)
        replacements = {}
        for field in fields(var):
            term = getattr(var, field.name)
            if issubclass(self.VAR, Datablock.VAR):
                getter = Datablock.VAR.LazyLoader(term)
            else:
                getter = eval(term)
            replacements[field.name] = getter
        var = replace(var, **replacements)
        self.log.detailed(f"Made {var=} from {spec=}")
        return var

    def leave_breadcrumbs_at_path(self, path, crumbs=None):
        """Bring a breadcrumb file into existence for the directory at *path*.

        *path* is ALWAYS a directory path -- never a file path.  With *crumbs*
        the breadcrumb is that named file inside it (``{path}/{crumbs}``);
        without, it is ``{path}.crumbs`` alongside it, since a directory topic
        has no filename to use.

        Existing content is never clobbered: a breadcrumb is only touched when
        nothing is there, which is the least this can do and still leave a mark.

        Returns the breadcrumb path.
        """
        if crumbs is not None:
            crumbpath = f"{path}/{crumbs}"
            ensure_path(path, storage_options=self.storage_options)
        else:
            crumbpath = f"{path}.crumbs"
        if not self.fs.exists(crumbpath):
            self.fs.touch(crumbpath)
        self.log.detailed(f"{self.anchor}: breadcrumb: {crumbpath}")
        return crumbpath
    
    #IDS: BEGIN
    #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
    # computed using the older version of these methods
    
    @staticmethod
    def is_specline(s):
        return isinstance(s, str) and (
            s.startswith('@') or s.startswith('$') or s.startswith('#')
        )
    
    @property
    def fqcn(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}"
    
    @property
    def version(self):
        """User-defined version of this Datablock subclass. Used in hash computation — do NOT include dbx version here."""
        return self.VERSION if hasattr(self, 'VERSION') else None

    @property
    def dbx_version(self):
        """The dbx library version. Recorded in the journal but NOT used in hash computation."""
        return __version__

    @property
    def session(self):
        """The run this block belongs to: given, or generated once and kept.

        One live instance keeps one session for as long as it is alive, and a
        whole build tree shares it, because `build_tree` and `Datastack.block`
        hand it down. That is what makes a run's journal entries findable
        together -- ``id`` identifies one row and ``hash`` one block, but
        neither says "these were written by the same run".

        Not part of :attr:`signature`, so which run built a block cannot
        change what the block IS.
        """
        if getattr(self, '_session_', None) is None:
            self._session_ = (uuid.uuid4().hex[:16]
                              if getattr(self, '_uuid16_', False) else str(uuid.uuid4()))
        return self._session_

    def _adopt(self, child, *, keyby: bool = False):
        """Hand *child* what it should inherit from this block.

        Called by the framework AFTER the user's hook returns, so a subclass
        that only overrides ``__block__`` never has to think about it.
        """
        kw = {'session': self.session}
        if keyby:
            keyby_val = getattr(self, 'keyby', None)
            if keyby_val is not None:
                kw['keyby'] = keyby_val
        return child.set(**kw)
    
    @property
    def revision(self):
        if not hasattr(self, '_revision'):
            self.log.detailed(f"--------------> COMPUTING revision")
            if self._revision_ is None:
                self.log.detailed(f"--------------> self._revision_ is None")
                gitrepo = (dataparts.DBX_USE_WORK_REPO
                           if dataparts.DBX_USE_WORK_REPO is not None
                           else dataparts.DBX_GIT_REPO)
                self._revision = gitrevision(log=self.log) if gitrepo is not None else None
                self.log.detailed(f"--------------> self._revision_: from gitrevision()")
            else:
                self.log.detailed(f"--------------> Using {self._revision_=}")
                self._revision = self._revision_
        return self._revision

    def __expand_spec__(self, expansion='repr', *, legacy: 'bool | None' = None,
                        legacy_typing: 'bool | None' = None):
        """
            . legacy: override LEGACY_NORM or LEGACY_SIGNATURE for the 'subsignature' expansion.
                None (default) = each block uses its own flag, i.e. the
                identity-bearing rendering. True/False forces the legacy or the
                modern form, and PROPAGATES to nested blocks, so the whole
                subtree is rendered the same way.

            . expansion: 'repr'|'quote'|'signature'
                . specline:      str starting with '@', '$' or '#'
                . datablock: Datablock object
                . obj:       object
            'repr':
                . FULL reduction
                    |obj:    repr(obj)
            'signature':
                . DATABLOCK reduction
                    |datablock: datablock.signature()
                    |specline:      repr(specline)
                    |obj:       repr(obj)
            'quote':
                . UNREDUCED spec:
                    |specline:      repr(specline)
                    |datablock: datablock.quote()
                    |obj:       repr(obj)  
        """
        legacy = self._legacy_norm() if legacy is None else legacy
        # The TYPING choice a nested child must inherit. Separate from *legacy*
        # above, which is the norm flag: signature(legacy=...) selects typing,
        # so passing the norm flag down there rendered children the new way
        # inside a parent pinned to the old one.
        legacy_typing = self._legacy_typing(legacy_typing)
        if legacy:
            keys = [field.name for field in self.VAR.__dataclass_fields__.values()]
        else:
            keys = sorted([field.name for field in self.VAR.__dataclass_fields__.values()])
        spec = {k: self.spec[k] if k in self.spec else getattr(self.var, k) for k in keys}
        _spec_ = {}
        if expansion == 'repr':
            #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
            # computed using the older version of these methods
            for k, v in spec.items():
                value = getattr(self.var, k)
                _spec_[k] = repr(value)
        elif expansion in ('signature', 'subsignature'):
            for k, v in spec.items():
                raw_v = self.spec[k] if (isinstance(getattr(self, 'spec', None), dict) and k in self.spec) else v
                if not self.is_specline(raw_v) and hasattr(self, 'var') and hasattr(self.var, '__dict__'):
                    for var_val in self.var.__dict__.values():
                        if isinstance(var_val, Datablock) and isinstance(getattr(var_val, 'spec', None), dict) and k in var_val.spec:
                            cand = var_val.spec[k]
                            if self.is_specline(cand):
                                raw_v = cand
                                break
                value = getattr(self.var, k, None)
                if self.is_specline(raw_v):
                    try:
                        eval_v = dataparts.eval(raw_v)
                        if isinstance(eval_v, Datablock):
                            _spec_[k] = eval_v.signature(
                                legacy_typing=legacy_typing, legacy_signature=legacy)
                        else:
                            _spec_[k] = raw_v
                    except Exception:
                        _spec_[k] = raw_v
                elif isinstance(value, Datablock):
                    _spec_[k] = value.signature(
                        legacy_typing=legacy_typing, legacy_signature=legacy)
                elif isinstance(value, str):
                    _spec_[k] = value
                elif legacy:
                    _spec_[k] = str(value)
                else:
                    # Stored as the value itself, so the embedding repr's it
                    # exactly once: 5 -> "5", '5' -> "'5'". No collision.
                    _spec_[k] = value
        elif expansion == 'quote' or expansion == 'cite':
            for k, v in spec.items():
                value = getattr(self.var, k)
                raw_v = self.spec[k] if (isinstance(getattr(self, 'spec', None), dict) and k in self.spec) else v
                if not self.is_specline(raw_v) and hasattr(self, 'var') and hasattr(self.var, '__dict__'):
                    for var_val in self.var.__dict__.values():
                        if isinstance(var_val, Datablock) and isinstance(getattr(var_val, 'spec', None), dict) and k in var_val.spec:
                            cand = var_val.spec[k]
                            if self.is_specline(cand):
                                raw_v = cand
                                break
                if self.is_specline(raw_v):
                    _spec_[k] = raw_v
                elif isinstance(value, Datablock):
                    # Nested blocks are ALWAYS single-line and un-deslashed,
                    # whatever the user-facing defaults are. A nested specline
                    # is stored as a string VALUE in this spec, so the outer
                    # repr() escapes any newline it contains to backslash-n --
                    # and `deslash` then strips the backslash, leaving a bare
                    # "n" and an unevaluable specline. `pretty` and `deslash`
                    # are presentation options for the OUTERMOST call only.
                    if expansion == 'quote':
                        _spec_[k] = value.quote(pretty=False, deslash=0)
                    else:
                        _spec_[k] = value.cite(pretty=False, deslash=0)
                else:
                    # The value itself, whatever its type, so the embedding
                    # repr's it exactly once. Strings had their own arm doing
                    # exactly this, which read as a special case and was not.
                    _spec_[k] = value
        else:
            raise ValueError(f"Unknown expansion: {repr(expansion)}")
        return _spec_
    
    @functools.cached_property
    def _rootkwargs_(self):
        rootkwargs = {}
        if self.url is not None:
            rootkwargs['url'] = self.url
        if self._anchor_ is not None:
            rootkwargs['anchor'] = self._anchor_
        return rootkwargs
    
    @functools.cached_property
    def _tailkwargs_(self):
        state = self.__getstate__()
        tailkwargs = {
            k: v
            for k, v in state.items()
            # 'session' groups a run's journal entries; pinning one into a
            # recorded quote would have inst() rejoin a run that is over.
            if k not in ['url', 'anchor', 'hash', 'spec', 'session']
        }
        self.log.detailed(f"{self.anchor}: _tailkwargs_: {tailkwargs=}")
        return tailkwargs
    
    def __repr_from_kwargs__(self, kwargs, anchor='anchor', *, quote_strs: bool = False):
        """Render ``kwargs`` as a ``k=v, ...`` argument list.

        ``quote_strs`` reprs string-valued kwargs, so ``url=abfss://x`` becomes
        ``url='abfss://x'``. It is named for what it does rather than for
        `rootkwargs`, because this method is called with the root kwargs, the
        `spec` dict AND the tailkwargs, and it quotes strings in all of them
        (``tag``, ``local``, ``keyby``, ... as well as ``url``/``anchor``).

        """
        def quotestr(v):
            return repr(v) if quote_strs and isinstance(v, str) else v
        kwargstrs = [f"{k}={quotestr(v)}" for k, v in kwargs.items()]
        kwargsrepr = ', '.join(kwargstrs)
        if anchor == 'anchor':
            _repr_ = f"{self.anchor}({kwargsrepr})"
        elif anchor == 'fqcn':
            _repr_ = f"{self.fqcn}({kwargsrepr})"
        elif anchor is None:
            _repr_ = f"({kwargsrepr})"
        else:
            raise ValueError(f"Unknown anchor: {repr(anchor)}")
        return _repr_
    
    @staticmethod
    def _split_top_level(text, seps=(', ',)):
        """Split ``text`` at ``seps`` occurrences that are NOT nested or quoted.

        Depth-aware over ``() [] {}`` and quote-aware over ``' "`` (honouring
        backslash escapes), so a separator inside a nested specline or a URL is
        left alone. Used only to choose where to break a citation for display;
        the pieces are re-joined verbatim, so a missed boundary costs
        readability, never correctness.
        """
        out, buf, depth, quote, esc = [], [], 0, None, False
        i = 0
        while i < len(text):
            c = text[i]
            if esc:
                buf.append(c); esc = False; i += 1; continue
            if c == '\\':
                buf.append(c); esc = True; i += 1; continue
            if quote is not None:
                buf.append(c)
                if c == quote:
                    quote = None
                i += 1
                continue
            if c in '\'"':
                quote = c; buf.append(c); i += 1; continue
            if c in '([{':
                depth += 1; buf.append(c); i += 1; continue
            if c in ')]}':
                depth -= 1; buf.append(c); i += 1; continue
            if depth == 0:
                hit = next((sp for sp in seps if text.startswith(sp, i)), None)
                if hit is not None:
                    buf.append(hit)
                    out.append(''.join(buf)); buf = []
                    i += len(hit)
                    continue
            buf.append(c); i += 1
        if buf:
            out.append(''.join(buf))
        return out

    def _cite_chunks(self, specline, indent):
        """Render a nested specline as indented implicit-concatenation chunks.

        A nested block lives in the spec as a STRING, so putting real newlines
        in it only makes the outer repr() show ``\\n`` escapes -- unreadable in a
        different way from one 4000-character line. Python's implicit string
        concatenation escapes that bind: ``('a' 'b')`` is one string, so the
        pieces can sit on their own indented lines and still evaluate to
        exactly the original specline. Correctness is structural here -- the
        chunks are ``repr``-ed and concatenated verbatim, so only the choice of
        break points is a judgement call.
        """
        # Break after the opening "$fqcn(", then at each top-level kwarg, then
        # inside spec={...} at each entry.
        head, _, rest = specline.partition('(')
        pieces = [head + '(']
        for kw in self._split_top_level(rest):
            if kw.startswith('spec={'):
                inner = kw[len('spec={'):]
                closing = inner.rfind('}')
                entries = inner[:closing] if closing != -1 else inner
                tail = inner[closing:] if closing != -1 else ''
                pieces.append('spec={')
                pieces.extend(self._split_top_level(entries))
                pieces.append(tail)
            else:
                pieces.append(kw)
        pieces = [p for p in pieces if p]
        body = f"\n{indent}".join(repr(p) for p in pieces)
        return f"(\n{indent}{body}\n{indent})"

    # Tailkwargs that quote()/cite() keep when `tailkwargs=False` (the default).
    #
    # The other ~25 tailkwargs are purely operational -- log verbosity, worker
    # counts, cache limits, start methods, timeouts -- and none of them change
    # what the block IS, so they are noise in a citation and they dominate it.
    #
    # `tag` is the one that cannot be dropped silently. It is NOT part of the
    # identity hash (subsignature() is built from _rootkwargs_ + spec), but
    # keyby='tag_version_shorthash' puts it in the artifact PATH, so a citation
    # without it re-evaluates to the same hash at a DIFFERENT key -- i.e. it
    # points at storage that does not hold the artifact you cited.
    CITE_KEEP_TAILKWARGS = ('tag',)

    def quote(self, *, deslash: int = 0, cite: bool = False, pretty: bool = False,
              tailkwargs: bool = True):
        """Return an evaluable ``$fqcn(...)`` specline for this block.

        ``tailkwargs=True`` (the default HERE) keeps every operational kwarg,
        because quote() is the **evaluable** form and those kwargs are part of
        reconstructing a working block -- ``local`` in particular decides where
        local artifacts are staged, so dropping it sends ``find_latest_ckpt``
        to a different directory even though the hash and key still match.
        :meth:`cite` defaults the other way: it is presentation-only, so it
        shows just ``CITE_KEEP_TAILKWARGS`` (``tag``) and omits the rest as
        noise.

        ``pretty=True`` wraps one kwarg per line. It stays OFF by default
        because the result is a specline that gets ``eval``-ed on the way back
        in (see ``dataparts.eval``), so formatting must be opt-in rather than
        silently changing what every caller emits.
        """
        mode = 'quote' if not cite else 'cite'
        quoted_spec = self.__expand_spec__(mode)
        def quotestr(x):
            return repr(x) if isinstance(x, str) else x
        kwargs = {**self._rootkwargs_, **{'spec': quoted_spec},}
        if tailkwargs:
            kwargs.update(**self._tailkwargs_)
        else:
            kwargs.update({
                k: v for k, v in self._tailkwargs_.items()
                if k in self.CITE_KEEP_TAILKWARGS
            })
        kwargstrs = [f"{k}={quotestr(v)}" for k, v in kwargs.items()]
        if pretty:
            # A FIXED 4-space indent, and the spec dict broken one entry per
            # line -- the two things that made the previous attempt unreadable:
            #
            #  * Aligning the indent to len("$fully.qualified.ClassName(")
            #    is 40-60 columns for these classes, so every continuation line
            #    began with a huge run of spaces. That is the "weird trailing
            #    whitespace": a lone line of blanks once anything re-wraps it.
            #  * Splitting only the top-level kwargs leaves the entire VAR
            #    on one enormous `spec={...}` line, which is exactly the part
            #    you wanted to read -- hence "no indentation".
            #
            # Do NOT pformat the joined string: pformat(str) returns that
            # string's *repr*, which turns the whole argument list into one
            # quoted positional ("takes 1 positional argument but 2 were
            # given"). Nested blocks are quoted NON-pretty (the defaults
            # above), because a nested specline is stored as a string value and
            # the outer repr would escape its newlines to backslash-n -- which
            # `deslash` then strips to a bare "n", corrupting the specline.
            IND = '    '
            parts = []
            for k, v in kwargs.items():
                if k == 'spec' and isinstance(v, dict):
                    rows = [f"{IND * 2}{sk!r}: {repr(sv)},\n" for sk, sv in v.items()]
                    parts.append(f"{IND}spec={{\n{''.join(rows)}{IND}}}")
                else:
                    parts.append(f"{IND}{k}={quotestr(v)}")
            quote = f"{self.fqcn}(\n" + ",\n".join(parts) + ",\n)"
        else:
            quote = f"{self.fqcn}({', '.join(kwargstrs)})"
        if deslash != 0:
            for i in range(deslash):
                quote = quote.replace('\\', '')
        if not cite:
            quote = f"${quote}"
        self.log.detailed(f"quote: ------------> {quoted_spec=}")
        self.log.detailed(f"quote: ------------> {quote=}")
        return quote

    def cite(self, *, deslash: int = 2, pretty: bool = True,
             tailkwargs: bool = False, _indent: str = ''):
        """Human-readable rendering of the block graph. **Presentation only.**

        Deliberately NOT evaluable -- :meth:`quote` is the evaluable form. That
        distinction is what makes this readable at any depth: because the output
        never has to survive ``eval``, a nested block is emitted as a real
        indented block rather than as a quoted specline string.

        The difference matters most where it used to hurt. Representing a child
        as a string means the parent's ``repr`` escapes it, and a
        grandchild ends up inside a string inside a string -- doubling
        backslashes at every level until the deep entries are unreadable no
        matter how they are wrapped. Recursing over the *object graph* instead
        removes the quoting entirely, so depth costs nothing but indentation.

        ``tailkwargs=False`` (default) shows only :attr:`CITE_KEEP_TAILKWARGS`.
        ``deslash`` is retained for compatibility and applied last; it is
        normally a no-op here, since the recursive form emits no escaped
        strings to begin with.
        """
        IND = '    '
        inner = _indent + IND
        lines = [f"${self.fqcn}("]
        for k, v in self._rootkwargs_.items():
            lines.append(f"{inner}{k}={v!r},")

        lines.append(f"{inner}spec={{")
        for sk in sorted(self.VAR.__dataclass_fields__):
            raw_v = self.spec[sk] if (isinstance(getattr(self, 'spec', None), dict) and sk in self.spec) else None
            val = getattr(self.var, sk)
            if isinstance(val, Datablock):
                rendered = val.cite(
                    deslash=0, pretty=pretty, tailkwargs=tailkwargs,
                    _indent=inner + IND,
                )
                lines.append(f"{inner}{IND}{sk!r}: {rendered},")
            elif self.is_specline(raw_v):
                lines.append(f"{inner}{IND}{sk!r}: {raw_v!r},")
            else:
                lines.append(f"{inner}{IND}{sk!r}: {val!r},")
        lines.append(f"{inner}}},")

        tail = (self._tailkwargs_ if tailkwargs else
                {k: v for k, v in self._tailkwargs_.items()
                 if k in self.CITE_KEEP_TAILKWARGS})
        for k, v in tail.items():
            lines.append(f"{inner}{k}={v!r},")
        lines.append(f"{_indent})")

        cite = '\n'.join(lines)
        if not pretty:
            cite = ' '.join(l.strip() for l in lines)
        for _ in range(max(deslash, 0)):
            cite = cite.replace('\\', '')
        self.log.detailed(f"cite: ------------> {cite=}")
        return cite

    def signature(self, *, deslash: bool = False, legacy: bool | None = None,
                  legacy_typing: bool | None = None,
                  legacy_signature: bool | None = None, pretty: bool = False):
        """The base identity string that :attr:`type` -- and hence :attr:`hash` and :attr:`code` -- is built from.

        Two independent opt-outs, because they were two different things
        sharing one name:

        *legacy_typing* renders leaves as text and nested blocks as embedded
        strings -- the pre-typing form. *legacy_signature* puts the root
        kwargs (url) into the identity -- the pre-LEGACY_NORM form, and the
        only thing that has ever made a signature non-relocatable. Pinning the
        typing does not turn it on.

        ``legacy=`` is the era switch and sets BOTH -- which is what it has
        always meant, back when they were one thing. The two named arguments
        override it individually.
        """
        if legacy_typing is None:
            legacy_typing = legacy
        if legacy_signature is None:
            legacy_signature = legacy
        legacy_typing = self._legacy_typing(legacy_typing)
        norm = self._legacy_norm() if legacy_signature is None else bool(legacy_signature)
        if pretty:
            import pprint
            return pprint.pformat(
                self.signaturedict(legacy_typing=legacy_typing, legacy_signature=norm,
                                   deslash=deslash), indent=2, width=120)
        if legacy_typing:
            #CAUTION! This branch is what already-built blocks hashed with, and
            # is the pre-change code verbatim. The NORM flag alone decides root
            # kwargs and quoting, exactly as before -- so a relocatable block
            # pinned for typing stays relocatable.
            sig_spec = self.__expand_spec__('signature', legacy=norm, legacy_typing=True)
            kwargs_dict = {**(self._rootkwargs_ if norm else {}), 'spec': sig_spec}
            sig = self.__repr_from_kwargs__(kwargs_dict, anchor=None, quote_strs=not norm)
        else:
            # Rendered FROM the typed dict, so the text and the dict cannot
            # disagree, and a leaf is quoted exactly when it is a string.
            # Root kwargs only on explicit opt-in: signature and hash are
            # relocatable, and nothing about typing changes that.
            root = ''.join(f"{k}={v!r}, " for k, v in self._rootkwargs_.items()) if norm else ''
            sig = f"({root}spec={self._typed_specdict(legacy=False)!r})"
        if deslash:
            sig = sig.replace('\\', '')
        self.log.detailed(f"signature: ------------> legacy={legacy}")
        self.log.detailed(f"signature: ------------>{sig=}")
        return sig

    def subsignature(self, *args, **kwargs):
        """Alias for :meth:`signature` for backwards compatibility."""
        return self.signature(*args, **kwargs)

    def norm(self, *args, **kwargs):
        """Alias for :meth:`signature` for backwards compatibility."""
        return self.signature(*args, **kwargs)


    @staticmethod
    def _parse_signature(signature: str) -> dict:
        """Parse a signature string like 'anchor(k1=v1, k2=v2)' into {k: v} dict."""
        signature = signature.strip()
        paren_start = signature.find('(')
        if paren_start == -1:
            return {}
        inner = signature[paren_start + 1:]
        if inner.endswith(')'):
            inner = inner[:-1]
        tokens = []
        depth = 0
        quote_char = None
        start = 0
        for i, c in enumerate(inner):
            if quote_char is not None:
                if c == quote_char and (i == 0 or inner[i - 1] != '\\'):
                    quote_char = None
            elif c in ('"', "'"):
                quote_char = c
            elif c in ('(', '[', '{'):
                depth += 1
            elif c in (')', ']', '}'):
                depth -= 1
            elif c == ',' and depth == 0:
                tokens.append(inner[start:i].strip())
                start = i + 1
        if start < len(inner):
            tokens.append(inner[start:].strip())
        result = {}
        for token in tokens:
            eq_idx = token.find('=')
            if eq_idx == -1:
                continue
            key = token[:eq_idx].strip()
            value = token[eq_idx + 1:].strip()
            result[key] = value
        return result

    @staticmethod
    def _parse_subsignature(*args, **kwargs):
        return Datablock._parse_signature(*args, **kwargs)

    @staticmethod
    def _split_top_level_items(inner: str, sep: str = ','):
        out, buf, depth, quote, esc = [], [], 0, None, False
        for c in inner:
            if esc:
                buf.append(c); esc = False; continue
            if c == '\\':
                buf.append(c); esc = True; continue
            if quote is not None:
                buf.append(c)
                if c == quote:
                    quote = None
                continue
            if c in ('"', "'"):
                quote = c; buf.append(c); continue
            if c in ('(', '[', '{'):
                depth += 1
            elif c in (')', ']', '}'):
                depth -= 1
            elif c == sep and depth == 0:
                out.append(''.join(buf)); buf = []
                continue
            buf.append(c)
        if buf:
            out.append(''.join(buf))
        return out

    @classmethod
    def _parse_dictstr(cls, text: str) -> dict:
        text = text.strip()
        if not (text.startswith('{') and text.endswith('}')):
            return {}
        out = {}
        for item in cls._split_top_level_items(text[1:-1]):
            if not item.strip():
                continue
            parts = cls._split_top_level_items(item, sep=':')
            if len(parts) < 2:
                return {}
            key, value = parts[0].strip(), ':'.join(parts[1:]).strip()
            unquoted = cls._unquote_str(key)
            out[unquoted if unquoted is not None else key] = value
        return out

    @staticmethod
    def _unquote_str(text: str):
        text = text.strip()
        if len(text) < 2 or text[0] != text[-1] or text[0] not in ('"', "'"):
            return None
        try:
            value = ast.literal_eval(text)
        except Exception:
            return None
        return value if isinstance(value, str) else None

    @staticmethod
    def _literal(text):
        if not isinstance(text, str):
            return text
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    @staticmethod
    def _is_signaturestr(text: str) -> bool:
        text = text.strip()
        if not text.endswith(')'):
            return False
        head, _, _ = text.partition('(')
        if head is text:
            return False
        return head == '' or all(p.isidentifier() for p in head.split('.'))

    @staticmethod
    def _is_subsignaturestr(text: str) -> bool:
        return Datablock._is_signaturestr(text)

    @classmethod
    def _structure_from_signature_text(cls, value):
        """Structure a signature value, recovering typed leaves where it can.

        A non-legacy signature is a faithful ``repr`` of a typed dict, so
        ``literal_eval`` reconstructs it exactly -- an ``int`` comes back an
        ``int``, not the substring ``'256'``. This is the read-side
        counterpart for anything holding a rendered signature rather than the
        block, such as a journal row.

        Falls back to `_structure_signatureval` when the text does not parse,
        which is the legacy rendering and anything carrying a specline.
        """
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value.strip())
        except Exception:
            return cls._structure_signatureval(value)

    @classmethod
    def _structure_signatureval(cls, value):
        if not isinstance(value, str):
            return value
        text = value.strip()
        inner = cls._unquote_str(text)
        if inner is not None:
            structured = cls._structure_signatureval(inner)
            if isinstance(structured, dict):
                return structured
            return value
        if text.startswith('{') and text.endswith('}'):
            parsed = cls._parse_dictstr(text)
            if parsed:
                return {k: cls._structure_signatureval(v) for k, v in parsed.items()}
            return value
        if cls._is_signaturestr(text):
            parsed = Datablock._parse_signature(text)
            if parsed:
                return {k: cls._structure_signatureval(v) for k, v in parsed.items()}
        return value

    @classmethod
    def _structure_subsignatureval(cls, value):
        return cls._structure_signatureval(value)

    @staticmethod
    def _parse_norm(*args, **kwargs):
        return Datablock._parse_signature(*args, **kwargs)

    @staticmethod
    def _is_normstr(*args, **kwargs):
        return Datablock._is_signaturestr(*args, **kwargs)

    @classmethod
    def _structure_normval(cls, *args, **kwargs):
        return cls._structure_signatureval(*args, **kwargs)

    def _journal_entry(self, journal: dict) -> 'DatajournalEntry':
        selectors = {k: journal[k] for k in ('entry_path', 'iloc', 'loc') if k in journal}
        filters = {k: v for k, v in journal.items() if k not in selectors}
        if len(selectors) != 1:
            raise ValueError(
                "journal must contain exactly one of 'entry_path', 'iloc', or "
                f"'loc'; got {sorted(selectors)}"
            )
        (key, value), = selectors.items()
        if key == 'entry_path':
            if filters:
                raise ValueError(
                    f"journal={{'entry_path': ...}} names one file, so the extra "
                    f"filters {sorted(filters)} cannot be applied; drop them or "
                    f"select with 'iloc'/'loc' instead"
                )
            fs, _ = fsspec.url_to_fs(value, **(self.storage_options or {}))
            with fs.open(value, 'rb') as f:
                _df = pd.read_parquet(f)
            return DatajournalEntry(_df.iloc[0].dropna(), storage_options=self.storage_options)
        return self.journal(**{key: value}, **filters)

    def diffsignature(
        self,
        other_signature: 'Datablock | DatajournalEntry | str | None' = ABSENT,
        *,
        journal: 'Datajournal | DatajournalEntry | dict | str | int | None' = None,
        raw: bool = False,
        deslash: bool = False,
        legacy: 'bool | None' = None,
        recursive: bool = True,
        report: bool = False,
        maxlen: 'int | None' = 160,
    ) -> 'dict | str':
        """Diff this datablock's signature against another signature, key by key."""
        if isinstance(other_signature, Datablock):
            other_signature = other_signature.signature(legacy=legacy)
        elif isinstance(other_signature, DatajournalEntry):
            other_signature = other_signature.read('signature') or other_signature.read('subsignature') or other_signature.read('norm') or ''
        elif (other_signature is None or other_signature is ABSENT) and journal is not None:
            _entry = self._journal_entry(journal)
            other_signature = _entry.read('signature') or _entry.read('subsignature') or _entry.read('norm') or ''

        def present(value):
            if value is ABSENT:
                return value
            if not raw:
                value = self._literal(value)
            if deslash and isinstance(value, str):
                return value.replace('\\', '')
            return value

        def diffdict(d1, d2):
            diff = {}
            for key in sorted(set(d1) | set(d2)):
                val1 = d1[key] if key in d1 else ABSENT
                val2 = d2[key] if key in d2 else ABSENT
                if isinstance(val1, dict) and isinstance(val2, dict):
                    valdiff = diffdict(val1, val2)
                    if len(valdiff) > 0:
                        diff[key] = valdiff
                else:
                    one, two = present(val1), present(val2)
                    if one is ABSENT or two is ABSENT or one != two or val1 != val2:
                        if not raw and one is not ABSENT and two is not ABSENT:
                            try:
                                indistinguishable = bool(one == two)
                            except Exception:
                                indistinguishable = False
                            if indistinguishable and val1 != val2:
                                one, two = val1, val2
                        diff[key] = (one, two)
            return diff

        parsed_self  = Datablock._parse_signature(self.signature(legacy=legacy))
        parsed_other = Datablock._parse_signature(other_signature or '')

        def _normalize_subsig_dict(d):
            if 'spec' not in d and d:
                root_keys = {'url', 'local', 'local_must_exist', 'storage_options', 'anchor', 'tag', 'revision', 'keyby', 'uuid16', 'redirect', 'validate_vars'}
                spec_part = {}
                root_part = {}
                for k, v in d.items():
                    if k in root_keys:
                        root_part[k] = v
                    else:
                        spec_part[k] = v
                if spec_part:
                    root_part['spec'] = spec_part
                    return root_part
            return d

        if 'spec' in parsed_self and 'spec' not in parsed_other:
            parsed_other = _normalize_subsig_dict(parsed_other)
        elif 'spec' not in parsed_self and 'spec' in parsed_other:
            parsed_self = _normalize_subsig_dict(parsed_self)

        if recursive:
            parsed_self = {k: self._structure_signatureval(v) for k, v in parsed_self.items()}
            parsed_other = {k: self._structure_signatureval(v) for k, v in parsed_other.items()}
        diff = diffdict(parsed_self, parsed_other)
        if not report:
            return diff
        return self.format_diff(diff, maxlen=maxlen)

    def diffsubsignature(self, *args, **kwargs):
        return self.diffsignature(*args, **kwargs)

    def diffsig(self, *args, **kwargs):
        """Alias for :meth:`diffsignature`."""
        return self.diffsignature(*args, **kwargs)

    def diffsubsig(self, *args, **kwargs):
        return self.diffsignature(*args, **kwargs)

    def diffnorm(self, *args, **kwargs):
        return self.diffsignature(*args, **kwargs)

    def signaturedict(self, *, legacy: 'bool | None' = None,
                      legacy_typing: 'bool | None' = None,
                      legacy_signature: 'bool | None' = None,
                      deslash: bool = False) -> dict:
        """The signature as a nested dict of correctly-typed values.

        Built from ``var`` via `_typed_specdict`, so an ``int`` field comes
        back an ``int``. Speclines stay strings.

        Under LEGACY_TYPING it is the old thing: the rendered signature parsed
        back into text leaves. *deslash* applies to that rendering before it is
        parsed -- stripping backslashes from the formatted output afterwards
        would eat the escapes ``repr`` put inside the leaves.
        """
        if legacy_typing is None:
            legacy_typing = legacy
        if legacy_signature is None:
            legacy_signature = legacy
        if self._legacy_typing(legacy_typing):
            parsed = Datablock._parse_signature(self.signature(
                legacy_typing=True, legacy_signature=legacy_signature, deslash=deslash))
            return {k: self._structure_from_signature_text(v) for k, v in parsed.items()}
        return {'spec': self._typed_specdict(legacy=False)}

    def sigdict(self, *, legacy: 'bool | None' = None, deslash: bool = False) -> dict:
        return self.signaturedict(legacy=legacy, deslash=deslash)

    def subsignaturedict(self, *, legacy: 'bool | None' = None, deslash: bool = False) -> dict:
        return self.signaturedict(legacy=legacy, deslash=deslash)

    def subsigdict(self, *, legacy: 'bool | None' = None, deslash: bool = False) -> dict:
        return self.signaturedict(legacy=legacy, deslash=deslash)

    def normdict(self, *args, **kwargs):
        return self.signaturedict(*args, **kwargs)

    def sig(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = True):
        """Alias for :meth:`signature` (defaults to pretty=True)."""
        return self.signature(deslash=deslash, legacy=legacy, pretty=pretty)

    def subsig(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = True):
        return self.signature(deslash=deslash, legacy=legacy, pretty=pretty)

    def typedict(self, *, deslash: bool = False, legacy: 'bool | None' = None,
                 legacy_typing: 'bool | None' = None,
                 legacy_signature: 'bool | None' = None) -> dict:
        """Return the full type structured as a dictionary."""
        return {
            'signature': self.signaturedict(
                legacy=legacy, legacy_typing=legacy_typing,
                legacy_signature=legacy_signature, deslash=deslash),
            'version': self.version,
            'paths': getattr(self, '_paths_', None),
            'topics': self.signature_topics(),
        }

    def tpdict(self, *, deslash: bool = False, legacy: 'bool | None' = None) -> dict:
        return self.typedict(deslash=deslash, legacy=legacy)

    def tp(self, *, deslash: bool = False, legacy: 'bool | None' = None, pretty: bool = True):
        """Alias for :meth:`type` (defaults to pretty=True)."""
        return self.type(deslash=deslash, legacy=legacy, pretty=pretty)

    def type(self, *, deslash: bool = False, legacy: 'bool | None' = None, pretty: bool = False):
        """Return the full type representation (signature + paths + version + topics)."""
        legacy = self._legacy_typing(legacy)
        if pretty:
            import pprint
            return pprint.pformat(
                self.typedict(deslash=deslash, legacy=legacy), indent=2, width=120)
        parts = [self.signature(deslash=deslash, legacy=legacy)]
        if getattr(self, '_paths_', None) is not None:
            parts.append(f"_paths_={getattr(self, '_paths_', None)}")
        parts.append(f"version={self.version}")
        parts.extend(self.signature_topics())
        tp = os.path.join(*parts)
        if deslash:
            tp = tp.replace('\\', '')
        return tp

    Diff = collections.namedtuple('Diff', ['subsig', 'topics', 'version'])

    def _topic_map(self, topics):
        """A TOPICS declaration as an ordered ``{path: value}`` map, or None.

        The structured counterpart of :meth:`signature_topics`: one entry per
        leaf, keyed by its ``'/'``-joined path, valued by the text that follows
        the ``=`` in that leaf's segment -- :data:`ABSENT` for a list-``TOPICS``
        entry, whose segment has no ``=`` at all. None for a block that declares
        no topics, which is the ``topics:None`` segment and NOT the same as the
        empty map of ``TOPICS = {}``.
        """
        if isinstance(topics, dict):
            out = {}
            # The era of the declaration in hand, which for the other side of a
            # difftopics() is not necessarily this block's own.
            modern = self._modern_topics(topics)

            def walk(node, prefix):
                if not isinstance(node, dict):
                    out['/'.join(prefix)] = self._topictext(node, modern)
                    return
                for name, child in node.items():
                    walk(child, prefix + (str(name),))

            for name, child in topics.items():
                walk(child, (str(name),))
            return out
        if isinstance(topics, list):
            return {str(name): ABSENT for name in topics}
        return None

    def _other_topics(self, other_topics, journal):
        """The other side of a :meth:`difftopics`, as ``(segments, map)``.

        Accepts a live block, a journal entry, a ``TOPICS`` declaration, or the
        ``str(dict)`` a journal records one as.
        """
        if other_topics is ABSENT:
            if journal is None:
                raise ValueError("difftopics needs other_topics= or journal=")
            other_topics = self._journal_entry(journal)
        if isinstance(other_topics, Datablock):
            return other_topics.signature_topics(), other_topics._topic_map(getattr(other_topics, 'TOPICS', None))
        if isinstance(other_topics, DatajournalEntry):
            # A journal records a list-TOPICS block as a mapping of DIRTOPIC,
            # so the two render alike from an entry even though they do not from
            # the blocks themselves. Compare two LIVE blocks to see that one.
            other_topics = other_topics.block.TOPICS
        elif isinstance(other_topics, str):
            other_topics = literal_topics(other_topics)
        topicmap = self._topic_map(other_topics)
        return self._render_topic_map(topicmap), topicmap

    @staticmethod
    def _render_topic_map(topicmap):
        if topicmap is None:
            return ("topics:None",)
        return tuple(f"topic:{path}" if value is ABSENT else f"topic:{path}={value}"
                     for path, value in topicmap.items())

    def difftopics(
        self,
        other_topics=ABSENT,
        *,
        journal: 'dict | None' = None,
        report: bool = False,
        maxlen: 'int | None' = 160,
    ) -> 'dict | str':
        """Diff this block's topics against another's, the way :attr:`signature` sees them.

        Compares :meth:`signature_topics` -- the very segments the signature is
        built from -- so the two agree by construction: the result is empty
        exactly when the topics contribute nothing to a difference in signature,
        and non-empty exactly when they do.

        Returns a **sparse** dict keyed by topic path, valued by
        ``(self_filename, other_filename)`` as those render into the signature,
        with :data:`ABSENT` for a path one side does not declare. A difference
        belonging to no single path -- a reordering, or ``TOPICS = {}`` against
        no TOPICS at all -- is reported under the :data:`SIGNATURE_TOPICS`
        sentinel key, carrying both segment tuples.

        Parameters
        ----------
        other_topics:
            A :class:`Datablock`, a :class:`DatajournalEntry`, a ``TOPICS``
            declaration (dict or list, ``None`` for a block declaring none), or
            the ``str(dict)`` form a journal records. Omit it to read the other
            side from *journal*.
        journal:
            Selector dict for the journal entry to compare against, as
            :meth:`diffsubsig`. Note that a journal records a list-``TOPICS``
            block as a mapping of :data:`DIRTOPIC`, so a list declaration and the
            equivalent dict one are indistinguishable once written -- against an
            entry they compare equal, against the live block they do not.
        report:
            Return readable text instead of the dict.
        maxlen:
            Truncate values longer than this in the *report* only.
        """
        mine = self.signature_topics()
        theirs, theirmap = self._other_topics(other_topics, journal)
        mymap = self._topic_map(getattr(self, 'TOPICS', None))

        diff = {}
        if tuple(mine) != tuple(theirs):
            for path in list(mymap or {}) + [p for p in (theirmap or {}) if p not in (mymap or {})]:
                one = (mymap or {}).get(path, ABSENT)
                two = (theirmap or {}).get(path, ABSENT)
                if one != two:
                    diff[path] = (one, two)
            if not diff:
                # They differ, but no single path does: a reordering, or the
                # empty-TOPICS/no-TOPICS distinction. Report the renderings.
                diff[SIGNATURE_TOPICS] = (tuple(mine), tuple(theirs))
        if not report:
            return diff
        return self.format_diff(diff, maxlen=maxlen)

    def diffversion(self, other_version=ABSENT, *, journal: 'dict | None' = None):
        """Diff this block's :attr:`version` against another's.

        Returns ``(self_version, other_version)`` when they differ, and ``None``
        when they do not -- so it is empty in the same sense the other two diffs
        are, and ``if block.diffversion(...)`` reads as "did the version move".

        Compared as :attr:`signature` renders them (``f"version={v}"``), so
        ``1`` and ``'1'`` are the same version -- they are the same signature,
        and this method exists to answer for the signature. Both values are
        reported as they are, so the type difference is still visible.

        Parameters
        ----------
        other_version:
            A :class:`Datablock`, a :class:`DatajournalEntry`, or a version
            value (``None`` being the version of a block declaring no
            ``VERSION``). Omit it to read the other side from *journal*.
        journal:
            Selector dict for the journal entry to compare against, as
            :meth:`diffsubsig`.
        """
        if other_version is ABSENT:
            if journal is None:
                raise ValueError("diffversion needs other_version= or journal=")
            other_version = self._journal_entry(journal)
        if isinstance(other_version, (Datablock, DatajournalEntry)):
            other_version = other_version.version
        mine = self.version
        if str(mine) == str(other_version):
            return None
        return (mine, other_version)

    def diffsig(self, *args, **kwargs):
        """Alias for :meth:`diffsubsignature`."""
        return self.diffsubsignature(*args, **kwargs)

    def diff(
        self,
        other=ABSENT,
        *,
        journal: 'dict | None' = None,
        report: bool = False,
        maxlen: 'int | None' = 160,
        **kwargs,
    ) -> 'Diff | tuple':
        """Diff this block against another across all three signature components."""
        if other is not ABSENT and not isinstance(other, (Datablock, DatajournalEntry)) and journal is None:
            raise TypeError(f"diff requires a Datablock or DatajournalEntry, got {type(other).__name__}: {other!r}")
        subsig = self.diffsubsignature(other, journal=journal, report=report, maxlen=maxlen, **kwargs)

        topics = self.difftopics(other, journal=journal, report=report, maxlen=maxlen)
        version = self.diffversion(other, journal=journal)
        if report and version is not None:
            version = f"self : {version[0]!r}\nother: {version[1]!r}"
        elif report:
            version = "no differences"
        return self.Diff(subsig, topics, version)

    @classmethod
    def format_diff(cls, diff: dict, *, maxlen: 'int | None' = 160) -> str:
        """Render a diff dict as one ``path`` + self/other per difference."""
        def crop(value):
            # repr() unconditionally: leaves are typed, so a bare rendering would
            # print the float 15.0 and the string '15.0' identically -- which is
            # exactly the distinction the report exists to show.
            text = repr(value)
            if maxlen is not None and len(text) > maxlen:
                text = f"{text[:maxlen]}... (+{len(text) - maxlen} chars)"
            return text

        def walk(node, path):
            for key, value in node.items():
                here = path + [str(key)]
                if isinstance(value, dict):
                    walk(value, here)
                else:
                    self_val, other_val = value
                    lines.append('.'.join(here))
                    lines.append(f"    self : {crop(self_val)}")
                    lines.append(f"    other: {crop(other_val)}")

        lines = []
        walk(diff, [])
        if not lines:
            return "no differences"
        return '\n'.join(lines)

    def __repr__(self, *, deslash: bool = True):
        # quote_strs is unconditional here, LEGACY_NORM or not: __repr__ is not
        # an input to signature, so quoting can only make the rendering more
        # faithful -- `url=abfss://x` is not evaluable at all, `url='abfss://x'`
        # is. Only signature()/subsignature() have to honour the legacy form.
        repr_spec = self.__expand_spec__('repr')
        r = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': repr_spec},
            **self._tailkwargs_,
        }, anchor='fqcn', quote_strs=True)
        self.log.detailed(f"__repr__(): ------------> {repr_spec=}")
        self.log.detailed(f"__repr__(): ------------> __repr__={r}")
        if deslash:
            r = r.replace('\\', '')
        return r

    def __str__(self):
        s = self.quote()
        s = s.replace('\\', '')
        return s
    
    @property
    def dfn(self):
        """The full definition (state) of this Datablock instance.

        Returns a dict containing ALL parameters — both the explicit parameters
        declared in ``Datablock.__init__`` (e.g. ``root``, ``tag``, ``revision``,
        ``keyby``, …) and any extra ``**kwargs`` that were passed at construction
        time.

        This is the dict that would be needed to reconstruct the block::

            block2 = MyBlock(**block1.dfn)
            assert block1.dfn == block2.dfn

        See also
        --------
        kwargs : The complementary property that returns *only* the dynamic
                 (non-explicit) parameters.
        """
        return self.__getstate__()

    @functools.cached_property
    def var(self):
        verbose = getattr(self, 'VERBOSE_VAR', False) or getattr(self, 'VERBOSE_CONFIG', False)
        log_fn = self.log.verbose if verbose else self.log.detailed
        log_fn(f"Forming var from spec: BEGIN")
        var = self._spec_to_var(self.spec)
        log_fn(f"Forming var from spec: END")
        return var

    # DEPRECATED ALIASES of .var
    @property
    def cfg(self):
        return self.var

    @property
    def config(self):
        return self.var

    @property
    def kwargs(self):
        """The dynamically-supplied keyword arguments of this Datablock instance.

        Returns a dict containing *only* the parameters that are NOT declared
        as explicit keyword arguments in ``Datablock.__init__``.  For example,
        if a block is created as::

            block = MyBlock(root='/data', my_custom_param=42)

        then ``block.kwargs == {'my_custom_param': 42}`` — the ``root`` key is
        excluded because it is an explicit parameter.

        These are the "user-defined" parameters that distinguish one block
        configuration from another within the same class.

        See also
        --------
        dfn : The complementary property that returns the full definition
              including explicit parameters.
        """
        explicit_keys = set(self.__explicit_params__())
        return {k: v for k, v in self.__getstate__().items() if k not in explicit_keys}
    
    def signature_topics(self):
        """The topic segments of :attr:`signature`, in the order it joins them.

        The one rendering of a block's topics into its identity: :attr:`signature`
        and :attr:`supersignature` join what this returns, and :meth:`difftopics`
        compares it. Two blocks whose signatures differ only in their topics are
        exactly the two whose ``signature_topics()`` differ -- which is what makes
        the diff answer the question the hash asks.
        """
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        if self._topicfiles is not None:
            # A leaf is named by its full path, so a nested topic reads
            # "topic:data/frames=None". A flat TOPICS has one-segment paths and
            # renders byte-identically to before -- the hash does not move.
            modern = self._modern_topics()
            return tuple(f"topic:{'/'.join(tp)}={self._topictext(self._topicnode(*tp), modern)}"
                         for tp in self.leaftopics())
        if hasattr(self, "TOPICS") and isinstance(self.TOPICS, list):
            return tuple(f"topic:{topic}" for topic in self.TOPICS)
        return ("topics:None",)

    def type(self, *, deslash: bool = False, legacy: 'bool | None' = None,
             legacy_typing: 'bool | None' = None,
             legacy_signature: 'bool | None' = None, pretty: bool = False):
        if legacy_typing is None:
            legacy_typing = legacy
        if legacy_signature is None:
            legacy_signature = legacy
        legacy_typing = self._legacy_typing(legacy_typing)
        if pretty:
            import pprint
            return pprint.pformat(
                self.typedict(deslash=deslash, legacy_typing=legacy_typing,
                              legacy_signature=legacy_signature), indent=2, width=120)
        parts = [self.signature(deslash=deslash, legacy_typing=legacy_typing,
                                legacy_signature=legacy_signature)]
        if self.__dict__.get('__redirected_paths__') is not None:
            parts.append(f"_redirected_paths_={self.__dict__['__redirected_paths__']}")
        parts.append(f"version={self.version}")
        parts.extend(self.signature_topics())
        tp = os.path.join(*parts)
        if deslash:
            tp = tp.replace('\\', '')
        return tp

    @property
    def hash(self): 
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hash
        # computed with the older code.
        if not hasattr(self, '_hash'):
            sha = hashlib.sha256()
            tp = self.type()
            sha.update(tp.encode())
            self._hash = sha.hexdigest()
            self.log.detailed(f"hash: ---------===---------> {tp=} ---> hash: {self._hash}")
        return self._hash

    @property
    def code(self):
        if not hasattr(self, '_code'): 
            if getattr(self, '_code_', None) is not None:
                self._code = self._code_
            elif getattr(self, '_subhash_', None) is not None:
                self._code = self._subhash_
            else:
                sha = hashlib.sha256()
                sig = self.signature()
                sha.update(sig.encode())
                self._code = sha.hexdigest()
                self.log.detailed(f"code: ---------===---------> {sig=} ---> code: {self._code}")
        return self._code

    @property
    def subhash(self):
        return self.code

    ### anchorage: begin
    @property
    def anchor(self):
        if self._anchor_ is not None:
            return self._anchor_
        return self.fqcn

    @property
    def tag(self):
        return self._tag_

    @property
    def key(self):
        """Return the key component based on self.keyby."""
        if self.keyby is None:
            key = None
        elif self.keyby == 'hash':
            key = self.hash
        elif self.keyby in ('code', 'subhash', 'superhash'):
            key = self.code
        elif self.keyby in ('signature', 'norm', 'subsignature'):
            key = self.signature()

        elif self.keyby == 'tag':
            key = self.tag
        elif self.keyby in ('taghash', 'tag_hash'):
            if self._tag_ is None:
                key = self.hash
            else:
                key = f"{self.tag}/{self.hash[:8]}"
        elif self.keyby == 'version_hash':
            if self.version is not None:
                key = f"version={self.version}/{self.hash[:8]}"
            else:
                key = self.hash
        elif self.keyby == 'tag_version_hash' or self.keyby == 'tag_version_shorthash':
            parts = []
            if self._tag_ is not None:
                parts.append(self.tag)
            if self.version is not None:
                parts.append(f"version={self.version}")
            if parts:
                if self.keyby == 'tag_version_shorthash':
                    parts.append(self.hash[:8])
                else:
                    parts.append(self.hash)
            else:
                if self.keyby == 'tag_version_shorthash':
                    parts.append(self.hash[:8])
                else:
                    parts.append(self.hash)
            key = '/'.join(parts)
        else:  
            raise NotImplementedError(f"keyby {repr(self.keyby)} is not implemented: missing override?")
        return key
    ### anchoracte: END
    #IDS: END

    #PATHS: BEGIN
    def __path__(
        self,
        *topicpath,
        ensure_dirpath: bool = False,
        bare: bool = False,
        local: bool = False,
    ):
        """Default path resolution for topics. May be implemented/overridden by specializations."""
        topicpath = self._normtopic(topicpath)
        node = self._topicnode(*topicpath)

        if isinstance(node, dict):
            return {name: self.path(*topicpath, name, ensure_dirpath=ensure_dirpath,
                                    bare=bare, local=local)
                    for name in node}
        if self._node_is_syntopic(node):
            return None

        dirpath = self.dirpath(*topicpath, local=local)
        if ensure_dirpath and dirpath is not None:
            ensure_path(dirpath, storage_options=self.storage_options)

        if self._node_is_dirtopic(node):
            return dirpath
        path = os.path.join(dirpath, node)
        self.log.detailed(f"{self.anchor}: path: {path}")
        if bare and path:
            fs = self.localfs if local else self.fs
            path = fs._strip_protocol(path)
        return path

    def path(
        self,
        *topicpath,
        ensure_dirpath: bool = False,
        bare: bool = False,
        local: bool = False,
    ):
        """Return the path for a topic, using self._redirected_paths_ if available, falling back onto anchorkeypath/topic."""
        if not local:
            red_paths = self._redirected_paths_
            if red_paths is not None:
                topicpath = self._normtopic(topicpath)
                if not topicpath:
                    res = red_paths
                else:
                    node = red_paths
                    for name in topicpath:
                        if isinstance(node, dict) and name in node:
                            node = node[name]
                        else:
                            node = None
                            break
                    res = node
                if res is not None:
                    if bare and isinstance(res, str):
                        fs = self.fs
                        res = fs._strip_protocol(res)
                    return res

        return self.__path__(
            *topicpath,
            ensure_dirpath=ensure_dirpath,
            bare=bare,
            local=local,
        )

    def ls(self, *topicpath, detail=False, local: bool = False):
        """List the contents at ``.path(*topicpath)`` using *fsspec*.

        If the path points to a file (i.e. a dict-TOPICS entry with a
        non-None filename), the parent directory is listed.  If the path
        is a directory (list-TOPICS, or dict-TOPICS with ``None``), it
        is listed directly.

        Parameters
        ----------
        topic : str
            The topic whose path to list.
        detail : bool, optional
            When *True* return full ``fsspec.ls`` dicts instead of plain
            path strings.

        Returns
        -------
        list[str] | list[dict]
            Listing of the path contents.
        """
        fs = self.localfs if local else self.fs
        topicpath = self._normtopic(topicpath)
        if self.is_topicgroup(*topicpath):
            # A group has no listing of its own: concatenate its leaves'.
            return [entry
                    for tp in self._leaves_under(*topicpath)
                    for entry in self.ls(*tp, detail=detail, local=local)]
        p = self.path(*topicpath, local=local)
        return ls_path(fs, p, self._is_dir_topic(*topicpath), detail=detail)

    def list(self, *topicpath, local: bool = False):
        """Detailed, recursive listing of every file under ``.path(*topicpath)``.

        Parallels :meth:`ls`, but recurses and returns full ``fsspec``
        detail dicts for all files (directory entries excluded) beneath the
        topic's path.  For a dict-TOPICS single-file topic the file itself
        is returned.  Returns an empty list when the path is absent.

        Parameters
        ----------
        topic : str
            The topic whose files to list.
        local : bool, optional
            When *True* operate on the local cache of the topic
            (``.path(topic, local=True)``) rather than the (possibly
            remote) canonical path.

        Returns
        -------
        list[dict]
            One ``fsspec`` detail dict per file, with ``name`` normalized
            to a fully-qualified path.
        """
        fs = self.localfs if local else self.fs
        topicpath = self._normtopic(topicpath)
        if self.is_topicgroup(*topicpath):
            return [entry
                    for tp in self._leaves_under(*topicpath)
                    for entry in self.list(*tp, local=local)]
        p = self.path(*topicpath, local=local)
        return list_path(fs, p, self._is_dir_topic(*topicpath))

    def size(self, *topicpath, local: bool = False):
        """Total size in bytes of all files under ``.path(*topicpath)``.

        Sums the ``size`` of every file reported by :meth:`list`.  Returns
        0 when the topic has no files.

        Parameters
        ----------
        topic : str
            The topic whose files to size.
        local : bool, optional
            When *True* size the local cache of the topic instead of the
            (possibly remote) canonical path.
        """
        return size(self.list(*self._normtopic(topicpath), local=local))


    def dirpath(
        self,
        *topicpath,
        ensure: bool = False,
        list: bool = False,
        local: bool = False,
    ):
        """The directory for a topic, one path segment per level.

        A group has a directory of its own -- ``dirpath('data')`` is the parent
        of ``dirpath('data', 'frames')`` -- so this answers for groups and
        leaves alike.  A :data:`SYNTOPIC` has no location and gives ``None``.

        A redirected topic answers with the directory of the path it is
        redirected to -- the path itself for a directory topic, its parent for a
        file one -- so a listing of a redirected block lists the data it
        actually reads. As in :meth:`path`, ``local=True`` is never redirected:
        the local cache is this block's own.
        """
        topicpath = self._normtopic(topicpath)
        if self._is_syntopic(*topicpath):
            # No location: nothing to name, and nothing to create for `ensure`.
            return None
        anchorkeypath = self.localanchorkeypath if local else self.anchorkeypath
        fs = self.localfs if local else self.fs
        if not local:
            redirected = self._redirect_dirpath(*topicpath)
            if redirected is not None:
                if list:
                    _lspath = redirected if redirected.endswith('/') else redirected + '/'
                    return fs.ls(_lspath)
                return redirected
        dirpath = os.path.join(anchorkeypath, *topicpath)
        if ensure:
            fs.makedirs(dirpath, exist_ok=True)
        if list:
            # Trailing "/" ensures Azure adlfs lists directory *contents*
            # rather than returning the virtual-directory marker itself.
            _lspath = dirpath if dirpath.endswith('/') else dirpath + '/'
            return fs.ls(_lspath)
        return dirpath

    def linklocal(self, topic, target: str|None = None):
        """Symlink *target* — a plain local filesystem path required by
        external tooling (e.g. TensorBoard) — to wherever *topic* resolves
        under local staging (``path(topic, local=True)``/``dirpath(topic,
        local=True)``).  When this block's url is itself local this is the
        topic's canonical path; otherwise it is the DBX_LOCAL staging path,
        so writers always see a real local path regardless of where
        url/DBX_ROOT points.

        For directory topics (list-TOPICS, or dict-TOPICS with a :data:`DIRTOPIC`
        value) *target* is linked to the topic directory itself. For file
        topics *target* is linked to the topic file path, with its parent
        directory created so a writer can create the file through the
        link. A no-op when *target* is ``None`` or already links to the
        resolved path; repointing a stale link and refusing to clobber a
        non-symlink at *target* are both logged.
        """
        if target is None:
            return self
        local_path = self.path(topic, local=True)
        if local_path is None:
            self.log.warning(
                f"linklocal: topic {topic!r} has no location (SYNTOPIC); "
                f"nothing to link {target} to"
            )
            return self
        if self._is_dir_topic(topic):
            self.localfs.makedirs(local_path, exist_ok=True)
        else:
            self.localfs.makedirs(os.path.dirname(local_path), exist_ok=True)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.lexists(target):
            if not os.path.islink(target):
                self.log.warning(
                    f"linklocal: {target} exists and is not a symlink; "
                    f"refusing to replace it with a link to {local_path}"
                )
                return self
            existing = os.readlink(target)
            if existing == local_path:
                return self
            self.log.info(
                f"linklocal: {target} was stale (linked to {existing}), "
                f"repointing to {local_path}"
            )
            try:
                os.remove(target)
            except OSError as e:
                self.log.warning(
                    f"linklocal: failed to remove stale symlink {target} -> {existing} ({e}); "
                    f"leaving it in place"
                )
                return self
        try:
            os.symlink(local_path, target)
        except OSError as e:
            self.log.warning(f"linklocal: failed to symlink {target} -> {local_path} ({e})")
        return self

    def paths(self):
        """``{topic: path}``, nested wherever TOPICS is."""
        return {topic: self.path(topic) for topic in self.topics()}

    def anchorpath(self):
        return self._anchorpath()

    @property
    def anchorkey(self):
        return self._anchorkey()

    @property
    def anchorkeypath(self):
        return self._anchorkeypath()

    @property
    def localanchorpath(self):
        return self._anchorpath(local=True)

    @property
    def localanchorkeypath(self):
        return self._anchorkeypath(local=True)

    def _anchorpath(self, anchor=None, *, local: bool = False):
        anchor = anchor or self.anchor
        fs, root = (self.localfs, self.localroot) if local else (self.fs, self.root)
        return fs_full_path(fs, os.path.join(root, anchor))

    def _anchorkey(self, anchor=None):
        anchor = anchor or self.anchor
        return os.path.join(anchor, self.key) if self.key else anchor

    def _anchorkeypath(self, anchor=None, *, local: bool = False):
        anchorkey = self._anchorkey(anchor)
        fs, root = (self.localfs, self.localroot) if local else (self.fs, self.root)
        bare = os.path.join(root, anchorkey) if anchorkey else root
        return fs_full_path(fs, bare)
    
    @staticmethod
    def _dbxanchorpathx(url, anchor, x, *, fqcn, ensure: bool = False, storage_options=None):
        """Return {url}/anchor/.dbx/fqcn/x — the anchor-level directory for artefact *x*."""
        fs, root = fsspec.url_to_fs(url, **(storage_options or {}))
        _dbxanchorpathx = fs_full_path(fs, os.path.join(root, anchor, ".dbx", fqcn, x))
        if ensure:
            fs.makedirs(_dbxanchorpathx, exist_ok=True)
        return _dbxanchorpathx

    def _dbxanchorhashpathx(self, x, ext=None, *, ensure_dirpath: bool = True, filename_prefix: str = ''):
        _dbxanchorhashpathx = os.path.join(self.anchorkeypath, ".journal", self.fqcn, x, self.hash)
        if ensure_dirpath:
            self.fs.makedirs(_dbxanchorhashpathx, exist_ok=True)
        if ext is None:
            ext = x
        xpath = os.path.join(_dbxanchorhashpathx, f'{filename_prefix}{self.fqcn}-{x}-{self.hash}-{self.dt}.{ext}')
        return xpath

    def _dbxjournalinstancepath(self, *, ensure_dirpath: bool = False, filename_prefix: str = ''):
        """
        Return {anchorkeypath}/.journal/{fqcn}/journal/{hash}/{fqcn}-{dt}.journal."""
        return self._dbxanchorhashpathx('journal', 'parquet', ensure_dirpath=ensure_dirpath, filename_prefix=filename_prefix)

    #PATHS: END

    #LOG LEVEL: BEGIN
    @property
    def info(self):
        return self.log.ist('info')
    
    @property
    def verbose(self):
        return self.log.ist('verbose')
    
    @property
    def debug(self):
        return self.log.ist('debug')
    
    @property
    def detailed(self):
        return self.log.ist('detailed')

    @property
    def log_volume(self):
        return LogVolume(
            info=self.info,
            verbose=self.verbose,
            debug=self.debug,
            detailed=self.detailed,
        )
    #LOG LEVEL: END

    #JOURNAL: BEGIN
    def _write_journal_dict(self, name, data, *, add_credentials: bool = False):
        if add_credentials:
            data = copy.deepcopy(data)
            data['hash'] = self.hash
            data['datetime'] = self.dt
        #
        ypath = self._dbxanchorhashpathx(name, 'yaml')
        write_yaml(data, ypath, storage_options=self.storage_options)
        assert self.fs.exists(ypath), f"path {ypath} does not exist after writing"
        self.log.detailed(f"WROTE: {name.upper()}: yaml: {ypath}")
        #
        pqpath = self._dbxanchorhashpathx(name, 'parquet')
        df = pd.DataFrame.from_records([{k: repr(v) for k, v in data.items()}])
        with self.fs.open(pqpath, 'wb') as f:
            df.to_parquet(f)
        assert self.fs.exists(pqpath), f"pqpath {pqpath} does not exist after writing"
        self.log.detailed(f"WROTE: {name.upper()}: parquet: {pqpath}")

    def _write_str(self, name, text):
        #
        path = self._dbxanchorhashpathx(name, 'txt')
        write_str(text, path, storage_options=self.storage_options)
        assert self.fs.exists(path), f"scopepath {path} does not exist after writing"
        self.log.detailed(f"WROTE: {name.upper()}: txt: {path}")

    def write_journal_entry(self, event: str, *, note: str = None, inline_note: bool = False,
                            message: str = None, inline_message: bool = False, journal_prefix: str = '',
                            redirection: 'str | dict | None' = None):
        """Write one journal entry for *event*, and return its ``entry_code``.

        ``entry_code`` is a fresh uuid per call, and it is the only field that
        identifies a *row*.  Everything else on an entry describes the block
        or the moment: ``hash`` and ``key`` are shared by every entry of that
        block, ``session`` by every entry of one run, and ``datetime``
        is only as unique as its resolution -- two entries written inside the
        same microsecond, or by two processes at once, collide.  So a caller
        holding an ``entry_code`` can address exactly the row it wrote:

            code = block.write_journal_entry(event='note')
            entry = block.journal(entry_code=code, loc=0)

        With one caveat that is a property of where entries live rather than
        of the code.  A journal *file* is per live instance -- its path is
        built from ``self.dt``, which does not move -- so a second call from
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
        entry_id = uuid.uuid4().hex[:16] if getattr(self, '_uuid16_', False) else str(uuid.uuid4())
        dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        code_seed = f"{self.hash}:{self.session}:{dt}:{event}:{entry_id}"
        code = hashlib.sha256(code_seed.encode('utf-8')).hexdigest()[:32]

        self._write_journal_dict('spec', self.spec)
        self._write_journal_dict('dfn', self.dfn)
        self._write_journal_dict('kwargs', self.kwargs)
        self._write_str('quote', self.quote())
        self._write_str('cite', self.cite())
        self._write_str('repr', self.__repr__())
        self._write_str('signature', self.signature())
        self._write_str('type', self.type())
        if note is not None and not inline_note:
            self._write_str('note', note)

        spec_path = self._dbxanchorhashpathx('spec', 'yaml')
        dfn_path = self._dbxanchorhashpathx('dfn', 'yaml')
        kwargs_path = self._dbxanchorhashpathx('kwargs', 'yaml')
        quote_path = self._dbxanchorhashpathx('quote', 'txt')
        cite_path = self._dbxanchorhashpathx('cite', 'txt')
        signature_path = self._dbxanchorhashpathx('signature', 'txt')
        repr_path = self._dbxanchorhashpathx('repr', 'txt')
        type_path = self._dbxanchorhashpathx('type', 'txt')
        if note is not None and not inline_note:
            note_path = self._dbxanchorhashpathx('note', 'txt')
            note_val = note_path
        else:
            note_val = note
        #
        logpath = self._dbxanchorhashpathx('log', ensure_dirpath=True)
        if logpath is not None:
            has_log = self.fs.exists(logpath)
        else:
            has_log = False
        #
        _TOPICS = getattr(self, 'TOPICS', None)
        topics_dict = ({name: copy.deepcopy(node) for name, node in _TOPICS.items()}
                       if isinstance(_TOPICS, dict)
                       else {topic: DIRTOPIC for topic in self.topics()})
        paths_dict = self.paths()
        #
        journal_path = self._dbxjournalinstancepath(ensure_dirpath=True, filename_prefix=journal_prefix)
        df = pd.DataFrame.from_records([{'datetime': dt,
                                         'build:start:datetime': self._build_start_dt,
                                         'build:end:datetime': self._build_end_dt,
                                         'version': self.version,
                                         'dbx_version': self.dbx_version,
                                         'revision': self.revision, 
                                         'url': self._url_,
                                         'anchor': self.anchor,
                                         'hash': self.hash,
                                         'keyby': self.keyby,
                                         'key': self.key,
                                         'anchorkeypath': self.anchorkeypath,
                                         'code': self.code,
                                         'session': self.session,
                                         'id': entry_id,
                                         'tag': self.tag,
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
        with self.fs.open(journal_path, 'wb') as f:
            df.to_parquet(f)
        
        tagstr = f"with tag {repr(self.tag)} " if self.tag is not None else ""
        self.log.debug(f"WROTE JOURNAL entry {entry_id} for event {repr(event)} {tagstr}"
                         f"to journal_path {journal_path}")
        return entry_id

    @staticmethod
    def Journal(anchor, loc: int = None, *, iloc: int = None, url=None, storage_options=None, log=None, n_workers=8, index=None, unnormalized: bool = False, **filter_kwargs):
        if log is None:
            log = Logger()
        if n_workers is None:
            n_workers = 8
        if loc is not None and iloc is not None:
            raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
        if url is None:
            url = os.environ.get('DBX_ROOT') or os.environ.get('DBX_URL')
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
        def read_entry_file(file):
            # Through `fs`, not by path: the glob above returns paths as that
            # filesystem names them -- protocol-stripped -- so handing one to
            # pandas reads it off the LOCAL disk, where a memory:// or remote
            # journal file is not. Every entry then "skipped as unreadable" and
            # the journal came back empty rather than failing.
            with fs.open(file, 'rb') as f:
                return pd.read_parquet(f, engine='pyarrow')

        df = None
        if len(parquet_files) > 0:
            dfs = []
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(read_entry_file, file) for file in parquet_files]
                future_to_file = {f: file for f, file in zip(futures, parquet_files)}
                for future in tqdm.tqdm(as_completed(futures), desc='Reading journal files', total=len(parquet_files)):
                    try:
                        _df = future.result()
                        _df['entry_path'] = fs_full_path(fs, future_to_file[future])
                    except Exception as e:
                        log.warning(f"Skipping unreadable journal file {future_to_file[future]}: {e}")
                        continue
                    dfs.append(_df)
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
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
                                        ('subhash', 'code'),
                                        ('uuid', 'session')):
                    if legacy in df.columns and current not in df.columns:
                        df = df.rename(columns={legacy: current})
                    elif legacy in df.columns:
                        df = df.drop(columns=[legacy])
                columns = [c for c in Datablock.JOURNAL_COLUMNS
                           if c in df.columns and c != 'event']
                # Anything unlisted keeps its place at the back, ahead of
                # 'event', so a column added later still shows up.
                columns += [c for c in df.columns if c not in set(columns + ['event'])]
                if 'event' in df.columns:
                    columns.append('event')
                df = df.sort_values('datetime', ascending=False)[columns].reset_index(drop=True)
                df = df.rename(columns={'build_log': 'log'})
            else:
                df = None
        journal = Datajournal(df, storage_options=storage_options, index=index,
                              unnormalized=unnormalized, **filter_kwargs)
        if loc is not None:
            result = DatajournalEntry(journal.loc[loc].dropna(), storage_options=storage_options)
        elif iloc is not None:
            result = DatajournalEntry(journal.iloc[iloc].dropna(), storage_options=storage_options)
        else:
            result = journal
        return result

    def journal(self, loc: int = None, *, iloc: int = None, url=None, storage_options=None, log=None, n_workers=8, index: str | None = None, unnormalized: bool = False, **filter_kwargs):
        if loc is not None and iloc is not None:
            raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
        return self.Journal(
            self.anchor,
            loc=loc,
            iloc=iloc,
            url=self._url_ if url is None else url,
            storage_options=self.storage_options if storage_options is None else storage_options,
            log=getattr(self, 'log', None) if log is None else log,
            n_workers=n_workers,
            index=index,
            unnormalized=unnormalized,
            **filter_kwargs,
        )

    def lastbuilt(self, index: str | None = None):
        """Return the most recent 'build:end' DatajournalEntry, or None."""
        j = self.journal(event='build:end', index=index)
        if len(j) == 0:
            return None
        return j.get(0, dropna=True)

    def running(self, index: str | None = None):
        """Return the latest 'build:start' DatajournalEntry with no matching 'build:end', or None."""
        j = self.journal(index=index)
        if len(j) == 0:
            return None
        started = set(j[j['event'] == 'build:start']['hash'])
        ended = set(j[j['event'] == 'build:end']['hash'])
        running_hashes = started - ended
        if not running_hashes:
            return None
        running_entries = j[(j['event'] == 'build:start') & (j['hash'].isin(running_hashes))]
        return DatajournalEntry(running_entries.iloc[0].dropna(), storage_options=self.storage_options)

def UNSAFE_clear_block_callable(block, topics=(), clear_dirpath=False, *, stack=None, idx=None, **kwargs):
    """Module-level callable for UNSAFE_clear_blocks (must be picklable)."""
    block.UNSAFE_clear(*topics, OVERRIDE=True, clear_dirpath=clear_dirpath)
    if stack is not None and idx is not None:
        stack._remove_block_path(idx)
    return block


def UNSAFE_copy_block_from_callable(block, anchorkeypath, overwrite=False, topicpaths=None, validate=True, always_copy_whole_dirpath=False):
    """Module-level callable for UNSAFE_copy_blocks_from (must be picklable).

    show_progress=False: UNSAFE_copy_blocks_from's executor already
    reports real aggregate per-block progress; each block's own
    (typically 1-topic, so always instantly "100%") bar would just
    flood the output otherwise.

    OVERRIDE=True: UNSAFE_copy_blocks_from already confirmed once at the
    top level; without this, each block's own UNSAFE_copy_from would
    re-prompt (or hang waiting on stdin in a worker process) once per block.
    """
    block.UNSAFE_copy_from(anchorkeypath, OVERRIDE=True, overwrite=overwrite, topicpaths=topicpaths, validate=validate, always_copy_whole_dirpath=always_copy_whole_dirpath, show_progress=False)
    return block


class DatablockValidityChecker:
    """Lightweight callable that checks if a block at index `idx` is valid."""
    def __init__(self, idx: int):
        self.idx = idx

    def __call__(self, stack):
        return stack.valid_block(self.idx)


class DatablockRedirectionChecker:
    """Lightweight callable that checks if a block at index `idx` is redirected."""
    def __init__(self, idx: int):
        self.idx = idx

    def __call__(self, stack):
        return stack.redirected_block(self.idx)


class DatablockValidationChecker:
    """Lightweight callable that checks if a block at index `idx` validates."""
    def __init__(self, idx: int, **kwargs):
        self.idx = idx
        self.kwargs = kwargs

    def __call__(self, stack):
        return stack.validate_block(self.idx, **self.kwargs)


class DatablockSignatureMatcher:
    """Lightweight callable that checks if a block at index `idx` matches signature, tag, and/or path pattern clauses."""
    def __init__(
        self,
        idx: int,
        signature_clauses: list[tuple] | None = None,
        tag_clauses: list[tuple] | None = None,
        path_clauses: list[tuple] | None = None,
    ):
        self.idx = idx
        self.signature_clauses = signature_clauses
        self.tag_clauses = tag_clauses
        self.path_clauses = path_clauses

    def __call__(self, stack):
        blk = stack.block(self.idx)
        if self.signature_clauses:
            sig = f"{getattr(blk, 'fqcn', blk.__class__.__name__)}{blk.signature()}"
            if not stack._matches_sig_clauses(sig, self.signature_clauses):
                return False
        if self.tag_clauses:
            tag = getattr(blk, 'tag', None)
            if not stack._matches_tag_clauses(tag, self.tag_clauses):
                return False
        if self.path_clauses:
            paths = stack._get_block_paths(blk)
            if not stack._matches_path_clauses(paths, self.path_clauses):
                return False
        return True


class Datastack(Datablock):
    """Abstract Datablock that orchestrates the building of multiple child
    Datablocks (blocks).

    Subclasses must implement:

        blocks() -> list[Datablock]
            Return the list of child Datablocks to be built.

    Parallelisation is controlled by two ``__init__``-only parameters
    (they are passed through to the Datablock ``__init__`` via ``**kwargs``
    and stored on ``self``, but do **not** affect the hash):

        parallelization : str | None
            Which CallableExecutor to use:
                None / 'inline'           → InlineCallableExecutor  (sequential)
                'multithreading'          → MultithreadingCallableExecutor
                'multiprocessing'         → MultiprocessingCallableExecutor
                'ray'                     → RayCallableExecutor
                'torch_multithreading'    → TorchMultithreadingCallableExecutor
                'torch_multiprocessing'   → TorchMultiprocessingCallableExecutor
        n_workers : int
            Passed straight through to the selected executor.
        devices : list[str] | str | None
            Required for torch parallelizations.  Devices are assigned to
            workers round-robin when ``n_workers > len(devices)``.

    Example
    -------
    ::

        class MyStack(Datastack):
            @dataclass
            class VAR(Datablock.VAR):
                path: str = None
                block_size: int = 100

            def blocks(self):
                n = self._total_items()
                return [
                    MyBlock(url=self.url, spec=dict(path=self.var.path, idx=i))
                    for i in range(math.ceil(n / self.var.block_size))
                ]

        stack = MyStack(root='/data', spec=dict(path='/input', block_size=100),
                        parallelization='multithreading', n_workers=4)
        stack.build()
    """

    class BlockMaker:
        """Lightweight callable that forms and optionally builds a block.

        Designed to be dispatched to a CallableExecutor so that both
        block *formation* (``__block__``) and *building* happen inside
        the worker, parallelizing the expensive Datablock instantiation.
        """
        def __init__(self, idx: int):
            self.idx = idx

        def __call__(self, stack, *, build=True):
            block = stack.__block__(self.idx)
            block.keyby = stack.keyby
            if build:
                block.build()
            del block
            gc.collect()

    @classmethod
    def _get_executors_(cls):
        """Lazily resolve executor classes (defined in dataparts)."""
        if not hasattr(cls, '_executors_cache'):
            cls._executors_cache = {
                "inline":                InlineCallableExecutor,
                "multithreading":        MultithreadingCallableExecutor,
                "multiprocessing":       MultiprocessingCallableExecutor,
                "ray":                   RayCallableExecutor,
                "torch_multithreading":  TorchMultithreadingCallableExecutor,
                "torch_multiprocessing": TorchMultiprocessingCallableExecutor,
            }
        return cls._executors_cache

    def __init__(self, *args, parallelization: str | None = None, n_workers: int = 1, devices: list | str | None = None, multiprocessing_start_method: str = 'spawn', worker_done_timeout_sec: int = 1000, shuffle_callables: bool = False, work_stealing: bool = False, **kwargs):
        super().__init__(*args, parallelization=parallelization, n_workers=n_workers, devices=devices, multiprocessing_start_method=multiprocessing_start_method, worker_done_timeout_sec=worker_done_timeout_sec, shuffle_callables=shuffle_callables, work_stealing=work_stealing, **kwargs)
        # Early validation only — executor_cls is a property so deepcopy/setstate paths work.
        executors = self._get_executors_()
        key = (self.parallelization or "inline").lower()
        if key not in executors:
            raise ValueError(
                f"Unknown parallelization {self.parallelization!r}. "
                f"Choose from {list(executors)}"
            )

    @property
    def executor_cls(self):
        """Resolve executor class from self.parallelization.

        Implemented as a property (not set in __init__) so that objects
        reconstructed via deepcopy / __setstate__ — which bypass __init__ —
        still return the correct class.
        """
        executors = self._get_executors_()
        key = (getattr(self, 'parallelization', None) or "inline").lower()
        if key not in executors:
            raise ValueError(
                f"Unknown parallelization {getattr(self, 'parallelization', None)!r}. "
                f"Choose from {list(executors)}"
            )
        return executors[key]

    # -- Abstract interface -------------------------------------------------------

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks.

        Subclasses **must** override this property.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement n_blocks"
        )

    def __block__(self, idx: int):
        """Return a single child :class:`Datablock` for the given index.

        Subclasses **must** override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __block__(idx)"
        )

    def block(self, idx: int):
        """Return the block at *idx*, lazily forming ``_blocks_`` if needed.

        Does **not** require :attr:`n_blocks` to be available — a block can
        be formed by index alone via :meth:`__block__`.  When ``n_blocks``
        *is* available it is used for bounds-checking.
        """
        if not hasattr(self, '_blocks_') or self._blocks_ is None:
            self._blocks_ = {}
        # Bounds-check when the count is known.
        try:
            n = self.n_blocks
            if idx < 0 or idx >= n:
                raise IndexError(
                    f"Block index {idx} out of range for "
                    f"{self.__class__.__name__} with {n} blocks"
                )
        except NotImplementedError:
            pass
        if idx not in self._blocks_:
            try:
                s = self.__block__(idx)
            except NotImplementedError:
                if self.__class__.blocks is not Datastack.blocks:
                    blist = self.blocks()
                    if 0 <= idx < len(blist):
                        s = blist[idx]
                    else:
                        raise IndexError(f"Block index {idx} out of range for {self.__class__.__name__} with {len(blist)} blocks")
                else:
                    raise
            s = self._adopt(s, keyby=True)
            self._blocks_[idx] = s
        return self._blocks_[idx]

    def blocks(self) -> list:
        """Return all blocks, forming them via :meth:`block` if needed."""
        n = self.n_blocks
        indices = tqdm.tqdm(range(n), desc=f"Forming {n} blocks") if n > 100 else range(n)
        return [self.block(idx) for idx in indices]

    def block_journal(self, **kwargs) -> Datajournal | None:
        """Return the Datajournal for child blocks, or None if no blocks exist or journal fails to load."""
        if self.n_blocks == 0:
            return None
        try:
            return self.block(0).journal(**kwargs)
        except Exception as e:
            self.log.detailed(f"block_journal: could not load journal for child blocks: {e}")
            return None

    def valid_block(self, idx: int) -> bool:
        """Return whether the block at index *idx* is valid."""
        if self._block_paths_topic():
            if self._check_block_path(idx):
                return True
        return self.block(idx).valid()

    def redirected_block(self, idx: int) -> bool:
        """Return whether the block at index *idx* is redirected."""
        return self.block(idx).redirected()

    def valid_blocks(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, **kwargs) -> pd.Series:
        """Return a pandas Series of booleans, one per block, indicating validity (parallelized)."""
        if false_only and true_only:
            raise ValueError("false_only and true_only are mutually exclusive")
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=bool)
        executors = self._get_executors_()
        if parallelization is not None:
            key = parallelization.lower()
        elif n_workers is not None:
            key = 'multithreading' if n_workers > 0 else 'inline'
        else:
            default_par = getattr(self, 'parallelization', None) or 'inline'
            key = default_par.lower()
        if key not in executors:
            raise ValueError(
                f"Unknown parallelization {key!r}. Choose from {list(executors)}"
            )
        executor_cls = executors[key]
        exec_kwargs = self._executor_kwargs(
            tag=f"CHECKING VALIDITY of {n} blocks [{self.__class__.__name__}]",
            n_workers=n_workers,
            executor_cls=executor_cls,
            **kwargs,
        )
        executor = executor_cls(**exec_kwargs)
        checkers = [self.DatablockValidityChecker(i) for i in range(n)]
        results = executor.exec_callables(checkers, self)
        series = pd.Series(results, dtype=bool)
        if false_only:
            return series[~series]
        if true_only:
            return series[series]
        return series

    def redirected_blocks(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, **kwargs) -> pd.Series:
        """Return a pandas Series of booleans, one per block, indicating redirection (parallelized)."""
        if false_only and true_only:
            raise ValueError("false_only and true_only are mutually exclusive")
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=bool)
        executors = self._get_executors_()
        if parallelization is not None:
            key = parallelization.lower()
        elif n_workers is not None:
            key = 'multithreading' if n_workers > 0 else 'inline'
        else:
            default_par = getattr(self, 'parallelization', None) or 'inline'
            key = default_par.lower()
        if key not in executors:
            raise ValueError(
                f"Unknown parallelization {key!r}. Choose from {list(executors)}"
            )
        executor_cls = executors[key]
        exec_kwargs = self._executor_kwargs(
            tag=f"CHECKING REDIRECTION of {n} blocks [{self.__class__.__name__}]",
            n_workers=n_workers,
            executor_cls=executor_cls,
            **kwargs,
        )
        executor = executor_cls(**exec_kwargs)
        checkers = [self.DatablockRedirectionChecker(i) for i in range(n)]
        results = executor.exec_callables(checkers, self)
        series = pd.Series(results, dtype=bool)
        if false_only:
            return series[~series]
        if true_only:
            return series[series]
        return series

    def validate_block(self, idx: int, **kwargs) -> bool:
        """Return whether the block at index *idx* validates."""
        if self.block(idx).validate(**kwargs):
            self._write_block_path(idx)
            return True
        else:
            self._remove_block_path(idx)
            return False

    def validate_blocks(
        self,
        parallelization: str | None = None,
        n_workers: int | None = None,
        work_stealing: bool | None = None,
        false_only: bool = False,
        true_only: bool = False,
        **kwargs,
    ) -> pd.Series:
        """Return a pandas Series of booleans, one per block, indicating validation result (parallelized)."""
        if false_only and true_only:
            raise ValueError("false_only and true_only are mutually exclusive")
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=bool)
        executors = self._get_executors_()
        if parallelization is not None:
            key = parallelization.lower()
        elif n_workers is not None:
            key = 'multithreading' if n_workers > 0 else 'inline'
        else:
            default_par = getattr(self, 'parallelization', None) or 'inline'
            key = default_par.lower()
        if key not in executors:
            raise ValueError(
                f"Unknown parallelization {key!r}. Choose from {list(executors)}"
            )
        executor_cls = executors[key]
        exec_kwargs = self._executor_kwargs(
            tag=f"VALIDATING {n} blocks [{self.__class__.__name__}]",
            n_workers=n_workers,
            executor_cls=executor_cls,
            **({"work_stealing": work_stealing} if work_stealing is not None else {}),
        )
        executor = executor_cls(**exec_kwargs)
        checkers = [self.DatablockValidationChecker(i, **kwargs) for i in range(n)]
        results = executor.exec_callables(checkers, self)
        series = pd.Series(results, dtype=bool)
        if false_only:
            return series[~series]
        if true_only:
            return series[series]
        return series

    @staticmethod
    def _normalize_pattern_spec(spec, *extra_patterns) -> list[tuple]:
        """Normalize pattern spec into list of tuples: OR of ANDs.

        - single string/pattern: `[(pattern,)]`
        - tuple: `[(p1, p2, ...)]` (ANDed)
        - list of strings: `[(s1,), (s2,), ...]` (ORed)
        - list of tuples: `[(p1, p2), (p3, p4)]` (OR of ANDs)
        - spec + extra_patterns: `[(spec, *extra_patterns)]` (ANDed)
        """
        if spec is None and not extra_patterns:
            return []

        if extra_patterns:
            first = [spec] if spec is not None else []
            return [tuple(first + list(extra_patterns))]

        if isinstance(spec, list):
            clauses = []
            for item in spec:
                if isinstance(item, tuple):
                    clauses.append(item)
                elif isinstance(item, list):
                    clauses.append(tuple(item))
                else:
                    clauses.append((item,))
            return clauses
        elif isinstance(spec, tuple):
            return [spec]
        else:
            return [(spec,)]

    @staticmethod
    def _match_single_sig_pattern(sig: str, p) -> bool:
        if isinstance(p, str):
            if p in sig:
                return True
            # Try key=value or key: value fuzzy match in dict/kwargs representations
            if '=' in p:
                k, v = p.split('=', 1)
                k, v = k.strip(), v.strip().strip("'\"")
                pattern_re = rf"['\"]?{re.escape(k)}['\"]?\s*[:=]\s*['\"]?[^,;)\n]*?{re.escape(v)}"
                if re.search(pattern_re, sig):
                    return True
            elif ':' in p:
                k, v = p.split(':', 1)
                k, v = k.strip(), v.strip().strip("'\"")
                pattern_re = rf"['\"]?{re.escape(k)}['\"]?\s*[:=]\s*['\"]?[^,;)\n]*?{re.escape(v)}"
                if re.search(pattern_re, sig):
                    return True
            try:
                if re.search(p, sig):
                    return True
            except re.error:
                pass
            return False
        elif isinstance(p, re.Pattern):
            return bool(p.search(sig))
        elif callable(p):
            return bool(p(sig))
        else:
            return str(p) in sig

    @classmethod
    def _matches_sig_clauses(cls, sig: str, clauses: list[tuple]) -> bool:
        if not clauses:
            return True
        for clause in clauses:
            if all(cls._match_single_sig_pattern(sig, p) for p in clause):
                return True
        return False

    @staticmethod
    def _match_single_tag_pattern(text: str | None, p) -> bool:
        if text is None:
            return False
        if isinstance(p, str):
            if p in text:
                return True
            if '=' in p:
                k, v = p.split('=', 1)
                k, v = k.strip(), v.strip().strip("'\"")
                pattern_re = rf"['\"]?{re.escape(k)}['\"]?\s*[:=]\s*['\"]?[^,;)\n]*?{re.escape(v)}"
                if re.search(pattern_re, text):
                    return True
            elif ':' in p:
                k, v = p.split(':', 1)
                k, v = k.strip(), v.strip().strip("'\"")
                pattern_re = rf"['\"]?{re.escape(k)}['\"]?\s*[:=]\s*['\"]?[^,;)\n]*?{re.escape(v)}"
                if re.search(pattern_re, text):
                    return True
            try:
                if re.search(p, text):
                    return True
            except re.error:
                pass
            return False
        elif isinstance(p, re.Pattern):
            return bool(p.search(text))
        elif callable(p):
            return bool(p(text))
        else:
            return str(p) in text

    @classmethod
    def _matches_tag_clauses(cls, tag: str | None, clauses: list[tuple]) -> bool:
        if not clauses:
            return True
        if tag is None:
            return False
        for clause in clauses:
            if all(cls._match_single_tag_pattern(tag, p) for p in clause):
                return True
        return False

    @classmethod
    def _collect_path_strings(cls, val, out: list[str]):
        if isinstance(val, str):
            out.append(val)
        elif isinstance(val, dict):
            for v in val.values():
                cls._collect_path_strings(v, out)
        elif isinstance(val, (list, tuple, set)):
            for v in val:
                cls._collect_path_strings(v, out)

    @classmethod
    def _get_block_paths(cls, blk) -> list[str]:
        """Extract all path strings for a block."""
        paths_list = []
        try:
            p = blk.paths()
            cls._collect_path_strings(p, paths_list)
        except Exception:
            pass
        if hasattr(blk, 'anchorkeypath'):
            try:
                akp = blk.anchorkeypath
                if akp and akp not in paths_list:
                    paths_list.append(akp)
            except Exception:
                pass
        return paths_list

    @classmethod
    def _matches_path_clauses(cls, block_paths: list[str], clauses: list[tuple]) -> bool:
        if not clauses:
            return True
        if not block_paths:
            return False
        for clause in clauses:
            if all(any(cls._match_single_tag_pattern(path_str, p) for path_str in block_paths) for p in clause):
                return True
        return False

    def find_blocks(
        self,
        signature=None,
        *patterns,
        tag=None,
        path=None,
        parallelization: str | None = None,
        n_workers: int | None = None,
        work_stealing: bool | None = None,
        **kwargs,
    ) -> list[int]:
        """Return a list of indices of all blocks matching signature, tag, and/or path pattern(s) (parallelized)."""
        executor_kw_names = {
            'executor_cls', 'worker_done_timeout_sec', 'shuffle_callables',
            'start_method', 'multiprocessing_start_method', 'devices'
        }
        extra_filter_kwargs = {k: v for k, v in kwargs.items() if k not in executor_kw_names}
        clean_kwargs = {k: v for k, v in kwargs.items() if k in executor_kw_names}

        extra_sig_patterns = [f"{k}={v}" for k, v in extra_filter_kwargs.items()]
        sig_clauses = self._normalize_pattern_spec(signature, *(list(patterns) + extra_sig_patterns))
        tag_clauses = self._normalize_pattern_spec(tag)
        path_clauses = self._normalize_pattern_spec(path)

        if not sig_clauses and not tag_clauses and not path_clauses:
            return []

        n = self.n_blocks
        if n == 0:
            return []

        executors = self._get_executors_()
        if parallelization is not None:
            key = parallelization.lower()
        elif n_workers is not None:
            key = 'multithreading' if n_workers > 0 else 'inline'
        else:
            default_par = getattr(self, 'parallelization', None) or 'inline'
            key = default_par.lower()
        if key not in executors:
            raise ValueError(
                f"Unknown parallelization {key!r}. Choose from {list(executors)}"
            )
        executor_cls = executors[key]
        exec_kwargs = self._executor_kwargs(
            tag=f"FINDING BLOCKS matching sig={sig_clauses} tag={tag_clauses} path={path_clauses} in {n} blocks [{self.__class__.__name__}]",
            n_workers=n_workers,
            executor_cls=executor_cls,
            **({"work_stealing": work_stealing} if work_stealing is not None else {}),
            **clean_kwargs,
        )
        executor = executor_cls(**exec_kwargs)
        matchers = [
            self.DatablockSignatureMatcher(
                i,
                signature_clauses=sig_clauses if sig_clauses else None,
                tag_clauses=tag_clauses if tag_clauses else None,
                path_clauses=path_clauses if path_clauses else None,
            )
            for i in range(n)
        ]
        results = executor.exec_callables(matchers, self)
        return [i for i, matched in enumerate(results) if matched]

    DatablockValidityChecker = DatablockValidityChecker
    DatablockRedirectionChecker = DatablockRedirectionChecker
    DatablockValidationChecker = DatablockValidationChecker
    DatablockSignatureMatcher = DatablockSignatureMatcher
    BlockValidChecker = DatablockValidityChecker
    BlockRedirectedChecker = DatablockRedirectionChecker
    BlockValidationChecker = DatablockValidationChecker
    BlockSignatureMatcher = DatablockSignatureMatcher

    # -- Default build logic ------------------------------------------------------

    def _executor_kwargs(self, tag: str | None = None, n_workers: int | None = None, executor_cls=None, **kwargs) -> dict:
        nw = n_workers if n_workers is not None else getattr(self, 'n_workers', 1)
        cls = executor_cls or self.executor_cls
        executor_kwargs = dict(
            n_workers=nw,
            tag=tag or f"EXECUTING [{self.__class__.__name__}]",
        )
        if hasattr(self, 'worker_done_timeout_sec') and self.worker_done_timeout_sec is not None:
            executor_kwargs['worker_done_timeout_sec'] = self.worker_done_timeout_sec
        if hasattr(self, 'shuffle_callables') and self.shuffle_callables:
            executor_kwargs['shuffle_callables'] = self.shuffle_callables
        if (hasattr(self, 'multiprocessing_start_method')
                and self.multiprocessing_start_method is not None
                and issubclass(cls, MultiprocessingCallableExecutor)):
            executor_kwargs['start_method'] = self.multiprocessing_start_method
        if getattr(self, 'devices', None) is not None:
            executor_kwargs['devices'] = self.devices
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = True
        executor_kwargs.update(kwargs)
        return executor_kwargs

    def __build__(self, *args, **kwargs):
        """Build all blocks using BlockMaker + the configured executor.

        Block formation (``__block__``) and building both happen inside
        the worker callables, so they are fully parallelized.
        """
        callables, callable_kwargs = self.__split__(*args, **kwargs)
        work_stealing_state = getattr(self, 'work_stealing', False)
        self.log.info(
            f"Building {self.__class__.__name__}: blocks using {len(callables)} callables, "
            f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}, work_stealing={work_stealing_state}"
        )
        executor_kwargs = self._executor_kwargs(
            tag=f"EXECUTING {len(callables)} callables [{self.__class__.__name__}]"
        )
        executor = self.executor_cls(**executor_kwargs)
        callable_results = executor.exec_callables(callables, self, **callable_kwargs)
        self.log.info(f"Stacking the results of {len(callable_results)} callables of {self.__class__.__name__}")
        result = self.__stack__(callable_results)
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return result

    def __split__(self, *args, **kwargs):
        callables = [self.BlockMaker(idx) for idx in range(self.n_blocks)]
        callable_kwargs = dict(build=True)
        return callables, callable_kwargs

    def __stack__(self, results=None):
        return self

    def _block_paths_topic(self) -> str | None:
        topics = self.topics()
        if 'block_paths' in topics:
            return 'block_paths'
        return None

    def _write_block_path(self, i: int):
        topic_name = self._block_paths_topic()
        if not topic_name:
            return
        block_dir = self.path(topic_name, ensure_dirpath=True)
        sentinel_path = os.path.join(block_dir, f"block_{i}.path")
        anchorkeypath = self.block(i).anchorkeypath
        with self.fs.open(sentinel_path, 'w') as f:
            f.write(anchorkeypath)
        if hasattr(self, '_built_block_set_cache'):
            self._built_block_set_cache.add(i)

    def _remove_block_path(self, i: int):
        topic_name = self._block_paths_topic()
        if not topic_name:
            return
        try:
            block_dir = self.path(topic_name)
            sentinel_path = os.path.join(block_dir, f"block_{i}.path")
            if self.fs.exists(sentinel_path):
                self.fs.rm(sentinel_path)
        except Exception:
            pass
        if hasattr(self, '_built_block_set_cache') and self._built_block_set_cache is not None:
            self._built_block_set_cache.discard(i)

    def _built_block_set(self) -> set[int]:
        if not hasattr(self, '_built_block_set_cache'):
            topic_name = self._block_paths_topic()
            if not topic_name:
                self._built_block_set_cache = set()
            else:
                try:
                    block_dir = self.path(topic_name)
                    if not self.fs.exists(block_dir):
                        self._built_block_set_cache = set()
                    else:
                        files = self.fs.ls(block_dir, detail=False)
                        indices = set()
                        for f in files:
                            fname = os.path.basename(f)
                            if fname.startswith('block_') and fname.endswith('.path'):
                                try:
                                    idx = int(fname.removeprefix('block_').removesuffix('.path'))
                                    indices.add(idx)
                                except ValueError:
                                    pass
                        self._built_block_set_cache = indices
                except Exception:
                    self._built_block_set_cache = set()
        return self._built_block_set_cache

    def _check_block_path(self, i: int) -> bool:
        topic_name = self._block_paths_topic()
        if not topic_name:
            return False
        if i in self._built_block_set():
            return True
        try:
            block_dir = self.path(topic_name)
            sentinel_path = os.path.join(block_dir, f"block_{i}.path")
            return self.fs.exists(sentinel_path)
        except Exception:
            return False

    def UNSAFE_clear_block(self, idx: int, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
        """Clear a single child block's data.

        Parameters
        ----------
        idx : int
            Index of the block to clear.
        *topics : str
            Forwarded to the block's ``UNSAFE_clear()``.
        OVERRIDE : bool
            If ``True``, skip the interactive confirmation.
        clear_dirpath : bool
            Forwarded to the block's ``UNSAFE_clear()``.
        """
        if not UNSAFE_allowed("UNSAFE_clear_block", OVERRIDE=OVERRIDE):
            return self.block(idx)

        blk = self.block(idx)
        return UNSAFE_clear_block_callable(blk, topics, clear_dirpath, stack=self, idx=idx)

    def UNSAFE_clear_blocks(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False, callable=UNSAFE_clear_block_callable):
        """Clear all block data, parallelized using the stack's builder settings.

        The interactive UNSAFE confirmation prompt is shown **once** at the
        stack level.  Individual ``block.UNSAFE_clear()`` calls are invoked
        with ``OVERRIDE=True`` so they do not re-prompt.

        Parameters
        ----------
        *topics : str
            Forwarded to each block's ``UNSAFE_clear()``.
        OVERRIDE : bool
            If ``True``, skip the interactive confirmation.
        clear_dirpath : bool
            Forwarded to each block's ``UNSAFE_clear()``.
        callable : callable, default UNSAFE_clear_block_callable
            Callable invoked per block to execute the clear operation.
        """
        if not UNSAFE_allowed("UNSAFE_clear_blocks", OVERRIDE=OVERRIDE):
            return self

        block_list = self.blocks()
        self.log.info(
            f"UNSAFE_clear_blocks: clearing {len(block_list)} blocks, "
            f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}"
        )
        self.write_journal_entry(event="UNSAFE_clear_blocks:begin")

        tag = f"CLEARING {len(block_list)} blocks [{self.__class__.__name__}, n_workers={self.n_workers}]"
        executor_kwargs = dict(n_workers=self.n_workers, tag=tag)
        if (hasattr(self, 'multiprocessing_start_method')
                and self.multiprocessing_start_method is not None
                and (self.parallelization or '').lower() in ('multiprocessing', 'torch_multiprocessing')):
            executor_kwargs['start_method'] = self.multiprocessing_start_method
        executor = callable_executor(self.parallelization, **executor_kwargs)

        callables = [functools.partial(callable, blk, topics, clear_dirpath, stack=self, idx=idx) for idx, blk in enumerate(block_list)]
        executor.exec_callables(callables)

        self.log.info(f"UNSAFE_clear_blocks complete: {self.__class__.__name__}")
        self.write_journal_entry(event="UNSAFE_clear_blocks:end")
        return self

    def UNSAFE_copy_blocks_from(self, anchorkeypath_callable, *, OVERRIDE: bool = False, overwrite: bool = False, topicpaths=None, validate: bool = True, always_copy_whole_dirpath: bool = False, callable=UNSAFE_copy_block_from_callable):
        """Copy each block's data from a per-block anchor path, parallelized using the stack's builder settings.

        Parameters
        ----------
        anchorkeypath_callable : callable
            Called as ``anchorkeypath_callable(block)`` for each block to obtain
            the ``anchorkeypath`` forwarded to that block's ``UNSAFE_copy_from()``.
        OVERRIDE : bool
            If ``True``, skip the interactive confirmation.
        overwrite, topicpaths, validate, always_copy_whole_dirpath :
            Forwarded to each block's ``UNSAFE_copy_from()``.
        callable : callable, default UNSAFE_copy_block_from_callable
            Callable invoked per block to execute the copy operation.
        """
        if not UNSAFE_allowed("UNSAFE_copy_blocks_from", OVERRIDE=OVERRIDE):
            return self

        block_list = self.blocks()
        work_stealing_state = getattr(self, 'work_stealing', False)
        self.log.info(
            f"UNSAFE_copy_blocks_from: copying {len(block_list)} blocks, "
            f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}, work_stealing={work_stealing_state}"
        )
        self.write_journal_entry(event="UNSAFE_copy_blocks_from:begin")

        blocks_iter = tqdm.tqdm(block_list, desc="UNSAFE_copy_blocks_from", unit="block", disable=not self.log_volume.info)
        callables = [
            functools.partial(callable, blk, anchorkeypath_callable(blk), overwrite, topicpaths, validate, always_copy_whole_dirpath)
            for blk in blocks_iter
        ]

        tag = f"COPYING {len(block_list)} blocks [{self.__class__.__name__}, n_workers={self.n_workers}]"
        executor_kwargs = dict(n_workers=self.n_workers, tag=tag)
        if hasattr(self, 'worker_done_timeout_sec') and self.worker_done_timeout_sec is not None:
            executor_kwargs['worker_done_timeout_sec'] = self.worker_done_timeout_sec
        if hasattr(self, 'shuffle_callables') and self.shuffle_callables:
            executor_kwargs['shuffle_callables'] = self.shuffle_callables
        if (hasattr(self, 'multiprocessing_start_method')
                and self.multiprocessing_start_method is not None
                and issubclass(self.executor_cls, MultiprocessingCallableExecutor)):
            executor_kwargs['start_method'] = self.multiprocessing_start_method
        if getattr(self, 'devices', None) is not None:
            executor_kwargs['devices'] = self.devices
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = True
        executor = self.executor_cls(**executor_kwargs)
        executor.exec_callables(callables)

        self.log.info(f"UNSAFE_copy_blocks_from complete: {self.__class__.__name__}")
        self.write_journal_entry(event="UNSAFE_copy_blocks_from:end")
        return self

    def UNSAFE_redirect_blocks(self, *, redirector: Callable = None, filter: dict = {}, validate: bool = False, OVERRIDE: bool = False, parallelization=None, n_workers=None):
        """Redirect each child block in the stack using redirector(block, stack, idx, journal=journal) callable.

        Parameters
        ----------
        redirector : Callable
            Callable with signature ``redirector(block, stack, idx, journal=journal) -> dict | None``.
            Returns kwargs for ``block.UNSAFE_redirect(**target)``, or None/empty if not redirecting.
        filter : dict, default {}
            Column filter kwargs passed to ``journal()`` when reading the child block's journal.
        validate : bool, default False
            If True, validates each block after redirection and considers invalid blocks as failures.
        OVERRIDE : bool, default False
            Must be True to allow unsafe redirection.
        parallelization : str, optional
            CallableExecutor to use. Defaults to ``self.parallelization``.
        n_workers : int, optional
            Worker count for parallel redirection and journal scanning.

        Returns
        -------
        tuple[int, int]
            (successes, total)
        """
        allowed = UNSAFE_allowed("UNSAFE_redirect_blocks", OVERRIDE=OVERRIDE)
        if redirector is None:
            raise ValueError("UNSAFE_redirect_blocks requires a redirector callable")
        block_list = self.blocks()
        total = len(block_list)
        if not allowed:
            return 0, total

        par = parallelization if parallelization is not None else getattr(self, 'parallelization', None)
        nw = n_workers if n_workers is not None else getattr(self, 'n_workers', 1)
        self.log.info(
            f"UNSAFE_redirect_blocks: redirecting {total} blocks, "
            f"parallelization={par}, n_workers={nw}"
        )
        self.write_journal_entry(event="UNSAFE_redirect_blocks:begin")

        try:
            blk0 = block_list[0] if total > 0 else None
            journal = blk0.journal(n_workers=nw, **(filter or {})) if blk0 is not None else None
        except Exception as e:
            self.log.detailed(f"UNSAFE_redirect_blocks: journal() lookup: {e}")
            journal = None

        tag = f"REDIRECTING {total} blocks [{self.__class__.__name__}, n_workers={nw}]"
        executor = callable_executor(par, n_workers=nw, tag=tag)

        callables = [functools.partial(_UNSAFE_redirect_block_callable, redirector, blk, self, idx, journal=journal, validate=validate) for idx, blk in enumerate(block_list)]
        results = executor.exec_callables(callables)

        successes = sum(1 for r in results if r is True) if results else 0

        self.log.info(f"UNSAFE_redirect_blocks complete: {self.__class__.__name__} ({successes}/{total} succeeded)")
        self.write_journal_entry(event="UNSAFE_redirect_blocks:end", note=f"{successes}/{total}")
        return successes, total


def _UNSAFE_redirect_block_callable(redirector, block, stack, idx, *, journal: Datajournal|None = None, validate: bool = False):
    target = redirector(block, stack, idx, journal=journal)
    if not target:
        return False
    kwargs = dict(target)
    if 'validate' not in kwargs:
        kwargs['validate'] = validate
    if 'journal' not in kwargs:
        kwargs['journal'] = journal
    kwargs['OVERRIDE'] = True
    redirected = block.UNSAFE_redirect(**kwargs)
    if validate and redirected:
        validated = stack.validate_block(idx)
        if not validated:
            return False
    return bool(redirected)



def _fscopy_item_callable(src_item, dst_item, storage_options):
    """Module-level callable for parallel directory copy in UNSAFE_copy_from.

    Copies a single item (file or subdirectory) from *src_item* to *dst_item*.
    When both endpoints are remote, a per-item temporary directory is used and
    removed immediately after the upload, so disk space is never accumulated
    across all parallel workers.
    """
    src_fs, _ = fsspec.url_to_fs(src_item, **(storage_options or {}))
    dst_fs, _ = fsspec.url_to_fs(dst_item, **(storage_options or {}))

    # Ensure the destination parent directory exists
    dst_parent = dst_item.rstrip('/').rsplit('/', 1)[0]
    if dst_parent:
        try:
            dst_fs.makedirs(dst_parent, exist_ok=True)
        except Exception:
            pass

    src_proto = getattr(src_fs, 'protocol', ())
    dst_proto = getattr(dst_fs, 'protocol', ())
    if 'file' in src_proto or 'file' in dst_proto:
        if 'file' in src_proto:
            dst_fs.put(src_item, dst_item, recursive=True)
        else:
            src_fs.get(src_item, dst_item, recursive=True)
    else:
        # Both endpoints are remote: stage through a per-item temp dir and
        # delete it immediately once the upload is done.
        # NOTE: dst_fs.put(local_path, remote_path) triggers a recursive call chain
        # in adlfs (_put → super._put → _put_file → ...).  Use dst_fs.open('wb')
        # streaming instead, which goes through the write API and avoids that path.
        tmpdir = tempfile.mkdtemp()
        try:
            basename = src_item.rstrip('/').rsplit('/', 1)[-1] or 'item'
            local_tmp = os.path.join(tmpdir, basename)
            src_fs.get(src_item, local_tmp, recursive=True)
            if os.path.isdir(local_tmp):
                # Directory: walk and stream each file, preserving structure
                for dirpath, _dirs, files in os.walk(local_tmp):
                    for fname in files:
                        local_f = os.path.join(dirpath, fname)
                        rel = os.path.relpath(local_f, local_tmp).replace(os.sep, '/')
                        remote_f = dst_item.rstrip('/') + '/' + rel
                        remote_parent = remote_f.rsplit('/', 1)[0]
                        if remote_parent:
                            try:
                                dst_fs.makedirs(remote_parent, exist_ok=True)
                            except Exception:
                                pass
                        # adlfs bug: commit_block_list doesn't pass overwrite=True,
                        # so delete first to avoid ResourceExistsError on commit.
                        try:
                            dst_fs.rm(remote_f)
                        except Exception:
                            pass
                        with open(local_f, 'rb') as lf, dst_fs.open(remote_f, 'wb') as rf:
                            shutil.copyfileobj(lf, rf)
            else:
                # Single file: stream directly to destination.
                # adlfs bug: commit_block_list doesn't pass overwrite=True,
                # so delete first to avoid ResourceExistsError on commit.
                try:
                    dst_fs.rm(dst_item)
                except Exception:
                    pass
                with open(local_tmp, 'rb') as lf, dst_fs.open(dst_item, 'wb') as rf:
                    shutil.copyfileobj(lf, rf)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


