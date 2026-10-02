"""Core framework classes: Datablock, Datastack, and remote execution.

The journals blocks record their builds in are in :mod:`dbx.journals`.

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
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass, field, fields, asdict, replace
import datetime
import functools
import gc
import hashlib
import inspect
import os
import random
import re
import shutil
import sys
import tempfile
import threading
from typing import Callable, Optional, Union
import uuid
import warnings


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
    default_datalake,
    InlineCallableExecutor,
    LogVolume,
    MultiprocessingCallableExecutor,
    MultithreadingCallableExecutor,
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
    remote,
    size,
    write_str,
    write_yaml,
)
from .journals import (
    normalize_journal_frame,
    datajournal,
    journal,
    CallableStr,
    CallableSignature,
    CallableSig,
    CallableType,
    CallableTp,
    Block,
    DatajournalEntry,
    DatajournalFrame,
    one_datalake,
    Datajournal,
    Datalog,
    DEFAULT_DATAJOURNAL,
    JOURNAL_DATETIME_FORMAT,
    execjournal,
    read_exec_journal,
    write_exec_journal,
    filter_journal_frame,
)
__version__ = "0.2.0"

class _SentinelMeta_(type):
    """A sentinel that is its own class: one object, usable as a value and in an annotation.

    ``x: int | None | ABSENT = ABSENT`` works as ``str | None`` does, because the
    sentinel is a type -- and pickle and copy keep it one object, since a class
    is always taken by reference. It renders as its ``REPR`` and is truthy
    unless it says otherwise; it is never instantiated.
    """

    def __repr__(cls):
        return cls.__dict__.get('REPR', cls.__name__)

    __str__ = __repr__

    def __bool__(cls):
        return cls.__dict__.get('TRUTH', True)

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"{cls!r} is a sentinel: use the class itself, not an instance of it")


class ABSENT(metaclass=_SentinelMeta_):
    """Marks a key present on only ONE side of a :meth:`Datablock.diffsig`.

    Needed because diffsubsig reports typed values: a key whose value *is* ``None``
    and a key that is missing entirely would otherwise both come back as
    ``None``, which are very different findings -- "this setting changed to None"
    versus "this setting did not exist when that build ran".
    """
    REPR = '<absent>'
    TRUTH = False


class SAME(metaclass=_SentinelMeta_):
    """A :class:`Datablock.Specialization`'s anchor when it is its block's own.

    The default, and the behaviour from before a specialization could name one:
    the narrower block is looked for in the journal of the block declaring it.
    Named otherwise when that block was built under another anchor -- a class
    since renamed, whose fqcn is the directory its journal is in.

    A class used as a value, as the topic markers are, so that it can stand in
    an annotation -- ``anchor: str | SAME = SAME`` -- as ``None`` does in
    ``str | None``. See `_SentinelMeta_`.
    """


class LegacyTopicsWarning(UserWarning):
    """A class's TOPICS declared in a spelling from before the topic markers.

    The canonical leaves are DATAFILE, DATADICT, DATADIR, DATASLICE and
    SYNTHETIC; a plain dict groups them. A bare filename, DIRTOPIC, SYNTOPIC,
    SLICETOPIC, DIR and the list form still work and still render as they
    always did -- respelling moves the hash -- and they are what a
    Specialization reconstructing an older block says.
    """


class SIGNATURE_TOPICS(metaclass=_SentinelMeta_):
    """The key under which :meth:`Datablock.difftopics` reports a whole-rendering
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
    REPR = '<signature topics>'


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


class TopicMarkerMeta(type):
    """Renders a topic marker as the declaration that made it.

    A leaf reaches :meth:`Datablock._topics_signature_` as ``str(node)``, and the
    journal records ``str(TOPICS)``, which ``repr``s it -- so a marker has to
    spell itself the same way in both, and that spelling has to be text that
    :func:`literal_topics` reads back into this very class.
    """

    #: ``{name: marker}`` for every marker a declaration may name, so a recorded
    #: ``{'masks': DIR}`` reads back as the class DIR while ``{'masks': 'DIR'}``
    #: stays a topic stored in a file called ``DIR``.
    REGISTRY = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        # A parameterised marker -- DATASLICE(idx='int'), DATAFILE('m.json'),
        # DATADIR('a note') -- is a subclass carrying `columns`, `filename` or
        # `note`, and is not a name anything may be declared under: it renders
        # as the call that made it and reads back through that call.
        if not any(namespace.get(k) for k in ('columns', 'filename', 'note')):
            TopicMarkerMeta.REGISTRY.setdefault(name, cls)

    def __repr__(cls):
        columns = cls.__dict__.get('columns')
        if not columns:
            return cls.__name__
        return f"{cls.__name__}({_render_columns_(columns)})"

    __str__ = __repr__


class TOPICMARKER(metaclass=TopicMarkerMeta):
    """Base of the topic markers: :class:`DATADIR`, :class:`SYNTHETIC`, ``DATAFILE``, ``DATASLICE``.

    A marker says what a topic IS.  The sentinels say it by what value they
    happen to be -- :data:`DIRTOPIC` is ``None`` and :data:`SYNTOPIC` the empty
    tuple -- which a reader has to know by heart and which a filename can be
    mistaken for.  A marker is a CLASS and never a string, so ``{'masks': DIR}``
    and ``{'masks': 'DIR'}`` are different declarations and stay different
    everywhere: the first is a directory topic, the second a topic stored in a
    file named ``DIR``.  Which is why a filename renders quoted under the flag
    and a marker does not -- see :meth:`Datablock._topictext_`.

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


def _check_note_(kind, note):
    """Refuse a note that would render into an ambiguous type string."""
    if not isinstance(note, str) or not note:
        raise TypeError(f"{kind} note must be a non-empty string, got {note!r}")
    if '/' in note:
        raise ValueError(
            f"{kind} note {note!r} may not contain '/': the marker is rendered into "
            f"the type string, whose segments are '/'-joined, so a '/' would let two "
            f"declarations render alike and collide onto one hash"
        )


class _NotedMarkerMeta_(TopicMarkerMeta):
    """Makes ``DATADIR('per-frame masks')`` a marker carrying that note.

    The note is documentation that reaches the identity: it renders into the
    type string, so writing one re-keys the block that writes it.  That is the
    price of it being recorded at all, and it is paid only by a declaration
    that asks for it -- ``DATADIR`` bare renders exactly as ``DIR`` always has.

    A call with no note returns the class ITSELF rather than an empty subclass,
    so ``DATADIR() is DATADIR`` and the two spell the same thing.
    """

    def __call__(cls, note=None):
        if note is None:
            return cls
        _check_note_(cls.__name__, note)
        return _NotedMarkerMeta_(cls.__name__, (cls,), {'note': note})

    def __repr__(cls):
        note = cls.__dict__.get('note')
        return cls.__name__ if note is None else f"{cls.__name__}({note!r})"

    __str__ = __repr__


class SYNTHETIC(TOPICMARKER, metaclass=_NotedMarkerMeta_):
    """A synthetic topic: presented, never stored.  :data:`SYNTOPIC` as a marker.

    ``path()`` and ``dirpath()`` are both ``None``, nothing is created, listed,
    copied or cleared for it, and it is vacuously valid.

    Takes an optional note -- ``SYNTHETIC('derived on read')`` -- which renders
    and so re-keys.  Bare ``SYNTHETIC`` renders as itself, exactly as before,
    so no existing declaration moves.
    """

    #: Unset on the bare marker, and set by the call that parameterises it.
    note = None


class DATADIR(TOPICMARKER, metaclass=_NotedMarkerMeta_):
    """A directory topic: the topic IS the directory.  :data:`DIRTOPIC` as a marker.

    A location, unlike :class:`SYNTHETIC` -- a real directory that merely has no
    filename inside it -- and it may say what is in it::

        TOPICS = {'masks': DATADIR('one PNG per frame')}

    The base of every directory marker: ``DATASLICE`` is one, because a slice
    IS a directory, and so is the deprecated :class:`DIR`. Every test that asks
    whether a topic is a directory asks ``is_topicmarker(node, DATADIR)``.
    """

    #: Unset on the bare marker, and set by the call that parameterises it.
    note = None


class DIR(DATADIR):
    """Deprecated: :class:`DATADIR`, under the name it had first.

    A subclass that renders as ``DIR``, not an alias: the rendering is the
    identity, so ``DIR = DATADIR`` would re-key every block that ever declared
    ``DIR``. A class declaring it is warned (`LegacyTopicsWarning`); respell it
    ``DATADIR`` and keep ``DIR`` in a ``Specialization(spec={}, topics=...)`` to
    reach what was built under it.
    """


class _DataFileMeta_(TopicMarkerMeta):
    """Makes ``DATAFILE('rows.csv', 'one row per frame')`` a marker carrying both."""

    def __call__(cls, filename, note=None):
        cls._check_filename_(filename)
        if note is not None:
            _check_note_(cls.__name__, note)
        return _DataFileMeta_(cls.__name__, (cls,),
                             {'filename': filename, 'note': note})

    def __repr__(cls):
        filename = cls.__dict__.get('filename')
        if not filename:
            return cls.__name__
        note = cls.__dict__.get('note')
        rendered = repr(filename) if note is None else f"{filename!r}, {note!r}"
        return f"{cls.__name__}({rendered})"

    __str__ = __repr__


class DATAFILE(TOPICMARKER, metaclass=_DataFileMeta_):
    """A topic stored as one file, named and optionally described::

        TOPICS = {'rows': DATAFILE('rows.csv', 'one row per frame')}

    The marker spelling of a bare filename.  ``DATAFILE('rows.csv')`` and
    ``'rows.csv'`` name the same file, and differ only in what they render as
    -- which is why adopting the marker re-keys the block, and why the bare
    string goes on working untouched for every block that has not.

    The note is documentation that reaches the identity.  A topic's filename
    says where the data is and nothing about what it holds; the note is the
    sentence a reader would otherwise have to go and find, recorded where the
    journal shows it.  It renders, so it is in the hash: changing the wording
    re-keys the block, the same as changing anything else that renders.
    """

    #: Unset on the bare marker, and set by the call that parameterises it.
    #: A bare DATAFILE names no file -- see :func:`_topic_filename_`.
    filename = None

    #: Unset unless the declaration gave one.
    note = None

    @staticmethod
    def _check_filename_(filename):
        """Refuse a filename that is not one, or that would render ambiguously."""
        if not isinstance(filename, str) or not filename:
            raise TypeError(
                f"DATAFILE filename must be a non-empty string, got {filename!r}"
            )
        if '/' in filename:
            raise ValueError(
                f"DATAFILE filename {filename!r} may not contain '/': the marker is "
                f"rendered into the type string, whose segments are '/'-joined, so a "
                f"'/' would let two declarations render alike and collide onto one hash"
            )


class _DataDictMeta_(_DataFileMeta_):
    """Makes ``DATADICT('meta.json', run=dict(id='str'))`` a marker carrying that schema.

    A call returns a SUBCLASS rather than an instance, exactly as
    ``DATASLICE``'s does, so everything a TOPICS declaration holds is a class
    and one test -- :func:`is_topicmarker` -- recognises the lot of them.
    """

    def __call__(cls, filename, *mapping, **typed):
        if mapping and (typed or len(mapping) > 1 or not isinstance(mapping[0], dict)):
            raise TypeError(
                f"{cls.__name__} takes a filename and then its keys as keywords -- "
                f"{cls.__name__}('meta.json', id='str') -- or as a single mapping "
                f"when a key is not an identifier"
            )
        cls._check_filename_(filename)
        schema = dict(mapping[0]) if mapping else dict(typed)
        cls._check_schema_(schema)
        return _DataDictMeta_(cls.__name__, (cls,),
                             {'filename': filename, 'schema': schema})

    def __repr__(cls):
        filename = cls.__dict__.get('filename')
        if not filename:
            return cls.__name__
        schema = cls.__dict__.get('schema') or {}
        rendered = _render_schema_(schema)
        return f"{cls.__name__}({filename!r}{', ' + rendered if rendered else ''})"

    __str__ = __repr__


class DATADICT(DATAFILE, metaclass=_DataDictMeta_):
    """A topic stored as ONE FILE holding a dict, declared with its schema::

        TOPICS = {'meta': DATADICT('meta.json', rows='int',
                                   run=dict(id='str', started='str'))}

    A file topic, not a directory: the filename is the marker's first argument
    and is where the data lands, so ``DATADICT('meta.json')`` and the plain
    string ``'meta.json'`` name the same file.  What the marker adds is the
    SHAPE of what is in it -- which a bare filename says nothing about, leaving
    every reader to open the file to find out, and leaving a schema free to
    change under artifacts that go on claiming to be the same block.

    A value is a dtype name as a string, or a nested ``dict(...)`` for a nested
    key.  Both render into the type string --
    ``topic:meta=DATADICT('meta.json', rows='int', run=dict(id='str'))`` -- so
    adding, dropping, retyping or REORDERING a key re-keys the block, the same
    way ``DATASLICE``'s columns do.

    Documentation, not enforcement: nothing yet checks that what is written
    matches what is declared.  The declaration is still worth having before the
    check exists, because it is what the check will be written against, and
    because it is in the hash either way.
    """

    #: Unset on the bare marker, and set by the call that parameterises it.
    #: A bare DATADICT names no file and cannot locate a topic -- see
    #: :func:`_topic_filename_`.
    filename = None

    #: ``{key: dtype | {key: ...}}``.  Empty when the file's shape is declared
    #: nowhere, which is the bare filename's behaviour under this spelling.
    schema = {}

    @classmethod
    def _check_schema_(cls, schema, _path=()):
        """Refuse a key or dtype that would render into an ambiguous type string."""
        for key, dtype in schema.items():
            where = '.'.join((*_path, str(key)))
            if not isinstance(key, str):
                raise TypeError(f"DATADICT key {where} must be a string, got {key!r}")
            if '/' in key:
                raise ValueError(
                    f"DATADICT key {where!r} may not contain '/': see the filename rule"
                )
            if isinstance(dtype, dict):
                cls._check_schema_(dtype, (*_path, key))
                continue
            if not isinstance(dtype, str):
                raise TypeError(
                    f"DATADICT dtype for {where!r} must be a string or a nested "
                    f"dict(...), got {dtype!r}"
                )
            if '/' in dtype:
                raise ValueError(
                    f"DATADICT dtype {dtype!r} for {where!r} may not contain '/': "
                    f"see the filename rule"
                )


#: One node of a TOPICS declaration: a marker -- ``DATAFILE('x')``, ``DATADICT(...)``,
#: ``DATADIR``, ``DATASLICE(...)``, ``SYNTHETIC`` -- or, in the older spellings a
#: Specialization may still have to say, a bare filename, ``DIRTOPIC`` (None),
#: ``SYNTOPIC`` (``()``) or ``SLICETOPIC``. A dict groups nodes under a name.
TopicNode = Union[type[TOPICMARKER], str, None, tuple[()], dict[str, 'TopicNode']]

#: A TOPICS declaration: ``{topic name: TopicNode}``.
Topics = dict[str, TopicNode]


class forward_property:
    """A class attribute declared forward: what every instance is, refined per instance.

    On the class it is *classvalue*, the forward declaration: what a caller
    holding only the class can be told, and a promise every instance keeps. On
    an instance it is the decorated method's value, which may say more --
    computed once, checked against the promise, and cached in the instance's
    ``__dict__`` under its own name, as with `functools.cached_property`.

    Made for a TOPICS the instance declares. A table reads its TAB's slice
    NAMES off the TAB class, before any tab exists; a tab's full declaration --
    the slice's columns -- follows from its VAR::

        @forward_property({'features': DATASLICE})
        def TOPICS(self):
            return {'features': DATASLICE({c: 'ndarray:float32' for c in self.feature_for_column})}

    The class says "a ``features`` slice"; each tab says which columns. A plain
    `property` cannot: on the class it is the property object, and a table
    reading one finds no slices and says nothing.

    The promise, as `refines` checks it -- raising here, when the instance's
    value is computed, rather than wherever it would otherwise fail:

    - a class (a topic marker included): the instance's value is it, a
      subclass of it, or an instance of it -- ``DATASLICE(final='ndarray:float32')``
      refines ``DATASLICE``, a part's ``Featuretab`` refines ``Datatab``, and
      ``'final'`` refines ``str``;
    - a dict: every key it declares is the instance's too, each refined in
      turn; the instance may declare more;
    - None: no promise;
    - anything else: equal, by value or as rendered.

    A non-data descriptor -- no ``__set__`` -- so assigning the attribute on an
    instance replaces it, as it would a cached_property. It is not pickled with
    the block (`Datablock.__getstate__` keeps the explicit parameters only), so
    an unpickled block computes it again from its own VAR.
    """

    def __init__(self, classvalue):
        self.classvalue = classvalue
        self.func = None
        self.name = None

    def __call__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        return self

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self.classvalue
        if self.func is None:
            raise TypeError(f"forward_property {self.name!r} decorates no method")
        value = self.func(obj)
        broken = refines(value, self.classvalue)
        if broken is not None:
            path, why = broken
            raise TypeError(
                f"{type(obj).__qualname__}.{self.name}{''.join(f'[{k!r}]' for k in path)}: {why}. "
                f"The class declares {self.name} = {self.classvalue!r} forward -- a promise every "
                f"instance keeps -- and this instance computed {value!r}."
            )
        obj.__dict__[self.name] = value
        return value

    def named(self, name: str) -> 'forward_property':
        """The same declaration under another attribute name -- as a table's TAB is its BLOCK."""
        other = forward_property(self.classvalue)(self.func)
        other.__set_name__(None, name)
        return other


def refines(value, promise, _path=()):
    """None when *value* keeps the forward declaration *promise*; else ``(path, why)``.

    See `forward_property` for what each kind of promise asks.
    """
    if promise is None:
        return None
    if isinstance(promise, dict):
        if not isinstance(value, dict):
            return _path, f"is {value!r}, where a dict is declared"
        for key, sub in promise.items():
            if key not in value:
                return _path, f"lacks {key!r}, which is declared"
            broken = refines(value[key], sub, _path + (key,))
            if broken is not None:
                return broken
        return None
    if isinstance(promise, type):
        if value is promise or (isinstance(value, type) and issubclass(value, promise)) \
                or isinstance(value, promise) or str(value) == str(promise):
            return None
        return _path, f"is {value!r}, which is not a {promise!r}"
    if value == promise or str(value) == str(promise):
        return None
    return _path, f"is {value!r}, not the declared {promise!r}"


def class_declaration(cls, name: str):
    """What *cls* itself declares under *name*, a `forward_property` read as its class value."""
    value = cls.__dict__.get(name)
    return value.classvalue if isinstance(value, forward_property) else value


def _check_forward_override_(cls, name: str) -> None:
    """Refuse a class that repeats an inherited `forward_property`'s class value as a plain attribute.

    Python finds an attribute on the class before its bases', so a plain one on
    a subclass hides the descriptor -- for the class AND for every instance. A
    copy of the forward declaration therefore replaces each instance's own
    value with the placeholder, and nothing says so: for a `Featuretab`, every
    tab then declares a ``features`` slice without columns, its hash moves, and
    whatever reads the columns fails far from here. A DIFFERENT value is taken
    as a deliberate override, and stands.
    """
    own = cls.__dict__.get(name)
    if own is None or isinstance(own, forward_property):
        return
    for base in cls.__mro__[1:]:
        inherited = base.__dict__.get(name)
        if inherited is None:
            continue
        if isinstance(inherited, forward_property) and str(own) == str(inherited.classvalue):
            raise TypeError(
                f"{cls.__qualname__}.{name} = {own!r} only repeats {base.__qualname__}.{name}'s "
                f"forward declaration, and hides it.\n"
                f"  {base.__qualname__}.{name} is a forward_property: {own!r} is what it says on "
                f"the class, while each instance computes its own, fuller value"
                + (f" -- {inherited.__doc__.strip().splitlines()[0].rstrip('.')}" if inherited.__doc__ else "") + ".\n"
                f"  A plain attribute here is found before {base.__qualname__}'s, on the class and on "
                f"every instance, so each {cls.__qualname__} would read the placeholder instead -- "
                f"changing its identity and losing what the instance would declare.\n"
                f"  Delete the line: {cls.__qualname__} inherits the declaration. To declare "
                f"something else, assign a different value, or a forward_property of its own."
            )
        return


def is_topicmarker(node, kind=TOPICMARKER):
    """True when *node* is the marker *kind*, or a parameterisation of it.

    Markers are classes, so this is the test that keeps a filename out: a
    string is never a marker however it is spelled.
    """
    return isinstance(node, type) and issubclass(node, kind)


def _render_columns_(columns):
    """A marker's ``columns`` as the call arguments that reconstruct it.

    Keyword form when every name is an identifier, which is what a declaration
    almost always looks like, and the mapping form when one is not -- ``DATASLICE``
    accepts both, so either rendering reads back as the same marker.  A column
    declared by the structure of the dict it holds renders as ``dict(...)``, as a
    ``DATADICT`` nested key does; a column declared by its MDS type renders as it
    always has.
    """
    if all(isinstance(name, str) and name.isidentifier() for name in columns):
        return ', '.join(f"{name}=dict({_render_schema_(coltype)})" if isinstance(coltype, dict)
                         else f"{name}={coltype!r}"
                         for name, coltype in columns.items())
    return repr(dict(columns))


def _render_schema_(schema):
    """A ``DATADICT`` schema as the call arguments that reconstruct it.

    Keyword form when every key is an identifier, and the mapping form when one
    is not -- the same rule :func:`_render_columns_` follows, and for the same
    reason: either rendering reads back as the same marker.  A nested key
    renders as ``dict(...)`` under the keyword form and as a plain dict literal
    under the mapping form, and :func:`_topic_literal_` reads both.
    """
    if not schema:
        return ''
    if not all(isinstance(key, str) and key.isidentifier() for key in schema):
        return repr(dict(schema))
    return ', '.join(
        f"{key}=dict({_render_schema_(dtype)})" if isinstance(dtype, dict)
        else f"{key}={dtype!r}"
        for key, dtype in schema.items()
    )


def _topic_filename_(node):
    """The filename a topic leaf stores its data under.

    A plain string IS the filename; a :class:`DATAFILE` carries one, and so
    does every marker that is one -- :class:`DATADICT` included.  Every
    other leaf -- a directory, a synthetic topic -- has no filename and never
    reaches here, since the callers test for those first.
    """
    if is_topicmarker(node, DATAFILE):
        if not node.filename:
            raise ValueError(
                f"a bare {node.__name__} names no file and so locates no topic; "
                f"declare it with one -- {node.__name__}('meta.json') -- or use the "
                f"filename alone"
            )
        return node.filename
    return node


def _legacy_topics_(topics, prefix=()):
    """The topics of a TOPICS declaration spelled the pre-marker way, as ``'a/b'`` paths."""
    if isinstance(topics, list):
        return [f"the list form ({', '.join(map(str, topics))})" if topics else "the list form"]
    if not isinstance(topics, dict):
        return []
    out = []
    for name, node in topics.items():
        path = prefix + (str(name),)
        if isinstance(node, dict):
            out.extend(_legacy_topics_(node, path))
        elif not is_topicmarker(node) or node is DIR:
            out.append('/'.join(path))
    return out


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
    return _topic_literal_(ast.parse(text, mode='eval').body)


def _topic_literal_(node):
    """One :mod:`ast` node of a recorded TOPICS as the value it stands for."""
    if isinstance(node, ast.Dict):
        return {_topic_literal_(k): _topic_literal_(v)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Name):
        return _marker_named_(node.id)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"recorded TOPICS: {ast.dump(node.func)} is not a marker")
        if node.func.id == 'dict' and not node.args:
            # A DATADICT nested key renders as dict(...). Read as the mapping it
            # spells, not through the registry: `dict` is not a marker, and this
            # stays a parse -- the keywords are themselves literals or dict()s.
            return {kw.arg: _topic_literal_(kw.value) for kw in node.keywords}
        marker = _marker_named_(node.func.id)
        args = [_topic_literal_(arg) for arg in node.args]
        return marker(*args, **{kw.arg: _topic_literal_(kw.value) for kw in node.keywords})
    return ast.literal_eval(node)


def _marker_named_(name):
    """The marker declared under *name*, or a ValueError naming what is known."""
    marker = TopicMarkerMeta.REGISTRY.get(name)
    if marker is None:
        raise ValueError(
            f"recorded TOPICS names {name!r}, which is not a topic marker; "
            f"known markers are {sorted(TopicMarkerMeta.REGISTRY)}"
        )
    return marker


def valid(*args, n_workers=None, summary=False, datalake=None, events=None, url=None, **kwargs):
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
        parser.add_argument("--datalake", "--url", dest="url", type=str, default=None, help="The datalake")
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
        url = one_datalake(datalake, url, 'valid')
        if url is not None:
            j_kwargs['datalake'] = url

        try:
            j = datajournal(anchor_key, **j_kwargs)
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


class _ClassOrInstance_:
    """A read-only attribute that answers on the class and on an instance alike.

    ``Cls.x`` is ``for_class(Cls)`` and ``obj.x`` is ``for_instance(obj)`` --
    so what a class says about all its instances (its anchor) and what one
    instance says about itself (an ``anchor=`` it was given) are one name.
    """

    def __init__(self, for_class, for_instance, doc=None):
        self.for_class, self.for_instance = for_class, for_instance
        self.__doc__ = doc

    def __get__(self, obj, owner=None):
        return self.for_class(owner) if obj is None else self.for_instance(obj)


#: The ``use_specializations`` a stack's ``use_block_specializations`` gives the
#: blocks it is forming, innermost last -- per thread, since blocks are formed
#: in parallel. A block's own ``use_specializations=`` wins over it.
_BLOCK_SPECIALIZATIONS = threading.local()


#: The journals stacks have read for the blocks they are forming, innermost
#: last -- per thread -- each as a `BlocksJournal`. A block of that anchor, in
#: that datalake, resolves its specializations against it instead of reading
#: the journal itself: which is what a block formed in a WORKER needs, since
#: the stack that read it is in another process.
_FORMING_JOURNALS = threading.local()

#: What builds a `Datajournal` or `DatajournalFrame` is named for it: old name -> new.
_RENAMED_DATAJOURNAL_BUILDERS_ = {
    'journal': 'datajournal',
    'Journal': 'Datajournal',
    'block_journal': 'block_datajournal',
    'tab_journal': 'tab_datajournal',
    'child_specialization_journal': 'child_specialization_datajournal',
}


class BlocksJournal:
    """A journal read once for a stack's blocks: those of *anchor*, stored in *datalake*.

    A plain class and not a namedtuple: it travels as a call argument, and a
    tuple there is taken apart and rebuilt, element by element, by `dbx.eval`.
    """
    __slots__ = ('journal', 'anchor', 'datalake')

    def __init__(self, journal, anchor, datalake):
        self.journal, self.anchor, self.datalake = journal, anchor, datalake

    def __getstate__(self):
        return (self.journal, self.anchor, self.datalake)

    def __setstate__(self, state):
        self.journal, self.anchor, self.datalake = state

    def __repr__(self):
        n = 'no' if self.journal is None else len(self.journal)
        return f"BlocksJournal({self.anchor!r} in {self.datalake!r}, {n} entries)"


@contextlib.contextmanager
def forming_with_journal(blocks_journal):
    """Blocks formed inside resolve against *blocks_journal* -- when it is theirs.

    What a block-forming callable wraps its forming in, having been handed the
    journal its stack read -- as a ctx kwarg, once per worker. None forms
    blocks as they would be formed anyway.
    """
    if blocks_journal is None or blocks_journal.journal is None:
        yield
        return
    stack = getattr(_FORMING_JOURNALS, 'stack', None)
    if stack is None:
        stack = _FORMING_JOURNALS.stack = []
    stack.append(blocks_journal)
    try:
        yield
    finally:
        stack.pop()


def _forming_journal_(anchor, datalake):
    """The journal a stack forming this block read for it, or None -- see `forming_with_journal`."""
    for bj in reversed(getattr(_FORMING_JOURNALS, 'stack', None) or ()):
        if bj.anchor == anchor and bj.datalake == datalake:
            return bj.journal
    return None


#: Pushed by a stack forming a block with no setting of its own, so that an
#: outer stack's -- or a query's -- does not reach it: a block formed for a
#: stack's cache must be formed as that stack forms it, whatever is in progress
#: around it.
_OWN_SETTING = object()


def _forming_with_specializations_():
    """The ``use_specializations`` the innermost stack forming a block asks for, or None."""
    stack = getattr(_BLOCK_SPECIALIZATIONS, 'stack', None)
    top = stack[-1] if stack else None
    return None if top is _OWN_SETTING else top


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

        self._url_ = URL as supplied to the dunders (a specline, or None);
                     the underscores are theirs, worn by what was supplied
        self.url   = that resolved to a real URL string
        self.fs    = fsspec filesystem object
        self.root  = protocol-free path (via fsspec.url_to_fs)
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
        'hash', 'code', 'tree', 'session', 'id',
        'datetime', 'build:start:datetime', 'build:end:datetime',
        'version', 'dbx_version', 'revision',
        'datalake', 'anchor', 'keyby', 'key', 'anchorkeypath', 'tag',
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

    @dataclass
    class VAR:
        class LazyLoader:
            """One VAR field: its term, resolved on first read and kept.

            Also the one place every VAR value passes on its way to the
            identity -- hash -> type() -> signature() -> _typed_specdict_() ->
            getattr(self.var, name) -- which is why the check that a value CAN
            be rendered deterministically lives here rather than at the two
            places that render one.
            """

            #: Distinct from None, which is an ordinary resolved value and a very
            #: common VAR default. With None as the sentinel a field holding one
            #: re-resolves on every read, so a specline evaluating to None was
            #: re-eval'd forever.
            _UNSET = object()

            # 1. Protocol and hooks ----------------------------------------

            def __init__(self, term, name=None, owner=None, exempt=False):
                self.term = term
                self.name = name
                self.owner = owner
                self.exempt = exempt
                self.value = self._UNSET

            def __call__(self):
                if self.value is self._UNSET:
                    if isinstance(self.term, str):
                        try:
                            self.value = dataparts.eval(self.term)
                        except Exception as e:
                            # Evaluated lazily -- often deep inside computing an
                            # identity -- so say which field of which block.
                            e.add_note(f"while evaluating {self.owner or ''}.VAR.{self.name or '<field>'} "
                                       f"= {self.term!r}")
                            raise
                    else:
                        # from_datablockable passes raw Python objects
                        self.value = self.term
                    self._check_renderable_()
                return self.value

            # 4. Helpers ---------------------------------------------------

            def _check_renderable_(self):
                """Refuse a value the identity cannot render deterministically.

                A VAR leaf that is neither a Datablock nor plain data falls
                through to repr() when the signature is rendered, and for a
                class with no content-bearing __repr__ that is
                ``<C object at 0x...>``: a memory address, inside the string the
                hash is taken over. The block then hashes differently on every
                construction, and since set() is a deepcopy-and-reconstruct the
                address moves on every call -- so a Datastack writes its children
                under one key and looks them up under another, finds nothing,
                and rebuilds over the top of what it already has.

                A specline is exempt whatever it resolves to: the identity
                renders the LINE for anything that does not resolve to a block,
                so the resolved object never reaches it.

                Structural rather than a search for the address. A __repr__
                added to quiet the symptom leaves the hole open -- two values
                with different content and one repr collide onto a single hash,
                and nothing here could tell.
                """
                if self.exempt or Datablock.is_specline(self.term):
                    return
                if self._renders_deterministically_(self.value):
                    return
                owner = self.owner or 'VAR'
                name = self.name or '<field>'
                raise TypeError(
                    f"{owner}.VAR.{name} holds a {type(self.value).__name__}, which "
                    f"this block's identity cannot render deterministically: a leaf "
                    f"that is neither a Datablock nor plain data reaches the "
                    f"signature through repr(), and an object without a "
                    f"content-bearing __repr__ renders its memory address there. The "
                    f"block would hash differently on every construction, and a stack "
                    f"would look its children up under a key it never wrote. Hold a "
                    f"Datablock, or a specline naming one, or reduce it to plain "
                    f"data; or record the exemption as {owner}.VAR_IDENTITY_EXEMPTIONS "
                    f"= {{{name!r}}}, which accepts whatever repr() makes of it"
                )

            @classmethod
            def _renders_deterministically_(cls, value):
                """True when repr(*value*) is a function of its content alone.

                Deliberately a whitelist. Everything on it renders the same in
                any process, from any address, so two blocks configured alike
                hash alike -- which is the whole of what identity promises.
                """
                if isinstance(value, Datablock):
                    return True
                if value is None or isinstance(value, (str, bytes, bool, int, float)):
                    return True
                if isinstance(value, (list, tuple, set, frozenset)):
                    return all(cls._renders_deterministically_(v) for v in value)
                if isinstance(value, dict):
                    return all(cls._renders_deterministically_(k)
                               and cls._renders_deterministically_(v)
                               for k, v in value.items())
                return False

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if isinstance(attr, Datablock.VAR.LazyLoader):
                return attr()
            return attr

    # DEPRECATED ALIAS: subclasses used to declare `class CONFIG(Datablock.CONFIG)`.
    # The name is kept so those declarations still resolve; __setstate__ maps a
    # subclass-declared CONFIG onto self.VAR (see _resolve_legacy_CONFIG_).
    CONFIG = VAR

    #: Spec keys whose upstream subtree the tree walks must not descend into:
    #: ``TREE_SKIP_VALIDATION`` for ``valid_var()``/``valid_tree()``,
    #: ``TREE_SKIP_BUILDING`` for ``build_tree()``.  A pair, named alike,
    #: because they are the same idea applied to the two walks -- and a field
    #: is routinely in both, as a warm-start source is: do not train it on my
    #: behalf, and do not hold its unfinished state against me.
    #:
    #: Tuples, both, and tuples in subclasses too.  Only membership is ever
    #: asked of them, so a set would do; declaring them alike is what stops a
    #: reader wondering what the difference is meant to mean.
    #:
    #: ``TREE_SKIP_VALIDATION`` supersedes the retired VALIDATE_CFG_EXEMPTIONS;
    #: ``TREE_SKIP_BUILDING`` supersedes BUILD_TREE_EXEMPTIONS.
    TREE_SKIP_VALIDATION = ()
    TREE_SKIP_BUILDING = ()

    #: Where this block coincides with a NARROWER one that was already built.
    #:
    #: A class grows: a new VAR field, a new topic, a bumped VERSION. Every
    #: block of it re-keys, and the topics that did not change are rebuilt for
    #: nothing. A :class:`Specialization` says that when the new fields hold
    #: the values given in its ``spec``, the topics it names ARE the topics of
    #: the block this class used to be -- whose identity is this one's with
    #: those fields dropped, those topics alone, and its own ``version`` if it
    #: names one. That block's hash is reconstructible from here (see
    #: :meth:`get_hash`), so its build can be found in the journal and read
    #: instead of repeated.
    #:
    #: Declared in preference order; the first one that both matches and
    #: resolves is the one used::
    #:
    #:     SPECIALIZATIONS = [
    #:         Datablock.Specialization(
    #:             spec=dict(window='hann'),
    #:             topics={'spectra': 'spectra.npy'},   # as the narrower block declared them
    #:             note="hann was the only window before the field existed",
    #:         ),
    #:     ]
    #:
    #: It is a claim about semantics that no reader can check from the code --
    #: that the two computations produce the same bytes -- which is what
    #: ``note`` is for.
    SPECIALIZATIONS = []

    #: Whether :attr:`SPECIALIZATIONS` are consulted, and whether installing one
    #: is recorded. ``True`` installs and records it in the journal, through
    #: :meth:`UNSAFE_redirect`; ``'memory'`` installs it on the instance and
    #: writes nothing; ``False`` declines to look.
    #:
    #: Recording is the default because an installed specialization is a claim
    #: about where a block's data came from, and a claim that is nowhere written
    #: down cannot be checked afterwards -- the journal entry is the only record
    #: that this block read another's build, and which specialization said it
    #: could. It also costs less, not more: the record includes the hidden
    #: ``.redirection`` topic, which every later construction reads instead of
    #: scanning the journal again.
    #:
    #: The per-instance ``use_specializations=`` overrides it; this is the
    #: default, and it lives here because it belongs next to the declaration it
    #: switches off.
    USE_SPECIALIZATIONS = True

    #: Constructor parameters that are NOT part of what a block IS: aids handed
    #: in to save it work, rather than properties of it. They are in the
    #: signature, so they are discoverable and typed like any other parameter,
    #: and out of :meth:`__explicit_params__` -- hence out of `parameters`,
    #: `dfn`, `quote()`, `cite()` and the journal record -- because a block
    #: reconstructed from any of those must come back the same block, and one
    #: that came back carrying a stale journal would not.
    TRANSIENT_PARAMS = ('specialization_journal', 'url', 'capture_output')

    #: VAR field names exempt from :meth:`VAR.LazyLoader._check_renderable_` --
    #: the check that a value can be rendered into the identity deterministically.
    #: An exemption does not take the field out of the identity: it goes on
    #: rendering through repr(), address and all, and the hash moves with it. It
    #: says only that this class knows, which is why it is written on the class
    #: rather than switched on from the environment -- it belongs where the next
    #: reader of the declaration will see it.
    VAR_IDENTITY_EXEMPTIONS = frozenset()

    #REDIRECT: BEGIN
    @dataclass(frozen=True)
    class Specialization:
        """One coincidence between this block and a narrower, already-built one.

        *spec* pins the VAR fields the narrower block never had, to the values
        at which the two computations agree. *topics* names the topics that
        come from it -- MY names, which are also its names -- in the order it
        declared them, since that order is in its identity. It is a DICT, the
        narrower block's own declaration -- ``{'tiles': SLICETOPIC, 'meta':
        'meta.json'}`` -- and is rendered as that, in that declaration's era:
        nothing of the narrower block's topics is taken from this class, which
        is what lets a specialization describe a block whose topics were
        SPELLED differently, as every block from before the topic markers was.
        (A Datatable still adds its TAB's slices, as its own identity always did.) *version* is the
        :attr:`VERSION` the narrower block carried, left :data:`ABSENT` to
        inherit this class's. The sentinel rather than ``None``, because
        ``None`` is what :attr:`version` reads as for a class that declares no
        VERSION at all -- a real value a specialization has to be able to name,
        and the one a class names on the day it starts versioning. *note* says
        why the coincidence holds; nothing else records it.

        Those three are the whole of an identity -- :meth:`typestr` is built from
        the spec, the version and the topics and nothing else -- so a
        specialization that names all three describes the narrower block
        COMPLETELY, rather than inheriting whatever the class happens to carry
        now. That is what *version* is for: without it a VERSION bump moved the
        reconstructed hash onto a block nobody ever built, and the only way to
        keep a specialization working was never to bump. The two cases it makes
        expressible are a bump that changed some topics and not others, and a
        bump that turned out not to change any.

        A pin is matched against the RENDERED spec (`_typed_specdict_`), not
        against the raw ``spec`` dict a caller passed: a field left at its
        default is absent from that dict, and a field left at its default is
        exactly the case this exists for. *version* is not matched against
        anything -- it is a statement about the OTHER block, not a condition on
        this one.

        *anchor* is where the narrower block was built: :data:`SAME`, the
        default, for this block's own anchor, or the anchor it had -- typically
        a class's fqcn before a rename. It says which journal to look in and
        nothing else: an identity names no class, so the reconstructed hash is
        the same wherever it was built.
        """
        #: The VAR fields the narrower block never had, each at the value that
        #: makes the two computations agree: ``{field: value}``.
        spec: dict
        #: The narrower block's TOPICS declaration, ``{name: TopicNode}``, in its
        #: own spelling -- or, read back from a record, that rendered as text.
        #: SAME when it declared exactly what this block does, as after a pure
        #: VAR rename (`redirect_vars`) -- which a block whose TOPICS it computes
        #: per instance has no other way to say.
        topics: Topics | str | SAME
        #: The narrower block's VERSION, or ABSENT to take this class's.
        version: int | str | None | ABSENT = ABSENT
        #: The anchor whose journal the narrower block is looked for in: SAME
        #: for this block's own, or the one it was built under.
        anchor: str | SAME = SAME
        #: Legacy serialization flags to reconstruct historical identities.
        #: Can be a collection of flags (e.g. ['signature', 'typing']), ['all']
        #: (or True) for all legacy flags, or None.
        legacy: tuple[str, ...] | list[str] | str | bool | None = field(default=None, kw_only=True)
        #: Whether to redirect all topics recorded by the matched build entry,
        #: rather than restricting redirection to the subset named in `topics`.
        #:
        #: By default (False), a specialization only redirects the topics declared
        #: in its `topics` mapping, leaving any other topics to be built by this block.
        #: When True, all topics recorded in the matched build's journal entry are
        #: redirected.
        #:
        #: NOTE: This is an UNSAFE escape hatch intended strictly for historical
        #: corner cases where a build wrote and recorded multiple topics, but an
        #: earlier hash implementation underdeclared topics (e.g. hashed on a single
        #: topic like 'count' before instance topics were added to the hash).
        #: Its use is discouraged unless absolutely necessary to reach such legacy builds.
        UNSAFE_redirect_all_topics: bool = field(default=False, kw_only=True)
        #: Specific subset of topics to redirect from the matched build. If None,
        #: redirects all topics declared in `topics` (or all recorded topics if
        #: `UNSAFE_redirect_all_topics` is True).
        redirect_topics: tuple[str, ...] | list[str] | None = field(default=None, kw_only=True)
        #: Mapping from current VAR field names to historical ones,
        #: ``{current_name: historical_name}``.  Mirrors `redirect_topics`
        #: for vars: when a VAR field has been renamed, the specialization
        #: reconstructs the historical identity under the old key name so
        #: the hash matches what was built before the rename.
        redirect_vars: dict | None = field(default=None, kw_only=True)
        #: Why the coincidence holds. Last -- and keyword-only, so that it is
        #: last in a subclass's constructor too, after its BLOCK or TAB.
        note: str = field(default='', kw_only=True)

        # 1. Protocol and hooks --------------------------------------------

        def __post_init__(self):
            # Frozen, so the dataclass can live on a class and be shared, and
            # so a hash cached against it cannot go stale underneath.
            object.__setattr__(self, 'spec', dict(self.spec))
            leg = self.legacy
            if leg is True:
                leg = ('all',)
            elif leg is False or leg is None:
                leg = None
            elif isinstance(leg, str):
                leg = ('all',) if leg == 'all' else (leg,)
            elif isinstance(leg, (list, tuple, set, frozenset)):
                if not leg:
                    leg = None
                elif 'all' in leg:
                    leg = ('all',)
                else:
                    leg = tuple(sorted(set(leg)))
            else:
                raise TypeError(f"Specialization legacy= must be a list/tuple of strings, 'all', True, or None; got {leg!r}")

            if leg is not None:
                allowed = {'all', 'signature', 'typing'}
                unknown = set(leg) - allowed
                if unknown:
                    raise ValueError(f"Specialization legacy= contains unrecognized flag(s) {sorted(unknown)!r}; allowed flags are {sorted(allowed)!r}")

            object.__setattr__(self, 'legacy', leg)
            object.__setattr__(self, 'UNSAFE_redirect_all_topics', self.UNSAFE_redirect_all_topics)
            topics = self.topics
            if topics is SAME or topics == 'SAME':      # the record form of SAME -- see to_dict
                topics = SAME
            elif isinstance(topics, str):
                topics = literal_topics(topics)         # the record form -- see to_dict
            if topics is SAME:
                pass
            elif isinstance(topics, (list, tuple)) and self.legacy:
                topics = tuple(topics)
            elif not isinstance(topics, dict):
                raise TypeError(
                    f"Specialization topics={self.topics!r} names topics without declaring "
                    f"them. Declare each as the narrower block did -- "
                    f"topics={{'spectra': 'spectra.npy', 'tiles': SLICETOPIC}} -- since "
                    f"nothing of a narrower block's identity is taken from this class's TOPICS."
                )
            # {name: node}: the topics AS THAT BLOCK DECLARED THEM. Their names
            # are still mine; their nodes, and so the era they are spelled in,
            # are its -- which is what a migration from the sentinels to the
            # markers changes and a list of names cannot say.
            object.__setattr__(self, 'topics', topics if topics is SAME or isinstance(topics, tuple) else dict(topics))
            if self.anchor is not SAME and not (isinstance(self.anchor, str) and self.anchor):
                raise TypeError(f"Specialization anchor= is SAME or an anchor string, got {self.anchor!r}")
            rv = self.redirect_vars
            if rv is not None:
                if not isinstance(rv, dict):
                    raise TypeError(f"Specialization redirect_vars= must be a dict or None, got {rv!r}")
                object.__setattr__(self, 'redirect_vars', dict(rv))


        def __hash__(self):
            # frozen=True would hash the field tuple, and one of those fields is
            # a dict. `key` is the same information, flattened.
            return hash(self.key)

        def __eq__(self, other):
            # By key, across the family: a table's specialization read back from
            # its journal is a Datatable.Specialization, and equals the plain
            # Datablock.Specialization the class declared when its TAB is SAME.
            if not isinstance(other, Datablock.Specialization):
                return NotImplemented
            return self.key == other.key

        def __repr__(self):
            return (f"Specialization(spec={dict(self.spec)!r}, "
                    f"topics={self.topics!r}"
                    + (f", version={self.version!r}" if self.version is not ABSENT else "")
                    + (f", anchor={self.anchor!r}" if self.anchor is not SAME else "")
                    + (f", legacy={list(self.legacy)!r}" if self.legacy is not None else "")
                    + (f", UNSAFE_redirect_all_topics={self.UNSAFE_redirect_all_topics!r}" if self.UNSAFE_redirect_all_topics else "")
                    + (f", redirect_vars={self.redirect_vars!r}" if self.redirect_vars else "")
                    + ''.join(f", {n}={v!r}" for n, v in self._extra_fields_() if v is not SAME)
                    + (f", note={self.note!r}" if self.note else "") + ")")

        # 2. Declared API --------------------------------------------------

        def to_dict(self):
            """The record form: literal, so it round-trips through the journal -- see `from_record`."""
            # The declaration as its rendering, which literal_topics reads back:
            # a marker is not a literal, and a record has to be.
            d = {'spec': dict(self.spec), 'topics': 'SAME' if self.topics is SAME else str(self.topics)}
            if self.version is not ABSENT:
                d['version'] = self.version
            if self.anchor is not SAME:
                d['anchor'] = self.anchor
            if self.legacy is not None:
                d['legacy'] = list(self.legacy)
            if self.UNSAFE_redirect_all_topics:
                d['UNSAFE_redirect_all_topics'] = self.UNSAFE_redirect_all_topics
            if self.redirect_topics is not None:
                d['redirect_topics'] = list(self.redirect_topics)
            if self.redirect_vars is not None:
                d['redirect_vars'] = dict(self.redirect_vars)
            d.update((n, v) for n, v in self._extra_fields_() if v is not SAME)
            if self.note:
                d['note'] = self.note
            return d

        @classmethod
        def from_record(cls, record: dict) -> 'Datablock.Specialization':
            """A Specialization from its `to_dict` record -- or from one written before.

            Records written while the declaration was a separate ``declared``
            field carry ``'topics'`` as the names and ``'declared'`` as the
            declaration's rendering; the declaration is what ``topics`` is now.
            """
            record = dict(record)
            if 'declared' in record:
                record['topics'] = record.pop('declared')
            legacy_flags = set()
            legacy_val = record.pop('legacy', None)
            if legacy_val is True:
                legacy_flags.add('all')
            elif isinstance(legacy_val, (list, tuple)):
                legacy_flags.update(legacy_val)
            elif isinstance(legacy_val, str):
                legacy_flags.add(legacy_val)
            if record.pop('legacy_typing', None):
                legacy_flags.add('typing')
            if record.pop('legacy_signature', None):
                legacy_flags.add('signature')
            if legacy_flags:
                record['legacy'] = sorted(legacy_flags)
            return cls(**record)

        # 3. Accessors -----------------------------------------------------

        @property
        def legacy_typing(self) -> bool:
            """Whether legacy typing is enabled for this specialization."""
            return bool(self.legacy and ('all' in self.legacy or 'typing' in self.legacy))

        @property
        def legacy_signature(self) -> bool:
            """Whether legacy signature is enabled for this specialization."""
            return bool(self.legacy and ('all' in self.legacy or 'signature' in self.legacy))

        # 4. Helpers -------------------------------------------------------

        #: The fields every Specialization has; a subclass's others -- a
        #: Datastack's BLOCK, a Datatable's TAB -- are recorded and keyed too.
        _base_fields = frozenset({'spec', 'topics', 'version', 'note', 'anchor',
                                  'legacy', 'UNSAFE_redirect_all_topics', 'redirect_topics',
                                  'redirect_vars'})

        def _extra_fields_(self):
            """``(name, value)`` of the fields a subclass adds, in declaration order."""
            return [(f.name, getattr(self, f.name)) for f in fields(self)
                    if f.name not in self._base_fields]

        @property
        def _block_(self):
            """The block class the narrower stack's type names -- SAME: none named here."""
            return SAME

        @property
        def key(self):
            """A stable identifier for caching and for the journal record.

            `version` is in it because `get_hash` caches per key: two
            specializations alike but for the version are two different
            identities, and one would otherwise be served the other's hash.
            """
            return (tuple(sorted(self.spec.items(), key=lambda kv: kv[0])),
                    ('SAME',) if self.topics is SAME else tuple(self.topics), self.version, str(self.topics),
                    None if self.anchor is SAME else self.anchor,
                    self.legacy,
                    self.UNSAFE_redirect_all_topics,
                    tuple(self.redirect_topics) if self.redirect_topics is not None else None,
                    tuple(sorted(self.redirect_vars.items())) if self.redirect_vars is not None else None,
                    # A subclass's field at SAME overrides nothing: that
                    # specialization IS the plain one, and keys as it.
                    tuple((n, v) for n, v in self._extra_fields_() if v is not SAME))

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
        #: The topics this redirection covers, when it covers only some. None
        #: means "whatever the other side records", which is the older meaning
        #: and still the common one.
        topics: list | None = None
        #: The :class:`Specialization` this redirection came from, when it came
        #: from one. It is what makes a specialized redirection legible as such
        #: -- in `redirection`, and in the journal entry that records it.
        specialization: Optional['Datablock.Specialization'] = None

    class SpecializationRow(dict, Specialization):
        """What one declared `Specialization` did, and why.

        A join of the declared `Specialization` and its resolution in the journal.
        A dict, like `Validation`, so every reader written against the mapping
        goes on working -- and a `Specialization` subclass, so attributes like
        `spec`, `topics`, `legacy`, etc. and equality comparison against
        `Specialization` work directly. Truthy exactly when it RESOLVED,
        which is the question being asked::

            row = block.specializations()[0]
            if not row:
                print(row)          # the whole story, in order

        Keys: `specialization`, `hash` (the narrower block's, reconstructed),
        `matches` (its pins fit this block), `why` (the one reason it did not
        work, None when nothing went wrong), `entry` (the journal entry it
        resolved to), `paths` ({topic: path}), `topics` (what it names) and
        `builds` (this block's topics that it does NOT name -- what a build
        would still have to produce).
        """
        def __hash__(self):
            return hash(self.key)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sp = self.get('specialization')
            if sp is not None:
                for f in ('spec', 'topics', 'version', 'anchor', 'legacy', 'UNSAFE_redirect_all_topics',
                          'redirect_topics', 'redirect_vars', 'note'):
                    object.__setattr__(self, f, getattr(sp, f, None))

        def __getattr__(self, name):
            if name in self:
                return self[name]
            sp = self.get('specialization')
            if sp is not None and hasattr(sp, name):
                return getattr(sp, name)
            raise AttributeError(f"{self.__class__.__name__!r} object has no attribute {name!r}")

        def _extra_fields_(self):
            sp = self.get('specialization')
            if sp is not None and hasattr(sp, '_extra_fields_'):
                return sp._extra_fields_()
            return ()

        def __eq__(self, other):
            if isinstance(other, Datablock.Specialization):
                return self.key == other.key
            return super().__eq__(other)

        def __bool__(self):
            return self.resolved

        @property
        def resolved(self) -> bool:
            """Whether it found a build to read, which is the whole question."""
            return self.get('paths') is not None

        @property
        def entry(self):
            return self.get('entry')

        @property
        def paths(self):
            return self.get('paths')

        @property
        def hash(self):
            return self.get('hash')

        @property
        def why(self):
            return self.get('why')

        @property
        def matches(self):
            return self.get('matches')

        @property
        def builds(self):
            return self.get('builds')

        def __repr__(self):
            sp = self.get('specialization')
            note = getattr(sp, 'note', '')
            if self.resolved:
                verdict = f"RESOLVED to journal entry {self['entry']}"
            elif self.get('matches'):
                verdict = "applies, but DID NOT RESOLVE"
            else:
                verdict = "DOES NOT APPLY"
            lines = [f"{self.__class__.__name__}: {verdict}"]
            if sp is not None:
                lines.append(f"    specialization {sp!r}")
            for label, value in (
                ('why', self.get('why')),
                ('note', note or None),
                ('hash', self.get('hash')),
                ('reads', ', '.join(self.get('topics') or []) or None),
                ('builds', ', '.join(self.get('builds') or []) or None),
            ):
                if value is not None:
                    lines.append(f"    {label:7} {value}")
            paths = self.get('paths')
            if paths:
                for i, (topic, path) in enumerate(paths.items()):
                    prefix = "    paths  " if i == 0 else "           "
                    lines.append(f"{prefix} {topic}: {path}")
            return '\n'.join(lines)

        __str__ = __repr__

    class Validation(dict):
        """``{topic: bool}`` -- whether each topic's data is where it is read from.

        A dict, so it says WHICH topic is missing; and False as a whole unless
        every topic is True, so ``if not valid:`` means what it reads as rather
        than "the report is empty". An empty one is True, as `valid_topics`
        answers for a block with no topics.
        """

        def __bool__(self):
            return all(self.values())

        def missing(self) -> list:
            """The topics that are not there."""
            return [t for t, ok in self.items() if not ok]

    fqcn = _ClassOrInstance_(
        lambda cls: f"{cls.__module__}.{cls.__name__}",
        lambda self: f"{type(self).__module__}.{type(self).__name__}",
        doc="``module.Name`` of the class -- on the class itself, too.")

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

    Diff = collections.namedtuple('Diff', ['subsig', 'topics', 'version'])

    #SPECIALIZE: BEGIN
    #: The events that count as "this hash has data": a build, or a redirection
    #: to one. A redirect entry records the paths AFTER redirection (it is
    #: written once they are installed), so following a chain costs nothing
    #: beyond reading the entry -- and cannot loop, since nothing is followed.
    SPECIALIZATION_EVENTS = ('build:end', 'UNSAFE_redirect')

    ### anchorage: begin
    #: The anchor every instance of this class is stored under, unless it is
    #: given an ``anchor=`` of its own. None means the class's `fqcn`.
    ANCHOR = None

    anchor = _ClassOrInstance_(
        lambda cls: cls.ANCHOR or cls.fqcn,
        lambda self: (self.__dict__.get('_anchor_')
                      or type(self).ANCHOR or type(self).fqcn),
        doc="""The directory a block is stored under, below its datalake.

        On the CLASS, the anchor every instance takes when not given its own --
        :attr:`ANCHOR`, else the `fqcn` -- which is what lets a stack find its
        blocks' journal from its ``BLOCK`` alone. On an instance, its own
        ``anchor=`` when it was given one.""")

    # 1. Protocol and hooks ------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _check_forward_override_(cls, 'TOPICS')
        legacy = _legacy_topics_(class_declaration(cls, 'TOPICS') or {})
        if legacy:
            warnings.warn(
                f"{cls.__qualname__}.TOPICS declares {', '.join(legacy)} in the pre-marker "
                f"spelling; declare them with DATAFILE, DATADICT, DATADIR, DATASLICE or "
                f"SYNTHETIC. Respelling moves the hash, so keep the old spelling in a "
                f"Specialization(spec={{}}, topics=<the old declaration>) to reach what was built.",
                LegacyTopicsWarning, stacklevel=2)
        if 'signature_topics' in cls.__dict__:
            # Renamed, and private: overridden, the old name would now be
            # ignored in silence -- and a block whose identity a subclass had
            # been shaping would take a different hash with no error at all.
            raise TypeError(
                f"{cls.__qualname__} defines signature_topics(), which is now the private "
                f"_topics_signature_() and is not for subclasses to override: a block's "
                f"topic identity is its TOPICS, and a narrower block's is the declaration "
                f"a Specialization carries -- Specialization(topics={{name: node, ...}})."
            )
        renamed = sorted(set(cls.__dict__) & set(_RENAMED_DATAJOURNAL_BUILDERS_))
        if renamed:
            # Nothing calls the old names any more: an override under one would
            # be ignored in silence.
            raise TypeError(
                f"{cls.__qualname__} defines {', '.join(renamed)}, renamed "
                + ', '.join(f"{old} -> {_RENAMED_DATAJOURNAL_BUILDERS_[old]}" for old in renamed)
                + ": what builds a Datajournal or DatajournalFrame is named for it."
            )

    def __init__(
        self,
        *,
        datalake: str = None,
        # The name `datalake` had first: accepted, never recorded. Every
        # quote() journalled before the rename spells it, and inst() evaluates
        # those.
        url: str = None,
        spec: Optional[Union[str, dict]] = None,
        anchor: str = None,
        tag: str|None = None,
        info: bool = None,
        verbose: bool = None,
        debug: bool = None,
        detailed: bool = None,
        # Accepted, never recorded: every dfn journalled before capture moved to
        # `dbx.exec(capture_output=True)` spells it. A block captures nothing;
        # its journal entry records the capture open around its build.
        capture_output: bool = None,
        revision: str = None,
        keyby: str = 'tag_version_shorthash',
        uuid16: bool = False,
        # Shared by every block of one build tree, and generated when not given. A
        # block's own identity does not depend on it, so it stays out of the
        # signature; it is how the journal groups the entries of a tree.
        tree: str | None = None,
        # When a read fails, follow a redirection recorded by UNSAFE_redirect()
        # and read from the entry it names instead. See :meth:`read`.
        redirect: bool = True,
        # Whether this block may read a narrower, already-built block's data
        # instead of rebuilding it -- see SPECIALIZATIONS. None defers to the
        # class's USE_SPECIALIZATIONS: True installs and records, 'memory'
        # installs without recording, False declines to look.
        use_specializations: 'bool | str | None' = None,
        # This instance's SPECIALIZATIONS, in place of the class's; None keeps
        # the class's and [] declares none. Where to look for an older build,
        # never what this block is: not in the hash.
        SPECIALIZATIONS: 'list[Datablock.Specialization | dict] | Datablock.Specialization | dict | None' = None,
        SPECIALIZATION: 'list[Datablock.Specialization | dict] | Datablock.Specialization | dict | None' = None,
        # A journal already read, for :meth:`_install_specialization_` to resolve
        # against instead of reading one itself. Operational, and transient: it
        # travels under a private state key so it never reaches `parameters`,
        # `quote()` or a pickle -- see __setstate__.
        specialization_journal=None,
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
        if 'datajournal' in kwargs:
            # Left to **kwargs it would be kept as a dynamic parameter and do
            # nothing: which session an entry goes under is the open
            # `with Datajournal()`'s to say, never a block's.
            raise TypeError(
                f"{type(self).__name__}: datajournal= is not a Datablock argument; "
                f"open `with dbx.Datajournal():` around the code that builds instead"
            )
        if SPECIALIZATIONS is None and SPECIALIZATION is not None:
            SPECIALIZATIONS = SPECIALIZATION
        # What `log` is named, and the levels it sets over the scope's, until
        # __setstate__ knows the block's key.
        self._log_name = self.fqcn
        self._log_levels = dict(info=info, verbose=verbose, debug=debug, detailed=detailed)
        if capture_output:
            self.log.warning(f"{type(self).__name__}: capture_output=True is ignored: a block captures nothing "
                             f"of its own. Run the command under dbx.exec(..., capture_output=True), "
                             f"or --capture-output, and its entries record that capture.")
        self._working_params_ = []
        self._uuid16_ = uuid16
        self._uuid = uuid.uuid4().hex[:16] if uuid16 else str(uuid.uuid4())  # unique per live instance, not preserved across serialization
        self.log.detailed(f"__init__: ------------------------------------------------> {tag=}")
        state = {
            'datalake': one_datalake(datalake, url, type(self).__name__),
            'spec': spec,
            'anchor': anchor,
            'tag': tag,
            'info': info,
            'verbose': verbose,
            'debug': debug,
            'detailed': detailed,
            'revision': revision,
            'keyby': keyby,
            'uuid16': uuid16,
            'tree': tree,
            'redirect': redirect,
            'use_specializations': use_specializations,
            'SPECIALIZATIONS': SPECIALIZATIONS,
            'validate_vars': validate_vars if validate_cfg is None else validate_cfg,
            '__specialization_journal__': specialization_journal,
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
        # What `log` sets over the scope's: None there defers to it.
        self._log_levels = dict(
            info=state.get('info', True),
            verbose=state.get('verbose', False),
            debug=state.get('debug', False),
            detailed=state.get('detailed', False),
        )
        self._working_params_ = []
        self._resolve_legacy_CONFIG_()

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
        # `capture_output` is a command's now, not a block's -- see __init__.
        # State pickled before carries it; pop it so it is never re-serialized.
        state.pop('capture_output', None)
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
        # 'url' is the key a state pickled before the rename carries.
        self._datalake_ = _unquote(state.get('datalake', state.get('url')))
        # Resolve specline datalakes (e.g. "$dbx.getenv('KEY')") to real paths.
        self.datalake = eval(self._datalake_) if self._datalake_ is not None else None
        if self.datalake is None:
            self.datalake = default_datalake()
        if self.datalake is None:
            raise ValueError(f"No datalake for {self.__class__.__name__}: pass datalake= or set "
                             f"DBX_DATALAKE (or DBX_ROOT, or DBX_URL)")

        self._local_ = _unquote(state.get('local'))
        if self._local_ == 'None':
            self._local_ = None
        self.local_must_exist = bool(_unquote(state.get('local_must_exist', False)))

        self.storage_options = _unquote(state.get('storage_options'))
        if isinstance(self.storage_options, str):
            try:
                self.storage_options = ast.literal_eval(self.storage_options)
            except Exception:
                pass
        if self.storage_options is None or not isinstance(self.storage_options, dict):
            self.storage_options = default_storage_options()

        self.fs, self.root = fsspec.url_to_fs(self.datalake, **self.storage_options)
        _url_protocol = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
        if _url_protocol in ('file', 'local', ''):
            # url/root is already local storage: local=True and local=False
            # must be identical, so DBX_LOCAL/local= are never consulted.
            self.local = self.datalake
            self.localfs, self.localroot = self.fs, self.root
        else:
            # Resolve specline LOCALs (e.g. "$dbx.getenv('KEY')") to real paths.
            self.local = eval(self._local_) if self._local_ is not None else None
            if self.local is None:
                self.local = os.environ.get('DBX_LOCAL') or '/tmp/dbx'
            if self.local is None:
                raise ValueError(f"No local for {self.__class__.__name__}: pass local= or set DBX_LOCAL")
            if self.local_must_exist and not os.path.isdir(self.local):
                raise FileNotFoundError(
                    f"local={self.local!r} for {self.__class__.__name__} does not "
                    f"exist (local_must_exist=True) -- provision/mount it before "
                    f"running (e.g. a dedicated scratch disk that must actually be "
                    f"attached), or construct with local_must_exist=False to let it "
                    f"be auto-created on demand instead."
                )
            self.localfs, self.localroot = fsspec.url_to_fs(self.local, **self.storage_options)
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
        self.keyby = _unquote(state.get('keyby', 'tag_version_shorthash'))
        if self.keyby not in (None, 'hash', 'code', 'subhash', 'superhash', 'norm', 'signature', 'subsignature', 'tag', 'taghash', 'tag_hash', 'version_hash', 'tag_version_hash', 'tag_version_shorthash', 'custom'):
            raise ValueError(f"keyby must be None, 'hash', 'code', 'signature', 'tag', 'taghash', 'tag_hash', 'version_hash', 'tag_version_hash', 'tag_version_shorthash', 'custom', got {self.keyby!r}")
        if self.keyby == 'tag' and self._tag_ is None:
            raise ValueError(
                f"keyby='tag' requires an explicit tag= argument, but none was provided for {self.__class__.__name__}"
            )
        self._uuid16_ = state.get('uuid16', False)
        self._tree_ = _unquote(state.get('tree'))
        if self._tree_ == 'None':
            self._tree_ = None
        # Redirection config: dict(code=..., filter=..., paths=...) or legacy bool
        self.redirect = state.get('redirect')
        # The value as given, so __getstate__ reproduces it: None means "ask
        # the class", and is the value that keeps it out of quote()/cite().
        self._use_specializations_ = _unquote(state.get('use_specializations'))
        if self._use_specializations_ == 'None':
            self._use_specializations_ = None
        # Not recorded: a stack's use_block_specializations is how this block was
        # FORMED, and is the stack's to record -- see Datastack._form_block_.
        forming = _forming_with_specializations_()
        self.use_specializations = (self._use_specializations_ if self._use_specializations_ is not None
                                    else forming if forming is not None
                                    else self.USE_SPECIALIZATIONS)
        # As records -- literals -- so that __getstate__, and so quote() and the
        # journal, carry them in a form that evaluates back.
        specs = state.get('SPECIALIZATIONS')
        if specs is None:
            specs = state.get('SPECIALIZATION')
        self._SPECIALIZATIONS_ = self._specialization_records_(specs)
        tab_specs = state.get('TAB_SPECIALIZATIONS')
        if tab_specs is None:
            tab_specs = state.get('TAB_SPECIALIZATION')
        if tab_specs is None:
            tab_specs = state.get('BLOCK_SPECIALIZATIONS')
        if tab_specs is None:
            tab_specs = state.get('BLOCK_SPECIALIZATION')
        self._TAB_SPECIALIZATIONS_ = self._specialization_records_(tab_specs)
        self.validate_vars = state.get('validate_vars', True)
        self._paths_ = None

        # Popped, not read: it must not reach `state_params`, which is what
        # becomes `self.parameters` and so `quote()`, `cite()` and the journal
        # record. A journal is a frame of every build this anchor ever had --
        # not something to render into a specline, and not something to carry
        # into a pickle. Transient by construction: `__getstate__` never emits
        # it, and a block unpickled in a worker reads an installed redirection
        # from its `.redirection` marker, not from the journal.
        specialization_journal = state.pop('__specialization_journal__', None)

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
        
        # Named with the hash (and tag if present) from here on.
        self._log_name = self._log_name_()
        if isinstance(self.redirect, dict):
            self._process_redirect_()
        self.__post_init__()
        if 'TOPICS' in self.__dict__:
            # TOPICS computed here are identity like any others -- but the
            # log's name above asked for the key, so the hash was taken, and
            # cached, from the class's TOPICS before these existed. Taken again.
            for cached in ('_hash', '_specialized_hashes_'):
                self.__dict__.pop(cached, None)
            self._log_name = self._log_name_()
        if self._SPECIALIZATIONS_ is not None:
            # After __post_init__, so the instance's own win over any a class
            # computes there, as they win over the class's.
            self.SPECIALIZATIONS = [self.Specialization.from_record(r) for r in self._SPECIALIZATIONS_]
        if getattr(self, '_TAB_SPECIALIZATIONS_', None) is not None:
            tab_cls = getattr(self, 'TAB', None) or getattr(self, 'BLOCK', None)
            spec_cls = getattr(tab_cls, 'Specialization', self.Specialization) if tab_cls else self.Specialization
            inst_tab_specs = [spec_cls.from_record(r) for r in self._TAB_SPECIALIZATIONS_]
            if 'TAB_SPECIALIZATIONS' in state or hasattr(self, 'TAB_SPECIALIZATIONS'):
                self.TAB_SPECIALIZATIONS = inst_tab_specs
            elif 'BLOCK_SPECIALIZATIONS' in state or hasattr(self, 'BLOCK_SPECIALIZATIONS'):
                self.BLOCK_SPECIALIZATIONS = inst_tab_specs
            else:
                self.TAB_SPECIALIZATIONS = inst_tab_specs
        if specialization_journal is None:
            specialization_journal = _forming_journal_(self.anchor, self.datalake)
        if specialization_journal is not None:
            # Kept for build() and get_redirection(), outside the state: never
            # pickled or copied, and shared with every sibling it was handed to.
            self.__dict__['__specialization_journal__'] = specialization_journal
        # Nothing is installed here: constructing a block, or asking whether it
        # is valid, reads no journal and writes nothing. A redirection installed
        # before is found in its `.redirection` marker; one not yet installed
        # is build()'s to install -- see there.
        self.log.detailed(f"======--------------> code: {self.code}")


    def __getstate__(self):
        # Serialization convention for explicit params (url, spec, anchor, …).
        # The underscores are the dunders' own -- __init__ and __setstate__ --
        # worn by the value that was supplied to them:
        #
        #   _{k}_ = the *original* value the user passed in (or None).
        #           This is what gets serialized so that the block can be
        #           faithfully reconstructed by __setstate__.
        #   {k}   = the *resolved* / post-processed value used at runtime.
        #           For most params the resolution is simple (e.g. eval of
        #           a default expression), but for ``url`` and ``local`` it
        #           involves evaluating speclines like ``$dbx.getenv('KEY')``.
        #
        # The loop below prefers _{k}_ over {k} to capture the original, which
        # for url/local is what keeps env() relocatable: serializing the
        # resolved path would pin a reconstruction to the machine that wrote it.
        _state = {}
        for k in self.__explicit_params__():
            if hasattr(self, f"_{k}_"):
                _state[k] = getattr(self, f"_{k}_")
            elif hasattr(self, k):
                _state[k] = getattr(self, k)
        #TODO: why does 'log' end up in self.parameters?
        for k in self.parameters:
            if k not in self.__explicit_params__() and k != 'log' and hasattr(self, k):
                if k in ('TAB_SPECIALIZATIONS', 'TAB_SPECIALIZATION', 'BLOCK_SPECIALIZATIONS', 'BLOCK_SPECIALIZATION') and getattr(self, '_TAB_SPECIALIZATIONS_', None) is not None:
                    _state[k] = self._TAB_SPECIALIZATIONS_
                else:
                    _state[k] = getattr(self, k)
        return _state

    @staticmethod
    def __explicit_params__():
        sig = inspect.signature(Datablock.__init__)
        return [
            p.name for p in sig.parameters.values()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and p.name != 'self'
            and p.name not in Datablock.TRANSIENT_PARAMS
        ]

    def __post_init__(self):
        ...

    def valid(self):
        red = self.redirection
        entry = red.entry if red is not None else None
        return self.__valid__(path=entry.block.anchorkeypath if entry is not None else None)

    def why_invalid(self, validation: str = 'valid') -> str | None:
        """Why this block is not valid, and what makes it so -- or None when it is.

        What a failure to read it should say, rather than name a missing file.
        *validation* 'validate' also asks validate() of a block whose data is
        there. Reads the journal when the block is neither built nor
        redirected, to say whether a specialization resolves for it.
        """
        return self._invalidity_(validation)[1]

    def _invalidity_(self, validation: str = 'valid') -> tuple[str | None, str | None]:
        """``(kind, why)``: *kind* 'fails_validation', 'owes', 'unadopted' or 'unbuilt' -- or ``(None, None)`` when valid."""
        owed = self.owedtopics()
        if self.valid() and not owed:
            if validation == 'validate' and not self.validate():
                return 'fails_validation', ("its data is there but fails validate(), which no build "
                                            "repairs: clear it -- UNSAFE_clear() -- and build it again")
            return None, None
        if self.redirected():
            return 'owes', (f"it reads {self.redirected_topics()} through a redirection, "
                            f"but owes {owed}, which only its build produces")
        if self.find_specialization() is not None:
            return 'unadopted', ("it was never built, but a specialization resolves for it and was "
                                 "never adopted: its specialize(), or its stack's specialize_blocks(), adopts it")
        return 'unbuilt', "it was never built, and no specialization resolves for it: its build() builds it"

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

    def __read__(self, *topicpath):
        raise NotImplementedError()

    def __expand_spec__(self, expansion='repr', *, legacy: 'bool | None' = None,
                        legacy_typing: 'bool | None' = None):
        """
            . legacy: override LEGACY_NORM or LEGACY_SIGNATURE for the 'subsignature' expansion.
                None (default) = each block uses its own flag, i.e. the
                identity-bearing rendering. True/False forces the legacy or the
                modern form, and PROPAGATES to nested blocks, so the whole
                subtree is rendered the same way.

            . expansion: 'repr'|'quote'|'cite'|'repr_all'|'signature'
                . specline:      str starting with '@', '$' or '#'
                . datablock: Datablock object
                . obj:       object
            'repr':
                . FULL reduction
                    |obj:    repr(obj)
            'signature':
                . DATABLOCK reduction
                    |datablock: datablock.signaturestr()
                    |specline:      repr(specline)
                    |obj:       repr(obj)
            'quote':
                . UNREDUCED spec:
                    |specline:      repr(specline)
                    |datablock: datablock.quote()
                    |obj:       repr(obj)  
            'repr_all':
                . as 'quote', but
                    |datablock: datablock.repr()
        """
        legacy = self._legacy_norm_() if legacy is None else legacy
        # The TYPING choice a nested child must inherit. Separate from *legacy*
        # above, which is the norm flag: signature(legacy=...) selects typing,
        # so passing the norm flag down there rendered children the new way
        # inside a parent pinned to the old one.
        legacy_typing = self._legacy_typing_(legacy_typing)
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
                            _spec_[k] = eval_v.signaturestr(
                                legacy_typing=legacy_typing, legacy_signature=legacy)
                        else:
                            _spec_[k] = raw_v
                    except Exception:
                        _spec_[k] = raw_v
                elif isinstance(value, Datablock):
                    _spec_[k] = value.signaturestr(
                        legacy_typing=legacy_typing, legacy_signature=legacy)
                elif isinstance(value, str):
                    _spec_[k] = value
                elif legacy:
                    _spec_[k] = str(value)
                else:
                    # Stored as the value itself, so the embedding repr's it
                    # exactly once: 5 -> "5", '5' -> "'5'". No collision.
                    _spec_[k] = value
        elif expansion in ('quote', 'cite', 'repr_all'):
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
                    elif expansion == 'repr_all':
                        _spec_[k] = value.repr(pretty=False, deslash=0)
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
        topicpath = self._normtopic_(topicpath)
        node = self._topicnode_(*topicpath)

        if isinstance(node, dict):
            return {name: self.__path__(*topicpath, name, ensure_dirpath=ensure_dirpath,
                                        bare=bare, local=local)
                    for name in node}
        if self._node_is_syntopic_(node):
            return None

        dirpath = self.dirpath(*topicpath, local=local, redirect=False)
        if ensure_dirpath and dirpath is not None:
            ensure_path(dirpath, storage_options=self.storage_options)

        if self._node_is_dirtopic_(node):
            return dirpath
        path = os.path.join(dirpath, _topic_filename_(node))
        self.log.detailed(f"{self.anchor}: path: {path}")
        if bare and path:
            fs = self.localfs if local else self.fs
            path = fs._strip_protocol(path)
        return path

    # 2. Declared API ------------------------------------------------------

    def set(self, **kw):
        """This block again, with *kw* replacing its non-identity parameters.

        ``spec`` is refused. Identity is sha256(:meth:`typestr`), and typestr() is
        built from the spec, so re-specifying here would hand back a DIFFERENT
        block through a method that reads as an amendment of this one -- and
        would carry this block's state, resolved against this block's identity,
        over to one it was never resolved for. Construct the other block
        instead, which says what it is doing::

            type(b)(**{**b.dfn, 'spec': {...}})

        Everything else is fair game: ``tag``, ``keyby``, ``local``, ``tree``,
        the log levels, the dynamic kwargs -- none of them reach the signature,
        so :attr:`hash` is the same on both sides of the call.
        """
        if 'spec' in kw:
            raise ValueError(
                f"{self.__class__.__name__}.set(): spec= is refused -- set() "
                f"amends a block's non-identity parameters, and a new spec is a "
                f"new block. Construct it: "
                f"type(b)(**{{**b.dfn, 'spec': {{...}}}})"
            )
        _kw = copy.deepcopy(self.__getstate__())
        _kw.update(kw)     
        return self.__class__(**_kw)

    def replace(self, **kw):
        """Alias of :meth:`set`, and refuses ``spec`` for the same reason."""
        return self.set(**kw)

    def valid_topic(self, *topicpath):
        """Validity of one topic, or of a whole group."""
        topicpath = self._normtopic_(topicpath)
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

    def validate(self, **kwargs):
        """Validate this block's data. Default implementation calls self.valid().

        Specializations may override it to perform custom validation logic.
        """
        return self.__validate__(**kwargs)

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

    def redirected_topics(self):
        """The top-level topics this block reads through a redirection.

        Empty when it is not redirected; every topic when the redirection is
        total. In between is a PARTIAL redirection, and this is the list a
        `__build__` must not write to -- :meth:`ownedtopics` is its complement,
        and the one to build.
        """
        paths = self._redirected_paths_
        if not paths:
            return []
        return [t for t in self.topics() if t in paths]

    def ownedtopics(self):
        """The top-level topics this block is RESPONSIBLE for producing.

        All of them unless it is redirected, and under a partial redirection
        the ones the redirection does not cover. A ``__build__`` that can be
        asked to build part of a block reads this; one that cannot may ignore
        it, and will simply rebuild what it always did -- into the redirected
        location, which is why :meth:`path` refuses to ensure a directory for a
        redirected topic.

        Not to be confused with :meth:`owedtopics`, one letter away and a
        subset of this: these are the topics that are MINE, those are the ones
        of mine that are not there yet. Ownership does not change when a build
        runs; owing does.
        """
        redirected = set(self.redirected_topics())
        return [t for t in self.topics() if t not in redirected]

    #: The name this went by before the pair with :meth:`owedtopics` made the
    #: distinction worth spelling: "build" said what to do with the topics and
    #: not which ones they were, so a reader had to guess whether it meant "must
    #: produce" or "has yet to produce" -- which are now two methods.
    buildtopics = ownedtopics

    def owedtopics(self):
        """The topics of :meth:`ownedtopics` that are not there -- :meth:`build`'s question.

        OWED, not OWNED: the subset of this block's own topics that it has yet
        to produce. Everything in :meth:`ownedtopics` is this block's to build;
        these are the ones it has not built.

        Empty unless the block is PARTIALLY redirected, which is the only case
        in which :meth:`valid` can answer True about topics that are not this
        block's to answer for. Under a specialization the redirected topics are
        another build's and are there by definition, so a `valid()` that reads
        only those -- a marker topic like :class:`~dbx.datatables.Datatable`'s
        ``done``, which is a perfectly good definition of validity for a table
        that built itself -- pronounces the block built before the topics it
        still owes exist. `build()` would then never call `__build__`, and the
        whole point of a specialization (reuse what did not change, build what
        did) would be unreachable for exactly the classes that most need it.

        Deliberately NOT folded into `valid()`. What validity means belongs to
        the class -- a marker is a real answer, and a block may know things
        about its data that the presence of a file does not say. Whether this
        block has produced what the redirection left to it is a different
        question, it is one this class can answer generically, and it is the
        only one `build()` has any business asking.
        """
        if self._redirected_paths_ is None:
            return []
        owned = self.ownedtopics()
        if not owned:
            return []
        # A non-empty topic list, so valid_topics answers with the per-topic
        # dict rather than the bare True it gives a block that declares none.
        validity = self.valid_topics(owned)
        return [t for t in owned if not validity[t]]

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
                yield from walk(child, prefix + (self._check_topicname_(name),))

        if not self.has_topics():
            return []
        if self._topics_is_list_:
            return [(self._check_topicname_(name),) for name in self.TOPICS]
        return [tp for name, child in self.TOPICS.items()
                for tp in walk(child, (self._check_topicname_(name),))]

    def has_topics(self):
        """True when this block declares named topics (list or dict TOPICS)."""
        return hasattr(self, 'TOPICS') and isinstance(self.TOPICS, (list, dict))

    def is_topicgroup(self, *topicpath):
        """True when *topicpath* names a group of topics rather than a leaf."""
        return isinstance(self._topicnode_(*topicpath), dict)

    def specialize(self, journal=None):
        """Install the applicable specialization without building any unspecialized topics."""
        if self.valid():
            return self
        if isinstance(journal, BlocksJournal):
            journal = _shared_journal_(journal, self)
        j = journal if journal is not None else self.__dict__.get('__specialization_journal__')
        if isinstance(j, BlocksJournal):
            j = _shared_journal_(j, self)
        self._install_specialization_(journal=j)
        return self

    def build(self, *args, deep: bool = False, **kwargs):
        # A redirected block answers its reads out of another entry's data (see
        # redirection), so building it would produce data that nothing
        # would go on to read. Declining is also what makes a redirect stick:
        # a build_tree() sweeping past would otherwise quietly rebuild the very
        # block someone redirected away from. Costs one journal read per
        # instance, which redirection caches.
        if self._redirected_paths_ is not None and not self.ownedtopics():
            entry = self.redirection.entry if self.redirection is not None else None
            whither = (f"journal entry {entry.block.id} (hash {entry.block.hash})"
                       if entry is not None else f"the paths {self._redirected_paths_}")
            self.log.info(
                f"BUILD ELIDED: {self.anchorkeypath} is REDIRECTED to {whither}, and reads "
                f"from there instead: nothing would read what a build of it wrote. Undo the "
                f"redirection, or construct with redirect=False, to build it anyway."
            )
            return self
        if not deep and self.valid() and not self.owedtopics():
            self.log.selected(f"Skipping existing datablock: {self.anchorkeypath}")
            return self
        # A SPECIALIZATION is installed here: an older build of a
        # narrower block that covers what this build would produce is adopted
        # rather than recomputed -- recorded, so every later construction reads
        # through it -- and the topics it does not cover are built. After
        # __post_init__, as construction is complete, since a class may compute
        # its TOPICS there and a specialization is named in terms of them.
        if not kwargs.pop('_specialized_', False):
            self.specialize(journal=self.__dict__.get('__specialization_journal__'))
        if self._redirected_paths_ is not None and not self.ownedtopics():
            entry = self.redirection.entry if self.redirection is not None else None
            whither = (f"journal entry {entry.block.id} (hash {entry.block.hash})"
                       if entry is not None else f"the paths {self._redirected_paths_}")
            self.log.info(
                f"BUILD ELIDED: {self.anchorkeypath} is REDIRECTED to {whither}, and reads "
                f"from there instead: nothing would read what a build of it wrote. Undo the "
                f"redirection, or construct with redirect=False, to build it anyway."
            )
            return self
        if self._redirected_paths_ is not None:
            # A PARTIAL redirection -- a specialization, or an explicit
            # topics= -- leaves topics with no data and no one else to read
            # them from, so this build is the thing that produces them.
            # Declining here is what made the whole point of a specialization
            # (reuse what did not change, build what did) unreachable.
            self.log.info(
                f"BUILD PARTIAL: {self.anchorkeypath} reads {self.redirected_topics()} "
                f"through a redirection and builds {self.ownedtopics()}."
            )
        try:
            # `owedtopics()` as well as `valid()`: a PARTIALLY redirected block
            # can be valid() by its own definition and still owe topics that
            # only this build will produce -- see owedtopics(). Empty for every
            # block that is not partially redirected, so nothing else moves.
            owed = self.owedtopics()
            if not self.valid() or owed:
                if owed and self.valid():
                    self.log.info(
                        f"BUILD OWED: {self.anchorkeypath} reports valid(), but "
                        f"{owed} are its own to build and are not there -- building "
                        f"rather than skipping."
                    )
                self.__pre_build__(*args, **kwargs)
                self.__build__(*args, **kwargs)
                self._build_end_dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
                self.__post_build__(*args, **kwargs)
            else:
                self.log.selected(f"Skipping existing datablock: {self.anchorkeypath}")
        except KeyboardInterrupt as e:
            self.__post_build__(*args, event="build:keyboard_interrupt", **kwargs)
            raise(e)
        except Exception as e:
            self.__post_build__(*args, event="build:exception", **kwargs)
            raise(e)
        return self

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
        callback = self._transfer_callback_(f"pull {os.path.basename(src.rstrip('/'))}", show_progress=show_progress)
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
        callback = self._transfer_callback_(f"push {os.path.basename(src.rstrip('/'))}", show_progress=show_progress)
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
        by :meth:`datajournal`.

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
            if self._is_syntopic_(*leaf):
                continue
            dirpath = self.dirpath(*leaf, ensure=True)
            node = self._topicnode_(*leaf)
            crumbs = None if self._node_is_dirtopic_(node) else node
            self.leave_breadcrumbs_at_path(dirpath, crumbs=crumbs)
        return self

    def build_tree(self, *args, exclude_self: bool = False, deep: bool = False, **kwargs):
        log = self.log
        log.verbose(f"Building tree for {self} with roots {self.spec.keys()}")
        def skip_cb(s):
            log.skip_subtree(s, "TREE_SKIP_BUILDING")
        
        for s, c in self._iter_var_blocks_('TREE_SKIP_BUILDING', skip_callback=skip_cb):
            if not deep and c.valid():
                log.skip_subtree(s, "already valid")
                continue
            self.write_journal_entry(event=f"build_tree:{s}:begin")
            with log.subtree(s):
                # A child built as part of this tree belongs to this run. VAR is
                # where it was constructed, which is too early to know that.
                self._adopt_(c).build_tree(*args, deep=deep, **kwargs)
            self.write_journal_entry(event=f"build_tree:{s}:end")
        if not exclude_self:
            self.build(*args, deep=deep, **kwargs)
        return self

    def valid_var(self, *, reduce=False):
        if not self.validate_vars:
            return True if reduce else {}
        results = {s: c.valid() for s, c in self._iter_var_blocks_('TREE_SKIP_VALIDATION')}
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
            for s, c in self._iter_var_blocks_('TREE_SKIP_VALIDATION')
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
        topicpath = self._normtopic_(topicpath)
        self._topicnode_(*topicpath)      # raises KeyError if it does not exist
        if len(topicpath) == 1:
            return self.__read__(topicpath[0])
        return self.__read__(*topicpath)

    def get_redirection(self, journal=None):
        """Where this block reads from instead, as a :attr:`Redirection`, or None.

        If *journal* is provided, use it directly to find matching redirection entries
        instead of reloading journal files from storage. :meth:`redirected` is the
        cheap question -- whether there is one -- and reads no journal.
        """
        if not getattr(self, 'redirect', False):
            return None
        if journal is None:
            journal = self.__dict__.get('__specialization_journal__')

        recorded = self._recorded_redirection_(journal=journal)
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
        topics = recorded.get('topics')
        spec_record = recorded.get('specialization')
        try:
            specialization = (self.Specialization.from_record(spec_record)
                              if isinstance(spec_record, dict) else None)
        except TypeError:
            # Recorded when a specialization could name its topics without
            # declaring them. The redirection it installed still holds; which
            # specialization installed it can no longer be reconstructed.
            specialization = None

        if specialization is not None and getattr(specialization, 'UNSAFE_redirect_all_topics', False):
            topics = None

        if paths is not None:
            if topics is not None:
                paths = self._mapped_paths_(paths, topic_map, topics)
            existing = self.__dict__.get('__redirected_paths__')
            if existing is None:
                red_yaml = os.path.join(self.anchorkeypath, '.redirection', 'paths.yaml')
                try:
                    if self.fs.exists(red_yaml):
                        existing = read_yaml(red_yaml, storage_options=self.storage_options)
                except Exception:
                    pass
            if existing:
                paths = {**paths, **existing}
            self.log.verbose(
                f"REDIRECTION: {self.anchorkeypath} reads from the given paths instead: {paths}"
            )
            self.__dict__['__redirected_paths__'] = paths
            return self.Redirection(paths=paths, entry=None, filter=None, topic_map=None,
                                    topics=topics, specialization=specialization)

        if filter is None and isinstance(recorded, dict) and not ('filter' in recorded or 'paths' in recorded or 'topic_map' in recorded or 'topics' in recorded):
            existing = self.__dict__.get('__redirected_paths__')
            if existing is None:
                red_yaml = os.path.join(self.anchorkeypath, '.redirection', 'paths.yaml')
                try:
                    if self.fs.exists(red_yaml):
                        existing = read_yaml(red_yaml, storage_options=self.storage_options)
                except Exception:
                    pass
            combined = {**recorded, **existing} if existing else recorded
            self.log.verbose(
                f"REDIRECTION: {self.anchorkeypath} reads from the given paths instead: {combined}"
            )
            self.__dict__['__redirected_paths__'] = combined
            return self.Redirection(paths=combined, entry=None, filter=None, topic_map=None)

        if specialization is not None and self._redirected_paths_ is not None:
            # A specialization recorded before its paths were: the .redirection
            # topic it wrote holds them, and saves finding its entry again.
            return self.Redirection(paths=self._redirected_paths_, entry=None, filter=filter,
                                    topic_map=topic_map, topics=topics,
                                    specialization=specialization)
        entry = self._redirect_entry_(filter, journal=journal)
        if entry is None:
            self.log.warning(f"redirection: filter {filter!r} matches no journal entry")
            return None

        resolved_paths = self._mapped_paths_(entry.block.paths(), topic_map, topics)
        self.__dict__['__redirected_paths__'] = resolved_paths

        self.log.verbose(
            f"REDIRECTION: {self.anchorkeypath} reads from journal entry {entry.block.id} "
            f"instead (hash {entry.block.hash}, event {entry.get('event')!r}, written "
            f"{entry.get('datetime')}), matched by {filter!r}"
            + (f", topics mapped {topic_map!r}" if topic_map else "")
            + (f", restricted to topics {topics!r}" if topics else "")
            + (f", as {specialization!r}" if specialization is not None else "")
        )
        return self.Redirection(paths=resolved_paths, entry=entry, filter=filter,
                                topic_map=topic_map, topics=topics,
                                specialization=specialization)

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

    def UNSAFE_redirect(self, *, redirector: Callable|None = None, journal: DatajournalFrame|None = None, filter: dict|None = None, topic_map: dict|None = None,
                        paths: dict|None = None, topics: list|None = None,
                        specialization: 'Datablock.Specialization | None' = None,
                        merge: bool = False,
                        dry_run: bool = False, dry_validate: bool = False,
                        validate: bool = False, remote: bool | Remote = False, OVERRIDE: bool = False):
        """Record that this block's topics are read from somewhere else and the location of this somewhere else.

        *topics* redirects only the topics it names. Every other topic reads as
        it would unredirected, so a block can take part of its data from
        elsewhere and build the rest -- which is the whole point of a
        *specialization*, and is available on its own for any redirection.

        *specialization* is the fourth way of saying where: one of this class's
        :attr:`SPECIALIZATIONS`, resolved to the narrower block's build through
        the hash reconstructed from it. It implies its own ``topics``, and is
        recorded as itself, so the journal says a specialization was installed
        and which one -- not merely that some paths were.

        *dry_run* resolves the whole thing and reports what it WOULD record --
        to stdout, and as the :class:`Redirection` it returns -- without
        installing it on this block, writing the hidden ``.redirection`` topic,
        or putting anything in the journal. The return type says which happened:
        a Redirection is a proposal, True is a redirection that is now in place.

        *dry_validate* adds the one question a dry run cannot answer by
        resolving: is the data actually THERE. A resolution reports the paths
        the target RECORDED, so a build whose data has since been cleared
        proposes just as cleanly as one whose data is intact. It installs the
        proposal in memory only, checks each redirected topic through it, and
        returns ``(Redirection, Validation)`` instead of the bare Redirection --
        the pair is how a caller tells a proposal that was checked from one that
        was not, which a None field could not say. It needs *dry_run*;
        ``validate=`` is the same question asked after a real redirection is
        installed.
        """
        if dry_validate and not dry_run:
            raise ValueError(
                "UNSAFE_redirect: dry_validate= reports on a redirection that is NOT being "
                "installed, so it wants dry_run=True. To check a redirection that is being "
                "installed, pass validate=True."
            )
        if not UNSAFE_allowed("UNSAFE_redirect", OVERRIDE=OVERRIDE):
            return False

        if specialization is not None:
            if filter is not None or paths is not None or redirector is not None:
                raise ValueError(
                    "UNSAFE_redirect: specialization= says where on its own; it does not "
                    "combine with redirector=/filter=/paths="
                )
            why = self._specialization_mismatch_(specialization)
            if why is not None:
                self.log.warning(f"UNSAFE_redirect: {specialization!r} does not apply: {why}")
                return False
            if getattr(specialization, 'UNSAFE_redirect_all_topics', False):
                topics = None
            elif getattr(specialization, 'redirect_topics', None) is not None:
                topics = list(specialization.redirect_topics) if topics is None else list(topics)
            else:
                topics = list(self._specialization_topics_(specialization)) if topics is None else list(topics)
            if journal is not None:
                journal = self._specialization_journal_(journal, specialization)

        explicit_journal = journal is not None
        if journal is None:
            try:
                journal = self.datajournal()
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
        if topics is not None:
            # Checked before anything is resolved or recorded: a restriction
            # naming none of this block's topics would redirect nothing, and
            # recording THAT claims a redirection is in place while every topic
            # still reads its own absent data -- and declines the build that
            # would have produced it. A topic_map whose TARGET is missing is a
            # different thing, documented on `_mapped_paths_`, and still allowed.
            named = set(self._toplevel_topics_(topics)) & set(self.topics())
            if not named:
                self.log.warning(
                    f"UNSAFE_redirect: topics={list(topics)!r} names none of this block's "
                    f"topics {self.topics()!r}; nothing to redirect"
                )
                return False

        entry = None
        target_paths = None
        redirect_record = None

        if specialization is not None:
            resolved = self._specialization_paths_(specialization, journal=journal)
            if resolved is None:
                self.log.warning(
                    f"UNSAFE_redirect: {specialization!r} applies, but no {'/'.join(self.SPECIALIZATION_EVENTS)} "
                    f"entry with hash {self.get_hash(specialization)} records data that is still "
                    f"there -- nothing to redirect to"
                )
                return False
            target_paths, entry = resolved
            target_paths = self._chase_redirected_paths_(target_paths)
            sp_topics = list(topics) if topics is not None else (
                None if getattr(specialization, 'UNSAFE_redirect_all_topics', False) else list(self._toplevel_topics_(self._specialization_topics_(specialization)))
            )
            redirect_record = {
                'filter': {'hash': self.get_hash(specialization)},
                'topics': sp_topics,
                'specialization': specialization.to_dict(),
                # Resolved already: recorded, so that where this block reads from
                # is answered from the record, not by finding the entry again in
                # a journal of thousands.
                'paths': target_paths,
            }

        # If the redirector gave us paths directly, use those immediately.
        redirector_paths = paths if (redirector is not None and isinstance(res, dict) and 'paths' in res) else None
        if redirector_paths is not None:
            target_paths = self._chase_redirected_paths_(redirector_paths)
            redirect_record = target_paths

        if target_paths is None and filter is not None and journal is not None:
            try:
                j = DatajournalFrame(journal, storage_options=getattr(journal, 'storage_options', self.storage_options), **dict(filter))
                if len(j) > 0:
                    entry = j.get(0, dropna=True) if hasattr(j, 'get') and 0 in j.index else DatajournalEntry(j.iloc[0].dropna(), storage_options=getattr(j, 'storage_options', self.storage_options))
                    target_paths = entry.block.paths()
                    if target_paths is None or not isinstance(target_paths, dict):
                        try:
                            target_paths = entry.inst(remote=remote).paths()
                        except Exception as e:
                            self.log.detailed(f"UNSAFE_redirect: entry.inst() failed: {e}")
                    if target_paths is not None:
                        target_paths = self._chase_redirected_paths_(target_paths)
                    redirect_record = (entry.block.id if topics is None else
                                       {'filter': dict(filter), 'topics': list(topics)})
            except Exception as e:
                self.log.warning(f"UNSAFE_redirect: filter {filter!r} failed on journal: {e}")

        if target_paths is None and paths is not None:
            target_paths = self._chase_redirected_paths_(paths)
            redirect_record = target_paths
            if topics is not None:
                # A bare paths dict cannot carry the restriction -- it IS the
                # record -- so the restriction has to be said in the dict form.
                redirect_record = {'paths': target_paths, 'topics': list(topics)}

        if target_paths is None and filter is None and explicit_journal:
            try:
                j = journal if isinstance(journal, DatajournalFrame) else DatajournalFrame(journal, storage_options=self.storage_options)
                if len(j) > 0:
                    entry = j.get(0, dropna=True) if hasattr(j, 'get') and 0 in j.index else DatajournalEntry(j.iloc[0].dropna(), storage_options=getattr(j, 'storage_options', self.storage_options))
                    target_paths = entry.block.paths()
                    if target_paths is None or not isinstance(target_paths, dict):
                        try:
                            target_paths = entry.inst(remote=remote).paths()
                        except Exception as e:
                            self.log.detailed(f"UNSAFE_redirect: entry.inst() failed: {e}")
                    if target_paths is not None:
                        target_paths = self._chase_redirected_paths_(target_paths)
                    redirect_record = entry.block.id
            except Exception as e:
                self.log.warning(f"UNSAFE_redirect: failed reading journal: {e}")

        if target_paths is None:
            self.log.warning(f"UNSAFE_redirect: no redirection for hash {self.hash}")
            return False

        remapped_paths = self._mapped_paths_(target_paths, topic_map, topics)
        remapped_paths = self._chase_redirected_paths_(remapped_paths)

        if dry_run:
            validation = None
            if dry_validate:
                # Installed in memory ONLY -- no hidden topic, no journal entry
                # -- so `path()` answers with the redirected paths and the check
                # runs against what a read would actually open. Restored in a
                # finally: a dry run that leaves the block redirected is not one.
                held = '__redirected_paths__' in self.__dict__
                previous = self.__dict__.get('__redirected_paths__')
                self.__dict__['__redirected_paths__'] = remapped_paths
                self.__dict__.pop('redirection', None)
                try:
                    # The REDIRECTED topics only. Under a partial redirection the
                    # rest are this block's own to build, and are absent exactly
                    # as they should be until it does.
                    validation = self.Validation(self.valid_topics(list(remapped_paths)))
                finally:
                    if held:
                        self.__dict__['__redirected_paths__'] = previous
                    else:
                        self.__dict__.pop('__redirected_paths__', None)
                    self.__dict__.pop('redirection', None)
            proposal = self.Redirection(
                paths=remapped_paths, entry=entry, filter=filter, topic_map=topic_map,
                topics=list(topics) if topics is not None else None,
                specialization=specialization)
            print(
                f"UNSAFE_redirect(dry_run=True) on {self.anchorkeypath}\n"
                f"DRY RUN -- nothing below has been done. In an actual run:\n"
                f"  - I would record {redirect_record!r} as this block's redirection\n"
                + (f"  - I would record it as the specialization {specialization!r}\n"
                   if specialization is not None else "")
                + (f"  - the redirection would come from journal entry {entry.block.id} "
                   f"(hash {entry.block.hash})\n" if entry is not None else "")
                + f"  - I would read these topics through the redirection: {remapped_paths!r}\n"
                + ((f"  - I checked those paths: all {len(validation)} topics are THERE\n"
                    if validation
                    else f"  - I checked those paths: {len(validation.missing())} of "
                         f"{len(validation)} topics are MISSING: {validation.missing()!r}\n")
                   if isinstance(validation, dict) else "")
                + f"  - I would still build these topics myself: "
                f"{[t for t in self.topics() if t not in remapped_paths]!r}"
            )
            return (proposal, validation) if dry_validate else proposal

        if merge:
            existing = dict(self._redirected_paths_) if self._redirected_paths_ is not None else {}
            if not existing:
                try:
                    red_yaml = os.path.join(self.anchorkeypath, '.redirection', 'paths.yaml')
                    if self.fs.exists(red_yaml):
                        existing = read_yaml(red_yaml, storage_options=self.storage_options) or {}
                except Exception:
                    pass
            combined_paths = {**existing, **remapped_paths}
            self._redirected_paths_ = combined_paths
        else:
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
            if topics is not None and not all(t in self._redirected_paths_ for t in self.topics()):
                val_res = self.valid_topics(list(remapped_paths))
            else:
                val_res = self.validate()
            is_val = bool(all(val_res.values())) if isinstance(val_res, dict) else bool(val_res)
            if not is_val:
                self.log.warning(f"UNSAFE_redirect: block {self.hash} remains invalid after redirection")
                return False

        self.log.verbose(f"UNSAFE_redirect: {self.hash} -> {redirect_record!r} (entry {code})")
        return True

    def UNSAFE_clear_redirection(self, *, OVERRIDE: bool = False) -> bool:
        """Stop this block reading elsewhere: undo :meth:`UNSAFE_redirect`, and nothing else.

        Unlike :meth:`UNSAFE_clear`, no data is touched -- neither this block's
        nor the block it was redirected to. What goes is the redirection: the
        hidden ``.redirection`` topic, the paths installed on this instance, and
        -- in the journal, whose latest redirection record is the one that
        holds -- an ``UNSAFE_clear_redirection`` entry that supersedes it.

        A specialized block BUILT again with specializations on will find its
        specialization again and redirect itself again; construct it with
        ``use_specializations=False`` to keep it reading its own. Returns
        whether there was a redirection to clear.
        """
        if not UNSAFE_allowed("UNSAFE_clear_redirection", OVERRIDE=OVERRIDE):
            return False
        had = self.redirected()
        try:
            red_dir = self.dirpath('.redirection')
            if self.fs.exists(red_dir):
                self.fs.rm(red_dir, recursive=True)
        except Exception as e:
            self.log.warning(f"UNSAFE_clear_redirection: could not remove .redirection "
                             f"for {self.anchorkeypath}: {e}")
        self._redirected_paths_ = None
        for cached in ('redirection', '__specialization__'):
            self.__dict__.pop(cached, None)
        self.write_journal_entry(event='UNSAFE_clear_redirection',
                                 redirection={'cleared': True}, journal_prefix='unredirect-')
        self.log.verbose(f"UNSAFE_clear_redirection: {self.anchorkeypath} "
                         f"{'reads its own data again' if had else 'was not redirected'}")
        return had

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

        def clear_topic(topicpath):
            # A group names a directory but holds no data of its own; clearing
            # it means clearing what is under it.
            if clear_dirpath:
                clear_path(self.dirpath(*topicpath, redirect=False), recursive=True)
                return
            for leaf in self._leaves_under_(*topicpath):
                clear_path(self.__path__(*leaf), recursive=self._is_dir_topic_(*leaf))

        if len(topics) == 0:
            for topic in self.topics():
                clear_topic((topic,))
            try:
                red_dir = self.dirpath('.redirection')
                if self.fs.exists(red_dir):
                    had_redirection = True
                    self.log.info(f"UNSAFE_clear: removing .redirection for {self.anchorkeypath} (the underlying block being redirected to is not affected)")
                    clear_path(red_dir, recursive=True)
            except Exception:
                pass
            self._redirected_paths_ = None
            self.__dict__.pop('redirection', None)
            self.__dict__.pop('__specialization__', None)
            self.__dict__.pop('__redirected_paths__', None)
            self.write_journal_entry(event="UNSAFE_clear", redirection={'cleared': True})
        else:
            cleared_topic_names = set()
            for topic in topics:
                norm_t = self._normtopic_((topic,))
                cleared_topic_names.add(norm_t[0])
                clear_topic(norm_t)
            try:
                red_dir = self.dirpath('.redirection')
                red_yaml = os.path.join(red_dir, 'paths.yaml')
                if self.fs.exists(red_yaml):
                    paths = read_yaml(red_yaml, storage_options=self.storage_options)
                    if isinstance(paths, dict):
                        updated = {k: v for k, v in paths.items() if k not in cleared_topic_names}
                        if not updated:
                            had_redirection = True
                            self.log.info(f"UNSAFE_clear: removing .redirection for {self.anchorkeypath} as all redirected topics were cleared")
                            clear_path(red_dir, recursive=True)
                            self._redirected_paths_ = None
                        else:
                            write_yaml(updated, red_yaml, storage_options=self.storage_options)
                            self._redirected_paths_ = updated
            except Exception as e:
                self.log.detailed(f"UNSAFE_clear: updating .redirection failed: {e}")
            self.__dict__.pop('redirection', None)
            self.__dict__.pop('__specialization__', None)
            self.__dict__.pop('__redirected_paths__', None)
            self.write_journal_entry(event=f"UNSAFE_clear:{list(topics)}", redirection={'cleared': True})

        msg = f"UNSAFE_clear: cleared block {self.hash}"
        if had_redirection:
            msg += " (redirection removed)"
        self.log.info(msg)
        return self

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
            Forwarded to :meth:`_UNSAFE_copy_topic_` for every topic; ignored
            by the base implementation but available to subclasses that
            override :meth:`_UNSAFE_copy_topic_` to accept additional
            per-topic options.
        """
        if not UNSAFE_allowed("UNSAFE_copy_from", OVERRIDE=OVERRIDE):
            return self
        if not overwrite:
            assert not self.valid(), f"Attempting to overwrite a valid Datablock {self}. Missing 'overwrite' argument?"
        fs, _ = self._url_to_fs_(anchorkeypath)
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
                self._UNSAFE_copy_topic_(
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
        journal entry (via :meth:`datajournal`) and forwards that entry's
        ``anchorkeypath`` as the copy source.

        Parameters
        ----------
        journal : dict
            Keyword arguments passed to :meth:`datajournal` to select the entry
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
        entry = self.datajournal(**journal)
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
        kwargs = {**self._rootkwargs_, **{'spec': quoted_spec},}
        if tailkwargs:
            kwargs.update(**self._tailkwargs_)
        else:
            kwargs.update({
                k: v for k, v in self._tailkwargs_.items()
                if k in self.CITE_KEEP_TAILKWARGS
            })
        quote = self._render_call_(kwargs, pretty=pretty, deslash=deslash, dollar=not cite)
        self.log.detailed(f"quote: ------------> {quoted_spec=}")
        self.log.detailed(f"quote: ------------> {quote=}")
        return quote

    def repr(self, *, deslash: int = 0, pretty: bool = False) -> str:
        """An evaluable ``$fqcn(...)`` specline carrying EVERY constructor kwarg.

        :meth:`quote` renders what reconstructs a working block, and leaves out
        what belongs to one run (``tree``); :meth:`cite`
        renders for reading. This renders the whole of :attr:`dfn`: spec and
        every other parameter, operational ones included -- what the block WAS,
        down to the run it was part of. ``url`` and ``anchor`` are rendered as
        :meth:`quote` renders them, only when given, so a block rooted by
        the environment's datalake stays relocatable. Private state
        (``__redirected_paths__``) is not a kwarg and is not rendered. A nested
        block renders as its own ``repr()``. *pretty* and *deslash* are as for
        :meth:`quote`.
        """
        self.tree   # generated on first access; render the one this block has
        kwargs = {**self._rootkwargs_, 'spec': self.__expand_spec__('repr_all')}
        kwargs.update({k: v for k, v in self.__getstate__().items()
                       if k not in ('datalake', 'url', 'anchor', 'spec') and not k.startswith('__')})
        r = self._render_call_(kwargs, pretty=pretty, deslash=deslash, dollar=True)
        self.log.detailed(f"repr: ------------> {r=}")
        return r

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

    def signaturestr(self, *, deslash: bool = False, legacy: bool | None = None,
                  legacy_typing: bool | None = None,
                  legacy_signature: bool | None = None, pretty: bool = False,
                  omit=(), redirect_vars=None):
        """The base identity string that `typestr` -- and hence `hash` and `code` -- is built from.

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

        *redirect_vars*, when given, is ``{current_name: historical_name}`` and
        is forwarded to `_typed_specdict_` so the rendered spec uses the
        historical key names and sort order.
        """
        if legacy_typing is None:
            legacy_typing = legacy
        if legacy_signature is None:
            legacy_signature = legacy
        legacy_typing = self._legacy_typing_(legacy_typing)
        norm = self._legacy_norm_() if legacy_signature is None else bool(legacy_signature)
        if pretty:
            import pprint
            return pprint.pformat(
                self.signature(legacy_typing=legacy_typing, legacy_signature=norm,
                                   deslash=deslash, omit=omit, redirect_vars=redirect_vars),
                indent=2, width=120)
        if legacy_typing:
            #CAUTION! This branch is what already-built blocks hashed with, and
            # is the pre-change code verbatim. The NORM flag alone decides root
            # kwargs and quoting, exactly as before -- so a relocatable block
            # pinned for typing stays relocatable.
            sig_spec = self.__expand_spec__('signature', legacy=norm, legacy_typing=True)
            if omit:
                sig_spec = {k: v for k, v in sig_spec.items() if k not in set(omit)}
            if redirect_vars:
                sig_spec = {redirect_vars.get(k, k): v for k, v in sig_spec.items()}
            kwargs_dict = {**(self._identity_rootkwargs_ if norm else {}), 'spec': sig_spec}
            sig = self.__repr_from_kwargs__(kwargs_dict, anchor=None, quote_strs=not norm)
        else:
            # Rendered FROM the typed dict, so the text and the dict cannot
            # disagree, and a leaf is quoted exactly when it is a string.
            # Root kwargs only on explicit opt-in: signature and hash are
            # relocatable, and nothing about typing changes that.
            root = ''.join(f"{k}={v!r}, " for k, v in self._identity_rootkwargs_.items()) if norm else ''
            sig = f"({root}spec={self._typed_specdict_(legacy=False, omit=omit, redirect_vars=redirect_vars)!r})"
        if deslash:
            sig = sig.replace('\\', '')
        self.log.detailed(f"signature: ------------> legacy={legacy}")
        self.log.detailed(f"signature: ------------>{sig=}")
        return sig

    def subsignaturestr(self, *args, **kwargs):
        """Alias for :meth:`signaturestr` for backwards compatibility."""
        return self.signaturestr(*args, **kwargs)

    def normstr(self, *args, **kwargs):
        """Alias for :meth:`signaturestr` for backwards compatibility."""
        return self.signaturestr(*args, **kwargs)

    def diffsignature(
        self,
        other_signature: 'Datablock | DatajournalEntry | str | None' = ABSENT,
        *,
        journal: 'DatajournalFrame | DatajournalEntry | dict | str | int | None' = None,
        raw: bool = False,
        deslash: bool = False,
        legacy: 'bool | None' = None,
        recursive: bool = True,
        report: bool = False,
        maxlen: 'int | None' = 160,
    ) -> 'dict | str':
        """Diff this datablock's signature against another signature, key by key."""
        if isinstance(other_signature, Datablock):
            other_signature = other_signature.signaturestr(legacy=legacy)
        elif isinstance(other_signature, DatajournalEntry):
            other_signature = other_signature.read('signature') or other_signature.read('subsignature') or other_signature.read('norm') or ''
        elif (other_signature is None or other_signature is ABSENT) and journal is not None:
            _entry = self._journal_entry_(journal)
            other_signature = _entry.read('signature') or _entry.read('subsignature') or _entry.read('norm') or ''

        def present(value):
            if value is ABSENT:
                return value
            if not raw:
                value = self._literal_(value)
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

        parsed_self  = Datablock._parse_signature_(self.signaturestr(legacy=legacy))
        parsed_other = Datablock._parse_signature_(other_signature or '')

        def _normalize_subsig_dict(d):
            if 'spec' not in d and d:
                root_keys = {'datalake', 'url', 'local', 'local_must_exist', 'storage_options', 'anchor', 'tag', 'revision', 'keyby', 'uuid16', 'redirect', 'validate_vars'}
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
            parsed_self = {k: self._structure_signatureval_(v) for k, v in parsed_self.items()}
            parsed_other = {k: self._structure_signatureval_(v) for k, v in parsed_other.items()}
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

    def signature(self, *, legacy: 'bool | None' = None,
                      legacy_typing: 'bool | None' = None,
                      legacy_signature: 'bool | None' = None,
                      deslash: bool = False, omit=(), redirect_vars=None) -> dict:
        """The signature as a nested dict of correctly-typed values.

        Built from ``var`` via `_typed_specdict_`, so an ``int`` field comes
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
        if self._legacy_typing_(legacy_typing):
            parsed = Datablock._parse_signature_(self.signaturestr(
                legacy_typing=True, legacy_signature=legacy_signature, deslash=deslash,
                omit=omit, redirect_vars=redirect_vars))
            return {k: self._structure_from_signature_text_(v) for k, v in parsed.items()}
        return {'spec': self._typed_specdict_(legacy=False, omit=omit, redirect_vars=redirect_vars)}

    def sig(self, *, legacy: 'bool | None' = None, deslash: bool = False) -> dict:
        return self.signature(legacy=legacy, deslash=deslash)

    def subsignature(self, *, legacy: 'bool | None' = None, deslash: bool = False) -> dict:
        return self.signature(legacy=legacy, deslash=deslash)

    def subsig(self, *, legacy: 'bool | None' = None, deslash: bool = False) -> dict:
        return self.signature(legacy=legacy, deslash=deslash)

    def norm(self, *args, **kwargs):
        return self.signature(*args, **kwargs)

    def sigstr(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = True):
        """Alias for :meth:`signaturestr` (defaults to pretty=True)."""
        return self.signaturestr(deslash=deslash, legacy=legacy, pretty=pretty)

    def subsigstr(self, *, deslash: bool = False, legacy: bool | None = None, pretty: bool = True):
        return self.signaturestr(deslash=deslash, legacy=legacy, pretty=pretty)

    def type(self, *, deslash: bool = False, legacy: 'bool | None' = None,
                 legacy_typing: 'bool | None' = None,
                 legacy_signature: 'bool | None' = None,
                 specialization: 'Datablock.Specialization | None' = None,
                 with_block: bool = True) -> dict:
        """The full type as a dictionary: the structured form of :meth:`typestr`.

        *specialization* describes the narrower block that one names instead,
        as ``typestr(specialization=)`` renders it; *with_block* is typestr's.
        """
        if specialization is None:
            return {
                'signature': self.signature(
                    legacy=legacy, legacy_typing=legacy_typing,
                    legacy_signature=legacy_signature, deslash=deslash),
                **self._type_entries_(with_block=with_block),
                'version': self.version,
                'paths': getattr(self, '_paths_', None),
                'topics': self._topics_signature_(),
            }
        version = self.version if specialization.version is ABSENT else specialization.version
        omit = tuple(specialization.spec)
        redirect_vars = getattr(specialization, 'redirect_vars', None)
        leg = legacy_typing if legacy_typing is not None else (legacy if legacy is not None else specialization.legacy_typing)
        return {
            'signature': {'spec': self._typed_specdict_(legacy=leg, omit=omit, redirect_vars=redirect_vars)},
            **self._type_entries_(specialization, with_block=with_block),
            'version': version,
            'paths': getattr(self, '_paths_', None),
            'topics': self._topics_signature_(list(self._specialization_topics_(specialization)),
                                              declared=self._specialization_topics_(specialization)),
        }

    def tp(self, *, deslash: bool = False, legacy: 'bool | None' = None) -> dict:
        return self.type(deslash=deslash, legacy=legacy)

    def tpstr(self, *, deslash: bool = False, legacy: 'bool | None' = None, pretty: bool = True):
        """Alias for :meth:`typestr` (defaults to pretty=True)."""
        return self.typestr(deslash=deslash, legacy=legacy, pretty=pretty)

    def difftopics(
        self,
        other_topics=ABSENT,
        *,
        journal: 'dict | None' = None,
        report: bool = False,
        maxlen: 'int | None' = 160,
    ) -> 'dict | str':
        """Diff this block's topics against another's, the way :attr:`signature` sees them.

        Compares :meth:`_topics_signature_` -- the very segments the signature is
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
        mine = self._topics_signature_()
        theirs, theirmap = self._other_topics_(other_topics, journal)
        mymap = self._topic_map_(getattr(self, 'TOPICS', None))

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
            other_version = self._journal_entry_(journal)
        if isinstance(other_version, (Datablock, DatajournalEntry)):
            other_version = other_version.version
        mine = self.version
        if str(mine) == str(other_version):
            return None
        return (mine, other_version)

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

    def typestr(self, *, deslash: bool = False, legacy: 'bool | None' = None,
             legacy_typing: 'bool | None' = None,
             legacy_signature: 'bool | None' = None, pretty: bool = False,
             specialization: 'Datablock.Specialization | None' = None,
             with_block: bool = True):
        """The identity string :attr:`hash` is the sha256 of.

        *specialization* renders the identity of the NARROWER block that one
        describes instead of this one's: its ``spec`` fields dropped, its
        ``topics`` alone, and its ``version`` when it names one. Its topics are
        rendered from the declaration it carries -- that block's own nodes, in
        that block's era -- and NOTHING of them is inherited from this class's
        TOPICS. (A Datatable adds its TAB's slices, as its own identity does.)
        See :meth:`get_hash`.

        *with_block* False computes it under the rules from before a stack's
        type named its BLOCK -- a Datatable's its TAB -- which is what
        reconstructing a stack built then needs. See :meth:`_type_entries_`.
        """
        omit, topics = ((), None) if specialization is None else (
            tuple(specialization.spec), list(self._specialization_topics_(specialization)))
        redirect_vars = getattr(specialization, 'redirect_vars', None) if specialization is not None else None
        version = self.version
        if specialization is not None and specialization.version is not ABSENT:
            version = specialization.version
        if specialization is not None and pretty:
            # type() describes THIS block, and rendering it under a
            # specialization's name would describe neither.
            raise ValueError("typestr(): pretty= and specialization= do not combine")
        if specialization is not None:
            if legacy_typing is None and specialization.legacy_typing:
                legacy_typing = True
            if legacy_signature is None and specialization.legacy_signature:
                legacy_signature = True
        if legacy_typing is None:
            legacy_typing = legacy
        if legacy_signature is None:
            legacy_signature = legacy
        legacy_typing = self._legacy_typing_(legacy_typing)
        if pretty:
            import pprint
            return pprint.pformat(
                self.type(deslash=deslash, legacy_typing=legacy_typing,
                              legacy_signature=legacy_signature), indent=2, width=120)
        # A redirection is emphatically NOT part of the identity: it says where
        # this block's data is read from, and a block does not become a
        # different block by being read from somewhere else. It used to be
        # appended here, which made hash() depend on WHEN it was first called
        # -- before a redirection was installed or after -- and moved
        # anchorkeypath, the journal directory and the redirection lookup along
        # with it. Masked, until keyby stopped naming the hash, by __setstate__
        # building the logger name out of self.key and caching _hash on the way.
        parts = [self.signaturestr(deslash=deslash, legacy_typing=legacy_typing,
                                legacy_signature=legacy_signature, omit=omit,
                                redirect_vars=redirect_vars)]
        # The narrower block's version when the specialization names one, this
        # class's otherwise. Taking it from the class unconditionally was what
        # made VERSION unbumpable while a specialization was live: the
        # reconstruction moved with the bump, onto an identity nothing built.
        parts.extend(f"{k}={v}" for k, v in self._type_entries_(specialization, with_block=with_block).items())
        parts.append(f"version={version}")
        parts.extend(self._topics_signature_(
            topics, declared=None if specialization is None else self._specialization_topics_(specialization)))
        tp = os.path.join(*parts)
        if deslash:
            tp = tp.replace('\\', '')
        return tp

    def get_hash(self, specialization: 'Datablock.Specialization | None' = None, *,
                 with_block: bool = True):
        """This block's hash, or the hash of one of its :attr:`SPECIALIZATIONS`.

        With *specialization*, the hash of the narrower block it describes --
        which is a real hash of a real identity, the one that block was built
        under, and so the one its journal entries are filed by. That is the
        whole mechanism: the reconstruction is a string operation on
        :meth:`typestr`, needing no access to the older class and no record that
        it ever existed.

        Cached per specialization, and never into ``_hash``: that one is this
        block's own, and a specialized hash is emphatically not it.

        *with_block* False: the hash under the rules from before a stack's type
        named its BLOCK -- see :meth:`typestr`. Not cached.
        """
        if not with_block:
            return hashlib.sha256(self.typestr(specialization=specialization,
                                               with_block=False).encode()).hexdigest()
        if specialization is None:
            if not hasattr(self, '_hash'):
                sha = hashlib.sha256()
                tp = self.typestr()
                sha.update(tp.encode())
                self._hash = sha.hexdigest()
                self.log.detailed(f"hash: ---------===---------> {tp=} ---> hash: {self._hash}")
            return self._hash
        cache = self.__dict__.setdefault('_specialized_hashes_', {})
        key = specialization.key
        if key not in cache:
            tp = self.typestr(specialization=specialization)
            cache[key] = hashlib.sha256(tp.encode()).hexdigest()
            self.log.detailed(f"get_hash({specialization!r}): {tp=} ---> {cache[key]}")
        return cache[key]

    def get_type(self, specialization: 'Datablock.Specialization | None' = None, **kwargs) -> dict:
        """``type(specialization=...)``: the structured identity, to pair with :meth:`get_hash`."""
        return self.type(specialization=specialization, **kwargs)

    def get_signature(self, specialization: 'Datablock.Specialization | None' = None, **kwargs) -> dict:
        """:meth:`signature`, or the narrower block's: the fields *specialization* pins, dropped."""
        if specialization is None:
            return self.signature(**kwargs)
        return self.get_type(specialization)['signature']

    def get_signaturestr(self, specialization: 'Datablock.Specialization | None' = None, **kwargs) -> str:
        """`signaturestr`, or the narrower block's: the fields *specialization* pins, dropped."""
        if specialization is None:
            return self.signaturestr(**kwargs)
        redirect_vars = getattr(specialization, 'redirect_vars', None)
        return self.signaturestr(omit=tuple(specialization.spec),
                                redirect_vars=redirect_vars, **kwargs)

    def get_typestr(self, specialization: 'Datablock.Specialization | None' = None, **kwargs) -> str:
        """``typestr(specialization=...)``: the identity :meth:`get_hash` is the sha256 of."""
        return self.typestr(specialization=specialization, **kwargs)

    def specialization_types(self) -> list:
        """:meth:`get_type` of every declared specialization, in declaration order.

        For looking at what each one reconstructs -- next to
        :meth:`specialization_typestrs`, and the hashes a journal is searched
        for, :meth:`specialization_hashes`. A reconstruction that differs from
        the identity on disk shows up here, before anything fails to resolve.
        """
        return [self.get_type(sp) for sp in (self.SPECIALIZATIONS or [])]

    def specialization_typestrs(self) -> list:
        """:meth:`get_typestr` of every declared specialization, in declaration order."""
        return [self.get_typestr(sp) for sp in (self.SPECIALIZATIONS or [])]

    def specialization_hashes(self) -> list:
        """:meth:`get_hash` of every declared specialization, in declaration order."""
        return [self.get_hash(sp) for sp in (self.SPECIALIZATIONS or [])]

    def matching_specializations(self):
        """The declared specializations whose pins this block satisfies, in order."""
        return [sp for sp in (self.SPECIALIZATIONS or [])
                if self._specialization_mismatch_(sp) is None]

    def specializations(self, journal=None):
        """One row per declared specialization, saying what it did.

        The debugging tool for this feature, and the reason a miss is never
        silent: a specialization that stopped matching -- a default that moved,
        a topic that was renamed, a build that was cleared -- shows up here as a
        row with a reason, rather than as a block that quietly rebuilds.
        """
        if isinstance(journal, BlocksJournal):
            journal = _shared_journal_(journal, self)
        rows = []
        for i, sp in enumerate(self.SPECIALIZATIONS or []):
            if getattr(sp, 'UNSAFE_redirect_all_topics', False):
                named = list(self.topics())
            elif getattr(sp, 'redirect_topics', None) is not None:
                named = list(self._toplevel_topics_(sp.redirect_topics))
            else:
                named = self._toplevel_topics_(self._specialization_topics_(sp))
            row = self.SpecializationRow(
                specialization=sp, hash=None, matches=False, why=None,
                entry=None, paths=None, topics=named,
                # This specialization's complement, which is what a build would
                # be left with if it were installed. Not `ownedtopics()`: that
                # answers for the redirection this block HAS, and this row is
                # about one it might not.
                builds=[t for t in self.topics() if t not in named],
            )
            try:
                why = self._specialization_mismatch_(sp)
            except (ValueError, KeyError) as e:
                row['why'] = str(e)
                rows.append(row)
                continue
            row['hash'] = self.get_hash(sp)
            if why is not None:
                row['why'] = why
                rows.append(row)
                continue
            row['matches'] = True
            reasons = []
            sp_j = journal
            if isinstance(sp_j, dict):
                anchor = self._specialization_anchor_(sp)
                sp_j = sp_j.get(anchor, sp_j.get(None))
            elif isinstance(sp_j, (list, tuple)):
                sp_j = sp_j[i] if i < len(sp_j) else None
            resolved = self._specialization_paths_(sp, journal=sp_j, why=reasons)
            if resolved is None:
                row['why'] = '; '.join(reasons) or (
                    f"no entry with hash {row['hash']} records data that is still there")
            else:
                row['paths'], entry = resolved
                row['entry'] = entry.block.id
            rows.append(row)
        return rows

    def find_specialization(self, journal=None):
        """The `SpecializationRow` build() would install here, or None -- found, NOT installed.

        The first, in declaration order, that applies, finds this block unbuilt
        where it covers, and resolves to data that is still there -- which is
        what `_install_specialization_` installs. Asked of a block
        constructed with `use_specializations=False`, it says what WOULD be
        installed without anything being redirected or recorded. For why each
        declared specialization did or did not apply, see `specializations()`.
        """
        if isinstance(journal, BlocksJournal):
            journal = _shared_journal_(journal, self)
        memo = {'journal': journal}
        for sp, (paths, entry) in self._specialization_candidates_(journal, _memo=memo):
            if getattr(sp, 'UNSAFE_redirect_all_topics', False):
                named = list(self.topics())
            elif getattr(sp, 'redirect_topics', None) is not None:
                named = list(self._toplevel_topics_(sp.redirect_topics))
            else:
                named = list(self._toplevel_topics_(self._specialization_topics_(sp)))
            return self.SpecializationRow(
                specialization=sp,
                hash=self.get_hash(sp),
                matches=True,
                why=None,
                entry=entry.block.id,
                paths=paths,
                topics=named,
                builds=[t for t in self.topics() if t not in named],
            )
        return None

    def UNSAFE_specialize(self, *, journal=None, dry_run: bool = False, dry_validate: bool = False, OVERRIDE: bool = False):
        """Install the applicable specialization AND record it in the journal.

        The explicit form of what build() does before building anything -- and
        construction never does: a redirection
        that outlives this instance, so the next run reads the narrower block's
        data without scanning the journal for it, and the journal says which
        specialization was installed and when. Here for a block constructed
        with ``use_specializations='memory'``, and for adopting one without
        building the topics it does not cover.

        *dry_run* reports what the first applicable specialization WOULD do and
        returns its proposed :class:`Redirection`, writing nothing -- the same
        handle :meth:`UNSAFE_redirect` has, *dry_validate* included, which makes
        the return a ``(Redirection, Validation)`` pair.
        """
        if not UNSAFE_allowed("UNSAFE_specialize", OVERRIDE=OVERRIDE):
            return None
        for sp in (self.SPECIALIZATIONS or []):
            if self._specialization_mismatch_(sp) is not None:
                continue
            res = self.UNSAFE_redirect(specialization=sp, journal=journal,
                                       dry_run=dry_run, dry_validate=dry_validate,
                                       OVERRIDE=True)
            if res:
                return res if dry_run else sp
        self.log.info(
            f"UNSAFE_specialize: nothing to install for {self.anchorkeypath}; "
            f"specializations(): {self.specializations()!r}"
        )
        return None

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
                topicpath = self._normtopic_(topicpath)
                if ensure_dirpath and topicpath and self._redirect_path_(*topicpath) is not None:
                    # ensure_dirpath=True is how this codebase asks for a place
                    # to WRITE, and a redirected topic's place belongs to
                    # another block. Under a partial redirection a __build__
                    # runs for the topics that are not redirected, and without
                    # this it would go on writing all of them -- into the data
                    # it was supposed to be reusing.
                    raise ValueError(
                        f"{self.__class__.__name__}: topic {'/'.join(topicpath)!r} is "
                        f"REDIRECTED to {self._redirect_path_(*topicpath)!r}, which belongs "
                        f"to another block; refusing to prepare it for writing. Build "
                        f"{self.ownedtopics()!r} instead (see ownedtopics()), or construct "
                        f"with redirect=False to build this block's own data."
                    )
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
        topicpath = self._normtopic_(topicpath)
        if self.is_topicgroup(*topicpath):
            # A group has no listing of its own: concatenate its leaves'.
            return [entry
                    for tp in self._leaves_under_(*topicpath)
                    for entry in self.ls(*tp, detail=detail, local=local)]
        p = self.path(*topicpath, local=local)
        return ls_path(fs, p, self._is_dir_topic_(*topicpath), detail=detail)

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
        topicpath = self._normtopic_(topicpath)
        if self.is_topicgroup(*topicpath):
            return [entry
                    for tp in self._leaves_under_(*topicpath)
                    for entry in self.list(*tp, local=local)]
        p = self.path(*topicpath, local=local)
        return list_path(fs, p, self._is_dir_topic_(*topicpath))

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
        return size(self.list(*self._normtopic_(topicpath), local=local))


    def dirpath(
        self,
        *topicpath,
        ensure: bool = False,
        list: bool = False,
        local: bool = False,
        redirect: bool = True,
    ):
        """The directory for a topic, one path segment per level.

        A group has a directory of its own -- ``dirpath('data')`` is the parent
        of ``dirpath('data', 'frames')`` -- so this answers for groups and
        leaves alike.  A `SYNTOPIC` has no location and gives ``None``.

        A redirected topic answers with the directory of the path it is
        redirected to -- the path itself for a directory topic, its parent for a
        file one -- so a listing of a redirected block lists the data it
        actually reads. As in `path()`, ``local=True`` is never redirected:
        the local cache is this block's own.
        """
        topicpath = self._normtopic_(topicpath)
        if self._is_syntopic_(*topicpath):
            # No location: nothing to name, and nothing to create for `ensure`.
            return None
        anchorkeypath = self.localanchorkeypath if local else self.anchorkeypath
        fs = self.localfs if local else self.fs
        if not local and redirect:
            redirected = self._redirect_dirpath_(*topicpath)
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
        if self._is_dir_topic_(topic):
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
        return self._anchorpath_()

    def write_journal_entry(self, event: str, *, note: str = None, inline_note: bool = False,
                            message: str = None, inline_message: bool = False, journal_prefix: str = '',
                            redirection: 'str | dict | None' = None):
        """Write one journal entry for *event*, under the current `Datajournal` session. See `Datajournal.write`."""
        return Datajournal.write(self, event, note=note, inline_note=inline_note,
                                 message=message, inline_message=inline_message,
                                 journal_prefix=journal_prefix, redirection=redirection)

    @staticmethod
    def Datajournal(anchor, loc: int = None, *, iloc: int = None, datalake=None, storage_options=None, log=None, n_workers=8, index=None, unnormalized: bool = False, url=None, **filter_kwargs):
        """*anchor*'s journal. See `Datajournal.read`."""
        return Datajournal.read(anchor, loc, iloc=iloc, datalake=one_datalake(datalake, url, 'Datajournal'),
                                storage_options=storage_options,
                                log=log, n_workers=n_workers, index=index,
                                unnormalized=unnormalized, **filter_kwargs)

    def datajournal(self, loc: int = None, *, iloc: int = None, datalake=None, storage_options=None, log=None, n_workers=None, index: str | None = None, unnormalized: bool = False, url=None, **filter_kwargs):
        """This block's anchor's journal, under this block's datalake. See `Datajournal.read`."""
        if loc is not None and iloc is not None:
            raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
        return Datajournal.read(
            self.anchor,
            loc=loc,
            iloc=iloc,
            datalake=self.datalake if (datalake is None and url is None) else one_datalake(datalake, url, 'datajournal'),
            storage_options=self.storage_options if storage_options is None else storage_options,
            log=getattr(self, 'log', None) if log is None else log,
            n_workers=n_workers,
            index=index,
            unnormalized=unnormalized,
            **filter_kwargs,
        )

    def lastbuilt(self, index: str | None = None):
        """Return the most recent 'build:end' DatajournalEntry, or None."""
        j = self.datajournal(event='build:end', index=index)
        if len(j) == 0:
            return None
        return j.get(0, dropna=True)

    def running(self, index: str | None = None):
        """Return the latest 'build:start' DatajournalEntry with no matching 'build:end', or None."""
        j = self.datajournal(index=index)
        if len(j) == 0:
            return None
        started = set(j[j['event'] == 'build:start']['hash'])
        ended = set(j[j['event'] == 'build:end']['hash'])
        running_hashes = started - ended
        if not running_hashes:
            return None
        running_entries = j[(j['event'] == 'build:start') & (j['hash'].isin(running_hashes))]
        return DatajournalEntry(running_entries.iloc[0].dropna(), storage_options=self.storage_options)

    # 3. Accessors ---------------------------------------------------------

    @property
    def url(self) -> str:
        """The name `datalake` had first."""
        return self.datalake

    @url.setter
    def url(self, value):
        self.datalake = value

    @property
    def is_local_fs(self):
        """True when this block's storage is on a local filesystem."""
        protocol = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
        return protocol in ('file', 'local', '')

    @functools.cached_property
    def redirection(self):
        """Where this block reads from instead, as a :attr:`Redirection`, or None.

        An informational property describing how this block is redirected.
        """
        return self.get_redirection()

    @property
    def version(self):
        """User-defined version of this Datablock subclass. Used in hash computation — do NOT include dbx version here."""
        return self.VERSION if hasattr(self, 'VERSION') else None

    @property
    def dbx_version(self):
        """The dbx library version. Recorded in the journal but NOT used in hash computation."""
        return __version__

    @property
    def tree(self):
        """The build tree this block belongs to: given, or generated once and kept.

        One live instance keeps one tree id for as long as it is alive, and a
        whole build tree shares it, because `build_tree` and `Datastack.block`
        hand it down. That is what makes a run's journal entries findable
        together -- ``id`` identifies one row and ``hash`` one block, but
        neither says "these were written by the same run".

        Not part of :attr:`signature`, so which run built a block cannot
        change what the block IS.
        """
        if getattr(self, '_tree_', None) is None:
            self._tree_ = (uuid.uuid4().hex[:16]
                           if getattr(self, '_uuid16_', False) else str(uuid.uuid4()))
        return self._tree_

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
        var = self._spec_to_var_(self.spec)
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

    @property
    def hash(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hash
        # computed with the older code.
        return self.get_hash()

    @property
    def specialization(self):
        """The :class:`Specialization` this block is reading through, or None."""
        sp = self.__dict__.get('__specialization__')
        if sp is not None:
            return sp
        red = self.redirection
        return red.specialization if red is not None else None

    #SPECIALIZE: END
    @property
    def code(self):
        if not hasattr(self, '_code'): 
            if getattr(self, '_code_', None) is not None:
                self._code = self._code_
            elif getattr(self, '_subhash_', None) is not None:
                self._code = self._subhash_
            else:
                sha = hashlib.sha256()
                sig = self.signaturestr()
                sha.update(sig.encode())
                self._code = sha.hexdigest()
                self.log.detailed(f"code: ---------===---------> {sig=} ---> code: {self._code}")
        return self._code

    @property
    def subhash(self):
        return self.code

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
            key = self.signaturestr()

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

    @property
    def anchorkey(self):
        return self._anchorkey_()

    @property
    def anchorkeypath(self):
        return self._anchorkeypath_()

    @property
    def localanchorpath(self):
        return self._anchorpath_(local=True)

    @property
    def localanchorkeypath(self):
        return self._anchorkeypath_(local=True)

    #PATHS: END

    #LOG LEVEL: BEGIN
    @property
    def log(self) -> Datalog:
        """This block's view of the current `Datalog`: named for the block, its own levels over the scope's.

        A view looks its levels up when it is written to, so the one kept here
        hears whatever ``with Datalog(...)`` is open then. A level the block
        was not given -- None -- is the scope's, then the environment's. Made
        again only when the block's name or levels have moved; never pickled.
        """
        d = self.__dict__
        name, levels = d.get('_log_name') or type(self).__name__, d.get('_log_levels', {})
        log = d.get('_log')
        if log is None or log.name != name or d.get('_log_made_from') != levels:
            log = d['_log'] = Datalog(name, **levels)
            d['_log_made_from'] = dict(levels)
        return log

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

    # 4. Helpers -----------------------------------------------------------

    @staticmethod
    def _coerce_to_annotation_(value, annotation):
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

    def _typed_specdict_(self, *, legacy: 'bool | None' = None, omit=(),
                         redirect_vars=None) -> dict:
        """The spec as real Python values -- ints as ints, blocks as sub-dicts.

        Built from ``self.var``, NOT by parsing the rendered signature. The
        rendering is text, and reading it back gives text: that is why
        ``sigdict()`` (now ``sig()``) used to report ``'256'`` for a field holding ``256``,
        with no coercion bug anywhere in sight.

        Speclines stay strings, since a specline IS a string; every other leaf
        is its declared type.

        *redirect_vars*, when given, is a ``{current_name: historical_name}``
        mapping.  Each current field is emitted under its historical name, and
        the keys are sorted by historical name so the rendered spec matches the
        identity of the build before the rename.
        """
        legacy = self._legacy_typing_(legacy)
        renames = redirect_vars or {}
        fields = self.VAR.__dataclass_fields__
        keys = [f.name for f in fields.values()]
        if not legacy:
            # Sort by the OUTPUT name: the historical name for renamed fields,
            # the current name for everything else.  This reproduces the sort
            # order the narrower block used, because *its* fields had those names.
            keys = sorted(keys, key=lambda k: renames.get(k, k))
        # *omit* drops fields the class did not used to have, so what is left
        # renders exactly as the narrower block rendered it. Dropping keys
        # cannot reorder the rest, which is what makes the reconstruction exact.
        if omit:
            keys = [k for k in keys if k not in set(omit)]

        out = {}
        for k in keys:
            out_key = renames.get(k, k)
            value = getattr(self.var, k)
            raw = self.spec[k] if (isinstance(getattr(self, 'spec', None), dict) and k in self.spec) else value
            if isinstance(value, Datablock):
                out[out_key] = value._typed_specdict_(legacy=legacy)
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
                out[out_key] = (evaluated._typed_specdict_(legacy=legacy)
                          if isinstance(evaluated, Datablock) else raw)
            else:
                out[out_key] = self._coerce_to_annotation_(value, fields[k].type)
        return out

    def _legacy_norm_(self) -> bool:
        """Whether the pre-LEGACY_NORM rendering applies: root kwargs, str()'d spec.

        Untouched by the typing change, and deliberately SEPARATE from
        :meth:`_legacy_typing_`. signature() and hash are relocatable -- free of
        url -- and only this flag has ever decided otherwise. Tying the typing
        opt-out to it would put a url into the identity of every block pinned
        for typing, which is the opposite of preserving it.
        """
        return bool(getattr(self, 'LEGACY_SIGNATURE', False)
                    or getattr(self, 'LEGACY_NORM', False))

    def _legacy_typing_(self, legacy: 'bool | None' = None) -> bool:
        """Whether to render the pre-typing way: text leaves, nested blocks as strings.

        Independent of :meth:`_legacy_norm_`: this chooses the TYPING, and that
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
    def _topictext_(node, modern):
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

    def _modern_topics_(self, topics=ABSENT) -> bool:
        """Whether *topics* is spelled with the markers rather than the sentinels.

        Derived from the declaration rather than announced by a flag: a TOPICS
        holding a marker is a modern one, and there is nothing else it could
        mean.  Derived on each call, so a TOPICS assigned or amended after the
        class body -- or computed per instance, as
        :class:`~dbx.datatables.DatatablePart`'s is -- is answered as it stands.

        A declaration holding both spellings has no era, and raises rather than
        rendering half of itself each way.
        """
        if topics is ABSENT:
            topics = getattr(self, 'TOPICS', None)
        modern, legacy = self._topic_spellings_(topics)
        if modern and legacy:
            raise ValueError(
                f"{self.__class__.__name__}: TOPICS mixes the topic markers with "
                f"the sentinels they replace -- {modern} against {legacy}. The two "
                f"render differently, so one declaration cannot be both: spell "
                f"every topic the one way or the other"
            )
        return bool(modern)

    def _topic_spellings_(self, topics, prefix=()):
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
                nested = self._topic_spellings_(node, path)
                modern.extend(nested[0])
                legacy.extend(nested[1])
            elif is_topicmarker(node):
                modern.append('/'.join(path))
            elif self._node_is_sentinel_(node):
                legacy.append('/'.join(path))
        return modern, legacy

    @staticmethod
    def _node_is_sentinel_(node):
        """True when node is one of the sentinels the markers replace."""
        return node is DIRTOPIC or (isinstance(node, tuple) and len(node) == 0)

    @property
    def _url_(self):
        return self._datalake_

    @_url_.setter
    def _url_(self, value):
        self._datalake_ = value

    @classmethod
    def _specialization_records_(cls, given):
        """*given* SPECIALIZATIONS as `Specialization.to_dict` records, or None for "the class's"."""
        if given is None:
            return None
        if isinstance(given, str):
            try:
                given = ast.literal_eval(given)     # as a quote() renders them, read back as text
            except Exception:
                import builtins, dbx
                cxt = dict(vars(dbx))
                spec_cls = getattr(cls, 'Specialization', Datablock.Specialization)
                cxt['Specialization'] = spec_cls
                cxt.setdefault('SAME', SAME)
                cxt.setdefault('ABSENT', ABSENT)
                given = builtins.eval(given, cxt)
        if isinstance(given, (Datablock.Specialization, dict)):
            given = [given]
        if not isinstance(given, (list, tuple)):
            raise TypeError(f"SPECIALIZATIONS= is a list of Specializations or None, got {given!r}")
        records = []
        for sp in given:
            if isinstance(sp, dict):
                sp = cls.Specialization.from_record(sp)
            if not isinstance(sp, Datablock.Specialization):
                raise TypeError(f"SPECIALIZATIONS= holds {sp!r}, which is not a Specialization")
            records.append(sp.to_dict())
        return records

    def _log_name_(self):
        """The logger's name: anchor and key -- so the hash -- and the tag when there is one."""
        log_name = f"{self.anchor}/{self.key}"
        if self._anchor_ is not None:
            log_name = f"{self.fqcn}: {log_name}"
        if self._tag_ is not None:
            log_name = f"{log_name} [{self._tag_}]"
        return log_name

    def _process_redirect_(self):
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
            target_entry = self._find_journal_entry_by_code_(code_of_target)
            if target_entry is None:
                raise ValueError(f"redirect failed: no journal entry found for code {code_of_target!r}")
            resolved_paths = target_entry.block.paths()

        elif filter_spec is not None:
            target_entry = self._find_journal_entry_by_filter_(filter_spec)
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

    def _find_journal_entry_by_code_(self, code: str):
        try:
            j = self.datajournal(id=code)
            if len(j) > 0:
                return DatajournalEntry(j.iloc[0].dropna(), storage_options=self.storage_options)
        except Exception:
            pass

        try:
            fs, root = fsspec.url_to_fs(self.datalake, **(self.storage_options or {}))
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

    def _find_journal_entry_by_filter_(self, filter_spec: Union[dict, str]):
        try:
            if isinstance(filter_spec, dict):
                j = self.datajournal(**filter_spec)
            elif isinstance(filter_spec, str):
                j = self.datajournal(event=filter_spec)
                if len(j) == 0:
                    j = self.datajournal(id=filter_spec)
            else:
                return None
            if len(j) > 0:
                return DatajournalEntry(j.iloc[0].dropna(), storage_options=self.storage_options)
        except Exception:
            pass
        return None

    def _resolve_legacy_CONFIG_(self):
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

    def _url_to_fs_(self, path):
        """Wrapper around ``fsspec.url_to_fs`` that injects ``self.storage_options``."""
        return fsspec.url_to_fs(path, **self.storage_options)


    @property
    def _topics_is_list_(self):
        """True when TOPICS is defined as a list (directory-per-topic mode)."""
        return hasattr(self, 'TOPICS') and isinstance(self.TOPICS, list)

    @property
    def _topicfiles_(self):
        """The effective topic → filename mapping.

        Returns TOPICS when it is a dict, otherwise None.  For a hierarchical
        TOPICS the values of group keys are themselves such mappings.
        """
        if hasattr(self, 'TOPICS') and isinstance(self.TOPICS, dict):
            return self.TOPICS
        return None

    @staticmethod
    def _normtopic_(topicpath):
        """Accept ``('data', 'frames')``, ``(('data', 'frames'),)`` and ``('data',)``.

        The tuple form lets a caller feed a :meth:`leaftopics` entry straight
        back in without unpacking it.
        """
        if len(topicpath) == 1 and isinstance(topicpath[0], (tuple, list)):
            return tuple(topicpath[0])
        return tuple(topicpath)

    def _topicnode_(self, *topicpath):
        """Resolve a topic path to its TOPICS entry.

        Returns a filename (``str``), :data:`DIRTOPIC`, :data:`SYNTOPIC`, or a
        ``dict`` for a group.  Raises KeyError naming the offending segment
        when the path does not exist, so a typo in a nested name says which
        level it failed at rather than surfacing as a missing file later.
        """
        topicpath = self._normtopic_(topicpath)
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
        self._modern_topics_()
        if self._topics_is_list_:
            if len(topicpath) > 1:
                raise KeyError(
                    f"list-TOPICS has no groups: {'/'.join(topicpath)} is nested"
                )
            if topicpath[0] not in self.TOPICS:
                raise KeyError(f"topic {topicpath[0]!r} not in {list(self.TOPICS)}")
            return DIRTOPIC

        node = self.TOPICS
        for i, name in enumerate(topicpath):
            self._check_topicname_(name)
            if not isinstance(node, dict):
                raise KeyError(
                    f"topic {'/'.join(topicpath[:i])!r} is a leaf, "
                    f"so it has no member {name!r}"
                )

            node = node[name]
            if not (isinstance(node, (dict, str)) or node is DIRTOPIC
                    or self._node_is_syntopic_(node) or is_topicmarker(node)):
                raise TypeError(
                    f"TOPICS entry {'/'.join(topicpath[:i+1])!r} is {node!r}; "
                    f"expected a filename, a topic marker, DIRTOPIC, SYNTOPIC, "
                    f"or a dict of these"
                )
        return node

    @staticmethod
    def _node_is_syntopic_(node):
        """True when node is :data:`SYNTOPIC` or the :class:`SYNTHETIC` marker."""
        return (isinstance(node, tuple) and len(node) == 0) or is_topicmarker(node, SYNTHETIC)

    @staticmethod
    def _check_topicname_(name):
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

    def _leaves_under_(self, *topicpath):
        """Leaf topic paths at or below *topicpath*, as full tuples from the root."""
        topicpath = self._normtopic_(topicpath)
        node = self._topicnode_(*topicpath)

        def walk(node, prefix):
            if not isinstance(node, dict):
                yield prefix
                return
            for name, child in node.items():
                yield from walk(child, prefix + (name,))

        return list(walk(node, topicpath))

    @staticmethod
    def _node_is_dirtopic_(node):
        """True when node is :data:`DIRTOPIC` or a :class:`DATADIR` marker.

        A parameterised marker is a subclass of the one it parameterises, so
        ``DATASLICE(idx='int')`` -- a slice IS a directory -- lands here too,
        and so does the deprecated :class:`DIR`.
        """
        return node is DIRTOPIC or is_topicmarker(node, DATADIR)

    def _is_dir_topic_(self, *topicpath):
        """True when the topic resolves to a directory rather than a file.

        True for list-TOPICS entries and for :data:`DIRTOPIC` leaves.  A
        :data:`SYNTOPIC` is neither, and neither is a group -- a group has a
        directory, but :meth:`path` describes it by its members.
        """
        topicpath = self._normtopic_(topicpath)
        if not topicpath or topicpath[0] is None:
            return False
        node = self._topicnode_(*topicpath)
        return self._node_is_dirtopic_(node)

    def _is_syntopic_(self, *topicpath):
        """True when the topic is declared :data:`SYNTOPIC` -- so it has no location.

        Only dict-TOPICS can declare one; every entry of a list-TOPICS is a
        directory.
        """
        topicpath = self._normtopic_(topicpath)
        if not topicpath or not self.has_topics() or self._topics_is_list_:
            return False
        try:
            return self._node_is_syntopic_(self._topicnode_(*topicpath))
        except (KeyError, TypeError):
            return False

    def _transfer_callback_(self, desc, *, show_progress: bool):
        """An fsspec transfer callback: a tqdm byte-progress bar when
        *show_progress*, otherwise fsspec's default no-op callback."""
        if not show_progress:
            return fsspec.callbacks.NoOpCallback()
        return fsspec.callbacks.TqdmCallback(tqdm_kwargs=dict(desc=desc, unit='B', unit_scale=True))

    def _iter_var_blocks_(self, exemptions_attr=None, skip_callback=None):
        """Yield (key, Datablock) pairs from self.var that are not in the given exemptions list."""
        exemptions = self._tree_skips_(exemptions_attr) if exemptions_attr else set()
        for s in self.spec.keys():
            if s in exemptions:
                if skip_callback:
                    skip_callback(s)
                continue
            c = getattr(self.var, s)
            if isinstance(c, Datablock):
                yield s, c

    def _tree_skips_(self, attr):
        """The spec keys *attr* exempts, as a set, refusing what cannot be one.

        A bare string is the error worth catching: ``TREE_SKIP_BUILDING =
        'ckpt_builder'`` -- the tuple written without its trailing comma --
        leaves membership testing SUBSTRINGS, so ``'ckpt'`` would be exempt and
        ``'ckpt_builderish'`` would not, and nothing would say so.

        The retired ``BUILD_TREE_EXEMPTIONS`` is refused here rather than
        ignored for the same reason: a class still declaring it would go on
        having its subtree built on its behalf, which for a warm-start source
        is a full training run nobody asked for.
        """
        retired = type(self).__dict__.get('BUILD_TREE_EXEMPTIONS')
        if retired is None:
            for klass in type(self).__mro__:
                if 'BUILD_TREE_EXEMPTIONS' in klass.__dict__:
                    retired = klass.__dict__['BUILD_TREE_EXEMPTIONS']
                    break
        if retired is not None:
            raise AttributeError(
                f"{type(self).__name__} declares BUILD_TREE_EXEMPTIONS, which is "
                f"retired -- rename it to TREE_SKIP_BUILDING. Left as it is, "
                f"build_tree() would descend into {tuple(retired)!r} and build "
                f"what the declaration meant to exempt."
            )
        value = getattr(self, attr, ())
        if isinstance(value, str):
            raise TypeError(
                f"{type(self).__name__}.{attr} is the string {value!r}, not a "
                f"tuple of spec keys -- a one-element tuple needs its trailing "
                f"comma: ({value!r},). As it stands, membership tests substrings."
            )
        return set(value)

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


    def _mapped_paths_(self, paths, topic_map, topics=None):
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
        # *topics* redirects only what it names, leaving every other topic to
        # read as it would unredirected -- which `path` already does for a
        # topic with no redirected path. Without it a redirection takes over
        # every topic the other side happens to record, which is right when the
        # other side IS this block elsewhere, and wrong when it is a narrower
        # block this one has grown past.
        if topics is not None:
            wanted = set(self._toplevel_topics_(topics))
            native_topics = [t for t in native_topics if t in wanted] or None
            if native_topics is None:
                self.log.warning(
                    f"redirection: topics={list(topics)!r} names none of this block's "
                    f"topics {self.topics()!r}; nothing is redirected"
                )
                return {}
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

    def _redirect_path_(self, *topicpath):
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

    def _redirect_dirpath_(self, *topicpath):
        """The directory of the path *topicpath* is redirected to, or None.

        A directory topic IS its path; a file topic's directory is the parent of
        the file it was redirected to -- which is what makes ``ls``, ``list``
        and ``size`` describe the redirected-to data rather than this block's
        empty one, since all three resolve through here.
        """
        redirected = self._redirect_path_(*topicpath)
        if redirected is None:
            return None
        if isinstance(redirected, dict):
            # A group: it has no path of its own, and its members carry theirs.
            return None
        node = self._topicnode_(*topicpath)
        if self._node_is_dirtopic_(node):
            return redirected
        return os.path.dirname(redirected)

    def _find_block_redirection_yaml_(self, topic: str, path: str) -> str | None:
        """Find the .redirection/paths.yaml for the block owning *path*, or None."""
        parts = [p for p in str(topic).split('/') if p]
        max_levels = len(parts) + 1
        d = os.path.dirname(path)
        for _ in range(max_levels):
            candidate = os.path.join(d, '.redirection', 'paths.yaml')
            try:
                if self.fs.exists(candidate):
                    return candidate
            except Exception:
                pass
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return None

    def _chase_redirected_paths_(self, paths: dict[str, str]) -> dict[str, str]:
        """Chase chained redirections for each topic until reaching terminal, non-redirected paths."""
        if not paths or not isinstance(paths, dict):
            return paths
        final_paths = {}
        for topic, path in paths.items():
            if not isinstance(path, str):
                final_paths[topic] = path
                continue
            curr_path = path
            visited = set()
            while curr_path and curr_path not in visited:
                visited.add(curr_path)
                red_yaml = self._find_block_redirection_yaml_(topic, curr_path)
                if red_yaml is None:
                    break
                try:
                    target_red_paths = read_yaml(red_yaml, storage_options=self.storage_options)
                    if isinstance(target_red_paths, dict) and topic in target_red_paths:
                        next_path = target_red_paths[topic]
                        if isinstance(next_path, str) and next_path != curr_path:
                            curr_path = next_path
                            continue
                except Exception:
                    pass
                break
            final_paths[topic] = curr_path
        return final_paths

    def _journal_hashdirpath_(self):
        """The directory holding THIS block's journal entries, and no others."""
        return Datajournal.dirpath(self, 'journal')

    def _recorded_redirection_(self, journal=None):
        """The latest redirection recorded for this block's hash, or None.

        Latest, because a redirection is a correction and the newest one is the
        one still meant.
        """
        if journal is not None and isinstance(journal, pd.DataFrame) and not journal.empty:
            if 'hash' in journal.columns and 'redirection' in journal.columns:
                mask = journal['hash'] == self.hash
                if 'anchor' in journal.columns:
                    mask = mask & (journal['anchor'] == self.anchor)
                sub = journal[mask]
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
            dirpath = self._journal_hashdirpath_()
            legacy_dirpath = os.path.join(
                Datablock._dbxanchorpathx_(self.datalake, self.anchor, 'journal',
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

    def _redirect_entry_(self, filter, journal=None):
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
                j = self.datajournal(**dict(filter))
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

    def _UNSAFE_copy_fs_(self, *, src_path, dst_path, recursive: bool = False):
        """Copy a single file or directory between two fsspec paths.

        fsspec does not implement a generic cross-filesystem ``.copy()``, so
        this dispatches to put/get directly when either side is local, and
        falls back to a temporary local directory when both are remote.
        """
        src_fs, _ = self._url_to_fs_(src_path)
        dst_fs, _ = self._url_to_fs_(dst_path)

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

    def _UNSAFE_copy_file_(self, src_path, dst_path):
        """Copy a single file, preferring a fast server-side blob copy.

        When *src_path* and *dst_path* resolve to the same non-local
        filesystem (e.g. two Azure blob paths in the same account), this
        does a direct blob-to-blob copy with no data transiting the local
        machine. Otherwise it falls back to :meth:`_UNSAFE_copy_fs_`
        (get+put, possibly via a local temporary directory), which is the
        only option when a real cross-filesystem or local-disk hop is
        required. Mirrors the ``use_server_side`` branch already used by
        :meth:`_UNSAFE_copy_topic_dir_` for whole-directory copies, exposed
        here for single-file copies (e.g. a subclass copying a subset of a
        topic's files instead of the whole directory).
        """
        src_fs, _ = self._url_to_fs_(src_path)
        dst_fs, _ = self._url_to_fs_(dst_path)
        if src_fs == dst_fs and 'file' not in getattr(src_fs, 'protocol', ()):
            dst_dir = os.path.dirname(dst_path)
            if dst_dir:
                dst_fs.makedirs(dst_dir, exist_ok=True)
            dst_fs.cp_file(src_path, dst_path)
        else:
            self._UNSAFE_copy_fs_(src_path=src_path, dst_path=dst_path, recursive=False)

    @staticmethod
    def _topicpaths_lookup_(topicpaths, topicpath):
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

    def _UNSAFE_copy_topic_file_(self, topic, anchorkeypath, *, topicpaths=None):
        """Copy the individual .path(topic) file."""
        topic = self._normtopic_((topic,))
        dst_path = self.path(*topic)
        if topicpaths is not None:
            _src_path = self._topicpaths_lookup_(topicpaths, topic)
        else:
            if self._topicfiles_ is None:
                raise ValueError(
                    f"Cannot copy topic file for {'/'.join(topic)!r}: TOPICS is not a dict "
                    f"(no filename mapping). Use always_copy_whole_dirpath=True for list-mode topics."
                )
            _src_path = os.path.join(*topic, _topic_filename_(self._topicnode_(*topic)))
        if dst_path is not None:
            src_path = os.path.join(anchorkeypath, _src_path)
            self.log.detailed(f"Copying file {src_path} to {dst_path}")
            self._UNSAFE_copy_fs_(src_path=src_path, dst_path=dst_path, recursive=False)

    def _UNSAFE_copy_topic_dir_(self, topic, anchorkeypath, *, topicpaths=None):
        """Copy the entire .dirpath(topic) directory."""
        topic = self._normtopic_((topic,))
        if topicpaths is not None:
            _src_path = self._topicpaths_lookup_(topicpaths, topic)
        else:
            _src_path = os.path.join(*topic)
        src_path = os.path.join(anchorkeypath, _src_path)
        src_fs, _ = self._url_to_fs_(src_path)
        dest_fs, _ = self._url_to_fs_(self.dirpath(*topic))
        use_server_side = (src_fs == dest_fs
                           and 'file' not in getattr(src_fs, 'protocol', ()))
        # Ensure dst dir pre-exists only for server-side copy (Azure) or list-mode topics.
        # For dict-mode fscopy, pre-creating dst causes fsspec to copy src INTO dst instead of AS dst.
        ensure = use_server_side or (self._topicfiles_ is None)
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
            # _fscopy_item_callable_ is a module-level function (picklable for multiprocessing).
            items = [src_fs.unstrip_protocol(p)
                     for p in src_fs.ls(src_path, detail=False)]
            if items:
                _storage_options = getattr(self, 'storage_options', {})
                _n_workers_ = getattr(self, 'n_workers', 1)
                _tag = f"fscopy {len(items)} items [{_src_path}]"
                _executor_kwargs_ = dict(n_workers=_n_workers_, tag=_tag)
                if (hasattr(self, 'multiprocessing_start_method')
                        and self.multiprocessing_start_method is not None
                        and (self.parallelization or '').lower() in ('multiprocessing', 'torch_multiprocessing')):
                    _executor_kwargs_['start_method'] = self.multiprocessing_start_method
                _executor = callable_executor(self.parallelization, **_executor_kwargs_)
                _callables = [
                    functools.partial(
                        _fscopy_item_callable_,
                        src_item,
                        dst_path.rstrip('/') + '/' + src_item.rstrip('/').rsplit('/', 1)[-1],
                        _storage_options,
                    )
                    for src_item in items
                ]
                _executor.exec_callables(_callables)
        else:
            self._UNSAFE_copy_fs_(src_path=src_path, dst_path=dst_path, recursive=True)

    def _UNSAFE_copy_topic_(self, topic, anchorkeypath, *, topicpaths=None, always_copy_whole_dirpath: bool = False, **kwargs):
        """Copy one topic's data from anchorkeypath into this Datablock.

        Dispatches to :meth:`_UNSAFE_copy_topic_dir_` or
        :meth:`_UNSAFE_copy_topic_file_` depending on TOPICS shape. Overriding
        this in a subclass is the extension point for customizing how a
        *specific* topic gets copied (e.g. copying only a subset of files
        instead of the whole directory) while leaving the rest of
        :meth:`UNSAFE_copy_from` (overwrite check, journal entries,
        post-copy validation) untouched -- see
        ``IJEPAsaurUSStill._UNSAFE_copy_topic_`` in soundworld for an example
        that restricts the ``ckpts`` topic to a subset of checkpoints.
        ``**kwargs`` is accepted (and ignored here) so subclasses can declare
        extra keyword-only parameters on their override without changing
        this base signature; :meth:`UNSAFE_copy_from` forwards its own
        ``**kwargs`` to every topic's call.
        """
        topic = self._normtopic_((topic,))
        if self.is_topicgroup(*topic):
            for leaf in self._leaves_under_(*topic):
                self._UNSAFE_copy_topic_(leaf, anchorkeypath, topicpaths=topicpaths,
                                         always_copy_whole_dirpath=always_copy_whole_dirpath,
                                         **kwargs)
            return
        if self._is_syntopic_(*topic):
            # No location on either side -- there is nothing to copy.
            self.log.verbose(f"Skipping SYNTOPIC topic {'/'.join(topic)}: it has no location")
            return
        # Use directory copy when:
        #  - always_copy_whole_dirpath is explicitly requested, OR
        #  - TOPICS is a list (self._topicfiles_ is None -> every topic IS a dir), OR
        #  - TOPICS is a dict but this topic maps to DIRTOPIC (directory-only topic)
        use_dir = (
            always_copy_whole_dirpath
            or self._topicfiles_ is None
            or (isinstance(self._topicfiles_, dict) and self._is_dir_topic_(*topic))
        )
        if use_dir:
            self.log.verbose(f"Using copy_topic_dir for topic {topic}: BEGIN")
            self._UNSAFE_copy_topic_dir_(topic, anchorkeypath, topicpaths=topicpaths)
            self.log.verbose(f"Using copy_topic_dir for topic {topic}: END")
        else:
            self.log.verbose(f"Using copy_topic_file for topic {topic}: BEGIN")
            self._UNSAFE_copy_topic_file_(topic, anchorkeypath, topicpaths=topicpaths)
            self.log.verbose(f"Using copy_topic_file for topic {topic}: END")

    def _spec_to_var_(self, spec):
        var = self.VAR(**spec)
        replacements = {}
        for field in fields(var):
            term = getattr(var, field.name)
            if issubclass(self.VAR, Datablock.VAR):
                # Named, so a refusal says which field is at fault rather than
                # leaving it to be bisected.
                getter = Datablock.VAR.LazyLoader(
                    term,
                    name=field.name,
                    owner=self.__class__.__name__,
                    exempt=field.name in (getattr(self, 'VAR_IDENTITY_EXEMPTIONS', None) or ()),
                )
            else:
                getter = eval(term)
            replacements[field.name] = getter
        var = replace(var, **replacements)
        # Guarded: rendering VAR evaluates every lazy field, which is no business of a log line.
        if self.log.ist('detailed'):
            self.log.detailed(f"Made {var=} from {spec=}")
        return var

    def _adopt_(self, child, *, keyby: bool = False):
        """Hand *child* what it should inherit from this block.

        Called by the framework AFTER the user's hook returns, so a subclass
        that only overrides ``__block__`` never has to think about it.
        """
        kw = {'tree': self.tree}
        if keyby:
            keyby_val = getattr(self, 'keyby', None)
            if keyby_val is not None:
                kw['keyby'] = keyby_val
        return child.set(**kw)

    @functools.cached_property
    def _rootkwargs_(self):
        """The root kwargs as quote(), cite() and repr() render them: ``datalake=``, ``anchor=``."""
        rootkwargs = {}
        if self._datalake_ is not None:
            rootkwargs['datalake'] = self._datalake_
        if self._anchor_ is not None:
            rootkwargs['anchor'] = self._anchor_
        return rootkwargs

    @functools.cached_property
    def _identity_rootkwargs_(self):
        """The root kwargs as a LEGACY signature renders them -- spelled ``url=``, as it always was.

        A block built under LEGACY_SIGNATURE or LEGACY_NORM carries its root
        kwargs in its identity, so the text ``url=...`` is in its hash, and the
        rename to datalake may not reach it.
        """
        rootkwargs = {}
        if self._datalake_ is not None:
            rootkwargs['url'] = self._datalake_
        if self._anchor_ is not None:
            rootkwargs['anchor'] = self._anchor_
        return rootkwargs

    @functools.cached_property
    def _tailkwargs_(self):
        state = self.__getstate__()
        tailkwargs = {
            k: v
            for k, v in state.items()
            # 'tree' groups a build tree's journal entries; pinning one into a
            # recorded quote would have inst() rejoin a tree that is over.
            if k not in ['datalake', 'url', 'anchor', 'hash', 'spec', 'tree',
                         '__redirected_paths__']
            # None means "ask the class", which is what every block that never
            # mentioned the feature says -- and saying it out loud in every
            # quote() would move the recorded text of blocks that have nothing
            # to do with specializations.
            and not (k in ('use_specializations', 'SPECIALIZATIONS', 'TAB_SPECIALIZATIONS', 'BLOCK_SPECIALIZATIONS') and v is None)
        }
        self.log.detailed(f"{self.anchor}: _tailkwargs_: {tailkwargs=}")
        return tailkwargs

    @staticmethod
    def _split_top_level_(text, seps=(', ',)):
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

    def _cite_chunks_(self, specline, indent):
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
        for kw in self._split_top_level_(rest):
            if kw.startswith('spec={'):
                inner = kw[len('spec={'):]
                closing = inner.rfind('}')
                entries = inner[:closing] if closing != -1 else inner
                tail = inner[closing:] if closing != -1 else ''
                pieces.append('spec={')
                pieces.extend(self._split_top_level_(entries))
                pieces.append(tail)
            else:
                pieces.append(kw)
        pieces = [p for p in pieces if p]
        body = f"\n{indent}".join(repr(p) for p in pieces)
        return f"(\n{indent}{body}\n{indent})"

    def _render_call_(self, kwargs, *, pretty: bool, deslash: int, dollar: bool) -> str:
        """``fqcn(k=v, ...)`` for *kwargs*: the rendering :meth:`quote` and :meth:`repr` share."""
        def quotestr(x):
            return repr(x) if isinstance(x, str) else x
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
        if dollar:
            quote = f"${quote}"
        return quote


    @staticmethod
    def _parse_signature_(signature: str) -> dict:
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
    def _split_top_level_items_(inner: str, sep: str = ','):
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
    def _parse_dictstr_(cls, text: str) -> dict:
        text = text.strip()
        if not (text.startswith('{') and text.endswith('}')):
            return {}
        out = {}
        for item in cls._split_top_level_items_(text[1:-1]):
            if not item.strip():
                continue
            parts = cls._split_top_level_items_(item, sep=':')
            if len(parts) < 2:
                return {}
            key, value = parts[0].strip(), ':'.join(parts[1:]).strip()
            unquoted = cls._unquote_str_(key)
            out[unquoted if unquoted is not None else key] = value
        return out

    @staticmethod
    def _unquote_str_(text: str):
        text = text.strip()
        if len(text) < 2 or text[0] != text[-1] or text[0] not in ('"', "'"):
            return None
        try:
            value = ast.literal_eval(text)
        except Exception:
            return None
        return value if isinstance(value, str) else None

    @staticmethod
    def _literal_(text):
        if not isinstance(text, str):
            return text
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    @staticmethod
    def _is_signaturestr_(text: str) -> bool:
        text = text.strip()
        if not text.endswith(')'):
            return False
        head, _, _ = text.partition('(')
        if head is text:
            return False
        return head == '' or all(p.isidentifier() for p in head.split('.'))


    @classmethod
    def _structure_from_signature_text_(cls, value):
        """Structure a signature value, recovering typed leaves where it can.

        A non-legacy signature is a faithful ``repr`` of a typed dict, so
        ``literal_eval`` reconstructs it exactly -- an ``int`` comes back an
        ``int``, not the substring ``'256'``. This is the read-side
        counterpart for anything holding a rendered signature rather than the
        block, such as a journal row.

        Falls back to `_structure_signatureval_` when the text does not parse,
        which is the legacy rendering and anything carrying a specline.
        """
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value.strip())
        except Exception:
            return cls._structure_signatureval_(value)

    @classmethod
    def _structure_signatureval_(cls, value):
        if not isinstance(value, str):
            return value
        text = value.strip()
        inner = cls._unquote_str_(text)
        if inner is not None:
            structured = cls._structure_signatureval_(inner)
            if isinstance(structured, dict):
                return structured
            return value
        if text.startswith('{') and text.endswith('}'):
            parsed = cls._parse_dictstr_(text)
            if parsed:
                return {k: cls._structure_signatureval_(v) for k, v in parsed.items()}
            return value
        if cls._is_signaturestr_(text):
            parsed = Datablock._parse_signature_(text)
            if parsed:
                return {k: cls._structure_signatureval_(v) for k, v in parsed.items()}
        return value

    @classmethod
    def _structure_subsignatureval_(cls, value):
        return cls._structure_signatureval_(value)


    def _journal_entry_(self, journal: dict) -> 'DatajournalEntry':
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
        return self.datajournal(**{key: value}, **filters)

    def _topic_map_(self, topics):
        """A TOPICS declaration as an ordered ``{path: value}`` map, or None.

        The structured counterpart of :meth:`_topics_signature_`: one entry per
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
            modern = self._modern_topics_(topics)

            def walk(node, prefix):
                if not isinstance(node, dict):
                    out['/'.join(prefix)] = self._topictext_(node, modern)
                    return
                for name, child in node.items():
                    walk(child, prefix + (str(name),))

            for name, child in topics.items():
                walk(child, (str(name),))
            return out
        if isinstance(topics, list):
            return {str(name): ABSENT for name in topics}
        return None

    def _other_topics_(self, other_topics, journal):
        """The other side of a :meth:`difftopics`, as ``(segments, map)``.

        Accepts a live block, a journal entry, a ``TOPICS`` declaration, or the
        ``str(dict)`` a journal records one as.
        """
        if other_topics is ABSENT:
            if journal is None:
                raise ValueError("difftopics needs other_topics= or journal=")
            other_topics = self._journal_entry_(journal)
        if isinstance(other_topics, Datablock):
            return other_topics._topics_signature_(), other_topics._topic_map_(getattr(other_topics, 'TOPICS', None))
        if isinstance(other_topics, DatajournalEntry):
            # A journal records a list-TOPICS block as a mapping of DIRTOPIC,
            # so the two render alike from an entry even though they do not from
            # the blocks themselves. Compare two LIVE blocks to see that one.
            other_topics = other_topics.block.TOPICS
        elif isinstance(other_topics, str):
            other_topics = literal_topics(other_topics)
        topicmap = self._topic_map_(other_topics)
        return self._render_topic_map_(topicmap), topicmap

    @staticmethod
    def _render_topic_map_(topicmap):
        if topicmap is None:
            return ("topics:None",)
        return tuple(f"topic:{path}" if value is ABSENT else f"topic:{path}={value}"
                     for path, value in topicmap.items())

    def _topics_signature_(self, topics=None, *, declared=None):
        """The topic segments of :attr:`signature`, in the order it joins them.

        The one rendering of a block's topics into its identity: :attr:`signature`
        and :attr:`supersignature` join what this returns, and :meth:`difftopics`
        compares it. Two blocks whose signatures differ only in their topics are
        exactly the two whose ``_topics_signature_()`` differ -- which is what makes
        the diff answer the question the hash asks.
        """
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        if declared is not None:
            if isinstance(declared, (list, tuple)):
                return tuple(f"topic:{topic}" for topic in declared)
            # A narrower block's own declaration: ITS nodes, in ITS order, in ITS
            # era -- a sentinel renders bare, as it did when that block was built.
            modern = self._modern_topics_(declared)
            return tuple(f"topic:{'/'.join(tp)}={self._topictext_(node, modern)}"
                         for tp, node in self._declared_leaves_(declared))
        if self._topicfiles_ is not None:
            # A leaf is named by its full path, so a nested topic reads
            # "topic:data/frames=None". A flat TOPICS has one-segment paths and
            # renders byte-identically to before -- the hash does not move.
            modern = self._modern_topics_()
            leaves = self.leaftopics() if topics is None else self._topic_leaves_(topics)
            return tuple(f"topic:{'/'.join(tp)}={self._topictext_(self._topicnode_(*tp), modern)}"
                         for tp in leaves)
        if hasattr(self, "TOPICS") and isinstance(self.TOPICS, list):
            names = self.TOPICS if topics is None else list(topics)
            return tuple(f"topic:{topic}" for topic in names)
        return ("topics:None",)

    @staticmethod
    def _declared_leaves_(declared, prefix=()):
        """``[(path, node)]`` for every leaf of a topic declaration, in its order."""
        out = []
        for name, node in declared.items():
            path = prefix + (str(name),)
            if isinstance(node, dict):
                out.extend(Datablock._declared_leaves_(node, path))
            else:
                out.append((path, node))
        return out

    def _topic_leaves_(self, topics):
        """The leaves of *topics*, in the order *topics* gives them.

        The order is the caller's, not this class's: a specialization names the
        topics in the order the narrower block declared them, and that order is
        in the identity being reconstructed. A group expands to its own leaves,
        in ITS declaration order, which is the only order it has.

        A name that is not a topic raises out of :meth:`_topicnode_`, naming it.
        """
        leaves = []
        for topic in topics:
            tp = self._normtopic_(topic if isinstance(topic, (tuple, list)) else (topic,))
            node = self._topicnode_(*tp)
            if isinstance(node, dict):
                leaves.extend(self._leaves_under_(*tp))
            else:
                leaves.append(tp)
        return leaves

    def _specialization_pin_(self, key, value):
        """A pinned value, rendered as `_typed_specdict_` renders that field.

        Compared as rendered rather than as given, because the rendering is
        what the hash is made of: ``1``, ``'1'`` and ``True`` are three
        different pins only if they render differently.
        """
        if isinstance(value, Datablock):
            return value._typed_specdict_()
        if self.is_specline(value):
            try:
                evaluated = dataparts.eval(value)
            except Exception:
                evaluated = None
            return (evaluated._typed_specdict_()
                    if isinstance(evaluated, Datablock) else value)
        return self._coerce_to_annotation_(value, self.VAR.__dataclass_fields__[key].type)

    def _specialization_topics_(self, specialization) -> 'Topics | tuple':
        """The narrower block's TOPICS declaration: its own, or -- for ``topics=SAME`` -- this block's."""
        return self.TOPICS if specialization.topics is SAME else specialization.topics

    def _specialization_mismatch_(self, specialization):
        """Why *specialization* does not describe this block, or None if it does.

        A pin naming a field this class does not have, or a topic it does not
        declare, RAISES rather than reporting a mismatch: that is a mistake in
        the declaration, and a declaration that quietly never matches is the
        one failure this feature cannot afford.
        """
        fields = self.VAR.__dataclass_fields__
        unknown = [k for k in specialization.spec if k not in fields]
        if unknown:
            raise ValueError(
                f"{self.__class__.__name__}.SPECIALIZATIONS: {specialization!r} pins "
                f"{unknown}, which {self.VAR.__name__} does not declare. A pin names a "
                f"VAR field this class has and the narrower block did not."
            )
        self._topic_leaves_(self._specialization_topics_(specialization))   # raises on a topic we do not declare
        typed = self._typed_specdict_()
        for k, v in specialization.spec.items():
            mine, pinned = typed[k], self._specialization_pin_(k, v)
            if repr(mine) != repr(pinned):
                return (f"it is for {k}={pinned!r} and this block has {k}={mine!r}, "
                        f"so it describes a different block -- which is what a pin is "
                        f"for, and not something to fix here")
        if (self.get_hash(specialization) == self.hash
                and self._specialization_anchor_(specialization) == self.anchor):
            # Under another anchor the same identity is the point: the block as
            # it was built before its class was renamed.
            return ("it reconstructs this block's own identity -- it drops no field, "
                    "no topic and no version, so there is no narrower block to read")
        return None

    def _specialization_journal_(self, journal, specialization):
        """Extract the journal specific to specialization if journal is a dict or list/tuple."""
        if isinstance(journal, dict):
            anchor = self._specialization_anchor_(specialization)
            return journal.get(anchor, journal.get(None))
        if isinstance(journal, (list, tuple)):
            try:
                idx = (self.SPECIALIZATIONS or []).index(specialization)
                return journal[idx] if idx < len(journal) else None
            except (ValueError, IndexError):
                return None
        return journal

    def _specialization_paths_(self, specialization, journal=None, why=None):
        """``{topic: path}`` for *specialization*, from the journal, or None.

        The entry is looked up by the reconstructed hash -- there is no other
        handle on the narrower block -- and its recorded paths are taken, cut
        down to the topics the specialization names. None when no entry has
        that hash, or when the paths it records are not there any more: a
        specialization that resolves to missing data has not resolved, and the
        next one should get its turn.

        *why* is a list to append the reason to, when there is no result. There
        are four different ways to come back with nothing and they call for four
        different things to be done about it, so "it did not resolve" is not an
        answer anyone can act on. :meth:`specializations` passes one.
        """
        def note(reason):
            # One line per reason: these are joined into a single `why`, and an
            # exception's own message may carry newlines of its own.
            if why is not None:
                why.append(' '.join(str(reason).split()))

        h = self.get_hash(specialization)
        anchor = self._specialization_anchor_(specialization)
        if isinstance(journal, BlocksJournal):
            journal = _shared_journal_(journal, self)
        journal = self._specialization_journal_(journal, specialization)
        try:
            if specialization.anchor is not SAME:
                j = self._journal_under_(anchor, journal, hash=h)
            else:
                j = self.datajournal(hash=h) if journal is None else DatajournalFrame(
                    journal, storage_options=self.storage_options, hash=h)
        except (FileNotFoundError, KeyError, TypeError) as e:
            self.log.detailed(f"specialization: no journal to resolve {h} in: {e}")
            note(f"there is no journal under {anchor!r} to look for hash {h} "
                 f"in ({e})")
            return None
        if len(j) == 0:
            note(f"the journal under {anchor!r} holds no entry at all for hash {h}, "
                 f"so the narrower block was never built "
                 f"{'here' if anchor == self.anchor else 'under that anchor'}")
            return None
        candidates = 0
        for i in range(len(j)):
            entry = DatajournalEntry(j.iloc[i].dropna(), storage_options=self.storage_options)
            if entry.get('event') not in self.SPECIALIZATION_EVENTS:
                continue
            candidates += 1
            recorded = entry.block.paths()
            if not isinstance(recorded, dict) or not recorded:
                note(f"entry {entry.block.id} records no paths at all")
                continue
            if getattr(specialization, 'UNSAFE_redirect_all_topics', False):
                wanted = list(self.topics())
            elif getattr(specialization, 'redirect_topics', None) is not None:
                wanted = list(self._toplevel_topics_(specialization.redirect_topics))
            else:
                wanted = self._toplevel_topics_(self._specialization_topics_(specialization))
            paths = {t: recorded[t] for t in wanted if t in recorded}
            if len(paths) < len(wanted):
                missing = sorted(set(wanted) - set(paths))
                self.log.verbose(
                    f"specialization: entry {entry.block.id} for hash {h} records no path "
                    f"for {missing}; skipping it"
                )
                note(f"entry {entry.block.id} records no path for {missing} -- a "
                 f"specialization is all of its topics or none of them")
                continue
            paths = self._chase_redirected_paths_(paths)
            gone = sorted(t for t, p in paths.items() if not self.valid_path(p))
            if gone:
                self.log.verbose(
                    f"specialization: entry {entry.block.id} for hash {h} records paths that "
                    f"are not there any more; skipping it"
                )
                note(f"entry {entry.block.id} records {gone} at paths that are not there "
                     f"any more -- that build has been cleared")
                continue
            return paths, entry
        if not candidates:
            note(f"the {len(j)} journal entries for hash {h} are all of other events; "
                 f"none is a {' or '.join(self.SPECIALIZATION_EVENTS)}")
        return None

    def _type_entries_(self, specialization=None, *, with_block: bool = True) -> dict:
        """Entries of this block's type beyond its signature, version and topics: ``{name: text}``.

        Rendered ``name=text``, after the signature and before the version --
        so, like everything in the type, they are identity. None here; a
        Datastack names its BLOCK, a Datatable its TAB.
        """
        return {}

    def _specialization_anchor_(self, specialization):
        """The anchor whose journal *specialization*'s narrower block is looked for in."""
        return self.anchor if specialization.anchor is SAME else specialization.anchor

    def _hash_journal_(self, specialization):
        """*specialization*'s narrower block's journal entries, for a block with no journal handed down.

        Read from that block's own directory -- `Datajournal.read_hash` --
        rather than the whole of its anchor's journal, every block's entries,
        which is what one block resolving its specialization used to read. The
        whole journal, read once per instance, when that block is not filed
        where its tag and version say.
        """
        anchor = self._specialization_anchor_(specialization)
        version = getattr(self, 'VERSION', None) if specialization.version is ABSENT else specialization.version
        j = Datajournal.read_hash(anchor, self.get_hash(specialization), tag=self.tag, version=version,
                                  datalake=self.datalake, storage_options=self.storage_options, log=self.log)
        if j is not None:
            return j
        return self._journal_under_(anchor)

    def _journal_under_(self, anchor, journal=None, **filter_kwargs):
        """The entries journalled under *anchor*, from *journal* when it holds any.

        A journal handed down is this block's anchor's -- unless a stack merged
        the other anchors its blocks' specializations name into it, which is
        what spares each block a read of its own. Otherwise *anchor*'s journal
        is read here, once per instance: kept on it, never in its state.
        """
        if isinstance(journal, dict):
            journal = journal.get(anchor, journal.get(None))
        if journal is not None and not isinstance(journal, (list, tuple)):
            if 'anchor' in getattr(journal, 'columns', ()):
                rows = journal[journal['anchor'] == anchor]
                if len(rows):
                    return DatajournalFrame(rows, storage_options=self.storage_options, **filter_kwargs)
            else:
                return DatajournalFrame(journal, storage_options=self.storage_options, **filter_kwargs)
        cache = self.__dict__.setdefault('__anchor_journals__', {})
        if anchor not in cache:
            try:
                cache[anchor] = Datajournal.read(
                    anchor, datalake=self.datalake, storage_options=self.storage_options, log=self.log)
            except FileNotFoundError:
                cache[anchor] = None
        if cache[anchor] is None:
            raise FileNotFoundError(f"no journal under {anchor!r} in {self.datalake!r}")
        return DatajournalFrame(cache[anchor], storage_options=self.storage_options, **filter_kwargs)

    def _toplevel_topics_(self, topics):
        """*topics* as TOP-LEVEL names, which is how a `paths` mapping is keyed."""
        return [tp[0] if isinstance(tp, (tuple, list)) else tp for tp in topics]

    def _specializing_(self):
        """Whether this block may consult :attr:`SPECIALIZATIONS` at all."""
        return bool(getattr(self, 'redirect', False)
                    and getattr(self, 'use_specializations', False)
                    and self.SPECIALIZATIONS)

    def _unbuilt_(self, topics=None):
        """True when NONE of the named topics are there, default all of them.

        Its own: resolved through ``__path__``, never ``path()``, which consults
        the redirection this is deciding whether to install.

        None rather than "not all", because a block with SOME of the named
        topics built is a block mid-build or half-cleared, and taking the rest
        from a narrower block would mix two computations' outputs under one
        hash.

        *topics* is what makes that question answerable for a PARTIAL
        specialization -- one that covers some topics and leaves the rest to
        this block. Asked over all of them, such a block stops being "unbuilt"
        the moment it builds the ones it owns, which is the first thing it does
        after installing the specialization: the topics the specialization
        covers are still absent from here, exactly as intended, and the answer
        flips anyway. Asked over the topics ONE specialization covers, it is
        the question that was always meant -- is there anything of MINE where
        this would have me read someone else's -- and it keeps the same answer
        for the life of the block.

        In the usual case the difference is invisible, because a resolved
        specialization writes ``.redirection`` and later constructions read
        that, and later builds do not ask again. It shows up when the memo is not there:
        cleared, never written because the resolving instance was configured
        ``use_specializations='memory'``, or written at another path because
        the instance that resolved was retagged afterwards. A cache that is
        load-bearing is not a cache.
        """
        topics = self.topics() if topics is None else self._toplevel_topics_(topics)
        if not topics:
            return False
        return not any(self.valid_path(self.__path__(t)) for t in topics)

    def _specialization_candidates_(self, journal=None, *, _memo=None):
        """Each specialization that could be installed, with its resolution, in declaration order.

        *_memo*, given, holds the journal: read the first time a candidate needs
        it, and kept there for the caller.
        """
        if isinstance(journal, BlocksJournal):
            journal = _shared_journal_(journal, self)
        memo = _memo if _memo is not None else {'journal': journal}
        for sp in (self.SPECIALIZATIONS or []):
            why = self._specialization_mismatch_(sp)
            if why is not None:
                self.log.detailed(f"specialization: {sp!r} does not apply: {why}")
                continue
            # Per specialization, over the topics IT covers -- see `_unbuilt_`.
            cov_topics = sp.redirect_topics if getattr(sp, 'redirect_topics', None) is not None else self._specialization_topics_(sp)
            if not self._unbuilt_(cov_topics):
                self.log.detailed(
                    f"specialization: {sp!r} does not apply: this block has data of "
                    f"its own under {self._toplevel_topics_(cov_topics)!r}, which a "
                    f"build wrote here; reading the rest from elsewhere would mix two "
                    f"computations under one hash"
                )
                continue
            sp_j = self._specialization_journal_(memo.get('journal'), sp)
            if sp_j is None and memo.get('journal') is None:
                # No journal handed down: this one specialization's entries,
                # from the narrower block's own directory -- kept for the redirect.
                key = ('hash_journal', sp.key)
                if key not in memo:
                    try:
                        memo[key] = self._hash_journal_(sp)
                    except FileNotFoundError:
                        memo[key] = None
                sp_j = memo[key]
            resolved = self._specialization_paths_(sp, journal=sp_j)
            if resolved is None:
                self.log.verbose(
                    f"SPECIALIZATION: {sp!r} applies to {self.anchorkeypath}, but no "
                    f"{'/'.join(self.SPECIALIZATION_EVENTS)} entry with hash "
                    f"{self.get_hash(sp)} records data that is still there"
                )
                continue
            yield sp, resolved

    def _install_specialization_(self, journal=None):
        """Read a narrower block's data instead of having none, or None.

        What :meth:`find_specialization` finds, installed: candidates
        that resolve are installed to satisfy missing topics.

        Installing it is recorded, through :meth:`UNSAFE_redirect`: the journal
        says this block read another's build and which specialization said it
        could, and the hidden ``.redirection`` topic it writes is what every
        later construction reads INSTEAD of scanning the journal again -- so the
        write happens once per block, not once per construction.
        ``use_specializations='memory'`` installs the paths on this instance and
        writes nothing, at the cost of resolving again next time.
        """
        # If the block is not specializing or already valid, do not install a redirection.
        if not self._specializing_() or self.valid():
            return None
        if isinstance(journal, BlocksJournal):
            journal = _shared_journal_(journal, self)
        # Read at most once -- and only if a candidate gets as far as resolving --
        # for finding one and for redirecting to it alike.
        memo = {'journal': journal}
        installed = []
        needed = {t for t in self.topics() if not self.valid_topic(t)}
        for sp, (paths, entry) in self._specialization_candidates_(journal, _memo=memo):
            offered = {t: p for t, p in paths.items() if t in needed and self.valid_path(p)}
            if not offered:
                continue
            if self.use_specializations != 'memory':
                sp_j = memo.get(('hash_journal', sp.key))
                if sp_j is None:
                    sp_j = self._specialization_journal_(memo.get('journal'), sp)
                if self.UNSAFE_redirect(specialization=sp, topics=list(offered.keys()), journal=sp_j, OVERRIDE=True, merge=True):
                    installed.append(sp)
                    needed -= set(offered.keys())
                    if not needed:
                        break
                continue
            if self._redirected_paths_ is None:
                self._redirected_paths_ = {}
            self._redirected_paths_.update(offered)
            self.__dict__.pop('redirection', None)
            self.__dict__['__specialization__'] = sp
            installed.append(sp)
            needed -= set(offered.keys())
            self.log.info(
                f"SPECIALIZATION (memory only, nothing recorded): {self.anchorkeypath} "
                f"reads {list(offered.keys())} through {sp!r}"
            )
            self.log.verbose(
                f"SPECIALIZATION: {self.anchorkeypath} reads {list(offered.keys())} "
                f"from journal entry {entry.block.id} (hash {self.get_hash(sp)}) instead, "
                f"as {sp!r}"
            )
            if not needed:
                break
        return installed[0] if len(installed) == 1 else (installed or None)

    def _anchorpath_(self, anchor=None, *, local: bool = False):
        anchor = anchor or self.anchor
        fs, root = (self.localfs, self.localroot) if local else (self.fs, self.root)
        return fs_full_path(fs, os.path.join(root, anchor))

    def _anchorkey_(self, anchor=None):
        anchor = anchor or self.anchor
        return os.path.join(anchor, self.key) if self.key else anchor

    def _anchorkeypath_(self, anchor=None, *, local: bool = False):
        anchorkey = self._anchorkey_(anchor)
        fs, root = (self.localfs, self.localroot) if local else (self.fs, self.root)
        bare = os.path.join(root, anchorkey) if anchorkey else root
        return fs_full_path(fs, bare)

    @staticmethod
    def _dbxanchorpathx_(url, anchor, x, *, fqcn, ensure: bool = False, storage_options=None):
        """Return {url}/anchor/.dbx/fqcn/x — the anchor-level directory for artefact *x*."""
        fs, root = fsspec.url_to_fs(url, **(storage_options or {}))
        _dbxanchorpathx_ = fs_full_path(fs, os.path.join(root, anchor, ".dbx", fqcn, x))
        if ensure:
            fs.makedirs(_dbxanchorpathx_, exist_ok=True)
        return _dbxanchorpathx_

    def _dbxanchorhashpathx_(self, x, ext=None, *, ensure_dirpath: bool = True, filename_prefix: str = ''):
        return Datajournal.path(self, x, ext, ensure_dirpath=ensure_dirpath, filename_prefix=filename_prefix)


    #LOG LEVEL: END

def UNSAFE_clear_block_callable(block, topics=(), clear_dirpath=False, *, stack=None, idx=None, **kwargs):
    """Module-level callable for UNSAFE_clear_blocks (must be picklable)."""
    block.UNSAFE_clear(*topics, OVERRIDE=True, clear_dirpath=clear_dirpath)
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
    """Lightweight callable that checks if a block at index `idx` is valid.

    *validation* 'valid' asks the block's valid(), 'validate' its validate(),
    and anything else is the stack's `valid_block`. With *with_path*, the
    answer comes as ``(ok, anchorkeypath)``: what a full pass records in the
    stack's manifest without forming every block a second time.
    """

    def __init__(self, idx: int, validation: str | None = None, with_path: bool = False):
        self.idx = idx
        self.validation = validation
        self.with_path = with_path

    def __call__(self, stack):
        if self.validation not in ('valid', 'validate'):
            return stack.valid_block(self.idx, validation=self.validation)
        block = stack.block(self.idx)
        ok = bool(block.validate() if self.validation == 'validate' else block.valid())
        return (ok, block.anchorkeypath) if self.with_path else ok


def _shared_journal_(caller, block):
    """The journal a stack read for its blocks, if *block* is one it was read for -- else None.

    Read under ONE anchor and ONE url, it holds the entries of the blocks
    stored there and no others: a block of another anchor, or rooted
    elsewhere, reads its own.
    """
    if caller is None:
        return None
    if isinstance(caller, (dict, list, tuple)):
        return caller
    if getattr(caller, 'journal', None) is None or getattr(caller, 'anchor', None) is None:
        return None
    if block.anchor != caller.anchor or (caller.datalake is not None and block.datalake != caller.datalake):
        return None
    return caller.journal


class DatablockRedirectionGetter:
    """Lightweight callable resolving the redirection of the block at index `idx`.

    *journal* is the one the stack read for all its blocks, under *anchor*. A
    block of another anchor -- a stack whose blocks are not all one kind --
    would find none of its entries there, so it reads its own instead.
    """

    def __init__(self, idx: int):
        self.idx = idx

    def __call__(self, stack, *, journal=None):
        with forming_with_journal(journal):
            block = stack.block(self.idx)
        return block.get_redirection(journal=_shared_journal_(journal, block))


class DatablockSpecializationFinder:
    """Lightweight callable finding the specialization the block at index `idx` would install.

    The block is formed with specializations OFF and not cached: finding must
    not install anything, and forming a block normally can.
    """

    def __init__(self, idx: int):
        self.idx = idx

    def __call__(self, stack, *, journal=None):
        with forming_with_journal(journal):
            block = stack._form_block_(self.idx, use_specializations=False)
        return block.find_specialization(journal=_shared_journal_(journal, block))


class DatablockSpecializationInstaller:
    """Lightweight callable installing the specialization the block at `idx` resolves -- building nothing.

    What the block's own build() would do first. Run over a stack's blocks by
    the stack's build(), which is the call that sanctions writing it.
    Returns ``(specialization | None, status)``, *status* one of:

    - ``'valid'``: its data is there, and passes *validation*;
    - ``'fails_validation'``: its data is there but fails validate(). Nothing
      but clearing and building it again makes it valid;
    - ``'adoption_fails_validation'``: what its specialization resolves to
      fails validate(), and was not adopted. Only building it without its
      specializations makes it valid;
    - ``'adopted'``: a specialization was installed, and the block is valid;
    - ``'owes'``: a specialization was installed, and topics are left owed;
    - ``'unresolved'``: no specialization resolves -- or *install* is False.
    """

    def __init__(self, idx: int, validation: str = 'valid', install: bool = True):
        self.idx = idx
        self.validation = validation
        self.install = install

    def _passes_(self, block) -> bool:
        return bool(block.validate()) if self.validation == 'validate' else True

    def __call__(self, stack, *, journal=None):
        with forming_with_journal(journal):
            block = stack._form_block_(self.idx)
        red_yaml = os.path.join(block.anchorkeypath, '.redirection', 'paths.yaml')
        is_recorded = block.fs.exists(red_yaml)
        if (is_recorded and block.valid()) or block.__valid__(path=None) or (block.valid() and not getattr(block, 'SPECIALIZATIONS', None)):
            return (None, 'valid' if self._passes_(block) else 'fails_validation')
        if not self.install:
            return (None, 'unresolved')
        sp = block._install_specialization_(journal=_shared_journal_(journal, block))
        if sp is None:
            return (None, 'unresolved')
        # Adopted is not built: a specialization that leaves topics owed leaves the block invalid.
        if not block.valid():
            return (sp, 'owes')
        if not self._passes_(block):
            # What was adopted fails the checks it was adopted to pass: not this block's.
            block.UNSAFE_clear_redirection(OVERRIDE=True)
            return (None, 'adoption_fails_validation')
        return (sp, 'adopted')


class DatablockRedirectionClearer:
    """Lightweight callable clearing the redirection of the block at index `idx`."""

    def __init__(self, idx: int):
        self.idx = idx

    def __call__(self, stack):
        return stack.block(self.idx).UNSAFE_clear_redirection(OVERRIDE=True)


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
            sig = f"{getattr(blk, 'fqcn', blk.__class__.__name__)}{blk.signaturestr()}"
            if not stack._matches_sig_clauses_(sig, self.signature_clauses):
                return False
        if self.tag_clauses:
            tag = getattr(blk, 'tag', None)
            if not stack._matches_tag_clauses_(tag, self.tag_clauses):
                return False
        if self.path_clauses:
            paths = stack._get_block_paths_(blk)
            if not stack._matches_path_clauses_(paths, self.path_clauses):
                return False
        return True


class InvalidBlocksError(RuntimeError):
    """Blocks of a stack that are not valid, found before anything fails reading them.

    Raised by `Datastack.build` for a stack that is valid, or reads an older
    build, over blocks that are not -- rather than returning a stack whose
    blocks would fail later, far from here, when something reads them; by
    anything that adopts or builds blocks, for blocks whose data fails
    validate(); and by whatever is about to read invalid blocks, before it
    starts. *invalid* is the indices of the blocks that are not valid;
    *state* says what the stack claims, *reader* who was about to read them,
    *validation* how they were found invalid.
    """

    #: How many of the invalid blocks the message names.
    SHOWN = 5

    #: Why a block fails validation, when whoever raises knows -- see `DatablockSpecializationInstaller`.
    REASONS = {
        'fails_validation': ("its data is there but fails validate(), which no build repairs: "
                             "clear it -- UNSAFE_clear() -- and build it again"),
        'adoption_fails_validation': ("what its specialization resolves to fails validate(), and was not "
                                      "adopted: build it without its specializations"),
    }

    def __init__(self, stack, invalid, *, state: str | None = None, reader: str | None = None,
                 validation: str = 'valid', reasons: dict | None = None):
        self.stack = stack
        self.invalid = list(invalid)
        reasons = reasons or {}
        n, k = stack.n_blocks, len(self.invalid)
        lines, kinds = [], set()
        for i in self.invalid[:self.SHOWN]:
            try:
                block = stack.block(i)
                if i in reasons:
                    kind, why = reasons[i], self.REASONS[reasons[i]]
                else:
                    kind, why = block._invalidity_(validation)
                kinds.add(kind)
                lines.append(f"  block {i}: {block.anchorkeypath}: {why}")
            except Exception as e:
                lines.append(f"  block {i}: cannot say why: {type(e).__name__}: {e}")
        if k > self.SHOWN:
            lines.append(f"  ... and {k - self.SHOWN} more")
        if kinds & {'fails_validation', 'adoption_fails_validation'}:
            own = [i for i in self.invalid if reasons.get(i, 'fails_validation') == 'fails_validation']
            adopted = [i for i in self.invalid if reasons.get(i) == 'adoption_fails_validation']
            remedy = " ".join(
                ([f"Clear what fails validate() and build it again: stack.UNSAFE_clear_blocks("
                  f"indices={own!r}, clear_done=False), then stack.build_blocks()."] if own else [])
                + ([f"Build {adopted!r} without their specializations: a stack constructed with "
                    f"use_block_specializations=False, then its build_blocks({adopted!r})."] if adopted else []))
        else:
            remedy = (("stack.specialize_blocks() adopts what resolves; " if 'unadopted' in kinds else "")
                      + "stack.build() adopts what resolves and builds the rest -- "
                        "stack.build_blocks() the same, without the stack's own topics.")
        super().__init__(
            (f"{reader}: " if reader else "")
            + f"{stack.anchorkeypath}: {k} of {n} blocks are not valid"
            + (f", but the stack {state}" if state else "")
            + (f" (validation={validation!r})" if validation != 'valid' else "") + ":\n"
            + "\n".join(lines) + f"\n{remedy}"
        )


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
            BLOCK = MyBlock
            @dataclass
            class VAR(Datablock.VAR):
                path: str = None
                block_size: int = 100

            def blocks(self):
                n = self._total_items()
                return [
                    MyBlock(datalake=self._datalake_, spec=dict(path=self.var.path, idx=i))
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

        def __call__(self, stack, *, build=True, journal=None):
            # *journal*, a BlocksJournal the stack read once and the executor
            # handed each worker, is what the block resolves its
            # specializations against, rather than reading the journal itself.
            with forming_with_journal(journal), stack._block_specializations_in_force_():
                block = stack.__block__(self.idx)
                block.keyby = stack.keyby
                if build:
                    block.build()
            del block
            gc.collect()

    #: The Datablock class this stack's blocks are -- every ``block(i)`` is one.
    #: A declaration: it says what the stack holds and is checked as each block
    #: is formed, but it does not locate anything -- a block's journal is under
    #: that block's url and anchor, which ``block(0)`` knows. A `Datatable`'s
    #: ``TAB`` is its BLOCK. Not declaring one is deprecated.
    BLOCK = None

    #: Classes already warned about a missing BLOCK, so a stack built in a loop warns once.
    _NO_BLOCK_WARNED = set()

    DatablockValidityChecker = DatablockValidityChecker
    DatablockRedirectionChecker = DatablockRedirectionChecker
    DatablockRedirectionGetter = DatablockRedirectionGetter
    DatablockRedirectionClearer = DatablockRedirectionClearer
    DatablockSpecializationFinder = DatablockSpecializationFinder
    DatablockSpecializationInstaller = DatablockSpecializationInstaller
    DatablockValidationChecker = DatablockValidationChecker
    DatablockSignatureMatcher = DatablockSignatureMatcher
    BlockValidChecker = DatablockValidityChecker
    BlockRedirectedChecker = DatablockRedirectionChecker
    BlockValidationChecker = DatablockValidationChecker
    BlockSignatureMatcher = DatablockSignatureMatcher

    # 1. Protocol and hooks ------------------------------------------------

    #: How a stack establishes that its blocks are valid -- see `valid_blocks`.
    VALIDATIONS = ('cross_check', 'valid', 'validate')
    VALIDATION = 'cross_check'
    #: How many blocks 'cross_check' compares with the manifest: block 0 and the rest at random.
    CROSS_CHECK_BLOCKS = 3

    def __init__(self, *args, parallelization: str | None = None, n_workers: int = 1, devices: list | str | None = None, multiprocessing_start_method: str = 'spawn', worker_done_timeout_sec: int = 1000, result_idle_timeout_sec: float | None = None, shuffle_callables: bool = False, work_stealing: bool = False, use_block_specializations: 'bool | str | None' = None, validation: str | None = None, cross_check_blocks: int | None = None, cross_check_seed: int | None = None, **kwargs):
        # The use_specializations the blocks are formed with -- see _form_block_.
        # Passed on only when given: every keyword of a stack reaches its
        # quote(), and a default spelled out would change that of every stack.
        # The same goes for how its blocks are validated -- see valid_blocks.
        if use_block_specializations is not None:
            kwargs['use_block_specializations'] = use_block_specializations
        if validation is not None:
            kwargs['validation'] = self._check_validation_(validation)
        if cross_check_blocks is not None:
            if not (isinstance(cross_check_blocks, int) and cross_check_blocks >= 1):
                raise ValueError(f"cross_check_blocks must be an int >= 1, got {cross_check_blocks!r}")
            kwargs['cross_check_blocks'] = cross_check_blocks
        if cross_check_seed is not None:
            kwargs['cross_check_seed'] = cross_check_seed
        super().__init__(*args, parallelization=parallelization, n_workers=n_workers, devices=devices, multiprocessing_start_method=multiprocessing_start_method, worker_done_timeout_sec=worker_done_timeout_sec, result_idle_timeout_sec=result_idle_timeout_sec, shuffle_callables=shuffle_callables, work_stealing=work_stealing, **kwargs)
        if self._block_class_() is None and type(self) not in Datastack._NO_BLOCK_WARNED:
            Datastack._NO_BLOCK_WARNED.add(type(self))
            warnings.warn(
                f"{type(self).__qualname__} declares no BLOCK: set BLOCK = <the Datablock class "
                f"its blocks are>. It will be required.", FutureWarning, stacklevel=2)
        # Early validation only — executor_cls is a property so deepcopy/setstate paths work.
        executors = self._get_executors_()
        key = (self.parallelization or "inline").lower()
        if key not in executors:
            raise ValueError(
                f"Unknown parallelization {self.parallelization!r}. "
                f"Choose from {list(executors)}"
            )

    def __block__(self, idx: int):
        """Return a single child :class:`Datablock` for the given index.

        Subclasses **must** override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __block__(idx)"
        )

    @dataclass(frozen=True, repr=False, eq=False)   # the base's repr (note last) and equality
    class Specialization(Datablock.Specialization):
        """A Datablock's Specialization, and the BLOCK the narrower stack's type names.

        *BLOCK*: SAME for this stack's own; a string for another fqcn; None for
        a stack built before a stack's type named its BLOCK at all.
        """
        BLOCK: str | SAME | None = SAME

        __hash__ = Datablock.Specialization.__hash__

        def __post_init__(self):
            super().__post_init__()
            if not (self.BLOCK is SAME or self.BLOCK is None or (isinstance(self.BLOCK, str) and self.BLOCK)):
                raise TypeError(f"Specialization BLOCK= is SAME, None or a block fqcn, got {self.BLOCK!r}")

        @property
        def _block_(self):
            return self.BLOCK

    def _type_entries_(self, specialization=None, *, with_block: bool = True) -> dict:
        """``{'BLOCK': fqcn}``: the block class is part of what a stack IS -- a Datatable calls it TAB.

        Without it a stack's hash disregarded its BLOCK -- an identity names no
        class, and a marker-spelled table carries none of its TAB's topics --
        so editing the BLOCK, or pointing it at another class, moved every
        block and left the stack valid over them. *with_block* False, or a
        *specialization* whose BLOCK is None, computes the type under the rules
        from before: no entry. A specialization's BLOCK string names another.
        """
        entries = super()._type_entries_(specialization, with_block=with_block)
        block = self._block_class_()
        if specialization is not None and specialization._block_ is not SAME:
            block = specialization._block_
        if not with_block or block is None:
            return entries
        return {**entries, 'BLOCK': block if isinstance(block, str) else block.fqcn}

    def specialize(self, journal=None, parallelization: str | None = None,
                   n_workers: int | None = None, **kwargs):
        """Install block and stack specializations without building any unspecialized topics."""
        # The blocks' journal is read by the adoption, and only if a block needs it.
        self.specialize_blocks(journal=journal, parallelization=parallelization,
                               n_workers=n_workers, **kwargs)
        stack_journal = None if isinstance(journal, BlocksJournal) else journal
        super().specialize(journal=stack_journal)
        return self

    def specialize_blocks(self, parallelization: str | None = None, n_workers: int | None = None,
                          journal=None, validation: str | None = None, **kwargs) -> pd.Series | None:
        """Install the specialization each block that is not valid resolves -- building nothing (parallelized).

        The block half of :meth:`specialize`, whatever the stack's own state:
        every block is asked whether it is valid, by *validation* -- see
        `valid_blocks` -- and those that are not adopt what their
        specializations resolve to. Returns the installed `Specialization`s by
        block index, or None when BLOCK declares none. Raises
        `InvalidBlocksError` when a block fails validate().
        """
        return self._install_block_specializations_(parallelization=parallelization, n_workers=n_workers,
                                                    journal=journal, validation=validation, **kwargs)

    def build(self, *args, deep: bool = False, **kwargs):
        # The blocks adopt what their specializations resolve to FIRST, before
        # the stack's own state is consulted. A stack whose build is then
        # elided -- itself adopted whole -- or skipped as valid says nothing
        # about blocks that moved to identities of their own; left unadopted,
        # they would read as unbuilt. The journal, when a block needs it, is
        # read once for the whole build -- see `_build_journal_`.
        self.__dict__['__building__'] = True
        validation = self._validation_()
        # deep=True takes no short cut: not the stack's validity, nor the manifest's -- every block is asked.
        if deep and validation == 'cross_check':
            validation = 'valid'
        try:
            _, invalid, failed, paths = self._adopt_block_specializations_(validation=validation)
            # Data that is there and fails validate() is repaired by nothing a build does.
            if failed:
                raise InvalidBlocksError(self, list(failed), validation=validation, reasons=failed)
            # A stack that is completely redirected answers from elsewhere;
            # its build is elided regardless of deep. An unredirected (or
            # partially redirected) stack that is already valid and owes no
            # topics has nothing of its OWN to build unless deep=True forces
            # it. Either way, it is done -- and so must its blocks be: those
            # that are not are built here, as build_blocks() builds them, and
            # the stack's own topics are left as they are. A block still not
            # valid after that is an error, said here rather than by whatever
            # reads it, far from here.
            # A stack with no topics of its own claims nothing by being
            # valid, which it is vacuously: its build is its blocks'.
            redirected = self._redirected_paths_ is not None and not self.ownedtopics()
            if not self.topics():
                self._build_owed_blocks_(invalid, validation, state=None)
                invalid = []
            if redirected or (not deep and self.valid() and not self.owedtopics()):
                self._build_owed_blocks_(invalid, validation, state=self._claim_(redirected))
                return super().build(*args, deep=deep, _specialized_=True, **kwargs)
            # The stack's own specialization; its blocks' are installed above.
            # Adopted whole, it elides the build that would have built them.
            Datablock.specialize(self)
            if invalid and self._redirected_paths_ is not None and not self.ownedtopics():
                self._build_owed_blocks_(invalid, validation, state=self._claim_(True))
                return super().build(*args, deep=deep, _specialized_=True, **kwargs)
            result = super().build(*args, deep=deep, _specialized_=True, **kwargs)
            # Datablock.build skips a valid stack, deep or not, and its blocks with it.
            full = 'validate' if validation == 'validate' else 'valid'
            invalid = [i for i in invalid if not self.valid_block(i, validation=full)]
            self._build_owed_blocks_(invalid, validation, state=self._claim_(False))
            if self.valid() and paths is not None:
                self._record_blocks_manifest_(paths)
            return result
        finally:
            self.__dict__.pop('__building__', None)
            self.__dict__.pop('__build_journal__', None)

    def _build_owed_blocks_(self, invalid, validation, *, state):
        """Build the blocks at *invalid* -- none of this stack's own topics -- and raise for any still not valid.

        What a stack's build does for blocks its own build will not reach: the
        stack is valid, or reads an older build, and so is done, but these are
        not. *state* is what the stack claims, for the error.
        """
        if not invalid:
            return
        self.log.info(f"{self.anchorkeypath}: the stack {state or 'has no topics of its own'}; "
                      f"building the {len(invalid)} of its {self.n_blocks} blocks that are not valid")
        self.build_blocks(invalid, validation=validation)
        full = 'validate' if validation == 'validate' else 'valid'
        still = [i for i in invalid if not self.valid_block(i, validation=full)]
        if still:
            raise InvalidBlocksError(self, still, state=f"{state}, and they were built" if state else
                                     "built them", validation=validation)

    @staticmethod
    def _claim_(redirected: bool) -> str:
        """What a stack that declines to build claims, for `InvalidBlocksError`."""
        return ("reads an older build (it is redirected), whose blocks are not these" if redirected
                else "is valid")

    def build_blocks(self, indices=None, validation: str | None = None, **kwargs):
        """Build the blocks that are not valid -- or those at *indices* -- and none of this stack's own topics.

        `build` does this too, for a stack that is valid or reads an older
        build -- done itself, but not its blocks; this is the same without the
        stack's own state consulted at all. The blocks adopt what their
        specializations resolve to first, and the rest are built as
        `__build__` builds them, by the stack's executor. Raises
        `InvalidBlocksError` for blocks that fail validate(), which a build
        does not repair.
        """
        validation = self._validation_(validation)
        # Within a build(), that build's: its journal read is shared, and it clears it.
        owns_build = '__building__' not in self.__dict__
        self.__dict__['__building__'] = True
        try:
            if indices is None:
                _, indices, failed, _ = self._adopt_block_specializations_(validation=validation)
                if failed:
                    raise InvalidBlocksError(self, list(failed), validation=validation, reasons=failed)
            indices = [int(i) for i in indices]
            if not indices:
                self.log.verbose(f"{self.anchorkeypath}: build_blocks: every block is valid")
                return self
            callables, callable_kwargs = self.__split__(**kwargs)
            callables = [callables[i] for i in indices]
            self.log.info(f"{self.anchorkeypath}: building {len(indices)} of {self.n_blocks} blocks")
            executor = self.executor_cls(**self._executor_kwargs_(
                tag=f"BUILDING {len(indices)} of {self.n_blocks} blocks [{self.__class__.__name__}]"))
            callable_kwargs = self._with_build_journal_(callables, callable_kwargs)
            executor.exec_callables(callables, self, **callable_kwargs)
            # A full pass, which records the manifest when every block is valid.
            self._check_blocks_('validate' if validation == 'validate' else 'valid')
            return self
        finally:
            if owns_build:
                self.__dict__.pop('__building__', None)
                self.__dict__.pop('__build_journal__', None)

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
        executor_kwargs = self._executor_kwargs_(
            tag=f"EXECUTING {len(callables)} callables [{self.__class__.__name__}]"
        )
        executor = self.executor_cls(**executor_kwargs)
        callable_kwargs = self._with_build_journal_(callables, callable_kwargs)
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

    # 2. Declared API ------------------------------------------------------

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
            self._blocks_[idx] = self._form_block_(idx)
        return self._blocks_[idx]

    def blocks(self) -> list:
        """Return all blocks, forming them via :meth:`block` if needed."""
        n = self.n_blocks
        indices = tqdm.tqdm(range(n), desc=f"Forming {n} blocks") if n > 100 else range(n)
        return [self.block(idx) for idx in indices]

    def block_datajournal(self, **kwargs) -> DatajournalFrame | None:
        """Return the DatajournalFrame for child blocks, or None if no blocks exist or journal fails to load."""
        if self.n_blocks == 0:
            return None
        try:
            return self._blocks_datajournal_(**kwargs)[0]
        except Exception as e:
            self.log.detailed(f"block_datajournal: could not load journal for child blocks: {e}")
            return None

    def child_specialization_datajournal(self):
        """The children's journal, read ONCE, for them to resolve against.

        A block that declares :attr:`SPECIALIZATIONS` resolves them in its
        build() -- never at construction -- and resolving means reading the
        journal. A stack's build builds every child, so a stack whose children
        are specialized would pay one journal read PER CHILD: invisible against
        a local directory, and a glob over ``**/*.parquet`` plus a parquet read
        per child against object storage, which is where these stacks live.

        A stack's build reads it once and hands it to the callables that adopt
        and build its blocks; this is the same read, for a caller that forms
        the children itself and hands it down as ``specialization_journal=``.
        A stack whose children declare no specializations should not call this
        at all -- there is nothing for them to resolve.

        Reading it means constructing child 0 -- for where its journal is, with
        specializations off and uncached -- which is itself a child
        construction, so the reentrant call is answered with None. The child 0
        the stack then caches resolves against the snapshot like the rest.

        A snapshot, deliberately. A child resolving against it cannot see an
        entry a SIBLING wrote during this same build -- which is right, since a
        specialization resolves to a build that predates this one, and a child
        reading a sibling's fresh entry would be resolving to data being
        written underneath it.
        """
        cached = self.__dict__.get('__child_journal__', ABSENT)
        if cached is not ABSENT:
            return cached
        block_cls = self._block_class_()
        if block_cls is not None:
            forming = _forming_journal_(block_cls.anchor, self._blocks_datalake_())
            if forming is not None:
                # Handed down with the callable that is forming this stack's
                # blocks -- read once, in the parent -- so not read here again.
                self.__dict__['__child_journal__'] = forming
                return forming
        if self.__dict__.get('__reading_child_journal__'):
            return None
        self.__dict__['__reading_child_journal__'] = True
        try:
            journal = self.block_datajournal()
        finally:
            self.__dict__.pop('__reading_child_journal__', None)
        n = 'no' if journal is None else len(journal)
        self.log.verbose(
            f"{self.__class__.__name__}: read the child journal once "
            f"({n} entries) for {self.n_blocks} children to resolve against"
        )
        self.__dict__['__child_journal__'] = journal
        return journal

    def valid_block(self, idx: int, validation: str | None = None) -> bool:
        """Whether the block at index *idx* is valid, by *validation* -- see `valid_blocks`."""
        validation = self._validation_(validation)
        if validation == 'cross_check' and self._blocks_cross_checked_():
            return True
        block = self.block(idx)
        return bool(block.validate() if validation == 'validate' else block.valid())

    def redirected_block(self, idx: int) -> bool:
        """Return whether the block at index *idx* is redirected."""
        return self.block(idx).redirected()

    def valid_blocks(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, validation: str | None = None, **kwargs) -> pd.Series:
        """Whether each block is valid, as a Series of booleans (parallelized).

        *validation*, this stack's own unless given:

        - ``'cross_check'``: the manifest -- recorded when a full pass last
          found every block valid -- vouches for all of them when block 0 and
          ``cross_check_blocks - 1`` others at random still have the paths it
          records; no data is looked at. Without a manifest, or when one does
          not match, every block is asked, as by 'valid'.
        - ``'valid'``: every block's valid() -- its data is there.
        - ``'validate'``: every block's validate() -- whatever its __validate__
          checks, which is valid() unless a block says otherwise.

        A full pass that finds every block valid, and this stack too, records
        the manifest.
        """
        if false_only and true_only:
            raise ValueError("false_only and true_only are mutually exclusive")
        series, _ = self._check_blocks_(validation, parallelization=parallelization,
                                        n_workers=n_workers, **kwargs)
        if false_only:
            return series[~series]
        if true_only:
            return series[series]
        return series

    def _check_blocks_(self, validation: str | None = None, parallelization: str | None = None,
                       n_workers: int | None = None, **kwargs):
        """``(valid, paths)``: whether each block is valid, and each block's path -- None when the manifest vouched."""
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=bool), []
        validation = self._validation_(validation)
        if validation == 'cross_check':
            if self._blocks_cross_checked_():
                return pd.Series(True, index=pd.RangeIndex(n), dtype=bool), None
            validation = 'valid'
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
        exec_kwargs = self._executor_kwargs_(
            tag=f"CHECKING VALIDITY ({validation}) of {n} blocks [{self.__class__.__name__}]",
            n_workers=n_workers,
            executor_cls=executor_cls,
            **kwargs,
        )
        executor = executor_cls(**exec_kwargs)
        checkers = [self.DatablockValidityChecker(i, validation=validation, with_path=True) for i in range(n)]
        results = executor.exec_callables(checkers, self)
        series = pd.Series([ok for ok, _ in results], dtype=bool)
        paths = [path for _, path in results]
        if series.all() and self.valid():
            self._record_blocks_manifest_(paths)
        return series, paths

    def blocks_redirected(self, parallelization: str | None = None, n_workers: int | None = None, false_only: bool = False, true_only: bool = False, **kwargs) -> pd.Series:
        """Whether each block is redirected, as a Series of booleans (parallelized).

        Each block's `redirected`: the cheap question, reading no journal. For
        WHERE they read from, see :meth:`get_block_redirections`.
        """
        if false_only and true_only:
            raise ValueError("false_only and true_only are mutually exclusive")
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=bool)
        results = self._exec_over_blocks_(
            [self.DatablockRedirectionChecker(i) for i in range(n)],
            tag=f"CHECKING REDIRECTION of {n} blocks [{self.__class__.__name__}]",
            parallelization=parallelization, n_workers=n_workers, **kwargs)
        series = pd.Series(results, dtype=bool)
        if false_only:
            return series[~series]
        if true_only:
            return series[series]
        return series

    #: The name `blocks_redirected` had first.
    redirected_blocks = blocks_redirected

    def get_block_redirections(self, parallelization: str | None = None, n_workers: int | None = None,
                               journal=None, redirected_only: bool = False, **kwargs) -> pd.Series:
        """Each block's `Redirection`, or None, as a Series indexed by block (parallelized).

        Resolving one reads the journal its entries are in, unlike
        :meth:`blocks_redirected`. The stack reads it ONCE -- its blocks'
        journal, through ``block(0)``, as :meth:`block_datajournal` does -- and
        hands it to every block; a block of another anchor reads its own. A
        *journal* passed in is used for every block as it is.
        *redirected_only* drops the Nones.
        """
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=object)
        shared = self._shared_blocks_journal_(journal)
        results = self._exec_over_blocks_(
            [self.DatablockRedirectionGetter(i) for i in range(n)], journal=shared,
            tag=f"RESOLVING REDIRECTION of {n} blocks [{self.__class__.__name__}]",
            parallelization=parallelization, n_workers=n_workers, **kwargs)
        series = pd.Series(results, dtype=object)
        if redirected_only:
            return series[series.notna()]
        return series

    def find_block_specializations(self, parallelization: str | None = None, n_workers: int | None = None,
                                   journal=None, found_only: bool = False, **kwargs) -> pd.Series:
        """The `Specialization` each block would install, or None, by block -- found, not installed (parallelized).

        Each block is formed with specializations off, so nothing is redirected
        or recorded: what this reports is what forming them normally WOULD do.
        The journal is read once, as :meth:`get_block_redirections` reads it.
        *found_only* drops the Nones.
        """
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=object)
        block_cls = self._block_class_()
        block_specs = self._block_specializations_()
        if not block_specs:
            series = pd.Series([None] * n, dtype=object)   # nothing declared: nothing to find
            return series[series.notna()] if found_only else series
        shared = self._shared_blocks_journal_(journal)
        results = self._exec_over_blocks_(
            [self.DatablockSpecializationFinder(i) for i in range(n)], journal=shared,
            tag=f"FINDING SPECIALIZATIONS of {n} blocks [{self.__class__.__name__}]",
            parallelization=parallelization, n_workers=n_workers, **kwargs)
        series = pd.Series(results, dtype=object)
        if found_only:
            return series[series.notna()]
        return series

    def _install_block_specializations_(self, parallelization: str | None = None,
                                        n_workers: int | None = None, journal=None,
                                        validation: str | None = None, **kwargs):
        """Install, block by block, the specialization each resolves -- only when BLOCK declares any.

        Returns the installed `Specialization`s by block index, or None when
        there is nothing to install. Raises `InvalidBlocksError` when a block
        fails validate() -- see `_adopt_block_specializations_`.
        """
        if self.n_blocks == 0 or self._block_class_() is None or not self._block_specializations_():
            return None
        specializations, _, failed, _ = self._adopt_block_specializations_(
            parallelization=parallelization, n_workers=n_workers, journal=journal,
            validation=validation, **kwargs)
        if failed:
            raise InvalidBlocksError(self, list(failed), validation=self._validation_(validation), reasons=failed)
        return specializations

    def _adopt_block_specializations_(self, parallelization: str | None = None,
                                      n_workers: int | None = None, journal=None,
                                      validation: str | None = None, **kwargs):
        """The blocks that are not valid adopt what their specializations resolve to: ``(specializations, invalid, failed, paths)``.

        Which blocks need one is asked of the blocks themselves, by
        *validation* -- see `valid_blocks` -- never of the stack: a stack's own
        validity says nothing about its blocks when it is redirected, its
        topics then being an older build's. The journal is read only when some
        block is not valid, once, here, and handed to every callable.

        *specializations* is the installed `Specialization`s by block index,
        or None when BLOCK declares none or there is no journal to resolve
        against; *invalid* the indices of the blocks STILL not valid --
        unresolved, or adopted with topics left owed; *failed* those that fail
        validate(), which no build as it stands repairs, as ``{index: why}`` --
        see `DatablockSpecializationInstaller`; *paths*
        every block's path, when a full pass found them -- else None.
        """
        n = self.n_blocks
        if n == 0:
            return None, [], {}, []
        validation = self._validation_(validation)
        kind = self._block_kind_()
        item_label = 'tabs' if kind == 'TAB' else 'blocks'

        valid, paths = self._check_blocks_(validation, parallelization=parallelization, n_workers=n_workers)
        invalid = list(valid[~valid].index)
        if not invalid:
            self.log.verbose(
                f"{self.__class__.__name__}: all {n} {item_label} already valid, "
                f"skipping specialization adoption"
            )
            return pd.Series([None] * n, dtype=object), [], {}, paths

        full = 'validate' if validation == 'validate' else 'valid'
        specs = self._block_class_() is not None and self._block_specializations_()
        shared = (journal if journal is not None else self._build_journal_()) if specs else None
        install = shared is not None
        tag = (f"SPECIALIZING {len(invalid)} of {n} {item_label} [{self.__class__.__name__}]" if install
               else f"CLASSIFYING {len(invalid)} of {n} invalid {item_label} [{self.__class__.__name__}]")
        results = self._exec_over_blocks_(
            [self.DatablockSpecializationInstaller(i, validation=full, install=install) for i in invalid],
            journal=shared, tag=tag, parallelization=parallelization, n_workers=n_workers, **kwargs)

        specializations = [None] * n
        by_status = {}
        for i, (sp, status) in zip(invalid, results):
            specializations[i] = sp
            by_status.setdefault(status, []).append(i)
        still_invalid = by_status.get('owes', []) + by_status.get('unresolved', [])
        failed = {i: status for status in ('fails_validation', 'adoption_fails_validation')
                  for i in by_status.get(status, [])}
        self.log.info(
            f"{self.__class__.__name__}: specialization adoption over {n} {item_label}: "
            f"{n - len(invalid) + len(by_status.get('valid', []))} already valid, "
            f"{len(by_status.get('adopted', []))} adopted, {len(by_status.get('owes', []))} adopted but owing, "
            f"{len(by_status.get('unresolved', []))} unresolved, {len(by_status.get('fails_validation', []))} "
            f"failing validate(), {len(by_status.get('adoption_fails_validation', []))} whose adoption failed it"
        )
        if not still_invalid and not failed and paths is not None and self.valid():
            self._record_blocks_manifest_(paths)
        return (pd.Series(specializations, dtype=object) if install else None), still_invalid, failed, paths

    def validate_block(self, idx: int, **kwargs) -> bool:
        """Return whether the block at index *idx* validates."""
        if self.block(idx).validate(**kwargs):
            return True
        # The manifest vouches for every block: not for this one.
        self._forget_blocks_manifest_()
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
        exec_kwargs = self._executor_kwargs_(
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
            'executor_cls', 'worker_done_timeout_sec', 'result_idle_timeout_sec',
            'shuffle_callables',
            'start_method', 'multiprocessing_start_method', 'devices'
        }
        extra_filter_kwargs = {k: v for k, v in kwargs.items() if k not in executor_kw_names}
        clean_kwargs = {k: v for k, v in kwargs.items() if k in executor_kw_names}

        extra_sig_patterns = [f"{k}={v}" for k, v in extra_filter_kwargs.items()]
        sig_clauses = self._normalize_pattern_spec_(signature, *(list(patterns) + extra_sig_patterns))
        tag_clauses = self._normalize_pattern_spec_(tag)
        path_clauses = self._normalize_pattern_spec_(path)

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
        exec_kwargs = self._executor_kwargs_(
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
        # The manifest vouches for every block: not for this one any more.
        self._forget_blocks_manifest_()
        blk = self.block(idx)
        return UNSAFE_clear_block_callable(blk, topics, clear_dirpath, stack=self, idx=idx)

    def UNSAFE_clear_blocks(self, *topics, indices=None, clear_done: bool = True, OVERRIDE: bool = False, clear_dirpath: bool = False, callable=UNSAFE_clear_block_callable):
        """Clear block data, parallelized using the stack's builder settings.

        The interactive UNSAFE confirmation prompt is shown **once** at the
        stack level.  Individual `block.UNSAFE_clear()` calls are invoked
        with `OVERRIDE=True` so they do not re-prompt.

        Parameters
        ----------
        *topics : str
            Forwarded to each block's `UNSAFE_clear()`.
        indices : sequence of int, optional
            If provided, only clear the blocks at these indices. If None,
            clears all blocks in the stack.
        clear_done : bool, default True
            If True and the stack has a 'done' topic that is valid, clear it
            so subsequent builds will rebuild the cleared blocks.
        OVERRIDE : bool
            If `True`, skip the interactive confirmation.
        clear_dirpath : bool
            Forwarded to each block's `UNSAFE_clear()`.
        callable : callable, default UNSAFE_clear_block_callable
            Callable invoked per block to execute the clear operation.
        """
        if not UNSAFE_allowed("UNSAFE_clear_blocks", OVERRIDE=OVERRIDE):
            return self
        # They may leave a block invalid: nothing vouches for the blocks until they are asked again.
        self._forget_blocks_manifest_()

        if indices is not None:
            idx_list = [int(i) for i in indices]
            block_list = [self.block(i) for i in idx_list]
            callables = [functools.partial(callable, blk, topics, clear_dirpath, stack=self, idx=i) for i, blk in zip(idx_list, block_list)]
        else:
            block_list = self.blocks()
            callables = [functools.partial(callable, blk, topics, clear_dirpath, stack=self, idx=idx) for idx, blk in enumerate(block_list)]

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

        executor.exec_callables(callables)

        if clear_done and getattr(self, 'has_topics', lambda: False)() and 'done' in self.topics() and self.valid_topic('done'):
            self.UNSAFE_clear('done', OVERRIDE=True)

        self.log.info(f"UNSAFE_clear_blocks complete: {self.__class__.__name__}")
        self.write_journal_entry(event="UNSAFE_clear_blocks:end")
        return self

    def UNSAFE_clear_block_redirections(self, *, OVERRIDE: bool = False,
                                        parallelization: str | None = None,
                                        n_workers: int | None = None, **kwargs) -> pd.Series:
        """Every block's `UNSAFE_clear_redirection`, parallelized; which ones had one, by block.

        Asks once, here, rather than once per block. No data is touched.
        """
        if not UNSAFE_allowed("UNSAFE_clear_block_redirections", OVERRIDE=OVERRIDE):
            return pd.Series([], dtype=bool)
        # They may leave a block invalid: nothing vouches for the blocks until they are asked again.
        self._forget_blocks_manifest_()
        n = self.n_blocks
        if n == 0:
            return pd.Series([], dtype=bool)
        self.write_journal_entry(event="UNSAFE_clear_block_redirections:begin")
        results = self._exec_over_blocks_(
            [self.DatablockRedirectionClearer(i) for i in range(n)],
            tag=f"CLEARING REDIRECTION of {n} blocks [{self.__class__.__name__}]",
            parallelization=parallelization, n_workers=n_workers, **kwargs)
        cleared = pd.Series(results, dtype=bool)
        self.__dict__.pop('block_redirections', None)
        self.write_journal_entry(event="UNSAFE_clear_block_redirections:end",
                                 note=f"{int(cleared.sum())}/{n}")
        return cleared

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
        # They may leave a block invalid: nothing vouches for the blocks until they are asked again.
        self._forget_blocks_manifest_()

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
        if getattr(self, 'result_idle_timeout_sec', None) is not None:
            executor_kwargs['result_idle_timeout_sec'] = self.result_idle_timeout_sec
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
            Column filter kwargs passed to ``datajournal()`` when reading the child block's journal.
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
        list
            What each block's :meth:`UNSAFE_redirect` returned, in block order:
            a ``(Redirection, Validation)`` pair under ``dry_run, dry_validate``,
            a bare :class:`Datablock.Redirection` under ``dry_run`` alone, True
            when a redirection was installed and False when it was refused, and
            None where the redirector declined a block entirely.
        """
        allowed = UNSAFE_allowed("UNSAFE_redirect_blocks", OVERRIDE=OVERRIDE)
        if redirector is None:
            raise ValueError("UNSAFE_redirect_blocks requires a redirector callable")
        block_list = self.blocks()
        total = len(block_list)
        if not allowed:
            return 0, total
        # They may leave a block invalid: nothing vouches for the blocks until they are asked again.
        self._forget_blocks_manifest_()

        par = parallelization if parallelization is not None else getattr(self, 'parallelization', None)
        nw = n_workers if n_workers is not None else getattr(self, 'n_workers', 1)
        self.log.info(
            f"UNSAFE_redirect_blocks: redirecting {total} blocks, "
            f"parallelization={par}, n_workers={nw}"
        )
        self.write_journal_entry(event="UNSAFE_redirect_blocks:begin")

        try:
            blk0 = block_list[0] if total > 0 else None
            journal = blk0.datajournal(n_workers=nw, **(filter or {})) if blk0 is not None else None
        except Exception as e:
            self.log.detailed(f"UNSAFE_redirect_blocks: datajournal() lookup: {e}")
            journal = None

        tag = f"REDIRECTING {total} blocks [{self.__class__.__name__}, n_workers={nw}]"
        executor = callable_executor(par, n_workers=nw, tag=tag)

        callables = [functools.partial(_UNSAFE_redirect_block_callable_, redirector, blk, self, idx, journal=journal, validate=validate) for idx, blk in enumerate(block_list)]
        results = list(executor.exec_callables(callables) or [])

        successes = sum(1 for r in results if _redirect_succeeded_(r))

        self.log.info(f"UNSAFE_redirect_blocks complete: {self.__class__.__name__} ({successes}/{total} succeeded)")
        self.write_journal_entry(event="UNSAFE_redirect_blocks:end", note=f"{successes}/{total}")
        return results

    # 3. Accessors ---------------------------------------------------------

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

    @property
    def n_blocks(self) -> int:
        """Return the number of blocks.

        Subclasses **must** override this property.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement n_blocks"
        )

    @functools.cached_property
    def block_redirections(self) -> pd.Series:
        """:meth:`get_block_redirections` with its defaults, resolved once."""
        return self.get_block_redirections()

    # 4. Helpers -----------------------------------------------------------

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

    @staticmethod
    def _normalize_pattern_spec_(spec, *extra_patterns) -> list[tuple]:
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
    def _match_single_sig_pattern_(sig: str, p) -> bool:
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
    def _matches_sig_clauses_(cls, sig: str, clauses: list[tuple]) -> bool:
        if not clauses:
            return True
        for clause in clauses:
            if all(cls._match_single_sig_pattern_(sig, p) for p in clause):
                return True
        return False

    @staticmethod
    def _match_single_tag_pattern_(text: str | None, p) -> bool:
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
    def _matches_tag_clauses_(cls, tag: str | None, clauses: list[tuple]) -> bool:
        if not clauses:
            return True
        if tag is None:
            return False
        for clause in clauses:
            if all(cls._match_single_tag_pattern_(tag, p) for p in clause):
                return True
        return False

    @classmethod
    def _collect_path_strings_(cls, val, out: list[str]):
        if isinstance(val, str):
            out.append(val)
        elif isinstance(val, dict):
            for v in val.values():
                cls._collect_path_strings_(v, out)
        elif isinstance(val, (list, tuple, set)):
            for v in val:
                cls._collect_path_strings_(v, out)

    @classmethod
    def _get_block_paths_(cls, blk) -> list[str]:
        """Extract all path strings for a block."""
        paths_list = []
        try:
            p = blk.paths()
            cls._collect_path_strings_(p, paths_list)
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
    def _matches_path_clauses_(cls, block_paths: list[str], clauses: list[tuple]) -> bool:
        if not clauses:
            return True
        if not block_paths:
            return False
        for clause in clauses:
            if all(any(cls._match_single_tag_pattern_(path_str, p) for path_str in block_paths) for p in clause):
                return True
        return False

    def _executor_kwargs_(self, tag: str | None = None, n_workers: int | None = None, executor_cls=None, **kwargs) -> dict:
        nw = n_workers if n_workers is not None else getattr(self, 'n_workers', 1)
        cls = executor_cls or self.executor_cls
        executor_kwargs = dict(
            n_workers=nw,
            tag=tag or f"EXECUTING [{self.__class__.__name__}]",
        )
        if hasattr(self, 'worker_done_timeout_sec') and self.worker_done_timeout_sec is not None:
            executor_kwargs['worker_done_timeout_sec'] = self.worker_done_timeout_sec
        if getattr(self, 'result_idle_timeout_sec', None) is not None:
            executor_kwargs['result_idle_timeout_sec'] = self.result_idle_timeout_sec
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

    @classmethod
    def _check_validation_(cls, validation: str) -> str:
        if validation not in cls.VALIDATIONS:
            raise ValueError(f"validation must be one of {cls.VALIDATIONS}, got {validation!r}")
        return validation

    def _validation_(self, validation: str | None = None) -> str:
        """The validation in force: *validation* when given, else this stack's -- see `valid_blocks`."""
        if validation is not None:
            return self._check_validation_(validation)
        return getattr(self, 'validation', None) or self.VALIDATION

    def _blocks_manifest_path_(self) -> str:
        """Where the record of this stack's valid blocks is: in its OWN directory, never redirected."""
        return os.path.join(self.anchorkeypath, '.blocks', 'manifest')

    def _read_blocks_manifest_(self) -> list[str] | None:
        """The block paths the manifest records, by index -- or None when there is none."""
        path = self._blocks_manifest_path_()
        try:
            if not self.fs.exists(path):
                return None
            with self.fs.open(path, 'r') as f:
                lines = f.read().splitlines()
        except Exception as e:
            self.log.warning(f"{self.anchorkeypath}: cannot read the blocks' manifest {path}: {e}")
            return None
        if not lines or not lines[0].startswith('n_blocks='):
            self.log.warning(f"{self.anchorkeypath}: the blocks' manifest {path} is malformed; ignoring it")
            return None
        paths = lines[1:]
        if len(paths) != int(lines[0].removeprefix('n_blocks=')):
            self.log.warning(f"{self.anchorkeypath}: the blocks' manifest {path} is truncated; ignoring it")
            return None
        return paths

    def _record_blocks_manifest_(self, paths: list[str]):
        """Record that every block is valid, and the path each one has: after a full pass over them, only."""
        path = self._blocks_manifest_path_()
        try:
            ensure_path(os.path.dirname(path), storage_options=self.storage_options)
            with self.fs.open(path, 'w') as f:
                f.write(f"n_blocks={len(paths)}\n" + "".join(f"{p}\n" for p in paths))
            self.__dict__['_blocks_cross_checked_cache'] = True
            self.log.verbose(f"{self.anchorkeypath}: recorded the manifest of {len(paths)} valid blocks")
        except Exception as e:
            self.log.warning(f"{self.anchorkeypath}: could not record the blocks' manifest at {path}: {e}")

    def _forget_blocks_manifest_(self):
        """Stop vouching for the blocks: something may have made one of them invalid."""
        self.__dict__['_blocks_cross_checked_cache'] = False
        try:
            path = self._blocks_manifest_path_()
            if self.fs.exists(path):
                self.fs.rm(path)
        except Exception as e:
            self.log.warning(f"{self.anchorkeypath}: could not remove the blocks' manifest: {e}")

    def _blocks_cross_checked_(self) -> bool:
        """Whether the manifest vouches for every block, cross-checked against the blocks as they are now.

        The manifest was recorded after a full pass found every block valid,
        with the path each had. Block 0, and ``cross_check_blocks - 1`` others
        drawn at random, are formed and their paths compared with it: whatever
        moves blocks to identities of their own -- BLOCK renamed, its TOPICS or
        VERSION changed, blocks reordered -- shows as a mismatch, and the
        manifest vouches for nothing. What it cannot see is data removed
        behind the stack's back: ``validation='valid'`` asks every block.
        Once per instance.
        """
        if '_blocks_cross_checked_cache' not in self.__dict__:
            self.__dict__['_blocks_cross_checked_cache'] = self._cross_check_blocks_()
        return self.__dict__['_blocks_cross_checked_cache']

    def _cross_check_blocks_(self) -> bool:
        recorded = self._read_blocks_manifest_()
        if recorded is None:
            return False
        n = self.n_blocks
        if len(recorded) != n:
            self.log.warning(f"{self.anchorkeypath}: the manifest records {len(recorded)} blocks, "
                             f"but there are {n} now; it vouches for nothing")
            return False
        if n == 0:
            return True
        k = getattr(self, 'cross_check_blocks', None) or self.CROSS_CHECK_BLOCKS
        seed = getattr(self, 'cross_check_seed', None)
        sample = [0] + sorted(random.Random(seed).sample(range(1, n), min(k - 1, n - 1)))
        self.log.verbose(f"{self.anchorkeypath}: cross-checking blocks {sample} of {n} against the manifest")
        for i in sample:
            current = self.block(i).anchorkeypath
            if current.rstrip('/') != recorded[i].rstrip('/'):
                self.log.warning(
                    f"{self.anchorkeypath}: the manifest records block {i} as {recorded[i]!r}, but it "
                    f"is {current!r} now; the manifest vouches for nothing, and every block is asked"
                )
                return False
        return True

    def _block_class_(self):
        """The class this stack's blocks are: its BLOCK, or None when it declares none."""
        return getattr(self, 'BLOCK', None)

    def _block_specializations_(self):
        """The specializations for this stack's blocks: from instance attributes, or the BLOCK class."""
        specs = getattr(self, 'BLOCK_SPECIALIZATIONS', None) or getattr(self, 'BLOCK_SPECIALIZATION', None)
        if specs is None:
            specs = getattr(self, 'TAB_SPECIALIZATIONS', None) or getattr(self, 'TAB_SPECIALIZATION', None)
        if specs is not None:
            block_cls = self._block_class_()
            spec_cls = getattr(block_cls, 'Specialization', Datablock.Specialization) if block_cls else Datablock.Specialization
            if isinstance(specs, str):
                records = self._specialization_records_(specs)
                return [spec_cls.from_record(r) for r in (records or [])]
            if isinstance(specs, (Datablock.Specialization, dict)):
                specs = [specs]
            else:
                specs = list(specs)
            return [spec_cls.from_record(sp) if isinstance(sp, dict) else sp for sp in specs]
        block_cls = self._block_class_()
        return getattr(block_cls, 'SPECIALIZATIONS', None) or []

    def _block_kind_(self) -> str:
        """'TAB' if this stack or its table declares TAB, else 'BLOCK'."""
        if getattr(self, 'TAB', None) is not None:
            return 'TAB'
        table = getattr(getattr(self, 'var', None), 'partition', None)
        table = getattr(table, 'datapoint_table', None)
        if table is not None and getattr(table, 'TAB', None) is not None:
            return 'TAB'
        return 'BLOCK'

    def _blocks_datajournal_(self, **kwargs):
        """``(journal, anchor, url)`` of this stack's blocks: read once, for all of them.

        From `BLOCK` when the stack declares one: the class's anchor under
        `_blocks_datalake_()` -- the blocks are taken to share it -- with no block
        formed at all. Otherwise from ``block(0)``, formed only to say where its
        journal is: uncached, and with specializations off, since forming it
        normally can install one and record it. Raises FileNotFoundError when
        nothing was ever journalled there.
        """
        block_cls = self._block_class_()
        kind = self._block_kind_()
        if block_cls is not None:
            anchor, lake = block_cls.anchor, self._blocks_datalake_()
            # And every other anchor BLOCK's specializations look in: a renamed
            # class's old journal, read here once rather than by every block.
            block_specs = self._block_specializations_()
            others = [a for a in dict.fromkeys(
                sp.anchor for sp in block_specs
                if sp.anchor is not SAME) if a != anchor]
            frames = []
            for a in [anchor] + others:
                try:
                    read_kwargs = dict(kwargs)
                    read_kwargs.setdefault('desc', f"Reading {a} ({kind}) journal files")
                    frames.append(Datajournal.read(a, datalake=lake, storage_options=self.storage_options,
                                                   log=self.log, **read_kwargs))
                except FileNotFoundError:
                    if a == anchor and not others:
                        raise
            if not frames:
                raise FileNotFoundError(f"no journal under {[anchor] + others!r} in {lake!r}")
            journal = frames[0] if len(frames) == 1 else DatajournalFrame(
                pd.concat(frames), storage_options=self.storage_options)
            return journal, anchor, lake
        first = self._form_block_(0, use_specializations=False)
        return first.datajournal(**kwargs), first.anchor, first.datalake

    def _build_journal_(self):
        """The `BlocksJournal` a build hands its callables, or None.

        Read only when there is something to resolve against it -- a BLOCK that
        declares SPECIALIZATIONS -- and once, here in the parent.
        """
        # Read once per build(): the blocks' adoption and the building of the
        # rest are both this build's -- see build() -- and share the one read.
        building = self.__dict__.get('__build_journal__', ABSENT)
        if building is not ABSENT:
            return building
        block_cls = self._block_class_()
        block_specs = self._block_specializations_()
        if block_cls is None or not block_specs:
            return None
        kind = self._block_kind_()
        item_label = 'tabs' if kind == 'TAB' else 'blocks'
        n_items = getattr(self, 'n_tabs', self.n_blocks)

        anchor = block_cls.anchor
        self.log.verbose(f"{self.__class__.__name__}: reading the {anchor} ({kind}) journal "
                         f"for {n_items} {item_label} to resolve against...")
        try:
            journal, anchor, lake = self._blocks_datajournal_()
        except FileNotFoundError:
            return None
        self.log.verbose(f"{self.__class__.__name__}: read the {anchor} ({kind}) journal once "
                         f"({len(journal)} entries) for {n_items} {item_label} to resolve against")
        journal = BlocksJournal(journal, anchor, lake)
        if self.__dict__.get('__building__'):
            self.__dict__['__build_journal__'] = journal
        return journal

    def _with_build_journal_(self, callables, callable_kwargs):
        """*callable_kwargs*, with the build's journal -- when the callables take one.

        A ctx kwarg, so the executor sends it once per worker, not once per
        callable. Only to callables whose ``__call__`` accepts ``journal=``: a
        BlockMaker subclass written before it did would be handed an argument
        it cannot take.
        """
        if not callables or 'journal' in callable_kwargs:
            return callable_kwargs
        try:
            params = inspect.signature(type(callables[0]).__call__).parameters
        except (TypeError, ValueError):
            return callable_kwargs
        takes = 'journal' in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        if not takes:
            return callable_kwargs
        journal = self._build_journal_()
        return callable_kwargs if journal is None else {**callable_kwargs, 'journal': journal}

    def _blocks_datalake_(self):
        """Where this stack's blocks are stored: its own datalake -- a DatatablePart's are its table's."""
        return self.datalake

    @contextlib.contextmanager
    def _block_specializations_in_force_(self, use_specializations='stack'):
        """This stack's ``use_block_specializations`` in force while a block is formed -- see `_form_block_`.

        Wherever a block of it is formed: `_form_block_`, and the makers that
        form and build blocks in the stack's workers, which call ``__block__``
        themselves -- and so, without it, formed every block with its own
        setting, whatever the stack's said.
        """
        wanted = (getattr(self, 'use_block_specializations', None)
                  if use_specializations == 'stack' else use_specializations)
        stack = getattr(_BLOCK_SPECIALIZATIONS, 'stack', None)
        if stack is None:
            stack = _BLOCK_SPECIALIZATIONS.stack = []
        stack.append(_OWN_SETTING if wanted is None else wanted)
        try:
            yield
        finally:
            stack.pop()

    def _form_block_(self, idx: int, *, use_specializations='stack'):
        """Block *idx*, formed and adopted -- uncached; :meth:`block` caches it.

        Formed with ``use_specializations`` as this stack's
        ``use_block_specializations`` says, unless the call names one: forming
        is inside ``__block__``, which is the subclass's, so the setting has to
        be in force while it runs, for the block's build to use it later. A
        block's own ``use_specializations=`` wins over it.
        """
        with self._block_specializations_in_force_(use_specializations):
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
            s = self._adopt_(s, keyby=True)
        declared = self._block_class_()
        if declared is not None and not isinstance(s, declared):
            raise TypeError(
                f"{self.__class__.__name__}.block({idx}) is a {type(s).__name__}, "
                f"not the {declared.__name__} its BLOCK declares"
            )
        return s

    def _exec_over_blocks_(self, callables, *, tag, parallelization=None, n_workers=None, journal=None, **kwargs):
        """Run one callable per block under this stack's executor -- chosen as `valid_blocks` chooses it."""
        executors = self._get_executors_()
        if parallelization is not None:
            key = parallelization.lower()
        elif n_workers is not None:
            key = 'multithreading' if n_workers > 0 else 'inline'
        else:
            key = (getattr(self, 'parallelization', None) or 'inline').lower()
        if key not in executors:
            raise ValueError(f"Unknown parallelization {key!r}. Choose from {list(executors)}")
        executor_cls = executors[key]
        executor = executor_cls(**self._executor_kwargs_(
            tag=tag, n_workers=n_workers, executor_cls=executor_cls, **kwargs))
        # The journal as a ctx kwarg: sent once per worker, not with each callable.
        ctx = {} if journal is None else {'journal': journal}
        return executor.exec_callables(callables, self, **ctx)

    def _shared_blocks_journal_(self, journal=None):
        """A `BlocksJournal` for these blocks: *journal* as given, or read once -- or None.

        Read only when BLOCK declares SPECIALIZATIONS: without them, a block not
        redirected answers from its own hash directory, and one redirected
        answers from its record -- neither needs the anchor's journal.
        """
        block_cls = self._block_class_()
        if journal is not None:
            return BlocksJournal(journal, block_cls.anchor if block_cls is not None else None,
                                 self._blocks_datalake_())
        block_specs = self._block_specializations_()
        if block_cls is None or not block_specs:
            return None
        kind = self._block_kind_()
        item_label = 'tabs' if kind == 'TAB' else 'blocks'
        n_items = getattr(self, 'n_tabs', self.n_blocks)
        anchor = block_cls.anchor
        self.log.verbose(f"{self.__class__.__name__}: reading the {anchor} ({kind}) journal "
                         f"for {n_items} {item_label} to resolve against...")
        try:
            journal, anchor, lake = self._blocks_datajournal_()
        except FileNotFoundError:
            return None                 # nothing to share: each block reads its own
        self.log.verbose(f"{self.__class__.__name__}: read the {anchor} ({kind}) journal once "
                         f"({len(journal)} entries) for {n_items} {item_label} to resolve against")
        return BlocksJournal(journal, anchor, lake)


def _redirect_succeeded_(result) -> bool:
    """Whether one block's ``UNSAFE_redirect`` result counts as a success.

    True is an installed redirection. A :class:`Datablock.Redirection` is a dry
    run's proposal, which counts as resolved. A ``(Redirection, Validation)``
    pair is a proposal whose data was looked for as well, and one that was not
    found is not something to count.
    """
    if result is True:
        return True
    if isinstance(result, tuple):
        proposal, validation = result
        return getattr(proposal, 'paths', None) is not None and bool(validation)
    return getattr(result, 'paths', None) is not None


def _UNSAFE_redirect_block_callable_(redirector, block, stack, idx, *, journal: DatajournalFrame|None = None, validate: bool = False):
    target = redirector(block, stack, idx, journal=journal)
    if not target:
        return None
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
    return redirected


def _fscopy_item_callable_(src_item, dst_item, storage_options):
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


