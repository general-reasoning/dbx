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
import atexit
import collections
from collections.abc import Iterable, Sequence
import copy
from dataclasses import dataclass, fields, asdict, replace, is_dataclass
import datetime
import functools
import gc
import hashlib
import importlib
import inspect
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import pprint as _pprint_
import queue
import re
import signal
import socket
import subprocess
import sys
import shutil
import tempfile
import threading
import time as time_module
import traceback as tb
from typing import Union, Optional, Sequence, Callable
import types
import uuid
import yaml


import git

import tqdm

# Disable tqdm's background TMonitor thread.
# The monitor races with explicit update() calls (causing the bar count to
# visually bounce) and is alive at fork() time, triggering the Python 3.12
# DeprecationWarning "This process is multi-threaded, use of fork() may lead
# to deadlocks".  We drive all updates explicitly so the monitor is unneeded.
tqdm.tqdm.monitor_interval = 0

import numpy as np

import fsspec

import pandas as pd



__eval__ = __builtins__['eval']

from .dataparts import *
__version__ = "0.0.2"

DBX_GIT_REPO = os.environ.get('DBX_GIT_REPO')
if DBX_GIT_REPO is None:
    try:
        import git
        _repo = git.Repo('.', search_parent_directories=True)
        DBX_GIT_REPO = _repo.working_tree_dir
    except (ImportError, Exception):
        pass
_DBX_GIT_REPO_ = DBX_GIT_REPO
DBX_USE_WORK_REPO = None
DBX_WORK_ROOT = None


def dbx_repos(repopath=None):
    if repopath is None:
        repopath = DBX_GIT_REPO
    if repopath is None:
        return None, None
    paths = repopath.split(':')
    if len(paths) == 1:
        return None, paths[0]
    elif len(paths) == 2:
        if '/dbx' in paths[0]:
            return paths[0], paths[1]
        else:
            return paths[1], paths[0]
    else:
        raise ValueError(f"Too many paths in repopath: {repopath}")

def dbx_revisions(revision):
    if isinstance(revision, str):
        if ':' in revision:
            parts = revision.split(':')
            if len(parts) == 2:
                return parts[0], parts[1]
            else:
                raise ValueError(f"Revisions string must have exactly one ':': {revision}")
        return None, revision
    elif isinstance(revision, tuple):
        if len(revision) == 2:
            return revision
        else:
            raise ValueError(f"Revisions tuple must have exactly two elements: {revision}")
    elif revision is None:
        return None, None
    else:
        raise ValueError(f"Unknown revision type: {type(revision)}")

def dbx_versions(version):
    if isinstance(version, str):
        if ':' in version:
            parts = version.split(':')
            if len(parts) == 2:
                return parts[0], parts[1]
            else:
                raise ValueError(f"Version string must have at most one ':': {version}")
        return None, version
    return None, None


def gitwrkreposetup(revision=None, *, gitrepo=None, reason: str = "", log=None):
    if log is None:
        log = Logger(name="gitwrkreposetup")
    global DBX_GIT_REPO
    global DBX_USE_WORK_REPO
    global DBX_WORK_ROOT
    
    dbx_repo, project_repo = dbx_repos(gitrepo)
    dbx_rev, project_rev = dbx_revisions(revision)

    def setup_wrkrepo(repo, rev, name):
        nonlocal log
        if repo is None:
            return None
        log.verbose(f"SETTING UP WORK REPO for {name} from {repo=} {reason} with revision {rev}")
        wrkroot = tempfile.TemporaryDirectory()
        package = os.path.basename(repo)
        wrkrepo = os.path.join(wrkroot.name, package)
        _pkgpath = gitclone(repo, wrkrepo)
        assert wrkrepo == _pkgpath
        if rev is not None:
            gitcheckout(wrkrepo, rev)
            log.verbose(f"Checked out {wrkrepo} to revision {rev}")
        else:
            log.verbose(f"Using {wrkrepo} at HEAD")
        sys.path.insert(0, wrkrepo)
        return wrkroot, wrkrepo

    use_wrkrepo = os.environ.get('DBX_USE_WORK_REPO') == 'True' or revision is not None
    if use_wrkrepo and DBX_USE_WORK_REPO is None:
        if DBX_GIT_REPO is None:
            raise ValueError("DBX_GIT_REPO is not set and could not be detected. Cannot setup temporary wrkrepo.")

        # --- Early dirty check on the ORIGINAL repos ----------------------
        # Must happen *before* cloning: the clone uses HEAD, so any
        # uncommitted changes would be silently lost.
        #
        # When revision is None (e.g. import-time wrkrepo setup) we only
        # warn — the caller may never write a journal or build anything.
        # When a specific revision IS requested the caller needs definitive
        # code identity, so a dirty tree is a hard error.
        def _check_dirty(path):
            if path is None:
                return
            repo = git.Repo(path)
            if repo.is_dirty() and not os.environ.get('DBX_DIRTY_REPO_OK'):
                msg = (
                    f"Dirty git repo: {path}: commit your changes before "
                    f"creating a work-repo clone (uncommitted changes would be lost)"
                )
                if revision is not None:
                    raise ValueError(msg)
                else:
                    log.info(f"WARNING: {msg}")
        _check_dirty(dbx_repo)
        _check_dirty(project_repo)
        # ------------------------------------------------------------------

        dbx_wrk = setup_wrkrepo(dbx_repo, dbx_rev, "dbx")
        project_wrk = setup_wrkrepo(project_repo, project_rev, "project")

        DBX_WORK_ROOT = (dbx_wrk[0] if dbx_wrk else None, project_wrk[0] if project_wrk else None)
        
        dbx_wrkrepo = dbx_wrk[1] if dbx_wrk else None
        project_wrkrepo = project_wrk[1] if project_wrk else None
        
        if dbx_wrkrepo:
            wrkrepo_str = f"{dbx_wrkrepo}:{project_wrkrepo}"
        else:
            wrkrepo_str = project_wrkrepo
        
        globals()['DBX_USE_WORK_REPO'] = wrkrepo_str
        os.environ['DBX_GIT_REPO'] = wrkrepo_str
        # Signal to worker processes (Ray, multiprocessing) that we are
        # operating from a wrkrepo and the dirty check should be skipped.
        os.environ['DBX_WORK_ROOT'] = wrkrepo_str
        
        if 'DBX_USE_WORK_REPO' in os.environ:
            del os.environ['DBX_USE_WORK_REPO']
            
        log.verbose(f"DBX_USE_WORK_REPO: {wrkrepo_str}")

    elif revision is not None and DBX_USE_WORK_REPO is not None:
        # Work repos already exist (cloned at HEAD during import).
        # Checkout the requested revision so that subsequent imports
        # (e.g. project modules evaluated from a journal entry quote)
        # pick up code from the correct commit.
        #
        # NOTE: modules already in sys.modules (including dbx itself)
        # will NOT be refreshed — only not-yet-imported modules benefit.
        dbx_wrkrepo, project_wrkrepo = dbx_repos(DBX_USE_WORK_REPO)
        if dbx_rev is not None and dbx_wrkrepo is not None:
            gitcheckout(dbx_wrkrepo, dbx_rev)
            log.verbose(f"Checked out existing work repo {dbx_wrkrepo} to revision {dbx_rev}")
        if project_rev is not None and project_wrkrepo is not None:
            gitcheckout(project_wrkrepo, project_rev)
            log.verbose(f"Checked out existing work repo {project_wrkrepo} to revision {project_rev}")
        # Clear finder caches so the import machinery sees the new files.
        importlib.invalidate_caches()


@dataclass
class LogVolume:
    """Bundle of log-verbosity settings."""
    info: bool = None
    verbose: bool = None
    debug: bool = None
    detailed: bool = None




def journal(cls_or_df, loc=None, *, iloc=None, url=None, storage_options=None, **filter_kwargs):
    """Retrieve or wrap a Datablock journal.

    Parameters
    ----------
    cls_or_df : type | str | pd.DataFrame
        A Datablock class, an anchor string, or a raw DataFrame.
    loc : int, optional
        If given, return a single :class:`JournalEntry` at this label index.
    iloc : int, optional
        If given, return a single :class:`JournalEntry` at this positional index.
        Mutually exclusive with *loc*.
    url : str, optional
        Storage URL.  Defaults to ``DBX_ROOT``.
    storage_options : dict, optional
        Storage options for fsspec.  Defaults to ``default_storage_options()``.
    **filter_kwargs
        Forwarded to :class:`JournalFrame` for filtering.

    Returns
    -------
    JournalFrame or JournalEntry
    """
    if loc is not None and iloc is not None:
        raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
    if isinstance(cls_or_df, pd.DataFrame):
        return JournalFrame(cls_or_df, storage_options=storage_options, **filter_kwargs)
    else:
        if isinstance(cls_or_df, str):
            anchor = cls_or_df
        else:
            anchor = cls_or_df.__module__ + "." + cls_or_df.__name__
        return Datablock.Journal(anchor, loc=loc, iloc=iloc, url=url, storage_options=storage_options, **filter_kwargs)



class JournalEntry(pd.Series):
    """A single row from a Datablock journal, with convenience accessors.

    Inherits from :class:`pandas.Series` so all standard pandas
    operations work.  Named properties expose journal-specific fields
    (``anchor``, ``hash``, ``url``, ``revision``, …).
    """
    def __init__(self, series: pd.Series, *, storage_options: dict = None,
                 logger: Logger = Logger(name="JournalEntry")):
        super().__init__(series)
        self.storage_options = storage_options or {}
        self.logger = logger

    def __tag__(self):
        return f"JournalEntry:{self.anchor}/{self.hash}"

    @property
    def anchor(self):
        return self.get('anchor')

    @property
    def superhash(self):
        return self.get('superhash')
    
    @property
    def hash(self):
        return self.get('hash')

    @property
    def url(self):
        return self.get('url')

    @property
    def revision(self):
        return self.get('revision')

    @property
    def gitrepo(self):
        return self.get('gitrepo')

    @property
    def version(self):
        return self.get('version')

    @property
    def keyby(self):
        return self.get('keyby', 'taghash')

    @property
    def tag(self):
        return self.get('tag')

    @property
    def key(self):
        """Reconstruct the key from journal fields, mirroring Datablock.key."""
        recorded_key = self.get('key')
        if recorded_key is not None and not (isinstance(recorded_key, float) and pd.isna(recorded_key)):
            return recorded_key
            
        keyby = self.keyby
        if keyby is None:
            key = None
        elif keyby == 'hash':
            key = self.hash
        elif keyby == 'superhash':
            key = self.superhash
        elif self.keyby == 'norm':
            key = self.norm()
        elif keyby == 'tag':
            key = self.tag
        elif keyby in ('taghash', 'tag_hash'):
            if self.tag is None:
                key = self.hash
            else:
                key = f"{self.tag}/{self.hash[:8]}"
        elif keyby == 'version_hash':
            if self.version is not None:
                key = f"version={self.version}/{self.hash[:8]}"
            else:
                key = self.hash
        elif keyby == 'tag_version_hash':
            parts = []
            if self.tag is not None:
                parts.append(self.tag)
            if self.version is not None:
                parts.append(f"version={self.version}")
            if parts:
                parts.append(self.hash[:8])
            else:
                parts.append(self.hash)
            key = '/'.join(parts)
        else:
            key = self.hash  # fallback
        return key

    @property
    def anchorkey(self):
        key = self.key
        return os.path.join(self.anchor, key) if key else self.anchor

    @property
    def root(self):
        """Protocol-free root derived from ``url`` via ``fsspec.url_to_fs``."""
        url = self.get('url')
        if url is None:
            return self.get('root')  # legacy fallback
        _, root = fsspec.url_to_fs(url, **self.storage_options)
        return root

    @property
    def anchorkeypath(self):
        recorded_path = self.get('anchorkeypath')
        if recorded_path is not None and not (isinstance(recorded_path, float) and pd.isna(recorded_path)):
            return recorded_path
            
        url = self.get('url')
        if url is None:
            # Legacy fallback when only 'root' is available
            root = self.get('root')
            return os.path.join(root, self.anchorkey) if root else self.anchorkey
        fs, root = fsspec.url_to_fs(url, **self.storage_options)
        return fs_full_path(fs, os.path.join(root, self.anchorkey))

    def read(self, *things, raw: bool = False, deslash: bool = False, safe: bool = False):
        def read_thing(thing):
            if hasattr(self, thing) and getattr(self, thing) is not None:
                path = getattr(self, thing)
                _, _ext = os.path.splitext(path)
                ext = _ext[1:]
                try:
                    if raw or ext == 'txt' or ext == 'log':
                        result = read_str(getattr(self, thing), storage_options=self.storage_options)
                    elif ext == 'yaml':
                        result = read_yaml(getattr(self, thing), safe=safe, storage_options=self.storage_options)
                    else:
                        raise ValueError(f"Uknown journal entry field extention for {thing}: {ext}")
                except FileNotFoundError:
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
        gitwrkreposetup(revision=revision, gitrepo=gitrepo, reason=f"because of evaluating a JournalEntry field {thing}")
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
            self.logger.info(f"Instantiating {self.__tag__()} with revision from journal entry {revision}")
        else:
            self.logger.info(f"Instantiating {self.__tag__()} with revision {revision}")
        if gitrepo == 'journal_entry':
            gitrepo = self.gitrepo
            self.logger.info(f"Instantiating {self.__tag__()} with gitrepo from journal entry {gitrepo}")
        else:
            self.logger.info(f"Instantiating {self.__tag__()} with gitrepo {gitrepo}")
        return self.eval('quote', eval=True, gitrepo=gitrepo, revision=revision)

    def inst(self, gitrepo=None, revision='journal_entry'):
        if gitrepo is None:
            gitrepo = DBX_GIT_REPO
        if gitrepo is None:
            gitrepo = 'journal_entry'
        return self.instantiate(gitrepo=gitrepo, revision=revision)

    @property
    def bid(self):
        """Reconstruct a Datablock.Bid from this journal entry."""
        return Datablock.Bid(
            hash=self.hash,
            version=self.version,
            revision=self.revision,
            dfn=self.read('dfn', safe=True) or {},
            kwargs=self.read('kwargs', safe=True) or {},
            spec=self.read('spec', safe=True) or {},
            quote=self.read('quote') or '',
            repr=self.read('repr') or '',
            norm=self.read('norm') or '',
            hashstr=self.read('hashstr') or '',
            supernorm=self.read('supernorm') or '',
            superhashstr=self.read('superhashstr') or '',
            superhash=self.superhash,
            anchor=self.anchor,
            tag=self.tag,
            key=self.key,
            keyby=self.keyby,
        )
    


class JournalFrame(pd.DataFrame):
    _metadata = ['storage_options', 'logger']

    def __init__(self, df: pd.DataFrame|None, *, storage_options: dict = None,
                 parse_datetimes: bool = True, logger: Logger = Logger(), **filter_kwargs):
        
        # Guard against an empty journal (no parquet files written yet).
        if df is None:
            df = pd.DataFrame()

        # Process the dataframe before calling super().__init__()
        if parse_datetimes:
            if 'datetime' in df.columns and not isinstance(df['datetime'].iloc[0], datetime.datetime): # TODO: use dtype?
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H-%M-%S.%f')
        for k, v in filter_kwargs.items():
            if k == 'date':
                if isinstance(v, str):
                    v = pd.to_datetime(v).date()
                elif isinstance(v, list):
                    v = [pd.to_datetime(x).date() for x in v]
                if isinstance(v, list):
                    df = df[df['datetime'].dt.date.isin(v)]
                else:
                    df = df[df['datetime'].dt.date == v]         
            elif k == 'hash':
                # Prefix match: allow short hashes
                if isinstance(v, list):
                    df = df[df['hash'].apply(lambda h: h.startswith(tuple(v)))]
                else:
                    df = df[df['hash'].str.startswith(v)]
            else:
                if k == 'datetime':
                    if isinstance(v, str):
                        v = datetime.datetime.strptime(v, '%Y-%m-%dT%H-%M-%S.%f')
                    elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                        v = [datetime.datetime.strptime(x, '%Y-%m-%dT%H-%M-%S.%f') for x in v]
                if isinstance(v, list):
                    df = df[df[k].isin(v)]
                else:
                    df = df[df[k] == v]
        
        # Initialize the DataFrame first
        super().__init__(df)
        
        # Set custom attributes AFTER super().__init__()
        self.storage_options = storage_options or {}
        self.logger = logger
            

    def get(self, entry:int, *, dropna: bool = False):
        entry = self.loc[entry]
        if dropna:
            entry = entry.dropna()
        return JournalEntry(entry, storage_options=self.storage_options)
    
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
                entry = JournalEntry(row, storage_options=self.storage_options)
                th = None
                th = entry.read(thing, raw=raw, safe=safe)
                entries.append(th)
            except Exception as exc: 
                self.logger.silent(f"JournalFrame: EXCEPTION when reading {thing}: {row=}, {entry=}, {th=}:\nEXCEPTION: {exc}")
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

    
def gitrevision(*, log=Logger()):
    repopath = DBX_USE_WORK_REPO if DBX_USE_WORK_REPO is not None else DBX_GIT_REPO
    if repopath is not None:
        d_repo, project_repo = dbx_repos(repopath)
        
        def get_rev(path):
            if path is None:
                return None
            try:
                repo = git.Repo(path)
                # Dirty check is now performed in gitwrkreposetup() before
                # cloning, so we don't need to repeat it here.
                return repo.head.commit.hexsha
            except Exception as e:
                # git ≥ 2.35.2 raises ValueError / GitCommandError when the
                # work-repo temp dir is owned by a different process (common
                # in DDP subprocesses spawned by Lightning's
                # SubprocessScriptLauncher).  Fall back gracefully: the
                # revision is for audit purposes only.
                log.warning(
                    f"gitrevision: could not read HEAD from {path!r}: {e}"
                )
                return None

        dbx_rev = get_rev(d_repo)
        project_rev = get_rev(project_repo)
        
        if dbx_rev:
            revision = f"{dbx_rev}:{project_rev}"
        else:
            revision = project_rev
            
        log.detailed(f"Obtained git revision for git repo(s) {repopath}: {revision}")
    else:
        revision = None
    return revision


def gitclone(repopath, newpath):
    git.Repo.clone_from(repopath, newpath)
    return newpath


def gitcheckout(repopath, revision):
    repo = git.Repo(repopath)
    repo.git.checkout(revision)
    return repopath


gitwrkreposetup(reason="datablocks import")


class Datablock:
    """
    Declare topics via TOPICS::

        TOPICS = ['images', 'masks']                        # directory topics
        TOPICS = {'images': 'images.csv', 'masks': None}    # file and directory topics
        TOPICS = {'data': 'data.parquet'}                   # single file topic

    TOPICS must be a list or a dict.  Every topic has a name.

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
    VERBOSE_CONFIG = False

    @dataclass
    class Bid: #BlockId
        hash: str
        version: str
        revision: str
        dfn: dict
        kwargs: dict
        spec: dict
        quote: str
        repr: str
        norm: str
        hashstr: str
        supernorm: str
        superhashstr: str
        superhash: str
        anchor: str
        tag: str
        key: str
        keyby: str

        def deslash(self, attr):
            a = getattr(self, attr)
            if isinstance(a, str):
                aa = a.replace('\\', '')
            else:
                aa = a
            return aa

        def fields(self):
            return {f.name: f.type for f in fields(self)}

        def to_dict(self, *, deslash: bool = False):
            d = {f.name: getattr(self, f.name) for f in fields(self)}
            if deslash:
                for k, v in d.items():
                    if isinstance(v, str):
                        d[k] = v.replace('\\', '')
            return d

    @dataclass
    class CONFIG:
        class LazyLoader:
            def __init__(self, term):
                self.term = term
                self.value = None
            def __call__(self):
                if self.value is None:
                    if isinstance(self.term, str):
                        self.value = eval(self.term)
                    else:
                        # from_datablockable passes raw Python objects
                        self.value = self.term
                return self.value

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if isinstance(attr, Datablock.CONFIG.LazyLoader):
                return attr()
            return attr

    def __init__(
        self,
        *,
        url: str = None,
        spec: Optional[Union[str, dict]] = None,
        anchor: str = None,
        hash: str|None = None,
        tag: str|None = None,
        info: bool = None,
        verbose: bool = None,
        debug: bool = None,
        detailed: bool = None,
        capture_output: bool = False,
        revision: str = None,
        keyby: str = 'tag_version_hash',
        uuid16: bool = False,
        validate_cfg: bool = True,
        storage_options: dict = None,
        **kwargs,
    ):# Initialize early logger for __post_init__ if needed, though usually hash is needed
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
            'hash': hash,
            'tag': tag,
            'info': info,
            'verbose': verbose,
            'debug': debug,
            'detailed': detailed,
            'capture_output': capture_output,
            'revision': revision,
            'keyby': keyby,
            'uuid16': uuid16,
            'validate_cfg': validate_cfg,
            'storage_options': storage_options,
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

        # Explicit parameters
        self.url = state.get('url')
        # Resolve specline URLs (e.g. "$dbx.getenv('KEY')") to real paths.
        self._url_ = eval(self.url) if self.url is not None else None
        if self._url_ is None:
            self._url_ = os.environ.get('DBX_ROOT')
        if self._url_ is None:
            raise ValueError(f"No url for {self.__class__.__name__}: pass url= or set DBX_ROOT")
        self.storage_options = state.get('storage_options')
        if self.storage_options is None:
            self.storage_options = default_storage_options()
        self.fs, self.root = fsspec.url_to_fs(self._url_, **self.storage_options)
        self._spec_ = state.get('spec')
        if self._spec_ is None:
            self.spec = asdict(self.CONFIG())
        else:
            self.spec = self._spec_
        self._anchor_ = state.get('anchor')
        self._hash_ = state.get('hash')
        self._superhash_ = state.get('superhash')
        self._tag_ = state.get('tag')
        
        self._revision_ = state.get('revision')
        self.capture_output = bool(state.get('capture_output', False))
        self.keyby = state.get('keyby', 'taghash')
        if self.keyby not in (None, 'hash', 'superhash', 'norm', 'tag', 'taghash', 'tag_hash', 'version_hash', 'tag_version_hash', 'custom'):
            raise ValueError(f"keyby must be None, 'hash', 'superhash', 'norm', 'tag', 'taghash', 'tag_hash', 'version_hash', 'tag_version_hash', 'custom', got {self.keyby!r}")
        if self.keyby == 'tag' and self._tag_ is None:
            raise ValueError(
                f"keyby='tag' requires an explicit tag= argument, but none was provided for {self.__class__.__name__}"
            )
        self._uuid16_ = state.get('uuid16', False)
        self.validate_cfg = state.get('validate_cfg', True)
        

        
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
        self.__post_init__()
        self.log.detailed(f"======--------------> bid: {self.bid}")

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
        # Override: serialize the raw specline, not the resolved _url_.
        _state['url'] = self.url
        
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

    def validtopic(self, topic):
        path = self.path(topic)
        valid = self.validpath(path)
        self.log.detailed(f"{self.anchor}: topic {topic} valid: {valid}")
        return valid
    
    def validtopics(self, topics=None, *, reduce: bool = False):
        result = None
        if topics is None:
            topics = self.topics()
        if topics:
            results = {
                topic:
                self.validtopic(topic) for topic in topics
            }
            if reduce:
                result = all(list(results.values()))
            else:
                result = results
        else:
            result = True  # no topics → always valid
        return result

    def validpath(self, path):
        if path is None:
            return True
        elif isinstance(path, dict):
            return all([self.validpath(p) for p in path.values()])
        elif isinstance(path, list):
            return all([self.validpath(p) for p in path])
        if path is None or path.endswith("None"): #If topic filename ends with 'None', it is considered to be valid by default
            result = True
        elif isinstance(path, dict):
            result = all([self.validpath(p) for p in path.values()])
        else:
            result = self.fs.exists(path)
        self.log.detailed(f"{self.anchor}: path {path} valid: {result}") 
        return result
    
    def validpaths(self, topics=None, *, reduce: bool = False):
        result = None
        if topics is None:
            topics = self.topics()
        results = {
            topic: self.validpath(self.path(topic))
            for topic in topics
        }
        if reduce:
            result = all(list(results.values()))
        else:
            result = results
        return result
    
    def valid(self):
        return self.validtopics(reduce=True)
    
    def topics(self):
        """Return the list of topic names.

        For dict-TOPICS, returns the keys.  For list-TOPICS, returns the list.
        Returns an empty list when TOPICS is not defined.
        """
        if not hasattr(self, 'TOPICS'):
            return []
        if isinstance(self.TOPICS, dict):
            return list(self.TOPICS.keys())
        if isinstance(self.TOPICS, list):
            return list(self.TOPICS)
        return []

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

        Returns TOPICS when it is a dict, otherwise None.
        """
        if hasattr(self, 'TOPICS') and isinstance(self.TOPICS, dict):
            return self.TOPICS
        return None

    def build(self, *args, **kwargs):
        if self.capture_output:
            logpath = self._dbxanchorhashpathx('log', ext='log', ensure_dirpath=True)
            self.log.verbose(f"-------------------- Capturing stdout/stderr to {logpath} ------------------")

            # Write to a local temp file; upload to remote logpath at the end.
            _local_log = tempfile.NamedTemporaryFile(
                mode='w', suffix='.log', prefix='dbx_capture_', delete=False, encoding='utf-8',
            )
            _fd_capture = FDCapture(_local_log)
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
                    _fd_capture.close()
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
                _fd_capture.close()
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
                _fd_capture.close()
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
                _fd_capture.close()
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
        if self.validate_cfg:
            valid_cfg = self.valid_cfg()
            if not all(list(valid_cfg.values())):
                raise ValueError(f"Not all upstream Datablocks in cfg are valid: {valid_cfg=}")
        self._build_start_dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        self._write_journal_entry(event="build:start",)
        return self

    def __post_build__(self, *args, event="build:end", **kwargs):
        self._write_journal_entry(event=event,)
        return self
    
    def __build__(self, *args, **kwargs):
        return self

    def getall(self, *, path=None, root='.'):
        for topic in self.topics():
            self.get(topic, path=path, root=root)

    def get(self, topic, *, path=None, root='.'):
        """Download datablock files to a local directory.

        By default, files are downloaded to a local directory that mirrors
        the remote storage layout::

            {root}/{anchor}/{key}/[topic/][file]

        When *path* is provided, files are downloaded directly to *path*
        and *root* is ignored.

        Delegates to :meth:`__get__` which can be overridden by
        subclasses to customise the download behaviour.

        Parameters
        ----------
        topic : str, optional
            The topic to download.  When ``None``, downloads the
            default topic (single-topic blocks) or the top-level
            directory (multi-TOPICS blocks).
        path : str, optional
            Explicit local destination directory.  When provided, *root*
            is ignored.
        root : str, default ``'.'``
            Local root directory.  The destination is formed as
            ``{root}/{anchor}/{key}``.

        Returns
        -------
        self
        """
        if path is None:
            path = os.path.join(root, self.anchor, self.key) if self.key else os.path.join(root, self.anchor)
        return self.__get__(topic, path=path)

    def __get__(self, topic, *, path='.'):
        """Default implementation: copy files from ``self.path(topic)`` to *path* using ``self.fs``."""
        src = self.path(topic)
        if src is None:
            src = self.dirpath(topic)
        if src is None or not self.fs.exists(src):
            self.log.warning(f"get: no path for topic={topic!r}, nothing to download")
            return self
        os.makedirs(path, exist_ok=True)
        if self.fs.isdir(src):
            self.fs.get(src, path, recursive=True)
        else:
            self.fs.get(src, os.path.join(path, os.path.basename(src)))
        return self


    def note(self, message: str | None = None, event: str = 'note', *, inline: bool = False):
        """Write a journal entry with the given *event* and optional *message*.

        The journal parquet file is prepended with ``{event}-`` so it can
        be distinguished from regular journal entries, but it still
        lives under the ``journal/`` directory and therefore is read
        by :meth:`journal`.

        Parameters
        ----------
        message : str, optional
            If provided, recorded in the journal ``message`` field.
        event : str, default 'note'
            The event name recorded in the journal (e.g. ``'keep'``,
            ``'note'``).
        inline : bool, default False
            When ``True`` the *message* string is stored directly in the
            journal record.  When ``False`` the message is written to a
            separate text file and the journal stores the file path.

        Returns
        -------
        self
        """
        self._write_journal_entry(
            event=event,
            message=message,
            inline_message=inline,
            journal_prefix=f'{event}-',
        )
        return self


    def keep(self, message: str | None = None):
        """Write a journal entry with event='keep' and optional *message*.

        Equivalent to ``self.note(message, event='keep', inline=True)``.
        """
        return self.note(message, event='keep', inline=True)

    def leave_breadcrumbs(self):
        topics = self.topics()
        if not topics:
            raise NotImplementedError(
                f"{self.__class__.__name__}.leave_breadcrumbs() requires TOPICS"
            )
        for topic in topics:
            self.dirpath(topic, ensure=True)
            self.leave_breadcrumbs_at_path(self.path(topic))
        return self

    def _iter_cfg_blocks(self, exemptions_attr=None, skip_callback=None):
        """Yield (key, Datablock) pairs from self.cfg that are not in the given exemptions list."""
        exemptions = set(getattr(self, exemptions_attr, ())) if exemptions_attr else set()
        for s in self.spec.keys():
            if s in exemptions:
                if skip_callback:
                    skip_callback(s)
                continue
            c = getattr(self.cfg, s)
            if isinstance(c, Datablock):
                yield s, c

    def build_tree(self, *args, exclude_self: bool = False, deep: bool = False, **kwargs):
        self.log.verbose(f"Building tree for {self} with roots {self.spec.keys()}")
        def skip_cb(s):
            self.log.verbose(f"------------------------ SKIPPING SUBTREE at {s} (BUILD_TREE_EXEMPTIONS) --------")
        
        for s, c in self._iter_cfg_blocks('BUILD_TREE_EXEMPTIONS', skip_callback=skip_cb):
            if not deep and c.valid():
                self.log.verbose(f"------------------------ SKIPPING SUBTREE at {s}: already valid --------")
                continue
            self._write_journal_entry(event=f"build_tree:{s}:begin")
            self.log.verbose(f"------------------------ BUILDING SUBTREE at {s}: BEGIN --------------------------------")
            c.build_tree(*args, deep=deep, **kwargs)   
            self.log.verbose(f"------------------------ BUILDING SUBTREE at {s}: END --------------------------------")
            self._write_journal_entry(event=f"build_tree:{s}:end")
        if not exclude_self:
            self.build(*args, **kwargs)
        return self
    
    def valid_cfg(self, *, reduce=False):
        if not self.validate_cfg:
            return True if reduce else {}
        results = {s: c.valid() for s, c in self._iter_cfg_blocks('VALIDATE_CFG_EXEMPTIONS')}
        if reduce:
            return all(list(results.values())) if results else True
        else:
            return results

    def valid_tree(self):
        """Return a nested dictionary mapping cfg keys to their valid status and the valid status of their subtrees."""
        if not self.validate_cfg:
            return {}
        return {
            s: {'valid': c.valid(), 'tree': c.valid_tree()}
            for s, c in self._iter_cfg_blocks('VALIDATE_CFG_EXEMPTIONS')
        }
    
    def read(self, topic):
        topics = self.topics()
        if topic not in topics:
            raise ValueError(f"Topic {repr(topic)} not in {topics}")
        return self.__read__(topic)
    
    def __read__(self, topic):
        raise NotImplementedError()
    
    def UNSAFE_clear(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
        if not UNSAFE_allowed("UNSAFE_clear", OVERRIDE=OVERRIDE):
            return self
        
        def clear_path(path, *, recursive=False, throw=False):
            if path is None:
                return
            self.log.verbose(f"removing {path}")
            try:
                if path.startswith("gs://"):
                    """
                    Circumvent bugs in fsspec.
                    """
                    from google.cloud import storage

                    client = storage.Client()
                    bits = path.removeprefix("gs://").split("/")
                    bucket_name = bits[0]
                    prefix = "/".join(bits[1:])
                    bucket = client.get_bucket(bucket_name)
                    blobs = bucket.list_blobs(prefix=prefix)
                    for blob in blobs:
                        blob.delete()
                else:
                    self.fs.rm(path, recursive=recursive)
            except Exception as e:
                self.log.warning(f"Error when trying to remove {path}")
                self.log.warning(f"EXCEPTION: {e}")
                if throw:
                    raise (e)
        if len(topics) == 0:
            all_topics = self.topics()
            if all_topics:
                for topic in all_topics:
                    if clear_dirpath:
                        clear_path(self.dirpath(topic), recursive=True)
                    else:
                        is_dir = self._topics_is_list or (self._topicfiles is not None and self._topicfiles.get(topic) is None)
                        clear_path(self.path(topic), recursive=is_dir)
            self._write_journal_entry(event="UNSAFE_clear")
        else:
            for topic in topics:
                if clear_dirpath:
                    clear_path(self.dirpath(topic), recursive=True)
                else:
                    is_dir = self._topics_is_list or (self._topicfiles is not None and self._topicfiles.get(topic) is None)
                    clear_path(self.path(topic), recursive=is_dir)
            self._write_journal_entry(event=f"UNSAFE_clear:{[topics]}")
        return self
    
    def UNSAFE_rename_from(self, anchor, *, OVERRIDE: bool = False, overwrite: bool = False, topicpaths=None, validate: bool = True, copy_dirpath: bool = False):
        if not UNSAFE_allowed("UNSAFE_rename_from", OVERRIDE=OVERRIDE):
            return self
        anchorkeypath = self._anchorkeypath(anchor)
        self._write_journal_entry(event="UNSAFE_rename_from:BEGIN", message=anchor, inline_message=True)
        result = self.UNSAFE_copy_from(anchorkeypath, overwrite=overwrite, topicpaths=topicpaths, validate=validate, copy_dirpath=copy_dirpath)
        self._write_journal_entry(event="UNSAFE_rename_from:END", message=anchor, inline_message=True)
        return result

    def UNSAFE_copy_from(self, anchorkeypath, *, overwrite: bool = False, topicpaths=None, validate: bool = True, copy_dirpath: bool = False):
        """Copy topic data from an external directory into this Datablock.

        Parameters
        ----------
        anchorkeypath : str
            Filesystem path to the source anchor+key directory containing
            the topic subdirectories (e.g. ``ckpts/``, ``logs/``).
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
        copy_dirpath : bool, default False
            If False (default), copies individual topic files via
            ``self.path(topic)``.  If True, copies entire topic
            directories via ``self.dirpath(topic)`` recursively.
        """
        def fscopy(*, src_path, dst_path, recursive: bool = False):
            # fsspec does not implement .copy, so use put/get or temporary directory
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

        def copy_topic_file(topic):
            """Copy the individual .path(topic) file."""
            dst_path = self.path(topic)
            if topicpaths is not None:
                _src_path = topicpaths[topic]
            else:
                _src_path = os.path.join(topic, self._topicfiles[topic])
            if dst_path is not None:
                src_path = os.path.join(anchorkeypath, _src_path)
                self.log.detailed(f"Copying file {src_path} to {dst_path}")
                fscopy(src_path=src_path, dst_path=dst_path, recursive=False)

        def copy_topic_dir(topic):
            """Copy the entire .dirpath(topic) directory."""
            dst_path = self.dirpath(topic)
            if topicpaths is not None:
                _src_path = topicpaths[topic]
            else:
                _src_path = topic
            src_path = os.path.join(anchorkeypath, _src_path)
            src_fs, _ = self._url_to_fs(src_path)
            if src_fs.exists(src_path):
                self.log.detailed(f"Copying directory {src_path} to {dst_path}")
                fscopy(src_path=src_path, dst_path=dst_path, recursive=True)

        if not overwrite:
            assert not self.valid(), f"Attempting to overwrite a valid Datablock {self}. Missing 'overwrite' argument?"
        fs, _ = self._url_to_fs(anchorkeypath)
        assert fs.isdir(anchorkeypath), f"Nonexistent hashpath {anchorkeypath}"
        self.log.verbose(f"Copying files from {anchorkeypath}: BEGIN")
        self._write_journal_entry(event="UNSAFE_copy_from:BEGIN", message=anchorkeypath, inline_message=True)
        try:
            topics = self.topics()
            if not topics:
                raise NotImplementedError(
                    f"{self.__class__.__name__}.UNSAFE_copy_from() requires TOPICS"
                )
            for topic in tqdm.tqdm(topics, desc="UNSAFE_copy_from", unit="topic"):
                if copy_dirpath:
                    copy_topic_dir(topic)
                else:
                    copy_topic_file(topic)
        
            self.log.verbose(f"Copying files from {anchorkeypath}: END")
            self._write_journal_entry(event="UNSAFE_copy_from:END", message=anchorkeypath, inline_message=True)
            if validate:
                assert self.valid(), f"Invalid Datablock after copy: {self}"
        except Exception as e:
            self.log.error(f"UNSAFE_copy_from: Error when trying to copy files from {anchorkeypath}")
            self.log.error(f"EXCEPTION: {e}")
            self._write_journal_entry(event="UNSAFE_copy_from:ERROR", message=anchorkeypath, inline_message=True)
            raise e
        return self

    # ALIAS
    def UNSAFE_pull(self, *args, **kwargs):
        return UNSAFE_copy_from(*args, **kwargs)

    def _spec_to_cfg(self, spec):
        config = self.CONFIG(**spec)
        replacements = {}
        for field in fields(config):
            term = getattr(config, field.name)
            if issubclass(self.CONFIG, Datablock.CONFIG):
                getter = Datablock.CONFIG.LazyLoader(term)
            else:
                getter = eval(term)
            replacements[field.name] = getter
        config = replace(config, **replacements)
        self.log.detailed(f"Made {config=} from {spec=}")
        return config

    def leave_breadcrumbs_at_path(self, path):
        with self.fs.open(path, "w") as f:
            f.write("")
    
    #IDS: BEGIN
    #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
    # computed using the older version of these methods
    @property
    def bid(self):
        return self.Bid(
            hash=self.hash,
            version=self.version,
            revision=self.revision,
            kwargs=self.kwargs,
            spec=self.spec,
            dfn=self.dfn,
            quote=self.quote(deslash=True),
            repr=self.__repr__(deslash=True),
            norm=self.norm(deslash=True),
            hashstr=self.hashstr,
            supernorm=self.supernorm(deslash=True),
            superhashstr=self.superhashstr,
            superhash=self.superhash,
            anchor=self.anchor,
            tag=self.tag,
            key=self.key,
            keyby=self.keyby,
        )
    
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
    def uuid(self):
        if not hasattr(self, '_uuid'):
            self._uuid = uuid.uuid4().hex[:16] if getattr(self, '_uuid16_', False) else str(uuid.uuid4())
        return self._uuid
    
    @property
    def revision(self):
        if not hasattr(self, '_revision'):
            self.log.detailed(f"--------------> COMPUTING revision")
            if self._revision_ is None:
                self.log.detailed(f"--------------> self._revision_ is None")
                gitrepo = DBX_USE_WORK_REPO if DBX_USE_WORK_REPO is not None else DBX_GIT_REPO
                self._revision = gitrevision(log=self.log) if gitrepo is not None else None
                self.log.detailed(f"--------------> self._revision_: from gitrevision()")
            else:
                self.log.detailed(f"--------------> Using {self._revision_=}")
                self._revision = self._revision_
        return self._revision

    def __expand_spec__(self, expansion='repr'):
        """
            . expansion: 'repr'|'quote'|'norm'
                . specline:      str starting with '@', '$' or '#'
                . datablock: Datablock object
                . obj:       object
            'repr':
                . FULL reduction
                    |obj:    repr(obj)
            'norm':
                . DATABLOCK reduction
                    |datablock: datablock.norm()
                    |specline:      repr(specline)
                    |obj:       repr(obj)
            'quote':
                . UNREDUCED spec:
                    |specline:      repr(specline)
                    |datablock: datablock.quote()
                    |obj:       repr(obj)  
        """
        keys = sorted([field.name for field in self.CONFIG.__dataclass_fields__.values()])
        spec = {k: self.spec[k] if k in self.spec else getattr(self.cfg, k) for k in keys}
        _spec_ = {}
        if expansion == 'repr':
            #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
            # computed using the older version of these methods
            for k, v in spec.items():
                value = getattr(self.cfg, k)
                _spec_[k] = repr(value)
        elif expansion == 'norm':
            for k, v in spec.items():
                value = getattr(self.cfg, k)
                if isinstance(value, Datablock):
                    _spec_[k] = value.norm()
                elif self.is_specline(v):
                    _spec_[k] = v
                elif isinstance(value, str):
                    _spec_[k] = value
                else:
                    _spec_[k] = repr(value)
        elif expansion == 'quote':
            for k, v in spec.items():
                value = getattr(self.cfg, k)
                if self.is_specline(v):
                    _spec_[k] = v
                elif isinstance(value, Datablock):
                    _spec_[k] = value.quote()
                elif isinstance(value, str):
                    _spec_[k] = value
                else:
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
        if self._hash_ is not None:
            rootkwargs['hash'] = self._hash_
        return rootkwargs
    
    @functools.cached_property
    def _tailkwargs_(self):
        state = self.__getstate__()
        tailkwargs = {
            k: v
            for k, v in state.items()
            if k not in ['url', 'anchor', 'hash', 'spec']          
        }
        self.log.detailed(f"{self.anchor}: _tailkwargs_: {tailkwargs=}")
        return tailkwargs
    
    def __repr_from_kwargs__(self, kwargs, anchor='anchor'):
        kwargstrs = [f"{k}={v}" for k, v in kwargs.items()]
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
    
    def quote(self, *, deslash: bool = False):
        quoted_spec = self.__expand_spec__('quote')
        def cite(x):
            return repr(x) if isinstance(x, str) else x
        kwargstrs = [f"{k}={cite(v)}" for k, v in {**self._rootkwargs_, **{'spec': quoted_spec}, **self._tailkwargs_}.items()]
        kwargsrepr = ', '.join(kwargstrs)
        quote = f"${self.fqcn}({kwargsrepr})"
        if deslash:
            quote = quote.replace('\\', '')
        self.log.detailed(f"quote: ------------> {quoted_spec=}")
        self.log.detailed(f"quote: ------------> {quote=}")
        return quote

    def norm(self, *, deslash: bool = False):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        norm_spec = self.__expand_spec__('norm')
        norm = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': norm_spec},
        }, anchor=None)
        if deslash:
            norm = norm.replace('\\', '')
        self.log.detailed(f"norm: ------------> {norm_spec=}")
        self.log.detailed(f"norm: ------------>{norm=}")
        return norm

    def supernorm(self, *, deslash: bool = False):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        supernorm_spec = self.__expand_spec__('norm')
        supernorm = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': supernorm_spec},
        }, anchor='fqcn')
        if deslash:
            supernorm = supernorm.replace('\\', '')
        self.log.detailed(f"supernorm: ------------> {supernorm_spec=}")
        self.log.detailed(f"supernorm: ------------>{supernorm=}")
        return supernorm

    def __repr__(self, *, deslash: bool = True):
        repr_spec = self.__expand_spec__('repr')
        r = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': repr_spec},
            **self._tailkwargs_,
        }, anchor='fqcn')
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
    def cfg(self):
        log_fn = self.log.verbose if getattr(self, 'VERBOSE_CONFIG', False) else self.log.detailed
        log_fn(f"Forming cfg from spec: BEGIN")
        cfg = self._spec_to_cfg(self.spec)
        log_fn(f"Forming cfg from spec: END")
        return cfg

    @property
    def config(self):
        return self.cfg
    
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
    def hashstr(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        if self._topicfiles is not None:
            topics = [f"topic:{topic}={file}" for topic, file in self._topicfiles.items()]
        elif hasattr(self, "TOPICS") and isinstance(self.TOPICS, list):
            topics = [f"topic:{topic}" for topic in self.TOPICS]
        else:
            topics = ["topics:None"]
        hashstr = os.path.join(
            self.norm(),
            f"version={self.version}",
            *topics,
        )
        return hashstr

    @property
    def superhashstr(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        if self._topicfiles is not None:
            topics = [f"topic:{topic}={file}" for topic, file in self._topicfiles.items()]
        elif hasattr(self, "TOPICS") and isinstance(self.TOPICS, list):
            topics = [f"topic:{topic}" for topic in self.TOPICS]
        else:
            topics = ["topics:None"]
        superhashstr = os.path.join(
            self.supernorm(),
            f"version={self.version}",
            *topics,
        )
        return superhashstr

    @property
    def hash(self): 
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hash
        # computed with the older code.
        if not hasattr(self, '_hash'): 
            if self._hash_ is not None:
                self._hash = self._hash_
            else:
                sha = hashlib.sha256()
                sha.update(self.hashstr.encode())
                self._hash = sha.hexdigest()
                self.log.detailed(f"hash: ---------===---------\u003e {self.hashstr=} ---\u003e hash: {self._hash}")
        return self._hash

    @property
    def superhash(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hash
        # computed with the older code.
        if not hasattr(self, '_superhash'): 
            if self._superhash_ is not None:
                self._superhash = self._superhash_
            else:
                sha = hashlib.sha256()
                sha.update(self.superhashstr.encode())
                self._superhash = sha.hexdigest()[:8]
                self.log.detailed(f"superhash: ---------===---------\u003e {self.superhashstr=} ---\u003e superhash: {self._superhash}")
        return self._superhash

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
        elif self.keyby == 'superhash':
            key = self.superhash
        elif self.keyby == 'norm':
            key = self.norm()
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
        elif self.keyby == 'tag_version_hash':
            parts = []
            if self._tag_ is not None:
                parts.append(self.tag)
            if self.version is not None:
                parts.append(f"version={self.version}")
            if parts:
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
    def path(
        self,
        topic,
        *,
        ensure_dirpath: bool = False,
    ):
        """Return the path for *topic*.

        For dict-TOPICS with a string value, returns ``dirpath/filename``.
        For dict-TOPICS with ``None`` value or list-TOPICS, returns the
        directory path (same as ``dirpath(topic)``).
        """

        dirpath = self.dirpath(topic)
        if ensure_dirpath and dirpath is not None:
            ensure_path(dirpath, storage_options=self.storage_options)

        if self._topics_is_list:
            # List-TOPICS: the topic IS a directory
            return dirpath

        topicfile = self._topicfiles[topic]
        if topicfile is None:
            # Dict-TOPICS with None value: directory-only topic
            return dirpath
        elif isinstance(topicfile, str):
            path = os.path.join(dirpath, topicfile)
        else:
            path = None
        self.log.detailed(f"{self.anchor}: path: {path}")
        return path

    def ls(self, topic, *, detail=False):
        """List the contents at ``.path(topic)`` using *fsspec*.

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
        p = self.path(topic)
        if p is None:
            return []
        if not self.fs.exists(p):
            return []
        # Determine if the path is a directory topic
        path_is_dir = (topic is not None and (
            self._topics_is_list or
            (self._topicfiles is not None and self._topicfiles.get(topic) is None)
        ))
        # Only fall back to the parent directory when the path genuinely
        # points to a file (dict-TOPICS with a non-None filename).
        # Skip this for directory-only topics where fs.isfile() may
        # return a false positive (e.g. Azure virtual directories).
        if not path_is_dir and self.fs.isfile(p):
            p = os.path.dirname(p)
        # Azure adlfs virtual directories: fs.ls("dir") may return only
        # the directory marker itself; a trailing "/" forces a prefix
        # listing that returns the directory's *contents*.
        if path_is_dir and not p.endswith('/'):
            p = p + '/'
        results = self.fs.ls(p, detail=detail)
        if detail:
            for entry in results:
                entry['name'] = fs_full_path(self.fs, entry['name'])
            return results
        return [fs_full_path(self.fs, x) for x in results]

    
    def dirpath(
        self,
        topic,
        *,
        ensure: bool = False,
        list: bool = False,
    ):  
        anchorkeypath = self.anchorkeypath
        dirpath = os.path.join(anchorkeypath, topic)
        if ensure:
            self.fs.makedirs(dirpath, exist_ok=True)
        if list:
            # Trailing "/" ensures Azure adlfs lists directory *contents*
            # rather than returning the virtual-directory marker itself.
            _lspath = dirpath if dirpath.endswith('/') else dirpath + '/'
            return self.fs.ls(_lspath)
        return dirpath

    def paths(self):
        topics = self.topics()
        paths = {topic: self.path(topic) for topic in topics}
        return paths

    def anchorpath(self):
        return self._anchorpath()

    @property
    def anchorkey(self):
        return self._anchorkey()

    @property
    def anchorkeypath(self):
        return self._anchorkeypath()

    def _anchorpath(self, anchor=None):
        anchor = anchor or self.anchor
        return fs_full_path(self.fs, os.path.join(self.root, anchor))

    def _anchorkey(self, anchor=None):
        anchor = anchor or self.anchor
        return os.path.join(anchor, self.key) if self.key else anchor

    def _anchorkeypath(self, anchor=None):
        anchorkey = self._anchorkey(anchor)
        bare = os.path.join(self.root, anchorkey) if anchorkey else self.root
        return fs_full_path(self.fs, bare)
    
    @staticmethod
    def _dbxanchorpathx(url, anchor, x, *, fqcn, ensure: bool = False, storage_options=None):
        """Return {url}/anchor/.dbx/fqcn/x — the anchor-level directory for artefact *x*."""
        fs, root = fsspec.url_to_fs(url, **(storage_options or {}))
        _dbxanchorpathx = fs_full_path(fs, os.path.join(root, anchor, ".dbx", fqcn, x))
        if ensure:
            fs.makedirs(_dbxanchorpathx, exist_ok=True)
        return _dbxanchorpathx

    def _dbxanchorhashpathx(self, x, ext=None, *, ensure_dirpath: bool = True, filename_prefix: str = ''):
        _dbxanchorpathx = Datablock._dbxanchorpathx(self._url_, self.anchor, x, fqcn=self.fqcn, storage_options=self.storage_options)
        _dbxanchorhashpathx = os.path.join(_dbxanchorpathx, self.hash)
        if ensure_dirpath:
            self.fs.makedirs(_dbxanchorhashpathx, exist_ok=True)
        if ext is None:
            ext = x
        xpath = os.path.join(_dbxanchorhashpathx, f'{filename_prefix}{self.fqcn}-{x}-{self.hash}-{self.dt}.{ext}')
        return xpath

    def _dbxjournalinstancepath(self, *, ensure_dirpath: bool = False, filename_prefix: str = ''):
        """
        Return /root/anchor/.dbx/journal/hash/{fqcn}-{dt}.journal."""
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

    def _write_journal_entry(self, event:str, *, message: str = None, inline_message: bool = False, journal_prefix: str = ''):
        self._write_journal_dict('spec', self.spec)
        self._write_journal_dict('dfn', self.dfn)
        self._write_journal_dict('kwargs', self.kwargs)
        self._write_str('quote', self.quote())
        self._write_str('repr', self.__repr__())
        self._write_str('norm', self.norm())
        self._write_str('supernorm', self.supernorm())
        self._write_str('hashstr', self.hashstr)
        if message is not None and not inline_message:
            self._write_str('message', message)
        #
        dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')

        spec_path = self._dbxanchorhashpathx('spec', 'yaml')
        dfn_path = self._dbxanchorhashpathx('dfn', 'yaml')
        kwargs_path = self._dbxanchorhashpathx('kwargs', 'yaml')
        quote_path = self._dbxanchorhashpathx('quote', 'txt')
        norm_path = self._dbxanchorhashpathx('norm', 'txt')
        repr_path = self._dbxanchorhashpathx('repr', 'txt')
        hashstr_path = self._dbxanchorhashpathx('hashstr', 'txt')
        supernorm_path = self._dbxanchorhashpathx('supernorm', 'txt')
        superhashstr_path = self._dbxanchorhashpathx('superhashstr', 'txt')
        if message is not None and not inline_message:
            message_path = self._dbxanchorhashpathx('message', 'txt')
            message = message_path
        else:
            message_path = None
        #
        logpath = self._dbxanchorhashpathx('log', ensure_dirpath=True)
        if logpath is not None:
            has_log = self.fs.exists(logpath)
        else:
            has_log = False
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
                                         'superhash': self.superhash,
                                         'keyby': self.keyby,
                                         'key': self.key,
                                         'anchorkeypath': self.anchorkeypath,
                                         'uuid': self.uuid,
                                         'tag': self.tag, 
                                         'log': logpath if has_log else None,
                                         'event': event,
                                         'spec': spec_path,
                                         'dfn': dfn_path,
                                         'kwargs': kwargs_path,
                                         'quote': quote_path,
                                         'norm': norm_path,
                                         'supernorm': supernorm_path,
                                         'repr': repr_path,
                                         'hashstr': hashstr_path,
                                         'superhashstr': superhashstr_path,
                                         'message': message,
                                         'gitrepo': DBX_GIT_REPO,
                                         'wrkrepo': DBX_USE_WORK_REPO,
        }])
        with self.fs.open(journal_path, 'wb') as f:
            df.to_parquet(f)
        
        tagstr = f"with tag {repr(self.tag)} " if self.tag is not None else ""
        self.log.debug(f"WROTE JOURNAL entry for event {repr(event)} {tagstr}"
                         f"to journal_path {journal_path}")

    @staticmethod
    def Journal(anchor, loc: int = None, *, iloc: int = None, url=None, storage_options=None, **filter_kwargs):
        if loc is not None and iloc is not None:
            raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
        if url is None:
            url = os.environ.get('DBX_ROOT')
        if storage_options is None:
            storage_options = default_storage_options()

        fs, root = fsspec.url_to_fs(url, **(storage_options or {}))
        log = Logger()

        journaldirpath = fs_full_path(fs, os.path.join(root, anchor, ".dbx"))

        if not fs.exists(journaldirpath):
            raise FileNotFoundError(
                f"Journal directory not found for {anchor!r}: {journaldirpath}\n"
                f"Check that the class name / anchor and url are correct."
            )

        files = fs.glob(os.path.join(journaldirpath, '**/journal/**/*.parquet'))
        parquet_files = files

        log.detailed(f"READING JOURNAL: from {journaldirpath=}, files: {parquet_files}")
        if len(parquet_files) > 0:
            dfs = []
            for file in parquet_files:
                try:
                    with fs.open(file, 'rb') as f:
                        _df = pd.read_parquet(f)
                except Exception as e:
                    log.warning(f"Skipping unreadable journal file {file}: {e}")
                    continue
                if 'revision' not in _df.columns:
                    _df = _df.rename(columns={'version': 'revision',})
                # Backward compat: rename legacy 'context' column to 'message'
                if 'context' in _df.columns and 'message' not in _df.columns:
                    _df = _df.rename(columns={'context': 'message'})
                # Backward compat: rename legacy 'build_datetime' to 'build:end:datetime'
                if 'build_datetime' in _df.columns and 'build:end:datetime' not in _df.columns:
                    _df = _df.rename(columns={'build_datetime': 'build:end:datetime'})

                dfs.append(_df)
            if len(dfs) > 0:
                df = pd.concat(dfs)
                leading = ['hash'] + (['uuid'] if 'uuid' in df.columns else []) + ['datetime']
                columns = leading + [c for c in df.columns if c not in set(leading + ['event'])] + ['event']
                df = df.sort_values('datetime', ascending=False)[columns].reset_index(drop=True)
                df = df.rename(columns={'build_log': 'log'})
            else:
                df = None
        else:
            df = None
        journal = JournalFrame(df, storage_options=storage_options, **filter_kwargs)
        if loc is not None:
            result = JournalEntry(journal.loc[loc].dropna(), storage_options=storage_options)
        elif iloc is not None:
            result = JournalEntry(journal.iloc[iloc].dropna(), storage_options=storage_options)
        else:
            result = journal
        return result

    def journal(self, loc: int = None, *, iloc: int = None, **filter_kwargs):
        if loc is not None and iloc is not None:
            raise ValueError("Specify at most one of 'loc' and 'iloc', not both.")
        return self.Journal(self.anchor, loc=loc, iloc=iloc, url=self._url_, storage_options=self.storage_options, **filter_kwargs)

    def lastbuilt(self):
        """Return the most recent 'build:end' JournalEntry, or None."""
        j = self.journal(event='build:end')
        if len(j) == 0:
            return None
        return j.get(0, dropna=True)

    def running(self):
        """Return the latest 'build:start' JournalEntry with no matching 'build:end', or None."""
        j = self.journal()
        if len(j) == 0:
            return None
        started = set(j[j['event'] == 'build:start']['hash'])
        ended = set(j[j['event'] == 'build:end']['hash'])
        running_hashes = started - ended
        if not running_hashes:
            return None
        running_entries = j[(j['event'] == 'build:start') & (j['hash'].isin(running_hashes))]
        return JournalEntry(running_entries.iloc[0].dropna(), storage_options=self.storage_options)

    #JOURNAL: END
    


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
            class CONFIG(Datablock.CONFIG):
                path: str = None
                block_size: int = 100

            def blocks(self):
                n = self._total_items()
                return [
                    MyBlock(url=self.url, spec=dict(path=self.cfg.path, idx=i))
                    for i in range(math.ceil(n / self.cfg.block_size))
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
                f"Unknown parallelization {self.parallelization!r}. "
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
            s = self.__block__(idx)
            s.keyby = self.keyby
            self._blocks_[idx] = s
        return self._blocks_[idx]

    def blocks(self) -> list:
        """Return all blocks, forming them via :meth:`block` if needed."""
        n = self.n_blocks
        indices = tqdm.tqdm(range(n), desc=f"Forming {n} blocks") if n > 100 else range(n)
        return [self.block(idx) for idx in indices]

    def valid_blocks(self) -> list[bool]:
        """Return a list of booleans, one per block, indicating validity."""
        return [s.valid() for s in self.blocks()]

    # -- Default build logic ------------------------------------------------------

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
        executor_kwargs = dict(
            n_workers=self.n_workers,
            tag=f"EXECUTING {len(callables)} callables [{self.__class__.__name__}]",
        )
        if hasattr(self, 'worker_done_timeout_sec') and self.worker_done_timeout_sec is not None:
            executor_kwargs['worker_done_timeout_sec'] = self.worker_done_timeout_sec
        if hasattr(self, 'shuffle_callables') and self.shuffle_callables:
            executor_kwargs['shuffle_callables'] = self.shuffle_callables
        if (hasattr(self, 'multiprocessing_start_method')
                and self.multiprocessing_start_method is not None
                and issubclass(self.executor_cls, MultiprocessingCallableExecutor)):
            executor_kwargs['start_method'] = self.multiprocessing_start_method
        # Torch executors require a 'devices' parameter.
        if getattr(self, 'devices', None) is not None:
            executor_kwargs['devices'] = self.devices
        if getattr(self, 'work_stealing', False):
            executor_kwargs['work_stealing'] = True
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

    def UNSAFE_clear_blocks(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
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
        """
        if not UNSAFE_allowed("UNSAFE_clear_blocks", OVERRIDE=OVERRIDE):
            return self

        block_list = self.blocks()
        self.log.info(
            f"UNSAFE_clear_blocks: clearing {len(block_list)} blocks, "
            f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}"
        )
        self._write_journal_entry(event="UNSAFE_clear_blocks:begin")

        tag = f"CLEARING {len(block_list)} blocks [{self.__class__.__name__}, n_workers={self.n_workers}]"
        executor_kwargs = dict(n_workers=self.n_workers, tag=tag)
        if (hasattr(self, 'multiprocessing_start_method')
                and self.multiprocessing_start_method is not None
                and (self.parallelization or '').lower() in ('multiprocessing', 'torch_multiprocessing')):
            executor_kwargs['start_method'] = self.multiprocessing_start_method
        executor = callable_executor(self.parallelization, **executor_kwargs)

        callables = [functools.partial(_clear_block_callable, blk, topics, clear_dirpath) for blk in block_list]
        executor.exec_callables(callables)

        self.log.info(f"UNSAFE_clear_blocks complete: {self.__class__.__name__}")
        self._write_journal_entry(event="UNSAFE_clear_blocks:end")
        return self

    def UNSAFE_rename_blocks_from(self, blockanchor, *, OVERRIDE: bool = False, overwrite: bool = False, topicpaths=None, validate: bool = True, copy_dirpath: bool = False):
        if not UNSAFE_allowed("UNSAFE_rename_blocks_from", OVERRIDE=OVERRIDE):
            return self

        block_list = self.blocks()
        self.log.info(
            f"UNSAFE_rename_blocks_from: renaming {len(block_list)} blocks, "
            f"executor={self.executor_cls.__name__}, n_workers={self.n_workers}"
        )
        self._write_journal_entry(event="UNSAFE_rename_blocks_from:begin", message=blockanchor, inline_message=True)

        tag = f"RENAMING {len(block_list)} blocks [{self.__class__.__name__}, n_workers={self.n_workers}]"
        executor_kwargs = dict(n_workers=self.n_workers, tag=tag)
        if (hasattr(self, 'multiprocessing_start_method')
                and self.multiprocessing_start_method is not None
                and (self.parallelization or '').lower() in ('multiprocessing', 'torch_multiprocessing')):
            executor_kwargs['start_method'] = self.multiprocessing_start_method
        executor = callable_executor(self.parallelization, **executor_kwargs)

        callables = [functools.partial(_rename_block_from_callable, blk, blockanchor, overwrite, topicpaths, validate, copy_dirpath) for blk in block_list]
        executor.exec_callables(callables)

        self.log.info(f"UNSAFE_rename_blocks_from complete: {self.__class__.__name__}")
        self._write_journal_entry(event="UNSAFE_rename_blocks_from:end", message=blockanchor, inline_message=True)
        return self


def _clear_block_callable(block, topics, clear_dirpath):
    """Module-level callable for UNSAFE_clear_blocks (must be picklable)."""
    block.UNSAFE_clear(*topics, OVERRIDE=True, clear_dirpath=clear_dirpath)
    return block

def _rename_block_from_callable(block, blockanchor, overwrite=False, topicpaths=None, validate=True, copy_dirpath=False):
    """Module-level callable for UNSAFE_rename_blocks_from (must be picklable)."""
    block.UNSAFE_rename_from(blockanchor, OVERRIDE=True, overwrite=overwrite, topicpaths=topicpaths, validate=validate, copy_dirpath=copy_dirpath)
    return block


def quotefn(fn, *args, tag="$", **kwargs):
    log = Logger()
    argstrs = [quote(arg) for arg in args]
    kwargstrs = [f"{k}={quote(v)}" for k, v in kwargs.items()]
    argkwargstr = ','.join(argstrs+kwargstrs)
    _quote = f"{tag}{fn}({argkwargstr})"
    log.detailed(f"Quoted {fn=}, {args=}, {kwargs=} to {repr(_quote)}")
    return _quote


def quote(obj, *args, tag="$", **kwargs):
    log = Logger()
    if not callable(obj):
        assert len(args) == 0, f"Nonempty args for a noncallable obj: {args}"
        assert len(kwargs) == 0, f"Nonempty kwargs for a noncallable obj: {kwargs}"
        if isinstance(obj, Datablock):
            _quote = obj.quote()
        elif isinstance(obj, str):
            _quote = repr(obj)
        else:
            _quote = obj
        log.detailed(f"===============> Quoted {obj=} to {repr(_quote)}")
    else:
        func = obj
        fn = f"{func.__module__}.{func.__qualname__}"
        _quote = quotefn(fn, *args, tag=tag, **kwargs)
    return _quote


def _build_block_with_to(block, *args, **kwargs):
    """Build helper for TorchXXX builders: the callable IS the block itself.

    The callable must have ``.to(device)`` — the TorchXXXCallableExecutor
    validates this and calls ``block.to(device)`` before invoking, then
    ``block.to('cpu')`` afterwards.  We implement ``__call__`` via this
    partial, and ``.to()`` is already on the block.
    """
    return block.build(*args, **kwargs)


class _TorchBlockCallable_:
    """Thin wrapper that makes a Datablock usable as a TorchXXX callable.

    The executor calls ``callable.to(device)(...).to('cpu')``, so we need
    ``.to()`` and ``__call__()`` on the same object.  This wrapper delegates
    both to the underlying block.
    """
    def __init__(self, block):
        if not hasattr(block, 'to') or not callable(getattr(block, 'to')):
            raise TypeError(
                f"{type(block).__name__} does not implement .to(device). "
                f"Datablocks used with TorchMultithreadingDatablocksBuilder / "
                f"TorchMultiprocessingDatablocksBuilder must define a .to() method."
            )
        self.block = block

    def to(self, device):
        self.block.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.block.build(*args, **kwargs)
        return self.block


class TorchMultithreadingDatablocksBuilder:
    """Builds Datablocks concurrently using threads with per-device placement.

    Delegates to :class:`TorchMultithreadingCallableExecutor`.
    Each block must implement ``.to(device)``.
    """

    def __init__(self, *, devices: list[str] = 'cuda', log: Logger = Logger()):
        if isinstance(devices, str):
            devices = [devices]
        self.devices = devices
        self.log = log
        self._executor = TorchMultithreadingCallableExecutor(
            devices=devices, log=log,
        )

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [_TorchBlockCallable_(block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks


class TorchMultiprocessingDatablocksBuilder:
    """Builds Datablocks concurrently using processes with per-device placement.

    Delegates to :class:`TorchMultiprocessingCallableExecutor`.
    Each block must implement ``.to(device)``.
    """

    def __init__(self, *, devices: list[str] = None, log: Logger = Logger()):
        if isinstance(devices, str):
            devices = [devices]
        self.devices = devices
        self.log = log
        self._executor = TorchMultiprocessingCallableExecutor(
            devices=devices, log=log,
        )

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [_TorchBlockCallable_(block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks



def _build_block(block, *args, **kwargs):
    return block.build(*args, **kwargs)

class MultithreadingDatablocksBuilder:
    """Builds Datablocks concurrently using threads, via MultithreadingCallableExecutor."""

    def __init__(self, *, n_workers: int = 1, batch_size: int = None, tag: str = "", log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log
        self._executor = MultithreadingCallableExecutor(n_workers=n_workers, batch_size=batch_size, tag=tag, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [functools.partial(_build_block, block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks


class MultiprocessingDatablocksBuilder:
    """Builds Datablocks concurrently using processes, via MultiprocessingCallableExecutor."""

    def __init__(self, *, n_workers: int = 1, batch_size: int = None, tag: str = "", log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log
        self._executor = MultiprocessingCallableExecutor(n_workers=n_workers, batch_size=batch_size, tag=tag, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [functools.partial(_build_block, block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks


class RayDatablocksBuilder:
    def __init__(self, *, n_workers: int = 1, batch_size: int = None, tag: str = "", revision=None, conda=None, log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log
        workers = [remote(revision=revision, conda=conda) for _ in range(n_workers)]
        self.executor = RayCallableExecutor(workers=workers, batch_size=batch_size, tag=tag, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        if len(blocks) > 0:
            callables = [functools.partial(_build_block, block) for block in blocks]
            results = self.executor.execute(callables, *ctx_args, **ctx_kwargs)
            
            # Update local blocks with built state from remote workers
            for block, res in zip(blocks, results):
                if res:
                    # RayCallableExecutor returns a flat list: [res1, res2, ...]
                    # We expect one result per block.
                    remote_block = res
                    state = remote_block.__getstate__()
                    # Update local block state from the remote result
                    block.__setstate__(state)

        return blocks


class InlineDatablocksBuilder:
    """Builds Datablocks sequentially in the local process, via InlineCallableExecutor."""
    def __init__(self, *, n_workers: int = 1, batch_size: int = None, tag: str = "", log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log
        self._executor = InlineCallableExecutor(n_workers=n_workers, batch_size=batch_size, tag=tag, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [functools.partial(_build_block, block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks


_DATABLOCKS_BUILDERS = {
    "inline":                InlineDatablocksBuilder,
    "multithreading":        MultithreadingDatablocksBuilder,
    "multiprocessing":       MultiprocessingDatablocksBuilder,
    "ray":                   RayDatablocksBuilder,
    "torch_multithreading":  TorchMultithreadingDatablocksBuilder,
    "torch_multiprocessing": TorchMultiprocessingDatablocksBuilder,
}


def select_builder(parallelization: str | None = None):
    """Return the datablocks-builder **class** for the given parallelization strategy.

    Parameters
    ----------
    parallelization : str or None
        One of ``'inline'`` (default), ``'multithreading'``,
        ``'multiprocessing'``, ``'ray'``, ``'torch_multithreading'``,
        ``'torch_multiprocessing'``.  Case-insensitive.
        ``None`` maps to ``'inline'``.

    Returns
    -------
    type
        The builder class (not an instance).
    """
    key = (parallelization or "inline").lower()
    cls = _DATABLOCKS_BUILDERS.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown parallelization {parallelization!r}. "
            f"Choose from {list(_DATABLOCKS_BUILDERS)}"
        )
    return cls


class SlurmRayCluster:
    """
    Manages a Ray cluster running inside a Slurm job.
    """
    def __init__(self, gpus=0, mem='8G', cpus=1, partition=None, nodes=1, nodelist=None, time='01:00:00', log=Logger()):
        self.log = log
        self.job_id = None
        self.ray_address = None
        self.ray_client_address = None
        
        # Create a temp dir for slurm logs
        dbx_slurm_home = os.path.expanduser('~/.dbx/slurm')
        os.makedirs(dbx_slurm_home, exist_ok=True)
        self.tmpdir = tempfile.mkdtemp(prefix='job_', dir=dbx_slurm_home)
        
        partition_str = f"#SBATCH --partition={partition}" if partition else ""
        nodelist_str = f"#SBATCH --nodelist={nodelist}" if nodelist else ""
        if gpus is None:
            gpu_str = "##SBATCH --gres=gpu:0"
        elif isinstance(gpus, int):
            gpu_str = f"#SBATCH --gres=gpu:{gpus}" if gpus > 0 else f"##SBATCH --gres=gpu:0"
        else:
            gpu_str = f"#SBATCH --gres=gpu:{gpus}"
        
        script = f"""#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
{partition_str}
{nodelist_str}
{gpu_str}
#SBATCH --job-name=dbx-ray
#SBATCH --output={self.tmpdir}/slurm-%j.out

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
# Try to get the IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{{print $1}}')
port=6379

echo "HEAD_NODE_IP: ${{head_node_ip}}"

# Start Ray head
echo "Starting Ray head on ${{head_node}} (${{head_node_ip}})"
srun --nodes=1 --ntasks=1 -w "$head_node" ray start --head --port=${{port}} --num-cpus={cpus} --num-gpus={gpus} --block &

# Start Ray workers
for ((i = 1; i < nodes; i++)); do
    node_i=${{nodes_array[$i]}}
    echo "Starting Ray worker on ${{node_i}}"
    srun --nodes=1 --ntasks=1 -w "$node_i" ray start --address="${{head_node_ip}}:${{port}}" --num-cpus={cpus} --num-gpus={gpus} --block &
done

wait
"""
        script_file = os.path.join(self.tmpdir, 'slurm_script.sh')
        with open(script_file, 'w') as f:
            f.write(script)
            
        cmd = ["sbatch", script_file]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to submit Slurm job: {res.stderr}")
            
        # Example output: "Submitted batch job 123456"
        self.job_id = res.stdout.strip().split()[-1]
        self.log.info(f"Submitted Slurm job {self.job_id} for Ray cluster")
        
        # Register for automatic cleanup on exit
        atexit.register(self.cancel)

        self.log.info("Waiting for Ray cluster to start (this may take a minute)...")
        start_time = time_module.time()
        try:
            while time_module.time() - start_time < 600: # 10 minutes max
                # Check if job is still in queue or running
                job_status_res = subprocess.run(["squeue", "-j", self.job_id, "-h", "-o", "%t"], capture_output=True, text=True)
                job_status = job_status_res.stdout.strip()
                
                if job_status == 'R':
                    # Job is running, try to get the head node (BatchHost)
                    batch_host_res = subprocess.run(["scontrol", "show", "job", self.job_id], capture_output=True, text=True)
                    match = re.search(r'BatchHost=(\S+)', batch_host_res.stdout)
                    if match:
                        head_node = match.group(1)
                        # Use the hostname directly (port is fixed at 6379 as per script)
                        addr = f"{head_node}:6379"
                        client_addr = f"ray://{head_node}:10001"
                        
                        # Check if port is open
                        try:
                            with socket.create_connection((head_node, 6379), timeout=1):
                                self.ray_address = addr
                                self.ray_client_address = client_addr
                                self.log.info(f"Ray cluster started at {head_node}:6379 (Client: {head_node}:10001)")
                                break
                        except (socket.timeout, ConnectionRefusedError):
                            pass

                if job_status_res.returncode == 0 and job_status:
                    if job_status in ['F', 'NF', 'TO', 'CA', 'CD']:
                        raise RuntimeError(f"Slurm job {self.job_id} failed or was cancelled (status: {job_status})")
                elif job_status_res.returncode != 0 and job_status_res.stderr:
                    # squeue might fail if job is already finished and moved to history
                    pass
                
                time_module.sleep(2)
        except KeyboardInterrupt:
            self.log.info(f"Interrupted while waiting for Ray cluster. Cancelling Slurm job {self.job_id}")
            self.cancel()
            raise

        if not self.ray_address:
            # Cleanup on timeout
            self.cancel()
            raise RuntimeError("Timed out waiting for Ray cluster to start on Slurm")
            
        # (Already logged cluster start in the loop)

    def cancel(self):
        """Cancel the Slurm job and cleanup temporary files."""
        if self.job_id:
            self.log.info(f"Cancelling Slurm job {self.job_id}")
            subprocess.run(["scancel", self.job_id])
            self.job_id = None
        
        if hasattr(self, 'tmpdir') and os.path.exists(self.tmpdir):
            try:
                shutil.rmtree(self.tmpdir)
            except Exception as e:
                self.log.debug(f"Failed to cleanup temp dir {self.tmpdir}: {e}")

class Remote:
    """
    A client-side proxy to a remote object running in a Ray Actor.
    """
    class RemoteObject:
        """
        Base class for remote proxies. Defined as a non-actor to support inheritance.
        """
        def __init__(self, obj):
            self._obj = obj

        def _wrap(self, val):
            """
            If val is a primitive (int, float, str, bool, None), return it directly.
            Otherwise, wrap it in a RemoteObject actor and return its handle.
            """
            if val is None or isinstance(val, (int, float, str, bool, list, dict, tuple)):
                return val
            
            # Module objects are often not stable when wrapped in new actors via Ray.
            # We return them directly (by value/pickle) to avoid crashing the worker process.

            if isinstance(val, types.ModuleType):
                return val
            
            # Universal proxying for all non-primitives.
            # Expanded call site with nested class as requested.
            # If actor creation fails (e.g. due to pickling issues), we fall back to returning by value.
            try:
                import ray
                return ray.remote(Remote.RemoteObject).remote(val)
            except Exception:
                return val

        def getattr(self, name):
            val = getattr(self._obj, name)
            return self._wrap(val)

        def call(self, name, *args, **kwargs):
            if hasattr(self, name) and name != 'obj' and not name.startswith('__'):
                attr = getattr(self, name)
            else:
                attr = getattr(self._obj, name)

            if not callable(attr):
                raise AttributeError(f"'{name}' is not callable")
            res = attr(*args, **kwargs)
            return self._wrap(res)

        def info(self, name):
            if hasattr(self, name):
                attr = getattr(self, name)
            else:
                attr = getattr(self._obj, name)
            is_call = callable(attr)
            return is_call, (None if is_call else self._wrap(attr))

        def environ(self, name):
            """Helper for verification of remote environment."""

            return os.environ.get(name)

    class RemoteDBX(RemoteObject):
        """
        Ray Actor that acts as a remote handle to the `dbx` module.
        Inherits directly from RemoteObject.
        """
        def __init__(self, revision=None):
            """
            Initialize the remote dbx instance.
            """
            
            # Import dbx after setting environment variables.
            # We import the package to get the full namespace.
            import dbx
            
            # Call this here, because os.environ got updated and/or a new revision may need to be checked out
            dbx.gitwrkreposetup(revision=revision, reason="because of RemoteDBX initialization")
                
            super().__init__(dbx)

        def apply(self, func, *args, **kwargs):
            res = func(*args, **kwargs)
            return self._wrap(res)

        def apply_batch(self, funcs_args_kwargs):
            """
            Execute a sequence of (func, args, kwargs) on the remote actor.
            """
            results = []
            for func, args, kwargs in funcs_args_kwargs:
                res = func(*args, **kwargs)
                results.append(self._wrap(res))
            return results

    def __init__(self, handle=None, *, revision=None, slurm=None):
        """
        Initialize the remote proxy.
        """
        if handle is not None and revision is not None:
            raise ValueError("Remote: Cannot provide both 'handle' and 'revision'")
        self._handle = handle
        self._slurm = slurm
        if handle is None:
            import ray
            self._handle = ray.remote(Remote.RemoteDBX).remote(revision=revision)

    def __del__(self):
        """
        Ensure Slurm job is cancelled when the Remote instance is deleted.
        """
        # Safely check for _slurm without triggering __getattr__
        slurm = self.__dict__.get('_slurm')
        if slurm:
            slurm.cancel()
            self._slurm = None

    def __getattr__(self, name):
        """
        Dispatch attribute access to the remote actor.
        """
        if name.startswith('_') and not name.startswith('__'):
            raise AttributeError(name)

        import ray
        is_callable, value = ray.get(self._handle.info.remote(name))
        
        if is_callable:
            def wrapper(*args, **kwargs):
                import ray
                res = ray.get(self._handle.call.remote(name, *args, **kwargs))
                return self._unwrap_or_proxy(res)
            return wrapper
        else:
            return self._unwrap_or_proxy(value)

    def __getstate__(self):
        """
        Return the state of the remote object.
        """
        import ray
        return ray.get(self._handle.call.remote('__getstate__'))

    def _unwrap_or_proxy(self, val):
        import ray
        if isinstance(val, ray.actor.ActorHandle):
            return Remote(val) # Recursive wrapping
        return val

    def run(self, func, *args, **kwargs):
        """
        Execute a callable on the remote actor.
        """
        import ray
        res = ray.get(self._handle.apply.remote(func, *args, **kwargs))
        return self._unwrap_or_proxy(res)

    def run_batch(self, funcs_args_kwargs):
        """
        Execute a sequence of (func, args, kwargs) on the remote actor in one round-trip.
        """
        import ray
        results = ray.get(self._handle.apply_batch.remote(funcs_args_kwargs))
        return [self._unwrap_or_proxy(res) for res in results]


def remote(*, revision=None, slurm=None, conda=None, log: Logger = Logger()):
    """
    Instantiate a remote dbx interpreter and return a Remote handle to it.
    """
    import ray
    dbx_env = {k: v for k, v in os.environ.items() if k.startswith('DBX')}
    
    if DBX_USE_WORK_REPO is not None:
        dbx_env['DBX_GIT_REPO'] = DBX_USE_WORK_REPO

    # If we are using a remote cluster, any path in /tmp on the login node will be inaccessible to workers.
    # We revert to the original repository path (usually in /home) which is shared.
    if slurm:
        dbx_env['DBX_GIT_REPO'] = _DBX_GIT_REPO_

    runtime_env = {'env_vars': dbx_env}
    if conda:
        runtime_env['conda'] = conda

    if slurm and slurm.ray_address:
        if ray.is_initialized():
            log.info("Re-initializing Ray to connect to Slurm Ray cluster...")
            ray.shutdown()
        # Use ray:// protocol for remote clusters to avoid shared /tmp/ray filesystem requirements
        address = getattr(slurm, 'ray_client_address', slurm.ray_address)
        ray.init(address=address, runtime_env=runtime_env, ignore_reinit_error=True)
    elif not ray.is_initialized():
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)
    
    log.verbose(f"INSTANTIATING Remote with env: {dbx_env}, revision: {revision}, slurm: {bool(slurm)}, conda: {conda}")
    return Remote(revision=revision, slurm=slurm)


def slurm_remote(*, revision=None, conda=None, gpus=0, mem='8G', cpus=1, partition=None, nodes=1, nodelist=None, time='01:00:00', log: Logger = Logger()):
    """
    Start a Slurm job with a Ray cluster and return a Remote handle to it.
    """
    cluster = SlurmRayCluster(gpus=gpus, mem=mem, cpus=cpus, partition=partition, nodes=nodes, nodelist=nodelist, time=time, log=log)
    return remote(revision=revision, slurm=cluster, conda=conda, log=log)


def slurm_exec(s=None, *, revision=None, conda=None, gpus=0, mem='8G', cpus=1, partition=None, nodes=1, nodelist=None, time='01:00:00', log: Logger = Logger(), **kwargs):
    if s is None:
        if len(sys.argv) < 2:
            raise ValueError(f"Too few args: {sys.argv}")
        s = sys.argv[1]
        for arg in sys.argv[2:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                try:
                    kwargs[k] = __eval__(v)
                except (NameError, SyntaxError):
                    kwargs[k] = v
    
    # Merge named args with kwargs (CLI overwrites programmatic defaults if passed)
    slurm_params = {
        'revision': revision, 'conda': conda, 'gpus': gpus, 'mem': mem,
        'cpus': cpus, 'partition': partition, 'nodes': nodes, 'nodelist': nodelist, 'time': time, 'log': log
    }
    for k in slurm_params:
        if k in kwargs:
            slurm_params[k] = kwargs.pop(k)
            
    r = None
    try:
        r = slurm_remote(**slurm_params)
        return r.run(eval, s, **kwargs)
    finally:
        if r is not None:
             # Force cancellation if r goes out of scope or process exits
             if hasattr(r, '_slurm') and r._slurm:
                 r._slurm.cancel()
                 r._slurm = None


def slurm_pprint(s=None, *, revision=None, conda=None, gpus=0, mem='8G', cpus=1, partition=None, nodes=1, nodelist=None, time='01:00:00', log: Logger = Logger(), **kwargs):
    _pprint_.pprint(slurm_exec(s, revision=revision, conda=conda, gpus=gpus, mem=mem, cpus=cpus, partition=partition, nodes=nodes, nodelist=nodelist, time=time, log=log, **kwargs))


class UNSAFE_datablock_journal_puller:
    def __init__(self, datablock_classname, idx, *, throw=True, clear=False, log=Logger(name="UNSAFE_datablock_journal_puller")):
        self.datablock_classname = datablock_classname
        self.idx = idx
        self.throw = throw
        self.clear = clear
        self.log = log
        
    def __call__(self, journal, datablocks=None):
        if datablocks is not None:
            datablock_handles = [datablock.norm() for datablock in datablocks]
        else:
            datablock_handles = None
        try:
            dbk = None
            anchorhashpath = None
            entry = journal(journal.index[self.idx])
            spec = entry.read('spec')
            dbk = eval(quotefn(self.datablock_classname, spec=spec))
            anchorhashpath = entry.anchorkeypath
            if datablock_handles is not None:
                if dbk.norm() in datablock_handles:
                    self.log.debug(f"Skipping datablock {dbk.norm()}: not in datablocks")
                    return None
            self.log.debug(f"Copying from {anchorhashpath} to {dbk}: BEGIN")
            dbk.UNSAFE_copy_from(anchorhashpath)
            self.log.debug(f"Copying from {anchorhashpath} to {dbk}: END")
            self.log.debug(f"VALID: {dbk.valid()}")
            if self.clear and dbk.valid():
                dbk = entry.inst()
                self.log.warning(f"Clearning datablock {_dbk_}")
                dbk.UNSAFE_clear()
            copied = True
        except Exception as e:
            copied = False
            if self.throw:
                raise(e)
            else:
                self.log.debug(f"Copying from {anchorhashpath} to {dbk}: EXCEPTION:\n{e}\nSkipping")
                self.log.debug(f"VALID: {dbk.valid()}")
        return dbk, copied
    

def UNSAFE_pull_datablocks_from_journal(datablock_classname, *, n_workers: int = 0, throw: bool = False, clear: bool = False, log: Logger = Logger(), event="build:end", revision=None, date=None, **other_journal_kwargs):
    """
    Pull datablocks from the journal based on specified criteria.

    Args:
        datablock_classname (str): The name of the Datablock class to pull.
        n_workers (int): Number of worker threads to use for pulling. 
            If 0, pulling is performed sequentially in the main thread.
        throw (bool): If True, exceptions encountered during pulling will be raised.
            If False, they will be logged as debug information.
        clear (bool): If True, valid datablocks will be cleared after being instantiated.
        log (Logger): Logger instance for status and debug messages.
        event (str): The journal event to filter by (e.g., "build:end").
        revision (Optional[str]): The journal revision to filter by.
        date (Optional[str/datetime]): The journal date to filter by.
        **other_journal_kwargs: Additional keyword arguments used to filter the journal query in addition to event, revision, and date.

    Returns:
        tuple[tuple[Datablock, ...], tuple[bool, ...]]: A tuple containing:
            - A tuple of instantiated Datablock objects.
            - A tuple of booleans indicating if each datablock was successfully copied.
    """
    log.verbose(f"Pulling {datablock_classname} ...")
    journal_kwargs = other_journal_kwargs.copy()
    if event is not None:
        journal_kwargs['event'] = event
    if revision is not None:
        journal_kwargs['revision'] = revision
    if date is not None:
        journal_kwargs['date'] = date
    log.verbose(f"Building journal using {journal_kwargs=} ...")
    _journal_ = journal(datablock_classname, **journal_kwargs)
    log.verbose(f"Found {len(_journal_)} journal entries ...")
    log.debug(f"{_journal_=}")

    _callables_ = [UNSAFE_datablock_journal_puller(datablock_classname, i, throw=throw, clear=clear, log=log) for i in range(len(_journal_))]
    log.verbose(f"Pulling ...")
    if n_workers == 0:
        results = []
        if log.ist('verbose'):
            _callables = tqdm.tqdm(_callables_)
            for _callable_ in _callables_:
                result = _callable_(_journal_)
                if result is not None:
                    results.append(result)
    else:
        executor = MultithreadingCallableExecutor(n_workers=n_workers, log=log)
        _results_ = executor.execute(_callables_, _journal_)
        results = [result for result in _results_ if result is not None]
    dbks, copied = zip(*results)
    pulled = sum([int(c) for c in copied])
    valids = sum([int(dbk.valid()) for dbk in dbks])
    log.info(f"Pulled {pulled} out of {len(copied)} datablocks.  Skipped {len(copied)-pulled}. Valids: {valids}")
    log.info(f"Done")
    return dbks, copied


