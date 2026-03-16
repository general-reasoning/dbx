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
import ray
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time as time_module
import traceback as tb
from typing import Union, Optional, Sequence, Callable
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

from scipy.stats import qmc

import pandas as pd
import torch
import torch.multiprocessing as mp


__eval__ = __builtins__['eval']
__version__ = "0.1.0"


DBXGITREPO = os.environ.get('DBXGITREPO')
if DBXGITREPO is None:
    try:
        import git
        _repo = git.Repo('.', search_parent_directories=True)
        DBXGITREPO = _repo.working_tree_dir
    except (ImportError, Exception):
        pass
_DBXGITREPO_ = DBXGITREPO
DBXUSEWRKREPO = None
DBXWRKROOT = None

def dbx_repos(repopath=None):
    if repopath is None:
        repopath = DBXGITREPO
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
    global DBXGITREPO
    global DBXUSEWRKREPO
    global DBXWRKROOT
    
    dbx_repo, project_repo = dbx_repos(gitrepo)
    dbx_rev, project_rev = dbx_revisions(revision)

    def setup_wrkrepo(repo, rev, name):
        nonlocal log
        if repo is None:
            return None
        log.info(f"SETTING UP TEMPORARY DBXWRKROOT for {name} from {repo=} {reason} with revision {rev}")
        wrkroot = tempfile.TemporaryDirectory()
        package = os.path.basename(repo)
        wrkrepo = os.path.join(wrkroot.name, package)
        _pkgpath = gitclone(repo, wrkrepo)
        assert wrkrepo == _pkgpath
        if rev is not None:
            gitcheckout(wrkrepo, rev)
            log.info(f"Checked out {wrkrepo} to revision {rev}")
        else:
            log.info(f"Using {wrkrepo} at HEAD")
        sys.path.insert(0, wrkrepo)
        return wrkroot, wrkrepo

    use_wrkrepo = os.environ.get('DBXUSEWRKREPO') == 'True' or revision is not None
    if use_wrkrepo and DBXUSEWRKREPO is None:
        if DBXGITREPO is None:
            raise ValueError("DBXGITREPO is not set and could not be detected. Cannot setup temporary wrkrepo.")
        
        dbx_wrk = setup_wrkrepo(dbx_repo, dbx_rev, "dbx")
        project_wrk = setup_wrkrepo(project_repo, project_rev, "project")

        DBXWRKROOT = (dbx_wrk[0] if dbx_wrk else None, project_wrk[0] if project_wrk else None)
        
        dbx_wrkrepo = dbx_wrk[1] if dbx_wrk else None
        project_wrkrepo = project_wrk[1] if project_wrk else None
        
        if dbx_wrkrepo:
            wrkrepo_str = f"{dbx_wrkrepo}:{project_wrkrepo}"
        else:
            wrkrepo_str = project_wrkrepo
        
        globals()['DBXUSEWRKREPO'] = wrkrepo_str
        os.environ['DBXGITREPO'] = wrkrepo_str
        
        if 'DBXUSEWRKREPO' in os.environ:
            del os.environ['DBXUSEWRKREPO']
            
        log.info(f"DBXUSEWRKREPO: {wrkrepo_str}")


def journal(cls_or_df, root=None, **kwargs):
    if isinstance(cls_or_df, pd.DataFrame):
        return JournalFrame(cls_or_df, **kwargs)
    else:
        return Datablock.Journal(cls_or_df, root, **kwargs)


class Logger:
    """Because Python logging is so cumbersome to initialize, configure and control, we have this."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        warning: bool = None,
        info: bool = None,
        verbose: bool = None,
        debug: bool = None,
        selected: bool = None,
        detailed: bool = None,
        selection: Union[str, Sequence[str]] = None,
        datetime: bool = True,
        stack_depth: int = 2,
    ):
        self.stack_depth = stack_depth
        self.name = name
        self.datetime = datetime
        self.allowed = ["ERROR"]
        
        _defaults_ = {
            'warning': True,
            'info': True,
            'verbose': False,
            'selected': True,
            'debug': False,
            'detailed': False,
            'selection': None,
        }
        def set_arg(name, locals):
            """Prioritize kwarg value, fall back to env var, then to default."""
            # Get kwarg value from local scope
            kwarg_value = locals.get(name)
            if kwarg_value is not None:
                result = kwarg_value
            else:
                # Generate env key: 'warning' -> 'DBXLOGWARNING'
                env_key = f'DBXLOG{name.upper()}'
                env_val = os.environ.get(env_key)
                if env_val is not None:
                    try:
                        result = __eval__(env_val)
                    except (NameError, SyntaxError):
                        result = env_val
                else:
                    result = _defaults_[name]
            setattr(self, f'_{name}_', result)
            if result and name != 'selection':
                self.allowed.append(name.upper())
        
        for argname in _defaults_.keys():
            set_arg(argname, locals())
        
        if self._selection_ is None:
            self._selection_ = []
        elif isinstance(self._selection_, str):
            self._selection_ = [self._selection_]
        if len(self._selection_) == 0:
            self._selection_ = os.environ.get('DBXLOGSELECTION')
            if self._selection_ is not None:
                self._selection_ = [s.strip() for s in self._selection_.split(',')]
            
    def get(self, key):
        return getattr(self, f"_{key}_")
    
    def ist(self, key):
        return getattr(self, f"_{key}_")


    def _print(self, prefix, msg):
        """
        #TODO: figure out why the next line causes things to hang sometimes
        stack = inspect.stack() 
        if self.stack_depth is None or self.stack_depth >= len(stack):
            func = None
        else:
            frame = stack[self.stack_depth]
            func = frame.function
        """
        func = None
        
        if self.name is None:
            if func is None:
                tag = ""
            else:
                tag = f"{func}: "
        else:
            if func is None:
                tag = f"{self.name}: "
            else:
                tag = f"{self.name}: {func}: "
        if prefix in self.allowed:
            dt = f"{datetime.datetime.now().isoformat()}: " if self.datetime else ""
            print(f"{prefix}: {dt}{tag}{msg}")

    def error(self, msg):
        self._print("ERROR", msg)

    def warning(self, msg):
        self._print("WARNING", msg)

    def info(self, msg):
        self._print("INFO", msg)

    def debug(self, msg):
        self._print("DEBUG", msg)

    def verbose(self, msg):
        self._print("VERBOSE", msg)

    def selected(self, msg):
        if self._selection_:
            try:
                frame = sys._getframe(self.stack_depth - 1)
                module = frame.f_globals.get('__name__')
                function = frame.f_code.co_name
                fqn = f"{module}.{function}"
                if fqn not in self._selection_:
                    return
            except (ValueError, AttributeError):
                pass
        self._print("SELECTED", msg)

    def detailed(self, msg):
        self._print("DETAILED", msg)

    def silent(self, mst):
        pass


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # Ensure immediate writing

    def flush(self):
        for f in self.files:
            f.flush()

    def __getattr__(self, name):
        """Proxy all missing attributes to the first file/stream."""
        return getattr(self.files[0], name)


class JournalEntry(pd.Series):
    def __init__(self, series: pd.Series, *, logger: Logger = Logger(name="JournalEntry")):
        super().__init__(series)
        self.logger = logger

    def __tag__(self):
        return f"JournalEntry:{self.anchor}/{self.hash}"

    @property
    def anchor(self):
        return self.get('anchor')

    @property
    def hash(self):
        return self.get('hash')

    @property
    def root(self):
        return self.get('root')

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
    def anchorhash(self):
        return os.path.join(self.anchor, self.hash)

    @property
    def anchorhashpath(self):
        return os.path.join(self.root, self.anchorhash)

    def read(self, *things, raw: bool = False, deslash: bool = False, safe: bool = False):
        def read_thing(thing):
            if hasattr(self, thing) and getattr(self, thing) is not None:
                path = getattr(self, thing)
                _, _ext = os.path.splitext(path)
                ext = _ext[1:]
                if raw or ext == 'txt' or ext == 'log':
                    result = read_str(getattr(self, thing))
                elif ext == 'yaml':
                    result = read_yaml(getattr(self, thing), safe=safe)
                else:
                    raise ValueError(f"Uknown journal entry field extention for {thing}: {ext}")
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
    
    def eval(self, thing, *, debug: bool = False, context={}, eval_term: bool = False, deslash: bool = False, gitrepo=None, revision=None):
        exc = None
        thingstr = self.read(thing, raw=True)
        if deslash:
            thingstr = thingstr.replace('\\', '')
        r = None
        # Call this here because a new revision may need to be checked out
        gitwrkreposetup(revision=revision, gitrepo=gitrepo, reason=f"because of evaluating a JournalEntry field {thing}")
        try:
            if eval_term:
                __eval_term__ = globals()['eval_term']
                r = __eval_term__(thingstr)
            else:
                r = __eval__(thingstr, globals(), context)
        except Exception as exc:
            if debug:
                breakpoint()
            else:
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
        return self.eval('quote', eval_term=True, gitrepo=gitrepo, revision=revision)

    def inst(self, gitrepo=None, revision='journal_entry'):
        if gitrepo is None:
            gitrepo = DBXGITREPO
        if gitrepo is None:
            gitrepo = 'journal_entry'
        return self.instantiate(gitrepo=gitrepo, revision=revision)
    

class JournalFrame(pd.DataFrame):
    def __init__(self, df: pd.DataFrame, *, parse_datetimes: bool = True, logger: Logger = Logger(), **kwargs):
        
        # Process the dataframe before calling super().__init__()
        if parse_datetimes:
            if 'datetime' in df.columns and not isinstance(df['datetime'].iloc[0], datetime.datetime): # TODO: use dtype?
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H-%M-%S.%f')
        for k, v in kwargs.items():
            if k == 'date':
                if isinstance(v, str):
                    v = pd.to_datetime(v).date()
                elif isinstance(v, list):
                    v = [pd.to_datetime(x).date() for x in v]
                if isinstance(v, list):
                    df = df[df['datetime'].dt.date.isin(v)]
                else:
                    df = df[df['datetime'].dt.date == v]         
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
        self.logger = logger
            

    def get(self, entry:int, *, dropna: bool = False):
        entry = self.loc[entry]
        if dropna:
            entry = entry.dropna()
        return JournalEntry(entry)
    
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
                entry = JournalEntry(row)
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
    repopath = DBXUSEWRKREPO if DBXUSEWRKREPO is not None else DBXGITREPO
    if repopath is not None:
        d_repo, project_repo = dbx_repos(repopath)
        
        def get_rev(path):
            if path is None:
                return None
            repo = git.Repo(path)
            if repo.is_dirty() and not os.environ.get('DBXDIRTYREPOK'):
                raise ValueError(f"Dirty git repo: {path}: commit your changes")
            return repo.head.commit.hexsha

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


gitwrkreposetup(reason="for initial import of dbx")


def make_google_cloud_storage_download_url(path):
    if not path.startswith("gs://"):
        return None
    _path = path.removeprefix("gs://")
    return f"https://storage.cloud.google.com/{_path}"


def get_named_const_and_cxt(name):
    bits = name.split(".")
    modbits = bits[:-1]
    cxt = {}
    if not modbits:
        return None, cxt
    prefix = None
    for modbit in modbits:
        if prefix is not None:
            modname = prefix + "." + modbit
        else:
            modname = modbit
        mod = importlib.import_module(modname)
        prefix = modname
        cxt[modname] = mod
    constname = bits[-1]
    const = getattr(mod, constname)
    return const, cxt


def eval_term(name):
    def get_named_args_kwargs(argkwargstr):
        args = []
        kwargs = {}
        if len(argkwargstr) > 0:
            bits = argkwargstr.split(",")
            for bit in bits:
                if "=" in bit:
                    k, v = bit.split("=")
                    val = __eval__(v)
                    kwargs[k] = val
                else:
                    arg = __eval__(bit)
                    args.append(arg)
        return args, kwargs

    def get_funcstr_argkwargstr(name):
        # TODO: replace with a regex
        lb = name.find("(")
        rb = name.rfind(")")
        if lb == -1 or rb == -1:
            funcstr = None
            argkwargstr = None
        else:
            funcstr = name[:lb]
            argkwargstr = name[lb + 1 : rb]
        return funcstr, argkwargstr

    Logger("eval_term").detailed(f" ====================> Evaluating term {repr(name)}")
    if isinstance(name, Iterable) and not isinstance(name, str):
        term = [eval_term(item) for item in name]
    elif isinstance(name, str):
        if name.startswith("@") or name.startswith("#") or name.startswith("$"):
            _name_ = name[1:]
            funcstr, _ = get_funcstr_argkwargstr(_name_)
            if funcstr is None:
                term, _ = get_named_const_and_cxt(_name_)
            else:
                _, cxt = get_named_const_and_cxt(funcstr)
                term = __eval__(_name_, cxt)
        else:
            term = name
    else:
        term = name
    return term


def eval(s=None, **kwargs):
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
    
    lb = s.find("(")
    lb = lb if lb != -1 else len(s)
    _, cxt = get_named_const_and_cxt(s[:lb])
    cxt.update(kwargs)
    r = __eval__(s, globals(), cxt)
    
    return r


def slurm_eval(s=None, *, revision=None, conda=None, gpus=0, mem='8G', cpus=1, partition=None, nodes=1, nodelist=None, time='01:00:00', log: Logger = Logger(), **kwargs):
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


def slurm_exec(s=None, *, revision=None, conda=None, gpus=0, mem='8G', cpus=1, partition=None, nodes=1, nodelist=None, time='01:00:00', log: Logger = Logger(), **kwargs):
    return slurm_eval(s, revision=revision, conda=conda, gpus=gpus, mem=mem, cpus=cpus, partition=partition, nodes=nodes, nodelist=nodelist, time=time, log=log, **kwargs)


def slurm_pprint(s=None, *, revision=None, conda=None, gpus=0, mem='8G', cpus=1, partition=None, nodes=1, nodelist=None, time='01:00:00', log: Logger = Logger(), **kwargs):
    _pprint_.pprint(slurm_eval(s, revision=revision, conda=conda, gpus=gpus, mem=mem, cpus=cpus, partition=partition, nodes=nodes, nodelist=nodelist, time=time, log=log, **kwargs))


def pprint(argstr=None, **kwargs):
    _pprint_.pprint(exec(argstr, **kwargs))


def exec(s=None, **kwargs):
    return eval(s, **kwargs)


def write_str(text, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "w") as f:
        f.write(text)
        log.detailed(f"WROTE {path}")


def read_str(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        text = f.read()
        log.detailed(f"READ {path}")
    return text


def write_yaml(data, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "w") as f:
        yaml.dump(data, f)
        log.detailed(f"WROTE {path}")


def read_yaml(path, *, log=Logger(), safe: bool = False, debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.BaseLoader) if safe else yaml.load(f, Loader=yaml.UnsafeLoader)
        log.detailed(f"READ {path}")
    return data


def write_json(data, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "w") as f:
        json.dump(data, f)
        log.detailed(f"WROTE {path}")


def read_json(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        data = json.load(f)
        log.detailed(f"READ {path}")
    return data


def write_tensor(tensor, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    array = tensor.numpy()
    with fs.open(path, "wb") as f:
        np.save(f, array)
        log.detailed(f"WROTE {path}")


def read_tensor(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "rb") as f:
        array = np.load(f)
        log.detailed(f"READ {path}")
        tensor = torch.from_numpy(array)
    return tensor


def write_tensors(path, *, log=Logger(), debug: bool = False, **tensors):
    arrays = {k: v.numpy() for k, v in tensors.items()}
    return write_npz(path, log=log, debug=debug, **arrays)


def read_tensors(path, *keys, log=Logger(), debug: bool = False):
    arrays = read_npz(path, *keys, log=log, debug=debug)
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    return tensors


def write_npz(path, *, log=Logger(), debug: bool = False, **kwargs):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "wb") as f:
        np.savez(f, **kwargs)
        log.detailed(f"WROTE {list(kwargs.keys())} to {path}")


def read_npz(path, *keys, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "rb") as f:
        data = np.load(f, allow_pickle=True)
        results = {k: data[k] for k in keys}
        log.detailed(f"READ {list(keys)} from {path}")
        return results
    

def write_pickle(obj, path):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(path):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, 'rb') as f:
        return pickle.load(f)


class IntRange(tuple):
    # TODO: ought to be a dataclass, but then isinstance(x, IntRange) might fail
    pass


class FloatRange(tuple):
    pass


class BoolRange(tuple):
    pass


def make_halton_sampling_kwargs_sequence(N, range_kwargs, *, seed=123, precision=4):
    log = Logger()

    def collect_bounds():
        lower, upper = [], []
        log.debug(f"range_kwargs: {range_kwargs}")
        for v in range_kwargs.values():
            log.debug(f"v: {v}")
            if isinstance(v, FloatRange) or isinstance(v, IntRange):
                log.debug(f"Caught a range value: {v}")
                lower.append(float(v[0]))
                upper.append(float(v[1]))
        log.debug(f"lower: {lower}, upper: {upper}")
        return lower, upper

    lower, upper = collect_bounds()
    halton = qmc.Halton(d=len(lower), seed=seed)
    halton.reset()  # TODO: REMOVE?
    sample = halton.random(N)
    if len(lower) > 0:
        ssample = qmc.scale(sample, lower, upper)
        kwargs_list = []
        for i in range(ssample.shape[0]):
            j = 0
            kwargs = {}
            for k, v in range_kwargs.items():
                if isinstance(v, FloatRange) or isinstance(v, IntRange):
                    if isinstance(v, IntRange):
                        kwargs[k] = int(round(ssample[i, j]))
                    else:
                        kwargs[k] = round(ssample[i, j], precision)
                    j += 1
                else:
                    kwargs[k] = v
            kwargs_list.append(kwargs)
    else:
        kwargs_list = [range_kwargs]
    return kwargs_list


class Datablock:
    """
    ROOT = 'protocol://path/to/root'
    TOPICFILES = {'topic', 'file.csv'} | TOPICFILE = 'file.csv'
    # protocol://path --- module/class/ --- topic [--- file]
    #        root           [anchor]        [topic]   [file]
    # root:       'protocol://path/to/root'
    # anchorpath: '{root}/modpath/class'|'{root}' if anchored|else
    # hashpath:   '{anchorpath}/{hash}|{anchorpath}/{hash}' if hash supplied through args|else
    # dirpath:    '{hashpath}/topic'|{hashpath}' if topic is not None|else
    # path:       '{dirpath}/{TOPICFILE}'|'{dirpath}' if TOPICFILE is not None|else
    
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
        handle: str
        hashstr: str
        anchor: str

        def deslash(self, attr):
            a = getattr(self, attr)
            if isinstance(a, str):
                aa = a.replace('\\', '')
            else:
                aa = a
            return aa

        def fields(self):
            return {f.name: f.type for f in fields(self)}

    @dataclass
    class CONFIG:
        class LazyLoader:
            def __init__(self, term):
                self.term = term
                self.value = None
            def __call__(self):
                if self.value is None:
                    self.value = eval_term(self.term)
                return self.value

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if isinstance(attr, Datablock.CONFIG.LazyLoader):
                return attr()
            return attr

    def __init__(
        self,
        *,
        root: str = None,
        spec: Optional[Union[str, dict]] = None,
        anchored: bool = True,
        hash: Optional[str] = None,
        tag: Optional[str] = None,
        info: bool = None,
        verbose: bool = None,
        debug: bool = None,
        detailed: bool = None,
        capture_output: bool = False,
        revision: str = None,
        device: str = 'cpu',
        uuid16: bool = False,
        **kwargs,
    ):
        self._working_params_ = []
        self._uuid16_ = uuid16
        self._uuid = uuid.uuid4().hex[:16] if uuid16 else str(uuid.uuid4())  # unique per live instance, not preserved across serialization
        state = {
            'root': root,
            'spec': spec,
            'anchored': anchored,
            'hash': hash,
            'tag': tag,
            'info': info,
            'verbose': verbose,
            'debug': debug,
            'detailed': detailed,
            'capture_output': capture_output,
            'revision': revision,
            'device': device,
            'uuid16': uuid16,
        }
        state.update(kwargs)
        self.__setstate__(state)
        
    def __setstate__(self, state):
        """NB: state keys should match __init__'s keyword arguments, with extra args properly captured in state."""
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
        self._root_ = state.get('root')
        self.root = self._root_
        if self.root is None:
            self.root = os.environ.get('DBXROOT')
        if self.root is None:
            raise ValueError(f"None root for {self.__class__.__name__}: maybe set DBXROOT?")
        self._spec_ = state.get('spec')
        if self._spec_ is None:
            self.spec = asdict(self.CONFIG())
        else:
            self.spec = self._spec_
        self.anchored = state.get('anchored', True)
        self._hash_ = state.get('hash')
        self._tag_ = state.get('tag')
        
        # Initialize early logger for __post_init__ if needed, though usually hash is needed
        self.log = Logger(
            f"{self.anchor}",
            debug=state.get('debug', False),
            verbose=state.get('verbose', False),
            detailed=state.get('detailed', False),
            info=state.get('info', True),
            stack_depth=None, #TODO: restore stack_depth default
        )
        self._revision_ = state.get('revision')
        self.capture_output = bool(state.get('capture_output', False))
        self.device = state.get('device', 'cpu')
        self._uuid16_ = state.get('uuid16', False)
        

        
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
        self.build_dt = None
        
        # Redefine logger with hash
        self.log = Logger(
            name=f"{self.anchor}/{self.hash}",
            debug=state.get('debug', False),
            verbose=state.get('verbose', False),
            detailed=state.get('detailed', False),
            info=state.get('info', True),
            stack_depth=None, #TODO: restore stack_depth default
        )
        self.__post_init__()
        self.log.detailed(f"======--------------> bid: {self.bid}")

    def __getstate__(self):
        _state = {}
        for k in self.__explicit_params__():
            if hasattr(self, f"_{k}_"):
                _state[k] = getattr(self, f"_{k}_")
            elif hasattr(self, k):
                _state[k] = getattr(self, k)
        
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
    
    def to(self, device):
        self.device = device
        return self

    def __post_init__(self):
        ...

    @property
    def bid(self):
        return self.Bid(
            hash=self.hash,
            version=self.version,
            revision=self.revision,
            kwargs=self.kwargs,
            spec=self.spec,
            dfn=self.dfn,
            quote=self.quote(),
            repr=self.__repr__(),
            handle=self.handle(),
            hashstr=self.hashstr,
            anchor=self.anchor,
        )
    
    def validtopic(self, topic=None):
        if topic is None:
            valid = self.validpath(self.path())
        else:
            valid = self.validpath(self.path(topic))
        self.log.detailed(f"{self.anchor}: topic {topic} valid: {valid}")
        return valid
    
    def validtopics(self, topics=None, *, reduce: bool = False):
        result = None
        if topics is None:
            if not self.has_topics():
                result = self.validtopic()
            else:
                topics = self.topics()
        if result is None:
            results = {
                topic:
                self.validtopic(topic) for topic in topics
            }
            if reduce:
                result = all(list(results.values()))
            else:
                result = results
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
            fs, _ = fsspec.url_to_fs(path)
            if 'file' not in fs.protocol:
                result = fsspec.filesystem("gcs").exists(path)
            else:
                result = os.path.exists(path) #TODO: Why not handle this case using fsspec? 
        self.log.detailed(f"{self.anchor}: path {path} valid: {result}") 
        return result
    
    def validpaths(self, topics=None, *, reduce: bool = False):
        result = None
        if topics is None:
            if not self.has_topics():
                results = [self.validpath(self.path())]
                if reduce:
                    result = all(results)
                else:
                    result = results
            else:
                topics = self.topics()
        if result is None:
            results = {
                topic: self.validpath(self.path(topic))
                for topic in topics
            }
            if reduce:
                result = all(list(results.values()))
            else:
                result = results
        return result
    
    def valid(self, topic=None):
        if topic is not None:
            return self.validtopic(topic)
        return self.validpaths(reduce=True)
    
    def topics(self):
        return list(self.TOPICFILES.keys()) if self.has_topics() else ([] if self.has_topic() else None)

    def has_topics(self):
        return hasattr(self, "TOPICFILES")
    
    def has_topic(self):
        return hasattr(self, "TOPICFILE")

    def build(self, *args, **kwargs):
        if self.capture_output:
            self.log.verbose(f"-------------------- Capturing stdout/stderr to {self._logpath()} ------------------")
            stdout = sys.stdout
            stderr = sys.stderr
            logpath = self._logpath()
            outfs, _ = fsspec.url_to_fs(logpath)
            captured_stream = outfs.open(logpath, "w", encoding="utf-8")
            sys.stdout = Tee(stdout, captured_stream)
            sys.stderr = Tee(stderr, captured_stream)
        try:
            if not self.valid():
                self.__pre_build__(*args, **kwargs)
                self.__build__(*args, **kwargs)
                self.build_dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
                self.__post_build__(*args, **kwargs)
            else:
                self.log.verbose(f"Skipping existing datablock: {self.hashpath()}")
        except KeyboardInterrupt as e:
            self.__post_build__(*args, event="build:keyboard_interrupt", **kwargs)
            raise(e)
        except Exception as e:
            self.__post_build__(*args, event="build:exception", **kwargs)
            raise(e)
        finally:
            if self.capture_output:
                sys.stdout = stdout
                sys.stderr = stderr
                captured_stream.close()
        return self

    def __pre_build__(self, *args, **kwargs):
        valid_cfg = self.valid_cfg()
        if not all(list(valid_cfg.values())):
            raise ValueError(f"Not all upstream Datablocks in cfg are valid: {valid_cfg=}")
        self._write_journal_entry(event="build:start",)
        return self

    def __post_build__(self, *args, event="build:end", **kwargs):
        self._write_journal_entry(event=event,)
        return self
    
    def __build__(self, *args, **kwargs):
        return self

    def leave_breadcrumbs(self):
        if hasattr(self, "TOPICFILES"):
            for topic in self.TOPICFILES:
                self.dirpath(topic, ensure=True)
                self.leave_breadcrumbs_at_path(self.path(topic))
        else:
            self.dirpath(ensure=True)
            self.leave_breadcrumbs_at_path(self.path())
        return self

    def build_tree(self, *args, exclude_self: bool = False, **kwargs):
        self.log.verbose(f"Building tree for {self} with roots {self.spec.keys()}")
        for s in self.spec.keys():
            c = getattr(self.cfg, s)
            if isinstance(c, Datablock):
                self._write_journal_entry(event=f"build_tree:{s}:begin")
                self.log.verbose(f"------------------------ BUILDING SUBTREE at {s}: BEGIN --------------------------------")
                c.build_tree(*args, **kwargs)   
                self.log.verbose(f"------------------------ BUILDING SUBTREE at {s}: END --------------------------------")
                self._write_journal_entry(event=f"build_tree:{s}:end")
        if not exclude_self:
            self.build(*args, **kwargs)
        return self
    
    def valid_cfg(self, *, reduce=False):
        results = {}
        for s in self.spec.keys():
            c = getattr(self.cfg, s)
            if isinstance(c, Datablock):
                results[s] = c.valid()
        if reduce:
            return all(list(results.values()))
        else:
            return results
    
    def read(self, topic=None):
        if self.has_topics():
            if topic not in self.TOPICFILES:
                raise ValueError(f"Topic {repr(topic)} not in {self.TOPICFILES}")
            _ =  self.__read__(topic)
        else:
            _ = self.__read__()
        return _
    
    def __read__(self, topic=None):
        raise NotImplementedError()
    
    def UNSAFE_clear(self, *topics, OVERRIDE: bool = False):
        if not OVERRIDE:
            response = input("ARE YOU SURE YOU WANT TO EXECUTE 'UNSAFE_clear'? [y/N]")
            if response.lower() != 'y':
                return self
        
        def clear_dirpath(dirpath, *, throw=False):
            self.log.info(f"removing {dirpath}")
            try:
                if dirpath.startswith("gs://"):
                    """
                    Circumvent bugs in fsspec and helm.data.utils
                    """
                    from google.cloud import storage

                    client = storage.Client()
                    bits = dirpath.removeprefix("gs://").split("/")
                    bucket_name = bits[0]
                    prefix = "/".join(bits[1:])
                    bucket = client.get_bucket(bucket_name)
                    blobs = bucket.list_blobs(prefix=prefix)
                    for blob in blobs:
                        blob.delete()
                else:
                    # fs = makefs(dirpath) # TODO: REMOVE
                    fs, _ = fsspec.url_to_fs(dirpath)
                    fs.rm(dirpath, recursive=True)
            except Exception as e:
                self.log.warning(f"Error when trying to remove {dirpath}")
                self.log.warning(f"EXCEPTION: {e}")
                if throw:
                    raise (e)
        if len(topics) == 0:
            if hasattr(self, "TOPICFILES"):
                for topic in self.TOPICFILES:
                    clear_dirpath(self.dirpath(topic))
            else:
                clear_dirpath(self.dirpath())
            self._write_journal_entry(event="UNSAFE_clear")
        else:
            for topic in topics:
                clear_dirpath(self.dirpath(topic))
            self._write_journal_entry(event=f"UNSAFE_clear:{[topics]}")
        return self
    
    def UNSAFE_copy_from(self, anchorhashpath, *, overwrite: bool = False, topicpaths=None, validate: bool = True):
        def fscopy(*, src_path, dst_path, recursive: bool = False):
            # fsspec does not implement .copy, so use put/get or temporary directory
            src_fs, _ = fsspec.url_to_fs(src_path)
            dst_fs, _ = fsspec.url_to_fs(dst_path)
            
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

        if not overwrite:
            assert not self.valid(), f"Attempting to overwrite a valid Datablock {self}. Missing 'overwrite' argument?"
        fs, _ = fsspec.url_to_fs(anchorhashpath)
        assert fs.isdir(anchorhashpath), f"Nonexistent hashpath {anchorhashpath}"
        self.log.verbose(f"Copying files from {anchorhashpath}: BEGIN")
        self._write_journal_entry(event="UNSAFE_copy_from:BEGIN", context=anchorhashpath, inline_context=True)
        try:
            if self.has_topics():
                for topic in self.topics():
                    dst_path = self.path(topic)
                    if dst_path is not None:
                        # File copy
                        if topicpaths is not None:
                            _src_path = topicpaths[topic]
                        else:
                            _src_path = os.path.join(topic, self.TOPICFILES[topic])
                        src_path = os.path.join(anchorhashpath, _src_path)
                        self.log.detailed(f"Copying file {src_path} to {dst_path}")
                        fscopy(src_path=src_path, dst_path=dst_path, recursive=False)
                    else:
                        # Dir copy
                        dst_path = self.dirpath(topic)
                        if topicpaths is not None:
                            _src_path = topicpaths[topic]
                        else:
                            _src_path = topic
                        src_path = os.path.join(anchorhashpath, _src_path)
                        src_fs, _ = fsspec.url_to_fs(src_path)
                        if src_fs.exists(src_path):
                            self.log.detailed(f"Copying directory {src_path} to {dst_path}")
                            fscopy(src_path=src_path, dst_path=dst_path, recursive=True)
            elif self.has_topic():
                dst_path = self.path()
                if dst_path is not None:
                    # File copy
                    if topicpaths is not None:
                        _src_path = topicpaths
                    else:
                        _src_path = self.TOPICFILE
                    src_path = os.path.join(anchorhashpath, _src_path)
                    self.log.detailed(f"Copying file {src_path} to {dst_path}")
                    fscopy(src_path=src_path, dst_path=dst_path, recursive=False)
                else:
                    # Dir copy
                    dst_path = self.dirpath()
                    if topicpaths is not None:
                        _src_path = topicpaths
                    else:
                        _src_path = ""
                    src_path = os.path.join(anchorhashpath, _src_path)
                    src_fs, _ = fsspec.url_to_fs(src_path)
                    if src_fs.exists(src_path):
                        self.log.detailed(f"Copying directory {src_path} to {dst_path}")
                        fscopy(src_path=src_path, dst_path=dst_path, recursive=True)
        
            self.log.verbose(f"Copying files from {anchorhashpath}: END")
            self._write_journal_entry(event="UNSAFE_copy_from:END", context=anchorhashpath, inline_context=True)
            if validate:
                assert self.valid(), f"Invalid Datablock after copy: {self}"
        except Exception as e:
            self.log.error(f"UNSAFE_copy_from: Error when trying to copy files from {anchorhashpath}")
            self.log.error(f"EXCEPTION: {e}")
            self._write_journal_entry(event="UNSAFE_copy_from:ERROR", context=anchorhashpath, inline_context=True)
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
                getter = eval_term(term)
            replacements[field.name] = getter
        config = replace(config, **replacements)
        self.log.detailed(f"Made {config=} from {spec=}")
        return config

    def leave_breadcrumbs_at_path(self, path):
        fs, _ = fsspec.url_to_fs(path)
        with fs.open(path, "w") as f:
            f.write("")
    
    #PATHS: BEGIN
    def path(
        self,
        topic=None,
        *,
        ensure_dirpath: bool = False,
    ):
        if topic is None:
            dirpath = self.dirpath()
            topicfiles = self.TOPICFILE if hasattr(self, 'TOPICFILE') else None
        else:
            dirpath = self.dirpath(topic)
            topicfiles = self.TOPICFILES[topic]
        if ensure_dirpath and dirpath is not None:
            self.ensure_path(dirpath)
        if isinstance(topicfiles, dict): 
            path = {topic: self.filepath(dirpath, topicfile) for topic, topicfile in topicfiles.items()}
        elif isinstance(topicfiles, list):
            path = [self.filepath(dirpath, topicfile) for topicfile in topicfiles]
        elif isinstance(topicfiles, str):
            path = self.filepath(dirpath, topicfiles)
        else:
            path = None
        self.log.detailed(f"{self.anchor}: path: {path}")
        return path
    
    def dirpath(
        self,
        topic=None,
        *,
        ensure: bool = False,
        list: bool = False,
    ):  
        hashpath = self.hashpath()
        if topic is not None:
            assert topic in self.TOPICFILES, f"Topic {repr(topic)} not in {self.TOPICFILES}"
            dirpath = os.path.join(hashpath, topic)
        else:
            dirpath = hashpath
        if ensure:
            fs, _ = fsspec.url_to_fs(dirpath)
            fs.makedirs(dirpath, exist_ok=True)
        if list:
            fs, _ = fsspec.url_to_fs(dirpath)
            return fs.ls(dirpath)
        return dirpath
    
    def filepath(
        self,
        dirpath,
        topicfile=None,
    ):
        if topicfile is None:
            path = None
        else:
            path = os.path.join(dirpath, topicfile) if topicfile is not None else None     
        return path
    
    def hashpath(self, *, ensure: bool = True):
        anchorpath = self.anchorpath()
        hashpath = os.path.join(anchorpath, self.hash)
        if ensure:
            fs, _ = fsspec.url_to_fs(hashpath)
            fs.makedirs(hashpath, exist_ok=True)
        return hashpath

    def ensure_path(self, path):
        fs, _ = fsspec.url_to_fs(path)
        fs.makedirs(path, exist_ok=True)
        return self

    def url(self, topic=None, *, redirect=None):
        path = self.path(topic)
        return make_google_cloud_storage_download_url(path)
    
    def paths(self):
        if self.has_topics:
            paths = {topic: self.path(topic) for topic in self.topics()}
        else:
            paths = self.path()
        return paths

    @property
    def anchor(self):
        anchor = (
            self.__module__
            + "."
            + self.__class__.__name__
        )
        return anchor

    @property
    def stump(self):
        return self.__class__.__name__

    def anchorpath(self):
        anchorpath = os.path.join(
            self.root,
            self.anchor,
        ) if self.anchored else self.root
        return anchorpath

    @property
    def anchorhash(self):
        anchorhash = os.path.join(
            self.anchor,
            self.hash,
        )
        return anchorhash

    @property
    def anchorhashpath(self):
        anchorhashpath = os.path.join(
            self.root,
            self.anchorhash,
        )
        return anchorhashpath


    @classmethod
    def _xanchorpath(cls, root, x, *, ensure: bool = False):
        xanchor = os.path.join(
            (
                cls.__module__
                + "."
                + cls.__name__
            ),
            f".{x}",
        )
        xanchorpath = os.path.join(
            root,
            xanchor,
        )
        if ensure:
            fs, _ = fsspec.url_to_fs(xanchorpath)
            fs.makedirs(xanchorpath, exist_ok=True)
        return xanchorpath
    
    def _xpath(self, x, ext=None, *, ensure: bool = True):
        xanchorpath = self._xanchorpath(self.root, x)
        xhashpath = os.path.join(
            xanchorpath,
            self.hash,
        )
        if ensure:
            fs, _ = fsspec.url_to_fs(xhashpath)
            fs.makedirs(xhashpath, exist_ok=True)
        if ext is None:
            ext = x
        xpath = os.path.join(xhashpath, f'{self.dt}.{ext}')
        return xpath
    
    ##REFACTOR: through _xanchorpath/_xpath: BEGIN
    @classmethod
    def _loganchorpath(cls, root):
        loganchor = os.path.join(
            (
                cls.__module__
                + "."
                + cls.__name__
            ),
            ".log",
        )
        loganchorpath = os.path.join(
            root,
            loganchor,
        )
        return loganchorpath

    @classmethod
    def _scopeanchorpath(cls, root):
        scopeanchor = os.path.join(
            (
                cls.__module__
                + "."
                + cls.__name__
            ),
            ".scope",
        )
        scopeanchorpath = os.path.join(
            root,
            scopeanchor,
        )
        return scopeanchorpath
    
    @classmethod
    def _stateanchorpath(cls, root):
        stateanchor = os.path.join(
            (
                cls.__module__
                + "."
                + cls.__name__
            ),
            ".state",
        )
        stateanchorpath = os.path.join(
            root,
            stateanchor,
        )
        return stateanchorpath

    def _logpath(self, *, ensure: bool = True):
        loganchorpath = self._loganchorpath(self.root)
        logdirpath = os.path.join(
            loganchorpath,
            self.hash,
        )
        if ensure:
            fs, _ = fsspec.url_to_fs(logdirpath)
            fs.makedirs(logdirpath, exist_ok=True)
        logpath = os.path.join(logdirpath, f'{self.dt}.log')
        return logpath

    def _scopepath(self, kind, *, ensure: bool = True):
        scopeanchorpath = self._scopeanchorpath(self.root)
        scopedirpath = os.path.join(
            scopeanchorpath,
            self.hash,
        )
        if ensure:
            fs, _ = fsspec.url_to_fs(scopedirpath)
            fs.makedirs(scopedirpath, exist_ok=True)
        if kind == 'yaml':
            scopepath = os.path.join(scopedirpath, f'{self.dt}.yaml')
        elif kind == 'parquet':
            scopepath = os.path.join(scopedirpath, f'{self.dt}.parquet')
        else:
            raise ValueError(f"Unknown path kind: {kind}")
        return scopepath
    
    def _statehashpath(self):
        stateanchorpath = self._stateanchorpath(self.root)
        return os.path.join(
            stateanchorpath,
            self.hash,
        )
    
    def _statepath(self, kind, *, ensure: bool = True):
        statehashpath = self._statehashpath()
        if ensure:
            fs, _ = fsspec.url_to_fs(statehashpath)
            fs.makedirs(statehashpath, exist_ok=True)
        if kind == 'yaml':
            statepath = os.path.join(statehashpath, f'{self.dt}.yaml')
        elif kind == 'parquet':
            statepath = os.path.join(statehashpath, f'{self.dt}.parquet')
        else:
            raise ValueError(f"Unknown path kind: {kind}")
        return statepath

    @staticmethod
    def _journalanchorpath(cls, root, *, ensure: bool = True):
        journalclassname = cls if isinstance(cls, str) else os.path.join(
            cls.__module__
            + "."
            + cls.__name__,
        )
        journalanchor = os.path.join(
            journalclassname,
            ".journal",
        )
        journalanchorpath = os.path.join(
            root,
            journalanchor,
        )
        if ensure:
            fs, _ = fsspec.url_to_fs(journalanchorpath)
            fs.makedirs(journalanchorpath, exist_ok=True)
        return journalanchorpath
    ##REFACTOR: through _xanchorpath/_xpath: END
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
    #LOG LEVEL: END


    #IDENTIFICATION: BEGIN
    #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
    # computed using the older version of these methods
    """
    
    """
    @staticmethod
    def is_specline(s):
        return isinstance(s, str) and (
            s.startswith('@') or s.startswith('$') or s.startswith('#')
        )
    
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
                gitrepo = DBXUSEWRKREPO if DBXUSEWRKREPO is not None else DBXGITREPO
                self._revision = gitrevision(log=self.log) if gitrepo is not None else None
                self.log.detailed(f"--------------> self._revision_: from gitrevision()")
            else:
                self.log.detailed(f"--------------> Using {self._revision_=}")
                self._revision = self._revision_
        return self._revision

    def __expand_spec__(self, expansion='repr'):
        """
            . expansion: 'repr'|'quote'|'handle'
                . specline:      str starting with '@', '$' or '#'
                . datablock: Datablock object
                . obj:       object
            'repr':
                . FULL reduction
                    |obj:    repr(obj)
            'handle':
                . DATABLOCK reduction
                    |datablock: datablock.handle()
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
        elif expansion == 'handle':
            for k, v in spec.items():
                value = getattr(self.cfg, k)
                if isinstance(value, Datablock):
                    _spec_[k] = value.handle()
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
        if self._root_ is not None:
            rootkwargs['root'] = self._root_
        if not self.anchored:
            rootkwargs['anchored'] = False
        if self._hash_ is not None:
            rootkwargs['hash'] = self._hash_
        return rootkwargs
    
    @functools.cached_property
    def _tailkwargs_(self):
        state = self.__getstate__()
        tailkwargs = {
            k: v
            for k, v in state.items()
            if k not in ['root', 'anchored', 'hash', 'spec']          
        }
        self.log.detailed(f"{self.anchor}: _tailkwargs_: {tailkwargs=}")
        return tailkwargs
    
    def __repr_from_kwargs__(self, kwargs, *, use_stump: bool = False):
        def cite(x):
            return repr(x) if isinstance(x, str) else x

        kwargstrs = [f"{k}={v}" for k, v in kwargs.items()]
        kwargsrepr = ', '.join(kwargstrs)
        if use_stump:
            _repr_ = f"{self.stump}({kwargsrepr})"
        else:
            _repr_ = f"{self.anchor}({kwargsrepr})"
        return _repr_
    
    def quote(self, *, deslash: bool = False):
        quoted_spec = self.__expand_spec__('quote')
        quote = "$" + self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': quoted_spec},
        })
        if deslash:
            quote = quote.replace('\\', '')
        self.log.detailed(f"quote: ------------> {quoted_spec=}")
        self.log.detailed(f"quote: ------------> {quote=}")
        return quote

    def handle(self, *, deslash: bool = False):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        repr_spec = self.__expand_spec__('handle')
        handle = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': repr_spec},
        })
        if deslash:
            handle = handle.replace('\\', '')
        self.log.detailed(f"handle: ------------> {repr_spec=}")
        self.log.detailed(f"handle: ------------>{handle=}")
        return handle

    def __repr__(self, *, deslash: bool = True):
        repr_spec = self.__expand_spec__('repr')
        r = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': repr_spec},
            **self._tailkwargs_,
        })
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
        """Returns ALL variables including explicit defaults and dynamically-supplied kwargs."""
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
        """Returns ONLY the dynamically-supplied kwargs and arguments not contained in the explicit parameters."""
        explicit_keys = set(self.__explicit_params__())
        return {k: v for k, v in self.__getstate__().items() if k not in explicit_keys}
    
    @property
    def hashstr(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        if hasattr(self, "TOPICFILES"):
            topics = [f"topic:{topic}={file}" for topic, file in self.TOPICFILES.items()]
        else:
            topics = ["topics:None"]
        hashstr = os.path.join(
            self.handle(),
            f"version={self.version}",
            *topics,
        )
        return hashstr
    
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
                self.log.detailed(f"hash: ---------===---------> {self.hashstr=} ---> hash: {self._hash}")
        return self._hash

    def getag(self, tag):
        new = self.set(tag=tag)
        new._write_journal_entry(event='tag')
        return new

    @property
    def tag(self):
        if not hasattr(self, '_tag'): 
            if self._tag_ is not None:
                self._tag = self._tag_
            else:
                self._tag = self.anchorhash
        return self._tag
    #IDENTIFICATION: END

    #JOURNAL: BEGIN
    def _write_journal_dict(self, name, data, *, add_credentials: bool = False):
        if add_credentials:
            data = copy.deepcopy(data)
            data['hash'] = self.hash
            data['datetime'] = self.dt
        #
        ypath = self._xpath(name, 'yaml')
        yfs, _ = fsspec.url_to_fs(ypath)
        write_yaml(data, ypath)
        assert yfs.exists(ypath), f"path {ypath} does not exist after writing"
        self.log.detailed(f"WROTE: {name.upper()}: yaml: {ypath}")
        #
        pqpath = self._xpath(name, 'parquet')
        pqfs, _ = fsspec.url_to_fs(pqpath)
        df = pd.DataFrame.from_records([{k: repr(v) for k, v in data.items()}])
        df.to_parquet(pqpath)
        assert pqfs.exists(pqpath), f"pqpath {pqpath} does not exist after writing"
        self.log.detailed(f"WROTE: {name.upper()}: parquet: {pqpath}")

    def _write_str(self, name, text):
        #
        path = self._xpath(name, 'txt')
        fs, _ = fsspec.url_to_fs(path)
        write_str(text, path)
        assert fs.exists(path), f"scopepath {path} does not exist after writing"
        self.log.detailed(f"WROTE: {name.upper()}: txt: {path}")

    def _write_journal_entry(self, event:str, *, context: str = None, inline_context: bool = False):
        self._write_journal_dict('spec', self.spec)
        self._write_journal_dict('dfn', self.dfn)
        self._write_journal_dict('kwargs', self.kwargs)
        self._write_str('quote', self.quote())
        self._write_str('repr', self.__repr__())
        self._write_str('handle', self.handle())
        self._write_str('hashstr', self.hashstr)
        if context is not None and not inline_context:
            self._write_str('context', context)
        #
        dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        filename = f"{self.hash}-{dt}"

        spec_path = self._xpath('spec', 'yaml')
        dfn_path = self._xpath('dfn', 'yaml')
        kwargs_path = self._xpath('kwargs', 'yaml')
        quote_path = self._xpath('quote', 'txt')
        handle_path = self._xpath('quote', 'txt')
        repr_path = self._xpath('repr', 'txt')
        hashstr_path = self._xpath('hashstr', 'txt')
        if context is not None and not inline_context:
            context_path = self._xpath('context', 'txt')
            context = context_path
        else:
            context_path = None
        #
        logpath = self._logpath()
        if logpath is not None:
            logfs, _ = fsspec.url_to_fs(logpath)
            has_log = logfs.exists(logpath)
        else:
            has_log = False
        #
        journal_path = os.path.join(self._journalanchorpath(self.__class__, self.root), f"{filename}.parquet")
        df = pd.DataFrame.from_records([{'datetime': dt,
                                         'build_datetime': self.build_dt,
                                         'version': self.version,
                                         'dbx_version': self.dbx_version,
                                         'revision': self.revision, 
                                         'root': self.root,
                                         'anchor': self.anchor,
                                         'hash': self.hash,
                                         'uuid': self.uuid,
                                         'tag': self.tag, 
                                         'log': logpath if has_log else None,
                                         'event': event,
                                         'spec': spec_path,
                                         'dfn': dfn_path,
                                         'kwargs': kwargs_path,
                                         'quote': quote_path,
                                         'handle': handle_path,
                                         'repr': repr_path,
                                         'hashstr': hashstr_path,
                                         'context': context,
                                         'gitrepo': DBXGITREPO,
                                         'wrkrepo': DBXUSEWRKREPO,
        }])
        df.to_parquet(journal_path)
        
        tagstr = f"with tag {repr(self.tag)} " if self.tag is not None else ""
        self.log.debug(f"WROTE JOURNAL entry for event {repr(event)} {tagstr}"
                         f"to journal_path {journal_path}")

    @staticmethod
    def Journal(cls, entry: int = None, *, root=None, **kwargs):
        if root is None:
            root = os.environ.get('DBXROOT')
        journaldirpath = Datablock._journalanchorpath(eval_term(cls), root)
        fs, _ = fsspec.url_to_fs(journaldirpath)
        files = list(fs.ls(journaldirpath))
        parquet_files = [f for f in files if f.endswith('.parquet')]

        log = Logger()
        log.detailed(f"READING JOURNAL: from {journaldirpath=}, files: {parquet_files}")
        if len(parquet_files) > 0:
            dfs = []
            for file in parquet_files:
                _df = pd.read_parquet(file)
                if 'revision' not in _df.columns:
                    _df = _df.rename(columns={'version': 'revision',})
                if 'kwargs' in _df.columns and 'state' not in _df.columns:
                    # Legacy entries: 'kwargs' was the state. 
                    # We map it to 'state' and also keep it as 'kwargs' (fallback).
                    _df['state'] = _df['kwargs']
                dfs.append(_df)
            df = pd.concat(dfs)
            leading = ['hash'] + (['uuid'] if 'uuid' in df.columns else []) + ['datetime']
            columns = leading + [c for c in df.columns if c not in set(leading + ['event'])] + ['event']
            df = df.sort_values('datetime', ascending=False)[columns].reset_index(drop=True)
            df = df.rename(columns={'build_log': 'log'})
        else:
            df = None
        journal = JournalFrame(df, **kwargs)
        if entry is not None:
            result = JournalEntry(journal.loc[entry].dropna())
        else:
            result = journal
        return result

    def journal(self, entry: int = None, **kwargs):
        return self.Journal(self.__class__,entry, root=self.root, **kwargs)
    #JOURNAL: END
    

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


class TorchMultithreadingDatablocksBuilder:
    def __init__(self, *, devices: list[str] = 'cuda', log: Logger = Logger()):
        if isinstance(devices, str):
            devices = [devices]
        self.devices = devices
        self.log = log

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        if len(blocks) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            progress_bar = tqdm.tqdm(total=len(blocks))
            block_lists = np.array_split(blocks, len(self.devices))
            block_offsets = np.cumsum([0] + [len(block_list) for block_list in block_lists])
            threads = [
                threading.Thread(target=self.__build_blocks__, args=(block_list, ctx_args, ctx_kwargs, block_offset, device, result_queue, done_queue, abort_event, progress_bar))
                for block_list, block_offset, device in zip(block_lists, block_offsets, self.devices)
            ]
            done_idxs = []
            for thread in threads:
                thread.start()
            while len(done_idxs) < len(blocks):
                success, idx, payload = result_queue.get()
                if success:
                    done_idxs.append(idx)
                    e = None
                else:
                    e = payload
                    self.log.info(f"Received error from block with index {idx}: {blocks[idx]}. Abandoning result_queue polling.")
                    break
            self.log.debug(f"Production loop done, feeding done_queue")
            for _ in range(len(self.devices)):
                done_queue.put(None)
            self.log.debug(f"Joining threads")
            for thread in threads:
                thread.join()
            if e is not None:
                self.log.verbose("Raising exception")
                raise e
            self.log.debug("Threads successfully joined")
        return blocks
    
    def __build_blocks__(self, blocks: Sequence[Datablock], ctx_args, ctx_kwargs, offset: int, device: str, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event, progress_bar):
        self.log.debug(f"Building {len(blocks)} feature blocks on device: {device}")
        device_ctx_args, device_ctx_kwargs = self.__args_kwargs_to_device__(ctx_args, ctx_kwargs, device)
        for i, block in enumerate(blocks):
            exception = None
            try:
                if abort_event.is_set():
                    break
                block.to(device).build(*device_ctx_args, **device_ctx_kwargs).to('cpu')
            except Exception as e:
                exception = e
                self.log.info(f"ERROR building feature block {block} on device: {device}")
            if exception is not None:
                result_queue.put((False, offset+i, exception))
                break
            result_queue.put((True, offset+i, None))
            progress_bar.update(1)
        del device_ctx_args, device_ctx_kwargs
        gc.collect()
        if exception is None:
            self.log.debug(f"Done building {len(blocks)} feature blocks on device: {device}")
        else:
            self.log.debug(f"Abandoning building {len(blocks)} feature blocks on device: {device} due to an exception")
        self.log.debug(f"Waiting on the done_queue on device: {device}")
        while True:
            item = done_queue.get()
            if item is None:
                self.log.debug(f"Done message received on the done_queue on device: {device}")
                break

    def __args_kwargs_to_device__(self, args, kwargs, device):
        device_args = [arg.to(device) if hasattr(arg, 'to') else arg for arg in args]
        device_kwargs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in kwargs.items()}
        return device_args, device_kwargs
    

class TorchMultiprocessingDatablocksBuilder(TorchMultithreadingDatablocksBuilder):
    def __init__(self, *, devices: list[str] = None, log: Logger = Logger()):
        if isinstance(devices, str):
            devices = [devices]
        self.devices = devices
        self.log = log

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        if len(blocks) > 0:
            result_queue = mp.Queue()
            done_queue = mp.Queue()
            abort_event = mp.Event()
            progress_bar = tqdm.tqdm(total=len(blocks))
            block_lists = np.array_split(blocks, len(self.devices))
            block_offsets = np.cumsum([0] + [len(block_list) for block_list in block_lists])
            processes = [
                mp.Process(target=self.__build_blocks__, args=(block_list, ctx_args, ctx_kwargs, block_offset, f"{i}", device, result_queue, done_queue, abort_event))
                for i, (block_list, block_offset, device) in enumerate(zip(block_lists, block_offsets, self.devices))
            ]
            self.log.verbose(f"Building {len(blocks)} feature blocks with {len(self.devices)} processes")
            done_idxs = []
            exc = None
            try:
                for process in processes:
                    process.start()
                for block in blocks:
                    del block
                gc.collect()
                while len(done_idxs) < len(blocks):
                    pexc, ptbstr = None, None
                    success, proc, idx, payload = result_queue.get()
                    if success:
                        done_idxs.append(idx)
                        progress_bar.update(1)
                    else:
                        pexc, ptbstr = payload
                        self.log.info(f"Received exception from process {proc}, block with index {idx}: {blocks[idx]}")
                        self.log.info(f"Exception: {pexc}")
                        self.log.info(f"Traceback:\n{ptbstr}")
                        self.log.info(f"Abandoning result_queue polling.")
                        break
                self.log.debug(f"Production loop done")
            except Exception as e:
                exc = e
                self.log.info(f"Caught exception in production loop\nException: {e}")
                tbstr = '\n'.join(tb.format_tb(e.__traceback__))
                self.log.info(f"Traceback:\n{tbstr}")
                abort_event.set()
            finally:
                self.log.debug(f"Feeding done_queue")
                for _ in self.devices:
                    done_queue.put(None)
                self.log.debug(f"Joining processes")
                for process in processes:
                    process.join()
                self.log.debug("Processes successfully joined")
            if pexc is not None:
                self.log.verbose(f"Reraising exception from process {proc}, block {idx}: {blocks[idx]}")
                raise(pexc)
            if exc is not None:
                self.log.verbose("Reraising production loop exception")
                raise(exc)
        return blocks
    
    def __build_blocks__(self, blocks: Sequence[Datablock], ctx_args, ctx_kwargs, offset: int, process: str, device: str, result_queue: mp.Queue, done_queue: mp.Queue, abort_event: mp.Event):
        self.log.debug(f"Building {len(blocks)} feature blocks on process: {process}, device: {device}")
        if device is not None:
            device_ctx_args, device_ctx_kwargs = self.__args_kwargs_to_device__(ctx_args, ctx_kwargs, device)
        else:
            device_ctx_args, device_ctx_kwargs = ctx_args, ctx_kwargs
        exception = None
        for i, block in enumerate(blocks):
            exception = None
            try:
                if abort_event.is_set():
                    break
                block.to(device).build(*device_ctx_args, **device_ctx_kwargs).to('cpu')
            except Exception as e:
                exception = e
                self.log.info(f"ERROR building datablock {block} on process: {process}, device: {device}")
            finally:
                del block
                gc.collect()
            if exception is not None:
                tbstr = '\n'.join(tb.format_tb(exception.__traceback__))
                result_queue.put((False, process, offset+i, (exception, tbstr)))
                break
            result_queue.put((True, process, offset+i, None))
        del device_ctx_args, device_ctx_kwargs
        gc.collect()
        if exception is None:
            self.log.debug(f"Done building {len(blocks)} datablocks on process: {process}, device: {device}")
        else:
            self.log.debug(f"Abandoning building {len(blocks)} datablocks on process: {process}, device: {device} due to an exception")
        self.log.debug(f"Waiting on the done_queue on process: {process}, device: {device}")
        while True:
            item = done_queue.get()
            if item is None:
                self.log.debug(f"Done message received on the done_queue on process: {process}, device: {device}")
                break


class _CallableExecutorBase_:
    """
    Abstract base that implements the fan-out / collect / join scaffold shared by
    MultithreadingCallableExecutor and MultiprocessingCallableExecutor.

    Wire protocol (result_queue items published by workers):
        success=True  → (True,  worker_idx, item_idx, payload)
        success=False → (False, worker_idx, item_idx, (exception, tbstr))

    Subclasses must implement:
        _n_workers  → int
        _make_queue()   → a Queue-like object
        _make_event()   → an Event-like object
        _make_worker(target, args) → a Thread/Process-like object with .start()/.join()
        _after_start(items, workers) → called after workers are started (default: no-op)
    """

    @property
    def _n_workers(self) -> int:
        raise NotImplementedError

    def _make_queue(self):
        raise NotImplementedError

    def _make_event(self):
        raise NotImplementedError

    def _make_worker(self, target, args):
        raise NotImplementedError

    def _after_start(self, items, workers):
        """Hook called in the main process right after all workers have been started."""
        pass

    # ------------------------------------------------------------------
    # Worker-side helper (called inside each worker)
    # ------------------------------------------------------------------
    def _run_items(self, items, ctx_args, ctx_kwargs, offset, worker_idx,
                   result_queue, done_queue, abort_event):
        """Run each item in *items* sequentially, accumulating results into
        batches of *self.batch_size* before putting them on *result_queue*.
        This amortises IPC / queue overhead when batch_size > 1.

        Wire protocol:
            success → (True,  worker_idx, [(item_idx, payload), ...])
            failure → (False, worker_idx, item_idx, (exception, tbstr))
        """
        worker_label = self._worker_label(worker_idx)
        self.log.debug(f"Executing {len(items)} callables on {worker_label}")
        batch_size = self.batch_size if (self.batch_size is not None and self.batch_size > 1) else 1
        exception = None
        batch = []  # list of (item_idx, payload)
        for i, item in enumerate(items):
            exception = None
            try:
                if abort_event.is_set():
                    break
                payload = item(*ctx_args, **ctx_kwargs)
                batch.append((offset + i, payload))
            except Exception as e:
                exception = e
                self.log.info(f"ERROR executing callable {offset+i} on {worker_label}")
            finally:
                self._after_item(item)
            if exception is not None:
                # Flush any accumulated results before reporting the error
                if batch:
                    result_queue.put((True, worker_idx, batch))
                    batch = []
                tbstr = '\n'.join(tb.format_tb(exception.__traceback__))
                result_queue.put((False, worker_idx, offset + i, (exception, tbstr)))
                break
            if len(batch) >= batch_size:
                result_queue.put((True, worker_idx, batch))
                batch = []
        # Flush any remaining results
        if batch and exception is None:
            result_queue.put((True, worker_idx, batch))
        gc.collect()
        if exception is None:
            self.log.debug(f"Done executing {len(items)} callables on {worker_label}")
        else:
            self.log.debug(f"Abandoning callables on {worker_label} due to exception")
        self.log.debug(f"Waiting on done_queue on {worker_label}")
        while True:
            if done_queue.get() is None:
                self.log.debug(f"Done signal received on {worker_label}")
                break

    def _after_item(self, item):
        """Hook called after each item is processed (e.g. to del block and gc)."""
        pass

    def _worker_label(self, worker_idx) -> str:
        return f"worker {worker_idx}"

    def _desc(self, streaming: bool) -> str:
        """Helper to format the progress bar description."""
        label = self._worker_label(0).split()[0].capitalize() # "Thread", "Process", etc.
        prefix = "Streaming " if streaming else ""
        desc = f"{prefix}{label}"
        if hasattr(self, 'tag') and self.tag:
            desc = f"{desc}: {self.tag}"
        if hasattr(self, 'batch_size') and self.batch_size is not None:
            desc = f"{desc} [bs={self.batch_size}]"
        return desc

    # ------------------------------------------------------------------
    # Main-process driver
    # ------------------------------------------------------------------
    def exec_callables(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute all callables and return results as a flat list.

        When *batch_size* is set, workers accumulate that many results before
        sending them back to the main process, amortising IPC overhead.  The
        progress bar advances in bursts of *batch_size* to reflect this.

        Always returns a plain list regardless of *batch_size*.  Use
        :meth:`exec_callables_streaming` explicitly when you need a generator.
        """
        payloads = [None] * len(callables)
        if len(callables) > 0:
            result_queue = self._make_queue()
            done_queue   = self._make_queue()
            abort_event  = self._make_event()
            progress_bar = tqdm.tqdm(total=len(callables), desc=self._desc(streaming=False))
            callable_lists   = np.array_split(callables, self._n_workers)
            callable_offsets = np.cumsum([0] + [len(cl) for cl in callable_lists])
            workers = [
                self._make_worker(
                    target=self._run_items,
                    args=(cl, ctx_args, ctx_kwargs, off, idx,
                          result_queue, done_queue, abort_event),
                )
                for idx, (cl, off) in enumerate(zip(callable_lists, callable_offsets))
            ]
            done_count = 0
            exc = None
            pexc = None
            try:
                for w in workers:
                    w.start()
                self._after_start(callables, workers)
                # Progress bar is created AFTER forking so child processes
                # do not inherit a live tqdm instance and redraw it on exit.
                progress_bar = tqdm.tqdm(total=len(callables), desc=self._desc(streaming=False))
                while done_count < len(callables):
                    msg = result_queue.get()
                    if msg[0]:  # success: (True, worker_idx, [(item_idx, payload), ...])
                        _, worker_idx, batch = msg
                        for item_idx, item_payload in batch:
                            payloads[item_idx] = item_payload
                            done_count += 1
                        progress_bar.update(len(batch))
                    else:       # failure: (False, worker_idx, item_idx, (exc, tbstr))
                        _, worker_idx, item_idx, (pexc, ptbstr) = msg
                        self.log.info(
                            f"Received exception from {self._worker_label(worker_idx)}, "
                            f"callable {item_idx}. Abandoning result_queue polling."
                        )
                        self.log.info(f"Exception: {pexc}")
                        self.log.info(f"Traceback:\n{ptbstr}")
                        abort_event.set()
                        break
                self.log.debug("Production loop done")
            except Exception as e:
                exc = e
                self.log.info(f"Exception in production loop: {e}")
                tbstr = '\n'.join(tb.format_tb(e.__traceback__))
                self.log.info(f"Traceback:\n{tbstr}")
                abort_event.set()
            finally:
                self.log.debug("Feeding done_queue")
                for _ in workers:
                    done_queue.put(None)
                self.log.debug("Joining workers")
                for w in workers:
                    w.join()
                self.log.debug("Workers successfully joined")
            if pexc is not None:
                self.log.verbose("Reraising exception from worker")
                raise pexc
            if exc is not None:
                self.log.verbose("Reraising production-loop exception")
                raise exc
        return payloads

    def exec_callables_streaming(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute callables and yield results in **input order**.

        A reorder buffer holds out-of-order arrivals from parallel workers;
        items are only yielded once all predecessors have been received.

        When *batch_size* > 1, results are yielded in lists of exactly
        *batch_size* items (or fewer for the final group), assembled in
        input order across worker-batch boundaries.  When *batch_size* is
        None or 1, individual payloads are yielded one at a time.

        The progress bar advances by the size of each received IPC batch,
        giving bursty updates that mirror the actual IPC rhythm.
        """
        if len(callables) > 0:
            result_queue = self._make_queue()
            done_queue   = self._make_queue()
            abort_event  = self._make_event()
            callable_lists   = np.array_split(callables, self._n_workers)
            callable_offsets = np.cumsum([0] + [len(cl) for cl in callable_lists])
            workers = [
                self._make_worker(
                    target=self._run_items,
                    args=(cl, ctx_args, ctx_kwargs, off, idx,
                          result_queue, done_queue, abort_event),
                )
                for idx, (cl, off) in enumerate(zip(callable_lists, callable_offsets))
            ]
            done_count = 0
            e = None
            try:
                for w in workers:
                    w.start()
                self._after_start(callables, workers)
                # Progress bar is created AFTER forking so child processes
                # do not inherit a live tqdm instance and redraw it on exit.
                progress_bar = tqdm.tqdm(total=len(callables), desc=self._desc(streaming=True))
                # Reorder buffer: holds payloads that arrived before their
                # predecessors, keyed by global item index.
                pending = {}        # item_idx -> payload
                next_to_yield = 0   # index of the next item to emit
                emit_buf = []       # accumulator for batch_size > 1 mode
                while done_count < len(callables):
                    msg = result_queue.get()
                    if msg[0]:  # success: (True, worker_idx, [(item_idx, payload), ...])
                        _, worker_idx, batch = msg
                        done_count += len(batch)
                        progress_bar.update(len(batch))
                        for item_idx, payload in batch:
                            pending[item_idx] = payload
                        # Drain pending in strict input order
                        while next_to_yield in pending:
                            p = pending.pop(next_to_yield)
                            next_to_yield += 1
                            if self.batch_size is not None and self.batch_size > 1:
                                emit_buf.append(p)
                                if len(emit_buf) >= self.batch_size:
                                    yield emit_buf
                                    emit_buf = []
                            else:
                                yield p
                    else:       # failure: (False, worker_idx, item_idx, (exc, tbstr))
                        _, worker_idx, item_idx, (e, ptbstr) = msg
                        abort_event.set()
                        break
                # Yield any remainder (last partial batch)
                if emit_buf:
                    yield emit_buf
            finally:
                for _ in workers:
                    done_queue.put(None)
                for w in workers:
                    w.join()
                if e is not None:
                    raise e
        else:
            return
            yield  # make this a generator

    def execute(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute all callables and return results as a flat list (same as exec_callables)."""
        return self.exec_callables(callables, *ctx_args, **ctx_kwargs)

    def execute_streaming(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute callables and yield results in input order (same as exec_callables_streaming)."""
        return self.exec_callables_streaming(callables, *ctx_args, **ctx_kwargs)


class MultithreadingCallableExecutor(_CallableExecutorBase_):
    def __init__(self, *, n_workers: int, batch_size: int = None, tag: str = "", log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log

    @property
    def _n_workers(self) -> int:
        return self.n_workers

    def _make_queue(self):
        return queue.Queue()

    def _make_event(self):
        return threading.Event()

    def _make_worker(self, target, args):
        return threading.Thread(target=target, args=args)

    def _worker_label(self, worker_idx) -> str:
        return f"thread {worker_idx}"


def _mp_worker_fn(target, args):
    """Module-level wrapper used by MultiprocessingCallableExecutor.

    Defined at module level (not as a closure) so it is picklable by name,
    which is required when multiprocessing uses the 'spawn' or 'forkserver'
    start method.  Sets TQDM_DISABLE so nested executors inside the worker
    do not produce progress bars.
    """
    import os
    os.environ['TQDM_DISABLE'] = '1'
    target(*args)


class MultiprocessingCallableExecutor(_CallableExecutorBase_):
    def __init__(self, *, n_workers: int, batch_size: int = None, tag: str = "", log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log

    @property
    def _n_workers(self) -> int:
        return self.n_workers

    def _make_queue(self):
        return mp.Queue()

    def _make_event(self):
        return mp.Event()

    def _make_worker(self, target, args):
        # Pass target and args explicitly so the module-level _mp_worker_fn
        # can be pickled by name (required for spawn/forkserver start methods).
        return mp.Process(target=_mp_worker_fn, args=(target, args))

    def _worker_label(self, worker_idx) -> str:
        return f"process {worker_idx}"

    def _after_start(self, items, workers):
        """Release main-process references so forked memory can be reclaimed."""
        for item in items:
            del item
        gc.collect()

    def _after_item(self, item):
        """Delete item reference inside the worker after each iteration."""
        del item
        gc.collect()


class RayCallableExecutor:
    def __init__(self, *, n_workers, batch_size: int = None, tag: str = "", revision=None, conda=None, log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log
        self.workers = [remote(revision=revision, conda=conda) for _ in range(n_workers)]

    def execute(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute all callables and return results as a flat list (same as exec_callables)."""
        return self.exec_callables(callables, *ctx_args, **ctx_kwargs)

    def execute_streaming(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute callables and yield results in input order (same as exec_callables_streaming)."""
        return self.exec_callables_streaming(callables, *ctx_args, **ctx_kwargs)

    def exec_callables(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        if len(callables) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            
            # Split callables among workers
            callable_lists = np.array_split(callables, self.n_workers)
            callable_offsets = np.cumsum([0] + [len(callable_list) for callable_list in callable_lists])
            
            threads = [
                threading.Thread(target=self.__exec_callables_batched__, 
                                 args=(worker, callable_list, ctx_args, ctx_kwargs, callable_offset, thread_idx, result_queue, done_queue, abort_event))
                for thread_idx, (worker, callable_list, callable_offset) in enumerate(zip(self.workers, callable_lists, callable_offsets))
            ]
            
            payloads = [None] * len(callables)
            done_idxs = []
            for thread in threads:
                thread.start()
                
            label = "Ray Batched"
            if self.tag:
                label = f"{label}: {self.tag}"
            if self.batch_size is not None:
                label = f"{label} [bs={self.batch_size}]"
            progress_bar = tqdm.tqdm(total=len(callables), desc=label)
            e = None
            while len(done_idxs) < len(callables):
                success, worker_idx, callable_idx, payload = result_queue.get()
                if success:
                    done_idxs.append(callable_idx)
                    payloads[callable_idx] = payload
                    progress_bar.update(1)
                else:
                    e = payload
                    self.log.info(f"Received error from callable with {callable_idx=} on worker {worker_idx}. Abandoning result_queue polling.")
                    break
                
            self.log.debug(f"Production loop done, feeding done_queue")
            for _ in range(self.n_workers):
                done_queue.put(None)
            self.log.debug(f"Joining threads")
            for thread in threads:
                thread.join()
                
            if e is not None:
                self.log.verbose("Raising exception")
                raise e
            self.log.debug("Workers successfully joined")
            return payloads
        return []

    def exec_callables_streaming(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        if len(callables) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            
            label = "Ray Streaming"
            if self.tag:
                label = f"{label}: {self.tag}"
            if self.batch_size is not None:
                label = f"{label} [bs={self.batch_size}]"
            progress_bar = tqdm.tqdm(total=len(callables), desc=label)
            
            # Split callables among workers
            callable_lists = np.array_split(callables, self.n_workers)
            callable_offsets = np.cumsum([0] + [len(callable_list) for callable_list in callable_lists])
            
            threads = [
                threading.Thread(target=self.__exec_callables_sequential__, 
                                 args=(worker, callable_list, ctx_args, ctx_kwargs, callable_offset, thread_idx, result_queue, done_queue, abort_event))
                for thread_idx, (worker, callable_list, callable_offset) in enumerate(zip(self.workers, callable_lists, callable_offsets))
            ]
            
            for thread in threads:
                thread.start()
                
            e = None
            done_count = 0
            batch = []
            try:
                while done_count < len(callables):
                    success, worker_idx, callable_idx, payload = result_queue.get()
                    if success:
                        done_count += 1
                        progress_bar.update(1)
                        if self.batch_size is not None and self.batch_size > 1:
                            batch.append(payload)
                            if len(batch) >= self.batch_size:
                                yield batch
                                batch = []
                        else:
                            yield payload
                    else:
                        e = payload
                        abort_event.set()
                        break
                if batch:
                    yield batch
            finally:
                for _ in range(self.n_workers):
                    done_queue.put(None)
                for thread in threads:
                    thread.join()
                if e is not None:
                    raise e
        else:
            return
            yield # make it a generator

    def __exec_callables_batched__(self, worker, callables: Sequence[Callable], ctx_args, ctx_kwargs, offset: int, thread_idx: int, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event):
        self.log.debug(f"Executing batch of {len(callables)} callables on worker {thread_idx}")
        try:
            batch_size = self.batch_size or len(callables)
            for i in range(0, len(callables), batch_size):
                if abort_event.is_set():
                    break
                chunk = callables[i : i + batch_size]
                batch_args = [(c, ctx_args, ctx_kwargs) for c in chunk]
                results = worker.run_batch(batch_args)
                for j, res in enumerate(results):
                    result_queue.put((True, thread_idx, offset + i + j, res))
        except Exception as e:
            result_queue.put((False, thread_idx, offset, e)) # reported at the offset of the batch
        
        while True:
            if done_queue.get() is None:
                break

    def __exec_callables_sequential__(self, worker, callables: Sequence[Callable], ctx_args, ctx_kwargs, offset: int, thread_idx: int, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event):
        self.log.debug(f"Executing {len(callables)} callables on worker {thread_idx}")
        batch_size = self.batch_size or 1
        exception = None
        for i in range(0, len(callables), batch_size):
            if abort_event.is_set():
                break
            chunk = callables[i : i + batch_size]
            try:
                if batch_size > 1:
                    batch_args = [(c, ctx_args, ctx_kwargs) for c in chunk]
                    results = worker.run_batch(batch_args)
                    for j, res in enumerate(results):
                        result_queue.put((True, thread_idx, offset + i + j, res))
                else:
                    callable = chunk[0]
                    self.log.detailed(f"EXECUTING callable {i+offset} on worker {thread_idx}: {callable}")
                    payload = worker.run(callable, *ctx_args, **ctx_kwargs)
                    self.log.detailed(f"EXECUTED callable {i+offset}: result: {payload}")
                    result_queue.put((True, thread_idx, offset + i, payload))
            except Exception as e:
                exception = e
                self.log.info(f"ERROR executing callable context on worker {thread_idx}")
                result_queue.put((False, thread_idx, offset + i, e))
                break
        
        gc.collect()
        if exception is None:
            self.log.debug(f"Done executing {len(callables)} callables on worker {thread_idx}")
        else:
            self.log.debug(f"Abandoning executing {len(callables)} callables on worker {thread_idx} due to an exception")
        
        self.log.debug(f"Waiting on the done_queue on worker {thread_idx}")
        while True:
            item = done_queue.get()
            if item is None:
                self.log.debug(f"Done message received on the done_queue on worker {thread_idx}")
                break


class InlineCallableExecutor:
    """Executes callables sequentially in the local process."""
    def __init__(self, *, n_workers: int = 1, batch_size: int = None, tag: str = "", log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log

    def execute(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute all callables and return results as a flat list (same as exec_callables)."""
        return self.exec_callables(callables, *ctx_args, **ctx_kwargs)

    def execute_streaming(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute callables and yield results in input order (same as exec_callables_streaming)."""
        return self.exec_callables_streaming(callables, *ctx_args, **ctx_kwargs)

    def exec_callables(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        payloads = []
        if len(callables) > 0:
            label = "Inline"
            if self.tag:
                label = f"{label}: {self.tag}"
            if self.batch_size is not None:
                label = f"{label} [bs={self.batch_size}]"
            progress_bar = tqdm.tqdm(total=len(callables), desc=label)
            for i, item in enumerate(callables):
                try:
                    payload = item(*ctx_args, **ctx_kwargs)
                    payloads.append(payload)
                    progress_bar.update(1)
                except Exception as e:
                    self.log.info(f"ERROR executing callable {i}")
                    raise e
            gc.collect()
        return payloads

    def exec_callables_streaming(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        if len(callables) > 0:
            label = "Inline Streaming"
            if self.tag:
                label = f"{label}: {self.tag}"
            if self.batch_size is not None:
                label = f"{label} [bs={self.batch_size}]"
            progress_bar = tqdm.tqdm(total=len(callables), desc=label)
            batch = []
            for i, item in enumerate(callables):
                try:
                    payload = item(*ctx_args, **ctx_kwargs)
                    progress_bar.update(1)
                    if self.batch_size is not None and self.batch_size > 1:
                        batch.append(payload)
                        if len(batch) >= self.batch_size:
                            yield batch
                            batch = []
                    else:
                        yield payload
                except Exception as e:
                    self.log.info(f"ERROR executing callable {i}")
                    raise e
            if batch:
                yield batch
            gc.collect()
        else:
            return
            yield

def _build_block(block, *args, **kwargs):
    return block.build(*args, **kwargs)

class MultithreadingDatablocksBuilder:
    """Builds Datablocks concurrently using threads, via MultithreadingCallableExecutor."""

    def __init__(self, *, n_workers: int = 1, log: Logger = Logger()):
        self.n_workers = n_workers
        self.log = log
        self._executor = MultithreadingCallableExecutor(n_workers=n_workers, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [functools.partial(_build_block, block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks


class MultiprocessingDatablocksBuilder:
    """Builds Datablocks concurrently using processes, via MultiprocessingCallableExecutor."""

    def __init__(self, *, n_workers: int = 1, log: Logger = Logger()):
        self.n_workers = n_workers
        self.log = log
        self._executor = MultiprocessingCallableExecutor(n_workers=n_workers, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [functools.partial(_build_block, block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks


class RayDatablocksBuilder:
    def __init__(self, *, n_workers: int = 1, revision=None, conda=None, log: Logger = Logger()):
        self.n_workers = n_workers
        self.log = log
        self.executor = RayCallableExecutor(n_workers=n_workers, revision=revision, conda=conda, log=log)

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
    def __init__(self, *, n_workers: int = 1, log: Logger = Logger()):
        self.n_workers = n_workers
        self.log = log
        self._executor = InlineCallableExecutor(n_workers=n_workers, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        callables = [functools.partial(_build_block, block) for block in blocks]
        self._executor.exec_callables(callables, *ctx_args, **ctx_kwargs)
        return blocks

    

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
                import shutil
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
            import types
            if isinstance(val, types.ModuleType):
                return val
            
            # Universal proxying for all non-primitives.
            # Expanded call site with nested class as requested.
            # If actor creation fails (e.g. due to pickling issues), we fall back to returning by value.
            try:
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
            import os
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

        is_callable, value = ray.get(self._handle.info.remote(name))
        
        if is_callable:
            def wrapper(*args, **kwargs):
                res = ray.get(self._handle.call.remote(name, *args, **kwargs))
                return self._unwrap_or_proxy(res)
            return wrapper
        else:
            return self._unwrap_or_proxy(value)

    def __getstate__(self):
        """
        Return the state of the remote object.
        """
        return ray.get(self._handle.call.remote('__getstate__'))

    def _unwrap_or_proxy(self, val):
        if isinstance(val, ray.actor.ActorHandle):
            return Remote(val) # Recursive wrapping
        return val

    def run(self, func, *args, **kwargs):
        """
        Execute a callable on the remote actor.
        """
        res = ray.get(self._handle.apply.remote(func, *args, **kwargs))
        return self._unwrap_or_proxy(res)

    def run_batch(self, funcs_args_kwargs):
        """
        Execute a sequence of (func, args, kwargs) on the remote actor in one round-trip.
        """
        results = ray.get(self._handle.apply_batch.remote(funcs_args_kwargs))
        return [self._unwrap_or_proxy(res) for res in results]


def remote(*, revision=None, slurm=None, conda=None, log: Logger = Logger()):
    """
    Instantiate a remote dbx interpreter and return a Remote handle to it.
    """
    dbx_env = {k: v for k, v in os.environ.items() if k.startswith('DBX')}
    
    if DBXUSEWRKREPO is not None:
        dbx_env['DBXGITREPO'] = DBXUSEWRKREPO

    # If we are using a remote cluster, any path in /tmp on the login node will be inaccessible to workers.
    # We revert to the original repository path (usually in /home) which is shared.
    if slurm:
        dbx_env['DBXGITREPO'] = _DBXGITREPO_

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


class UNSAFE_datablock_journal_puller:
    def __init__(self, datablock_classname, idx, *, throw=True, clear=False, log=Logger(name="UNSAFE_datablock_journal_puller")):
        self.datablock_classname = datablock_classname
        self.idx = idx
        self.throw = throw
        self.clear = clear
        self.log = log
        
    def __call__(self, journal, datablocks=None):
        if datablocks is not None:
            datablock_handles = [datablock.handle() for datablock in datablocks]
        else:
            datablock_handles = None
        try:
            dbk = None
            anchorhashpath = None
            entry = journal(journal.index[self.idx])
            spec = entry.read('spec')
            dbk = eval_term(quotefn(self.datablock_classname, spec=spec))
            anchorhashpath = entry.anchorhashpath
            if datablock_handles is not None:
                if dbk.handle() in datablock_handles:
                    self.log.debug(f"Skipping datablock {dbk.handle()}: not in datablocks")
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
