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

import numpy as np

import fsspec

from scipy.stats import qmc

import pandas as pd
import torch
import torch.multiprocessing as mp


__eval__ = __builtins__['eval']


DBXGITREPO = os.environ.get('DBXGITREPO')
if DBXGITREPO is None:
    try:
        import git
        _repo = git.Repo('.', search_parent_directories=True)
        DBXGITREPO = _repo.working_tree_dir
    except (ImportError, Exception):
        pass
_DBXGITREPO_ = DBXGITREPO
DBXWRKREPO = None
DBXWRKROOT = None

def gitwrkreposetup(revision=None, *, reason: str = "", log=None):
    if log is None:
        log = Logger(name="gitwrkreposetup")
    global DBXGITREPO
    tbstr = ''.join(tb.format_stack())
    log.silent(f"gitwrkreposetup callstack:\n{tbstr}")
    global DBXWRKREPO
    global DBXWRKROOT
    use_dbxgitrepo = os.environ.get('DBXGITREPO') or revision is not None
    if use_dbxgitrepo and DBXWRKREPO is None:
        if DBXGITREPO is None:
            raise ValueError("DBXGITREPO is not set and could not be detected. Cannot setup temporary wrkrepo.")
        log.info(f"SETTING UP TEMPORARY DBXWRKROOT from {DBXGITREPO=} {reason} with revision {revision}")
        DBXWRKROOT = tempfile.TemporaryDirectory()
        package = os.path.basename(DBXGITREPO)
        globals()['DBXWRKREPO'] = os.path.join(DBXWRKROOT.name, package)
        _pkgpath = gitclone(DBXGITREPO, globals()['DBXWRKREPO'])
        assert globals()['DBXWRKREPO'] == _pkgpath
        sys.path.insert(0, globals()['DBXWRKREPO'])
        # ENSURE that any child process gets the same DBXGITREPO and does not spawn its own temporary repo
        os.environ['DBXGITREPO'] = globals()['DBXWRKREPO']
        if 'DBXWRKREPO' in os.environ:
            del os.environ['DBXWRKREPO']
        log.info(f"DBXWRKROOT: {DBXWRKROOT.name}, DBXWRKREPO: {globals()['DBXWRKREPO']}")
        log.silent(f"os.environ['DBXGITREPO'] = {os.environ['DBXGITREPO']}, os.environ['DBXWRKREPO'] = {os.environ.get('DBXWRKREPO')}")
    if revision is not None:
        gitcheckout(globals()['DBXWRKREPO'], revision)
        log.info(f"Checked out {globals()['DBXWRKREPO']} to revision {revision}")
    else:
        log.info(f"Using {globals()['DBXWRKREPO']} at HEAD")


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
            if result:
                self.allowed.append(name.upper())
        
        for argname in _defaults_.keys():
            set_arg(argname, locals())

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

    def __repr__(self):
        return f"JournalEntry:{self.anchor}/{self.hash}"

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
                result = {k: v.replace('\\', '') for k, v in result.items()}
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
        gitwrkreposetup(revision=revision, reason=f"because of evaluating a JournalEntry field {thing}")
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
            self.logger.info(f"Instantiating {self} with revision from journal entry {revision}")
        else:
            self.logger.info(f"Instantiating {self} with revision {revision}")
        if gitrepo == 'journal_entry':
            gitrepo = self.gitrepo
            self.logger.info(f"Instantiating {self} with gitrepo from journal entry {gitrepo}")
        else:
            self.logger.info(f"Instantiating {self} with gitrepo {gitrepo}")
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
    repopath = DBXWRKREPO if DBXWRKREPO is not None else DBXGITREPO
    if repopath is not None:
        repo = git.Repo(repopath)
        if repo.is_dirty():
            raise ValueError(f"Dirty git repo: {repopath}: commit your changes: {DBXWRKREPO=}, {DBXGITREPO=}")
        revision = repo.head.commit.hexsha
        log.detailed(f"Obtained git revision for git repo {repopath}: {revision}, {DBXWRKREPO=}, {DBXGITREPO=}")
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
    @dataclass
    class Bid: #BlockId
        hash: str
        key: str
        version: str
        revision: str
        kwargs: dict
        spec: dict
        quote: str
        repr: str
        handle: str
        knob: str
        hashstr: str
        keystr: str
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
        **kwargs,
    ):
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
            'kwargs': kwargs,
        }
        self.__setstate__(state)
        
    def __setstate__(self, state):
        """NB: state keys should match __init__'s keyword arguments, with extra args in 'kwargs' dict"""
        state.pop('gitrepo', None)
        
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
        
        self.cfg = self._spec_to_cfg(self.spec)
        self.config = self.cfg # alias
        
        self._working_params_ = ['cfg', 'config', 'log', 'dt', 'build_dt']
        kwargs = state.get('kwargs', {})
        for key in kwargs.keys():
            assert key not in self.__explicit_params__() + self._working_params_, \
                f"Key {key} in kwargs conflicts with __explicit_params__() + _working_params_: {self.__explicit_params__() + self._working_params_}"
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        handled_keys = set(self.__explicit_params__()) | {'kwargs'}
        other_keys = [k for k in state if k not in handled_keys]
        assert len(other_keys) == 0, f"Unknown keys in state: {other_keys}"
            
        # self.parameters used for state retrieval
        self.parameters = self.__explicit_params__() + list(kwargs.keys())
        
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
        _kwargs = {
            k: getattr(self, k)
            for k in self.parameters if k not in self.__explicit_params__() and k != 'log' and hasattr(self, k)
        }
        _state['kwargs'] = _kwargs
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
        kwargs = _kw.pop('kwargs', {})
        _kw.update(**kwargs) # Merge old explicit and indirect params
        _kw.update(**kw)     # Update with NEW params (prioritized)
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
            key=self.key,   
            version=self.version,
            revision=self.revision,
            spec=self.spec,
            kwargs=self.kwargs,
            quote=self.quote(),
            repr=self.__repr__(),
            handle=self.handle(),
            knob=self.knob(),
            hashstr=self.hashstr,
            keystr=self.keystr,
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
    
    def valid(self,):
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
    
    def UNSAFE_copy_from(self, anchorhashpath, *, overwrite: bool = False, topicpaths=None):
        def fscopy(*, src_path, dst_path):
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
                    dst_fs.put(src_path, dst_path)
                else:
                    # Source is remote, destination is local
                    src_fs.get(src_path, dst_path)
            else:
                # Both are remote, use temporary directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, os.path.basename(src_path))
                    src_fs.get(src_path, tmp_path)
                    dst_fs.put(tmp_path, dst_path)

        if not overwrite:
            assert not self.valid(), f"Attempting to overwrite a valid Datablock {self}. Missing 'overwrite' argument?"
        fs, _ = fsspec.url_to_fs(anchorhashpath)
        assert fs.isdir(anchorhashpath), f"Nonexistent hashpath {anchorhashpath}"
        self.log.debug(f"Copying files from {anchorhashpath}: BEGIN")
        self._write_journal_entry(event="UNSAFE_copy_from:BEGIN", context=anchorhashpath, inline_context=True)
        try:
            if self.has_topics():
                for topic in self.topics():
                    dst_path = self.path(topic)
                    if dst_path is not None:
                        if topicpaths is not None:
                            _src_path = topicpaths[topic]
                        else:
                            _src_path = os.path.join(topic, self.TOPICFILES[topic])
                        src_path = os.path.join(anchorhashpath, _src_path)
                        self.log.detailed(f"Copying {src_path} to {dst_path}")
                        fscopy(src_path=src_path, dst_path=dst_path)
            elif self.has_topic():
                dst_path = self.path()
                if dst_path is not None:
                    if topicpaths is not None:
                        _src_path = topicpaths
                    else:
                        _src_path = self.TOPICFILE
                    src_path = os.path.join(anchorhashpath, _src_path)
                    self.log.detailed(f"Copying {src_path} to {dst_path}")
                    fscopy(src_path=src_path, dst_path=dst_path)
            self.log.debug(f"Copying files from {anchorhashpath}: END")
            self._write_journal_entry(event="UNSAFE_copy_from:END", context=anchorhashpath, inline_context=True)
            assert self.valid(), f"Invalid Datablock after copy: {self}"
        except Exception as e:
            self.log.error(f"UNSAFE_copy_from: Error when trying to copy files from {anchorhashpath}")
            self.log.error(f"EXCEPTION: {e}")
            self._write_journal_entry(event="UNSAFE_copy_from:ERROR", context=anchorhashpath, inline_context=True)
            raise e

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
    def _kwargsanchorpath(cls, root):
        kwargsanchor = os.path.join(
            (
                cls.__module__
                + "."
                + cls.__name__
            ),
            ".kwargs",
        )
        kwargsanchorpath = os.path.join(
            root,
            kwargsanchor,
        )
        return kwargsanchorpath

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
    
    def _kwargshashpath(self):
        kwargsanchorpath = self._kwargsanchorpath(self.root)
        return os.path.join(
            kwargsanchorpath,
            self.hash,
        )
    
    def _kwargspath(self, kind, *, ensure: bool = True):
        kwargshashpath = self._kwargshashpath()
        if ensure:
            fs, _ = fsspec.url_to_fs(kwargshashpath)
            fs.makedirs(kwargshashpath, exist_ok=True)
        if kind == 'yaml':
            kwargspath = os.path.join(kwargshashpath, f'{self.dt}.yaml')
        elif kind == 'parquet':
            kwargspath = os.path.join(kwargshashpath, f'{self.dt}.parquet')
        else:
            raise ValueError(f"Unknown path kind: {kind}")
        return kwargspath

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
        if hasattr(self, 'VERSION'):
            version = self.VERSION
        else:
            version = None
        return version
    
    @property
    def uuid(self):
        if not hasattr(self, '_uuid'):
            self._uuid = str(uuid.uuid4())
        return self._uuid
    
    @property
    def revision(self):
        if not hasattr(self, '_revision'):
            self.log.detailed(f"--------------> COMPUTING revision")
            if self._revision_ is None:
                self.log.detailed(f"--------------> self._revision_ is None")
                gitrepo = DBXWRKREPO if DBXWRKREPO is not None else DBXGITREPO
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
            if k not in ['root', 'anchored', 'hash', 'spec', 'kwargs']          
        }
        if 'kwargs' in state:
            tailkwargs.update(state['kwargs'])
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

    def knob(self, *, deslash: bool = False):
        repr_spec = self.__expand_spec__('handle')
        knob = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': repr_spec},
        }, use_stump=True)
        if deslash:
            knob = knob.replace('\\', '')
        self.log.detailed(f"knob: ------------> {repr_spec=}")
        self.log.detailed(f"knob: ------------>{knob=}")
        return knob
    
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
    def kwargs(self):
        return self.__getstate__()
    
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
    def keystr(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        if hasattr(self, "TOPICFILES"):
            topics = [f"topic:{topic}={file}" for topic, file in self.TOPICFILES.items()]
        else:
            topics = ["topics:None"]
        keystr = os.path.join(
            self.knob(),
            f"version={self.version}",
            *topics,
        )
        return keystr
    
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

    @property
    def tag(self):
        if not hasattr(self, '_tag'): 
            if self._tag_ is not None:
                self._tag = self._tag_
            else:
                self._tag = self.anchorhash
        return self._tag
    
    @property
    def key(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed with the older code.
        if not hasattr(self, '_key'): 
            sha = hashlib.sha256()
            sha.update(self.keystr.encode())
            self._key = sha.hexdigest()
            self.log.detailed(f"key: ---------===---------> {self.keystr=} ---> key: {self._key}")
        return self._key
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
        self._write_journal_dict('kwargs', self.kwargs)
        self._write_str('quote', self.quote())
        self._write_str('repr', self.__repr__())
        self._write_str('handle', self.handle())
        self._write_str('hashstr', self.hashstr)
        self._write_str('keystr', self.keystr)
        if context is not None and not inline_context:
            self._write_str('context', context)
        #
        dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        filename = f"{self.hash}-{dt}"

        spec_path = self._xpath('spec', 'yaml')
        kwargs_path = self._xpath('kwargs', 'yaml')
        quote_path = self._xpath('quote', 'txt')
        handle_path = self._xpath('quote', 'txt')
        repr_path = self._xpath('repr', 'txt')
        hashstr_path = self._xpath('hashstr', 'txt')
        keystr_path = self._xpath('keystr', 'txt')
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
                                         'revision': self.revision, 
                                         'root': self.root,
                                         'anchor': self.anchor,
                                         'hash': self.hash,
                                         'key': self.key,
                                         'tag': self.tag, 
                                         'log': logpath if has_log else None,
                                         'event': event,
                                         'spec': spec_path,
                                         'kwargs': kwargs_path,
                                         'quote': quote_path,
                                         'handle': handle_path,
                                         'repr': repr_path,
                                         'hashstr': hashstr_path,
                                         'keystr': keystr_path,
                                         'context': context,
                                         'gitrepo': DBXGITREPO,
                                         'wrkrepo': DBXWRKREPO,
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
                dfs.append(_df)
            df = pd.concat(dfs)
            # TODO: FIX uuid; currently not unique
            columns = ['hash', 'datetime'] + [c for c in df.columns if c not in ('hash', 'datetime', 'event', 'uuid')] + ['event']
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


class MultithreadingCallableExecutor:
    def __init__(self, *, n_threads, log: Logger = Logger()):
        self.n_threads = n_threads
        self.log = log

    #ALIAS
    def execute(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        return self.exec_callables(callables, *ctx_args, **ctx_kwargs)
    
    def exec_callables(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        if len(callables) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            progress_bar = tqdm.tqdm(total=len(callables))
            callable_lists = np.array_split(callables, self.n_threads)
            callable_offsets = np.cumsum([0] + [len(callable_list) for callable_list in callable_lists])
            threads = [
                threading.Thread(target=self.__exec_callables__, args=(callable_list, ctx_args, ctx_kwargs, callable_offset, thread_idx, result_queue, done_queue, abort_event))
                for thread_idx, (callable_list, callable_offset) in enumerate(zip(callable_lists, callable_offsets))
            ]
            payloads = [[] for _ in range(len(callables))]
            done_idxs = []
            for thread in threads:
                thread.start()
            while len(done_idxs) < len(callables):
                success, thread_idx, callable_idx, payload = result_queue.get()
                if success:
                    done_idxs.append(callable_idx)
                    e = None
                    payloads[callable_idx].append(payload)
                    progress_bar.update(1)
                else:
                    e = payload
                    self.log.info(f"Received error from callable with {callable_idx=} on thread {thread_idx}. Abandoning result_queue polling.")
                    break
            self.log.debug(f"Production loop done, feeding done_queue")
            for _ in range(self.n_threads):
                done_queue.put(None)
            self.log.debug(f"Joining threads")
            for thread in threads:
                thread.join()
            if e is not None:
                self.log.verbose("Raising exception")
                raise e
            self.log.debug("Threads successfully joined")
        flat_payloads = [payload for sublist in payloads for payload in sublist]
        return flat_payloads
    
    def __exec_callables__(self, callables: Sequence[Callable], ctx_args, ctx_kwargs, offset: int, thread_idx: int, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event):
        self.log.debug(f"Executing {len(callables)} callables on thread: {thread_idx}")
        for i, callable in enumerate(callables):
            self.log.detailed(f"EXECUTING callable {i+offset} on {thread_idx=}: {callable}")
            exception = None
            try:
                if abort_event.is_set():
                    break
                payload = callable(*ctx_args, **ctx_kwargs)
                self.log.detailed(f"EXECUTED callable {i+offset}: result: {payload}")
            except Exception as e:
                exception = e
                self.log.info(f"ERROR executing callable {callable} on {thread_idx=}")
            if exception is not None:
                result_queue.put((False, thread_idx, offset+i, exception))
                break
            result_queue.put((True, thread_idx, offset+i, payload))
        gc.collect()
        if exception is None:
            self.log.debug(f"Done executing {len(callables)} callables on {thread_idx=}")
        else:
            self.log.debug(f"Abandoning executing {len(callables)} callables on {thread_idx=} due to an exception")
        self.log.debug(f"Waiting on the done_queue on {thread_idx}")
        while True:
            item = done_queue.get()
            if item is None:
                self.log.debug(f"Done message received on the done_queue on {thread_idx=}")
                break


class MultithreadingDatablocksBuilder:
    def __init__(self, *, n_threads: int = 1, log: Logger = Logger()):
        self.n_threads = n_threads
        self.log = log

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        if len(blocks) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            progress_bar = tqdm.tqdm(total=len(blocks))
            block_lists = np.array_split(blocks, self.n_threads)
            block_offsets = np.cumsum([0] + [len(block_list) for block_list in block_lists])
            threads = [
                threading.Thread(target=self.__build_blocks__, args=(block_list, ctx_args, ctx_kwargs, block_offset, thread_idx, result_queue, done_queue, abort_event, progress_bar))
                for thread_idx, (block_list, block_offset) in enumerate(zip(block_lists, block_offsets))
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
            for _ in range(self.n_threads):
                done_queue.put(None)
            self.log.debug(f"Joining threads")
            for thread in threads:
                thread.join()
            if e is not None:
                self.log.verbose("Raising exception")
                raise e
            self.log.debug("Threads successfully joined")
        return blocks
    

    def __build_blocks__(self, blocks: Sequence[Datablock], ctx_args, ctx_kwargs, offset: int, thread_idx: int, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event, progress_bar):
        self.log.debug(f"Building {len(blocks)} feature blocks on thread: {thread_idx}")
        for i, block in enumerate(blocks):
            exception = None
            try:
                if abort_event.is_set():
                    break
                block.build(*ctx_args, **ctx_kwargs).to('cpu')
            except Exception as e:
                exception = e
                self.log.info(f"ERROR building feature block {block} on thread: {thread_idx}")
            if exception is not None:
                result_queue.put((False, offset+i, exception))
                break
            result_queue.put((True, offset+i, None))
            progress_bar.update(1)
        gc.collect()
        if exception is None:
            self.log.debug(f"Done building {len(blocks)} feature blocks on thread: {thread_idx}")
        else:
            self.log.debug(f"Abandoning building {len(blocks)} feature blocks on thread: {thread_idx} due to an exception")
        self.log.debug(f"Waiting on the done_queue on thread: {thread_idx}")
        while True:
            item = done_queue.get()
            if item is None:
                self.log.debug(f"Done message received on the done_queue on thread: {thread_idx}")
                break


class MultiprocessingDatablockBuilder:
    def __init__(self, *, n_processes: int = 1, log: Logger = Logger()):
        self.n_processes = n_processes
        self.log = log

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        if len(blocks) > 0:
            result_queue = mp.Queue()
            done_queue = mp.Queue()
            abort_event = mp.Event()
            progress_bar = tqdm.tqdm(total=len(blocks))
            block_lists = np.array_split(blocks, self.n_processes)
            block_offsets = np.cumsum([0] + [len(block_list) for block_list in block_lists])
            processes = [
                mp.Process(target=self.__build_blocks__, args=(block_list, ctx_args, ctx_kwargs, block_offset, i, result_queue, done_queue, abort_event))
                for i, (block_list, block_offset) in enumerate(zip(block_lists, block_offsets))
            ]
            self.log.verbose(f"Building {len(blocks)} feature blocks with {self.n_processes} processes")
            done_idxs = []
            exc = None
            try:
                for process in processes:
                    process.start()
                # Clear references to blocks in main process to save memory/allow pickling if needed (though copy happens on fork/spawn)
                # In multiprocessing spawn (default on some OS, optional on Linux), blocks need to be pickled.
                # Assuming linux fork by default but being safe.
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
                        self.log.info(f"Received exception from process {proc}, block with index {idx}")
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
                for _ in range(self.n_processes):
                    done_queue.put(None)
                self.log.debug(f"Joining processes")
                for process in processes:
                    process.join()
                self.log.debug("Processes successfully joined")
            
            if pexc is not None:
                self.log.verbose(f"Reraising exception from process {proc}, block {idx}")
                raise(pexc)
            if exc is not None:
                self.log.verbose("Reraising production loop exception")
                raise(exc)
        return blocks

    def __build_blocks__(self, blocks: Sequence[Datablock], ctx_args, ctx_kwargs, offset: int, process_idx: int, result_queue: mp.Queue, done_queue: mp.Queue, abort_event: mp.Event):
        self.log.debug(f"Building {len(blocks)} feature blocks on process: {process_idx}")
        exception = None
        for i, block in enumerate(blocks):
            exception = None
            try:
                if abort_event.is_set():
                    break
                # Assuming .build() method exists and works in subprocess
                block.build(*ctx_args, **ctx_kwargs)
            except Exception as e:
                exception = e
                self.log.info(f"ERROR building datablock {block} on process: {process_idx}")
            finally:
                del block
                gc.collect()
            
            if exception is not None:
                tbstr = '\n'.join(tb.format_tb(exception.__traceback__))
                result_queue.put((False, process_idx, offset+i, (exception, tbstr)))
                break
            result_queue.put((True, process_idx, offset+i, None))
        
        # Cleanup
        gc.collect()
        if exception is None:
            self.log.debug(f"Done building {len(blocks)} datablocks on process: {process_idx}")
        else:
            self.log.debug(f"Abandoning building {len(blocks)} datablocks on process: {process_idx} due to an exception")
        
        self.log.debug(f"Waiting on the done_queue on process: {process_idx}")
        while True:
            item = done_queue.get()
            if item is None:
                self.log.debug(f"Done message received on the done_queue on process: {process_idx}")
                break

    

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
            val = getattr(self.obj, name)
            return self._wrap(val)

        def call(self, name, *args, **kwargs):
            if hasattr(self, name) and name != 'obj': # Avoid recursion or direct access to internal state
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
        if hasattr(self, '_slurm') and self._slurm:
            self._slurm.cancel()
            self._slurm = None

    def __getattr__(self, name):
        """
        Dispatch attribute access to the remote actor.
        """
        is_callable, value = ray.get(self._handle.info.remote(name))
        
        if is_callable:
            def wrapper(*args, **kwargs):
                res = ray.get(self._handle.call.remote(name, *args, **kwargs))
                return self._unwrap_or_proxy(res)
            return wrapper
        else:
            return self._unwrap_or_proxy(value)

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


def remote(*, revision=None, slurm=None, conda=None, log: Logger = Logger()):
    """
    Instantiate a remote dbx interpreter and return a Remote handle to it.
    """
    dbx_env = {k: v for k, v in os.environ.items() if k.startswith('DBX')}
    
    if DBXWRKREPO is not None:
        dbx_env['DBXGITREPO'] = DBXWRKREPO

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


class RemoteCallableExecutor:
    def __init__(self, *, n_workers, revision=None, conda=None, log: Logger = Logger()):
        self.n_workers = n_workers
        self.log = log
        self.workers = [remote(revision=revision, conda=conda) for _ in range(n_workers)]

    #ALIAS
    def execute(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        return self.exec_callables(callables, *ctx_args, **ctx_kwargs)

    def exec_callables(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        if len(callables) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            progress_bar = tqdm.tqdm(total=len(callables))
            
            # Split callables among workers
            callable_lists = np.array_split(callables, self.n_workers)
            callable_offsets = np.cumsum([0] + [len(callable_list) for callable_list in callable_lists])
            
            threads = [
                threading.Thread(target=self.__exec_callables__, 
                                 args=(worker, callable_list, ctx_args, ctx_kwargs, callable_offset, thread_idx, result_queue, done_queue, abort_event))
                for thread_idx, (worker, callable_list, callable_offset) in enumerate(zip(self.workers, callable_lists, callable_offsets))
            ]
            
            payloads = [[] for _ in range(len(callables))]
            done_idxs = []
            for thread in threads:
                thread.start()
                
            e = None
            while len(done_idxs) < len(callables):
                success, worker_idx, callable_idx, payload = result_queue.get()
                if success:
                    done_idxs.append(callable_idx)
                    e = None
                    payloads[callable_idx].append(payload)
                else:
                    e = payload
                    self.log.info(f"Received error from callable with {callable_idx=} on worker {worker_idx}. Abandoning result_queue polling.")
                    break
                progress_bar.update(1)
                
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
        flat_payloads = [payload for sublist in payloads for payload in sublist]
        return flat_payloads

    def __exec_callables__(self, worker, callables: Sequence[Callable], ctx_args, ctx_kwargs, offset: int, thread_idx: int, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event):
        self.log.debug(f"Executing {len(callables)} callables on worker {thread_idx}")
        for i, callable in enumerate(callables):
            self.log.detailed(f"EXECUTING callable {i+offset} on worker {thread_idx}: {callable}")
            exception = None
            try:
                if abort_event.is_set():
                    break
                payload = worker.run(callable, *ctx_args, **ctx_kwargs)
                self.log.detailed(f"EXECUTED callable {i+offset}: result: {payload}")
            except Exception as e:
                exception = e
                self.log.info(f"ERROR executing callable {callable} on worker {thread_idx}")
            
            if exception is not None:
                result_queue.put((False, thread_idx, offset+i, exception))
                break
            result_queue.put((True, thread_idx, offset+i, payload))
        
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


class RemoteDatablocksBuilder:
    def __init__(self, *, n_workers: int = 1, revision=None, conda=None, log: Logger = Logger()):
        self.n_workers = n_workers
        self.log = log
        self.executor = RemoteCallableExecutor(n_workers=n_workers, revision=revision, conda=conda, log=log)

    def build_blocks(self, blocks: Sequence[Datablock], *ctx_args, **ctx_kwargs):
        if len(blocks) > 0:
            def build_block(block, *args, **kwargs):
                return block.build(*args, **kwargs)

            callables = [functools.partial(build_block, block) for block in blocks]
            results = self.executor.execute(callables, *ctx_args, **ctx_kwargs)
            
            # Update local blocks with built state from remote workers
            for block, result_list in zip(blocks, results):
                if result_list:
                    # RemoteCallableExecutor returns a list of lists: [[res]]
                    # We expect one result per block.
                    remote_block = result_list[0]
                    # Update local block state from the remote result
                    block.__setstate__(remote_block.__getstate__())

        return blocks


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
            anchorhashpath = os.path.join(entry.root, entry.anchor, entry.hash)
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
        executor = MultithreadingCallableExecutor(n_threads=n_workers, log=log)
        _results_ = executor.execute(_callables_, _journal_)
        results = [result for result in _results_ if result is not None]
    dbks, copied = zip(*results)
    pulled = sum([int(c) for c in copied])
    valids = sum([int(dbk.valid()) for dbk in dbks])
    log.info(f"Pulled {pulled} out of {len(copied)} datablocks.  Skipped {len(copied)-pulled}. Valids: {valids}")
    log.info(f"Done")
    return dbks, copied
