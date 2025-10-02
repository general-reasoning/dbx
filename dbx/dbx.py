from collections.abc import Iterable
import copy
from dataclasses import dataclass, fields, asdict, replace, is_dataclass
import datetime
import gc
import hashlib
import importlib
import inspect
import json
import multiprocessing as mp
import os
import pickle
import queue
import sys
import threading
import time
import traceback as tb
import typing
from typing import Union, Optional, Sequence, Callable
import uuid
import yaml

import git

import tqdm
import rich

import numpy as np

import fsspec

from scipy.stats import qmc

import pandas as pd
import pyarrow.parquet as pq
import torch


class Logger:
    """Because Python logging is so cumbersome to initialize, configure and control, we have this."""

    def __init__(
        self,
        *,
        warning: bool = True,
        info: bool = True,
        verbose: bool = False,
        debug: bool = False,
        select: bool = False,
        detailed: bool = False,
        name: Optional[str] = None,
        stack_depth: int = 2,
    ):
        self._warning = eval(os.environ.get('DBXWARNING', str(warning)))
        self._info = eval(os.environ.get('DBXINFO', str(info)))
        self._verbose = eval(os.environ.get('DBXVERBOSE', str(verbose)))
        self._select = eval(os.environ.get('DBXSELECT', str(select)))
        self._debug = eval(os.environ.get('DBXDEBUG', str(debug)))
        self._detailed = eval(os.environ.get('DBXDETAILED', str(detailed)))
        self.allowed = []
        if self._warning:
            self.allowed.append("WARNING")
        if self._info:
            self.allowed.append("INFO")
        if self._debug:
            self.allowed.append("DEBUG")
        if self._verbose:
            self.allowed.append("VERBOSE")
        if self._select:
            self.allowed.append("SELECT")
        if self._detailed:
            self.allowed.append("DETAIL")
        self.stack_depth = stack_depth
        self.name = name

    def _print(self, prefix, msg):
        if self.name is None:
            stack = inspect.stack()
            frame = stack[self.stack_depth-1]
            func = frame.function
            name = f"{func}"
        else:
            name = self.name
        if prefix in self.allowed:
            print(f"{prefix}: {name}: {msg}")

    def warning(self, msg):
        self._print("WARNING", msg)

    def info(self, msg):
        self._print("INFO", msg)

    def debug(self, msg):
        self._print("DEBUG", msg)

    def verbose(self, msg):
        self._print("VERBOSE", msg)

    def select(self, msg):
        self._print("SELECT", msg)

    def detailed(self, msg):
        self._print("DETAILED", msg)

    def silent(self, mst):
        pass


def journal(cls, root=None):
    return Datablock.Journal(cls, root)


def scopes(cls, root=None):
    return Datablock.Scopes(cls, root)


def gitrevision(repopath, *, log=Logger()):
    if repopath is not None:
        repo = git.Repo(repopath)
        if repo.is_dirty():
            raise ValueError(f"Dirty git repo: {repopath}: commit your changes")
        branch = repo.active_branch.name
        reponame = os.path.basename(repopath)
        revision = f"{reponame}:{repo.rev_parse(branch).hexsha}"
        log.verbose(f"Obtained git revision for git repo {repopath}: {revision}")
    else:
        revision = None
    return revision


def make_download_url(path):
    if not path.startswith("gs://"):
        return None
    _path = path.removeprefix("gs://")
    return f"https://storage.cloud.google.com/{_path}"


def get_named_const_and_cxt(name):
    bits = name.split(".")
    modbits = bits[:-1]
    prefix = None
    cxt = {}
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
                    val = eval(v)
                    kwargs[k] = val
                else:
                    arg = eval(bit)
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

    if isinstance(name, Iterable) and not isinstance(name, str):
        term = [eval_term(item) for item in name]
    elif isinstance(name, str):
        if not (name.startswith("[") and name.endswith("]")) and not name.startswith("@"):
            term = name
        else:
            if name.startswith("@"):
                _name_ = name[1:-1] if name.endswith('#') else name[1:]
            else:
                _name_ = name[1:-1]
            funcstr, _ = get_funcstr_argkwargstr(_name_)
            if funcstr is None:
                term, _ = get_named_const_and_cxt(_name_)
            else:
                _, cxt = get_named_const_and_cxt(funcstr)
                term = eval(_name_, cxt)
    else:
        term = name
    return term


def exec(s=None):
    if s is None:
        if len(sys.argv) > 2:
            raise ValueError(f"Too many args: {sys.argv}")
        elif len(sys.argv) == 1:
            raise ValueError(f"Too few args: {sys.argv}")
        s = sys.argv[1]
    lb = s.find("(")
    lb = lb if lb != -1 else len(s)
    _, cxt = get_named_const_and_cxt(s[:lb])
    r = eval(s, globals(), cxt)
    return r


def exec_print(argstr=None):
    print(exec(argstr))


def write_yaml(data, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "w") as f:
        yaml.dump(data, f)
        log.debug(f"WROTE {path}")


def read_yaml(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        data = yaml.safe_load(f)
        log.verbose(f"READ {path}")
    return data


def write_json(data, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "w") as f:
        json.dump(data, f)
        log.debug(f"WROTE {path}")


def read_json(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        data = json.load(f)
        log.debug(f"READ {path}")
    return data


def write_tensor(tensor, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    array = tensor.numpy()
    with fs.open(path, "wb") as f:
        np.save(f, array)
        log.debug(f"WROTE {path}")


def read_tensor(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "rb") as f:
        array = np.load(f)
        log.debug(f"READ {path}")
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
        log.debug(f"WROTE {list(kwargs.keys())} to {path}")


def read_npz(path, *keys, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "rb") as f:
        data = np.load(f)
        results = [data[k] for k in keys]
        log.debug(f"READ {list(keys)} from {path}")
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


def quote(obj):
    if isinstance(obj, str) and obj.startswith("@"):
        quote = obj
    else:
        quote = f"@{repr(obj)}#"
    return quote


class Datashard:
    def build(self):
          pass
     
    def __str__(self):
        return repr(self)
    
    def to(self, device):
        return self


class Datablock(Datashard):
    """
    ROOT = 'protocol://path/to/root'
    TOPICFILES = {'topic', 'file.csv'} | TOPICFILE = 'file.csv'
    # protocol://path --- module/class/ --- topic --- file 
    #        root           [anchor]        [topic]   [file]
    # root:       'protocol://path/to/root'
    # anchorpath: '{root}/modpath/class'|'{root}' if anchored|else
    # hashpath:   '{anchorpath}/{hash}|{anchorpath}/{hash}' if hash supplied through args|else
    # dirpath:    '{hashpath}/topic'|{hashpath}' if topic is not None|else
    # path:       '{dirpath}/{TOPICFILE}'|'{dirpath}' if TOPICFILE is not None|else
    
    """
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
        root: str = None,
        *,
        spec: Optional[Union[str, dict]] = None,
        anchored: bool = True,
        hash: Optional[str] = None,
        tag: Optional[str] = None,
        info: bool = True,
        verbose: bool = False,
        debug: bool = False,
        detailed: bool = False,
        capture_build_output: bool = False,
        gitrepo: str  = None,
        **kwargs,
    ):
        self.__setstate__(dict(
            root=root,
            spec=spec,
            anchored=anchored,
            hash=hash,
            tag=tag,
            info=info,
            verbose=verbose,
            debug=debug,
            detailed=detailed,
            capture_build_output=capture_build_output,
            gitrepo=gitrepo,
            **kwargs,
        ))
        
    def __setstate__(
        self,
        kwargs,
    ):
        processed = []
        self.root = kwargs.get('root')
        self._autoroot = False
        if self.root is None:
            self.root = os.environ.get('DBXROOT')
            self._autoroot = True
        if self.root is None:
            raise ValueError(f"None root for {self.__class__.__name__}: maybe set DBXROOT?")
        processed.append('root')
        self.spec = kwargs.get('spec')
        processed.append('spec')
        self.anchored = kwargs.get('anchored')
        processed.append('anchored')
        self._hash = kwargs.get('hash')
        processed.append('hash')
        self.tag = kwargs.get('tag')
        processed.append('tag')
        #
        self.info = eval(os.environ.get('DBXINFO', str(kwargs.get('info'))))
        self.verbose = eval(os.environ.get('DBXVERBOSE', str(kwargs.get('verbose'))))
        self.debug = eval(os.environ.get('DBXDEBUG', str(kwargs.get('debug'))))
        self.detailed = eval(os.environ.get('DBXDETAILED', str(kwargs.get('detailed'))))
        self.gitrepo = os.environ.get('DBXREPO', kwargs.get('gitrepo'))
        self.capture_build_output = bool(kwargs.get('capture_build_output'))
        processed.extend(['verbose', 'debug', 'gitrepo', 'capture_build_output'])  
        self.log = Logger(
            debug=self.debug,
            verbose=self.verbose,
            detailed=self.detailed,
            info=self.info,
            name=self.anchor(),
        )
        #
        if isinstance(self.spec, str):
            self.spec = read_json(self.spec, debug=self.debug)
        if self.spec is None:
            self.spec = asdict(self.CONFIG())
        self.config = self._spec_to_config(self.spec)
        self._scope = None
        self._autohash = self._hash is None

        for k, v in kwargs.items():
            if k not in processed:
                setattr(self, k, v)
        self.parameters = list(kwargs.keys())
        self.__post_init__()

    def __getstate__(self):
        return dict(
            root=self.root if not self._autoroot else None,
            spec=self.spec,
            anchored=self.anchored,
            hash=self.hash if not self._autohash else None,
            tag=self.tag,
            info=self.info,
            verbose=self.verbose,
            debug=self.debug,
            detailed=self.detailed,
            capture_build_output=self.capture_build_output,
            gitrepo=self.gitrepo,
            **{k: getattr(self, k) for k in self.parameters 
             if k not in ('root', 'spec', 'anchored', 'hash', 'tag', 'info', 'verbose', 'debug', 'detailed', 'capture_build_output', 'gitrepo')
            }
        )

    def __post_init__(self):
        ...

    def __repr__(self):
        if self._autoroot:
            argstr = f"spec={repr(self.spec)}"
        else:
            argstr = ', '.join((repr(self.root), f"spec={repr(self.spec)}"))
        kwargslist = []
        if not self.anchored:
            kwargslist.append('anchored=False')
        if not self._autohash:
            kwargslist.append(f'hash={self._hash}')
        if len(kwargslist) > 0:
            kwargstr = ', '.join(kwargslist)
            argskwargsrepr = argstr + ', ' + kwargstr
        else:
            argskwargsrepr = argstr
        r = f"{self.anchor()}({argskwargsrepr})"
        self.log.detailed(f"{self.anchor()}: repr: {r}")
        return r
    
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

    def kwargs(self):
        return self.__getstate__()

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
    
    def path(
        self,
        topic=None,
        *,
        ensure_dirpath: bool = False,
    ):
        if topic is None:
            dirpath = self.dirpath()
            topicfiles = self.TOPICFILE
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
        self.log.detailed(f"{self.anchor()}: path: {path}")
        return path

    def ensure_path(self, path):
        fs, _ = fsspec.url_to_fs(path)
        fs.makedirs(path, exist_ok=True)
        return self

    def url(self, topic=None, *, redirect=None):
        path = self.path(topic)
        return make_download_url(path)
    
    def paths(self):
        if self.has_topics:
            paths = {topic: self.path(topic) for topic in self.topics()}
        else:
            paths = self.path()
        return paths

    def validpath(self, path):
        if isinstance(path, dict):
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
        self.log.detailed(f"{self.anchor()}: path {path} valid: {result}") 
        return result
    
    def valid(self,):
        results = []
        if hasattr(self, "TOPICFILES"):
            results += [
                self.validpath(self.path(topic))
                for topic in self.TOPICFILES
            ]
        else:
            results += [self.validpath(self.path())]
        result = all(results)
        self.log.detailed(f"{self.anchor()}: {results=}")
        return result
    
    def topics(self):
        return list(self.TOPICFILES.keys()) if self.has_topics() else []

    def has_topics(self):
        return hasattr(self, "TOPICFILES")

    def build(self, *args, **kwargs):
        if self.capture_build_output:
            stdout = sys.stdout
            outfs, _ = fsspec.url_to_fs(self.capture_out)
            captured_stdout_stream = outfs.open(self.logpath(), "w", encoding="utf-8")
            sys.stdout = captured_stdout_stream
        try:
            if not self.valid():
                self.__pre_build__(*args, **kwargs).__build__(*args, **kwargs).__post_build__(*args, **kwargs)
            else:
                self.log.verbose(f"Skipping existing datablock: {self.hashpath()}")
        finally:
            if self.capture_build_output:
                sys.stdout = stdout
                captured_stdout_stream.close()
        return self

    def __pre_build__(self, *args, **kwargs):
        self._write_scope()
        self._write_journal_entry(event="build:start",)
        return self

    def __build__(self, *args, **kwargs):
        return self

    def __post_build__(self, *args, **kwargs):
        self._write_journal_entry(event="build:end",)
        return self
    
    @property
    def revision(self):
        if not hasattr(self, '_revision'):
            self._revision = gitrevision(self.gitrepo, log=self.log) if self.gitrepo is not None else None
        return self._revision

    def leave_breadcrumbs(self):
        if hasattr(self, "TOPICFILES"):
            for topic in self.TOPICFILES:
                self.dirpath(topic, ensure=True)
                self.leave_breadcrumbs_at_path(self.path(topic))
        else:
            self.dirpath(ensure=True)
            self.leave_breadcrumbs_at_path(self.path())
        return self

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
    
    def UNSAFE_clear(self):
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
        if hasattr(self, "TOPICFILES"):
            for topic in self.TOPICFILES:
                clear_dirpath(self.dirpath(topic))
        else:
            clear_dirpath(self.dirpath())
        self._write_journal_entry(event="UNSAFE_clear")
        return self
    
    def anchor(self):
        anchor = (
            self.__module__
            + "."
            + self.__class__.__name__
        )
        return anchor
    
    def anchorpath(self):
        anchorpath = os.path.join(
            self.root,
            self.anchor(),
        ) if self.anchored else self.root
        return anchorpath
    
    @property
    def hash(self):
        if self._hash is None:
            if hasattr(self, "TOPICFILES"):
                topics = [f"_topic_{topic}={file}" for topic, file in self.TOPICFILES.items()]
            else:
                topics = ["None"]
            hivehandle = os.path.join(
                *topics,
                *[f"{key}={repr(val)}" for key, val in self._scope_.items()]
            )
            sha = hashlib.sha256()
            sha.update(hivehandle.encode())
            self._hash = sha.hexdigest()
        return self._hash
            
    def scope(self):
        yscopepath = self.Scopepath(self.root, self.hash, 'yaml')
        yfs, _ = fsspec.url_to_fs(yscopepath)
        if not yfs.exists(yscopepath):
            return
        scope = read_yaml(yscopepath)
        return scope
    
    @staticmethod
    def Scopes(cls, root):
        cls = eval_term(cls)
        if root is None:
            root = os.environ.get('DBXROOT')
        scopeanchorpath = cls._scopeanchorpath(root)
        fs, _ = fsspec.url_to_fs(scopeanchorpath)
        if not fs.exists(scopeanchorpath):
            df = None
        else:
            paths = list(fs.ls(scopeanchorpath))
            scopefiles_ = [
                os.path.join(path, 'scope.parquet') 
                for path in paths
            ]
            scopefiles = [f for f in scopefiles_ if fs.exists(f)]
            hashes = [os.path.dirname(f).removeprefix(scopeanchorpath).removeprefix('/') for f in scopefiles]
            if len(scopefiles) > 0:
                dfs = []
                for scopefile in scopefiles:
                    _df = pd.read_parquet(scopefile)
                    dfs.append(_df)
                df = pd.concat(dfs)
                df.index = hashes
            else:
                df = pd.DataFrame(index=hashes)
            df = df.reset_index().rename(columns={'index': 'hash'})
        return df

    @staticmethod
    def Journal(cls, root):
        cls = eval_term(cls)
        if root is None:
            root = os.environ.get('DBXROOT')
        journaldirpath = cls._journalanchorpath(root)
        fs, _ = fsspec.url_to_fs(journaldirpath)
        files = list(fs.ls(journaldirpath))
        parquet_files = [f for f in files if f.endswith('.parquet')]
        
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
        return df

    def scopes(self):
        return self.Scopes(self.__class__, self.root)

    def journal(self):
        return self.Journal(self.__class__, self.root)

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
    def _scopepath(cls, root, hash, *, ensure: bool = True):
        scopeanchorpath = cls._scopeanchorpath(root)
        scopepath = os.path.join(
            scopeanchorpath,
            hash,
        )
        if ensure:
            fs, _ = fsspec.url_to_fs(scopepath)
            fs.makedirs(scopepath, exist_ok=True)
        return scopepath
    
    @classmethod
    def Scopepath(cls, root, hash, kind, *, ensure: bool = True):
        if kind == 'yaml':
            return os.path.join(cls._scopepath(root, hash, ensure=ensure), 'scope.yaml')
        elif kind == 'parquet':
            return os.path.join(cls._scopepath(root, hash, ensure=ensure), 'scope.parquet')
        else:
            raise ValueError(f"Unknown configpath kind: {kind}")
        
    def logpath(self, *, ensure: bool = True):
        return os.path.join(self._scopepath(self.root, self.hash, ensure=ensure), 'log', f'{self.uuid}.log')

    @classmethod
    def _journalanchorpath(cls, root, *, ensure: bool = True):
        journalanchor = os.path.join(
            cls.__module__
            + "."
            + cls.__name__,
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

    def hashpath(self, *, ensure: bool = True):
        anchorpath = self.anchorpath()
        hashpath = os.path.join(anchorpath, self.hash)
        if ensure:
            fs, _ = fsspec.url_to_fs(hashpath)
            fs.makedirs(hashpath, exist_ok=True)
        return hashpath

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

    @property
    def _scope_(self):
        if self._scope is None:
            scope = {}
            for k, v in self.spec.items():
                value = getattr(self.config, k)
                if isinstance(v, str) and v.endswith('#') and isinstance(value, Datablock):
                    scope[k] = f"@{value.anchor()}/{value.hash}"
                elif isinstance(v, str) and v.endswith('#'):
                    v = v.removesuffix('#')
                    scope[k] = v
                elif is_dataclass(v):
                    scope[k] = asdict(v) #TODO: call to a TBD recursive _scope_ on the container?
                else:
                    scope[k] = repr(v)
            scope['version'] = self.version
            self._scope = scope
        return self._scope

    def _write_scope(self):
        #
        yscopepath = self.Scopepath(self.root, self.hash, 'yaml')
        yfs, _ = fsspec.url_to_fs(yscopepath)
        write_yaml(self._scope_, yscopepath)
        assert yfs.exists(yscopepath), f"scopepath {yscopepath} does not exist after writing"
        self.log.debug(f"WROTE: SCOPE: yaml: {yscopepath}")
        #
        pscopepath = self.Scopepath(self.root, self.hash, 'parquet')
        pfs, _ = fsspec.url_to_fs(pscopepath)
        scopedf = pd.DataFrame.from_records([self._scope_])
        scopedf.to_parquet(pscopepath)
        assert pfs.exists(pscopepath), f"scopepath {pscopepath} does not exist after writing"
        #
        self.log.debug(f"WROTE: SCOPE: parquet: {pscopepath}")

    def _write_journal_entry(self, event:str):
        hash = self.hash
        dt = str(datetime.datetime.now()).replace(' ', '-')
        key = f"{hash}-{dt}"

        kwargs_path = os.path.join(self._journalanchorpath(self.root), f"{key}.yaml")
        write_yaml(self.kwargs(), kwargs_path)
        #
        logpath = self.logpath()
        if logpath is not None:
            logfs, _ = fsspec.url_to_fs(logpath)
            has_log = logfs.exists(logpath)
        else:
            has_log = False
        #
        journal_path = os.path.join(self._journalanchorpath(self.root), f"{key}.parquet")
        df = pd.DataFrame.from_records([{'datetime': dt,
                                         'version': self.version,
                                         'revision': self.revision, 
                                         'hash': hash,
                                         'tag': self.tag, 
                                         'log': logpath if has_log else None,
                                         'event': event,
                                         'kwargs': kwargs_path,
        }])
        df.to_parquet(journal_path)
        
        self.log.debug(f"Wrote JOURNAL entry for event {repr(event)} with tag {repr(self.tag)} "
                         f"to journal_path {journal_path} and kwargs_path {kwargs_path}")
    
    def _spec_to_config(self, spec):
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
        return config

    def leave_breadcrumbs_at_path(self, path):
        fs, _ = fsspec.url_to_fs(path)
        with fs.open(path, "w") as f:
            f.write("")


def datablock_method(
    datablock_cls,
    method,
    method_kwargs,
    *,
    root: str = None,
    spec: Optional[Union[str,dict]] = None,
    anchored: bool = True,
    hash: Optional[str] = None,
    tag: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    capture_build_output: bool = False,
    gitrepo: str  = None,
    kwargs: dict = dict(),
):
    datablock_cls = eval_term(datablock_cls)
    datablock = datablock_cls(root,
                              spec=spec, 
                              anchored=anchored, 
                              hash=hash,
                              tag=tag,
                              verbose=verbose, 
                              debug=debug, 
                              capture_build_output=capture_build_output, 
                              gitrepo=gitrepo,
                              **kwargs)
    method_callable = getattr(datablock, method)
    return method_callable(**method_kwargs)

    
class DatabatchBuilder:
    def __init__(self, *, verbose: bool = False, debug: bool = False,):
        self.verbose = verbose
        self.debug = debug
        self.log = Logger(verbose=verbose, debug=debug)
        
    def __call__(self, datablock_build_method_args_kwargs_list):
        N = len(datablock_build_method_args_kwargs_list)
        for i, (args, kwargs) in enumerate(datablock_build_method_args_kwargs_list):
            self.log.verbose(f"Building {i}-th datablock out of {N}")
            datablock_method(*args, **kwargs)
        

class Databatch(Datablock):
    DATABLOCK = None
    TOPICFILE = "summary.csv"

    def __init__(
        self,
        root: str = None,
        spec: Optional[Union[str,dict]] = None,
        *,
        anchored: bool = True,
        hash: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
        capture_build_output: bool = False,
        gitrepo: str  = None,
        builder: DatabatchBuilder = None,
    ):
        super().__init__(root,
                         spec=spec,
                         anchored=anchored,
                         hash=hash,
                         verbose=verbose,
                         debug=debug,
                         capture_build_output=capture_build_output, 
                         gitrepo=gitrepo)
        self._builder = builder

    @property
    def builder(self):
        if isinstance(self._builder, str):
            self._builder = eval_term(self._builder)
        return self._builder

    def datablocks(self) -> typing.Iterable[Datablock]:
        raise NotImplementedError()
        return self

    def __build__(self,):
        if self.builder is None:
            n_datablocks = len(self.datablocks())
            if self.verbose or self.debug:
                iterator = enumerate(self.datablocks())
            else:
                iterator = tqdm.tqdm(enumerate(self.datablocks()))
            for i, datablock in iterator:
                self.log.verbose(f"Building {i}-th datablock out of {n_datablocks}")
                datablock.build()
        else:
            datablock_method_args_kwargs_list = []
            for i, datablock in enumerate(self.datablocks()):
                tagi = f"run:{i}:{self.hash}"
                datablock_method_args_kwargs = (
                    (self.DATABLOCK, 'build', {}),
                    datablock.kwargs(),
                )
                datablock_method_args_kwargs_list.append(datablock_method_args_kwargs)
            self.builder(datablock_method_args_kwargs_list)
        return self

    def run(self):
        return self.build()
    
    def datablock_scopes(self):
        return self.Scopes(self.DATABLOCK, self.root)

    def datablock_journal(self):
        return self.Journal(self.DATABLOCK, self.root)


class MultiprocessProgressTracker:
    """Wrapper for a rich.progress tracker that can be shared across processes."""

    def __init__(self, tasks):
        ctx = mp.get_context('spawn')
        self.mp_values = {
            task.id: ctx.Value('i', task.completed)
            for task in tasks
        }

    def advance(self, id, amount):
        with self.mp_values[id].get_lock():
            self.mp_values[id].value += amount

    def __getitem__(self, id):
        return self.mp_values[id].value


class MultiprocessProgress:
    """Wrapper for a rich.progress bar that can be shared across processes."""

    def __init__(self, pb):
        self.pb = pb
        self.tracker = MultiprocessProgressTracker(self.pb.tasks)
        self.should_stop = False

    def _update_progress(self):
        while not self.should_stop:
            for task in self.pb.tasks:
                self.pb.update(task.id, completed=self.tracker[task.id])
            time.sleep(0.1)

    def __enter__(self):
        self._thread = threading.Thread(target=self._update_progress)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self.should_stop = True
        self._thread.join()


def datablock_method_multiprocessing(
        id,
        datablock_method_args_kwargs_list,
        progress_bar=None,
        progress_task=None,
):
        args, kwargs = datablock_method_args_kwargs_list[id]
        result = datablock_method(*args, **kwargs)
        if progress_bar is not None and progress_task is not None:
            progress_bar.advance(progress_task, 1)
        return result
    

class TorchMultiprocessingDatabatchBuilder(DatabatchBuilder):
    def __init__(self, *, num_gpus: int = 1):
        self.num_gpus = num_gpus
    
    def __call__(self, datablock_method_args_kwargs_list):
        N = len(datablock_method_args_kwargs_list)
        pb = rich.progress.Progress() 
        pb.add_task(
            "Speed: ",
            progress_type="speed",
            total=None
        )
        slide_task = pb.add_task(
            f"Building ...",
            progress_type="slide_progress",
            total=N,
        )
        pb.start()
        with MultiprocessProgress(pb) as mp_pb:
            torch.multiprocessing.spawn(
                datablock_method_multiprocessing,
                args=(datablock_method_args_kwargs_list,
                      mp_pb.tracker,
                      slide_task,
                ),       
                nprocs=self.num_gpus,
            )


class MultithreadingDatashardBuilder:
    def __init__(self, *, devices: list[str] = 'cuda', log: Logger = Logger()):
        if isinstance(devices, str):
            devices = [devices]
        self.devices = devices
        self.log = log

    def build_shards(self, shards: Sequence[Datashard], *ctx_args, **ctx_kwargs):
        if len(shards) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            progress_bar = tqdm.tqdm(total=len(shards))
            shard_lists = np.array_split(shards, len(self.devices))
            shard_offsets = np.cumsum([0] + [len(shard_list) for shard_list in shard_lists])
            threads = [
                threading.Thread(target=self.__build_shards__, args=(shard_list, ctx_args, ctx_kwargs, shard_offset, device, result_queue, done_queue, abort_event, progress_bar))
                for shard_list, shard_offset, device in zip(shard_lists, shard_offsets, self.devices)
            ]
            done_idxs = []
            for thread in threads:
                thread.start()
            while len(done_idxs) < len(shards):
                success, idx, payload = result_queue.get()
                if success:
                    done_idxs.append(idx)
                    e = None
                else:
                    e = payload
                    self.log.info(f"Received error from shard with index {idx}: {shards[idx]}. Abandoning result_queue polling.")
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
        return shards
    
    def __build_shards__(self, shards: Sequence[Datashard], ctx_args, ctx_kwargs, offset: int, device: str, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event, progress_bar):
        self.log.debug(f"Building {len(shards)} feature shards on device: {device}")
        device_ctx_args, device_ctx_kwargs = self.__args_kwargs_to_device__(ctx_args, ctx_kwargs, device)
        for i, shard in enumerate(shards):
            exception = None
            try:
                if abort_event.is_set():
                    break
                shard.to(device).build(*device_ctx_args, **device_ctx_kwargs).to('cpu')
            except Exception as e:
                exception = e
                self.log.info(f"ERROR building feature shard {shard} on device: {device}")
            if exception is not None:
                result_queue.put((False, offset+i, exception))
                break
            result_queue.put((True, offset+i, None))
            progress_bar.update(1)
        del device_ctx_args, device_ctx_kwargs
        gc.collect()
        if exception is None:
            self.log.debug(f"Done building {len(shards)} feature shards on device: {device}")
        else:
            self.log.debug(f"Abandoning building {len(shards)} feature shards on device: {device} due to an exception")
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
    


