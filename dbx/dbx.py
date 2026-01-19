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
import pickle
import pprint as _pprint_
import queue
import sys
import threading
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


def journal(cls, root=None):
    return eval_term(cls).Journal(root)


class Logger:
    """Because Python logging is so cumbersome to initialize, configure and control, we have this."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        warning: bool = True,
        info: bool = True,
        verbose: bool = False,
        debug: bool = False,
        selected: bool = True,
        detailed: bool = False,
        datetime: bool = True,
        stack_depth: int = 2,
    ):
        self._warning_ = eval(os.environ.get('DBXLOGWARNING', str(warning)))
        self._info_ = eval(os.environ.get('DBXLOGINFO', str(info)))
        self._verbose_ = eval(os.environ.get('DBXLOGVERBOSE', str(verbose)))
        self._selected_ = eval(os.environ.get('DBXLOGSELECTED', str(selected)))
        self._debug_ = eval(os.environ.get('DBXLOGDEBUG', str(debug)))
        self._detailed_ = eval(os.environ.get('DBXLOGDETAILED', str(detailed)))
        
        self.allowed = ["ERROR"]
        if self._warning_:
            self.allowed.append("WARNING")
        if self._info_:
            self.allowed.append("INFO")
        if self._debug_:
            self.allowed.append("DEBUG")
        if self._verbose_:
            self.allowed.append("VERBOSE")
        if self._selected_:
            self.allowed.append("SELECTED")
        if self._detailed_:
            self.allowed.append("DETAILED")
        self.stack_depth = stack_depth
        self.name = name
        self.datetime = datetime

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


class JournalEntry(pd.Series):
    def __init__(self, series: pd.Series, *, logger: Logger = Logger(name="JournalEntry")):
        super().__init__(series)
        self.logger = logger

    def read(self, *things, raw: bool = False, deslash: bool = False):
        def read_thing(thing):
            if hasattr(self, thing) and getattr(self, thing) is not None:
                path = getattr(self, thing)
                _, _ext = os.path.splitext(path)
                ext = _ext[1:]
                if raw or ext == 'txt' or ext == 'log':
                    result = read_str(getattr(self, thing))
                elif ext == 'yaml':
                    result = read_yaml(getattr(self, thing))
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
    
    def eval(self, thing, *, debug: bool = False, context={}, revision=None):
        _eval_ = globals()['eval']
        exc = None
        thingstr = self.read(thing, raw=True)
        r = None
        try:
            r = _eval_(thingstr, globals(), context)
        except Exception as e:
            if debug:
                exc = e
                breakpoint()
            else:
                raise exc
        return r
    

class JournalFrame(pd.DataFrame):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def list(self, thing, *, take: str = 'last', sortby: Optional[str] = None, raw: bool = False):
        if take == 'last':
            unique_rows = self.groupby('hash').last()
        elif take == 'first':
            unique_rows = self.groupby('hash').first()
        elif take == 'all':
            unique_rows = self.set_index('hash')
        else:
            raise ValueError(f"Unknown take value: {take}")
        hashes = []
        entries = []
        for hash, row in unique_rows.iterrows():
            entries.append(JournalEntry(row).read(thing, raw=raw))
            hashes.append(hash)
        if raw:
            thingsframe = pd.DataFrame.from_dict({hash: entry for hash, entry in zip(hashes, entries)}, orient='index')
            thingsframe.columns = [thing]
            thingsframe.index.name = 'hash'
            thingsframe = thingsframe.reset_index()
        else:
            thingsframe = pd.DataFrame.from_records(entries)
        thingsframe['hash'] = hashes
        if sortby is not None and sortby in thingsframe.columns:
            thingsframe = thingsframe.sort_values(sortby).set_index(sortby).reset_index() # force sortby to be the first column
        return thingsframe

    
def gitrevision(repopath, *, log=Logger()):
    if repopath is not None:
        repo = git.Repo(repopath)
        if repo.is_dirty():
            raise ValueError(f"Dirty git repo: {repopath}: commit your changes")
        branch = repo.active_branch.name
        reponame = os.path.basename(repopath)
        revision = f"{reponame}:{repo.rev_parse(branch).hexsha}"
        log.selected(f"Obtained git revision for git repo {repopath}: {revision}")
    else:
        revision = None
    return revision


def make_google_cloud_storage_download_url(path):
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
                term = eval(_name_, cxt)
        else:
            term = name
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


def pprint(argstr=None):
    _pprint_.pprint(exec(argstr))


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


def read_yaml(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        data = yaml.safe_load(f)
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
    # protocol://path --- module/class/ --- topic --- file 
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
        version: str
        revision: str
        kwargs: dict
        spec: dict
        quote: str
        repr: str
        hashhandle: str

        def deslash(self, attr):
            a = getattr(self, attr)
            if isinstance(a, str):
                aa = a.replace('\\', '')
            else:
                aa = a
            return aa

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
        info: bool = True,
        verbose: bool = False,
        debug: bool = False,
        detailed: bool = False,
        capture_output: bool = False,
        revision: str = None,
        gitrepo: str  = None,
        device: str = 'cpu',
        **kwargs,
    ):
        self.__setstate__(
            root=root,
            spec=spec,
            anchored=anchored,
            hash=hash,
            tag=tag,
            info=info,
            verbose=verbose,
            debug=debug,
            detailed=detailed,
            capture_output=capture_output,
            revision=revision,
            gitrepo=gitrepo,
            device=device,
            **kwargs,
        )
        
    def __setstate__(
        self,
        *,
        root: str = None,
        spec: Optional[Union[str, dict]] = None,
        anchored: bool = True,
        hash: Optional[str] = None,
        tag: Optional[str] = None,
        info: bool = True,
        verbose: bool = False,
        debug: bool = False,
        detailed: bool = False,
        capture_output: bool = False,
        revision: str = None,
        gitrepo: str  = None,
        device: str = 'cpu',
        **kwargs,
    ):
        """NB: signature must match __init__'s"""
        self._root_ = root
        if self._root_ is None:
            self.root = os.environ.get('DBXROOT')
        if self.root is None:
            raise ValueError(f"None root for {self.__class__.__name__}: maybe set DBXROOT?")
        self._spec_ = spec
        if self._spec_ is None:
            self.spec = asdict(self.CONFIG())
        else:
            self.spec = self._spec_
        self.anchored = anchored
        self._hash_ = hash
        self.tag = tag
        self.log = Logger(
            f"{self.anchor}",
            debug=kwargs.get('debug'),
            verbose=kwargs.get('verbose'),
            detailed=kwargs.get('detailed'),
            info=kwargs.get('info'),
            stack_depth=None, #TODO: restore stack_depth default
        )
        self._revision_ = revision
        self.gitrepo = os.environ.get('DBXREPO', gitrepo)
        self.capture_output = bool(capture_output)
        self.device = device  
        #
        self.cfg = self._spec_to_cfg(self.spec)
        self.config = self.cfg # alias
        #
        for k, v in kwargs.items():
            setattr(self, k, v)
        #
        self.parameters = [
            'root', 'spec', 'anchored', 'hash', 'tag', 'info', 'verbose', 'debug',
            'detailed', 'capture_output', 'revision', 'gitrepo', 'device'
        ] + list(kwargs.keys())
        #
        self.dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        self.build_dt = None
        self.__post_init__()
        # redefined the logger using self.hash, which is generally invalid before __post_init__ (e.g., TOPICFILES may be undefined)
        self.log = Logger(
            name=f"{self.anchor}/{self.hash}",
            debug=debug,
            verbose=verbose,
            detailed=detailed,
            info=info,
            stack_depth=None, #TODO: restore stack_depth default
        )
        self.log.detailed(f"======--------------> bid: {self.bid}")

    def __getstate__(self):
        return {
            k: getattr(self, f"_{k}_") if hasattr(self, f"_{k}_") else getattr(self, k)
            for k in self.parameters
        }
    
    def set(self, **kwargs):
        _kwargs = copy.deepcopy(self.__getstate__())
        _kwargs.update(**kwargs)
        return self.__class__(**_kwargs)
    
    def replace(self, **kwargs):
        return self.set(**kwargs)
    
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
            spec=self.spec,
            kwargs=self.kwargs,
            quote=self.quote(),
            repr=self.__repr__(),
            hashhandle=self.hashhandle,
        )
    
    def validtopic(self, topic=None):
        if topic is None:
            valid = self.validpath(self.path())
        else:
            valid = self.validpath(self.path(topic))
        self.log.detailed(f"{self.anchor}: topic {topic} valid: {valid}")
        return valid
    
    def validtopics(self, topics):
        results = [
            self.validtopic(topic) for topic in topics
        ]
        return all(results)

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
        self.log.detailed(f"{self.anchor}: path {path} valid: {result}") 
        return result
    
    def validpaths(self,):
        results = {}
        if hasattr(self, "TOPICFILES"):
            for topic in self.TOPICFILES:
                results[topic] = self.validpath(self.path(topic))
        else:
            results += [self.validpath(self.path())]
        return results
    
    def valid(self,):
        validpaths = self.validpaths()
        if hasattr(self, "TOPICFILES"):
            results = validpaths.values()
        else:
            results = validpaths
        result = all(results)
        self.log.detailed(f"{self.anchor}: {results=}")
        return result
    
    def topics(self):
        return list(self.TOPICFILES.keys()) if self.has_topics() else []

    def has_topics(self):
        return hasattr(self, "TOPICFILES")
    
    def has_topic(self):
        return hasattr(self, "TOPICFILE")

    def build(self, *args, **kwargs):
        if self.capture_output:
            self.log.verbose(f"-------------------- Capturing stdout to {self._logpath()} ------------------")
            stdout = sys.stdout
            logpath = self._logpath()
            outfs, _ = fsspec.url_to_fs(logpath)
            captured_stdout_stream = outfs.open(logpath, "w", encoding="utf-8")
            sys.stdout = Tee(stdout, captured_stdout_stream)
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
                captured_stdout_stream.close()
        return self

    def __pre_build__(self, *args, **kwargs):
        self._write_journal_dict('spec', self.spec)
        self._write_journal_dict('kwargs', self.kwargs)
        self._write_str('quote', self.quote())
        self._write_str('repr', self.__repr__())
        self._write_str('hashhandle', self.hashhandle)
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

    def build_tree(self, *args, **kwargs):
        for s in self.spec.keys():
            c = getattr(self.cfg, s)
            if isinstance(c, Datablock):
                self.log.verbose(f"------------------------ Building subtree at {s} --------------------------------")
                c.build_tree(*args, **kwargs)   
        return self.build(*args, **kwargs)
    
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
    
    def UNSAFE_copy_from(self, anchorpath, *, overwrite: bool = False):
        if not overwrite:
            assert not self.valid(), f"Attempting to overwrite a valid Datablock {self}. Missing 'overwrite' argument?"
        fs, _ = fsspec.url_to_fs(anchorpath)
        hashpath = os.path.join(anchorpath, self.hash) 
        assert fs.isdir(hashpath), f"Nonexistent hashpath {hashpath}"
        self.log.verbose(f"Copying files from {hashpath}: BEGIN")
        if self.has_topics():
            for topic in self.topics():
                path = self.path(topic)
                if path is not None:
                    _path = os.path.join(hashpath, self.TOPICFILES[topic])
                    self.log.verbose(f"Copying {_path} to {path}")
                    fsspec.copy(_path, path)
        elif self.has_topic():
            path = self.path(topic)
            if path is not None:
                    _path = os.path.join(hashpath, self.TOPICFILES[topic])
                    self.log.verbose(f"Copying {_path} to {path}")
                    fsspec.copy(_path, path)
        self.log.verbose(f"Copying files from {hashpath}: END")
        assert self.valid(), f"Invalid Datablock after copy: {self}"

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
    
    def anchorpath(self):
        anchorpath = os.path.join(
            self.root,
            self.anchor,
        ) if self.anchored else self.root
        return anchorpath

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
        #DEBUG
        breakpoint()

        if not hasattr(self, '_revision'):
            self.log.selected(f"--------------> COMPUTING revision")
            if self._revision_ is None:
                self.log.selected(f"--------------> self._revision_ is None")
                self._revision = gitrevision(self.gitrepo, log=self.log) if self.gitrepo is not None else None
                self.log.selected(f"--------------> self._revision_: from gitrevision({self.gitrepo}")
            else:
                self.log.selected(f"--------------> Using {self._revision_=}")
                self._revision = self._revision_
        return self._revision

    def __expand_spec__(self, expansion='repr'):
        """
            . expansion: 'repr'|'quote'
            . spec lines
                . a specline is a str starting with '@', '$' or '#'
                . a strline: a non-specline str
                . an objline: a non-str object
            'repr':
                . FULL reduction
                    |specline: repr(eval_term(specline)) -- repr-reduction
                    |strline:  strline
                    |objline:  repr(objline)
            'quote':
                . UNREDUCED spec:
                    |specline: specline
                    |strline: strline
                    |Datablock objline: objline.quote()
                    |non-Datablock objline: repr(objline)  
        """
        _spec = {}
        if expansion == 'repr':
            #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
            # computed using the older version of these methods
            for k, v in self.spec.items():
                value = getattr(self.cfg, k)
                if isinstance(v, str):
                    if self.is_specline(v):
                        _spec[k] = repr(value)
                    else: 
                        _spec[k] = v
                else:
                    _spec[k] = repr(v)
        elif expansion == 'quote':
            for k, v in self.spec.items():
                value = getattr(self.cfg, k)
                if isinstance(value, str):
                    _spec[k] = v
                elif isinstance(value, Datablock):
                    _spec[k] = value.quote()
                else:
                    _spec[k] = repr(value)
        else:
            raise ValueError(f"Unknown expansion: {expansion}")
        return _spec
    
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
        tailkwargs = {
            k: v
            for k, v in self.__getstate__().items()
            if k not in ['root', 'anchored', 'hash', 'spec']          
        }
        self.log.detailed(f"{self.anchor}: _tailkwargs_: {tailkwargs=}")
        return tailkwargs
    
    def __repr_from_kwargs__(self, kwargs):
        def cite(x):
            return repr(x) if isinstance(x, str) else x

        kwargstrs = [f"{k}={cite(v)}" for k, v in kwargs.items()]
        kwargsrepr = ', '.join(kwargstrs)
        return f"{self.anchor}({kwargsrepr})"
    
    def quote(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        expanded_spec = self.__expand_spec__('full')
        quote = "$" + self.__repr_from_kwargs__({
            **{'spec': expanded_spec},
        })
        self.log.detailed(f"quote: ------------> {expanded_spec=}")
        self.log.detailed(f"quote: ------------> {quote=}")
        return quote
    
    def __repr__(self):
        expanded_spec = self.__expand_spec__('repr')
        r = self.__repr_from_kwargs__({
            **self._rootkwargs_,
            **{'spec': expanded_spec},
            **self._tailkwargs_,
        })
        self.log.detailed(f"__repr__(): ------------> {expanded_spec=}")
        self.log.detailed(f"__repr__(): ------------> __repr__={r}")
        return r
    
    def __str__(self):
        s = self.quote()
        s = s.replace('\\', '')
        return s
    
    @property
    def kwargs(self):
        return self.__getstate__()
    
    @property
    def hashhandle(self):
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hashes
        # computed using the older version of these methods
        if hasattr(self, "TOPICFILES"):
            topics = [f"topic:{topic}={file}" for topic, file in self.TOPICFILES.items()]
        else:
            topics = ["topics:None"]
        hashhandle = os.path.join(
            self.__repr__(),
            f"version={self.version}",
            *topics,
        )
        return hashhandle
    
    @property
    def hash(self): 
        #CAUTION! Changing this code may invalidate Datablocks that have already been computed and identified by their hash
        # computed with the older code.
        if not hasattr(self, '_hash'): 
            if self._hash_ is not None:
                self._hash = self._hash_
            else:
                sha = hashlib.sha256()
                sha.update(self.hashhandle.encode())
                self._hash = sha.hexdigest()
                self.log.detailed(f"hash: ---------===---------> {self.hashhandle=} ---> hash: {self._hash}")
        return self._hash
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
        self.log.debug(f"WROTE: {name.upper()}: yaml: {ypath}")
        #
        pqpath = self._xpath(name, 'parquet')
        pqfs, _ = fsspec.url_to_fs(pqpath)
        df = pd.DataFrame.from_records([{k: repr(v) for k, v in data.items()}])
        df.to_parquet(pqpath)
        assert pqfs.exists(pqpath), f"pqpath {pqpath} does not exist after writing"
        self.log.debug(f"WROTE: {name.upper()}: parquet: {pqpath}")

    def _write_str(self, name, text):
        #
        path = self._xpath(name, 'txt')
        fs, _ = fsspec.url_to_fs(path)
        write_str(text, path)
        assert fs.exists(path), f"scopepath {path} does not exist after writing"
        self.log.debug(f"WROTE: {name.upper()}: txt: {path}")

    def _write_journal_entry(self, event:str):
        hash = self.hash
        dt = datetime.datetime.now().isoformat().replace(' ', '-').replace(':', '-')
        key = f"{hash}-{dt}"

        spec_path = self._xpath('spec', 'yaml')
        kwargs_path = self._xpath('kwargs', 'yaml')
        quote_path = self._xpath('quote', 'txt')
        hashhandle_path = self._xpath('hashhandle', 'txt')
        repr_path = self._xpath('repr', 'txt')
        #
        logpath = self._logpath()
        if logpath is not None:
            logfs, _ = fsspec.url_to_fs(logpath)
            has_log = logfs.exists(logpath)
        else:
            has_log = False
        #
        journal_path = os.path.join(self._journalanchorpath(self.__class__, self.root), f"{key}.parquet")
        df = pd.DataFrame.from_records([{'datetime': dt,
                                         'build_datetime': self.build_dt,
                                         'version': self.version,
                                         'revision': self.revision, 
                                         'root': self.root,
                                         'anchor': self.anchor,
                                         'hash': hash,
                                         'tag': self.tag, 
                                         'log': logpath if has_log else None,
                                         'event': event,
                                         'spec': spec_path,
                                         'kwargs': kwargs_path,
                                         'quote': quote_path,
                                         'repr': repr_path,
                                         'hashhandle': hashhandle_path,
        }])
        df.to_parquet(journal_path)
        
        tagstr = "with tag {repr(self.tag)} " if self.tag is not None else ""
        self.log.debug(f"WROTE JOURNAL entry for event {repr(event)} {tagstr}"
                         f"to journal_path {journal_path}")

    #TODO: REMOVE in favor of Journal().list('spec')
    @staticmethod
    def Specs(cls, root):
        cls = eval_term(cls)
        if root is None:
            root = os.environ.get('DBXROOT')
        specanchorpath = cls._xanchorpath(root, 'spec')
        fs, _ = fsspec.url_to_fs(specanchorpath)
        if not fs.exists(specanchorpath):
            df = None
        else:
            paths = list(fs.ls(specanchorpath))
            specfiles_ = list(itertools.chain.from_iterable(
                fs.ls(path) for path in paths
            ))
            specfiles = [f for f in specfiles_ if fs.exists(f) and f.endswith('.parquet')]
            hashes = [os.path.dirname(f).removeprefix(specanchorpath).removeprefix('/') for f in specfiles]
            if len(specfiles) > 0:
                dfs = []
                for specfile in specfiles:
                    _df = pd.read_parquet(specfile)
                    dfs.append(_df)
                df = pd.concat(dfs)
                df.index = hashes
            else:
                df = pd.DataFrame(index=hashes)
            df = df.reset_index().rename(columns={'index': 'hash'})
        return df

    #TODO: REMOVE in favor of Journal().list('kwargs')
    @staticmethod
    def Kwargs(cls, root):
        cls = eval_term(cls)
        if root is None:
            root = os.environ.get('DBXROOT')
        kwargsanchorpath = cls._kwargsanchorpath(root)
        fs, _ = fsspec.url_to_fs(kwargsanchorpath)
        if not fs.exists(kwargsanchorpath):
            df = None
        else:
            paths = list(fs.ls(kwargsanchorpath))
            kwargsfiles_ = list(itertools.chain.from_iterable(
                fs.ls(path) for path in paths
            ))
            kwargsfiles = [f for f in kwargsfiles_ if fs.exists(f) and f.endswith('.parquet')]
            hashes = [os.path.dirname(f).removeprefix(kwargsanchorpath).removeprefix('/') for f in kwargsfiles]
            if len(kwargsfiles) > 0:
                dfs = []
                for kwargsfile in kwargsfiles:
                    _df = pd.read_parquet(kwargsfile)
                    dfs.append(_df)
                df = pd.concat(dfs)
                df.index = hashes
            else:
                df = pd.DataFrame(index=hashes)
            df = df.reset_index().rename(columns={'index': 'hash'})
        return df

    @classmethod
    def Journal(cls, entry: int = None, *, root=None):
        if root is None:
            root = os.environ.get('DBXROOT')
        journaldirpath = Datablock._journalanchorpath(eval_term(cls), root)
        fs, _ = fsspec.url_to_fs(journaldirpath)
        files = list(fs.ls(journaldirpath))
        parquet_files = [f for f in files if f.endswith('.parquet')]

        log = Logger()
        log.debug(f"READING JOURNAL: from {journaldirpath=}, files: {parquet_files}")
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
        if entry is not None:
            result = JournalEntry(df.loc[entry])
        else:
            result = JournalFrame(df)
        return result

    def journal(self, entry: int = None):
        return self.Journal(entry, root=self.root)
    #JOURNAL: END
    

def quote(obj, *args, tag="$", **kwargs):
    log = Logger()
    if not callable(obj):
        assert len(args) == 0, f"Nonempty args for a noncallable obj: {args}"
        assert len(kwargs) == 0, f"Nonempty kwargs for a noncallable obj: {kwargs}"
        if isinstance(obj, Datablock):
            _quote = obj.beta
        else:
            _quote = repr(obj)
        log.detailed(f"===============> Quoted {obj=} to {repr(_quote)}")
    else:
        func = obj
        argstrs = [quote(arg) for arg in args]
        kwargstrs = [f"{k}={quote(v)}" for k, v in kwargs.items()]
        argkwargstr = ','.join(argstrs+kwargstrs)
        _quote = f"{tag}{func.__module__}.{func.__qualname__}({argkwargstr})"
        log.detailed(f"Quoted {func=}, {args=}, {kwargs=} to {repr(_quote)}")
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
                else:
                    e = payload
                    self.log.info(f"Received error from callable with {callable_idx=} on thread {thread_idx}. Abandoning result_queue polling.")
                    break
                progress_bar.update(1)
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
        return payloads
    
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
    