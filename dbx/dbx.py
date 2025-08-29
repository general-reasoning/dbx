from collections.abc import Iterable
from dataclasses import dataclass, fields, asdict, replace
import datetime
import hashlib
import importlib
import inspect
import json
import logging
import os
import sys
import traceback as tb
import typing
from typing import Union, Optional
import uuid

import git

import tqdm

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
        name: Optional[str] = None,
        stack_depth: int = 2,
    ):
        self.allowed = []
        if warning:
            self.allowed.append("WARNING")
        if info:
            self.allowed.append("INFO")
        if debug:
            self.allowed.append("DEBUG")
        if verbose:
            self.allowed.append("VERBOSE")
        if select:
            self.allowed.append("SELECT")
        self.stack_depth = stack_depth
        self.name = name
        if self.name is None:
            self.name = f"{inspect.stack()[stack_depth].function}"

    def _print(self, prefix, msg):
        if prefix in self.allowed:
            print(f"{prefix}: {self.name}: {msg}")

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

    def silent(self, mst):
        pass


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
                    val = eval_term(v)
                    kwargs[k] = val
                else:
                    arg = eval_term(bit)
                    args.append(arg)
        return args, kwargs

    def get_named_func_args_kwargs(name):
        # TODO: replace with a regex
        lb = name.find("(")
        rb = name.rfind(")")
        if lb == -1 or rb == -1:
            lb = name.find("[")
            rb = name.find("]")
        if lb == -1 or rb == -1:
            func = None
            args = None
            kwargs = None
        else:
            funcstr = name[:lb]
            func, _ = get_named_const_and_cxt(funcstr)
            argkwargstr = name[lb + 1 : rb]
            args, kwargs = get_named_args_kwargs(argkwargstr)
        return func, args, kwargs

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
            func, args, kwargs = get_named_func_args_kwargs(_name_)
            if func is None:
                term, _ = get_named_const_and_cxt(_name_)
            else:
                term = func(*args, **kwargs)
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
    rb = s.rfind(")")
    _, cxt = get_named_const_and_cxt(s[:lb])
    r = eval(s, globals(), cxt)
    return r


def exec_print(argstr=None):
    print(exec(argstr))


def write_json(data, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "w") as f:
        json.dump(data, f)
        log.info(f"WROTE {path}")


def read_json(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        data = json.load(f)
        log.info(f"READ {path}")
    return data


def write_tensor(tensor, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    array = tensor.numpy()
    with fs.open(path, "wb") as f:
        np.save(f, array)
        log.info(f"WROTE {path}")

def read_tensor(path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "rb") as f:
        array = np.load(f)
        log.info(f"READ {path}")
        tensor = torch.from_numpy(array)
    return tensor


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
    FILES = {'topic', 'file.csv'} | FILE = 'file.csv'
    # protocol://path --- module/class/# --- topic --- file 
    #        root           [anchor]        [topic]   [file]
    # root:       'protocol://path/to/root'
    # anchorpath: '{root}/modpath/class'|'{root}' if not unmoored|else
    # hashpath:   '{anchorpath}/#/{hash}|{anchorpath}/{hash}' if hash supplied through args|else
    # dirpath:    '{hashpath}/topic'|{hashpath}' if topic is not None|else
    # path:       '{dirpath}/{FILE}'|'{dirpath}' if FILE is not None|else
    
    """
    @dataclass
    class CONFIG:
        class LazyGetter:
            def __init__(self, term):
                self.term = term
                self.value = None
            def __call__(self):
                if self.value is None:
                    self.value = eval_term(self.term)
                return self.value

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if isinstance(attr, Datablock.CONFIG.LazyGetter):
                return attr()
            return attr

    def __init__(
        self,
        root: str = None,
        verbose: bool = False,
        debug: bool = False,
        *,
        cfg: Optional[Union[str, dict]] = None,
        hash: Optional[str] = None,
        unmoored: bool = False,
        capture_build_output: bool = False,
        gitrepo: str  = None,
    ):
        self.__setstate__(dict(
            root=root,
            verbose=verbose,
            debug=debug,
            cfg=cfg,
            hash=hash,
            unmoored=unmoored,
            capture_build_output=capture_build_output,
            gitrepo=gitrepo,
        ))
        
    def __setstate__(
        self,
        kwargs,
    ):
        self.root = kwargs.get('root')
        self.verbose = kwargs.get('verbose')
        self.debug = kwargs.get('debug')
        self.cfg = kwargs.get('cfg')
        self._hash = kwargs.get('hash')
        self.unmoored = kwargs.get('unmoored')
        self.capture_build_output = kwargs.get('capture_build_output')
        self.gitrepo = kwargs.get('gitrepo')

        if self.root is None:
            self.root = os.environ.get('DBKSPACE')
            if self.root is None:
                mod = importlib.import_module(self.__module__)
                if hasattr(mod, 'DBKSPACE'):
                    self.root = getattr(
                    mod,
                    'DBKSPACE',
                )
            if self.root is None:
                if hasattr(self, 'ROOT'):
                    self.root = self.ROOT
                
        self.log = Logger(
            debug=self.debug,
            verbose=self.verbose,
            name=self.anchor(),
        )
        if self.gitrepo is None:
                mod = importlib.import_module(self.__module__)
                if hasattr(mod, 'DBKREPO'):
                    self.gitrepo = getattr(
                    mod,
                    'DBKREPO',
                )
        if isinstance(self.cfg, str):
            self.cfg = read_json(self.cfg, debug=self.debug)
        if self.cfg is None:
            self.cfg = asdict(self.CONFIG())
        self.config = self._cfg_to_config(self.cfg)
        self._autohash = self._hash is None
        self.tag = None
        self.__post_init__()


    def __getstate__(self):
        return self.kwargs()


    def __post_init__(self):
        ...

    @property
    def uuid(self):
        if not hasattr(self, '_uuid'):
            self._uuid = str(uuid.uuid4())
        return self._uuid

    def kwargs(self):
        return dict(
            root=self.root,
            verbose=self.verbose,
            debug=self.debug,
            cfg=self.cfg,
            hash=self._hash if not self._autohash else None,
            unmoored=self.unmoored,
            capture_build_output=self.capture_build_output,
            gitrepo=self.gitrepo,
        )

    def path(
        self,
        topic=None,
        *,
        ensure_dirpath: bool = False,
    ):
        if topic is None:
            dirpath = self.dirpath()
            topicfile = self.FILE
        else:
            dirpath = self.dirpath(topic)
            topicfile = self.FILES[topic]
        path = os.path.join(dirpath, topicfile) if topicfile is not None else dirpath
        if ensure_dirpath:
            self.ensure_path(dirpath)      
        return path

    def ensure_path(self, path):
        fs, _ = fsspec.url_to_fs(path)
        fs.makedirs(path, exist_ok=True)
        return self

    def url(self, topic=None, *, redirect=None):
        path = self.path(topic)
        return make_download_url(path)

    def valid(
        self,
    ):
        def validpath(path):
            if path.endswith("None"): #If topic filename ends with 'None', it is considered to be valid by default
                result = True
            else:
                fs, _ = fsspec.url_to_fs(path)
                if 'file' not in fs.protocol:
                    result = fsspec.filesystem("gcs").exists(path)
                else:
                    result = os.path.exists(path) #TODO: Why not handle this case using fsspec? 
            self.log.debug(f"path {path} valid: {result}") 
            return result

        results = []

        if hasattr(self, "FILES"):
            results += [
                validpath(self.path(topic))
                for topic in self.FILES
            ]
        else:
            results += [validpath(self.path())]
        result = all(results)
        self.log.debug(f"{results=}")
        return result

    def build(self, tag:str = None):
        if self.capture_build_output:
            stdout = sys.stdout
            outfs, _ = fsspec.url_to_fs(self.capture_out)
            captured_stdout_stream = outfs.open(self.logpath(), "w", encoding="utf-8")
            sys.stdout = captured_stdout_stream
        try:
            if not self.valid():
                self.__pre_build__(tag).__build__().__post_build__(tag)
            else:
                self.log.verbose(f"Skipping existing datablock: {self.dirpath()}")
        finally:
            if self.capture_build_output:
                sys.stdout = stdout
                captured_stdout_stream.close()
        return self

    def __pre_build__(self, tag: str = None):
        self._write_scope(version=self.version)
        self._write_journal_entry(event="build:start", tag=tag, version=self.version)
        self.tag = tag
        return self

    def __build__(self):
        return self

    def __post_build__(self, tag: str = None):
        if tag is not None:
            assert tag == self.tag, f"Tag mismatch: {tag=} != {self.tag=}"
        self.tag = None
        self._write_journal_entry(event="build:end", tag=tag, version=self.version)
        return self
    
    @property
    def version(self):
        if not hasattr(self, '_version'):
            self._version = gitrevision(self.gitrepo, log=self.log) if self.gitrepo is not None else None
        return self._version


    def leave_breadcrumbs(self):
        if hasattr(self, "FILES"):
            for topic in self.FILES:
                self.dirpath(topic, ensure=True)
                self._leave_breadcrumbs_at_path(self.path(topic))
        else:
            self.dirpath(ensure=True)
            self._leave_breadcrumbs_at_path(self.path())
        return self

    def read(self, topic=None):
        raise NotImplementedError()
        return self
    
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
        if hasattr(self, "FILES"):
            for topic in self.FILES:
                clear_dirpath(self.dirpath(topic))
        else:
            clear_dirpath(self.dirpath())
        self._write_journal_entry(event="UNSAFE_clear", version=self.version)
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
        ) if not self.unmoored else self.root
        return anchorpath
    
    @property
    def hash(self):
        if self._hash is None:
            hivehandle = os.path.join(
                *[f"{key}={val}" for key, val in self.cfg.items()] #TODO: scopecfg?
            )
            sha = hashlib.sha256()
            sha.update(hivehandle.encode())
            self._hash = sha.hexdigest()
        return self._hash
            
    def scope(self):
        versionpath = self.Versionpath(self.root, self.hash)
        vfs, _ = fsspec.url_to_fs(versionpath)
        with vfs.open(versionpath, 'r') as vf:
            version = vf.read()
        #
        jconfigpath = self.Cfgpath(self.root, self.hash, 'json')
        jcfs, _ = fsspec.url_to_fs(jconfigpath)
        with jcfs.open(jconfigpath, 'r') as jcf:
            cfg = json.load(jcf)
        return version, cfg
    
    @staticmethod
    def Scopes(cls, root):
        scopeanchorpath = cls._scopeanchorpath(root)
        fs, _ = fsspec.url_to_fs(scopeanchorpath)
        paths = list(fs.ls(scopeanchorpath))
        hashes = [f.removeprefix(scopeanchorpath).removeprefix('/') for f in paths]
        cfgfiles_ = [os.path.join(path, 'cfg.parquet') for path in paths]
        cfgfiles = [f for f in cfgfiles_ if fs.exists(f)]
        if len(cfgfiles) > 0:
            ds = pq.ParquetDataset(cfgfiles, filesystem=fs)
            df = ds.read().to_pandas()
            df.index = hashes
        else:
            df = pd.DataFrame(index=hashes)
        verfiles = [os.path.join(path, 'version') for path in paths]
        versions = []
        for verfile in verfiles:
            #verfs = makefs(scopeanchorpath) # TODO: REMOVE
            verfs, _ = fsspec.url_to_fs(scopeanchorpath)
            with verfs.open(verfile, 'r') as verf:
                version = verf.read()
                versions.append(version)
        df['version'] = versions
        # The index is the dirpath to the file, ending in the hash, so we extract it and make a column
        df['hash'] = [path.split('/')[-1] for path in df.index]
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def Journal(cls, root):
        journaldirpath = cls._journalanchorpath(root)
        fs, _ = fsspec.url_to_fs(journaldirpath)
        files = list(fs.ls(journaldirpath))
        if len(files) > 0:
            ds = pq.ParquetDataset(files, filesystem=fs)
            df = ds.read().to_pandas().sort_values('datetime', ascending=False)
        else:
            df = pd.DataFrame()
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
            "scope",
            "#",
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
    def Versionpath(cls, root, hash, *, ensure: bool = True):
        return os.path.join(cls._scopepath(root, hash, ensure=ensure), 'version')
    
    @classmethod
    def Cfgpath(cls, root, hash, kind, *, ensure: bool = True):
        if kind == 'json':
            return os.path.join(cls._scopepath(root, hash, ensure=ensure), 'cfg.json')
        elif kind == 'parquet':
            return os.path.join(cls._scopepath(root, hash, ensure=ensure), 'cfg.parquet')
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
            "journal",
        )
        journalanchorpath = os.path.join(
            root,
            journalanchor,
            '#',
        )
        if ensure:
            fs, _ = fsspec.url_to_fs(journalanchorpath)
            fs.makedirs(journalanchorpath, exist_ok=True)
        return journalanchorpath

    def hashpath(self, *, ensure: bool = True):
        anchorpath = self.anchorpath()
        if self._autohash:
            hashpath = os.path.join(anchorpath, '#', self.hash)
        else:
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
            assert topic in self.FILES, f"Topic {repr(topic)} not in {self.FILES}"
            dirpath = os.path.join(hashpath, topic)
        else:
            dirpath = hashpath
        if ensure:
            fs, _ = fsspec.url_to_fs(dirpath)
            fs.makedirs(dirpath, exist_ok=True)
        return dirpath

    def _make_scopecfg(self):
        scopecfg = {}
        for k, v in self.cfg.items():
            value = getattr(self.config, k)
            if isinstance(v, str) and v.endswith('#') and isinstance(value, Datablock):
                scopecfg[k] = f"{v}#{value.hash()}"
            else:
                scopecfg[k] = v
        return scopecfg

    def _write_scope(self, *, version):
        versionpath = self.Versionpath(self.root, self.hash)
        vfs, _ = fsspec.url_to_fs(versionpath)
        with vfs.open(versionpath, 'w') as vf:
            vf.write(str(version))
        assert vfs.exists(versionpath), f"Versionpath {versionpath} does not exist after writing"
        #
        scopecfg = self._make_scopecfg()
        #
        jconfigpath = self.Cfgpath(self.root, self.hash, 'json')
        jcfs, _ = fsspec.url_to_fs(jconfigpath)
        with jcfs.open(jconfigpath, 'w') as jcf:
            json.dump(scopecfg, jcf)
        assert jcfs.exists(jconfigpath), f"Configpath {jconfigpath} does not exist after writing"
        #
        pconfigpath = self.Cfgpath(self.root, self.hash, 'parquet')
        pcfs, _ = fsspec.url_to_fs(pconfigpath)
        cfgdf = pd.DataFrame.from_records([scopecfg])
        cfgdf.to_parquet(pconfigpath)
        assert pcfs.exists(pconfigpath), f"Configpath {pconfigpath} does not exist after writing"
        #
        self.log.verbose(f"wrote SCOPE: versionpath: {versionpath} and configpaths: {jconfigpath} and {pconfigpath}")

    def _write_journal_entry(self, event:str, tag:str=None, *, version):
        hash = self.hash
        dt = str(datetime.datetime.now()).replace(' ', '-')
        path = os.path.join(self._journalanchorpath(self.root), f"{hash}-{dt}.parquet")
        df = pd.DataFrame.from_records([{'hash': hash, 
                                         'uuid': self.uuid, 
                                         'version': version, 
                                         'event': event, 
                                         'tag': tag, 
                                         'datetime': dt, 
                                         'build_log': self.logpath()}])
        self.log.verbose(f"Wrote JOURNAL entry for event {repr(event)} with tag {repr(tag)} to path {path}")
        df.to_parquet(path)
    
    def _cfg_to_config(self, cfg):
        config = self.CONFIG(**cfg)
        replacements = {}
        for field in fields(config):
            term = getattr(config, field.name)
            if issubclass(self.CONFIG, Datablock.CONFIG):
                getter = Datablock.CONFIG.LazyGetter(term)
            else:
                getter = eval_term(term)
            replacements[field.name] = getter
        config = replace(config, **replacements)
        return config

    def _leave_breadcrumbs_at_path(self, path):
        fs, _ = fsspec.url_to_fs(path)
        with fs.open(path, "w") as f:
            f.write("")


def datablock_method(
    datablock_cls,
    method,
    *,
    root: str = None,
    verbose: bool = False,
    debug: bool = False,
    cfg: dict,
    hash: Optional[str] = None,
    unmoored: bool = False,
    capture_build_output: bool = False,
    gitrepo: str  = None,
    kwargs: dict = dict(),
):
    datablock_cls = eval_term(datablock_cls)
    datablock = datablock_cls(root=root, verbose=verbose, debug=debug, cfg=cfg, hash=hash, unmoored=unmoored, capture_build_output=capture_build_output, gitrepo=gitrepo)
    method_callable = getattr(datablock, method)
    return method_callable(**kwargs)

    
class BatchBuilder:
    @property
    def tag(self):
        raise NotImplementedError()
    
    def __call__(datablock_cls, kwargslist):
        raise NotImplementedError()
        

class Databatch(Datablock):
    DATABLOCK = None
    FILE = "summary.csv"

    def __init__(
        self,
        root: str = None,
        verbose: bool = False,
        debug: bool = False,
        builder: BatchBuilder = None,
        *,
        cfg: Optional[Union[str,dict]] = None,
    ):
        super().__init__(root, verbose, debug, cfg=cfg)
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
                tagi = f"run:{i}:{self.hash}"
                if tag is not None:
                    tagi = f"{tag}:{tagi}"
                datablock.build(tag=tagi)
        else:
            tag = self.tag or (self.runner.tag if hasattr(self.runner, 'tag') else None)
            dataset_method_build_kwargslist = []
            for i, datablock in enumerate(self.datablocks()):
                tagi = f"run:{i}:{self.hash}"
                if tag is not None:
                    tagi = f"{tag}:{tagi}"
                dataset_method_build_kwargs = dict(
                    **datablock.kwargs(),
                    **{'kwargs': dict(tag=tagi,),}
                )
                dataset_method_build_kwargslist.append(dataset_method_build_kwargs)
            self.builder(self.DATABLOCK, dataset_method_build_kwargslist)
        return self

    def run(self):
        return self.build(tag=self.tag)
    
    def datablock_scopes(self):
        return self.Scopes(self.DATABLOCK, self.root)

    def datablock_journal(self):
        return self.Journal(self.DATABLOCK, self.root)

    
