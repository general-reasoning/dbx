from collections.abc import Iterable
from dataclasses import dataclass, fields
import datetime
import hashlib
import importlib
import inspect
import json
import os
import typing

import fsspec

from scipy.stats import qmc

import pandas as pd
import pyarrow.parquet as pq


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

    def _print(self, prefix, msg, *, stack_depth=2):
        if prefix in self.allowed:
            print(f"{prefix}: {inspect.stack()[stack_depth].function}: {msg}")

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


def make_download_url(path):
    if not path.startswith("gs://"):
        return None
    _path = path.removeprefix("gs://")
    return f"https://storage.cloud.google.com/{_path}"


def get_named_term(name):
    def get_named_const_and_cxt(name):
        bits = name.split(".")
        modbits = bits[:-1]
        prefix = None
        ctx = {}
        for modbit in modbits:
            if prefix is not None:
                modname = prefix + "." + modbit
            else:
                modname = modbit
            mod = importlib.import_module(modname)
            ctx[modname] = mod
            prefix = modname
        constname = bits[-1]
        const = getattr(mod, constname)
        return const, ctx

    def get_named_funcstr_and_argkwargstr(name):
        # TODO: replace with a regex
        lb = name.find("(")
        rb = name.find(")")
        if lb == -1 or rb == -1:
            lb = name.find("[")
            rb = name.find("]")
        if lb == -1 or rb == -1:
            funcstr = None
            argkwargstr = None
        else:
            funcstr = name[:lb]
            argkwargstr = name[lb + 1 : rb]
        return funcstr, argkwargstr

    if isinstance(name, Iterable) and not isinstance(name, str):
        term = [get_named_term(item) for item in name]
    elif isinstance(name, str):
        if not name.startswith("[") or not name.endswith("]"):
            term = name
        else:
            _name_ = name[1:-1]
            funcstr, argkwargstr = get_named_funcstr_and_argkwargstr(_name_)
            if funcstr is None:
                term, _ = get_named_const_and_cxt(_name_)
            else:
                _, ctx = get_named_const_and_cxt(funcstr)
                term = eval(f"{funcstr}({argkwargstr})", globals(), ctx)
    else:
        term = name
    return term


def write_json(data, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "w") as f:
        json.dump(data, f)
        log.info(f"WROTE {path}")


def read_json(data, path, *, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "r") as f:
        data = json.load(f)
        log.info(f"READ {path}")
    return data


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
        TOPICS = {'topic', 'file.csv'} | TOPIC = 'file.csv'
    """
    @dataclass
    class CONFIG:
        ...

    def __init__(
        self,
        root: str = None,
        verbose: bool = False,
        debug: bool = False,
        version: str  = None,
        *,
        cfg: dict,
    ):
        self.root = root
        if self.root is None:
            self.root = os.environ.get('DATALAKE', None)
        self.verbose = verbose
        self.debug = debug
        self.version = version
        self.log = Logger(
            debug=debug,
            verbose=verbose,
        )
        self.cfg = cfg
        self.config = self._inject_cfg(cfg)
        self.__post_init__()

    def __post_init__(self):
        ...

    def kwargs(self):
        return dict(
            root=self.root,
            verbose=self.verbose,
            debug=self.debug,
            version=self.version,
            cfg=self.cfg,
        )

    def path(
        self,
        topic=None,
    ):
        if topic is None:
            path = os.path.join(
                self._dirpath(),
                self.TOPIC,
            )
        else:
            path = os.path.join(self._dirpath(topic), self.TOPICS[topic])
        return path

    def url(self, topic=None, *, redirect=None):
        path = self.path(topic)
        return make_download_url(path)

    def valid(
        self,
    ):
        def validpath(path):
            if path.endswith("None"): #TODO: clarify
                return True
            elif path.startswith("gs://"): #TODO: generalize to other filesystems
                result = fsspec.filesystem("gcs").exists(path)
            else:
                result = os.path.exists(path)
            return result

        results = []

        if hasattr(self, "TOPICS"):
            results += [
                validpath(self.path(topic))
                for topic in self.TOPICS
                if self.TOPICS[topic] is not None
            ]
        else:
            results += [validpath(self.path())]
        result = all(results)
        return result

    def build(self, tag:str = None):
        return self.__pre_build__(tag).__build__(tag).__post_build__(tag)

    def __pre_build__(self, tag: str = None):
        self._write_scope()
        return self

    def __build__(self, tag: str = None):
        return self

    def __post_build__(self, tag:str = None):
        self._write_journal_entry(event="build", tag=tag)
        return self

    def read(self, topic=None):
        raise NotImplementedError()
        return self
    
    def UNSAFE_clear(self):
        def clear_dirpath(dirpath, *, throw=False):
            self.log.info(f"removing {dirpath} to overwrite output")
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
        if hasattr(self, "TOPICS"):
            for topic in self.TOPICS:
                clear_dirpath(self._dirpath(topic))
        else:
            clear_dirpath(self._dirpath())
        self._write_journal_entry(event="UNSAFE_clear")
        return self
    
    def hash(self): 
        hivehandle = os.path.join(
            *[f"{key}={val}" for key, val in self.cfg.items()]
        )
        sha = hashlib.sha256()
        sha.update(hivehandle.encode())
        hash = sha.hexdigest()
        return hash
            
    def scope(self):
        versionpath = self.Versionpath(self.root, self.hash())
        vfs, _ = fsspec.url_to_fs(versionpath)
        with vfs.open(versionpath, 'r') as vf:
            version = vf.read()
        #
        jconfigpath = self.Configpath(self.root, self.hash(), 'json')
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
        cfgfiles = [os.path.join(path, 'config.parquet') for path in paths]
        ds = pq.ParquetDataset(cfgfiles, filesystem=fs)
        df = ds.read().to_pandas()
        df.index = hashes
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

    def scopes(self):
        return self.Scopes(self.__class__, self.root)

    def journal(self):
        return self.Journal(self.__class__, self.root)
    
    @staticmethod
    def Journal(cls, root):
        journaldirpath = cls._journalanchorpath(root)
        #fs = makefs(journaldirpath) #TODO: REMOVE
        fs, _ = fsspec.url_to_fs(journaldirpath)
        ds = pq.ParquetDataset(list(fs.ls(journaldirpath)), filesystem=fs)
        df = ds.read().to_pandas()
        return df

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
    def _scopepath(cls, root, hash):
        scopeanchorpath = cls._scopeanchorpath(root)
        scopepath = os.path.join(
            scopeanchorpath,
            hash,
        )
        return scopepath

    @classmethod
    def _versionpath(cls, root, hash):
        return os.path.join(cls._scopepath(root, hash), 'version')
    
    @classmethod
    def _cfgpath(cls, root, hash, kind):
        if kind == 'json':
            return os.path.join(cls._scopepath(root, hash), 'cfg.json')
        elif kind == 'parquet':
            return os.path.join(cls._scopepath(root, hash), 'cfg.parquet')
        else:
            raise ValueError(f"Unknown configpath kind: {kind}")

    @classmethod
    def _journalanchorpath(cls, root):
        journalanchor = os.path.join(
            cls.__module__
            + "."
            + cls.__name__,
            "journal",
        )
        journalanchorpath = os.path.join(
            root,
            journalanchor,
        )
        return journalanchorpath
    
    def _dirpath(
        self,
        topic=None,
    ):
        anchor = (
            self.__module__
            + "."
            + self.__class__.__name__
            + "/#"
        )
        anchorpath = os.path.join(
            self.root,
            anchor,
        )
        if topic is not None:
            assert topic in self.TOPICS, f"Topic {topic} not in {self.TOPICS}"
            dirpath = os.path.join(anchorpath, self.hash(), topic)
        else:
            dirpath = os.path.join(anchorpath, self.hash())
        return dirpath

    def _write_scope(self):
        versionpath = self._versionpath(self.root, self.hash())
        vfs, _ = fsspec.url_to_fs(versionpath)
        with vfs.open(versionpath, 'w') as vf:
            vf.write(str(self.version))
        #
        jconfigpath = self._cfgpath(self.root, self.hash(), 'json')
        jcfs, _ = fsspec.url_to_fs(jconfigpath)
        with jcfs.open(jconfigpath, 'w') as jcf:
            json.dump(self.cfg, jcf)
        pconfigpath = self._cfgpath(self.root, self.hash(), 'parquet')
        cfgdf = pd.DataFrame.from_records([self.cfg])
        cfgdf.to_parquet(pconfigpath)
        self.log.verbose(f"Wrote SCOPE: versionpath: {versionpath} and configpaths: {jconfigpath} and {pconfigpath}")
    
    def _write_journal_entry(self, event:str, tag:str=None):
        hash = self.hash()
        dt = str(datetime.datetime.now()).replace(' ', '-')
        path = os.path.join(self._journalanchorpath(self.root), f"{hash}-{dt}.parquet")
        df = pd.DataFrame.from_records([{'hash': hash, 'version': self.version, 'event': event, 'tag': tag, 'datetime': dt}])
        self.log.verbose(f"Wrote JOURNAL entry for event {repr(event)} with tag {repr(tag)} to path {path}")
        df.to_parquet(path)
    
    def _inject_cfg(self, cfg):
        config = self.CONFIG(**cfg)
        for field in fields(config):
            setattr(self, field.name, get_named_term(getattr(config, field.name)))
        return config


def datablock_build(
    *,
    datablock_cls,
    root: str = None,
    verbose: bool = False,
    debug: bool = False,
    version: str  = None,
    tag: str  = None,
    cfg: dict,
):
    datablock_cls = get_named_term(datablock_cls)
    datablock = datablock_cls(root=root, verbose=verbose, debug=debug, version=version, cfg=cfg)
    return datablock.build(tag)


TAG = str
class BatchRunner:
    @property
    def tag(self):
        raise NotImplementedError()
    
    def __call__(func, kwargslist):
        raise NotImplementedError()
        

class Databatch(Datablock):
    DATABLOCK = None
    TOPIC = "summary.csv"

    def __init__(
        self,
        root: str = None,
        verbose: bool = False,
        debug: bool = False,
        version: str  = None,
        runner: BatchRunner = None,
        *,
        cfg: dict,
    ):
        super().__init__(root, verbose, debug, version, cfg=cfg)
        self.runner = runner

    def datablocks(self) -> typing.Iterable[Datablock]:
        raise NotImplementedError()
        return self

    def submit(self):
        kwargslist = [
            dict(
                datablock_cls=self.DATABLOCK,
                tag=self.runner.tag,
                **datablock.kwargs(),
            )
            for datablock in self.datablocks()
        ]
        if self.runner is None:
            for datablock in self.datablocks():
                datablock.build(tag="submit")
        else:
            self.runner(datablock_build, kwargslist)
        return self
    
    def summary(self, tag=None):
        return self.build(tag)

    def datablock_scopes(self):
        return self.Scopes(self.DATABLOCK, self.root)

    def datablock_journal(self):
        return self.Journal(self.DATABLOCK, self.root)

    
