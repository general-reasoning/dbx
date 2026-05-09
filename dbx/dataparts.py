"""
Standalone utility functions and classes for dbx.

This module contains components that do **not** depend on ``Datablock``:
Logger, Tee, I/O helpers, term evaluator, callable executors, etc.
"""
import collections
from collections.abc import Iterable, Sequence
import datetime
import functools
import gc
import hashlib
import importlib
import inspect
import json
import multiprocessing as mp
import os
import pickle
import pprint as _pprint_
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time as time_module
import traceback as tb
from typing import Union, Optional, Sequence, Callable
import uuid

import numpy as np
import fsspec
import pandas as pd
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
import tqdm
import yaml

# Disable tqdm's background TMonitor thread (see dbx.py for rationale).
tqdm.tqdm.monitor_interval = 0

__eval__ = __builtins__['eval'] if isinstance(__builtins__, dict) else getattr(__builtins__, 'eval')


def getenv(key: str) -> str:
    """Return the value of environment variable *key*.

    Intended to be referenced in speclines so that handles contain the
    symbolic ``$dbx.getenv('KEY')`` rather than the resolved path.
    """
    return os.environ[key]


def env(key: str) -> str:
    """Return a specline that lazily resolves an environment variable.

    Usage::

        from dbx import env
        MyBlock(root=env('FACTION_DATAROOT'), spec=dict(path=env('MY_PATH')))

    The returned string is a specline (``$``-prefixed) that the existing
    dbx eval/expand machinery keeps symbolic in handles and evaluates to
    the real value in cfg / __init__.
    """
    if isinstance(key, str) and key.startswith('$'):
        # Already a specline — idempotent.
        return key
    return f"$dbx.getenv('{key}')"


class Logger:
    """Lightweight, printf-style logger with level gating.

    Levels (from most to least severe): ``ERROR``, ``WARNING``, ``INFO``,
    ``VERBOSE``, ``SELECTED``, ``DEBUG``, ``DETAILED``.

    Each level can be enabled via constructor keyword, environment
    variable (``DBX_LOG_<LEVEL>``), or built-in default.

    Parameters
    ----------
    name : str, optional
        Prefix printed before every message.
    warning, info, verbose, debug, selected, detailed : bool, optional
        Override the default enable/disable for each level.
    selection : str or list[str], optional
        Fully-qualified function names that ``selected()`` should emit
        for.  Comma-separated string or list.  Falls back to
        ``DBX_LOG_SELECTION``.
    datetime : bool
        Include ISO-8601 timestamp in output (default ``True``).
    stack_depth : int
        Caller-frame offset for function-name tagging (default ``2``).
    """

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
                # Generate env key: 'warning' -> 'DBX_LOG_WARNING'
                env_key = f'DBX_LOG_{name.upper()}'
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
            self._selection_ = [s.strip() for s in self._selection_.split(',')]
        if len(self._selection_) == 0:
            self._selection_ = os.environ.get('DBX_LOG_SELECTION')
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

    def _fmt(self, msg, args):
        """Format message stdlib-style: msg % args when args are provided."""
        if args:
            try:
                return msg % args
            except (TypeError, ValueError):
                return f"{msg} {args}"
        return msg

    def error(self, msg, *args, **kwargs):
        self._print("ERROR", self._fmt(msg, args))

    def warning(self, msg, *args, **kwargs):
        self._print("WARNING", self._fmt(msg, args))

    def warn(self, msg, *args, **kwargs):
        self._print("WARNING", self._fmt(msg, args))

    def info(self, msg, *args, **kwargs):
        self._print("INFO", self._fmt(msg, args))

    def debug(self, msg, *args, **kwargs):
        self._print("DEBUG", self._fmt(msg, args))

    def verbose(self, msg, *args, **kwargs):
        self._print("VERBOSE", self._fmt(msg, args))

    def selected(self, msg, *args, **kwargs):
        if not self._selection_:
            return
        frame = sys._getframe(self.stack_depth - 1)
        module = frame.f_globals.get('__name__')
        qualname = frame.f_code.co_qualname  # e.g. "Datablock.tag" (Python 3.11+)
        function = frame.f_code.co_name      # e.g. "tag"
        fqn_full  = f"{module}.{qualname}"   # "dbx.datablocks.Datablock.tag"
        fqn_short = f"{module}.{function}"   # "dbx.datablocks.tag"
        if fqn_full not in self._selection_ and fqn_short not in self._selection_:
            return
        self._print("SELECTED", self._fmt(msg, args))

    def detailed(self, msg, *args, **kwargs):
        self._print("DETAILED", self._fmt(msg, args))

    def silent(self, msg, *args, **kwargs):
        pass



class Tee:
    """Write to multiple file-like objects simultaneously.

    Every :meth:`write` and :meth:`flush` is forwarded to all wrapped
    streams.  Attribute access for anything else is proxied to the
    first stream.
    """

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


def ensure_path(path, *, storage_options=None):
    """Create *path* and all parent directories (``fsspec``-aware)."""
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    fs.makedirs(path, exist_ok=True)


def fs_full_path(fs, path):
    """Return *path* with the protocol prefix restored for remote filesystems.

    For local (``file``) filesystems the bare path is returned unchanged
    so that it remains compatible with Python's built-in ``open()``.
    For remote filesystems (``abfs``, ``gcs``, ``memory``, …) the protocol
    is re-attached via ``fs.unstrip_protocol()``.

    Special case: ``adlfs`` (Azure) — ``unstrip_protocol()`` drops the
    ``@account.dfs.core.windows.net`` portion.  We reconstruct the full
    ``abfss://container@account.dfs.core.windows.net/path`` form so that
    downstream consumers (e.g. MosaicML) can extract the account name.
    """
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    if protocol in ('file', 'local', ''):
        return path
    # adlfs: unstrip_protocol() loses the account — reconstruct manually.
    if protocol in ('abfs', 'az') and hasattr(fs, 'account_name') and fs.account_name:
        # path = "container/sub/path" → split into container + rest
        parts = path.split('/', 1)
        container = parts[0]
        rest = parts[1] if len(parts) > 1 else ''
        host = f"{fs.account_name}.dfs.core.windows.net"
        return f"abfss://{container}@{host}/{rest}"
    return fs.unstrip_protocol(path)


def parse_storage_options(raw: str | None) -> dict:
    """Parse a ``key=val;key=val;...`` string into a dict.

    Used to decode the ``DBX_STORAGE_OPTIONS`` environment variable.
    Returns an empty dict when *raw* is ``None`` or empty.

    Examples
    --------
    >>> parse_storage_options("account_name=myacct;account_key=secret")
    {'account_name': 'myacct', 'account_key': 'secret'}
    >>> parse_storage_options(None)
    {}
    """
    if not raw:
        return {}
    opts = {}
    for pair in raw.split(';'):
        pair = pair.strip()
        if not pair:
            continue
        if '=' not in pair:
            raise ValueError(
                f"Invalid DBX_STORAGE_OPTIONS fragment {pair!r}: expected key=value"
            )
        k, v = pair.split('=', 1)
        opts[k.strip()] = v.strip()
    return opts


def default_storage_options() -> dict:
    """Return storage options from ``DBX_STORAGE_OPTIONS`` env var, or ``{}``."""
    return parse_storage_options(os.environ.get('DBX_STORAGE_OPTIONS'))


def UNSAFE_allowed(what: str, *, OVERRIDE: bool = False):
    """Prompt the user for confirmation before an unsafe operation.

    Returns ``True`` if the user confirms or *OVERRIDE* is ``True``.
    """
    if not OVERRIDE:
        response = input(f"ARE YOU SURE YOU WANT TO EXECUTE UNSAFE CODE: {what}? [y/N]")
        if response.lower() != 'y':
            return False
    return True



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



def eval(name):
    """Evaluate a dbx term expression.

    Strings prefixed with ``@``, ``$``, or ``#`` are resolved as
    Python expressions in a context that includes imported modules.
    Iterables are evaluated element-wise.  Plain strings and other
    objects are returned as-is.
    """
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

    Logger("eval").detailed(f" ====================> Evaluating term {repr(name)}")
    if isinstance(name, Iterable) and not isinstance(name, str):
        term = [eval(item) for item in name]
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



def exec(s=None, **kwargs):
    """Parse and execute a dbx expression from *s* or ``sys.argv``.

    When *s* is ``None``, the expression and keyword arguments are
    read from the command line (``sys.argv[1:]``).
    """
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


def pprint(argstr=None, **kwargs):
    """Execute a dbx expression and pretty-print the result."""
    _pprint_.pprint(exec(argstr, **kwargs))


def write_str(text, path, *, storage_options=None, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "w") as f:
        f.write(text)
        log.detailed(f"WROTE {path}")


def read_str(path, *, storage_options=None, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "r") as f:
        text = f.read()
        log.detailed(f"READ {path}")
    return text


def write_yaml(data, path, *, storage_options=None, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "w") as f:
        yaml.dump(data, f)
        log.detailed(f"WROTE {path}")


def read_yaml(path, *, storage_options=None, log=Logger(), safe: bool = False, debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.BaseLoader) if safe else yaml.load(f, Loader=yaml.UnsafeLoader)
        log.detailed(f"READ {path}")
    return data


def write_json(data, path, *, storage_options=None, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "w") as f:
        json.dump(data, f)
        log.detailed(f"WROTE {path}")


def read_json(path, *, storage_options=None, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "r") as f:
        data = json.load(f)
        log.detailed(f"READ {path}")
    return data


def write_tensor(tensor, path, *, storage_options=None, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    array = tensor.numpy()
    with fs.open(path, "wb") as f:
        np.save(f, array)
        log.detailed(f"WROTE {path}")


def read_tensor(path, *, storage_options=None, log=Logger(), debug: bool = False):
    import torch as _torch
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "rb") as f:
        array = np.load(f)
        log.detailed(f"READ {path}")
        tensor = _torch.from_numpy(array)
    return tensor


def write_tensors(path, *, log=Logger(), debug: bool = False, **tensors):
    arrays = {k: v.numpy() for k, v in tensors.items()}
    return write_npz(path, log=log, debug=debug, **arrays)


def read_tensors(path, *keys, log=Logger(), debug: bool = False):
    import torch as _torch
    arrays = read_npz(path, *keys, log=log, debug=debug)
    tensors = {k: _torch.from_numpy(v) for k, v in arrays.items()}
    return tensors


def write_npz(path, *, storage_options=None, log=Logger(), debug: bool = False, **kwargs):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "wb") as f:
        np.savez(f, **kwargs)
        log.detailed(f"WROTE {list(kwargs.keys())} to {path}")


def read_npz(path, *keys, storage_options=None, log=Logger(), debug: bool = False):
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, "rb") as f:
        data = np.load(f, allow_pickle=True)
        results = {k: data[k] for k in keys}
        log.detailed(f"READ {list(keys)} from {path}")
        return results
    

def write_pickle(obj, path, *, storage_options=None):
    """Pickle *obj* to *path* (any ``fsspec`` URL)."""
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(path, *, storage_options=None):
    """Unpickle and return the object stored at *path*."""
    fs, _ = fsspec.url_to_fs(path, **(storage_options or {}))
    with fs.open(path, 'rb') as f:
        return pickle.load(f)


def write_frame(frame: pd.DataFrame, path, *, storage_options=None, log=Logger(), **kwargs):
    """Write a pandas DataFrame to *path* (any fsspec URL).

    The serialisation format is chosen by the file extension:
    - ``.parquet`` (default / recommended) — written via ``pyarrow``.
    - ``.csv``                             — written as UTF-8 CSV.

    Extra keyword arguments are forwarded to the underlying writer.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    fs, fpath = fsspec.url_to_fs(path, **(storage_options or {}))
    if path.endswith('.csv'):
        with fs.open(path, 'w', encoding='utf-8') as f:
            frame.to_csv(f, **kwargs)
    else:
        table = pa.Table.from_pandas(frame)
        with fs.open(path, 'wb') as f:
            pq.write_table(table, f, **kwargs)
    log.detailed(f"WROTE frame {frame.shape} to {path}")


def read_frame(path, *, storage_options=None, log=Logger(), **kwargs) -> pd.DataFrame:
    """Read a pandas DataFrame from *path* (any fsspec URL).

    Format is inferred from the file extension (see :func:`write_frame`).
    Extra keyword arguments are forwarded to the underlying reader.
    """
    import pyarrow.parquet as pq

    fs, fpath = fsspec.url_to_fs(path, **(storage_options or {}))
    if path.endswith('.csv'):
        with fs.open(path, 'r', encoding='utf-8') as f:
            frame = pd.read_csv(f, **kwargs)
    else:
        with fs.open(path, 'rb') as f:
            frame = pq.read_table(f, **kwargs).to_pandas()
    log.detailed(f"READ frame {frame.shape} from {path}")
    return frame




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
        if hasattr(self, '_n_workers'):
            desc = f"{desc} [nw={self._n_workers}]"
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
    """Execute callables concurrently using threads.

    Parameters
    ----------
    n_workers : int
        Number of threads.
    batch_size : int, optional
        Accumulate this many results before IPC transfer.
    tag : str
        Label for the progress bar.
    """

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
    """Execute callables concurrently using separate processes.

    Uses ``torch.multiprocessing`` for CUDA-safe forking.  Main-process
    references are released after workers start to allow memory reclamation.

    Parameters
    ----------
    n_workers : int
        Number of worker processes.
    batch_size : int, optional
        Accumulate this many results before IPC transfer.
    tag : str
        Label for the progress bar.
    """

    def __init__(self, *, n_workers: int, batch_size: int = None, tag: str = "",
                 start_method: str = 'spawn', log: Logger = Logger()):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.tag = tag
        self.log = log
        # Use 'spawn' by default to avoid fork-safety issues with HTTP clients
        # (e.g. Azure SDK, fsspec AzureBlobFileSystem) and CUDA contexts.
        # Override with 'fork' for faster startup when all objects are fork-safe.
        if _TORCH_AVAILABLE:
            import torch.multiprocessing as _mp
        else:
            import multiprocessing as _mp
        self._mp = _mp.get_context(start_method)

    @property
    def _n_workers(self) -> int:
        return self.n_workers

    def _make_queue(self):
        return self._mp.Queue()

    def _make_event(self):
        return self._mp.Event()

    def _make_worker(self, target, args):
        # Pass target and args explicitly so the module-level _mp_worker_fn
        # can be pickled by name (required for spawn/forkserver start methods).
        return self._mp.Process(target=_mp_worker_fn, args=(target, args))

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
    """Execute callables on pre-created Ray actor workers.

    Requires the ``ray`` optional dependency (``pip install datablocks[ray]``).

    Parameters
    ----------
    n_workers : int
        Number of Ray actors (ignored when *workers* is provided).
    workers : list, optional
        Pre-created Ray actor handles.
    worker_factory : callable, optional
        Factory returning a new Ray actor.  Called *n_workers* times.
    batch_size : int, optional
        When set, workers process items in batches of this size.
    tag : str
        Label for the progress bar.
    """

    def __init__(self, *, n_workers: int = 1, workers=None, worker_factory=None,
                 batch_size: int = None, tag: str = "", log: Logger = Logger()):
        if workers is not None:
            self.workers = workers
            self.n_workers = len(workers)
        elif worker_factory is not None:
            self.workers = [worker_factory() for _ in range(n_workers)]
            self.n_workers = n_workers
        else:
            raise ValueError(
                "RayCallableExecutor requires either 'workers' (pre-created) "
                "or 'worker_factory' (callable returning a worker)"
            )
        self.batch_size = batch_size
        self.tag = tag
        self.log = log

    def execute(self, callables: Sequence[Callable], *ctx_args, **ctx_kwargs):
        """Execute all callables; streams chunked results if batch_size is set."""
        if self.batch_size is not None:
            return self.exec_callables_streaming(callables, *ctx_args, **ctx_kwargs)
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
            label = f"{label} [nw={self.n_workers}]"
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
            label = f"{label} [nw={self.n_workers}]"
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
    """Execute callables sequentially in the local process.

    Useful as a no-parallelism baseline and for debugging.
    """
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



class _TorchCallableExecutorMixin_:
    """Mixin that adds device-management to a ``_CallableExecutorBase_`` subclass.

    Subclasses of this mixin combine it with ``MultithreadingCallableExecutor``
    or ``MultiprocessingCallableExecutor`` and override ``_run_items`` so that
    each callable is:

    1. validated (must have a ``.to()`` method),
    2. moved to the worker's device via ``callable.to(device)``,
    3. executed,
    4. moved back via ``callable.to('cpu')``.

    Context args / kwargs that have a ``.to()`` method are also moved to the
    worker's device.
    """

    def __init__(self, *, devices: list[str] = 'cuda', batch_size: int = None,
                 tag: str = "", log: Logger = Logger()):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for TorchMulti*CallableExecutor. "
                "Install it with: pip install datablocks[torch]"
            )
        if isinstance(devices, str):
            devices = [devices]
        # Initialise the concrete executor base (Thread or Process variant).
        super().__init__(n_workers=len(devices), batch_size=batch_size, tag=tag, log=log)
        self.devices = devices

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_callable(item):
        """Raise ``TypeError`` if *item* lacks a ``.to()`` method; return *item*.

        Designed to be chained::

            self._validate_callable(c).to(device)(...)
        """
        if not hasattr(item, 'to') or not callable(getattr(item, 'to')):
            raise TypeError(
                f"{type(item).__name__} does not implement .to(device). "
                f"Callables used with TorchMultithreadingCallableExecutor / "
                f"TorchMultiprocessingCallableExecutor must define a .to() method."
            )
        return item

    @staticmethod
    def _args_kwargs_to_device(args, kwargs, device):
        device_args = [a.to(device) if hasattr(a, 'to') else a for a in args]
        device_kwargs = {k: v.to(device) if hasattr(v, 'to') else v
                         for k, v in kwargs.items()}
        return device_args, device_kwargs

    # ------------------------------------------------------------------
    # Override worker loop
    # ------------------------------------------------------------------
    def _run_items(self, items, ctx_args, ctx_kwargs, offset, worker_idx,
                   result_queue, done_queue, abort_event):
        device = self.devices[worker_idx]
        worker_label = self._worker_label(worker_idx)
        self.log.debug(f"Executing {len(items)} callables on {worker_label} (device={device})")

        device_ctx_args, device_ctx_kwargs = self._args_kwargs_to_device(
            ctx_args, ctx_kwargs, device,
        )

        batch_size = self.batch_size if (self.batch_size is not None and self.batch_size > 1) else 1
        exception = None
        batch = []
        for i, item in enumerate(items):
            exception = None
            try:
                if abort_event.is_set():
                    break
                payload = self._validate_callable(item).to(device)(*device_ctx_args, **device_ctx_kwargs)
                item.to('cpu')
                batch.append((offset + i, payload))
            except Exception as e:
                exception = e
                self.log.info(f"ERROR executing callable {offset+i} on {worker_label} (device={device})")
            finally:
                self._after_item(item)
            if exception is not None:
                if batch:
                    result_queue.put((True, worker_idx, batch))
                    batch = []
                tbstr = '\n'.join(tb.format_tb(exception.__traceback__))
                result_queue.put((False, worker_idx, offset + i, (exception, tbstr)))
                break
            if len(batch) >= batch_size:
                result_queue.put((True, worker_idx, batch))
                batch = []
        if batch and exception is None:
            result_queue.put((True, worker_idx, batch))

        del device_ctx_args, device_ctx_kwargs
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


class TorchMultithreadingCallableExecutor(_TorchCallableExecutorMixin_,
                                          MultithreadingCallableExecutor):
    """Multithreading callable executor with per-worker device management.

    Each worker is assigned a device from *devices*.  Before execution,
    each callable is validated for ``.to()``, moved to the device,
    executed, then moved back to cpu.
    """
    pass


class TorchMultiprocessingCallableExecutor(_TorchCallableExecutorMixin_,
                                           MultiprocessingCallableExecutor):
    """Multiprocessing callable executor with per-worker device management."""
    pass


_CALLABLE_EXECUTORS = {
    "inline":                InlineCallableExecutor,
    "multithreading":        MultithreadingCallableExecutor,
    "multiprocessing":       MultiprocessingCallableExecutor,
    "ray":                   RayCallableExecutor,
    "torch_multithreading":  TorchMultithreadingCallableExecutor,
    "torch_multiprocessing": TorchMultiprocessingCallableExecutor,
}


def select_executor(parallelization: str | None = None):
    """Return the callable-executor **class** for the given parallelization strategy.

    Parameters
    ----------
    parallelization : str or None
        One of ``'inline'`` (default), ``'multithreading'``,
        ``'multiprocessing'``, ``'torch_multithreading'``,
        ``'torch_multiprocessing'``.  Case-insensitive.
        ``None`` maps to ``'inline'``.

    Returns
    -------
    type
        The executor class (not an instance).
    """
    key = (parallelization or "inline").lower()
    cls = _CALLABLE_EXECUTORS.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown parallelization {parallelization!r}. "
            f"Choose from {list(_CALLABLE_EXECUTORS)}"
        )
    return cls


def callable_executor(parallelization: str = None, **kwargs):
    """Create a callable-executor instance for the given parallelization strategy.

    Parameters
    ----------
    parallelization : str or None
        One of ``'inline'`` (default), ``'multithreading'``,
        ``'multiprocessing'``, ``'torch_multithreading'``,
        ``'torch_multiprocessing'``.  ``None`` maps to ``'inline'``.
    **kwargs
        Forwarded to the executor constructor (e.g. ``n_workers``, ``tag``).

    Returns
    -------
    A callable-executor instance.
    """
    return select_executor(parallelization)(**kwargs)


# ---------------------------------------------------------------------------
# @tagged pipeline decorator
# ---------------------------------------------------------------------------

_TAGGED_SKIP_DEFAULTS = frozenset({'tag', 'url'})


def _make_tag(func: callable, sig: inspect.Signature, arguments: dict,
              skip: frozenset) -> str:
    """Build a human-readable call string from bound + defaulted arguments.

    Only non-default values (and required positional args) are included, so
    the result reads like the minimal call a user would type.

    Example::

        "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('CPTAC_206020', single=1)"
    """
    pos_args, kw_args = [], []
    for name, param in sig.parameters.items():
        if name in skip:
            continue
        value = arguments[name]
        is_required = param.default is inspect.Parameter.empty
        is_positional = param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        if is_required:
            if is_positional:
                pos_args.append(repr(value))
            else:
                kw_args.append(f"{name}={repr(value)}")
        elif value != param.default:
            kw_args.append(f"{name}={repr(value)}")

    qualname = f"{func.__module__}.{func.__qualname__}"
    return f"{qualname}({', '.join(pos_args + kw_args)})"


def tagged(func=None, *, skip: frozenset = _TAGGED_SKIP_DEFAULTS):
    """Decorator for pipeline functions that auto-computes a call-string tag.

    When the decorated function is called with ``tag=None`` (or tag is
    omitted), the decorator synthesises a tag of the form::

        "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('CPTAC_206020', single=1)"

    showing only the arguments that differ from their defaults.  If ``tag`` is
    supplied explicitly (including by an upstream pipeline that already computed
    its own tag), it is passed through unchanged.

    The decorated function receives ``tag`` as a normal keyword argument and
    need not know whether it was supplied by the caller or synthesised here.

    Usage::

        @tagged
        def my_pipeline(name, *, tag=None, url=None, n_workers=1):
            clip = MyClip(...)
            clip.tag = tag   # propagate down to the datablock
            return clip

    Parameters
    ----------
    skip : frozenset
        Parameter names to exclude from the generated tag string.  Defaults to
        ``{'tag', 'url'}`` — operational overrides that are not part of a
        pipeline's logical identity.
    """
    if func is None:
        return functools.partial(tagged, skip=skip)

    sig = inspect.signature(func)
    if 'tag' not in sig.parameters:
        raise TypeError(
            f"@tagged: {func.__qualname__} must have a 'tag' parameter "
            f"(e.g. tag: str | None = None)"
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get('tag') is None:
            bound.arguments['tag'] = _make_tag(func, sig, bound.arguments, skip)
        return func(*bound.args, **bound.kwargs)

    return wrapper
