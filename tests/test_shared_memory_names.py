"""MosaicML streaming's shared memory is named per process, in EVERY module that names it.

Each of `prefix`, `util`, `shared` and `dataset` imported `_get_path` by name; patched in
two of them, a StreamingDataset in every process named its arrays ``000000_*`` -- one
segment shared by all, mapped at the wrong size by all but its creator.
"""
import os

import pytest

pytest.importorskip("streaming")

from dbx.datastreams import SharedMemoryManager


def test_every_module_names_shared_memory_by_process():
    import streaming.base.dataset as dataset
    import streaming.base.shared as shared
    import streaming.base.shared.prefix as prefix
    import streaming.base.util as util
    SharedMemoryManager.enable_pid_prefixes()
    for module in (prefix, util, shared, dataset):
        assert module._get_path(0, 'shard_states') == f"p{os.getpid()}_0000_shard_states", module.__name__


def _hold_open(path, ready, go, errors):
    from dbx.datastreams import open_datastream
    try:
        ds = open_datastream(path)
        ready.wait()                  # every process has its dataset open at once
        assert len(ds) > 0
        go.wait()
    except Exception as e:            # noqa: BLE001 -- reported to the parent
        errors.put(f"{type(e).__name__}: {e}")
        ready.abort()


def _write_mds(path, n_rows, rows_per_shard):
    from streaming import MDSWriter
    with MDSWriter(out=str(path), columns={'x': 'int'}, size_limit=rows_per_shard * 64) as w:
        for i in range(n_rows):
            w.write({'x': i})


def test_processes_with_streams_of_different_shard_counts_open_at_once(tmp_path):
    """The failure itself: concurrent datasets whose shard_states differ in size, one process each."""
    import multiprocessing as mp
    paths = []
    for k, n in enumerate((5, 400, 1200)):
        p = tmp_path / f"mds{k}"
        _write_mds(p, n, rows_per_shard=8)
        paths.append(str(p))
    ctx = mp.get_context('spawn')
    ready, go, errors = ctx.Barrier(len(paths)), ctx.Barrier(len(paths)), ctx.Queue()
    procs = [ctx.Process(target=_hold_open, args=(p, ready, go, errors)) for p in paths]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    found = [errors.get() for _ in range(errors.qsize())] if not errors.empty() else []
    assert not found, found
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
