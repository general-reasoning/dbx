"""``reuse_workers=True``: an executor's workers, started by its first call, run every later call's callables too.

Nothing else about ``exec_callables`` changes -- results in input order, the
progress bar, the journal carried, a failure raised -- and without it nothing
changes at all: each call starts and stops workers of its own, as it always has.
"""
import functools
import os
import threading

import pytest

from dbx.databits import callable_executor
from dbx.datablocks import DataJournal


def _whoami(i, scale=1):
    """Top level, so a spawned worker unpickles it by name: the item, scaled, and who ran it."""
    return i * scale, os.getpid(), threading.get_ident()


def _fails(i):
    raise RuntimeError(f"boom {i}")


@pytest.mark.parametrize("parallelization, who", [('multiprocessing', 1), ('multithreading', 2)])
def test_three_calls_run_on_the_workers_the_first_one_started(parallelization, who):
    ex = callable_executor(parallelization, n_workers=2, reuse_workers=True, tag='reused')
    try:
        runners = []
        for scale in (1, 10, 100):
            out = ex.exec_callables([functools.partial(_whoami, i, scale) for i in range(8)])
            assert [o[0] for o in out] == [i * scale for i in range(8)], "results in input order"
            runners.append({o[who] for o in out})
        assert len(set().union(*runners)) <= 2, "no worker started after the first call's"
        workers = ex._pool_['workers']
        assert all(w.is_alive() for w in workers)
    finally:
        ex.close()
    assert ex._pool_ is None and not any(w.is_alive() for w in workers)
    ex.close()      # again: harmless


def test_by_default_each_call_starts_workers_of_its_own():
    ex = callable_executor('multiprocessing', n_workers=2, tag='fresh')
    first = {pid for _, pid, _ in ex.exec_callables([functools.partial(_whoami, i) for i in range(4)])}
    second = {pid for _, pid, _ in ex.exec_callables([functools.partial(_whoami, i) for i in range(4)])}
    assert not first & second
    assert getattr(ex, '_pool_', None) is None


def test_a_failure_raises_stops_the_workers_and_the_next_call_starts_anew():
    ex = callable_executor('multiprocessing', n_workers=2, reuse_workers=True, tag='failing')
    try:
        ex.exec_callables([functools.partial(_whoami, i) for i in range(2)])
        workers = ex._pool_['workers']
        with pytest.raises(RuntimeError, match="boom"):
            ex.exec_callables([functools.partial(_fails, i) for i in range(2)])
        assert ex._pool_ is None and not any(w.is_alive() for w in workers)
        assert [o[0] for o in ex.exec_callables([functools.partial(_whoami, i) for i in range(3)])] == [0, 1, 2]
    finally:
        ex.close()


def test_the_journal_is_carried_to_reused_workers(tmp_path, monkeypatch):
    from test_datajournal import _built_under_the_scope, block
    monkeypatch.setenv('DBX_ROOT', str(tmp_path))
    ex = callable_executor('multiprocessing', n_workers=2, reuse_workers=True, tag='journaled')
    try:
        with DataJournal() as dj:
            first = ex.exec_callables([functools.partial(_built_under_the_scope, i) for i in range(2)])
            second = ex.exec_callables([functools.partial(_built_under_the_scope, i) for i in range(2, 4)])
    finally:
        ex.close()
    assert first + second == [block(tmp_path, x=100 + i).hash for i in range(4)]
    assert len(dj.written_entries()) == 8, "each build's start and end"


@pytest.mark.parametrize("parallelization", ['inline', 'multithreading'])
def test_every_executor_takes_it_and_closes(parallelization):
    ex = callable_executor(parallelization, n_workers=2, reuse_workers=True)
    assert [o[0] for o in ex.exec_callables([functools.partial(_whoami, i) for i in range(3)])] == [0, 1, 2]
    ex.close()


def test_a_distributed_core_partition_starts_its_workers_once(tmp_path, monkeypatch):
    """The scan, every Lloyd pass, the assignment and the coverage, on one set of workers."""
    from dbx.databits import MultithreadingCallableExecutor
    from test_core_partition import core, points
    tbl = points(tmp_path)
    started = []
    original = MultithreadingCallableExecutor._make_worker_

    def counting(self, target, args, idx):
        started.append(idx)
        return original(self, target, args, idx)
    monkeypatch.setattr(MultithreadingCallableExecutor, '_make_worker_', counting)
    p = core(tmp_path, tbl, fractions=(0.5,), clustering='distributed',
             parallelization='multithreading', n_workers=2)
    assert p.valid() and len(started) == 2
