"""Datalog: levels resolved from the scope, worker-led lines, and subtree banners that nest across workers."""
import functools
import re

import pytest

from dbx.datablocks import Datablock, DATAFILE
from dbx.dataparts import callable_executor
from dbx.journals import Datalog


LEAD = re.compile(r"^(?P<worker>\S+):\s+(?P<dt>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}): (?P<rest>.*)$")


def _lines(capsys):
    return [l for l in capsys.readouterr().out.splitlines() if LEAD.match(l)]


def _rest(line):
    return LEAD.match(line)['rest']


@pytest.fixture(autouse=True)
def quiet_env(monkeypatch):
    for level in Datalog.LEVELS:
        monkeypatch.delenv(f"DBX_LOG_{level.upper()}", raising=False)


class TestLevels:

    def test_scope_turns_a_level_on_for_views_made_before_it(self, capsys):
        log = Datalog("x")
        log.verbose("off")
        with Datalog(verbose=True):
            log.verbose("on")
        log.verbose("off again")
        assert [_rest(l) for l in _lines(capsys)] == ["VERBOSE: x: on"]

    def test_view_level_wins_over_scope(self):
        with Datalog(verbose=True):
            assert Datalog(verbose=False).ist('verbose') is False
            assert Datalog().ist('verbose') is True

    def test_inner_scope_wins(self):
        with Datalog(verbose=True):
            with Datalog(verbose=False):
                assert not Datalog().ist('verbose')
            assert Datalog().ist('verbose')

    def test_env_under_scope(self, monkeypatch):
        monkeypatch.setenv('DBX_LOG_VERBOSE', 'True')
        assert Datalog().ist('verbose')
        with Datalog(verbose=False):
            assert not Datalog().ist('verbose')

    def test_line_leads_with_worker_then_time(self, capsys):
        Datalog("x").info("hello")
        (line,) = _lines(capsys)
        m = LEAD.match(line)
        assert m['worker'] == 'main'
        assert m['rest'] == "INFO: x: hello"


class TestSubtree:

    def test_banners_nest_and_line_up(self, capsys):
        log = Datalog()
        with Datalog(verbose=True):
            with log.subtree('a'):
                with log.subtree('b'):
                    log.skip_subtree('c', 'already valid')
        lines = _lines(capsys)
        assert [_rest(l) for l in lines] == [
            "------------>>>--- BUILDING SUBTREE at a",
            "------------------>>>--- BUILDING SUBTREE at b",
            "------------------------xxx--- SKIPPING SUBTREE at c: already valid",
            "------------------<<<--- BUILDING SUBTREE at b",
            "------------<<<--- BUILDING SUBTREE at a",
        ]
        # The banners start at the same column: worker and time are fixed-width.
        assert len({l.index('---') for l in lines}) == 1

    def test_failed_subtree_closes_and_restores_depth(self, capsys):
        log = Datalog()
        with Datalog(verbose=True):
            with pytest.raises(RuntimeError):
                with log.subtree('a'):
                    assert Datalog.depth() == 1
                    raise RuntimeError
        assert Datalog.depth() == 0
        assert _rest(_lines(capsys)[-1]) == "------------<<<--- BUILDING SUBTREE at a"

    def test_quiet_without_verbose(self, capsys):
        with Datalog().subtree('a'):
            pass
        assert _lines(capsys) == []


def _work(i):
    Datalog("w").verbose(f"{i} at {Datalog.depth()}")
    return Datalog.worker(), Datalog.depth()


class TestCarried:

    @pytest.mark.parametrize('parallelization, kind', [('multithreading', 't'), ('multiprocessing', 'p')])
    def test_scope_depth_and_worker_reach_workers(self, parallelization, kind, capfd):
        ex = callable_executor(parallelization, n_workers=2)
        with Datalog(verbose=True):
            with Datalog().subtree('a'):
                results = ex.exec_callables([functools.partial(_work, i) for i in range(2)])
        assert sorted(results) == [(f"main/{kind}0", 1), (f"main/{kind}1", 1)]
        out = capfd.readouterr().out
        assert re.search(rf"^main/{kind}\d:\s+\S+: VERBOSE: w: 0 at 1$", out, re.M)
        assert Datalog.worker() == 'main' and Datalog.depth() == 0


class TestDatablockLog:

    class Leaf(Datablock):
        TOPICS = {'out': DATAFILE('out.txt')}

        def __build__(self):
            with open(self.path('out'), 'w') as f:
                f.write('x')

    def test_block_level_over_scope(self, tmp_path):
        b = self.Leaf(datalake=str(tmp_path), verbose=False)
        with Datalog(verbose=True):
            assert not b.log.ist('verbose')
            assert self.Leaf(datalake=str(tmp_path)).log.ist('verbose')

    def test_block_log_named_for_block(self, tmp_path):
        b = self.Leaf(datalake=str(tmp_path))
        assert b.log.name == b._log_name_()
        assert 'log' not in b.__dict__
