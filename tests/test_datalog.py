"""Datalog: levels resolved from the scope, worker-led lines, and subtree banners that nest across workers."""
import functools
import re

import pytest

from dbx.datablocks import Datablock, DATAFILE
from dbx.databits import callable_executor
from dbx.journals import Datalog, log_section


LEAD = re.compile(r"^(?P<worker>\S+?):-*\s+(?P<dt>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}): (?P<rest>.*)$")


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
        assert line.startswith('main:' + '-' * (Datalog.WORKER_WIDTH - len('main')) + ' ')


class TestSubtree:

    def test_tree_is_a_level_of_its_own_and_follows_verbose_where_unset(self, capsys, monkeypatch):
        """DBX_LOG_TREE shows the banners alone; set nowhere, they come with VERBOSE, as they always did."""
        for var in ('DBX_LOG_TREE', 'DBX_LOG_VERBOSE'):
            monkeypatch.delenv(var, raising=False)
        log = Datalog()

        def shown():
            with log.subtree('a'):
                log.verbose('inside')
            return [_rest(l) for l in _lines(capsys)]

        assert shown() == []
        with Datalog(verbose=True):
            assert shown() == ["------------>>>--- BUILDING SUBTREE at a", "VERBOSE: inside",
                               "------------<<<--- BUILDING SUBTREE at a"]
        with Datalog(verbose=True, tree=False):
            assert shown() == ["VERBOSE: inside"]
        monkeypatch.setenv('DBX_LOG_TREE', '1')
        assert shown() == ["------------>>>--- BUILDING SUBTREE at a", "------------<<<--- BUILDING SUBTREE at a"]

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


class TestSection:
    def test_brackets_its_body_at_verbose(self, capsys):
        log = Datalog(verbose=True)
        with log.section("copying"):
            log.verbose("inside")
        assert [_rest(l) for l in _lines(capsys)] == [
            "VERBOSE: >>>--- copying", "VERBOSE: inside", "VERBOSE: <<<--- copying"]

    def test_closes_when_its_body_raises(self, capsys):
        with pytest.raises(KeyError):
            with Datalog(verbose=True).section("copying"):
                raise KeyError("x")
        assert _rest(_lines(capsys)[-1]) == "VERBOSE: <<<--- copying: raised KeyError"

    def test_at_the_level_given(self, capsys):
        with Datalog(verbose=True, detailed=False).section("var", 'detailed'):
            pass
        assert _lines(capsys) == []
        with Datalog(detailed=True).section("var", 'detailed'):
            pass
        assert [_rest(l) for l in _lines(capsys)] == ["DETAILED: >>>--- var", "DETAILED: <<<--- var"]

    def test_any_log_with_the_level_will_do(self):
        class Log:
            def __init__(self):
                self.lines = []
            def verbose(self, msg):
                self.lines.append(msg)
        log = Log()
        with log_section(log, "reading"):
            log.verbose("inside")
        assert log.lines == [">>>--- reading", "inside", "<<<--- reading"]


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
        label = 't' if kind == 't' else ''      # a process is its bare number, a thread t<i>
        assert sorted(results) == [(f"main:{label}0", 1), (f"main:{label}1", 1)]
        out = capfd.readouterr().out
        assert re.search(rf"^main:{label}\d:-*\s+\S+: VERBOSE: w: 0 at 1$", out, re.M)
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
