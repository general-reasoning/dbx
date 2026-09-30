"""
Output capture is a COMMAND's: ``dbx.exec(..., capture_output=True)``, or
``--capture-output`` on the command line.

The command's process tees its stdout/stderr to a master `OutputCapture`,
opened when the command starts and closed when it is over; every worker
PROCESS a dbx executor sends its work to runs each callable inside a capture
of its own, whose path comes home with the result. The exec journal records
them all as ``output_captures``, the master at index 0, and
``ExecjournalEntry.output()`` prints one. A block captures nothing: its
journal entry's ``log`` is the capture open around its build.
"""
import functools
import os
import subprocess
import sys
import textwrap
import warnings
from dataclasses import dataclass

import pytest

import dbx
from dbx.datablocks import Datablock
from dbx.journals import OutputCapture


@pytest.fixture(autouse=True)
def lake(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.setenv('DBX_ROOT', str(tmp_path))


class Loud(Datablock):
    """Prints while it builds -- through Python, and straight to fd 1."""
    TOPICS = {'result': 'result.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        print(f"PY x={self.var.x} pid={os.getpid()}", flush=True)
        os.write(1, f"FD x={self.var.x}\n".encode())
        with open(self.path('result', ensure_dirpath=True), 'w') as f:
            f.write("built")


class Failing(Datablock):
    TOPICS = {'result': 'result.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        pass

    def __build__(self):
        print("about to fail", flush=True)
        raise RuntimeError("intentional build failure")


def build_loud(x, root):
    """Module-level, so that a spawned worker can import it."""
    Loud(datalake=root, spec={'x': x}).build()
    return x


def fail_loud(root):
    print(f"WORKER FAILING pid={os.getpid()}", flush=True)
    Failing(datalake=root).build()


def _exec(s, **kwargs):
    """`dbx.exec`, printing to fd 1 and 2 as it does outside pytest.

    pytest swaps sys.stdout for a buffer of its own -- again at the start of
    every test, so a fixture cannot undo it -- and a print then never reaches
    fd 1, where a capture, like a terminal, reads.
    """
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        return dbx.exec(s, **kwargs)
    finally:
        sys.stdout, sys.stderr = saved


def _row(tmp_path, iloc=0):
    return dbx.execjournal(datalake=str(tmp_path), iloc=iloc)


def _captures(row):
    return [str(p) for p in row['output_captures']]


def _read(path):
    with open(path) as f:
        return f.read()


def _logs(row):
    return [str(v) for v in row.datajournal(index=None)['log']]


class TestTheMasterCapture:

    def test_a_command_that_captures_has_one_master_capture(self, tmp_path):
        _exec("print('PY'); os.write(1, b'FD' + bytes([10])); 42", capture_output=True, os=os)
        captures = _captures(_row(tmp_path))
        assert len(captures) == 1
        assert os.path.basename(captures[0]).startswith('master-')
        text = _read(captures[0])
        assert 'PY' in text and 'FD' in text

    def test_written_under_the_commands_session(self, tmp_path):
        _exec("1", capture_output=True)
        row = _row(tmp_path)
        assert os.path.dirname(_captures(row)[0]) == OutputCapture.dirpath(str(tmp_path), row['session'])

    def test_a_command_that_does_not_capture_records_none(self, tmp_path):
        _exec("1")
        assert _captures(_row(tmp_path)) == []

    def test_closed_when_the_command_is_over(self, tmp_path):
        before = (os.fstat(1).st_ino, os.fstat(2).st_ino)
        _exec("1", capture_output=True)
        assert OutputCapture.current() is None
        assert (os.fstat(1).st_ino, os.fstat(2).st_ino) == before

    def test_a_failing_command_is_captured_too(self, tmp_path):
        with pytest.raises(ZeroDivisionError):
            _exec("print('before the fall'); 1/0", capture_output=True)
        assert OutputCapture.current() is None
        assert 'before the fall' in _read(_captures(_row(tmp_path))[0])

    def test_capture_output_is_not_a_name_for_the_statements(self, tmp_path):
        assert _exec("'capture_output' in dir()", capture_output=True) is False

    def test_a_nested_exec_joins_the_capture(self, tmp_path):
        _exec("dbx.exec('print(1)', capture_output=True); dbx.exec('print(2)')",
                 capture_output=True, dbx=dbx)
        rows = dbx.execjournal(datalake=str(tmp_path), index=None)
        assert len(rows) == 3
        masters = {_captures(rows.get(i))[0] for i in range(3)}
        assert len(masters) == 1


class TestABlockRecordsTheCapture:

    def test_its_log_is_the_capture_open_around_its_build(self, tmp_path):
        _exec("Loud(datalake=root, spec={'x': 1}).build()", capture_output=True,
                 Loud=Loud, root=str(tmp_path))
        row = _row(tmp_path)
        assert _logs(row) == _captures(row)[:1]
        assert 'PY x=1' in row.datajournal(iloc=0).read('log')

    def test_no_capture_no_log(self, tmp_path):
        _exec("Loud(datalake=root, spec={'x': 1}).build()", Loud=Loud, root=str(tmp_path))
        assert _logs(_row(tmp_path)) == ['None']

    def test_capture_output_is_accepted_ignored_and_not_recorded(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            blk = Loud(datalake=str(tmp_path), capture_output=True)
        assert 'capture_output' not in blk.dfn
        assert not hasattr(blk, 'capture_output')

    def test_a_dfn_recorded_with_it_reconstructs_the_same_block(self, tmp_path):
        blk = Loud(datalake=str(tmp_path), spec={'x': 3})
        again = Loud(**{**blk.dfn, 'capture_output': False})
        assert again.dfn == blk.dfn and again.hash == blk.hash


class TestWorkerCaptures:

    def test_a_worker_process_captures_each_callable(self, tmp_path):
        _exec("ex.exec_callables([functools.partial(build, x, root) for x in range(3)])",
                 capture_output=True, ex=dbx.MultiprocessingCallableExecutor(n_workers=2),
                 functools=functools, build=build_loud, root=str(tmp_path))
        row = _row(tmp_path)
        master, *workers = _captures(row)
        assert os.path.basename(master).startswith('master-')
        assert len(workers) == 3
        assert all(os.path.basename(w).startswith('worker-') for w in workers)
        texts = [_read(w) for w in workers]
        assert sorted(x for x in range(3) for t in texts if f'PY x={x}' in t and f'FD x={x}' in t) == [0, 1, 2]
        # Each block records the capture of the process that built it.
        assert sorted(_logs(row)) == sorted(workers)

    def test_a_worker_thread_writes_to_the_master(self, tmp_path):
        _exec("ex.exec_callables([functools.partial(build, x, root) for x in range(3)])",
                 capture_output=True, ex=dbx.MultithreadingCallableExecutor(n_workers=2),
                 functools=functools, build=build_loud, root=str(tmp_path))
        row = _row(tmp_path)
        captures = _captures(row)
        assert len(captures) == 1
        assert set(_logs(row)) == set(captures)
        text = _read(captures[0])
        assert all(f'PY x={x}' in text for x in range(3))

    def test_a_failed_workers_capture_comes_home(self, tmp_path):
        with pytest.raises(Exception):
            _exec("ex.exec_callables([functools.partial(fail, root)])",
                     capture_output=True, ex=dbx.MultiprocessingCallableExecutor(n_workers=1),
                     functools=functools, fail=fail_loud, root=str(tmp_path))
        master, *workers = _captures(_row(tmp_path))
        assert len(workers) == 1
        assert 'WORKER FAILING' in _read(workers[0])

    def test_no_capture_nothing_carried(self, tmp_path):
        _exec("ex.exec_callables([functools.partial(build, 1, root)])",
                 ex=dbx.MultiprocessingCallableExecutor(n_workers=1),
                 functools=functools, build=build_loud, root=str(tmp_path))
        row = _row(tmp_path)
        assert _captures(row) == []
        assert not os.path.exists(OutputCapture.dirpath(str(tmp_path), row['session']))


class TestOutput:

    @pytest.fixture
    def row(self, tmp_path):
        _exec("print('MASTER TEXT'); ex.exec_callables([functools.partial(build, x, root) for x in range(2)])",
                 capture_output=True, ex=dbx.MultiprocessingCallableExecutor(n_workers=1),
                 functools=functools, build=build_loud, root=str(tmp_path))
        return _row(tmp_path)

    def test_the_master_by_default(self, row, capfd):
        capfd.readouterr()
        row.output()
        assert 'MASTER TEXT' in capfd.readouterr().out

    def test_by_index(self, row, capfd):
        capfd.readouterr()
        row.output(1)
        assert capfd.readouterr().out == _read(_captures(row)[1])

    def test_by_basename_prefix(self, row, capfd):
        capfd.readouterr()
        row.output(basename='master')
        assert 'MASTER TEXT' in capfd.readouterr().out

    def test_by_basename_regex(self, row, capfd):
        name = os.path.basename(_captures(row)[2])
        capfd.readouterr()
        row.output(basename=r'worker-.*' + name[-15:].replace('.', r'\.'))
        assert capfd.readouterr().out == _read(_captures(row)[2])

    def test_a_basename_must_match_exactly_one(self, row):
        with pytest.raises(LookupError, match='matches 2'):
            row.output(basename='worker-')
        with pytest.raises(LookupError, match='matches 0'):
            row.output(basename='nothing-like-it')

    def test_an_index_out_of_range(self, row):
        with pytest.raises(IndexError, match='has 3'):
            row.output(3)

    def test_not_both(self, row):
        with pytest.raises(ValueError):
            row.output(0, basename='master')

    def test_a_basename_finds_a_capture_the_row_does_not_list(self, row, capfd):
        """A worker that was killed never sent its path home; its capture is still in the session."""
        stray = os.path.join(os.path.dirname(_captures(row)[0]), 'worker-elsewhere-1-x.log')
        with open(stray, 'w') as f:
            f.write('STRAY')
        capfd.readouterr()
        row.output(basename='worker-elsewhere')
        assert capfd.readouterr().out == 'STRAY'

    def test_a_command_that_captured_nothing(self, tmp_path):
        _exec("1")
        with pytest.raises(LookupError, match='captured no output'):
            _row(tmp_path).output()


def _run_cli(tmp_path, *argv, entry='pprint'):
    env = dict(os.environ, DBX_ROOT=str(tmp_path), DBX_DIRTY_REPO_OK='1', DBX_USE_WORK_REPO='False')
    code = f"import sys, dbx; sys.argv = {['dbx.' + entry, *argv]!r}; sys.exit(dbx.{entry}())"
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                          cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), timeout=120)


class TestTheCommandLine:

    def test_pprint_captures_what_it_prints_of_the_result(self, tmp_path):
        proc = _run_cli(tmp_path, '--capture-output', "print('ON THE WAY'); 'THE RESULT'")
        assert proc.returncode == 0, proc.stderr
        assert 'THE RESULT' in proc.stdout
        row = _row(tmp_path)
        assert row['exec'] == "print('ON THE WAY'); 'THE RESULT'"
        text = _read(_captures(row)[0])
        assert 'ON THE WAY' in text and 'THE RESULT' in text

    def test_the_flag_may_come_after_the_expression(self, tmp_path):
        proc = _run_cli(tmp_path, "'X'", '--capture-output')
        assert proc.returncode == 0, proc.stderr
        assert len(_captures(_row(tmp_path))) == 1

    def test_exec_takes_it_too(self, tmp_path):
        proc = _run_cli(tmp_path, '--capture-output', "print('EXEC OUT')", entry='exec')
        assert proc.returncode == 0, proc.stderr
        assert 'EXEC OUT' in _read(_captures(_row(tmp_path))[0])

    def test_without_it_nothing_is_captured(self, tmp_path):
        proc = _run_cli(tmp_path, "'X'")
        assert proc.returncode == 0, proc.stderr
        assert _captures(_row(tmp_path)) == []

    def test_the_pinned_runner_captures_its_rendering(self, tmp_path):
        """Phase 2 of a pinned run renders outside exec; the runner opens the capture around both."""
        from dbx import dataparts
        runner = tmp_path / 'runner.py'
        runner.write_text(dataparts._PINNED_RUNNER.format(
            render=textwrap.indent(dataparts._PINNED_RENDER['pprint'], '    '),
            capture_flag=dataparts.CAPTURE_OUTPUT_FLAG))
        env = dict(os.environ, DBX_ROOT=str(tmp_path), DBX_DIRTY_REPO_OK='1', DBX_USE_WORK_REPO='False')
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env['PYTHONPATH'] = os.pathsep.join([repo] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        proc = subprocess.run([sys.executable, str(runner), '--capture-output', "'RENDERED'"],
                              capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert 'RENDERED' in _read(_captures(_row(tmp_path))[0])
