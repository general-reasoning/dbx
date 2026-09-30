"""
``OutputTee`` tees fd 1 and 2 to a log file — it does not swallow them.

Two properties matter, and they are the two a ``sys.stdout``-replacing tee
could not deliver together:

1. **Pass-through.** Everything still reaches the real terminal.  Turning on
   ``capture_output`` must not make a long build go silent.
2. **Completeness.** Because the capture is at the file-descriptor level,
   writes from C extensions and subprocesses land in the log too.  A
   Python-level tee sees only what goes through the Python stream object.

These run the probe in a subprocess: the assertions are about what the OS
hands to fd 1/2, which pytest's own capture would otherwise intercept.
"""
import os
import subprocess
import sys
import textwrap

import pytest


PROBE = textwrap.dedent("""
    import os, subprocess, sys
    from dbx.dataparts import OutputTee

    logname = sys.argv[1]
    with open(logname, 'w') as lf:
        cap = OutputTee(lf)
        print("PY_STDOUT")                       # via sys.stdout
        os.write(1, b"RAW_FD1\\n")                # as a C extension writes
        subprocess.run(["echo", "SUBPROC"], check=True)
        sys.stderr.write("PY_STDERR\\n")
        cap.close()
""")


@pytest.fixture
def run_probe(tmp_path):
    def _run():
        script = tmp_path / 'probe.py'
        script.write_text(PROBE)
        logname = tmp_path / 'captured.log'
        env = dict(os.environ, DBX_DIRTY_REPO_OK='1')
        env.pop('DBX_USE_WORK_REPO', None)
        proc = subprocess.run(
            [sys.executable, str(script), str(logname)],
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0, proc.stderr
        return proc, logname.read_text()
    return _run


class TestPassThrough:
    """capture_output must not mean 'go silent'."""

    def test_python_stdout_still_reaches_the_terminal(self, run_probe):
        proc, _ = run_probe()
        assert 'PY_STDOUT' in proc.stdout

    def test_raw_fd_writes_still_reach_the_terminal(self, run_probe):
        proc, _ = run_probe()
        assert 'RAW_FD1' in proc.stdout

    def test_subprocess_output_still_reaches_the_terminal(self, run_probe):
        proc, _ = run_probe()
        assert 'SUBPROC' in proc.stdout

    def test_stderr_still_reaches_the_terminal(self, run_probe):
        proc, _ = run_probe()
        assert 'PY_STDERR' in proc.stderr


class TestCompleteness:
    """The log gets everything, including what a Python-level tee would miss."""

    def test_python_stdout_is_logged(self, run_probe):
        _, log = run_probe()
        assert 'PY_STDOUT' in log

    def test_raw_fd_writes_are_logged(self, run_probe):
        """A sys.stdout tee never sees this — it bypasses the Python stream."""
        _, log = run_probe()
        assert 'RAW_FD1' in log

    def test_subprocess_output_is_logged(self, run_probe):
        """Subprocesses inherit the fds, not sys.stdout."""
        _, log = run_probe()
        assert 'SUBPROC' in log

    def test_stderr_is_logged(self, run_probe):
        _, log = run_probe()
        assert 'PY_STDERR' in log


class TestRestoration:

    def test_close_restores_the_descriptors(self, tmp_path):
        """Leaking the dup2 would redirect the rest of the process to a pipe."""
        from dbx.dataparts import OutputTee

        before = (os.fstat(1).st_ino, os.fstat(2).st_ino)
        with open(tmp_path / 'x.log', 'w') as lf:
            cap = OutputTee(lf)
            os.write(1, b'something\n')
            cap.close()
        after = (os.fstat(1).st_ino, os.fstat(2).st_ino)
        assert before == after


LINGERING = textwrap.dedent("""
    import subprocess, sys, time
    from dbx.dataparts import OutputTee

    logname = sys.argv[1]
    with open(logname, 'w') as lf:
        cap = OutputTee(lf)
        print("BEFORE", flush=True)
        # Inherits the tee's pipe as its stdout, and outlives the tee -- as
        # multiprocessing's resource tracker, or a Ray raylet, does.
        child = subprocess.Popen([sys.executable, "-c",
                                  "import time; time.sleep(1.5); print('AFTER', flush=True)"])
        start = time.monotonic()
        cap.close()
        print(f"CLOSED IN {time.monotonic() - start:.2f}", flush=True)
        child.wait()
""")


class TestAProcessThatOutlivesTheTee:
    """close() must not wait for the pipe to reach end-of-file: a descendant holding it would hang it."""

    @pytest.fixture
    def run_lingering(self, tmp_path):
        script = tmp_path / 'lingering.py'
        script.write_text(LINGERING)
        logname = tmp_path / 'captured.log'
        env = dict(os.environ, DBX_DIRTY_REPO_OK='1')
        env.pop('DBX_USE_WORK_REPO', None)
        proc = subprocess.run(
            [sys.executable, str(script), str(logname)],
            capture_output=True, text=True, env=env, timeout=60,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0, proc.stderr
        return proc, logname.read_text()

    def test_close_does_not_wait_for_it(self, run_lingering):
        proc, _ = run_lingering
        closed_in = float(proc.stdout.split('CLOSED IN ')[1].split()[0])
        assert closed_in < 1.0

    def test_what_came_before_is_logged(self, run_lingering):
        _, log = run_lingering
        assert 'BEFORE' in log

    def test_what_it_writes_after_still_reaches_the_terminal(self, run_lingering):
        proc, log = run_lingering
        assert 'AFTER' in proc.stdout
        assert 'AFTER' not in log
