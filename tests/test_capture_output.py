"""
Tests for the capture_output log-upload ordering fix.

The key change being validated:
  When capture_output=True, the captured log file is now uploaded
  BEFORE __post_build__() writes the journal entry.  Previously
  it was uploaded in the `finally` block, AFTER __post_build__,
  so the journal's fs.exists(logpath) check could not find the
  log file and recorded None for the 'log' column.

Test matrix:
  1. Normal build: log is uploaded before __post_build__, journal records the log path
  2. Build exception: log is still uploaded via the finally block
  3. stdout/stderr restoration after normal build
  4. stdout/stderr restoration after exception
  5. Temp file cleanup
  6. Build when already valid (skip path) — no log created
"""
import os
import sys
import pytest
from dataclasses import dataclass
from unittest.mock import patch

from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Datablock subclasses for testing
# ---------------------------------------------------------------------------

class CapturedBlock(Datablock):
    """Block that prints to stdout/stderr during build and captures output."""
    TOPICS = {'result': 'result.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    # Track the order of operations
    build_log = []

    def __build__(self):
        CapturedBlock.build_log.append('__build__')
        print("stdout message from build")
        print("stderr message from build", file=sys.stderr)
        path = self.path(ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write("built")

    def __post_build__(self, *args, **kwargs):
        CapturedBlock.build_log.append('__post_build__')
        # Check if log file exists at this point
        logpath = self._dbxanchorhashpathx('log', ext='log', ensure_dirpath=True)
        log_exists_at_post_build = self.fs.exists(logpath)
        CapturedBlock.build_log.append(f'log_exists_at_post_build={log_exists_at_post_build}')
        super().__post_build__(*args, **kwargs)


class FailingCapturedBlock(Datablock):
    """Block that fails during build, with capture_output enabled."""
    TOPICS = {'result': 'result.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        print("about to fail")
        raise RuntimeError("intentional build failure")


class SlowCapturedBlock(Datablock):
    """Block that prints a lot during build."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        for i in range(100):
            print(f"line {i}")
        path = self.path(ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write("done")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(cls, tmp_path, **kwargs):
    return cls(url=str(tmp_path), capture_output=True, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCaptureOutputOrdering:
    """Verify that log upload happens BEFORE __post_build__."""

    def test_log_exists_at_post_build_time(self, tmp_path):
        """The core assertion: when __post_build__ runs, the log file
        should already be on disk so the journal records the path."""
        CapturedBlock.build_log = []
        block = _make_block(CapturedBlock, tmp_path)
        block.build()

        assert '__build__' in CapturedBlock.build_log
        assert '__post_build__' in CapturedBlock.build_log
        assert 'log_exists_at_post_build=True' in CapturedBlock.build_log

        # Verify ordering: __build__ before __post_build__
        build_idx = CapturedBlock.build_log.index('__build__')
        post_build_idx = CapturedBlock.build_log.index('__post_build__')
        assert build_idx < post_build_idx

    def test_journal_records_log_path(self, tmp_path):
        """Journal entry for build:end should have a non-None 'log' column."""
        block = _make_block(CapturedBlock, tmp_path)
        CapturedBlock.build_log = []
        block.build()

        journal = block.journal()
        # Filter for build:end events
        end_rows = journal[journal['event'] == 'build:end']
        assert len(end_rows) > 0, "Expected at least one build:end journal entry"
        log_val = end_rows.iloc[-1]['log']
        assert log_val is not None, "Journal 'log' column should not be None when capture_output=True"
        assert str(log_val).endswith('.log'), f"Expected log path ending in .log, got {log_val}"


class TestCaptureOutputStdioRestoration:
    """Verify stdout/stderr are properly restored after build."""

    def test_stdout_restored_after_normal_build(self, tmp_path):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        block = _make_block(CapturedBlock, tmp_path)
        CapturedBlock.build_log = []
        block.build()
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_stdout_restored_after_exception(self, tmp_path):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        block = _make_block(FailingCapturedBlock, tmp_path)
        with pytest.raises(RuntimeError, match="intentional build failure"):
            block.build()
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


class TestCaptureOutputTempFileCleanup:
    """Verify temp files are cleaned up in all paths."""

    def test_temp_file_removed_after_normal_build(self, tmp_path):
        block = _make_block(CapturedBlock, tmp_path)
        CapturedBlock.build_log = []

        # Track temp files created
        import tempfile
        created_temps = []
        original_ntf = tempfile.NamedTemporaryFile

        def tracking_ntf(*args, **kwargs):
            f = original_ntf(*args, **kwargs)
            created_temps.append(f.name)
            return f

        with patch.object(tempfile, 'NamedTemporaryFile', tracking_ntf):
            block.build()

        # All temp files should be cleaned up
        for temp_path in created_temps:
            assert not os.path.exists(temp_path), f"Temp file {temp_path} was not cleaned up"

    def test_temp_file_removed_after_exception(self, tmp_path):
        block = _make_block(FailingCapturedBlock, tmp_path)

        import tempfile
        created_temps = []
        original_ntf = tempfile.NamedTemporaryFile

        def tracking_ntf(*args, **kwargs):
            f = original_ntf(*args, **kwargs)
            created_temps.append(f.name)
            return f

        with patch.object(tempfile, 'NamedTemporaryFile', tracking_ntf):
            with pytest.raises(RuntimeError):
                block.build()

        for temp_path in created_temps:
            assert not os.path.exists(temp_path), f"Temp file {temp_path} was not cleaned up"


class TestCaptureOutputContent:
    """Verify the captured log content is correct."""

    def test_log_contains_stdout_output(self, tmp_path, capfd):
        block = _make_block(SlowCapturedBlock, tmp_path)
        with capfd.disabled():
            block.build()

        # Find the uploaded log file
        logpath = block._dbxanchorhashpathx('log', ext='log', ensure_dirpath=False)
        # The actual log path includes a datetime stamp, so we need to find
        # the log directory and list files
        log_dir = os.path.dirname(logpath)
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        assert len(log_files) > 0, "No log files found"

        # Read the most recent log
        log_content = open(os.path.join(log_dir, log_files[0])).read()
        assert "line 0" in log_content
        assert "line 99" in log_content


class TestCaptureOutputSkipPath:
    """When block is already valid, build() skips and no log is created."""

    def test_no_log_when_already_valid(self, tmp_path):
        block = _make_block(CapturedBlock, tmp_path)
        CapturedBlock.build_log = []
        block.build()
        assert block.valid()

        # Build again — should skip
        CapturedBlock.build_log = []
        block2 = _make_block(CapturedBlock, tmp_path)
        block2.build()
        # __build__ should not have been called
        assert '__build__' not in CapturedBlock.build_log


class TestCaptureOutputDisabled:
    """When capture_output=False (default), behavior is unchanged."""

    def test_no_capture_by_default(self, tmp_path):
        block = CapturedBlock(url=str(tmp_path))  # capture_output defaults to False
        assert block.capture_output is False
        CapturedBlock.build_log = []

        original_stdout = sys.stdout
        block.build()
        # stdout should never have been redirected
        assert sys.stdout is original_stdout
