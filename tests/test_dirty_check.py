"""Tests for the early dirty-check in gitwrkreposetup().

These tests use a *real* temporary git repo (not the user's working tree)
to verify that gitwrkreposetup() raises before cloning when the source
repo has uncommitted changes, and that DBX_DIRTY_REPO_OK suppresses the error.
"""
import os
import subprocess
import tempfile

import pytest

import dbx.datablocks as datablocks
from dbx.datablocks import gitwrkreposetup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_repo(tmp_path: str) -> str:
    """Create a minimal git repo with one committed file."""
    repo_dir = os.path.join(tmp_path, "project")
    os.makedirs(repo_dir)
    subprocess.check_call(["git", "init"], cwd=repo_dir,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@test.com"], cwd=repo_dir)
    subprocess.check_call(["git", "config", "user.name", "Test"], cwd=repo_dir)
    filepath = os.path.join(repo_dir, "hello.txt")
    with open(filepath, "w") as f:
        f.write("hello\n")
    subprocess.check_call(["git", "add", "."], cwd=repo_dir,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo_dir,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return repo_dir


def _dirty(repo_dir: str):
    """Add an uncommitted change to the repo."""
    filepath = os.path.join(repo_dir, "hello.txt")
    with open(filepath, "a") as f:
        f.write("dirty\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDirtyCheck:

    def test_dirty_repo_raises(self, tmp_path, monkeypatch):
        """gitwrkreposetup must raise ValueError for a dirty source repo."""
        repo_dir = _make_clean_repo(str(tmp_path))
        _dirty(repo_dir)

        # Point DBX_GIT_REPO at the dirty repo and force work-repo creation.
        monkeypatch.setattr(datablocks, 'DBX_GIT_REPO', repo_dir)
        monkeypatch.setattr(datablocks, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(datablocks, 'DBX_WORK_ROOT', None)
        monkeypatch.setenv('DBX_USE_WORK_REPO', 'True')
        monkeypatch.delenv('DBX_DIRTY_REPO_OK', raising=False)

        with pytest.raises(ValueError, match="Dirty git repo"):
            gitwrkreposetup(reason="test")

    def test_clean_repo_succeeds(self, tmp_path, monkeypatch):
        """gitwrkreposetup must not raise for a clean source repo."""
        repo_dir = _make_clean_repo(str(tmp_path))

        monkeypatch.setattr(datablocks, 'DBX_GIT_REPO', repo_dir)
        monkeypatch.setattr(datablocks, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(datablocks, 'DBX_WORK_ROOT', None)
        monkeypatch.setenv('DBX_USE_WORK_REPO', 'True')
        monkeypatch.delenv('DBX_DIRTY_REPO_OK', raising=False)

        # Should not raise
        gitwrkreposetup(reason="test")

    def test_dirty_repo_ok_suppresses(self, tmp_path, monkeypatch):
        """DBX_DIRTY_REPO_OK=1 should suppress the dirty check."""
        repo_dir = _make_clean_repo(str(tmp_path))
        _dirty(repo_dir)

        monkeypatch.setattr(datablocks, 'DBX_GIT_REPO', repo_dir)
        monkeypatch.setattr(datablocks, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(datablocks, 'DBX_WORK_ROOT', None)
        monkeypatch.setenv('DBX_USE_WORK_REPO', 'True')
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')

        # Should not raise despite dirty repo
        gitwrkreposetup(reason="test")
