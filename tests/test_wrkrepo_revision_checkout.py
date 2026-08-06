"""
Tests for gitwrkreposetup() revision checkout in existing work repos.

When work repos were already cloned (DBX_USE_WORK_REPO is set) and
gitwrkreposetup() is called with a specific revision, it should
checkout that revision in the existing clones rather than silently
doing nothing.
"""
import importlib
import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call

import dbx.datablocks as dbxmod
import dbx.dataparts as dataparts_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path, *, commits=1):
    """Create a real git repo with the given number of commits.

    Returns a list of commit SHAs (oldest first).
    """
    os.makedirs(path, exist_ok=True)
    subprocess.check_call(["git", "init"], cwd=path,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "t@t.com"], cwd=path)
    subprocess.check_call(["git", "config", "user.name", "T"], cwd=path)

    shas = []
    for i in range(commits):
        fpath = os.path.join(path, f"file_{i}.txt")
        with open(fpath, "w") as f:
            f.write(f"content {i}\n")
        subprocess.check_call(["git", "add", "."], cwd=path,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "commit", "-m", f"commit {i}"], cwd=path,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path
        ).decode().strip()
        shas.append(sha)
    return shas


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGitwrkreposetupRevisionCheckout:
    """Tests for the elif branch: revision checkout in existing work repos."""

    def test_checkout_called_on_existing_project_wrkrepo(self, monkeypatch, tmp_path):
        """When DBX_USE_WORK_REPO points to an existing project-only work repo,
        gitwrkreposetup(revision=...) should checkout the revision there."""
        project_wrk = str(tmp_path / "project")
        shas = _init_git_repo(project_wrk, commits=2)
        old_rev = shas[0]

        # Simulate state after initial import-time clone (at HEAD = shas[1])
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', project_wrk)
        monkeypatch.setattr(dataparts_mod, 'DBX_WORK_ROOT', (None, tmp_path))
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', project_wrk)

        # Verify we start at HEAD (shas[1])
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_wrk
        ).decode().strip()
        assert head == shas[1]

        # Call with old revision — should checkout
        dbxmod.gitwrkreposetup(revision=old_rev, reason="test")

        # Verify the work repo is now at the old revision
        head_after = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_wrk
        ).decode().strip()
        assert head_after == old_rev

    def test_checkout_called_on_both_repos(self, monkeypatch, tmp_path):
        """When DBX_USE_WORK_REPO has both dbx and project paths,
        both repos are checked out to their respective revisions."""
        dbx_wrk = str(tmp_path / "dbx")
        project_wrk = str(tmp_path / "project")
        dbx_shas = _init_git_repo(dbx_wrk, commits=2)
        project_shas = _init_git_repo(project_wrk, commits=2)

        wrkrepo_str = f"{dbx_wrk}:{project_wrk}"
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', wrkrepo_str)
        monkeypatch.setattr(dataparts_mod, 'DBX_WORK_ROOT', (tmp_path, tmp_path))
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', wrkrepo_str)

        combined_rev = f"{dbx_shas[0]}:{project_shas[0]}"
        dbxmod.gitwrkreposetup(revision=combined_rev, reason="test")

        dbx_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dbx_wrk
        ).decode().strip()
        project_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_wrk
        ).decode().strip()
        assert dbx_head == dbx_shas[0]
        assert project_head == project_shas[0]

    def test_only_project_rev_checks_out_project_only(self, monkeypatch, tmp_path):
        """When revision has no ':' (project-only), only project is checked out."""
        dbx_wrk = str(tmp_path / "dbx")
        project_wrk = str(tmp_path / "project")
        dbx_shas = _init_git_repo(dbx_wrk, commits=2)
        project_shas = _init_git_repo(project_wrk, commits=2)

        wrkrepo_str = f"{dbx_wrk}:{project_wrk}"
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', wrkrepo_str)
        monkeypatch.setattr(dataparts_mod, 'DBX_WORK_ROOT', (tmp_path, tmp_path))
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', wrkrepo_str)

        # Project-only revision (no ':')
        dbxmod.gitwrkreposetup(revision=project_shas[0], reason="test")

        # dbx should stay at HEAD
        dbx_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dbx_wrk
        ).decode().strip()
        assert dbx_head == dbx_shas[1], "dbx repo should remain at HEAD"

        # project should be checked out
        project_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_wrk
        ).decode().strip()
        assert project_head == project_shas[0]

    def test_noop_when_no_revision(self, monkeypatch, tmp_path):
        """When revision is None, existing work repos should not be touched."""
        project_wrk = str(tmp_path / "project")
        shas = _init_git_repo(project_wrk, commits=2)

        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', project_wrk)
        monkeypatch.setattr(dataparts_mod, 'DBX_WORK_ROOT', (None, tmp_path))
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', project_wrk)

        dbxmod.gitwrkreposetup(revision=None, reason="test")

        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_wrk
        ).decode().strip()
        assert head == shas[1], "Should remain at HEAD when revision=None"

    def test_importlib_caches_invalidated(self, monkeypatch, tmp_path):
        """importlib.invalidate_caches() is called after checkout."""
        project_wrk = str(tmp_path / "project")
        shas = _init_git_repo(project_wrk, commits=2)

        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', project_wrk)
        monkeypatch.setattr(dataparts_mod, 'DBX_WORK_ROOT', (None, tmp_path))
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', project_wrk)

        with patch.object(importlib, 'invalidate_caches') as mock_invalidate:
            dbxmod.gitwrkreposetup(revision=shas[0], reason="test")
            mock_invalidate.assert_called_once()

    def test_checkout_changes_file_content(self, monkeypatch, tmp_path):
        """After checkout, files in the work repo reflect the old revision."""
        project_wrk = str(tmp_path / "project")
        os.makedirs(project_wrk)
        subprocess.check_call(["git", "init"], cwd=project_wrk,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "config", "user.email", "t@t.com"], cwd=project_wrk)
        subprocess.check_call(["git", "config", "user.name", "T"], cwd=project_wrk)

        # Commit 1: file has "old content"
        marker = os.path.join(project_wrk, "marker.py")
        with open(marker, "w") as f:
            f.write("VERSION = 'old'\n")
        subprocess.check_call(["git", "add", "."], cwd=project_wrk,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "commit", "-m", "old"], cwd=project_wrk,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        old_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_wrk
        ).decode().strip()

        # Commit 2: file has "new content"
        with open(marker, "w") as f:
            f.write("VERSION = 'new'\n")
        subprocess.check_call(["git", "add", "."], cwd=project_wrk,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "commit", "-m", "new"], cwd=project_wrk,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Verify file is at "new"
        with open(marker) as f:
            assert "new" in f.read()

        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', project_wrk)
        monkeypatch.setattr(dataparts_mod, 'DBX_WORK_ROOT', (None, tmp_path))
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', project_wrk)

        # Checkout old revision
        dbxmod.gitwrkreposetup(revision=old_sha, reason="test")

        # File should now have "old content"
        with open(marker) as f:
            content = f.read()
        assert "old" in content
        assert "new" not in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
