"""
Tests for the dirty-repo check in gitwrkreposetup() and gitrevision().

After the refactor, the dirty check lives in gitwrkreposetup() (before
cloning), not in gitrevision().  gitrevision() now simply returns the
commit hash without dirtiness validation.

Strategy: we use both mock repos (for gitrevision) and real temporary
git repos (for gitwrkreposetup dirty-check paths).
"""
import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock
import dbx.datablocks as dbxmod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_repo(*, dirty=False, hexsha="abc123"):
    """Return a mock git.Repo."""
    repo = MagicMock()
    repo.is_dirty.return_value = dirty
    repo.head.commit.hexsha = hexsha
    return repo


# ---------------------------------------------------------------------------
# gitrevision() — no longer checks dirtiness
# ---------------------------------------------------------------------------

class TestGitrevisionNoDirtyCheck:

    def test_dirty_repo_returns_hexsha(self, monkeypatch):
        """gitrevision() must not raise even for dirty repos (check moved)."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBX_GIT_REPO', '/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        mock_repo = _make_mock_repo(dirty=True, hexsha="deadbeef")
        with patch('dbx.datablocks.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dbxmod.gitrevision()
        assert rev == "deadbeef"

    def test_clean_repo_returns_hexsha(self, monkeypatch):
        """Clean repo should return a hexsha."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBX_GIT_REPO', '/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        mock_repo = _make_mock_repo(dirty=False, hexsha="cafebabe")
        with patch('dbx.datablocks.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dbxmod.gitrevision()
        assert rev == "cafebabe"

    def test_wrkrepo_set_returns_hexsha(self, monkeypatch):
        """When wrkrepo is active, gitrevision still returns hexsha."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', '/tmp/wrk/project')
        monkeypatch.setattr(dbxmod, 'DBX_GIT_REPO', '/tmp/wrk/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        mock_repo = _make_mock_repo(dirty=True, hexsha="deadc0de")
        with patch('dbx.datablocks.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dbxmod.gitrevision()
        assert rev == "deadc0de"


# ---------------------------------------------------------------------------
# gitwrkreposetup() sets DBX_WORK_ROOT in the environment
# ---------------------------------------------------------------------------

class TestGitwrkreposetupSetsEnvVar:

    def test_dbxwrkroot_set_in_env_after_setup(self, monkeypatch, tmp_path):
        """gitwrkreposetup() must set DBX_WORK_ROOT in os.environ."""
        # Create a real clean repo so the dirty check passes.
        repo_dir = str(tmp_path / "project")
        os.makedirs(repo_dir)
        subprocess.check_call(["git", "init"], cwd=repo_dir,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "config", "user.email", "t@t.com"], cwd=repo_dir)
        subprocess.check_call(["git", "config", "user.name", "T"], cwd=repo_dir)
        with open(os.path.join(repo_dir, "f.txt"), "w") as f:
            f.write("x\n")
        subprocess.check_call(["git", "add", "."], cwd=repo_dir,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "commit", "-m", "init"], cwd=repo_dir,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBX_WORK_ROOT', None)
        monkeypatch.setattr(dbxmod, 'DBX_GIT_REPO', repo_dir)
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        import tempfile as _tempfile
        # setup_wrkrepo computes: wrkrepo = os.path.join(tmpdir.name, basename(repo))
        # so we need gitclone to return that same path.
        clone_parent = tmp_path / "wrkroot"
        clone_parent.mkdir()
        expected_wrkrepo = str(clone_parent / "project")

        fake_tmpdir = MagicMock(spec=_tempfile.TemporaryDirectory)
        fake_tmpdir.name = str(clone_parent)

        with (
            patch('dbx.datablocks.dbx_repos', return_value=(None, repo_dir)),
            patch('dbx.datablocks.gitclone', return_value=expected_wrkrepo),
            patch('dbx.datablocks.gitcheckout', return_value=expected_wrkrepo),
            patch('dbx.datablocks.tempfile.TemporaryDirectory', return_value=fake_tmpdir),
            patch('dbx.datablocks.sys') as mock_sys,
        ):
            mock_sys.path = []
            dbxmod.gitwrkreposetup(revision='HEAD', reason="test")

        assert 'DBX_WORK_ROOT' in os.environ, \
            "DBX_WORK_ROOT should be set after gitwrkreposetup()"

        # Manual cleanup (gitwrkreposetup bypasses monkeypatch).
        os.environ.pop('DBX_WORK_ROOT', None)
        os.environ.pop('DBX_GIT_REPO', None)

    def test_dbxwrkroot_not_set_when_no_wrkrepo_needed(self, monkeypatch):
        """No revision and no env flag → no wrkrepo, no DBX_WORK_ROOT."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBX_GIT_REPO', '/fake/gitrepo')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)
        monkeypatch.delenv('DBX_USE_WORK_REPO', raising=False)

        dbxmod.gitwrkreposetup(revision=None, reason="test")

        assert 'DBX_WORK_ROOT' not in os.environ, \
            "DBX_WORK_ROOT should NOT be set when no wrkrepo is created"
