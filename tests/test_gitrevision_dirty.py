"""
Tests for the dirty-repo check bypass in gitrevision() and the DBXWRKROOT
environment variable set by gitwrkreposetup().

Strategy: we mock git.Repo so we never need a real git repository, and we
patch the module-level globals (DBX_USE_WORK_REPO, DBXGITREPO, DBXWRKROOT) as
well as os.environ to exercise each code path in isolation.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import dbx.dbx as dbxmod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dirty_repo(hexsha="abc123"):
    """Return a mock git.Repo that reports is_dirty() == True."""
    repo = MagicMock()
    repo.is_dirty.return_value = True
    repo.head.commit.hexsha = hexsha
    return repo


def _make_clean_repo(hexsha="abc123"):
    """Return a mock git.Repo that reports is_dirty() == False."""
    repo = MagicMock()
    repo.is_dirty.return_value = False
    repo.head.commit.hexsha = hexsha
    return repo


# ---------------------------------------------------------------------------
# gitrevision() dirty-check: normal (no wrkrepo) paths
# ---------------------------------------------------------------------------

class TestGitrevisionDirtyCheck:

    def test_dirty_repo_raises_without_wrkrepo(self, monkeypatch):
        """Dirty repo should raise when no wrkrepo is active."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBXGITREPO', '/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)
        monkeypatch.delenv('DBX_DIRTY_REPO_OK', raising=False)

        mock_repo = _make_dirty_repo()
        with patch('dbx.dbx.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            with pytest.raises(ValueError, match="Dirty git repo"):
                dbxmod.gitrevision()

    def test_clean_repo_succeeds_without_wrkrepo(self, monkeypatch):
        """Clean repo should return a hexsha without error."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBXGITREPO', '/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)
        monkeypatch.delenv('DBX_DIRTY_REPO_OK', raising=False)

        mock_repo = _make_clean_repo("deadbeef")
        with patch('dbx.dbx.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dbxmod.gitrevision()
        assert rev == "deadbeef"

    def test_dirty_repo_ok_with_dbxdirtyrepok(self, monkeypatch):
        """DBXDIRTYREPOK env var should bypass the dirty check."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBXGITREPO', '/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')

        mock_repo = _make_dirty_repo("cafebabe")
        with patch('dbx.dbx.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dbxmod.gitrevision()
        assert rev == "cafebabe"


# ---------------------------------------------------------------------------
# gitrevision() dirty-check: wrkrepo paths
# ---------------------------------------------------------------------------

class TestGitrevisionSkipsWhenWrkrepo:

    def test_dirty_repo_ok_when_dbxusewrkrepo_global_set(self, monkeypatch):
        """Master process: DBX_USE_WORK_REPO global set → dirty check skipped."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', '/tmp/wrk/project')
        monkeypatch.setattr(dbxmod, 'DBXGITREPO', '/tmp/wrk/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)
        monkeypatch.delenv('DBX_DIRTY_REPO_OK', raising=False)

        mock_repo = _make_dirty_repo("deadc0de")
        with patch('dbx.dbx.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dbxmod.gitrevision()
        assert rev == "deadc0de"
        # is_dirty should never have been called (or at least not caused a raise)

    def test_dirty_repo_ok_when_dbxwrkroot_env_set(self, monkeypatch):
        """Worker process: DBXWRKROOT env var set → dirty check skipped."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBXGITREPO', '/fake/project')
        monkeypatch.setenv('DBX_WORK_ROOT', '/tmp/wrk/project')
        monkeypatch.delenv('DBX_DIRTY_REPO_OK', raising=False)

        mock_repo = _make_dirty_repo("0xdeadbeef")
        with patch('dbx.dbx.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dbxmod.gitrevision()
        assert rev == "0xdeadbeef"


# ---------------------------------------------------------------------------
# gitwrkreposetup() sets DBXWRKROOT in the environment
# ---------------------------------------------------------------------------

class TestGitwrkreposetupSetsEnvVar:

    def test_dbxwrkroot_set_in_env_after_setup(self, monkeypatch, tmp_path):
        """gitwrkreposetup() must set DBXWRKROOT=True in os.environ."""
        # Pre-conditions: no wrkrepo yet, a valid DBXGITREPO
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBXWRKROOT', None)
        monkeypatch.setattr(dbxmod, 'DBXGITREPO', '/fake/gitrepo')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        # Provide a fake clone destination
        fake_wrk_dir = tmp_path / "gitrepo"
        fake_wrk_dir.mkdir()

        # A real TemporaryDirectory whose .name points at tmp_path so the
        # os.path.join(wrkroot.name, package) path exists.
        import tempfile
        fake_tmpdir = MagicMock(spec=tempfile.TemporaryDirectory)
        fake_tmpdir.name = str(tmp_path)

        with (
            patch('dbx.dbx.dbx_repos', return_value=(None, '/fake/gitrepo')),
            patch('dbx.dbx.gitclone', return_value=str(fake_wrk_dir)),
            patch('dbx.dbx.gitcheckout', return_value=str(fake_wrk_dir)),
            patch('dbx.dbx.tempfile.TemporaryDirectory', return_value=fake_tmpdir),
            patch('dbx.dbx.sys') as mock_sys,
        ):
            mock_sys.path = []
            # revision=something triggers use_wrkrepo=True inside gitwrkreposetup
            dbxmod.gitwrkreposetup(revision='HEAD', reason="test")

        assert os.environ.get('DBX_WORK_ROOT') == str(fake_wrk_dir), \
            "DBX_WORK_ROOT should be set to the wrkrepo path in os.environ after gitwrkreposetup()"

        # gitwrkreposetup writes directly to os.environ (bypassing monkeypatch),
        # so we must clean up manually to prevent leaking into later tests.
        os.environ.pop('DBX_WORK_ROOT', None)
        os.environ.pop('DBX_GIT_REPO', None)

    def test_dbxwrkroot_not_set_when_no_wrkrepo_needed(self, monkeypatch):
        """gitwrkreposetup() with no revision and DBX_USE_WORK_REPO env not 'True'
        should NOT set DBXWRKROOT (no wrkrepo is created)."""
        monkeypatch.setattr(dbxmod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dbxmod, 'DBXGITREPO', '/fake/gitrepo')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)
        monkeypatch.delenv('DBX_USE_WORK_REPO', raising=False)  # env flag also off

        dbxmod.gitwrkreposetup(revision=None, reason="test")

        assert 'DBX_WORK_ROOT' not in os.environ, \
            "DBX_WORK_ROOT should NOT be set when no wrkrepo is created"
