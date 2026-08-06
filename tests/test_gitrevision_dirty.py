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
from unittest.mock import patch, MagicMock, PropertyMock
import git as gitmod
import dbx.datablocks as dbxmod
import dbx.dataparts as dataparts_mod


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
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        mock_repo = _make_mock_repo(dirty=True, hexsha="deadbeef")
        with patch('dbx.dataparts.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dataparts_mod.gitrevision()
        assert rev == "deadbeef"

    def test_clean_repo_returns_hexsha(self, monkeypatch):
        """Clean repo should return a hexsha."""
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        mock_repo = _make_mock_repo(dirty=False, hexsha="cafebabe")
        with patch('dbx.dataparts.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dataparts_mod.gitrevision()
        assert rev == "cafebabe"

    def test_wrkrepo_set_returns_hexsha(self, monkeypatch):
        """When wrkrepo is active, gitrevision still returns hexsha."""
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', '/tmp/wrk/project')
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/tmp/wrk/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        mock_repo = _make_mock_repo(dirty=True, hexsha="deadc0de")
        with patch('dbx.dataparts.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dataparts_mod.gitrevision()
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

        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_WORK_ROOT', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', repo_dir)
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
            patch('dbx.dataparts.dbx_repos', return_value=(None, repo_dir)),
            patch('dbx.dataparts.gitclone', return_value=expected_wrkrepo),
            patch('dbx.dataparts.gitcheckout', return_value=expected_wrkrepo),
            patch('dbx.dataparts.tempfile.TemporaryDirectory', return_value=fake_tmpdir),
            patch('dbx.dataparts.sys') as mock_sys,
        ):
            mock_sys.path = []
            dataparts_mod.gitwrkreposetup(revision='HEAD', reason="test")

        assert 'DBX_WORK_ROOT' in os.environ, \
            "DBX_WORK_ROOT should be set after gitwrkreposetup()"

        # Manual cleanup (gitwrkreposetup bypasses monkeypatch).
        os.environ.pop('DBX_WORK_ROOT', None)
        os.environ.pop('DBX_GIT_REPO', None)

    def test_dbxwrkroot_not_set_when_no_wrkrepo_needed(self, monkeypatch):
        """No revision and no env flag → no wrkrepo, no DBX_WORK_ROOT."""
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/fake/gitrepo')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)
        monkeypatch.delenv('DBX_USE_WORK_REPO', raising=False)

        dataparts_mod.gitwrkreposetup(revision=None, reason="test")

        assert 'DBX_WORK_ROOT' not in os.environ, \
            "DBX_WORK_ROOT should NOT be set when no wrkrepo is created"


# ---------------------------------------------------------------------------
# gitrevision() — graceful fallback for DDP ownership / invalid-repo errors
# (commit 30051b3)
# ---------------------------------------------------------------------------

class TestGitrevisionOwnershipErrors:
    """
    Cover the two exception paths observed during DDP (Lightning
    SubprocessScriptLauncher) training where the work-repo tmp dir is
    owned by a different process:

    rank2: git.Repo(path) raises InvalidGitRepositoryError because the
           tmp dir is not a git repo at all.
    rank1: git.Repo(path) succeeds but repo.head.commit.hexsha raises
           ValueError("SHA is empty, possible dubious ownership …").

    In both cases gitrevision() must return None without propagating the
    exception.
    """

    def test_invalid_git_repo_returns_none(self, monkeypatch):
        """git.Repo() raises InvalidGitRepositoryError → gitrevision() returns None (rank2 scenario)."""
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/tmp/tmpbl7jey0b/autopath')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        with patch('dbx.dataparts.git') as mock_git:
            mock_git.Repo.side_effect = gitmod.exc.InvalidGitRepositoryError(
                '/tmp/tmpbl7jey0b/autopath'
            )
            rev = dataparts_mod.gitrevision()

        assert rev is None

    def test_dubious_ownership_hexsha_raises_returns_none(self, monkeypatch):
        """repo.head.commit.hexsha raises ValueError (safe-directory) → gitrevision() returns None (rank1 scenario)."""
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/tmp/tmpbl7jey0b/autopath')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        mock_commit = MagicMock()
        type(mock_commit).hexsha = PropertyMock(
            side_effect=ValueError(
                "SHA is empty, possible dubious ownership in the repository at "
                "/tmp/tmpbl7jey0b/autopath"
            )
        )
        mock_repo = MagicMock()
        mock_repo.head.commit = mock_commit

        with patch('dbx.dataparts.git') as mock_git:
            mock_git.Repo.return_value = mock_repo
            rev = dataparts_mod.gitrevision()

        assert rev is None

    def test_invalid_repo_does_not_raise(self, monkeypatch):
        """gitrevision() must not propagate any exception from git.Repo()."""
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/tmp/some/invalid/path')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        with patch('dbx.dataparts.git') as mock_git:
            mock_git.Repo.side_effect = Exception("unexpected git failure")
            # Must not raise — any exception from get_rev() is caught.
            rev = dataparts_mod.gitrevision()

        assert rev is None

    def test_project_repo_fails_dbx_repo_succeeds(self, monkeypatch):
        """When only the project repo raises, dbx_rev still forms the revision string."""
        # Provide a colon-separated path so dbx_repos() returns two distinct repos.
        # dbx_repos() puts the path containing '/dbx' first as d_repo.
        monkeypatch.setattr(dataparts_mod, 'DBX_USE_WORK_REPO', None)
        monkeypatch.setattr(dataparts_mod, 'DBX_GIT_REPO', '/fake/dbx:/fake/project')
        monkeypatch.delenv('DBX_WORK_ROOT', raising=False)

        good_repo = _make_mock_repo(hexsha='aabbccdd')
        bad_commit = MagicMock()
        type(bad_commit).hexsha = PropertyMock(
            side_effect=ValueError("SHA is empty, dubious ownership")
        )
        bad_repo = MagicMock()
        bad_repo.head.commit = bad_commit

        def repo_factory(path, *args, **kwargs):
            if 'dbx' in path:
                return good_repo
            raise gitmod.exc.InvalidGitRepositoryError(path)

        with patch('dbx.dataparts.git') as mock_git:
            mock_git.Repo.side_effect = repo_factory
            rev = dataparts_mod.gitrevision()

        # dbx_rev='aabbccdd' is truthy → revision = f"{dbx_rev}:{project_rev}"
        assert rev == 'aabbccdd:None'
