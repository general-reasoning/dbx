"""
Tests for the DBX_LOCAL / local= resolution invariant on Datablock.

Verifies that DBX_LOCAL (and the local= constructor kwarg) is only ever
consulted when DBX_ROOT/DBX_URL (i.e. url=) points at a non-local
filesystem. When url= is already local, local=True must be a no-op:
localfs/localroot alias fs/root regardless of DBX_LOCAL or local=.
"""
import os
import pytest

from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.delenv('DBX_LOCAL', raising=False)


# ---------------------------------------------------------------------------
# url= is local: DBX_LOCAL / local= must never be consulted
# ---------------------------------------------------------------------------

class TestLocalUrlIgnoresLocalStaging:

    def test_localfs_and_localroot_alias_fs_and_root(self, tmp_path):
        block = Datablock(url=str(tmp_path))
        assert block.localfs is block.fs
        assert block.localroot == block.root

    def test_dbx_local_env_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_LOCAL', '/tmp/dbx_should_not_be_used')
        block = Datablock(url=str(tmp_path))
        assert block.localroot == block.root
        assert block.localroot != '/tmp/dbx_should_not_be_used'

    def test_local_kwarg_is_ignored(self, tmp_path):
        block = Datablock(url=str(tmp_path), local='/tmp/dbx_should_not_be_used')
        assert block.localroot == block.root
        assert block.localroot != '/tmp/dbx_should_not_be_used'

    def test_path_local_true_matches_local_false(self, tmp_path):
        class SingleTopicBlock(Datablock):
            TOPICS = {'output': 'output.txt'}

        block = SingleTopicBlock(url=str(tmp_path))
        assert block.path('output', local=True) == block.path('output', local=False)
        assert block.dirpath('output', local=True) == block.dirpath('output', local=False)


# ---------------------------------------------------------------------------
# url= is non-local: DBX_LOCAL / local= are consulted
# ---------------------------------------------------------------------------

class TestNonLocalUrlConsultsLocalStaging:

    def test_dbx_local_env_is_used(self, tmp_path, monkeypatch):
        staging = tmp_path / 'staging'
        monkeypatch.setenv('DBX_LOCAL', str(staging))
        block = Datablock(url="memory://dbx_test_nonlocal")
        assert block.localroot == str(staging)
        assert block.localfs is not block.fs

    def test_local_kwarg_overrides_env(self, tmp_path, monkeypatch):
        env_staging = tmp_path / 'from_env'
        kwarg_staging = tmp_path / 'from_kwarg'
        monkeypatch.setenv('DBX_LOCAL', str(env_staging))
        block = Datablock(url="memory://dbx_test_nonlocal", local=str(kwarg_staging))
        assert block.localroot == str(kwarg_staging)

    def test_defaults_to_tmp_dbx_when_unset(self):
        block = Datablock(url="memory://dbx_test_nonlocal")
        assert block.localroot == '/tmp/dbx'

    def test_path_local_true_differs_from_local_false(self, tmp_path, monkeypatch):
        staging = tmp_path / 'staging'
        monkeypatch.setenv('DBX_LOCAL', str(staging))

        class SingleTopicBlock(Datablock):
            TOPICS = {'output': 'output.txt'}

        block = SingleTopicBlock(url="memory://dbx_test_nonlocal")
        assert block.path('output', local=True) != block.path('output', local=False)
        assert block.path('output', local=True).startswith(str(staging))
