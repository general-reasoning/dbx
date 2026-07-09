"""
Tests for Datablock.synclocal(topic, *, suffix, key, validate, latest).

Generalizes the find-latest-checkpoint-and-pull-it-if-missing pattern:
list a directory topic, filter by suffix, sort by a parsed key, and
either sync every missing entry (latest=False) or sync only the
newest entry that passes an optional validate() check, falling back
to progressively older entries otherwise (latest=True).
"""
import os
import re
import pytest
import fsspec

from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.delenv('DBX_LOCAL', raising=False)


@pytest.fixture
def mem_url():
    uid = os.urandom(4).hex()
    return f"memory://dbx_test_synclocal_{uid}"


@pytest.fixture(autouse=True)
def _clear_memory_fs():
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    yield
    fs.store.clear()


class CkptBlock(Datablock):
    """Directory topic holding many checkpoint-shaped files."""
    TOPICS = {'ckpts': None}


def step_key(name):
    return int(re.search(r'ckpt_step_(\d+)\.pt', name).group(1))


def make_ckpts(block, steps, *, via_fs=False):
    dirpath = block.dirpath('ckpts', ensure=True)
    for step in steps:
        p = os.path.join(dirpath, f'ckpt_step_{step}.pt')
        if via_fs:
            with block.fs.open(p, 'w') as f:
                f.write(f'ckpt at step {step}')
        else:
            with open(p, 'w') as f:
                f.write(f'ckpt at step {step}')


# ---------------------------------------------------------------------------
# latest=True
# ---------------------------------------------------------------------------

class TestSyncLocalLatest:

    def test_pulls_highest_key(self, tmp_path):
        block = CkptBlock(url=str(tmp_path / 'store'))
        make_ckpts(block, [10, 100, 20, 5])
        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True)
        assert result.endswith('ckpt_step_100.pt')
        assert os.path.isfile(result)

    def test_falls_back_on_validate_failure(self, tmp_path):
        block = CkptBlock(url=str(tmp_path / 'store'))
        make_ckpts(block, [10, 100, 20, 5])

        def validate(path):
            return 'step_100' not in path and 'step_20' not in path

        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True, validate=validate)
        assert result.endswith('ckpt_step_10.pt')

    def test_returns_none_when_all_invalid(self, tmp_path):
        block = CkptBlock(url=str(tmp_path / 'store'))
        make_ckpts(block, [10, 20])
        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True, validate=lambda p: False)
        assert result is None

    def test_returns_none_when_no_entries(self, tmp_path):
        block = CkptBlock(url=str(tmp_path / 'store'))
        block.dirpath('ckpts', ensure=True)
        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True)
        assert result is None

    def test_suffix_filters_out_non_matching_entries(self, tmp_path):
        block = CkptBlock(url=str(tmp_path / 'store'))
        make_ckpts(block, [10, 100])
        dirpath = block.dirpath('ckpts')
        with open(os.path.join(dirpath, 'ckpt_step_999.tmp'), 'w') as f:
            f.write('incomplete')
        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True)
        assert result.endswith('ckpt_step_100.pt')

    def test_nonlocal_url_stages_via_dbx_local(self, tmp_path, monkeypatch, mem_url):
        staging = tmp_path / 'staging'
        monkeypatch.setenv('DBX_LOCAL', str(staging))
        block = CkptBlock(url=mem_url)
        make_ckpts(block, [5, 50], via_fs=True)
        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True)
        assert result.startswith(str(staging))
        assert os.path.isfile(result)
        with open(result) as f:
            assert f.read() == 'ckpt at step 50'

    def test_already_local_entries_are_not_repulled(self, tmp_path, monkeypatch, mem_url):
        staging = tmp_path / 'staging'
        monkeypatch.setenv('DBX_LOCAL', str(staging))
        block = CkptBlock(url=mem_url)
        make_ckpts(block, [5, 50], via_fs=True)

        block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True)

        calls = []
        original_pull = block.pull
        monkeypatch.setattr(block, 'pull', lambda *a, **kw: calls.append((a, kw)) or original_pull(*a, **kw))

        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=True)
        assert result.endswith('ckpt_step_50.pt')
        assert calls == []


# ---------------------------------------------------------------------------
# latest=False
# ---------------------------------------------------------------------------

class TestSyncLocalAll:

    def test_pulls_every_matching_entry_sorted(self, tmp_path):
        block = CkptBlock(url=str(tmp_path / 'store'))
        make_ckpts(block, [10, 100, 20, 5])
        result = block.synclocal('ckpts', suffix='.pt', key=step_key, latest=False)
        assert [os.path.basename(p) for p in result] == [
            'ckpt_step_5.pt', 'ckpt_step_10.pt', 'ckpt_step_20.pt', 'ckpt_step_100.pt',
        ]
        for p in result:
            assert os.path.isfile(p)

    def test_default_key_is_lexical(self, tmp_path):
        block = CkptBlock(url=str(tmp_path / 'store'))
        dirpath = block.dirpath('ckpts', ensure=True)
        for name in ('b.txt', 'a.txt', 'c.txt'):
            with open(os.path.join(dirpath, name), 'w') as f:
                f.write(name)
        result = block.synclocal('ckpts')
        assert [os.path.basename(p) for p in result] == ['a.txt', 'b.txt', 'c.txt']
