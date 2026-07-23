"""
Tests for Datablock.list(topic) and Datablock.topicsize(topic).

Verifies:
1. list() returns [] and topicsize() returns 0 before build.
2. list() on a single-file topic returns that file's detail dict.
3. topicsize() on a single-file topic equals the file's byte length.
4. list() on a directory topic recurses over all files (incl. nested).
5. topicsize() on a directory topic sums all (nested) file sizes.
6. list() excludes directory entries (files only).
7. local=True operates on the local cache.
"""
import os
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class SingleTopicBlock(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write("0123456789")  # 10 bytes


class DirTopicBlock(Datablock):
    """TOPICS[topic]=None — artifact is a directory with nested files."""
    TOPICS = {'images': None}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        dirpath = self.dirpath('images', ensure=True)
        for i in range(3):
            with open(os.path.join(dirpath, f'file_{i}.bin'), 'wb') as f:
                f.write(b'\x00' * 16)
        # A nested subdirectory to exercise recursion.
        nested = os.path.join(dirpath, 'sub')
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, 'deep.bin'), 'wb') as f:
            f.write(b'\x00' * 32)


def _make(cls, tmp_path, **kwargs):
    return cls(url=str(tmp_path), **kwargs)


class TestBeforeBuild:

    def test_list_empty(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        assert block.list('output') == []

    def test_topicsize_zero(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        assert block.topicsize('output') == 0


class TestSingleFileTopic:

    def test_list_returns_the_file(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        block.build()
        result = block.list('output')
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert os.path.basename(result[0]['name']) == 'output.txt'
        assert os.path.isabs(result[0]['name'])

    def test_topicsize_matches_bytes(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        block.build()
        assert block.topicsize('output') == 10


class TestDirTopic:

    def test_list_recurses(self, tmp_path):
        block = _make(DirTopicBlock, tmp_path)
        block.__build__()
        result = block.list('images')
        basenames = sorted(os.path.basename(e['name']) for e in result)
        # 3 top-level files + 1 nested file, no directory entries.
        assert basenames == ['deep.bin', 'file_0.bin', 'file_1.bin', 'file_2.bin']
        for e in result:
            assert isinstance(e, dict)
            assert e.get('type') != 'directory'

    def test_topicsize_sums_all(self, tmp_path):
        block = _make(DirTopicBlock, tmp_path)
        block.__build__()
        # 3 * 16 + 32 = 80
        assert block.topicsize('images') == 80


class TestLocal:

    def test_list_and_size_local(self, tmp_path):
        """When url is local, local=True mirrors the default behavior."""
        block = _make(SingleTopicBlock, tmp_path)
        block.build()
        result = block.list('output', local=True)
        assert len(result) == 1
        assert block.topicsize('output', local=True) == 10
