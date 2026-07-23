"""
Tests for JournalEntry.ls / .list / .topicsize.

These mirror Datablock.ls/.list/.topicsize but resolve the topic path from
the journal entry's recorded ``paths`` field rather than a live block.

Verifies:
1. entry.paths / entry.topics parse the recorded stringified dicts.
2. entry.list() on a single-file topic returns that file's detail dict.
3. entry.topicsize() equals the file's byte length.
4. entry.list() on a directory topic recurses (incl. nested files).
5. entry.topicsize() sums all (nested) file sizes.
6. entry.ls()/.list()/.topicsize() agree with the block's own methods.
7. An unrecorded topic raises KeyError.
"""
import os
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, JournalEntry


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
        nested = os.path.join(dirpath, 'sub')
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, 'deep.bin'), 'wb') as f:
            f.write(b'\x00' * 32)


def _built(cls, tmp_path):
    """Build a block and return (block, its last journal entry)."""
    block = cls(url=str(tmp_path))
    block.build()
    entry = block.lastbuilt()
    assert isinstance(entry, JournalEntry)
    return block, entry


class TestRecordedFields:

    def test_paths_and_topics_single(self, tmp_path):
        block, entry = _built(SingleTopicBlock, tmp_path)
        assert 'output' in entry.paths
        assert os.path.basename(entry.paths['output']) == 'output.txt'
        assert entry.topics.get('output') == 'output.txt'

    def test_topics_marks_dir(self, tmp_path):
        block, entry = _built(DirTopicBlock, tmp_path)
        assert 'images' in entry.paths
        assert entry.topics.get('images') is None


class TestSingleFileTopic:

    def test_list_returns_the_file(self, tmp_path):
        block, entry = _built(SingleTopicBlock, tmp_path)
        result = entry.list('output')
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert os.path.basename(result[0]['name']) == 'output.txt'
        assert os.path.isabs(result[0]['name'])

    def test_topicsize_matches_bytes(self, tmp_path):
        block, entry = _built(SingleTopicBlock, tmp_path)
        assert entry.topicsize('output') == 10


class TestDirTopic:

    def test_list_recurses(self, tmp_path):
        block, entry = _built(DirTopicBlock, tmp_path)
        result = entry.list('images')
        basenames = sorted(os.path.basename(e['name']) for e in result)
        assert basenames == ['deep.bin', 'file_0.bin', 'file_1.bin', 'file_2.bin']
        for e in result:
            assert e.get('type') != 'directory'

    def test_topicsize_sums_all(self, tmp_path):
        block, entry = _built(DirTopicBlock, tmp_path)
        # 3 * 16 + 32 = 80
        assert entry.topicsize('images') == 80


class TestMatchesBlock:

    def test_agrees_with_block_single(self, tmp_path):
        block, entry = _built(SingleTopicBlock, tmp_path)
        assert sorted(entry.ls('output')) == sorted(block.ls('output'))
        assert entry.topicsize('output') == block.topicsize('output')

    def test_agrees_with_block_dir(self, tmp_path):
        block, entry = _built(DirTopicBlock, tmp_path)
        assert entry.topicsize('images') == block.topicsize('images')
        entry_names = sorted(os.path.basename(e['name']) for e in entry.list('images'))
        block_names = sorted(os.path.basename(e['name']) for e in block.list('images'))
        assert entry_names == block_names


class TestMissingTopic:

    def test_unknown_topic_raises(self, tmp_path):
        block, entry = _built(SingleTopicBlock, tmp_path)
        with pytest.raises(KeyError):
            entry.list('nonexistent')
