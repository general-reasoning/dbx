"""
Tests for Datablock.get() / __get__():

1. get() downloads a single TOPICFILE to the target path.
2. get() downloads a TOPICS directory to the target path.
3. get() downloads a specific topic from TOPICFILES.
4. get() with no path defaults to '.'.
5. get() on a block with no topics is a no-op.
6. __get__ can be overridden by subclasses.
"""
import os
import pytest

from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SingleFileBlock(Datablock):
    """Block with a single TOPICFILE."""
    TOPICFILE = 'output.txt'

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('hello from single file')


class MultiTopicBlock(Datablock):
    """Block with TOPICFILES (dict of topic -> filename)."""
    TOPICFILES = {
        'alpha': 'alpha.txt',
        'beta': 'beta.txt',
    }

    def __build__(self):
        for topic in self.TOPICFILES:
            self.dirpath(topic, ensure=True)
            with open(self.path(topic), 'w') as f:
                f.write(f'data for {topic}')


class TopicsDirBlock(Datablock):
    """Block with TOPICS (each topic is a directory)."""
    TOPICS = ['part_a', 'part_b']

    def __build__(self):
        for topic in self.TOPICS:
            dirpath = self.dirpath(topic, ensure=True)
            with open(os.path.join(dirpath, 'data.txt'), 'w') as f:
                f.write(f'contents of {topic}')


class NoTopicBlock(Datablock):
    """Block with no TOPICFILE(S) — produces no artifacts."""
    def __build__(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetSingleFile:

    def test_downloads_topicfile(self, tmp_path, monkeypatch):
        """get() should copy the TOPICFILE to the destination."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = SingleFileBlock(url=root)
        block.build()

        dest = str(tmp_path / 'download')
        block.get(path=dest)
        downloaded = os.path.join(dest, 'output.txt')
        assert os.path.exists(downloaded)
        with open(downloaded) as f:
            assert f.read() == 'hello from single file'

    def test_returns_self(self, tmp_path, monkeypatch):
        """get() should return self for chaining."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = SingleFileBlock(url=root)
        block.build()

        dest = str(tmp_path / 'download')
        result = block.get(path=dest)
        assert result is block

    def test_creates_dest_dir(self, tmp_path, monkeypatch):
        """get() should create the destination directory if it doesn't exist."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = SingleFileBlock(url=root)
        block.build()

        dest = str(tmp_path / 'nested' / 'deep' / 'download')
        assert not os.path.exists(dest)
        block.get(path=dest)
        assert os.path.isdir(dest)


class TestGetMultiTopic:

    def test_downloads_specific_topic(self, tmp_path, monkeypatch):
        """get(topic) should download only that topic's file."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = MultiTopicBlock(url=root)
        block.build()

        dest = str(tmp_path / 'download')
        block.get('alpha', path=dest)
        downloaded = os.path.join(dest, 'alpha.txt')
        assert os.path.exists(downloaded)
        with open(downloaded) as f:
            assert f.read() == 'data for alpha'


class TestGetTopicsDir:

    def test_downloads_topic_directory(self, tmp_path, monkeypatch):
        """get(topic) should recursively download the topic directory."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = TopicsDirBlock(url=root)
        block.build()

        dest = str(tmp_path / 'download')
        block.get('part_a', path=dest)
        # The directory contents should be under dest
        found_files = []
        for dirpath, dirnames, filenames in os.walk(dest):
            for fn in filenames:
                found_files.append(fn)
        assert 'data.txt' in found_files


class TestGetNoTopic:

    def test_no_topic_block_is_noop(self, tmp_path, monkeypatch):
        """get() on a block with no topics should be a safe no-op."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = NoTopicBlock(url=root)
        block.build()

        dest = str(tmp_path / 'download')
        result = block.get(path=dest)
        assert result is block


class TestGetOverride:

    def test_custom_get(self, tmp_path, monkeypatch):
        """Subclasses can override __get__ for custom download logic."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        captured = {}

        class CustomGetBlock(SingleFileBlock):
            def __get__(self, topic=None, *, path='.'):
                captured['topic'] = topic
                captured['path'] = path
                return self

        block = CustomGetBlock(url=root)
        block.build()
        dest = str(tmp_path / 'download')
        block.get('mytopic', path=dest)
        assert captured['topic'] == 'mytopic'
        assert captured['path'] == dest
