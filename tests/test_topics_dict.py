"""
Tests for TOPICS as a dict (dict-TOPICS mode).

When TOPICS is a dict, it should behave exactly like TOPICFILES:
- Keys are topic names, values are filenames (str) or None (directory topics).
- path(topic) returns the file path when value is a string, None when value is None.
- dirpath(topic) returns the directory path.
- valid()/validtopic() work correctly for both file and dir topics.
- UNSAFE_clear handles file vs dir topics correctly.
- hashstr uses the topic:topic=file format.
"""
import os
import pytest

from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Test blocks
# ---------------------------------------------------------------------------

class DictTopicsFileBlock(Datablock):
    """TOPICS as a dict, all values are filenames."""
    TOPICS = {
        'alpha': 'alpha.csv',
        'beta': 'beta.csv',
    }

    def __build__(self):
        for topic in self.TOPICS:
            self.dirpath(topic, ensure=True)
            with open(self.path(topic), 'w') as f:
                f.write(f'data for {topic}')


class DictTopicsMixedBlock(Datablock):
    """TOPICS as a dict with mixed file and directory values."""
    TOPICS = {
        'logs': 'train.log',
        'checkpoints': None,
    }

    def __build__(self):
        # logs topic: write a file
        self.dirpath('logs', ensure=True)
        with open(self.path('logs'), 'w') as f:
            f.write('log data')
        # checkpoints topic: write files inside a directory
        ckpt_dir = self.dirpath('checkpoints', ensure=True)
        with open(os.path.join(ckpt_dir, 'model.pt'), 'w') as f:
            f.write('checkpoint data')


class DictTopicsDirOnlyBlock(Datablock):
    """TOPICS as a dict where all values are None (all directory topics)."""
    TOPICS = {
        'images': None,
        'masks': None,
    }

    def __build__(self):
        for topic in self.TOPICS:
            d = self.dirpath(topic, ensure=True)
            with open(os.path.join(d, 'data.bin'), 'w') as f:
                f.write(f'{topic} data')


class ListTopicsBlock(Datablock):
    """TOPICS as a list — existing behavior, used for comparison."""
    TOPICS = ['part_a', 'part_b']

    def __build__(self):
        for topic in self.TOPICS:
            d = self.dirpath(topic, ensure=True)
            with open(os.path.join(d, 'data.txt'), 'w') as f:
                f.write(f'{topic} data')


# ---------------------------------------------------------------------------
# 1. Basic path/dirpath
# ---------------------------------------------------------------------------

class TestDictTopicsPaths:

    def test_path_returns_filepath_for_str_value(self, tmp_path, monkeypatch):
        """path(topic) should return a file path when TOPICS[topic] is a string."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        p = block.path('alpha')
        assert p.endswith('alpha.csv')

    def test_path_returns_none_for_none_value(self, tmp_path, monkeypatch):
        """path(topic) should return None when TOPICS[topic] is None."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsMixedBlock(url=str(tmp_path))
        p = block.path('checkpoints')
        assert p is None

    def test_dirpath_returns_directory(self, tmp_path, monkeypatch):
        """dirpath(topic) should return the topic directory."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        d = block.dirpath('alpha')
        assert d.endswith('/alpha')

    def test_topics_returns_keys(self, tmp_path, monkeypatch):
        """topics() should return the dict keys."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        assert set(block.topics()) == {'alpha', 'beta'}


# ---------------------------------------------------------------------------
# 2. Build and valid
# ---------------------------------------------------------------------------

class TestDictTopicsBuildValid:

    def test_build_and_valid_file_topics(self, tmp_path, monkeypatch):
        """Build should create files and valid() should return True."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        block.build()
        assert block.valid()
        for topic in block.TOPICS:
            assert block.validtopic(topic)
            assert os.path.isfile(block.path(topic))

    def test_build_and_valid_mixed_topics(self, tmp_path, monkeypatch):
        """Build should work with mixed file/dir topics."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsMixedBlock(url=str(tmp_path))
        block.build()
        assert block.valid()
        # File topic
        assert block.validtopic('logs')
        assert os.path.isfile(block.path('logs'))
        # Dir topic
        assert block.validtopic('checkpoints')
        assert os.path.isdir(block.dirpath('checkpoints'))

    def test_build_and_valid_dir_only_topics(self, tmp_path, monkeypatch):
        """Build should work when all topics are directories."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsDirOnlyBlock(url=str(tmp_path))
        block.build()
        assert block.valid()
        for topic in block.TOPICS:
            assert block.validtopic(topic)


# ---------------------------------------------------------------------------
# 3. UNSAFE_clear
# ---------------------------------------------------------------------------

class TestDictTopicsClear:

    def test_clear_all_file_topics(self, tmp_path, monkeypatch):
        """UNSAFE_clear should remove all file-based topics."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        block.build()
        assert block.valid()
        block.UNSAFE_clear(OVERRIDE=True)
        assert not block.valid()

    def test_clear_specific_topic(self, tmp_path, monkeypatch):
        """UNSAFE_clear('alpha') should remove only that topic."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        block.build()
        block.UNSAFE_clear('alpha', OVERRIDE=True)
        assert not block.validtopic('alpha')
        assert block.validtopic('beta')

    def test_clear_mixed_topics(self, tmp_path, monkeypatch):
        """UNSAFE_clear should handle mixed file/dir topics."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsMixedBlock(url=str(tmp_path))
        block.build()
        assert block.valid()
        block.UNSAFE_clear(OVERRIDE=True)
        assert not block.valid()


# ---------------------------------------------------------------------------
# 4. get() with dict-TOPICS
# ---------------------------------------------------------------------------

class TestDictTopicsGet:

    def test_get_file_topic(self, tmp_path, monkeypatch):
        """get(topic) should download a file topic."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = DictTopicsFileBlock(url=root)
        block.build()
        dest = str(tmp_path / 'download')
        block.get('alpha', path=dest)
        assert os.path.isfile(os.path.join(dest, 'alpha.csv'))

    def test_get_dir_topic(self, tmp_path, monkeypatch):
        """get(topic) should download a directory topic."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / 'store')
        block = DictTopicsDirOnlyBlock(url=root)
        block.build()
        dest = str(tmp_path / 'download')
        block.get('images', path=dest)
        # Should have downloaded the directory contents
        found = []
        for dirpath, _, filenames in os.walk(dest):
            found.extend(filenames)
        assert 'data.bin' in found


# ---------------------------------------------------------------------------
# 5. hashstr format
# ---------------------------------------------------------------------------

class TestDictTopicsHash:

    def test_dict_topics_hash_uses_topicfiles_format(self, tmp_path, monkeypatch):
        """Dict-TOPICS hashstr should use topic:topic=file format."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        assert 'topic:alpha=alpha.csv' in block.hashstr
        assert 'topic:beta=beta.csv' in block.hashstr

    def test_list_topics_hash_unchanged(self, tmp_path, monkeypatch):
        """List-TOPICS hashstr should still use topic:topic format (no =)."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = ListTopicsBlock(url=str(tmp_path))
        assert 'topic:part_a' in block.hashstr
        assert '=' not in block.hashstr.split('topic:part_a')[1].split('/')[0]


# ---------------------------------------------------------------------------
# 6. _topics_is_list / _topicfiles helpers
# ---------------------------------------------------------------------------

class TestTopicsHelpers:

    def test_topics_is_list_true_for_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = ListTopicsBlock(url=str(tmp_path))
        assert block._topics_is_list is True

    def test_topics_is_list_false_for_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        assert block._topics_is_list is False

    def test_topicfiles_returns_dict_topics(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = DictTopicsFileBlock(url=str(tmp_path))
        assert block._topicfiles is block.TOPICS

    def test_topicfiles_returns_none_for_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        block = ListTopicsBlock(url=str(tmp_path))
        assert block._topicfiles is None
