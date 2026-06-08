"""Tests that UNSAFE_clear() handles None paths as a no-op.

When TOPICFILES[topic] = None the artifact is a directory, not a file.
path(topic) returns None in that case.  clear_path(None) must be a
silent no-op — not an AttributeError on NoneType.startswith().

Also covers mixed TOPICFILES where some topics have files and some are
directory-only (path → None).
"""
import os
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample Datablock subclasses
# ---------------------------------------------------------------------------

class AllDirTopics(Datablock):
    """Every topic is a directory (TOPICFILES value is None).
    path(topic) returns None for all topics.
    """
    TOPICFILES = {'images': None, 'masks': None}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        for topic in self.TOPICFILES:
            dirpath = self.dirpath(topic, ensure=True)
            with open(os.path.join(dirpath, 'data.bin'), 'wb') as f:
                f.write(b'\x00' * 16)

    def validtopic(self, topic=None):
        if topic is None:
            return all(self.validtopic(t) for t in self.TOPICFILES)
        d = self.dirpath(topic)
        return os.path.isdir(d) and bool(os.listdir(d))

    def valid(self, topic=None):
        if topic is not None:
            return self.validtopic(topic)
        return all(self.validtopic(t) for t in self.TOPICFILES)


class MixedTopics(Datablock):
    """Some topics are files, some are directories (None).
    path('logs') returns a filepath; path('checkpoints') returns None.
    """
    TOPICFILES = {'logs': 'train.log', 'checkpoints': None}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        # logs topic: write a file
        self.dirpath('logs', ensure=True)
        with open(self.path('logs'), 'w') as f:
            f.write('epoch=1\n')
        # checkpoints topic: write files in a directory
        ckpt_dir = self.dirpath('checkpoints', ensure=True)
        with open(os.path.join(ckpt_dir, 'model.pt'), 'wb') as f:
            f.write(b'\x01' * 32)

    def validtopic(self, topic=None):
        if topic is None:
            return all(self.validtopic(t) for t in self.TOPICFILES)
        if self.TOPICFILES[topic] is not None:
            p = self.path(topic)
            return p is not None and os.path.isfile(p)
        else:
            d = self.dirpath(topic)
            return os.path.isdir(d) and bool(os.listdir(d))

    def valid(self, topic=None):
        if topic is not None:
            return self.validtopic(topic)
        return all(self.validtopic(t) for t in self.TOPICFILES)


class NoTopicfileBlock(Datablock):
    """No TOPICFILE or TOPICFILES at all — path() returns None."""

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        self.dirpath(ensure=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(cls, tmp_path, **kwargs):
    return cls(url=str(tmp_path), **kwargs)


# ---------------------------------------------------------------------------
# Tests: all-directory topics (every path(topic) is None)
# ---------------------------------------------------------------------------

class TestClearAllDirTopics:
    """TOPICFILES = {'images': None, 'masks': None} — path() always None."""

    def test_path_returns_none_for_dir_topics(self, tmp_path):
        block = _make_block(AllDirTopics, tmp_path)
        for topic in block.TOPICFILES:
            assert block.path(topic) is None

    def test_clear_all_does_not_raise(self, tmp_path):
        """UNSAFE_clear() must not crash when path(topic) is None."""
        block = _make_block(AllDirTopics, tmp_path)
        block.__build__()
        assert block.valid()
        # This used to raise: AttributeError: 'NoneType' has no attribute 'startswith'
        block.UNSAFE_clear(OVERRIDE=True)
        # Should return self
        assert isinstance(block.UNSAFE_clear(OVERRIDE=True), AllDirTopics)

    def test_clear_specific_dir_topic_does_not_raise(self, tmp_path):
        """UNSAFE_clear('images') must not crash when path('images') is None."""
        block = _make_block(AllDirTopics, tmp_path)
        block.__build__()
        assert block.valid()
        block.UNSAFE_clear('images', OVERRIDE=True)

    def test_clear_with_clear_dirpath_removes_dirs(self, tmp_path):
        """clear_dirpath=True should remove the actual topic directories."""
        block = _make_block(AllDirTopics, tmp_path)
        block.__build__()
        assert block.valid()
        block.UNSAFE_clear(OVERRIDE=True, clear_dirpath=True)
        for topic in block.TOPICFILES:
            assert not os.path.exists(block.dirpath(topic))

    def test_clear_returns_self(self, tmp_path):
        block = _make_block(AllDirTopics, tmp_path)
        block.__build__()
        result = block.UNSAFE_clear(OVERRIDE=True)
        assert result is block


# ---------------------------------------------------------------------------
# Tests: mixed topics (some paths None, some not)
# ---------------------------------------------------------------------------

class TestClearMixedTopics:
    """TOPICFILES = {'logs': 'train.log', 'checkpoints': None}."""

    def test_path_returns_none_only_for_dir_topic(self, tmp_path):
        block = _make_block(MixedTopics, tmp_path)
        assert block.path('logs') is not None
        assert block.path('checkpoints') is None

    def test_clear_all_does_not_raise(self, tmp_path):
        """Clearing all topics must handle the None-path topic gracefully."""
        block = _make_block(MixedTopics, tmp_path)
        block.__build__()
        assert block.valid()
        block.UNSAFE_clear(OVERRIDE=True)

    def test_clear_file_topic_removes_file(self, tmp_path):
        """Clearing the file-based topic should remove its file."""
        block = _make_block(MixedTopics, tmp_path)
        block.__build__()
        log_path = block.path('logs')
        assert os.path.isfile(log_path)
        block.UNSAFE_clear('logs', OVERRIDE=True)
        assert not os.path.exists(log_path)

    def test_clear_dir_topic_does_not_raise(self, tmp_path):
        """Clearing the dir-based topic (path=None) must not crash."""
        block = _make_block(MixedTopics, tmp_path)
        block.__build__()
        assert block.valid('checkpoints')
        block.UNSAFE_clear('checkpoints', OVERRIDE=True)

    def test_clear_dir_topic_with_clear_dirpath(self, tmp_path):
        """clear_dirpath=True on the dir-based topic removes its directory."""
        block = _make_block(MixedTopics, tmp_path)
        block.__build__()
        ckpt_dir = block.dirpath('checkpoints')
        assert os.path.isdir(ckpt_dir)
        block.UNSAFE_clear('checkpoints', OVERRIDE=True, clear_dirpath=True)
        assert not os.path.exists(ckpt_dir)
        # logs topic should be untouched
        assert os.path.isfile(block.path('logs'))

    def test_rebuild_after_clear(self, tmp_path):
        """Full cycle: build → clear → rebuild."""
        block = _make_block(MixedTopics, tmp_path)
        block.__build__()
        assert block.valid()
        block.UNSAFE_clear(OVERRIDE=True, clear_dirpath=True)
        block.__build__()
        assert block.valid()


# ---------------------------------------------------------------------------
# Tests: no TOPICFILE at all (path() returns None)
# ---------------------------------------------------------------------------

class TestClearNoTopicfile:
    """Datablock with no TOPICFILE/TOPICFILES — path() is None."""

    def test_path_returns_none(self, tmp_path):
        block = _make_block(NoTopicfileBlock, tmp_path)
        assert block.path() is None

    def test_clear_does_not_raise(self, tmp_path):
        """UNSAFE_clear on a block with no TOPICFILE should be a no-op."""
        block = _make_block(NoTopicfileBlock, tmp_path)
        block.__build__()
        # path() is None here — clear_path(None) must not crash
        block.UNSAFE_clear(OVERRIDE=True)

    def test_clear_returns_self(self, tmp_path):
        block = _make_block(NoTopicfileBlock, tmp_path)
        result = block.UNSAFE_clear(OVERRIDE=True)
        assert result is block
