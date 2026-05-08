"""
Tests for Datablock.ls(topic).

Verifies:
1. ls() on a single-topic block lists the containing directory.
2. ls(topic) on a multi-topic block lists each topic directory.
3. ls() returns [] when no path exists (pre-build).
4. ls(detail=True) returns dicts with 'name', 'size', 'type' keys.
5. ls() on a TOPICS-only block works with overridden path().
6. ls() on a dir-valued topic (TOPICFILES[topic]=None) lists dir contents.
7. ls() on a no-topic block returns [].
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

class SingleTopicBlock(Datablock):
    """Minimal single-topic Datablock."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write(f"built:{self.cfg.label}")


class MultiTopicBlock(Datablock):
    """Multi-topic Datablock."""
    TOPICFILES = {'alpha': 'alpha.txt', 'beta': 'beta.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n: str = "'3'"

    def __build__(self):
        for topic in self.TOPICFILES:
            self.dirpath(topic, ensure=True)
            with open(self.path(topic), 'w') as f:
                f.write(f"{topic}:{self.cfg.n}")


class DirTopicBlock(Datablock):
    """TOPICFILES[topic]=None — artifact is a directory."""
    TOPICFILES = {'images': None, 'masks': None}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        for topic in self.TOPICFILES:
            dirpath = self.dirpath(topic, ensure=True)
            for i in range(3):
                with open(os.path.join(dirpath, f'file_{i}.bin'), 'wb') as f:
                    f.write(b'\x00' * 16)


class TopicsBlock(Datablock):
    """TOPICS-only block with custom path()."""
    TOPICS = ['frames', 'poses']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def path(self, topic=None, *, ensure_dirpath=False):
        kp = self.anchorkeypath
        if ensure_dirpath:
            os.makedirs(kp, exist_ok=True)
        if topic is None:
            return kp
        elif topic == 'frames':
            return os.path.join(kp, 'data', 'frames.pt')
        elif topic == 'poses':
            return os.path.join(kp, 'data', 'poses.json')
        else:
            raise ValueError(f"Unknown topic: {topic}")

    def __build__(self):
        for topic in self.TOPICS:
            p = self.path(topic)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                f.write(f"built:{topic}")


class NoTopicBlock(Datablock):
    """Datablock with no TOPICFILE or TOPICFILES."""

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(cls, tmp_path, **kwargs):
    return cls(url=str(tmp_path), **kwargs)


# ---------------------------------------------------------------------------
# 1. Single-topic ls()
# ---------------------------------------------------------------------------

class TestLsSingleTopic:

    def test_ls_empty_before_build(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        result = block.ls()
        assert result == []

    def test_ls_after_build(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        block.build()
        result = block.ls()
        # path() points to the file; ls() should list its parent dir
        assert len(result) >= 1
        basenames = [os.path.basename(p) for p in result]
        assert 'output.txt' in basenames

    def test_ls_returns_full_paths(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        block.build()
        result = block.ls()
        for p in result:
            assert os.path.isabs(p)


# ---------------------------------------------------------------------------
# 2. Multi-topic ls(topic)
# ---------------------------------------------------------------------------

class TestLsMultiTopic:

    def test_ls_topic_empty_before_build(self, tmp_path):
        block = _make(MultiTopicBlock, tmp_path)
        result = block.ls('alpha')
        assert result == []

    def test_ls_topic_after_build(self, tmp_path):
        block = _make(MultiTopicBlock, tmp_path)
        block.build()
        result = block.ls('alpha')
        basenames = [os.path.basename(p) for p in result]
        assert 'alpha.txt' in basenames

    def test_ls_each_topic(self, tmp_path):
        block = _make(MultiTopicBlock, tmp_path)
        block.build()
        for topic, filename in block.TOPICFILES.items():
            result = block.ls(topic)
            basenames = [os.path.basename(p) for p in result]
            assert filename in basenames

    def test_ls_no_topic_lists_anchorkeypath(self, tmp_path):
        """ls() without topic on a multi-topic block lists anchorkeypath."""
        block = _make(MultiTopicBlock, tmp_path)
        block.build()
        result = block.ls()
        # anchorkeypath contains subdirectories for each topic
        basenames = [os.path.basename(p) for p in result]
        assert 'alpha' in basenames
        assert 'beta' in basenames


# ---------------------------------------------------------------------------
# 3. detail=True
# ---------------------------------------------------------------------------

class TestLsDetail:

    def test_ls_detail_returns_dicts(self, tmp_path):
        block = _make(SingleTopicBlock, tmp_path)
        block.build()
        result = block.ls(detail=True)
        assert len(result) >= 1
        for entry in result:
            assert isinstance(entry, dict)
            assert 'name' in entry

    def test_ls_detail_topic(self, tmp_path):
        block = _make(MultiTopicBlock, tmp_path)
        block.build()
        result = block.ls('alpha', detail=True)
        assert len(result) >= 1
        names = [os.path.basename(e['name']) for e in result]
        assert 'alpha.txt' in names


# ---------------------------------------------------------------------------
# 4. TOPICS-only block
# ---------------------------------------------------------------------------

class TestLsTopicsOnly:

    def test_ls_topic_after_build(self, tmp_path):
        block = _make(TopicsBlock, tmp_path)
        block.build()
        result = block.ls('frames')
        basenames = [os.path.basename(p) for p in result]
        assert 'frames.pt' in basenames

    def test_ls_topic_poses(self, tmp_path):
        block = _make(TopicsBlock, tmp_path)
        block.build()
        result = block.ls('poses')
        basenames = [os.path.basename(p) for p in result]
        assert 'poses.json' in basenames


# ---------------------------------------------------------------------------
# 5. Dir-valued topics (TOPICFILES[topic]=None)
# ---------------------------------------------------------------------------

class TestLsDirTopic:

    def test_ls_dir_topic_lists_contents(self, tmp_path):
        block = _make(DirTopicBlock, tmp_path)
        block.__build__()
        result = block.ls('images')
        assert len(result) == 3
        basenames = sorted(os.path.basename(p) for p in result)
        assert basenames == ['file_0.bin', 'file_1.bin', 'file_2.bin']

    def test_ls_dir_topic_masks(self, tmp_path):
        block = _make(DirTopicBlock, tmp_path)
        block.__build__()
        result = block.ls('masks')
        assert len(result) == 3

    def test_ls_dir_topic_detail(self, tmp_path):
        block = _make(DirTopicBlock, tmp_path)
        block.__build__()
        result = block.ls('images', detail=True)
        assert len(result) == 3
        for entry in result:
            assert isinstance(entry, dict)
            assert entry['size'] == 16


# ---------------------------------------------------------------------------
# 6. No-topic block
# ---------------------------------------------------------------------------

class TestLsNoTopic:

    def test_ls_returns_empty_list(self, tmp_path):
        """A block with no TOPICFILE/TOPICFILES has path()=None → ls() returns []."""
        block = _make(NoTopicBlock, tmp_path)
        assert block.ls() == []
