"""
Tests for Datablock.ls(topic).

Verifies:
1. ls() on a single-topic block lists the containing directory.
2. ls(topic) on a multi-topic block lists each topic directory.
3. ls() returns [] when no path exists (pre-build).
4. ls(detail=True) returns dicts with 'name', 'size', 'type' keys.
5. ls() on a TOPICS-only block works with overridden path().
6. ls() on a dir-valued topic (TOPICS[topic]=None) lists dir contents.
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
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        path = self.path(ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")


class MultiTopicBlock(Datablock):
    """Multi-topic Datablock."""
    TOPICS = {'alpha': 'alpha.txt', 'beta': 'beta.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n: str = "'3'"

    def __build__(self):
        for topic in self.TOPICS:
            self.dirpath(topic, ensure=True)
            with open(self.path(topic), 'w') as f:
                f.write(f"{topic}:{self.cfg.n}")


class DirTopicBlock(Datablock):
    """TOPICS[topic]=None — artifact is a directory."""
    TOPICS = {'images': None, 'masks': None}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        for topic in self.TOPICS:
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
    """Datablock with no TOPICS or TOPICS."""

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
        for topic, filename in block.TOPICS.items():
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
# 5. Dir-valued topics (TOPICS[topic]=None)
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
        """A block with no TOPICS/TOPICS has path()=None → ls() returns []."""
        block = _make(NoTopicBlock, tmp_path)
        assert block.ls() == []


# ---------------------------------------------------------------------------
# 7. List-TOPICS directory mode (no custom path() override)
# ---------------------------------------------------------------------------

class ListTopicsDirBlock(Datablock):
    """TOPICS as a list — each topic is a directory.

    Unlike TopicsBlock above, this does NOT override path(), so it
    exercises the default path() → dirpath() flow for list-TOPICS.
    """
    TOPICS = ['gold_table', 'cell_catalog', 'cell_assignments']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        for topic in self.TOPICS:
            d = self.dirpath(topic, ensure=True)
            for i in range(2):
                with open(os.path.join(d, f'part_{i}.parquet'), 'w') as f:
                    f.write(f'{topic}:part_{i}')


class TestLsListTopicsDir:
    """ls(topic) on list-TOPICS must list only that topic's directory."""

    def test_ls_topic_lists_only_requested_topic(self, tmp_path):
        """ls('cell_assignments') must NOT return sibling topics."""
        block = _make(ListTopicsDirBlock, tmp_path)
        block.__build__()
        result = block.ls('cell_assignments')
        basenames = [os.path.basename(p) for p in result]
        # Should contain the files inside cell_assignments, not sibling dirs
        assert 'part_0.parquet' in basenames
        assert 'part_1.parquet' in basenames
        # Must NOT contain sibling topic directories
        assert 'gold_table' not in basenames
        assert 'cell_catalog' not in basenames

    def test_ls_each_topic_is_isolated(self, tmp_path):
        """Each topic's ls() should return only its own contents."""
        block = _make(ListTopicsDirBlock, tmp_path)
        block.__build__()
        for topic in block.TOPICS:
            result = block.ls(topic)
            assert len(result) == 2, f"Expected 2 files in {topic}, got {len(result)}"
            basenames = [os.path.basename(p) for p in result]
            assert 'part_0.parquet' in basenames
            assert 'part_1.parquet' in basenames

    def test_ls_no_topic_lists_all_topics(self, tmp_path):
        """ls() with no topic should list the anchorkeypath (all topic dirs)."""
        block = _make(ListTopicsDirBlock, tmp_path)
        block.__build__()
        result = block.ls()
        basenames = [os.path.basename(p) for p in result]
        for topic in block.TOPICS:
            assert topic in basenames

    def test_ls_topic_detail(self, tmp_path):
        """ls(topic, detail=True) should return dicts for topic contents."""
        block = _make(ListTopicsDirBlock, tmp_path)
        block.__build__()
        result = block.ls('gold_table', detail=True)
        assert len(result) == 2
        for entry in result:
            assert isinstance(entry, dict)
            assert 'name' in entry


# ---------------------------------------------------------------------------
# 8. Dict-TOPICS with None values (directory topics)
# ---------------------------------------------------------------------------

class DictTopicsNoneBlock(Datablock):
    """TOPICS as a dict where all values are None — each topic is a directory."""
    TOPICS = {
        'gold_table': None,
        'cell_catalog': None,
        'cell_assignments': None,
    }

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        for topic in self.TOPICS:
            d = self.dirpath(topic, ensure=True)
            for i in range(2):
                with open(os.path.join(d, f'part_{i}.parquet'), 'w') as f:
                    f.write(f'{topic}:part_{i}')


class DictTopicsMixedNoneBlock(Datablock):
    """TOPICS as a dict with mixed file and None (directory) values."""
    TOPICS = {
        'summary': 'summary.json',
        'cell_assignments': None,
    }

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        # File topic
        self.dirpath('summary', ensure=True)
        with open(self.path('summary'), 'w') as f:
            f.write('{"status": "ok"}')
        # Directory topic
        d = self.dirpath('cell_assignments', ensure=True)
        for i in range(3):
            with open(os.path.join(d, f'shard_{i}.parquet'), 'w') as f:
                f.write(f'shard_{i}')


class TestLsDictTopicsNone:
    """ls(topic) on dict-TOPICS with None values must list that topic dir."""

    def test_ls_topic_lists_only_requested_topic(self, tmp_path):
        """ls('cell_assignments') must NOT return sibling topics."""
        block = _make(DictTopicsNoneBlock, tmp_path)
        block.__build__()
        result = block.ls('cell_assignments')
        basenames = [os.path.basename(p) for p in result]
        assert 'part_0.parquet' in basenames
        assert 'part_1.parquet' in basenames
        assert 'gold_table' not in basenames
        assert 'cell_catalog' not in basenames

    def test_ls_each_topic_is_isolated(self, tmp_path):
        """Each topic's ls() should return only its own contents."""
        block = _make(DictTopicsNoneBlock, tmp_path)
        block.__build__()
        for topic in block.TOPICS:
            result = block.ls(topic)
            assert len(result) == 2, f"Expected 2 files in {topic}, got {len(result)}"

    def test_ls_mixed_file_topic(self, tmp_path):
        """ls() on a file-valued topic should list the containing dir."""
        block = _make(DictTopicsMixedNoneBlock, tmp_path)
        block.__build__()
        result = block.ls('summary')
        basenames = [os.path.basename(p) for p in result]
        assert 'summary.json' in basenames

    def test_ls_mixed_dir_topic(self, tmp_path):
        """ls() on a None-valued topic should list the topic dir contents."""
        block = _make(DictTopicsMixedNoneBlock, tmp_path)
        block.__build__()
        result = block.ls('cell_assignments')
        basenames = [os.path.basename(p) for p in result]
        assert len(result) == 3
        assert 'shard_0.parquet' in basenames
        # Must NOT contain sibling topic
        assert 'summary' not in basenames
