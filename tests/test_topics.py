"""
Tests for the TOPICS class attribute.

TOPICS is a list of topic names (no filenames) for Datablock subclasses
that override path() to provide fully custom path generation.

Verifies:
1. has_topics() returns True for TOPICS-only classes.
2. topics() returns the TOPICS list.
3. path() override works (no TOPICFILES required).
4. dirpath(topic) derives from path(topic) for TOPICS-only classes.
5. valid() / validpaths() work with overridden path().
6. build/read round-trip works.
7. hashstr includes TOPICS in the hash.
8. leave_breadcrumbs() raises NotImplementedError for TOPICS-only.
9. UNSAFE_copy_from() raises NotImplementedError for TOPICS-only.
10. keyby options work with TOPICS.
11. Wrapper (datablock()) lifts TOPICS correctly.
12. Pickle serialization works.
"""
import os
import pickle
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock
from dbx.datawraps import datablock, Datablockable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_topics')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample subclass: TOPICS with custom path()
# ---------------------------------------------------------------------------

class TopicsBlock(Datablock):
    """Datablock with TOPICS (no TOPICFILES) and a custom path()."""
    TOPICS = ['frames', 'poses']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'default'"

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
        return self

    def __read__(self, topic=None):
        with open(self.path(topic), 'r') as f:
            return f.read()


class SingleTopicBlock(Datablock):
    """Datablock with one topic in TOPICS and a custom path()."""
    TOPICS = ['output']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def path(self, topic=None, *, ensure_dirpath=False):
        kp = self.anchorkeypath
        if topic is None:
            return kp
        elif topic == 'output':
            return os.path.join(kp, 'result.txt')
        else:
            raise ValueError(f"Unknown topic: {topic}")

    def __build__(self):
        p = self.path('output')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            f.write("output_data")
        return self

    def __read__(self, topic=None):
        with open(self.path(topic), 'r') as f:
            return f.read()


# ---------------------------------------------------------------------------
# Datablockable with TOPICS (for wrapper tests)
# ---------------------------------------------------------------------------

class TopicsDatblockable:
    """Datablockable with TOPICS for wrapper tests."""
    TOPICS = ['frames', 'poses']

    @dataclass
    class CONFIG:
        label: str = 'default'

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def path(self, topic=None, *, ensure_dirpath=False):
        kp = self.anchorkeypath
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
        return self

    def __read__(self, topic=None):
        with open(self.path(topic), 'r') as f:
            return f.read()


# ---------------------------------------------------------------------------
# 1. has_topics() / topics()
# ---------------------------------------------------------------------------

class TestTopicsDiscovery:

    def test_has_topics_true(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert block.has_topics() is True

    def test_has_topic_false(self):
        """TOPICS-only blocks should not report has_topic()."""
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert block.has_topic() is False

    def test_topics_returns_list(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert block.topics() == ['frames', 'poses']

    def test_no_topicfiles_attr(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert not hasattr(block, 'TOPICFILES')

    def test_no_topicfile_attr(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert not hasattr(block, 'TOPICFILE')


# ---------------------------------------------------------------------------
# 2. Custom path()
# ---------------------------------------------------------------------------

class TestTopicsPath:

    def test_path_none_returns_keypath(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert block.path() == block.anchorkeypath

    def test_path_frames(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        p = block.path('frames')
        assert p.endswith('data/frames.pt')
        assert block.anchorkeypath in p

    def test_path_poses(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        p = block.path('poses')
        assert p.endswith('data/poses.json')
        assert block.anchorkeypath in p

    def test_path_unknown_topic_raises(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        with pytest.raises(ValueError, match="Unknown topic"):
            block.path('nonexistent')


# ---------------------------------------------------------------------------
# 3. dirpath(topic) for TOPICS-only
# ---------------------------------------------------------------------------

class TestTopicsDirpath:

    def test_dirpath_none_returns_keypath(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert block.dirpath() == block.anchorkeypath

    def test_dirpath_topic_derives_from_path(self):
        """For TOPICS-only, dirpath(topic) should be dirname of path(topic)."""
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        p = block.path('frames')
        assert block.dirpath('frames') == os.path.dirname(p)

    def test_dirpath_topic_poses(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        p = block.path('poses')
        assert block.dirpath('poses') == os.path.dirname(p)


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------

class TestTopicsValidation:

    def test_valid_false_before_build(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert block.valid() is False

    def test_valid_true_after_build(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path))
        block.build()
        assert block.valid() is True

    def test_valid_single_topic(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path))
        block.build()
        assert block.valid('frames') is True
        assert block.valid('poses') is True

    def test_validpaths(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path))
        block.build()
        vp = block.validpaths()
        assert isinstance(vp, dict)
        assert vp['frames'] is True
        assert vp['poses'] is True

    def test_validpaths_before_build(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        vp = block.validpaths()
        assert isinstance(vp, dict)
        assert vp['frames'] is False
        assert vp['poses'] is False


# ---------------------------------------------------------------------------
# 5. Build + read round-trip
# ---------------------------------------------------------------------------

class TestTopicsBuildRead:

    def test_build_creates_files(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path))
        block.build()
        assert os.path.exists(block.path('frames'))
        assert os.path.exists(block.path('poses'))

    def test_read_frames(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path))
        block.build()
        assert block.read('frames') == 'built:frames'

    def test_read_poses(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path))
        block.build()
        assert block.read('poses') == 'built:poses'

    def test_read_invalid_topic_raises(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path))
        block.build()
        with pytest.raises(ValueError, match="not in"):
            block.read('nonexistent')

    def test_single_topic_build_read(self, tmp_path):
        block = SingleTopicBlock(root=str(tmp_path))
        block.build()
        assert block.valid() is True
        assert block.read('output') == 'output_data'


# ---------------------------------------------------------------------------
# 6. hashstr includes TOPICS
# ---------------------------------------------------------------------------

class TestTopicsHash:

    def test_hashstr_includes_topics(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert 'topic:frames' in block.hashstr
        assert 'topic:poses' in block.hashstr

    def test_hashstr_no_equals(self):
        """TOPICS hashstr entries should NOT have '=' (no filename)."""
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        for part in block.hashstr.split(os.sep):
            if part.startswith('topic:'):
                assert '=' not in part

    def test_different_topics_different_hash(self):
        """Blocks with different TOPICS should have different hashes."""
        block1 = TopicsBlock(root='/tmp/dbx_test_topics')
        block2 = SingleTopicBlock(root='/tmp/dbx_test_topics')
        assert block1.hash != block2.hash


# ---------------------------------------------------------------------------
# 7. leave_breadcrumbs raises
# ---------------------------------------------------------------------------

class TestTopicsBreadcrumbs:

    def test_leave_breadcrumbs_raises(self, tmp_path):
        """TOPICS-only blocks should raise NotImplementedError from leave_breadcrumbs."""
        block = TopicsBlock(root=str(tmp_path))
        with pytest.raises(NotImplementedError, match="leave_breadcrumbs"):
            block.leave_breadcrumbs()


# ---------------------------------------------------------------------------
# 8. UNSAFE_copy_from raises
# ---------------------------------------------------------------------------

class TestTopicsCopyFrom:

    def test_unsafe_copy_from_raises(self, tmp_path):
        """TOPICS-only blocks should raise NotImplementedError from UNSAFE_copy_from."""
        block = TopicsBlock(root=str(tmp_path))
        src = str(tmp_path / 'src')
        os.makedirs(src, exist_ok=True)
        with pytest.raises(NotImplementedError, match="UNSAFE_copy_from"):
            block.UNSAFE_copy_from(src)


# ---------------------------------------------------------------------------
# 9. keyby works with TOPICS
# ---------------------------------------------------------------------------

class TestTopicsKeyby:

    def test_keyby_hash(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path), keyby='hash')
        block.build()
        assert block.valid() is True
        assert block.hash in block.path('frames')

    def test_keyby_tag(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path), keyby='tag', tag='v1')
        block.build()
        assert block.valid() is True
        assert 'v1' in block.path('frames')

    def test_keyby_taghash(self, tmp_path):
        block = TopicsBlock(root=str(tmp_path), keyby='taghash', tag='exp1')
        block.build()
        assert block.valid() is True
        assert 'exp1' in block.path('frames')
        assert block.hash[:8] in block.path('frames')


# ---------------------------------------------------------------------------
# 10. bid fields
# ---------------------------------------------------------------------------

class TestTopicsBid:

    def test_bid_keyby(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        assert block.bid.keyby == 'hash'

    def test_bid_tag(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics', tag='t1')
        assert block.bid.tag == 't1'


# ---------------------------------------------------------------------------
# 11. Pickle serialization
# ---------------------------------------------------------------------------

class TestTopicsSerialization:

    def test_pickle_round_trip(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.topics() == ['frames', 'poses']
        assert restored.has_topics() is True
        assert restored.hash == block.hash

    def test_pickle_preserves_paths(self):
        block = TopicsBlock(root='/tmp/dbx_test_topics')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.path('frames') == block.path('frames')
        assert restored.path('poses') == block.path('poses')


# ---------------------------------------------------------------------------
# 12. Wrapper (datablock()) with TOPICS
# ---------------------------------------------------------------------------

class TestTopicsWrapper:

    def test_wrapper_lifts_topics(self):
        Wrapped = datablock(TopicsDatblockable)
        block = Wrapped(root='/tmp/dbx_test_topics')
        assert hasattr(block, 'TOPICS')
        assert block.TOPICS == ['frames', 'poses']

    def test_wrapper_has_topics(self):
        Wrapped = datablock(TopicsDatblockable)
        block = Wrapped(root='/tmp/dbx_test_topics')
        assert block.has_topics() is True

    def test_wrapper_topics_list(self):
        Wrapped = datablock(TopicsDatblockable)
        block = Wrapped(root='/tmp/dbx_test_topics')
        assert block.topics() == ['frames', 'poses']

    def test_wrapper_build_read(self, tmp_path):
        Wrapped = datablock(TopicsDatblockable)
        block = Wrapped(root=str(tmp_path))
        block.build()
        assert block.valid() is True
        assert block.read('frames') == 'built:frames'
        assert block.read('poses') == 'built:poses'

    def test_wrapper_no_topicfiles(self):
        Wrapped = datablock(TopicsDatblockable)
        block = Wrapped(root='/tmp/dbx_test_topics')
        assert not hasattr(TopicsDatblockable, 'TOPICFILES') or TopicsDatblockable.TOPICFILES == {}
