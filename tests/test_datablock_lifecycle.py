"""
Lifecycle tests for Datablock: .build(), .leave_breadcrumbs(), .valid(), .UNSAFE_clear().

Verifies the canonical lifecycle:
    1. A fresh Datablock starts with valid() == False
    2. build() produces output and results in valid() == True
    3. UNSAFE_clear() removes the output and results in valid() == False

Also tests:
    - leave_breadcrumbs() creates empty topic files (valid() == True)
    - build() is a no-op when already valid
    - UNSAFE_clear() with specific topics only clears those topics
    - Multi-topic Datablock lifecycle
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
    """Minimal single-topic Datablock that writes a marker file on build."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        path = self.path()
        self.dirpath(ensure=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")


class MultiTopicBlock(Datablock):
    """Multi-topic Datablock that writes a file per topic on build."""
    TOPICFILES = {'alpha': 'alpha.txt', 'beta': 'beta.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n: str = "'3'"

    def __build__(self):
        for topic in self.TOPICFILES:
            path = self.path(topic)
            self.dirpath(topic, ensure=True)
            with open(path, 'w') as f:
                f.write(f"{topic}:{self.cfg.n}")


class CountingBlock(Datablock):
    """Tracks how many times __build__ is called (to verify skip-if-valid)."""
    TOPICFILE = 'counter.txt'
    _build_count = 0

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        CountingBlock._build_count += 1
        path = self.path()
        self.dirpath(ensure=True)
        with open(path, 'w') as f:
            f.write(str(CountingBlock._build_count))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(cls, tmp_path, **kwargs):
    """Instantiate a Datablock subclass rooted in a pytest tmp_path."""
    return cls(root=str(tmp_path), **kwargs)


# ---------------------------------------------------------------------------
# 1. Initial state: valid() == False
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_single_topic_starts_invalid(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        assert block.valid() is False

    def test_multi_topic_starts_invalid(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        assert block.valid() is False


# ---------------------------------------------------------------------------
# 2. build() → valid() == True
# ---------------------------------------------------------------------------

class TestBuild:

    def test_build_makes_single_topic_valid(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        assert block.valid() is False
        block.build()
        assert block.valid() is True

    def test_build_makes_multi_topic_valid(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        assert block.valid() is False
        block.build()
        assert block.valid() is True

    def test_build_creates_topic_file(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        block.build()
        path = block.path()
        assert os.path.isfile(path)
        with open(path) as f:
            assert "built:" in f.read()

    def test_build_creates_all_topic_files(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        for topic in block.TOPICFILES:
            path = block.path(topic)
            assert os.path.isfile(path)

    def test_build_returns_self(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        result = block.build()
        assert result is block

    def test_build_is_noop_when_already_valid(self, tmp_path):
        CountingBlock._build_count = 0
        block = _make_block(CountingBlock, tmp_path)
        block.build()
        assert CountingBlock._build_count == 1
        # Second build should skip __build__
        block.build()
        assert CountingBlock._build_count == 1


# ---------------------------------------------------------------------------
# 3. UNSAFE_clear() → valid() == False
# ---------------------------------------------------------------------------

class TestUNSAFEClear:

    def test_clear_after_build_makes_invalid(self, tmp_path):
        """The core lifecycle: invalid → build → valid → clear → invalid."""
        block = _make_block(SingleTopicBlock, tmp_path)
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        block.UNSAFE_clear(OVERRIDE=True)
        assert block.valid() is False

    def test_clear_after_build_multi_topic(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        assert block.valid() is True
        block.UNSAFE_clear(OVERRIDE=True)
        assert block.valid() is False

    def test_clear_removes_topic_file(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        block.build()
        path = block.path()
        assert os.path.exists(path)
        block.UNSAFE_clear(OVERRIDE=True)
        assert not os.path.exists(path)

    def test_clear_removes_all_topic_dirs(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        for topic in block.TOPICFILES:
            assert os.path.exists(block.path(topic))
        block.UNSAFE_clear(OVERRIDE=True)
        for topic in block.TOPICFILES:
            assert not os.path.exists(block.dirpath(topic))

    def test_clear_specific_topic(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        # Clear only 'alpha', 'beta' should remain
        block.UNSAFE_clear('alpha', OVERRIDE=True)
        assert not os.path.exists(block.dirpath('alpha'))
        assert os.path.exists(block.path('beta'))

    def test_clear_returns_self(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        block.build()
        result = block.UNSAFE_clear(OVERRIDE=True)
        assert result is block

    def test_rebuild_after_clear(self, tmp_path):
        """After clearing, build() should work again."""
        block = _make_block(SingleTopicBlock, tmp_path)
        block.build()
        assert block.valid() is True
        block.UNSAFE_clear(OVERRIDE=True)
        assert block.valid() is False
        block.build()
        assert block.valid() is True


# ---------------------------------------------------------------------------
# 4. leave_breadcrumbs()
# ---------------------------------------------------------------------------

class TestLeaveBreadcrumbs:

    def test_breadcrumbs_make_single_topic_valid(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        assert block.valid() is False
        block.leave_breadcrumbs()
        assert block.valid() is True

    def test_breadcrumbs_make_multi_topic_valid(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        assert block.valid() is False
        block.leave_breadcrumbs()
        assert block.valid() is True

    def test_breadcrumbs_create_empty_files(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        block.leave_breadcrumbs()
        path = block.path()
        assert os.path.isfile(path)
        assert os.path.getsize(path) == 0

    def test_breadcrumbs_create_all_topic_files(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.leave_breadcrumbs()
        for topic in block.TOPICFILES:
            path = block.path(topic)
            assert os.path.isfile(path)
            assert os.path.getsize(path) == 0

    def test_breadcrumbs_returns_self(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        result = block.leave_breadcrumbs()
        assert result is block

    def test_clear_after_breadcrumbs(self, tmp_path):
        """Breadcrumbs should be clearable just like built artifacts."""
        block = _make_block(SingleTopicBlock, tmp_path)
        block.leave_breadcrumbs()
        assert block.valid() is True
        block.UNSAFE_clear(OVERRIDE=True)
        assert block.valid() is False


# ---------------------------------------------------------------------------
# 5. Full lifecycle: invalid → build → valid → clear → invalid → rebuild
# ---------------------------------------------------------------------------

class TestFullLifecycle:

    def test_single_topic_full_cycle(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        # Step 1: starts invalid
        assert block.valid() is False
        # Step 2: build makes it valid
        block.build()
        assert block.valid() is True
        assert os.path.isfile(block.path())
        # Step 3: clear makes it invalid
        block.UNSAFE_clear(OVERRIDE=True)
        assert block.valid() is False
        # Step 4: rebuild
        block.build()
        assert block.valid() is True

    def test_multi_topic_full_cycle(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        block.UNSAFE_clear(OVERRIDE=True)
        assert block.valid() is False
        block.build()
        assert block.valid() is True
