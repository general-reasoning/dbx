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
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")


class MultiTopicBlock(Datablock):
    """Multi-topic Datablock that writes a file per topic on build."""
    TOPICS = {'alpha': 'alpha.txt', 'beta': 'beta.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n: str = "'3'"

    def __build__(self):
        for topic in self.TOPICS:
            path = self.path(topic, ensure_dirpath=True)
            with open(path, 'w') as f:
                f.write(f"{topic}:{self.cfg.n}")




class DirTopicBlock(Datablock):
    """Datablock where TOPICS[topic]=None — artifact is a whole directory.

    path(topic) returns None (validpath → True by convention), but the
    dirpath(topic) is the real artifact.  UNSAFE_clear must remove it
    recursively.
    """
    TOPICS = {'images': None, 'masks': None}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        for topic in self.TOPICS:
            dirpath = self.dirpath(topic, ensure=True)
            # Write one or more files inside the topic directory.
            with open(os.path.join(dirpath, 'data.bin'), 'wb') as f:
                f.write(b'\x00' * 16)

    def validtopic(self, topic):
        """For dir-valued topics, validity = dirpath exists and is non-empty."""
        d = self.dirpath(topic)
        return os.path.isdir(d) and bool(os.listdir(d))

    def valid(self):
        return all(self.validtopic(t) for t in self.TOPICS)


class MixedTopicBlock(Datablock):
    """TOPICS dict where some values are None (directory artifact) and some are
    filenames — mirrors the PoseGridReel pattern:
        {'gold_table': None, 'cell_catalog': 'cell_catalog.parquet', 'shards': None}
    """
    TOPICS = {
        'gold_table': None,
        'cell_catalog': 'cell_catalog.parquet',
        'shards': None,
    }

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        # Directory topics: populate the dirpath
        for topic in ('gold_table', 'shards'):
            dirpath = self.dirpath(topic, ensure=True)
            with open(os.path.join(dirpath, 'data.bin'), 'wb') as f:
                f.write(b'\x00' * 8)
        # File topic
        path = self.path('cell_catalog', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write('catalog')

    def validtopic(self, topic):
        if self.TOPICS[topic] is None:
            d = self.dirpath(topic)
            return os.path.isdir(d) and bool(os.listdir(d))
        return os.path.isfile(self.path(topic))

    def valid(self):
        return all(self.validtopic(t) for t in self.TOPICS)



class CountingBlock(Datablock):
    """Tracks how many times __build__ is called (to verify skip-if-valid)."""
    TOPICS = {'counter': 'counter.txt'}
    _build_count = 0

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass

    def __build__(self):
        CountingBlock._build_count += 1
        path = self.path('counter', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(str(CountingBlock._build_count))


class NestedBlock(Datablock):
    """Datablock containing another Datablock to test valid_tree and valid_cfg."""
    TOPICS = {'nested': 'nested.txt'}
    
    @dataclass
    class CONFIG(Datablock.CONFIG):
        child: SingleTopicBlock = None
        
    def __build__(self):
        path = self.path('nested', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write("nested")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(cls, tmp_path, **kwargs):
    """Instantiate a Datablock subclass rooted in a pytest tmp_path."""
    return cls(url=str(tmp_path), **kwargs)


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
        path = block.path('output')
        assert os.path.isfile(path)
        with open(path) as f:
            assert "built:" in f.read()

    def test_build_creates_all_topic_files(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        for topic in block.TOPICS:
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

    # -- Core lifecycle (default clear_dirpath=False) --------------------------

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

    # -- clear_dirpath=False (default): removes files, preserves dirs ----------

    def test_default_removes_single_topic_file(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        block.build()
        path = block.path('output')
        assert os.path.exists(path)
        block.UNSAFE_clear(OVERRIDE=True)
        assert not os.path.exists(path)
        # directory should still exist
        assert os.path.isdir(block.dirpath('output'))

    def test_default_removes_multi_topic_files_preserves_dirs(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        for topic in block.TOPICS:
            assert os.path.exists(block.path(topic))
        block.UNSAFE_clear(OVERRIDE=True)
        for topic in block.TOPICS:
            assert not os.path.exists(block.path(topic))
            assert os.path.isdir(block.dirpath(topic))

    def test_default_specific_topic_removes_file_preserves_dir(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        block.UNSAFE_clear('alpha', OVERRIDE=True)
        assert not os.path.exists(block.path('alpha'))
        assert os.path.isdir(block.dirpath('alpha'))
        # beta untouched
        assert os.path.exists(block.path('beta'))

    # -- clear_dirpath=True: removes entire directories ------------------------

    def test_clear_dirpath_removes_single_topic_dir(self, tmp_path):
        block = _make_block(SingleTopicBlock, tmp_path)
        block.build()
        topic_dir = block.dirpath('output')
        assert os.path.isdir(topic_dir)
        block.UNSAFE_clear(OVERRIDE=True, clear_dirpath=True)
        assert not os.path.exists(topic_dir)

    def test_clear_dirpath_removes_multi_topic_dirs(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        block.UNSAFE_clear(OVERRIDE=True, clear_dirpath=True)
        for topic in block.TOPICS:
            assert not os.path.exists(block.dirpath(topic))

    def test_clear_dirpath_specific_topic_removes_dir(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        block.UNSAFE_clear('alpha', OVERRIDE=True, clear_dirpath=True)
        assert not os.path.exists(block.dirpath('alpha'))
        # beta untouched
        assert os.path.isdir(block.dirpath('beta'))
        assert os.path.exists(block.path('beta'))

    def test_clear_dirpath_rebuild(self, tmp_path):
        """Full cycle with clear_dirpath=True."""
        block = _make_block(MultiTopicBlock, tmp_path)
        block.build()
        assert block.valid() is True
        block.UNSAFE_clear(OVERRIDE=True, clear_dirpath=True)
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
        path = block.path('output')
        assert os.path.isfile(path)
        assert os.path.getsize(path) == 0

    def test_breadcrumbs_create_all_topic_files(self, tmp_path):
        block = _make_block(MultiTopicBlock, tmp_path)
        block.leave_breadcrumbs()
        for topic in block.TOPICS:
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
        assert os.path.isfile(block.path('output'))
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


# ---------------------------------------------------------------------------
# 6. UNSAFE_copy_from()
# ---------------------------------------------------------------------------

class TestUNSAFECopyFrom:

    # -- copy_dirpath=False (default): copies individual files -----------------

    def test_copy_single_topic_file(self, tmp_path):
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(SingleTopicBlock, src_root)
        src.build()
        assert src.valid() is True

        dst = _make_block(SingleTopicBlock, dst_root)
        assert dst.valid() is False
        dst.UNSAFE_copy_from(src.anchorkeypath)
        assert dst.valid() is True
        # The file content should match
        with open(dst.path('output')) as f:
            content = f.read()
        assert "built:" in content

    def test_copy_multi_topic_files(self, tmp_path):
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(MultiTopicBlock, src_root)
        src.build()

        dst = _make_block(MultiTopicBlock, dst_root)
        dst.UNSAFE_copy_from(src.anchorkeypath)
        assert dst.valid() is True
        for topic in dst.TOPICS:
            assert os.path.isfile(dst.path(topic))

    def test_copy_default_preserves_only_files(self, tmp_path):
        """Default copy_dirpath=False copies files, not extra dir contents."""
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(SingleTopicBlock, src_root)
        src.build()
        # Add an extra file in the source topic dirpath that isn't the topic file
        extra = os.path.join(src.dirpath('output'), "extra.txt")
        with open(extra, "w") as f:
            f.write("extra")

        dst = _make_block(SingleTopicBlock, dst_root)
        dst.UNSAFE_copy_from(src.anchorkeypath)
        assert dst.valid() is True
        # The extra file should NOT have been copied
        assert not os.path.exists(os.path.join(dst.dirpath('output'), "extra.txt"))

    # -- copy_dirpath=True: copies entire directories --------------------------

    def test_copy_dirpath_single_topic(self, tmp_path):
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(SingleTopicBlock, src_root)
        src.build()
        # Add an extra file in the source topic dirpath
        extra = os.path.join(src.dirpath('output'), "extra.txt")
        with open(extra, "w") as f:
            f.write("extra")

        dst = _make_block(SingleTopicBlock, dst_root)
        dst.UNSAFE_copy_from(src.anchorkeypath, copy_dirpath=True)
        assert dst.valid() is True
        # The extra file SHOULD have been copied with copy_dirpath=True
        assert os.path.exists(os.path.join(dst.dirpath('output'), "extra.txt"))

    def test_copy_dirpath_multi_topic(self, tmp_path):
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(MultiTopicBlock, src_root)
        src.build()
        # Add extra files in each topic dirpath
        for topic in src.TOPICS:
            extra = os.path.join(src.dirpath(topic), "bonus.txt")
            with open(extra, "w") as f:
                f.write(f"bonus_{topic}")

        dst = _make_block(MultiTopicBlock, dst_root)
        dst.UNSAFE_copy_from(src.anchorkeypath, copy_dirpath=True)
        assert dst.valid() is True
        for topic in dst.TOPICS:
            assert os.path.exists(os.path.join(dst.dirpath(topic), "bonus.txt"))

    def test_copy_returns_self(self, tmp_path):
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(SingleTopicBlock, src_root)
        src.build()
        dst = _make_block(SingleTopicBlock, dst_root)
        result = dst.UNSAFE_copy_from(src.anchorkeypath)
        assert result is dst

    def test_copy_then_clear_then_recopy(self, tmp_path):
        """Full cycle: copy → valid → clear → invalid → copy again → valid."""
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(SingleTopicBlock, src_root)
        src.build()

        dst = _make_block(SingleTopicBlock, dst_root)
        dst.UNSAFE_copy_from(src.anchorkeypath)
        assert dst.valid() is True
        dst.UNSAFE_clear(OVERRIDE=True)
        assert dst.valid() is False
        dst.UNSAFE_copy_from(src.anchorkeypath)
        assert dst.valid() is True

    def test_copy_mixed_topics(self, tmp_path):
        """TOPICS dict with a mix of None (dir) and filename values.

        Regression test: UNSAFE_copy_from used to call copy_topic_file for
        None-valued topics, crashing with TypeError on os.path.join(topic, None).
        """
        src_root = tmp_path / "src"
        dst_root = tmp_path / "dst"
        src = _make_block(MixedTopicBlock, src_root)
        src.build()
        assert src.valid() is True

        dst = _make_block(MixedTopicBlock, dst_root)
        assert dst.valid() is False
        dst.UNSAFE_copy_from(src.anchorkeypath)
        assert dst.valid() is True
        # Dir-valued topics should be populated
        for topic in ('gold_table', 'shards'):
            d = dst.dirpath(topic)
            assert os.path.isdir(d) and os.listdir(d), f"{topic} dir missing or empty"
        # File-valued topic
        assert os.path.isfile(dst.path('cell_catalog'))


# ---------------------------------------------------------------------------
# 7. TOPICS[topic]=None — directory-valued topics
# ---------------------------------------------------------------------------

class TestDirTopicClear:
    """TOPICS[topic]=None means the artifact is a directory, not a file.

    UNSAFE_clear() must remove it recursively (the fix on datablocks.py
    lines 897 and 910: is_dir = TOPICS.get(topic) is None).
    """

    def test_build_creates_topic_dirs(self, tmp_path):
        block = _make_block(DirTopicBlock, tmp_path)
        block.__build__()
        for topic in block.TOPICS:
            assert os.path.isdir(block.dirpath(topic))
            assert os.listdir(block.dirpath(topic))

    def test_valid_after_build(self, tmp_path):
        block = _make_block(DirTopicBlock, tmp_path)
        assert not block.valid()
        block.__build__()
        assert block.valid()

    def test_clear_all_removes_topic_dirs_recursively(self, tmp_path):
        """UNSAFE_clear() with no topics should remove each dir-valued topic dir."""
        block = _make_block(DirTopicBlock, tmp_path)
        block.__build__()
        assert block.valid()
        block.UNSAFE_clear(OVERRIDE=True, clear_dirpath=True)
        for topic in block.TOPICS:
            assert not os.path.exists(block.dirpath(topic))

    def test_clear_specific_topic_removes_only_that_dir(self, tmp_path):
        """UNSAFE_clear('images') should only remove images/, not masks/."""
        block = _make_block(DirTopicBlock, tmp_path)
        block.__build__()
        assert block.valid()
        block.UNSAFE_clear('images', OVERRIDE=True, clear_dirpath=True)
        assert not os.path.exists(block.dirpath('images'))
        assert os.path.isdir(block.dirpath('masks'))
        assert os.listdir(block.dirpath('masks'))


# ---------------------------------------------------------------------------
# 8. valid_cfg() and valid_tree()
# ---------------------------------------------------------------------------

class TestValidTree:
    
    def test_valid_tree_invalid_initial(self, tmp_path):
        from dbx.datablocks import quote
        child = SingleTopicBlock(url=str(tmp_path / "child"))
        parent = NestedBlock(
            url=str(tmp_path / "parent"),
            spec={'child': quote(child)},
        )
        
        # Initially both are invalid
        assert parent.valid() is False
        assert child.valid() is False
        
        assert parent.valid_cfg() == {'child': False}
        assert parent.valid_cfg(reduce=True) is False
        
        assert parent.valid_tree() == {
            'child': {
                'valid': False,
                'tree': {}
            }
        }
        
    def test_valid_tree_mixed_validity(self, tmp_path):
        from dbx.datablocks import quote
        child = SingleTopicBlock(url=str(tmp_path / "child"))
        parent = NestedBlock(
            url=str(tmp_path / "parent"),
            spec={'child': quote(child)},
        )
        
        # Build child only
        child.build()
        assert child.valid() is True
        assert parent.valid() is False
        
        assert parent.valid_cfg() == {'child': True}
        assert parent.valid_cfg(reduce=True) is True
        
        assert parent.valid_tree() == {
            'child': {
                'valid': True,
                'tree': {}
            }
        }
        
        # Build parent
        parent.build()
        assert parent.valid() is True
        
        # Clear child (parent remains valid but its upstream is invalid)
        child.UNSAFE_clear(OVERRIDE=True)
        assert child.valid() is False
        assert parent.valid() is True
        
        assert parent.valid_cfg() == {'child': False}
        assert parent.valid_tree() == {
            'child': {
                'valid': False,
                'tree': {}
            }
        }
