"""
Serialization roundtrip tests for Datablock and Datastack.

Verifies that Datablock and Datastack instances survive:
    1. copy.deepcopy()
    2. pickle.dumps() / pickle.loads()  (full ser/des)
    3. manual __getstate__() → __setstate__() roundtrip

Each path must preserve:
    - Identity: hash, tag, anchor, url, cfg, keyby
    - Functionality: valid(), build(), executor_cls (Datastack)
"""
import copy
import math
import os
import pickle
import tempfile
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, Datastack


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.fixture
def url(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Sample subclasses
# ---------------------------------------------------------------------------

class SimpleBlock(Datablock):
    """Minimal single-topic Datablock for serialization tests."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'test'"

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")

    def __read__(self, topic):
        with open(self.path('output'), 'r') as f:
            return f.read()


class TaggedBlock(Datablock):
    """Datablock with an explicit tag."""
    TOPICS = {'tagged': 'tagged.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        value: int = 42

    def __build__(self):
        path = self.path('tagged', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"v={self.cfg.value}")

    def __read__(self, topic):
        with open(self.path('tagged'), 'r') as f:
            return f.read()


class MultiTopicBlock(Datablock):
    """Multi-topic Datablock."""
    TOPICS = {'alpha': 'alpha.txt', 'beta': 'beta.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n: int = 5

    def __build__(self):
        for topic in self.TOPICS:
            self.dirpath(topic, ensure=True)
            with open(self.path(topic), 'w') as f:
                f.write(f"{topic}:{self.cfg.n}")


class StackBlock(Datablock):
    """Trivial block for Datastack tests."""
    TOPICS = {'block': 'block.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        idx: int = 0

    def __build__(self):
        path = self.path('block', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"block:{self.cfg.idx}")

    def __read__(self, topic):
        with open(self.path('block'), 'r') as f:
            return f.read()


class SimpleStack(Datastack):
    """Concrete Datastack for serialization tests."""
    TOPICS = {'stack_meta': 'stack_meta.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        total_items: int = 6
        block_size: int = 2

    @property
    def n_blocks(self):
        return math.ceil(self.cfg.total_items / self.cfg.block_size)

    def __block__(self, idx):
        return StackBlock(url=self.url, spec=dict(idx=idx))

    def blocks(self):
        return [self.__block__(i) for i in range(self.n_blocks)]

    def __read__(self, topic):
        return f"stack:{self.n_blocks}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_identity_preserved(original, restored):
    """Assert that identity-defining attributes survived the roundtrip."""
    assert restored.hash == original.hash, "hash mismatch"
    assert restored.url == original.url, "url mismatch"
    assert restored.anchor == original.anchor, "anchor mismatch"
    assert restored.keyby == original.keyby, "keyby mismatch"
    assert restored.fqcn == original.fqcn, "fqcn mismatch"
    assert restored.spec == original.spec, "spec mismatch"
    # cfg fields
    for field in vars(original.cfg):
        assert getattr(restored.cfg, field) == getattr(original.cfg, field), \
            f"cfg.{field} mismatch"


def _roundtrip_deepcopy(obj):
    return copy.deepcopy(obj)


def _roundtrip_pickle(obj):
    return pickle.loads(pickle.dumps(obj))


def _roundtrip_state(obj):
    """Manual __getstate__ → new instance via __setstate__."""
    state = obj.__getstate__()
    restored = obj.__class__.__new__(obj.__class__)
    restored.__setstate__(state)
    return restored


ROUNDTRIPS = [
    pytest.param(_roundtrip_deepcopy, id="deepcopy"),
    pytest.param(_roundtrip_pickle, id="pickle"),
    pytest.param(_roundtrip_state, id="getstate-setstate"),
]


# ===========================================================================
# Datablock serialization tests
# ===========================================================================

class TestDatablockSerialization:
    """Datablock identity and functionality survive all roundtrip paths."""

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_identity_preserved(self, url, roundtrip):
        block = SimpleBlock(url=url, spec=dict(label="'hello'"))
        restored = roundtrip(block)
        _assert_identity_preserved(block, restored)

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_tag_preserved(self, url, roundtrip):
        block = TaggedBlock(url=url, tag="my-tag", spec=dict(value=99))
        restored = roundtrip(block)
        _assert_identity_preserved(block, restored)
        assert restored._tag_ == "my-tag"
        assert restored.tag == "my-tag"

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_custom_anchor_preserved(self, url, roundtrip):
        block = SimpleBlock(url=url, anchor="custom/anchor")
        restored = roundtrip(block)
        assert restored.anchor == "custom/anchor"
        _assert_identity_preserved(block, restored)

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_multi_topic_identity(self, url, roundtrip):
        block = MultiTopicBlock(url=url, spec=dict(n=10))
        restored = roundtrip(block)
        _assert_identity_preserved(block, restored)
        assert restored.TOPICS == block.TOPICS

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_valid_false_before_build(self, url, roundtrip):
        """A fresh block is invalid; the clone should also be invalid."""
        block = SimpleBlock(url=url)
        restored = roundtrip(block)
        assert block.valid() is False
        assert restored.valid() is False

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_build_after_roundtrip(self, url, roundtrip):
        """A cloned block can be built independently."""
        block = SimpleBlock(url=url, spec=dict(label="'rt'"))
        restored = roundtrip(block)
        restored.build()
        assert restored.valid() is True
        assert restored.read('output') == "built:'rt'"

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_keyby_variants(self, url, roundtrip):
        """All keyby modes survive the roundtrip."""
        for keyby in ('hash', 'tag', 'taghash', 'tag_hash', 'version_hash', 'tag_version_hash', 'tag_version_shorthash'):
            kwargs = dict(keyby=keyby)
            if keyby == 'tag':
                kwargs['tag'] = 'test-tag'
            block = SimpleBlock(url=url, **kwargs)
            restored = roundtrip(block)
            assert restored.keyby == keyby
            assert restored.key == block.key


class TestDatablockSetPreservation:
    """Datablock.set() uses deepcopy internally — verify it works."""

    def test_set_preserves_identity(self, url):
        block = SimpleBlock(url=url, tag="v1", spec=dict(label="'orig'"))
        clone = block.set(tag="v2")
        assert clone.tag == "v2"
        assert clone.hash == block.hash  # spec unchanged, hash same
        assert clone.cfg.label == "'orig'"

    def test_set_changes_spec(self, url):
        block = SimpleBlock(url=url, spec=dict(label="'a'"))
        clone = block.set(spec=dict(label="'b'"))
        assert clone.cfg.label == "'b'"
        assert clone.hash != block.hash  # spec changed


# ===========================================================================
# Datastack serialization tests
# ===========================================================================

class TestDatastackSerialization:
    """Datastack identity, executor_cls, and build survive all roundtrip paths."""

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_identity_preserved(self, url, roundtrip):
        stack = SimpleStack(url=url, spec=dict(total_items=6, block_size=2))
        restored = roundtrip(stack)
        _assert_identity_preserved(stack, restored)

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_executor_cls_default(self, url, roundtrip):
        """Default (inline) executor_cls survives roundtrip."""
        from dbx.dataparts import InlineCallableExecutor
        stack = SimpleStack(url=url)
        restored = roundtrip(stack)
        assert restored.executor_cls is InlineCallableExecutor

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_executor_cls_multithreading(self, url, roundtrip):
        """Multithreading executor_cls survives roundtrip."""
        from dbx.dataparts import MultithreadingCallableExecutor
        stack = SimpleStack(url=url, parallelization='multithreading', n_workers=4)
        restored = roundtrip(stack)
        assert restored.executor_cls is MultithreadingCallableExecutor
        assert restored.n_workers == 4

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_executor_cls_multiprocessing(self, url, roundtrip):
        """Multiprocessing executor_cls survives roundtrip."""
        from dbx.dataparts import MultiprocessingCallableExecutor
        stack = SimpleStack(url=url, parallelization='multiprocessing', n_workers=2)
        restored = roundtrip(stack)
        assert restored.executor_cls is MultiprocessingCallableExecutor

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_n_blocks_preserved(self, url, roundtrip):
        stack = SimpleStack(url=url, spec=dict(total_items=10, block_size=3))
        restored = roundtrip(stack)
        assert restored.n_blocks == stack.n_blocks

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_build_after_roundtrip(self, url, roundtrip):
        """A roundtripped Datastack can still build all its blocks."""
        stack = SimpleStack(url=url, spec=dict(total_items=4, block_size=2))
        restored = roundtrip(stack)
        restored.build()
        blocks = restored.blocks()
        for blk in blocks:
            assert blk.valid(), f"Block {blk.cfg.idx} invalid after build"

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_tag_on_stack(self, url, roundtrip):
        stack = SimpleStack(url=url, tag="stack-v1")
        restored = roundtrip(stack)
        assert restored.tag == "stack-v1"
        _assert_identity_preserved(stack, restored)


class TestDatastackSetPreservation:
    """Datastack.set() (which deepcopies internally) must preserve executor_cls."""

    def test_set_tag_preserves_executor(self, url):
        """This is the exact bug that motivated the executor_cls property fix."""
        from dbx.dataparts import MultithreadingCallableExecutor
        stack = SimpleStack(
            url=url,
            parallelization='multithreading',
            n_workers=4,
            tag="v1",
        )
        clone = stack.set(tag="v2")
        assert clone.tag == "v2"
        assert clone.executor_cls is MultithreadingCallableExecutor
        assert clone.n_workers == 4

    def test_set_spec_preserves_executor(self, url):
        from dbx.dataparts import MultiprocessingCallableExecutor
        stack = SimpleStack(
            url=url,
            parallelization='multiprocessing',
            n_workers=2,
            spec=dict(total_items=6, block_size=2),
        )
        clone = stack.set(spec=dict(total_items=10, block_size=5))
        assert clone.executor_cls is MultiprocessingCallableExecutor
        assert clone.cfg.total_items == 10
        assert clone.n_blocks == 2

    def test_set_then_build(self, url):
        """set() clone can build successfully."""
        stack = SimpleStack(
            url=url,
            parallelization='multithreading',
            n_workers=2,
            spec=dict(total_items=4, block_size=2),
        )
        clone = stack.set(tag="built-clone")
        clone.build()
        for blk in clone.blocks():
            assert blk.valid()


# ===========================================================================
# Nested Datablock inside spec — the real-world trigger
# ===========================================================================

class OuterBlock(Datablock):
    """A Datablock whose spec contains another Datablock (the deepcopy trigger)."""
    TOPICS = {'outer': 'outer.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        name: str = "'outer'"

    def __build__(self):
        path = self.path('outer', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"outer:{self.cfg.name}")

    def __read__(self, topic):
        with open(self.path('outer'), 'r') as f:
            return f.read()


class TestNestedDatablockInSpec:
    """When a Datablock's spec/kwargs contains another Datablock,
    deepcopy must not crash on the inner block."""

    @pytest.mark.parametrize("roundtrip", ROUNDTRIPS)
    def test_deepcopy_block_with_block_attr(self, url, roundtrip):
        inner = SimpleBlock(url=url, spec=dict(label="'inner'"))
        outer = OuterBlock(url=url, inner_block=inner)
        restored = roundtrip(outer)
        _assert_identity_preserved(outer, restored)
        # The inner block attribute should also survive
        assert hasattr(restored, 'inner_block')

    def test_set_with_nested_block(self, url):
        """set() on a block that holds another block in kwargs."""
        inner = SimpleBlock(url=url, spec=dict(label="'inner'"))
        outer = OuterBlock(url=url, inner_block=inner)
        clone = outer.set(tag="cloned")
        assert clone.tag == "cloned"
        assert hasattr(clone, 'inner_block')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
