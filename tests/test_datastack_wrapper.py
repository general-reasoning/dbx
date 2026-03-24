"""
Tests for dbx.datastack() — the Datastackable wrapper.

With MI (multiple inheritance), the wrapper IS-A Datastack and IS-A Datastackable.
The Datastackable's __init__ is NOT called when wrapped — Datablock.__init__
provides cfg, verbose, log, etc.

Verifies:
1. Protocol validation (missing shard(), SHARD raises TypeError).
2. Wrapper class structure (name, bases, __wrapped__).
3. shards() creates wrapped Datablocks from shard() results.
4. __build__ orchestrates shard building correctly.
5. CONFIG lifting works.
6. Optional read() → __read__ delegation.
7. Serialization round-trip (pickle).
8. from_datastackable classmethod.
"""
import os
import pickle
import tempfile
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, Datastack
from dbx.datawraps import datablock, datastack


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_datastack_wrapper')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample Datablockable shard class
# ---------------------------------------------------------------------------

class ItemProcessor:
    """A Datablockable that writes a marker file per shard."""
    TOPICFILE = 'item.txt'

    @dataclass
    class CONFIG:
        item_id: int = None

    def __init__(self, *, cfg=None, **_):
        # Only called for standalone (unwrapped) use
        self.cfg = cfg
        self.built = False

    def build(self, *args, **kwargs):
        self.built = True
        # When wrapped, self IS the Datablock, so self.path() works directly
        path = self.path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(f"item:{self.cfg.item_id}")
        return self

    def read(self, topic=None):
        path = self.path()
        with open(path, 'r') as f:
            return f.read()


# ---------------------------------------------------------------------------
# Sample Datastackable class
# ---------------------------------------------------------------------------

class BatchProcessor:
    """A Datastackable that shards items into ItemProcessors."""
    SHARD = ItemProcessor
    TOPICFILE = 'batch_meta.txt'

    @dataclass
    class CONFIG:
        n_items: int = 5

    def __init__(self, *, cfg=None, **_):
        # Only called for standalone (unwrapped) use
        self.cfg = cfg

    @property
    def n_shards(self):
        return self.cfg.n_items

    def shard(self, idx):
        return ItemProcessor(
            cfg=ItemProcessor.CONFIG(item_id=idx),
        )

    def read(self, topic=None):
        return f"batch with {self.cfg.n_items} items"


class NoReadBatchProcessor:
    """A Datastackable without read()."""
    SHARD = ItemProcessor

    @dataclass
    class CONFIG:
        n_items: int = 3

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    @property
    def n_shards(self):
        return self.cfg.n_items

    def shard(self, idx):
        return ItemProcessor(
            cfg=ItemProcessor.CONFIG(item_id=idx),
        )


# ---------------------------------------------------------------------------
# 1. Protocol validation
# ---------------------------------------------------------------------------

class TestProtocolValidation:

    def test_missing_shard_raises(self):
        class Bad:
            SHARD = ItemProcessor
            @property
            def n_shards(self): return 0
        with pytest.raises(TypeError, match="shard"):
            datastack(Bad)

    def test_missing_n_shards_raises(self):
        class Bad:
            SHARD = ItemProcessor
            def shard(self, idx): return None
        with pytest.raises(TypeError, match="n_shards"):
            datastack(Bad)

    def test_missing_shard_class_raises(self):
        class Bad:
            @property
            def n_shards(self): return 0
            def shard(self, idx): return None
        with pytest.raises(TypeError, match="SHARD"):
            datastack(Bad)


# ---------------------------------------------------------------------------
# 2. Wrapper class structure
# ---------------------------------------------------------------------------

class TestWrapperStructure:

    def test_name(self):
        Wrapped = datastack(BatchProcessor)
        assert Wrapped.__name__ == 'BatchProcessor_Datastack'

    def test_is_datastack_subclass(self):
        Wrapped = datastack(BatchProcessor)
        assert issubclass(Wrapped, Datastack)

    def test_is_datablock_subclass(self):
        Wrapped = datastack(BatchProcessor)
        assert issubclass(Wrapped, Datablock)

    def test_is_datastackable_subclass(self):
        """With MI, the wrapper is also a subclass of the user class."""
        Wrapped = datastack(BatchProcessor)
        assert issubclass(Wrapped, BatchProcessor)

    def test_wrapped_reference(self):
        Wrapped = datastack(BatchProcessor)
        assert Wrapped.__wrapped__ is BatchProcessor

    def test_module_is_caller(self):
        Wrapped = datastack(BatchProcessor)
        assert Wrapped.__module__ == __name__

    def test_shard_block_stored(self):
        Wrapped = datastack(BatchProcessor)
        assert hasattr(Wrapped, '_ShardBlock_')
        assert issubclass(Wrapped._ShardBlock_, Datablock)
        assert Wrapped._ShardBlock_.__wrapped__ is ItemProcessor


# ---------------------------------------------------------------------------
# 3. CONFIG lifting
# ---------------------------------------------------------------------------

class TestConfigLifting:

    def test_config_becomes_datablock_config_subclass(self):
        Wrapped = datastack(BatchProcessor)
        assert issubclass(Wrapped.CONFIG, Datablock.CONFIG)

    def test_config_fields_preserved(self):
        Wrapped = datastack(BatchProcessor)
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(Wrapped.CONFIG)}
        assert 'n_items' in field_names


# ---------------------------------------------------------------------------
# 4. Shards conversion
# ---------------------------------------------------------------------------

class TestShardsConversion:

    def test_shards_returns_datablocks(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(root=tmp, spec=dict(n_items=3))
            shard_list = stack.shards()
            assert len(shard_list) == 3
            for s in shard_list:
                assert isinstance(s, Datablock)

    def test_shard_configs_match(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(root=tmp, spec=dict(n_items=4))
            shard_list = stack.shards()
            ids = [s.cfg.item_id for s in shard_list]
            assert ids == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 5. Build orchestration
# ---------------------------------------------------------------------------

class TestBuild:

    def test_inline_build(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(root=tmp, spec=dict(n_items=3))
            stack.build()
            shards = stack.shards()
            for s in shards:
                assert s.valid(), f"Shard item_id={s.cfg.item_id} not built"
                content = s.read()
                assert content == f"item:{s.cfg.item_id}"

    def test_multithreading_build(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(
                root=tmp,
                spec=dict(n_items=4),
                parallelization='multithreading',
                n_workers=2,
            )
            stack.build()
            shards = stack.shards()
            for s in shards:
                assert s.valid(), f"Shard item_id={s.cfg.item_id} not built"

    def test_build_returns_self(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(root=tmp, spec=dict(n_items=1))
            result = stack.build()
            assert result is stack


# ---------------------------------------------------------------------------
# 6. Optional __read__ delegation
# ---------------------------------------------------------------------------

class TestReadDelegation:

    def test_read_delegates_when_defined(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(root=tmp, spec=dict(n_items=5))
            assert stack.__read__() == "batch with 5 items"

    def test_no_read_when_not_defined(self):
        Wrapped = datastack(NoReadBatchProcessor)
        assert 'read' not in NoReadBatchProcessor.__dict__
        # The wrapper should NOT have __read__ (inherits Datablock's stub)


# ---------------------------------------------------------------------------
# 7. Serialization
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_pickle_roundtrip(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(root=tmp, spec=dict(n_items=3))
            restored = pickle.loads(pickle.dumps(stack))
            assert isinstance(restored, Datastack)
            assert isinstance(restored, BatchProcessor)
            assert restored.cfg.n_items == 3


# ---------------------------------------------------------------------------
# 8. from_datastackable
# ---------------------------------------------------------------------------

class TestFromDatastackable:

    def _make_obj(self, **overrides):
        defaults = dict(
            cfg=BatchProcessor.CONFIG(n_items=7),
        )
        defaults.update(overrides)
        return BatchProcessor(**defaults)

    def test_spec_extracted(self):
        Wrapped = datastack(BatchProcessor)
        obj = self._make_obj()
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped.from_datastackable(obj, root=tmp)
            assert stack.cfg.n_items == 7

    def test_spec_propagated(self):
        Wrapped = datastack(BatchProcessor)
        obj = self._make_obj()
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped.from_datastackable(obj, root=tmp)
            assert stack.cfg.n_items == 7

    def test_type_check_rejects_wrong_type(self):
        Wrapped = datastack(BatchProcessor)
        with pytest.raises(TypeError, match="Expected an instance of"):
            Wrapped.from_datastackable("not_a_batch", root='/tmp')

    def test_result_is_datastack(self):
        Wrapped = datastack(BatchProcessor)
        obj = self._make_obj()
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped.from_datastackable(obj, root=tmp)
            assert isinstance(stack, Datastack)
            assert isinstance(stack, Datablock)
            assert isinstance(stack, BatchProcessor)


# ---------------------------------------------------------------------------
# 9. No CONFIG — equivalent to class CONFIG: pass
# ---------------------------------------------------------------------------

class NoConfigBatchProcessor:
    """A Datastackable with no CONFIG at all.

    Should behave identically to having `class CONFIG: pass`.
    """
    SHARD = ItemProcessor

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    @property
    def n_shards(self):
        return 2

    def shard(self, idx):
        return ItemProcessor(cfg=ItemProcessor.CONFIG(item_id=idx))


class TestNoConfigDatastack:
    """A Datastackable with no CONFIG should be treated identically to one
    with `class CONFIG: pass` — wrapper succeeds, CONFIG is an empty
    Datablock.CONFIG subclass, cfg/hash/shards all work.
    """

    def test_wrapping_succeeds(self):
        Wrapped = datastack(NoConfigBatchProcessor)
        assert issubclass(Wrapped, Datastack)
        assert issubclass(Wrapped, NoConfigBatchProcessor)

    def test_config_is_datablock_config_subclass(self):
        """Synthesised CONFIG must be a Datablock.CONFIG subclass with no extra fields."""
        from dataclasses import fields as dc_fields
        Wrapped = datastack(NoConfigBatchProcessor)
        assert issubclass(Wrapped.CONFIG, Datablock.CONFIG)
        base_fields = {f.name for f in dc_fields(Datablock.CONFIG)}
        wrapper_fields = {f.name for f in dc_fields(Wrapped.CONFIG)}
        assert wrapper_fields == base_fields

    def test_instantiation(self, tmp_path):
        Wrapped = datastack(NoConfigBatchProcessor)
        stack = Wrapped(root=str(tmp_path))
        assert isinstance(stack, Datastack)
        assert isinstance(stack, Datablock)

    def test_cfg_accessible(self, tmp_path):
        Wrapped = datastack(NoConfigBatchProcessor)
        stack = Wrapped(root=str(tmp_path))
        assert stack.cfg is not None
        assert isinstance(stack.cfg, Datablock.CONFIG)

    def test_hash_works(self, tmp_path):
        Wrapped = datastack(NoConfigBatchProcessor)
        stack = Wrapped(root=str(tmp_path))
        assert isinstance(stack.hash, str)

    def test_n_shards(self, tmp_path):
        Wrapped = datastack(NoConfigBatchProcessor)
        stack = Wrapped(root=str(tmp_path))
        assert stack.n_shards == 2

    def test_shards_are_datablocks(self, tmp_path):
        Wrapped = datastack(NoConfigBatchProcessor)
        stack = Wrapped(root=str(tmp_path))
        shard_list = stack.shards()
        assert len(shard_list) == 2
        for s in shard_list:
            assert isinstance(s, Datablock)
