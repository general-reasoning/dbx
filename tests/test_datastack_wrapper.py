"""
Tests for dbx.datastack() — the Datastackable wrapper.

Verifies:
1. Protocol validation (missing __shards__, SHARD raises TypeError).
2. Wrapper class structure (name, bases, __wrapped__).
3. shards() converts Datablockable instances to Datablocks.
4. __build__ orchestrates shard building correctly.
5. CONFIG lifting works.
6. Optional __read__ delegation.
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

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.paths = paths
        self.cfg = cfg
        self.verbose = verbose
        self.detailed = detailed
        self.debug = debug
        self.log = log
        self.device = device
        self.built = False

    def __build__(self, *args, **kwargs):
        import fsspec
        path = self.paths
        fs, _ = fsspec.url_to_fs(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        with fs.open(path, 'w') as f:
            f.write(f"item:{self.cfg.item_id}")
        self.built = True
        return self

    def __read__(self, topic=None):
        import fsspec
        path = self.paths
        fs, _ = fsspec.url_to_fs(path)
        with fs.open(path, 'r') as f:
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

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.paths = paths
        self.cfg = cfg
        self.verbose = verbose
        self.detailed = detailed
        self.debug = debug
        self.log = log
        self.device = device

    def __shards__(self):
        return [
            ItemProcessor(
                paths=None,  # will be overridden by from_datablockable
                cfg=ItemProcessor.CONFIG(item_id=i),
                verbose=self.verbose,
                detailed=self.detailed,
                debug=self.debug,
                log=self.log,
                device=self.device,
            )
            for i in range(self.cfg.n_items)
        ]

    def __read__(self, topic=None):
        return f"batch with {self.cfg.n_items} items"


class NoReadBatchProcessor:
    """A Datastackable without __read__."""
    SHARD = ItemProcessor

    @dataclass
    class CONFIG:
        n_items: int = 3

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.cfg = cfg
        self.verbose = verbose
        self.detailed = detailed
        self.debug = debug
        self.log = log
        self.device = device

    def __shards__(self):
        return [
            ItemProcessor(
                paths=None,
                cfg=ItemProcessor.CONFIG(item_id=i),
                verbose=self.verbose,
                detailed=self.detailed,
                debug=self.debug,
                log=self.log,
                device=self.device,
            )
            for i in range(self.cfg.n_items)
        ]


# ---------------------------------------------------------------------------
# 1. Protocol validation
# ---------------------------------------------------------------------------

class TestProtocolValidation:

    def test_missing_shards_raises(self):
        class Bad:
            SHARD = ItemProcessor
        with pytest.raises(TypeError, match="__shards__"):
            datastack(Bad)

    def test_missing_shard_raises(self):
        class Bad:
            def __shards__(self): return []
        with pytest.raises(TypeError, match="SHARD"):
            datastack(Bad)


# ---------------------------------------------------------------------------
# 2. Wrapper class structure
# ---------------------------------------------------------------------------

class TestWrapperStructure:

    def test_name(self):
        Wrapped = datastack(BatchProcessor)
        assert Wrapped.__name__ == '_BatchProcessor_Datastack_'

    def test_is_datastack_subclass(self):
        Wrapped = datastack(BatchProcessor)
        assert issubclass(Wrapped, Datastack)

    def test_is_datablock_subclass(self):
        Wrapped = datastack(BatchProcessor)
        assert issubclass(Wrapped, Datablock)

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
        assert not hasattr(NoReadBatchProcessor, '__read__')
        # The wrapper should NOT have __read__ either (inherits Datablock's)


# ---------------------------------------------------------------------------
# 7. Serialization
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_pickle_roundtrip(self):
        Wrapped = datastack(BatchProcessor)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped(root=tmp, spec=dict(n_items=3))
            restored = pickle.loads(pickle.dumps(stack))
            assert isinstance(restored.obj, BatchProcessor)
            assert restored.cfg.n_items == 3


# ---------------------------------------------------------------------------
# 8. from_datastackable
# ---------------------------------------------------------------------------

class TestFromDatastackable:

    def _make_obj(self, **overrides):
        from dbx.databits import Logger
        defaults = dict(
            paths=None,
            cfg=BatchProcessor.CONFIG(n_items=7),
            verbose=True,
            detailed=False,
            debug=True,
            log=Logger(),
            device='cuda:0',
        )
        defaults.update(overrides)
        return BatchProcessor(**defaults)

    def test_spec_extracted(self):
        Wrapped = datastack(BatchProcessor)
        obj = self._make_obj()
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped.from_datastackable(obj, root=tmp)
            assert stack.cfg.n_items == 7

    def test_verbose_propagated(self):
        Wrapped = datastack(BatchProcessor)
        obj = self._make_obj(verbose=True)
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped.from_datastackable(obj, root=tmp)
            assert stack.verbose is True

    def test_device_propagated(self):
        Wrapped = datastack(BatchProcessor)
        obj = self._make_obj(device='cuda:0')
        with tempfile.TemporaryDirectory() as tmp:
            stack = Wrapped.from_datastackable(obj, root=tmp)
            assert stack.device == 'cuda:0'

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
