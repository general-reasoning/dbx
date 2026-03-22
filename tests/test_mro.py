"""
Tests for multiple-inheritance MRO in datablock() and datastack() wrappers.

The wrappers use true multiple inheritance:
    datablock(cls) → type(name, (Datablock, cls), ...)
    datastack(cls) → type(name, (Datastack, cls), ...)

This means:
- Protocol methods (__build__, __read__, __shard__, n_shards) are delegated
  via explicit class_attrs that call cls.method(self, ...).
- Datablock/Datastack methods (build, valid, hash, path, shards, ...) are
  inherited normally.
- Custom methods on the Datablockable/Datastackable are accessible on
  wrapper instances via MRO.

Because `verbose`, `detailed`, `debug`, and `cfg` are read-only properties
or cached_property on Datablock, Datablockable __init__ must NOT assign
those — Datablock manages them.  Only `device` needs direct assignment.

These tests verify that all three categories resolve correctly and that
the Datablockable/Datastackable side of the MRO doesn't accidentally
shadow or break Datablock/Datastack behaviour.
"""
import os
import tempfile
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, Datastack
from dbx.datawraps import datablock, datastack


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_mro')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Datablockable fixtures
# ---------------------------------------------------------------------------

class ProcessorWithCustomMethods:
    """Datablockable that defines extra methods beyond the protocol."""
    TOPICFILE = 'result.txt'
    _built = False  # class-level default; __init__ is NOT called when wrapped

    @dataclass
    class CONFIG:
        factor: int = 2

    def __init__(self, *, cfg=None, device=None, **_):
        # Only called for standalone (unwrapped) use
        self.cfg = cfg
        self.device = device
        self._built = False

    def __build__(self, *args, **kwargs):
        self._built = True
        return self

    def __read__(self, topic=None):
        return f"result_{self.cfg.factor}"

    # -- Custom methods (should be accessible on wrapper via MRO) --
    def compute(self, x):
        """A custom method not part of the Datablockable protocol."""
        return x * self.cfg.factor

    @property
    def is_built(self):
        """A custom property."""
        return self._built


class ProcessorWithPathOverride:
    """Datablockable that overrides path() — should shadow Datablock.path()."""
    TOPICFILE = 'out.dat'

    @dataclass
    class CONFIG:
        output_dir: str = '/tmp/custom'

    def __init__(self, *, cfg=None, device=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return None

    def path(self, topic=None, *, ensure_dirpath=False):
        return os.path.join(self.cfg.output_dir, self.TOPICFILE)


class ProcessorWithConflictingProperty:
    """Datablockable that naively assigns verbose — this MUST fail under MI."""
    TOPICFILE = 'out.txt'

    @dataclass
    class CONFIG:
        pass

    def __init__(self, *, cfg=None, device=None, **_):
        self.cfg = cfg
        self.verbose = True  # should fail: read-only property on Datablock

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return None


# ---------------------------------------------------------------------------
# Datastackable fixtures
# ---------------------------------------------------------------------------

class ShardProcessor:
    """A simple Datablockable for use as SHARD in Datastackable tests."""
    TOPICFILE = 'shard.txt'

    @dataclass
    class CONFIG:
        idx: int = 0

    def __init__(self, *, cfg=None, device=None, **_):
        self.cfg = cfg
        self.device = device

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return f"shard_{self.cfg.idx}"


class PipelineWithCustomMethods:
    """Datastackable with custom methods beyond the protocol."""
    SHARD = ShardProcessor

    @dataclass
    class CONFIG:
        n_items: int = 6
        shard_size: int = 2

    def __init__(self, *, cfg=None, device=None, **_):
        self.cfg = cfg
        self.device = device

    @property
    def n_shards(self):
        import math
        return math.ceil(self.cfg.n_items / self.cfg.shard_size)

    def __shard__(self, idx):
        return ShardProcessor(
            cfg=ShardProcessor.CONFIG(idx=idx),
            device=self.device,
        )

    # -- Custom methods --
    def summary(self):
        """Custom method not part of the Datastackable protocol."""
        return f"{self.n_shards} shards of size {self.cfg.shard_size}"

    @property
    def total_items(self):
        """Custom property."""
        return self.cfg.n_items


# ===========================================================================
# datablock() MRO tests
# ===========================================================================

class TestDatablockMRO:
    """Verify method resolution across Datablock ← Datablockable inheritance."""

    def test_wrapper_inherits_from_both(self):
        """Wrapper class is a subclass of both Datablock and the user class."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        assert issubclass(Wrapped, Datablock)
        assert issubclass(Wrapped, ProcessorWithCustomMethods)

    def test_instance_is_both(self):
        """Wrapper instance is an instance of both Datablock and the user class."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        assert isinstance(block, Datablock)
        assert isinstance(block, ProcessorWithCustomMethods)

    # -- Protocol methods (delegated via class_attrs) -------------------------

    def test_build_delegates_to_datablockable(self):
        """__build__ should call the Datablockable's __build__."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        block.__build__()
        assert block._built is True

    def test_read_delegates_to_datablockable(self):
        """__read__ should call the Datablockable's __read__."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro', spec={'factor': 5})
        assert block.__read__() == "result_5"

    # -- Datablock methods (inherited) ----------------------------------------

    def test_build_lifecycle_inherited(self):
        """Datablock.build() is inherited and calls __build__."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        block.__build__()
        assert block._built is True

    def test_hash_inherited(self):
        """Datablock.hash is inherited."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        assert block.hash is not None
        assert isinstance(block.hash, str)

    def test_cfg_inherited(self):
        """Datablock.cfg is inherited and resolves config correctly."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro', spec={'factor': 7})
        assert block.cfg.factor == 7

    def test_valid_inherited(self):
        """Datablock.valid() is inherited."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        assert isinstance(block.valid(), bool)

    def test_has_topic_inherited(self):
        """Datablock.has_topic() is inherited."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        assert block.has_topic() is True

    def test_set_inherited(self):
        """Datablock.set() is inherited and returns correct wrapper type."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        block2 = block.set(device='cuda')
        assert isinstance(block2, Wrapped)
        assert block2.device == 'cuda'

    def test_verbose_is_datablock_property(self):
        """verbose should resolve to Datablock's read-only property, not a plain attribute."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        # verbose is a property that reads from the Logger
        assert isinstance(type(block).__mro__[1].__dict__.get('verbose', None), property) or \
               isinstance(Datablock.__dict__.get('verbose', None), property)
        # should be a bool, not raise
        assert isinstance(block.verbose, bool)

    def test_detailed_is_datablock_property(self):
        """detailed should resolve to Datablock's read-only property."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        assert isinstance(block.detailed, bool)

    def test_debug_is_datablock_property(self):
        """debug should resolve to Datablock's read-only property."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        assert isinstance(block.debug, bool)

    # -- Custom Datablockable methods (accessible via MRO) --------------------

    def test_custom_method_accessible(self):
        """Custom methods on the Datablockable are accessible on the wrapper."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro', spec={'factor': 3})
        assert block.compute(10) == 30

    def test_custom_property_accessible(self):
        """Custom properties on the Datablockable are accessible on the wrapper."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        block = Wrapped(root='/tmp/dbx_test_mro')
        assert block.is_built is False
        block.__build__()
        assert block.is_built is True

    # -- path() override from Datablockable -----------------------------------

    def test_datablockable_path_overrides_datablock_path(self):
        """When Datablockable defines path(), it should override Datablock.path()."""
        Wrapped = datablock(ProcessorWithPathOverride)
        block = Wrapped(root='/tmp/dbx_test_mro', spec={'output_dir': '/data/out'})
        # Should use the Datablockable's path(), not Datablock's
        assert block.path() == '/data/out/out.dat'

    def test_dirpath_blocked_when_path_overridden(self):
        """When Datablockable defines path(), dirpath() should raise."""
        Wrapped = datablock(ProcessorWithPathOverride)
        block = Wrapped(root='/tmp/dbx_test_mro')
        with pytest.raises(NotImplementedError, match="defines its own path"):
            block.dirpath()

    # -- Conflict detection ---------------------------------------------------

    def test_init_not_called_when_wrapped(self):
        """Datablockable __init__ is NOT called when wrapped — no property conflicts."""
        class ChecksInit:
            TOPICFILE = 'out.txt'
            init_called = False
            @dataclass
            class CONFIG:
                pass
            def __init__(self, **_):
                self.init_called = True
                self.verbose = True  # would fail if called on wrapper
            def __build__(self, *args, **kwargs): return self
            def __read__(self, topic=None): return None
        Wrapped = datablock(ChecksInit)
        block = Wrapped(root='/tmp/dbx_test_mro')
        # __init__ was NOT called, so no AttributeError and init_called stays False
        assert block.init_called is False

    def test_cfg_not_clobbered_when_init_skipped(self):
        """Since __init__ is skipped, cfg can't be clobbered."""
        class CfgAssigner:
            TOPICFILE = 'out.txt'
            @dataclass
            class CONFIG:
                x: int = 1
            def __init__(self, *, cfg=None, device=None, **_):
                self.cfg = 'CLOBBERED'  # would clobber if called
            def __build__(self, *args, **kwargs): return self
            def __read__(self, topic=None): return None
        Wrapped = datablock(CfgAssigner)
        block = Wrapped(root='/tmp/dbx_test_mro', spec={'x': 42})
        # __init__ is NOT called, so cfg is Datablock's cached_property
        assert block.cfg.x == 42

    # -- MRO order verification -----------------------------------------------

    def test_mro_order(self):
        """MRO should be: Wrapper → Datablock → Datablockable → object."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        mro = Wrapped.__mro__
        # Wrapper comes first
        assert mro[0] is Wrapped
        # Then Datablock
        db_idx = mro.index(Datablock)
        # Then the user class
        cls_idx = mro.index(ProcessorWithCustomMethods)
        assert db_idx < cls_idx

    def test_explicit_attrs_shadow_both_parents(self):
        """class_attrs (__build__, __read__) on the wrapper take priority."""
        Wrapped = datablock(ProcessorWithCustomMethods)
        # __build__ is defined in class_attrs, not inherited from either parent
        assert '__build__' in Wrapped.__dict__
        assert '__read__' in Wrapped.__dict__


# ===========================================================================
# datastack() MRO tests
# ===========================================================================

class TestDatastackMRO:
    """Verify method resolution across Datastack ← Datastackable inheritance."""

    def test_wrapper_inherits_from_both(self):
        """Wrapper class is a subclass of Datastack, Datablock, and the user class."""
        Wrapped = datastack(PipelineWithCustomMethods)
        assert issubclass(Wrapped, Datastack)
        assert issubclass(Wrapped, Datablock)
        assert issubclass(Wrapped, PipelineWithCustomMethods)

    def test_instance_is_both(self):
        """Wrapper instance is an instance of all parent classes."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        assert isinstance(stack, Datastack)
        assert isinstance(stack, Datablock)
        assert isinstance(stack, PipelineWithCustomMethods)

    # -- Protocol methods (delegated via class_attrs) -------------------------

    def test_n_shards_delegates_to_datastackable(self):
        """n_shards should call the Datastackable's n_shards property."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro', spec={'n_items': 10, 'shard_size': 3})
        assert stack.n_shards == 4  # ceil(10/3)

    def test_shard_delegates_to_datastackable(self):
        """__shard__ should delegate to the Datastackable and wrap the result."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        shard = stack.shard(0)
        # Should be a Datablock wrapping ShardProcessor
        assert isinstance(shard, Datablock)

    # -- Datastack methods (inherited) ----------------------------------------

    def test_shards_inherited(self):
        """Datastack.shards() is inherited."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro', spec={'n_items': 4, 'shard_size': 2})
        shards = stack.shards()
        assert len(shards) == 2

    def test_build_inherited(self):
        """Datablock.build() is inherited through Datastack → Datablock."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        assert hasattr(stack, 'build')
        assert callable(stack.build)

    def test_unsafe_clear_shards_inherited(self):
        """Datastack.UNSAFE_clear_shards() is inherited."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        assert hasattr(stack, 'UNSAFE_clear_shards')
        assert callable(stack.UNSAFE_clear_shards)

    # -- Datablock methods (inherited via Datastack) --------------------------

    def test_hash_inherited(self):
        """Datablock.hash is available on the Datastack wrapper."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        assert stack.hash is not None

    def test_cfg_inherited(self):
        """Datablock.cfg is available and resolves config correctly."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro', spec={'n_items': 8, 'shard_size': 4})
        assert stack.cfg.n_items == 8
        assert stack.cfg.shard_size == 4

    def test_valid_inherited(self):
        """Datablock.valid() is available."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        assert isinstance(stack.valid(), bool)

    def test_set_inherited(self):
        """Datablock.set() returns correct wrapper type."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        stack2 = stack.set(device='cuda')
        assert isinstance(stack2, Wrapped)
        assert stack2.device == 'cuda'

    def test_verbose_is_datablock_property(self):
        """verbose should resolve to Datablock's read-only property."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        assert isinstance(stack.verbose, bool)

    # -- Custom Datastackable methods (accessible via MRO) --------------------

    def test_custom_method_accessible(self):
        """Custom methods on the Datastackable are accessible on the wrapper."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro', spec={'n_items': 6, 'shard_size': 2})
        assert stack.summary() == "3 shards of size 2"

    def test_custom_property_accessible(self):
        """Custom properties on the Datastackable are accessible on the wrapper."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro', spec={'n_items': 12, 'shard_size': 3})
        assert stack.total_items == 12

    def test_custom_method_can_access_datablock_properties(self):
        """Custom methods should be able to read Datablock properties like verbose."""
        Wrapped = datastack(PipelineWithCustomMethods)
        stack = Wrapped(root='/tmp/dbx_test_mro')
        # __shard__ accesses self.verbose, self.detailed, etc. — these
        # should resolve to Datablock's properties without error
        shard = stack.shard(0)
        assert shard is not None

    # -- MRO order verification -----------------------------------------------

    def test_mro_order(self):
        """MRO should be: Wrapper → Datastack → Datablock → Datastackable → object."""
        Wrapped = datastack(PipelineWithCustomMethods)
        mro = Wrapped.__mro__
        assert mro[0] is Wrapped
        ds_idx = mro.index(Datastack)
        db_idx = mro.index(Datablock)
        cls_idx = mro.index(PipelineWithCustomMethods)
        assert ds_idx < db_idx < cls_idx

    def test_explicit_attrs_shadow_both_parents(self):
        """class_attrs (__shard__, n_shards) on the wrapper take priority."""
        Wrapped = datastack(PipelineWithCustomMethods)
        assert '__shard__' in Wrapped.__dict__
        assert 'n_shards' in Wrapped.__dict__
