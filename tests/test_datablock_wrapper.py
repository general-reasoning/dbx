"""
Tests for dbx.datablock() — the Datablockable wrapper.

With MI (multiple inheritance), the wrapper IS-A Datablock and IS-A Datablockable.
The Datablockable's __init__ is NOT called when wrapped — Datablock.__init__
provides cfg, verbose, log, etc.

Verifies:
1. Protocol validation (missing __build__, __read__ raises TypeError).
2. Wrapper class has correct name, bases, and __wrapped__ attribute.
3. TOPICFILES / TOPICFILE are correctly lifted from the Datablockable class.
4. CONFIG is lifted; user CONFIG not inheriting Datablock.CONFIG gets auto-wrapped.
5. The wrapper instance IS both Datablock and Datablockable.
6. __build__ and __read__ delegate to the Datablockable class methods.
7. VERSION is lifted when present.
8. Anchor uses caller module, not dbx.datablocks.
9. Serialization round-trip works (pickle).
10. from_datablockable creates wrapper from raw instances.
"""
import os
import pickle
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock
from dbx.datawraps import datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_wrapper')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample Datablockable classes
# ---------------------------------------------------------------------------

class MultiTopicProcessor:
    """A multi-topic Datablockable with plain CONFIG (not inheriting Datablock.CONFIG)."""
    TOPICFILES = {'features': 'features.pt', 'metadata': 'meta.json'}
    built = False  # class-level default (since __init__ is NOT called when wrapped)
    build_args = None

    @dataclass
    class CONFIG:
        model_name: str = 'resnet50'
        layer: str = 'avgpool'

    def __init__(self, *, cfg=None, **_):
        # Only called for standalone (unwrapped) use
        self.cfg = cfg
        self.built = False
        self.build_args = None

    def __build__(self, *args, **kwargs):
        self.built = True
        self.build_args = (args, kwargs)
        return self

    def __read__(self, topic=None):
        return f"data_for_{topic}"


class SingleTopicProcessor:
    """A single-topic Datablockable."""
    TOPICFILE = 'output.csv'

    @dataclass
    class CONFIG:
        delimiter: str = ','

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return "single_topic_data"


class InheritingConfigProcessor:
    """A Datablockable whose CONFIG already inherits from Datablock.CONFIG."""
    TOPICFILE = 'pca_result.npz'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n_components: str = '10'

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic):
        return f"result_{self.cfg.n_components}"


class EmptyConfigProcessor:
    """A Datablockable with an explicit empty CONFIG."""
    TOPICFILE = 'empty_cfg.pt'

    @dataclass
    class CONFIG:
        pass

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return None


class NoConfigProcessor:
    """A Datablockable with no CONFIG at all."""
    TOPICFILE = 'out.txt'

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return None


class NoTopicProcessor:
    """A Datablockable with NO TOPICFILE or TOPICFILES."""
    _build_count = 0

    @dataclass
    class CONFIG:
        label: str = 'noop'

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        NoTopicProcessor._build_count += 1
        return self

    def __read__(self, topic=None):
        return "no_topic_data"


class PathOverrideProcessor:
    """A no-topic Datablockable that overrides path() to return dirpath().

    This mimics config-wrapper Datablocks (like Lightning) that define a
    custom path() returning the key-level directory instead of None.
    Before the valid() fix, this would cause valid() to return False
    because the directory doesn't exist yet.
    """
    _build_count = 0

    @dataclass
    class CONFIG:
        variant: str = 'base'

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def path(self, topic=None, *, ensure_dirpath=False):
        return self.dirpath()

    def __build__(self, *args, **kwargs):
        PathOverrideProcessor._build_count += 1
        return self

    def __read__(self, topic=None):
        return "path_override_data"


class VersionedProcessor:
    """A Datablockable with a VERSION attribute."""
    TOPICFILE = 'versioned.txt'
    VERSION = 'v3'

    @dataclass
    class CONFIG:
        pass

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return "versioned_data"


# ---------------------------------------------------------------------------
# 1. Protocol validation
# ---------------------------------------------------------------------------

class TestProtocolValidation:

    def test_missing_build_raises(self):
        class Bad:
            TOPICFILE = 'x.txt'
            def __read__(self, topic): ...
        with pytest.raises(TypeError, match="__build__"):
            datablock(Bad)

    def test_missing_read_raises(self):
        class Bad:
            TOPICFILE = 'x.txt'
            def __build__(self): ...
        with pytest.raises(TypeError, match="__read__"):
            datablock(Bad)

    def test_missing_topics_accepted(self):
        """TOPICFILES/TOPICFILE may be set at class level or not (dynamic setting
        is no longer supported since __init__ is not called when wrapped)."""
        class NoClassLevelTopics:
            def __init__(self, *, cfg=None, **_):
                self.TOPICFILE = 'dynamic.txt'  # only works standalone
            def __build__(self, *args, **kwargs): return self
            def __read__(self, topic=None): return None
        # Should NOT raise
        Wrapped = datablock(NoClassLevelTopics)
        assert issubclass(Wrapped, Datablock)


# ---------------------------------------------------------------------------
# 2. Wrapper class structure
# ---------------------------------------------------------------------------

class TestWrapperClassStructure:

    def test_name(self):
        Wrapped = datablock(MultiTopicProcessor)
        assert Wrapped.__name__ == 'MultiTopicProcessor_Datablock'

    def test_is_datablock_subclass(self):
        Wrapped = datablock(MultiTopicProcessor)
        assert issubclass(Wrapped, Datablock)

    def test_is_datablockable_subclass(self):
        """With MI, the wrapper is also a subclass of the user class."""
        Wrapped = datablock(MultiTopicProcessor)
        assert issubclass(Wrapped, MultiTopicProcessor)

    def test_wrapped_reference(self):
        Wrapped = datablock(MultiTopicProcessor)
        assert Wrapped.__wrapped__ is MultiTopicProcessor

    def test_module_is_caller(self):
        Wrapped = datablock(MultiTopicProcessor)
        assert Wrapped.__module__ == __name__


# ---------------------------------------------------------------------------
# 3. TOPICFILES / TOPICFILE lifting
# ---------------------------------------------------------------------------

class TestTopicLifting:

    def test_topicfiles_lifted(self):
        Wrapped = datablock(MultiTopicProcessor)
        assert Wrapped.TOPICFILES == {'features': 'features.pt', 'metadata': 'meta.json'}

    def test_topicfile_lifted(self):
        Wrapped = datablock(SingleTopicProcessor)
        assert Wrapped.TOPICFILE == 'output.csv'

    def test_no_topics_not_lifted(self):
        Wrapped = datablock(NoTopicProcessor)
        assert not hasattr(Wrapped, 'TOPICFILE')
        assert not hasattr(Wrapped, 'TOPICFILES')


# ---------------------------------------------------------------------------
# 4. CONFIG lifting
# ---------------------------------------------------------------------------

class TestConfigLifting:

    def test_plain_config_becomes_datablock_config_subclass(self):
        Wrapped = datablock(MultiTopicProcessor)
        assert issubclass(Wrapped.CONFIG, Datablock.CONFIG)

    def test_inheriting_config_preserved(self):
        Wrapped = datablock(InheritingConfigProcessor)
        assert Wrapped.CONFIG is InheritingConfigProcessor.CONFIG

    def test_no_config_gets_empty(self):
        Wrapped = datablock(NoConfigProcessor)
        assert issubclass(Wrapped.CONFIG, Datablock.CONFIG)

    def test_config_fields_preserved(self):
        Wrapped = datablock(MultiTopicProcessor)
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(Wrapped.CONFIG)}
        assert 'model_name' in field_names
        assert 'layer' in field_names


# ---------------------------------------------------------------------------
# 5. Instance type (MI — no inner object)
# ---------------------------------------------------------------------------

class TestMIInstance:

    def test_instance_is_datablock(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert isinstance(block, Datablock)

    def test_instance_is_datablockable(self):
        """With MI, the wrapper instance IS-A Datablockable."""
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert isinstance(block, MultiTopicProcessor)

    def test_cfg_available(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.cfg is not None
        assert block.cfg.model_name == 'resnet50'

    def test_keyby_available(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper', keyby='handle')
        assert block.keyby == 'handle'

    def test_init_not_called(self):
        """Datablockable's __init__ is NOT called when wrapped."""
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        # built defaults to class-level False, not the __init__ value
        assert block.built is False


# ---------------------------------------------------------------------------
# 6. Delegation
# ---------------------------------------------------------------------------

class TestDelegation:

    def test_build_delegates(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        block.__build__('arg1', key='val')
        assert block.built is True
        assert block.build_args == (('arg1',), {'key': 'val'})

    def test_read_delegates(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        result = block.__read__('features')
        assert result == "data_for_features"


# ---------------------------------------------------------------------------
# 7. VERSION lifting
# ---------------------------------------------------------------------------

class TestVersionLifting:

    def test_version_lifted(self):
        Wrapped = datablock(VersionedProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.version == 'v3'

    def test_version_none_when_absent(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.version is None


# ---------------------------------------------------------------------------
# 8. Serialization
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_pickle_roundtrip(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        restored = pickle.loads(pickle.dumps(block))
        assert isinstance(restored, Datablock)
        assert isinstance(restored, MultiTopicProcessor)
        assert restored.cfg.model_name == block.cfg.model_name

    def test_set_creates_new_instance(self):
        Wrapped = datablock(MultiTopicProcessor)
        block1 = Wrapped(root='/tmp/dbx_test_wrapper')
        block2 = block1.set(tag='newtag')
        assert block2.tag == 'newtag'
        assert isinstance(block2, MultiTopicProcessor)


# ---------------------------------------------------------------------------
# 9. from_datablockable
# ---------------------------------------------------------------------------

class TestFromDatablockable:

    def _make_processor(self, **overrides):
        """Create a standalone MultiTopicProcessor instance with sensible defaults."""
        defaults = dict(
            cfg=MultiTopicProcessor.CONFIG(model_name='vit_b', layer='layer4'),
        )
        defaults.update(overrides)
        return MultiTopicProcessor(**defaults)

    def test_spec_extracted_from_cfg(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert block.cfg.model_name == 'vit_b'
        assert block.cfg.layer == 'layer4'

    def test_spec_propagated(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert block.cfg.model_name == 'vit_b'

    def test_kwargs_override_propagated_attrs(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper', tag='override')
        assert block.tag == 'override'

    def test_type_check_rejects_wrong_type(self):
        Wrapped = datablock(MultiTopicProcessor)
        with pytest.raises(TypeError, match="Expected an instance of"):
            Wrapped.from_datablockable("not_a_processor", root='/tmp')

    def test_result_is_both_types(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert isinstance(block, Datablock)
        assert isinstance(block, MultiTopicProcessor)

    def test_hash_matches_direct_construction(self):
        """from_datablockable with same spec should produce same hash as direct construction."""
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block_from = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        block_direct = Wrapped(root='/tmp/dbx_test_wrapper', spec=dict(model_name='vit_b', layer='layer4'))
        assert block_from.hash == block_direct.hash


# ---------------------------------------------------------------------------
# 10. No-topic wrapper: valid() == True, build() skips
# ---------------------------------------------------------------------------

class TestNoTopicWrapper:

    def test_valid_returns_true_immediately(self, tmp_path):
        """A datablock()-wrapped Datablockable with no TOPICFILE(S) is always valid."""
        Wrapped = datablock(NoTopicProcessor)
        block = Wrapped(root=str(tmp_path))
        assert block.valid() is True

    def test_path_returns_none(self, tmp_path):
        Wrapped = datablock(NoTopicProcessor)
        block = Wrapped(root=str(tmp_path))
        assert block.path() is None

    def test_build_skips_because_valid(self, tmp_path):
        """Since valid() is True, build() should skip __build__ entirely."""
        NoTopicProcessor._build_count = 0
        Wrapped = datablock(NoTopicProcessor)
        block = Wrapped(root=str(tmp_path))
        block.build()
        assert NoTopicProcessor._build_count == 0, "__build__ should not be called when valid() is True"

    def test_isinstance_checks(self, tmp_path):
        Wrapped = datablock(NoTopicProcessor)
        block = Wrapped(root=str(tmp_path))
        assert isinstance(block, Datablock)
        assert isinstance(block, NoTopicProcessor)

    def test_read_delegates(self, tmp_path):
        Wrapped = datablock(NoTopicProcessor)
        block = Wrapped(root=str(tmp_path))
        assert block.__read__() == "no_topic_data"

    def test_path_override_valid_despite_missing_dir(self, tmp_path):
        """A Datablockable that overrides path() to return dirpath() should
        still be valid when it has no TOPICFILE(S) — the fix short-circuits
        before path() is ever called."""
        Wrapped = datablock(PathOverrideProcessor)
        block = Wrapped(root=str(tmp_path))
        # dirpath() points to a nonexistent directory, but valid() should
        # not even check it.
        assert not os.path.exists(block.dirpath())
        assert block.valid() is True

    def test_path_override_build_skips(self, tmp_path):
        """build() should skip __build__ for path-overriding no-topic blocks."""
        PathOverrideProcessor._build_count = 0
        Wrapped = datablock(PathOverrideProcessor)
        block = Wrapped(root=str(tmp_path))
        block.build()
        assert PathOverrideProcessor._build_count == 0
