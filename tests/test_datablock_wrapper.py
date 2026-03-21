"""
Tests for dbx.datablock() — the Datablockable wrapper.

Verifies:
1. Protocol validation (missing __build__, __read__, TOPICFILES raises TypeError).
2. Wrapper class has correct name, bases, and __wrapped__ attribute.
3. TOPICFILES / TOPICFILE are correctly lifted.
4. CONFIG is lifted; user CONFIG not inheriting Datablock.CONFIG gets auto-wrapped.
5. __post_init__ instantiates the inner object with resolved paths and cfg.
6. __build__ and __read__ delegate to the inner object.
7. VERSION is lifted when present.
8. Anchor uses caller module, not dbx.datablocks.
9. Serialization round-trip works (pickle).
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

    @dataclass
    class CONFIG:
        model_name: str = 'resnet50'
        layer: str = 'avgpool'

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        self.cfg = cfg
        self.verbose = verbose
        self.detailed = detailed
        self.debug = debug
        self.log = log
        self.device = device
        self.built = False
        self.build_args = None

    def __build__(self, *args, **kwargs):
        self.built = True
        self.build_args = (args, kwargs)
        return self

    def __read__(self, topic):
        return f"data_for_{topic}"


class SingleTopicProcessor:
    """A single-topic Datablockable using TOPICFILE."""
    TOPICFILE = 'output.csv'

    @dataclass
    class CONFIG:
        delimiter: str = ','

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return "single_topic_data"


class InheritingConfigProcessor:
    """A Datablockable whose CONFIG already inherits from Datablock.CONFIG."""
    TOPICFILES = {'result': 'result.npz'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n_components: str = '10'

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic):
        return f"result_{self.cfg.n_components}"


class VersionedProcessor:
    """A Datablockable with VERSION."""
    TOPICFILES = {'data': 'data.pt'}
    VERSION = 'v3'

    @dataclass
    class CONFIG:
        pass

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        pass

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic):
        return None


class NoConfigProcessor:
    """A Datablockable with no CONFIG at all."""
    TOPICFILE = 'out.txt'

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        pass

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return "no_config"


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
        """TOPICFILES/TOPICFILE may be set in __init__, so class-level absence is fine."""
        class NoClassLevelTopics:
            def __init__(self, *, cfg, verbose, detailed, debug, log, device):
                self.cfg = cfg
                self.TOPICFILE = 'dynamic.txt'  # set at instance level
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
# 5. Inner object instantiation
# ---------------------------------------------------------------------------

class TestInnerObjectInstantiation:

    def test_obj_created_on_construction(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert hasattr(block, 'obj')
        assert isinstance(block.obj, MultiTopicProcessor)

    def test_obj_receives_cfg(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.obj.cfg is block.cfg

    def test_obj_receives_device(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.obj.device == block.device


# ---------------------------------------------------------------------------
# 6. Delegation
# ---------------------------------------------------------------------------

class TestDelegation:

    def test_build_delegates(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        # build() won't actually run because valid() may be True;
        # call __build__ directly
        block.__build__('arg1', key='val')
        assert block.obj.built is True
        assert block.obj.build_args == (('arg1',), {'key': 'val'})

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
        assert isinstance(restored.obj, MultiTopicProcessor)
        assert restored.obj.cfg.model_name == block.obj.cfg.model_name

    def test_set_creates_new_instance(self):
        Wrapped = datablock(MultiTopicProcessor)
        block1 = Wrapped(root='/tmp/dbx_test_wrapper')
        block2 = block1.set(device='cuda')
        assert block2.device == 'cuda'
        assert isinstance(block2.obj, MultiTopicProcessor)


# ---------------------------------------------------------------------------
# 9. from_datablockable
# ---------------------------------------------------------------------------

class TestFromDatablockable:

    def _make_processor(self, **overrides):
        """Create a MultiTopicProcessor instance with sensible defaults."""
        from dbx.databits import Logger
        defaults = dict(
            cfg=MultiTopicProcessor.CONFIG(model_name='vit_b', layer='layer4'),
            verbose=True,
            detailed=False,
            debug=True,
            log=Logger(),
            device='cuda:1',
        )
        defaults.update(overrides)
        return MultiTopicProcessor(**defaults)

    def test_spec_extracted_from_cfg(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert block.cfg.model_name == 'vit_b'
        assert block.cfg.layer == 'layer4'

    def test_verbose_propagated(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor(verbose=True)
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert block.verbose is True

    def test_device_propagated(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor(device='cuda:1')
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert block.device == 'cuda:1'

    def test_debug_propagated(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor(debug=True)
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert block.debug is True

    def test_kwargs_override_propagated_attrs(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor(device='cuda:1')
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper', device='cpu')
        assert block.device == 'cpu'

    def test_type_check_rejects_wrong_type(self):
        Wrapped = datablock(MultiTopicProcessor)
        with pytest.raises(TypeError, match="Expected an instance of"):
            Wrapped.from_datablockable("not_a_processor", root='/tmp')

    def test_result_is_datablock(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert isinstance(block, Datablock)

    def test_inner_obj_created(self):
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        assert hasattr(block, 'obj')
        assert isinstance(block.obj, MultiTopicProcessor)

    def test_hash_matches_direct_construction(self):
        """from_datablockable with same spec should produce same hash as direct construction."""
        Wrapped = datablock(MultiTopicProcessor)
        obj = self._make_processor()
        block_from = Wrapped.from_datablockable(obj, root='/tmp/dbx_test_wrapper')
        block_direct = Wrapped(root='/tmp/dbx_test_wrapper', spec=dict(model_name='vit_b', layer='layer4'))
        assert block_from.hash == block_direct.hash


# ---------------------------------------------------------------------------
# 10. Instance-level TOPICFILE(S) propagation
# ---------------------------------------------------------------------------

class InstanceTopicFileProcessor:
    """Datablockable that sets TOPICFILE in __init__ based on cfg."""

    @dataclass
    class CONFIG:
        output_name: str = 'result'

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        self.cfg = cfg
        self.verbose = verbose
        self.detailed = detailed
        self.debug = debug
        self.log = log
        self.device = device
        self.TOPICFILE = f'{cfg.output_name}.csv'

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return 'instance_data'


class InstanceTopicFilesProcessor:
    """Datablockable that sets TOPICFILES in __init__ based on cfg."""

    @dataclass
    class CONFIG:
        prefix: str = 'out'

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        self.cfg = cfg
        self.verbose = verbose
        self.detailed = detailed
        self.debug = debug
        self.log = log
        self.device = device
        self.TOPICFILES = {
            'data': f'{cfg.prefix}_data.pt',
            'meta': f'{cfg.prefix}_meta.json',
        }

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic):
        return f'{topic}_content'


class BothClassAndInstanceTopicProcessor:
    """Class-level TOPICFILE should take precedence (no instance propagation)."""
    TOPICFILE = 'class_level.txt'

    @dataclass
    class CONFIG:
        pass

    def __init__(self, *, cfg, verbose, detailed, debug, log, device):
        self.cfg = cfg
        # Also set at instance level — should NOT override class-level
        self.TOPICFILE = 'instance_level.txt'

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return None


class TestInstanceTopicPropagation:
    """Tests for TOPICFILE(S) defined in __init__ instead of at class level."""

    def test_instance_topicfile_propagated(self):
        """TOPICFILE set in __init__ should be available on the wrapper instance."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert hasattr(block, 'TOPICFILE')
        assert block.TOPICFILE == 'result.csv'

    def test_instance_topicfile_reflects_cfg(self):
        """TOPICFILE set from cfg in __init__ should reflect the spec."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper', spec={'output_name': 'features'})
        assert block.TOPICFILE == 'features.csv'

    def test_instance_topicfiles_propagated(self):
        """TOPICFILES set in __init__ should be available on the wrapper instance."""
        Wrapped = datablock(InstanceTopicFilesProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert hasattr(block, 'TOPICFILES')
        assert block.TOPICFILES == {
            'data': 'out_data.pt',
            'meta': 'out_meta.json',
        }

    def test_instance_topicfiles_reflects_cfg(self):
        """TOPICFILES set from cfg in __init__ should reflect the spec."""
        Wrapped = datablock(InstanceTopicFilesProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper', spec={'prefix': 'train'})
        assert block.TOPICFILES == {
            'data': 'train_data.pt',
            'meta': 'train_meta.json',
        }

    def test_class_level_topicfile_not_overridden_by_instance(self):
        """When TOPICFILE is defined at class level, instance-level should NOT override it."""
        Wrapped = datablock(BothClassAndInstanceTopicProcessor)
        # The class-level TOPICFILE is lifted as a class attribute
        assert Wrapped.TOPICFILE == 'class_level.txt'
        # When instantiated, the class-level attribute takes priority
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.TOPICFILE == 'class_level.txt'

    def test_no_class_level_topicfile_attribute(self):
        """Wrapper class should NOT have TOPICFILE as class attribute when only set in __init__."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        # Not a class-level attribute
        assert not hasattr(Wrapped, 'TOPICFILE')
        # But IS available on instances after construction
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert hasattr(block, 'TOPICFILE')

    def test_instance_topicfile_inner_obj_has_it(self):
        """The inner Datablockable object should have the TOPICFILE attribute."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert hasattr(block.obj, 'TOPICFILE')
        assert block.obj.TOPICFILE == block.TOPICFILE

    def test_instance_topicfiles_has_topics(self):
        """Block with instance-level TOPICFILES should report has_topics()=True."""
        Wrapped = datablock(InstanceTopicFilesProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.has_topics()
        assert set(block.topics()) == {'data', 'meta'}

    def test_instance_topicfile_has_topic(self):
        """Block with instance-level TOPICFILE should report has_topic()=True."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.has_topic()

    def test_build_delegates_with_instance_topicfile(self):
        """__build__ should still delegate correctly when TOPICFILE is set in __init__."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        block.__build__()
        # No error = success

    def test_read_delegates_with_instance_topicfiles(self):
        """__read__ should delegate correctly when TOPICFILES is set in __init__."""
        Wrapped = datablock(InstanceTopicFilesProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.__read__('data') == 'data_content'

    def test_pickle_roundtrip_instance_topicfile(self):
        """Pickle roundtrip should work for instance-level TOPICFILE blocks."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper', spec={'output_name': 'myfile'})
        restored = pickle.loads(pickle.dumps(block))
        assert hasattr(restored, 'TOPICFILE')
        assert restored.TOPICFILE == 'myfile.csv'

    def test_set_preserves_instance_topicfile(self):
        """block.set() should reconstruct correctly and re-propagate instance TOPICFILE."""
        Wrapped = datablock(InstanceTopicFileProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper', spec={'output_name': 'alpha'})
        block2 = block.set(device='cuda')
        assert hasattr(block2, 'TOPICFILE')
        assert block2.TOPICFILE == 'alpha.csv'
