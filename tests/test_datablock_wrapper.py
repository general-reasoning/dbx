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

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.paths = paths
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
        return f"data_for_{topic}_at_{self.paths[topic]}"


class SingleTopicProcessor:
    """A single-topic Datablockable using TOPICFILE."""
    TOPICFILE = 'output.csv'

    @dataclass
    class CONFIG:
        delimiter: str = ','

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.paths = paths
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic=None):
        return f"single_topic_data_at_{self.paths}"


class InheritingConfigProcessor:
    """A Datablockable whose CONFIG already inherits from Datablock.CONFIG."""
    TOPICFILES = {'result': 'result.npz'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n_components: str = '10'

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.paths = paths
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

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.paths = paths

    def __build__(self, *args, **kwargs):
        return self

    def __read__(self, topic):
        return None


class NoConfigProcessor:
    """A Datablockable with no CONFIG at all."""
    TOPICFILE = 'out.txt'

    def __init__(self, *, paths, cfg, verbose, detailed, debug, log, device):
        self.paths = paths

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

    def test_missing_topics_raises(self):
        class Bad:
            def __build__(self): ...
            def __read__(self, topic): ...
        with pytest.raises(TypeError, match="TOPICFILES or TOPICFILE"):
            datablock(Bad)


# ---------------------------------------------------------------------------
# 2. Wrapper class structure
# ---------------------------------------------------------------------------

class TestWrapperClassStructure:

    def test_name(self):
        Wrapped = datablock(MultiTopicProcessor)
        assert Wrapped.__name__ == '_MultiTopicProcessor_Datablock_'

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

    def test_obj_receives_paths_dict(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert isinstance(block.obj.paths, dict)
        assert set(block.obj.paths.keys()) == {'features', 'metadata'}

    def test_obj_receives_cfg(self):
        Wrapped = datablock(MultiTopicProcessor)
        block = Wrapped(root='/tmp/dbx_test_wrapper')
        assert block.obj.cfg is block.cfg


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
        expected_path = block.path('features')
        assert result == f"data_for_features_at_{expected_path}"


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
        assert restored.obj.paths == block.obj.paths

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
            paths={'features': '/fake/features.pt', 'metadata': '/fake/meta.json'},
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
