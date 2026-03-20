"""
Comprehensive tests for Datablock.kwargs and Datablock.dfn properties.

kwargs: Returns only the dynamically-supplied keyword arguments (not declared
        in Datablock.__init__'s explicit parameter list).

dfn:    Returns the full definition / state dict — explicit params + kwargs —
        sufficient to reconstruct the block via MyBlock(**block.dfn).
"""
import os
import pickle
import copy
import pytest
from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Test Datablock subclasses
# ---------------------------------------------------------------------------

class SimpleBlock(Datablock):
    """A minimal block with no extra class-level config."""
    def __build__(self):
        pass

class VersionedBlock(Datablock):
    """A block with a custom VERSION."""
    VERSION = "v2"
    def __build__(self):
        pass


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# dfn: returns all parameters (explicit + kwargs)
# ---------------------------------------------------------------------------

class TestDfn:

    def test_dfn_contains_explicit_params(self):
        """dfn should include explicit Datablock.__init__ params like root, tag, device."""
        block = SimpleBlock(root='/tmp/test', tag='mytag', device='cuda')
        d = block.dfn
        assert d['root'] == '/tmp/test'
        assert d['tag'] == 'mytag'
        assert d['device'] == 'cuda'

    def test_dfn_contains_dynamic_kwargs(self):
        """dfn should also include user-supplied **kwargs."""
        block = SimpleBlock(root='/tmp/test', alpha=1, beta='two')
        d = block.dfn
        assert d['alpha'] == 1
        assert d['beta'] == 'two'

    def test_dfn_includes_defaults(self):
        """dfn should include explicit params even when left at their defaults."""
        block = SimpleBlock(root='/tmp/test')
        d = block.dfn
        # anchored defaults to True, capture_output to False, device to 'cpu'
        assert d['anchored'] is True
        assert d['capture_output'] is False
        assert d['device'] == 'cpu'

    def test_dfn_matches_getstate(self):
        """dfn should be identical to __getstate__()."""
        block = SimpleBlock(root='/tmp/test', x=10)
        assert block.dfn == block.__getstate__()

    def test_dfn_can_reconstruct_block(self):
        """A block built from dfn should have the same dfn as the original."""
        block1 = SimpleBlock(root='/tmp/a', alpha=1, tag='t')
        block2 = SimpleBlock(**block1.dfn)
        assert block1.dfn == block2.dfn

    def test_dfn_is_fresh_dict(self):
        """Each call to dfn should return a new dict (not a cached reference)."""
        block = SimpleBlock(root='/tmp/test', x=1)
        d1 = block.dfn
        d2 = block.dfn
        assert d1 == d2
        assert d1 is not d2  # different dict objects

    def test_dfn_with_none_values(self):
        """Explicit params set to None should still appear in dfn."""
        block = SimpleBlock(root='/tmp/test', tag=None, revision=None)
        d = block.dfn
        assert 'tag' in d
        assert d['tag'] is None


# ---------------------------------------------------------------------------
# kwargs: returns only dynamic (non-explicit) parameters
# ---------------------------------------------------------------------------

class TestKwargs:

    def test_kwargs_excludes_explicit_params(self):
        """kwargs should not contain root, tag, revision, device, etc."""
        block = SimpleBlock(root='/tmp/test', tag='t', device='cuda', my_param=42)
        kw = block.kwargs
        assert 'root' not in kw
        assert 'tag' not in kw
        assert 'device' not in kw
        assert 'anchored' not in kw

    def test_kwargs_contains_only_dynamic(self):
        """kwargs should contain exactly the user-supplied **kwargs."""
        block = SimpleBlock(root='/tmp/test', alpha=1, beta=2)
        kw = block.kwargs
        assert kw == {'alpha': 1, 'beta': 2}

    def test_kwargs_empty_when_no_extras(self):
        """kwargs should be empty if no extra kwargs were passed."""
        block = SimpleBlock(root='/tmp/test')
        assert block.kwargs == {}

    def test_kwargs_is_subset_of_dfn(self):
        """Every key in kwargs should also appear in dfn, but not vice versa."""
        block = SimpleBlock(root='/tmp/test', extra=99)
        kw = block.kwargs
        d = block.dfn
        for k in kw:
            assert k in d
            assert d[k] == kw[k]
        # dfn has more keys than kwargs
        assert len(d) > len(kw)

    def test_kwargs_with_many_extras(self):
        """kwargs should handle many dynamic params correctly."""
        extras = {f'param_{i}': i for i in range(20)}
        block = SimpleBlock(root='/tmp/test', **extras)
        kw = block.kwargs
        for k, v in extras.items():
            assert kw[k] == v
        assert len(kw) == 20


# ---------------------------------------------------------------------------
# dfn / kwargs interaction with set()
# ---------------------------------------------------------------------------

class TestDfnKwargsWithSet:

    def test_set_preserves_kwargs(self):
        """set() should preserve existing kwargs and allow overrides."""
        block1 = SimpleBlock(root='/tmp/a', x=1, y=2)
        block2 = block1.set(y=3, z=4)
        assert block2.kwargs['x'] == 1
        assert block2.kwargs['y'] == 3
        assert block2.kwargs['z'] == 4

    def test_set_preserves_explicit_in_dfn(self):
        """set() should preserve explicit params in dfn."""
        block1 = SimpleBlock(root='/tmp/a', tag='orig', x=1)
        block2 = block1.set(x=2)
        assert block2.dfn['root'] == '/tmp/a'
        assert block2.dfn['tag'] == 'orig'
        assert block2.dfn['x'] == 2

    def test_set_does_not_mutate_original(self):
        """set() should not mutate the original block."""
        block1 = SimpleBlock(root='/tmp/a', x=1)
        block2 = block1.set(x=2)
        assert block1.kwargs['x'] == 1
        assert block2.kwargs['x'] == 2


# ---------------------------------------------------------------------------
# dfn / kwargs after serialization
# ---------------------------------------------------------------------------

class TestDfnKwargsSerialization:

    def test_pickle_preserves_kwargs(self):
        """kwargs should survive pickle round-trip."""
        block = SimpleBlock(root='/tmp/test', alpha='abc', num=42)
        restored = pickle.loads(pickle.dumps(block))
        assert restored.kwargs == block.kwargs

    def test_pickle_preserves_dfn(self):
        """dfn should survive pickle round-trip."""
        block = SimpleBlock(root='/tmp/test', alpha='abc', num=42, tag='mytag')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.dfn == block.dfn

    def test_legacy_kwargs_dict_in_state(self):
        """Older pickles stored a 'kwargs' dict in the state. Verify backward compat."""
        old_state = {
            'root': '/tmp/legacy',
            'kwargs': {'a': 100, 'b': 200},
            'anchored': True,
            'revision': 'test',
        }
        block = SimpleBlock.__new__(SimpleBlock)
        block.__setstate__(old_state)
        assert block.kwargs['a'] == 100
        assert block.kwargs['b'] == 200
        assert block.dfn['root'] == '/tmp/legacy'

    def test_legacy_state_dict_in_state(self):
        """Older pickles also had 'state' sub-dict. Verify backward compat."""
        old_state = {
            'root': '/tmp/legacy',
            'state': {'c': 300},
            'anchored': True,
            'revision': 'test',
        }
        block = SimpleBlock.__new__(SimpleBlock)
        block.__setstate__(old_state)
        assert block.kwargs['c'] == 300
        assert block.dfn['root'] == '/tmp/legacy'
