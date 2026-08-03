"""
Backward compatibility for the CONFIG -> VAR rename.

``Datablock.CONFIG`` used to be the name of the spec dataclass and ``.cfg`` /
``.config`` the name of the instantiated-and-resolved property.  They are now
``Datablock.VAR`` and ``.var``.  A subclass written against the old names must
keep working unchanged:

    * ``class CONFIG(Datablock.CONFIG)`` still resolves, and ``__setstate__``
      binds it to ``self.VAR`` so the rest of the machinery sees it;
    * ``.cfg`` and ``.config`` remain aliases of ``.var``;
    * neither alias changes the block's identity (spec/hash/key).

The ``validate_cfg`` -> ``validate_vars`` rename keeps no alias, but state
serialized under the old key is still read.  ``VALIDATE_CFG_EXEMPTIONS`` is
retired outright in favour of ``TREE_SKIP_VALIDATION``: silently ignoring it
would re-enable the very validation it was suppressing, so it raises.
"""
import copy
import os
import pickle
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.fixture
def url(tmp_path):
    return str(tmp_path)


class LegacyBlock(Datablock):
    """A subclass still written in terms of the deprecated CONFIG name."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'legacy'"
        n: int = 3

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}:{self.config.n}")

    def __read__(self, topic):
        with open(self.path('output'), 'r') as f:
            return f.read()


class ModernBlock(Datablock):
    """The same block written in terms of VAR."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = "'legacy'"
        n: int = 3

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.var.label}:{self.var.n}")

    def __read__(self, topic):
        with open(self.path('output'), 'r') as f:
            return f.read()


class ParentBlock(Datablock):
    """A block whose spec holds another Datablock, so validation descends."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        child: object = None

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write('parent')


class DerivedLegacyBlock(LegacyBlock):
    """A subclass of a legacy block that does not redeclare the spec."""
    pass


class OverridingBlock(LegacyBlock):
    """A modern VAR override on top of a legacy CONFIG base must win."""

    @dataclass
    class VAR(Datablock.VAR):
        label: str = "'overridden'"
        n: int = 7


def test_legacy_CONFIG_is_bound_to_VAR(url):
    b = LegacyBlock(url=url)
    assert b.VAR is LegacyBlock.CONFIG
    assert b.var.label == "'legacy'"
    assert b.var.n == 3


def test_cfg_and_config_alias_var(url):
    b = LegacyBlock(url=url)
    assert b.cfg is b.var
    assert b.config is b.var


def test_modern_VAR_is_not_shadowed_by_the_alias(url):
    b = ModernBlock(url=url)
    assert b.VAR is ModernBlock.VAR
    assert b.var.label == "'legacy'"


def test_legacy_and_modern_specs_agree(url):
    legacy = LegacyBlock(url=url)
    modern = ModernBlock(url=url)
    assert legacy.spec == modern.spec


def test_legacy_CONFIG_inherited_through_a_subclass(url):
    b = DerivedLegacyBlock(url=url)
    assert b.VAR is LegacyBlock.CONFIG
    assert b.var.n == 3


def test_VAR_override_beats_an_inherited_legacy_CONFIG(url):
    b = OverridingBlock(url=url)
    assert b.VAR is OverridingBlock.VAR
    assert b.var.label == "'overridden'"
    assert b.var.n == 7


@pytest.mark.parametrize('roundtrip', [
    copy.deepcopy,
    lambda b: pickle.loads(pickle.dumps(b)),
    lambda b: type(b)(**b.__getstate__()),
])
def test_legacy_block_survives_serialization(url, roundtrip):
    original = LegacyBlock(url=url, spec=dict(label="'x'", n=5))
    restored = roundtrip(original)
    assert restored.VAR is LegacyBlock.CONFIG
    assert restored.var.label == "'x'"
    assert restored.var.n == 5
    assert restored.hash == original.hash
    assert restored.key == original.key


def test_legacy_block_builds(url):
    b = LegacyBlock(url=url)
    b.build()
    assert b.valid()
    assert b.read('output') == "built:'legacy':3"


# ---------------------------------------------------------------------------
# validate_cfg -> validate_vars
# ---------------------------------------------------------------------------

def _legacy_state(url):
    """A dfn/state dict as it was recorded before the rename."""
    state = ModernBlock(url=url).__getstate__()
    assert state.pop('validate_vars') is True
    state['validate_cfg'] = False
    return state


def test_legacy_validate_cfg_is_honored_by_init(url):
    """Reconstructing from a dfn recorded before the rename still works."""
    restored = ModernBlock(**_legacy_state(url))
    assert restored.validate_vars is False
    # The legacy key must not survive as a dynamic kwarg: it does not reach
    # norm(), but it would be re-serialized forever and drift quote()/cite()
    # -- and hence the journal -- from an otherwise identical block.
    assert 'validate_cfg' not in restored.__getstate__()
    assert 'validate_cfg' not in restored.kwargs
    assert restored.__getstate__()['validate_vars'] is False
    assert 'validate_cfg=' not in restored.quote()  # NB: tmp_path contains the test name
    assert restored.hash == ModernBlock(url=url).hash


def test_legacy_validate_cfg_is_honored_by_setstate(url):
    """Unpickling bypasses __init__ and hands the old key straight to state."""
    restored = ModernBlock.__new__(ModernBlock)
    restored.__setstate__(_legacy_state(url))
    assert restored.validate_vars is False
    assert 'validate_cfg' not in restored.__getstate__()


def test_validate_vars_is_not_part_of_identity(url):
    """It changes what build() checks, not what the block *is*.

    norm() is built from url/anchor/hash and spec only, so neither the
    explicit params nor any dynamic kwarg reaches the hash.
    """
    checked = ModernBlock(url=url)
    unchecked = ModernBlock(url=url, validate_vars=False)
    assert checked.norm() == unchecked.norm()
    assert checked.hash == unchecked.hash
    assert checked.key == unchecked.key


# ---------------------------------------------------------------------------
# VALIDATE_CFG_EXEMPTIONS is retired
# ---------------------------------------------------------------------------

def test_retired_exemptions_attr_raises(url):
    class Stale(ParentBlock):
        VALIDATE_CFG_EXEMPTIONS = ('child',)

    with pytest.raises(AttributeError, match='TREE_SKIP_VALIDATION'):
        Stale(url=url, spec=dict(child=ModernBlock(url=url)))


def test_TREE_SKIP_VALIDATION_replaces_it(url):
    class Skipping(ParentBlock):
        TREE_SKIP_VALIDATION = ('child',)

    spec = dict(child=ModernBlock(url=url))
    # The child is never built, so without the exemption it reports invalid.
    assert ParentBlock(url=url, spec=spec).valid_var() == {'child': False}
    assert Skipping(url=url, spec=spec).valid_var() == {}
    assert Skipping(url=url, spec=spec).valid_tree() == {}
