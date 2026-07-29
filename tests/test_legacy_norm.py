"""Tests for string quoting in ``norm()`` and the :attr:`Datablock.LEGACY_NORM` escape hatch.

``__repr_from_kwargs__`` is shared by three callers, two of which are identity:

    norm()      -> hashstr -> hash -> key -> storage path
    supernorm() -> superhashstr -> superhash
    __repr__()  -> display / journal only

so quoting strings there unconditionally would move every existing hash and
orphan every artifact already stored under the old one. The split is:

* ``__repr__`` always quotes -- it is not an input to ``hashstr``.
* ``norm``/``supernorm`` quote only when ``LEGACY_NORM`` is False (the default,
  i.e. every new subclass). Subclasses whose artifacts predate the change set
  it to True and keep the exact bytes their hashes were computed from.

The pinned hash below is the load-bearing test: it was computed with dbx at the
commit BEFORE quoting existed, so it fails if the legacy path ever drifts.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


PIN_URL = '/tmp/dbx-legacy-norm-pin'


class LegacyBlock(Datablock):
    """Stands in for a subclass with artifacts already built and keyed."""
    LEGACY_NORM = True
    VERSION = 7
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = 'lbl'
        size: int = 3

    def __build__(self):
        pass


class ModernBlock(Datablock):
    """No LEGACY_NORM -- inherits the default False, so strings are quoted."""
    VERSION = 7
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = 'lbl'
        size: int = 3

    def __build__(self):
        pass


def _pin(cls):
    return cls(url=PIN_URL, spec=dict(label='lbl', size=3)).set(tag='pin')


class TestDefault:

    def test_datablock_default_is_not_legacy(self):
        """New subclasses must get the corrected form without opting in."""
        assert Datablock.LEGACY_NORM is False
        assert ModernBlock.LEGACY_NORM is False

    def test_legacy_is_opt_in_and_inherited(self):
        class Child(LegacyBlock):
            pass
        assert Child.LEGACY_NORM is True


class TestLegacyNormIsByteStable:

    def test_pinned_hash(self):
        """Computed with dbx BEFORE string quoting existed. Do not update."""
        assert _pin(LegacyBlock).hash == (
            '067daa07f7bc70c43d923133b42e018753a77a91d1de949abc199ad4b03f329f'
        )

    def test_url_is_not_quoted(self):
        norm = _pin(LegacyBlock).norm()
        assert f"url={PIN_URL}" in norm
        assert f"url={PIN_URL!r}" not in norm

    def test_supernorm_url_is_not_quoted(self):
        assert f"url={PIN_URL}" in _pin(LegacyBlock).supernorm()

    def test_non_string_spec_value_is_repr_d_twice(self):
        """int 3 renders "'3'" -- the collision the legacy form is stuck with."""
        assert "'size': '3'" in _pin(LegacyBlock).norm()


class TestModernNormQuotesStrings:

    def test_url_is_quoted(self):
        norm = _pin(ModernBlock).norm()
        assert f"url={PIN_URL!r}" in norm

    def test_supernorm_url_is_quoted(self):
        assert f"url={PIN_URL!r}" in _pin(ModernBlock).supernorm()

    def test_non_string_spec_value_is_repr_d_once(self):
        assert "'size': 3" in _pin(ModernBlock).norm()

    def test_identity_differs_from_legacy(self):
        """Same url, spec, version and topics -- only the flag differs.

        This is why every pre-existing subclass had to be marked: leaving one
        unmarked silently re-keys it.
        """
        assert _pin(ModernBlock).hash != _pin(LegacyBlock).hash
        assert _pin(ModernBlock).superhash != _pin(LegacyBlock).superhash


class TestSpecValueCollision:
    """``n=5`` vs ``n='5'`` collided onto one norm under the legacy form."""

    @dataclass
    class _C(Datablock.CONFIG):
        v: object = None

    def _pair(self, cls):
        return (cls(url=PIN_URL, spec=dict(v=5)),
                cls(url=PIN_URL, spec=dict(v='5')))

    def test_legacy_collides(self):
        class L(Datablock):
            LEGACY_NORM = True
            TOPICS = {'output': 'output.txt'}
            CONFIG = TestSpecValueCollision._C
            def __build__(self): pass
        a, b = self._pair(L)
        assert a.norm() == b.norm()
        assert a.hash == b.hash

    def test_modern_distinguishes(self):
        class M(Datablock):
            TOPICS = {'output': 'output.txt'}
            CONFIG = TestSpecValueCollision._C
            def __build__(self): pass
        a, b = self._pair(M)
        assert a.norm() != b.norm()
        assert a.hash != b.hash
        assert "'v': 5" in a.norm()
        assert "'v': '5'" in b.norm()


class TestReprAlwaysQuotes:
    """__repr__ is not hashed, so it quotes for legacy blocks too."""

    @pytest.mark.parametrize('cls', [LegacyBlock, ModernBlock])
    def test_url_is_quoted(self, cls):
        assert f"url={PIN_URL!r}" in repr(_pin(cls))

    @pytest.mark.parametrize('cls', [LegacyBlock, ModernBlock])
    def test_string_tailkwarg_is_quoted(self, cls):
        """`tag` is a tailkwarg, and __repr_from_kwargs__ receives those too --
        hence the flag is named quote_strs, not quote_rootkwargs."""
        assert "tag='pin'" in repr(_pin(cls))


class TestQuoteStrsFlagIsOptIn:
    """The helper itself must default to the unquoted form."""

    def test_default_leaves_strings_bare(self):
        b = _pin(LegacyBlock)
        assert b.__repr_from_kwargs__({'a': 'x'}, anchor=None) == '(a=x)'

    def test_flag_quotes_strings_only(self):
        b = _pin(LegacyBlock)
        out = b.__repr_from_kwargs__({'a': 'x', 'n': 3}, anchor=None,
                                     quote_strs=True)
        assert out == "(a='x', n=3)"
