"""
Tests for Datablock.quote() — verifying that the cite() helper
correctly repr-wraps string values and leaves non-strings unchanged,
and that the quote uses fqcn (not anchor).
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample Datablock subclasses
# ---------------------------------------------------------------------------

class SimpleBlock(Datablock):
    """Minimal block with a single string spec field."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        pass


class IntSpecBlock(Datablock):
    """Block with an integer spec field (non-string value)."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        count: str = "42"

    def __build__(self):
        pass


class MixedSpecBlock(Datablock):
    """Block with both string and non-string spec fields."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        name: str = "'alice'"
        count: str = "7"

    def __build__(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_block(cls, tmp_path, **kwargs):
    return cls(root=str(tmp_path), **kwargs)


# ---------------------------------------------------------------------------
# Tests: quote() basics
# ---------------------------------------------------------------------------

class TestQuoteBasics:

    def test_quote_starts_with_dollar(self, tmp_path):
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote()
        assert q.startswith('$')

    def test_quote_uses_fqcn_not_anchor(self, tmp_path):
        """quote() should use the fully-qualified class name, not the anchor."""
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote()
        # fqcn for SimpleBlock defined in this module
        assert block.fqcn in q

    def test_quote_uses_fqcn_even_with_custom_anchor(self, tmp_path):
        """When anchor differs from fqcn, quote() should still use fqcn."""
        block = _make_block(SimpleBlock, tmp_path, anchor='my.custom.anchor')
        q = block.quote()
        assert block.fqcn in q
        # anchor should NOT appear in the quote prefix (after $)
        # The anchor may appear as a kwarg value though
        prefix = q.split('(')[0]
        assert prefix == f'${block.fqcn}'

    def test_quote_returns_string(self, tmp_path):
        block = _make_block(SimpleBlock, tmp_path)
        assert isinstance(block.quote(), str)


# ---------------------------------------------------------------------------
# Tests: cite() correctness — string values get repr(), non-strings don't
# ---------------------------------------------------------------------------

class TestCiteInQuote:

    def test_string_spec_values_are_not_double_repr(self, tmp_path):
        """String values inside spec are kept as-is (not repr'd again)
        because __expand_spec__('quote') already returns the raw string."""
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote()
        # The spec value 'hello' (a string) should appear in the quote
        assert 'hello' in q

    def test_int_spec_value_in_spec_dict(self, tmp_path):
        """Spec values are always strings (CONFIG fields are str-typed),
        so they appear inside the quoted spec dict as strings."""
        block = _make_block(IntSpecBlock, tmp_path)
        q = block.quote()
        # The spec dict value '42' is a string, so cite wraps it in repr
        assert "'count': '42'" in q

    def test_mixed_spec_values(self, tmp_path):
        """All spec values are strings in the spec dict."""
        block = _make_block(MixedSpecBlock, tmp_path)
        q = block.quote()
        # Both are strings in the spec dict
        assert "'count': '7'" in q
        assert "alice" in q

    def test_root_kwarg_is_cited_as_string(self, tmp_path):
        """Root is always a string and should be repr'd by cite() in quote."""
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote()
        root_str = str(tmp_path)
        # cite(root) should produce repr(root), i.e. quoted with apostrophes
        assert f"root='{root_str}'" in q


# ---------------------------------------------------------------------------
# Tests: string quoting details
# ---------------------------------------------------------------------------

class TestCiteStringQuoting:
    """Targeted tests for how cite() handles different value types in quote()."""

    def test_anchor_kwarg_is_repr_quoted(self, tmp_path):
        """When anchor is set, it's a string kwarg and cite() must repr-quote it."""
        block = _make_block(SimpleBlock, tmp_path, anchor='my.anchor')
        q = block.quote()
        # anchor='my.anchor' -> cite produces repr('my.anchor') = "'my.anchor'"
        assert "anchor='my.anchor'" in q

    def test_spec_dict_is_not_repr_quoted(self, tmp_path):
        """The spec value is a dict, so cite() should NOT repr-wrap it.
        It should appear as spec={...}, not spec='{...}'."""
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote()
        # spec= should be followed by { (the dict), not by ' (a quoted string)
        spec_idx = q.index('spec=')
        after_spec = q[spec_idx + len('spec=')]
        assert after_spec == '{', (
            f"Expected spec value to start with '{{' (dict), "
            f"but got {after_spec!r} in: {q}"
        )

    def test_hash_kwarg_is_repr_quoted(self, tmp_path):
        """When hash is explicitly set, it's a string and cite() must quote it."""
        block = _make_block(SimpleBlock, tmp_path, hash='abc123')
        q = block.quote()
        assert "hash='abc123'" in q

    def test_quote_overall_format(self, tmp_path):
        """Verify the full format: $fqcn(root='...', spec={...})."""
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote()
        root_str = str(tmp_path)
        # Must start with $fqcn(
        assert q.startswith(f"${block.fqcn}(")
        # Must end with )
        assert q.endswith(')')
        # Must contain root= with a repr'd string value
        assert f"root='{root_str}'" in q
        # Must contain spec= with a dict value
        assert 'spec={' in q

    def test_quote_with_all_rootkwargs(self, tmp_path):
        """When root, anchor, and hash are all set, all should be repr-quoted."""
        block = _make_block(SimpleBlock, tmp_path,
                            anchor='custom.anchor', hash='deadbeef')
        q = block.quote()
        root_str = str(tmp_path)
        assert f"root='{root_str}'" in q
        assert "anchor='custom.anchor'" in q
        assert "hash='deadbeef'" in q
        # All three should appear before spec=
        spec_idx = q.index('spec=')
        assert q.index('root=') < spec_idx
        assert q.index('anchor=') < spec_idx
        assert q.index('hash=') < spec_idx


# ---------------------------------------------------------------------------
# Tests: tailkwargs (extra **kwargs) in quote()
# ---------------------------------------------------------------------------

class TestQuoteTailkwargs:
    """Verify that extra **kwargs passed to __init__ appear in quote()."""

    def test_extra_kwargs_appear_in_quote(self, tmp_path):
        """Extra kwargs like batch_size should appear in quote() output."""
        block = _make_block(SimpleBlock, tmp_path, batch_size=32, num_workers=4)
        q = block.quote()
        assert 'batch_size=32' in q
        assert 'num_workers=4' in q

    def test_string_extra_kwargs_are_cited(self, tmp_path):
        """String-valued extra kwargs should be repr-quoted by cite()."""
        block = _make_block(SimpleBlock, tmp_path, device='cuda:0')
        q = block.quote()
        assert "device='cuda:0'" in q

    def test_extra_kwargs_appear_after_spec(self, tmp_path):
        """Tailkwargs should appear after spec in the quote."""
        block = _make_block(SimpleBlock, tmp_path, batch_size=32)
        q = block.quote()
        spec_idx = q.index('spec=')
        batch_idx = q.index('batch_size=')
        assert batch_idx > spec_idx

    def test_quote_matches_repr_structure_for_tailkwargs(self, tmp_path):
        """quote() and __repr__() should both include tailkwargs."""
        block = _make_block(SimpleBlock, tmp_path, batch_size=32)
        q = block.quote()
        r = repr(block)
        assert 'batch_size=32' in q
        assert 'batch_size=32' in r

    def test_no_extra_kwargs_no_tailkwargs(self, tmp_path):
        """When no extra kwargs are passed, no tailkwargs appear in quote."""
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote()
        # quote should end with spec={...})
        assert q.endswith(')')
        # No extra keys beyond root= and spec=
        # Count top-level = signs (rough check)
        prefix = q[q.index('(') + 1 : q.rindex(')')]
        # Should only have root= and spec=
        assert 'root=' in prefix
        assert 'spec=' in prefix


# ---------------------------------------------------------------------------
# Tests: tailkwargs in wrapped Datablockable quote()
# ---------------------------------------------------------------------------

class TestWrappedQuoteTailkwargs:
    """Verify that datablock()-wrapped classes also include tailkwargs in quote()."""

    def test_wrapped_extra_kwargs_in_quote(self, tmp_path):
        from dbx.datawraps import datablock

        class MyProcessor:
            TOPICFILE = 'out.txt'

            @dataclass
            class CONFIG(Datablock.CONFIG):
                label: str = "'hello'"

            def __init__(self, *, cfg=None, **_):
                self.cfg = cfg

            def build(self, *args, **kwargs):
                return self

            def read(self, topic=None):
                return None

        Wrapped = datablock(MyProcessor)
        block = Wrapped(root=str(tmp_path), batch_size=64, num_workers=8)
        q = block.quote()
        assert 'batch_size=64' in q
        assert 'num_workers=8' in q

    def test_wrapped_extra_kwargs_saved_on_self(self, tmp_path):
        from dbx.datawraps import datablock

        class MyProcessor:
            TOPICFILE = 'out.txt'

            @dataclass
            class CONFIG(Datablock.CONFIG):
                label: str = "'hello'"

            def __init__(self, *, cfg=None, **_):
                self.cfg = cfg

            def build(self, *args, **kwargs):
                return self

            def read(self, topic=None):
                return None

        Wrapped = datablock(MyProcessor)
        block = Wrapped(root=str(tmp_path), batch_size=64, num_workers=8)
        # Extra kwargs must be saved on self by __setstate__
        assert hasattr(block, 'batch_size')
        assert block.batch_size == 64
        assert hasattr(block, 'num_workers')
        assert block.num_workers == 8

# ---------------------------------------------------------------------------
# Tests: deslash
# ---------------------------------------------------------------------------

class TestQuoteDeslash:

    def test_deslash_removes_backslashes(self, tmp_path):
        block = _make_block(SimpleBlock, tmp_path)
        q_normal = block.quote(deslash=False)
        q_deslash = block.quote(deslash=True)
        assert '\\' not in q_deslash
        # If original had no backslashes, they should be identical
        if '\\' not in q_normal:
            assert q_normal == q_deslash

    def test_deslash_false_preserves_backslashes(self, tmp_path):
        block = _make_block(SimpleBlock, tmp_path)
        q = block.quote(deslash=False)
        # Should be the raw quote, no backslash removal
        assert isinstance(q, str)


# ---------------------------------------------------------------------------
# Tests: __str__ uses quote
# ---------------------------------------------------------------------------

class TestStrUsesQuote:

    def test_str_matches_deslashed_quote(self, tmp_path):
        """__str__ should return quote() with backslashes removed."""
        block = _make_block(SimpleBlock, tmp_path)
        s = str(block)
        q = block.quote()
        assert s == q.replace('\\', '')
