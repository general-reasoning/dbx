"""
Tests for build_tree(deep=...) parameter.

deep=False (default): skip subtrees whose root node is already valid.
deep=True:            unconditionally recurse and build every node.
"""
import os
import pytest
from dataclasses import dataclass

from dbx.datablocks import SAME, Datablock
from dbx.databits import quote


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.fixture
def url(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Build-tracking infrastructure
# ---------------------------------------------------------------------------

# Module-level dict so we can count how many times each block was built.
_build_counts = {}


def _reset_counts():
    _build_counts.clear()


def _record_build(label):
    _build_counts[label] = _build_counts.get(label, 0) + 1


# ---------------------------------------------------------------------------
# Datablock subclasses forming a 3-level tree:
#   Root -> Mid -> Leaf
# ---------------------------------------------------------------------------

class Leaf(Datablock):
    TOPICS = {'leaf': 'leaf.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = "'leaf'"

    def __build__(self):
        _record_build(self.var.label)
        path = self.path('leaf', ensure_dirpath=True)
        with self.fs.open(path, 'w') as f:
            f.write(f"leaf:{self.var.label}")


class Mid(Datablock):
    TOPICS = {'mid': 'mid.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = "'mid'"
        dep: str = "'none'"  # will be overridden with a quoted Leaf

    def __build__(self):
        _record_build(self.var.label)
        path = self.path('mid', ensure_dirpath=True)
        with self.fs.open(path, 'w') as f:
            f.write(f"mid:{self.var.label}")


class Root(Datablock):
    TOPICS = {'root': 'root.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = "'root'"
        child: str = "'none'"  # will be overridden with a quoted Mid

    def __build__(self):
        _record_build(self.var.label)
        path = self.path('root', ensure_dirpath=True)
        with self.fs.open(path, 'w') as f:
            f.write(f"root:{self.var.label}")


class MidGrown(Mid):
    """`Mid` with a field it never had, at the value at which the two agree: a `Mid` built is its past."""
    SPECIALIZATIONS = [Datablock.Specialization(spec={'m': 0}, topics=SAME, anchor=f'{__name__}.Mid',
                                                note="a Mid, before m")]

    @dataclass
    class VAR(Mid.VAR):
        m: int = 0


class RootWithExemptions(Root):
    """Same as Root but exempts 'child' from build_tree traversal."""
    TREE_SKIP_BUILDING = ('child',)
    TREE_SKIP_VALIDATION = ('child',)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tree(url):
    """Build a Root -> Mid -> Leaf tree and return (root, mid, leaf)."""
    leaf = Leaf(datalake=url, spec=dict(label="'L'"))
    mid_spec = dict(
        label="'M'",
        dep=quote(leaf),
    )
    mid = Mid(datalake=url, spec=mid_spec)
    root_spec = dict(
        label="'R'",
        child=quote(mid),
    )
    root = Root(datalake=url, spec=root_spec)
    return root, mid, leaf


def _make_tree_with_exemptions(url):
    """Build a RootWithExemptions -> Mid -> Leaf tree."""
    leaf = Leaf(datalake=url, spec=dict(label="'L'"))
    mid_spec = dict(
        label="'M'",
        dep=quote(leaf),
    )
    mid = Mid(datalake=url, spec=mid_spec)
    root_spec = dict(
        label="'R'",
        child=quote(mid),
    )
    root = RootWithExemptions(datalake=url, spec=root_spec)
    return root, mid, leaf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildTreeDeep:

    def test_deep_builds_all_nodes(self, url):
        """deep=True builds every node in the tree unconditionally."""
        root, mid, leaf = _make_tree(url)
        _reset_counts()

        root.build_tree(deep=True)

        # All three nodes should have been built
        assert _build_counts.get("'L'", 0) == 1, "Leaf should be built"
        assert _build_counts.get("'M'", 0) == 1, "Mid should be built"
        assert _build_counts.get("'R'", 0) == 1, "Root should be built"
        assert root.valid()

    def test_shallow_builds_all_when_none_valid(self, url):
        """When nothing is valid, shallow (default) builds everything too."""
        root, mid, leaf = _make_tree(url)
        _reset_counts()

        root.build_tree()  # deep=False by default

        assert _build_counts.get("'L'", 0) == 1
        assert _build_counts.get("'M'", 0) == 1
        assert _build_counts.get("'R'", 0) == 1
        assert root.valid()

    def test_shallow_skips_valid_subtree(self, url):
        """When the entire tree is already valid, shallow skips all children."""
        root, mid, leaf = _make_tree(url)

        # Build everything first
        root.build_tree(deep=True)
        assert root.valid()

        # Now rebuild in shallow mode — no child should be rebuilt
        _reset_counts()
        root.build_tree()  # deep=False

        # Children were valid, so they should NOT have been re-entered
        assert _build_counts.get("'L'", 0) == 0, "Leaf should be skipped (valid)"
        assert _build_counts.get("'M'", 0) == 0, "Mid should be skipped (valid)"
        # Root itself is also valid, and build() skips valid blocks
        assert _build_counts.get("'R'", 0) == 0, "Root should be skipped (valid)"

    def test_deep_rebuilds_valid_subtree(self, url):
        """deep=True descends into valid subtrees and re-invokes build on each."""
        root, mid, leaf = _make_tree(url)

        # Build everything first
        root.build_tree(deep=True)
        assert root.valid()

        # Rebuild with deep=True — it should descend into every node.
        # However build() itself also skips valid blocks, so __build__
        # won't fire again. The key test is that build_tree *enters*
        # the subtree (doesn't skip it).  We verify by checking that
        # the recursive call was made, even if build() is a no-op.
        # To test the actual descent, we invalidate just the leaf.
        os.remove(leaf.path('leaf'))
        assert not leaf.valid()

        _reset_counts()
        root.build_tree(deep=True)

        # deep=True descended into mid (even though mid is valid),
        # which descended into leaf and rebuilt it
        assert _build_counts.get("'L'", 0) == 1, "Leaf should be rebuilt"

    def test_shallow_skips_valid_mid_even_if_leaf_invalid(self, url):
        """Shallow mode checks only the immediate child; if mid is valid,
        it won't descend to discover that leaf is invalid."""
        root, mid, leaf = _make_tree(url)

        # Build everything
        root.build_tree(deep=True)
        assert root.valid()
        assert mid.valid()
        assert leaf.valid()

        # Invalidate just the leaf
        os.remove(leaf.path('leaf'))
        assert not leaf.valid()
        # But mid is still valid (its own TOPICS exists)
        assert mid.valid()

        _reset_counts()
        root.build_tree()  # shallow

        # Mid is valid → shallow skips it entirely, so leaf is NOT rebuilt
        assert _build_counts.get("'L'", 0) == 0, "Leaf should NOT be rebuilt in shallow mode"
        assert _build_counts.get("'M'", 0) == 0, "Mid should be skipped (valid)"

    def test_shallow_builds_invalid_child(self, url):
        """Shallow mode still builds a child that is itself invalid."""
        root, mid, leaf = _make_tree(url)

        # Build everything
        root.build_tree(deep=True)

        # Invalidate mid (but leave leaf valid)
        os.remove(mid.path('mid'))
        assert not mid.valid()
        assert leaf.valid()

        _reset_counts()
        root.build_tree()  # shallow

        # Mid is invalid → shallow enters it.
        # Inside mid's build_tree, leaf is valid → skipped.
        assert _build_counts.get("'M'", 0) == 1, "Mid should be rebuilt"
        assert _build_counts.get("'L'", 0) == 0, "Leaf should be skipped (valid)"

    def test_exemptions_respected_with_deep(self, url):
        """TREE_SKIP_BUILDING skip subtrees even when deep=True."""
        root, mid, leaf = _make_tree_with_exemptions(url)
        _reset_counts()

        root.build_tree(deep=True)

        # 'child' is exempted, so mid and leaf should NOT be built
        assert _build_counts.get("'M'", 0) == 0, "Mid should be exempted"
        assert _build_counts.get("'L'", 0) == 0, "Leaf should be exempted"
        # Root itself should still be built
        assert _build_counts.get("'R'", 0) == 1, "Root should be built"

    @pytest.mark.parametrize('deep', [False, True, 'try_specialize_children'])
    def test_an_exempt_child_with_a_past_is_adopted_not_built(self, url, deep):
        """TREE_SKIP_BUILDING never builds a child -- but adopts its past: that computes nothing, and the child is valid."""
        leaf = Leaf(datalake=url, spec=dict(label="'L'"))
        built = Mid(datalake=url, spec=dict(label="'M'", dep=quote(leaf))).build_tree()
        grown = MidGrown(datalake=url, spec=dict(label="'M'", dep=quote(leaf)))
        assert grown.hash != built.hash and not grown.valid()
        root = RootWithExemptions(datalake=url, spec=dict(label="'R'", child=quote(grown)))
        _reset_counts()
        root.build_tree(deep=deep)
        assert _build_counts.get("'M'", 0) == 0 and _build_counts.get("'L'", 0) == 0, "nothing of the child built"
        again = MidGrown(datalake=url, spec=dict(label="'M'", dep=quote(leaf)))
        assert again.valid() and again.redirected() and again.path('mid') == built.path('mid')

    @pytest.mark.parametrize('deep', [False, 'try_specialize_children'])
    def test_an_exempt_child_under_a_valid_one_is_adopted(self, url, deep):
        """A valid child vouches for what it reads, not what it exempts: an evaluator's still is adopted below it."""
        leaf = Leaf(datalake=url, spec=dict(label="'L'"))
        built = Mid(datalake=url, spec=dict(label="'M'", dep=quote(leaf))).build_tree()
        grown = MidGrown(datalake=url, spec=dict(label="'M'", dep=quote(leaf)))
        evaluator = RootWithExemptions(datalake=url, spec=dict(label="'E'", child=quote(grown))).build()
        assert evaluator.valid() and not grown.valid()
        root = Root(datalake=url, spec=dict(label="'R'", child=quote(evaluator)))
        _reset_counts()
        root.build_tree(deep=deep)
        assert _build_counts.get("'M'", 0) == 0 and _build_counts.get("'E'", 0) == 0
        assert MidGrown(datalake=url, spec=dict(label="'M'", dep=quote(leaf))).valid()

    def test_an_exempt_child_with_no_past_is_left_as_it_is(self, url):
        root, mid, leaf = _make_tree_with_exemptions(url)
        _reset_counts()
        root.build_tree()
        assert not mid.valid() and _build_counts.get("'M'", 0) == 0

    def test_exclude_self_with_deep(self, url):
        """exclude_self=True + deep=True builds children but not root."""
        root, mid, leaf = _make_tree(url)
        _reset_counts()

        root.build_tree(deep=True, exclude_self=True)

        assert _build_counts.get("'L'", 0) == 1, "Leaf should be built"
        assert _build_counts.get("'M'", 0) == 1, "Mid should be built"
        assert _build_counts.get("'R'", 0) == 0, "Root should NOT be built"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ═══════════════════════════════════════════════════════════════════════
#  The two skip declarations
# ═══════════════════════════════════════════════════════════════════════

class TestTreeSkipDeclarations:
    """``TREE_SKIP_BUILDING`` and ``TREE_SKIP_VALIDATION`` -- one idea, two walks.

    Both are tuples, on the base and in every subclass. Only membership is
    ever asked of them, so a set would work; declaring them alike is what
    stops a reader wondering what the difference was meant to mean.
    """

    def test_the_base_declares_both_as_empty_tuples(self):
        assert Datablock.TREE_SKIP_BUILDING == ()
        assert Datablock.TREE_SKIP_VALIDATION == ()

    @pytest.mark.pinned
    def test_the_retired_name_is_refused_not_ignored(self, url):
        """A class still declaring BUILD_TREE_EXEMPTIONS must fail loudly.

        Ignored, it would go on having its subtree built on its behalf -- and
        for the warm-start source this exists to exempt, that is a full
        training run nobody asked for. Silence is the one unacceptable
        outcome of a rename.
        """
        class Retired(Root):
            VERSION = 1
            BUILD_TREE_EXEMPTIONS = ('child',)

        root, _, _ = _make_tree_with_exemptions(url)
        retired = Retired(url=url, spec=dict(root.spec))
        with pytest.raises(AttributeError, match='TREE_SKIP_BUILDING'):
            retired.build_tree(exclude_self=True)

    @pytest.mark.pinned
    def test_a_bare_string_is_refused(self, url):
        """The tuple written without its trailing comma.

        Left alone, membership tests SUBSTRINGS: 'chi' would be exempt and
        'children' would not, and nothing would say so.
        """
        class Stringy(Root):
            VERSION = 1
            TREE_SKIP_BUILDING = 'child'

        root, _, _ = _make_tree_with_exemptions(url)
        stringy = Stringy(url=url, spec=dict(root.spec))
        with pytest.raises(TypeError, match='trailing comma'):
            stringy.build_tree(exclude_self=True)
