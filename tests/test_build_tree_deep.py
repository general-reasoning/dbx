"""
Tests for build_tree(deep=...) parameter.

deep=False (default): skip subtrees whose root node is already valid.
deep=True:            unconditionally recurse and build every node.
"""
import os
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, quote


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
    TOPICFILE = 'leaf.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'leaf'"

    def __build__(self):
        _record_build(self.cfg.label)
        self.dirpath(ensure=True)
        with self.fs.open(self.path(), 'w') as f:
            f.write(f"leaf:{self.cfg.label}")


class Mid(Datablock):
    TOPICFILE = 'mid.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'mid'"
        dep: str = "'none'"  # will be overridden with a quoted Leaf

    def __build__(self):
        _record_build(self.cfg.label)
        self.dirpath(ensure=True)
        with self.fs.open(self.path(), 'w') as f:
            f.write(f"mid:{self.cfg.label}")


class Root(Datablock):
    TOPICFILE = 'root.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'root'"
        child: str = "'none'"  # will be overridden with a quoted Mid

    def __build__(self):
        _record_build(self.cfg.label)
        self.dirpath(ensure=True)
        with self.fs.open(self.path(), 'w') as f:
            f.write(f"root:{self.cfg.label}")


class RootWithExemptions(Root):
    """Same as Root but exempts 'child' from build_tree traversal."""
    BUILD_TREE_EXEMPTIONS = ('child',)
    VALIDATE_CFG_EXEMPTIONS = ('child',)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tree(url):
    """Build a Root -> Mid -> Leaf tree and return (root, mid, leaf)."""
    leaf = Leaf(url=url, spec=dict(label="'L'"))
    mid_spec = dict(
        label="'M'",
        dep=quote(leaf),
    )
    mid = Mid(url=url, spec=mid_spec)
    root_spec = dict(
        label="'R'",
        child=quote(mid),
    )
    root = Root(url=url, spec=root_spec)
    return root, mid, leaf


def _make_tree_with_exemptions(url):
    """Build a RootWithExemptions -> Mid -> Leaf tree."""
    leaf = Leaf(url=url, spec=dict(label="'L'"))
    mid_spec = dict(
        label="'M'",
        dep=quote(leaf),
    )
    mid = Mid(url=url, spec=mid_spec)
    root_spec = dict(
        label="'R'",
        child=quote(mid),
    )
    root = RootWithExemptions(url=url, spec=root_spec)
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
        os.remove(leaf.path())
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
        os.remove(leaf.path())
        assert not leaf.valid()
        # But mid is still valid (its own TOPICFILE exists)
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
        os.remove(mid.path())
        assert not mid.valid()
        assert leaf.valid()

        _reset_counts()
        root.build_tree()  # shallow

        # Mid is invalid → shallow enters it.
        # Inside mid's build_tree, leaf is valid → skipped.
        assert _build_counts.get("'M'", 0) == 1, "Mid should be rebuilt"
        assert _build_counts.get("'L'", 0) == 0, "Leaf should be skipped (valid)"

    def test_exemptions_respected_with_deep(self, url):
        """BUILD_TREE_EXEMPTIONS skip subtrees even when deep=True."""
        root, mid, leaf = _make_tree_with_exemptions(url)
        _reset_counts()

        root.build_tree(deep=True)

        # 'child' is exempted, so mid and leaf should NOT be built
        assert _build_counts.get("'M'", 0) == 0, "Mid should be exempted"
        assert _build_counts.get("'L'", 0) == 0, "Leaf should be exempted"
        # Root itself should still be built
        assert _build_counts.get("'R'", 0) == 1, "Root should be built"

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
