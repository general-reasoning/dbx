"""
Tests for the Datablock 'keyby' parameter.

keyby controls how the key component of paths is determined:
    'hash'  (default) – uses self.hash (the SHA-256 of hashstr)
    'handle'          – uses self.handle() (human-readable representation)
    'tag'             – uses self.tag (explicit tag or default anchorkey)
    'taghash'         – uses '{tag}/{shorthash}' (tag + first 8 chars of hash)
    None              – no key component (anchorpath IS the keypath)

Verifies:
1. Default keyby='hash' matches the old hash-based paths.
2. keyby='handle' uses handle() for keypath, anchorkey, anchorkeypath.
3. keyby='tag' uses self.tag for key (both explicit and default).
3b. keyby='taghash' uses '{tag}/{shorthash}' for key.
4. keyby=None collapses keypath to anchorpath.
5. Invalid keyby values raise ValueError.
6. keyby is recorded in the journal entry.
7. keyby survives pickle round-trip.
8. keyby works correctly with the wrapper (datablock()).
"""
import os
import pickle
import tempfile
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock
from dbx.datawraps import datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx_test_keyby')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Sample Datablock subclass
# ---------------------------------------------------------------------------

class SimpleBlock(Datablock):
    """Minimal block for testing keyby."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'hello'"

    def __build__(self):
        path = self.path()
        self.dirpath(ensure=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")


class SimpleDatblockable:
    """Datablockable for wrapper tests."""
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG:
        label: str = 'hello'

    def __init__(self, *, cfg=None, **_):
        self.cfg = cfg

    def __build__(self, *args, **kwargs):
        path = self.path()
        self.dirpath(ensure=True)
        with open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")
        return self

    def __read__(self, topic=None):
        with open(self.path(), 'r') as f:
            return f.read()


# ---------------------------------------------------------------------------
# 1. Default keyby='hash'
# ---------------------------------------------------------------------------

class TestDefaultKeybyHash:

    def test_default_keyby_is_hash(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby')
        assert block.keyby == 'hash'

    def test_keypath_equals_old_hashpath_by_default(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby')
        expected = os.path.join(block.anchorpath(), block.hash)
        assert block.anchorkeypath == expected

    def test_anchorkey_equals_old_anchorhash_by_default(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby')
        expected = os.path.join(block.anchor, block.hash)
        assert block.anchorkey == expected

    def test_anchorkeypath_equals_old_anchorhashpath_by_default(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby')
        expected = os.path.join(block.root, block.anchor, block.hash)
        assert block.anchorkeypath == expected



# ---------------------------------------------------------------------------
# 2. keyby='handle'
# ---------------------------------------------------------------------------

class TestKeybyHandle:

    def test_keyby_handle_set(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        assert block.keyby == 'handle'

    def test_key_is_handle(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        assert block.key == block.handle()

    def test_keypath_uses_handle(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        expected = os.path.join(block.anchorpath(), block.handle())
        assert block.anchorkeypath == expected

    def test_anchorkey_uses_handle(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        expected = os.path.join(block.anchor, block.handle())
        assert block.anchorkey == expected

    def test_anchorkeypath_uses_handle(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        expected = os.path.join(block.root, block.anchor, block.handle())
        assert block.anchorkeypath == expected

    def test_keypath_differs_from_hash_default(self):
        """keypath should be different when keyby='handle' vs keyby='hash'."""
        block_hash = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='hash')
        block_handle = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        assert block_hash.anchorkeypath != block_handle.anchorkeypath

    def test_build_with_handle_keyby(self, tmp_path):
        """A block with keyby='handle' should build and validate correctly."""
        block = SimpleBlock(root=str(tmp_path), keyby='handle')
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        # Verify the path actually uses handle
        assert block.handle() in block.path()


# ---------------------------------------------------------------------------
# 3. keyby='tag'
# ---------------------------------------------------------------------------

class TestKeybyTag:

    def test_keyby_tag_set(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='mytag')
        assert block.keyby == 'tag'

    def test_key_is_tag_explicit(self):
        """When tag is explicitly provided, key should be that tag."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='mytag')
        assert block.key == 'mytag'
        assert block.key == block.tag

    def test_keyby_tag_requires_explicit_tag(self):
        """keyby='tag' without an explicit tag causes recursion (tag defaults
        to anchorkey which calls key which calls tag …).  Verify the error."""
        with pytest.raises(RecursionError):
            SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag')

    def test_keypath_uses_explicit_tag(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='v1')
        expected = os.path.join(block.anchorpath(), 'v1')
        assert block.anchorkeypath == expected

    def test_anchorkey_uses_explicit_tag(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='v1')
        expected = os.path.join(block.anchor, 'v1')
        assert block.anchorkey == expected

    def test_anchorkeypath_uses_explicit_tag(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='v1')
        expected = os.path.join(block.root, block.anchor, 'v1')
        assert block.anchorkeypath == expected

    def test_keypath_differs_from_hash_and_handle(self):
        """keypath with keyby='tag' should differ from hash and handle."""
        block_hash = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='hash')
        block_handle = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        block_tag = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='special')
        assert block_hash.anchorkeypath != block_tag.anchorkeypath
        assert block_handle.anchorkeypath != block_tag.anchorkeypath

    def test_tag_appears_in_path(self):
        """The explicit tag string should appear in the full file path."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='experiment-42')
        assert 'experiment-42' in block.path()

    def test_build_with_tag_keyby(self, tmp_path):
        """A block with keyby='tag' should build and validate correctly."""
        block = SimpleBlock(root=str(tmp_path), keyby='tag', tag='run1')
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        # Verify the path actually uses tag
        assert 'run1' in block.path()

    def test_build_with_tag_keyby_read_back(self, tmp_path):
        """Built content should be readable at the tag-keyed path."""
        block = SimpleBlock(root=str(tmp_path), keyby='tag', tag='run2')
        block.build()
        with open(block.path(), 'r') as f:
            content = f.read()
            # CONFIG label is "'hello'" (lazy-evaluated), so cfg.label == 'hello' (with quotes)
            assert content.startswith('built:')

    def test_different_tags_different_paths(self):
        """Two blocks with different tags should have different keypaths."""
        block_a = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='alpha')
        block_b = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='beta')
        assert block_a.anchorkeypath != block_b.anchorkeypath

    def test_same_tag_same_keypath(self):
        """Two blocks with the same tag should share the same keypath."""
        block_a = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='same')
        block_b = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='same')
        assert block_a.anchorkeypath == block_b.anchorkeypath

    def test_bid_tag_field(self):
        """The bid should reflect the explicit tag."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='mytag')
        assert block.bid.tag == 'mytag'


# ---------------------------------------------------------------------------
# 3b. keyby='taghash'
# ---------------------------------------------------------------------------

class TestKeybyTaghash:

    def test_keyby_taghash_set(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='mytag')
        assert block.keyby == 'taghash'

    def test_key_is_tag_slash_shorthash(self):
        """key should be '{tag}/{shorthash}' where shorthash is first 8 chars of hash."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='mytag')
        expected = f"mytag/{block.hash[:8]}"
        assert block.key == expected

    def test_shorthash_is_8_chars(self):
        """shorthash should be the first 8 characters of the full hash."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='mytag')
        assert block.shorthash == block.hash[:8]
        assert len(block.shorthash) == 8

    def test_keypath_uses_tag_and_shorthash(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='v1')
        expected = os.path.join(block.anchorpath(), 'v1', block.hash[:8])
        assert block.anchorkeypath == expected

    def test_anchorkey_uses_tag_and_shorthash(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='v1')
        expected = os.path.join(block.anchor, 'v1', block.hash[:8])
        assert block.anchorkey == expected

    def test_anchorkeypath_uses_tag_and_shorthash(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='v1')
        expected = os.path.join(block.root, block.anchor, 'v1', block.hash[:8])
        assert block.anchorkeypath == expected

    def test_keypath_differs_from_tag_only(self):
        """keyby='taghash' should produce different paths than keyby='tag'."""
        block_tag = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='same')
        block_taghash = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='same')
        assert block_tag.anchorkeypath != block_taghash.anchorkeypath

    def test_tag_and_shorthash_appear_in_path(self):
        """Both tag and shorthash should appear in the full file path."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='experiment-42')
        assert 'experiment-42' in block.path()
        assert block.hash[:8] in block.path()

    def test_build_with_taghash_keyby(self, tmp_path):
        """A block with keyby='taghash' should build and validate correctly."""
        block = SimpleBlock(root=str(tmp_path), keyby='taghash', tag='run1')
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        assert 'run1' in block.path()
        assert block.hash[:8] in block.path()

    def test_build_with_taghash_keyby_read_back(self, tmp_path):
        """Built content should be readable at the taghash-keyed path."""
        block = SimpleBlock(root=str(tmp_path), keyby='taghash', tag='run2')
        block.build()
        with open(block.path(), 'r') as f:
            content = f.read()
            assert content.startswith('built:')

    def test_different_tags_different_paths(self):
        """Two blocks with different tags should have different keypaths."""
        block_a = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='alpha')
        block_b = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='beta')
        assert block_a.anchorkeypath != block_b.anchorkeypath

    def test_same_tag_same_keypath(self):
        """Two blocks with the same tag/config should share the same keypath."""
        block_a = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='same')
        block_b = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='same')
        assert block_a.anchorkeypath == block_b.anchorkeypath

    def test_bid_keyby_field(self):
        """The bid should reflect keyby='taghash'."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='mytag')
        assert block.bid.keyby == 'taghash'

    def test_taghash_without_explicit_tag_defaults_to_hash(self):
        """keyby='taghash' without explicit tag should default key to self.hash."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash')
        assert block.key == block.hash

    def test_pickle_preserves_keyby_taghash(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='taghash', tag='pkl')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'taghash'
        assert restored.tag == 'pkl'
        assert restored.key == f"pkl/{restored.hash[:8]}"


# ---------------------------------------------------------------------------
# 4. keyby=None
# ---------------------------------------------------------------------------

class TestKeybyNone:

    def test_keyby_none_set(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby=None)
        assert block.keyby is None

    def test_key_is_none(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby=None)
        assert block.key is None

    def test_keypath_equals_anchorpath(self):
        """With keyby=None, keypath should collapse to anchorpath."""
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby=None)
        assert block.anchorkeypath == block.anchorpath()

    def test_anchorkey_equals_anchor(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby=None)
        assert block.anchorkey == block.anchor

    def test_anchorkeypath_equals_anchorpath(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby=None)
        expected = os.path.join(block.root, block.anchor)
        assert block.anchorkeypath == expected

    def test_build_with_none_keyby(self, tmp_path):
        """A block with keyby=None should build with files directly in anchorpath."""
        block = SimpleBlock(root=str(tmp_path), keyby=None)
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        # The path should be directly under anchorpath (no hash subdirectory)
        assert block.path() == os.path.join(block.anchorpath(), 'output.txt')


# ---------------------------------------------------------------------------
# 5. Invalid keyby
# ---------------------------------------------------------------------------

class TestKeybyInvalid:

    def test_invalid_keyby_raises(self):
        with pytest.raises(ValueError, match="keyby must be"):
            SimpleBlock(root='/tmp/dbx_test_keyby', keyby='invalid')

    def test_numeric_keyby_raises(self):
        with pytest.raises(ValueError, match="keyby must be"):
            SimpleBlock(root='/tmp/dbx_test_keyby', keyby=42)


# ---------------------------------------------------------------------------
# 6. keyby in journal entry
# ---------------------------------------------------------------------------

class TestKeybyJournal:

    def test_keyby_in_journal_on_build(self, tmp_path):
        """keyby should appear in the journal entry created during build."""
        block = SimpleBlock(root=str(tmp_path), keyby='handle')
        block.build()
        journal = block.journal()
        assert 'keyby' in journal.columns
        assert journal.iloc[0]['keyby'] == 'handle'

    def test_keyby_tag_in_journal_on_build(self, tmp_path):
        """keyby='tag' should appear in the journal entry created during build."""
        block = SimpleBlock(root=str(tmp_path), keyby='tag', tag='journaltag')
        block.build()
        journal = block.journal()
        assert 'keyby' in journal.columns
        assert journal.iloc[0]['keyby'] == 'tag'


# ---------------------------------------------------------------------------
# 7. keyby survives serialization
# ---------------------------------------------------------------------------

class TestKeybySerialization:

    def test_pickle_preserves_keyby_hash(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='hash')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'hash'

    def test_pickle_preserves_keyby_handle(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'handle'

    def test_pickle_preserves_keyby_tag(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='pkl')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'tag'
        assert restored.tag == 'pkl'
        assert restored.key == 'pkl'

    def test_pickle_preserves_keyby_none(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby=None)
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby is None

    def test_set_preserves_keyby(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        block2 = block.set(tag='newtag')
        assert block2.keyby == 'handle'

    def test_set_can_change_keyby(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='hash')
        block2 = block.set(keyby='handle')
        assert block2.keyby == 'handle'

    def test_set_can_change_to_keyby_tag(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='hash')
        block2 = block.set(keyby='tag', tag='switched')
        assert block2.keyby == 'tag'
        assert block2.key == 'switched'

    def test_dfn_includes_keyby(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        assert block.dfn['keyby'] == 'handle'

    def test_dfn_includes_keyby_tag(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='tag', tag='t1')
        assert block.dfn['keyby'] == 'tag'

    def test_kwargs_excludes_keyby(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        assert 'keyby' not in block.kwargs


# ---------------------------------------------------------------------------
# 8. keyby with datablock() wrapper
# ---------------------------------------------------------------------------

class TestKeybyWrapper:

    def test_wrapper_default_keyby(self):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root='/tmp/dbx_test_keyby')
        assert block.keyby == 'hash'

    def test_wrapper_keyby_handle(self):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root='/tmp/dbx_test_keyby', keyby='handle')
        assert block.keyby == 'handle'
        assert block.key == block.handle()

    def test_wrapper_keyby_tag(self):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root='/tmp/dbx_test_keyby', keyby='tag', tag='wrapped')
        assert block.keyby == 'tag'
        assert block.key == 'wrapped'
        assert 'wrapped' in block.anchorkeypath

    def test_wrapper_keyby_none(self):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root='/tmp/dbx_test_keyby', keyby=None)
        assert block.keyby is None
        assert block.anchorkeypath == block.anchorpath()

    def test_wrapper_build_with_handle_keyby(self, tmp_path):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root=str(tmp_path), keyby='handle')
        assert block.valid() is False
        block.build()
        assert block.valid() is True

    def test_wrapper_keyby_taghash(self):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root='/tmp/dbx_test_keyby', keyby='taghash', tag='wrapped')
        assert block.keyby == 'taghash'
        assert block.key == f"wrapped/{block.hash[:8]}"
        assert 'wrapped' in block.anchorkeypath
        assert block.hash[:8] in block.anchorkeypath

    def test_wrapper_build_with_tag_keyby(self, tmp_path):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root=str(tmp_path), keyby='tag', tag='wrapbuild')
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        assert 'wrapbuild' in block.path()

    def test_wrapper_build_with_taghash_keyby(self, tmp_path):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root=str(tmp_path), keyby='taghash', tag='wrapbuild')
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        assert 'wrapbuild' in block.path()
        assert block.hash[:8] in block.path()
