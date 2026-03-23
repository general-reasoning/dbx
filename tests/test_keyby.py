"""
Tests for the Datablock 'keyby' parameter.

keyby controls how the key component of paths is determined:
    'hash'  (default) – uses self.hash (the SHA-256 of hashstr)
    'handle'          – uses self.handle() (human-readable representation)
    None              – no key component (anchorpath IS the keypath)

Verifies:
1. Default keyby='hash' matches the old hash-based paths.
2. keyby='handle' uses handle() for keypath, anchorkey, anchorkeypath.
3. keyby=None collapses keypath to anchorpath.
4. Invalid keyby values raise ValueError.
5. keyby is recorded in the journal entry.
6. keyby survives pickle round-trip.
7. keyby works correctly with the wrapper (datablock()).
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
        assert block.keypath() == expected

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
        assert block.keypath() == expected

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
        assert block_hash.keypath() != block_handle.keypath()

    def test_build_with_handle_keyby(self, tmp_path):
        """A block with keyby='handle' should build and validate correctly."""
        block = SimpleBlock(root=str(tmp_path), keyby='handle')
        assert block.valid() is False
        block.build()
        assert block.valid() is True
        # Verify the path actually uses handle
        assert block.handle() in block.path()


# ---------------------------------------------------------------------------
# 3. keyby=None
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
        assert block.keypath() == block.anchorpath()

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
# 4. Invalid keyby
# ---------------------------------------------------------------------------

class TestKeybyInvalid:

    def test_invalid_keyby_raises(self):
        with pytest.raises(ValueError, match="keyby must be"):
            SimpleBlock(root='/tmp/dbx_test_keyby', keyby='invalid')

    def test_numeric_keyby_raises(self):
        with pytest.raises(ValueError, match="keyby must be"):
            SimpleBlock(root='/tmp/dbx_test_keyby', keyby=42)


# ---------------------------------------------------------------------------
# 5. keyby in journal entry
# ---------------------------------------------------------------------------

class TestKeybyJournal:

    def test_keyby_in_journal_on_build(self, tmp_path):
        """keyby should appear in the journal entry created during build."""
        block = SimpleBlock(root=str(tmp_path), keyby='handle')
        block.build()
        journal = block.journal()
        assert 'keyby' in journal.columns
        assert journal.iloc[0]['keyby'] == 'handle'


# ---------------------------------------------------------------------------
# 6. keyby survives serialization
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

    def test_dfn_includes_keyby(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        assert block.dfn['keyby'] == 'handle'

    def test_kwargs_excludes_keyby(self):
        block = SimpleBlock(root='/tmp/dbx_test_keyby', keyby='handle')
        assert 'keyby' not in block.kwargs


# ---------------------------------------------------------------------------
# 7. keyby with datablock() wrapper
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

    def test_wrapper_keyby_none(self):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root='/tmp/dbx_test_keyby', keyby=None)
        assert block.keyby is None
        assert block.keypath() == block.anchorpath()

    def test_wrapper_build_with_handle_keyby(self, tmp_path):
        Wrapped = datablock(SimpleDatblockable)
        block = Wrapped(root=str(tmp_path), keyby='handle')
        assert block.valid() is False
        block.build()
        assert block.valid() is True
