"""Tests for keyby='version_hash' on Datablock.

version_hash produces "version={version}/{hash}" when VERSION is set,
otherwise falls back to plain hash — mirroring taghash but keyed on version.
"""
import pytest
from dataclasses import dataclass
from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class VersionedBlock(Datablock):
    """Datablock with a VERSION class attribute."""
    VERSION = "v3"
    TOPICFILE = 'out.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        x: int = 1

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write(f"x={self.cfg.x}")

    def __read__(self, topic=None):
        with open(self.path(), 'r') as f:
            return f.read()


class UnversionedBlock(Datablock):
    """Datablock without a VERSION — version property returns None."""
    TOPICFILE = 'out.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        y: int = 2

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write(f"y={self.cfg.y}")

    def __read__(self, topic=None):
        with open(self.path(), 'r') as f:
            return f.read()


@pytest.fixture(autouse=True)
def _allow_dirty(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKeybyVersionHash:

    def test_versioned_key_format(self, tmp_path):
        """When VERSION is set, key should be 'version={version}/{hash}'."""
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash')
        assert block.key == f"version={block.version}/{block.hash}"
        assert block.key.startswith("version=v3/")

    def test_unversioned_falls_back_to_hash(self, tmp_path):
        """When VERSION is absent, key should be just the hash."""
        block = UnversionedBlock(url=str(tmp_path), keyby='version_hash')
        assert block.version is None
        assert block.key == block.hash

    def test_key_changes_with_version(self, tmp_path):
        """Two subclasses with different VERSIONs produce different keys even if hash matches."""
        class V1(Datablock):
            VERSION = "1"
            TOPICFILE = 'a.txt'
        class V2(Datablock):
            VERSION = "2"
            TOPICFILE = 'a.txt'

        b1 = V1(url=str(tmp_path), keyby='version_hash')
        b2 = V2(url=str(tmp_path), keyby='version_hash')
        # Even if hashes happen to differ, the version prefix must differ
        assert b1.key.startswith("version=1/")
        assert b2.key.startswith("version=2/")

    def test_construction_succeeds(self, tmp_path):
        """keyby='version_hash' must not raise at construction time."""
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash')
        assert block.keyby == 'version_hash'

    def test_invalid_keyby_still_rejected(self, tmp_path):
        """Unknown keyby values must still raise ValueError."""
        with pytest.raises(ValueError, match="keyby must be"):
            Datablock(url=str(tmp_path), keyby='nonexistent')

    def test_build_and_read(self, tmp_path):
        """A version_hash block can be built and read normally."""
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash', spec=dict(x=42))
        block.build()
        assert block.valid()
        assert block.read() == "x=42"

    def test_pickle_roundtrip(self, tmp_path):
        """keyby='version_hash' survives pickle serialization."""
        import pickle
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'version_hash'
        assert restored.key == block.key

    def test_deepcopy_roundtrip(self, tmp_path):
        """keyby='version_hash' survives deepcopy."""
        import copy
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash')
        restored = copy.deepcopy(block)
        assert restored.keyby == 'version_hash'
        assert restored.key == block.key
        assert restored.hash == block.hash
