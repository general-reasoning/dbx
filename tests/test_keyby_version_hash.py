"""Tests for keyby='version_hash', 'tag_hash', and 'tag_version_hash' on Datablock.

- version_hash:      "version={version}/{hash[:8]}", falls back to hash
- tag_hash:          alias for 'taghash' — "{tag}/{hash[:8]}", falls back to hash
- tag_version_hash:  "{tag}/version={version}/{hash[:8]}", skipping None parts,
                     falls back to hash when both tag and version are None
"""
import copy
import pickle
import pytest
from dataclasses import dataclass
from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class VersionedBlock(Datablock):
    """Datablock with a VERSION class attribute."""
    VERSION = "v3"
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        x: int = 1

    def __build__(self):
        path = self.path('out', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"x={self.cfg.x}")

    def __read__(self, topic):
        with open(self.path('out'), 'r') as f:
            return f.read()


class UnversionedBlock(Datablock):
    """Datablock without a VERSION — version property returns None."""
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        y: int = 2

    def __build__(self):
        path = self.path('out', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"y={self.cfg.y}")

    def __read__(self, topic):
        with open(self.path('out'), 'r') as f:
            return f.read()


@pytest.fixture(autouse=True)
def _allow_dirty(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ===========================================================================
# version_hash
# ===========================================================================

class TestKeybyVersionHash:

    def test_versioned_key_format(self, tmp_path):
        """When VERSION is set, key should be 'version={version}/{hash[:8]}'."""
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash')
        assert block.key == f"version={block.version}/{block.hash[:8]}"
        assert block.key.startswith("version=v3/")

    def test_unversioned_falls_back_to_hash(self, tmp_path):
        """When VERSION is absent, key should be just the hash."""
        block = UnversionedBlock(url=str(tmp_path), keyby='version_hash')
        assert block.version is None
        assert block.key == block.hash

    def test_key_changes_with_version(self, tmp_path):
        """Two subclasses with different VERSIONs produce different key prefixes."""
        class V1(Datablock):
            VERSION = "1"
            TOPICS = {'a': 'a.txt'}
        class V2(Datablock):
            VERSION = "2"
            TOPICS = {'a': 'a.txt'}

        b1 = V1(url=str(tmp_path), keyby='version_hash')
        b2 = V2(url=str(tmp_path), keyby='version_hash')
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
        assert block.valid(topic=None)
        assert block.read('out') == "x=42"

    def test_pickle_roundtrip(self, tmp_path):
        """keyby='version_hash' survives pickle serialization."""
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'version_hash'
        assert restored.key == block.key

    def test_deepcopy_roundtrip(self, tmp_path):
        """keyby='version_hash' survives deepcopy."""
        block = VersionedBlock(url=str(tmp_path), keyby='version_hash')
        restored = copy.deepcopy(block)
        assert restored.keyby == 'version_hash'
        assert restored.key == block.key
        assert restored.hash == block.hash


# ===========================================================================
# tag_hash (alias for taghash)
# ===========================================================================

class TestKeybyTagHash:

    def test_tag_hash_matches_taghash(self, tmp_path):
        """tag_hash and taghash must produce identical keys."""
        a = VersionedBlock(url=str(tmp_path), keyby='taghash', tag='mytag')
        b = VersionedBlock(url=str(tmp_path), keyby='tag_hash', tag='mytag')
        assert a.key == b.key

    def test_tag_hash_without_tag_falls_back(self, tmp_path):
        """tag_hash without tag= falls back to hash, just like taghash."""
        block = VersionedBlock(url=str(tmp_path), keyby='tag_hash')
        assert block.key == block.hash

    def test_tag_hash_with_tag(self, tmp_path):
        """tag_hash with tag= produces '{tag}/{hash[:8]}'."""
        block = VersionedBlock(url=str(tmp_path), keyby='tag_hash', tag='run1')
        assert block.key == f"run1/{block.hash[:8]}"

    def test_tag_hash_pickle_roundtrip(self, tmp_path):
        block = VersionedBlock(url=str(tmp_path), keyby='tag_hash', tag='t')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'tag_hash'
        assert restored.key == block.key


# ===========================================================================
# tag_version_hash
# ===========================================================================

class TestKeybyTagVersionHash:

    def test_tag_and_version(self, tmp_path):
        """Both tag and version present: '{tag}/version={version}/{hash[:8]}'."""
        block = VersionedBlock(url=str(tmp_path), keyby='tag_version_hash', tag='exp1')
        expected = f"exp1/version=v3/{block.hash[:8]}"
        assert block.key == expected

    def test_tag_only(self, tmp_path):
        """Tag present, no version: '{tag}/{hash[:8]}'."""
        block = UnversionedBlock(url=str(tmp_path), keyby='tag_version_hash', tag='exp2')
        assert block.version is None
        expected = f"exp2/{block.hash[:8]}"
        assert block.key == expected

    def test_version_only(self, tmp_path):
        """No tag, version present: 'version={version}/{hash[:8]}'."""
        block = VersionedBlock(url=str(tmp_path), keyby='tag_version_hash')
        expected = f"version=v3/{block.hash[:8]}"
        assert block.key == expected

    def test_neither_tag_nor_version(self, tmp_path):
        """Neither tag nor version: falls back to full hash."""
        block = UnversionedBlock(url=str(tmp_path), keyby='tag_version_hash')
        assert block.version is None
        assert block.key == block.hash

    def test_build_and_read(self, tmp_path):
        """tag_version_hash block can build and read."""
        block = VersionedBlock(url=str(tmp_path), keyby='tag_version_hash', tag='b', spec=dict(x=7))
        block.build()
        assert block.valid(topic=None)
        assert block.read('out') == "x=7"

    def test_pickle_roundtrip(self, tmp_path):
        block = VersionedBlock(url=str(tmp_path), keyby='tag_version_hash', tag='t')
        restored = pickle.loads(pickle.dumps(block))
        assert restored.keyby == 'tag_version_hash'
        assert restored.key == block.key

    def test_deepcopy_roundtrip(self, tmp_path):
        block = VersionedBlock(url=str(tmp_path), keyby='tag_version_hash', tag='t')
        restored = copy.deepcopy(block)
        assert restored.key == block.key
