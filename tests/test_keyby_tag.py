"""Test that keyby='tag' without an explicit tag fails at construction time."""
import pytest
from dbx.datablocks import Datablock


class TestKeybyTagRequiresTag:

    def test_keyby_tag_without_tag_raises(self, tmp_path):
        """keyby='tag' with no tag= must raise ValueError, not recurse."""
        with pytest.raises(ValueError, match="keyby='tag' requires an explicit tag="):
            Datablock(url=str(tmp_path), keyby='tag')

    def test_keyby_tag_with_tag_succeeds(self, tmp_path):
        """keyby='tag' with an explicit tag= must not raise."""
        block = Datablock(url=str(tmp_path), keyby='tag', tag='my_tag')
        assert block.key == 'my_tag'

    def test_keyby_taghash_without_tag_succeeds(self, tmp_path):
        """keyby='taghash' (default) without tag= must still work (falls back to hash)."""
        block = Datablock(url=str(tmp_path), keyby='taghash')
        assert block.key == block.hash
