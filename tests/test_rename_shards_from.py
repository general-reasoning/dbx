"""Tests for UNSAFE_rename_from and UNSAFE_rename_blocks_from.

Verifies:
1. UNSAFE_rename_from copies data from a source anchor into the block.
2. UNSAFE_rename_blocks_from dispatches UNSAFE_rename_from to each block
   via _rename_block_from_callable (validates the callable name fix).
"""
import os
import pytest
from dataclasses import dataclass
from unittest.mock import patch, call

from dbx.datablocks import Datablock, Datastack


# ---------------------------------------------------------------------------
# Test blocks
# ---------------------------------------------------------------------------

class RenameBlock(Datablock):
    """Block for rename tests."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        value: str = 'default'

    def __build__(self):
        path = self.path(ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(self.cfg.value)

    def __read__(self, topic=None):
        with open(self.path(), 'r') as f:
            return f.read()


class RenameStackBlock(Datablock):
    """Block for stack rename tests."""
    TOPICS = {'block': 'block.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        idx: int = 0

    def __build__(self):
        path = self.path(ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write(f"block:{self.cfg.idx}")

    def __read__(self, topic=None):
        with open(self.path(), 'r') as f:
            return f.read()


class RenameStack(Datastack):
    """Stack for rename tests."""
    TOPICS = {'stack': 'stack.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        n_blocks_total: int = 2

    @property
    def n_blocks(self):
        return self.cfg.n_blocks_total

    def __block__(self, idx):
        return RenameStackBlock(url=self.url, spec=dict(idx=idx))

    def blocks(self):
        return [self.__block__(i) for i in range(self.n_blocks)]


@pytest.fixture(autouse=True)
def _allow_dirty(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# UNSAFE_rename_from (Datablock)
# ---------------------------------------------------------------------------

class TestUNSAFERenameFrom:

    def test_rename_from_invokes_copy_from(self, tmp_path):
        """UNSAFE_rename_from should call UNSAFE_copy_from with the correct anchorkeypath."""
        block = RenameBlock(url=str(tmp_path), spec=dict(value='hello'))
        old_anchor = 'old.module.Block'
        expected_path = block._anchorkeypath(old_anchor)

        with patch.object(block, 'UNSAFE_copy_from', return_value=block) as mock_copy:
            result = block.UNSAFE_rename_from(old_anchor, OVERRIDE=True)

        mock_copy.assert_called_once_with(
            expected_path, overwrite=False, topicpaths=None, validate=True, copy_dirpath=False
        )
        assert result is block

    def test_rename_from_passes_overwrite(self, tmp_path):
        """overwrite flag should be forwarded to UNSAFE_copy_from."""
        block = RenameBlock(url=str(tmp_path), spec=dict(value='hello'))

        with patch.object(block, 'UNSAFE_copy_from', return_value=block) as mock_copy:
            block.UNSAFE_rename_from('some.anchor', OVERRIDE=True, overwrite=True)

        _, kwargs = mock_copy.call_args
        assert kwargs['overwrite'] is True

    def test_rename_from_passes_copy_dirpath(self, tmp_path):
        """copy_dirpath flag should be forwarded to UNSAFE_copy_from."""
        block = RenameBlock(url=str(tmp_path), spec=dict(value='hello'))

        with patch.object(block, 'UNSAFE_copy_from', return_value=block) as mock_copy:
            block.UNSAFE_rename_from('some.anchor', OVERRIDE=True, copy_dirpath=True)

        _, kwargs = mock_copy.call_args
        assert kwargs['copy_dirpath'] is True

    def test_rename_from_skipped_without_override(self, tmp_path):
        """Without OVERRIDE, the rename should be skipped (returns self)."""
        block = RenameBlock(url=str(tmp_path), spec=dict(value='hello'))

        with patch('builtins.input', return_value='n'):
            with patch.object(block, 'UNSAFE_copy_from') as mock_copy:
                result = block.UNSAFE_rename_from('some.anchor')

        mock_copy.assert_not_called()
        assert result is block


# ---------------------------------------------------------------------------
# UNSAFE_rename_blocks_from (Datastack)
# ---------------------------------------------------------------------------

class TestUNSAFERenameBlocksFrom:

    def test_rename_blocks_from_calls_rename_from_on_each_block(self, tmp_path):
        """UNSAFE_rename_blocks_from should call UNSAFE_rename_from on each block.

        This test validates that:
        - _rename_block_from_callable is correctly referenced (no NameError)
        - Each block's UNSAFE_rename_from is called with the right anchor
        """
        stack = RenameStack(url=str(tmp_path), spec=dict(n_blocks_total=3))
        old_anchor = 'old.module.Shard'

        with patch.object(RenameStackBlock, 'UNSAFE_rename_from', return_value=None) as mock_rename:
            result = stack.UNSAFE_rename_blocks_from(old_anchor, OVERRIDE=True)

        assert result is stack
        assert mock_rename.call_count == 3
        for c in mock_rename.call_args_list:
            assert c[0][0] == old_anchor  # first positional arg is the anchor
            assert c[1]['OVERRIDE'] is True

    def test_rename_blocks_from_forwards_overwrite(self, tmp_path):
        """overwrite should be forwarded to each block's UNSAFE_rename_from."""
        stack = RenameStack(url=str(tmp_path), spec=dict(n_blocks_total=2))

        with patch.object(RenameStackBlock, 'UNSAFE_rename_from', return_value=None) as mock_rename:
            stack.UNSAFE_rename_blocks_from('old.anchor', OVERRIDE=True, overwrite=True)

        for c in mock_rename.call_args_list:
            assert c[1]['overwrite'] is True

    def test_rename_blocks_from_forwards_copy_dirpath(self, tmp_path):
        """copy_dirpath should be forwarded to each block's UNSAFE_rename_from."""
        stack = RenameStack(url=str(tmp_path), spec=dict(n_blocks_total=2))

        with patch.object(RenameStackBlock, 'UNSAFE_rename_from', return_value=None) as mock_rename:
            stack.UNSAFE_rename_blocks_from('old.anchor', OVERRIDE=True, copy_dirpath=True)

        for c in mock_rename.call_args_list:
            assert c[1]['copy_dirpath'] is True

    def test_rename_blocks_from_returns_self(self, tmp_path):
        """UNSAFE_rename_blocks_from should return the stack itself."""
        stack = RenameStack(url=str(tmp_path), spec=dict(n_blocks_total=2))

        with patch.object(RenameStackBlock, 'UNSAFE_rename_from', return_value=None):
            result = stack.UNSAFE_rename_blocks_from('old.anchor', OVERRIDE=True)

        assert result is stack
