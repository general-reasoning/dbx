"""Tests for UNSAFE_allowed() and its use in UNSAFE_clear()."""
import os
import pytest
from unittest.mock import patch, MagicMock
from dbx.datablocks import UNSAFE_allowed, Datablock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# UNSAFE_allowed unit tests
# ---------------------------------------------------------------------------

class TestUNSAFEAllowed:

    def test_override_true_skips_prompt(self):
        """OVERRIDE=True should return True without prompting."""
        with patch('builtins.input') as mock_input:
            result = UNSAFE_allowed("test_action", OVERRIDE=True)
        mock_input.assert_not_called()
        assert result is True

    def test_user_confirms_y(self):
        """User typing 'y' should return True."""
        with patch('builtins.input', return_value='y') as mock_input:
            result = UNSAFE_allowed("test_action")
        mock_input.assert_called_once()
        assert result is True

    def test_user_confirms_uppercase_Y(self):
        """User typing 'Y' should also return True (case-insensitive)."""
        with patch('builtins.input', return_value='Y'):
            result = UNSAFE_allowed("test_action")
        assert result is True

    def test_user_declines_n(self):
        """User typing 'n' should return False."""
        with patch('builtins.input', return_value='n'):
            result = UNSAFE_allowed("test_action")
        assert result is False

    def test_user_declines_empty(self):
        """Empty input (just Enter) should return False (default is N)."""
        with patch('builtins.input', return_value=''):
            result = UNSAFE_allowed("test_action")
        assert result is False

    def test_user_declines_random_string(self):
        """Any non-'y' input should return False."""
        with patch('builtins.input', return_value='yes'):
            result = UNSAFE_allowed("test_action")
        assert result is False

    def test_prompt_contains_what(self):
        """The prompt shown to the user should include the 'what' description."""
        with patch('builtins.input', return_value='n') as mock_input:
            UNSAFE_allowed("my_special_action")
        prompt = mock_input.call_args[0][0]
        assert "my_special_action" in prompt


# ---------------------------------------------------------------------------
# UNSAFE_clear integration tests
# ---------------------------------------------------------------------------

class MyBlock(Datablock):
    TOPICFILE = 'output.txt'
    def __build__(self):
        pass


class TestUNSAFEClear:

    def _make_block(self):
        return MyBlock(root='/tmp/dbx_test_clear')

    def test_clear_with_override_proceeds(self):
        """UNSAFE_clear(OVERRIDE=True) must not prompt and must attempt clearing."""
        block = self._make_block()
        # Patch out the actual filesystem removal so we don't touch disk
        with patch.object(block, '_write_journal_entry'), \
             patch('fsspec.url_to_fs') as mock_url_to_fs:
            mock_fs = MagicMock()
            mock_url_to_fs.return_value = (mock_fs, None)
            result = block.UNSAFE_clear(OVERRIDE=True)
        # Should return self (fluent API)
        assert result is block

    def test_clear_aborts_when_user_declines(self):
        """UNSAFE_clear() should return self without touching fs when user says n."""
        block = self._make_block()
        with patch('builtins.input', return_value='n'), \
             patch('fsspec.url_to_fs') as mock_url_to_fs:
            result = block.UNSAFE_clear()
        # fs operations must never be called
        mock_url_to_fs.assert_not_called()
        assert result is block

    def test_clear_proceeds_when_user_confirms(self):
        """UNSAFE_clear() should attempt clearing when user types 'y'."""
        block = self._make_block()
        with patch('builtins.input', return_value='y'), \
             patch.object(block, '_write_journal_entry'), \
             patch('fsspec.url_to_fs') as mock_url_to_fs:
            mock_fs = MagicMock()
            mock_url_to_fs.return_value = (mock_fs, None)
            result = block.UNSAFE_clear()
        assert result is block
        # The filesystem rm should have been reached
        mock_fs.rm.assert_called()

    def test_clear_override_does_not_call_input(self):
        """With OVERRIDE=True, input() must never be called."""
        block = self._make_block()
        with patch('builtins.input') as mock_input, \
             patch.object(block, '_write_journal_entry'), \
             patch('fsspec.url_to_fs') as mock_url_to_fs:
            mock_fs = MagicMock()
            mock_url_to_fs.return_value = (mock_fs, None)
            block.UNSAFE_clear(OVERRIDE=True)
        mock_input.assert_not_called()
