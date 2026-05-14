"""
Tests for Datablock.flag(message, *, inline=False).

Verifies:
  1. A journal entry with event='flag' is written
  2. inline=False stores context as a file path pointing to the message
  3. inline=True stores context as the literal message string
  4. Multiple flags from separate instances produce separate journal rows
  5. .flag() returns self (fluent API)
"""
import os
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
# Test Datablock
# ---------------------------------------------------------------------------

class FlagBlock(Datablock):
    TOPICFILE = 'out.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        pass


def _make(tmp_path, **kw):
    return FlagBlock(url=str(tmp_path), **kw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlagJournalEntry:

    def test_flag_creates_journal_entry(self, tmp_path):
        b = _make(tmp_path)
        b.flag("hello")
        j = b.journal()
        flag_rows = j[j['event'] == 'flag']
        assert len(flag_rows) == 1

    def test_flag_event_value(self, tmp_path):
        b = _make(tmp_path)
        b.flag("test msg")
        j = b.journal()
        assert 'flag' in j['event'].values

    def test_flag_records_hash(self, tmp_path):
        b = _make(tmp_path)
        b.flag("check hash")
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        assert row['hash'] == b.hash


class TestFlagContextNotInline:
    """inline=False (default): context is written to a file; journal stores the path."""

    def test_context_is_file_path(self, tmp_path):
        b = _make(tmp_path)
        b.flag("stored in file")
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        ctx = row['context']
        assert ctx is not None
        # Should be a filesystem path, not the raw message
        assert ctx != "stored in file"
        assert os.path.exists(ctx)

    def test_context_file_contains_message(self, tmp_path):
        b = _make(tmp_path)
        b.flag("payload text")
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        with open(row['context']) as f:
            assert f.read().strip() == "payload text"


class TestFlagContextInline:
    """inline=True: context is the literal message string in the journal."""

    def test_context_is_literal_message(self, tmp_path):
        b = _make(tmp_path)
        b.flag("inline msg", inline=True)
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        assert row['context'] == "inline msg"

    def test_no_context_file_written_for_inline(self, tmp_path):
        b = _make(tmp_path)
        b.flag("no file", inline=True)
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        ctx = row['context']
        # The value is the raw string, not a path that exists on disk
        assert not os.path.exists(ctx)


class TestFlagMultipleInstances:
    """Each Datablock instance has a unique dt, so separate instances
    produce separate journal parquet files that aggregate correctly."""

    def test_multiple_flags_from_separate_instances(self, tmp_path):
        for msg in ("first", "second", "third"):
            b = _make(tmp_path)
            b.flag(msg)
        j = _make(tmp_path).journal()
        flag_rows = j[j['event'] == 'flag']
        assert len(flag_rows) == 3

    def test_mixed_inline_modes_across_instances(self, tmp_path):
        b1 = _make(tmp_path)
        b1.flag("file-ctx")

        b2 = _make(tmp_path)
        b2.flag("inline-ctx", inline=True)

        j = _make(tmp_path).journal()
        rows = j[j['event'] == 'flag'].sort_values('datetime').reset_index(drop=True)
        assert len(rows) == 2
        # First: file path
        assert os.path.exists(rows.loc[0, 'context'])
        # Second: literal string
        assert rows.loc[1, 'context'] == "inline-ctx"


class TestFlagReturnsSelf:

    def test_returns_self(self, tmp_path):
        b = _make(tmp_path)
        result = b.flag("msg")
        assert result is b

    def test_returns_self_inline(self, tmp_path):
        b = _make(tmp_path)
        result = b.flag("msg", inline=True)
        assert result is b
