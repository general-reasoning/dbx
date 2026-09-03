"""
Tests for Datablock.note(message=None, event='note', *, inline=False).

Verifies:
  1. A journal entry with the given event is written
  2. The journal parquet filename is prefixed with '{event}-'
  3. inline=False stores message as a file path pointing to the text
  4. inline=True stores message as the literal string
  5. Multiple notes from separate instances produce separate journal rows
  6. .note() returns self (fluent API)
  7. Default event is 'note'
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

class NoteBlock(Datablock):
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        pass


def _make(tmp_path, **kw):
    return NoteBlock(url=str(tmp_path), **kw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoteDatajournalEntry:

    def test_note_creates_journal_entry(self, tmp_path):
        b = _make(tmp_path)
        b.note("hello", event="flag")
        j = b.journal()
        rows = j[j['event'] == 'flag']
        assert len(rows) == 1

    def test_note_default_event_is_note(self, tmp_path):
        """note('msg') without explicit event should use event='note'."""
        b = _make(tmp_path)
        b.note("default event test")
        j = b.journal()
        assert 'note' in j['event'].values

    def test_note_custom_event(self, tmp_path):
        b = _make(tmp_path)
        b.note("test msg", event="myevent")
        j = b.journal()
        assert 'myevent' in j['event'].values

    def test_note_records_hash(self, tmp_path):
        b = _make(tmp_path)
        b.note("check hash", event="flag")
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        assert row['hash'] == b.hash

    def test_note_review_event(self, tmp_path):
        b = _make(tmp_path)
        b.note("needs review", event="review")
        j = b.journal()
        rows = j[j['event'] == 'review']
        assert len(rows) == 1


class TestNoteParquetPrefix:
    """note() should produce a parquet file prefixed with '{event}-'."""

    def test_parquet_filename_starts_with_event(self, tmp_path):
        b = _make(tmp_path)
        b.note("hello", event="flag")
        dbx_dir = str(tmp_path)
        all_parquets = []
        for root, dirs, files in os.walk(dbx_dir):
            if os.path.basename(root) != b.hash:
                continue
            parent = os.path.basename(os.path.dirname(root))
            if parent != 'journal':
                continue
            for f in files:
                if f.endswith('.parquet'):
                    all_parquets.append(f)
        flag_parquets = [f for f in all_parquets if f.startswith('flag-')]
        assert len(flag_parquets) >= 1, f"Expected flag- prefixed parquets, got: {all_parquets}"

    def test_default_event_prefix_is_note(self, tmp_path):
        b = _make(tmp_path)
        b.note("saved")
        dbx_dir = str(tmp_path)
        all_parquets = []
        for root, dirs, files in os.walk(dbx_dir):
            if os.path.basename(root) != b.hash:
                continue
            parent = os.path.basename(os.path.dirname(root))
            if parent != 'journal':
                continue
            for f in files:
                if f.endswith('.parquet'):
                    all_parquets.append(f)
        note_parquets = [f for f in all_parquets if f.startswith('note-')]
        assert len(note_parquets) >= 1, f"Expected note- prefixed parquets, got: {all_parquets}"

    def test_custom_event_prefix(self, tmp_path):
        b = _make(tmp_path)
        b.note("saved", event="checkpoint")
        dbx_dir = str(tmp_path)
        all_parquets = []
        for root, dirs, files in os.walk(dbx_dir):
            if os.path.basename(root) != b.hash:
                continue
            parent = os.path.basename(os.path.dirname(root))
            if parent != 'journal':
                continue
            for f in files:
                if f.endswith('.parquet'):
                    all_parquets.append(f)
        cp_parquets = [f for f in all_parquets if f.startswith('checkpoint-')]
        assert len(cp_parquets) >= 1, f"Expected checkpoint- prefixed parquets, got: {all_parquets}"


class TestNoteMessageNotInline:
    """inline=False (default): message is written to a file; journal stores the path."""

    def test_message_is_file_path(self, tmp_path):
        b = _make(tmp_path)
        b.note("stored in file", event="flag")
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        msg = row['note']
        assert msg is not None
        # Should be a filesystem path, not the raw message
        assert msg != "stored in file"
        assert os.path.exists(msg)

    def test_message_file_contains_message(self, tmp_path):
        b = _make(tmp_path)
        b.note("payload text", event="flag")
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        with open(row['note']) as f:
            assert f.read().strip() == "payload text"


class TestNoteMessageInline:
    """inline=True: message is the literal message string in the journal."""

    def test_message_is_literal_message(self, tmp_path):
        b = _make(tmp_path)
        b.note("inline msg", event="flag", inline=True)
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        assert row['note'] == "inline msg"

    def test_no_message_file_written_for_inline(self, tmp_path):
        b = _make(tmp_path)
        b.note("no file", event="flag", inline=True)
        j = b.journal()
        row = j[j['event'] == 'flag'].iloc[-1]
        msg = row['note']
        # The value is the raw string, not a path that exists on disk
        assert not os.path.exists(msg)


class TestNoteNoMessage:
    """note() with no message should have None/NaN message."""

    def test_no_message(self, tmp_path):
        b = _make(tmp_path)
        b.note(event="ping")
        j = b.journal()
        row = j[j['event'] == 'ping'].iloc[-1]
        import pandas as pd
        assert pd.isna(row.get('message', None)) or row.get('note') is None

    def test_no_message_default_event(self, tmp_path):
        b = _make(tmp_path)
        b.note()
        j = b.journal()
        row = j[j['event'] == 'note'].iloc[-1]
        import pandas as pd
        assert pd.isna(row.get('message', None)) or row.get('note') is None


class TestNoteMultipleInstances:
    """Each Datablock instance has a unique dt, so separate instances
    produce separate journal parquet files that aggregate correctly."""

    def test_multiple_notes_from_separate_instances(self, tmp_path):
        for msg in ("first", "second", "third"):
            b = _make(tmp_path)
            b.note(msg, event="flag")
        j = _make(tmp_path).journal()
        flag_rows = j[j['event'] == 'flag']
        assert len(flag_rows) == 3

    def test_mixed_inline_modes_across_instances(self, tmp_path):
        b1 = _make(tmp_path)
        b1.note("file-msg", event="flag")

        b2 = _make(tmp_path)
        b2.note("inline-msg", event="flag", inline=True)

        j = _make(tmp_path).journal()
        rows = j[j['event'] == 'flag'].sort_values('datetime').reset_index(drop=True)
        assert len(rows) == 2
        # First: file path
        assert os.path.exists(rows.loc[0, 'note'])
        # Second: literal string
        assert rows.loc[1, 'note'] == "inline-msg"


class TestNoteReturnsSelf:

    def test_returns_self(self, tmp_path):
        b = _make(tmp_path)
        result = b.note("msg", event="flag")
        assert result is b

    def test_returns_self_inline(self, tmp_path):
        b = _make(tmp_path)
        result = b.note("msg", event="flag", inline=True)
        assert result is b

    def test_returns_self_no_message(self, tmp_path):
        b = _make(tmp_path)
        result = b.note(event="ping")
        assert result is b

    def test_returns_self_bare(self, tmp_path):
        b = _make(tmp_path)
        result = b.note()
        assert result is b
