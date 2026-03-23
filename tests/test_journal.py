"""
Tests for dbx.journal() / Datablock.Journal():

1. FileNotFoundError is raised when the journal directory does not exist.
2. A JournalFrame is returned when the directory exists but has no parquet files.
3. entry=N correctly returns the Nth JournalEntry.
4. Regression: root must not be passed positionally (would silently land in
   the 'entry' slot of Datablock.Journal before the fix).
"""
import os
import datetime
import pytest
import pandas as pd

import dbx.datablocks as dbxmod
from dbx.datablocks import journal, Datablock, JournalFrame, JournalEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _journal_dir(root, cls) -> str:
    """Return the journal directory path for a given class and root."""
    anchor = cls.__module__ + "." + cls.__name__
    return Datablock._journalanchorpath(anchor, root, ensure=False)


def _write_fake_journal_entry(journal_dir: str, hash_val: str = "abc123", event: str = "build"):
    """Write a minimal parquet journal entry into journal_dir."""
    os.makedirs(journal_dir, exist_ok=True)
    now = datetime.datetime.now()
    df = pd.DataFrame([{
        'hash':     hash_val,
        'datetime': now,
        'event':    event,
        'anchor':   'test.FakeBlock',
        'root':     '/tmp/dbx_test',
    }])
    path = os.path.join(journal_dir, f"{event}.parquet")
    df.to_parquet(path)
    return path


# A minimal Datablock subclass to use as the anchor
class FakeBlock(Datablock):
    def __build__(self):
        pass


# ---------------------------------------------------------------------------
# 1. FileNotFoundError when journal directory is missing
# ---------------------------------------------------------------------------

class TestJournalMissingDir:

    def test_raises_filenotfounderror(self, tmp_path, monkeypatch):
        """journal() must raise FileNotFoundError when the journal dir doesn't exist."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / "nonexistent_root")
        # Do NOT create the directory

        with pytest.raises(FileNotFoundError, match="Journal directory not found"):
            journal(FakeBlock, root=root)

    def test_error_message_contains_class_name(self, tmp_path, monkeypatch):
        """The FileNotFoundError message should mention the class anchor."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / "nonexistent_root")

        with pytest.raises(FileNotFoundError) as exc_info:
            journal(FakeBlock, root=root)

        assert "FakeBlock" in str(exc_info.value)

    def test_error_message_contains_path(self, tmp_path, monkeypatch):
        """The FileNotFoundError message should include the resolved path."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / "nonexistent_root")
        expected_dir = _journal_dir(root, FakeBlock)

        with pytest.raises(FileNotFoundError) as exc_info:
            journal(FakeBlock, root=root)

        assert expected_dir in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Empty journal dir → JournalFrame(None)
# ---------------------------------------------------------------------------

class TestJournalEmptyDir:

    def test_returns_journalframe_when_no_parquets(self, tmp_path, monkeypatch):
        """An existing but empty journal dir should return a JournalFrame wrapping None."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        # Create the journal directory with no parquet files
        journal_dir = _journal_dir(root, FakeBlock)
        os.makedirs(journal_dir, exist_ok=True)

        result = journal(FakeBlock, root=root)
        assert isinstance(result, JournalFrame)


# ---------------------------------------------------------------------------
# 3. entry= parameter returns a JournalEntry
# ---------------------------------------------------------------------------

class TestJournalEntryParam:

    def test_entry_returns_journal_entry(self, tmp_path, monkeypatch):
        """journal(cls, entry=0, root=...) should return the 0th JournalEntry."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        journal_dir = _journal_dir(root, FakeBlock)
        _write_fake_journal_entry(journal_dir, hash_val="deadbeef", event="build")

        result = journal(FakeBlock, entry=0, root=root)
        assert isinstance(result, JournalEntry)

    def test_entry_none_returns_journalframe(self, tmp_path, monkeypatch):
        """journal(cls, root=...) with no entry should return the full JournalFrame."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        journal_dir = _journal_dir(root, FakeBlock)
        _write_fake_journal_entry(journal_dir, hash_val="cafebabe", event="build")

        result = journal(FakeBlock, root=root)
        assert isinstance(result, JournalFrame)
        assert len(result) == 1

    def test_entry_value_matches_row(self, tmp_path, monkeypatch):
        """The returned JournalEntry should contain the correct hash."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        journal_dir = _journal_dir(root, FakeBlock)
        _write_fake_journal_entry(journal_dir, hash_val="myhash42", event="build")

        entry = journal(FakeBlock, entry=0, root=root)
        assert entry.get('hash') == "myhash42"


# ---------------------------------------------------------------------------
# 4. Regression: root must not silently become 'entry'
# ---------------------------------------------------------------------------

class TestJournalRootPassedCorrectly:

    def test_root_kwarg_is_not_mistaken_for_entry(self, tmp_path, monkeypatch):
        """
        Before the fix, `journal(cls, root)` passed root as the second positional
        arg to Datablock.Journal, which would interpret it as 'entry'.  This test
        confirms root is forwarded as a keyword arg and the directory check uses
        the actual root path.
        """
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        # Use a non-existent path so we get a clear FileNotFoundError (not a
        # TypeError about entry being a string, which is what the old bug caused).
        bad_root = str(tmp_path / "missing")
        with pytest.raises(FileNotFoundError):
            journal(FakeBlock, root=bad_root)


# A second Datablock subclass for prefix-filtering tests
class OtherBlock(Datablock):
    def __build__(self):
        pass


def _write_prefixed_entry(journal_dir, classname, hash_val="abc", event="build"):
    """Write a parquet entry with a classname-prefixed filename."""
    os.makedirs(journal_dir, exist_ok=True)
    now = datetime.datetime.now()
    dt = now.isoformat().replace(' ', '-').replace(':', '-')
    filename = f"{classname}-{hash_val}-{dt}"
    df = pd.DataFrame([{
        'hash': hash_val,
        'datetime': now,
        'event': event,
        'anchor': classname,
        'root': '/tmp/dbx_test',
    }])
    path = os.path.join(journal_dir, f"{filename}.parquet")
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# 5. prefix filtering
# ---------------------------------------------------------------------------

class TestJournalPrefixFiltering:

    def test_no_prefix_returns_all(self, tmp_path, monkeypatch):
        """Without prefix, Journal() returns entries from all classes."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        jdir = _journal_dir(root, FakeBlock)
        fake_anchor = FakeBlock.__module__ + "." + FakeBlock.__name__
        other_anchor = OtherBlock.__module__ + "." + OtherBlock.__name__
        _write_prefixed_entry(jdir, fake_anchor, hash_val="aaa")
        _write_prefixed_entry(jdir, other_anchor, hash_val="bbb")

        result = journal(FakeBlock, root=root)
        assert isinstance(result, JournalFrame)
        assert len(result) == 2

    def test_prefix_filters_by_classname(self, tmp_path, monkeypatch):
        """With prefix=anchor, only matching entries are returned."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        jdir = _journal_dir(root, FakeBlock)
        fake_anchor = FakeBlock.__module__ + "." + FakeBlock.__name__
        other_anchor = OtherBlock.__module__ + "." + OtherBlock.__name__
        _write_prefixed_entry(jdir, fake_anchor, hash_val="aaa")
        _write_prefixed_entry(jdir, other_anchor, hash_val="bbb")

        result = Datablock.Journal(
            fake_anchor, root=root, prefix=fake_anchor + "-",
        )
        assert isinstance(result, JournalFrame)
        assert len(result) == 1
        assert result.iloc[0]['hash'] == "aaa"

    def test_prefix_no_match_returns_empty(self, tmp_path, monkeypatch):
        """Prefix that matches nothing yields JournalFrame(None)."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        jdir = _journal_dir(root, FakeBlock)
        fake_anchor = FakeBlock.__module__ + "." + FakeBlock.__name__
        _write_prefixed_entry(jdir, fake_anchor, hash_val="aaa")

        result = Datablock.Journal(
            fake_anchor, root=root, prefix="nonexistent.Class-",
        )
        assert isinstance(result, JournalFrame)
        # No matching files → wraps None → length 0
        assert len(result) == 0

    def test_prefix_none_is_default(self, tmp_path, monkeypatch):
        """prefix=None (default) behaves identically to no prefix."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        jdir = _journal_dir(root, FakeBlock)
        fake_anchor = FakeBlock.__module__ + "." + FakeBlock.__name__
        _write_prefixed_entry(jdir, fake_anchor, hash_val="xyz")

        result = Datablock.Journal(fake_anchor, root=root, prefix=None)
        assert len(result) == 1
