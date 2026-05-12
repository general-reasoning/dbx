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
    return Datablock._dbxanchorpathx(root, anchor, 'journal', fqcn=anchor)


def _write_fake_journal_entry(journal_dir: str, hash_val: str = "abc123", event: str = "build"):
    """Write a minimal parquet journal entry into journal_dir."""
    import fsspec
    fs, jdir = fsspec.url_to_fs(journal_dir)
    fs.makedirs(jdir, exist_ok=True)
    now = datetime.datetime.now()
    df = pd.DataFrame([{
        'hash':     hash_val,
        'datetime': now,
        'event':    event,
        'anchor':   'test.FakeBlock',
        'url':      '/tmp/dbx_test',
    }])
    path = os.path.join(jdir, f"{event}.parquet")
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
            journal(FakeBlock, url=root)

    def test_error_message_contains_class_name(self, tmp_path, monkeypatch):
        """The FileNotFoundError message should mention the class anchor."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / "nonexistent_root")

        with pytest.raises(FileNotFoundError) as exc_info:
            journal(FakeBlock, url=root)

        assert "FakeBlock" in str(exc_info.value)

    def test_error_message_contains_path(self, tmp_path, monkeypatch):
        """The FileNotFoundError message should include the resolved path."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path / "nonexistent_root")
        expected_dir = _journal_dir(root, FakeBlock)

        with pytest.raises(FileNotFoundError) as exc_info:
            journal(FakeBlock, url=root)

        # journal() without fqcn checks {anchor}/.dbx, not the fqcn-qualified path
        anchor = FakeBlock.__module__ + "." + FakeBlock.__name__
        assert anchor in str(exc_info.value)
        assert ".dbx" in str(exc_info.value)


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
        import fsspec
        fs, jdir = fsspec.url_to_fs(journal_dir)
        fs.makedirs(jdir, exist_ok=True)

        result = journal(FakeBlock, url=root)
        assert isinstance(result, JournalFrame)


# ---------------------------------------------------------------------------
# 3. entry= parameter returns a JournalEntry
# ---------------------------------------------------------------------------

class TestJournalEntryParam:

    def test_entry_returns_journal_entry(self, tmp_path, monkeypatch):
        """journal(cls, entry=0, url=...) should return the 0th JournalEntry."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        journal_dir = _journal_dir(root, FakeBlock)
        _write_fake_journal_entry(journal_dir, hash_val="deadbeef", event="build")

        result = journal(FakeBlock, entry=0, url=root)
        assert isinstance(result, JournalEntry)

    def test_entry_none_returns_journalframe(self, tmp_path, monkeypatch):
        """journal(cls, url=...) with no entry should return the full JournalFrame."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        journal_dir = _journal_dir(root, FakeBlock)
        _write_fake_journal_entry(journal_dir, hash_val="cafebabe", event="build")

        result = journal(FakeBlock, url=root)
        assert isinstance(result, JournalFrame)
        assert len(result) == 1

    def test_entry_value_matches_row(self, tmp_path, monkeypatch):
        """The returned JournalEntry should contain the correct hash."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        journal_dir = _journal_dir(root, FakeBlock)
        _write_fake_journal_entry(journal_dir, hash_val="myhash42", event="build")

        entry = journal(FakeBlock, entry=0, url=root)
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
            journal(FakeBlock, url=bad_root)


# A second Datablock subclass for classname-filtering tests
class OtherBlock(Datablock):
    def __build__(self):
        pass


def _write_journal_in_hash_dir(journal_dir, classname, hash_val="abc", event="build"):
    """Write a parquet entry inside a hash subdirectory, mirroring _dbxanchorhashpathx layout."""
    hash_dir = os.path.join(journal_dir, hash_val)
    import fsspec
    fs, hdir = fsspec.url_to_fs(hash_dir)
    fs.makedirs(hdir, exist_ok=True)
    now = datetime.datetime.now()
    dt = now.isoformat().replace(' ', '-').replace(':', '-')
    filename = f"{classname}-journal-{hash_val}-{dt}"
    df = pd.DataFrame([{
        'hash': hash_val,
        'datetime': now,
        'event': event,
        'anchor': classname,
        'url': '/tmp/dbx_test',
    }])
    _, hdir = fsspec.url_to_fs(hash_dir)
    path = os.path.join(hdir, f"{filename}.parquet")
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# 5. fqcn filtering
# ---------------------------------------------------------------------------

class TestJournalFqcnSubdirectories:

    def test_returns_all_fqcn_entries(self, tmp_path, monkeypatch):
        """Journal() returns entries from all fqcn subdirectories under an anchor."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        fake_anchor = FakeBlock.__module__ + "." + FakeBlock.__name__
        other_anchor = OtherBlock.__module__ + "." + OtherBlock.__name__
        # Write entries into separate fqcn-qualified journal dirs
        fake_jdir = Datablock._dbxanchorpathx(root, fake_anchor, 'journal', fqcn=fake_anchor)
        other_jdir = Datablock._dbxanchorpathx(root, fake_anchor, 'journal', fqcn=other_anchor)
        _write_journal_in_hash_dir(fake_jdir, fake_anchor, hash_val="aaa")
        _write_journal_in_hash_dir(other_jdir, other_anchor, hash_val="bbb")

        result = Datablock.Journal(fake_anchor, url=root)
        assert isinstance(result, JournalFrame)
        assert len(result) == 2

    def test_single_fqcn_found(self, tmp_path, monkeypatch):
        """Journal() finds entries when only one fqcn subdirectory exists."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        fake_anchor = FakeBlock.__module__ + "." + FakeBlock.__name__
        fake_jdir = Datablock._dbxanchorpathx(root, fake_anchor, 'journal', fqcn=fake_anchor)
        _write_journal_in_hash_dir(fake_jdir, fake_anchor, hash_val="xyz")

        result = Datablock.Journal(fake_anchor, url=root)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 6. journal() finds entries regardless of anchor form
# ---------------------------------------------------------------------------

class BuildableBlock(Datablock):
    """Block that actually builds a file so a journal entry gets written."""
    TOPICFILE = 'output.txt'

    def __build__(self):
        self.dirpath(ensure=True)
        with open(self.path(), 'w') as f:
            f.write('built')


class TestJournalAnchorForms:

    def test_default_anchor_journal(self, tmp_path, monkeypatch):
        """journal() finds entries when using the default anchor (fqcn)."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        block = BuildableBlock(url=root)
        block.build()
        result = block.journal()
        assert isinstance(result, JournalFrame)
        assert len(result) >= 1

    def test_custom_anchor_journal(self, tmp_path, monkeypatch):
        """journal() finds entries when using a custom anchor."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        block = BuildableBlock(url=root, anchor='my.custom.anchor')
        block.build()
        result = block.journal()
        assert isinstance(result, JournalFrame)
        assert len(result) >= 1

    def test_static_journal_default_anchor(self, tmp_path, monkeypatch):
        """Datablock.Journal(anchor) finds entries for default anchor."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        block = BuildableBlock(url=root)
        block.build()
        anchor = block.anchor
        result = Datablock.Journal(anchor, url=root)
        assert isinstance(result, JournalFrame)
        assert len(result) >= 1

    def test_static_journal_custom_anchor(self, tmp_path, monkeypatch):
        """Datablock.Journal(anchor) finds entries for custom anchor."""
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)
        block = BuildableBlock(url=root, anchor='custom.anchor')
        block.build()
        result = Datablock.Journal('custom.anchor', url=root)
        assert isinstance(result, JournalFrame)
        assert len(result) >= 1

    def test_multiple_builds_all_found(self, tmp_path, monkeypatch):
        """Builds with different specs create distinct journal entries, all found."""
        from dataclasses import dataclass
        monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
        root = str(tmp_path)

        class CfgBlock(Datablock):
            TOPICFILE = 'out.txt'
            @dataclass
            class CONFIG(Datablock.CONFIG):
                label: str = 'a'
            def __build__(self):
                self.dirpath(ensure=True)
                with open(self.path(), 'w') as f:
                    f.write(self.cfg.label)

        b1 = CfgBlock(url=root, spec={'label': 'first'})
        b1.build()
        b2 = CfgBlock(url=root, spec={'label': 'second'})
        b2.build()
        # Both share the same anchor, journal should find both
        result = b1.journal()
        assert len(result) >= 2
