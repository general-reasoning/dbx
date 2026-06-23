"""
Tests for Datablock.keep(message=None).

Verifies:
  1. keep() writes a journal entry with event='keep'
  2. The journal parquet file is prefixed with 'keep-'
  3. No KEEP sentinel file is created in anchorkeypath
  4. journal() reads keep entries (glob matches keep-*.parquet)
  5. message is recorded as inline message when provided
  6. keep() returns self (fluent API)
  7. keep() works before and after build without side-effects
  8. keep() delegates to note('keep', ..., inline=True)
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


@pytest.fixture
def url(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Sample subclass
# ---------------------------------------------------------------------------

class SimpleBlock(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'test'"

    def __build__(self):
        path = self.path(ensure_dirpath=True)
        with self.fs.open(path, 'w') as f:
            f.write(f"built:{self.cfg.label}")


def _make(url, **kw):
    return SimpleBlock(url=url, **kw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKeepJournalEntry:

    def test_keep_writes_journal_entry(self, url):
        """keep() should create a journal entry with event='keep'."""
        block = _make(url)
        block.keep()
        j = block.journal()
        keep_rows = j[j['event'] == 'keep']
        assert len(keep_rows) == 1

    def test_keep_event_value(self, url):
        """The journal should contain an event named 'keep'."""
        block = _make(url)
        block.keep()
        j = block.journal()
        assert 'keep' in j['event'].values

    def test_keep_records_correct_hash(self, url):
        """The journal entry should record the correct block hash."""
        block = _make(url)
        block.keep()
        j = block.journal()
        row = j[j['event'] == 'keep'].iloc[-1]
        assert row['hash'] == block.hash


class TestKeepParquetPrefix:

    def test_parquet_filename_starts_with_keep(self, url):
        """The journal parquet file written by keep() should start with 'keep-'."""
        block = _make(url)
        block.keep()
        # Find journal parquet files under .dbx/ (the full path includes fqcn subdirs)
        dbx_dir = os.path.join(url, block.anchor, '.dbx')
        all_parquets = []
        for root, dirs, files in os.walk(dbx_dir):
            # Only look inside journal/ directories
            if os.path.basename(root) != block.hash:
                continue
            parent = os.path.basename(os.path.dirname(root))
            if parent != 'journal':
                continue
            for f in files:
                if f.endswith('.parquet'):
                    all_parquets.append(f)
        keep_parquets = [f for f in all_parquets if f.startswith('keep-')]
        assert len(keep_parquets) >= 1, f"Expected keep- prefixed parquets, got: {all_parquets}"


class TestKeepNoSentinelFile:

    def test_no_keep_marker_file(self, url):
        """keep() should NOT create a KEEP file in anchorkeypath."""
        block = _make(url)
        block.keep()
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        assert not block.fs.exists(keeppath)


class TestKeepReadByJournal:

    def test_journal_reads_keep_entries(self, url):
        """journal() should include keep entries (the glob matches keep-*.parquet)."""
        block = _make(url)
        block.keep(message='marker')
        j = block.journal()
        assert len(j) > 0
        assert 'keep' in j['event'].values

    def test_journal_filter_by_event(self, url):
        """journal(event='keep') should return only keep entries."""
        block = _make(url)
        block.build()
        block2 = _make(url)
        block2.keep(message='important')
        j = block.journal(event='keep')
        assert len(j) >= 1
        assert all(j['event'] == 'keep')


class TestKeepMessage:

    def test_keep_with_message_records_inline(self, url):
        """keep(message=...) should record message as inline in the journal."""
        block = _make(url)
        block.keep(message='do not delete')
        j = block.journal()
        row = j[j['event'] == 'keep'].iloc[-1]
        assert row['message'] == 'do not delete'

    def test_keep_without_message_has_no_message(self, url):
        """keep() with no message should have None/NaN message."""
        block = _make(url)
        block.keep()
        j = block.journal()
        row = j[j['event'] == 'keep'].iloc[-1]
        import pandas as pd
        assert pd.isna(row.get('message', None)) or row.get('message') is None


class TestKeepReturnsSelf:

    def test_returns_self(self, url):
        """keep() should return self for chaining."""
        block = _make(url)
        result = block.keep()
        assert result is block

    def test_returns_self_with_message(self, url):
        """keep(message=...) should return self for chaining."""
        block = _make(url)
        result = block.keep(message='note')
        assert result is block


class TestKeepLifecycle:

    def test_keep_before_build(self, url):
        """keep() should work even before build."""
        block = _make(url)
        assert not block.valid()
        block.keep(message='pre-build marker')
        j = block.journal()
        assert 'keep' in j['event'].values

    def test_keep_after_build(self, url):
        """keep() should work after build without disturbing existing files."""
        block = _make(url)
        block.build()
        assert block.valid()
        block2 = _make(url)
        block2.keep(message='post-build')
        # Block still valid
        assert block.valid()
        j = block.journal()
        assert 'keep' in j['event'].values

    def test_multiple_keeps_produce_multiple_entries(self, url):
        """Calling keep() multiple times should produce multiple journal entries."""
        for msg in ('first', 'second', 'third'):
            b = _make(url)
            b.keep(message=msg)
        j = _make(url).journal()
        keep_rows = j[j['event'] == 'keep']
        assert len(keep_rows) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
