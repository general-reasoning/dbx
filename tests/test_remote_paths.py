"""
Tests for remote-storage path correctness using the ``memory://`` filesystem.

Verifies that Datablock path methods, journal writing, and validation all work
correctly when the URL uses a non-local fsspec protocol.  The ``memory://``
filesystem is used as an in-process mock — no actual remote infrastructure
is needed.

Key things tested:
    1. anchorkeypath / anchorpath / dirpath / path return protocol-prefixed URLs
    2. _dbxanchorhashpathx returns protocol-prefixed URLs
    3. build() + valid() round-trip on memory://
    4. Journal entries are written and readable on memory://
    5. JournalEntry.anchorkeypath works on real journal data (url-based, no root)
    6. fs_full_path leaves local paths alone, prefixes remote ones
"""
import os
import datetime
import pytest
import pandas as pd
import fsspec

from dbx.datablocks import Datablock, JournalEntry, JournalFrame, journal
from dbx.dataparts import fs_full_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.fixture
def mem_url():
    """Return a unique memory:// URL for each test."""
    uid = os.urandom(4).hex()
    return f"memory://dbx_test_{uid}"


@pytest.fixture(autouse=True)
def _clear_memory_fs():
    """Ensure the MemoryFileSystem is empty before each test."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    yield
    fs.store.clear()


# ---------------------------------------------------------------------------
# Sample Datablocks
# ---------------------------------------------------------------------------

class MemSingleTopic(Datablock):
    """Single-topic Datablock that writes via fsspec (works on any backend)."""
    TOPICS = {'output': 'output.txt'}

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with self.fs.open(path, 'w') as f:
            f.write('hello from memory')


class MemMultiTopic(Datablock):
    """Multi-topic Datablock that writes via fsspec."""
    TOPICS = {'alpha': 'a.txt', 'beta': 'b.txt'}

    def __build__(self):
        for topic in self.TOPICS:
            self.dirpath(topic, ensure=True)
            with self.fs.open(self.path(topic), 'w') as f:
                f.write(f'{topic}:data')


# ---------------------------------------------------------------------------
# 1. fs_full_path behaviour
# ---------------------------------------------------------------------------

class TestFsFullPath:

    def test_local_path_unchanged(self):
        fs, _ = fsspec.url_to_fs('/tmp/foo')
        assert fs_full_path(fs, '/tmp/foo') == '/tmp/foo'

    def test_memory_path_gets_protocol(self):
        fs, root = fsspec.url_to_fs('memory://bucket/root')
        result = fs_full_path(fs, root)
        assert result.startswith('memory://')
        assert 'bucket/root' in result

    def test_memory_subpath_gets_protocol(self):
        fs, root = fsspec.url_to_fs('memory://bucket/root')
        sub = os.path.join(root, 'child', 'file.txt')
        result = fs_full_path(fs, sub)
        assert result.startswith('memory://')
        assert 'child/file.txt' in result


# ---------------------------------------------------------------------------
# 2. anchorkeypath / anchorpath / dirpath / path return full URLs on memory://
# ---------------------------------------------------------------------------

class TestPathsOnMemory:

    def test_anchorkeypath_has_protocol(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        assert block.anchorkeypath.startswith('memory://')

    def test_anchorpath_has_protocol(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        assert block.anchorpath().startswith('memory://')

    def test_dirpath_has_protocol(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        assert block.dirpath().startswith('memory://')

    def test_path_has_protocol(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        assert block.path('output').startswith('memory://')

    def test_multi_topic_dirpath_has_protocol(self, mem_url):
        block = MemMultiTopic(url=mem_url)
        for topic in block.TOPICS:
            assert block.dirpath(topic).startswith('memory://')

    def test_multi_topic_path_has_protocol(self, mem_url):
        block = MemMultiTopic(url=mem_url)
        for topic in block.TOPICS:
            assert block.path(topic).startswith('memory://')

    def test_local_paths_bare(self, tmp_path):
        """On local fs, paths should remain bare (no file:// prefix)."""
        block = MemSingleTopic(url=str(tmp_path))
        assert not block.anchorkeypath.startswith('file://')
        assert not block.path('output').startswith('file://')


# ---------------------------------------------------------------------------
# 3. _dbxanchorhashpathx returns protocol-prefixed URLs
# ---------------------------------------------------------------------------

class TestDbxAnchorHashPathX:

    def test_returns_protocol_prefixed_url(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        xpath = block._dbxanchorhashpathx('journal', 'parquet', ensure_dirpath=False)
        assert xpath.startswith('memory://')

    def test_local_returns_bare_path(self, tmp_path):
        block = MemSingleTopic(url=str(tmp_path))
        xpath = block._dbxanchorhashpathx('journal', 'parquet', ensure_dirpath=False)
        assert not xpath.startswith('file://')


# ---------------------------------------------------------------------------
# 4. build() + valid() round-trip on memory://
# ---------------------------------------------------------------------------

class TestBuildOnMemory:

    def test_single_topic_lifecycle(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        assert block.valid(topic=None) is False
        block.build()
        assert block.valid(topic=None) is True

    def test_multi_topic_lifecycle(self, mem_url):
        block = MemMultiTopic(url=mem_url)
        assert block.valid(topic=None) is False
        block.build()
        assert block.valid(topic=None) is True

    def test_built_file_is_readable(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        block.build()
        with block.fs.open(block.path('output'), 'r') as f:
            content = f.read()
        assert content == 'hello from memory'

    def test_clear_after_build(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        block.build()
        assert block.valid(topic=None) is True
        block.UNSAFE_clear(OVERRIDE=True)
        assert block.valid(topic=None) is False

    def test_leave_breadcrumbs_on_memory(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        assert block.valid(topic=None) is False
        block.leave_breadcrumbs()
        assert block.valid(topic=None) is True

    def test_dirpath_ensure_on_memory(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        dp = block.dirpath(ensure=True)
        assert block.fs.isdir(dp)

    def test_ls_on_memory(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        block.build()
        files = block.ls()
        assert len(files) >= 1


# ---------------------------------------------------------------------------
# 5. Journal writing + reading on memory://
# ---------------------------------------------------------------------------

class TestJournalOnMemory:

    def test_journal_entry_written_on_build(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        block.build()
        j = block.journal()
        assert isinstance(j, JournalFrame)
        assert len(j) >= 1

    def test_journal_entry_has_correct_url(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        block.build()
        j = block.journal()
        entry = j.get(0)
        assert entry.url == mem_url

    def test_journal_entry_has_correct_hash(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        block.build()
        j = block.journal()
        entry = j.get(0)
        assert entry.hash == block.hash

    def test_static_journal_on_memory(self, mem_url):
        block = MemSingleTopic(url=mem_url)
        block.build()
        j = Datablock.Journal(block.anchor, url=mem_url)
        assert isinstance(j, JournalFrame)
        assert len(j) >= 1


# ---------------------------------------------------------------------------
# 6. JournalEntry.anchorkeypath works with url (not root)
# ---------------------------------------------------------------------------

class TestJournalEntryPaths:

    def test_anchorkeypath_from_url(self):
        """JournalEntry with 'url' but no 'root' should produce correct paths."""
        data = {
            'url': 'memory://bucket/root',
            'anchor': 'my.module.MyBlock',
            'hash': 'abc123def456',
            'superhash': 'ab12cd34',
        }
        entry = JournalEntry(pd.Series(data))
        akp = entry.anchorkeypath
        assert akp.startswith('memory://')
        assert 'my.module.MyBlock' in akp
        assert 'abc123def456' in akp

    def test_anchorkeypath_from_url_with_hash(self):
        data = {
            'url': 'memory://bucket/root',
            'anchor': 'my.module.MyBlock',
            'hash': 'abc123def456',
            'superhash': 'ab12cd34',
        }
        entry = JournalEntry(pd.Series(data))
        akp = entry.anchorkeypath
        assert akp.startswith('memory://')
        assert 'abc123def456' in akp

    def test_anchorkeypath_local_url_no_prefix(self):
        """JournalEntry with local url should produce bare paths."""
        data = {
            'url': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': 'abc123def456',
            'superhash': 'ab12cd34',
        }
        entry = JournalEntry(pd.Series(data))
        assert not entry.anchorkeypath.startswith('file://')
        assert '/tmp/dbx_test' in entry.anchorkeypath

    def test_root_property_from_url(self):
        """JournalEntry.root should derive from url."""
        data = {
            'url': 'memory://bucket/root',
            'anchor': 'test',
            'hash': 'abc',
        }
        entry = JournalEntry(pd.Series(data))
        assert entry.root == '/bucket/root'

    def test_root_property_legacy_fallback(self):
        """JournalEntry.root should fall back to 'root' field when no url."""
        data = {
            'root': '/legacy/path',
            'anchor': 'test',
            'hash': 'abc',
        }
        entry = JournalEntry(pd.Series(data))
        assert entry.root == '/legacy/path'

    def test_journal_entry_from_real_build(self, mem_url):
        """JournalEntry produced by a real build has correct anchorkeypath."""
        block = MemSingleTopic(url=mem_url)
        block.build()
        j = block.journal()
        entry = j.get(0)
        # Entry should have url, not root
        assert entry.url == mem_url
        assert entry.get('root') is None
        # anchorkeypath should start with protocol
        akp = entry.anchorkeypath
        assert akp.startswith('memory://')
        assert entry.anchor in akp
