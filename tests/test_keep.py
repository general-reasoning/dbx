"""
Tests for Datablock.keep(msg=None).

Verifies that keep() creates a KEEP marker file in the anchorkeypath
directory, optionally with a message inside.
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
    TOPICFILE = 'output.txt'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = "'test'"

    def __build__(self):
        self.dirpath(ensure=True)
        with self.fs.open(self.path(), 'w') as f:
            f.write(f"built:{self.cfg.label}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKeep:

    def test_keep_creates_file(self, url):
        """keep() should create a KEEP file in anchorkeypath."""
        block = SimpleBlock(url=url)
        block.keep()
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        assert block.fs.exists(keeppath)

    def test_keep_empty_content_by_default(self, url):
        """KEEP file should be empty when no msg is given."""
        block = SimpleBlock(url=url)
        block.keep()
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        with block.fs.open(keeppath, 'r') as f:
            content = f.read()
        assert content == ''

    def test_keep_with_message(self, url):
        """keep(msg=...) should write the message into the KEEP file."""
        block = SimpleBlock(url=url)
        block.keep(msg='do not delete — needed for experiment X')
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        with block.fs.open(keeppath, 'r') as f:
            content = f.read()
        assert content == 'do not delete — needed for experiment X'

    def test_keep_returns_self(self, url):
        """keep() should return self for chaining."""
        block = SimpleBlock(url=url)
        result = block.keep()
        assert result is block

    def test_keep_before_build(self, url):
        """keep() should work even before build (creates anchorkeypath)."""
        block = SimpleBlock(url=url)
        assert not block.valid()
        block.keep(msg='pre-build marker')
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        assert block.fs.exists(keeppath)

    def test_keep_after_build(self, url):
        """keep() should work after build without disturbing existing files."""
        block = SimpleBlock(url=url)
        block.build()
        assert block.valid()
        block.keep(msg='post-build')
        # Block still valid
        assert block.valid()
        # KEEP file exists
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        assert block.fs.exists(keeppath)
        with block.fs.open(keeppath, 'r') as f:
            assert f.read() == 'post-build'

    def test_keep_overwrites_existing(self, url):
        """Calling keep() twice should overwrite the previous KEEP file."""
        block = SimpleBlock(url=url)
        block.keep(msg='first')
        block.keep(msg='second')
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        with block.fs.open(keeppath, 'r') as f:
            assert f.read() == 'second'

    def test_keep_with_none_msg(self, url):
        """keep(msg=None) should produce an empty file (same as no arg)."""
        block = SimpleBlock(url=url)
        block.keep(msg=None)
        keeppath = os.path.join(block.anchorkeypath, 'KEEP')
        with block.fs.open(keeppath, 'r') as f:
            assert f.read() == ''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
