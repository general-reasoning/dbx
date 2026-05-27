"""
Tests for the Datablock.is_local_fs property.

Verifies that is_local_fs correctly identifies local filesystem protocols
('file', 'local', '') and non-local protocols (e.g. 'memory', 's3').
"""

import os
import unittest
from unittest.mock import patch, PropertyMock
from dbx.datablocks import Datablock


class TestIsLocalFs(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_local_bare_path(self):
        """A bare path like /tmp/... should use the local filesystem."""
        block = Datablock(url="/tmp/dbx_test_local_fs")
        self.assertTrue(block.is_local_fs)

    def test_local_file_url(self):
        """An explicit file:// URL should be recognized as local."""
        block = Datablock(url="file:///tmp/dbx_test_local_fs")
        self.assertTrue(block.is_local_fs)

    def test_memory_fs(self):
        """A memory:// URL should NOT be local."""
        block = Datablock(url="memory://dbx_test_local_fs")
        self.assertFalse(block.is_local_fs)

    def test_protocol_tuple(self):
        """When fs.protocol is a tuple, the first element is used."""
        block = Datablock(url="/tmp/dbx_test_local_fs")
        # Local filesystem protocol can be a tuple ('file', 'local')
        # on some fsspec versions; either way the property should handle it.
        original_protocol = block.fs.protocol
        try:
            # Simulate tuple protocol
            block.fs.protocol = ('file', 'local')
            self.assertTrue(block.is_local_fs)

            block.fs.protocol = ('s3', 's3a')
            self.assertFalse(block.is_local_fs)
        finally:
            block.fs.protocol = original_protocol

    def test_protocol_string(self):
        """When fs.protocol is a string, it's used directly."""
        block = Datablock(url="/tmp/dbx_test_local_fs")
        original_protocol = block.fs.protocol
        try:
            block.fs.protocol = 'file'
            self.assertTrue(block.is_local_fs)

            block.fs.protocol = 'local'
            self.assertTrue(block.is_local_fs)

            block.fs.protocol = ''
            self.assertTrue(block.is_local_fs)

            block.fs.protocol = 's3'
            self.assertFalse(block.is_local_fs)
        finally:
            block.fs.protocol = original_protocol

    def test_survives_pickle_roundtrip(self):
        """is_local_fs should work after pickling and unpickling."""
        import pickle
        block = Datablock(url="/tmp/dbx_test_local_fs")
        self.assertTrue(block.is_local_fs)

        data = pickle.dumps(block)
        restored = pickle.loads(data)
        self.assertTrue(restored.is_local_fs)


class TestIsLocalFsSubclass(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_subclass_inherits_is_local_fs(self):
        """Subclasses of Datablock should inherit is_local_fs."""
        class MyBlock(Datablock):
            pass

        block = MyBlock(url="/tmp/dbx_test_local_fs")
        self.assertTrue(block.is_local_fs)


if __name__ == "__main__":
    unittest.main()
