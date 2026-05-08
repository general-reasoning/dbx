import os
import unittest
import pandas as pd
import fsspec
from dbx.datablocks import Datablock, JournalEntry
from dbx.dataparts import fs_full_path

class TestAnchorKeyPath(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_datablock_anchorkeypath(self):
        class MyBlock(Datablock):
            pass
        
        block = MyBlock(url="/tmp/dbx_test")
        # Default keyby='taghash'; with no tag set, key falls back to hash
        self.assertTrue(hasattr(block, 'anchorkey'))
        self.assertTrue(hasattr(block, 'anchorkeypath'))
        
        expected_anchorkey = os.path.join(block.anchor, block.hash)
        self.assertEqual(block.anchorkey, expected_anchorkey)
        
        # anchorkeypath uses fs_full_path: for local fs, returns bare path
        expected_anchorkeypath = fs_full_path(
            block.fs, os.path.join(block.root, expected_anchorkey)
        )
        self.assertEqual(block.anchorkeypath, expected_anchorkeypath)
        # For local filesystem, fs_full_path should return the bare path
        self.assertFalse(block.anchorkeypath.startswith('file://'))

    def test_journal_entry_anchorkeypath_with_url(self):
        """JournalEntry.anchorkeypath should derive root from url."""
        data = {
            'url': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde'
        }
        series = pd.Series(data)
        entry = JournalEntry(series)
        
        self.assertTrue(hasattr(entry, 'anchorkey'))
        self.assertTrue(hasattr(entry, 'anchorkeypath'))
        
        expected_anchorkey = os.path.join(data['anchor'], data['hash'])
        self.assertEqual(entry.anchorkey, expected_anchorkey)
        
        fs, root = fsspec.url_to_fs(data['url'])
        expected_anchorkeypath = fs_full_path(fs, os.path.join(root, expected_anchorkey))
        self.assertEqual(entry.anchorkeypath, expected_anchorkeypath)

    def test_journal_entry_anchorkeypath_legacy_root(self):
        """JournalEntry.anchorkeypath still works with legacy 'root' field."""
        data = {
            'root': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde'
        }
        series = pd.Series(data)
        entry = JournalEntry(series)
        
        expected_anchorkey = os.path.join(data['anchor'], data['hash'])
        expected_anchorkeypath = os.path.join(data['root'], expected_anchorkey)
        self.assertEqual(entry.anchorkeypath, expected_anchorkeypath)

    def test_journal_entry_backward_compat_aliases(self):
        """anchorhash and anchorhashpath still work as aliases on JournalEntry."""
        data = {
            'url': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde'
        }
        series = pd.Series(data)
        entry = JournalEntry(series)
        self.assertEqual(entry.anchorhash, os.path.join(data['anchor'], data['hash']))
        # anchorhashpath uses the same url-based resolution
        fs, root = fsspec.url_to_fs(data['url'])
        expected = fs_full_path(fs, os.path.join(root, entry.anchorhash))
        self.assertEqual(entry.anchorhashpath, expected)

if __name__ == "__main__":
    unittest.main()
