import os
import unittest
import pandas as pd
import fsspec
from dbx.datablocks import Datablock, DatajournalEntry
from dbx.dataparts import fs_full_path

class TestAnchorKeyPath(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

    def test_datablock_anchorkeypath(self):
        class MyBlock(Datablock):
            pass
        
        block = MyBlock(url="/tmp/dbx_test")
        # Default keyby='tag_version_hash'; key includes version + hash components
        self.assertTrue(hasattr(block, 'anchorkey'))
        self.assertTrue(hasattr(block, 'anchorkeypath'))
        
        expected_anchorkey = os.path.join(block.anchor, block.key)
        self.assertEqual(block.anchorkey, expected_anchorkey)
        
        # anchorkeypath uses fs_full_path: for local fs, returns bare path
        expected_anchorkeypath = fs_full_path(
            block.fs, os.path.join(block.root, expected_anchorkey)
        )
        self.assertEqual(block.anchorkeypath, expected_anchorkeypath)
        # For local filesystem, fs_full_path should return the bare path
        self.assertFalse(block.anchorkeypath.startswith('file://'))

    def test_journal_entry_anchorkeypath_with_url(self):
        """DatajournalEntry.anchorkeypath should derive root from url."""
        data = {
            'url': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde',
            'subhash': 'ab12cd34',
        }
        series = pd.Series(data)
        entry = DatajournalEntry(series)
        
        # They are Datablock-shaped, so they live on the Block view, not on
        # the entry -- which would otherwise fall through to column lookup.
        self.assertFalse(hasattr(entry, 'anchorkey'))
        self.assertTrue(hasattr(entry.block, 'anchorkey'))
        self.assertTrue(hasattr(entry.block, 'anchorkeypath'))
        
        expected_anchorkey = os.path.join(data['anchor'], data['hash'][:8])
        self.assertEqual(entry.block.anchorkey, expected_anchorkey)
        
        fs, root = fsspec.url_to_fs(data['url'])
        expected_anchorkeypath = fs_full_path(fs, os.path.join(root, expected_anchorkey))
        self.assertEqual(entry.block.anchorkeypath, expected_anchorkeypath)

    def test_journal_entry_anchorkeypath_legacy_root(self):
        """DatajournalEntry.anchorkeypath still works with legacy 'root' field."""
        data = {
            'root': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde',
            'subhash': 'ab12cd34',
        }
        series = pd.Series(data)
        entry = DatajournalEntry(series)
        
        expected_anchorkey = os.path.join(data['anchor'], data['hash'][:8])
        expected_anchorkeypath = os.path.join(data['root'], expected_anchorkey)
        self.assertEqual(entry.block.anchorkeypath, expected_anchorkeypath)


if __name__ == "__main__":
    unittest.main()
