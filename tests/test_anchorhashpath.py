import os
import unittest
import pandas as pd
from dbx.datablocks import Datablock, JournalEntry

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
        
        expected_anchorkeypath = os.path.join(block.root, expected_anchorkey)
        self.assertEqual(block.anchorkeypath, expected_anchorkeypath)

    def test_journal_entry_anchorkeypath(self):
        data = {
            'root': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde'
        }
        series = pd.Series(data)
        entry = JournalEntry(series)
        
        self.assertTrue(hasattr(entry, 'anchorkey'))
        self.assertTrue(hasattr(entry, 'anchorkeypath'))
        
        expected_anchorkey = os.path.join(data['anchor'], data['hash'])
        self.assertEqual(entry.anchorkey, expected_anchorkey)
        
        expected_anchorkeypath = os.path.join(data['root'], expected_anchorkey)
        self.assertEqual(entry.anchorkeypath, expected_anchorkeypath)

    def test_journal_entry_backward_compat_aliases(self):
        """anchorhash and anchorhashpath still work as aliases on JournalEntry."""
        data = {
            'root': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde'
        }
        series = pd.Series(data)
        entry = JournalEntry(series)
        self.assertEqual(entry.anchorhash, entry.anchorkey)
        self.assertEqual(entry.anchorhashpath, entry.anchorkeypath)

if __name__ == "__main__":
    unittest.main()
