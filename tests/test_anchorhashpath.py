import os
import unittest
import pandas as pd
from dbx.dbx import Datablock, JournalEntry

class TestAnchorHashPath(unittest.TestCase):
    def test_datablock_anchorhashpath(self):
        class MyBlock(Datablock):
            pass
        
        block = MyBlock(root="/tmp/dbx_test")
        # Manually set hash if needed, but Datablock should compute it
        self.assertTrue(hasattr(block, 'anchorhash'))
        self.assertTrue(hasattr(block, 'anchorhashpath'))
        
        expected_anchorhash = os.path.join(block.anchor, block.hash)
        self.assertEqual(block.anchorhash, expected_anchorhash)
        
        expected_anchorhashpath = os.path.join(block.root, expected_anchorhash)
        self.assertEqual(block.anchorhashpath, expected_anchorhashpath)

    def test_journal_entry_anchorhashpath(self):
        data = {
            'root': '/tmp/dbx_test',
            'anchor': 'my.module.MyBlock',
            'hash': '12345abcde'
        }
        series = pd.Series(data)
        entry = JournalEntry(series)
        
        self.assertTrue(hasattr(entry, 'anchorhash'))
        self.assertTrue(hasattr(entry, 'anchorhashpath'))
        
        expected_anchorhash = os.path.join(data['anchor'], data['hash'])
        self.assertEqual(entry.anchorhash, expected_anchorhash)
        
        expected_anchorhashpath = os.path.join(data['root'], expected_anchorhash)
        self.assertEqual(entry.anchorhashpath, expected_anchorhashpath)

if __name__ == "__main__":
    unittest.main()
