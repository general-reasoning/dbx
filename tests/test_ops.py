import unittest
import pickle
import os
from dbx.datablocks import Datablock

class MyTestBlock(Datablock):
    VERSION = "v1"

class TestDatablockOps(unittest.TestCase):
    def test_ops_returns_getstate(self):
        block = MyTestBlock(datalake="/tmp", my_param="hello", extra_val=123, revision="test")
        state = block.__getstate__()
        self.assertEqual(block.dfn, state)
        self.assertEqual(block.ops['my_param'], "hello")
        self.assertEqual(block.ops['extra_val'], 123)
        
    def test_defn_returns_all_args(self):
        block = MyTestBlock(datalake="/tmp", my_param="hello", extra_val=123, info=True, revision="test")
        
        # ops drops explicit params like 'root', 'revision' and 'info'
        self.assertEqual(block.ops, {'my_param': 'hello', 'extra_val': 123})
        
        # dfn yields state consisting of ALL __init__ + **kwargs parameters
        defn = block.dfn
        self.assertEqual(defn['datalake'], "/tmp")
        self.assertEqual(defn['info'], True)
        self.assertEqual(defn['my_param'], "hello")
        self.assertEqual(defn['extra_val'], 123)
        self.assertIsNone(defn['anchor']) # verifying default property fallback

    def test_set_method_updates_ops(self):
        block1 = MyTestBlock(datalake="/tmp/a", a=1, b=2, revision="test")
        block2 = block1.set(b=3, c=4)
        
        self.assertEqual(block2.ops['a'], 1)  # Preserved
        self.assertEqual(block2.ops['b'], 3)  # Overwritten
        self.assertEqual(block2.ops['c'], 4)  # Added
        self.assertEqual(block2.dfn['datalake'], "/tmp/a") # Preserved through definition
        
        # Verify old block is unchanged
        self.assertEqual(block1.ops['b'], 2)
        self.assertNotIn('c', block1.ops)

    def test_serialization(self):
        block = MyTestBlock(datalake="/tmp/serialize", alpha="abc", x=10, revision="test")
        
        # Serialize
        serialized = pickle.dumps(block)
        
        # Deserialize
        deserialized = pickle.loads(serialized)
        
        # Assert ops are identical
        self.assertEqual(block.ops, deserialized.ops)
        self.assertEqual(deserialized.ops['alpha'], "abc")
        self.assertEqual(deserialized.ops['x'], 10)
        self.assertEqual(deserialized.dfn['datalake'], "/tmp/serialize")

    def test_legacy_kwargs_format_unpickling(self):
        # Manually mimic a state dict from an older version
        # Older version had explicit 'kwargs' and 'state' dicts
        old_state_dict = {
            'url': '/tmp/legacy',
            'kwargs': {'a': 100, 'b': 200},
            'state': {'c': 300},
            'anchor': None,
            'revision': 'test'
        }
        
        block = MyTestBlock.__new__(MyTestBlock)
        block.__setstate__(old_state_dict)
        
        self.assertEqual(block.ops['a'], 100)
        self.assertEqual(block.ops['b'], 200)
        self.assertEqual(block.ops['c'], 300)
        
        self.assertEqual(block.dfn['datalake'], '/tmp/legacy')

if __name__ == "__main__":
    unittest.main()
