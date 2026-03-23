import unittest
import pickle
import os
from dbx.datablocks import Datablock

class MyTestBlock(Datablock):
    VERSION = "v1"

class TestDatablockKwargs(unittest.TestCase):
    def test_kwargs_returns_getstate(self):
        block = MyTestBlock(root="/tmp", my_param="hello", extra_val=123, revision="test")
        state = block.__getstate__()
        self.assertEqual(block.dfn, state)
        self.assertEqual(block.kwargs['my_param'], "hello")
        self.assertEqual(block.kwargs['extra_val'], 123)
        
    def test_defn_returns_all_args(self):
        block = MyTestBlock(root="/tmp", my_param="hello", extra_val=123, info=True, revision="test")
        
        # kwargs drops explicit params like 'root', 'revision' and 'info'
        self.assertEqual(block.kwargs, {'my_param': 'hello', 'extra_val': 123})
        
        # dfn yields state consisting of ALL __init__ + **kwargs parameters
        defn = block.dfn
        self.assertEqual(defn['root'], "/tmp")
        self.assertEqual(defn['info'], True)
        self.assertEqual(defn['my_param'], "hello")
        self.assertEqual(defn['extra_val'], 123)
        self.assertIsNone(defn['anchor']) # verifying default property fallback

    def test_set_method_updates_kwargs(self):
        block1 = MyTestBlock(root="/tmp/a", a=1, b=2, revision="test")
        block2 = block1.set(b=3, c=4)
        
        self.assertEqual(block2.kwargs['a'], 1)  # Preserved
        self.assertEqual(block2.kwargs['b'], 3)  # Overwritten
        self.assertEqual(block2.kwargs['c'], 4)  # Added
        self.assertEqual(block2.dfn['root'], "/tmp/a") # Preserved through definition
        
        # Verify old block is unchanged
        self.assertEqual(block1.kwargs['b'], 2)
        self.assertNotIn('c', block1.kwargs)

    def test_serialization(self):
        block = MyTestBlock(root="/tmp/serialize", alpha="abc", x=10, revision="test")
        
        # Serialize
        serialized = pickle.dumps(block)
        
        # Deserialize
        deserialized = pickle.loads(serialized)
        
        # Assert kwargs are identical
        self.assertEqual(block.kwargs, deserialized.kwargs)
        self.assertEqual(deserialized.kwargs['alpha'], "abc")
        self.assertEqual(deserialized.kwargs['x'], 10)
        self.assertEqual(deserialized.dfn['root'], "/tmp/serialize")

    def test_legacy_kwargs_format_unpickling(self):
        # Manually mimic a state dict from an older version
        # Older version had explicit 'kwargs' and 'state' dicts
        old_state_dict = {
            'root': '/tmp/legacy',
            'kwargs': {'a': 100, 'b': 200},
            'state': {'c': 300},
            'anchor': None,
            'revision': 'test'
        }
        
        block = MyTestBlock.__new__(MyTestBlock)
        block.__setstate__(old_state_dict)
        
        self.assertEqual(block.kwargs['a'], 100)
        self.assertEqual(block.kwargs['b'], 200)
        self.assertEqual(block.kwargs['c'], 300)
        
        self.assertEqual(block.dfn['root'], '/tmp/legacy')

if __name__ == "__main__":
    unittest.main()
