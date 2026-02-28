import ray
import os
import sys
sys.path.insert(0, os.getcwd())
from dbx.dbx import remote

def test_minimal():
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    
    print("Instantiating Remote handle...")
    r = remote(env={"MINIMAL_TEST": "passed"})
    
    print("Testing remote function call (environ)...")
    res = r.environ("MINIMAL_TEST")
    print(f"Remote MINIMAL_TEST: {res}")
    assert res == "passed"
    
    print("Testing universal proxying (accessing Logger class)...")
    logger_proxy = r.Logger
    print(f"Logger proxy handle: {logger_proxy}")
    # accessing _handle directly for internal verification
    assert hasattr(logger_proxy, '_handle')
    
    print("Minimal RPC verification PASSED.")

if __name__ == '__main__':
    try:
        test_minimal()
    finally:
        ray.shutdown()
