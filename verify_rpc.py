import os
import ray
import sys

# Ensure the local dbx directory is in the path so we can import it
sys.path.insert(0, os.getcwd())

from dbx.dbx import remote

def test_rpc():
    print("Initializing Ray...")
    # Initialize Ray. If it's already running, this will just connect.
    ray.init(ignore_reinit_error=True)

    # Set a local DBX variable to test propagation
    os.environ["DBX_PROP_TEST"] = "success"
    
    print(f"Instantiating remote dbx via remote() with env={{'TEST_VAR': 'hello_remote'}} (and DBX_PROP_TEST=success from local env)")
    # Initialize using the helper function. Returns a Remote instance.
    remote_dbx = remote(env={"TEST_VAR": "hello_remote"})

    # 1. Verify environment variable setting via get_env proxy method
    print("Verifying environment variables in remote actor...")
    remote_test_var = remote_dbx.environ("TEST_VAR")
    print(f"Remote TEST_VAR: {remote_test_var}")
    assert remote_test_var == "hello_remote"

    # 2. Verify propagation of custom variable
    remote_prop = remote_dbx.environ("DBX_PROP_TEST")
    print(f"Remote DBX_PROP_TEST: {remote_prop}")
    assert remote_prop == "success"
    print("Verification of propagated variables confirmed.")
    
    # 3. Test nested proxying: Instantiate a Datablocks remotely
    print("Testing nested proxying: remote_dbx.Datablocks()...")
    # Datablocks is a class. Calling it should return a Remote instance wrapping a RemoteObject actor
    db = remote_dbx.Datablocks()
    print(f"Remote Datablocks proxy: {db}")
    
    # 4. Call a function directly through the proxy.
    print("Calling remote_dbx.gitrevision(os.getcwd())...")
    rev = remote_dbx.gitrevision(os.getcwd())
    print(f"Remote gitrevision result: {rev}")
    assert rev is not None

    print("Remote Universal Proxy Verification Successful!")

if __name__ == "__main__":
    try:
        test_rpc()
    finally:
        ray.shutdown()
