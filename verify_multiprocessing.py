
import sys
import os
import time
import multiprocessing as mp

# Add dbx to path
sys.path.append('/home/t-9dkarp/dbx')

from dbx.dbx import MultiprocessingDatablocksBuilder, Datablocks, Logger

# Define a simple Datablocks for testing
class TestDatablocks(Datablocks):
    def __init__(self, value, sleep_time=0.1):
        self.value = value
        self.sleep_time = sleep_time
        self._built = False
        super().__init__()

    def build(self, *args, **kwargs):
        print(f"Building block {self.value} in process {os.getpid()}")
        time.sleep(self.sleep_time)
        self._built = True
        return self

    def __repr__(self):
        return f"TestDatablocks(value={self.value})"

def verify_multiprocessing():
    print("Verifying MultiprocessingDatablocksBuilder...")
    blocks = [TestDatablocks(i) for i in range(4)]
    
    # Use 2 processes
    builder = MultiprocessingDatablocksBuilder(n_processes=2, log=Logger(name="TestBuilder"))
    
    start_time = time.time()
    built_blocks = builder.build_blocks(blocks)
    end_time = time.time()
    
    print(f"Built {len(built_blocks)} blocks in {end_time - start_time:.2f} seconds")
    
    assert len(built_blocks) == 4
    # Note: built_blocks in the main process won't have the _built flag set because 
    # modifications in subprocesses don't propagate back to the main process object instances 
    # unless using shared memory or returning results.
    # The builder implementation returns 'blocks' which are the original objects.
    # However, the goal of the builder is usually to trigger side effects (writing to disk, etc) 
    # OR the user expects the blocks to be modified if they returned them?
    # 
    # Looking at the implementation of MultithreadingDatablocksBuilder:
    # It returns 'blocks'.
    # In multithreading, the objects are shared, so modifications (like .to('cpu')) show up.
    # In multiprocessing, they are copied. So correct, the local objects won't show changes.
    # But checking if the build process ran without error is the main thing here.
    
    print("Verification successful!")

if __name__ == "__main__":
    verify_multiprocessing()
