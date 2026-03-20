
import ray
import time
import functools
from dbx import datablocks
from dbx.datablocks import RayCallableExecutor

def work_fn(seconds):
    time.sleep(seconds)
    return seconds

def main():
    print("Initializing Ray...")
    ray.init(ignore_reinit_error=True)

    tasks_count = 5
    task_duration = 2.0 # seconds
    
    print(f"\n--- Running {tasks_count} tasks of {task_duration}s each with 1 worker ---")
    print("Initializing executor with 1 worker...")
    executor_1 = RayCallableExecutor(n_workers=1)
    # Warmup / Wait for workers? RayCallableExecutor init creates workers immediately but they might take a moment.
    # But usually ray.remote() returns a handle immediately. The actual startup happens async.
    # However, let's just measure execute().
    
    start_time_1 = time.time()
    callables_1 = [functools.partial(work_fn, task_duration) for _ in range(tasks_count)]
    executor_1.execute(callables_1)
    end_time_1 = time.time()
    duration_1 = end_time_1 - start_time_1
    print(f"Duration with 1 worker: {duration_1:.4f} seconds")

    print(f"\n--- Running {tasks_count} tasks of {task_duration}s each with 5 workers ---")
    print("Initializing executor with 5 workers...")
    executor_5 = RayCallableExecutor(n_workers=5)
    
    start_time_5 = time.time()
    callables_5 = [functools.partial(work_fn, task_duration) for _ in range(tasks_count)]
    executor_5.execute(callables_5)
    end_time_5 = time.time()
    duration_5 = end_time_5 - start_time_5
    print(f"Duration with 5 workers: {duration_5:.4f} seconds")

    speedup = duration_1 / duration_5
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Validation
    if speedup > 2.0:
        print("SUCCESS: Speedup is significant (> 2.0x)")
    else:
        print("WARNING: Speedup is lower than expected.")

    ray.shutdown()

if __name__ == "__main__":
    main()
