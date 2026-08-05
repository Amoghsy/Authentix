"""
timers.py

This module contains context manager timers to measure block and service execution latencies.
"""

import sys
import time


class CodeTimer:
    """
    Context manager to profile code blocks execution duration in milliseconds.
    """
    def __init__(self):
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0


if __name__ == "__main__":
    print("Executing self-test for backend/app/utils/timers.py...")
    try:
        with CodeTimer() as timer:
            time.sleep(0.05)
            
        print(f"Elapsed duration: {timer.elapsed_ms:.2f} ms")
        assert 45.0 <= timer.elapsed_ms <= 150.0  # Allow buffer for scheduler delays
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
