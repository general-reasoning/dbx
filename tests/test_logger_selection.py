
import os
import pytest
import sys
from dbx.datablocks import Logger

# Custom Logger that captures whether _print was called
class CaptureLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.printed = False
        self.last_prefix = None
        self.last_msg = None

    def _print(self, prefix, msg):
        self.printed = True
        self.last_prefix = prefix
        self.last_msg = msg
        super()._print(prefix, msg)

def caller_func(logger, msg):
    logger.selected(msg)

def dummmy_func(logger, msg):
    logger.selected(msg)

def test_logger_selection_explicit():
    # Test that selection works when explicitly provided in __init__
    # Note: When running with pytest, the module name might be just 'test_logger_selection'
    # depending on how pytest is invoked. We'll use the actual __name__ here.
    fqn = f"{__name__}.caller_func"
    logger = CaptureLogger(selection=[fqn])
    
    # Call from allowed function
    caller_func(logger, "Should print")
    assert logger.printed is True
    logger.printed = False
    
    # Call from disallowed function
    dummmy_func(logger, "Should not print")
    assert logger.printed is False

def test_logger_selection_string():
    fqn = f"{__name__}.caller_func"
    logger = CaptureLogger(selection=f"{fqn}, other.func")
    
    caller_func(logger, "Should print")
    assert logger.printed is True
    logger.printed = False

def test_logger_selection_none():
    logger = CaptureLogger(selection=None)
    
    caller_func(logger, "Should print")
    assert logger.printed is True
    logger.printed = False
    
    dummmy_func(logger, "Should also print")
    assert logger.printed is True

def test_logger_selection_env(monkeypatch):
    fqn = f"{__name__}.caller_func"
    monkeypatch.setenv('DBX_LOG_SELECTION', fqn)
    logger = CaptureLogger()
    
    caller_func(logger, "Should print")
    assert logger.printed is True
    logger.printed = False
    
    dummmy_func(logger, "Should not print")
    assert logger.printed is False

if __name__ == "__main__":
    # Manual run
    logger = CaptureLogger(selection=['__main__.caller_func'])
    caller_func(logger, "Manual test")
    if logger.printed:
        print("Manual test passed!")
    else:
        print("Manual test failed!")
