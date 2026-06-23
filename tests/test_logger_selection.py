
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

def test_logger_selection_none_is_silent():
    """When selection is None/empty, selected() should not print anything."""
    logger = CaptureLogger(selection=None)
    
    caller_func(logger, "Should NOT print")
    assert logger.printed is False
    
    dummmy_func(logger, "Should also NOT print")
    assert logger.printed is False

def test_logger_selection_unset_env_is_silent(monkeypatch):
    """When DBX_LOG_SELECTION is not set, selected() should be silent."""
    monkeypatch.delenv('DBX_LOG_SELECTION', raising=False)
    logger = CaptureLogger()
    
    caller_func(logger, "Should NOT print")
    assert logger.printed is False

def test_logger_selection_env(monkeypatch):
    fqn = f"{__name__}.caller_func"
    monkeypatch.setenv('DBX_LOG_SELECTION', fqn)
    logger = CaptureLogger()
    
    caller_func(logger, "Should print")
    assert logger.printed is True
    logger.printed = False
    
    dummmy_func(logger, "Should not print")
    assert logger.printed is False


def test_logger_selected_default_stack_depth():
    """selected() works with default stack_depth=2."""
    fqn = f"{__name__}.caller_func"
    logger = CaptureLogger(selection=[fqn])
    assert logger.stack_depth == 2  # default
    caller_func(logger, "Should print with default stack_depth")
    assert logger.printed is True


def test_logger_selected_stack_depth_none_breaks():
    """stack_depth=None causes TypeError in selected() when selection is set."""
    fqn = f"{__name__}.caller_func"
    logger = CaptureLogger(selection=[fqn], stack_depth=None)
    with pytest.raises(TypeError):
        caller_func(logger, "Should raise TypeError")


def test_datablock_logger_has_working_stack_depth(monkeypatch, tmp_path):
    """Datablock's Logger should have a valid stack_depth so selected() works."""
    from dbx.datablocks import Datablock

    monkeypatch.setenv('DBX_ROOT', str(tmp_path))
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.setenv('DBX_LOG_SELECTED', 'True')

    class SimpleBlock(Datablock):
        TOPICS = {'out': 'out.txt'}
        def __build__(self, *a, **kw): return self
        def __read__(self, topic): return None

    block = SimpleBlock(url=str(tmp_path))
    assert block.log.stack_depth is not None
    assert isinstance(block.log.stack_depth, int)

    # selected() should not raise
    block.log.selected("Test message from Datablock")


# ---------------------------------------------------------------------------
# Class-method selection via co_qualname
# ---------------------------------------------------------------------------

class Widget:
    """Test class with a method that calls logger.selected()."""
    def do_work(self, logger, msg):
        logger.selected(msg)

    @classmethod
    def class_work(cls, logger, msg):
        logger.selected(msg)


class OtherWidget:
    """A different class — should not match Widget selections."""
    def do_work(self, logger, msg):
        logger.selected(msg)


def test_selection_matches_class_qualified_fqn():
    """Selection with module.Class.method matches a method call."""
    fqn = f"{__name__}.Widget.do_work"
    logger = CaptureLogger(selection=[fqn])
    Widget().do_work(logger, "Should print via class-qualified fqn")
    assert logger.printed is True


def test_selection_short_form_matches_method():
    """Selection with module.method (no class) still matches a method call."""
    fqn = f"{__name__}.do_work"
    logger = CaptureLogger(selection=[fqn])
    Widget().do_work(logger, "Should print via short fqn")
    assert logger.printed is True


def test_selection_class_qualified_rejects_wrong_class():
    """Selection for Widget.do_work should NOT match OtherWidget.do_work."""
    fqn = f"{__name__}.Widget.do_work"
    logger = CaptureLogger(selection=[fqn])
    OtherWidget().do_work(logger, "Should NOT print")
    assert logger.printed is False


def test_selection_class_qualified_classmethod():
    """Selection with module.Class.class_work matches a classmethod call."""
    fqn = f"{__name__}.Widget.class_work"
    logger = CaptureLogger(selection=[fqn])
    Widget.class_work(logger, "Should print via classmethod")
    assert logger.printed is True


if __name__ == "__main__":
    # Manual run
    logger = CaptureLogger(selection=['__main__.caller_func'])
    caller_func(logger, "Manual test")
    if logger.printed:
        print("Manual test passed!")
    else:
        print("Manual test failed!")

