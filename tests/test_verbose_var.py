import os
import pytest
from dbx.datablocks import Datablock, Logger

# Mock Logger to capture calls
class MockLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def verbose(self, msg):
        self.calls.append(('verbose', msg))
        super().verbose(msg)

    def detailed(self, msg):
        self.calls.append(('detailed', msg))
        super().detailed(msg)

# Test Datablock subclass
class MyBlock(Datablock):
    def __build__(self):
        pass

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_ROOT', '/tmp/dbx')
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    # Patch Logger in dbx.datablocks module
    monkeypatch.setattr('dbx.datablocks.Logger', MockLogger)

def _calls(block, level):
    return [msg for lvl, msg in block.log.calls
            if lvl == level and 'Forming var from spec' in msg]

def test_verbose_var_false():
    """Verify that VERBOSE_VAR=False uses log.detailed for var formation."""
    class BlockFalse(MyBlock):
        VERBOSE_VAR = False

    b = BlockFalse()
    # Trigger var formation
    _ = b.var

    # Check if 'detailed' was used for BEGIN/END
    detailed_calls = _calls(b, 'detailed')
    assert len(detailed_calls) == 2
    assert any("BEGIN" in msg for msg in detailed_calls)
    assert any("END" in msg for msg in detailed_calls)

    # Check that 'verbose' was NOT used for this specific message
    assert len(_calls(b, 'verbose')) == 0

def test_verbose_var_true():
    """Verify that VERBOSE_VAR=True uses log.verbose for var formation."""
    class BlockTrue(MyBlock):
        VERBOSE_VAR = True

    b = BlockTrue()
    # Trigger var formation
    _ = b.var

    # Check if 'verbose' was used for BEGIN/END
    verbose_calls = _calls(b, 'verbose')
    assert len(verbose_calls) == 2
    assert any("BEGIN" in msg for msg in verbose_calls)
    assert any("END" in msg for msg in verbose_calls)

    # Check that 'detailed' was NOT used for this specific message
    assert len(_calls(b, 'detailed')) == 0

def test_verbose_var_missing():
    """Verify that missing VERBOSE_VAR defaults to False (log.detailed)."""
    b = MyBlock()
    # MyBlock doesn't define VERBOSE_VAR, so it inherits False from Datablock
    _ = b.var

    assert len(_calls(b, 'detailed')) == 2

def test_legacy_verbose_config_true():
    """The deprecated VERBOSE_CONFIG spelling still selects log.verbose."""
    class LegacyBlock(MyBlock):
        VERBOSE_CONFIG = True

    b = LegacyBlock()
    _ = b.var

    assert len(_calls(b, 'verbose')) == 2
    assert len(_calls(b, 'detailed')) == 0
