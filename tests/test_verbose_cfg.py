import os
import pytest
from dbx.dbx import Datablock, Logger

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
    # Patch Logger in dbx.dbx module
    monkeypatch.setattr('dbx.dbx.Logger', MockLogger)

def test_verbose_config_false():
    """Verify that VERBOSE_CONFIG=False uses log.detailed for cfg formation."""
    class BlockFalse(MyBlock):
        VERBOSE_CONFIG = False
        
    b = BlockFalse()
    # Trigger cfg formation
    _ = b.cfg
    
    # Check if 'detailed' was used for BEGIN/END
    detailed_calls = [msg for level, msg in b.log.calls if level == 'detailed' and 'Forming cfg from spec' in msg]
    assert len(detailed_calls) == 2
    assert any("BEGIN" in msg for msg in detailed_calls)
    assert any("END" in msg for msg in detailed_calls)
    
    # Check that 'verbose' was NOT used for this specific message
    verbose_calls = [msg for level, msg in b.log.calls if level == 'verbose' and 'Forming cfg from spec' in msg]
    assert len(verbose_calls) == 0

def test_verbose_config_true():
    """Verify that VERBOSE_CONFIG=True uses log.verbose for cfg formation."""
    class BlockTrue(MyBlock):
        VERBOSE_CONFIG = True
        
    b = BlockTrue()
    # Trigger cfg formation
    _ = b.cfg
    
    # Check if 'verbose' was used for BEGIN/END
    verbose_calls = [msg for level, msg in b.log.calls if level == 'verbose' and 'Forming cfg from spec' in msg]
    assert len(verbose_calls) == 2
    assert any("BEGIN" in msg for msg in verbose_calls)
    assert any("END" in msg for msg in verbose_calls)
    
    # Check that 'detailed' was NOT used for this specific message
    detailed_calls = [msg for level, msg in b.log.calls if level == 'detailed' and 'Forming cfg from spec' in msg]
    assert len(detailed_calls) == 0

def test_verbose_config_missing():
    """Verify that missing VERBOSE_CONFIG defaults to False (log.detailed)."""
    b = MyBlock()
    # MyBlock doesn't define VERBOSE_CONFIG, so it inherits False from Datablock
    _ = b.cfg
    
    detailed_calls = [msg for level, msg in b.log.calls if level == 'detailed' and 'Forming cfg from spec' in msg]
    assert len(detailed_calls) == 2
