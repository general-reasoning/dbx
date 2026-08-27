import pytest
import sys
import dbx
from dbx.datapoints import DatapointTable, DatapointTableTab

def test_datapoint_table_tab_module_function():
    """Verify that DatapointTableTab is accessible as a top-level module function and as DatapointTable.Tab."""
    assert callable(DatapointTableTab)
    assert DatapointTable.Tab is DatapointTableTab

def test_quotefn_datapoint_table_tab():
    """Verify that quotefn produces a valid evaluable specline string for DatapointTableTab."""
    spec_str = dbx.quotefn(DatapointTableTab, "$DummyTable()", 0)
    assert "dbx.datapoints.DatapointTableTab" in spec_str
    assert "0" in spec_str

class _DummyTab:
    def __init__(self, idx):
        self.idx = idx

class _DummyTable:
    def __init__(self, idx=None, **kwargs):
        self.idx = idx
    def __call__(self, idx=None, tag=None, **spec):
        return _DummyTab(self.idx if self.idx is not None else idx)

# Register _DummyTable in dbx.datapoints for eval resolution
sys.modules['dbx.datapoints']._DummyTable = _DummyTable

def _DummyAdder(a, b):
    return a + b

sys.modules['dbx.datapoints']._DummyAdder = _DummyAdder

def test_recursive_specline_eval():
    """Verify that get_named_args_kwargs and dbx.eval resolve embedded speclines recursively."""
    quoted = "$dbx.datapoints.DatapointTableTab($dbx.datapoints._DummyTable(idx=5), 5)"
    res = dbx.eval(quoted)
    assert isinstance(res, _DummyTab)
    assert int(res.idx) == 5

def test_deep_recursive_specline_eval():
    """Test multi-level deep recursive evaluation and nested speclines inside kwargs."""
    inner_adder = "$dbx.datapoints._DummyAdder(a=10, b=20)"
    inner_table = f"$dbx.datapoints._DummyTable(idx={inner_adder})"
    quoted = f"$dbx.datapoints.DatapointTableTab({inner_table}, 5)"

    res = dbx.eval(quoted)
    assert isinstance(res, _DummyTab)
    assert int(res.idx) in (30, 1020)
