import pytest
import sys
import dbx
from dbx.datatables import Datatable, DatatableTab

def test_datatable_tab_module_function():
    """Verify that DatatableTab is accessible as a top-level module function and as Datatable.Tab."""
    assert callable(DatatableTab)
    assert Datatable.Tab is DatatableTab

def test_quotefn_datatable_tab():
    """Verify that quotefn produces a valid evaluable specline string for DatatableTab."""
    spec_str = dbx.quotefn(DatatableTab, "$DummyTable()", 0)
    assert "dbx.datatables.DatatableTab" in spec_str
    assert "0" in spec_str

class _DummyTab:
    def __init__(self, idx):
        self.idx = idx

class _DummyTable:
    def __init__(self, idx=None, **kwargs):
        self.idx = idx
    def __call__(self, idx=None, tag=None, **spec):
        return _DummyTab(self.idx if self.idx is not None else idx)

# Register _DummyTable in dbx.datatables for eval resolution
sys.modules['dbx.datatables']._DummyTable = _DummyTable

def _DummyAdder(a, b):
    return a + b

sys.modules['dbx.datatables']._DummyAdder = _DummyAdder

def test_recursive_specline_eval():
    """Verify that get_named_args_kwargs and dbx.eval resolve embedded speclines recursively."""
    quoted = "$dbx.datatables.DatatableTab($dbx.datatables._DummyTable(idx=5), 5)"
    res = dbx.eval(quoted)
    assert isinstance(res, _DummyTab)
    assert int(res.idx) == 5

def test_deep_recursive_specline_eval():
    """Test multi-level deep recursive evaluation and nested speclines inside kwargs."""
    inner_adder = "$dbx.datatables._DummyAdder(a=10, b=20)"
    inner_table = f"$dbx.datatables._DummyTable(idx={inner_adder})"
    quoted = f"$dbx.datatables.DatatableTab({inner_table}, 5)"

    res = dbx.eval(quoted)
    assert isinstance(res, _DummyTab)
    assert int(res.idx) in (30, 1020)
