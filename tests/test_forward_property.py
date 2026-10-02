"""
`forward_property`: a cached_property that also answers on the class.

Made for a TOPICS the instance declares -- a table reads its TAB's slice names
off the class, while a tab's columns follow from its VAR.
"""
import pytest

from dbx.datablocks import Datablock, forward_property, class_declaration
from dbx.datatables import DATASLICE


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Counted:
    calls = 0

    @forward_property(str)          # the promise: a str
    def value(self):
        """Computed once per instance."""
        type(self).calls += 1
        return f'on {id(self)}'


def test_the_class_reads_the_class_value():
    assert Counted.value is str


def test_an_instance_computes_once_and_caches():
    Counted.calls = 0
    c = Counted()
    first = c.value
    assert first == f'on {id(c)}' and c.value is first
    assert Counted.calls == 1
    assert c.__dict__['value'] is first


def test_assigning_on_an_instance_replaces_it():
    c = Counted()
    c.value = 'assigned'
    assert c.value == 'assigned'
    assert Counted.value is str


def test_it_keeps_the_docstring():
    assert Counted.__dict__['value'].__doc__ == "Computed once per instance."


class Columns(Datablock):
    VAR_COLUMNS = ('a', 'b')

    @forward_property({'rows': DATASLICE})
    def TOPICS(self):
        return {'rows': DATASLICE({c: 'int' for c in self.VAR_COLUMNS})}


def test_a_block_declares_on_the_class_and_computes_on_the_instance(tmp_path):
    assert Columns.TOPICS == {'rows': DATASLICE}
    block = Columns(datalake=str(tmp_path))
    assert block.TOPICS['rows'].columns == {'a': 'int', 'b': 'int'}
    assert class_declaration(Columns, 'TOPICS') == {'rows': DATASLICE}


def test_repeating_the_class_value_on_a_subclass_is_refused():
    """The copy would hide the instance's declaration -- the columns -- silently."""
    with pytest.raises(TypeError, match="Delete the line"):
        class Copy(Columns):
            TOPICS = {'rows': DATASLICE}


def test_a_different_declaration_on_a_subclass_stands(tmp_path):
    class Other(Columns):
        TOPICS = {'other': DATASLICE(x='int')}

    # Each DATASLICE(...) call makes its own class: compare the declarations as rendered.
    assert str(Other(datalake=str(tmp_path)).TOPICS) == str({'other': DATASLICE(x='int')})


def test_a_partition_part_answers_on_the_class_with_a_dict():
    """Its TOPICS are its table's, per instance; on the class there is no table, and it says so as a dict."""
    from dbx.datatables import DatatablePart
    assert DatatablePart.TOPICS == {}


def test_a_partition_part_declares_its_tab_forward():
    """Every part's tabs are Datatabs; which one is the partitioned table's -- and so is its BLOCK."""
    from dbx.datatables import Datatab, DatatablePart
    assert DatatablePart.TAB is Datatab and DatatablePart.BLOCK is Datatab


class Promising(Datablock):
    """Declares a slice forward; the instance's value is whatever `answer` says."""
    answer = None

    @forward_property({'rows': DATASLICE})
    def TOPICS(self):
        return self.answer


def test_an_instance_keeps_the_promise_or_is_refused(tmp_path):
    from dbx.datablocks import DATAFILE

    class Refines(Promising):
        answer = {'rows': DATASLICE(x='int'), 'more': DATAFILE('m.txt')}   # more is allowed

    assert 'more' in Refines(datalake=str(tmp_path)).TOPICS

    class Breaks(Promising):
        answer = {'rows': DATAFILE('rows.txt')}

    with pytest.raises(TypeError, match=r"Breaks.TOPICS\['rows'\]: is DATAFILE\('rows.txt'\), which is not a DATASLICE"):
        Breaks(datalake=str(tmp_path)).TOPICS

    class Drops(Promising):
        answer = {'other': DATASLICE}

    with pytest.raises(TypeError, match=r"lacks 'rows', which is declared"):
        Drops(datalake=str(tmp_path)).TOPICS


def test_the_refusal_explains_itself():
    with pytest.raises(TypeError) as e:
        class Copy(Columns):
            TOPICS = {'rows': DATASLICE}
    msg = str(e.value)
    assert "only repeats Columns.TOPICS's forward declaration" in msg
    assert "found before Columns's, on the class and on every instance" in msg
    assert "Delete the line" in msg
