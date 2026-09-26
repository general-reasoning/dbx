"""
``(slice, column, key)``: one entry of a column that holds a dict.

A slice may carry a column whose value is a dict -- an annotation record, say
-- when what a caller wants is one entry of it. ``dataset()`` and ``data()``
take the triple and pass on only that entry (or, for a list of keys, a dict of
just those), and a `Datacollator` pair may be the same triple.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from dbx.datablocks import Datablock
from dbx.datatables import SLICETOPIC, Datatab
from dbx.datastreams import column_spec, merge_column_specs, project_column
from dbx.featuretables import Datacollator
from dbx.probes import label_vector


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class AnnotatedTab(Datatab):
    TOPICS = {'samples': SLICETOPIC, 'annotations': SLICETOPIC}

    @dataclass
    class VAR(Datatab.VAR):
        n: int = 4

    def __build__(self):
        spec = {'samples': {'x': 'ndarray:float32'},
                'annotations': {'annotations': 'json', 'idx': 'int'}}
        with self.slice_writers(spec) as writers:
            for i in range(self.var.n):
                writers['samples'].write({'x': np.full(3, i, dtype=np.float32)})
                writers['annotations'].write({
                    'annotations': {'label': i % 2, 'site': f"s{i}", 'area': 0.5 * i},
                    'idx': i,
                })


@pytest.fixture
def tab(tmp_path):
    return AnnotatedTab(url=str(tmp_path), spec=dict(n=4)).build()


class TestDatasetAndData:

    def test_a_triple_takes_one_entry_as_rows_are_assembled(self, tab):
        ds = tab.dataset(('annotations', 'annotations', 'label'))
        assert [ds[i]['annotations'] for i in range(4)] == [{'annotations': i % 2} for i in range(4)]

    def test_a_list_of_keys_is_a_dict_of_just_those(self, tab):
        row = tab.dataset(('annotations', 'annotations', ['label', 'site']))[1]
        assert row['annotations']['annotations'] == {'label': 1, 'site': 's1'}

    def test_alongside_other_slices_and_columns(self, tab):
        row = tab.dataset('samples', ('annotations', 'annotations', 'label'), ('annotations', 'idx'))[3]
        assert row['annotations'] == {'annotations': 1, 'idx': 3}
        assert row['samples']['x'].tolist() == [3.0, 3.0, 3.0]

    def test_data_takes_it_too(self, tab):
        got = tab.data(('annotations', 'annotations', 'label'))
        assert got == {'annotations': {'annotations': [0, 1, 0, 1]}}
        stacked = tab.data(('annotations', 'annotations', 'label'), concat=True)
        assert np.asarray(stacked['annotations']['annotations']).tolist() == [0, 1, 0, 1]

    def test_asked_whole_and_in_part_it_is_read_whole(self, tab):
        row = tab.dataset(('annotations', 'annotations'), ('annotations', 'annotations', 'label'))[0]
        assert row['annotations']['annotations'] == {'label': 0, 'site': 's0', 'area': 0.0}

    def test_two_keys_asked_separately_are_both_taken(self, tab):
        row = tab.dataset(('annotations', 'annotations', 'label'),
                          ('annotations', 'annotations', 'site'))[2]
        assert row['annotations']['annotations'] == {'label': 0, 'site': 's2'}

    def test_a_missing_key_says_which(self, tab):
        with pytest.raises(KeyError, match=r"no key\(s\) \['nope'\]"):
            tab.dataset(('annotations', 'annotations', 'nope'))[0]

    def test_a_key_of_a_column_that_is_not_a_dict(self, tab):
        with pytest.raises(TypeError, match="not a dict"):
            tab.data(('annotations', 'idx', 'label'))


class TestCollator:
    """A collator pair may be a triple -- which is what a probe's labels need."""

    def test_on_a_stacked_slice(self, tab):
        c = Datacollator(spec=dict(signals=[('samples', 'x')],
                                   labels=[('annotations', 'annotations', 'label')]))
        assert c.slices() == ['samples', 'annotations']
        data = tab.data(*c.slices(), concat=True)
        signals, labels = c(data)
        assert signals.shape == (4, 3)
        assert np.asarray(labels).reshape(-1).tolist() == [0, 1, 0, 1]
        assert label_vector(c, data).tolist() == [0, 1, 0, 1]

    def test_on_rows(self, tab):
        c = Datacollator(spec=dict(signals=[('samples', 'x')],
                                   labels=[('annotations', 'annotations', 'label')]))
        ds = tab.dataset(*c.slices())
        _, labels = c([ds[i] for i in range(4)])
        assert labels.reshape(-1).tolist() == [0, 1, 0, 1]

    def test_a_list_of_keys_is_one_pair_per_key(self):
        c = Datacollator(spec=dict(signals=[('annotations', 'annotations', ['label', 'area'])]))
        assert c.signal_pairs == (('annotations', 'annotations', 'label'),
                                  ('annotations', 'annotations', 'area'))

    def test_a_plain_pair_is_unchanged(self):
        c = Datacollator(spec=dict(signals=[('samples', 'x')], labels=[('annotations', 'idx')]))
        assert c.signal_pairs == (('samples', 'x'),) and c.label_pairs == (('annotations', 'idx'),)


class TestSpecs:

    def test_column_spec(self):
        assert column_spec('a') == ('a', None)
        assert column_spec(('a', 'k')) == ('a', 'k')
        assert column_spec(('a', ['k', 'j'])) == ('a', ('k', 'j'))

    def test_merge(self):
        assert merge_column_specs(['a', ('b', 'k')]) == ['a', ('b', 'k')]
        assert merge_column_specs([('a', 'k'), 'a']) == ['a']
        assert merge_column_specs([('a', 'k'), ('a', 'j')]) == [('a', ('k', 'j'))]
        assert merge_column_specs([('a', 'k'), ('a', 'k')]) == [('a', 'k')]

    def test_project(self):
        v = {'k': 1, 'j': 2}
        assert project_column(v, None) is v
        assert project_column(v, 'k') == 1
        assert project_column(v, ('j',)) == {'j': 2}


def test_a_feature_table_parses_the_triple_as_a_table_does():
    from dbx.featuretables import _UpstreamSlices
    items = _UpstreamSlices._norm_items((('annotations', 'annotations', 'label'), 'features'))
    assert items == [('annotations', [('annotations', 'label')]), ('features', None)]
    assert _UpstreamSlices._norm_items((('annotations', 'annotations', ['a', 'b']),)) == [
        ('annotations', [('annotations', ('a', 'b'))])]


from dbx.datatables import DATASLICE


class DeclaredTab(Datatab):
    """Declares its dict column by structure, and so writes with no columns passed."""
    TOPICS = {'annotations': DATASLICE(annotations=dict(cohort='str', recurrence='int'), idx='int')}

    def __build__(self):
        with self.slice_writers() as writers:
            for i in range(3):
                writers['annotations'].write({'annotations': {'cohort': f"c{i}", 'recurrence': i},
                                              'idx': i})


class TestADictColumnDeclaredByItsStructure:

    def test_it_renders_the_structure(self):
        m = DATASLICE(annotations=dict(cohort='str', recurrence='int'), idx='int')
        assert repr(m) == "DATASLICE(annotations=dict(cohort='str', recurrence='int'), idx='int')"

    def test_it_is_in_the_hash(self, tmp_path):
        text = DeclaredTab(url=str(tmp_path)).typestr()
        assert "topic:annotations=DATASLICE(annotations=dict(cohort='str', recurrence='int'), idx='int')" in text

    def test_the_writer_gets_json(self, tmp_path):
        t = DeclaredTab(url=str(tmp_path))
        assert t.declared_columns('annotations') == {'annotations': 'json', 'idx': 'int'}
        assert t.declared_schema('annotations') == {
            'annotations': {'cohort': 'str', 'recurrence': 'int'}, 'idx': 'int'}

    def test_built_and_read_back_by_key(self, tmp_path):
        t = DeclaredTab(url=str(tmp_path)).build()
        assert t.data(('annotations', 'annotations', 'recurrence')) == {
            'annotations': {'annotations': [0, 1, 2]}}
        assert t.dataset(('annotations', 'annotations', 'cohort'))[2] == {
            'annotations': {'annotations': 'c2'}}

    def test_an_explicit_json_agrees_with_the_declaration(self, tmp_path):
        t = DeclaredTab(url=str(tmp_path))
        assert t._writable_columns({'annotations': {'annotations': 'json', 'idx': 'int'}},
                                   ['annotations']) == {'annotations': {'annotations': 'json', 'idx': 'int'}}

    def test_a_structure_that_would_render_ambiguously_is_refused(self):
        with pytest.raises(ValueError, match="may not contain '/'"):
            DATASLICE(annotations=dict(**{'a/b': 'str'}))
        with pytest.raises(TypeError):
            DATASLICE(annotations=dict(cohort=3))
