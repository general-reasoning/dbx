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
    return AnnotatedTab(datalake=str(tmp_path), spec=dict(n=4)).build()


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
    """Lists are several side by side; tuples are depth."""

    def test_column_specs(self):
        from dbx.datastreams import column_specs
        assert column_specs('a') == [('a', None)]
        assert column_specs(['a', 'b']) == [('a', None), ('b', None)]
        assert column_specs(('a', 'k')) == [('a', ('k',))]
        assert column_specs(('a', 's', 'c')) == [('a', ('s', 'c'))]
        assert column_specs(('a', ('s', 'c'))) == [('a', ('s', 'c'))]          # nested tuples: more depth
        assert column_specs(('a', ['k', ('s', 'c')])) == [('a', [('k',), ('s', 'c')])]
        assert column_specs(('a', 's', ['c', 'n'])) == [('a', [('s', 'c'), ('s', 'n')])]
        assert column_specs([('a', 's', 'c'), 'b']) == [('a', ('s', 'c')), ('b', None)]
        with pytest.raises(ValueError, match="must end a path"):
            column_specs(('a', ['k', 'j'], 'x'))

    def test_slice_spec(self):
        from dbx.datastreams import slice_spec
        assert slice_spec('s') == ('s', None)
        assert slice_spec(('s', 'c')) == ('s', [('c', None)])
        assert slice_spec(('s', ['c', 'd'])) == ('s', [('c', None), ('d', None)])
        assert slice_spec(('s', ('c', 'd'))) == ('s', [('c', ('d',))]), "a tuple is depth"
        assert slice_spec(('s', 'c', 'k1', 'k2')) == ('s', [('c', ('k1', 'k2'))])

    def test_merge(self):
        assert merge_column_specs(['a', ('b', 'k')]) == ['a', ('b', ('k',))]
        assert merge_column_specs([('a', 'k'), 'a']) == ['a']
        assert merge_column_specs([('a', 'k'), ('a', 'j')]) == [('a', [('k',), ('j',)])]
        assert merge_column_specs([('a', 'k'), ('a', 'k')]) == [('a', ('k',))]
        assert merge_column_specs([('a', 'k'), ('a', ('s', 'c'))]) == [('a', [('k',), ('s', 'c')])]

    def test_project(self):
        v = {'k': 1, 'j': 2}
        assert project_column(v, None) is v
        assert project_column(v, 'k') == 1
        assert project_column(v, ('k',)) == 1
        assert project_column(v, [('j',)]) == {'j': 2}
        nested = {'k': 1, 's': {'c': {'d': 7}, 'e': 8}}
        assert project_column(nested, ('s', 'c', 'd')) == 7
        assert project_column(nested, [('k',), ('s', 'c', 'd')]) == {'k': 1, 's': {'c': {'d': 7}}}

def test_a_feature_table_parses_the_triple_as_a_table_does():
    from dbx.featuretables import UpstreamTabSlices
    items = UpstreamTabSlices._norm_items_((('annotations', 'annotations', 'label'), 'features'))
    assert items == [('annotations', [('annotations', ('label',))]), ('features', None)]
    assert UpstreamTabSlices._norm_items_((('annotations', 'annotations', ['a', 'b']),)) == [
        ('annotations', [('annotations', [('a',), ('b',)])])]
    assert UpstreamTabSlices._norm_items_((('annotations', 'annotations', 'a', 'b'),)) == [
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
        text = DeclaredTab(datalake=str(tmp_path)).typestr()
        assert "topic:annotations=DATASLICE(annotations=dict(cohort='str', recurrence='int'), idx='int')" in text

    def test_the_writer_gets_json(self, tmp_path):
        t = DeclaredTab(datalake=str(tmp_path))
        assert t.declared_columns('annotations') == {'annotations': 'json', 'idx': 'int'}
        assert t.declared_schema('annotations') == {
            'annotations': {'cohort': 'str', 'recurrence': 'int'}, 'idx': 'int'}

    def test_built_and_read_back_by_key(self, tmp_path):
        t = DeclaredTab(datalake=str(tmp_path)).build()
        assert t.data(('annotations', 'annotations', 'recurrence')) == {
            'annotations': {'annotations': [0, 1, 2]}}
        assert t.dataset(('annotations', 'annotations', 'cohort'))[2] == {
            'annotations': {'annotations': 'c2'}}

    def test_an_explicit_json_agrees_with_the_declaration(self, tmp_path):
        t = DeclaredTab(datalake=str(tmp_path))
        assert t._writable_columns_({'annotations': {'annotations': 'json', 'idx': 'int'}},
                                   ['annotations']) == {'annotations': {'annotations': 'json', 'idx': 'int'}}

    def test_a_structure_that_would_render_ambiguously_is_refused(self):
        with pytest.raises(ValueError, match="may not contain '/'"):
            DATASLICE(annotations=dict(**{'a/b': 'str'}))
        with pytest.raises(TypeError):
            DATASLICE(annotations=dict(cohort=3))


class BadRowTab(Datatab):
    TOPICS = {'annotations': DATASLICE(annotations=dict(cohort='str', recurrence='int'), idx='int')}

    @dataclass
    class VAR(Datatab.VAR):
        row: dict = None

    def __build__(self):
        with self.slice_writers() as writers:
            writers['annotations'].write({'annotations': self.var.row, 'idx': 0})


class NestedTab(Datatab):
    TOPICS = {'a': DATASLICE(rec=dict(site=dict(name='str', code='int')))}

    @dataclass
    class VAR(Datatab.VAR):
        row: dict = None

    def __build__(self):
        with self.slice_writers() as writers:
            writers['a'].write({'rec': self.var.row})


class TestTheDeclaredStructureIsChecked:

    @pytest.mark.parametrize('row, error, match', [
        ({'cohort': 'c'}, ValueError, r"missing \['recurrence'\]"),
        ({'cohort': 'c', 'recurrence': 1, 'extra': 2}, ValueError, r"not declared \['extra'\]"),
        ('not a dict', TypeError, "declared a dict"),
    ])
    def test_a_row_is_refused_when_written(self, tmp_path, row, error, match):
        with pytest.raises(error, match=match):
            BadRowTab(datalake=str(tmp_path), spec=dict(row=row)).build()

    def test_a_row_that_matches_is_written(self, tmp_path):
        t = BadRowTab(datalake=str(tmp_path), spec=dict(row={'cohort': 'c', 'recurrence': 1})).build()
        assert t.data(('annotations', 'annotations', 'cohort')) == {'annotations': {'annotations': ['c']}}

    def test_nested_structure_too(self, tmp_path):
        with pytest.raises(ValueError, match=r"\['site'\].*missing \['code'\]"):
            NestedTab(datalake=str(tmp_path), spec=dict(row={'site': {'name': 'x'}})).build()
        NestedTab(datalake=str(tmp_path / 'ok'), spec=dict(row={'site': {'name': 'x', 'code': 1}})).build()

    def test_a_key_the_slice_does_not_declare_is_refused_before_reading(self, tmp_path):
        t = DeclaredTab(datalake=str(tmp_path))            # not built: nothing is read
        with pytest.raises(KeyError, match=r"declares keys \['cohort', 'recurrence'\], not \['stage'\]"):
            t.dataset(('annotations', 'annotations', 'stage'))
        with pytest.raises(KeyError, match="not"):
            t.data(('annotations', 'annotations', ['cohort', 'stage']))

    def test_a_key_of_a_column_declared_scalar_is_refused(self, tmp_path):
        with pytest.raises(TypeError, match="declares column 'idx' as 'int'"):
            DeclaredTab(datalake=str(tmp_path)).dataset(('annotations', 'idx', 'k'))

    def test_an_undeclared_slice_is_not_checked(self, tab):
        """AnnotatedTab passes its columns to the writer: there is no declaration to hold it to."""
        assert tab.dataset(('annotations', 'annotations', 'site'))[0] == {'annotations': {'annotations': 's0'}}


def test_a_table_holds_a_key_to_its_tabs_declaration(tmp_path):
    """A table's slices are declared by its tabs, not in its own TOPICS."""
    from dbx.datatables import Datatable

    class DeclaredTable(Datatable):
        TAB = DeclaredTab

        @property
        def n_tabs(self):
            return 2

    table = DeclaredTable(datalake=str(tmp_path))
    with pytest.raises(KeyError, match=r"not \['stage'\]"):
        table.dataset(('annotations', 'annotations', 'stage'))
    table.build()
    assert table.data(('annotations', 'annotations', 'recurrence'))['annotations']['annotations'] == [0, 1, 2] * 2



# ---------------------------------------------------------------------------
# Nested to any depth, and a key that is a path through it
# ---------------------------------------------------------------------------

DEEP = dict(label='int', site=dict(code='int', geo=dict(lat='float', lon='float')))


class DeepTab(Datatab):
    TOPICS = {'annotations': DATASLICE(annotations=dict(DEEP), idx='int')}

    def __build__(self):
        with self.slice_writers() as writers:
            for i in range(3):
                writers['annotations'].write({'annotations': {
                    'label': i % 2,
                    'site': {'code': 10 + i, 'geo': {'lat': 1.0 * i, 'lon': -1.0 * i}},
                }, 'idx': i})


class TestNested:

    def test_any_depth_renders_and_reads_back(self):
        from dbx.datablocks import DATADICT, literal_topics
        m = DATASLICE(annotations=dict(DEEP))
        assert repr(m) == ("DATASLICE(annotations=dict(label='int', site=dict(code='int', "
                           "geo=dict(lat='float', lon='float'))))")
        d = DATADICT('meta.json', run=dict(a=dict(b=dict(c=dict(d='int')))))
        assert repr(d) == "DATADICT('meta.json', run=dict(a=dict(b=dict(c=dict(d='int')))))"
        back = literal_topics(str({'m': m, 'd': d}))
        assert repr(back['m']) == repr(m) and repr(back['d']) == repr(d)

    def test_a_path_reads_a_deep_entry(self, tmp_path):
        t = DeepTab(datalake=str(tmp_path)).build()
        assert t.data(('annotations', 'annotations', ('site', 'geo', 'lat'))) == {
            'annotations': {'annotations': [0.0, 1.0, 2.0]}}
        row = t.dataset(('annotations', 'annotations', ['label', ('site', 'code')]))[1]
        assert row['annotations']['annotations'] == {'label': 1, 'site': {'code': 11}}

    def test_a_path_is_checked_against_the_declaration(self, tmp_path):
        t = DeepTab(datalake=str(tmp_path))
        with pytest.raises(KeyError, match=r"declares keys \['lat', 'lon'\] under 'site.geo', not \['alt'\]"):
            t.dataset(('annotations', 'annotations', ('site', 'geo', 'alt')))
        with pytest.raises(TypeError, match="declares 'label' as 'int', which has no key 'x'"):
            t.dataset(('annotations', 'annotations', ('label', 'x')))

    def test_a_deep_row_is_checked_when_written(self, tmp_path):
        class Bad(DeepTab):
            def __build__(self):
                with self.slice_writers() as writers:
                    writers['annotations'].write({'annotations': {
                        'label': 0, 'site': {'code': 1, 'geo': {'lat': 0.0}}}, 'idx': 0})
        with pytest.raises(ValueError, match=r"\['site'\]\['geo'\].*missing \['lon'\]"):
            Bad(datalake=str(tmp_path)).build()

    def test_a_collator_path(self, tmp_path):
        t = DeepTab(datalake=str(tmp_path)).build()
        c = Datacollator(spec=dict(
            signals=[('annotations', 'annotations', ['label', ('site', 'geo', 'lat')])],
            labels=[('annotations', 'annotations', ('site', 'code'))]))
        assert c.signal_pairs == (('annotations', 'annotations', 'label'),
                                  ('annotations', 'annotations', ('site', 'geo', 'lat')))
        assert c.label_pairs == (('annotations', 'annotations', ('site', 'code')),)
        data = t.data(*c.slices(), concat=True)
        assert label_vector(c, data).tolist() == [10, 11, 12]
        _, labels = c([t.dataset(*c.slices())[i] for i in range(3)])
        assert labels.reshape(-1).tolist() == [10, 11, 12]

    def test_a_probe_names_a_path_column(self):
        from dbx.probes import _pair_key_
        assert _pair_key_(('annotations', 'annotations', ('site', 'code'))) == 'annotations.annotations.site.code'



class TestListsSideBySideTuplesDeeper:

    def test_several_columns_one_of_them_narrowed(self, tmp_path):
        t = DeepTab(datalake=str(tmp_path)).build()
        row = t.dataset(('annotations', [('annotations', 'site', 'code'), 'idx']))[1]
        assert row['annotations'] == {'annotations': 11, 'idx': 1}

    def test_the_path_may_be_spelled_flat_or_nested(self, tmp_path):
        t = DeepTab(datalake=str(tmp_path)).build()
        flat = t.data(('annotations', 'annotations', 'site', 'geo', 'lat'))
        nested = t.data(('annotations', 'annotations', ('site', 'geo', 'lat')))
        also = t.data(('annotations', ('annotations', 'site', 'geo', 'lat')))
        assert flat == nested == also == {'annotations': {'annotations': [0.0, 1.0, 2.0]}}

    def test_several_keys_under_a_path(self, tmp_path):
        t = DeepTab(datalake=str(tmp_path)).build()
        row = t.dataset(('annotations', 'annotations', 'site', 'geo', ['lat', 'lon']))[2]
        assert row['annotations']['annotations'] == {'site': {'geo': {'lat': 2.0, 'lon': -2.0}}}

    def test_a_collator_reads_pairs_the_same_way(self, tmp_path):
        c = Datacollator(spec=dict(signals=[
            ('annotations', 'annotations', 'site', 'geo', 'lat'),
            ('annotations', [('annotations', 'label'), 'idx']),
        ]))
        assert c.signal_pairs == (
            ('annotations', 'annotations', ('site', 'geo', 'lat')),
            ('annotations', 'annotations', 'label'),
            ('annotations', 'idx'),
        )
