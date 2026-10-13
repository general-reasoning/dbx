"""A journal entry records what can be read from its block: ``slices``, its own and upstream, with their columns."""
import ast

import pytest

pytest.importorskip("torch")

from dbx.datatables import DATASLICE, Datatable, Datatablet  # noqa: E402
from test_featuretablet import DummySampleTable, _feature_table  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


def _entry_slices(block):
    j = block.datajournal()
    return ast.literal_eval(j.iloc[-1]['slices'])


def test_a_tables_entry_says_what_its_tablets_and_upstream_hold(tmp_path):
    table = _feature_table(str(tmp_path), 'features')
    assert table.topics() == ['done'], "a table's own topics are its markers alone"
    slices = _entry_slices(table)
    assert set(slices) == set(table.slices(recursive=True)) >= {'features', 'samples', 'labels'}
    assert slices['features'] == table.tablet(0).declared_schema('features')
    assert slices == table.readable_slices()


def test_a_tablets_entry_says_the_same_of_itself(tmp_path):
    tablet = _feature_table(str(tmp_path), 'features').tablet(0)
    assert set(tablet.readable_slices()) == set(tablet.slices(recursive=True))
    assert tablet.readable_slices()['features'] == tablet.declared_schema('features')


def test_slices_not_yet_readable_are_named_and_not_raised(tmp_path, monkeypatch):
    """Journaled before its tablets can be asked -- a part's partition not dealt, say -- a slice's columns are None."""
    table = DummySampleTable(datalake=str(tmp_path), spec=dict(samples_per_tablet=5), tag='samples')

    def unreadable(self, idx):
        raise FileNotFoundError("not dealt")
    monkeypatch.setattr(type(table), 'tablet', unreadable)
    slices = table.readable_slices()
    assert set(slices) == set(table.slices(recursive=True)) and all(v is None for v in slices.values())
    table.write_journal_entry(event='note')         # does not raise


def test_slices_read_back_right_after_topics_and_paths(tmp_path):
    columns = list(_feature_table(str(tmp_path), 'features').datajournal().columns)
    assert columns[columns.index('topics'):columns.index('topics') + 3] == ['topics', 'paths', 'slices']


def test_a_note_reads_back_right_after_slices(tmp_path):
    table = _feature_table(str(tmp_path), 'features')
    table.note('what this table is for', event='note', inline=True)
    columns = list(table.datajournal().columns)
    assert columns[columns.index('topics'):columns.index('topics') + 4] == ['topics', 'paths', 'slices', 'note']


class Annotated(Datatablet):
    """A tablet whose dict column declares its keys."""
    TOPICS = {'meta': DATASLICE(meta=dict(case_id='str', cohort='str', clinical='json')),
              'x': DATASLICE(x='int32')}


def test_a_dict_column_is_recorded_with_its_keys(tmp_path):
    tablet = Annotated(datalake=str(tmp_path), tag='annotated')
    slices = tablet.readable_slices()
    assert slices['meta'] == {'meta': {'case_id': 'str', 'cohort': 'str', 'clinical': 'json'}}
    assert slices['x'] == {'x': 'int32'}
    tablet.write_journal_entry(event='note')
    assert ast.literal_eval(tablet.datajournal().iloc[0]['slices']) == slices


def test_a_plain_tables_slices_are_its_tablets(tmp_path):
    """A table reading nothing upstream -- no _route_ -- owns its slices, declared by its tablets."""
    table = DummySampleTable(datalake=str(tmp_path), spec=dict(samples_per_tablet=5), tag='samples').build()
    assert not hasattr(table, '_route_')
    slices = table.readable_slices()
    assert set(slices) == set(table.slices(recursive=True))
    assert slices == {s: table.tablet(0).declared_schema(s) for s in slices}




def test_rebuilding_a_valid_tables_tablets_is_journaled_on_the_table(tmp_path):
    """build_blocks leaves an entry of the table's own, with its slices -- beside its build:end, not over it."""
    table = DummySampleTable(datalake=str(tmp_path), spec=dict(samples_per_tablet=5), tag='samples').build()
    events = lambda: sorted(table.datajournal()['event'])
    assert events() == ['build:end']
    table.build_blocks()                            # every tablet valid: nothing built, nothing written
    assert events() == ['build:end']
    table.build_blocks([0])                         # the same instance that wrote build:end
    assert events() == ['build:end', 'build_blocks:end']
    entry = table.datajournal(event='build_blocks:end').iloc[-1]
    assert ast.literal_eval(entry['slices']) == table.readable_slices()
    assert entry['note'] == f"1 of {table.n_blocks} blocks: [0]"


def test_update_is_a_note_under_its_own_event(tmp_path):
    table = _feature_table(str(tmp_path), 'features')
    assert table.update('slices re-recorded', inline=True) is table
    entry = table.datajournal(event='update').iloc[-1]
    assert entry['note'] == 'slices re-recorded'
    assert ast.literal_eval(entry['slices']) == table.readable_slices()
    assert 'build:end' in set(table.datajournal()['event']), "beside the build's entry, not over it"
