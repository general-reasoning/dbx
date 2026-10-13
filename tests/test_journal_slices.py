"""A journal entry records what can be read from its block: ``slices``, its own and upstream, with their columns."""
import ast

import pytest

pytest.importorskip("torch")

from dbx.datatables import Datatable  # noqa: E402
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
    assert slices['features'] == table.tablet(0).declared_columns('features')
    assert slices == table.readable_slices()


def test_a_tablets_entry_says_the_same_of_itself(tmp_path):
    tablet = _feature_table(str(tmp_path), 'features').tablet(0)
    assert set(tablet.readable_slices()) == set(tablet.slices(recursive=True))
    assert tablet.readable_slices()['features'] == tablet.declared_columns('features')


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

