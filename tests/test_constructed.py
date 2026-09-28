"""
dbx.constructed(): the block journal entries of what was built, redirected or copied in.
"""
import pytest

import dbx
from dbx.datablocks import DATAFILE, Datablock
from dbx.journals import DatajournalFrame


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Apple(Datablock):
    TOPICS = {'x': DATAFILE('x.txt')}

    def __build__(self):
        with open(self.path('x', ensure_dirpath=True), 'w') as f:
            f.write('apple')


class Pear(Apple):
    pass


@pytest.fixture
def lake(tmp_path):
    Apple(datalake=str(tmp_path)).build()
    Pear(datalake=str(tmp_path)).build()
    Apple(datalake=str(tmp_path), tag='second').build()
    # Journalled, and not a construction.
    Apple(datalake=str(tmp_path)).write_journal_entry(event='note:checked')
    return str(tmp_path)


def test_one_anchor(lake):
    frame = dbx.constructed(Apple, datalake=lake)
    assert isinstance(frame, DatajournalFrame)
    assert set(frame['anchor']) == {Apple.anchor} and len(frame) == 2
    assert set(frame['event']) == {'build:end'}
    assert len(dbx.constructed(Apple.anchor, datalake=lake)) == 2


def test_every_anchor_newest_first(lake):
    frame = dbx.constructed(datalake=lake)
    assert set(frame['anchor']) == {Apple.anchor, Pear.anchor} and len(frame) == 3
    assert list(frame['datetime']) == sorted(frame['datetime'], reverse=True)


def test_what_is_not_a_construction_is_left_out(lake):
    assert 'note:checked' not in set(dbx.constructed(datalake=lake)['event'])


def test_event_none_is_every_event(lake):
    everything = dbx.constructed(Apple, event=None, datalake=lake)
    assert set(everything['event']) == {'build:end', 'note:checked'}


def test_another_event_filter(lake):
    notes = dbx.constructed(event='^note:', datalake=lake)
    assert set(notes['event']) == {'note:checked'} and len(notes) == 1


def test_an_empty_lake(tmp_path):
    frame = dbx.constructed(datalake=str(tmp_path))
    assert isinstance(frame, DatajournalFrame) and len(frame) == 0
