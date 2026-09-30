"""
`ExecjournalFrame`: how the exec journal displays, and what a frame derived from it is.

The display is compact -- short ids and sessions, timestamps to the second,
path lists as counts, nothing repeated, no traceback -- and the data is not:
every value is there, in full, to index, filter and read. A short id is enough
to get a row by. A selection, a slice or a sort is an `ExecjournalFrame` still,
as a selection of a block journal is a `DatajournalFrame`.
"""
from dataclasses import dataclass

import pandas as pd
import pytest

import dbx
from dbx.datablocks import Datablock
from dbx.journals import DatajournalFrame, ExecjournalEntry, ExecjournalFrame


@pytest.fixture(autouse=True)
def lake(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.setenv('DBX_ROOT', str(tmp_path))


class Built(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


@pytest.fixture
def ej(tmp_path):
    dbx.exec("Built(datalake=root, spec={'x': 1}).build()  # built", Built=Built, root=str(tmp_path))
    dbx.exec("1 + 1  # added")
    with pytest.raises(ZeroDivisionError):
        dbx.exec("1/0  # failed")
    return dbx.execjournal()


class TestTheDisplay:

    def test_ids_and_sessions_are_short(self, ej):
        text = repr(ej)
        for full in list(ej['id']) + list(ej['session']):
            assert full not in text
            assert f"{full[:8]}…" in text

    def test_id_is_not_repeated_beside_the_index(self, ej):
        shown = ExecjournalFrame._display_frame_(ej)
        assert 'id' not in shown.columns and shown.index.name == 'id'

    def test_with_a_numbered_index_the_id_column_is_short(self, tmp_path):
        dbx.exec("1")
        shown = ExecjournalFrame._display_frame_(dbx.execjournal(index=None))
        assert shown['id'].iloc[0].endswith('…') and len(shown['id'].iloc[0]) == 9

    def test_timestamps_to_the_second_and_the_start_once(self, ej):
        shown = ExecjournalFrame._display_frame_(ej)
        assert 'exec:start:datetime' not in shown.columns
        assert all('.' not in v for v in shown['datetime'])
        assert list(shown.columns[:2]) == ['datetime', 'exec']

    def test_paths_are_counted_and_the_traceback_left_out(self, ej):
        shown = ExecjournalFrame._display_frame_(ej)
        assert list(shown['datajournal_entries']) == ['[0]', '[0]', '[1]']
        assert 'traceback' not in shown.columns and 'exception' in shown.columns

    def test_comment_and_success_last(self, ej):
        assert list(ExecjournalFrame._display_frame_(ej).columns[-2:]) == ['comment', 'success']

    def test_the_data_is_untouched(self, ej):
        repr(ej)
        ej._repr_html_()
        assert all(len(i) == 36 for i in ej.index)
        assert ej['traceback'].notna().sum() == 1
        assert '.' in ej['datetime'].iloc[0]

    def test_every_column_whatever_the_width_option(self, ej):
        with pd.option_context('display.width', 40, 'display.max_columns', 3):
            text = repr(ej)
        for col in ('exec', 'exception', 'session', 'comment', 'success'):
            assert col in text

    def test_show_full_prints_every_value(self, ej, capsys):
        ej.show(full=True, width=10_000, max_colwidth=10_000)
        out = capsys.readouterr().out
        assert ej['id'].iloc[0] in out and 'traceback' in out

    def test_show_compact(self, ej, capsys):
        ej.show(width=10_000)
        out = capsys.readouterr().out
        assert ej['id'].iloc[0] not in out and ej['id'].iloc[0][:8] in out

    def test_an_empty_journal(self, tmp_path):
        repr(dbx.execjournal())


class TestAShortIdIsEnough:

    def test_get_by_the_displayed_id(self, ej):
        full = ej.index[1]
        assert ej.get(full[:8])['id'] == full
        assert ej(f"{full[:8]}…")['id'] == full
        assert dbx.execjournal(loc=full[:8])['id'] == full

    def test_a_full_id_still(self, ej):
        assert ej.get(ej.index[0])['id'] == ej.index[0]

    def test_a_prefix_of_several_is_an_error(self, ej):
        with pytest.raises(KeyError, match='of 3'):
            ej.get('')

    def test_a_prefix_of_none_is_an_error(self, ej):
        with pytest.raises(KeyError, match='of 0'):
            ej.get('not-an-id')

    def test_loc_stays_exact(self, ej):
        with pytest.raises(KeyError):
            ej.loc[ej.index[0][:8]]


class TestDerivedFrames:

    def test_a_selection_is_an_execjournal_frame_with_its_datalake(self, ej):
        sub = ej[['exec', 'comment']]
        assert type(sub) is ExecjournalFrame
        assert sub.datalake == ej.datalake
        assert ej.index[0][:8] + '…' in repr(sub)

    def test_a_filter_a_sort_a_head(self, ej):
        for derived in (ej[ej['success'] == False], ej.sort_values('exec'), ej.head(1)):
            assert type(derived) is ExecjournalFrame
            assert derived.datalake == ej.datalake

    def test_a_filtered_row_reaches_what_it_wrote(self, ej):
        built = ej[ej['comment'] == 'built']
        entry = built.get(built.index[0])
        assert isinstance(entry, ExecjournalEntry)
        assert len(entry.dataentries()) == 1

    def test_a_block_journal_selection_is_not_read_again(self, ej, tmp_path):
        frame = dbx.datajournal(Built, datalake=str(tmp_path), storage_options={'k': 1})
        sub = frame[['anchor', 'hash']]
        assert type(sub) is DatajournalFrame
        # Normalizing it again would add `type` and `signature` back.
        assert list(sub.columns) == ['anchor', 'hash']
        assert sub.storage_options == {'k': 1}
        built = frame[frame['event'] == 'build:end']
        assert type(built) is DatajournalFrame and built.get(built.index[0])['anchor'] == frame['anchor'].iloc[0]
