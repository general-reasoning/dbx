"""UNSAFE_compact: an anchor's block journal, and the exec journal, rewritten as one parquet file each -- no entry dropped."""
import os
from dataclasses import dataclass

import pytest

import dbx
from dbx.datablocks import DATAFILE, Datablock
from dbx.journals import DataJournal, ExecJournal, PipeJournal, read_exec_journal, write_exec_journal
from dbx.pipelines import Pipeline


class Leaf(Datablock):
    TOPICS = {'x': DATAFILE('x.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 1

    def __build__(self):
        with open(self.path('x', ensure_dirpath=True), 'w') as f:
            f.write(str(self.var.n))


class Other(Leaf):
    pass


class Entry(Pipeline):
    JOURNAL = True

    @dataclass
    class VAR(Pipeline.VAR):
        n: int = 1

    def __create__(self):
        return Leaf(spec={'n': self.var.n})


@pytest.fixture
def lake(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
    return str(tmp_path)


def _entry_files(lake, anchor):
    return DataJournal._journal_files_(__import__('fsspec').filesystem('file'), os.path.join(lake, anchor),
                                       n_workers=2, log=dbx.journals.Datalog())


def _rows(frame):
    return sorted(map(tuple, frame.drop(columns=['entry_path']).astype(str).values.tolist()))


def test_compact_rewrites_an_anchors_entries_as_one_file(lake):
    for n in range(4):
        Leaf(spec={'n': n}).build()
        Leaf(spec={'n': n}).note(f'noted {n}')
    Other().build()
    before = DataJournal.read(Leaf().anchor, datalake=lake)
    assert len(_entry_files(lake, Leaf().anchor)) == 8

    dry = DataJournal.UNSAFE_compact([Leaf().anchor], dry_run=True)
    assert len(_entry_files(lake, Leaf().anchor)) == 8 and dry['removed'].tolist() == [8]

    frame = DataJournal.UNSAFE_compact([Leaf().anchor], OVERRIDE=True)
    (row,) = frame.to_dict('records')
    assert (row['entries'], row['removed'], row['kept']) == (8, 8, 0)
    files = _entry_files(lake, Leaf().anchor)
    assert files == [os.path.join(lake, Leaf().anchor, DataJournal.COMPACTED.format(anchor=Leaf().anchor))]
    after = DataJournal.read(Leaf().anchor, datalake=lake)
    assert _rows(after) == _rows(before), "every entry, as it was"
    assert len(_entry_files(lake, Other().anchor)) == 1, "another anchor is not touched"
    assert Leaf(spec={'n': 2}).lastbuilt() is not None


def test_entries_written_after_are_compacted_with_the_compacted_ones(lake):
    Leaf(spec={'n': 1}).build()
    DataJournal.UNSAFE_compact([Leaf().anchor], OVERRIDE=True)
    Leaf(spec={'n': 2}).build()
    assert len(DataJournal.read(Leaf().anchor, datalake=lake)) == 2
    assert len(_entry_files(lake, Leaf().anchor)) == 2
    frame = DataJournal.UNSAFE_compact([Leaf().anchor], OVERRIDE=True)
    assert frame['entries'].tolist() == [2] and len(_entry_files(lake, Leaf().anchor)) == 1
    assert len(DataJournal.read(Leaf().anchor, datalake=lake)) == 2


def test_the_anchors_default_to_every_anchor(lake):
    Leaf().build()
    Other().build()
    frame = DataJournal.UNSAFE_compact(OVERRIDE=True)
    assert sorted(frame['anchor']) == sorted(dbx.anchors(lake))


def test_redirections_are_left_where_they_are(lake):
    src, dst = Leaf(spec={'n': 1}), Leaf(spec={'n': 2})
    src.build()
    dst.UNSAFE_redirect(paths=src.paths(), OVERRIDE=True)
    assert dst.redirected()
    frame = DataJournal.UNSAFE_compact([Leaf().anchor], OVERRIDE=True)
    assert frame['kept'].tolist() == [1]
    again = Leaf(spec={'n': 2})
    assert again.redirected() and again.path('x') == src.path('x')


def test_compact_asks_first(lake, monkeypatch):
    Leaf().build()
    monkeypatch.setattr('builtins.input', lambda prompt: 'n')
    assert DataJournal.UNSAFE_compact([Leaf().anchor]) is None
    assert len(_entry_files(lake, Leaf().anchor)) == 1


def test_a_compacted_entrypoint_still_counts_as_constructed(lake):
    built, unbuilt = Entry(spec={'n': 1}), Entry(spec={'n': 2})
    built.create()
    unbuilt.create()
    built.entrypoint.build()
    DataJournal.UNSAFE_compact([Leaf().anchor], OVERRIDE=True)
    assert list(PipeJournal.collect(construct_only=True)['hash']) == [built.hash]


@pytest.mark.parametrize('parallelization', ['inline', 'multithreading'])
def test_the_exec_journal_compacts_to_one_file(lake, parallelization):
    for i in range(5):
        write_exec_journal(f"print({i})  # step {i}", datalake=lake)
    before = read_exec_journal(lake)
    result = ExecJournal.UNSAFE_compact(datalake=lake, parallelization=parallelization, OVERRIDE=True)
    assert (result['entries'], result['removed']) == (5, 5)
    assert os.listdir(ExecJournal.dirpath(lake)) == [ExecJournal.COMPACTED]
    after = read_exec_journal(lake)
    assert sorted(after['id']) == sorted(before['id']) and sorted(after['comment']) == sorted(before['comment'])
    write_exec_journal("print(5)  # step 5", datalake=lake)
    assert len(read_exec_journal(lake)) == 6
    assert ExecJournal.UNSAFE_compact(datalake=lake, OVERRIDE=True)['entries'] == 6
    assert len(read_exec_journal(lake)) == 6


def test_a_dry_run_writes_and_checks_the_compacted_file_then_leaves_nothing(lake):
    for n in range(3):
        Leaf(spec={'n': n}).build()
    anchordir = os.path.join(lake, Leaf().anchor)
    listing = sorted(os.path.join(d, f) for d, _, fs in os.walk(anchordir) for f in fs)
    frame = DataJournal.UNSAFE_compact([Leaf().anchor], dry_run=True)
    assert frame['verified'].tolist() == [True]
    assert sorted(os.path.join(d, f) for d, _, fs in os.walk(anchordir) for f in fs) == listing


def test_a_compacted_file_that_does_not_read_back_as_written_leaves_the_journal_as_it_was(lake, monkeypatch):
    for n in range(3):
        Leaf(spec={'n': n}).build()
    anchordir = os.path.join(lake, Leaf().anchor)
    listing = sorted(os.path.join(d, f) for d, _, fs in os.walk(anchordir) for f in fs)

    def differs(*args, **kwargs):
        raise AssertionError("read back otherwise")

    monkeypatch.setattr('pandas.testing.assert_frame_equal', differs)
    with pytest.raises(AssertionError, match='read back otherwise'):
        DataJournal.UNSAFE_compact([Leaf().anchor], parallelization='inline', OVERRIDE=True)
    assert sorted(os.path.join(d, f) for d, _, fs in os.walk(anchordir) for f in fs) == listing


def test_an_exec_dry_run_checks_and_leaves_nothing(lake):
    for i in range(3):
        write_exec_journal(f"print({i})", datalake=lake)
    names = sorted(os.listdir(ExecJournal.dirpath(lake)))
    assert ExecJournal.UNSAFE_compact(datalake=lake, dry_run=True)['verified'] is True
    assert sorted(os.listdir(ExecJournal.dirpath(lake))) == names


def test_an_unreadable_exec_file_is_left_as_it_is(lake):
    for i in range(3):
        write_exec_journal(f"print({i})", datalake=lake)
    empty = os.path.join(ExecJournal.dirpath(lake), 'exec_empty.parquet')
    open(empty, 'wb').close()
    result = ExecJournal.UNSAFE_compact(datalake=lake, OVERRIDE=True)
    assert (result['entries'], result['removed'], result['kept']) == (3, 3, 1)
    assert sorted(os.listdir(ExecJournal.dirpath(lake))) == sorted([ExecJournal.COMPACTED, 'exec_empty.parquet'])
    assert len(read_exec_journal(lake)) == 3
