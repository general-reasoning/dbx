"""
`Datajournal`: the handle a block writes its journal through and reads it with.

Constructing one touches no storage; ``read()`` reads, ``write()`` writes. A
handle has a ``session`` fixed for its lifetime and written to every entry,
and remembers every entry path it wrote (``written_entries()``).
"""
import copy
import functools
import os
import pickle
import time
from dataclasses import dataclass

import pandas as pd
import pytest

import dbx
from dbx.datablocks import (DEFAULT_DATAJOURNAL, Datablock, Datajournal,
                            DatajournalEntry, DatajournalFrame, Datastack)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Built(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


def block(tmp_path, x=1, **kwargs):
    return Built(datalake=str(tmp_path), spec={'x': x}, **kwargs)


class Stack(Datastack):
    BLOCK = Built
    TOPICS = {'meta': 'meta.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 2

    @property
    def n_blocks(self):
        return self.var.n

    def __block__(self, idx):
        return Built(datalake=self.url, spec={'x': idx})


class TestReadingIsAQuery:
    """Reading a journal is a query over storage: it takes no handle, and a handle reads nothing."""

    def test_construction_reads_nothing(self, tmp_path):
        """A root that does not exist is only an error once something reads it."""
        Datajournal()
        with pytest.raises(FileNotFoundError):
            Datajournal.read('some.Anchor', datalake=str(tmp_path / 'nowhere'))

    def test_read_returns_the_frame(self, tmp_path):
        b = block(tmp_path)
        b.build()
        frame = Datajournal.read(b.anchor, datalake=str(tmp_path))
        assert isinstance(frame, DatajournalFrame) and len(frame) == 1
        assert isinstance(Datajournal.read(b.anchor, loc=0, datalake=str(tmp_path)), DatajournalEntry)

    def test_a_block_reads_under_its_own_datalake(self, tmp_path, monkeypatch):
        """Whatever the default datalake is."""
        monkeypatch.setenv('DBX_ROOT', str(tmp_path / 'elsewhere'))
        b = block(tmp_path / 'lake')
        b.build()
        assert len(b.journal()) == 1


class TestABlockHasNoJournalOfItsOwn:
    """Which session an entry goes under is the open ``with``'s to say, never a block's."""

    def test_datajournal_is_not_an_argument(self, tmp_path):
        with pytest.raises(TypeError, match=r"with dbx\.Datajournal\(\)"):
            block(tmp_path, datajournal=Datajournal())

    def test_with_no_scope_open_the_process_default(self, tmp_path):
        assert Datajournal.current() is DEFAULT_DATAJOURNAL
        b = block(tmp_path)
        b.build()
        assert b.journal(loc=0).block.session == DEFAULT_DATAJOURNAL.session

    def test_a_handle_is_shared_not_copied(self):
        dj = Datajournal()
        assert copy.copy(dj) is dj and copy.deepcopy(dj) is dj


def _identity(b):
    return dict(hash=b.hash, code=b.code, key=b.key, anchorkeypath=b.anchorkeypath,
                signature=b.signaturestr(), type=b.typestr(), quote=b.quote(), cite=b.cite())


@pytest.mark.pinned
class TestAJournalIsNotIdentity:
    """Which journal a block writes through says nothing about what the block IS.

    A journal is where the record of a build goes. If it reached the
    signature, the same configuration run under two `dbx.exec` commands --
    two sessions -- would be two blocks, keyed apart, built twice, and each
    unable to find the other's data.
    """

    @pytest.mark.parametrize('cls, spec', [(Built, {'x': 3}), (Stack, {'n': 2})])
    def test_every_way_of_getting_one_leaves_identity_alone(self, tmp_path, cls, spec):
        url = str(tmp_path)
        ref = _identity(cls(url=url, spec=spec))
        # Two sessions -- two commands -- and a block carried across each way it travels.
        for dj in (Datajournal(), Datajournal()):
            with dj:
                scoped = cls(url=url, spec=spec)
                assert Datajournal.current() is dj
                for b in (scoped, scoped.set(tag=None), pickle.loads(pickle.dumps(scoped))):
                    assert _identity(b) == ref

    def test_nor_that_of_the_blocks_it_is_handed_to(self, tmp_path):
        plain = Stack(url=str(tmp_path))
        with Datajournal():
            scoped = Stack(url=str(tmp_path))
            for i in range(plain.n_blocks):
                assert _identity(scoped.block(i)) == _identity(plain.block(i))

    def test_the_guard_distinct_blocks_stay_distinct(self, tmp_path):
        """So the parity above cannot pass by everything collapsing to one value."""
        assert block(tmp_path, x=1).hash != block(tmp_path, x=2).hash


class TestSession:

    def test_fixed_for_the_lifetime_and_distinct_between_handles(self):
        dj = Datajournal()
        assert dj.session == dj.session
        assert Datajournal().session != dj.session

    def test_written_to_every_entry(self, tmp_path):
        with Datajournal() as dj:
            b = block(tmp_path)
            b.write_journal_entry(event='note', journal_prefix='a-')
            b.write_journal_entry(event='note', journal_prefix='b-')
        j = b.journal()
        assert list(j['session']) == [dj.session] * 2
        assert b.journal(loc=0).block.session == dj.session

    def test_distinct_from_the_tree(self, tmp_path):
        with Datajournal() as dj:
            block(tmp_path, tree='TREE').write_journal_entry(event='note')
        entry = block(tmp_path).journal(loc=0).block
        assert (entry.tree, entry.session) == ('TREE', dj.session)

    def test_survives_pickling(self):
        """The session crosses; the written paths are the copy's own to collect."""
        with Datajournal() as dj:
            dj._record_('/some/entry.parquet')
        again = pickle.loads(pickle.dumps(dj))
        assert again.session == dj.session and again.written_entries() == []

    def test_a_filter(self, tmp_path):
        with Datajournal() as mine:
            block(tmp_path, x=1).build()
        with Datajournal():
            block(tmp_path, x=2).build()
        assert len(block(tmp_path).journal(session=mine.session)) == 1


class TestWrittenEntries:

    def test_every_entry_path_once_in_write_order(self, tmp_path):
        with Datajournal() as dj:
            b = block(tmp_path)
            b.write_journal_entry(event='note', journal_prefix='a-')
            b.write_journal_entry(event='note', journal_prefix='b-')
            b.write_journal_entry(event='note', journal_prefix='a-')   # rewrites a-
        paths = dj.written_entries()
        assert [os.path.basename(p)[:2] for p in paths] == ['a-', 'b-']
        assert all(os.path.exists(p) for p in paths)

    def test_only_this_handles(self, tmp_path):
        dj = Datajournal()
        block(tmp_path, x=1).build()
        assert dj.written_entries() == []


class TestLegacySessionColumn:
    """``session`` named the TREE until it was renamed; now it is its own column.

    A journal spanning both has rows of each kind, and each is read by its own
    meaning: decided per row, by whether the row has a ``tree``.
    """

    def test_old_and_new_rows_in_one_journal(self, tmp_path):
        with Datajournal() as dj:
            b = block(tmp_path, tree='NEW-TREE')
            b.write_journal_entry(event='note')
        new_path = b.journal(loc=0)['entry_path']
        old = pd.read_parquet(new_path).drop(columns=['tree', 'session', 'id'])
        old['session'] = 'OLD-TREE'
        old['datetime'] = '2020-01-01T00-00-00.000000'
        old.to_parquet(new_path.replace('-journal-', '-journal-old-'))

        j = b.journal()
        new_row, old_row = j.iloc[0], j.iloc[1]
        assert (new_row['tree'], new_row['session']) == ('NEW-TREE', dj.session)
        assert old_row['tree'] == 'OLD-TREE' and pd.isna(old_row['session'])


class TestWithScope:
    """``with Datajournal()`` is the session of every entry written while it is open."""

    def test_entries_inside_go_under_it(self, tmp_path):
        with Datajournal() as dj:
            assert Datajournal.current() is dj
            b = block(tmp_path)
            b.build()
        assert Datajournal.current() is DEFAULT_DATAJOURNAL
        assert dj.written_entries() == [b.journal(loc=0)['entry_path']]

    def test_decided_when_it_writes_not_when_it_was_made(self, tmp_path):
        """Built after the inner ``with`` closed: the next one out, and nothing to the closed one."""
        with Datajournal() as outer:
            with Datajournal() as inner:
                b = block(tmp_path)
            b.build()
        assert inner.written_entries() == []
        assert outer.written_entries() == [b.journal(loc=0)['entry_path']]

    def test_with_none_left_the_default(self, tmp_path):
        with Datajournal() as dj:
            b = block(tmp_path)
        b.build()
        assert dj.written_entries() == []
        assert b.journal(loc=0).block.session == DEFAULT_DATAJOURNAL.session

    def test_the_innermost_wins_and_nesting_unwinds(self):
        with Datajournal() as outer:
            with Datajournal() as inner:
                assert Datajournal.current() is inner
            assert Datajournal.current() is outer
        assert Datajournal.current() is DEFAULT_DATAJOURNAL

    def test_threads_see_the_main_threads(self, tmp_path):
        """Why it is process-wide and not a ContextVar: a thread pool's callables build blocks too."""
        from concurrent.futures import ThreadPoolExecutor
        with Datajournal() as dj:
            with ThreadPoolExecutor(2) as ex:
                seen = list(ex.map(lambda i: Datajournal.current(), range(2)))
        assert all(j is dj for j in seen)

    def test_a_threads_own_is_its_own(self):
        """A ``with`` opened in a thread is that thread's: its siblings, and the main thread, do not see it."""
        import threading
        opened, release, seen = threading.Event(), threading.Event(), {}

        def worker():
            with Datajournal() as mine:
                seen['worker'], seen['mine'] = Datajournal.current(), mine
                opened.set()
                release.wait(10)
            seen['after'] = Datajournal.current()

        with Datajournal() as dj:
            t = threading.Thread(target=worker)
            t.start()
            opened.wait(10)
            seen['main'] = Datajournal.current()
            release.set()
            t.join(10)
        assert seen['worker'] is seen['mine'] and seen['main'] is dj
        assert seen['after'] is dj

    def test_dfn_yaml_carries_no_journal(self, tmp_path):
        from dbx.dataparts import read_yaml
        b = block(tmp_path)
        b.build()
        assert 'datajournal' not in read_yaml(b.journal(loc=0)['dfn'], safe=True)


class TestExec:
    """One `dbx.exec` is one session, and its exec-journal row lists what it wrote."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def _row(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        return read_exec_journal(datalake=str(tmp_path), iloc=0)

    def test_blocks_share_one_session_and_the_row_lists_their_entries(self, tmp_path):
        a, b = dbx.exec("a = Built(datalake=root, spec={'x': 1}); b = Built(datalake=root, spec={'x': 2}); "
                        "a.build(); b.build(); (a, b)", Built=Built, root=str(tmp_path))
        row = self._row(tmp_path)
        session = row['session']
        assert session != DEFAULT_DATAJOURNAL.session
        # The command is over, so its scope is closed and they write elsewhere now.
        assert Datajournal.current() is DEFAULT_DATAJOURNAL
        entries = [blk.journal(hash=blk.hash, loc=0) for blk in (a, b)]
        assert sorted(row['datajournal_entries']) == sorted(e['entry_path'] for e in entries)
        assert {e.block.session for e in entries} == {session}

    def test_a_block_constructed_deep_inside_is_covered(self, tmp_path):
        def pipeline():
            blk = Built(datalake=str(tmp_path), spec={'x': 5})
            blk.build()
            return None
        dbx.exec("pipeline()", pipeline=pipeline)
        assert len(self._row(tmp_path)['datajournal_entries']) == 1

    def test_a_failing_command_still_records_what_it_wrote(self, tmp_path):
        with pytest.raises(ZeroDivisionError):
            dbx.exec("Built(datalake=root, spec={'x': 1}).build(); 1/0", Built=Built, root=str(tmp_path))
        assert len(self._row(tmp_path)['datajournal_entries']) == 1

    def test_one_row_written_when_the_command_is_over(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        seen = []
        dbx.exec("seen.append(len(read(datalake=root)))", seen=seen, read=read_exec_journal, root=str(tmp_path))
        assert seen == [0]
        assert len(read_exec_journal(datalake=str(tmp_path))) == 1

    def test_a_failing_command_is_one_row_too(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        with pytest.raises(ZeroDivisionError):
            dbx.exec("1/0")
        assert len(read_exec_journal(datalake=str(tmp_path))) == 1

    def test_the_scope_closes_with_the_command(self, tmp_path):
        dbx.exec("1 + 1")
        assert Datajournal.current() is DEFAULT_DATAJOURNAL

    def test_a_nested_exec_joins_the_session(self, tmp_path):
        outer = dbx.exec("dbx.exec('Built(datalake=root, spec={\"x\": 3})', Built=Built, root=root)",
                         Built=Built, root=str(tmp_path), dbx=dbx)
        from dbx.dataparts import read_exec_journal
        rows = read_exec_journal(datalake=str(tmp_path))
        assert rows['session'].nunique() == 1 and len(rows) == 2


class TestExecjournal:
    """The exec journal's frame and entry, and the way back to what a command wrote."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def _run(self, tmp_path):
        return dbx.exec("a = Built(datalake=root, spec={'x': 1}); b = Built(datalake=root, spec={'x': 2}); "
                        "a.build(); b.build(); (a, b)  # two", Built=Built, root=str(tmp_path))

    def test_the_types(self, tmp_path):
        from dbx.dataparts import ExecjournalEntry, ExecjournalFrame
        self._run(tmp_path)
        assert isinstance(dbx.journal(), ExecjournalFrame)
        assert isinstance(dbx.journal(iloc=0), ExecjournalEntry)
        row_id = dbx.journal(iloc=0)['id']
        assert isinstance(dbx.journal().get(row_id), ExecjournalEntry)

    def test_an_entry_gives_back_what_it_wrote_in_order(self, tmp_path):
        a, b = self._run(tmp_path)
        entries = dbx.journal(iloc=0).dataentries()
        assert all(isinstance(e, DatajournalEntry) for e in entries)
        assert [e.block.hash for e in entries] == [a.hash, b.hash]
        assert [e.block.typestr() for e in entries] == [a.typestr(), b.typestr()]

    def test_a_filter_then_entries(self, tmp_path):
        self._run(tmp_path)
        dbx.exec("Built(datalake=root, spec={'x': 9}).build()  # other", Built=Built, root=str(tmp_path))
        row_id = dbx.journal(comment='two', iloc=0)['id']
        assert len(dbx.journal(id=row_id).dataentries()) == 2
        assert len(dbx.journal().dataentries()) == 3

    def test_a_command_that_wrote_nothing(self, tmp_path):
        dbx.exec("1 + 1")
        assert dbx.journal(iloc=0).dataentries() == []

    def test_rerun_executes_it_again_as_a_new_command(self, tmp_path):
        """A rebuild of a valid block writes nothing, so the command writes a note -- which always writes."""
        s = "b = Built(datalake=root, spec={'x': 4}); b.write_journal_entry(event='note'); b.hash"
        h = dbx.exec(s, Built=Built, root=str(tmp_path))
        first = dbx.journal(iloc=0)
        assert first.rerun(Built=Built, root=str(tmp_path)) == h
        rows = dbx.journal()
        assert len(rows) == 2 and rows['exec'].nunique() == 1
        assert rows.iloc[0]['session'] != first['session']
        assert len(dbx.journal(iloc=0).dataentries()) == 1


class TestRepr:
    """``repr()``: every kwarg, where ``quote()`` keeps only what reconstructs a block."""

    def test_it_carries_what_quote_leaves_out(self, tmp_path):
        b = block(tmp_path, tree='T-1')
        r, q = b.repr(), b.quote()
        for text in ("tree='T-1'", 'use_specializations=None'):
            assert text in r and text not in q

    def test_every_dfn_key_is_rendered(self, tmp_path):
        b = block(tmp_path)
        r = b.repr()
        # url and anchor as quote() renders them: only when given, so that a
        # block rooted by DBX_ROOT stays relocatable.
        assert all(f"{k}=" in r for k in b.dfn if not k.startswith('__') and k not in ('url', 'anchor'))

    def test_the_tree_is_the_one_the_block_has(self, tmp_path):
        """Generated on first access -- and repr() renders it, not the None before it."""
        b = block(tmp_path)
        assert f"tree={b.tree!r}" in b.repr()

    def test_evaluable_back_to_the_same_block(self, tmp_path):
        b = block(tmp_path, tag='t')
        again = dbx.eval(b.repr())
        assert (again.hash, again.tree, again.repr()) == (b.hash, b.tree, b.repr())

    def test_a_nested_block_renders_as_its_own_repr(self, tmp_path):
        inner = block(tmp_path, tree='INNER')

        class Outer(Datablock):
            TOPICS = {'o': 'o.txt'}

            @dataclass
            class VAR(Datablock.VAR):
                child: Datablock = None

        outer = Outer(datalake=str(tmp_path), spec={'child': inner})
        assert repr(inner.repr()) in outer.repr()
        assert repr(inner.quote()) in outer.quote()

    def test_recorded_in_the_journal_and_read_back(self, tmp_path):
        b = block(tmp_path)
        b.build()
        entry = b.journal(loc=0)
        assert entry.block.repr() == b.repr()
        assert entry.read('repr') == b.repr()
        d = entry.block.to_dict()
        assert (d['repr'], d['quote'], d['cite']) == (b.repr(), b.quote(), b.cite())


class TestExecTimes:

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def test_start_and_end_with_datetime_the_start(self, tmp_path):
        import time
        from dbx.dataparts import read_exec_journal
        dbx.exec("sleep(0.05)", sleep=time.sleep)
        row = read_exec_journal(datalake=str(tmp_path), iloc=0)
        assert row['datetime'] == row['exec:start:datetime']
        assert row['exec:end:datetime'] > row['exec:start:datetime']

    def test_a_failing_command_has_both(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        with pytest.raises(ZeroDivisionError):
            dbx.exec("1/0")
        row = read_exec_journal(datalake=str(tmp_path), iloc=0)
        assert row['exec:start:datetime'] and row['exec:end:datetime']


class TestJournalIndex:
    """`dbx.journal()` indexes by ``id`` by default; ``index=None`` numbers it."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def test_the_exec_journal_by_default(self, tmp_path):
        dbx.exec("1 + 1")
        frame = dbx.journal()
        row_id = frame.iloc[0]['id']
        assert list(frame.index) == [row_id]
        assert dbx.journal(loc=row_id)['exec'] == '1 + 1'
        assert dbx.journal(loc=row_id)['id'] == row_id, "the column stays a column"

    def test_a_block_journal_by_default(self, tmp_path):
        b = block(tmp_path)
        b.build()
        entry_id = b.journal(iloc=0)['id']
        assert dbx.journal(Built, datalake=str(tmp_path), loc=entry_id).block.id == entry_id

    def test_none_numbers_it(self, tmp_path):
        dbx.exec("1 + 1")
        assert list(dbx.journal(index=None).index) == [0]
        assert dbx.journal(index=None, loc=0)['exec'] == '1 + 1'

    def test_a_journal_with_no_id_column_is_left_numbered(self):
        frame = dbx.journal(pd.DataFrame({'hash': ['h'], 'anchor': ['a.B']}))
        assert list(frame.index) == [0]

    def test_an_explicit_column_must_exist(self):
        with pytest.raises(KeyError):
            dbx.journal(pd.DataFrame({'hash': ['h']}), index='id')


class TestExecjournalShape:

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def test_exec_first_comment_last(self, tmp_path):
        dbx.exec("1 + 1  # why")
        cols = list(dbx.journal().columns)
        assert cols[0] == 'exec' and cols[-1] == 'comment'
        assert 'datajournal_entries' in cols and 'written_entries' not in cols

    def test_a_row_under_the_old_column_name_still_reads(self, tmp_path):
        import os
        dbx.exec("Built(datalake=root, spec={'x': 1}).build()", Built=Built, root=str(tmp_path))
        exec_dir = os.path.join(str(tmp_path), '.journal', 'exec')
        [f] = os.listdir(exec_dir)
        old = pd.read_parquet(os.path.join(exec_dir, f)).rename(
            columns={'datajournal_entries': 'written_entries'})
        old.to_parquet(os.path.join(exec_dir, f))
        dbx.exec("2 + 2")
        rows = dbx.journal(index=None)
        assert 'written_entries' not in rows.columns
        assert [len(v) for v in rows['datajournal_entries']] == [0, 1]

    def test_datajournal_is_a_frame_of_what_it_wrote(self, tmp_path):
        a, b = dbx.exec("a = Built(datalake=root, spec={'x': 1}); b = Built(datalake=root, spec={'x': 2}); "
                        "a.build(); b.build(); (a, b)", Built=Built, root=str(tmp_path))
        entry = dbx.journal(iloc=0)
        frame = entry.datajournal()
        assert isinstance(frame, DatajournalFrame)
        assert list(frame['hash']) == [a.hash, b.hash]
        assert list(frame['entry_path']) == [e['entry_path'] for e in entry.dataentries()]
        assert list(dbx.journal().datajournal()['hash']) == [a.hash, b.hash]

    def test_a_command_that_wrote_nothing_has_an_empty_frame(self, tmp_path):
        dbx.exec("1 + 1")
        frame = dbx.journal(iloc=0).datajournal()
        assert isinstance(frame, DatajournalFrame) and len(frame) == 0

    def test_rerun_prints_the_shell_line_first(self, tmp_path, capsys):
        dbx.exec("x = \"$HOME\"; 1 + 1  # c", )
        dbx.journal(iloc=0).rerun()
        out = capsys.readouterr().out
        assert out.splitlines()[0] == 'dbx.pprint "x = \\"\\$HOME\\"; 1 + 1  # c"'


class TestFilterPatterns:
    """A filter value is a substring, a regex, or a glob -- any one matching is a match."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    @pytest.fixture
    def ids(self):
        return ['a6ff00', 'b0a6c1', 'c1d2e3']

    @pytest.mark.parametrize('pattern, expected', [
        ('^a6', ['a6ff00']),                  # regex, anchored
        ('a6', ['a6ff00', 'b0a6c1']),         # substring
        ('*a6*', ['a6ff00', 'b0a6c1']),       # glob, anywhere
        ('a6*', ['a6ff00']),                  # glob, at the start
        ('*e3', ['c1d2e3']),                  # glob, at the end
        ('^zz', []),
    ])
    def test_a_frame(self, ids, pattern, expected):
        frame = dbx.journal(pd.DataFrame({'id': ids, 'hash': ['h'] * 3}), id=pattern, index=None)
        assert sorted(frame['id']) == expected

    def test_the_block_journal(self, tmp_path):
        b = block(tmp_path)
        b.build()
        entry_id = b.journal(iloc=0)['id']
        assert len(b.journal(id=f'^{entry_id[:4]}')) == 1
        assert len(b.journal(id=f'*{entry_id[2:6]}*')) == 1
        assert len(b.journal(id='^nomatch')) == 0

    def test_the_exec_journal(self, tmp_path):
        dbx.exec("1 + 1")
        row_id = dbx.journal(iloc=0)['id']
        assert len(dbx.journal(id=f'^{row_id[:4]}')) == 1
        assert len(dbx.journal(id=f'*{row_id[3:7]}*')) == 1


class Other(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        fail: bool = False

    def __build__(self):
        if self.var.fail:
            raise RuntimeError("boom")
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


class TestConstructed:
    """What a command CONSTRUCTED: built, redirected, or copied in -- by anchor."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def _run(self, tmp_path):
        return dbx.exec(
            "a = Built(datalake=root, spec={'x': 1}); b = Built(datalake=root, spec={'x': 2}); o = Other(datalake=root); "
            "a.build(); b.build(); o.build(); "
            "n = Built(datalake=root, spec={'x': 3}); n.write_journal_entry(event='note'); (a, b, o)",
            Built=Built, Other=Other, root=str(tmp_path))

    def test_by_anchor(self, tmp_path):
        a, b, o = self._run(tmp_path)
        entry = dbx.journal(iloc=0)
        assert entry.anchors() == sorted({a.anchor, o.anchor})
        got = entry.constructed()
        assert set(got) == {a.anchor, o.anchor}
        assert sorted(got[a.anchor]['hash']) == sorted([a.hash, b.hash]), "the note is not a construction"
        assert list(got[o.anchor]['hash']) == [o.hash]
        one = entry.constructed(a.anchor)
        assert isinstance(one, DatajournalFrame) and len(one) == 2
        assert set(one['event']) == {'build:end'}

    def test_a_failed_build_was_not_constructed(self, tmp_path):
        with pytest.raises(RuntimeError):
            dbx.exec("Other(datalake=root, spec={'fail': True}).build()", Other=Other, root=str(tmp_path))
        entry = dbx.journal(iloc=0)
        assert len(entry.datajournal()) == 1 and entry.datajournal().iloc[0]['event'] == 'build:exception'
        assert entry.constructed() == {}

    def test_other_filters_add_to_the_event_default(self, tmp_path):
        a, b, o = self._run(tmp_path)
        entry = dbx.journal(iloc=0)
        assert list(entry.constructed(a.anchor, hash=f"^{b.hash}")['hash']) == [b.hash]
        # event=None drops the default: the note is back.
        assert len(entry.constructed(a.anchor, event=None)) == 3

    def test_the_events_are_matched_exactly(self):
        from dbx.journals import _constructed_
        frame = DatajournalFrame(pd.DataFrame({
            'anchor': ['s.S'] * 5,
            'hash': ['h1', 'h2', 'h3', 'h4', 'h5'],
            'event': ['UNSAFE_redirect', 'UNSAFE_redirect_blocks:end', 'build:end',
                      'build_tree:x:end', 'UNSAFE_copy_from:END'],
        }))
        assert list(_constructed_(frame, 's.S', {})['hash']) == ['h1', 'h3', 'h5']

    def test_over_a_frame_of_commands(self, tmp_path):
        a, b, o = self._run(tmp_path)
        dbx.exec("Built(datalake=root, spec={'x': 9}).build()", Built=Built, root=str(tmp_path))
        got = dbx.journal().constructed()
        assert len(got[a.anchor]) == 3 and len(got[o.anchor]) == 1


class TestTwoJournals:
    """dbx.datajournal() and dbx.execjournal() -- and dbx.journal(), which picks one."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def test_each_says_which_it_reads(self, tmp_path):
        from dbx.dataparts import ExecjournalFrame
        b = block(tmp_path)
        dbx.exec("b.build()", b=b)
        assert isinstance(dbx.datajournal(Built, datalake=str(tmp_path)), DatajournalFrame)
        assert isinstance(dbx.datajournal(b), DatajournalFrame)
        assert isinstance(dbx.execjournal(), ExecjournalFrame)
        assert dbx.execjournal(iloc=0)['exec'] == 'b.build()'
        assert dbx.datajournal(b, iloc=0).block.hash == b.hash

    def test_journal_picks_the_same_one(self, tmp_path):
        b = block(tmp_path)
        dbx.exec("b.build()", b=b)
        assert dbx.journal(iloc=0)['id'] == dbx.execjournal(iloc=0)['id']
        assert dbx.journal(b, iloc=0)['id'] == dbx.datajournal(b, iloc=0)['id']

    def test_datajournal_needs_to_be_told_what(self):
        with pytest.raises(TypeError, match="execjournal"):
            dbx.datajournal(None)


class Timed(Datablock):
    """Sleeps *delay* seconds, then builds -- or fails, with *fail*."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        delay: float = 0.0
        fail: bool = False

    def __build__(self):
        time.sleep(self.var.delay)
        if self.var.fail:
            raise RuntimeError("boom")
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


class FailingStack(Datastack):
    """Two `Timed` blocks: the first builds after 6s, the second fails after 3s.

    So the failure reaches the parent first, and the success only while it
    drains the workers -- the result that used to go down with the raise.
    """
    BLOCK = Timed
    TOPICS = {'meta': 'meta.txt'}

    @property
    def n_blocks(self):
        return 2

    def __block__(self, idx):
        return Timed(datalake=self.url, spec={'delay': 6.0 if idx == 0 else 3.0, 'fail': idx == 1})


def _built_under_the_scope(i, delay=0.0):
    """Top level, so a spawned worker can unpickle it by name."""
    time.sleep(delay)
    b = Built(datalake=os.environ['DBX_ROOT'], spec={'x': 100 + i})
    b.build()
    return b.hash


def _fails_after(delay):
    time.sleep(delay)
    raise RuntimeError("boom")


class TestAcrossProcesses:
    """A process executor carries the open scope to its workers and brings what they wrote back."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def test_a_stack_built_in_worker_processes(self, tmp_path):
        s = dbx.exec("s = Stack(datalake=root, parallelization='multiprocessing', n_workers=2); s.build(); s",
                     Stack=Stack, root=str(tmp_path))
        e = dbx.execjournal(iloc=0)
        frame = e.datajournal()
        assert sorted(frame['hash']) == sorted([s.hash] + [b.hash for b in s.blocks()])
        assert set(frame['session']) == {e['session']}

    def test_a_block_that_failed_in_a_worker(self, tmp_path):
        with pytest.raises(Exception):
            dbx.exec("FailingStack(datalake=root, parallelization='multiprocessing', n_workers=2).build()",
                     FailingStack=FailingStack, root=str(tmp_path))
        frame = dbx.execjournal(iloc=0).datajournal()
        events = frame[frame['anchor'] == Timed.anchor]['event']
        # The failed block's own entry, from its worker -- the stack records a
        # build:exception of its own, in this process -- and its sibling's,
        # which succeeded after the failure had stopped the collecting.
        assert sorted(events) == ['build:end', 'build:exception']

    def test_results_are_the_callables_values(self, tmp_path):
        from dbx.dataparts import callable_executor
        with Datajournal() as dj:
            hashes = callable_executor('multiprocessing', n_workers=2).exec_callables(
                [functools.partial(_built_under_the_scope, i) for i in range(2)])
        assert hashes == [block(tmp_path, x=100 + i).hash for i in range(2)]
        assert len(dj.written_entries()) == 2

    def test_with_no_scope_open_the_process_default_is_carried(self, tmp_path):
        """The default is the bottom of every stack: a worker writes under the dispatcher's, not its own."""
        from dbx.dataparts import callable_executor
        hashes = callable_executor('multiprocessing', n_workers=1).exec_callables(
            [functools.partial(_built_under_the_scope, 0)])
        entry = block(tmp_path, x=100).journal(iloc=0)
        assert hashes == [entry.block.hash]
        assert entry.block.session == DEFAULT_DATAJOURNAL.session
        assert entry['entry_path'] in DEFAULT_DATAJOURNAL.written_entries()

    def test_a_stream_keeps_what_arrived_after_a_failure(self, tmp_path):
        from dbx.dataparts import callable_executor
        with Datajournal() as dj:
            stream = callable_executor('multiprocessing', n_workers=2).exec_callables_streaming(
                [functools.partial(_fails_after, 3.0), functools.partial(_built_under_the_scope, 0, 6.0)])
            with pytest.raises(RuntimeError, match="boom"):
                list(stream)
        assert dj.written_entries() == [block(tmp_path, x=100).journal(iloc=0)['entry_path']]

    def test_threads_are_not_wrapped(self):
        from dbx.dataparts import callable_executor
        with Datajournal():
            assert callable_executor('multithreading', n_workers=2).exec_callables(
                [lambda: 1, lambda: 2]) == [1, 2]


_CHILD_BUILDS = """
import os
from dataclasses import dataclass
import dbx
from dbx.datablocks import Datablock

class Child(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('child')

%s
"""


def _run_child(body):
    """A process dbx did not start: a plain subprocess, inheriting the environment."""
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c', _CHILD_BUILDS % body], check=True,
                   env={**os.environ, 'DBX_USE_WORK_REPO': 'False', 'DBX_DIRTY_REPO_OK': 'True'})


def _dies_after_building():
    """Top level, for a spawned worker: builds, then exits without a word -- as a killed worker does."""
    _built_under_the_scope(7)
    os._exit(1)


class TestSessionIndex:
    """A command session indexes what is written under it, in storage, from whichever process wrote it."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def test_a_command_indexes_its_entries(self, tmp_path):
        b = dbx.exec("b = Built(datalake=root, spec={'x': 1}); b.build(); b", Built=Built, root=str(tmp_path))
        row = dbx.execjournal(iloc=0)
        entry_path = b.journal(loc=0)['entry_path']
        assert Datajournal.session_entries(row['session'], datalake=str(tmp_path)) == [entry_path]

    def test_nothing_is_indexed_outside_a_command(self, tmp_path):
        block(tmp_path).build()
        with Datajournal():
            block(tmp_path, x=2).build()
        assert not os.path.exists(os.path.join(str(tmp_path), '.journal', 'sessions'))

    def test_a_process_the_command_started_is_part_of_it(self, tmp_path):
        dbx.exec("run(\"Child(datalake=os.environ['DBX_ROOT']).build()\")", run=_run_child)
        row = dbx.execjournal(iloc=0)
        frame = row.datajournal()
        assert list(frame['anchor']) == ['__main__.Child']
        assert list(frame['session']) == [row['session']]

    def test_a_command_run_in_that_process_joins_the_session(self, tmp_path):
        dbx.exec("run(\"dbx.exec('Child(datalake=os.environ[\\\\'DBX_ROOT\\\\']).build()', Child=Child)\")",
                 run=_run_child)
        rows = dbx.execjournal()
        assert rows['session'].nunique() == 1 and len(rows) == 2
        outer = rows[rows['exec'].str.startswith('run(')].iloc[0]
        assert list(dbx.execjournal(id=outer['id'], iloc=0).datajournal()['anchor']) == ['__main__.Child']

    def test_a_worker_that_never_returned(self, tmp_path):
        from dbx.dataparts import callable_executor
        with pytest.raises(Exception):
            dbx.exec("ex.exec_callables([die])", ex=callable_executor('multiprocessing', n_workers=1),
                     die=_dies_after_building)
        row = dbx.execjournal(iloc=0)
        assert list(row['datajournal_entries']) == []           # it never came back
        frame = row.datajournal()                                # but its index did
        assert list(frame['hash']) == [block(tmp_path, x=107).hash]

    def test_the_environment_is_restored(self):
        from dbx.journals import JOURNAL_DATALAKE_ENV, JOURNAL_SESSION_ENV
        dbx.exec("1 + 1")
        assert JOURNAL_SESSION_ENV not in os.environ and JOURNAL_DATALAKE_ENV not in os.environ

    def test_a_row_from_before_the_index(self, tmp_path):
        import shutil
        dbx.exec("Built(datalake=root, spec={'x': 1}).build()", Built=Built, root=str(tmp_path))
        shutil.rmtree(os.path.join(str(tmp_path), '.journal', 'sessions'))
        assert len(dbx.execjournal(iloc=0).dataentries()) == 1
