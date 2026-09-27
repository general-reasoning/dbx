"""
`Datajournal`: the handle a block writes its journal through and reads it with.

Constructing one touches no storage; ``read()`` reads, ``write()`` writes. A
handle has a ``session`` fixed for its lifetime and written to every entry,
and remembers every entry path it wrote (``written_entries()``).
"""
import copy
import os
import pickle
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
    return Built(url=str(tmp_path), spec={'x': x}, **kwargs)


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
        return Built(url=self.url, spec={'x': idx})


class TestAHandleIsNotTheRecords:

    def test_construction_reads_nothing(self, tmp_path):
        """A root that does not exist is only an error once something reads it."""
        dj = Datajournal(str(tmp_path / 'nowhere'))
        with pytest.raises(FileNotFoundError):
            dj.read('some.Anchor')

    def test_read_returns_the_frame(self, tmp_path):
        b = block(tmp_path)
        b.build()
        frame = Datajournal(str(tmp_path)).read(b.anchor)
        assert isinstance(frame, DatajournalFrame) and len(frame) == 1
        assert isinstance(Datajournal(str(tmp_path)).read(b.anchor, loc=0), DatajournalEntry)

    def test_the_block_url_wins_over_the_handles(self, tmp_path):
        """A block reads and writes under its own url, whatever its handle's default is."""
        b = block(tmp_path / 'lake', datajournal=Datajournal(str(tmp_path / 'elsewhere')))
        b.build()
        assert len(b.journal()) == 1


class TestTheBlockWritesThroughIt:

    def test_default_is_the_process_journal(self, tmp_path):
        assert block(tmp_path).datajournal is DEFAULT_DATAJOURNAL

    def test_a_given_journal_is_used(self, tmp_path):
        dj = Datajournal()
        b = block(tmp_path, datajournal=dj)
        b.build()
        assert b.datajournal is dj
        assert dj.written_entries() == [b.journal(loc=0)['entry_path']]

    def test_it_is_handed_down_not_copied(self, tmp_path):
        """.set() deep-copies a block's state, which is how _adopt() hands it to a child."""
        dj = Datajournal()
        b = block(tmp_path, datajournal=dj)
        assert b.set(tag='t').datajournal is dj
        assert copy.deepcopy(dj) is dj

    def test_a_stack_hands_it_to_its_blocks(self, tmp_path):
        dj = Datajournal()
        stack = Stack(url=str(tmp_path), datajournal=dj)
        assert all(stack.block(i).datajournal is dj for i in range(stack.n_blocks))



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
        explicit = cls(url=url, spec=spec, datajournal=Datajournal('memory://elsewhere', n_workers=3))
        for b in (explicit, explicit.set(tag=None), pickle.loads(pickle.dumps(explicit))):
            assert _identity(b) == ref
        with Datajournal() as dj:
            scoped = cls(url=url, spec=spec)
            assert scoped.datajournal is dj and _identity(scoped) == ref

    def test_nor_that_of_the_blocks_it_is_handed_to(self, tmp_path):
        plain = Stack(url=str(tmp_path))
        given = Stack(url=str(tmp_path), datajournal=Datajournal())
        for i in range(plain.n_blocks):
            assert _identity(given.block(i)) == _identity(plain.block(i))

    def test_the_guard_distinct_blocks_stay_distinct(self, tmp_path):
        """So the parity above cannot pass by everything collapsing to one value."""
        assert block(tmp_path, x=1).hash != block(tmp_path, x=2).hash


class TestSession:

    def test_fixed_for_the_lifetime_and_distinct_between_handles(self):
        dj = Datajournal()
        assert dj.session == dj.session
        assert Datajournal().session != dj.session

    def test_written_to_every_entry(self, tmp_path):
        dj = Datajournal()
        b = block(tmp_path, datajournal=dj)
        b.write_journal_entry(event='note', journal_prefix='a-')
        b.write_journal_entry(event='note', journal_prefix='b-')
        j = b.journal()
        assert list(j['session']) == [dj.session] * 2
        assert b.journal(loc=0).block.session == dj.session

    def test_distinct_from_the_tree(self, tmp_path):
        dj = Datajournal()
        b = block(tmp_path, datajournal=dj, tree='TREE')
        b.write_journal_entry(event='note')
        entry = b.journal(loc=0).block
        assert (entry.tree, entry.session) == ('TREE', dj.session)

    def test_survives_pickling(self):
        dj = Datajournal(n_workers=3)
        again = pickle.loads(pickle.dumps(dj))
        assert again.session == dj.session and again.n_workers == 3

    def test_a_filter(self, tmp_path):
        mine, theirs = Datajournal(), Datajournal()
        block(tmp_path, x=1, datajournal=mine).build()
        block(tmp_path, x=2, datajournal=theirs).build()
        assert len(block(tmp_path).journal(session=mine.session)) == 1


class TestWrittenEntries:

    def test_every_entry_path_once_in_write_order(self, tmp_path):
        dj = Datajournal()
        b = block(tmp_path, datajournal=dj)
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
        dj = Datajournal()
        b = block(tmp_path, datajournal=dj, tree='NEW-TREE')
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
    """``with Datajournal()`` is the journal of every block that writes while it is open."""

    def test_blocks_inside_use_it(self, tmp_path):
        with Datajournal() as dj:
            b = block(tmp_path)
            assert b.datajournal is dj
        assert b.datajournal is DEFAULT_DATAJOURNAL

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

    def test_the_innermost_wins_and_nesting_unwinds(self, tmp_path):
        b = block(tmp_path)
        with Datajournal() as outer:
            with Datajournal() as inner:
                assert b.datajournal is inner
            assert b.datajournal is outer
        assert Datajournal.current() is None

    def test_an_explicit_one_wins(self, tmp_path):
        mine = Datajournal()
        with Datajournal():
            assert block(tmp_path, datajournal=mine).datajournal is mine

    def test_threads_see_it(self, tmp_path):
        """Why it is process-wide and not a ContextVar: a thread pool's callables construct blocks too."""
        from concurrent.futures import ThreadPoolExecutor
        with Datajournal() as dj:
            with ThreadPoolExecutor(2) as ex:
                seen = list(ex.map(lambda i: block(tmp_path, x=i).datajournal, range(2)))
        assert all(j is dj for j in seen)

    def test_dfn_yaml_stays_safe_to_load(self, tmp_path):
        from dbx.dataparts import read_yaml
        dj = Datajournal(n_workers=3)
        b = block(tmp_path, datajournal=dj)
        b.build()
        dfn = read_yaml(b.journal(loc=0)['dfn'], safe=True)
        assert dfn['datajournal'] == repr(dj)


class TestExec:
    """One `dbx.exec` is one session, and its exec-journal row lists what it wrote."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def _row(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        return read_exec_journal(url=str(tmp_path), iloc=0)

    def test_blocks_share_one_session_and_the_row_lists_their_entries(self, tmp_path):
        a, b = dbx.exec("a = Built(url=root, spec={'x': 1}); b = Built(url=root, spec={'x': 2}); "
                        "a.build(); b.build(); (a, b)", Built=Built, root=str(tmp_path))
        row = self._row(tmp_path)
        session = row['session']
        assert session != DEFAULT_DATAJOURNAL.session
        # The command is over, so its scope is closed and they write elsewhere now.
        assert a.datajournal is b.datajournal is DEFAULT_DATAJOURNAL
        entries = [blk.journal(hash=blk.hash, loc=0) for blk in (a, b)]
        assert sorted(row['datajournal_entries']) == sorted(e['entry_path'] for e in entries)
        assert {e.block.session for e in entries} == {session}

    def test_a_block_constructed_deep_inside_is_covered(self, tmp_path):
        def pipeline():
            blk = Built(url=str(tmp_path), spec={'x': 5})
            blk.build()
            return None
        dbx.exec("pipeline()", pipeline=pipeline)
        assert len(self._row(tmp_path)['datajournal_entries']) == 1

    def test_a_failing_command_still_records_what_it_wrote(self, tmp_path):
        with pytest.raises(ZeroDivisionError):
            dbx.exec("Built(url=root, spec={'x': 1}).build(); 1/0", Built=Built, root=str(tmp_path))
        assert len(self._row(tmp_path)['datajournal_entries']) == 1

    def test_one_row_written_when_the_command_is_over(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        seen = []
        dbx.exec("seen.append(len(read(url=root)))", seen=seen, read=read_exec_journal, root=str(tmp_path))
        assert seen == [0]
        assert len(read_exec_journal(url=str(tmp_path))) == 1

    def test_a_failing_command_is_one_row_too(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        with pytest.raises(ZeroDivisionError):
            dbx.exec("1/0")
        assert len(read_exec_journal(url=str(tmp_path))) == 1

    def test_the_scope_closes_with_the_command(self, tmp_path):
        dbx.exec("1 + 1")
        assert Datajournal.current() is None

    def test_a_nested_exec_joins_the_session(self, tmp_path):
        outer = dbx.exec("dbx.exec('Built(url=root, spec={\"x\": 3})', Built=Built, root=root)",
                         Built=Built, root=str(tmp_path), dbx=dbx)
        from dbx.dataparts import read_exec_journal
        rows = read_exec_journal(url=str(tmp_path))
        assert rows['session'].nunique() == 1 and len(rows) == 2


class TestExecjournal:
    """The exec journal's frame and entry, and the way back to what a command wrote."""

    @pytest.fixture(autouse=True)
    def lake(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DBX_ROOT', str(tmp_path))

    def _run(self, tmp_path):
        return dbx.exec("a = Built(url=root, spec={'x': 1}); b = Built(url=root, spec={'x': 2}); "
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
        entries = dbx.journal(iloc=0).entries()
        assert all(isinstance(e, DatajournalEntry) for e in entries)
        assert [e.block.hash for e in entries] == [a.hash, b.hash]
        assert [e.block.typestr() for e in entries] == [a.typestr(), b.typestr()]

    def test_a_filter_then_entries(self, tmp_path):
        self._run(tmp_path)
        dbx.exec("Built(url=root, spec={'x': 9}).build()  # other", Built=Built, root=str(tmp_path))
        row_id = dbx.journal(comment='two', iloc=0)['id']
        assert len(dbx.journal(id=row_id).entries()) == 2
        assert len(dbx.journal().entries()) == 3

    def test_a_command_that_wrote_nothing(self, tmp_path):
        dbx.exec("1 + 1")
        assert dbx.journal(iloc=0).entries() == []

    def test_rerun_executes_it_again_as_a_new_command(self, tmp_path):
        """A rebuild of a valid block writes nothing, so the command writes a note -- which always writes."""
        s = "b = Built(url=root, spec={'x': 4}); b.write_journal_entry(event='note'); b.hash"
        h = dbx.exec(s, Built=Built, root=str(tmp_path))
        first = dbx.journal(iloc=0)
        assert first.rerun(Built=Built, root=str(tmp_path)) == h
        rows = dbx.journal()
        assert len(rows) == 2 and rows['exec'].nunique() == 1
        assert rows.iloc[0]['session'] != first['session']
        assert len(dbx.journal(iloc=0).entries()) == 1


class TestRepr:
    """``repr()``: every kwarg, where ``quote()`` keeps only what reconstructs a block."""

    def test_it_carries_what_quote_leaves_out(self, tmp_path):
        b = block(tmp_path, tree='T-1', datajournal=Datajournal(n_workers=3))
        r, q = b.repr(), b.quote()
        for text in ("tree='T-1'", "datajournal=dbx.Datajournal(n_workers=3)", 'use_specializations=None'):
            assert text in r and text not in q

    def test_every_dfn_key_is_rendered(self, tmp_path):
        b = block(tmp_path, datajournal=Datajournal())
        r = b.repr()
        # url and anchor as quote() renders them: only when given, so that a
        # block rooted by DBX_ROOT stays relocatable.
        assert all(f"{k}=" in r for k in b.dfn if not k.startswith('__') and k not in ('url', 'anchor'))

    def test_the_tree_is_the_one_the_block_has(self, tmp_path):
        """Generated on first access -- and repr() renders it, not the None before it."""
        b = block(tmp_path)
        assert f"tree={b.tree!r}" in b.repr()

    def test_evaluable_back_to_the_same_block(self, tmp_path):
        b = block(tmp_path, tag='t', datajournal=Datajournal(n_workers=3))
        again = dbx.eval(b.repr())
        assert (again.hash, again.tree, again.repr()) == (b.hash, b.tree, b.repr())
        assert isinstance(again.datajournal, Datajournal) and again.datajournal.n_workers == 3

    def test_a_nested_block_renders_as_its_own_repr(self, tmp_path):
        inner = block(tmp_path, tree='INNER')

        class Outer(Datablock):
            TOPICS = {'o': 'o.txt'}

            @dataclass
            class VAR(Datablock.VAR):
                child: Datablock = None

        outer = Outer(url=str(tmp_path), spec={'child': inner})
        assert repr(inner.repr()) in outer.repr()
        assert repr(inner.quote()) in outer.quote()

    def test_recorded_in_the_journal_and_read_back(self, tmp_path):
        b = block(tmp_path, datajournal=Datajournal())
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
        row = read_exec_journal(url=str(tmp_path), iloc=0)
        assert row['datetime'] == row['exec:start:datetime']
        assert row['exec:end:datetime'] > row['exec:start:datetime']

    def test_a_failing_command_has_both(self, tmp_path):
        from dbx.dataparts import read_exec_journal
        with pytest.raises(ZeroDivisionError):
            dbx.exec("1/0")
        row = read_exec_journal(url=str(tmp_path), iloc=0)
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
        assert dbx.journal(Built, url=str(tmp_path), loc=entry_id).block.id == entry_id

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
        dbx.exec("Built(url=root, spec={'x': 1}).build()", Built=Built, root=str(tmp_path))
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
        a, b = dbx.exec("a = Built(url=root, spec={'x': 1}); b = Built(url=root, spec={'x': 2}); "
                        "a.build(); b.build(); (a, b)", Built=Built, root=str(tmp_path))
        entry = dbx.journal(iloc=0)
        frame = entry.datajournal()
        assert isinstance(frame, DatajournalFrame)
        assert list(frame['hash']) == [a.hash, b.hash]
        assert list(frame['entry_path']) == [e['entry_path'] for e in entry.entries()]
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
            "a = Built(url=root, spec={'x': 1}); b = Built(url=root, spec={'x': 2}); o = Other(url=root); "
            "a.build(); b.build(); o.build(); "
            "n = Built(url=root, spec={'x': 3}); n.write_journal_entry(event='note'); (a, b, o)",
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
            dbx.exec("Other(url=root, spec={'fail': True}).build()", Other=Other, root=str(tmp_path))
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
        from dbx.dataparts import _constructed_
        frame = DatajournalFrame(pd.DataFrame({
            'anchor': ['s.S'] * 5,
            'hash': ['h1', 'h2', 'h3', 'h4', 'h5'],
            'event': ['UNSAFE_redirect', 'UNSAFE_redirect_blocks:end', 'build:end',
                      'build_tree:x:end', 'UNSAFE_copy_from:END'],
        }))
        assert list(_constructed_(frame, 's.S', {})['hash']) == ['h1', 'h3', 'h5']

    def test_over_a_frame_of_commands(self, tmp_path):
        a, b, o = self._run(tmp_path)
        dbx.exec("Built(url=root, spec={'x': 9}).build()", Built=Built, root=str(tmp_path))
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
        assert isinstance(dbx.datajournal(Built, url=str(tmp_path)), DatajournalFrame)
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
