"""
A stack reads its blocks' journal ONCE per build, not once per block.

A block whose TAB declares SPECIALIZATIONS resolves them as it is formed, and
resolving reads the journal. Each block-forming callable used to read it for
itself -- `__block__` formed the block without the table's shared journal, and
`_adopt_`'s `.set()` formed it again without it -- which on a real lake was a
3700-file read per tab, repeated for every one of hundreds of tabs.

The parent reads it once, hands it to the executor as a ctx kwarg -- one copy
per worker, not per callable -- and each callable forms its block with it.
"""
from dataclasses import dataclass

import pytest

from dbx.datablocks import DIRTOPIC, Datablock, Datajournal
from dbx.datatables import Datatab, Datatable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Tab(Datatab):
    TOPICS = {'rows': 'rows.txt', 'extra': 'extra.txt'}
    # Narrower than this tab, and never built: every tab tries it and fails to
    # resolve -- the case in which nothing is installed and nothing carries over.
    SPECIALIZATIONS = [Datablock.Specialization(spec={}, topics={'rows': 'rows.txt'})]

    @dataclass
    class VAR(Datatab.VAR):
        tab_idx: int = 0

    def __build__(self):
        for topic in ('rows', 'extra'):
            with open(self.path(topic, ensure_dirpath=True), 'w') as f:
                f.write(f"{topic}-{self.var.tab_idx}")


class Table(Datatable):
    TAB = Tab
    TOPICS = {'tab_paths': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datatable.VAR):
        n: int = 6

    @property
    def n_tabs(self):
        return self.var.n

    def __tab__(self, idx, **spec):
        return super().__tab__(idx, tab_idx=idx, **spec)


def _count_full_reads(monkeypatch):
    reads = []
    original = Datajournal.read

    def counting(anchor, *args, **kwargs):
        reads.append(anchor)
        return original(anchor, *args, **kwargs)

    monkeypatch.setattr(Datajournal, 'read', staticmethod(counting))
    return reads


@pytest.mark.parametrize('parallelization', ['inline', 'multithreading'])
def test_a_table_build_reads_its_tabs_journal_once(tmp_path, monkeypatch, parallelization):
    # Something journalled under the tabs' anchor already, so there is a journal to read.
    Table(datalake=str(tmp_path), spec={'n': 1}).build()
    reads = _count_full_reads(monkeypatch)
    Table(datalake=str(tmp_path), spec={'n': 6}, parallelization=parallelization,
          n_workers=2).build()
    tab_reads = [a for a in reads if a == Tab.anchor]
    assert len(tab_reads) <= 1, f"{len(tab_reads)} reads of the tabs' journal for 6 tabs"


# ---------------------------------------------------------------------------
# A redirection a specialization installed answers from its record, not the journal
# ---------------------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test_specializations import TestOneJournalReadForAWholeTable, adopted, v1table  # noqa: E402


@pytest.fixture
def grown(tmp_path, monkeypatch):
    v1table(tmp_path, spec={'n': 3}).build()
    TestOneJournalReadForAWholeTable._grow_the_tab(monkeypatch)
    return tmp_path


def test_the_record_carries_the_paths(grown):
    tab = adopted(v1table(grown, spec={'n': 3}).tab(0))  # installs, and records
    entry = tab.datajournal(event='^UNSAFE_redirect$', hash=tab.hash, iloc=0, index=None)
    assert 'paths' in entry.block.redirection


def test_where_a_specialized_tab_reads_from_costs_no_journal_read(grown, monkeypatch):
    adopted(v1table(grown, spec={'n': 3}).tab(0))         # installed, recorded
    reads = _count_full_reads(monkeypatch)
    fresh = v1table(grown, spec={'n': 3}).tab(0)          # forming it reads nothing at all
    assert fresh.get_redirection().specialization is not None
    fresh.valid()           # False -- the grown `extra` is unbuilt -- and asked cheaply
    assert fresh.redirected_topics() == ['rows']
    assert reads == [], f"{len(reads)} full journal read(s)"


def test_an_older_record_without_paths_answers_from_the_marker(grown, monkeypatch):
    """Records written before the paths were: the .redirection topic holds them."""
    from dbx.datablocks import Datablock
    tab = adopted(v1table(grown, spec={'n': 3}).tab(0))
    original = Datablock._recorded_redirection_

    def without_paths(self, journal=None):
        rec = original(self, journal=journal)
        return {k: v for k, v in rec.items() if k != 'paths'} if isinstance(rec, dict) else rec

    monkeypatch.setattr(Datablock, '_recorded_redirection_', without_paths)
    reads = _count_full_reads(monkeypatch)
    fresh = v1table(grown, spec={'n': 3}).tab(0)          # forming it reads nothing
    red = fresh.get_redirection()
    assert red is not None and red.paths == tab.get_redirection().paths
    assert reads == []


def test_a_redirected_tab_formed_again_records_nothing_more(grown):
    table = v1table(grown, spec={'n': 3})
    tab = adopted(table.tab(0))
    before = len(tab.datajournal(event='^UNSAFE_redirect$', hash=tab.hash, index=None))
    for _ in range(3):
        adopted(v1table(grown, spec={'n': 3}).tab(0))   # finds the marker: installs nothing
    after = len(tab.datajournal(event='^UNSAFE_redirect$', hash=tab.hash, index=None))
    assert after == before == 1


@pytest.mark.parametrize('parallelization', ['inline', 'multithreading'])
def test_valid_tabs_reads_the_journal_once(tmp_path, monkeypatch, parallelization):
    Table(datalake=str(tmp_path), spec={'n': 6}).build()
    reads = _count_full_reads(monkeypatch)
    table = Table(datalake=str(tmp_path), spec={'n': 6}, parallelization=parallelization, n_workers=2)
    assert table.valid_tabs().all()
    assert len([a for a in reads if a == Tab.anchor]) <= 1


def test_a_part_filters_its_built_tabs_in_the_parent_by_default(tmp_path):
    from dbx.datatables import DatatablePartition
    table = Table(datalake=str(tmp_path / 't'), spec={'n': 4})
    part = DatatablePartition(datalake=str(tmp_path / 'p'), spec=dict(
        datapoint_table=table, fractions=[0.5, 0.5], partition_slice=0)).fold(0)
    assert part.filter_built_tabs is True


# ---------------------------------------------------------------------------
# A table caches its tabs' journal on the instance, never in its state
# ---------------------------------------------------------------------------

import pickle  # noqa: E402


class PlainTab(Datatab):
    TOPICS = {'rows': 'rows.txt'}

    @dataclass
    class VAR(Datatab.VAR):
        tab_idx: int = 0

    def __build__(self):
        with open(self.path('rows', ensure_dirpath=True), 'w') as f:
            f.write('x')


class PlainTable(Table):
    TAB = PlainTab


def test_a_table_carries_no_journal_in_its_state(tmp_path):
    """A cached journal lives on the instance, never in __getstate__: a copy starts clean."""
    import copy
    Table(datalake=str(tmp_path), spec={'n': 2}).build()
    table = Table(datalake=str(tmp_path), spec={'n': 2})
    table.child_specialization_datajournal()               # read, and cached on the instance
    assert '__child_journal__' in table.__dict__
    state = table.__getstate__()
    assert not any('journal' in k for k in state if k.startswith('__'))
    for twin in (pickle.loads(pickle.dumps(table)), copy.deepcopy(table)):
        assert '__child_journal__' not in twin.__dict__


def test_a_copy_reads_the_journal_once_when_it_needs_it(tmp_path, monkeypatch):
    """A worker's copy reads nothing to form its tabs, and a build of it reads once."""
    Table(datalake=str(tmp_path), spec={'n': 4}).build()
    table = Table(datalake=str(tmp_path), spec={'n': 4})
    blob = pickle.dumps(table)
    reads = _count_full_reads(monkeypatch)
    twin = pickle.loads(blob)
    assert reads == [], "unpickling alone reads nothing"
    for i in range(4):
        twin.tab(i)
    assert reads == [], "forming tabs resolves nothing, so reads nothing"
    assert twin.child_specialization_datajournal() is not None
    twin.child_specialization_datajournal()
    assert len([a for a in reads if a == Tab.anchor]) == 1


def test_nothing_is_read_for_a_table_whose_tabs_declare_no_specializations(tmp_path, monkeypatch):
    PlainTable(datalake=str(tmp_path), spec={'n': 3}).build()
    reads = _count_full_reads(monkeypatch)
    table = PlainTable(datalake=str(tmp_path), spec={'n': 3})
    pickle.loads(pickle.dumps(table)).tab(0)
    table.get_tab_redirections(parallelization='inline')
    assert table.find_tab_specializations().tolist() == [None, None, None]
    assert reads == []


def test_the_journal_is_not_in_the_tables_definition(tmp_path):
    Table(datalake=str(tmp_path), spec={'n': 2}).build()
    table = Table(datalake=str(tmp_path), spec={'n': 2})
    table.tab(0)
    assert '__child_journal__' not in table.dfn and 'child_journal' not in table.quote()


def test_already_built_table_shortcircuits_build_and_build_tree(tmp_path, monkeypatch):
    Table(datalake=str(tmp_path), spec={'n': 4}).build()
    reads = _count_full_reads(monkeypatch)
    table = Table(datalake=str(tmp_path), spec={'n': 4})
    assert table.valid()
    table.build()
    assert reads == [], "build() on an already-valid table must not read tab journals"
    table.build_tree()
    assert reads == [], "build_tree() on an already-valid table must not read tab journals"


def test_build_tree_deep_asks_every_tab(tmp_path, monkeypatch):
    """deep=True takes no short cut -- neither the table's validity nor its manifest's: every tab is asked.

    And reads no journal for it: every tab being valid, none has a specialization to resolve.
    """
    Table(datalake=str(tmp_path), spec={'n': 4}).build()
    reads = _count_full_reads(monkeypatch)
    asked = []
    original = Tab.valid
    monkeypatch.setattr(Tab, 'valid', lambda self: asked.append(self.var.tab_idx) or original(self))

    Table(datalake=str(tmp_path), spec={'n': 4}).build_tree()
    assert asked == [], "the manifest vouches for the tabs of a built table"

    Table(datalake=str(tmp_path), spec={'n': 4}).build_tree(deep=True)
    assert sorted(set(asked)) == [0, 1, 2, 3], "build_tree(deep=True) must ask every tab"
    assert [a for a in reads if a == Tab.anchor] == []
