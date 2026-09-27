"""
A Specialization may look for the narrower block under another anchor.

A block's journal is under its anchor, and a class's anchor defaults to its
fqcn -- so renaming a class leaves everything it built in a journal the renamed
class never reads. ``anchor=`` points a specialization at that journal. The
identity is unaffected: it names no class, so the old block's hash is
reconstructed exactly as it would be under the old name.
"""
from dataclasses import dataclass

import pytest

from dbx.datablocks import DIRTOPIC, SAME, Datablock, Datajournal
from dbx.datatables import Datatab, Datatable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Before(Datatab):
    """The class as it was named when it was built."""
    TOPICS = {'rows': 'rows.txt'}

    @dataclass
    class VAR(Datatab.VAR):
        tab_idx: int = 0

    def __build__(self):
        with open(self.path('rows', ensure_dirpath=True), 'w') as f:
            f.write(f"rows-{self.var.tab_idx}")


class After(Datatab):
    """The same block renamed: same spec, topics and version, so the same hash."""
    TOPICS = {'rows': 'rows.txt'}
    SPECIALIZATIONS = [Datablock.Specialization(
        spec={}, topics={'rows': 'rows.txt'}, anchor=Before.anchor,
        note="renamed only")]

    @dataclass
    class VAR(Datatab.VAR):
        tab_idx: int = 0

    def __build__(self):
        raise AssertionError("a renamed block should be read, not rebuilt")


class BeforeTable(Datatable):
    TAB = Before
    TOPICS = {'tab_paths': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(Datatable.VAR):
        n: int = 4

    @property
    def n_tabs(self):
        return self.var.n

    def __tab__(self, idx, **spec):
        return super().__tab__(idx, tab_idx=idx, **spec)


class AfterTable(BeforeTable):
    TAB = After


def _count_reads(monkeypatch):
    reads = []
    original = Datajournal.read

    def counting(self, anchor, *args, **kwargs):
        reads.append(anchor)
        return original(self, anchor, *args, **kwargs)

    monkeypatch.setattr(Datajournal, 'read', counting)
    return reads


def test_same_is_the_default_and_round_trips():
    sp = Datablock.Specialization(spec={}, topics={'rows': 'rows.txt'})
    assert sp.anchor is SAME
    assert 'anchor' not in sp.to_dict()
    named = Datablock.Specialization(spec={}, topics={'rows': 'rows.txt'}, anchor='pkg.Old')
    again = Datablock.Specialization(**named.to_dict())
    assert again.anchor == 'pkg.Old' and again.key == named.key != sp.key


def test_an_anchor_must_be_one():
    with pytest.raises(TypeError, match="SAME or an anchor string"):
        Datablock.Specialization(spec={}, topics={'rows': 'rows.txt'}, anchor=3)


def test_a_renamed_block_reads_what_its_old_name_built(tmp_path):
    Before(datalake=str(tmp_path), spec={'tab_idx': 2}).build()
    after = After(datalake=str(tmp_path), spec={'tab_idx': 2})
    assert after.anchor != Before.anchor
    assert after.hash == Before(datalake=str(tmp_path), spec={'tab_idx': 2}).hash
    assert after.redirected_topics() == ['rows']
    assert after.valid()
    with open(after.path('rows')) as f:
        assert f.read() == 'rows-2'


def test_under_its_own_anchor_the_same_identity_is_still_refused(tmp_path):
    same = Datablock.Specialization(spec={}, topics={'rows': 'rows.txt'})
    assert "own identity" in Before(datalake=str(tmp_path))._specialization_mismatch_(same)


def test_a_table_reads_each_anchor_once_for_all_its_tabs(tmp_path, monkeypatch):
    BeforeTable(datalake=str(tmp_path), spec={'n': 4}).build()
    reads = _count_reads(monkeypatch)
    table = AfterTable(datalake=str(tmp_path), spec={'n': 4})
    tabs = [table.tab(i) for i in range(4)]
    assert all(t.redirected_topics() == ['rows'] for t in tabs)
    for i, t in enumerate(tabs):
        with open(t.path('rows')) as f:
            assert f.read() == f'rows-{i}'
    old_reads = [a for a in reads if a == Before.anchor]
    assert len(old_reads) <= 1, f"{len(old_reads)} reads of the old anchor's journal for 4 tabs"
