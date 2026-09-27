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


# ---------------------------------------------------------------------------
# SPECIALIZATIONS= on the instance, in place of the class's
# ---------------------------------------------------------------------------

import copy  # noqa: E402
import pickle  # noqa: E402


def _renamed_by_instance(url, **kw):
    """`Before` built; a class that declares nothing, given the rename per instance."""
    return Unspecialized(datalake=url, spec={'tab_idx': 2}, SPECIALIZATIONS=[
        Datablock.Specialization(spec={}, topics={'rows': 'rows.txt'}, anchor=Before.anchor)], **kw)


class Unspecialized(Datatab):
    TOPICS = {'rows': 'rows.txt'}

    @dataclass
    class VAR(Datatab.VAR):
        tab_idx: int = 0


def test_an_instance_declares_its_own(tmp_path):
    Before(datalake=str(tmp_path), spec={'tab_idx': 2}).build()
    assert Unspecialized.SPECIALIZATIONS in (None, [])
    block = _renamed_by_instance(str(tmp_path))
    assert block.redirected_topics() == ['rows']
    assert block.valid()


def test_none_keeps_the_class_and_empty_declares_none(tmp_path):
    Before(datalake=str(tmp_path), spec={'tab_idx': 2}).build()
    # First: once one construction installs the redirection, its .redirection
    # marker answers every later one, whatever they declare.
    off = After(datalake=str(tmp_path), spec={'tab_idx': 2}, SPECIALIZATIONS=[])
    assert off.SPECIALIZATIONS == [] and off.redirected_topics() == []
    assert After(datalake=str(tmp_path), spec={'tab_idx': 2}).redirected_topics() == ['rows']


def test_they_are_not_the_identity(tmp_path):
    plain = Unspecialized(datalake=str(tmp_path), spec={'tab_idx': 2})
    assert _renamed_by_instance(str(tmp_path)).hash == plain.hash


def test_they_travel_with_the_block(tmp_path):
    block = _renamed_by_instance(str(tmp_path))
    for twin in (pickle.loads(pickle.dumps(block)), copy.deepcopy(block), block.set(verbose=True)):
        assert [sp.key for sp in twin.SPECIALIZATIONS] == [sp.key for sp in block.SPECIALIZATIONS]


def test_a_quote_carries_them_and_evaluates_back(tmp_path):
    import dbx
    block = _renamed_by_instance(str(tmp_path))
    q = block.quote()
    assert 'SPECIALIZATIONS=' in q
    again = dbx.eval(q)
    assert [sp.key for sp in again.SPECIALIZATIONS] == [sp.key for sp in block.SPECIALIZATIONS]
    assert 'SPECIALIZATIONS' not in Unspecialized(datalake=str(tmp_path)).quote()


def test_anything_but_specializations_is_refused(tmp_path):
    with pytest.raises(TypeError, match="not a Specialization"):
        Unspecialized(datalake=str(tmp_path), SPECIALIZATIONS=['rows'])
