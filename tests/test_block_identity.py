"""
A stack's type names its BLOCK -- a table's, its TAB -- so the class is identity.

It did not: an identity names no class, and a table spelled with the topic
markers carries none of its TAB's topics, so pointing a stack at another block
class left its hash, and its `done`, exactly where they were.

The entry is the class's fqcn: ``BLOCK=<fqcn>`` -- ``TAB=<fqcn>`` for a table --
after the spec, before the version and the topics. ``with_block=False`` computes
a type under the rules from before, as reconstructing a stack built then needs;
a Specialization says the same with ``BLOCK=None`` -- a table's with ``TAB=None``.
"""
from dataclasses import dataclass

import pytest

from dbx.datablocks import DATAFILE, SAME, Datablock, Datastack
from dbx.datatables import DATASLICE, Datatab, Datatable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Leaf(Datablock):
    TOPICS = {'x': DATAFILE('x.txt')}


class OtherLeaf(Datablock):
    TOPICS = {'x': DATAFILE('x.txt')}


class Stack(Datastack):
    BLOCK = Leaf


class OtherStack(Datastack):
    BLOCK = OtherLeaf


class TabA(Datatab):
    TOPICS = {'a': DATASLICE(i='int')}


class TabB(Datatab):
    TOPICS = {'a': DATASLICE(i='int')}


class TableA(Datatable):
    TAB = TabA


class TableB(Datatable):
    TAB = TabB


def _made(cls, tmp_path):
    return cls(datalake=str(tmp_path))


class TestAStackNamesItsBlock:

    def test_its_type_says_which(self, tmp_path):
        stack = _made(Stack, tmp_path)
        assert f"/BLOCK={Leaf.fqcn}/version=" in stack.typestr()
        assert stack.type()['BLOCK'] == Leaf.fqcn

    def test_another_block_class_is_another_stack(self, tmp_path):
        assert _made(Stack, tmp_path).hash != _made(OtherStack, tmp_path).hash

    def test_the_old_rules_did_not_tell_them_apart(self, tmp_path):
        a, b = _made(Stack, tmp_path), _made(OtherStack, tmp_path)
        assert 'BLOCK=' not in a.typestr(with_block=False)
        assert a.get_hash(with_block=False) == b.get_hash(with_block=False)
        assert 'BLOCK' not in a.type(with_block=False)

    def test_a_stack_declaring_no_block_names_none(self, tmp_path):
        class Bare(Datastack):
            pass
        with pytest.warns(FutureWarning):
            bare = _made(Bare, tmp_path)
        assert 'BLOCK=' not in bare.typestr()


class TestATableNamesItsTab:

    def test_as_TAB_and_not_BLOCK(self, tmp_path):
        tp = _made(TableA, tmp_path).typestr()
        assert f"/TAB={TabA.fqcn}/version=" in tp and 'BLOCK=' not in tp

    def test_another_TAB_is_another_table(self, tmp_path):
        assert _made(TableA, tmp_path).hash != _made(TableB, tmp_path).hash

    def test_editing_the_TAB_in_place_does_not_rekey_the_table(self, tmp_path, monkeypatch):
        """The limit of naming the fqcn: the class is named, not its contents.

        See the KNOWN GAP in Datatable's docstring.
        """
        before = _made(TableA, tmp_path).hash
        monkeypatch.setattr(TabA, 'TOPICS', {'a': DATASLICE(i='int', j='int')})
        assert _made(TableA, tmp_path).hash == before


class TestASpecializationSaysWhichRules:

    def test_BLOCK_None_reconstructs_the_old_rules(self, tmp_path):
        stack = _made(Stack, tmp_path)
        sp = Datastack.Specialization(spec={}, topics={}, BLOCK=None)
        assert 'BLOCK=' not in stack.get_typestr(sp)

    def test_BLOCK_SAME_is_this_stacks_own_and_a_string_another(self, tmp_path):
        stack = _made(Stack, tmp_path)
        own = Datastack.Specialization(spec={}, topics={})
        assert own.BLOCK is SAME and f"BLOCK={Leaf.fqcn}" in stack.get_typestr(own)
        other = Datastack.Specialization(spec={}, topics={}, BLOCK='pkg.OldLeaf')
        assert "BLOCK=pkg.OldLeaf" in stack.get_typestr(other)

    def test_it_round_trips(self):
        sp = Datastack.Specialization(spec={}, topics={'x': 'x.txt'}, BLOCK=None)
        again = Datastack.Specialization.from_record(sp.to_dict())
        assert again.BLOCK is None and again.key == sp.key
        assert Datastack.Specialization(spec={}, topics={'x': 'x.txt'}).key != sp.key

    def test_anything_else_is_refused(self):
        with pytest.raises(TypeError, match="BLOCK= is SAME, None or a block fqcn"):
            Datastack.Specialization(spec={}, topics={}, BLOCK=Leaf)

    def test_a_block_specialization_has_no_BLOCK_and_a_tables_calls_it_TAB(self, tmp_path):
        assert not hasattr(Datablock.Specialization(spec={}, topics={}), 'BLOCK')
        sp = Datatable.Specialization(spec={}, topics={}, TAB=None)
        assert not hasattr(sp, 'BLOCK')
        assert 'TAB=' not in _made(TableA, tmp_path).get_typestr(sp)
        assert f"TAB={TabA.fqcn}" in _made(TableA, tmp_path).get_typestr(
            Datatable.Specialization(spec={}, topics={}))
        with pytest.raises(TypeError, match="TAB= is SAME, None or a TAB fqcn"):
            Datatable.Specialization(spec={}, topics={}, TAB=TabA)

    def test_a_plain_block_specialization_names_the_stacks_own(self, tmp_path):
        """SAME, as Datastack.Specialization's default: nothing to say about the block class."""
        stack = _made(Stack, tmp_path)
        assert f"BLOCK={Leaf.fqcn}" in stack.get_typestr(Datablock.Specialization(spec={}, topics={}))
