"""`Datablock.pretty`: a live block, read as its journal entry reads it.

`blk.pretty` is a `Pretty`, as `entry.pretty` is, over what an entry written
for the block now would record. So `blk.pretty.quote()` is the `quote()` of
the entry its build writes, and so for the rest. `quote()` and `cite()` both
take `specline=False`, which drops the leading specline anchor.
"""
from dataclasses import dataclass

import pytest

from dbx import DATAFILE
from dbx.datablocks import BuildJournal, Datablock, Pretty


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Leaf(Datablock):
    TOPICS = {'output': DATAFILE('output.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = 'leaf'
        size: int = 3

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('leaf')


class Top(Datablock):
    TOPICS = {'output': DATAFILE('output.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        leaf: object = None
        epochs: int = 10

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('top')


@pytest.fixture
def top(tmp_path):
    leaf = Leaf(datalake=str(tmp_path), spec=dict(label='leaf', size=3)).set(tag='leaf')
    return Top(datalake=str(tmp_path), spec=dict(leaf=leaf, epochs=10),
               local=str(tmp_path / 'local')).set(tag='top')


@pytest.fixture
def entry(top):
    """The last entry *top*'s build wrote."""
    top.var.leaf.build()
    with BuildJournal() as journal:
        top.build()
    [*_, ref] = journal.written_entries()
    return BuildJournal.read_refs([ref]).get(0, dropna=True)


class TestBlockPretty:

    def test_a_block_has_one(self, top):
        assert isinstance(top.pretty, Pretty)

    @pytest.mark.pinned
    @pytest.mark.parametrize('kwargs', [{}, {'pretty': False}, {'deslash': True}, {'pretty': False, 'deslash': True},
                                        {'specline': False}, {'pretty': False, 'specline': False}])
    def test_quote_is_its_entrys(self, top, entry, kwargs):
        assert top.pretty.quote(**kwargs) == entry.pretty.quote(**kwargs)

    @pytest.mark.pinned
    @pytest.mark.parametrize('kwargs', [{}, {'deslash': True}, {'specline': False}])
    def test_cite_is_its_entrys(self, top, entry, kwargs):
        assert top.pretty.cite(**kwargs) == entry.pretty.cite(**kwargs)

    @pytest.mark.parametrize('name', ['repr', 'signaturestr', 'typestr', 'paths', 'topics', 'signature', 'type'])
    def test_the_rest_is_its_entrys(self, top, entry, name):
        assert getattr(top.pretty, name)() == getattr(entry.pretty, name)()

    @pytest.mark.parametrize('name', ['hash', 'anchor', 'tag', 'version', 'key', 'keyby', 'anchorkeypath',
                                      'datalake'])
    def test_its_properties_are_its_entrys(self, top, entry, name):
        assert getattr(top.pretty, name) == getattr(entry.pretty, name)

    def test_its_TOPICS_are_its_entrys(self, top, entry):
        """By their text: a topic marker, `DATAFILE(...)`, is equal to another only as itself."""
        assert str(top.pretty.TOPICS) == str(entry.pretty.TOPICS)

    def test_unquoted_quote_evaluates_as_the_block(self, top):
        """`pretty=False` is the record: the evaluable specline, as the block's own quote."""
        assert top.pretty.quote(pretty=False) == top.quote()

    def test_it_writes_nothing(self, top, tmp_path):
        before = sorted(p for p in tmp_path.rglob('*'))
        top.pretty.quote()
        assert sorted(p for p in tmp_path.rglob('*')) == before


class TestSpecline:

    def test_quote_starts_with_its_anchor(self, top):
        assert top.pretty.quote().startswith('$')
        assert top.pretty.quote(pretty=False).startswith('$')

    @pytest.mark.parametrize('kwargs', [{}, {'pretty': False}])
    def test_quote_without_it_is_the_rest(self, top, kwargs):
        assert top.pretty.quote(specline=False, **kwargs) == top.pretty.quote(**kwargs)[1:]

    def test_cite_without_it_is_the_rest(self, top):
        assert top.pretty.cite().startswith('$')
        assert top.pretty.cite(specline=False) == top.pretty.cite()[1:]

    def test_only_the_leading_anchor_goes(self, top):
        """A nested block's own `$` stays: it is not the leading one."""
        quote = top.pretty.quote(specline=False)
        assert not quote.startswith('$')
        assert '$' in quote

    @pytest.mark.parametrize('anchor', ['@', '#', '$'])
    def test_any_anchor(self, anchor):
        assert Pretty._unanchored_(f"{anchor}a.B(x=1)") == 'a.B(x=1)'
        assert Pretty._unanchored_('a.B(x=1)') == 'a.B(x=1)'
