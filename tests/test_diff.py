"""``diff()`` — the three components a signature is made of.

A ``signature`` is a norm, a version and the topics, joined; a hash is that
string. So ``diffnorm()``, ``diffversion()`` and ``difftopics()`` between them
account for every way two blocks can hash differently, and ``diff()`` returns
all three at once.

``difftopics()`` compares ``signature_topics()`` — the very segments the
signature is built from — so the two cannot drift: a topic difference the
signature sees is one the diff reports, and vice versa. The tests here assert
that equivalence directly, over a spread of TOPICS shapes, rather than trusting
the two renderings to stay in step.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import (
    ABSENT,
    DIRTOPIC,
    SIGNATURE_TOPICS,
    SYNTOPIC,
    Datablock,
)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch, tmp_path):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.setenv('DBX_ROOT', str(tmp_path))


class Block(Datablock):
    VERSION = 1
    TOPICS = {'out': 'out.txt', 'data': {'frames': DIRTOPIC}}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1

    def __build__(self):
        with open(self.path('out', ensure_dirpath=True), 'w') as f:
            f.write('data')


class Renamed(Block):
    TOPICS = {'out': 'renamed.txt', 'data': {'frames': DIRTOPIC}}


class Reordered(Block):
    """The same topics, declared the other way round."""
    TOPICS = {'data': {'frames': DIRTOPIC}, 'out': 'out.txt'}


class Extra(Block):
    TOPICS = {'out': 'out.txt', 'data': {'frames': DIRTOPIC}, 'log': 'run.log'}


class Deeper(Block):
    TOPICS = {'out': 'out.txt', 'data': {'frames': {'raw': DIRTOPIC}}}


class Synthetic(Block):
    TOPICS = {'out': 'out.txt', 'data': {'frames': SYNTOPIC}}


class Listed(Datablock):
    VERSION = 1
    TOPICS = ['out', 'data']


class Dicted(Datablock):
    """The same names as Listed, declared as directory topics."""
    VERSION = 1
    TOPICS = {'out': DIRTOPIC, 'data': DIRTOPIC}


class Untopicked(Datablock):
    VERSION = 1


class Empty(Datablock):
    VERSION = 1
    TOPICS = {}


class Versioned(Block):
    VERSION = 2


class Unversioned(Block):
    VERSION = None


ALL = [Block, Renamed, Reordered, Extra, Deeper, Synthetic,
       Listed, Dicted, Untopicked, Empty]


def block(cls, tmp_path, **kwargs):
    return cls(url=str(tmp_path), **kwargs)


class TestSignatureTopics:
    """The rendering the signature is built from, exposed."""

    def test_it_is_what_the_signature_joins(self, tmp_path):
        b = block(Block, tmp_path)
        for segment in b.signature_topics():
            assert segment in b.signature()

    def test_a_leaf_is_named_by_its_full_path(self, tmp_path):
        assert block(Block, tmp_path).signature_topics() == (
            'topic:out=out.txt', 'topic:data/frames=None',
        )

    def test_a_list_declaration_renders_without_a_filename(self, tmp_path):
        assert block(Listed, tmp_path).signature_topics() == ('topic:out', 'topic:data')

    def test_no_topics_renders_as_such(self, tmp_path):
        assert block(Untopicked, tmp_path).signature_topics() == ('topics:None',)

    def test_empty_topics_renders_as_nothing(self, tmp_path):
        assert block(Empty, tmp_path).signature_topics() == ()

    def test_the_signature_uses_the_same_segments(self, tmp_path):
        b = block(Block, tmp_path)
        for segment in b.signature_topics():
            assert segment in b.signature()


class TestDifftopicsAgreesWithSignature:
    """The equivalence the method exists for, asserted both ways."""

    @pytest.mark.parametrize('one', ALL)
    @pytest.mark.parametrize('two', ALL)
    def test_empty_exactly_when_the_segments_agree(self, one, two, tmp_path):
        a, b = block(one, tmp_path), block(two, tmp_path)
        same_segments = a.signature_topics() == b.signature_topics()
        assert bool(a.difftopics(b)) is not same_segments

    @pytest.mark.parametrize('other', ALL)
    def test_a_difference_means_a_different_hash(self, other, tmp_path):
        """Nothing but the topics varies across these, so a reported difference
        is a hash that moved, and no difference is a hash that did not."""
        a, b = block(Block, tmp_path), block(other, tmp_path)
        assert bool(a.difftopics(b)) is (a.hash != b.hash)

    def test_reordering_is_a_difference(self, tmp_path):
        """Segments are JOINED, so their order is part of the identity even
        though no one topic changed."""
        a, b = block(Block, tmp_path), block(Reordered, tmp_path)
        assert a.hash != b.hash
        assert a.difftopics(b) == {
            SIGNATURE_TOPICS: (a.signature_topics(), b.signature_topics())
        }

    def test_empty_topics_against_none_is_a_difference(self, tmp_path):
        a, b = block(Untopicked, tmp_path), block(Empty, tmp_path)
        assert a.hash != b.hash
        assert list(a.difftopics(b)) == [SIGNATURE_TOPICS]

    def test_a_list_and_the_equivalent_dict_differ(self, tmp_path):
        """`topic:out` against `topic:out=None`: same names, different rendering."""
        a, b = block(Listed, tmp_path), block(Dicted, tmp_path)
        assert a.hash != b.hash
        assert a.difftopics(b) == {'out': (ABSENT, 'None'), 'data': (ABSENT, 'None')}


class TestDifftopicsReports:

    def test_a_renamed_file_is_reported_at_its_path(self, tmp_path):
        assert block(Block, tmp_path).difftopics(block(Renamed, tmp_path)) == {
            'out': ('out.txt', 'renamed.txt')
        }

    def test_an_added_topic_is_absent_on_one_side(self, tmp_path):
        assert block(Block, tmp_path).difftopics(block(Extra, tmp_path)) == {
            'log': (ABSENT, 'run.log')
        }

    def test_a_nested_path_is_reported_whole(self, tmp_path):
        diff = block(Block, tmp_path).difftopics(block(Deeper, tmp_path))
        assert diff == {'data/frames': ('None', ABSENT), 'data/frames/raw': (ABSENT, 'None')}

    def test_a_syntopic_is_distinguishable_from_a_dirtopic(self, tmp_path):
        assert block(Block, tmp_path).difftopics(block(Synthetic, tmp_path)) == {
            'data/frames': ('None', '()')
        }

    def test_identical_blocks_report_nothing(self, tmp_path):
        assert block(Block, tmp_path).difftopics(block(Block, tmp_path)) == {}

    def test_report_renders_text(self, tmp_path):
        text = block(Block, tmp_path).difftopics(block(Renamed, tmp_path), report=True)
        assert 'out' in text and 'out.txt' in text and 'renamed.txt' in text

    def test_report_says_so_when_there_is_nothing(self, tmp_path):
        assert block(Block, tmp_path).difftopics(block(Block, tmp_path), report=True) == 'no differences'


class TestDifftopicsOtherSides:
    """A block, a declaration, a journal entry -- the same comparison."""

    def test_a_topics_dict(self, tmp_path):
        b = block(Block, tmp_path)
        assert b.difftopics({'out': 'out.txt', 'data': {'frames': DIRTOPIC}}) == {}
        assert b.difftopics({'out': 'other.txt', 'data': {'frames': DIRTOPIC}}) == {
            'out': ('out.txt', 'other.txt')
        }

    def test_a_topics_list(self, tmp_path):
        assert block(Listed, tmp_path).difftopics(['out', 'data']) == {}

    def test_none_means_a_block_declaring_no_topics(self, tmp_path):
        assert block(Untopicked, tmp_path).difftopics(None) == {}
        assert list(block(Block, tmp_path).difftopics(None)) == ['out', 'data/frames']

    def test_the_str_dict_a_journal_records(self, tmp_path):
        b = block(Block, tmp_path)
        assert b.difftopics(str({'out': 'out.txt', 'data': {'frames': DIRTOPIC}})) == {}

    def test_a_journal_entry(self, tmp_path):
        """`journal=` reads the block's OWN anchor, so the other class is
        pointed at Block's anchor to have a journal to be compared against."""
        built = block(Block, tmp_path)
        built.build()
        assert block(Block, tmp_path).difftopics(journal=dict(loc=0)) == {}
        renamed = Renamed(url=str(tmp_path), anchor=built.anchor)
        assert renamed.difftopics(journal=dict(event='build:end', loc=0)) == {
            'out': ('renamed.txt', 'out.txt')
        }

    def test_a_journal_entry_passed_directly(self, tmp_path):
        block(Block, tmp_path).build()
        entry = block(Block, tmp_path).journal(loc=0)
        assert block(Renamed, tmp_path).difftopics(entry) == {'out': ('renamed.txt', 'out.txt')}

    def test_an_other_side_is_required(self, tmp_path):
        with pytest.raises(ValueError):
            block(Block, tmp_path).difftopics()


class TestDiffversion:

    def test_none_when_the_versions_agree(self, tmp_path):
        assert block(Block, tmp_path).diffversion(block(Block, tmp_path)) is None

    def test_the_pair_when_they_do_not(self, tmp_path):
        assert block(Block, tmp_path).diffversion(block(Versioned, tmp_path)) == (1, 2)

    def test_a_bare_version_value(self, tmp_path):
        b = block(Block, tmp_path)
        assert b.diffversion(1) is None
        assert b.diffversion(7) == (1, 7)

    def test_it_compares_as_the_signature_renders(self, tmp_path):
        """`version=1` either way, so the hash cannot tell them apart and
        neither does this -- though both values are still reported."""
        b = block(Block, tmp_path)
        assert b.diffversion('1') is None
        assert b.hash == Block(url=str(tmp_path)).hash

    def test_an_undeclared_version_is_None_not_absent(self, tmp_path):
        a, b = block(Block, tmp_path), block(Unversioned, tmp_path)
        assert a.diffversion(b) == (1, None)
        assert b.diffversion(None) is None

    def test_it_agrees_with_the_hash(self, tmp_path):
        for other in (Block, Versioned, Unversioned):
            a, b = block(Block, tmp_path), block(other, tmp_path)
            assert bool(a.diffversion(b)) is (a.hash != b.hash)

    def test_a_journal_entry(self, tmp_path):
        built = block(Block, tmp_path)
        built.build()
        assert block(Block, tmp_path).diffversion(journal=dict(loc=0)) is None
        versioned = Versioned(url=str(tmp_path), anchor=built.anchor)
        assert versioned.diffversion(journal=dict(event='build:end', loc=0)) == (2, 1)

    def test_an_other_side_is_required(self, tmp_path):
        """None is a version, so it cannot double as 'nothing passed'."""
        with pytest.raises(ValueError):
            block(Block, tmp_path).diffversion()


class TestDiff:

    def test_it_is_a_triple(self, tmp_path):
        d = block(Block, tmp_path).diff(block(Block, tmp_path))
        assert isinstance(d, tuple) and len(d) == 3

    def test_its_parts_are_named(self, tmp_path):
        d = block(Block, tmp_path).diff(block(Renamed, tmp_path))
        subsig, topics, version = d
        assert d.subsig is subsig and d.topics is topics and d.version is version

    def test_the_parts_are_what_the_three_methods_return(self, tmp_path):
        a, b = block(Block, tmp_path, spec=dict(x=2)), block(Versioned, tmp_path)
        d = a.diff(b)
        assert d.subsig == a.diffsubsignature(b.subsignature())
        assert d.topics == a.difftopics(b)
        assert d.version == a.diffversion(b)

    def test_nothing_differs_between_identical_blocks(self, tmp_path):
        assert not any(block(Block, tmp_path).diff(block(Block, tmp_path)))

    def test_a_spec_change_shows_up_in_the_subsig(self, tmp_path):
        d = block(Block, tmp_path, spec=dict(x=1)).diff(block(Block, tmp_path, spec=dict(x=2)))
        assert d.subsig and not d.topics and not d.version

    def test_a_topic_change_shows_up_in_the_topics(self, tmp_path):
        d = block(Block, tmp_path).diff(block(Renamed, tmp_path))
        assert d.topics and not d.version

    def test_a_version_change_shows_up_in_the_version(self, tmp_path):
        d = block(Block, tmp_path).diff(block(Versioned, tmp_path))
        assert d.version and not d.topics

    def test_any_of_it_means_a_different_hash(self, tmp_path):
        """Every component of the signature is covered by one of the three."""
        a = block(Block, tmp_path)
        others = [block(cls, tmp_path) for cls in (Block, Renamed, Reordered, Extra,
                                                   Versioned, Unversioned)]
        others.append(block(Block, tmp_path, spec=dict(x=99)))
        others.append(block(Block, tmp_path, tag='tagged'))
        for other in others:
            assert any(a.diff(other)) is (a.hash != other.hash), other

    def test_a_journal_entry(self, tmp_path):
        built = block(Block, tmp_path)
        built.build()
        assert not any(block(Block, tmp_path).diff(journal=dict(loc=0)))
        renamed = Renamed(url=str(tmp_path), anchor=built.anchor)
        d = renamed.diff(journal=dict(event='build:end', loc=0))
        assert d.topics == {'out': ('renamed.txt', 'out.txt')}

    def test_report_renders_all_three(self, tmp_path):
        d = block(Block, tmp_path, spec=dict(x=2)).diff(block(Versioned, tmp_path), report=True)
        assert all(isinstance(part, str) for part in d)
        assert 'renamed' not in d.topics and d.topics == 'no differences'
        assert 'self : 1' in d.version and 'other: 2' in d.version

    def test_diffsubsignature_options_are_forwarded(self, tmp_path):
        a = block(Block, tmp_path, spec=dict(x=1))
        b = block(Block, tmp_path, spec=dict(x=2))
        assert a.diff(b, recursive=False).subsig == a.diffsubsignature(b.subsignature(), recursive=False)

    def test_an_other_side_is_required(self, tmp_path):
        with pytest.raises(ValueError):
            block(Block, tmp_path).diff()

    def test_it_refuses_something_that_is_neither(self, tmp_path):
        with pytest.raises(TypeError):
            block(Block, tmp_path).diff('a norm string')
