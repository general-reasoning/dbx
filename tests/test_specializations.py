"""A block reads a narrower, already-built block's data instead of rebuilding it.

A class grows a VAR field and a topic. Every block of it re-keys, and the topics
that did not change are rebuilt for nothing. A :class:`Datablock.Specialization`
says that at a given value of the new field, the topics it names ARE the topics
of the block this class used to be -- whose identity is this one's with that
field dropped and those topics alone, and whose hash is therefore reconstructible
from here. The build is found in the journal by that hash and read in place.

The evolution is modelled as two classes sharing one ``anchor``, because a test
cannot redefine a class in the middle of itself. A real class keeps its anchor
by keeping its name, which is the same thing.
"""
import os
import pickle

import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock

ANCHOR = 'Spectra'


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class V1(Datablock):
    """The class before it grew."""
    TOPICS = {'spectra': 'spectra.npy'}

    @dataclass
    class VAR(Datablock.VAR):
        sr: int = 16000

    def __build__(self):
        with open(self.path('spectra', ensure_dirpath=True), 'w') as f:
            f.write(f"spectra-{self.var.sr}")

    def __read__(self, topic):
        with open(self.path(topic)) as f:
            return f.read()


class V2(V1):
    """The same class, grown a `window` field and a `phases` topic."""
    TOPICS = {'spectra': 'spectra.npy', 'phases': 'phases.npy'}

    @dataclass
    class VAR(V1.VAR):
        window: str = 'hann'

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec=dict(window='hann'),
            topics=['spectra'],
            note="hann was the only window there was before the field existed",
        ),
    ]

    def __build__(self):
        for topic in self.buildtopics():
            with open(self.path(topic, ensure_dirpath=True), 'w') as f:
                f.write(f"{topic}-{self.var.sr}")


def v1(url, **kw):
    return V1(url=str(url), anchor=ANCHOR, spec={'sr': 16000}, **kw)


def v2(url, **kw):
    spec = kw.pop('spec', {'sr': 16000})
    return V2(url=str(url), anchor=ANCHOR, spec=spec, **kw)


@pytest.fixture
def built(tmp_path):
    """The narrower block, built."""
    b = v1(tmp_path)
    b.build()
    return b


class TestTheReconstructedIdentity:
    """The hash of the narrower block, computed from the wider one."""

    def test_it_is_the_narrower_blocks_hash(self, tmp_path):
        assert v2(tmp_path).get_hash(V2.SPECIALIZATIONS[0]) == v1(tmp_path).hash

    def test_it_is_not_this_blocks_own_hash(self, tmp_path):
        block = v2(tmp_path)
        assert block.get_hash(V2.SPECIALIZATIONS[0]) != block.hash

    def test_the_type_drops_the_field_and_the_topic(self, tmp_path):
        t = v2(tmp_path).type(specialization=V2.SPECIALIZATIONS[0])
        assert 'window' not in t
        assert 'phases' not in t
        assert t == v1(tmp_path).type()

    def test_hash_is_cached_apart_from_the_blocks_own(self, tmp_path):
        block = v2(tmp_path)
        own, specialized = block.hash, block.get_hash(V2.SPECIALIZATIONS[0])
        assert block.hash == own            # not clobbered by the specialized one
        assert block.get_hash(V2.SPECIALIZATIONS[0]) == specialized

    def test_a_pin_on_an_unknown_field_raises(self, tmp_path):
        block = v2(tmp_path)
        with pytest.raises(ValueError, match='does not declare'):
            block._specialization_mismatch(
                Datablock.Specialization(spec=dict(nosuch=1), topics=['spectra']))

    def test_an_unknown_topic_raises(self, tmp_path):
        block = v2(tmp_path)
        with pytest.raises(KeyError):
            block._specialization_mismatch(
                Datablock.Specialization(spec=dict(window='hann'), topics=['nosuch']))


class TestMatching:

    def test_a_field_left_at_its_default_matches(self, tmp_path):
        """The case the feature exists for: the new field was never mentioned."""
        assert v2(tmp_path).matching_specializations() == V2.SPECIALIZATIONS

    def test_the_pinned_value_given_explicitly_matches(self, tmp_path):
        block = v2(tmp_path, spec={'sr': 16000, 'window': 'hann'})
        assert block.matching_specializations() == V2.SPECIALIZATIONS

    def test_another_value_does_not(self, tmp_path):
        block = v2(tmp_path, spec={'sr': 16000, 'window': 'hamming'})
        assert block.matching_specializations() == []
        assert "pinned 'hann'" in block.specializations()[0]['why']

    def test_a_specialization_that_drops_nothing_is_refused(self, tmp_path):
        """It reconstructs this block's own identity, so there is no other block."""
        block = v2(tmp_path)
        why = block._specialization_mismatch(
            Datablock.Specialization(spec={}, topics=['spectra', 'phases']))
        assert 'own identity' in why


class TestReadingThroughIt:

    def test_the_topic_resolves_to_the_narrower_blocks_path(self, tmp_path, built):
        assert v2(tmp_path).path('spectra') == built.path('spectra')

    def test_and_reads_its_data(self, tmp_path, built):
        assert v2(tmp_path).read('spectra') == 'spectra-16000'

    def test_the_grown_topic_stays_this_blocks_own(self, tmp_path, built):
        block = v2(tmp_path)
        assert block.path('phases').startswith(block.anchorkeypath)

    def test_it_reports_which_specialization(self, tmp_path, built):
        assert v2(tmp_path).specialization == V2.SPECIALIZATIONS[0]

    def test_nothing_is_installed_when_there_is_no_build_to_find(self, tmp_path):
        block = v2(tmp_path)
        assert block.specialization is None
        assert block.redirected_topics() == []

    def test_nothing_is_installed_when_the_block_has_its_own_data(self, tmp_path, built):
        """A block built on its own terms has nothing to borrow."""
        v2(tmp_path, use_specializations=False).build()
        assert v2(tmp_path).specialization is None

    def test_a_partial_build_does_not_undo_it(self, tmp_path, built):
        """The built half is `phases`; `spectra` is still the narrower block's."""
        v2(tmp_path).build()
        later = v2(tmp_path)
        assert later.path('spectra') == built.path('spectra')
        assert later.valid() is True

    def test_use_specializations_False_declines(self, tmp_path, built):
        assert v2(tmp_path, use_specializations=False).specialization is None

    def test_redirect_False_declines(self, tmp_path, built):
        assert v2(tmp_path, redirect=False).specialization is None


class TestBuildingTheRest:

    def test_the_topics_divide(self, tmp_path, built):
        block = v2(tmp_path)
        assert block.redirected_topics() == ['spectra']
        assert block.buildtopics() == ['phases']

    def test_invalid_until_the_rest_is_built(self, tmp_path, built):
        assert v2(tmp_path).valid() is False

    def test_build_produces_only_the_rest(self, tmp_path, built):
        block = v2(tmp_path)
        block.build()
        assert block.valid() is True
        assert block.read('phases') == 'phases-16000'
        assert block.path('spectra') == built.path('spectra')

    def test_a_total_redirection_still_declines_to_build(self, tmp_path, built):
        block = v2(tmp_path)
        block.UNSAFE_redirect(paths={'spectra': built.path('spectra'),
                                     'phases': built.path('spectra')}, OVERRIDE=True)
        assert block.buildtopics() == []
        block.build()   # declines, loudly, and does not raise
        assert block.read('phases') == 'spectra-16000'

    def test_writing_to_a_redirected_topic_is_refused(self, tmp_path, built):
        """A __build__ that ignores buildtopics() must not clobber the source."""
        block = v2(tmp_path)
        with pytest.raises(ValueError, match='REDIRECTED'):
            block.path('spectra', ensure_dirpath=True)
        assert built.read('spectra') == 'spectra-16000'

    def test_reading_a_redirected_topic_is_not(self, tmp_path, built):
        assert v2(tmp_path).path('spectra') is not None


class TestItTravelsWithTheBlock:
    """Resolving costs a journal scan, so it is resolved once."""

    def test_pickle(self, tmp_path, built):
        block = v2(tmp_path)
        back = pickle.loads(pickle.dumps(block))
        assert back.path('spectra') == block.path('spectra')

    def test_set_of_an_operational_parameter(self, tmp_path, built):
        block = v2(tmp_path)
        assert block.set(tag='t').path('spectra') == block.path('spectra')

    def test_a_relocation_resolves_again(self, tmp_path, built):
        """The carried paths are absolute, so a new url is a new question."""
        block = v2(tmp_path)
        moved = block.set(url=str(tmp_path / 'elsewhere'))
        assert moved.redirected_topics() == []

    def test_it_stays_out_of_quote(self, tmp_path, built):
        """A quote carrying absolute paths would not relocate."""
        q = v2(tmp_path).quote()
        assert '__redirected_paths__' not in q
        assert 'use_specializations' not in q

    def test_a_block_that_never_heard_of_them_keeps_its_quote(self, tmp_path):
        assert 'use_specializations' not in v1(tmp_path).quote()


class TestRecordingIt:

    def test_installing_one_records_it(self, tmp_path, built):
        v2(tmp_path)
        later = v2(tmp_path, use_specializations=False)
        assert later.redirected_topics() == ['spectra']
        assert later.redirection.specialization == V2.SPECIALIZATIONS[0]

    def test_memory_mode_writes_nothing(self, tmp_path, built):
        block = v2(tmp_path, use_specializations='memory')
        assert block.redirected_topics() == ['spectra']
        assert v2(tmp_path, use_specializations=False).redirected_topics() == []

    def test_the_record_is_what_later_constructions_read(self, tmp_path, built):
        """So the journal is scanned once per block, not once per construction."""
        v2(tmp_path)
        later = v2(tmp_path)
        assert later.redirected_topics() == ['spectra']
        # It came off the recorded redirection, not off a fresh resolution.
        assert later.__dict__.get('__specialization__') is None

    def test_unsafe_specialize_is_the_explicit_form(self, tmp_path, built):
        block = v2(tmp_path, use_specializations='memory')
        assert block.UNSAFE_specialize(OVERRIDE=True) == V2.SPECIALIZATIONS[0]
        later = v2(tmp_path, use_specializations=False)
        assert later.redirection.topics == ['spectra']

    def test_unsafe_specialize_reports_when_there_is_nothing(self, tmp_path):
        assert v2(tmp_path).UNSAFE_specialize(OVERRIDE=True) is None

    def test_it_reports_which_specialization_in_memory_mode_too(self, tmp_path, built):
        assert v2(tmp_path, use_specializations='memory').specialization == V2.SPECIALIZATIONS[0]


class TestTheReport:
    """A miss is never silent."""

    def test_it_names_the_hash_it_looked_for(self, tmp_path):
        row = v2(tmp_path).specializations()[0]
        assert row['matches'] is True
        assert row['hash'] == v1(tmp_path).hash
        assert row['hash'] in row['why']

    def test_it_reports_a_resolved_one(self, tmp_path, built):
        row = v2(tmp_path).specializations()[0]
        assert row['why'] is None
        assert row['paths'] == {'spectra': built.path('spectra')}
        assert row['entry'] is not None

    def test_it_reports_a_pin_that_no_longer_matches(self, tmp_path, built):
        row = v2(tmp_path, spec={'sr': 16000, 'window': 'hamming'}).specializations()[0]
        assert row['matches'] is False
        assert 'hamming' in row['why']


class TestPartialRedirectionOnItsOwn:
    """`topics=` restricts any redirection, specialization or not."""

    def test_it_redirects_only_what_it_names(self, tmp_path, built):
        block = v2(tmp_path, use_specializations=False)
        block.UNSAFE_redirect(paths={'spectra': built.path('spectra'),
                                     'phases': built.path('spectra')},
                              topics=['spectra'], OVERRIDE=True)
        assert block.redirected_topics() == ['spectra']
        assert block.buildtopics() == ['phases']

    def test_the_restriction_is_recorded(self, tmp_path, built):
        block = v2(tmp_path, use_specializations=False)
        block.UNSAFE_redirect(paths={'spectra': built.path('spectra'),
                                     'phases': built.path('spectra')},
                              topics=['spectra'], OVERRIDE=True)
        later = v2(tmp_path, use_specializations=False)
        assert later.redirected_topics() == ['spectra']

    def test_naming_no_topic_of_this_block_redirects_nothing(self, tmp_path, built):
        block = v2(tmp_path, use_specializations=False)
        assert block.UNSAFE_redirect(paths={'spectra': built.path('spectra')},
                                     topics=['nosuch'], OVERRIDE=True) is False


class TestTheIdentityIsNotTheRedirection:
    """Installing a redirection must not move the block it is installed on."""

    def test_hash_is_the_same_before_and_after(self, tmp_path, built):
        block = v2(tmp_path, keyby='tag', tag='t', use_specializations=False)
        before, path_before = block.hash, block.anchorkeypath
        block.UNSAFE_redirect(paths={'spectra': built.path('spectra')}, OVERRIDE=True)
        assert block.hash == before
        assert block.anchorkeypath == path_before

    def test_hash_does_not_depend_on_when_it_is_first_asked(self, tmp_path, built):
        """The bug this replaced: _hash cached before or after moved the answer."""
        cold = v2(tmp_path, keyby='tag', tag='t')       # resolves on construction
        warm = v2(tmp_path, keyby='tag', tag='t', use_specializations=False)
        assert cold.hash == warm.hash
