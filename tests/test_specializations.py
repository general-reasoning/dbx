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
from dbx.datapoints import DIRTOPIC, DatapointTab, DatapointTable

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
        for topic in self.ownedtopics():
            with open(self.path(topic, ensure_dirpath=True), 'w') as f:
                f.write(f"{topic}-{self.var.sr}")


class V3(V2):
    """The same class once more, with VERSION bumped because `phases` changed.

    `spectra` did not, and neither did the computation behind it -- which is a
    claim only a specialization that names the OLDER version can make. V1
    declared no VERSION at all, so the value to name is ``None``, which is why
    the field's "inherit" default has to be a sentinel rather than None.
    """

    VERSION = 2

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec=dict(window='hann'),
            topics=['spectra'],
            version=None,
            note="the bump was for `phases`; `spectra` is what it always was",
        ),
    ]


def v1(url, **kw):
    return V1(url=str(url), anchor=ANCHOR, spec={'sr': 16000}, **kw)


def v2(url, **kw):
    spec = kw.pop('spec', {'sr': 16000})
    return V2(url=str(url), anchor=ANCHOR, spec=spec, **kw)


def v3(url, **kw):
    spec = kw.pop('spec', {'sr': 16000})
    return V3(url=str(url), anchor=ANCHOR, spec=spec, **kw)


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
        assert block.ownedtopics() == ['phases']

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
        assert block.ownedtopics() == []
        block.build()   # declines, loudly, and does not raise
        assert block.read('phases') == 'spectra-16000'

    def test_writing_to_a_redirected_topic_is_refused(self, tmp_path, built):
        """A __build__ that ignores ownedtopics() must not clobber the source."""
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

    def test_dry_run_reports_without_writing(self, tmp_path, built, capsys):
        block = v2(tmp_path, use_specializations=False)
        proposal = block.UNSAFE_specialize(dry_run=True, OVERRIDE=True)
        assert proposal.specialization == V2.SPECIALIZATIONS[0]
        assert proposal.paths == {'spectra': built.path('spectra')}
        out = capsys.readouterr().out
        assert 'DRY RUN -- nothing below has been done' in out
        assert 'would record' in out
        assert block.redirected_topics() == []                    # not installed
        assert v2(tmp_path, use_specializations=False).redirected_topics() == []   # not recorded

    def test_a_dry_run_redirection_is_a_proposal_not_a_True(self, tmp_path, built):
        block = v2(tmp_path, use_specializations=False)
        assert block.UNSAFE_redirect(paths={'spectra': built.path('spectra')},
                                     dry_run=True, OVERRIDE=True).paths is not None
        assert block.redirected_topics() == []

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
        assert block.ownedtopics() == ['phases']

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


# ---------------------------------------------------------------------------
# A TABLE, which is where the two assumptions above stop holding
# ---------------------------------------------------------------------------
#
# Everything above is a plain Datablock, whose `valid()` is `valid_topics()`
# and which writes nowhere but its own topics. A DatapointTable is neither:
# its `valid()` is the `done` MARKER, and its split ensures the `tab_paths`
# directory before it does anything else. Both are reasonable on their own and
# both break under a partial redirection, so this is the shape a specialization
# has to survive to be usable by the classes that most want one -- a table
# whose tabs hold terabytes and whose own topics are a few kilobytes derived
# from them.

TABLE_ANCHOR = 'Rows'


class RowTab(DatapointTab):
    """One tab, one file. No slices: this is about the TABLE's topics."""

    VERSION = 1
    TOPICS = {'rows': 'rows.txt'}

    @dataclass
    class VAR(DatapointTab.VAR):
        tab_idx: int = 0

    def __build__(self):
        with open(self.path('rows', ensure_dirpath=True), 'w') as f:
            f.write(f"rows-{self.var.tab_idx}\n")

    def __read__(self, *topicpath):
        with open(self.path('rows')) as f:
            return f.read()


class RowTableV1(DatapointTable):
    """The table before it grew a topic."""

    VERSION = 1
    TAB = RowTab
    TOPICS = {'summary': 'summary.txt', 'tab_paths': DIRTOPIC, 'done': 'done'}

    @dataclass
    class VAR(DatapointTable.VAR):
        n: int = 2

    @property
    def n_tabs(self):
        return self.var.n

    def __tab__(self, idx, **spec):
        return super().__tab__(idx, tab_idx=idx, **spec)

    def __stack__(self, results=None):
        if not self.valid_topic('summary'):
            with open(self.path('summary', ensure_dirpath=True), 'w') as f:
                f.write(''.join(self.tab(i).read('rows')
                                for i in range(self.n_tabs)))
        return super().__stack__(results)       # writes `done`

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath and topicpath[0] in ('summary', 'report'):
            with open(self.path(*topicpath)) as f:
                return f.read()
        return super().__read__(*topicpath)


class RowTableV2(RowTableV1):
    """The same table, grown a `report` topic derived from what the tabs hold.

    No VAR field was added and the TAB did not move, so the specialization pins
    nothing and names the three topics the older table declared.
    """

    TOPICS = {'summary': 'summary.txt', 'report': 'report.txt',
              'tab_paths': DIRTOPIC, 'done': 'done'}

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={},
            topics=['summary', 'tab_paths', 'done'],
            note="`report` is derived from the tabs, which did not re-key",
        ),
    ]

    def __stack__(self, results=None):
        if not self.valid_topic('report'):
            with open(self.path('report', ensure_dirpath=True), 'w') as f:
                f.write(f"report over {self.n_tabs} tabs\n")
        return super().__stack__(results)       # summary, then `done`


def v1table(url, **kw):
    return RowTableV1(url=str(url), anchor=TABLE_ANCHOR, spec={'n': 2}, **kw)


def v2table(url, **kw):
    return RowTableV2(url=str(url), anchor=TABLE_ANCHOR, spec={'n': 2}, **kw)


@pytest.fixture
def built_table(tmp_path):
    """The narrower TABLE, built: two tabs, a summary, a done marker."""
    t = v1table(tmp_path)
    t.build()
    assert t.valid()
    return t


class TestASpecializedTable:
    """A table reads its older self's topics and builds only the new one."""

    def test_the_reconstruction_is_the_older_tables_hash(self, tmp_path):
        assert v2table(tmp_path).get_hash(RowTableV2.SPECIALIZATIONS[0]) == \
            v1table(tmp_path).hash

    def test_it_resolves_to_the_older_build(self, tmp_path, built_table):
        table = v2table(tmp_path)
        assert table.specialization == RowTableV2.SPECIALIZATIONS[0]
        assert table.redirected_topics() == ['summary', 'tab_paths', 'done']
        assert table.ownedtopics() == ['report']

    def test_a_marker_valid_does_not_skip_the_build(self, tmp_path, built_table):
        """The failure this exists to stop, and it was SILENT.

        `DatapointTable.valid()` is `valid_topic('done')`, and `done` is one of
        the topics the specialization redirects -- so the table reports itself
        built, off another build's marker, before `report` exists. `build()`
        used to ask `valid()` and nothing else, skip, and return a table that
        raises the first time anything reads the topic it was grown for. No
        error, no warning, and a topic that never gets made however many times
        you rebuild.
        """
        table = v2table(tmp_path)
        assert table.valid()                       # `done`, through the redirection
        assert not table.valid_topic('report')     # ... and yet
        assert table.owedtopics() == ['report']    # which is what build() asks
        table.build()
        assert table.valid_topic('report')
        assert table.read('report') == "report over 2 tabs\n"

    def test_the_split_does_not_ensure_a_redirected_directory(self, tmp_path,
                                                              built_table):
        """The second failure, which was loud but total.

        `path(ensure_dirpath=True)` on a redirected topic raises -- correctly,
        since it would be creating a directory inside another block's data --
        and `DatapointTable.__split__` called exactly that on `tab_paths`
        before doing anything else. The tab machinery was therefore not merely
        unnecessary for a specialized table, it was unreachable.
        """
        table = v2table(tmp_path)
        with pytest.raises(Exception):
            table.path('tab_paths', ensure_dirpath=True)
        table.build()                              # the split no longer calls it
        assert table.valid_topic('report')

    def test_no_sentinel_is_written_into_the_other_blocks_tab_paths(
            self, tmp_path, built_table):
        """A redirection covering `tab_paths` can only be one that left the TAB
        alone, so the sentinels already there name the same tabs. Ours would be
        a write into its data."""
        sentinels = sorted(os.listdir(built_table.path('tab_paths')))
        v2table(tmp_path).build()
        assert sorted(os.listdir(built_table.path('tab_paths'))) == sentinels

    def test_the_older_topics_are_not_rebuilt(self, tmp_path, built_table):
        table = v2table(tmp_path)
        table.build()
        assert table.path('summary') == built_table.path('summary')
        assert table.path('report').startswith(table.anchorkeypath)
        assert table.read('summary') == built_table.read('summary')

    def test_a_second_build_owes_nothing(self, tmp_path, built_table):
        v2table(tmp_path).build()
        again = v2table(tmp_path)
        assert again.owedtopics() == []
        again.build()
        assert again.valid_topic('report')

    def test_an_unspecialized_table_still_builds_whole(self, tmp_path):
        """Nothing above may change what an ordinary table does."""
        table = v2table(tmp_path, use_specializations=False)
        assert table.owedtopics() == []            # not redirected: owes nothing
        table.build()
        assert table.valid()
        for topic in ('summary', 'report', 'done'):
            assert table.valid_topic(topic)


class TestOwedTopics:
    """`owedtopics()` is empty for anything that is not partially redirected."""

    def test_an_ordinary_unbuilt_block_owes_nothing(self, tmp_path):
        assert v2(tmp_path).owedtopics() == []

    def test_an_ordinary_built_block_owes_nothing(self, tmp_path):
        block = v2(tmp_path, use_specializations=False)
        block.build()
        assert block.owedtopics() == []

    def test_a_specialized_block_owes_what_it_did_not_redirect(self, tmp_path,
                                                               built):
        block = v2(tmp_path)
        assert block.redirected_topics() == ['spectra']
        assert block.owedtopics() == ['phases']
        block.build()
        assert block.owedtopics() == []


class TestASpecializationAcrossAVersionBump:
    """`version=` completes the reconstruction.

    `type()` is the spec, the version and the topics and nothing else, and a
    specialization used to name only two of the three -- the version came from
    the class, so a bump moved the reconstruction onto an identity nobody had
    ever built and the only way to keep a specialization working was never to
    bump. Naming all three describes the narrower block completely.
    """

    SP = V3.SPECIALIZATIONS[0]

    def test_it_reconstructs_the_older_versions_hash(self, tmp_path):
        assert v3(tmp_path).get_hash(self.SP) == v1(tmp_path).hash

    def test_the_bump_did_move_this_blocks_own_hash(self, tmp_path):
        assert v3(tmp_path).hash != v2(tmp_path).hash

    def test_without_the_version_it_would_reconstruct_nothing_built(self, tmp_path):
        """What the field is for: the same specialization without it names
        (this spec, version=2, spectra), which no build ever had."""
        unversioned = Datablock.Specialization(
            spec=dict(window='hann'), topics=['spectra'])
        block = v3(tmp_path)
        assert block.get_hash(unversioned) != block.get_hash(self.SP)
        assert block.get_hash(unversioned) != v1(tmp_path).hash

    def test_the_version_is_in_the_rendered_type(self, tmp_path):
        block = v3(tmp_path)
        assert 'version=2' in block.type()
        assert 'version=None' in block.type(specialization=self.SP)

    def test_it_resolves_and_builds_only_what_it_did_not_cover(self, tmp_path,
                                                               built):
        block = v3(tmp_path)
        assert block.specialization == self.SP
        assert block.redirected_topics() == ['spectra']
        assert block.ownedtopics() == ['phases']
        block.build()
        assert block.read('spectra') == 'spectra-16000'     # V1's bytes
        assert block.read('phases') == 'phases-16000'       # freshly built

    def test_the_version_survives_the_journal_record(self, tmp_path, built):
        """The record is literal, so `version` has to round-trip through it --
        a redirection read back must say which specialization installed it, and
        two alike but for the version are two different claims."""
        v3(tmp_path).build()
        later = v3(tmp_path, use_specializations=False)
        assert later.redirection.specialization == self.SP
        assert later.redirection.specialization.version is None

    def test_a_version_is_not_matched_against_this_block(self, tmp_path):
        """It states what the OTHER block was; it is not a condition on this
        one, so it can never be the reason a specialization does not apply."""
        assert v3(tmp_path)._specialization_mismatch(self.SP) is None


class TestTheOwnedOwedPair:
    """Two names one letter apart, so what separates them is worth pinning."""

    def test_owned_is_what_this_block_must_produce(self, tmp_path, built):
        block = v2(tmp_path)
        assert block.ownedtopics() == ['phases']

    def test_owed_is_the_subset_not_yet_there(self, tmp_path, built):
        block = v2(tmp_path)
        assert block.owedtopics() == ['phases']
        block.build()
        assert block.ownedtopics() == ['phases']   # ownership does not change
        assert block.owedtopics() == []            # owing does

    def test_owed_is_always_a_subset_of_owned(self, tmp_path, built):
        block = v2(tmp_path)
        assert set(block.owedtopics()) <= set(block.ownedtopics())

    def test_buildtopics_is_the_old_name(self, tmp_path):
        """Kept so a caller written against it still resolves."""
        assert Datablock.buildtopics is Datablock.ownedtopics
        block = v2(tmp_path)
        assert block.buildtopics() == block.ownedtopics()
