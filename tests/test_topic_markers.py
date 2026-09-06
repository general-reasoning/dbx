"""Topic markers -- a topic declared by what it IS, not by a sentinel value.

``DIRTOPIC`` is ``None`` and ``SYNTOPIC`` is ``()``: values chosen so they
cannot collide with a filename, which a reader still has to know by heart.
``DIR`` and ``SYNTHETIC`` say the same things as themselves, and ``DATASLICE`` says
the one thing no sentinel could -- what columns the slice has, which is the
reason the markers exist at all.  Under ``SLICETOPIC`` a slice's columns lived
in whatever dict ``__build__`` happened to hand the writers: they reached no
hash, so a block could change shape and go on claiming to be the same block.

A declaration holding a marker IS a marker declaration -- nothing announces it,
and a declaration mixing the two spellings is refused rather than rendered half
each way.  The spellings differ: a marker renders as itself where the sentinel
renders as its value, and a filename renders quoted where the older spelling
renders it bare.  So adopting the markers re-keys a block, and every block
spelled the older way keeps its rendering, and its hashes, exactly.

The other half: a ``DatapointTable`` spelled with the markers stops carrying its
TAB's slice topics in its own signature.  Another block's topics were never its
identity to hold, and the TAB is already part of the table.
"""
import os
from dataclasses import dataclass

import pytest

import dbx
from dbx.datablocks import DIR, DIRTOPIC, SYNTHETIC, SYNTOPIC, Datablock, literal_topics

pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

from dbx.datapoints import DATASLICE, SLICETOPIC, DatapointTab, DatapointTable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Marked(Datablock):
    """One of each marker, beside an ordinary file topic."""

    VERSION = 1
    TOPICS = {'data': 'data.txt', 'masks': DIR, 'cache': SYNTHETIC}

    def __build__(self):
        with open(self.path('data', ensure_dirpath=True), 'w') as f:
            f.write('data')
        os.makedirs(self.dirpath('masks'), exist_ok=True)
        with open(os.path.join(self.dirpath('masks'), 'm.bin'), 'wb') as f:
            f.write(b'\x00')


class Sentinels(Datablock):
    """The same block in the spelling the markers replace."""

    VERSION = 1
    TOPICS = {'data': 'data.txt', 'masks': DIRTOPIC, 'cache': SYNTOPIC}

    def __build__(self):
        pass


class MarkedTab(DatapointTab):
    VERSION = 1
    TOPICS = {'numbers': DATASLICE(idx='int', square='int'), 'note': 'note.txt'}

    @dataclass
    class VAR(DatapointTab.VAR):
        n: int = 3

    def __build__(self):
        with self.slice_writers() as writers:
            for i in range(self.var.n):
                writers['numbers'].write({'idx': i, 'square': i * i})
        with self.fs.open(self.path('note', ensure_dirpath=True), 'w') as f:
            f.write("tab note")


class MarkedTable(DatapointTable):
    VERSION = 1
    TAB = MarkedTab
    TOPICS = {'tab_paths': DIR, 'done': 'done'}

    @dataclass
    class VAR(DatapointTable.VAR):
        n_tabs_: int = 2

    @property
    def n_tabs(self):
        return self.var.n_tabs_


def block(cls, tmp_path, **kwargs):
    return cls(url=str(tmp_path), **kwargs)


# ---------------------------------------------------------------------------
# The declaration decides the era
# ---------------------------------------------------------------------------

class TestTheDeclarationDecidesTheEra:
    """Nothing announces the era: holding a marker is what makes one."""

    def test_a_declaration_holding_a_marker_quotes_its_filenames(self, tmp_path):
        assert block(Marked, tmp_path).signature_topics() == (
            "topic:data='data.txt'", 'topic:masks=DIR', 'topic:cache=SYNTHETIC',
        )

    def test_a_declaration_holding_none_renders_as_it_always_has(self, tmp_path):
        assert block(Sentinels, tmp_path).signature_topics() == (
            'topic:data=data.txt', 'topic:masks=None', 'topic:cache=()',
        )

    def test_a_filename_belongs_to_neither_spelling(self, tmp_path):
        """Only filenames means nothing to detect, so it renders the older way
        -- which is what keeps every block that has one addressable."""
        class Files(Datablock):
            TOPICS = {'data': 'data.txt', 'more': 'more.txt'}

            def __build__(self):
                pass

        assert block(Files, tmp_path).signature_topics() == (
            'topic:data=data.txt', 'topic:more=more.txt',
        )

    def test_a_marker_beside_a_sentinel_is_refused(self, tmp_path):
        class Mixed(Datablock):
            TOPICS = {'masks': DIR, 'cache': SYNTOPIC}

            def __build__(self):
                pass

        with pytest.raises(ValueError, match="mixes the topic markers"):
            block(Mixed, tmp_path).type()

    def test_the_error_names_both_sides(self, tmp_path):
        class Mixed(Datablock):
            TOPICS = {'masks': DIR, 'old': DIRTOPIC}

            def __build__(self):
                pass

        with pytest.raises(ValueError, match=r"\['masks'\].*\['old'\]"):
            block(Mixed, tmp_path).type()

    def test_a_nested_mixture_is_refused(self, tmp_path):
        class Mixed(Datablock):
            TOPICS = {'data': {'frames': DIR, 'masks': DIRTOPIC}}

            def __build__(self):
                pass

        with pytest.raises(ValueError, match=r"data/frames.*data/masks"):
            block(Mixed, tmp_path).path('data', 'frames')

    def test_the_two_slice_spellings_are_a_mixture_too(self, tmp_path):
        class Mixed(DatapointTab):
            TOPICS = {'a': DATASLICE(idx='int'), 'b': SLICETOPIC}

            def __build__(self):
                pass

        with pytest.raises(ValueError, match="mixes the topic markers"):
            block(Mixed, tmp_path).type()

    def test_an_inherited_sentinel_is_a_mixture(self, tmp_path):
        """A subclass adding a marker to a base's sentinels has to respell them,
        and re-keys by doing so. Which is the whole of what adopting them costs."""
        class Extended(Sentinels):
            TOPICS = {'extra': DIR, **Sentinels.TOPICS}

            def __build__(self):
                pass

        with pytest.raises(ValueError, match="mixes the topic markers"):
            block(Extended, tmp_path).type()


class TestMarkersRenderAsThemselves:

    def test_a_slice_renders_with_its_columns(self, tmp_path):
        assert "topic:numbers=DATASLICE(idx='int', square='int')" in block(MarkedTab, tmp_path).type()

    def test_a_bare_slice_renders_as_the_bare_marker(self, tmp_path):
        class Bare(DatapointTab):
            TOPICS = {'numbers': DATASLICE}

            def __build__(self):
                pass

        assert 'topic:numbers=DATASLICE' in block(Bare, tmp_path).type()

    def test_a_markers_own_arguments_keep_their_quotes(self, tmp_path):
        assert "DATASLICE(idx='int', square='int')" in block(MarkedTab, tmp_path).type()

    def test_the_markers_are_exported_from_the_package(self):
        assert (dbx.DIR, dbx.SYNTHETIC, dbx.DATASLICE) == (DIR, SYNTHETIC, DATASLICE)


class TestAMarkerBehavesAsTheSentinelItReplaces:

    def test_dir_is_a_directory_topic(self, tmp_path):
        b = block(Marked, tmp_path)
        assert b.path('masks') == b.dirpath('masks')

    def test_synthetic_has_no_location(self, tmp_path):
        assert block(Marked, tmp_path).path('cache') is None

    def test_a_synthetic_topic_cannot_hold_a_block_back(self, tmp_path):
        b = block(Marked, tmp_path)
        b.build()
        assert b.valid()

    def test_a_slice_is_a_directory_topic(self, tmp_path):
        t = block(MarkedTab, tmp_path)
        assert t.path('numbers') == t.dirpath('numbers')

    def test_a_built_slice_reads_back(self, tmp_path):
        t = block(MarkedTab, tmp_path)
        t.build()
        assert t.data('numbers') == {'numbers': {'idx': [0, 1, 2], 'square': [0, 1, 4]}}


@pytest.mark.pinned
class TestAMarkerIsNotItsName:
    """Within one declaration, a marker and a file named for it stay apart.

    ``{'masks': DIR, 'x': 'DIR'}`` is a directory topic beside a topic stored in
    a file called ``DIR``.  A leaf renders into the type string as its own text,
    so left bare the two would both render ``=DIR`` and one block's two topics
    would collide onto one segment while meaning different things.  Quoting the
    filename is what keeps them apart -- in the journal, where both are only ever
    text in a column, as much as in the hash.

    Across declarations they can still collide: a sentinel-spelled block renders
    its filenames bare, so its ``{'masks': 'DIR'}`` renders as a marker-spelled
    block's ``{'masks': DIR}`` does.  Closing that would mean re-rendering every
    block that predates the markers, which is the one thing the era exists to
    avoid.  Hence *within*: what one declaration says must be unambiguous.
    """

    def test_a_marker_and_a_file_named_for_it_render_apart(self, tmp_path):
        class Both(Datablock):
            TOPICS = {'masks': DIR, 'x': 'DIR'}

            def __build__(self):
                pass

        assert block(Both, tmp_path).signature_topics() == (
            'topic:masks=DIR', "topic:x='DIR'",
        )

    def test_a_parameterised_marker_and_the_string_of_its_call(self, tmp_path):
        class Both(DatapointTab):
            TOPICS = {'numbers': DATASLICE(idx='int'), 'x': "DATASLICE(idx='int')"}

            def __build__(self):
                pass

        marker, filename = block(Both, tmp_path).signature_topics()
        assert marker == "topic:numbers=DATASLICE(idx='int')"
        assert filename != marker.replace('numbers', 'x')

    def test_a_recorded_marker_reads_back_as_the_marker(self):
        assert literal_topics(str({'masks': DIR, 'cache': SYNTHETIC})) == {
            'masks': DIR, 'cache': SYNTHETIC,
        }

    def test_a_recorded_filename_reads_back_as_a_filename(self):
        read = literal_topics(str({'masks': 'DIR'}))
        assert read == {'masks': 'DIR'} and read['masks'] is not DIR

    def test_a_recorded_slice_keeps_its_columns(self):
        read = literal_topics(str({'numbers': DATASLICE(idx='int', square='int')}))
        assert read['numbers'].columns == {'idx': 'int', 'square': 'int'}

    def test_an_unknown_name_is_not_evaluated(self):
        with pytest.raises(ValueError, match="not a topic marker"):
            literal_topics("{'masks': shutil}")


@pytest.mark.pinned
class TestASlicesColumnsAreItsIdentity:
    """A slice's columns are its shape, and its shape belongs in its hash.

    Two slices that differ in what they hold must not address one directory:
    the second build would read the first's shards and find columns it never
    wrote.  Order counts with the rest -- MDS reads a row back in the order it
    was written, so a permutation is a different slice too.
    """

    def _tab(self, tmp_path, columns):
        class Tab(DatapointTab):
            TOPICS = {'numbers': DATASLICE(**columns)}

            def __build__(self):
                pass

        return block(Tab, tmp_path)

    def test_the_declared_columns_are_in_the_hash(self, tmp_path):
        a = self._tab(tmp_path, {'idx': 'int'})
        b = self._tab(tmp_path, {'idx': 'int', 'square': 'int'})
        assert a.hash != b.hash

    def test_retyping_a_column_re_keys(self, tmp_path):
        a = self._tab(tmp_path, {'idx': 'int'})
        b = self._tab(tmp_path, {'idx': 'str'})
        assert a.hash != b.hash

    def test_reordering_re_keys(self, tmp_path):
        a = self._tab(tmp_path, {'idx': 'int', 'square': 'int'})
        b = self._tab(tmp_path, {'square': 'int', 'idx': 'int'})
        assert a.hash != b.hash

    def test_the_same_columns_agree(self, tmp_path):
        """The guard: without it every assertion above could pass by everything
        differing from everything."""
        a = self._tab(tmp_path, {'idx': 'int', 'square': 'int'})
        b = self._tab(tmp_path, {'idx': 'int', 'square': 'int'})
        assert a.hash == b.hash


@pytest.mark.pinned
class TestTheDeclarationIsWhatGetsWritten:
    """What a block declares and what it writes may not differ.

    The declaration is in the hash.  A build that wrote other columns would
    leave the hash asserting a shape the data does not have -- which is worse
    than not declaring the columns at all, because it looks checked.
    """

    def test_columns_that_disagree_are_refused(self, tmp_path):
        class Disagreeing(DatapointTab):
            TOPICS = {'numbers': DATASLICE(idx='int')}

            def __build__(self):
                with self.slice_writers({'numbers': {'idx': 'str'}}):
                    pass

        with pytest.raises(ValueError, match="declared"):
            block(Disagreeing, tmp_path).build()

    def test_a_reordering_is_a_disagreement(self, tmp_path):
        class Reordered(DatapointTab):
            TOPICS = {'numbers': DATASLICE(idx='int', square='int')}

            def __build__(self):
                with self.slice_writers({'numbers': {'square': 'int', 'idx': 'int'}}):
                    pass

        with pytest.raises(ValueError, match="declared"):
            block(Reordered, tmp_path).build()

    def test_a_restatement_is_allowed(self, tmp_path):
        class Restated(DatapointTab):
            TOPICS = {'numbers': DATASLICE(idx='int')}

            def __build__(self):
                with self.slice_writers({'numbers': {'idx': 'int'}}) as writers:
                    writers['numbers'].write({'idx': 0})

        b = block(Restated, tmp_path)
        b.build()
        assert b.data('numbers') == {'numbers': {'idx': [0]}}


class TestSliceWritersTakesTheDeclaration:

    def test_no_argument_is_needed(self, tmp_path):
        t = block(MarkedTab, tmp_path)
        t.build()
        assert t.declared_columns('numbers') == {'idx': 'int', 'square': 'int'}
        assert t.data('numbers')['numbers']['square'] == [0, 1, 4]

    def test_a_bare_slice_still_needs_its_columns_passed(self, tmp_path):
        class Bare(DatapointTab):
            TOPICS = {'numbers': DATASLICE}

            def __build__(self):
                with self.slice_writers():
                    pass

        with pytest.raises(ValueError, match="no columns"):
            block(Bare, tmp_path).build()

    def test_the_sentinel_is_unaffected(self, tmp_path):
        class Sentinel(DatapointTab):
            TOPICS = {'numbers': SLICETOPIC}

            def __build__(self):
                with self.slice_writers({'numbers': {'idx': 'int'}}) as writers:
                    writers['numbers'].write({'idx': 0})

        b = block(Sentinel, tmp_path)
        b.build()
        assert b.declared_columns('numbers') is None
        assert b.data('numbers') == {'numbers': {'idx': [0]}}

    def test_a_missing_slice_is_still_refused(self, tmp_path):
        class TwoSlices(DatapointTab):
            TOPICS = {'numbers': SLICETOPIC, 'letters': SLICETOPIC}

            def __build__(self):
                with self.slice_writers({'numbers': {'idx': 'int'}}):
                    pass

        with pytest.raises(ValueError, match="letters"):
            block(TwoSlices, tmp_path).build()


class TestSliceColumnsAreChecked:

    def test_a_column_name_with_a_slash_is_refused(self):
        with pytest.raises(ValueError, match="may not contain"):
            DATASLICE(**{'a/b': 'int'})

    def test_a_column_type_with_a_slash_is_refused(self):
        with pytest.raises(ValueError, match="may not contain"):
            DATASLICE(idx='ndarray/int8')

    def test_a_column_type_must_be_a_string(self):
        with pytest.raises(TypeError, match="must be a string"):
            DATASLICE(idx=int)

    def test_a_name_that_is_not_an_identifier_uses_the_mapping_form(self):
        marker = DATASLICE({'my col': 'int'})
        assert marker.columns == {'my col': 'int'}
        assert str(marker) == "DATASLICE({'my col': 'int'})"
        assert literal_topics(str({'s': marker}))['s'].columns == {'my col': 'int'}

    def test_a_mapping_and_keywords_together_are_refused(self):
        with pytest.raises(TypeError, match="as keywords"):
            DATASLICE({'a': 'int'}, b='int')


# ---------------------------------------------------------------------------
# A table stops carrying the TAB's topics
# ---------------------------------------------------------------------------

class TestAMarkedTableCarriesOnlyItsOwnTopics:

    def test_the_tabs_slices_are_not_in_the_tables_signature(self, tmp_path):
        segments = block(MarkedTable, tmp_path).signature_topics()
        assert not any('numbers' in segment for segment in segments)

    def test_a_table_spelled_the_older_way_still_carries_them(self, tmp_path):
        class LegacyTab(DatapointTab):
            TOPICS = {'numbers': SLICETOPIC}

            def __build__(self):
                pass

        class LegacyTable(DatapointTable):
            TAB = LegacyTab

            @property
            def n_tabs(self):
                return 1

        assert 'topic:numbers=SLICETOPIC' in block(LegacyTable, tmp_path).signature_topics()

    def test_the_table_still_routes_the_tabs_slices(self, tmp_path):
        """Identity is what stops carrying them; routing never did."""
        assert block(MarkedTable, tmp_path).slices() == ('numbers',)

    def test_the_table_reads_its_tabs_slices(self, tmp_path):
        table = block(MarkedTable, tmp_path)
        table.build()
        assert table.data('numbers')['numbers']['idx'] == [0, 1, 2, 0, 1, 2]


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------

class TestTheJournalRecordsMarkers:

    def test_the_recorded_topics_are_markers(self, tmp_path):
        b = block(Marked, tmp_path)
        b.build()
        assert b.journal(loc=0).block.TOPICS == {
            'data': 'data.txt', 'masks': DIR, 'cache': SYNTHETIC,
        }

    def test_the_entry_answers_what_kind_each_topic_is(self, tmp_path):
        b = block(Marked, tmp_path)
        b.build()
        recorded = b.journal(loc=0).block
        assert recorded._is_dir_topic('masks') and not recorded._is_dir_topic('data')
        assert recorded._is_syntopic('cache') and not recorded._is_syntopic('masks')

    def test_a_recorded_slice_keeps_its_columns(self, tmp_path):
        t = block(MarkedTab, tmp_path)
        t.build()
        recorded = t.journal(loc=0).block.TOPICS
        assert recorded['numbers'].columns == {'idx': 'int', 'square': 'int'}

    def test_a_block_does_not_differ_from_its_own_entry(self, tmp_path):
        b = block(Marked, tmp_path)
        b.build()
        assert b.difftopics(str(b.TOPICS)) == {}

    def test_a_marker_differs_from_the_sentinel_it_replaces(self, tmp_path):
        assert block(Marked, tmp_path).difftopics(str(Sentinels.TOPICS)) == {
            'data': ("'data.txt'", 'data.txt'),
            'masks': ('DIR', 'None'),
            'cache': ('SYNTHETIC', '()'),
        }
