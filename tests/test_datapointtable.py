"""Tests for the :class:`~dbx.datapoints.DatapointTab` / :class:`~dbx.datapoints.DatapointTable`
scaffolding.

Covers TOPICS synthesis from SLICETOPIC entries, where a tab's shards land relative to
its table, the all-or-nothing validity rule, the merged per-slice index and its
ordering, and the two ways a built table reads back (``data`` and ``dataset``).
"""
import json
import os

os.environ.setdefault('DBX_DIRTY_REPO_OK', '1')

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

# torch and mosaicml-streaming are optional extras, and dbx.datastreams needs
# both at import time. importorskip skips this module when either is absent,
# rather than failing collection -- which takes the entire suite down before
# any test runs. Install them and these run as normal.
pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

from dbx.datapoints import (
    DIRTOPIC,
    SLICETOPIC,
    DatapointTab,
    DatapointTable,
    DatapointPartition,
)
from dbx.datastreams import (
    ZipIterableStreamingDatasets,
    ZipStreamingDataset,
)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


# ---------------------------------------------------------------------------
# A minimal tab/table pair
# ---------------------------------------------------------------------------

class LetterTab(DatapointTab):
    """Writes ``n`` items into two lockstep slices, plus a non-slice topic."""

    VERSION = 1
    TOPICS = {'numbers': SLICETOPIC, 'letters': SLICETOPIC, 'note': 'note.txt'}

    @dataclass
    class VAR(DatapointTab.VAR):
        n: int = 3
        base: int = 0
        fail: bool = False

    COLUMNS = {
        'numbers': {'idx': 'int', 'square': 'int'},
        'letters': {'idx': 'int', 'label': 'str'},
    }

    def __build__(self, stage=None):
        with self.slice_writers(self.COLUMNS, stage=stage) as writers:
            for i in range(self.var.n):
                k = self.var.base + i
                writers['numbers'].write({'idx': k, 'square': k * k})
                writers['letters'].write({'idx': k, 'label': f"lbl{k}"})
                if self.var.fail and i == self.var.n - 1:
                    raise RuntimeError("boom")
        with self.fs.open(self.path('note', ensure_dirpath=True), 'w') as f:
            f.write(f"tab note")

    def __stats__(self, slice_name):
        return {'n_samples': len(self.data(slice_name))}


class LetterTable(DatapointTable):
    VERSION = 1
    TAB = LetterTab

    @dataclass
    class VAR(DatapointTable.VAR):
        n_tabs_: int = 3
        per_tab: int = 3
        fail: bool = False

    @property
    def n_tabs(self):
        return self.var.n_tabs_

    def __tab__(self, idx):
        return super().__tab__(idx, n=self.var.per_tab,
                               base=idx * self.var.per_tab, fail=self.var.fail)

    def __stats__(self, slice_name):
        per_tab = [self.tab(i).stats(slice_name) for i in range(self.n_tabs)]
        return {'n_samples': sum(s['n_samples'] for s in per_tab)}


@pytest.fixture
def table(tmp_path):
    return LetterTable(url=str(tmp_path), spec=dict(n_tabs_=3, per_tab=3))


@pytest.fixture
def built_table(table):
    table.build()
    return table


# ---------------------------------------------------------------------------
# TOPICS synthesis
# ---------------------------------------------------------------------------

class TestTopicsFromSlices:

    def test_tab_data_group_is_synthesized(self):
        assert LetterTab.TOPICS['numbers'] == SLICETOPIC
        assert LetterTab.TOPICS['letters'] == SLICETOPIC

    def test_declared_topics_are_kept(self):
        assert LetterTab.TOPICS['note'] == 'note.txt'

    def test_table_slices_equal_tab_slices(self):
        """Table slices come from TAB, not from table's own TOPICS."""
        assert LetterTable.slices == LetterTab.slices
        assert 'numbers' not in LetterTable.TOPICS

    def test_table_keeps_its_inherited_topics(self):
        assert LetterTable.TOPICS['tabs'] is DIRTOPIC
        assert LetterTable.TOPICS['done'] == 'done'

    def test_docstring_example_subclass_extends_topics_and_slices(self):
        """The DatapointTab docstring's DebuggableFrameTab example."""
        class Debuggable(LetterTab):
            TOPICS = {'debug': {'plots': DIRTOPIC}, 'depth': SLICETOPIC}

        assert Debuggable.TOPICS['debug'] == {'plots': DIRTOPIC}
        assert 'depth' in Debuggable.slices

    def test_topics_do_not_accumulate_down_the_hierarchy(self):
        """A subclass declaring TOPICS does not accumulate parent TOPICS unless explicitly specified."""
        class BaseTab(DatapointTab):
            TOPICS = {'samples': SLICETOPIC, 'meta': 'meta.json'}

        class SubTab(BaseTab):
            TOPICS = {'report': 'report.json'}

        class ExplicitSubTab(BaseTab):
            TOPICS = {'report': 'report.json', **BaseTab.TOPICS}

        assert SubTab.TOPICS == {'report': 'report.json'}
        assert ExplicitSubTab.TOPICS == {'report': 'report.json', 'samples': SLICETOPIC, 'meta': 'meta.json'}

    def test_slice_topics_are_not_in_table_topics(self):
        """Slice topics belong to the tab; they must not appear in the table's TOPICS."""
        assert 'numbers' not in LetterTable.TOPICS
        assert 'letters' not in LetterTable.TOPICS

    def test_slices_are_in_the_signature(self, table):
        assert 'topic:numbers=SLICETOPIC' in table.type()
        assert 'topic:letters=SLICETOPIC' in table.type()

    def test_renaming_a_slice_rekeys_the_table(self, tmp_path):
        class GlyphTab(LetterTab):
            TOPICS = {'numbers': SLICETOPIC, 'glyphs': SLICETOPIC}

        class GlyphTable(LetterTable):
            TAB = GlyphTab

        assert 'glyphs' in GlyphTable.slices
        assert 'glyphs' not in GlyphTable.TOPICS
        a = LetterTable(url=str(tmp_path), spec=dict(n_tabs_=3, per_tab=3))
        b = GlyphTable(url=str(tmp_path), spec=dict(n_tabs_=3, per_tab=3))
        assert a.hash != b.hash

    def test_unknown_slice_is_rejected(self, table):
        with pytest.raises(KeyError, match='unknown slice'):
            table.dataset('nope')
            table.dataset('nope')


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

class TestPlacement:

    def test_tab_slices_live_under_tab_anchorkeypath(self, table):
        tab = table.tab(0)
        assert tab.path('numbers').startswith(tab.anchorkeypath)

    def test_non_slice_topics_stay_under_the_tabs_own_key(self, table):
        tab = table.tab(0)
        assert tab.path('note').startswith(tab.anchorkeypath)

    def test_a_tab_without_a_table_can_address_its_slices_and_topics(self, tmp_path):
        lone = LetterTab(url=str(tmp_path), spec=dict(n=2))
        assert lone.path('note') == os.path.join(
            lone.anchorkeypath, 'note', 'note.txt',
        )
        assert lone.path('numbers').startswith(lone.anchorkeypath)


# ---------------------------------------------------------------------------
# Building and validity
# ---------------------------------------------------------------------------

class TestBuild:

    def test_table_is_invalid_before_build(self, table):
        assert table.valid() is False

    def test_table_is_valid_after_build(self, built_table):
        assert built_table.valid() is True

    def test_every_tab_is_valid(self, built_table):
        assert all(built_table.tab(i).valid() for i in range(built_table.n_tabs))

    def test_rebuild_is_a_no_op(self, built_table):
        before = json.loads(open(built_table.tab(0).slice_index_path('numbers')).read())
        built_table.build()
        after = json.loads(open(built_table.tab(0).slice_index_path('numbers')).read())
        assert before == after

    def test_staged_build_lands_the_same_data(self, tmp_path):
        """``stage=True`` is the remote-storage path; on local storage it must
        produce the same results as writing in place."""
        direct = LetterTable(url=str(tmp_path / 'direct'), spec=dict(n_tabs_=1))
        staged = LetterTable(url=str(tmp_path / 'staged'), spec=dict(n_tabs_=1))
        direct.tab(0).build(stage=False)
        staged.tab(0).build(stage=True)
        assert staged.tab(0).valid()
        assert staged.tab(0).data('letters') == direct.tab(0).data('letters')

    def test_staged_upload_sends_the_index_last(self, tmp_path, monkeypatch):
        """valid_slice() reads a non-empty index.json as "built", so an upload
        interrupted partway must not have landed the index before the shards
        it names."""
        tab = LetterTab(url=str(tmp_path), spec=dict(n=3, base=0))

        uploaded = []
        original = type(tab.fs).put_file

        def recording_put_file(self, src, dest, **kwargs):
            uploaded.append(os.path.basename(dest))
            return original(self, src, dest, **kwargs)

        monkeypatch.setattr(type(tab.fs), 'put_file', recording_put_file)
        tab.build(stage=True)

        assert uploaded.count('index.json') == len(tab.slices)
        runs, current = [], []
        for name in uploaded:
            current.append(name)
            if name == 'index.json':
                runs.append(current)
                current = []
        assert not current, f"files uploaded after the last index: {current}"
        for run in runs:                       # one run per slice
            assert run[-1] == 'index.json'
            assert len(run) > 1, "a slice uploaded its index and no shards"

    def test_a_failed_build_leaves_the_tab_invalid(self, tmp_path):
        broken = LetterTable(url=str(tmp_path), spec=dict(n_tabs_=1, fail=True))
        tab = broken.tab(0)
        with pytest.raises(RuntimeError, match='boom'):
            tab.build()
        assert tab.valid() is False
        assert tab.validtopic('numbers') is False

    def test_an_empty_slice_is_valid(self, tmp_path):
        """``MDSWriter.finish()`` writes an index.json even when nothing was
        written through it; presence of index.json counts as built and valid."""
        empty = LetterTable(url=str(tmp_path), spec=dict(n_tabs_=1, per_tab=0))
        tab = empty.tab(0)
        tab.build()
        assert os.path.exists(tab.slice_index_path('numbers'))


# ---------------------------------------------------------------------------
# The merged per-slice index
# ---------------------------------------------------------------------------

class TestTabIndexes:

    def test_one_index_per_tab_and_slice(self, built_table):
        for idx in range(built_table.n_tabs):
            tab = built_table.tab(idx)
            for name in built_table.slices:
                assert os.path.exists(tab.slice_index_path(name))


# ---------------------------------------------------------------------------
# Reading: data / dataset / stats
# ---------------------------------------------------------------------------

class TestRead:

    def test_tab_data_returns_its_own_samples(self, built_table):
        assert [s['label'] for s in built_table.tab(1).data('letters')] == \
            ['lbl3', 'lbl4', 'lbl5']

    def test_table_data_concatenates_the_tabs(self, built_table):
        assert [s['idx'] for s in built_table.data('numbers')] == list(range(9))

    def test_data_with_no_slice_returns_every_slice(self, built_table):
        data = built_table.data()
        assert set(data) == set(built_table.slices)
        assert len(data['numbers']) == 9

    def test_read_addresses_a_slice(self, built_table):
        assert built_table.read('letters') == built_table.data('letters')

    def test_dataset_zips_every_slice(self, built_table):
        ds = built_table.dataset()
        assert len(ds) == 9
        assert set(ds[0]) == {'idx', 'square', 'label'}

    def test_dataset_is_index_aligned(self, built_table):
        ds = built_table.dataset()
        for i in range(len(ds)):
            sample = ds[i]
            assert sample['square'] == sample['idx'] ** 2
            assert sample['label'] == f"lbl{sample['idx']}"

    def test_dataset_opens_only_the_named_slices(self, built_table):
        ds = built_table.dataset('letters')
        assert set(ds[0]) == {'idx', 'label'}

    def test_tab_dataset_covers_only_that_tab(self, built_table):
        ds = built_table.tab(2).dataset()
        assert len(ds) == 3
        assert [ds[i]['idx'] for i in range(3)] == [6, 7, 8]

    def test_stats_per_slice(self, built_table):
        assert built_table.stats('numbers') == {'n_samples': 9}
        assert built_table.tab(0).stats('letters') == {'n_samples': 3}

    def test_stats_with_no_slice_returns_every_slice(self, built_table):
        assert built_table.stats() == {'numbers': {'n_samples': 9},
                                      'letters': {'n_samples': 9}}


# ---------------------------------------------------------------------------
# dataset(mode=...)
# ---------------------------------------------------------------------------

class TestDatasetMode:

    def test_map_mode_is_the_default(self, built_table):
        assert isinstance(built_table.dataset(), ZipStreamingDataset)

    def test_iter_mode_gives_the_iterable_zip(self, built_table):
        ds = built_table.dataset(mode='iter', batch_size=3)
        assert isinstance(ds, ZipIterableStreamingDatasets)

    def test_iter_mode_yields_what_map_mode_indexes(self, built_table):
        mapped = built_table.dataset()
        expected = [mapped[i] for i in range(len(mapped))]
        assert list(built_table.dataset(mode='iter', batch_size=3)) == expected

    def test_iter_mode_respects_projection(self, built_table):
        ds = built_table.dataset(mode='iter', batch_size=3,
                                 columns={'numbers': ['idx']})
        assert set(next(iter(ds))) == {'idx', 'label'}

    def test_iter_mode_demands_a_batch_size(self, built_table):
        """StreamingDataset only complains on the first batch, from inside a
        worker; dataset() asks where the mode was chosen."""
        with pytest.raises(ValueError, match='batch_size'):
            built_table.dataset(mode='iter')

    def test_map_mode_needs_no_batch_size(self, built_table):
        assert len(built_table.dataset()) == 9

    def test_unknown_mode_is_rejected(self, built_table):
        with pytest.raises(ValueError, match="'map' or 'iter'"):
            built_table.dataset(mode='streaming')


# ---------------------------------------------------------------------------
# flush_every: shard boundaries that coincide across slices
# ---------------------------------------------------------------------------

class SizedTab(DatapointTab):
    """Two slices whose samples differ wildly in size, so that a byte-driven
    shard boundary in one lands nowhere near the other's."""

    VERSION = 1
    TOPICS = {'small': SLICETOPIC, 'big': SLICETOPIC}

    @dataclass
    class VAR(DatapointTab.VAR):
        n: int = 24
        width: int = 400
        flush_every: int = None
        size_limit: int = None
        skew: bool = False

    COLUMNS = {'small': {'idx': 'int'},
               'big': {'idx': 'int', 'payload': 'str'}}

    def __build__(self):
        with self.slice_writers(self.COLUMNS,
                                flush_every=self.var.flush_every,
                                size_limit=self.var.size_limit) as writers:
            for i in range(self.var.n):
                writers['small'].write({'idx': i})
                writers['big'].write({'idx': i, 'payload': 'x' * self.var.width})
            if self.var.skew:
                writers['big'].write({'idx': -1, 'payload': 'extra'})


class SizedTable(DatapointTable):
    VERSION = 1
    TAB = SizedTab

    @dataclass
    class VAR(DatapointTable.VAR):
        n: int = 24
        width: int = 400
        flush_every: int = None
        size_limit: int = None
        skew: bool = False

    @property
    def n_tabs(self):
        return 1

    def __tab__(self, idx):
        return super().__tab__(
            idx, n=self.var.n, width=self.var.width,
            flush_every=self.var.flush_every, size_limit=self.var.size_limit,
            skew=self.var.skew,
        )


def sized(tmp_path, **spec):
    table = SizedTable(url=str(tmp_path), spec=spec)
    table.build()
    return table


class TestFlushEvery:

    def test_byte_budget_alone_shards_the_slices_differently(self, tmp_path):
        """The problem flush_every exists to solve: sized by bytes, a slice
        of 4-byte ints and a slice of 400-byte strings break apart at
        completely different sample counts."""
        table = sized(tmp_path, size_limit=4096)
        assert table.shard_sizes('small') != table.shard_sizes('big')

    def test_flush_every_makes_the_boundaries_coincide(self, tmp_path):
        table = sized(tmp_path, flush_every=8)
        assert table.shard_sizes('small') == table.shard_sizes('big') == [8, 8, 8]

    def test_trailing_partial_shard_is_kept(self, tmp_path):
        """A count that is not a multiple of flush_every leaves a short last
        shard -- in both slices, at the same place."""
        table = sized(tmp_path, n=20, flush_every=8)
        assert table.shard_sizes('small') == table.shard_sizes('big') == [8, 8, 4]

    def test_exact_multiple_leaves_no_empty_shard(self, tmp_path):
        """finish() flushes only what is pending, so a sample count that is
        an exact multiple must not append a zero-sample shard."""
        table = sized(tmp_path, n=16, flush_every=8)
        assert table.shard_sizes('big') == [8, 8]

    def test_the_data_still_reads_back_whole(self, tmp_path):
        table = sized(tmp_path, flush_every=8)
        assert [s['idx'] for s in table.data('small')] == list(range(24))
        assert len(table.dataset()) == 24

    def test_size_limit_firing_first_is_an_error(self, tmp_path):
        """flush_every does not override size_limit -- whichever comes first
        ends the shard -- so a size_limit that fires first would silently
        undo the alignment."""
        with pytest.raises(ValueError, match='size_limit'):
            sized(tmp_path, flush_every=64, size_limit=4096)

    def test_slices_written_out_of_lockstep_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='lockstep'):
            sized(tmp_path, flush_every=8, skew=True)

    def test_non_positive_flush_every_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='flush_every must be positive'):
            sized(tmp_path, flush_every=0)

    def test_aligned_slices_may_be_zipped_with_shuffle(self, tmp_path):
        """The point of the whole exercise: identically-sharded slices pass
        the iterator-mode alignment check with shuffling on."""
        table = sized(tmp_path, flush_every=8)
        ds = table.dataset(mode='iter', batch_size=4, shuffle=True,
                           shuffle_seed=17)
        assert isinstance(ds, ZipIterableStreamingDatasets)

    def test_a_shuffled_zip_stays_aligned(self, tmp_path):
        """Shuffled and still paired: 'big' carries idx too, so a mispairing
        shows up as the two slices' idx disagreeing."""
        table = sized(tmp_path, flush_every=8)
        ds = table.dataset(mode='iter', batch_size=4, shuffle=True,
                           shuffle_seed=17, columns={'small': ['idx'],
                                                     'big': ['idx']},
                           shared={'idx'}, validate_shared=True)
        seen = [s['idx'] for s in ds]
        assert sorted(seen) == list(range(24))
        assert seen != list(range(24)), "shuffle=True did not shuffle"

    def test_differently_sharded_slices_are_refused_a_shuffled_zip(self, tmp_path):
        table = sized(tmp_path, size_limit=4096)
        with pytest.raises(ValueError, match='shard boundaries'):
            table.dataset(mode='iter', batch_size=4, shuffle=True,
                          shuffle_seed=17)


# ---------------------------------------------------------------------------
# Unimplemented hooks name themselves
# ---------------------------------------------------------------------------

class TestScaffoldingErrors:

    LONE = dict()

    def test_build_names_the_slices_to_write(self, tmp_path):
        class Bare(DatapointTab):
            TOPICS = {'a': SLICETOPIC, 'b': SLICETOPIC}

        # __build__ directly, not build(): build() journals first, and
        # journaling resolves every topic path.
        # The hook's own message is what is under test.
        with pytest.raises(NotImplementedError, match="'a', 'b'"):
            Bare(url=str(tmp_path), spec=dict(self.LONE)).__build__()

    def test_missing_tab_class(self, tmp_path):
        class NoTab(DatapointTable):
            TOPICS = {'a': SLICETOPIC, **DatapointTable.TOPICS}

            @property
            def n_tabs(self):
                return 1

        with pytest.raises(NotImplementedError, match='TAB'):
            NoTab(url=str(tmp_path)).build()

    def test_missing_n_tabs(self, tmp_path):
        class NoCount(DatapointTable):
            TAB = LetterTab

        with pytest.raises(NotImplementedError, match='n_tabs'):
            NoCount(url=str(tmp_path)).build()

    def test_slice_writers_requires_every_slice(self, tmp_path):
        class Partial(DatapointTab):
            TOPICS = {'a': SLICETOPIC, 'b': SLICETOPIC}

            def __build__(self):
                with self.slice_writers({'a': {'x': 'int'}}):
                    pass

        with pytest.raises(ValueError, match=r"slice\(s\) \['b'\]"):
            Partial(url=str(tmp_path), spec=dict(self.LONE)).__build__()


# ---------------------------------------------------------------------------
# Cache root
# ---------------------------------------------------------------------------

class TestCache:

    def test_cacheroot_defaults_under_local(self, table):
        assert table.cacheroot == os.path.join(table.localroot, 'streaming')

    def test_cacheroot_honours_an_explicit_cache(self, tmp_path):
        explicit = LetterTable(url=str(tmp_path), spec=dict(n_tabs_=1),
                               cache=str(tmp_path / 'scratch'))
        assert explicit.cacheroot == str(tmp_path / 'scratch')

    def test_tabs_inherit_the_tables_cache(self, tmp_path):
        explicit = LetterTable(url=str(tmp_path), spec=dict(n_tabs_=1),
                               cache=str(tmp_path / 'scratch'))
        assert explicit.tab(0).cacheroot == str(tmp_path / 'scratch')

    def test_an_unset_cache_stays_unset_in_the_handle(self, table):
        """cacheroot resolves lazily, so no machine's absolute path is baked
        into the block's handle or journal."""
        assert getattr(table, 'cache', None) is None
        assert 'streaming' not in table.quote()


# ---------------------------------------------------------------------------
# Placement is the tab's own key
# ---------------------------------------------------------------------------

class TestTabKey:

    def test_slice_dir_and_own_topics_share_the_key(self, table):
        tab = table.tab(0)
        assert tab.path('numbers').startswith(tab.anchorkeypath)
        assert tab.anchorkeypath.endswith(tab.key)


# ---------------------------------------------------------------------------
# datastream(): one slice, unzipped
# ---------------------------------------------------------------------------

class TestDatastream:

    def test_datastream_returns_one_unzipped_slice(self, built_table):
        ds = built_table.datastream('letters')
        assert len(ds) == 9
        assert set(ds[0]) >= {'idx', 'label'}
        assert 'square' not in ds[0]

    def test_dataset_is_the_zip_of_datastreams(self, built_table):
        zipped = built_table.dataset()
        assert len(zipped.datasets) == len(built_table.slices)
        assert len(zipped) == len(built_table.datastream('numbers'))

    def test_datastream_rejects_an_unknown_slice(self, built_table):
        with pytest.raises(KeyError, match='unknown slice'):
            built_table.datastream('nope')


# ---------------------------------------------------------------------------
# Zip policy: projection, shared keys, conflicts
# ---------------------------------------------------------------------------

class TestZipPolicy:

    def test_columns_project_a_slice(self, built_table):
        ds = built_table.dataset(columns=[('numbers', 'square')])
        assert set(ds[0]) == {'square', 'idx', 'label'}

    def test_columns_reject_an_unopened_slice(self, built_table):
        with pytest.raises(KeyError, match='not among'):
            built_table.dataset('letters', columns=[('numbers', 'square')])

    def test_shared_keys_are_validated(self, built_table):
        """'idx' is written to both slices in lockstep, so it must agree."""
        ds = built_table.dataset(shared={'idx'}, validate_shared=True,
                                 on_conflict='error')
        assert [ds[i]['idx'] for i in range(len(ds))] == list(range(9))

    def test_an_unshared_collision_can_be_an_error(self, built_table):
        with pytest.raises(KeyError, match="supplied by both source"):
            built_table.dataset(on_conflict='error')[0]

    def test_default_merge_is_unchanged(self, built_table):
        """No policy arguments: a plain last-wins merge, as before."""
        ds = built_table.dataset()
        assert ds[0]['idx'] == 0 and ds[0]['square'] == 0


# ---------------------------------------------------------------------------
# Topics besides the slices
# ---------------------------------------------------------------------------

class NestedTab(LetterTab):
    """A tab with a file topic and a nested topic group besides its slices."""

    TOPICS = {'note': 'note.txt', 'debug': {'plots': DIRTOPIC, 'log': 'run.log'}, **LetterTab.TOPICS}


class NestedTable(LetterTable):
    TAB = NestedTab


class TestExtraTopics:
    """Anything in TOPICS that is not a slice is an ordinary Datablock topic."""

    def _table(self, tmp_path):
        return NestedTable(url=str(tmp_path), spec=dict(n_tabs_=1))

    def test_extra_topics_merge_with_the_synthesized_data_group(self):
        assert NestedTab.TOPICS == {
            'numbers': SLICETOPIC,
            'letters': SLICETOPIC,
            'note': 'note.txt',
            'debug': {'plots': DIRTOPIC, 'log': 'run.log'},
        }

    def test_nested_extra_groups_are_addressable(self, tmp_path):
        tab = self._table(tmp_path).tab(0)
        assert tab.path('debug', 'log') == os.path.join(
            tab.anchorkeypath, 'debug', 'log', 'run.log',
        )
        assert set(tab.path('debug')) == {'plots', 'log'}

    def test_extra_topics_stay_under_the_tabs_own_key(self, tmp_path):
        table = self._table(tmp_path)
        tab = table.tab(0)
        assert tab.path('debug', 'log').startswith(tab.anchorkeypath)
        assert tab.path('numbers').startswith(tab.anchorkeypath)

    def test_extra_topics_are_in_the_signature(self, tmp_path):
        tab = self._table(tmp_path).tab(0)
        assert 'topic:debug/log=run.log' in tab.type()
        assert ('debug', 'log') in tab.leaftopics()

    def test_extra_topics_count_towards_validity(self, tmp_path):
        """A tab whose slices landed but whose extra topic did not is unbuilt."""
        table = self._table(tmp_path)
        tab = table.tab(0)
        tab.build()                              # LetterTab.__build__ writes 'note'
        assert tab.validtopic('numbers') is True
        assert tab.validtopic('debug') is False  # nothing writes it
        assert tab.valid() is False

    def test_read_defers_extra_topics_to_the_subclass(self, tmp_path):
        tab = self._table(tmp_path).tab(0)
        with pytest.raises(NotImplementedError, match='note'):
            tab.read('note')


# ---------------------------------------------------------------------------
# Block-shuffled sampling
# ---------------------------------------------------------------------------

class TestBlockShuffleSampler:

    def test_covers_every_index_exactly_once(self):
        from dbx.datastreams import BlockShuffleSampler
        s = BlockShuffleSampler(n=100, block_size=8, seed=1)
        assert sorted(s) == list(range(100))
        assert len(s) == 100

    def test_working_set_stays_within_a_block(self):
        """The whole point: consecutive draws stay near each other, so a
        shard cache can help.  A global shuffle would not."""
        from dbx.datastreams import BlockShuffleSampler
        block = 16
        order = list(BlockShuffleSampler(n=320, block_size=block, seed=7))
        for start in range(0, len(order), block):
            chunk = order[start:start + block]
            assert max(chunk) - min(chunk) < block, chunk

    def test_order_is_not_the_identity(self):
        from dbx.datastreams import BlockShuffleSampler
        order = list(BlockShuffleSampler(n=256, block_size=8, seed=3))
        assert order != list(range(256))

    def test_reproducible_for_a_seed(self):
        from dbx.datastreams import BlockShuffleSampler
        a = list(BlockShuffleSampler(n=64, block_size=8, seed=5))
        b = list(BlockShuffleSampler(n=64, block_size=8, seed=5))
        c = list(BlockShuffleSampler(n=64, block_size=8, seed=6))
        assert a == b and a != c

    def test_set_epoch_reshuffles(self):
        from dbx.datastreams import BlockShuffleSampler
        s = BlockShuffleSampler(n=64, block_size=8, seed=5)
        first = list(s)
        s.set_epoch(1)
        assert list(s) != first

    def test_fixed_epoch_ignores_set_epoch(self):
        """A validation sampler must score the same subset every epoch."""
        from dbx.datastreams import BlockShuffleSampler
        s = BlockShuffleSampler(n=64, block_size=8, seed=5, fixed_epoch=True)
        first = list(s)
        s.set_epoch(3)
        assert list(s) == first

    def test_ragged_tail_block(self):
        from dbx.datastreams import BlockShuffleSampler
        assert sorted(BlockShuffleSampler(n=10, block_size=4, seed=0)) == list(range(10))

    def test_resume_state_skips_what_was_consumed(self):
        from dbx.datastreams import BlockShuffleSampler
        s = BlockShuffleSampler(n=64, block_size=8, seed=2)
        it = iter(s)
        consumed = [next(it) for _ in range(20)]
        state = s.state_dict()
        assert state['consumed'] == 20

        resumed = BlockShuffleSampler(n=64, block_size=8, seed=2)
        resumed.load_state_dict(state)
        rest = list(resumed)
        assert len(rest) == 44
        assert consumed + rest == list(BlockShuffleSampler(n=64, block_size=8, seed=2))

    def test_rejects_bad_block_size(self):
        from dbx.datastreams import BlockShuffleSampler
        with pytest.raises(ValueError, match='block_size'):
            BlockShuffleSampler(n=10, block_size=0)


class TestTableSampler:

    def test_shard_sizes_come_from_the_merged_index(self, built_table):
        sizes = built_table.shard_sizes('numbers')
        assert sizes and sum(sizes) == len(built_table.datastream('numbers'))

    def test_n_rows_matches_the_datastream(self, built_table):
        assert built_table.n_rows('numbers') == len(built_table.datastream('numbers'))

    def test_every_slice_agrees_on_length(self, built_table):
        """The lockstep contract, read off the indexes without opening them."""
        assert len({built_table.n_rows(s) for s in built_table.slices}) == 1
        assert len(built_table.verify_slice_row_counts_match()) == len(built_table.slices)

    def test_sampler_defaults_chunk_size_to_the_shard_capacity(self, built_table):
        sampler = built_table.chunk_shuffle_sampler('numbers')
        assert sampler.chunk_size == built_table.max_rows_per_shard('numbers')
        assert sampler.chunk_size == max(built_table.shard_sizes('numbers'))
        assert len(sampler) == built_table.n_rows('numbers')

    def test_sampler_covers_the_whole_table(self, built_table):
        assert sorted(built_table.chunk_shuffle_sampler('numbers')) == list(range(built_table.n_rows('numbers')))

    def test_sampler_honours_an_explicit_chunk_size(self, built_table):
        assert built_table.chunk_shuffle_sampler('numbers', chunk_size=2).chunk_size == 2
        assert built_table.block_shuffle_sampler('numbers', block_size=2).block_size == 2

    def test_sampler_drives_a_dataloader_over_the_zip(self, built_table):
        from torch.utils.data import DataLoader
        ds = built_table.dataset()
        loader = DataLoader(ds, sampler=built_table.chunk_shuffle_sampler('numbers', seed=1), batch_size=2)
        seen = [int(i) for batch in loader for i in batch['idx']]
        assert sorted(seen) == list(range(len(ds)))


# ---------------------------------------------------------------------------
# Valid Tab Fast Path and Sentinel Files
# ---------------------------------------------------------------------------

class TestValidTabAndSentinels:

    def test_sentinels_written_on_build(self, tmp_path):
        tbl = LetterTable(url=str(tmp_path / "sentinel_test"), spec=dict(n_tabs_=3))
        for i in range(3):
            assert not tbl.valid_tab(i)
            assert not tbl._check_tab_path(i)
        
        tbl.build()
        
        for i in range(3):
            assert tbl._check_tab_path(i)
            assert tbl.valid_tab(i)
            sentinel_path = os.path.join(tbl.path('tab_paths'), f"tab_{i}.path")
            assert os.path.exists(sentinel_path)
            with open(sentinel_path) as f:
                assert f.read().strip() == tbl.tab(i).anchorkeypath

    def test_fallback_when_tab_paths_topic_missing(self, tmp_path):
        class NoTabPathsTable(LetterTable):
            TOPICS = {'tabs': DIRTOPIC, 'done': 'done'}

        tbl = NoTabPathsTable(url=str(tmp_path / "nobuilt_test"), spec=dict(n_tabs_=2))
        assert 'tab_paths' not in tbl.topics()
        assert not tbl.valid_tab(0)
        
        tbl.build()
        assert tbl.valid_tab(0)
        assert tbl.valid_tab(1)

    def test_find_tabs(self, tmp_path):
        tbl = LetterTable(
            url=str(tmp_path / "find_table"),
            spec=dict(n_tabs_=4),
        )
        assert tbl.find_tabs("base=0") == [0]
        assert tbl.find_blocks("base=3") == [1]
        assert tbl.find_tabs(["base=6"]) == [2]
        assert tbl.find_tabs(["LetterTab", "base=9"]) == [3]

        # Match multiple
        assert tbl.find_tabs("LetterTab") == [0, 1, 2, 3]
        assert tbl.find_blocks("LetterTab") == [0, 1, 2, 3]

        # No match
        assert tbl.find_tabs("nonexistent") == []

    def test_parallel_filtering_skips_already_valid_tabs(self, tmp_path):
        tbl = LetterTable(
            url=str(tmp_path / "filter_test"),
            spec=dict(n_tabs_=4),
            parallelization="multithreading",
            n_workers=2,
        )
        # Manually build tab 0 and tab 2
        tbl.tab(0).build()
        tbl._write_tab_path(0)
        tbl.tab(2).build()
        tbl._write_tab_path(2)

        assert tbl.valid_tab(0)
        assert not tbl.valid_tab(1)
        assert tbl.valid_tab(2)
        assert not tbl.valid_tab(3)

        # Build whole table — should filter out 0 and 2, only building 1 and 3
        tbl.build()
        for i in range(4):
            assert tbl.valid_tab(i)
            assert tbl._check_tab_path(i)

    def test_filter_built_tabs_option(self, tmp_path):
        tbl_default = LetterTable(url=str(tmp_path / "filter_default"), spec=dict(n_tabs_=2))
        assert tbl_default.filter_built_tabs is False

        tbl_explicit = LetterTable(url=str(tmp_path / "filter_explicit"), spec=dict(n_tabs_=2), filter_built_tabs=True)
        assert tbl_explicit.filter_built_tabs is True

    def test_valid_tabs_and_valid_blocks(self, tmp_path):
        tbl = LetterTable(
            url=str(tmp_path / "valid_tabs_test"),
            spec=dict(n_tabs_=4),
            parallelization="multithreading",
            n_workers=2,
        )
        assert isinstance(tbl.valid_tabs(), pd.Series)
        assert isinstance(tbl.valid_blocks(), pd.Series)
        assert tbl.valid_tabs().tolist() == [False, False, False, False]
        assert tbl.valid_blocks().tolist() == [False, False, False, False]

        # Test parallelization override
        assert tbl.valid_tabs(parallelization="inline").tolist() == [False, False, False, False]
        assert tbl.valid_tabs(n_workers=1).tolist() == [False, False, False, False]

        tbl.tab(0).build()
        tbl._write_tab_path(0)
        tbl.tab(2).build()
        tbl._write_tab_path(2)

        assert tbl.valid_tabs().tolist() == [True, False, True, False]
        assert tbl.valid_blocks().tolist() == [True, False, True, False]
        assert tbl.valid_tabs(parallelization="inline").tolist() == [True, False, True, False]

        # Test false_only and true_only
        assert tbl.valid_tabs(false_only=True).index.tolist() == [1, 3]
        assert tbl.valid_tabs(true_only=True).index.tolist() == [0, 2]
        with pytest.raises(ValueError, match="mutually exclusive"):
            tbl.valid_tabs(false_only=True, true_only=True)

        assert tbl.valid_tab(0) is True
        assert tbl.valid_tab(1) is False
        assert tbl.valid_block(0) is True
        assert tbl.valid_block(1) is False

        tbl.build()
        assert tbl.valid_tabs().tolist() == [True, True, True, True]
        assert tbl.valid_blocks().tolist() == [True, True, True, True]
        assert tbl.valid_tabs(false_only=True).empty
        assert tbl.valid_tabs(true_only=True).index.tolist() == [0, 1, 2, 3]

    def test_redirected_tabs_and_redirected_blocks(self, tmp_path):
        src_tbl = LetterTable(
            url=str(tmp_path / "red_src"),
            spec=dict(n_tabs_=3),
        ).build()
        dst_tbl = LetterTable(
            url=str(tmp_path / "red_dst"),
            spec=dict(n_tabs_=3),
            parallelization="multithreading",
            n_workers=2,
        )
        assert isinstance(dst_tbl.redirected_tabs(), pd.Series)
        assert isinstance(dst_tbl.redirected_blocks(), pd.Series)
        assert dst_tbl.redirected_tabs().tolist() == [False, False, False]
        assert dst_tbl.redirected_blocks().tolist() == [False, False, False]

        # Test parallelization override
        assert dst_tbl.redirected_tabs(parallelization="inline").tolist() == [False, False, False]

        dst_tbl.tab(1).UNSAFE_redirect(paths=src_tbl.tab(1).paths(), OVERRIDE=True)

        assert dst_tbl.tab(1).redirected() is True
        assert dst_tbl.tab(0).redirected() is False
        assert dst_tbl.redirected_tab(1) is True
        assert dst_tbl.redirected_tab(0) is False
        assert dst_tbl.redirected_block(1) is True
        assert dst_tbl.redirected_block(0) is False
        assert dst_tbl.redirected_tabs().tolist() == [False, True, False]
        assert dst_tbl.redirected_blocks().tolist() == [False, True, False]
        assert dst_tbl.redirected_tabs(parallelization="inline").tolist() == [False, True, False]

        # Test false_only and true_only
        assert dst_tbl.redirected_tabs(false_only=True).index.tolist() == [0, 2]
        assert dst_tbl.redirected_tabs(true_only=True).index.tolist() == [1]
        with pytest.raises(ValueError, match="mutually exclusive"):
            dst_tbl.redirected_tabs(false_only=True, true_only=True)

    def test_fold_valid_and_redirected_tabs(self, tmp_path):
        src_tbl = LetterTable(
            url=str(tmp_path / "fold_src"),
            spec=dict(n_tabs_=4),
        ).build()
        dst_tbl = LetterTable(
            url=str(tmp_path / "fold_dst"),
            spec=dict(n_tabs_=4),
        )
        dst_tbl.tab(0).build()
        dst_tbl._write_tab_path(0)
        dst_tbl.tab(2).UNSAFE_redirect(paths=src_tbl.tab(2).paths(), OVERRIDE=True)

        partition = DatapointPartition(
            url=str(tmp_path / "partition"),
            validate_vars=False,
            spec=dict(
                datapoint_table=dst_tbl,
                fractions=[0.5, 0.5],
                partition_slice='letters',
            ),
        ).build()

        fold0 = partition.fold(0)
        # fold0 covers a subset of tabs
        valid_res = fold0.valid_tabs()
        red_res = fold0.redirected_tabs()
        assert isinstance(valid_res, pd.Series)
        assert isinstance(red_res, pd.Series)
        assert len(valid_res) == fold0.n_tabs
        assert len(red_res) == fold0.n_tabs
        assert all(isinstance(v, (bool, np.bool_)) for v in valid_res)
        assert all(isinstance(r, (bool, np.bool_)) for r in red_res)


