"""Every slice gets one sample per item, and a build that breaks that says so.

A tab's slices are read back by index: sample *i* of one is taken to describe
what sample *i* of another does.  A tab that writes one slice and skips another
for some item -- an early ``continue``, a per-item exception swallowed, a branch
that writes two of three -- shifts one slice against the rest from that item on.

Nothing downstream can see it.  A map-mode zip pairs index *i* with index *i*
whatever the rows mean, so every row after the skip pairs one item's data with
another's, no read raises, and the artifacts look exactly like correct ones.
The training signal is wrong and everything reports success.

So the count is taken on every build.  It used to be installed only when
`slice_writers` was given ``flush_every``, which made the guarantee a side
effect of wanting row-sized shards -- a coupling nothing stated and nobody
would guess, and one a single-slice table or a table not read in ``mode='iter'``
had no reason to ask for.
"""
import shutil
from dataclasses import dataclass

import pytest

pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

from dbx.datapoints import SLICETOPIC, DatapointTab, DatapointTable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class TwoSlices(DatapointTab):
    """Writes ``n`` items into two slices, optionally skipping one item of 'b'."""

    VERSION = 1
    TOPICS = {'a': SLICETOPIC, 'b': SLICETOPIC}

    @dataclass
    class VAR(DatapointTab.VAR):
        n: int = 4
        skip: int = None
        flush_every: int = None

    def __build__(self):
        columns = {'a': {'i': 'int'}, 'b': {'i': 'int'}}
        with self.slice_writers(columns, flush_every=self.var.flush_every) as writers:
            for i in range(self.var.n):
                writers['a'].write({'i': i})
                if i != self.var.skip:
                    writers['b'].write({'i': i})


def tab(tmp_path, **spec):
    return TwoSlices(url=str(tmp_path), spec=spec)


@pytest.mark.pinned
class TestASliceGetsOneSamplePerItem:
    """A tab that writes its slices unequally must not produce a valid table.

    This is the contract the slices exist under: sample *i* of one describes
    what sample *i* of another describes.  Break it and every zipped row after
    the break is wrong, silently -- so the check has to be on the build that
    caused it, not on the reader that cannot tell.
    """

    def test_a_ragged_build_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match='lockstep'):
            tab(tmp_path, n=4, skip=2).build()

    def test_it_is_refused_without_flush_every(self, tmp_path):
        """The gap: the count used to be installed only alongside shard
        alignment, so a tab that never asked for it was never checked."""
        with pytest.raises(ValueError, match='lockstep'):
            tab(tmp_path, n=4, skip=2, flush_every=None).build()

    def test_it_is_refused_with_flush_every_too(self, tmp_path):
        with pytest.raises(ValueError, match='lockstep'):
            tab(tmp_path, n=8, skip=5, flush_every=4).build()

    def test_a_lockstep_build_is_not(self, tmp_path):
        """The guard: a check that refused every build would pass the three
        above while making the library useless."""
        t = tab(tmp_path, n=4)
        t.build()
        assert t.valid()
        assert t.data('a')['a']['i'] == t.data('b')['b']['i'] == [0, 1, 2, 3]

    def test_the_refusal_names_the_counts(self, tmp_path):
        """Which slice fell behind, and by how much, is the whole diagnosis."""
        with pytest.raises(ValueError, match=r"\{'a': 4, 'b': 3\}"):
            tab(tmp_path, n=4, skip=2).build()


class TestValidateReadsTheFinishedIndexes:
    """Defence in depth: the write path is not the only way a slice arrives."""

    def test_a_slice_whose_index_disagrees_fails_validate(self, tmp_path):
        long, short = tab(tmp_path, n=4), tab(tmp_path, n=2)
        long.build()
        short.build()
        # A half-uploaded slice: 'b' claims two rows where the rest has four.
        shutil.copy(short.slice_index_path('b'), long.slice_index_path('b'))
        with pytest.raises(ValueError, match='lockstep'):
            long.validate()

    def test_a_whole_tab_validates(self, tmp_path):
        t = tab(tmp_path, n=4)
        t.build()
        assert t.validate()

    def test_an_unbuilt_tab_is_answered_before_any_index_is_read(self, tmp_path):
        """`valid()` comes first, so a tab with nothing on disk says False
        rather than raising about row counts it cannot read."""
        assert tab(tmp_path, n=4).validate() is False


class TestASingleSliceTabIsUnaffected:
    """The count costs an increment per write and answers vacuously here."""

    def test_it_builds_and_reads(self, tmp_path):
        class OneSlice(DatapointTab):
            TOPICS = {'only': SLICETOPIC}

            def __build__(self):
                with self.slice_writers({'only': {'i': 'int'}}) as writers:
                    for i in range(3):
                        writers['only'].write({'i': i})

        t = OneSlice(url=str(tmp_path))
        t.build()
        assert t.validate() and t.data('only') == {'only': {'i': [0, 1, 2]}}


class TestATableRefusesARaggedTab:

    def test_the_build_stops_at_the_tab_that_did_it(self, tmp_path):
        class RaggedTable(DatapointTable):
            VERSION = 1
            TAB = TwoSlices

            @property
            def n_tabs(self):
                return 2

            def __tab__(self, idx, tag=None):
                return super().__tab__(idx, n=4, skip=2 if idx == 1 else None)

        with pytest.raises(ValueError, match='lockstep'):
            RaggedTable(url=str(tmp_path)).build()
