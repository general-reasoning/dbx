"""Which columns the slices share is declared on the block, not at each call.

`shared=` and `validate_shared=` were per-call parameters of `dataset()`, so
the read-time alignment check was configured by every consumer rather than once
by the author.  Which columns two slices hold in common is a fact about how
they were WRITTEN: the caller is the wrong party to know it, and every caller of
one table needs the same answer.

So a `DatapointTab` or `DatapointTable` declares ``shared_slice_columns``, and a
block reading an upstream block's slices alongside its own -- a `DatafeatureTab`
over its `DatapointTab` -- declares ``shared_upstream_column`` for the column
that answers the question spanning the two: is feature row *i* the features OF
sample row *i*?

Both are kwargs and not VAR: they change no bytes, only what is checked on a
read.  And neither defaults to the intersection of the declared columns, because
`_check_shared` runs inside `_merge` and so is paid on every row of every epoch:
dbx cannot know whether that intersection is two scalars or a
``(datapoints_per_row, ...)`` ndarray, and imposing an unbounded per-row cost
the author never chose is worse than not checking.  Whoever declared the slices
knows which key is cheap.
"""
from dataclasses import dataclass

import pytest

pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

from dbx.datapoints import SLICETOPIC, DatapointTab, DatapointTable
from dbx.datafeatures import _UpstreamSlices


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Samples(DatapointTab):
    """Two slices carrying one bookkeeping column, written in lockstep."""

    VERSION = 1
    TOPICS = {'a': SLICETOPIC, 'b': SLICETOPIC}

    @dataclass
    class VAR(DatapointTab.VAR):
        n: int = 4

    def __build__(self):
        columns = {'a': {'idx': 'int', 'x': 'int'}, 'b': {'idx': 'int', 'y': 'int'}}
        with self.slice_writers(columns) as writers:
            for i in range(self.var.n):
                writers['a'].write({'idx': i, 'x': i * 10})
                writers['b'].write({'idx': i, 'y': i * 100})


class SampleTable(DatapointTable):
    VERSION = 1
    TAB = Samples

    @property
    def n_tabs(self):
        return 2


class Derived(_UpstreamSlices, DatapointTab):
    """The shape a `DatafeatureTab` has: own slice plus an upstream block's.

    Carries the upstream's bookkeeping column through, which is what makes the
    cross-block alignment checkable at all.
    """

    VERSION = 1
    TOPICS = {'d': SLICETOPIC}
    UPSTREAM_VAR = ('source',)

    @dataclass
    class VAR(DatapointTab.VAR):
        source: Samples = None
        offset: int = 0

    def __build__(self):
        upstream = self.var.source.data(('a', ['idx']))['a']['idx']
        with self.slice_writers({'d': {'idx': 'int', 'z': 'int'}}) as writers:
            for i in upstream:
                writers['d'].write({'idx': int(i) + self.var.offset, 'z': int(i) ** 2})


@pytest.fixture
def samples(tmp_path):
    s = Samples(url=str(tmp_path), shared_slice_columns='idx')
    s.build()
    return s


class TestTheDeclarationIsTheDefault:

    def test_a_caller_that_says_nothing_gets_the_check(self, samples):
        ds = samples.dataset()
        assert ds.shared == {'idx'} and ds.validate_shared is True

    def test_a_block_declaring_nothing_checks_nothing(self, tmp_path):
        plain = Samples(url=str(tmp_path))
        plain.build()
        ds = plain.dataset()
        assert not ds.shared and ds.validate_shared is False

    def test_a_caller_passing_shared_is_not_validated_unasked(self, tmp_path):
        """What validate_shared=False meant before this defaulting existed."""
        plain = Samples(url=str(tmp_path))
        plain.build()
        assert plain.dataset(shared={'idx'}).validate_shared is False

    def test_a_caller_may_override_the_declaration(self, samples):
        assert samples.dataset(validate_shared=False).validate_shared is False
        assert samples.dataset(shared={'x'}).shared == {'x'}

    def test_a_table_declares_it_the_same_way(self, tmp_path):
        table = SampleTable(url=str(tmp_path), shared_slice_columns='idx')
        table.build()
        ds = table.dataset()
        assert ds.shared == {'idx'} and ds.validate_shared is True

    def test_a_bare_str_is_one_column_and_not_four(self, samples):
        """'idx' is iterable, so left unnormalised it reads as ('i','d','x')."""
        assert samples.shared_slice_columns == ('idx',)
        assert samples.dataset().shared == {'idx'}

    def test_a_sequence_is_taken_as_given(self, tmp_path):
        s = Samples(url=str(tmp_path), shared_slice_columns=['idx', 'x'])
        assert s.shared_slice_columns == ('idx', 'x')

    def test_it_survives_a_reconstruction(self, samples):
        assert samples.set().shared_slice_columns == ('idx',)

    def test_it_is_not_in_the_identity(self, tmp_path):
        """It changes no bytes, only what is checked on a read."""
        assert Samples(url=str(tmp_path), shared_slice_columns='idx').hash == \
               Samples(url=str(tmp_path)).hash


class TestWhatTheCheckCatches:

    def test_aligned_slices_read(self, samples):
        assert samples.dataset()[0] == {'a': {'idx': 0, 'x': 0}, 'b': {'idx': 0, 'y': 0}}

    def test_a_misaligned_pairing_is_caught(self, samples):
        """The comparison itself: two rows whose shared column disagrees.

        A shift is what the check exists for, so pair row 0 of one slice with
        row 1 of the other and read the merge.
        """
        ds = samples.dataset()
        with pytest.raises(ValueError, match='not aligned'):
            ds._merge(0, [ds.datasets[0][0], ds.datasets[1][1]])

    def test_one_source_has_nothing_to_align_with(self, samples):
        """A single-slice read must not be refused for naming a shared column
        that only it carries -- there is no pairing to check."""
        assert samples.dataset('a')[0] == {'a': {'idx': 0, 'x': 0}}


class TestAKeyOnlyOneSourceCarriesIsRefused:
    """A shared key present in one source is compared against nothing.

    The loop passes it on every row, so the alignment it was named to check
    goes unchecked -- and reads as checked, which is worse than not asking.
    It is the failure mode of declaring the columns once on the block: the
    block cannot know which slices a caller will read together.
    """

    def test_a_projection_that_drops_it_is_refused(self, samples):
        with pytest.raises(ValueError, match='fewer than two'):
            samples.dataset(('a', ['idx', 'x']), ('b', ['y']))[0]

    def test_the_refusal_names_the_key(self, samples):
        with pytest.raises(ValueError, match=r"\['idx'\]"):
            samples.dataset(('a', ['idx']), ('b', ['y']))[0]

    def test_a_key_no_source_carries_is_refused(self, tmp_path):
        s = Samples(url=str(tmp_path), shared_slice_columns='nonesuch')
        s.build()
        with pytest.raises(ValueError, match='fewer than two'):
            s.dataset()[0]

    def test_it_is_settled_once_rather_than_per_row(self, samples):
        ds = samples.dataset()
        assert ds._shared_comparable is False
        ds[0]
        assert ds._shared_comparable is True


class TestTheUpstreamDeclaration:

    @pytest.fixture
    def derived(self, samples, tmp_path):
        d = Derived(url=str(tmp_path), spec=dict(source=samples),
                    shared_upstream_column='idx')
        d.build()
        return d

    def test_it_defaults_the_cross_block_comparison(self, derived):
        ds = derived.dataset('d', 'a')
        assert ds.shared == {'idx'} and ds.validate_shared is True

    def test_an_aligned_pair_reads(self, derived):
        assert derived.dataset('d', 'a')[0] == {'d': {'idx': 0, 'z': 0}, 'a': {'idx': 0, 'x': 0}}

    def test_a_shifted_derived_slice_is_caught(self, samples, tmp_path):
        """Exactly the fault the declaration exists for: the derived rows are
        no longer the ones computed from the samples they are paired with."""
        shifted = Derived(url=str(tmp_path), spec=dict(source=samples, offset=1),
                          shared_upstream_column='idx')
        shifted.build()
        with pytest.raises(ValueError, match='not aligned'):
            shifted.dataset('d', 'a')[0]

    def test_without_the_declaration_the_shift_reads_clean(self, samples, tmp_path):
        """What the default buys. Nothing raises, and every row pairs a derived
        value with the wrong sample."""
        shifted = Derived(url=str(tmp_path), spec=dict(source=samples, offset=1))
        shifted.build()
        assert shifted.dataset('d', 'a')[0] == {'d': {'idx': 1, 'z': 0}, 'a': {'idx': 0, 'x': 0}}

    def test_a_bare_str_is_normalised(self, derived):
        assert derived.shared_upstream_column == ('idx',)

    def test_it_takes_precedence_over_the_slice_declaration(self, samples, tmp_path):
        d = Derived(url=str(tmp_path), spec=dict(source=samples),
                    shared_upstream_column='idx', shared_slice_columns='z')
        d.build()
        assert d.dataset('d', 'a').shared == {'idx'}

    def test_the_slice_declaration_still_applies_alone(self, samples, tmp_path):
        d = Derived(url=str(tmp_path), spec=dict(source=samples),
                    shared_slice_columns='idx')
        d.build()
        assert d.dataset('d', 'a').shared == {'idx'}

    def test_it_is_not_in_the_identity(self, samples, tmp_path):
        assert Derived(url=str(tmp_path), spec=dict(source=samples),
                       shared_upstream_column='idx').hash == \
               Derived(url=str(tmp_path), spec=dict(source=samples)).hash
