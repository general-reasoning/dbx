"""`BuildJournalFrame.diff` -- the datablocks of one journal, side by side: a column per hash, a row per difference.

`BuildJournal.diff` is the same over an anchor's whole journal, read for it.
"""
from dataclasses import dataclass

import pandas as pd
import pytest

from dbx.datablocks import DATAFILE, BuildJournal, BuildJournalFrame, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch, tmp_path):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))


class Built(Datablock):
    VERSION = 1
    TOPICS = {'out': DATAFILE('out.txt'), 'data': {'frames': DATAFILE('frames.txt')}}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 1
        y: str = 'a'

    def __build__(self):
        with open(self.path('out', ensure_dirpath=True), 'w') as f:
            f.write('data')


class Extra(Built):
    VERSION = 2
    TOPICS = {'out': DATAFILE('renamed.txt'), 'data': {'frames': DATAFILE('frames.txt')}, 'log': DATAFILE('run.log')}


ANCHOR = 'diffed'
HEADERS = ['tag', 'VERSION', 'TOPICS']


def built(tmp_path, cls=Built, **kwargs):
    dblk = cls(datalake=str(tmp_path), anchor=ANCHOR, **kwargs)
    dblk.build()
    return dblk


def diff(tmp_path, dblks, mode='signature'):
    return BuildJournal.diff([d.hash for d in dblks], mode, anchor=ANCHOR, datalake=str(tmp_path))


class TestSignature:

    def test_a_row_per_differing_leaf_a_column_per_hash(self, tmp_path):
        a = built(tmp_path, spec=dict(x=1, y='a'))
        b = built(tmp_path, spec=dict(x=2, y='a'))
        d = diff(tmp_path, [a, b])
        assert list(d.columns) == [a.hash, b.hash]
        assert list(d.index) == HEADERS + ['spec.x']
        assert list(d.loc['spec.x']) == [1, 2]

    def test_values_are_typed(self, tmp_path):
        a = built(tmp_path, spec=dict(x=1, y='1'))
        b = built(tmp_path, spec=dict(x=1, y='a'))
        assert list(diff(tmp_path, [a, b]).loc['spec.y']) == ['1', 'a']

    def test_three_hashes(self, tmp_path):
        dblks = [built(tmp_path, spec=spec) for spec in (dict(x=1), dict(x=1, y='b'), dict(x=3))]
        d = diff(tmp_path, dblks)
        assert list(d.index) == HEADERS + ['spec.x', 'spec.y']
        assert list(d.loc['spec.x']) == [1, 1, 3]

    def test_transposed(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        d = diff(tmp_path, [a, b]).transpose()
        assert list(d.index) == [a.hash, b.hash] and list(d.columns) == HEADERS + ['spec.x']

    def test_nothing_differing_is_the_headers_alone(self, tmp_path):
        a = built(tmp_path)
        d = BuildJournal.diff([a.hash, a.hash[:12]], anchor=ANCHOR, datalake=str(tmp_path))
        assert list(d.index) == HEADERS and list(d.columns) == [a.hash, a.hash[:12]]


class TestTheHeaders:
    """`tag`, `VERSION` and `TOPICS` are always there, first, the same in every hash or not."""

    def test_version_topics_and_tag(self, tmp_path):
        a, b = built(tmp_path, tag='first'), built(tmp_path, cls=Extra)
        d = diff(tmp_path, [a, b])
        assert list(d.loc['VERSION']) == [1, 2]
        assert list(d.loc['TOPICS']) == [('out', 'data'), ('out', 'data', 'log')]
        assert list(d.loc['tag']) == ['first', BuildJournal.DIFF_NA]

    def test_in_ops_too(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        assert list(diff(tmp_path, [a, b], mode='ops').index) == HEADERS


class TestOps:

    def test_the_ops_of_each_hash(self, tmp_path):
        a = built(tmp_path, spec=dict(x=1), n_viz_workers=3)
        b = built(tmp_path, spec=dict(x=2))
        d = diff(tmp_path, [a, b], mode='ops')
        assert list(d.index) == HEADERS + ['n_viz_workers']
        assert list(d.loc['n_viz_workers']) == [3, BuildJournal.DIFF_NA]


class TestRefusals:

    @pytest.mark.parametrize('mode', ['hash', 'version', 'topics'])
    def test_an_unknown_mode(self, tmp_path, mode):
        a = built(tmp_path)
        with pytest.raises(ValueError):
            BuildJournal.diff([a.hash, a.hash], mode, anchor=ANCHOR, datalake=str(tmp_path))

    def test_one_hash(self, tmp_path):
        a = built(tmp_path)
        with pytest.raises(ValueError):
            BuildJournal.diff([a.hash], anchor=ANCHOR, datalake=str(tmp_path))

    def test_an_unknown_hash(self, tmp_path):
        a = built(tmp_path)
        with pytest.raises(LookupError):
            BuildJournal.diff([a.hash, 'f' * 64], anchor=ANCHOR, datalake=str(tmp_path))

    def test_hashes_after_the_list(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        with pytest.raises(TypeError):
            a.buildjournal().diff([a.hash], b.hash)


class TestOnTheFrame:
    """A datablock's `buildjournal()` is a `BuildJournalFrame`, and its `diff` takes the list of hashes alone."""

    def test_a_datablocks_journal(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        d = a.buildjournal().diff([a.hash[:14], b.hash[:11]])
        assert list(d.columns) == [a.hash[:14], b.hash[:11]] and list(d.loc['spec.x']) == [1, 2]

    def test_a_hash_is_a_pattern_as_a_filter_is(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        d = a.buildjournal().diff([f"^{a.hash[:6]}", f"{b.hash[:6]}*"])
        assert list(d.loc['spec.x']) == [1, 2]

    def test_a_pattern_matching_several_hashes_is_refused(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        with pytest.raises(LookupError, match='matches 2 hashes'):
            a.buildjournal().diff(['.*', b.hash])

    def test_pandas_diff_is_still_pandas(self, tmp_path):
        a = built(tmp_path)
        frame = a.buildjournal()
        numbers = frame.assign(n=range(len(frame)))[['n']]
        pd.testing.assert_frame_equal(pd.DataFrame(numbers.diff()), pd.DataFrame(numbers).diff())
        pd.testing.assert_frame_equal(pd.DataFrame(numbers.diff(2)), pd.DataFrame(numbers).diff(2))
        pd.testing.assert_frame_equal(pd.DataFrame(numbers.diff(periods=2)), pd.DataFrame(numbers).diff(periods=2))


class TestRowsAlreadyRead:
    """Rows read once, wrapped as a `BuildJournalFrame`, diff as the journal read for it would -- nothing read again."""

    @pytest.fixture
    def read(self, tmp_path):
        a, b, c = (built(tmp_path, spec=dict(x=x)) for x in (1, 2, 3))
        return (a, b, c), BuildJournal.read(ANCHOR, datalake=str(tmp_path), index=None)

    def test_reads_nothing_again(self, read, monkeypatch):
        (a, b, _), journal = read
        monkeypatch.setattr(BuildJournal, 'read', classmethod(lambda cls, *a, **k: pytest.fail('read again')))
        assert list(journal.diff([a.hash, b.hash]).loc['spec.x']) == [1, 2]

    def test_as_reading_it_would(self, read, tmp_path):
        (a, b, c), journal = read
        hashes = [a.hash[:8], b.hash[:8], c.hash[:8]]
        assert journal.diff(hashes).equals(BuildJournal.diff(hashes, anchor=ANCHOR, datalake=str(tmp_path)))

    def test_a_plain_frame_of_rows(self, read):
        """As hyp holds the pipeline journal: a DataFrame, wrapped."""
        (a, b, _), journal = read
        d = BuildJournalFrame(pd.DataFrame(journal).reset_index(drop=True)).diff([a.hash, b.hash])
        assert list(d.loc['spec.x']) == [1, 2]
