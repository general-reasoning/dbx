"""`BuildJournal.diff` -- the datablocks of one journal, side by side: a column per hash, a row per difference."""
from dataclasses import dataclass

import pytest

from dbx.datablocks import DATAFILE, BuildJournal, Datablock


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


def built(tmp_path, cls=Built, **kwargs):
    dblk = cls(datalake=str(tmp_path), anchor=ANCHOR, **kwargs)
    dblk.build()
    return dblk


def diff(tmp_path, mode, *dblks, **kwargs):
    return BuildJournal.diff(mode, *(d.hash for d in dblks), anchor=ANCHOR, datalake=str(tmp_path), **kwargs)


class TestSignature:

    def test_a_row_per_differing_leaf_a_column_per_hash(self, tmp_path):
        a = built(tmp_path, spec=dict(x=1, y='a'))
        b = built(tmp_path, spec=dict(x=2, y='a'))
        d = diff(tmp_path, 'signature', a, b)
        assert list(d.columns) == [a.hash, b.hash]
        assert list(d.index) == ['spec.x']
        assert list(d.loc['spec.x']) == [1, 2]

    def test_values_are_typed(self, tmp_path):
        a = built(tmp_path, spec=dict(x=1, y='1'))
        b = built(tmp_path, spec=dict(x=1, y='a'))
        assert list(diff(tmp_path, 'signature', a, b).loc['spec.y']) == ['1', 'a']

    def test_three_hashes(self, tmp_path):
        dblks = [built(tmp_path, spec=dict(x=x)) for x in (1, 1, 3)]
        dblks[1] = built(tmp_path, spec=dict(x=1, y='b'))
        d = diff(tmp_path, 'signature', *dblks)
        assert sorted(d.index) == ['spec.x', 'spec.y']
        assert list(d.loc['spec.x']) == [1, 1, 3]

    def test_transpose(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        d = diff(tmp_path, 'signature', a, b, transpose=True)
        assert list(d.index) == [a.hash, b.hash] and list(d.columns) == ['spec.x']

    def test_nothing_differing_is_empty(self, tmp_path):
        a = built(tmp_path)
        d = BuildJournal.diff('signature', a.hash, a.hash[:12], anchor=ANCHOR, datalake=str(tmp_path))
        assert d.empty and list(d.columns) == [a.hash, a.hash[:12]]

    def test_a_hash_prefix(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        d = BuildJournal.diff('signature', a.hash[:10], b.hash[:10], anchor=ANCHOR, datalake=str(tmp_path))
        assert list(d.columns) == [a.hash[:10], b.hash[:10]]


class TestTopics:

    def test_only_the_topics_some_hash_lacks(self, tmp_path):
        a, b = built(tmp_path), built(tmp_path, cls=Extra)
        d = diff(tmp_path, 'topics', a, b)
        # 'out' is in both, under different files: present in every hash, so not listed.
        assert list(d.index) == ['log']
        assert d.loc['log', a.hash] == BuildJournal.DIFF_NA
        assert d.loc['log', b.hash] != BuildJournal.DIFF_NA


class TestVersion:

    def test_one_row(self, tmp_path):
        a, b = built(tmp_path), built(tmp_path, cls=Extra)
        d = diff(tmp_path, 'version', a, b)
        assert list(d.index) == ['version'] and list(d.loc['version']) == [1, 2]

    def test_the_same_version_is_empty(self, tmp_path):
        a, b = built(tmp_path, spec=dict(x=1)), built(tmp_path, spec=dict(x=2))
        assert diff(tmp_path, 'version', a, b).empty


class TestOps:

    def test_the_ops_of_each_hash(self, tmp_path):
        a = built(tmp_path, spec=dict(x=1), n_viz_workers=3)
        b = built(tmp_path, spec=dict(x=2))
        d = diff(tmp_path, 'ops', a, b)
        assert list(d.index) == ['n_viz_workers']
        assert list(d.loc['n_viz_workers']) == [3, BuildJournal.DIFF_NA]


class TestRefusals:

    def test_an_unknown_mode(self, tmp_path):
        a = built(tmp_path)
        with pytest.raises(ValueError):
            BuildJournal.diff('hash', a.hash, a.hash, anchor=ANCHOR, datalake=str(tmp_path))

    def test_one_hash(self, tmp_path):
        a = built(tmp_path)
        with pytest.raises(ValueError):
            BuildJournal.diff('signature', a.hash, anchor=ANCHOR, datalake=str(tmp_path))

    def test_an_unknown_hash(self, tmp_path):
        a = built(tmp_path)
        with pytest.raises(LookupError):
            BuildJournal.diff('signature', a.hash, 'f' * 64, anchor=ANCHOR, datalake=str(tmp_path))
