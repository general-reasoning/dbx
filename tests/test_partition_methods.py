"""
`DatatablePartition`: whole tabs dealt to folds, by groupby / stratifyby / balance / seed.

Each tab here is one "slide": `rows` rows, every one carrying the slide's
patient and cohort in a `meta` slice -- the columns the partition reads,
named by its collator's groupby and stratifyby roles.
"""
import json
from dataclasses import dataclass

import numpy as np
import pytest

from dbx.datablocks import Datablock
from dbx.datatables import DATASLICE, Datacollator, Datatab, Datatable, DatatablePartition


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Slide(Datatab):
    TOPICS = {'meta': DATASLICE(patient='str', cohort='str', info='json')}

    @dataclass
    class VAR(Datatab.VAR):
        patient: str = ''
        cohort: str = ''
        rows: int = 1
        mixed: bool = False         # a second patient on the last row

    def __build__(self):
        with self.slice_writers() as writers:
            for i in range(self.var.rows):
                patient = 'other' if self.var.mixed and i == self.var.rows - 1 else self.var.patient
                cohort = self.var.cohort or None
                writers['meta'].write({'patient': patient, 'cohort': cohort or '',
                                       'info': {'cohort': cohort, 'patient': patient}})


#: (patient, cohort, rows): 3 cohorts, some patients with two slides, sizes varying 100x.
SLIDES = [('p1', 'LUAD', 1000), ('p1', 'LUAD', 800), ('p2', 'LUAD', 50), ('p3', 'LUAD', 60),
          ('p4', 'BRCA', 5000), ('p4', 'BRCA', 4000), ('p5', 'BRCA', 70), ('p8', 'BRCA', 90),
          ('p6', 'OV', 40), ('p7', 'OV', 30), ('p7', 'OV', 6000), ('p9', 'OV', 20)]


class Slides(Datatable):
    TAB = Slide

    @dataclass
    class VAR(Datatable.VAR):
        slides: list = None

    @property
    def n_tabs(self):
        return len(self.var.slides)

    def __tab__(self, idx, **kw):
        s = self.var.slides[idx]
        return super().__tab__(idx, patient=s[0], cohort=s[1], rows=s[2], mixed=bool(s[3:]) and s[3])


def table(tmp_path, slides=SLIDES):
    return Slides(datalake=str(tmp_path), spec={'slides': [list(s) for s in slides]}).build()


INFO = ('meta', 'info')
PATIENT, COHORT = INFO + ('patient',), INFO + ('cohort',)


def collator(groupby=None, stratifyby=None):
    columns = {k: v for k, v in (('groupby', groupby), ('stratifyby', stratifyby)) if v is not None}
    return Datacollator(spec=dict(columns=columns)) if columns else None


def partition(tmp_path, tbl, groupby=None, stratifyby=None, build=True, **spec):
    p = DatatablePartition(datalake=str(tmp_path), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta',
        collator=collator(groupby, stratifyby), **spec))
    return p.build() if build else p


def folds_of(p):
    return [set(f) for f in p.read('tabs')]


def test_every_tab_is_in_exactly_one_fold(tmp_path):
    p = partition(tmp_path, table(tmp_path), groupby=PATIENT, stratifyby=COHORT)
    a, b = folds_of(p)
    assert a.isdisjoint(b) and a | b == set(range(len(SLIDES)))


def test_a_group_lands_in_one_fold(tmp_path):
    p = partition(tmp_path, table(tmp_path), groupby=PATIENT, balance='tabs')
    fold_of = {i: k for k, f in enumerate(folds_of(p)) for i in f}
    for patient in {s[0] for s in SLIDES}:
        tabs = [i for i, s in enumerate(SLIDES) if s[0] == patient]
        assert len({fold_of[i] for i in tabs}) == 1, patient


def test_the_fractions_hold_within_each_stratum(tmp_path):
    p = partition(tmp_path, table(tmp_path), stratifyby=COHORT, balance='tabs')
    for f in p.read('summary')['folds']:
        # 4 slides per cohort, halved: 2 and 2 in each fold.
        assert {c: v['tabs'] for c, v in f['strata'].items()} == {'BRCA': 2, 'LUAD': 2, 'OV': 2}


def test_the_seed_decides_and_repeats(tmp_path):
    tbl = table(tmp_path)
    one = folds_of(partition(tmp_path / 'a', tbl, groupby=PATIENT, stratifyby=COHORT, seed=0))
    again = folds_of(partition(tmp_path / 'b', tbl, groupby=PATIENT, stratifyby=COHORT, seed=0))
    assert one == again
    seeds = {json.dumps([sorted(f) for f in folds_of(partition(tmp_path / f's{s}', tbl, stratifyby=COHORT, seed=s))])
             for s in range(8)}
    assert len(seeds) > 1, "different seeds deal differently"


def test_balance_counts_rows_or_tabs(tmp_path):
    tbl = table(tmp_path)
    by_tabs = partition(tmp_path / 't', tbl, balance='tabs').read('summary')['folds']
    assert [f['tabs'] for f in by_tabs] == [6, 6]
    by_rows = partition(tmp_path / 'r', tbl, balance='rows').read('summary')['folds']
    rows = [f['rows'] for f in by_rows]
    assert sum(rows) == sum(s[2] for s in SLIDES) and min(rows) > 0


def test_the_summary_says_what_each_fold_holds(tmp_path):
    tbl = table(tmp_path)
    s = partition(tmp_path, tbl, groupby=PATIENT, stratifyby=COHORT).read('summary')
    assert s['method'] == 'random' and s['n_tabs'] == len(SLIDES) and s['skipped'] == []
    tags = [t for f in s['folds'] for t in f['tags']]
    assert sorted(tags) == sorted(tbl.tab(i).tag for i in range(len(SLIDES)))


def test_a_tab_with_no_value_is_skipped_and_said(tmp_path, caplog):
    slides = SLIDES + [('p10', '', 10)]          # cohort None
    p = partition(tmp_path, table(tmp_path, slides), stratifyby=COHORT)
    s = p.read('summary')
    assert [k['idx'] for k in s['skipped']] == [len(SLIDES)] and 'no stratifyby value' in s['skipped'][0]['why']
    assert len(SLIDES) not in set().union(*folds_of(p))


def test_a_tab_holding_two_values_is_not_implemented(tmp_path):
    slides = SLIDES[:-1] + [('p9', 'OV', 20, True)]
    with pytest.raises(NotImplementedError, match="hold more than one value"):
        partition(tmp_path, table(tmp_path, slides), groupby=PATIENT)


def test_a_group_spanning_strata_is_refused(tmp_path):
    slides = SLIDES + [('p1', 'OV', 10)]          # p1 is LUAD elsewhere
    with pytest.raises(ValueError, match="span more than one stratum"):
        partition(tmp_path, table(tmp_path, slides), groupby=PATIENT, stratifyby=COHORT)


def test_largest_first_takes_none_of_the_new_choices(tmp_path):
    with pytest.raises(ValueError, match="largest_first' takes no"):
        DatatablePartition(datalake=str(tmp_path), spec=dict(
            datatable=table(tmp_path), fractions=[0.5, 0.5], partition_slice='meta',
            method='largest_first', collator=collator(stratifyby=COHORT)))


def test_largest_first_deals_as_before(tmp_path):
    p = partition(tmp_path, table(tmp_path), method='largest_first')
    rows = [s[2] for s in SLIDES]
    # Descending rows, each to the fold with the larger deficit -- by hand:
    want, have, target = [[], []], [0, 0], [sum(rows) / 2] * 2
    for i in sorted(range(len(rows)), key=lambda i: rows[i], reverse=True):
        k = int(np.argmax([target[0] - have[0], target[1] - have[1]]))
        want[k].append(i)
        have[k] += rows[i]
    assert p.read('tabs') == [sorted(w) for w in want]


def test_a_collator_with_other_roles_is_refused(tmp_path):
    c = Datacollator(spec=dict(columns={'groupby': PATIENT, 'signals': [INFO]}))
    with pytest.raises(ValueError, match="declares \\['signals'\\] too"):
        DatatablePartition(datalake=str(tmp_path), spec=dict(
            datatable=table(tmp_path), fractions=[0.5, 0.5], partition_slice='meta', collator=c))


@pytest.mark.parametrize('stratifyby', [COHORT, None])
def test_the_partition_from_before_the_collator_is_reached(tmp_path, monkeypatch, stratifyby):
    """Its table was datapoint_table, and groupby and stratifyby were fields of
    its own -- as a collator holding them renders, under its old module name."""
    tbl = table(tmp_path)

    @dataclass
    class OldVAR(Datablock.VAR):
        datapoint_table: Datatable
        fractions: list
        partition_slice: int | str
        method: str = 'random'
        seed: int = 0
        groupby: tuple | list | None = None
        stratifyby: tuple | list | None = None
        balance: str = 'rows'

    with monkeypatch.context() as m:
        m.setattr(DatatablePartition, 'VAR', OldVAR)
        m.setattr(DatatablePartition, 'SPECIALIZATIONS', [])
        m.setattr(DatatablePartition, '__post_init__', Datablock.__post_init__)
        old = DatatablePartition(datalake=str(tmp_path), spec=dict(
            datapoint_table=tbl, fractions=[0.5, 0.5], partition_slice='meta',
            groupby=PATIENT, stratifyby=stratifyby, seed=3))
        old_hash = old.hash
    new = partition(tmp_path, tbl, groupby=PATIENT, stratifyby=stratifyby, seed=3, build=False)
    assert new.hash != old_hash
    assert old_hash in new.specialization_hashes()
