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


def test_a_tab_holding_two_values_is_split_into_pieces(tmp_path):
    slides = SLIDES[:-1] + [('p9', 'OV', 20, True)]     # its last row is patient 'other'
    p = partition(tmp_path, table(tmp_path, slides), groupby=PATIENT)
    entries = [e for f in p.read('tabs') for e in f]
    last = len(slides) - 1
    assert last not in entries
    assert sorted((e for e in entries if isinstance(e, dict)), key=lambda e: e['groupby']) == [
        {'tab': last, 'groupby': 'other'}, {'tab': last, 'groupby': 'p9'}]
    s = p.read('summary')
    assert s['n_pieces'] == 2 and sum(f['pieces'] for f in s['folds']) == 2
    assert {t for f in s['folds'] for t in f['tags'] if '#' in t} == {f"{p.datatable.tab(last).tag}#other",
                                                                     f"{p.datatable.tab(last).tag}#p9"}


def test_a_group_spanning_strata_is_refused(tmp_path):
    slides = SLIDES + [('p1', 'OV', 10)]          # p1 is LUAD elsewhere
    with pytest.raises(ValueError, match="span more than one stratum"):
        partition(tmp_path, table(tmp_path, slides), groupby=PATIENT, stratifyby=COHORT)


def test_largest_first_takes_none_of_the_new_choices(tmp_path):
    with pytest.raises(ValueError, match="largest_first' takes no"):
        DatatablePartition(datalake=str(tmp_path), spec=dict(
            datatable=table(tmp_path), fractions=[0.5, 0.5], partition_slice='meta',
            method='largest_first', collator=collator(stratifyby=COHORT)))


@pytest.mark.parametrize('mixed', [False, True])
def test_largest_first_deals_as_before(tmp_path, mixed):
    # It reads no partition column, so a mixed tab is a tab like any other to it, and is not split.
    slides = SLIDES[:-1] + [('p9', 'OV', 20, True)] if mixed else SLIDES
    p = partition(tmp_path, table(tmp_path, slides), method='largest_first')
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


def test_a_partition_of_constant_tabs_is_what_it_was(tmp_path):
    """Pinned from dbx 07cd119, before tabs could be split: no partition or fold of constant tabs moves."""
    tbl = table(tmp_path)
    p = partition(tmp_path / 'gs', tbl, groupby=PATIENT, stratifyby=COHORT)
    assert p.hash == '3318e720fcb934c1fd3b563c7dc3f36487c8f63733a4414fdd0c9bccecba6a57'
    assert p.read('tabs') == [[3, 6, 7, 9, 10, 11], [0, 1, 2, 4, 5, 8]]
    assert p.part(0).hash == 'c6c4a6c63b633cb4ffee9cc7e1ba5025653a53294314f7d87e1279b4fe4ce166'
    assert 'n_pieces' not in p.read('summary')
    p = partition(tmp_path / 'none', tbl)
    assert p.hash == '1229ccb8cfc81d5dc775aea8d497d19339f0f2b3bf3413bcbf0575127b134dbc'
    assert p.read('tabs') == [[0, 2, 3, 4, 5, 7, 9, 11], [1, 6, 8, 10]]
    assert p.part(0).hash == '9aed45e329a46a642d46ba8f067fcb8fd382e18601ad0bb842f13698d7ae20ef'


# Bags: tabs whose rows each carry their own patient and cohort -- several of each in a tab.

class Bag(Datatab):
    TOPICS = {'meta': DATASLICE(row='int', patient='str', label='int', info='json'),
              'x': DATASLICE(row='int', features='ndarray:float32')}

    @dataclass
    class VAR(Datatab.VAR):
        bag: int = 0
        rows: list = None           # [patient | None, cohort] per row

    def __build__(self):
        with self.slice_writers() as writers:
            for i, (patient, cohort) in enumerate(self.var.rows):
                row, label = 1000 * self.var.bag + i, COHORTS.index(cohort)
                # Separable by label, so a probe has something to find.
                x = np.random.default_rng(row).normal(size=4).astype(np.float32) + 3 * np.eye(4, dtype=np.float32)[label]
                writers['meta'].write({'row': row, 'patient': patient or '', 'label': label,
                                       'info': {'patient': patient, 'cohort': cohort}})
                writers['x'].write({'row': row, 'features': x})


COHORTS = ['A', 'B']

#: Six bags: some of one patient, most of several; p2 in two bags; one row of no patient.
BAGS = [[('p1', 'A'), ('p1', 'A'), ('p2', 'A')],
        [('p3', 'B'), ('p3', 'B')],
        [('p2', 'A'), ('p4', 'B'), ('p4', 'B'), (None, 'B')],
        [('p5', 'A')] * 3,
        [('p6', 'B'), ('p7', 'B')],
        [('p8', 'A'), ('p8', 'A'), ('p9', 'B')]]


class Bags(Datatable):
    TAB = Bag

    @dataclass
    class VAR(Datatable.VAR):
        bags: list = None

    @property
    def n_tabs(self):
        return len(self.var.bags)

    def __tab__(self, idx, **kw):
        return super().__tab__(idx, bag=idx, rows=[list(r) for r in self.var.bags[idx]])


def bags(tmp_path):
    return Bags(datalake=str(tmp_path), spec={'bags': [[list(r) for r in b] for b in BAGS]}).build()


def bag_partition(tmp_path, tbl, groupby=PATIENT, stratifyby=COHORT, **spec):
    return DatatablePartition(datalake=str(tmp_path), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta',
        collator=collator(groupby, stratifyby), **spec)).build()


def built_parts(p, **kw):
    return [p.part(k).build(**kw) for k in range(p.n_folds())]


def part_meta(part):
    return part.data('meta')['meta']


def test_every_row_is_in_one_fold_or_skipped_and_said(tmp_path):
    tbl = bags(tmp_path)
    p = bag_partition(tmp_path / 'p', tbl)
    rows = [part_meta(f)['row'] for f in built_parts(p)]
    assert set(rows[0]).isdisjoint(rows[1])
    every = {1000 * b + i: r for b, bag in enumerate(BAGS) for i, r in enumerate(bag)}
    skipped = p.read('summary')['skipped']
    assert [(s['piece'], s['rows']) for s in skipped] == [({'tab': 2, 'groupby': None, 'stratifyby': 'B'}, 1)]
    assert 'no groupby value' in skipped[0]['why']
    assert sorted(rows[0] + rows[1]) == sorted(r for r, (patient, _) in every.items() if patient is not None)
    assert len(rows[0]) + len(rows[1]) + sum(s['rows'] for s in skipped) == len(every)


def test_a_groups_rows_are_in_one_fold_across_tabs_and_pieces(tmp_path):
    for seed in range(4):
        p = bag_partition(tmp_path / f's{seed}', bags(tmp_path), seed=seed)
        fold_of = {}
        for k, f in enumerate(built_parts(p)):
            for info in part_meta(f)['info']:
                fold_of.setdefault(info['patient'], set()).add(k)
        assert all(len(ks) == 1 for ks in fold_of.values()), (seed, fold_of)


def test_a_piece_holds_its_value_in_every_row_and_slice_alike(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    n_pieces = 0
    for fold in built_parts(p):
        for i, entry in enumerate(fold.tab_indices):
            if not isinstance(entry, dict):
                continue
            n_pieces += 1
            piece = fold.tab(i)
            meta, x = piece.data('meta')['meta'], piece.data('x')['x']
            assert {(m['patient'], m['cohort']) for m in meta['info']} == {(entry['groupby'], entry['stratifyby'])}
            assert piece.n_rows('meta') == piece.n_rows('x') == len(meta['row']) > 0
            assert meta['row'] == x['row']              # row i of one slice is row i of the other
            assert piece.tag == f"{p.datatable.tab(entry['tab']).tag}#{entry['groupby']}#{entry['stratifyby']}"
    assert n_pieces == p.read('summary')['n_pieces'] - 1        # all but the skipped one


def test_a_piece_records_its_source_rows_which_take_the_same_piece_of_the_source(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    for part in built_parts(p):
        for i, entry in enumerate(part.tab_indices):
            if not isinstance(entry, dict):
                continue
            piece, source = part.tab(i), p.datatable.tab(entry['tab'])
            rows = piece.read('source_rows')
            assert rows.dtype == np.int64 and list(rows) == sorted(rows)
            # The indices, applied to the source, are the piece -- what a tab built from the source row by row reuses.
            src = source.data('x')['x']
            assert [src['row'][r] for r in rows] == piece.data('x')['x']['row']
            info = source.data('meta')['meta']['info']
            assert [r for r, v in enumerate(info) if (v['patient'], v['cohort']) == (entry['groupby'], entry['stratifyby'])] == list(rows)


def test_fold_is_the_name_part_had_first(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    assert p.fold(1).hash == p.part(1).hash


def test_the_fractions_hold_within_each_stratum_counting_pieces(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path), groupby=None, balance='tabs')
    folds = p.read('summary')['folds']
    # Split by cohort only: A is bags 0, 3 and a piece each of 2 and 5; B bags 1, 4 and the others.
    for cohort in COHORTS:
        assert [f['strata'][cohort]['tabs'] for f in folds] == [2, 2], cohort


def test_the_same_seed_deals_the_same_pieces_with_the_same_hashes(tmp_path):
    tbl = bags(tmp_path)
    one, two = bag_partition(tmp_path / 'a', tbl, seed=5), bag_partition(tmp_path / 'b', tbl, seed=5)
    assert one.read('tabs') == two.read('tabs')
    pieces = [e for f in one.read('tabs') for e in f if isinstance(e, dict)]
    assert pieces
    again = DatatablePartition(datalake=str(tmp_path / 'a'), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta', collator=collator(PATIENT, COHORT), seed=5))
    assert [one.tab(e).hash for e in pieces] == [again.tab(e).hash for e in pieces]
    assert len({one.tab(e).hash for e in pieces}) == len(pieces)


def test_a_fold_builds_its_pieces_in_parallel(tmp_path):
    from dbx.datatables import DatatablePart
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    for k in range(p.n_folds()):
        fold = DatatablePart(datalake=str(tmp_path / 'p'), spec=dict(partition=p, fold=k),
                             parallelization='multiprocessing', n_workers=2)
        assert fold.hash == p.part(k).hash
        fold.build()
        assert all(p.part(k).valid_tab(i, validation='valid') for i in range(fold.n_tabs))


def test_a_piece_that_cannot_be_recovered_says_so(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    with pytest.raises(ValueError, match="cannot be recovered"):
        p.tab({'tab': 0, 'groupby': 'p4', 'stratifyby': 'B'}).build()


def test_a_mixed_tab_upstream_of_a_feature_table_is_not_implemented(tmp_path):
    from test_probes import DummyModelEvaluatorFactory, DummySampleTable
    from dbx import Featuretable
    url = str(tmp_path)
    samples = DummySampleTable(datalake=url, spec=dict(samples_per_tab=4), tag='samples').build()
    features = Featuretable(datalake=url, devices=['cpu'], tag='features', spec=dict(
        upstream=samples, evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
        collator=Datacollator(spec=dict(columns={'signals': [('samples', 'samples')], 'labels': [('labels', 'labels')]})))).build()
    with pytest.raises(NotImplementedError, match="DataslicesUpstream"):
        DatatablePartition(datalake=url, spec=dict(
            datatable=features, fractions=[0.5, 0.5], partition_slice='features',
            collator=Datacollator(spec=dict(columns={'stratifyby': ('labels', 'labels')}, recursive=True)))).build()


# Probes on folds holding pieces.

def probe(tmp_path, fit, ev, **spec):
    from dbx import FeatureAffineLogisticProbe
    return FeatureAffineLogisticProbe(datalake=str(tmp_path), spec=dict(
        fit_table=fit, eval_table=ev,
        collator=Datacollator(spec=dict(columns={'signals': [('x', 'features')], 'labels': [('meta', 'label')]})),
        **spec))


def test_a_probe_fits_and_scores_on_folds_holding_pieces(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    fit, ev = built_parts(p)
    assert any(isinstance(e, dict) for e in fit.tab_indices + ev.tab_indices)
    pr = probe(tmp_path / 'probe', fit, ev).build()
    assert sorted(pr.read('classes').tolist()) == [0, 1]
    assert len(pr.read('fit', 'labels')) == fit.n_rows('meta')
    assert len(pr.read('eval', 'labels')) == ev.n_rows('meta')


def test_pieces_of_one_tab_in_two_folds_are_disjoint_and_one_piece_twice_is_not(tmp_path):
    tbl = bags(tmp_path)
    for seed in range(20):
        p = bag_partition(tmp_path / f's{seed}', tbl, seed=seed)
        sources = [{e['tab'] for e in f if isinstance(e, dict)} for f in p.read('tabs')]
        if sources[0] & sources[1]:
            break
    else:
        pytest.fail("no seed puts two pieces of one tab in different folds")
    fit, ev = built_parts(p)
    probe(tmp_path / 'ok', fit, ev)._check_disjoint_()
    with pytest.raises(ValueError, match="share"):
        probe(tmp_path / 'twice', fit, fit)._check_disjoint_()


def test_a_probe_says_a_piece_is_not_built_before_it_starts(tmp_path):
    from dbx.datablocks import InvalidBlocksError
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    fit, ev = built_parts(p)
    # A built fold, and one of its pieces gone since: the fold vouches, the piece does not.
    i = next(i for i, e in enumerate(fit.tab_indices) if isinstance(e, dict))
    fit.UNSAFE_clear_blocks('meta', indices=[i], clear_done=False, OVERRIDE=True)
    with pytest.raises(InvalidBlocksError, match=rf"(?s)reading \['x', 'meta'\] from its tabs; a part's pieces are "
                                                 rf"built by the part: part\.build\(\).*block {i}:"):
        probe(tmp_path / 'probe', fit, ev).build()


def test_stratifying_by_the_label_makes_a_tab_of_two_labels_valid_samples(tmp_path):
    tbl = bags(tmp_path)
    LABEL = ('meta', 'label')
    whole = DatatablePartition(datalake=str(tmp_path / 'whole'), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta')).build()
    with pytest.raises(ValueError, match="can carry only one label"):
        probe(tmp_path / 'refused', *built_parts(whole), tab_aggregation='mean').build()
    p = bag_partition(tmp_path / 'p', tbl, groupby=None, stratifyby=LABEL, balance='tabs')
    fit, ev = built_parts(p)
    pr = probe(tmp_path / 'probe', fit, ev, tab_aggregation='mean').build()
    # One sample per tab or piece, each of one label.
    assert len(pr.read('fit', 'labels')) == fit.n_tabs and len(pr.read('eval', 'labels')) == ev.n_tabs
