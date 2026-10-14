"""
`DatatablePartition`: whole tablets dealt to folds, by stratifyby / partitionby / balance / seed.

Each tablet here is one "slide": `rows` rows, every one carrying the slide's
patient and cohort in a `meta` slice -- the columns the partition reads,
named by its collator's stratifyby and partitionby roles.
"""
import json
from dataclasses import dataclass

import numpy as np
import pytest

from dbx.datablocks import Datablock
from dbx.datatables import DATASLICE, Datacollator, Datatablet, Datatable
from dbx.dataparts import (DatamapCarbonTable, DatamapTable, DatatablePart, DatatablePartition, MappedDatatablet,
                           MappedFeaturetablet)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Slide(Datatablet):
    TOPICS = {'meta': DATASLICE(patient='str', cohort='str', info='json')}

    @dataclass
    class VAR(Datatablet.VAR):
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
    TABLET = Slide

    @dataclass
    class VAR(Datatable.VAR):
        slides: list = None

    @property
    def n_tablets(self):
        return len(self.var.slides)

    def __tablet__(self, idx, **kw):
        s = self.var.slides[idx]
        return super().__tablet__(idx, patient=s[0], cohort=s[1], rows=s[2], mixed=bool(s[3:]) and s[3])


def table(tmp_path, slides=SLIDES):
    return Slides(datalake=str(tmp_path), spec={'slides': [list(s) for s in slides]}).build()


INFO = ('meta', 'info')
PATIENT, COHORT = INFO + ('patient',), INFO + ('cohort',)


def collator(stratifyby=None, partitionby=None):
    columns = {k: v for k, v in (('stratifyby', stratifyby), ('partitionby', partitionby)) if v is not None}
    return Datacollator(spec=dict(columns=columns)) if columns else None


def partition(tmp_path, tbl, stratifyby=None, partitionby=None, build=True, **spec):
    p = DatatablePartition(datalake=str(tmp_path), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta',
        collator=collator(stratifyby, partitionby), **spec))
    return p.build() if build else p


def folds_of(p):
    return [set(f) for f in p.read('tablets')]


def test_every_tablet_is_in_exactly_one_fold(tmp_path):
    p = partition(tmp_path, table(tmp_path), stratifyby=PATIENT, partitionby=COHORT)
    a, b = folds_of(p)
    assert a.isdisjoint(b) and a | b == set(range(len(SLIDES)))


def test_a_stratum_lands_in_one_fold(tmp_path):
    p = partition(tmp_path, table(tmp_path), stratifyby=PATIENT, balance='tablets')
    fold_of = {i: k for k, f in enumerate(folds_of(p)) for i in f}
    for patient in {s[0] for s in SLIDES}:
        tablets = [i for i, s in enumerate(SLIDES) if s[0] == patient]
        assert len({fold_of[i] for i in tablets}) == 1, patient


def test_the_fractions_hold_within_each_partition(tmp_path):
    p = partition(tmp_path, table(tmp_path), partitionby=COHORT, balance='tablets')
    for f in p.read('summary')['folds']:
        # 4 slides per cohort, halved: 2 and 2 in each fold.
        assert {c: v['tabs'] for c, v in f['partitions'].items()} == {'BRCA': 2, 'LUAD': 2, 'OV': 2}


def test_the_seed_decides_and_repeats(tmp_path):
    tbl = table(tmp_path)
    one = folds_of(partition(tmp_path / 'a', tbl, stratifyby=PATIENT, partitionby=COHORT, seed=0))
    again = folds_of(partition(tmp_path / 'b', tbl, stratifyby=PATIENT, partitionby=COHORT, seed=0))
    assert one == again
    seeds = {json.dumps([sorted(f) for f in folds_of(partition(tmp_path / f's{s}', tbl, partitionby=COHORT, seed=s))])
             for s in range(8)}
    assert len(seeds) > 1, "different seeds deal differently"


def test_balance_counts_rows_or_tablets(tmp_path):
    tbl = table(tmp_path)
    by_tablets = partition(tmp_path / 't', tbl, balance='tablets').read('summary')['folds']
    assert [f['tabs'] for f in by_tablets] == [6, 6]
    by_rows = partition(tmp_path / 'r', tbl, balance='rows').read('summary')['folds']
    rows = [f['rows'] for f in by_rows]
    assert sum(rows) == sum(s[2] for s in SLIDES) and min(rows) > 0


def test_the_summary_says_what_each_fold_holds(tmp_path):
    tbl = table(tmp_path)
    s = partition(tmp_path, tbl, stratifyby=PATIENT, partitionby=COHORT).read('summary')
    assert s['method'] == 'random' and s['n_tablets'] == len(SLIDES) and s['skipped'] == []
    tags = [t for f in s['folds'] for t in f['tags']]
    assert sorted(tags) == sorted(tbl.tablet(i).tag for i in range(len(SLIDES)))


def test_a_tablet_with_no_value_is_skipped_and_said(tmp_path, caplog):
    slides = SLIDES + [('p10', '', 10)]          # cohort None
    p = partition(tmp_path, table(tmp_path, slides), partitionby=COHORT)
    s = p.read('summary')
    assert [k['idx'] for k in s['skipped']] == [len(SLIDES)] and 'no partitionby value' in s['skipped'][0]['why']
    assert len(SLIDES) not in set().union(*folds_of(p))


def test_a_tablet_holding_two_values_is_split_into_pieces(tmp_path):
    slides = SLIDES[:-1] + [('p9', 'OV', 20, True)]     # its last row is patient 'other'
    p = partition(tmp_path, table(tmp_path, slides), stratifyby=PATIENT)
    entries = [e for f in p.read('tablets') for e in f]
    last = len(slides) - 1
    assert last not in entries
    assert sorted((e for e in entries if isinstance(e, dict)), key=lambda e: e['stratifyby']) == [
        {'tab': last, 'stratifyby': 'other'}, {'tab': last, 'stratifyby': 'p9'}]
    s = p.read('summary')
    assert s['n_pieces'] == 2 and sum(f['pieces'] for f in s['folds']) == 2
    assert {t for f in s['folds'] for t in f['tags'] if '#' in t} == {f"{p.datatable.tablet(last).tag}#other",
                                                                     f"{p.datatable.tablet(last).tag}#p9"}


def test_a_stratum_spanning_partitions_is_refused(tmp_path):
    slides = SLIDES + [('p1', 'OV', 10)]          # p1 is LUAD elsewhere
    with pytest.raises(ValueError, match="span more than one partition"):
        partition(tmp_path, table(tmp_path, slides), stratifyby=PATIENT, partitionby=COHORT)


def test_largest_first_takes_none_of_the_new_choices(tmp_path):
    with pytest.raises(ValueError, match="largest_first' takes no"):
        DatatablePartition(datalake=str(tmp_path), spec=dict(
            datatable=table(tmp_path), fractions=[0.5, 0.5], partition_slice='meta',
            method='largest_first', collator=collator(partitionby=COHORT)))


@pytest.mark.parametrize('mixed', [False, True])
def test_largest_first_deals_as_before(tmp_path, mixed):
    # It reads no partition column, so a mixed tablet is a tablet like any other to it, and is not split.
    slides = SLIDES[:-1] + [('p9', 'OV', 20, True)] if mixed else SLIDES
    p = partition(tmp_path, table(tmp_path, slides), method='largest_first')
    rows = [s[2] for s in SLIDES]
    # Descending rows, each to the fold with the larger deficit -- by hand:
    want, have, target = [[], []], [0, 0], [sum(rows) / 2] * 2
    for i in sorted(range(len(rows)), key=lambda i: rows[i], reverse=True):
        k = int(np.argmax([target[0] - have[0], target[1] - have[1]]))
        want[k].append(i)
        have[k] += rows[i]
    assert p.read('tablets') == [sorted(w) for w in want]


def test_a_collator_with_other_roles_is_refused(tmp_path):
    c = Datacollator(spec=dict(columns={'stratifyby': PATIENT, 'signals': [INFO]}))
    with pytest.raises(ValueError, match="declares \\['signals'\\] too"):
        DatatablePartition(datalake=str(tmp_path), spec=dict(
            datatable=table(tmp_path), fractions=[0.5, 0.5], partition_slice='meta', collator=c))


def test_a_partition_of_constant_tablets_is_what_it_was(tmp_path):
    """Pinned from dbx 07cd119, before tabs could be split: no deal of constant tabs moves.

    Its hash has moved twice. On 2026-10-09 *coalesce*, *min_tab_rows* and *max_tab_rows* became
    fields, which a specialization pinning them to None reached past. On 2026-10-10 the roles
    were renamed -- groupby is stratifyby, the old stratifyby partitionby -- and VERSION became 2,
    so that a collator naming stratifyby, which hashes as before and means the opposite, is not
    taken for one built before: what was built is rebuilt. The deal is what stays, pinned here.
    """
    tbl = table(tmp_path)
    p = partition(tmp_path / 'gs', tbl, stratifyby=PATIENT, partitionby=COHORT)
    assert '3318e720fcb934c1fd3b563c7dc3f36487c8f63733a4414fdd0c9bccecba6a57' not in p.specialization_hashes()
    assert p.read('tablets') == [[3, 6, 7, 9, 10, 11], [0, 1, 2, 4, 5, 8]]
    assert p.part(0).tablet_indices == [3, 6, 7, 9, 10, 11]
    assert 'n_pieces' not in p.read('summary')
    p = partition(tmp_path / 'none', tbl)
    assert p.read('tablets') == [[0, 2, 3, 4, 5, 7, 9, 11], [1, 6, 8, 10]]
    assert p.part(0).tablet_indices == [0, 2, 3, 4, 5, 7, 9, 11]


# Bags: tablets whose rows each carry their own patient and cohort -- several of each in a tablet.

class Bag(Datatablet):
    TOPICS = {'meta': DATASLICE(row='int', patient='str', label='int', info='json'),
              'x': DATASLICE(row='int', features='ndarray:float32')}

    @dataclass
    class VAR(Datatablet.VAR):
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
    TABLET = Bag

    @dataclass
    class VAR(Datatable.VAR):
        bags: list = None

    @property
    def n_tablets(self):
        return len(self.var.bags)

    def __tablet__(self, idx, **kw):
        return super().__tablet__(idx, bag=idx, rows=[list(r) for r in self.var.bags[idx]])


def bags(tmp_path):
    return Bags(datalake=str(tmp_path), spec={'bags': [[list(r) for r in b] for b in BAGS]}).build()


def bag_partition(tmp_path, tbl, stratifyby=PATIENT, partitionby=COHORT, **spec):
    return DatatablePartition(datalake=str(tmp_path), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta',
        collator=collator(stratifyby, partitionby), **spec)).build()


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
    assert [(s['piece'], s['rows']) for s in skipped] == [({'tab': 2, 'stratifyby': None, 'partitionby': 'B'}, 1)]
    assert 'no stratifyby value' in skipped[0]['why']
    assert sorted(rows[0] + rows[1]) == sorted(r for r, (patient, _) in every.items() if patient is not None)
    assert len(rows[0]) + len(rows[1]) + sum(s['rows'] for s in skipped) == len(every)


def test_a_stratums_rows_are_in_one_fold_across_tablets_and_pieces(tmp_path):
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
        for i, entry in enumerate(fold.tablet_indices):
            if not isinstance(entry, dict):
                continue
            n_pieces += 1
            piece = fold.tablet(i)
            meta, x = piece.data('meta')['meta'], piece.data('x')['x']
            assert {(m['patient'], m['cohort']) for m in meta['info']} == {(entry['stratifyby'], entry['partitionby'])}
            assert piece.n_rows('meta') == piece.n_rows('x') == len(meta['row']) > 0
            assert meta['row'] == x['row']              # row i of one slice is row i of the other
            assert piece.tag == f"{p.datatable.tablet(entry['tab']).tag}#{entry['stratifyby']}#{entry['partitionby']}"
    assert n_pieces == p.read('summary')['n_pieces'] - 1        # all but the skipped one


def test_a_pieces_source_slice_points_back_and_takes_the_same_piece_of_the_source(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    for part in built_parts(p):
        for i, entry in enumerate(part.tablet_indices):
            if not isinstance(entry, dict):
                continue
            piece, source = part.tablet(i), p.datatable.tablet(entry['tab'])
            rows = piece.source_index()[:, 1]
            assert rows.dtype == np.int64 and list(rows) == sorted(rows)
            assert set(piece.data(('source', 'tab'))['source']['tab']) == {entry['tab']}
            # The indices, applied to the source, are the piece -- what a tablet built from the source row by row reuses.
            src = source.data('x')['x']
            assert [src['row'][r] for r in rows] == piece.data('x')['x']['row']
            info = source.data('meta')['meta']['info']
            assert [r for r, v in enumerate(info) if (v['patient'], v['cohort']) == (entry['stratifyby'], entry['partitionby'])] == list(rows)


def test_fold_is_the_name_part_had_first(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    assert p.fold(1).hash == p.part(1).hash


def test_the_fractions_hold_within_each_partition_counting_pieces(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path), stratifyby=None, balance='tablets')
    folds = p.read('summary')['folds']
    # Split by cohort only: A is bags 0, 3 and a piece each of 2 and 5; B bags 1, 4 and the others.
    for cohort in COHORTS:
        assert [f['partitions'][cohort]['tabs'] for f in folds] == [2, 2], cohort


def test_the_same_seed_deals_the_same_pieces_with_the_same_hashes(tmp_path):
    tbl = bags(tmp_path)
    one, two = bag_partition(tmp_path / 'a', tbl, seed=5), bag_partition(tmp_path / 'b', tbl, seed=5)
    assert one.read('tablets') == two.read('tablets')
    pieces = [e for f in one.read('tablets') for e in f if isinstance(e, dict)]
    assert pieces
    again = DatatablePartition(datalake=str(tmp_path / 'a'), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta', collator=collator(PATIENT, COHORT), seed=5))
    def piece_hashes(partition):
        return [partition.part(k).tablet(i).hash for k in range(partition.n_folds())
                for i, e in enumerate(partition.tablets_indices(k)) if isinstance(e, dict)]
    assert piece_hashes(one) == piece_hashes(again)
    assert len(set(piece_hashes(one))) == len(pieces)


def test_a_fold_builds_its_pieces_in_parallel(tmp_path):
    from dbx.dataparts import DatatablePart
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    for k in range(p.n_folds()):
        fold = DatatablePart(datalake=str(tmp_path / 'p'), spec=dict(partition=p, fold=k),
                             parallelization='multiprocessing', n_workers=2)
        assert fold.hash == p.part(k).hash
        fold.build()
        assert all(p.part(k).valid_tablet(i, validation='valid') for i in range(fold.n_tablets))


def test_a_piece_that_cannot_be_recovered_says_so(tmp_path):
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    datamap = DatamapTable(datalake=str(tmp_path / 'p'), spec=dict(partition=p, fold=0))
    with pytest.raises(ValueError, match="cannot be recovered"):
        datamap._entry_rows_({'tab': 0, 'stratifyby': 'p4', 'partitionby': 'B'}, 0)


def _features(url):
    from test_probes import DummyModelEvaluatorFactory, DummySampleTable
    from dbx import Featuretable
    samples = DummySampleTable(datalake=url, spec=dict(samples_per_tablet=4), tag='samples').build()
    return Featuretable(datalake=url, devices=['cpu'], tag='features', spec=dict(
        upstream=samples, evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
        collator=Datacollator(spec=dict(columns={'signals': [('samples', 'samples')], 'labels': [('labels', 'labels')]})))).build()


@pytest.mark.parametrize('coalesce', [None, 'pack'])
def test_a_mixed_feature_tablet_is_cut_layer_by_layer(tmp_path, coalesce):
    """A piece of a feature tablet is the same rows of its features and, as its upstream, of its samples.

    Every tablet alternates its label, so every one is cut, by a column upstream of
    the features: each piece copies its own rows of the features, and its upstream
    copies the same rows of the samples -- the layers kept, nothing re-evaluated.
    """
    url = str(tmp_path)
    features = _features(url)
    # Seed 2 splits each tablet's pieces across the folds, so a packed tablet pools pieces of both tablets.
    spec = dict(coalesce='pack', min_tablet_rows=3) if coalesce else {}
    p = DatatablePartition(datalake=url, spec=dict(
        datatable=features, fractions=[0.5, 0.5], partition_slice='features', seed=2,
        collator=Datacollator(spec=dict(columns={'partitionby': ('labels', 'labels')}, recursive=True)), **spec)).build()
    copies = 0
    for k in range(p.n_folds()):
        part = p.part(k).build()
        for i in range(part.n_tablets):
            tablet = part.tablet(i)
            if part.datamap.whole(i) is not None:
                continue
            copies += 1
            assert isinstance(tablet, MappedFeaturetablet) and isinstance(tablet.var.upstream, MappedDatatablet)
            assert 'source' not in tablet.slices() and 'source' in tablet.var.upstream.slices(), "written once, below"
            src = tablet.source_index()                         # read through the upstream
            got = tablet.data('features', 'samples', 'labels')
            feats = [features.tablet(int(t)).data('features')['features'] for t, _ in src]
            ups = [features.tablet(int(t)).var.upstream.data('samples', 'labels') for t, _ in src]
            for j, (t, r) in enumerate(src):
                for c, v in got['features'].items():
                    assert np.array_equal(v[j], feats[j][c][r])
                assert np.array_equal(got['samples']['samples'][j], ups[j]['samples']['samples'][r])
                assert got['labels']['labels'][j] == ups[j]['labels']['labels'][r]
    assert copies


def test_a_carbon_map_refuses_a_table_not_laid_out_alike(tmp_path):
    from test_probes import DummySampleTable
    url = str(tmp_path)
    features = _features(url)
    p = DatatablePartition(datalake=url, spec=dict(
        datatable=features, fractions=[0.5, 0.5], partition_slice='features', seed=2,
        collator=Datacollator(spec=dict(columns={'partitionby': ('labels', 'labels')}, recursive=True)))).build()
    datamap = p.part(0).datamap
    alike = DatamapCarbonTable(datalake=url, spec=dict(template=datamap, datatable=features.var.upstream))
    assert [t for t, _ in alike.record(0)] == [t for t, _ in datamap.record(0)]
    unlike = DummySampleTable(datalake=url, spec=dict(samples_per_tablet=5), tag='unlike').build()
    with pytest.raises(ValueError, match="not laid out alike"):
        DatamapCarbonTable(datalake=url, spec=dict(template=datamap, datatable=unlike)).record(0)


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
    assert any(isinstance(e, dict) for e in fit.tablet_indices + ev.tablet_indices)
    pr = probe(tmp_path / 'probe', fit, ev).build()
    assert sorted(pr.read('classes').tolist()) == [0, 1]
    assert len(pr.read('fit', 'labels')) == fit.n_rows('meta')
    assert len(pr.read('eval', 'labels')) == ev.n_rows('meta')


def test_pieces_of_one_tablet_in_two_folds_are_disjoint_and_one_piece_twice_is_not(tmp_path):
    tbl = bags(tmp_path)
    for seed in range(20):
        p = bag_partition(tmp_path / f's{seed}', tbl, seed=seed)
        sources = [{e['tab'] for e in f if isinstance(e, dict)} for f in p.read('tablets')]
        if sources[0] & sources[1]:
            break
    else:
        pytest.fail("no seed puts two pieces of one tablet in different folds")
    fit, ev = built_parts(p)
    probe(tmp_path / 'ok', fit, ev)._check_disjoint_()
    with pytest.raises(ValueError, match="share"):
        probe(tmp_path / 'twice', fit, fit)._check_disjoint_()


def test_a_probe_says_a_piece_is_not_built_before_it_starts(tmp_path):
    from dbx.datablocks import InvalidBlocksError
    p = bag_partition(tmp_path / 'p', bags(tmp_path))
    fit, ev = built_parts(p)
    # A built fold, and one of its pieces gone since: the fold vouches, the piece does not.
    i = next(i for i, e in enumerate(fit.tablet_indices) if isinstance(e, dict))
    fit.UNSAFE_clear_blocks('meta', indices=[i], clear_done=False, OVERRIDE=True)
    with pytest.raises(InvalidBlocksError, match=rf"(?s)reading \['x', 'meta'\] from its tablets; a part's pieces are "
                                                 rf"built by the part: part\.build\(\).*block {i}:"):
        probe(tmp_path / 'probe', fit, ev).build()


def test_a_part_of_a_partition_not_built_is_not_valid(tmp_path):
    """Asked, it says no -- rather than reading tablets its partition has not dealt."""
    p = partition(tmp_path, table(tmp_path), build=False)
    assert not p.valid() and not p.part(0).dealt
    assert p.part(0).valid() is False
    assert p.part(0).specialize().valid() is False, "nothing to specialize, and nothing read"
    p.build()
    assert p.part(0).dealt and p.part(0).build().valid()

