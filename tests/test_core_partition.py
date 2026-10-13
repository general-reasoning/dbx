"""
`DatatableCorePartition`: rows clustered by their values, dealt nearest first into folds of fresh tablets.

Each tablet here holds points from four well-separated blobs, one per label, so
the clusters k-means finds are the labels, and a row's distance to its
center says how central it is.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from dbx.datatables import DATASLICE, Datacollator, Datatablet, Datatable
from dbx.dataparts import DatatableCarbonCorePartition, DatatableCorePart, DatatableCorePartition, DatatableCoreTablet


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


BLOBS = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)


class Points(Datatablet):
    TOPICS = {'pts': DATASLICE(x='ndarray:float32', patient='str', label='int', site='int'),
              'img': DATASLICE(big='ndarray:float32')}

    @dataclass
    class VAR(Datatablet.VAR):
        k: int = 0
        n: int = 40
        offset: int = 0         # shifts every point: a table laid out alike, holding other values

    def __build__(self):
        rng = np.random.default_rng(self.var.k)
        with self.slice_writers() as writers:
            for i in range(self.var.n):
                label = (i + self.var.k) % 4
                x = BLOBS[label] + rng.normal(scale=1.0, size=2).astype(np.float32) + self.var.offset
                # site cuts across the blobs: every site holds rows of every label.
                writers['pts'].write({'x': x, 'patient': f"p{self.var.k}_{label}", 'label': label,
                                      'site': (i // 4) % 3})
                writers['img'].write({'big': np.full(8, self.var.k * 1000 + i, dtype=np.float32)})


class PointTable(Datatable):
    TABLET = Points

    @dataclass
    class VAR(Datatable.VAR):
        n_tablets: int = 6
        n: int = 40
        offset: int = 0

    @property
    def n_tablets(self):
        return self.var.n_tablets

    def __tablet__(self, idx, **kw):
        return super().__tablet__(idx, k=idx, n=self.var.n, offset=self.var.offset)


def points(tmp_path, **spec):
    return PointTable(datalake=str(tmp_path / 'points'), spec=spec).build()


X, LABEL, PATIENT, SITE = ('pts', 'x'), ('pts', 'label'), ('pts', 'patient'), ('pts', 'site')


def core(tmp_path, table, fractions=(0.8, 0.2), build=True, coreby=(X,), stratifyby=None, partitionby=None,
         carry=None, name='core', parallelization=None, n_workers=1, **spec):
    columns = {'coreby': list(coreby)}
    if stratifyby is not None:
        columns['stratifyby'] = stratifyby
    if partitionby is not None:
        columns['partitionby'] = partitionby
    if carry is not None:
        columns['carry'] = list(carry)
    p = DatatableCorePartition(datalake=str(tmp_path / name), parallelization=parallelization, n_workers=n_workers,
                               spec=dict(datatable=table, fractions=list(fractions), partition_slice='pts',
                                         collator=Datacollator(spec=dict(columns=columns)),
                                         n_clusters=4, **spec))
    return p.build() if build else p


def every_row(table):
    return {(t, r) for t in range(table.n_tablets) for r in range(table.var.n)}


def test_fractions_summing_to_one_deal_every_row_once_and_sample_every_cluster(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl)
    layouts = [p.layout(k) for k in range(2)]
    dealt = [{(int(t), int(r)) for t, r, _ in fl} for fl in layouts]
    assert dealt[0].isdisjoint(dealt[1]) and dealt[0] | dealt[1] == every_row(tbl)
    for fl in layouts:          # source order: a core tablet reads a few neighbouring tablets
        assert [tuple(r) for r in fl[:, :2]] == sorted(tuple(r) for r in fl[:, :2])
    sizes = [np.bincount(fl[:, 2], minlength=4) for fl in layouts]
    for c in range(4):
        n = sizes[0][c] + sizes[1][c]
        assert abs(sizes[0][c] - 0.8 * n) <= 1, (c, sizes)
    s = p.read('summary')
    assert s['n_dealt'] == len(every_row(tbl)) and s['n_undealt'] == 0


def test_the_clusters_are_the_blobs(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl)
    labels = {(t, r): l for t in range(tbl.n_tablets) for r, l in enumerate(tbl.tablet(t).data(LABEL)['pts']['label'])}
    # Each cluster holds one label.
    by_cluster = {}
    for k in range(2):
        for t, r, c in p.layout(k):
            by_cluster.setdefault(int(c), set()).add(labels[(int(t), int(r))])
    assert sorted(len(v) for v in by_cluster.values()) == [1, 1, 1, 1]


def test_a_coreset_takes_the_rows_nearest_each_center(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, fractions=(0.1, 0.1))
    centers = p.read('centers')
    xs = {(t, r): x for t in range(tbl.n_tablets) for r, x in enumerate(tbl.tablet(t).data(X)['pts']['x'])}
    taken = {(int(t), int(r)): int(c) for k in range(2) for t, r, c in p.layout(k)}
    assert abs(len(taken) - 0.2 * len(xs)) <= 8          # 10% of each of 4 clusters, per fold, rounded
    for c in range(4):
        dist = {key: float(np.linalg.norm(x - centers[c])) for key, x in xs.items()
                if int(np.argmin(np.linalg.norm(centers - x, axis=1))) == c}
        inside = [d for key, d in dist.items() if key in taken]
        outside = [d for key, d in dist.items() if key not in taken]
        assert inside and max(inside) <= min(outside)
    s = p.read('summary')
    assert s['n_undealt'] == len(xs) - len(taken)


def test_a_group_is_dealt_whole(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, stratifyby=PATIENT, fractions=(0.5, 0.5))
    patients = {(t, r): pa for t in range(tbl.n_tablets) for r, pa in enumerate(tbl.tablet(t).data(PATIENT)['pts']['patient'])}
    fold_of = {}
    for k in range(2):
        for t, r, _ in p.layout(k):
            fold_of.setdefault(patients[(int(t), int(r))], set()).add(k)
    assert len(fold_of) == len(set(patients.values())) and all(len(v) == 1 for v in fold_of.values())


def test_core_tablets_carry_what_the_collator_names_and_point_back(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, carry=[LABEL], rows_per_tablet=16)
    assert p.carried_slices() == ['pts'] and p.read('columns') == {'pts': {'x': 'ndarray:float32', 'label': 'int'}}
    for k in range(2):
        part = DatatableCorePart(datalake=str(tmp_path / 'core'), spec=dict(partition=p, fold=k),
                                 parallelization='multiprocessing', n_workers=2)
        assert part.hash == p.part(k).hash and isinstance(part.tablet(0), DatatableCoreTablet)
        part.build()
        assert part.slices() == ('pts', 'source')
        fl = p.layout(k)
        assert part.n_tablets == -(-len(fl) // 16)
        n = 0
        for i in range(part.n_tablets):
            piece = part.tablet(i)
            assert piece.valid() and sorted(piece.slices()) == ['pts', 'source']
            pts = piece.data('pts')['pts']
            assert sorted(pts) == ['label', 'x']            # no patient, no img
            src = piece.source_index()
            assert len(src) == piece.n_rows('pts') == piece.n_rows('source')
            assert [tuple(r) for r in src] == [tuple(r) for r in fl[n:n + len(src), :2]]
            for (t, r), x, label in zip(src, pts['x'], pts['label']):
                row = tbl.tablet(int(t)).data('pts')['pts']
                np.testing.assert_array_equal(x, row['x'][r])
                assert label == row['label'][r]
            n += len(src)
        assert n == len(fl)


def test_core_tablets_declare_the_carried_columns_as_the_table_declares_them(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, carry=[LABEL], rows_per_tablet=16)
    assert p.carried_declarations() == {'pts': {'x': 'ndarray:float32', 'label': 'int'}}
    part = p.part(0)
    assert part.tablet(0).declared_schema('pts') == {'x': 'ndarray:float32', 'label': 'int'}
    assert part.readable_slices() == {'pts': {'x': 'ndarray:float32', 'label': 'int'}, 'source': {'tab': 'int', 'row': 'int'}}


def test_distributed_clustering_finds_the_same_blobs_and_leaves_no_scratch(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, clustering='distributed', parallelization='multiprocessing', n_workers=2)
    labels = {(t, r): l for t in range(tbl.n_tablets) for r, l in enumerate(tbl.tablet(t).data(LABEL)['pts']['label'])}
    by_cluster = {}
    for k in range(2):
        for t, r, c in p.layout(k):
            by_cluster.setdefault(int(c), set()).add(labels[(int(t), int(r))])
    assert sorted(len(v) for v in by_cluster.values()) == [1, 1, 1, 1]
    assert not p.fs.exists(f"{p.anchorkeypath}/.scratch")
    assert p.read('summary')['clustering'] == 'distributed'


def carbon(tmp_path, table, template, carry=(('img', 'big'),), name='core', build=True, **spec):
    q = DatatableCarbonCorePartition(datalake=str(tmp_path / name), spec=dict(
        datatable=table, template=template, collator=Datacollator(spec=dict(columns={'carry': list(carry)})), **spec))
    return q.build() if build else q


def test_a_table_laid_out_alike_takes_the_template_s_layout_without_clustering(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, rows_per_tablet=16)
    other = PointTable(datalake=str(tmp_path / 'other'), spec=dict(offset=100)).build()
    q = carbon(tmp_path, other, p)
    assert q.n_folds() == p.n_folds() and q.rows_per_tablet == 16, "the template's folds and tablets"
    assert q.read('summary')['template'] == p.hash
    for k in range(2):
        np.testing.assert_array_equal(q.layout(k), p.layout(k))
        assert q.tablets_indices(k) == p.tablets_indices(k)
        part = q.part(k).build()
        assert part.slices() == ('img', 'source')
        src = np.concatenate([part.tablet(i).source_index() for i in range(part.n_tablets)])
        np.testing.assert_array_equal(src, p.layout(k)[:, :2])
        big = np.concatenate([part.tablet(i).data('img', concat=True)['img']['big'] for i in range(part.n_tablets)])
        np.testing.assert_array_equal(big[:, 0], src[:, 0] * 1000 + src[:, 1])
    unlike = PointTable(datalake=str(tmp_path / 'unlike'), spec=dict(n=39)).build()
    with pytest.raises(ValueError, match="not laid out alike"):
        carbon(tmp_path, unlike, p, name='unlike_core')


def test_its_own_rows_per_tablet_cuts_the_same_rows_differently(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, rows_per_tablet=16)
    q = carbon(tmp_path, tbl, p, carry=[X], rows_per_tablet=5)
    for k in range(2):
        np.testing.assert_array_equal(q.layout(k), p.layout(k))
        assert all(end - start <= 5 for start, end in q.tablets_indices(k))
        assert len(q.tablets_indices(k)) > len(p.tablets_indices(k))


def test_a_carbon_copy_is_a_template_too(tmp_path):
    """A chain of tables computed one from another all take the first one's layout."""
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, rows_per_tablet=16)
    other = PointTable(datalake=str(tmp_path / 'other'), spec=dict(offset=100)).build()
    q = carbon(tmp_path, other, p, name='q')
    third = PointTable(datalake=str(tmp_path / 'third'), spec=dict(offset=200)).build()
    r = carbon(tmp_path, third, q, carry=[X], name='r')
    assert r.rows_per_tablet == 16 and r.read('summary')['template'] == q.hash
    for k in range(2):
        np.testing.assert_array_equal(r.layout(k), p.layout(k))
        x = np.concatenate([r.part(k).build().tablet(i).data('pts', concat=True)['pts']['x']
                            for i in range(len(r.tablets_indices(k)))])
        assert (x >= 200 - 10).all(), "its own table's values"


@pytest.mark.parametrize('spec, match', [
    (dict(carry=()), "no 'carry'"),
    (dict(rows_per_tablet=0), "rows_per_tablet must be"),
])
def test_what_a_carbon_copy_refuses(tmp_path, spec, match):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, build=False)
    with pytest.raises(ValueError, match=match):
        carbon(tmp_path, tbl, p, build=False, **spec)


def test_a_carbon_copy_takes_only_carry_and_a_core_template(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, build=False)
    with pytest.raises(ValueError, match="names only 'carry'"):
        DatatableCarbonCorePartition(datalake=str(tmp_path / 'core'), spec=dict(
            datatable=tbl, template=p, collator=Datacollator(spec=dict(columns={'coreby': [X], 'carry': [X]}))))
    with pytest.raises(ValueError, match="template must be a DatatableCorePartition"):
        DatatableCarbonCorePartition(datalake=str(tmp_path / 'core'), spec=dict(
            datatable=tbl, template=tbl, collator=Datacollator(spec=dict(columns={'carry': [X]}))))


def test_a_core_partition_s_layout_is_retired_but_its_hash_is_kept(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, build=False)
    with pytest.raises(ValueError, match="layout is retired"):
        core(tmp_path, tbl, build=False, name='again', layout=p)
    assert 'layout' in DatatableCorePartition.VAR.__dataclass_fields__, (
        "a VAR's defaults are its identity: removing the field moves every core partition's hash")


def test_the_same_spec_is_the_same_partition(tmp_path):
    tbl = points(tmp_path)
    one, two = core(tmp_path, tbl, name='a'), core(tmp_path, tbl, name='b')
    assert one.hash == two.hash
    for k in range(2):
        np.testing.assert_array_equal(one.layout(k), two.layout(k))
        assert one.core_tablet(k, 0).hash == two.core_tablet(k, 0).hash
    assert core(tmp_path, tbl, name='c', seed=1, build=False).hash != one.hash


@pytest.mark.parametrize('spec, match', [
    (dict(fractions=(0.8, 0.4)), "sum to at most 1"),
    (dict(coreby=()), "no 'coreby'"),
    (dict(clustering='gpu'), "clustering must be one of"),
    (dict(method='largest_first'), "takes no method or balance"),
])
def test_what_a_core_partition_refuses(tmp_path, spec, match):
    tbl = points(tmp_path)
    with pytest.raises(ValueError, match=match):
        core(tmp_path, tbl, build=False, **spec)


def test_a_probe_fits_on_core_parts_of_a_feature_table(tmp_path):
    """Features clustered, labels carried from upstream: the core tablets hold both, and read nothing upstream."""
    from test_probes import DummyModelEvaluatorFactory, DummySampleTable
    from dbx import FeatureAffineLogisticProbe, Featuretable
    url = str(tmp_path)
    samples = DummySampleTable(datalake=url, spec=dict(samples_per_tablet=40), tag='samples').build()
    features = Featuretable(datalake=url, devices=['cpu'], tag='features', spec=dict(
        upstream=samples, evaluator_factory=DummyModelEvaluatorFactory(spec=dict(capture_final=True)),
        collator=Datacollator(spec=dict(columns={'signals': [('samples', 'samples')], 'labels': [('labels', 'labels')]})))).build()
    p = DatatableCorePartition(datalake=url, spec=dict(
        datatable=features, fractions=[0.5, 0.5], partition_slice='features', n_clusters=4, rows_per_tablet=16,
        collator=Datacollator(spec=dict(columns={'coreby': [('features', 'final')], 'carry': [('labels', 'labels')]},
                                        recursive=True)))).build()
    fit, ev = p.part(0).build(), p.part(1).build()
    assert fit.slices() == ('features', 'labels', 'source')
    probe = FeatureAffineLogisticProbe(datalake=url, spec=dict(
        fit_table=fit, eval_table=ev,
        collator=Datacollator(spec=dict(columns={'signals': [('features', 'final')], 'labels': [('labels', 'labels')]})))).build()
    assert sorted(probe.read('classes').tolist()) == [0, 1]
    assert len(probe.read('fit', 'labels')) + len(probe.read('eval', 'labels')) == 80


def column(tbl, pair):
    return {(t, r): v for t in range(tbl.n_tablets) for r, v in enumerate(tbl.tablet(t).data(pair)[pair[0]][pair[1]])}


@pytest.mark.parametrize('fractions', [(0.5, 0.5), (0.1, 0.1)])
def test_the_fractions_hold_within_each_partition(tmp_path, fractions):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, fractions=fractions, partitionby=SITE)
    site = column(tbl, SITE)
    total = np.bincount(list(site.values()))
    for k, f in enumerate(fractions):
        got = np.bincount([site[(int(t), int(r))] for t, r, _ in p.layout(k)], minlength=len(total))
        # Each partition's cells carry their rounding into the next: within a row of its share.
        assert np.all(np.abs(got - f * total) <= 1), (k, got, f * total)


def test_strata_within_partitions_are_dealt_whole_and_a_stratum_across_partitions_is_refused(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, fractions=(0.5, 0.5), stratifyby=PATIENT, partitionby=LABEL)
    patient = column(tbl, PATIENT)
    fold_of = {}
    for k in range(2):
        for t, r, _ in p.layout(k):
            fold_of.setdefault(patient[(int(t), int(r))], set()).add(k)
    assert all(len(v) == 1 for v in fold_of.values())
    with pytest.raises(ValueError, match="spans more than one partition"):
        core(tmp_path, tbl, name='across', stratifyby=PATIENT, partitionby=SITE)


def test_coverage_measures_how_the_folds_fill_out_the_table(tmp_path):
    tbl = points(tmp_path)
    # Every cluster a candidate, so the distances are exact, and checked by brute force.
    p = core(tmp_path, tbl, fractions=(0.5, 0.5), coverage_sample=100, coverage_clusters=4)
    cov = p.read('coverage')
    assert sorted(cov) == sorted(['levels', 'sample', 'scale', 'coverage', 'coverage_mean', 'baseline',
                                  'baseline_mean', 'separation', 'cluster_rows', 'cluster_tv', 'empty_clusters'])
    assert cov['coverage'].shape == (3, 4) and cov['separation'].shape == (2, 4)
    np.testing.assert_array_equal(cov['coverage'][2], 0)        # the folds together are every row
    xs = column(tbl, X)
    sample = [tuple(int(v) for v in r) for r in cov['sample']]
    assert len(sample) == 100 and len(set(sample)) == 100
    for k in range(2):
        core_x = np.stack([xs[(int(t), int(r))] for t, r, _ in p.layout(k)])
        exact = [np.linalg.norm(core_x - xs[i], axis=1).min() for i in sample]
        np.testing.assert_allclose(cov['coverage'][k], np.quantile(exact, cov['levels']), rtol=1e-5, atol=1e-5)
    every = np.stack([xs[i] for i in sample])
    table_x = np.stack(list(xs.values()))
    keys = list(xs)
    nn = [np.sort(np.linalg.norm(table_x - every[j], axis=1))[1] for j in range(len(sample))]
    np.testing.assert_allclose(cov['scale'], np.quantile(nn, cov['levels']), rtol=1e-5, atol=1e-5)
    assert cov['cluster_rows'][0].sum() == len(xs) and cov['cluster_rows'][1:].sum() == len(xs)
    assert np.all(cov['cluster_tv'] < 0.05) and np.all(cov['empty_clusters'] == 0)
    assert np.all(cov['separation'] > 0)


def test_a_coreset_s_coverage_has_a_random_baseline_to_be_read_against(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, fractions=(0.1, 0.1), coverage_sample=100)
    cov = p.read('coverage')
    assert np.all(np.isfinite(cov['coverage'])) and np.all(np.isfinite(cov['baseline']))
    assert np.all(cov['coverage'][:, -1] >= cov['coverage'][:, 0])       # max >= median
    assert np.all(cov['coverage'] > 0)


def test_coverage_is_measured_in_the_workers_too_and_taken_by_a_carbon_copy(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, clustering='distributed', parallelization='multiprocessing', n_workers=2,
             coverage_sample=50)
    cov = p.read('coverage')
    np.testing.assert_array_equal(cov['coverage'][2], 0)
    other = PointTable(datalake=str(tmp_path / 'other'), spec=dict(offset=100)).build()
    q = carbon(tmp_path, other, p)
    for key, value in q.read('coverage').items():
        np.testing.assert_array_equal(value, cov[key])


def test_a_core_part_and_a_carbon_copy_s_build_with_their_partitions_workers(tmp_path):
    tbl = points(tmp_path)
    p = core(tmp_path, tbl, build=False, parallelization='multiprocessing', n_workers=2)
    q = DatatableCarbonCorePartition(datalake=str(tmp_path / 'core'), parallelization='multithreading', n_workers=4,
                                     spec=dict(datatable=tbl, template=p,
                                               collator=Datacollator(spec=dict(columns={'carry': [X]}))))
    assert (p.part(0).parallelization, p.part(0).n_workers) == ('multiprocessing', 2)
    assert (q.part(0).parallelization, q.part(0).n_workers) == ('multithreading', 4)
