"""
`DatatablePartition`'s *coalesce*: a cut tab made whole again, small entries packed -- and the probe that reads them.

*coalesce* decides what a part's tabs are once the deal is done, never which
part a row goes to. So the tests compare each mode's parts with *coalesce*
None's, row by row, by the rows of the table their ``source`` names.

The table is DATAPARTITION.md's: five tabs of rows (stratum, partition), s1 and s3
in p1, the others in p2, with a feature and a label per row for a probe -- 0 and 1 by turns,
so a stratum's rows carry both.
"""
from dataclasses import dataclass, field

import numpy as np
import pytest

from dbx.datablocks import Datablock
from dbx.datatables import DATASLICE, Datacollator, Datatab, Datatable
from dbx.dataparts import DatatablePackTab, DatatablePartition, DatatabPiece
from test_partition_methods import COHORT, PATIENT, bag_partition, bags, built_parts  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


#: Each tab's rows, (stratum, partition).
TABS = [
    [('s1', 'p1')] * 4,
    [('s2', 'p2'), ('s2', 'p2'), ('s3', 'p1'), ('s3', 'p1')],
    [('s1', 'p1')] * 3,
    [('s4', 'p2'), ('s4', 'p2'), ('s6', 'p2'), ('s6', 'p2')],
    [('s5', 'p2'), ('s3', 'p1'), ('s5', 'p2'), ('s5', 'p2')],
]
STRATUM, PARTITION = ('meta', 'stratum'), ('meta', 'partition')


class RowsTab(Datatab):
    TOPICS = {'meta': DATASLICE(row='int', stratum='str', partition='str', label='int'),
              'x': DATASLICE(row='int', features='ndarray:float32')}

    @dataclass
    class VAR(Datatab.VAR):
        tab: int = 0
        rows: list = field(default_factory=list)

    def __build__(self):
        with self.slice_writers() as writers:
            for i, (g, s) in enumerate(self.var.rows):
                row, label = 100 * self.var.tab + i, i % 2      # a stratum's rows carry both labels
                x = np.random.default_rng(row).normal(size=3).astype(np.float32) + 3 * label
                writers['meta'].write({'row': row, 'stratum': g, 'partition': s, 'label': label})
                writers['x'].write({'row': row, 'features': x})


class RowsTable(Datatable):
    TAB = RowsTab

    @dataclass
    class VAR(Datatable.VAR):
        tabs: list = None

    @property
    def n_tabs(self):
        return len(self.var.tabs)

    def __tab__(self, idx, **kw):
        return super().__tab__(idx, tab=idx, rows=[list(r) for r in self.var.tabs[idx]])


def rows_table(tmp_path, tabs=TABS):
    return RowsTable(datalake=str(tmp_path), spec={'tabs': [[list(r) for r in t] for t in tabs]}).build()


def split(tmp_path, tbl, stratifyby=STRATUM, partitionby=None, seed=4, **spec):
    columns = {k: v for k, v in (('stratifyby', stratifyby), ('partitionby', partitionby)) if v is not None}
    return DatatablePartition(datalake=str(tmp_path), spec=dict(
        datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta', seed=seed,
        collator=Datacollator(spec=dict(columns=columns)), **spec)).build()


def rows_of(part) -> list:
    """A built part's rows, as the (tab, row) of the partitioned table each one is."""
    return sorted(map(tuple, part.source_index().tolist()))


def test_coalesce_moves_no_row(tmp_path):
    """Whatever the mode, part k holds exactly the rows coalesce=None's part k does."""
    tbl = rows_table(tmp_path)
    for partitionby in (None, PARTITION):
        cut = [rows_of(p) for p in built_parts(split(tmp_path / f'none{bool(partitionby)}', tbl, partitionby=partitionby))]
        for mode, spec in (('whole', {}), ('pack', {'min_tab_rows': 3, 'max_tab_rows': 4})):
            p = split(tmp_path / f'{mode}{bool(partitionby)}', tbl, partitionby=partitionby, coalesce=mode, **spec)
            assert [rows_of(part) for part in built_parts(p)] == cut, (mode, partitionby)


def test_whole_rejoins_a_cut_tab_whose_pieces_went_to_one_part(tmp_path):
    """DATAPARTITION.md's stratifyby figure: tabs 3 and 4 are cut, and each one's pieces all go to part 1."""
    tbl = rows_table(tmp_path)
    cut = split(tmp_path / 'none', tbl)
    assert [e['tab'] for e in cut.read('tabs')[1] if isinstance(e, dict)] == [1, 3, 3, 4, 4]
    p = split(tmp_path / 'whole', tbl, coalesce='whole')
    assert p.read('tabs') == [[0, {'tab': 1, 'stratifyby': 's2'}, 2], [{'tab': 1, 'stratifyby': 's3'}, 3, 4]]
    assert p.read('summary')['rejoined'] == ['tab_000003', 'tab_000004']
    # Rejoined, a tab is the table's own, read where it is: no piece of it to build.
    assert not any(isinstance(p.part(1).tab(i), DatatabPiece) for i in (1, 2))


def test_whole_rejoins_a_tab_whose_pieces_lie_in_two_partitions_and_counts_it_in_both(tmp_path):
    """Both pieces of tab 4 go to part 0, s3's in p1 and s5's in p2: whole, it holds rows of both partitions."""
    tbl = rows_table(tmp_path)
    cut = split(tmp_path / 'none', tbl, partitionby=PARTITION)
    assert {(4, 's3'), (4, 's5')} <= {(e['tab'], e['stratifyby']) for e in cut.read('tabs')[0] if isinstance(e, dict)}
    p = split(tmp_path / 'p', tbl, partitionby=PARTITION, coalesce='whole')
    assert 4 in p.read('tabs')[0]
    # The rows per partition are the deal's, however the tabs hold them; tab 4 is a tab of each.
    partitions = lambda q: {st: c['rows'] for st, c in q.read('summary')['folds'][0]['partitions'].items()}
    assert partitions(p) == partitions(cut)
    assert all(c['tabs'] >= 1 for c in p.read('summary')['folds'][0]['partitions'].values())


def test_whole_keeps_cut_a_tab_with_a_piece_in_no_part(tmp_path):
    """Bag 2's row of no patient is skipped: the bag made whole would bring it back."""
    tbl = bags(tmp_path)
    for seed in range(30):
        p = bag_partition(tmp_path / f's{seed}', tbl, seed=seed, coalesce='whole')
        parts = {k for k, f in enumerate(p.read('tabs')) for e in f if isinstance(e, dict) and e['tab'] == 2}
        if len(parts) == 1:
            break
    else:
        pytest.fail("no seed deals bag 2's pieces to one part")
    assert all(2 not in f for f in p.read('tabs'))


def test_pack_pools_the_small_entries_and_keeps_the_rest(tmp_path):
    tbl = rows_table(tmp_path)
    p = split(tmp_path / 'p', tbl, coalesce='pack', min_tab_rows=3, max_tab_rows=4)
    for k, entries in enumerate(p.read('tabs')):
        part = p.part(k).build()
        packs = [e for e in entries if isinstance(e, dict) and 'pack' in e]
        kept = [e for e in entries if e not in packs]
        # Whole tabs and pieces of at least 3 rows stay; every packed tab takes at most 4.
        assert all(isinstance(e, int) and len(TABS[e]) >= 3 for e in kept), entries
        for i, e in enumerate(entries):
            if e in packs:
                tab = part.tab(i)
                assert isinstance(tab, DatatablePackTab)
                assert sum(seg[2] for seg in e['segments']) == tab.n_rows('meta') <= 4
        assert p.read('summary')['folds'][k]['packed']['tabs'] == len(packs)


def test_a_packed_tab_holds_the_rows_its_source_slice_names_in_every_slice(tmp_path):
    tbl = rows_table(tmp_path)
    p = split(tmp_path / 'p', tbl, partitionby=PARTITION, coalesce='pack', min_tab_rows=3)
    seen = 0
    for part in built_parts(p):
        for i, e in enumerate(part.tab_indices):
            if not (isinstance(e, dict) and 'pack' in e):
                continue
            tab = part.tab(i)
            meta, x = tab.data('meta')['meta'], tab.data('x')['x']
            for j, (t, r) in enumerate(tab.source_index().tolist()):
                source = tbl.tab(t)
                assert meta['row'][j] == source.data('meta')['meta']['row'][r] == x['row'][j]
                assert np.array_equal(x['features'][j], source.data('x')['x']['features'][r])
                seen += 1
    assert seen


@pytest.mark.parametrize('spec, match', [
    ({'coalesce': 'pack'}, 'min_tab_rows'),
    ({'coalesce': 'pack', 'min_tab_rows': 4, 'max_tab_rows': 3}, 'max_tab_rows'),
    ({'min_tab_rows': 2}, "coalesce='pack' only"),
    ({'coalesce': 'tight'}, 'coalesce must be one of'),
])
def test_coalesce_is_checked_where_it_is_given(tmp_path, spec, match):
    with pytest.raises(ValueError, match=match):
        DatatablePartition(datalake=str(tmp_path), spec=dict(
            datatable=rows_table(tmp_path), fractions=[0.5, 0.5], partition_slice='meta', **spec))


def test_a_partition_from_before_the_roles_were_renamed_is_not_taken_for_one_now(tmp_path, monkeypatch):
    """Before 2026-10-10 a collator's stratifyby divided its values between the parts; now it keeps them in one.

    The collator spells the same either way, so only the partition's VERSION tells them apart, and no
    specialization may reach past it: a partition built before is rebuilt, never read as one built now.
    """
    tbl = rows_table(tmp_path)
    collator = Datacollator(spec=dict(columns={'stratifyby': STRATUM}))
    spec = dict(datatable=tbl, fractions=[0.5, 0.5], partition_slice='meta', collator=collator)
    with monkeypatch.context() as m:
        m.setattr(DatatablePartition, '__module__', 'dbx.datatables')
        m.setattr(DatatablePartition, 'VAR', DatatablePartition._DealVAR_)
        m.setattr(DatatablePartition, 'VERSION', 1)
        m.setattr(DatatablePartition, '__post_init__', Datablock.__post_init__)
        old_hash = DatatablePartition(datalake=str(tmp_path), spec=spec).hash
    for mode in (None, 'whole'):
        new = DatatablePartition(datalake=str(tmp_path), spec=dict(spec, coalesce=mode))
        assert new.hash != old_hash
        assert old_hash not in [new.get_hash(sp) for sp in new._specialization_variants_()]
        assert not [sp for sp in DatatablePartition.SPECIALIZATIONS if new._pins_match_(sp)]


def test_a_piece_stored_under_dbx_datatables_is_reached(tmp_path):
    p = split(tmp_path / 'p', rows_table(tmp_path))
    piece = next(p.tab(e) for e in p.read('tabs')[0] if isinstance(e, dict))
    reached = [(piece._specialization_anchor_(sp), piece.get_hash(sp)) for sp in type(piece).SPECIALIZATIONS]
    assert ('dbx.datatables.DatatabPiece', piece.hash) in reached


# The probe: rows aggregated by a column, wherever they lie.

def agg_probe(tmp_path, fit, ev, aggregateby=PATIENT, **spec):
    from dbx import FeatureAffineLogisticProbe
    columns = {'signals': [('x', 'features')], 'labels': [('meta', 'label')]}
    if aggregateby is not None:
        columns['aggregateby'] = aggregateby
    return FeatureAffineLogisticProbe(datalake=str(tmp_path), spec=dict(
        fit_table=fit, eval_table=ev, collator=Datacollator(spec=dict(columns=columns)), **spec))


def test_an_aggregate_is_the_mean_of_its_rows_and_label_across_packed_tabs(tmp_path):
    tbl = bags(tmp_path)
    p = bag_partition(tmp_path / 'p', tbl, coalesce='pack', min_tab_rows=3)
    fit, ev = built_parts(p)
    assert any(isinstance(e, dict) and 'pack' in e for e in fit.tab_indices + ev.tab_indices)
    pr = agg_probe(tmp_path / 'probe', fit, ev, aggregation='mean').build()
    for side, part in (('fit', fit), ('eval', ev)):
        meta, x = part.data('meta')['meta'], part.data('x')['x']
        cells = {}
        for info, label, feats in zip(meta['info'], meta['label'], x['features']):
            cells.setdefault((info['patient'], int(label)), []).append(feats)
        want = np.array(sorted(np.mean(v, axis=0, dtype=np.float64).tolist() for v in cells.values()))
        got = np.array(sorted(pr.read(side, 'features').numpy().astype(np.float64).tolist()))
        assert len(pr.read(side, 'labels')) == len(cells)
        assert np.allclose(got, want, atol=1e-5)


def test_the_rows_of_one_value_carrying_two_labels_are_two_samples(tmp_path):
    """Each stratum's rows alternate labels 0, 1: aggregated by stratum, each stratum is a sample per label it carries."""
    p = split(tmp_path / 'p', rows_table(tmp_path), coalesce='whole')
    fit, ev = built_parts(p)
    pr = agg_probe(tmp_path / 'probe', fit, ev, aggregateby=STRATUM, aggregation='mean').build()
    for side, part in (('fit', fit), ('eval', ev)):
        meta = part.data('meta')['meta']
        cells = sorted({(g, int(label)) for g, label in zip(meta['stratum'], meta['label'])})
        assert sorted(pr.read(side, 'labels').tolist()) == sorted(label for _, label in cells)
        assert len({g for g, _ in cells}) < len(cells)


def test_two_tables_sharing_an_aggregateby_value_are_refused(tmp_path):
    """Aggregated by cohort, of a partition dividing every cohort between the parts: every cohort is in both tables."""
    fit, ev = built_parts(bag_partition(tmp_path / 'p', bags(tmp_path)))
    with pytest.raises(ValueError, match=r"share \d+ value\(s\) of aggregateby"):
        agg_probe(tmp_path / 'probe', fit, ev, aggregateby=COHORT).build()


def test_parts_of_two_partitions_of_one_table_sharing_rows_are_refused(tmp_path):
    """Packed tabs are tagged by where they are, so only their rows can say they overlap.

    No bag holds 5 rows, so every entry is packed, and no tab of one part is a tab of the other.
    """
    tbl = bags(tmp_path)
    a = bag_partition(tmp_path / 'a', tbl, seed=0, coalesce='pack', min_tab_rows=5)
    b = bag_partition(tmp_path / 'b', tbl, seed=1, coalesce='pack', min_tab_rows=5)
    fit, ev = a.part(0).build(), b.part(0).build()
    assert all(isinstance(e, dict) and 'pack' in e for e in fit.tab_indices + ev.tab_indices)
    assert set(map(tuple, fit.source_index().tolist())) & set(map(tuple, ev.source_index().tolist()))
    with pytest.raises(ValueError, match="share .* of its rows"):
        agg_probe(tmp_path / 'probe', fit, ev, aggregateby=None)._check_disjoint_()
    # Two parts of one partition are disjoint by its deal.
    agg_probe(tmp_path / 'ok', *built_parts(a), aggregateby=None)._check_disjoint_()


def test_aggregation_takes_its_rows_by_an_aggregateby_column(tmp_path):
    fit, ev = built_parts(bag_partition(tmp_path / 'p', bags(tmp_path)))
    with pytest.raises(ValueError, match="aggregateby column"):
        agg_probe(tmp_path / 'probe', fit, ev, aggregateby=None, aggregation='mean')
    with pytest.raises(ValueError, match="unknown signal_reduction"):
        agg_probe(tmp_path / 'reduce', fit, ev, signal_reduction='max')
