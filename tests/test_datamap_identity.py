"""The identities the datamap refactor must keep: a partition's, its parts', and every block built on a part.

See DATAPARTITION.md, "TODO: three choices, untangled -- datamaps, and tables
mapped along them", Identities. The hashes below were taken from the code
before the refactor (dbx 0e6ca42). A part enters the identity of a block built
on it as its rendered spec only -- ``{'fold': k, 'partition': {...}}`` -- so the
part's and the partition's classes may change, and their VARs, versions and
topics may not.
"""
from dataclasses import dataclass

import pytest

from dbx.datablocks import DATAFILE, SAME, Datablock
from dbx.datatables import DATASLICE, Datacollator, Datatable, Datatablet
from dbx.dataparts import DatatablePartition


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Bag(Datatablet):
    TOPICS = {'tiles': DATASLICE(tile='int'), 'annotations': DATASLICE(annotations=dict(case_id='str', cohort='str'))}

    @dataclass
    class VAR(Datatablet.VAR):
        k: int = 0


class Clip(Datatable):
    TABLET = Bag

    @dataclass
    class VAR(Datatable.VAR):
        n_tablets: int = 4

    @property
    def n_tablets(self):
        return self.var.n_tablets

    def __tablet__(self, idx, **kw):
        return super().__tablet__(idx, k=idx)


class Downstream(Datablock):
    """A block built on a part, as a deep feature clip is built on a CPTAC fold."""
    TOPICS = {'out': DATAFILE('out.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        upstream: Datatable = None


def _partition(tmp_path):
    collator = Datacollator(spec=dict(columns={'stratifyby': ('annotations', 'annotations', 'case_id'),
                                               'partitionby': ('annotations', 'annotations', 'cohort')}))
    return DatatablePartition(datalake=str(tmp_path), spec=dict(
        datatable=Clip(datalake=str(tmp_path)), fractions=[0.6, 0.2, 0.2], partition_slice='tiles',
        collator=collator))


def _past(block, sp):
    """*block*'s hash as *sp* says it rendered before -- the identity a specialization reaches."""
    return block.get_hash(sp)


def _built_on(block, field, nested):
    """The hash of a block built on *block*, as it hashed when *block* rendered as its past *nested*."""
    return block.get_hash(type(block).Specialization(spec={}, topics=SAME, nested={field: nested}))


@pytest.mark.pinned
def test_a_partition_its_parts_and_what_is_built_on_them_are_reached(tmp_path):
    """Every CPTAC partition, part, deep feature clip and probe built is reached only if these are.

    The partition keeps its hash. A part was a `DatatablePart`, its VAR ``{partition,
    fold}``; since 2026-10-16 it is the table pulled back along the part's map, a
    `MappedDatatable` with a hash of its own -- and the specialization it declares
    reaches the part's old identity, for itself and for every block built on it.
    """
    partition = _partition(tmp_path)
    part = partition.part(1).pullback()
    (past,) = [sp for sp in type(part).SPECIALIZATIONS if part._pins_match_(sp)]
    downstream = Downstream(datalake=str(tmp_path), spec=dict(upstream=part))
    assert partition.hash == PINNED['partition']
    assert _past(part, past) == PINNED['part']
    assert _built_on(downstream, 'upstream', past) == PINNED['downstream']


PINNED = {
    'partition': 'bdef27cc10ce637f5672c19aa92ab50047ffcefb6a8aef73aa0d18d73e8e2db2',
    'part': '0f44b7a7d4a4439b3981eb1f1fa6445170d59a029706ae99af8aa445c48286cb',
    'downstream': '35aa6d78373ccd416530a9bf73cdd08325a86810e6e39c0644e3cfa898c6b5b0',
}


@pytest.mark.pinned
def test_a_feature_table_s_part_is_reached_as_the_datatable_part_it_was(tmp_path):
    """A part of a feature table was a DatatablePart, adopted as a FeaturetablePart; it is a MappedFeaturetable now.

    Its upstream -- the feature table's, pulled back along the part's map -- was never in
    its VAR: its specialization leaves it out, determined by the map, the table and the
    collator, and so reaches the identity it was built under. The hash was taken from the
    code before the change (dbx 7fd8eea).
    """
    import sys
    sys.path.insert(0, __import__('os').path.dirname(__file__))
    from test_partition_methods import _features
    from dbx.dataparts import MappedFeaturetable
    url = str(tmp_path)
    p = DatatablePartition(datalake=url, spec=dict(datatable=_features(url), fractions=[0.5, 0.5],
                                                   partition_slice='features'))
    new = p.part(0).pullback()
    assert type(new) is MappedFeaturetable
    pasts = [sp for sp in type(new).SPECIALIZATIONS if new._pins_match_(sp)]
    assert {sp.anchor for sp in pasts} == {'dbx.dataparts.FeaturetablePart', 'dbx.dataparts.DatatablePart'}
    assert {_past(new, sp) for sp in pasts} == {OLD_FEATURE_PART}


OLD_FEATURE_PART = 'fcbff0b55efe01efc65ac52154752edcf73880b866249ec9aece189558ed3f22'


@pytest.mark.pinned
def test_a_core_partition_keeps_its_identity_and_reaches_its_old_path(tmp_path):
    """A core partition's hash -- its spec, version and TOPICS, a topic's description included -- is pinned.

    Every CPTAC core's clustering was built under it; on 2026-10-16 a reworded description of its
    ``layout`` topic moved it, unseen. Its class was `DatatableCorePartition` until 2026-10-17, and its
    specialization reaches what was built at that class's path. The hash is the code's at dbx 7fd8eea.
    """
    import sys
    sys.path.insert(0, __import__('os').path.dirname(__file__))
    from test_core_partition import core, PointTable
    p = core(tmp_path, PointTable(datalake=str(tmp_path / 'points'), spec={}), build=False)
    assert p.hash == CORE_PARTITION
    (old,) = [sp for sp in type(p).SPECIALIZATIONS if p._pins_match_(sp)]
    assert old.anchor == 'dbx.dataparts.DatatableCorePartition' and p.get_hash(old) == CORE_PARTITION


CORE_PARTITION = 'b72da7e7d75a8ffe9da2d32a57ad1e2ba4e77d63c849ef5ebcc1ef292853b3d4'


def test_map_and_mapped_tablets_are_filed_by_hash_and_tagged_by_their_table(tmp_path):
    """A map tablet's and a mapped tablet's path is its hash alone; its tag, its table's and its index. No identity moves."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_core_partition import core, points
    table = points(tmp_path).set(tag='clip')
    partition = core(tmp_path, table).set(tag='deal')
    datamap = partition.part(0)
    assert datamap.tag == 'deal:part0'
    lot = datamap.tablet(0)
    assert lot.tag == 'deal:part0:000000'
    assert os.path.basename(os.path.dirname(lot.anchorkeypath)) == 'dbx.dataparts.PartitionDatamapTablet'
    mapped = datamap.pullback(table).tablet(0)
    assert mapped.tag == 'clip along deal:part0:000000'
    assert os.path.basename(os.path.dirname(mapped.anchorkeypath)) == 'dbx.dataparts.MappedDatatablet'
    assert datamap.pullback(table).build().valid()


def test_a_mapped_tablet_filed_under_its_tag_is_reached_where_it_was_filed(tmp_path, monkeypatch):
    """Until 2026-10-17 a mapped tablet was filed under its tag, its partition's `tag_of`; now by its hash.

    Its hash is unchanged, and a specialization naming that tag (`Specialization(tag=...)`) reaches it
    where it was filed: rebuilt now, the pullback copies nothing again.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_core_partition import core, points
    from dbx.dataparts import MappedDatatablet
    table = points(tmp_path)
    monkeypatch.setattr(MappedDatatablet, 'KEYBY', 'tag_version_shorthash')
    partition = core(tmp_path, table)
    before = partition.part(0).pullback(table)
    before_tag = lambda i: partition.tag_of(0, i)
    # As then: each tablet tagged as its partition says, and filed under that tag.
    monkeypatch.setattr(type(before), '__tablet__', _tagged_as_then_(type(before).__tablet__, before_tag))
    before.build_tree()
    built = {i: before.tablet(i).anchorkeypath for i in range(before.n_tablets)}
    assert all(f"/{before_tag(i)}/" in p for i, p in built.items())
    monkeypatch.undo()
    after = core(tmp_path, table).part(0).pullback(table)
    assert after.hash == before.hash and after.tablet(0).hash == before.tablet(0).hash
    after.build_tree()
    for i, path in built.items():
        tablet = after.tablet(i)
        assert tablet.valid() and tablet.anchorkeypath != path
        assert os.path.dirname(tablet.path('source')).startswith(path)          # read where it was filed
        assert not os.path.exists(os.path.join(tablet.anchorkeypath, 'source'))  # nothing copied again


def _tagged_as_then_(tablet, tag_of):
    """A pullback's `__tablet__`, tagging tablet *i* ``tag_of(i)`` as it did until 2026-10-17."""
    def __tablet__(self, idx, **kw):
        return tablet(self, idx, **kw).set(tag=tag_of(idx))
    return __tablet__


def test_a_part_of_whole_tablets_builds_no_map(tmp_path):
    """Every tablet taken whole, no lot is read: build_tree builds the pullback, valid, and not its map."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_core_partition import points
    table = points(tmp_path)
    partition = DatatablePartition(datalake=str(tmp_path), spec=dict(datatable=table, fractions=[0.5, 0.5],
                                                                     partition_slice='pts')).build()
    pulled = partition.part(0).pullback()
    assert all(pulled.datamap.whole(i) is not None for i in range(pulled.n_tablets))
    assert pulled.TREE_SKIP_BUILDING == ('datamap',)
    pulled.build_tree()
    assert pulled.valid() and pulled.valid_var(reduce=True)
    assert not pulled.datamap.valid(), "its map was built"


def test_a_part_of_whole_tablets_builds_no_map_and_another_table_pulls_back_along_it(tmp_path):
    """Every tablet taken whole, no lot is read: build_tree builds the pullback, valid, and not its map's tablets.

    The map is still there -- the same block, in the pullback's VAR -- and another table laid out alike
    is pulled back along it, still unbuilt: each tablet that table's own, as the partition records it.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_core_partition import points
    table = points(tmp_path)
    partition = DatatablePartition(datalake=str(tmp_path), spec=dict(datatable=table, fractions=[0.5, 0.5],
                                                                     partition_slice='pts')).build()
    datamap = partition.part(0)
    pulled = datamap.pullback()
    assert all(datamap.whole(i) is not None for i in range(pulled.n_tablets))
    assert pulled.TREE_SKIP_BUILDING == ('datamap',)
    pulled.build_tree()
    assert pulled.valid() and pulled.valid_var(reduce=True)
    assert not datamap.valid(), "its map was built"
    other = points(tmp_path, offset=100)
    along = datamap.pullback(other).build_tree()
    assert along.valid() and not datamap.valid()
    for i in range(along.n_tablets):
        assert along.tablet(i).hash == other.tablet(datamap.whole(i)).hash
        assert along.tablet(i).hash != pulled.tablet(i).hash
