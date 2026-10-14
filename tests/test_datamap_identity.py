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

from dbx.datablocks import DATAFILE, Datablock
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


@pytest.mark.pinned
def test_a_partition_its_parts_and_what_is_built_on_them_keep_their_hashes(tmp_path):
    """Every CPTAC partition, fold, deep feature clip and probe built is reached only if these stay put.

    A part is the table pulled back along its fold's map, whatever classes do
    it: the partition's VAR, version and topics, and the part's VAR ``{partition,
    fold}``, version, BLOCK and TOPICS, are the identities of everything built.
    """
    partition = _partition(tmp_path)
    part = partition.part(1)
    downstream = Downstream(datalake=str(tmp_path), spec=dict(upstream=part))
    assert {'partition': partition.hash, 'part': part.hash, 'downstream': downstream.hash} == PINNED


PINNED = {
    'partition': 'bdef27cc10ce637f5672c19aa92ab50047ffcefb6a8aef73aa0d18d73e8e2db2',
    'part': '0f44b7a7d4a4439b3981eb1f1fa6445170d59a029706ae99af8aa445c48286cb',
    'downstream': '35aa6d78373ccd416530a9bf73cdd08325a86810e6e39c0644e3cfa898c6b5b0',
}


@pytest.mark.pinned
def test_a_feature_table_s_part_hashes_as_it_did_as_a_datatable_part(tmp_path):
    """A part of a feature table is a FeaturetablePart since 2026-10-13, and its identity is what it was.

    The class is in its path, not its hash: a part built as a DatatablePart is
    adopted where it is, by FeaturetablePart's specialization, only because the
    hashes agree.
    """
    import sys
    sys.path.insert(0, __import__('os').path.dirname(__file__))
    from test_partition_methods import _features
    from dbx.dataparts import DatatablePart, FeaturetablePart
    url = str(tmp_path)
    p = DatatablePartition(datalake=url, spec=dict(datatable=_features(url), fractions=[0.5, 0.5],
                                                   partition_slice='features'))
    new = p.part(0)
    assert type(new) is FeaturetablePart
    assert new.hash == DatatablePart(datalake=url, spec=dict(partition=p, fold=0)).hash
