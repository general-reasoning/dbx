"""Unit tests for DatatablePartition, and the tables pulled back along its parts."""

import os
from dataclasses import dataclass
import numpy as np
import pytest

from dbx.dataparts import MappedDatatable

from dbx import (
    Datatablet,
    Datatable,
    DatatablePartition,
    SLICETOPIC,
)


class DummyTablet(Datatablet):
    TOPICS = {'numbers': SLICETOPIC, 'letters': SLICETOPIC}

    @dataclass
    class VAR(Datatablet.VAR):
        n: int = 3
        base: int = 0

    COLUMNS = {
        'numbers': {'idx': 'int', 'val': 'int'},
        'letters': {'idx': 'int', 'lbl': 'str'},
    }

    def __build__(self):
        with self.slice_writers(self.COLUMNS) as writers:
            for i in range(self.var.n):
                k = self.var.base + i
                writers['numbers'].write({'idx': k, 'val': k * 10})
                writers['letters'].write({'idx': k, 'lbl': f"item_{k}"})


class DummyTable(Datatable):
    TABLET = DummyTablet

    @dataclass
    class VAR(Datatable.VAR):
        tablet_sizes: tuple[int, ...] = (5, 3, 2, 4)

    @property
    def n_tablets(self):
        return len(self.var.tablet_sizes)

    def __tablet__(self, idx):
        offset = sum(self.var.tablet_sizes[:idx])
        # Through super(), so the tablet inherits the table's url. Constructing
        # self.TABLET directly leaves url unset, and the tablet lands under DBX_ROOT
        # instead of under the table it belongs to.
        return super().__tablet__(
            idx,
            tag=f"tab_{idx}",
            n=self.var.tablet_sizes[idx],
            base=offset,
        )


@pytest.fixture
def table(tmp_path):
    tbl = DummyTable(datalake=str(tmp_path / 'table'))
    tbl.build()
    return tbl


def test_datatable_partition_and_fold(table, tmp_path):
    partition = DatatablePartition(
        datalake=str(tmp_path / 'partition'),
        spec=dict(datatable=table, fractions=[0.5, 0.5], partition_slice=0),
    ).build()

    assert partition.valid()
    assert partition.n_parts() == 2

    tablets0 = [partition.part(0).pullback().tablet(i) for i in range(partition.part(0).n_tablets)]
    tablets1 = [partition.part(1).pullback().tablet(i) for i in range(partition.part(1).n_tablets)]
    indices0 = partition.tablets_indices(0)
    indices1 = partition.tablets_indices(1)

    assert set(indices0).isdisjoint(set(indices1))
    assert len(indices0) + len(indices1) == table.n_tablets

    fold0 = partition.part(0).pullback()
    fold1 = partition.part(1).pullback()

    assert isinstance(fold0, MappedDatatable)
    assert fold0.var.datatable is None, "the partition's own table"
    assert not hasattr(fold0.var, 'datapoints_per_row')
    assert fold0.datamap.var.partition.datatable == table
    assert fold0.datatable == table
    assert fold0.datapoints_per_row == 1
    assert fold0.n_tablets == len(indices0)
    assert fold1.n_tablets == len(indices1)

    # data() is column-major and keyed by slice, so a row count comes from a
    # column rather than from len() of the returned mapping.
    data0 = fold0.data('numbers')['numbers']['idx']
    data1 = fold1.data('numbers')['numbers']['idx']

    assert len(data0) + len(data1) == table.n_rows('numbers')
    assert fold0.n_rows('numbers') == len(data0)

    ds0 = fold0.dataset()
    assert len(ds0) == len(data0)


def test_partition_slice_parameter(table, tmp_path):
    p_by_str = DatatablePartition(
        datalake=str(tmp_path / 'partition_str'),
        spec=dict(datatable=table, fractions=[0.5, 0.5], partition_slice='letters'),
    ).build()
    assert p_by_str.valid()

    p_by_int = DatatablePartition(
        datalake=str(tmp_path / 'partition_int'),
        spec=dict(datatable=table, fractions=[0.5, 0.5], partition_slice=1),
    ).build()
    assert p_by_int.valid()
    assert p_by_str.tablets_indices(0) == p_by_int.tablets_indices(0)


def test_a_part_resolves_its_tablets_redirections_through_the_table(table, tmp_path):
    """A part's tablet i is the table's tablet its lot takes whole; so is its redirection."""
    partition = DatatablePartition(
        datalake=str(tmp_path / 'partition_r'),
        spec=dict(datatable=table, fractions=[0.5, 0.5], partition_slice=0),
    ).build()
    part = partition.part(0).pullback()
    real = part.datamap.entries[0]
    other = DummyTable(datalake=str(tmp_path / 'elsewhere')).build()
    table.tablet(real).UNSAFE_redirect(paths=other.tablet(real).paths(), OVERRIDE=True)

    got = part.get_tablet_redirections(parallelization='inline')
    assert got.index.tolist() == list(range(part.n_tablets))
    assert got[0].paths == other.tablet(real).paths()
    assert part.tablets_redirected(parallelization='inline')[0]


def test_a_part_builds_with_its_partitions_workers(table, tmp_path):
    """Each part is built by its own build(), one after another: each takes the workers the partition has."""
    spec = dict(datatable=table, fractions=[0.5, 0.5], partition_slice=0)
    partition = DatatablePartition(datalake=str(tmp_path / 'partition'), spec=spec,
                                   parallelization='multiprocessing', n_workers=3)
    part = partition.part(1).pullback()
    assert (part.parallelization, part.n_workers) == ('multiprocessing', 3)
    assert part.hash == DatatablePartition(datalake=str(tmp_path / 'partition'), spec=spec).part(1).pullback().hash, \
        "workers are no identity"
    inline = DatatablePartition(datalake=str(tmp_path / 'partition'), spec=spec).part(0).pullback()
    assert (inline.parallelization, inline.n_workers) == (None, 1)
