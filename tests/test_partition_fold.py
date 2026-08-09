"""Unit tests for DatapointPartition and DatapointFold."""

import os
from dataclasses import dataclass
import numpy as np
import pytest

from dbx import (
    DatapointTab,
    DatapointTable,
    DatapointPartition,
    DatapointFold,
    SLICETOPIC,
)


class DummyTab(DatapointTab):
    TOPICS = {'numbers': SLICETOPIC, 'letters': SLICETOPIC}

    @dataclass
    class VAR(DatapointTab.VAR):
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


class DummyTable(DatapointTable):
    TAB = DummyTab

    @dataclass
    class VAR(DatapointTable.VAR):
        tab_sizes: tuple[int, ...] = (5, 3, 2, 4)

    @property
    def n_tabs(self):
        return len(self.var.tab_sizes)

    def __tab__(self, idx):
        offset = sum(self.var.tab_sizes[:idx])
        return self.TAB(
            spec=dict(n=self.var.tab_sizes[idx], base=offset),
            tag=f"tab_{idx}",
        )


@pytest.fixture
def table(tmp_path):
    tbl = DummyTable(url=str(tmp_path / 'table'))
    tbl.build()
    return tbl


def test_datapoint_partition_and_fold(table, tmp_path):
    partition = DatapointPartition(
        url=str(tmp_path / 'partition'),
        spec=dict(datapoint_table=table, fractions=[0.5, 0.5], partition_slice=0),
    ).build()

    assert partition.valid()
    assert partition.n_folds() == 2

    tabs0 = partition.tabs(0)
    tabs1 = partition.tabs(1)
    indices0 = partition.tabs_indices(0)
    indices1 = partition.tabs_indices(1)

    assert set(indices0).isdisjoint(set(indices1))
    assert len(indices0) + len(indices1) == table.n_tabs

    fold0 = partition.fold(0)
    fold1 = partition.fold(1)

    assert isinstance(fold0, DatapointFold)
    assert fold0.n_tabs == len(indices0)
    assert fold1.n_tabs == len(indices1)

    data0 = fold0.data('numbers')
    data1 = fold1.data('numbers')

    assert len(data0) + len(data1) == table.n_rows('numbers')
    assert fold0.n_rows('numbers') == len(data0)

    ds0 = fold0.dataset()
    assert len(ds0) == len(data0)


def test_partition_slice_parameter(table, tmp_path):
    p_by_str = DatapointPartition(
        url=str(tmp_path / 'partition_str'),
        spec=dict(datapoint_table=table, fractions=[0.5, 0.5], partition_slice='letters'),
    ).build()
    assert p_by_str.valid()

    p_by_int = DatapointPartition(
        url=str(tmp_path / 'partition_int'),
        spec=dict(datapoint_table=table, fractions=[0.5, 0.5], partition_slice=1),
    ).build()
    assert p_by_int.valid()
    assert p_by_str.tabs_indices(0) == p_by_int.tabs_indices(0)
