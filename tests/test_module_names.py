"""
dbx's classes live in the modules they are defined in, under the names they have now.

The modules were once ``dbx.datapoints``, ``dbx.datafeatures``, ``dbx.datamodels``
and ``dbx.dataprobes``, and the classes had other names (``DatapointTab``,
``DatafeatureTable``, ...). Those names are gone: a class's `fqcn` -- the anchor
its artifacts are stored under -- is its real module and name. What was stored
under an old module name is reached by a specialization naming that anchor.
"""
import importlib
from dataclasses import dataclass

import pytest

import dbx
from dbx import backbones, datatables, featuretables, probes
from dbx.datatables import DATASLICE, DatatablePartition, Datatab, Datatable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.mark.parametrize('cls', [datatables.Datatab, datatables.Datatable, datatables.DatatablePartition,
                                 datatables.DatatablePart, featuretables.Featuretab, featuretables.Featuretable,
                                 featuretables.BipolarFeaturetab, featuretables.BipolarFeaturetable,
                                 datatables.Datacollator, probes.FeatureStatsProbe])
def test_a_class_is_recorded_under_its_own_module(cls):
    assert cls.__module__ in ('dbx.datatables', 'dbx.featuretables', 'dbx.probes')
    assert cls.__module__ == importlib.import_module(cls.__module__).__name__


@pytest.mark.parametrize('name', ['dbx.datapoints', 'dbx.datafeatures', 'dbx.datamodels', 'dbx.dataprobes',
                                  'dbx.databackbones'])
def test_the_old_modules_are_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


@pytest.mark.parametrize('name', ['DatapointTab', 'DatapointTable', 'DatapointPartition', 'DatapointFold',
                                  'DatatabBase', 'UpstreamTabSlices', 'DatafeatureTab', 'DatafeatureTable',
                                  'FeatureTab', 'FeatureTable', 'BipolarDatafeatureTab', 'DatafeatureStatsProbe',
                                  'DatamodelEvaluatorFactory', 'DataformerEvaluator', 'Datastill', 'DatapointTableTab'])
def test_the_old_names_are_gone(name):
    assert not any(hasattr(m, name) for m in (dbx, backbones, datatables, featuretables, probes))


class Rows(Datatab):
    TOPICS = {'rows': DATASLICE(i='int', k='int')}

    @dataclass
    class VAR(Datatab.VAR):
        n: int = 3

    def __build__(self):
        with self.slice_writers() as writers:
            for i in range(self.var.n):
                writers['rows'].write({'i': i, 'k': 0})


class Table(Datatable):
    TAB = Rows

    @property
    def n_tabs(self):
        return 4


def test_a_partition_stored_under_the_old_module_name_is_reached(tmp_path, monkeypatch):
    """Under dbx.datapoints its table was datapoint_table, and groupby a field of its own."""
    from dbx.datablocks import Datablock
    from dbx.datatables import Datacollator
    table = Table(datalake=str(tmp_path)).build()

    @dataclass
    class OldVAR(Datablock.VAR):
        datapoint_table: Datatable
        fractions: list[float]
        partition_slice: int | str
        method: str = 'random'
        seed: int = 0
        groupby: tuple | list | None = None
        stratifyby: tuple | list | None = None
        balance: str = 'rows'

    with monkeypatch.context() as m:
        m.setattr(DatatablePartition, '__module__', 'dbx.datapoints')
        m.setattr(DatatablePartition, 'VAR', OldVAR)
        m.setattr(DatatablePartition, '__post_init__', Datablock.__post_init__)
        old = DatatablePartition(datalake=str(tmp_path), spec=dict(
            datapoint_table=table, fractions=[0.5, 0.5], partition_slice='rows', groupby=('rows', 'k'), balance='tabs'))
        old_anchor, old_hash = old.anchor, old.hash
    new = DatatablePartition(datalake=str(tmp_path), spec=dict(
        datatable=table, fractions=[0.5, 0.5], partition_slice='rows', balance='tabs',
        collator=Datacollator(spec=dict(columns={'groupby': ('rows', 'k')}))))
    assert (old_anchor, new.anchor) == ('dbx.datapoints.DatatablePartition', 'dbx.datatables.DatatablePartition')
    reached = [(new._specialization_anchor_(sp), new.get_hash(sp)) for sp in DatatablePartition.SPECIALIZATIONS
               if new._pins_match_(sp)]
    assert (old_anchor, old_hash) in reached
