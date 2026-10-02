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
                                 featuretables.Datacollator, probes.FeatureStatsProbe])
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
    TOPICS = {'rows': DATASLICE(i='int')}

    @dataclass
    class VAR(Datatab.VAR):
        n: int = 3

    def __build__(self):
        with self.slice_writers() as writers:
            for i in range(self.var.n):
                writers['rows'].write({'i': i})


class Table(Datatable):
    TAB = Rows

    @property
    def n_tabs(self):
        return 4


def test_a_partition_stored_under_the_old_module_name_is_adopted(tmp_path, monkeypatch):
    table = Table(datalake=str(tmp_path)).build()

    def partition():
        return DatatablePartition(datalake=str(tmp_path), spec=dict(
            datapoint_table=table, fractions=[0.5, 0.5], partition_slice='rows', balance='tabs'))

    with monkeypatch.context() as m:
        m.setattr(DatatablePartition, '__module__', 'dbx.datapoints')
        old = partition().build()
        old_tabs, old_anchor, old_hash = old.read('tabs'), old.anchor, old.hash
    new = partition()
    assert (old_anchor, new.anchor) == ('dbx.datapoints.DatatablePartition', 'dbx.datatables.DatatablePartition')
    assert new.hash == old_hash, "the module is the anchor, not the identity"
    new.build()                                   # adopts: nothing left to build
    assert sorted(new.redirected_topics()) == ['summary', 'tabs']
    assert new.read('tabs') == old_tabs
