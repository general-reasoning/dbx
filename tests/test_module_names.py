"""
dbx's classes live in the modules they are defined in, under the names they have now.

The modules were once ``dbx.datapoints``, ``dbx.datafeatures``, ``dbx.datamodels``
and ``dbx.dataprobes``, and the classes had other names (``DatapointTab``,
``DatafeatureTable``, ...). Those names are gone: a class's `fqcn` -- the anchor
its artifacts are stored under -- is its real module and name. What was stored
under an old module name is reached by a specialization naming that anchor -- except
partitions, rebuilt since their roles were renamed (see DATAPARTITION.md, "Identities").
"""
import importlib

import pytest

import dbx
from dbx import backbones, dataparts, datatables, featuretables, probes


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.mark.parametrize('cls', [datatables.Datatab, datatables.Datatable, dataparts.DatatablePartition,
                                 dataparts.DatatablePart, featuretables.Featuretab, featuretables.Featuretable,
                                 featuretables.BipolarFeaturetab, featuretables.BipolarFeaturetable,
                                 datatables.Datacollator, dataparts.DatatabPiece, dataparts.DatatableCorePartition,
                                 dataparts.DatatableCorePart, dataparts.DatatableCoreTab, dataparts.DatatablePackTab,
                                 probes.FeatureStatsProbe])
def test_a_class_is_recorded_under_its_own_module(cls):
    assert cls.__module__ in ('dbx.datatables', 'dbx.dataparts', 'dbx.featuretables', 'dbx.probes')
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
    assert not any(hasattr(m, name) for m in (dbx, backbones, datatables, dataparts, featuretables, probes))
