"""``dbx.datapoints`` and ``dbx.datafeatures`` still resolve, and still mean it.

The files are now ``dbx/datatables.py`` and ``dbx/featuretables.py``, which
would have been a cosmetic change if a module name went nowhere.  It goes two
places that outlive the source:

* `fqcn` is a class's ``__module__`` plus its name, `anchor` falls back to
  `fqcn`, and `anchorkeypath` and the journal directory are built from that.
  Every artifact built by a class defined in dbx is stored under a path
  spelling the module it was defined in -- and `DatafeatureTab` and its table
  are configured by spec rather than subclassed, so it is their own fqcn, not
  a downstream subclass's.
* `quotefn` renders ``fn.__module__`` into a specline, and a specline stands in
  a spec as text.  A spec naming `DatapointTableTab` would hash differently
  under a new module name.

So the classes keep ``__module__`` as it was, and the old dotted paths are
aliased onto the same module objects -- the way ``dbx.datamodels`` already is
onto :mod:`dbx.databackbones`.  Neither half is a courtesy to importers: the
strings are recorded in artifacts on disk, which do not get to be migrated.

Aliases and not forwarding shims, because a forwarder has a namespace of its
own.  ``monkeypatch.setattr(dbx.datapoints, 'StreamingDataset', ...)`` would
patch the forwarder while the code under test read the name out of its own
module, leaving the patch a silent no-op -- and a test that believed it had
stubbed a remote read would perform one instead.  Which is not hypothetical:
`test_tab_stream_cache` does exactly that patch, and under a forwarding shim it
hung on a real download.
"""
from dataclasses import dataclass

import pytest

pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

import dbx
import dbx.datafeatures
import dbx.datapoints
from dbx import datatables, featuretables

#: The fqcn every artifact of these classes is stored under. Hard-coded, because
#: the whole point is that they are not free to move.
LEGACY_FQCNS = [
    (datatables.DatapointBase, 'dbx.datapoints.DatapointBase'),
    (datatables.DatapointTab, 'dbx.datapoints.DatapointTab'),
    (datatables.DatapointTable, 'dbx.datapoints.DatapointTable'),
    (datatables.DatapointPartition, 'dbx.datapoints.DatapointPartition'),
    (datatables.DatapointFold, 'dbx.datapoints.DatapointFold'),
    (featuretables.Datacollator, 'dbx.datafeatures.Datacollator'),
    (featuretables.DatafeatureTab, 'dbx.datafeatures.DatafeatureTab'),
    (featuretables.DatafeatureTable, 'dbx.datafeatures.DatafeatureTable'),
    (featuretables.BipolarDatafeatureTab, 'dbx.datafeatures.BipolarDatafeatureTab'),
    (featuretables.BipolarDatafeatureTable, 'dbx.datafeatures.BipolarDatafeatureTable'),
]


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.mark.pinned
class TestAClassKeepsTheFqcnItsArtifactsAreStoredUnder:
    """A class defined in dbx may not change the module it reports.

    `anchor` falls back to `fqcn`, so the fqcn is a directory name on disk --
    for artifacts that were written before this file was named what it is now.
    A class that reported a new one would look in a directory nothing was ever
    written to, find nothing, and rebuild: the data is not lost, it is
    duplicated and orphaned, and nothing raises.
    """

    @pytest.mark.parametrize('cls,fqcn', LEGACY_FQCNS,
                             ids=[f for _, f in LEGACY_FQCNS])
    def test_the_fqcn_is_unchanged(self, cls, fqcn):
        assert f"{cls.__module__}.{cls.__name__}" == fqcn

    def test_a_specline_naming_one_is_unchanged(self, tmp_path):
        """`quotefn` renders __module__, and a specline stands in a spec as
        text -- so this one is in a hash, not merely in a path."""
        quoted = dbx.quotefn(datatables.DatapointTableTab, "$DummyTable()", 0)
        assert 'dbx.datapoints.DatapointTableTab' in quoted

    def test_a_subclass_elsewhere_reports_its_own_module(self):
        """The guard. Pinning by __module__ rather than by special-casing fqcn
        is what makes this hold: Python gives every class the module it was
        defined in, so the pin cannot be inherited."""
        class Sub(datatables.Datatab):
            VERSION = 1
            TOPICS = {'s': datatables.SLICETOPIC}

            def __build__(self):
                pass

        assert Sub.__module__ == __name__
        assert Sub.__module__ != 'dbx.datapoints'


class TestTheOldModulePathsAreTheSameModule:

    def test_they_are_the_very_same_object(self):
        import sys
        assert sys.modules['dbx.datapoints'] is datatables
        assert sys.modules['dbx.datafeatures'] is featuretables

    def test_patching_one_name_is_visible_under_the_other(self, monkeypatch):
        """The reason they are aliases. A forwarding shim would make this a
        silent no-op, and a stub that does not take is worse than none."""
        sentinel = object()
        monkeypatch.setattr(dbx.datapoints, 'StreamingDataset', sentinel)
        assert datatables.StreamingDataset is sentinel


    def test_the_public_names_import(self):
        assert dbx.datapoints.DatapointTab is datatables.DatapointTab
        assert dbx.datafeatures.DatafeatureTab is featuretables.DatafeatureTab

    def test_the_new_short_names_import_from_either(self):
        assert dbx.datapoints.Datatab is datatables.Datatab
        assert dbx.datafeatures.FeatureTab is featuretables.FeatureTab

    def test_a_private_name_resolves(self):
        """`_UpstreamSlices` is imported by name elsewhere, and `import *`
        would have skipped it."""
        assert dbx.datafeatures._UpstreamSlices is featuretables._UpstreamSlices

    def test_a_module_constant_resolves(self):
        assert dbx.datapoints.SLICETOPIC is datatables.SLICETOPIC

    def test_dir_reports_the_module(self):
        assert 'DatapointTab' in dir(dbx.datapoints)

    def test_an_unknown_name_still_raises(self):
        with pytest.raises(AttributeError):
            dbx.datapoints.no_such_name

    def test_they_resolve_as_attributes_of_the_package(self):
        """A recorded specline is a dotted path off `dbx`, so the old names
        have to resolve without the caller having imported them."""
        import importlib
        assert importlib.import_module('dbx').datapoints is datatables
        assert importlib.import_module('dbx.datafeatures') is featuretables

    def test_a_recorded_specline_still_evaluates(self):
        import sys

        class _Table:
            def __init__(self, idx=None, **kwargs):
                self.idx = idx

            def __call__(self, idx=None, tag=None, **spec):
                return ('tab', self.idx if self.idx is not None else idx)

        sys.modules['dbx.datapoints']._ShimTable = _Table
        try:
            got = dbx.eval("$dbx.datapoints.DatapointTableTab("
                           "$dbx.datapoints._ShimTable(idx=7), 7)")
            assert got == ('tab', 7)
        finally:
            del sys.modules['dbx.datapoints']._ShimTable
