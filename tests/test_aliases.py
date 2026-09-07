"""The short names are the classes themselves, so no identity can move.

``Datatab``/``Datatable`` and ``FeatureTab``/``FeatureTable`` are assignments,
not subclasses.  An alias that were a subclass would carry its own ``VAR`` and
its own name, and a block declared from it would be a different block than one
declared from the original -- the difference showing up as a hash and a storage
path, quietly, on the day someone modernised an import.

A block's hash is sha256 of its ``type()``, which is signature + version +
topics and names no class at all.  The one place a class name is recorded is
`fqcn`, in the storage path, and that is the name of the subclass being built --
not of the base it was declared from.  So switching a declaration to the short
spelling keeps both.  Renaming the subclass is what would move them.
"""
from dataclasses import dataclass

import pytest

pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

import dbx
from dbx.datapoints import (
    SLICETOPIC,
    Datatab,
    Datatable,
    DatapointTab,
    DatapointTable,
)
from dbx.datafeatures import (
    DatafeatureTab,
    DatafeatureTable,
    FeatureTab,
    FeatureTable,
)

ALIASES = [
    ('Datatab', Datatab, DatapointTab),
    ('Datatable', Datatable, DatapointTable),
    ('FeatureTab', FeatureTab, DatafeatureTab),
    ('FeatureTable', FeatureTable, DatafeatureTable),
]


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


@pytest.mark.pinned
class TestAnAliasIsTheClass:
    """The short name must not be a second class.

    Every identity claim below rests on this one: two spellings of one class
    cannot disagree about anything, and two classes always can.
    """

    @pytest.mark.parametrize('name,alias,original', ALIASES)
    def test_the_alias_is_the_original(self, name, alias, original):
        assert alias is original

    @pytest.mark.parametrize('name,alias,original', ALIASES)
    def test_it_keeps_the_original_name(self, name, alias, original):
        """So `fqcn`, and the storage path built from it, is untouched."""
        assert alias.__name__ == original.__name__


@pytest.mark.pinned
class TestDeclaringFromTheAliasChangesNothing:
    """The same class, declared two ways, is one block.

    Identity is what a hash addresses and a path resolves to, so a spelling
    that moved either would orphan every artifact built under the other.
    """

    @staticmethod
    def _cell(base):
        class Cell(base):
            VERSION = 1
            TOPICS = {'s': SLICETOPIC}

            @dataclass
            class VAR(base.VAR):
                n: int = 3

            def __build__(self):
                pass

        return Cell

    def test_the_hash_is_the_same(self, tmp_path):
        a = self._cell(DatapointTab)(url=str(tmp_path), spec=dict(n=3))
        b = self._cell(Datatab)(url=str(tmp_path), spec=dict(n=3))
        assert a.hash == b.hash

    def test_the_type_string_is_the_same(self, tmp_path):
        a = self._cell(DatapointTab)(url=str(tmp_path), spec=dict(n=3))
        b = self._cell(Datatab)(url=str(tmp_path), spec=dict(n=3))
        assert a.type() == b.type()

    def test_the_storage_path_is_the_same(self, tmp_path):
        a = self._cell(DatapointTab)(url=str(tmp_path), spec=dict(n=3))
        b = self._cell(Datatab)(url=str(tmp_path), spec=dict(n=3))
        assert a.anchorkeypath == b.anchorkeypath

    def test_no_base_class_name_reaches_the_type_string(self, tmp_path):
        """Why the above holds rather than happening to: the identity renders
        spec, version and topics, and none of them names a class."""
        text = self._cell(DatapointTab)(url=str(tmp_path), spec=dict(n=3)).type()
        assert 'DatapointTab' not in text and 'Datatab' not in text

    def test_a_renamed_subclass_does_move_the_path(self, tmp_path):
        """The guard: the path is not simply insensitive to everything."""
        a = self._cell(DatapointTab)(url=str(tmp_path), spec=dict(n=3))

        class Renamed(DatapointTab):
            VERSION = 1
            TOPICS = {'s': SLICETOPIC}

            @dataclass
            class VAR(DatapointTab.VAR):
                n: int = 3

            def __build__(self):
                pass

        assert Renamed(url=str(tmp_path), spec=dict(n=3)).anchorkeypath != a.anchorkeypath


class TestTheAliasesAreExported:

    @pytest.mark.parametrize('name,alias,original', ALIASES)
    def test_from_the_package(self, name, alias, original):
        assert getattr(dbx, name) is original
