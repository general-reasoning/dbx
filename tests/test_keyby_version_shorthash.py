"""keyby='version_shorthash', the default: a block's tag is a label, not a location -- except a tablet's.

Under the old default, 'tag_version_shorthash', a block's tag was in its path,
so tagging a block that was already built moved it. 'version_shorthash' keys
an untagged block exactly as that did, and a tagged one as if it were not
tagged. A `Datatablet` keeps 'tag_version_shorthash' as its KEYBY: tables'
tablets are built by the thousand under their tags, and none may move.
"""
import os
import pickle
from dataclasses import dataclass

import pytest

import dbx
from dbx.datablocks import DATAFILE, Datablock
from dbx.journals import Block as JournalBlock, PipeJournal
from dbx.pipelines import pipeline
from test_partition_coalesce import RowsTable, TABLETS, split  # noqa: E402
from test_partition_methods import built_parts  # noqa: E402
from test_core_partition import core, points  # noqa: E402


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Leaf(Datablock):
    TOPICS = {'x': DATAFILE('x.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 1

    def __build__(self):
        with open(self.path('x', ensure_dirpath=True), 'w') as f:
            f.write(str(self.var.n))


class VersionedLeaf(Leaf):
    VERSION = '2'


OLD = 'tag_version_shorthash'


def _keyed(path, tag):
    return tag in path.split(os.sep)


# ---------------------------------------------------------------------------
# An untagged block is where it always was
# ---------------------------------------------------------------------------

@pytest.mark.pinned
@pytest.mark.parametrize('cls', [Leaf, VersionedLeaf])
def test_an_untagged_blocks_path_is_the_old_defaults(tmp_path, cls):
    """With or without a VERSION: 'version_shorthash' is 'tag_version_shorthash' with no tag to put in."""
    new, old = cls(datalake=str(tmp_path), spec={'n': 3}), cls(datalake=str(tmp_path), spec={'n': 3}, keyby=OLD)
    assert new.keyby == 'version_shorthash'
    assert new.key == old.key and new.anchorkeypath == old.anchorkeypath
    assert new.key == (f"version=2/{new.hash[:8]}" if cls is VersionedLeaf else new.hash[:8])
    old.build()
    assert new.valid()


@pytest.mark.parametrize('cls', [Leaf, VersionedLeaf])
def test_the_journal_reconstructs_the_key(tmp_path, cls):
    block = cls(datalake=str(tmp_path), spec={'n': 3}, tag='X')
    assert JournalBlock._key_from_('version_shorthash', block.hash, 'X', block.version) == block.key


def test_version_hash_still_spells_the_whole_hash_without_a_version(tmp_path):
    block = Leaf(datalake=str(tmp_path), keyby='version_hash')
    assert block.key == block.hash


# ---------------------------------------------------------------------------
# A tagged block: its tag is a label, unless it asks to be keyed by it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cls', [Leaf, VersionedLeaf])
def test_a_tag_moves_nothing_by_default(tmp_path, cls):
    """The case this is for: tagging a block already built finds it built."""
    cls(datalake=str(tmp_path), spec={'n': 3}).build()
    tagged = cls(datalake=str(tmp_path), spec={'n': 3}, tag='GIGAPATH_DEEP_CPTAC_602020_TEST')
    assert tagged.tag == 'GIGAPATH_DEEP_CPTAC_602020_TEST'
    assert not _keyed(tagged.anchorkeypath, tagged.tag)
    assert tagged.anchorkeypath == cls(datalake=str(tmp_path), spec={'n': 3}).anchorkeypath
    assert tagged.valid()


@pytest.mark.parametrize('cls', [Leaf, VersionedLeaf])
def test_an_explicit_tag_keyby_is_honoured(tmp_path, cls):
    tagged = cls(datalake=str(tmp_path), spec={'n': 3}, tag='X', keyby=OLD)
    assert tagged.key == '/'.join(['X'] + ([f"version={tagged.version}"] if tagged.version else []) + [tagged.hash[:8]])
    cls(datalake=str(tmp_path), spec={'n': 3}).build()
    assert not tagged.valid(), "under its tag, apart from the untagged build"


def test_an_explicit_keyby_survives_quote_pickle_and_set(tmp_path):
    block = Leaf(datalake=str(tmp_path), tag='X', keyby='hash')
    for again in (dbx.eval(block.quote()), pickle.loads(pickle.dumps(block)), block.set(verbose=True)):
        assert again.keyby == 'hash' and again.key == block.hash
    assert Leaf(datalake=str(tmp_path), tag='X').set(keyby=OLD).key.startswith('X/')


def test_a_quote_recorded_under_the_old_default_keeps_its_path(tmp_path):
    """Every quote spells its keyby, so one journalled before reconstructs where it was built."""
    old = Leaf(datalake=str(tmp_path), spec={'n': 3}, tag='X', keyby=OLD)
    assert "keyby='tag_version_shorthash'" in old.quote()
    assert dbx.eval(old.quote()).anchorkeypath == old.anchorkeypath


@pytest.mark.pinned
def test_the_default_moves_no_hash(tmp_path):
    """keyby is no identity: neither the default nor a tag changes a hash, typed or legacy."""
    a = Leaf(datalake=str(tmp_path), spec={'n': 3})
    b = Leaf(datalake=str(tmp_path), spec={'n': 3}, tag='X', keyby=OLD)
    assert a.hash == b.hash
    assert a.signaturestr(legacy_typing=True) == b.signaturestr(legacy_typing=True)


def test_a_tag_keyed_narrower_build_is_found_by_its_directory(tmp_path):
    """A specialization of a block at the new default finds what was built, tagged, under the old one."""
    class Narrow(Leaf):
        pass

    class Grown(Leaf):
        TOPICS = {'x': DATAFILE('x.txt'), 'y': DATAFILE('y.txt')}
        SPECIALIZATIONS = [Datablock.Specialization(spec={}, topics={'x': DATAFILE('x.txt')})]

        def __build__(self):
            for topic in self.ownedtopics():
                with open(self.path(topic, ensure_dirpath=True), 'w') as f:
                    f.write(topic)

    Narrow(datalake=str(tmp_path), anchor=Grown.anchor, spec={'n': 3}, tag='X', keyby=OLD).build()
    grown = Grown(datalake=str(tmp_path), spec={'n': 3}, tag='X')
    assert grown.find_specialization() is not None


# ---------------------------------------------------------------------------
# A table's tablets keep their tags in their paths, built and read back alike
# ---------------------------------------------------------------------------

def _tag_keyed(tablet):
    return (tablet.keyby == OLD and _keyed(tablet.anchorkeypath, tablet.tag)
            and tablet.anchorkeypath == tablet.set(keyby=OLD).anchorkeypath)


def test_a_tables_tablets_are_tag_keyed_built_and_read(tmp_path):
    table = RowsTable(datalake=str(tmp_path), spec={'tablets': [[list(r) for r in t] for t in TABLETS]})
    assert table.keyby == 'version_shorthash'
    formed = [table.__tablet__(i) for i in range(table.n_tablets)]     # as an override's super() forms it
    assert all(_tag_keyed(t) for t in formed)
    table.build()
    for i in range(table.n_tablets):
        tablet = table.tablet(i)
        assert _tag_keyed(tablet) and tablet.tag == f"tab_{i:06d}"
        assert tablet.anchorkeypath == formed[i].anchorkeypath
        assert tablet.valid(), "built where it is read"


def test_a_pickled_or_requoted_table_still_leaves_its_tablets_their_own_keyby(tmp_path):
    """A table spells its keyby once round-tripped; it is still the default, so it is handed to no tablet."""
    table = RowsTable(datalake=str(tmp_path), spec={'tablets': [[list(r) for r in t] for t in TABLETS]}).build()
    for again in (pickle.loads(pickle.dumps(table)), dbx.eval(table.quote()), table.set(verbose=True)):
        assert all(_tag_keyed(again.tablet(i)) and again.tablet(i).valid() for i in range(again.n_tablets))


def test_an_explicit_keyby_on_a_table_is_handed_to_its_tablets(tmp_path):
    table = RowsTable(datalake=str(tmp_path), keyby='hash',
                      spec={'tablets': [[list(r) for r in t] for t in TABLETS]})
    assert table.key == table.hash
    table.build()
    for i in range(table.n_tablets):
        tablet = table.tablet(i)
        assert tablet.keyby == 'hash' and tablet.key == tablet.hash and tablet.valid()


@pytest.mark.parametrize('coalesce', [None, 'pack'])
def test_a_parts_pieces_and_packed_tablets_are_tag_keyed_built_and_read(tmp_path, coalesce):
    table = RowsTable(datalake=str(tmp_path), spec={'tablets': [[list(r) for r in t] for t in TABLETS]}).build()
    spec = {'coalesce': 'pack', 'min_tablet_rows': 3, 'max_tablet_rows': 4} if coalesce else {}
    partition = split(tmp_path, table, **spec)
    kinds = set()
    for part in built_parts(partition):
        for i in range(part.n_tablets):
            tablet = part.tablet(i)
            kinds.add(type(tablet).__name__)
            assert _tag_keyed(tablet) and tablet.valid(), (type(tablet).__name__, tablet.anchorkeypath)
    assert 'MappedDatatablet' in kinds


def test_core_tablets_are_tag_keyed_built_and_read(tmp_path):
    partition = core(tmp_path, points(tmp_path, n_tablets=3, n=20))
    for fold in range(2):
        part = partition.part(fold).pullback(None, partition.var.collator).build()
        for i in range(part.n_tablets):
            tablet = part.tablet(i)
            assert type(tablet).__name__ == 'MappedDatatablet'
            assert _tag_keyed(tablet) and tablet.valid()
            assert tablet.anchorkeypath == partition.part(fold).pullback(None, partition.var.collator).tablet(i).anchorkeypath


# ---------------------------------------------------------------------------
# Pipelines: their journal is collected by scanning, so both layouts are found
# ---------------------------------------------------------------------------

@pipeline
def tagged_leaf(name, *, var_n: int = 1, datalake=None):
    return Leaf(datalake=datalake, spec={'n': var_n}, tag=name)


def test_the_pipejournal_finds_creations_keyed_either_way(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
    new = tagged_leaf('a')
    old = new.set(keyby=OLD)
    assert old.anchorkeypath != new.anchorkeypath and _keyed(old.anchorkeypath, old.tag)
    old.create()
    new.create()
    frame = PipeJournal.collect(datalake=str(tmp_path))
    assert set(frame['anchorkeypath']) == {old.anchorkeypath, new.anchorkeypath}
    assert set(frame['hash']) == {new.hash}
    assert len(dbx.pipejournal(datalake=str(tmp_path), tag='tagged_leaf')) == 2
