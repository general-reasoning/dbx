"""
A Specialization declares the narrower block's topics; nothing is inherited.

Reconstructing an older block's identity used to render its topic NAMES with
the current class's nodes and era -- so a block whose topics were spelled with
the sentinels (SLICETOPIC) could not be reconstructed from the class once it
was respelled with the markers (DATASLICE): the reconstruction came out in the
new spelling, a hash nothing was built under, and was then refused as "this
block's own identity".
"""
import pytest

from dbx.datablocks import DIRTOPIC, Datablock
from dbx.datatables import DATASLICE, SLICETOPIC, Datatab, Datatable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class OldTab(Datatab):
    """Spelled with the sentinel, as it was when it was built."""
    TOPICS = {'tiles': SLICETOPIC}


class NewTab(Datatab):
    """The same tab respelled with a marker -- a new identity -- specialized to the old one."""
    TOPICS = {'tiles': DATASLICE(tile='ndarray:uint8')}
    SPECIALIZATIONS = [Datablock.Specialization(
        spec={}, topics={'tiles': SLICETOPIC},
        note="respelled only: the slice holds what it always did")]


class OldTable(Datatable):
    TAB = OldTab
    TOPICS = {'tab_paths': DIRTOPIC, 'done': 'done'}


from dbx.datablocks import DIR, DATAFILE  # noqa: E402


class NewTable(Datatable):
    """Respelled with the markers; its own identity no longer carries the TAB's slices."""
    TAB = NewTab
    TOPICS = {'tab_paths': DIR, 'done': DATAFILE('done')}
    SPECIALIZATIONS = [Datablock.Specialization(
        spec={}, topics={'tab_paths': DIRTOPIC, 'done': 'done'},
        note="respelled only")]


class TestARespelledBlockReconstructsTheOldOne:

    def test_a_tab(self, tmp_path):
        new, old = NewTab(url=str(tmp_path)), OldTab(url=str(tmp_path))
        sp = NewTab.SPECIALIZATIONS[0]
        assert new.hash != old.hash, "the respelling is a new identity"
        assert new.get_hash(sp) == old.hash
        assert new.get_typestr(sp) == old.typestr()
        assert new._specialization_mismatch(sp) is None
        assert 'topic:tiles=SLICETOPIC' in new.get_typestr(sp)

    def test_a_table_adds_its_tabs_slices_as_the_old_era_did(self, tmp_path):
        new, old = NewTable(url=str(tmp_path)), OldTable(url=str(tmp_path))
        sp = NewTable.SPECIALIZATIONS[0]
        assert 'topic:tiles=SLICETOPIC' in old.typestr()          # the sentinel era accumulated
        assert 'topic:tiles' not in new.typestr()                  # the marker era does not
        assert new.get_typestr(sp) == old.typestr()
        assert new.get_hash(sp) == old.hash


class TestNothingIsInherited:

    def test_a_list_of_names_is_refused(self):
        with pytest.raises(TypeError, match="names topics without declaring them"):
            Datablock.Specialization(spec={}, topics=['tiles'])

    def test_the_declaration_not_the_class_is_rendered(self, tmp_path):
        new = NewTab(url=str(tmp_path))
        other = Datablock.Specialization(spec={}, topics={'tiles': 'tiles.bin'})
        assert "topic:tiles='tiles.bin'" not in new.get_typestr(other)   # sentinel-era: bare
        assert 'topic:tiles=tiles.bin' in new.get_typestr(other)

    def test_a_record_round_trips(self):
        sp = Datablock.Specialization(spec={'w': 'hann'},
                                      topics={'tiles': DATASLICE(tile='ndarray:uint8'), 'x': 'x.npy'},
                                      version=None, note='n')
        again = Datablock.Specialization(**sp.to_dict())
        # A marker read back is a new object: equal as rendered, which is what an identity is.
        assert again.key == sp.key and repr(again.declared) == repr(sp.declared)
        assert again.topics == ('tiles', 'x')


class TestLooking:

    def test_types_typestrs_hashes_and_signatures(self, tmp_path):
        new, old = NewTab(url=str(tmp_path)), OldTab(url=str(tmp_path))
        sp = NewTab.SPECIALIZATIONS[0]
        assert new.specialization_hashes() == [old.hash]
        assert new.specialization_typestrs() == [old.typestr()]
        [t] = new.specialization_types()
        assert t == new.get_type(sp) and t['topics'] == old.type()['topics']
        assert t['version'] == old.type()['version']
        assert new.get_signature(sp) == old.signature()
        assert new.get_signaturestr(sp) == old.signaturestr()
        assert new.get_type() == new.type() and new.get_typestr() == new.typestr()

    def test_a_block_with_none_lists_none(self, tmp_path):
        old = OldTab(url=str(tmp_path))
        assert old.specialization_types() == old.specialization_typestrs() == old.specialization_hashes() == []


def test_overriding_the_old_public_name_is_refused():
    with pytest.raises(TypeError, match="now the private _topics_signature_"):
        class Tempted(Datablock):
            def signature_topics(self, topics=None):
                return ()
