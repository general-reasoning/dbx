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

from dbx.datablocks import ABSENT, DIRTOPIC, SAME, Datablock
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
    SPECIALIZATIONS = [Datatable.Specialization(
        spec={}, topics={'tab_paths': DIRTOPIC, 'done': 'done'}, TAB=None,
        note="respelled only; built before a table's type named its TAB")]


class TestARespelledBlockReconstructsTheOldOne:

    def test_a_tab(self, tmp_path):
        new, old = NewTab(datalake=str(tmp_path)), OldTab(datalake=str(tmp_path))
        sp = NewTab.SPECIALIZATIONS[0]
        assert new.hash != old.hash, "the respelling is a new identity"
        assert new.get_hash(sp) == old.hash
        assert new.get_typestr(sp) == old.typestr()
        assert new._specialization_mismatch_(sp) is None
        assert 'topic:tiles=SLICETOPIC' in new.get_typestr(sp)

    def test_a_table_adds_its_tabs_slices_as_the_old_era_did(self, tmp_path):
        new, old = NewTable(datalake=str(tmp_path)), OldTable(datalake=str(tmp_path))
        sp = NewTable.SPECIALIZATIONS[0]
        assert 'topic:tiles=SLICETOPIC' in old.typestr()          # the sentinel era accumulated
        assert 'topic:tiles' not in new.typestr()                  # the marker era does not
        # Built before a table's type named its TAB: under those rules.
        assert new.get_typestr(sp) == old.typestr(with_block=False)
        assert new.get_hash(sp) == old.get_hash(with_block=False)


class TestNothingIsInherited:

    def test_a_list_of_names_is_refused(self):
        with pytest.raises(TypeError, match="names topics without declaring them"):
            Datablock.Specialization(spec={}, topics=['tiles'])

    def test_the_declaration_not_the_class_is_rendered(self, tmp_path):
        new = NewTab(datalake=str(tmp_path))
        other = Datablock.Specialization(spec={}, topics={'tiles': 'tiles.bin'})
        assert "topic:tiles='tiles.bin'" not in new.get_typestr(other)   # sentinel-era: bare
        assert 'topic:tiles=tiles.bin' in new.get_typestr(other)

    def test_a_record_round_trips(self):
        sp = Datablock.Specialization(spec={'w': 'hann'},
                                      topics={'tiles': DATASLICE(tile='ndarray:uint8'), 'x': 'x.npy'},
                                      version=None, note='n')
        again = Datablock.Specialization.from_record(sp.to_dict())
        # A marker read back is a new object: equal as rendered, which is what an identity is.
        assert again.key == sp.key and repr(again.topics) == repr(sp.topics)
        assert tuple(again.topics) == ('tiles', 'x')

    def test_a_record_from_before_topics_was_the_declaration_still_reads(self):
        """Records once carried the names as `topics` and the declaration as `declared`."""
        old = {'spec': {}, 'topics': ['tiles'], 'declared': "{'tiles': 'SLICETOPIC'}"}
        sp = Datablock.Specialization.from_record(old)
        assert sp.topics == {'tiles': 'SLICETOPIC'} and tuple(sp.topics) == ('tiles',)

    def test_note_is_last(self):
        sp = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, anchor='a.B', note='why')
        assert repr(sp).endswith("note='why')") and list(sp.to_dict())[-1] == 'note'
        with pytest.raises(TypeError):
            Datablock.Specialization({}, {'x': 'x.txt'}, ABSENT, SAME, 'why')   # keyword-only


class TestLegacySpecialization:

    def test_legacy_default_and_none(self):
        sp = Datablock.Specialization(spec={}, topics={'x': 'x.txt'})
        assert sp.legacy is None
        assert sp.legacy_typing is False
        assert sp.legacy_signature is False
        assert 'legacy' not in sp.to_dict()

        sp_none = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=None)
        assert sp_none.legacy is None
        assert sp_none.legacy_typing is False
        assert sp_none.legacy_signature is False

        sp_false = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=False)
        assert sp_false.legacy is None

    def test_legacy_true_and_all(self):
        sp_true = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=True)
        assert sp_true.legacy == ('all',)
        assert sp_true.legacy_typing is True
        assert sp_true.legacy_signature is True
        assert sp_true.to_dict()['legacy'] == ['all']

        sp_all_str = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy='all')
        assert sp_all_str.legacy == ('all',)
        assert sp_all_str.key == sp_true.key

        sp_all_list = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=['all'])
        assert sp_all_list.legacy == ('all',)
        assert sp_all_list.key == sp_true.key

    def test_legacy_individual_flags(self):
        sp_typing = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=['typing'])
        assert sp_typing.legacy == ('typing',)
        assert sp_typing.legacy_typing is True
        assert sp_typing.legacy_signature is False
        assert sp_typing.to_dict()['legacy'] == ['typing']

        sp_sig = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=['signature'])
        assert sp_sig.legacy == ('signature',)
        assert sp_sig.legacy_typing is False
        assert sp_sig.legacy_signature is True
        assert sp_sig.to_dict()['legacy'] == ['signature']

        sp_both = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=['signature', 'typing'])
        assert sp_both.legacy == ('signature', 'typing')
        assert sp_both.legacy_typing is True
        assert sp_both.legacy_signature is True
        assert sp_both.to_dict()['legacy'] == ['signature', 'typing']

        # Order invariance
        sp_both_rev = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=['typing', 'signature'])
        assert sp_both_rev.legacy == ('signature', 'typing')
        assert sp_both.key == sp_both_rev.key

    def test_legacy_invalid_flags(self):
        with pytest.raises(ValueError, match="unrecognized flag"):
            Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=['invalid'])

        with pytest.raises(TypeError, match="must be a list/tuple of strings"):
            Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=123)

    def test_legacy_from_record_backward_compatibility(self):
        # Round trip modern
        sp = Datablock.Specialization(spec={}, topics={'x': 'x.txt'}, legacy=['signature', 'typing'])
        restored = Datablock.Specialization.from_record(sp.to_dict())
        assert restored.legacy == ('signature', 'typing')
        assert restored.legacy_typing is True
        assert restored.legacy_signature is True

        # Historical record with legacy=True
        hist_true = {'spec': {}, 'topics': "{'x': 'x.txt'}", 'legacy': True}
        restored_true = Datablock.Specialization.from_record(hist_true)
        assert restored_true.legacy == ('all',)
        assert restored_true.legacy_typing is True
        assert restored_true.legacy_signature is True

        # Historical record with legacy_typing and legacy_signature fields
        hist_fields = {'spec': {}, 'topics': "{'x': 'x.txt'}", 'legacy_typing': True, 'legacy_signature': True}
        restored_fields = Datablock.Specialization.from_record(hist_fields)
        assert restored_fields.legacy == ('signature', 'typing')
        assert restored_fields.legacy_typing is True
        assert restored_fields.legacy_signature is True

        # Historical record with only legacy_typing
        hist_typing = {'spec': {}, 'topics': "{'x': 'x.txt'}", 'legacy_typing': True}
        restored_typing = Datablock.Specialization.from_record(hist_typing)
        assert restored_typing.legacy == ('typing',)
        assert restored_typing.legacy_typing is True
        assert restored_typing.legacy_signature is False


class TestLooking:

    def test_types_typestrs_hashes_and_signatures(self, tmp_path):
        new, old = NewTab(datalake=str(tmp_path)), OldTab(datalake=str(tmp_path))
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
        old = OldTab(datalake=str(tmp_path))
        assert old.specialization_types() == old.specialization_typestrs() == old.specialization_hashes() == []


def test_overriding_the_old_public_name_is_refused():
    with pytest.raises(TypeError, match="now the private _topics_signature_"):
        class Tempted(Datablock):
            def signature_topics(self, topics=None):
                return ()
