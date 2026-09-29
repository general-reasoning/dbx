"""A block whose VAR field was renamed reads the build made under the old name.

Renaming a field re-keys every block of the class, though nothing about what
it computes has changed. ``redirect_vars={current: historical}`` reconstructs
the old identity by rendering each renamed field under its historical name --
and in the historical sort order, which a rename can change.

As in test_specializations, the evolution is two classes sharing one anchor.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import DATAFILE, Datablock

ANCHOR = 'Renamed'


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Old(Datablock):
    """Before the rename: `layer` sorts after `kappa`."""
    TOPICS = {'features': DATAFILE('features.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        layer: str = 'final'
        kappa: float = 0.5

    def __build__(self):
        with open(self.path('features', ensure_dirpath=True), 'w') as f:
            f.write(f"features-{self.var.layer}-{self.var.kappa}")

    def __read__(self, topic):
        with open(self.path(topic)) as f:
            return f.read()


class New(Old):
    """After the rename: `feature` sorts before `kappa`."""

    @dataclass
    class VAR(Datablock.VAR):
        feature: str = 'final'
        kappa: float = 0.5

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, topics={'features': DATAFILE('features.txt')},
            redirect_vars={'feature': 'layer'},
            note="VAR field renamed from layer to feature"),
    ]

    def __build__(self):
        with open(self.path('features', ensure_dirpath=True), 'w') as f:
            f.write(f"features-{self.var.feature}-{self.var.kappa}")


SP = New.SPECIALIZATIONS[0]


def old(url):
    return Old(datalake=str(url), anchor=ANCHOR, spec={'layer': 'final', 'kappa': 0.5})


def new(url, **kw):
    return New(datalake=str(url), anchor=ANCHOR, spec={'feature': 'final', 'kappa': 0.5}, **kw)


class TestTheReconstructedIdentity:

    def test_the_rename_is_a_new_identity(self, tmp_path):
        assert new(tmp_path).hash != old(tmp_path).hash

    def test_it_is_the_old_blocks_hash(self, tmp_path):
        assert new(tmp_path).get_hash(SP) == old(tmp_path).hash
        assert new(tmp_path).get_typestr(SP) == old(tmp_path).typestr()

    def test_the_historical_name_and_order_are_rendered(self, tmp_path):
        assert list(new(tmp_path).get_type(SP)['signature']['spec']) == ['kappa', 'layer']
        assert list(new(tmp_path).signature()['spec']) == ['feature', 'kappa']

    def test_the_pretty_form_renders_it_too(self, tmp_path):
        pretty = new(tmp_path).get_signaturestr(SP, pretty=True)
        assert "'layer'" in pretty and "'feature'" not in pretty

    def test_it_is_not_refused_as_the_blocks_own_identity(self, tmp_path):
        assert new(tmp_path)._specialization_mismatch_(SP) is None

    def test_it_keys_apart_from_the_plain_specialization(self):
        plain = Datablock.Specialization(spec={}, topics={'features': DATAFILE('features.txt')})
        assert SP.key != plain.key


class TestReadingThroughIt:

    def test_the_old_build_is_read(self, tmp_path):
        built = old(tmp_path)
        built.build()
        block = new(tmp_path).specialize()
        assert block.specialization == SP
        assert block.path('features') == built.path('features')
        assert block.read('features') == 'features-final-0.5'
        assert block.valid() is True

    def test_the_found_row_carries_it(self, tmp_path):
        """A SpecializationRow is a Specialization too: it must hash as the declared one."""
        old(tmp_path).build()
        block = new(tmp_path)
        row = block.find_specialization()
        assert row and row.redirect_vars == {'feature': 'layer'}
        assert row.key == SP.key
        assert block.get_hash(row) == block.get_hash(SP) == old(tmp_path).hash


class TestTheDeclaration:

    def test_a_non_dict_is_refused(self):
        with pytest.raises(TypeError, match="redirect_vars"):
            Datablock.Specialization(spec={}, topics={}, redirect_vars=[('feature', 'layer')])

    def test_a_record_round_trips(self):
        again = Datablock.Specialization.from_record(SP.to_dict())
        assert again.redirect_vars == {'feature': 'layer'} and again.key == SP.key

    def test_it_is_in_the_repr(self):
        assert "redirect_vars={'feature': 'layer'}" in repr(SP)
