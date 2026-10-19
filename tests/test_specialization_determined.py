"""`Specialization.determined`: a field the past never had, left out whatever it holds, because other fields determine it.

A pin leaves a field out of the past rendering only at the value it names. A
field holding a block that varies with the rest of the VAR -- a mapped feature
table's ``upstream``, its source's upstream pulled back along its map -- has no
one value to name. `determined` says which fields fix it, and leaves it out.
"""
from dataclasses import dataclass

import pytest

from dbx.datablocks import DATAFILE, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Seed(Datablock):
    TOPICS = {'seed': DATAFILE('seed.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 0


class Narrow(Datablock):
    """The block as it was: an ``x`` alone."""
    TOPICS = {'out': DATAFILE('out.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 0

    def __build__(self):
        with open(self.path('out', ensure_dirpath=True), 'w') as f:
            f.write(str(self.var.x))


class Wide(Datablock):
    """The block as it is: ``x``, and a block ``derived`` from it -- left out of the past, whatever it holds."""
    TOPICS = {'out': DATAFILE('out.txt')}
    SPECIALIZATIONS = [Datablock.Specialization(
        spec={}, topics={'out': DATAFILE('out.txt')}, anchor=f'{__name__}.Narrow',
        determined={'derived': ('x',)}, note="derived is Seed(n=x): fixed by x, which the past keeps")]

    @dataclass
    class VAR(Datablock.VAR):
        x: int = 0
        derived: Datablock = None


class Downstream(Datablock):
    """A block built on either."""
    TOPICS = {'down': DATAFILE('down.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        upstream: Datablock = None


def _wide(url, x):
    return Wide(datalake=url, spec=dict(x=x, derived=Seed(datalake=url, spec=dict(n=x))))


def test_a_determined_field_is_left_out_whatever_it_holds(tmp_path):
    url = str(tmp_path)
    for x in (1, 2):
        wide, narrow = _wide(url, x), Narrow(datalake=url, spec=dict(x=x))
        assert wide.hash != narrow.hash
        (sp,) = Wide.SPECIALIZATIONS
        assert wide.get_hash(sp) == narrow.hash, "the past rendering is the narrow block's, whatever derived holds"


def test_a_block_built_on_it_finds_its_past_through_the_nested_rendering(tmp_path):
    """Offered only because the narrow block was built: a past nothing recorded is a combination that never was."""
    url = str(tmp_path)
    Narrow(datalake=url, spec=dict(x=3)).build()
    old = Downstream(datalake=url, spec=dict(upstream=Narrow(datalake=url, spec=dict(x=3))))
    new = Downstream(datalake=url, spec=dict(upstream=_wide(url, 3)))
    assert new.hash != old.hash
    assert old.hash in [new.get_hash(v) for v in new._specialization_variants_()]


def test_what_determined_refuses():
    with pytest.raises(ValueError, match="pinned in spec= too"):
        Datablock.Specialization(spec={'derived': None}, topics={}, determined={'derived': ('x',)})
    with pytest.raises(ValueError, match="left out themselves"):
        Datablock.Specialization(spec={}, topics={}, determined={'a': ('b',), 'b': ('x',)})
    with pytest.raises(ValueError, match="must name the fields"):
        Datablock.Specialization(spec={}, topics={}, determined={'a': ()})
    with pytest.raises(TypeError, match="maps fields"):
        Datablock.Specialization(spec={}, topics={}, determined=['a'])


def test_a_claim_about_fields_a_block_lacks_does_not_match_it(tmp_path):
    sp = Datablock.Specialization(spec={}, topics={}, determined={'derived': ('nowhere',)})
    assert not _wide(str(tmp_path), 1)._pins_match_(sp)


def test_it_round_trips_as_a_record_and_leaves_the_keys_of_others_as_they_were():
    (sp,) = Wide.SPECIALIZATIONS
    assert Datablock.Specialization.from_record(sp.to_dict()) == sp
    assert sp.to_dict()['determined'] == {'derived': ['x']}
    plain = Datablock.Specialization(spec={}, topics={'out': DATAFILE('out.txt')})
    assert not any(isinstance(e, tuple) and e[:1] == ('determined',) for e in plain.key), "no trace when unset"
    assert sp.omitted == ('derived',)


def test_a_determined_field_this_class_lacks_is_a_mistake_said_loudly(tmp_path):
    """As a pin naming a field the class lacks: a declaration that quietly never matches is the failure that costs."""
    wide = _wide(str(tmp_path), 1)
    sp = Datablock.Specialization(spec={}, topics={'out': DATAFILE('out.txt')}, determined={'derived': ('nowhere',)})
    with pytest.raises(ValueError, match="does not declare"):
        wide._specialization_mismatch_(sp)
