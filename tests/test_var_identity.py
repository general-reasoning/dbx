"""A VAR value that cannot be rendered deterministically is refused.

A block's hash is sha256 of its type string, and the type string renders every
VAR leaf: a Datablock as its own spec, a specline as the line, and anything
else through ``repr()``.  For an object with no content-bearing ``__repr__``
that is ``<C object at 0x...>`` -- a memory address, in the hash.

Which makes the block's identity a function of where it happened to be
allocated.  ``set()`` is a deepcopy and reconstruct, and `Datastack.block()`
reaches every child through it, so the address moved on every call: the stack
wrote its children under one key and looked them up under another, found
nothing, and rebuilt over what it had already written.  Two directories, one
logical block, no error anywhere.

The check is at `Datablock.VAR.LazyLoader`, the one funnel every VAR value
passes through before the identity can reach it, and it is structural.  A
``__repr__`` added to quiet the symptom would leave the hole open: two values
with different content and one repr collide onto a single hash, and nothing in
dbx could tell.
"""
from dataclasses import dataclass

import pytest

from dbx import dataparts
from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Plain:
    """No __repr__, so repr() gives an address."""

    def __init__(self, n=1):
        self.n = n


class Held(Datablock):
    VERSION = 1
    TOPICS = {'out': 'out.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        thing: object = None

    def __build__(self):
        pass


class Upstream(Datablock):
    VERSION = 1
    TOPICS = {'out': 'out.txt'}

    def __build__(self):
        pass


def block(cls, tmp_path, **spec):
    return cls(url=str(tmp_path), spec=spec)


@pytest.mark.pinned
class TestAHashIsAFunctionOfConfigurationAlone:
    """The same configuration, reached twice, is one block.

    Everything downstream of the hash assumes it: a key names a directory, a
    stack looks its children up by the key it wrote them under, and `valid()`
    answers about the path the key resolves to.  A hash that moves between two
    constructions of one configuration does not merely mislabel -- it silently
    duplicates the artifact and orphans the first copy.
    """

    def test_a_second_construction_agrees(self, tmp_path):
        assert block(Held, tmp_path, thing=[1, 'a']).hash == \
               block(Held, tmp_path, thing=[1, 'a']).hash

    def test_set_does_not_move_it(self, tmp_path):
        b = block(Held, tmp_path, thing={'k': [1, 2.5, None, True]})
        hashes = [b.hash]
        for _ in range(3):
            b = b.set()
            hashes.append(b.hash)
        assert len(set(hashes)) == 1

    def test_a_datablock_held_in_var_does_not_move_it(self, tmp_path):
        b = block(Held, tmp_path, thing=Upstream(url=str(tmp_path)))
        assert b.set().set().hash == b.hash

    def test_different_configurations_stay_different(self, tmp_path):
        """The guard: a hash that ignored its VAR would pass every test above."""
        assert block(Held, tmp_path, thing=[1]).hash != block(Held, tmp_path, thing=[2]).hash


class TestWhatIsRefused:

    def test_a_plain_object_is_refused(self, tmp_path):
        with pytest.raises(TypeError, match="cannot render deterministically"):
            block(Held, tmp_path, thing=Plain()).hash

    def test_the_refusal_names_the_class_and_the_field(self, tmp_path):
        """Without the name it is an hour of bisecting which field it was."""
        with pytest.raises(TypeError, match=r"Held\.VAR\.thing holds a Plain"):
            block(Held, tmp_path, thing=Plain()).hash

    def test_it_is_refused_inside_a_container(self, tmp_path):
        with pytest.raises(TypeError, match="cannot render deterministically"):
            block(Held, tmp_path, thing={'a': [Plain()]}).hash

    def test_it_is_refused_before_anything_reads_the_hash(self, tmp_path):
        """The funnel precedes the identity, so the first read of the field
        raises -- not some later rendering of it."""
        with pytest.raises(TypeError, match="cannot render deterministically"):
            block(Held, tmp_path, thing=Plain()).var.thing

    def test_an_object_with_a_repr_is_refused_too(self, tmp_path):
        """Structural, not a search for the address: a __repr__ silences the
        symptom while two values with different content and one repr still
        collide onto a single hash."""
        class Reprd(Plain):
            def __repr__(self):
                return '<Reprd>'

        with pytest.raises(TypeError, match="cannot render deterministically"):
            block(Held, tmp_path, thing=Reprd()).hash


class TestWhatPasses:

    @pytest.mark.parametrize('value', [
        None, 'text', 0, 1.5, True, b'bytes',
        [1, 'a'], (1, 'a'), {'a', 'b'}, frozenset({'a'}),
        {'k': [1, {'deep': ('nested', None)}]},
    ])
    def test_plain_data_and_containers_of_it(self, tmp_path, value):
        assert block(Held, tmp_path, thing=value).hash

    def test_a_datablock(self, tmp_path):
        assert block(Held, tmp_path, thing=Upstream(url=str(tmp_path))).hash

    def test_a_specline_whatever_it_resolves_to(self, tmp_path, monkeypatch):
        """The identity renders the LINE for anything that does not resolve to a
        block, so what it resolves to never reaches the hash."""
        monkeypatch.setattr(dataparts, 'eval', lambda term: Plain())
        b = block(Held, tmp_path, thing='$something.opaque')
        assert isinstance(b.var.thing, Plain)
        assert "'$something.opaque'" in b.signature()


class TestTheExemption:

    def test_an_exempt_field_is_not_checked(self, tmp_path):
        class Exempt(Held):
            VAR_IDENTITY_EXEMPTIONS = {'thing'}

        assert block(Exempt, tmp_path, thing=Plain()).hash

    def test_an_exemption_does_not_take_the_field_out_of_the_identity(self, tmp_path):
        """What it buys is the check, not determinism: the value goes on
        rendering through repr(), address and all."""
        class Exempt(Held):
            VAR_IDENTITY_EXEMPTIONS = {'thing'}

        b = block(Exempt, tmp_path, thing=Plain())
        assert b.set().hash != b.hash

    def test_it_exempts_only_the_field_named(self, tmp_path):
        class Exempt(Held):
            VAR_IDENTITY_EXEMPTIONS = {'other'}

        with pytest.raises(TypeError, match="cannot render deterministically"):
            block(Exempt, tmp_path, thing=Plain()).hash


class TestTheLazyLoaderResolvesOnce:
    """`None` is an ordinary resolved value, so it cannot be the memo's sentinel."""

    def test_a_specline_resolving_to_none_is_evaluated_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(dataparts, 'eval', lambda term: calls.append(term))
        loader = Datablock.VAR.LazyLoader('$nothing', name='thing', owner='Held')
        assert loader() is None and loader() is None and loader() is None
        assert calls == ['$nothing']

    def test_a_field_holding_none_resolves_to_none(self, tmp_path):
        assert block(Held, tmp_path, thing=None).var.thing is None

    def test_the_check_runs_once_rather_than_per_read(self, tmp_path):
        checked = []
        loader = Datablock.VAR.LazyLoader(Plain(), name='thing', owner='Held', exempt=True)
        original = type(loader)._check_renderable
        try:
            type(loader)._check_renderable = lambda self: checked.append(self.name)
            loader(), loader(), loader()
        finally:
            type(loader)._check_renderable = original
        assert checked == ['thing']
