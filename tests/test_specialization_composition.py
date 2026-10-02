"""
A block finds what it built before even when blocks NESTED in it were respelled.

A nested block renders inline in its container's identity, as its spec. So a
rename inside it moves its container's hash -- and the container's own
specializations, which rename only its own fields, could not reach back. Now
each nested block contributes its own past: a block looking for its old builds
renders each nested block as one of THAT block's own specializations says it
was, recursively. A rename is declared once, by the class whose field it is.

The chain mirrors a BipolarDeepFeatureBag:

    Top(upstream=Mid(upstream=Source, collator=Leaf))   was   Top(featuretab=Mid(datapoint_tab=Source, collator=Leaf))
    Leaf(columns={'signals': ..., 'labels': ...}, recursive=False)   was   Leaf(signals=..., labels=...)
"""
from dataclasses import dataclass, field

import pytest

from dbx.datablocks import DATAFILE, SAME, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Built(Datablock):
    TOPICS = {'out': DATAFILE('out.txt')}

    def __build__(self):
        with open(self.path('out', ensure_dirpath=True), 'w') as f:
            f.write(type(self).__name__)


class Source(Built):
    @dataclass
    class VAR(Datablock.VAR):
        n: int = 1


# --- before -----------------------------------------------------------------

class OldLeaf(Datablock):
    TOPICS = {}

    @dataclass
    class VAR(Datablock.VAR):
        signals: list = field(default_factory=list)
        labels: list = field(default_factory=list)


class OldMid(Built):
    @dataclass
    class VAR(Datablock.VAR):
        datapoint_tab: Datablock = None
        collator: Datablock = None


class OldTop(Built):
    @dataclass
    class VAR(Datablock.VAR):
        featuretab: Datablock = None
        k: int = 1


class OldHolder(Built):
    @dataclass
    class VAR(Datablock.VAR):
        mid: Datablock = None


# --- after: each class declares only its own past ---------------------------

class Leaf(Datablock):
    TOPICS = {}

    @dataclass
    class VAR(Datablock.VAR):
        columns: dict = field(default_factory=dict)
        recursive: bool = False

    SPECIALIZATIONS = [Datablock.Specialization(
        spec={'recursive': False}, topics=SAME,
        redirect_vars={'columns.signals': 'signals', 'columns.labels': 'labels'},
        note="signals and labels were fields of their own; there was no recursive")]


class Mid(Built):
    @dataclass
    class VAR(Datablock.VAR):
        upstream: Datablock = None
        collator: Datablock = None

    SPECIALIZATIONS = [Datablock.Specialization(
        spec={}, topics=SAME, redirect_vars={'upstream': 'datapoint_tab'}, note="upstream was datapoint_tab")]


class Top(Built):
    @dataclass
    class VAR(Datablock.VAR):
        upstream: Datablock = None
        k: int = 1

    SPECIALIZATIONS = [Datablock.Specialization(
        spec={}, topics=SAME, redirect_vars={'upstream': 'featuretab'}, note="upstream was featuretab")]


class Holder(Built):
    """Nothing of its own was renamed: only what it holds."""
    @dataclass
    class VAR(Datablock.VAR):
        mid: Datablock = None


SIGNALS, LABELS = [('tiles', 'tile')], []


def old_chain(url):
    src = Source(datalake=url, anchor='Source').build()
    leaf = OldLeaf(datalake=url, anchor='Leaf', spec={'signals': SIGNALS, 'labels': LABELS})
    mid = OldMid(datalake=url, anchor='Mid', spec={'datapoint_tab': src, 'collator': leaf}).build()
    top = OldTop(datalake=url, anchor='Top', spec={'featuretab': mid}).build()
    holder = OldHolder(datalake=url, anchor='Holder', spec={'mid': mid}).build()
    return mid, top, holder


def new_chain(url, recursive=False):
    src = Source(datalake=url, anchor='Source')
    leaf = Leaf(datalake=url, anchor='Leaf', spec={'columns': {'signals': SIGNALS, 'labels': LABELS},
                                                    'recursive': recursive})
    mid = Mid(datalake=url, anchor='Mid', spec={'upstream': src, 'collator': leaf})
    top = Top(datalake=url, anchor='Top', spec={'upstream': mid})
    holder = Holder(datalake=url, anchor='Holder', spec={'mid': mid})
    return mid, top, holder


def test_a_restructured_block_renders_as_it_was():
    leaf = Leaf(spec={'columns': {'signals': SIGNALS, 'labels': LABELS}})
    old = OldLeaf(spec={'signals': SIGNALS, 'labels': LABELS})
    sp = Leaf.SPECIALIZATIONS[0]
    assert leaf._typed_specdict_(omit=tuple(sp.spec), redirect_vars=sp.redirect_vars) == old._typed_specdict_()


@pytest.mark.parametrize('level', ['mid', 'top', 'holder'])
def test_every_level_finds_and_adopts_its_old_build(tmp_path, level):
    url = str(tmp_path)
    old = dict(zip(('mid', 'top', 'holder'), old_chain(url)))
    new = dict(zip(('mid', 'top', 'holder'), new_chain(url)))
    block = new[level]
    assert block.hash != old[level].hash, "the respelling is a new identity"
    row = block.find_specialization()
    assert row is not None and row.hash == old[level].hash
    block.build()                                   # adopts: nothing to build
    assert block.redirected_topics() == ['out'] and block.valid()


def test_the_holder_declares_nothing_of_its_own():
    assert not Holder.SPECIALIZATIONS, "its nested blocks' pasts are theirs"


def test_a_past_whose_pins_do_not_hold_is_not_tried(tmp_path):
    """recursive=True never existed before: no old build can be this block, and none is adopted."""
    url = str(tmp_path)
    old_chain(url)
    mid, top, holder = new_chain(url, recursive=True)
    assert mid.find_specialization() is None
    assert top.find_specialization() is None
    assert holder.find_specialization() is None


def test_a_nested_class_decides_its_own_past(tmp_path, monkeypatch):
    """Take the Leaf's specialization away -- touching neither Mid nor Top -- and neither finds its old build."""
    url = str(tmp_path)
    old_chain(url)
    monkeypatch.setattr(Leaf, 'SPECIALIZATIONS', [])
    mid, top, _ = new_chain(url)
    assert mid.find_specialization() is None
    assert top.find_specialization() is None


def test_a_composed_specialization_round_trips_as_a_record(tmp_path):
    url = str(tmp_path)
    old_chain(url)
    _, top, _ = new_chain(url)
    composed = top.find_specialization().specialization
    assert composed.nested, "found through its nested block's past"
    back = Datablock.Specialization.from_record(composed.to_dict())
    assert back == composed and top.get_hash(back) == top.get_hash(composed)


def test_rendering_a_past_leaves_the_block_as_it_is():
    """The dotted move works on a copy: rendering the past twice gives the same, and the block's own value is intact."""
    columns = {'signals': SIGNALS, 'labels': LABELS}
    leaf = Leaf(spec={'columns': columns})
    sp = Leaf.SPECIALIZATIONS[0]
    first = leaf._typed_specdict_(omit=tuple(sp.spec), redirect_vars=sp.redirect_vars)
    assert leaf._typed_specdict_(omit=tuple(sp.spec), redirect_vars=sp.redirect_vars) == first
    assert leaf.var.columns == {'signals': SIGNALS, 'labels': LABELS}
