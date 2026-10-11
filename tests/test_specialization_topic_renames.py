"""A topic renamed since a block was built is read under its new name from where the old one recorded it.

``redirect_topics`` as a mapping, ``{current_name: historical_name}``, is to a
topic what ``redirect_vars`` is to a VAR field: the narrower block's identity
names the topic as it was (``topics=``), and its build recorded the path under
that name; this block reads it under the name it has now.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import DATAFILE, Datablock

ANCHOR = 'Indices'


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Before(Datablock):
    """The class with its topic under its old name."""
    TOPICS = {'tabs': DATAFILE('tabs.json')}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 3

    def __build__(self):
        with open(self.path('tabs', ensure_dirpath=True), 'w') as f:
            f.write(f"indices-{self.var.n}")


class After(Datablock):
    """The same class, its topic renamed -- the same file, now ``tablets``."""
    VERSION = 2
    TOPICS = {'tablets': DATAFILE('tabs.json')}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 3

    SPECIALIZATIONS = [
        Datablock.Specialization(
            spec={}, version=None, topics={'tabs': DATAFILE('tabs.json')}, redirect_topics={'tablets': 'tabs'},
            note="the topic tabs was renamed tablets"),
    ]

    def __build__(self):
        with open(self.path('tablets', ensure_dirpath=True), 'w') as f:
            f.write(f"rebuilt-{self.var.n}")

    def __read__(self, topic):
        with open(self.path(topic)) as f:
            return f.read()


def test_the_old_identity_names_the_old_topic(tmp_path):
    before = Before(datalake=str(tmp_path), anchor=ANCHOR, spec={'n': 3})
    after = After(datalake=str(tmp_path), anchor=ANCHOR, spec={'n': 3})
    assert after.hash != before.hash
    assert after.specialization_hashes() == [before.hash]


def test_a_renamed_topic_is_adopted_and_read_under_its_new_name(tmp_path):
    before = Before(datalake=str(tmp_path), anchor=ANCHOR, spec={'n': 3}).build()
    after = After(datalake=str(tmp_path), anchor=ANCHOR, spec={'n': 3})
    assert not after.valid()
    after.specialize()
    assert after.valid() and not after.owedtopics()
    assert after.redirected_topics() == ['tablets']
    assert after.path('tablets') == before.path('tabs')
    assert after.read('tablets') == "indices-3"


def test_a_list_still_names_topics_under_the_names_they_were_recorded(tmp_path):
    Before(datalake=str(tmp_path), anchor=ANCHOR, spec={'n': 3}).build()
    listed = Datablock.Specialization(spec={}, version=None, topics={'tabs': DATAFILE('tabs.json')},
                                      redirect_topics=['tablets'])
    after = After(datalake=str(tmp_path), anchor=ANCHOR, spec={'n': 3}, SPECIALIZATIONS=[listed])
    after.specialize()
    assert not after.valid(), "a list reads a topic only under the name it was recorded under"


def test_the_mapping_is_part_of_the_specializations_key_and_record():
    a = Datablock.Specialization(spec={}, topics={'tabs': DATAFILE('tabs.json')}, redirect_topics={'tablets': 'tabs'})
    b = Datablock.Specialization(spec={}, topics={'tabs': DATAFILE('tabs.json')}, redirect_topics=['tablets'])
    assert a.key != b.key
    assert a.to_dict()['redirect_topics'] == {'tablets': 'tabs'}
    assert Datablock.Specialization.from_record(a.to_dict()) == a
