"""A parameterised topic marker pickles -- to a worker, inside a specialization -- as the call that made it."""
import pickle

import pytest

from dbx.datablocks import DATADICT, DATADIR, DATAFILE, DIR, SYNTHETIC, Datablock
from dbx.datatables import DATASLICE

MARKERS = [DATAFILE, DATADIR, DIR, SYNTHETIC, DATASLICE,
           DATAFILE('tabs.json'), DATAFILE('tabs.json', 'one list of tab indices per fold'),
           DATADIR('a note'), SYNTHETIC('derived on read'),
           DATADICT('bag.npz', mean='ndarray', sign='ndarray'),
           DATASLICE(tile='ndarray:uint8'), DATASLICE(annotations=dict(case_id='str', cohort='str'))]


@pytest.mark.parametrize("marker", MARKERS, ids=repr)
def test_a_marker_round_trips_to_its_own_rendering(marker):
    back = pickle.loads(pickle.dumps(marker))
    assert repr(back) == repr(marker)
    assert type(back) is type(marker)
    if marker.__qualname__ == repr(marker):
        assert back is marker, "a declared marker is the very one"


def test_a_specialization_naming_old_topics_pickles():
    sp = Datablock.Specialization(spec={}, version=1,
                                  topics={'tabs': DATAFILE('tabs.json', 'one list of tab indices per fold')},
                                  redirect_topics={'tablets': 'tabs'})
    back = pickle.loads(pickle.dumps(sp))
    assert back == sp and back.key == sp.key
