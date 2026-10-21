"""``UNSAFE_ship`` — copying topics OUT, to readers who have no dbx.

The outbound mirror of ``UNSAFE_copy_from``: a target per topic, and only the topics
named. ``UNSAFE_ship_redirect`` then points the block at what it shipped, so the
shipped copy is the single source of truth rather than a second one that can drift.
"""
import json
import os

import pytest

from dbx.datablocks import DATADIR, DATAFILE, SYNTHETIC, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Cfg(Datablock):
    """A generated config directory, a facts file, and a topic with no location."""
    TOPICS = {'configs': DATADIR('hydra config dir'),
              'facts': DATAFILE('facts.json'),
              'scratch': SYNTHETIC}

    def __build__(self):
        data = os.path.join(self.path('configs', ensure_dirpath=True), 'data')
        os.makedirs(data, exist_ok=True)
        with open(os.path.join(data, 'bc.yaml'), 'w') as f:
            f.write('source: abfss://x\n')
        with open(self.path('facts', ensure_dirpath=True), 'w') as f:
            json.dump({'windows': 321337}, f)
        return self

    def __read__(self, topic='facts'):
        if topic == 'facts':
            with open(self.path('facts')) as f:
                return json.load(f)
        return self.path('configs')


@pytest.fixture
def block(tmp_path):
    return Cfg(url=str(tmp_path / 'store')).build()


def test_ship_copies_a_file_topic_and_a_dir_topic(block, tmp_path):
    repo = tmp_path / 'repo'
    shipped = block.UNSAFE_ship({'configs': str(repo / 'configs')},
                                facts=str(repo / 'facts.json'),
                                OVERRIDE=True, show_progress=False)
    assert shipped == {'configs': str(repo / 'configs'), 'facts': str(repo / 'facts.json')}
    assert (repo / 'configs' / 'data' / 'bc.yaml').is_file()
    assert json.loads((repo / 'facts.json').read_text()) == {'windows': 321337}


@pytest.mark.pinned
def test_an_unnamed_topic_is_not_shipped(block, tmp_path):
    """Shipping one topic must not drag the others along: a block ships its generated
    config and keeps its checkpoints to itself."""
    repo = tmp_path / 'repo'
    block.UNSAFE_ship(facts=str(repo / 'facts.json'), OVERRIDE=True, show_progress=False)
    assert (repo / 'facts.json').is_file()
    assert not (repo / 'configs').exists()


def test_a_syntopic_ships_nothing(block, tmp_path):
    repo = tmp_path / 'repo'
    assert block.UNSAFE_ship(scratch=str(repo / 'scratch'),
                             OVERRIDE=True, show_progress=False) == {}
    assert not (repo / 'scratch').exists()


def test_an_unknown_topic_is_refused(block, tmp_path):
    with pytest.raises(ValueError, match='not topics of this block'):
        block.UNSAFE_ship(nope=str(tmp_path / 'x'), OVERRIDE=True, show_progress=False)


def test_naming_no_topic_is_refused(block):
    with pytest.raises(ValueError, match='names no topic'):
        block.UNSAFE_ship(OVERRIDE=True, show_progress=False)


@pytest.mark.pinned
def test_an_existing_target_is_not_silently_replaced(block, tmp_path):
    """A target is somewhere else's data — a repo file, a shared directory. Replacing
    one is a thing to ask for, not a thing to discover afterwards."""
    target = str(tmp_path / 'repo' / 'facts.json')
    block.UNSAFE_ship(facts=target, OVERRIDE=True, show_progress=False)
    with pytest.raises(FileExistsError):
        block.UNSAFE_ship(facts=target, OVERRIDE=True, show_progress=False)
    block.UNSAFE_ship(facts=target, overwrite=True, OVERRIDE=True, show_progress=False)


@pytest.mark.pinned
def test_an_invalid_block_is_not_shipped(tmp_path):
    """Shipping sends data out. A block that has not built has none to send."""
    unbuilt = Cfg(url=str(tmp_path / 'store2'))
    with pytest.raises(AssertionError):
        unbuilt.UNSAFE_ship(facts=str(tmp_path / 'repo' / 'f.json'),
                            OVERRIDE=True, show_progress=False)


@pytest.mark.pinned
def test_ship_redirect_makes_the_shipped_copy_the_one_that_is_read(block, tmp_path):
    """The point of shipping-with-redirect: one copy, not two that drift."""
    target = tmp_path / 'repo' / 'facts.json'
    own = block.path('facts')
    block.UNSAFE_ship_redirect(facts=str(target), OVERRIDE=True, show_progress=False)

    assert block.redirected()
    assert block.path('facts') == str(target)
    with open(target, 'w') as f:
        json.dump({'windows': 999}, f)
    assert block.read('facts') == {'windows': 999}

    os.remove(own)                      # the block's own copy is no longer consulted
    assert block.read('facts') == {'windows': 999}
    assert block.valid()


@pytest.mark.pinned
def test_ship_redirect_leaves_unnamed_topics_reading_their_own_data(block, tmp_path):
    store = str(tmp_path / 'store')
    block.UNSAFE_ship_redirect(facts=str(tmp_path / 'repo' / 'facts.json'),
                               OVERRIDE=True, show_progress=False)
    assert block.path('configs').startswith(store)
