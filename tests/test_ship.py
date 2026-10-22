"""``UNSAFE_ship`` — copying topics OUT, to readers who have no dbx.

The outbound mirror of ``UNSAFE_copy_from``: a target per topic, and only the topics
named. ``UNSAFE_ship_redirect`` then points the block at what it shipped, so the
shipped copy is the single source of truth rather than a second one that can drift.
"""
import json
import os
import pathlib

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
    shipped = block.UNSAFE_ship({'configs': str(repo / 'configs'),
                                 'facts': str(repo / 'facts.json')},
                                OVERRIDE=True, show_progress=False)
    assert shipped == {'configs': str(repo / 'configs'), 'facts': str(repo / 'facts.json')}
    assert (repo / 'configs' / 'data' / 'bc.yaml').is_file()
    assert json.loads((repo / 'facts.json').read_text()) == {'windows': 321337}


@pytest.mark.pinned
def test_an_unnamed_topic_is_not_shipped(block, tmp_path):
    """Shipping one topic must not drag the others along: a block ships its generated
    config and keeps its checkpoints to itself."""
    repo = tmp_path / 'repo'
    block.UNSAFE_ship({'facts': str(repo / 'facts.json')}, OVERRIDE=True,
                      show_progress=False)
    assert (repo / 'facts.json').is_file()
    assert not (repo / 'configs').exists()


def test_a_syntopic_ships_nothing(block, tmp_path):
    repo = tmp_path / 'repo'
    assert block.UNSAFE_ship({'scratch': str(repo / 'scratch')},
                             OVERRIDE=True, show_progress=False) == {}
    assert not (repo / 'scratch').exists()


def test_an_unknown_topic_is_refused(block, tmp_path):
    with pytest.raises(ValueError, match='not topics of this block'):
        block.UNSAFE_ship({'nope': str(tmp_path / 'x')}, OVERRIDE=True,
                          show_progress=False)


def test_naming_no_topic_is_refused(block):
    with pytest.raises(ValueError, match='names no topic'):
        block.UNSAFE_ship(OVERRIDE=True, show_progress=False)


@pytest.mark.pinned
def test_an_existing_target_is_not_silently_replaced(block, tmp_path):
    """A target is somewhere else's data — a repo file, a shared directory. Replacing
    one is a thing to ask for, not a thing to discover afterwards."""
    target = str(tmp_path / 'repo' / 'facts.json')
    block.UNSAFE_ship({'facts': target}, OVERRIDE=True, show_progress=False)
    with pytest.raises(FileExistsError):
        block.UNSAFE_ship({'facts': target}, OVERRIDE=True, show_progress=False)
    block.UNSAFE_ship({'facts': target}, overwrite=True, OVERRIDE=True,
                      show_progress=False)


@pytest.mark.pinned
def test_an_invalid_block_is_not_shipped(tmp_path):
    """Shipping sends data out. A block that has not built has none to send."""
    unbuilt = Cfg(url=str(tmp_path / 'store2'))
    with pytest.raises(AssertionError):
        unbuilt.UNSAFE_ship({'facts': str(tmp_path / 'repo' / 'f.json')},
                            OVERRIDE=True, show_progress=False)


@pytest.mark.pinned
def test_ship_redirect_makes_the_shipped_copy_the_one_that_is_read(block, tmp_path):
    """The point of shipping-with-redirect: one copy, not two that drift."""
    target = tmp_path / 'repo' / 'facts.json'
    own = block.path('facts')
    block.UNSAFE_ship_redirect({'facts': str(target)}, OVERRIDE=True,
                               show_progress=False)

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
    block.UNSAFE_ship_redirect({'facts': str(tmp_path / 'repo' / 'facts.json')},
                               OVERRIDE=True, show_progress=False)
    assert block.path('configs').startswith(store)


# --- __relocate_shipment__: sending bytes other than the stored ones ---

class Rewriting(Cfg):
    """Ships `facts` with a root rewritten to an env expression, as a portable config."""

    def __relocate_shipment__(self, topic, target, workdir, root_env=None):
        if topic != 'facts' or root_env is None:
            return super().__relocate_shipment__(topic, target, workdir)
        out = os.path.join(workdir(), 'facts.json')
        with open(out, 'w') as f:
            json.dump({'windows': 321337, 'root': '${oc.env:%s}' % root_env}, f)
        return out


def test_the_default_hook_ships_the_topic_itself(block, tmp_path):
    """And stages nothing: a huge topic is copied where it lies."""
    target = str(tmp_path / 'repo' / 'facts.json')
    block.UNSAFE_ship({'facts': target}, OVERRIDE=True, show_progress=False)
    assert json.loads(pathlib.Path(target).read_text()) == {'windows': 321337}


def test_a_relocated_topic_ships_what_the_hook_wrote(tmp_path):
    source = Rewriting(url=str(tmp_path / 'store')).build()
    target = str(tmp_path / 'repo' / 'facts.json')
    source.UNSAFE_ship({'facts': target}, relocate={'root_env': 'SOUNDCLONE_DATAROOT'},
                       OVERRIDE=True, show_progress=False)

    assert json.loads(pathlib.Path(target).read_text()) == {
        'windows': 321337, 'root': '${oc.env:SOUNDCLONE_DATAROOT}'}
    # The block's own copy is untouched -- relocation changes what is SENT.
    assert source.read('facts') == {'windows': 321337}


def test_the_hook_s_workdir_is_removed_afterwards(tmp_path):
    seen = []

    class Noting(Rewriting):
        def __relocate_shipment__(self, topic, target, workdir, root_env=None):
            out = super().__relocate_shipment__(topic, target, workdir, root_env=root_env)
            if root_env is not None and topic == 'facts':
                seen.append(os.path.dirname(out))
            return out

    source = Noting(url=str(tmp_path / 'store')).build()
    source.UNSAFE_ship({'facts': str(tmp_path / 'repo' / 'facts.json')},
                       relocate={'root_env': 'ROOT'}, OVERRIDE=True, show_progress=False)
    assert seen and not os.path.exists(seen[0])


@pytest.mark.pinned
def test_ship_redirect_refuses_a_relocation(tmp_path):
    """Redirecting onto a relocated shipment would make `read()` answer with bytes the
    block never built -- the opposite of the guarantee ship_redirect exists for."""
    source = Rewriting(url=str(tmp_path / 'store')).build()
    with pytest.raises(ValueError, match="not this block's data"):
        source.UNSAFE_ship_redirect({'facts': str(tmp_path / 'repo' / 'facts.json')},
                                    relocate={'root_env': 'ROOT'}, OVERRIDE=True,
                                    show_progress=False)


@pytest.mark.pinned
@pytest.mark.parametrize("target_exists", [False, True], ids=['fresh', 'existing'])
def test_a_directory_topic_ships_its_contents_whatever_is_at_the_target(block, tmp_path,
                                                                        target_exists):
    """`<target>/data/bc.yaml`, never `<target>/configs/data/bc.yaml`.

    fsspec nests a source DIRECTORY under a destination that already exists, so without
    a trailing slash the shape depended on whether anything had created the target --
    which on object storage an earlier partial transfer does. A dataset shipped to the
    lake came out one level deep, and the config pointing at it was then wrong.
    """
    target = tmp_path / 'repo' / 'configs'
    if target_exists:
        target.mkdir(parents=True)
    block.UNSAFE_ship({'configs': str(target)}, OVERRIDE=True, overwrite=True,
                      show_progress=False)

    assert (target / 'data' / 'bc.yaml').is_file()
    assert not (target / 'configs').exists()


@pytest.mark.pinned
def test_a_directory_topic_ships_its_contents_to_a_remote_target(block, tmp_path):
    """The same, where the destination is not a local filesystem -- `put` rather than
    `get`, which is the path a lake ship takes."""
    import fsspec
    memory = fsspec.filesystem('memory')
    for path in list(memory.store):
        memory.rm(path)
    memory.makedirs('/repo/configs', exist_ok=True)          # the target pre-exists

    block.UNSAFE_ship({'configs': 'memory:///repo/configs'}, OVERRIDE=True,
                      overwrite=True, show_progress=False)

    got = sorted(p.lstrip('/') for p in memory.find('/repo/configs'))
    assert got == ['repo/configs/data/bc.yaml'], got
