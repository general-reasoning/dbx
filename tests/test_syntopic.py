"""
``SYNTOPIC`` — a synthetic topic, one the block never stores.

``DIRTOPIC`` and ``SYNTOPIC`` are both "no filename", but they answer different
questions: a ``DIRTOPIC`` topic IS a location (a real directory that happens to
hold no single named file), whereas a synthetic one has no location at all.
Nothing on the filesystem is named, created, listed, copied or cleared for a
``SYNTOPIC``, and — since a topic that was never going to be written cannot be
missing — it cannot hold a block back from being valid.

The two must never collapse into each other, which is why ``SYNTOPIC`` is ``()``
and not another ``None``-alike — these tests pin that separation.
"""
import os

import pytest

import dbx
from dbx.datablocks import DIRTOPIC, SYNTOPIC, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Mixed(Datablock):
    """One of each kind of topic."""
    TOPICS = {'data': 'data.txt', 'masks': DIRTOPIC, 'cache': SYNTOPIC}

    def __build__(self):
        with open(self.path('data', ensure_dirpath=True), 'w') as f:
            f.write('data')
        os.makedirs(self.dirpath('masks'), exist_ok=True)
        with open(os.path.join(self.dirpath('masks'), 'm.bin'), 'wb') as f:
            f.write(b'\x00')


@pytest.fixture
def block(tmp_path):
    return Mixed(url=str(tmp_path))


class TestSYNTOPICIsItsOwnMarker:

    def test_syntopic_is_the_empty_tuple(self):
        assert SYNTOPIC == ()

    def test_exported_from_the_package(self):
        assert dbx.SYNTOPIC is SYNTOPIC

    def test_syntopic_is_not_dirtopic(self):
        """The whole point: 'no location' and 'is a directory' must not merge."""
        assert SYNTOPIC is not DIRTOPIC
        assert SYNTOPIC != DIRTOPIC

    def test_syntopic_is_falsy_like_dirtopic(self):
        """Existing `if not topicfile` style checks keep behaving."""
        assert not SYNTOPIC and not DIRTOPIC


class TestSYNTOPICHasNoLocation:

    def test_path_is_none(self, block):
        assert block.path('cache') is None

    def test_dirpath_is_none(self, block):
        assert block.dirpath('cache') is None

    def test_the_other_topics_are_unaffected(self, block):
        assert block.path('data').endswith('data.txt')
        assert block.path('masks') == block.dirpath('masks')

    def test_it_is_neither_a_dir_topic_nor_a_file_topic(self, block):
        assert block._is_syntopic('cache')
        assert not block._is_dir_topic('cache')
        assert block._is_dir_topic('masks')
        assert not block._is_syntopic('masks')
        assert not block._is_syntopic('data')

    def test_paths_records_it_as_none(self, block):
        assert block.paths()['cache'] is None

    def test_it_is_still_a_topic(self, block):
        assert 'cache' in block.topics()


class TestSYNTOPICCreatesNothing:

    def test_ensure_dirpath_creates_nothing(self, block):
        assert block.path('cache', ensure_dirpath=True) is None
        assert not os.path.exists(os.path.join(block.anchorkeypath, 'cache'))

    def test_dirpath_ensure_creates_nothing(self, block):
        assert block.dirpath('cache', ensure=True) is None
        assert not os.path.exists(os.path.join(block.anchorkeypath, 'cache'))

    def test_leave_breadcrumbs_skips_it(self, tmp_path):
        # No DIRTOPIC topic here: leave_breadcrumbs() opens path(topic) as a file,
        # which already fails on a DIRTOPIC topic independently of SYNTOPIC.
        class FileAndSyn(Datablock):
            TOPICS = {'data': 'data.txt', 'cache': SYNTOPIC}
            def __build__(self): pass

        b = FileAndSyn(url=str(tmp_path))
        b.leave_breadcrumbs()
        assert not os.path.exists(os.path.join(b.anchorkeypath, 'cache'))
        assert os.path.exists(b.path('data'))


class TestSYNTOPICIsVacuouslyValid:

    def test_a_syntopic_does_not_block_validity(self, block):
        block.build()
        assert block.validtopic('cache')
        assert block.valid()

    def test_a_block_of_only_syntopics_is_valid(self, tmp_path):
        class AllSyn(Datablock):
            TOPICS = {'a': SYNTOPIC, 'b': SYNTOPIC}

            def __build__(self):
                pass

        b = AllSyn(url=str(tmp_path))
        b.build()
        assert b.valid()

    def test_a_missing_real_topic_still_invalidates(self, block):
        """Guard: SYNTOPIC must not make validity vacuous for the OTHER topics."""
        assert not block.valid()


class TestSYNTOPICIsInertOnTheFilesystem:

    def test_ls_and_list_are_empty(self, block):
        block.build()
        assert block.ls('cache') == []
        assert block.list('cache') == []

    def test_size_is_zero(self, block):
        block.build()
        assert block.size('cache') == 0

    def test_clear_leaves_the_other_topics_alone(self, block):
        block.build()
        block.UNSAFE_clear(OVERRIDE=True)
        assert not os.path.exists(block.path('data'))

    def test_clear_dirpath_does_not_raise(self, block):
        block.build()
        block.UNSAFE_clear(OVERRIDE=True, clear_dirpath=True)


class TestSYNTOPICInTheJournal:

    def test_recorded_and_read_back(self, block):
        block.build()
        entry = block.journal(iloc=-1)
        assert entry.block.TOPICS['cache'] == SYNTOPIC
        assert entry.block.paths()['cache'] is None
        assert entry.block._is_syntopic('cache')
        assert not entry.block._is_dir_topic('cache')

    def test_entry_listing_is_empty(self, block):
        block.build()
        entry = block.journal(iloc=-1)
        assert entry.block.ls('cache') == []
        assert entry.block.size('cache') == 0

    def test_dirtopic_and_syntopic_stay_distinct_through_the_journal(self, block):
        """A round trip through str(dict) must not turn () into None."""
        block.build()
        topics = block.journal(iloc=-1).block.TOPICS
        assert topics['masks'] is DIRTOPIC
        assert topics['cache'] == SYNTOPIC
        assert topics['masks'] is not topics['cache']


class TestSYNTOPICInTheSignature:

    def test_a_syntopic_is_part_of_identity(self, tmp_path):
        """Declaring a topic SYNTOPIC is still a declaration; it must be recorded."""
        class WithCache(Datablock):
            TOPICS = {'data': 'data.txt', 'cache': SYNTOPIC}
            def __build__(self): pass

        class WithoutCache(Datablock):
            TOPICS = {'data': 'data.txt'}
            def __build__(self): pass

        a = WithCache(url=str(tmp_path), anchor='shared')
        b = WithoutCache(url=str(tmp_path), anchor='shared')
        assert a.type() != b.type()
        assert a.hash != b.hash

    def test_it_renders_as_the_empty_tuple(self, tmp_path):
        """Pinned so the recorded form is a decision, not an accident."""
        assert 'topic:cache=()' in Mixed(url=str(tmp_path)).type()

    def test_syntopic_and_dirtopic_give_different_signatures(self, tmp_path):
        class AsSyn(Datablock):
            TOPICS = {'x': SYNTOPIC}
            def __build__(self): pass

        class AsDirTopic(Datablock):
            TOPICS = {'x': DIRTOPIC}
            def __build__(self): pass

        assert (AsSyn(url=str(tmp_path), anchor='s').type()
                != AsDirTopic(url=str(tmp_path), anchor='s').type())
