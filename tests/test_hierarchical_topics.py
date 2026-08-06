"""
Hierarchical TOPICS: a dict value may itself be a dict of topics.

    TOPICS = {
        'data': {'frames': DIRTOPIC, 'annotations': SYNTOPIC, 'index': 'index.csv'},
        'model': 'model.pt',
    }

Every topic-addressing method takes one name per level -- ``path('data',
'frames')``, ``read('data', 'annotations')``, ``validtopic('data')``.  A GROUP
is addressable too: it has a directory, its ``path()`` is the dict of its
members' paths, and its validity is the conjunction of theirs.

The hard constraint is backward compatibility: a flat TOPICS must produce a
byte-identical signature, and therefore the same hash and storage paths, as
before hierarchy existed.
"""
import hashlib
import os

import pytest

from dbx.datablocks import DIRTOPIC, SYNTOPIC, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Nested(Datablock):
    TOPICS = {
        'data': {'frames': DIRTOPIC, 'annotations': SYNTOPIC, 'index': 'index.csv'},
        'model': 'model.pt',
    }

    def __build__(self):
        os.makedirs(self.dirpath('data', 'frames'), exist_ok=True)
        with open(os.path.join(self.dirpath('data', 'frames'), 'f0.bin'), 'wb') as f:
            f.write(b'0123')
        with open(self.path('data', 'index', ensure_dirpath=True), 'w') as f:
            f.write('idx')
        with open(self.path('model', ensure_dirpath=True), 'w') as f:
            f.write('m')

    def __read__(self, *topicpath):
        return '/'.join(topicpath)


class Deep(Datablock):
    """Three levels, to check nothing is hard-coded to two."""
    TOPICS = {'a': {'b': {'c': 'c.txt'}}}

    def __build__(self):
        with open(self.path('a', 'b', 'c', ensure_dirpath=True), 'w') as f:
            f.write('deep')


@pytest.fixture
def block(tmp_path):
    return Nested(url=str(tmp_path))


@pytest.fixture
def built(tmp_path):
    b = Nested(url=str(tmp_path))
    b.build()
    return b


class TestEnumeration:

    def test_topics_lists_the_top_level(self, block):
        assert block.topics() == ['data', 'model']

    def test_leaftopics_walks_depth_first(self, block):
        assert block.leaftopics() == [
            ('data', 'frames'), ('data', 'annotations'), ('data', 'index'), ('model',),
        ]

    def test_leaftopics_of_a_flat_block_is_one_tuple_per_name(self, tmp_path):
        class Flat(Datablock):
            TOPICS = {'a': 'a.txt', 'b': DIRTOPIC}
            def __build__(self): pass
        assert Flat(url=str(tmp_path)).leaftopics() == [('a',), ('b',)]

    def test_group_detection(self, block):
        assert block.is_topicgroup('data')
        assert not block.is_topicgroup('data', 'frames')
        assert not block.is_topicgroup('model')


class TestAddressing:

    def test_leaf_paths(self, block):
        assert block.path('data', 'frames') == block.dirpath('data', 'frames')
        assert block.path('data', 'index').endswith('data/index/index.csv')
        assert block.path('model').endswith('model/model.pt')

    def test_a_syntopic_leaf_has_no_location(self, block):
        assert block.path('data', 'annotations') is None
        assert block.dirpath('data', 'annotations') is None

    def test_group_path_is_the_dict_of_its_members(self, block):
        p = block.path('data')
        assert set(p) == {'frames', 'annotations', 'index'}
        assert p['annotations'] is None
        assert p['index'] == block.path('data', 'index')

    def test_group_dirpath_is_the_parent_of_its_members(self, block):
        assert block.dirpath('data', 'frames').startswith(block.dirpath('data') + '/')

    def test_nesting_is_mirrored_on_disk(self, block):
        assert block.dirpath('data', 'frames').endswith('/data/frames')

    def test_three_levels(self, tmp_path):
        b = Deep(url=str(tmp_path))
        assert b.path('a', 'b', 'c').endswith('a/b/c/c.txt')
        b.build()
        assert b.valid()

    def test_a_leaftopics_tuple_can_be_fed_straight_back(self, block):
        for tp in block.leaftopics():
            assert block.path(tp) == block.path(*tp)

    def test_paths_mirrors_the_declared_shape(self, block):
        paths = block.paths()
        assert set(paths) == {'data', 'model'}
        assert isinstance(paths['data'], dict)
        assert paths['data']['index'] == block.path('data', 'index')


class TestBadPaths:

    def test_unknown_top_level_name(self, block):
        with pytest.raises(KeyError, match='nope'):
            block.path('nope')

    def test_unknown_nested_name_names_its_level(self, block):
        with pytest.raises(KeyError, match='frames2'):
            block.path('data', 'frames2')

    def test_descending_into_a_leaf(self, block):
        with pytest.raises(KeyError, match='leaf'):
            block.path('model', 'deeper')

    def test_no_topic_at_all_is_a_TypeError(self, block):
        """It was a missing-positional-argument TypeError before varargs."""
        with pytest.raises(TypeError):
            block.path()

    def test_a_slash_in_a_name_is_rejected(self, tmp_path):
        """It would make the signature ambiguous against a real nesting."""
        class Slashed(Datablock):
            TOPICS = {'data/frames': DIRTOPIC}
            def __build__(self): pass
        with pytest.raises(ValueError, match="may not contain"):
            Slashed(url=str(tmp_path)).signature


class TestValidity:

    def test_leaf_validity(self, built):
        assert built.validtopic('data', 'frames')
        assert built.validtopic('data', 'index')

    def test_a_syntopic_leaf_is_vacuously_valid(self, block):
        assert block.validtopic('data', 'annotations')

    def test_group_validity_is_the_conjunction_of_its_leaves(self, built):
        assert built.validtopic('data')

    def test_group_is_invalid_when_a_member_is_missing(self, block):
        os.makedirs(block.dirpath('data', 'frames'), exist_ok=True)
        assert block.validtopic('data', 'frames')     # dir exists
        assert not block.validtopic('data', 'index')  # file does not
        assert not block.validtopic('data')           # so the group does not

    def test_validtopics_reports_per_top_level_topic(self, built):
        assert built.validtopics() == {'data': True, 'model': True}

    def test_valid_covers_the_whole_tree(self, block, built):
        assert built.valid()
        assert not block.__class__(url=block.url + '/elsewhere').valid()


class TestReadAndListing:

    def test_read_takes_a_topic_path(self, built):
        assert built.read('data', 'annotations') == 'data/annotations'

    def test_read_rejects_an_unknown_path(self, built):
        with pytest.raises(KeyError):
            built.read('data', 'nope')

    def test_read_of_a_flat_topic_still_passes_one_argument(self, tmp_path):
        """Existing single-argument __read__ overrides must keep working."""
        seen = []

        class Flat(Datablock):
            TOPICS = {'out': 'out.txt'}
            def __build__(self): pass
            def __read__(self, topic):
                seen.append(topic)
                return topic

        assert Flat(url=str(tmp_path)).read('out') == 'out'
        assert seen == ['out']

    def test_ls_of_a_leaf(self, built):
        assert [os.path.basename(x) for x in built.ls('data', 'frames')] == ['f0.bin']

    def test_ls_of_a_group_concatenates_its_leaves(self, built):
        names = {os.path.basename(x) for x in built.ls('data')}
        assert 'f0.bin' in names and 'index.csv' in names

    def test_size_of_a_group_sums_its_leaves(self, built):
        assert built.size('data') == built.size('data', 'frames') + built.size('data', 'index')
        assert built.size('data') == 4 + 3


class TestClearAndBreadcrumbs:

    def test_clearing_a_group_clears_its_leaves(self, built):
        assert built.valid()
        built.UNSAFE_clear('data', OVERRIDE=True)
        assert not built.validtopic('data', 'index')
        assert built.validtopic('model')          # untouched

    def test_clearing_everything(self, built):
        built.UNSAFE_clear(OVERRIDE=True)
        assert not built.valid()

    def test_breadcrumbs_cover_every_leaf(self, block):
        block.leave_breadcrumbs()
        assert block.valid()

    def test_breadcrumbs_skip_syntopics(self, block):
        block.leave_breadcrumbs()
        assert not os.path.exists(os.path.join(block.anchorkeypath, 'data', 'annotations'))


class TestSignature:

    def test_a_nested_leaf_is_named_by_its_full_path(self, block):
        assert 'topic:data/frames=None' in block.signature
        assert 'topic:data/index=index.csv' in block.signature
        assert 'topic:model=model.pt' in block.signature

    def test_nesting_changes_identity(self, tmp_path):
        """{'a': {'b': X}} and {'a': X} are different declarations."""
        class Grouped(Datablock):
            TOPICS = {'a': {'b': 'x.txt'}}
            def __build__(self): pass

        class Flat(Datablock):
            TOPICS = {'a': 'x.txt'}
            def __build__(self): pass

        a = Grouped(url=str(tmp_path), anchor='s')
        b = Flat(url=str(tmp_path), anchor='s')
        assert a.signature != b.signature
        assert a.hash != b.hash

    def test_hash_is_the_sha256_of_the_signature(self, block):
        assert block.hash == hashlib.sha256(block.signature.encode()).hexdigest()


class TestJournal:

    def test_the_declared_shape_is_recorded(self, built):
        topics = built.journal(iloc=-1).topics
        assert topics == {
            'data': {'frames': None, 'annotations': (), 'index': 'index.csv'},
            'model': 'model.pt',
        }

    def test_recorded_paths_are_nested(self, built):
        paths = built.journal(iloc=-1).paths
        assert paths['data']['index'] == built.path('data', 'index')

    def test_entry_addresses_topics_by_path(self, built):
        entry = built.journal(iloc=-1)
        assert entry._is_dir_topic('data', 'frames')
        assert entry._is_syntopic('data', 'annotations')
        assert entry.is_topicgroup('data')
        assert not entry.is_topicgroup('model')

    def test_entry_listing_matches_the_block(self, built):
        entry = built.journal(iloc=-1)
        assert entry.size('data') == built.size('data')
        assert {os.path.basename(x) for x in entry.ls('data', 'frames')} == {'f0.bin'}
