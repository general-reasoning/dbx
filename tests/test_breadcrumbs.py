"""
``leave_breadcrumbs_at_path(path, crumbs=None)`` takes a DIRECTORY path.

It used to be handed ``path(topic)``, which for a directory topic IS the
directory -- so opening it for writing raised IsADirectoryError and
``leave_breadcrumbs()`` was unusable on any block with one.  Now the caller
passes ``dirpath(topic)`` and says what to write inside it:

    crumbs='data.txt'  ->  {dirpath}/data.txt   (a file topic's own file)
    crumbs=None        ->  {dirpath}.crumbs     (a directory topic's marker)
"""
import os

import pytest

from dbx.datablocks import DIR, NULL, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class FileTopics(Datablock):
    TOPICS = {'output': 'output.txt', 'meta': 'meta.json'}
    def __build__(self): pass


class DirTopic(Datablock):
    TOPICS = {'masks': DIR}
    def __build__(self): pass


class ListTopics(Datablock):
    TOPICS = ['images', 'masks']
    def __build__(self): pass


class Mixed(Datablock):
    TOPICS = {'output': 'output.txt', 'masks': DIR, 'cache': NULL}
    def __build__(self): pass


class TestDirectoryTopicsNoLongerRaise:
    """The regression this fixes."""

    def test_dir_topic_breadcrumbs_do_not_raise(self, tmp_path):
        DirTopic(url=str(tmp_path)).leave_breadcrumbs()

    def test_list_topics_breadcrumbs_do_not_raise(self, tmp_path):
        ListTopics(url=str(tmp_path)).leave_breadcrumbs()

    def test_mixed_topics_breadcrumbs_do_not_raise(self, tmp_path):
        Mixed(url=str(tmp_path)).leave_breadcrumbs()


class TestWhereTheBreadcrumbLands:

    def test_dir_topic_gets_a_sibling_crumbs_file(self, tmp_path):
        b = DirTopic(url=str(tmp_path))
        b.leave_breadcrumbs()
        crumb = b.dirpath('masks') + '.crumbs'
        assert os.path.isfile(crumb)
        # ...and NOT inside the directory, which stays empty.
        assert os.listdir(b.dirpath('masks')) == []

    def test_file_topic_breadcrumb_is_its_own_file(self, tmp_path):
        b = FileTopics(url=str(tmp_path))
        b.leave_breadcrumbs()
        for topic in b.TOPICS:
            assert os.path.isfile(b.path(topic))
            assert os.path.getsize(b.path(topic)) == 0

    def test_list_topic_gets_a_sibling_crumbs_file(self, tmp_path):
        b = ListTopics(url=str(tmp_path))
        b.leave_breadcrumbs()
        for topic in b.TOPICS:
            assert os.path.isfile(b.dirpath(topic) + '.crumbs')

    def test_null_topic_gets_nothing(self, tmp_path):
        b = Mixed(url=str(tmp_path))
        b.leave_breadcrumbs()
        assert not os.path.exists(os.path.join(b.anchorkeypath, 'cache'))
        assert not os.path.exists(os.path.join(b.anchorkeypath, 'cache.crumbs'))


class TestLeaveBreadcrumbsAtPathDirectly:

    @pytest.fixture
    def dirpath(self, tmp_path):
        d = tmp_path / 'topicdir'
        d.mkdir()
        return str(d)

    def test_crumbs_names_a_file_inside(self, tmp_path, dirpath):
        b = FileTopics(url=str(tmp_path))
        out = b.leave_breadcrumbs_at_path(dirpath, crumbs='inside.txt')
        assert out == f"{dirpath}/inside.txt"
        assert os.path.isfile(out)

    def test_no_crumbs_appends_dot_crumbs(self, tmp_path, dirpath):
        b = FileTopics(url=str(tmp_path))
        out = b.leave_breadcrumbs_at_path(dirpath)
        assert out == f"{dirpath}.crumbs"
        assert os.path.isfile(out)

    def test_the_directory_is_created_for_a_named_crumb(self, tmp_path):
        b = FileTopics(url=str(tmp_path))
        missing = str(tmp_path / 'not-there-yet')
        out = b.leave_breadcrumbs_at_path(missing, crumbs='x.txt')
        assert os.path.isfile(out)

    def test_existing_content_is_not_clobbered(self, tmp_path, dirpath):
        """A breadcrumb marks absence; it must never erase a real artifact."""
        b = FileTopics(url=str(tmp_path))
        target = os.path.join(dirpath, 'real.txt')
        with open(target, 'w') as f:
            f.write('precious')

        b.leave_breadcrumbs_at_path(dirpath, crumbs='real.txt')

        with open(target) as f:
            assert f.read() == 'precious'

    def test_is_idempotent(self, tmp_path, dirpath):
        b = FileTopics(url=str(tmp_path))
        first = b.leave_breadcrumbs_at_path(dirpath)
        second = b.leave_breadcrumbs_at_path(dirpath)
        assert first == second
        assert os.path.isfile(first)


class TestValidityIsPreserved:
    """Breadcrumbs exist to make a block read as valid; that must still hold."""

    def test_file_topics_become_valid(self, tmp_path):
        b = FileTopics(url=str(tmp_path))
        assert b.valid() is False
        b.leave_breadcrumbs()
        assert b.valid() is True

    def test_dir_topic_becomes_valid(self, tmp_path):
        b = DirTopic(url=str(tmp_path))
        b.leave_breadcrumbs()
        assert b.valid() is True

    def test_a_breadcrumbed_block_is_skipped_by_build(self, tmp_path):
        """Which is the point of a breadcrumb: it stands in for the artifact.

        build() short-circuits on valid(), so the crumb suppresses the build
        rather than being overwritten by it.
        """
        class Real(Datablock):
            TOPICS = {'output': 'output.txt'}
            def __build__(self):
                with open(self.path('output', ensure_dirpath=True), 'w') as f:
                    f.write('real')

        b = Real(url=str(tmp_path))
        b.leave_breadcrumbs()
        b.build()
        with open(b.path('output')) as f:
            assert f.read() == ''        # still the crumb, never built

        b.UNSAFE_clear(OVERRIDE=True)
        b.build()
        with open(b.path('output')) as f:
            assert f.read() == 'real'    # cleared, so the build runs
