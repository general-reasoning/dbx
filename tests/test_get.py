"""
Tests for Datablock.pull()/push() and pulltopic()/pulltopics()/pushtopic()/pushtopics():

1. pull()/push() are generic src/dest copies, no-op when src == dest.
2. pulltopic() with no path pulls to local staging (path(topic, local=True)).
3. pulltopic(path=...) pulls to os.path.join(root, path) instead.
4. pulltopic() handles both file topics and directory topics.
5. pulltopics() fans out over all topics, disambiguating explicit paths per-topic.
6. pushtopic()/pushtopics() are the upload-side counterparts.
7. pulltopic()/pushtopic() can be overridden by subclasses.
"""
import os
import pytest

from dbx.datablocks import Datablock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    monkeypatch.delenv('DBX_LOCAL', raising=False)


@pytest.fixture
def mem_url():
    """Return a unique memory:// URL for each test."""
    uid = os.urandom(4).hex()
    return f"memory://dbx_test_get_{uid}"


@pytest.fixture(autouse=True)
def _clear_memory_fs():
    import fsspec
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    yield
    fs.store.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SingleFileBlock(Datablock):
    """Block with a single TOPICS."""
    TOPICS = {'output': 'output.txt'}

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with open(path, 'w') as f:
            f.write('hello from single file')


class MultiTopicBlock(Datablock):
    """Block with TOPICS (dict of topic -> filename)."""
    TOPICS = {
        'alpha': 'alpha.txt',
        'beta': 'beta.txt',
    }

    def __build__(self):
        for topic in self.TOPICS:
            self.dirpath(topic, ensure=True)
            with open(self.path(topic), 'w') as f:
                f.write(f'data for {topic}')


class TopicsDirBlock(Datablock):
    """Block with TOPICS (each topic is a directory)."""
    TOPICS = ['part_a', 'part_b']

    def __build__(self):
        for topic in self.TOPICS:
            dirpath = self.dirpath(topic, ensure=True)
            with open(os.path.join(dirpath, 'data.txt'), 'w') as f:
                f.write(f'contents of {topic}')


class MemSingleFileBlock(Datablock):
    """Like SingleFileBlock, but writes via self.fs (works on any fsspec backend)."""
    TOPICS = {'output': 'output.txt'}

    def __build__(self):
        path = self.path('output', ensure_dirpath=True)
        with self.fs.open(path, 'w') as f:
            f.write('hello from single file')


# ---------------------------------------------------------------------------
# pull() / push() generic primitives
# ---------------------------------------------------------------------------

class TestPullPush:

    def test_pull_file(self, tmp_path):
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        block.build()
        src = block.path('output')
        dest = str(tmp_path / 'download' / 'output.txt')
        result = block.pull(src, dest)
        assert result is block
        assert os.path.isfile(dest)
        with open(dest) as f:
            assert f.read() == 'hello from single file'

    def test_pull_directory(self, tmp_path):
        """fsspec's fs.get(dir, dest, recursive=True) nests dir's basename
        under dest (no trailing slash on src), so contents land at
        dest/<basename(src)>/... rather than directly under dest."""
        block = TopicsDirBlock(url=str(tmp_path / 'store'))
        block.build()
        src = block.dirpath('part_a')
        dest = str(tmp_path / 'download')
        block.pull(src, dest)
        assert os.path.isfile(os.path.join(dest, 'part_a', 'data.txt'))

    def test_pull_noop_when_src_equals_dest(self, tmp_path):
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        block.build()
        src = block.path('output')
        # Same path: must not error and must not need a destination fs op.
        block.pull(src, src)
        assert os.path.isfile(src)

    def test_push_file(self, tmp_path):
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        src = str(tmp_path / 'local' / 'output.txt')
        os.makedirs(os.path.dirname(src))
        with open(src, 'w') as f:
            f.write('uploaded content')
        dest = block.path('output')
        result = block.push(src, dest)
        assert result is block
        assert os.path.isfile(dest)
        with open(dest) as f:
            assert f.read() == 'uploaded content'

    def test_push_noop_when_src_equals_dest(self, tmp_path):
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        block.build()
        dest = block.path('output')
        block.push(dest, dest)
        assert os.path.isfile(dest)

    def test_pull_missing_source_warns_and_noops(self, tmp_path):
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        result = block.pull(os.path.join(str(tmp_path), 'store', 'nope'), str(tmp_path / 'dest'))
        assert result is block
        assert not os.path.exists(str(tmp_path / 'dest'))


# ---------------------------------------------------------------------------
# pulltopic(): default (local staging) destination
# ---------------------------------------------------------------------------

class TestPullTopicDefaultLocal:

    def test_local_url_is_a_noop(self, tmp_path):
        """When url is itself local, local staging aliases the canonical path."""
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        block.build()
        result = block.pulltopic('output')
        assert result is block
        assert os.path.isfile(block.path('output'))

    def test_nonlocal_url_stages_to_dbx_local(self, tmp_path, monkeypatch, mem_url):
        staging = tmp_path / 'staging'
        monkeypatch.setenv('DBX_LOCAL', str(staging))
        block = MemSingleFileBlock(url=mem_url)
        block.build()
        block.pulltopic('output')
        local_path = block.path('output', local=True)
        assert local_path.startswith(str(staging))
        assert os.path.isfile(local_path)
        with open(local_path) as f:
            assert f.read() == 'hello from single file'


# ---------------------------------------------------------------------------
# pulltopic(path=...): explicit destination, joined with root
# ---------------------------------------------------------------------------

class TestPullTopicExplicitPath:

    def test_downloads_to_root_join_path(self, tmp_path):
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        block.build()
        dest = str(tmp_path / 'download' / 'output.txt')
        block.pulltopic('output', path=dest)
        assert os.path.isfile(dest)
        with open(dest) as f:
            assert f.read() == 'hello from single file'

    def test_root_and_path_are_joined(self, tmp_path):
        """path is joined with root, not overridden by it."""
        block = SingleFileBlock(url=str(tmp_path / 'store'))
        block.build()
        local_root = str(tmp_path / 'local')
        block.pulltopic('output', path='nested/output.txt', root=local_root)
        expected = os.path.join(local_root, 'nested', 'output.txt')
        assert os.path.isfile(expected)

    def test_downloads_specific_topic(self, tmp_path):
        block = MultiTopicBlock(url=str(tmp_path / 'store'))
        block.build()
        dest = str(tmp_path / 'download' / 'alpha.txt')
        block.pulltopic('alpha', path=dest)
        assert os.path.isfile(dest)
        with open(dest) as f:
            assert f.read() == 'data for alpha'

    def test_downloads_topic_directory(self, tmp_path):
        block = TopicsDirBlock(url=str(tmp_path / 'store'))
        block.build()
        dest = str(tmp_path / 'download')
        block.pulltopic('part_a', path=dest)
        # fsspec nests the source dir's basename under dest (see test_pull_directory)
        assert os.path.isfile(os.path.join(dest, 'part_a', 'data.txt'))


# ---------------------------------------------------------------------------
# pulltopics(): fans out over every topic
# ---------------------------------------------------------------------------

class TestPullTopics:

    def test_default_pulls_every_topic_to_local_staging(self, tmp_path):
        block = MultiTopicBlock(url=str(tmp_path / 'store'))
        block.build()
        result = block.pulltopics()
        assert result is block
        assert os.path.isfile(block.path('alpha'))
        assert os.path.isfile(block.path('beta'))

    def test_explicit_path_disambiguates_per_topic(self, tmp_path):
        """Every topic would otherwise collide on the same explicit path;
        pulltopics() nests each topic under its own name. For file topics
        that name becomes the exact destination filename (the topic name
        itself, not the remote basename)."""
        block = MultiTopicBlock(url=str(tmp_path / 'store'))
        block.build()
        dest = str(tmp_path / 'download')
        block.pulltopics(path=dest)
        assert os.path.isfile(os.path.join(dest, 'alpha'))
        assert os.path.isfile(os.path.join(dest, 'beta'))
        with open(os.path.join(dest, 'alpha')) as f:
            assert f.read() == 'data for alpha'


# ---------------------------------------------------------------------------
# pushtopic() / pushtopics(): upload-side counterpart
# ---------------------------------------------------------------------------

class TestPushTopic:

    def test_pushes_from_explicit_path(self, tmp_path, monkeypatch, mem_url):
        monkeypatch.setenv('DBX_LOCAL', str(tmp_path / 'staging'))
        block = SingleFileBlock(url=mem_url)
        src = str(tmp_path / 'local' / 'output.txt')
        os.makedirs(os.path.dirname(src))
        with open(src, 'w') as f:
            f.write('local content')
        result = block.pushtopic('output', path=src, root='.')
        assert result is block
        with block.fs.open(block.path('output')) as f:
            assert f.read().decode() == 'local content'

    def test_pushes_from_local_staging_by_default(self, tmp_path, monkeypatch, mem_url):
        staging = tmp_path / 'staging'
        monkeypatch.setenv('DBX_LOCAL', str(staging))
        block = SingleFileBlock(url=mem_url)
        local_path = block.path('output', local=True)
        os.makedirs(os.path.dirname(local_path))
        with open(local_path, 'w') as f:
            f.write('staged content')
        block.pushtopic('output')
        with block.fs.open(block.path('output')) as f:
            assert f.read().decode() == 'staged content'


class TestPushTopics:

    def test_explicit_path_disambiguates_per_topic(self, tmp_path):
        """Mirror of the pulltopics() disambiguation: for file topics, the
        source at root/path/<topic> is the exact file to upload."""
        block = MultiTopicBlock(url=str(tmp_path / 'store2'))
        src_root = str(tmp_path / 'upload')
        os.makedirs(src_root)
        for topic in ('alpha', 'beta'):
            with open(os.path.join(src_root, topic), 'w') as f:
                f.write(f'uploaded {topic}')
        result = block.pushtopics(path=src_root)
        assert result is block
        with open(block.path('alpha')) as f:
            assert f.read() == 'uploaded alpha'
        with open(block.path('beta')) as f:
            assert f.read() == 'uploaded beta'


# ---------------------------------------------------------------------------
# Overriding pulltopic() in subclasses
# ---------------------------------------------------------------------------

class TestPullTopicOverride:

    def test_custom_pulltopic(self, tmp_path):
        """Subclasses can override pulltopic() for custom download logic."""
        root = str(tmp_path / 'store')
        captured = {}

        class CustomPullBlock(SingleFileBlock):
            def pulltopic(self, topic, *, path=None, root='.'):
                captured['topic'] = topic
                captured['path'] = path
                return self

        block = CustomPullBlock(url=root)
        block.build()
        dest = str(tmp_path / 'download')
        block.pulltopic('mytopic', path=dest)
        assert captured['topic'] == 'mytopic'
        assert captured['path'] == dest
