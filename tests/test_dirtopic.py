"""
``DIRTOPIC`` — the self-documenting spelling of a directory topic in dict-TOPICS.

``DIRTOPIC`` *is* ``None``, so this is a naming change and nothing else: a block
written with ``DIRTOPIC`` and one written with a bare ``None`` must be
indistinguishable, down to the hash.  These tests pin that equivalence so the
constant cannot quietly drift into a distinct sentinel, which would split every
existing directory topic off its stored artifacts.
"""
import os

import pytest

import dbx
from dbx.datablocks import DIRTOPIC, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class WithDIR(Datablock):
    TOPICS = {'logs': 'train.log', 'checkpoints': DIRTOPIC}

    def __build__(self):
        self.dirpath('checkpoints', ensure=True)
        with open(self.path('logs', ensure_dirpath=True), 'w') as f:
            f.write('log')
        with open(os.path.join(self.dirpath('checkpoints'), 'ckpt.pt'), 'w') as f:
            f.write('ckpt')


class WithNone(Datablock):
    TOPICS = {'logs': 'train.log', 'checkpoints': None}

    def __build__(self):
        WithDIR.__build__(self)


class TestDIRTOPICIsNone:

    def test_dirtopic_is_none(self):
        assert DIRTOPIC is None

    def test_exported_from_the_package(self):
        assert dbx.DIRTOPIC is DIRTOPIC

    def test_topics_dict_is_literally_the_same(self):
        assert WithDIR.TOPICS == WithNone.TOPICS


class TestDIRTOPICBehavesAsADirectoryTopic:

    @pytest.fixture
    def block(self, tmp_path):
        return WithDIR(url=str(tmp_path))

    def test_dir_topic_path_is_the_dirpath(self, block):
        assert block.path('checkpoints') == block.dirpath('checkpoints')

    def test_file_topic_path_is_not(self, block):
        assert block.path('logs') == os.path.join(block.dirpath('logs'), 'train.log')

    def test_is_dir_topic(self, block):
        assert block._is_dir_topic('checkpoints')
        assert not block._is_dir_topic('logs')

    def test_builds_and_validates(self, block):
        block.build()
        assert block.valid()


class TestDIRTOPICDoesNotChangeIdentity:
    """A rename that moved the hash would orphan every stored artifact."""

    def test_signature_is_unaffected_by_the_spelling(self, tmp_path):
        a = WithDIR(url=str(tmp_path), anchor='shared')
        b = WithNone(url=str(tmp_path), anchor='shared')
        assert a.signature() == b.signature()
        assert a.hash == b.hash
        assert a.key == b.key

    def test_dir_topic_renders_as_none_in_the_signature(self, tmp_path):
        """The recorded form is still `topic:name=None`, not `topic:name=DIRTOPIC`."""
        assert 'topic:checkpoints=None' in WithDIR(url=str(tmp_path)).type()


class TestDIRTOPICInTheJournal:

    def test_journal_records_the_dir_topic_as_none(self, tmp_path):
        b = WithDIR(url=str(tmp_path))
        b.build()
        entry = b.journal(iloc=-1)
        assert entry.topics['checkpoints'] is None
        assert entry.topics['logs'] == 'train.log'
        assert entry._is_dir_topic('checkpoints')
        assert not entry._is_dir_topic('logs')
