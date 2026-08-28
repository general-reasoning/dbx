import hashlib
from dataclasses import dataclass
import pandas as pd
import pytest

from dbx.datablocks import Datablock, Datajournal, journal, DatajournalEntry


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class SampleBlock(Datablock):
    VERSION = 1
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        param: str = 'value'

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('hello')


def test_signature_and_norm_alias(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    assert block.signature() == block.norm()
    assert "(spec={'param': 'value'})" in block.signature()


def test_signature_and_hashes(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    sig = block.signature()
    tp = block.type()

    assert sig in tp
    assert f"version={block.version}" in tp
    assert "topic:output=output.txt" in tp

    expected_hash = hashlib.sha256(tp.encode()).hexdigest()
    expected_code = hashlib.sha256(sig.encode()).hexdigest()

    assert block.hash == expected_hash
    assert block.code == expected_code
    assert block.subhash == expected_code


def test_super_properties_removed(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    assert not hasattr(block, 'superhash')
    assert not hasattr(block, 'supernorm')
    assert not hasattr(block, 'supersignature')
    assert not hasattr(block, 'bid')


def test_journal_writing_and_indexing(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    block.build()

    # Test standard journal call without index
    j_default = block.journal()
    assert isinstance(j_default, Datajournal)
    assert 'code' in j_default.columns
    assert 'signature' in j_default.columns
    assert 'superhash' not in j_default.columns

    # Test journal with index='hash'
    j_indexed_hash = block.journal(index='hash')
    assert j_indexed_hash.index.name == 'hash'
    assert block.hash in j_indexed_hash.index

    # Test standalone journal function with index='code'
    j_indexed_code = journal(SampleBlock, url=str(tmp_path), index='code')
    assert j_indexed_code.index.name == 'code'
    assert block.code in j_indexed_code.index


def test_invalid_index_raises_keyerror(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    block.build()

    with pytest.raises(KeyError):
        block.journal(index='non_existent_column')
