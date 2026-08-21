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


def test_subsignature_and_norm_alias(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    assert block.subsignature() == block.norm()
    assert "(spec={'param': 'value'})" in block.subsignature()


def test_signature_and_hashes(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    subsig = block.subsignature()
    sig = block.signature()

    assert subsig in sig
    assert f"version={block.version}" in sig
    assert "topic:output=output.txt" in sig

    expected_hash = hashlib.sha256(sig.encode()).hexdigest()
    expected_subhash = hashlib.sha256(subsig.encode()).hexdigest()

    assert block.hash == expected_hash
    assert block.subhash == expected_subhash


def test_super_properties_removed(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    assert not hasattr(block, 'superhash')
    assert not hasattr(block, 'supernorm')
    assert not hasattr(block, 'supersignature')


def test_bid_fields(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    bid = block.bid
    fields = bid.fields()

    assert 'subsignature' in fields
    assert 'subhash' in fields
    assert 'hash' in fields
    assert 'superhash' not in fields
    assert 'supernorm' not in fields
    assert 'supersignature' not in fields

    assert bid.subsignature == block.subsignature(deslash=True)
    assert bid.subhash == block.subhash
    assert bid.hash == block.hash


def test_journal_writing_and_indexing(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    block.build()

    # Test standard journal call without index
    j_default = block.journal()
    assert isinstance(j_default, Datajournal)
    assert 'subhash' in j_default.columns
    assert 'subsignature' in j_default.columns
    assert 'superhash' not in j_default.columns

    # Test journal with index='hash'
    j_indexed_hash = block.journal(index='hash')
    assert j_indexed_hash.index.name == 'hash'
    assert block.hash in j_indexed_hash.index

    # Test standalone journal function with index='subhash'
    j_indexed_subhash = journal(SampleBlock, url=str(tmp_path), index='subhash')
    assert j_indexed_subhash.index.name == 'subhash'
    assert block.subhash in j_indexed_subhash.index


def test_invalid_index_raises_keyerror(tmp_path):
    block = SampleBlock(url=str(tmp_path))
    block.build()

    with pytest.raises(KeyError):
        block.journal(index='non_existent_column')
