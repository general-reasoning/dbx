"""
``hashstr`` -> ``signature`` and ``superhashstr`` -> ``supersignature``.

The property is the string that :attr:`Datablock.hash` is the sha256 of, so
the rename must be pure: same bytes, same hash, same key.  It is also a
recorded field -- ``signature.txt``, a journal column, and a ``Bid`` field --
and journals written before the rename recorded those columns under the old
names, so reading them has to keep working.
"""
import hashlib

import pandas as pd
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock, DatajournalEntry


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Solo(Datablock):
    """Actually writes its topic, so build() gets past the validity check."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = 'solo'

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


@pytest.fixture
def block(tmp_path):
    return Solo(url=str(tmp_path))


@pytest.fixture
def built(tmp_path):
    b = Solo(url=str(tmp_path), spec=dict(label='solo')).set(tag='solo')
    b.build()
    return b


# ---------------------------------------------------------------------------
# The method itself
# ---------------------------------------------------------------------------

class TestSignatureMethod:

    def test_signature_is_what_hash_hashes(self, block):
        assert block.hash == hashlib.sha256(block.signature().encode()).hexdigest()

    def test_subsignature_is_what_subhash_hashes(self, block):
        assert block.subhash == hashlib.sha256(
            block.subsignature().encode()).hexdigest()

    def test_signature_is_built_from_subsignature(self, block):
        assert block.subsignature() in block.signature()
        assert f"version={block.version}" in block.signature()

    def test_deslash_parameter(self, block):
        assert '\\' not in block.signature(deslash=True)
        assert '\\' not in block.subsignature(deslash=True)

    def test_the_old_names_are_gone(self, block):
        assert not hasattr(block, 'hashstr')
        assert not hasattr(block, 'superhashstr')
        assert not hasattr(block, 'superhash')
        assert not hasattr(block, 'supersignature')


# ---------------------------------------------------------------------------
# Bid
# ---------------------------------------------------------------------------

class TestSignatureInBid:

    def test_bid_field_names(self, block):
        fields = block.bid.fields()
        assert 'signature' in fields and 'subsignature' in fields and 'subhash' in fields
        assert 'hashstr' not in fields and 'superhashstr' not in fields and 'superhash' not in fields

    def test_bid_values_match_the_properties(self, block):
        assert block.bid.signature == block.signature(deslash=True)
        assert block.bid.subsignature == block.subsignature(deslash=True)
        assert block.bid.subhash == block.subhash

    def test_bid_to_dict_covers_them(self, block):
        d = block.bid.to_dict()
        assert d['signature'] == block.signature(deslash=True)
        assert d['subsignature'] == block.subsignature(deslash=True)
        assert d['subhash'] == block.subhash


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------

class TestSignatureInJournal:

    def test_build_writes_signature_txt(self, built):
        entry = built.journal(iloc=-1)
        assert entry.signature is not None, "journal has no signature column"
        assert '-signature-' in entry.signature
        assert entry.signature.endswith('.txt')
        assert entry.read('signature') == built.signature()

    def test_build_writes_subsignature_txt(self, built):
        entry = built.journal(iloc=-1)
        assert entry.subsignature is not None
        assert entry.read('subsignature') == built.subsignature()

    def test_journal_entry_bid_carries_them(self, built):
        bid = built.journal(iloc=-1).bid
        assert bid.signature == built.signature(deslash=True)
        assert bid.subsignature == built.subsignature(deslash=True)
        assert bid.subhash == built.subhash


# ---------------------------------------------------------------------------
# Journals written before the rename
# ---------------------------------------------------------------------------

class TestPreRenameJournals:

    def _entry(self, **columns):
        return DatajournalEntry(pd.Series({'hash': 'abc', 'anchor': 'a.B', **columns}))

    def test_legacy_hashstr_column_is_read_as_signature(self):
        entry = self._entry(hashstr='/j/x-hashstr-1.txt',
                            norm='/j/x-norm-1.txt')
        assert entry.signature == '/j/x-hashstr-1.txt'
        assert entry.subsignature == '/j/x-norm-1.txt'

    def test_new_column_wins_when_both_are_present(self):
        entry = self._entry(signature='/j/new.txt', hashstr='/j/old.txt')
        assert entry.signature == '/j/new.txt'

    def test_nan_in_the_new_column_still_falls_back(self):
        entry = self._entry(signature=float('nan'), hashstr='/j/old.txt')
        assert entry.signature == '/j/old.txt'

    def test_absent_in_both_degrades_to_none(self):
        entry = self._entry()
        assert entry.signature is None
        assert entry.subsignature is None
        assert entry.read('signature') is None

    def test_nan_in_both_degrades_to_none(self):
        entry = self._entry(signature=float('nan'), hashstr=float('nan'))
        assert entry.signature is None

    def test_a_real_mixed_era_journal_reads_both_rows(self, built):
        """End to end: an old row and a new row concatenated into one frame."""
        new_row = built.journal(iloc=-1)
        old_row = pd.Series({**dict(new_row), 'signature': None,
                             'subsignature': None,
                             'hashstr': '/j/legacy-hashstr.txt',
                             'norm': '/j/legacy-norm.txt'})
        frame = pd.DataFrame([old_row, pd.Series(dict(new_row))])

        assert DatajournalEntry(frame.iloc[0]).signature == '/j/legacy-hashstr.txt'
        assert DatajournalEntry(frame.iloc[1]).signature == new_row.signature
