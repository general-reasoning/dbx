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
# The property itself
# ---------------------------------------------------------------------------

class TestSignatureProperty:

    def test_signature_is_what_hash_hashes(self, block):
        assert block.hash == hashlib.sha256(block.signature.encode()).hexdigest()

    def test_supersignature_is_what_superhash_hashes(self, block):
        assert block.superhash == hashlib.sha256(
            block.supersignature.encode()).hexdigest()[:8]

    def test_signature_is_built_from_norm(self, block):
        assert block.norm() in block.signature
        assert f"version={block.version}" in block.signature

    def test_supersignature_is_built_from_supernorm(self, block):
        assert block.supernorm() in block.supersignature

    def test_the_two_differ(self, block):
        """supersignature anchors on the fqcn; a silent alias would hide that."""
        assert block.signature != block.supersignature

    def test_the_old_names_are_gone(self, block):
        assert not hasattr(block, 'hashstr')
        assert not hasattr(block, 'superhashstr')


# ---------------------------------------------------------------------------
# Bid
# ---------------------------------------------------------------------------

class TestSignatureInBid:

    def test_bid_field_names(self, block):
        fields = block.bid.fields()
        assert 'signature' in fields and 'supersignature' in fields
        assert 'hashstr' not in fields and 'superhashstr' not in fields

    def test_bid_values_match_the_properties(self, block):
        assert block.bid.signature == block.signature
        assert block.bid.supersignature == block.supersignature

    def test_bid_to_dict_covers_them(self, block):
        d = block.bid.to_dict()
        assert d['signature'] == block.signature
        assert d['supersignature'] == block.supersignature


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------

class TestSignatureInJournal:

    def test_build_writes_signature_txt(self, built):
        entry = built.journal(iloc=-1)
        assert entry.signature is not None, "journal has no signature column"
        assert '-signature-' in entry.signature
        assert entry.signature.endswith('.txt')
        assert entry.read('signature') == built.signature

    def test_build_writes_supersignature_txt(self, built):
        entry = built.journal(iloc=-1)
        assert entry.supersignature is not None
        assert entry.read('supersignature') == built.supersignature

    def test_journal_entry_bid_carries_them(self, built):
        bid = built.journal(iloc=-1).bid
        assert bid.signature == built.signature
        assert bid.supersignature == built.supersignature


# ---------------------------------------------------------------------------
# Journals written before the rename
# ---------------------------------------------------------------------------

class TestPreRenameJournals:

    def _entry(self, **columns):
        return DatajournalEntry(pd.Series({'hash': 'abc', 'anchor': 'a.B', **columns}))

    def test_legacy_hashstr_column_is_read_as_signature(self):
        entry = self._entry(hashstr='/j/x-hashstr-1.txt',
                            superhashstr='/j/x-superhashstr-1.txt')
        assert entry.signature == '/j/x-hashstr-1.txt'
        assert entry.supersignature == '/j/x-superhashstr-1.txt'

    def test_new_column_wins_when_both_are_present(self):
        entry = self._entry(signature='/j/new.txt', hashstr='/j/old.txt')
        assert entry.signature == '/j/new.txt'

    def test_nan_in_the_new_column_still_falls_back(self):
        """A journal spanning the rename has BOTH columns, NaN-filled per row.

        Concatenating a pre- and a post-rename entry gives every row both
        columns, so the old row's `signature` is NaN rather than missing --
        a plain .get(name, default) would return the NaN and read() would
        then try to open it as a path.
        """
        entry = self._entry(signature=float('nan'), hashstr='/j/old.txt')
        assert entry.signature == '/j/old.txt'

    def test_absent_in_both_degrades_to_none(self):
        entry = self._entry()
        assert entry.signature is None
        assert entry.supersignature is None
        assert entry.read('signature') is None

    def test_nan_in_both_degrades_to_none(self):
        entry = self._entry(signature=float('nan'), hashstr=float('nan'))
        assert entry.signature is None

    def test_a_real_mixed_era_journal_reads_both_rows(self, built):
        """End to end: an old row and a new row concatenated into one frame."""
        new_row = built.journal(iloc=-1)
        old_row = pd.Series({**dict(new_row), 'signature': None,
                             'supersignature': None,
                             'hashstr': '/j/legacy-hashstr.txt',
                             'superhashstr': '/j/legacy-superhashstr.txt'})
        frame = pd.DataFrame([old_row, pd.Series(dict(new_row))])

        assert DatajournalEntry(frame.iloc[0]).signature == '/j/legacy-hashstr.txt'
        assert DatajournalEntry(frame.iloc[1]).signature == new_row.signature
