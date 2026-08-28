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

    def test_type_is_what_hash_hashes(self, block):
        assert block.hash == hashlib.sha256(block.type().encode()).hexdigest()

    def test_signature_is_what_code_hashes(self, block):
        assert block.code == hashlib.sha256(block.signature().encode()).hexdigest()

    def test_type_is_built_from_signature(self, block):
        assert block.signature() in block.type()
        assert f"version={block.version}" in block.type()

    def test_deslash_parameter(self, block):
        assert '\\' not in block.type(deslash=True)
        assert '\\' not in block.signature(deslash=True)

    def test_the_old_names_are_gone(self, block):
        assert not hasattr(block, 'hashstr')
        assert not hasattr(block, 'superhashstr')
        assert not hasattr(block, 'superhash')
        assert not hasattr(block, 'supersignature')
        assert not hasattr(block, 'bid')
        assert not hasattr(Datablock, 'Bid')


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------

class TestSignatureInJournal:

    def test_build_writes_type_txt(self, built):
        entry = built.journal(iloc=-1)
        assert entry.type is not None, "journal has no type column"
        assert '-type-' in entry.type
        assert entry.type.endswith('.txt')
        assert entry.read('type') == built.type()

    def test_build_writes_signature_txt(self, built):
        entry = built.journal(iloc=-1)
        assert entry.signature is not None
        assert entry.read('signature') == built.signature()


# ---------------------------------------------------------------------------
# Journals written before the rename
# ---------------------------------------------------------------------------

class TestPreRenameJournals:

    def _entry(self, **columns):
        return DatajournalEntry(pd.Series({'hash': 'abc', 'anchor': 'a.B', **columns}))

    def test_legacy_hashstr_column_is_read_as_type(self):
        entry = self._entry(hashstr='/j/x-hashstr-1.txt',
                            norm='/j/x-norm-1.txt')
        assert entry.type == '/j/x-hashstr-1.txt'
        assert entry.signature == '/j/x-norm-1.txt'

    def test_new_column_wins_when_both_are_present(self):
        entry = self._entry(type='/j/new.txt', signature='/j/old.txt')
        assert entry.type == '/j/new.txt'

    def test_nan_in_the_new_column_still_falls_back(self):
        entry = self._entry(type=float('nan'), signature='/j/old.txt')
        assert entry.type == '/j/old.txt'

    def test_absent_in_both_degrades_to_none(self):
        entry = self._entry()
        assert entry.type is None
        assert entry.signature is None
        assert entry.read('type') is None
        assert entry.read('signature') is None

    def test_nan_in_both_degrades_to_none(self):
        entry = self._entry(type=float('nan'), signature=float('nan'))
        assert entry.type is None

    def test_a_real_mixed_era_journal_reads_both_rows(self, built):
        """End to end: an old row and a new row concatenated into one frame."""
        new_row = built.journal(iloc=-1)
        old_row = pd.Series({**dict(new_row), 'type': None,
                             'signature': None,
                             'subsignature': None,
                             'hashstr': '/j/legacy-hashstr.txt',
                             'norm': '/j/legacy-norm.txt'})
        frame = pd.DataFrame([old_row, pd.Series(dict(new_row))])

        assert DatajournalEntry(frame.iloc[0]).type == '/j/legacy-hashstr.txt'
        assert DatajournalEntry(frame.iloc[0]).signature == '/j/legacy-norm.txt'
        assert DatajournalEntry(frame.iloc[1]).type == new_row.type
        assert DatajournalEntry(frame.iloc[1]).signature == new_row.signature
