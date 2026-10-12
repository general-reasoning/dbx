"""DataJournal._journal_files_: the entry files the journal globs find, listed a level at a time."""
import re

import fsspec
import pandas as pd
import pytest

from dbx.journals import DataJournal, Datalog

GLOBS = ('{a}/.dbx/*/journal/**/*.parquet', '{a}/**/.journal/*/journal/**/*.parquet')


def _parquet_(fs, path):
    fs.makedirs(path.rsplit('/', 1)[0], exist_ok=True)
    with fs.open(path, 'wb') as f:
        pd.DataFrame([{'event': 'build:end'}]).to_parquet(f)


def _lake_(fs, anchor):
    """An anchor with entries at two depths, a legacy .dbx entry, side parquet files, and block data."""
    entries = [
        f'{anchor}/k1/.journal/pkg.Block/journal/h1/e1.parquet',
        f'{anchor}/k1/.journal/pkg.Block/journal/h1/e2.parquet',
        f'{anchor}/tag/k2/.journal/pkg.Block/journal/h2/e3.parquet',
        f'{anchor}/.dbx/pkg.Block/journal/h3/e4.parquet',
    ]
    others = [
        # Siblings of journal/: parquet files, but not entries.
        f'{anchor}/k1/.journal/pkg.Block/spec/h1/s.parquet',
        f'{anchor}/tag/k2/.journal/pkg.Block/kwargs/h2/k.parquet',
        # The block's own data.
        f'{anchor}/k1/data/part-0.parquet',
        f'{anchor}/k1/journal/not-under-dot-journal.parquet',
    ]
    for p in entries + others:
        _parquet_(fs, p)
    return sorted(entries)


@pytest.mark.parametrize('protocol', ['file', 'memory'])
def test_journal_files_are_what_the_globs_find_less_the_sibling_dirs(tmp_path, protocol):
    fs = fsspec.filesystem(protocol)
    anchor = str(tmp_path / 'Block') if protocol == 'file' else f'/lake-{tmp_path.name}/Block'
    expected = [fs._strip_protocol(p) for p in _lake_(fs, anchor)]

    found = DataJournal._journal_files_(fs, anchor, n_workers=4, log=Datalog())
    assert found == expected

    globbed = {f for g in GLOBS for f in fs.glob(g.format(a=anchor))}
    # The globs find every entry file the listing does; what they find beyond
    # it is in journal/'s siblings, which the listing does not enter.
    assert set(found) <= globbed
    assert all(re.search(r'/\.journal/[^/]+/(spec|kwargs)/', f) for f in globbed - set(found))


def test_journal_files_of_a_missing_anchor_is_empty(tmp_path):
    fs = fsspec.filesystem('file')
    assert DataJournal._journal_files_(fs, str(tmp_path / 'nowhere'), n_workers=4, log=Datalog()) == []
