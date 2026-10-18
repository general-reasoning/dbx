"""BuildJournal._journal_files_: the entry files the journal globs find, listed a level at a time."""
import re

import fsspec
import pandas as pd
import pytest

from dbx.journals import BuildJournal, Datalog

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

    found = BuildJournal._journal_files_(fs, anchor, n_workers=4, log=Datalog())
    assert found == expected

    globbed = {f for g in GLOBS for f in fs.glob(g.format(a=anchor))}
    # The globs find every entry file the listing does; what they find beyond
    # it is in journal/'s siblings, which the listing does not enter.
    assert set(found) <= globbed
    assert all(re.search(r'/\.journal/[^/]+/(spec|kwargs)/', f) for f in globbed - set(found))


def test_journal_files_of_a_missing_anchor_is_empty(tmp_path):
    fs = fsspec.filesystem('file')
    assert BuildJournal._journal_files_(fs, str(tmp_path / 'nowhere'), n_workers=4, log=Datalog()) == []


def test_list_is_the_listing_as_full_paths(tmp_path):
    fs = fsspec.filesystem('memory')
    lake = f'memory://lake-{tmp_path.name}'
    expected = _lake_(fs, f'/lake-{tmp_path.name}/Block')
    files = BuildJournal.list('Block', datalake=lake, n_workers=4, storage_options={})
    assert files == [f'memory://{p}' for p in expected]


def test_list_of_a_missing_anchor_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BuildJournal.list('Block', datalake=str(tmp_path), storage_options={})


def test_list_tells_progress_the_files_found_so_far_with_no_total(tmp_path):
    _lake_(fsspec.filesystem('file'), str(tmp_path / 'Block'))
    told = []
    BuildJournal.list('Block', datalake=str(tmp_path), n_workers=4, storage_options={},
                     progress=lambda desc, n, total: told.append((n, total)))
    assert told and all(total is None for _, total in told)
    assert [n for n, _ in told] == sorted(n for n, _ in told) and told[-1][0] == 4


def test_read_frame_tells_progress_each_file_read_of_the_total(tmp_path):
    fs = fsspec.filesystem('file')
    for i in range(4):
        path = str(tmp_path / f'Block/k{i}/.journal/pkg.Block/journal/h{i}/e.parquet')
        fs.makedirs(path.rsplit('/', 1)[0], exist_ok=True)
        with fs.open(path, 'wb') as f:
            pd.DataFrame([{'event': 'build:end', 'datetime': f'2026-10-0{i + 1}T00-00-00.000000'}]).to_parquet(f)
    files = BuildJournal.list('Block', datalake=str(tmp_path), storage_options={})
    told = []
    BuildJournal.read_frame(files, n_workers=2, storage_options={}, unnormalized=True,
                           progress=lambda desc, done, total: told.append((done, total)))
    assert told == [(i, 4) for i in range(1, 5)]


class _Stop_(Exception):
    pass


def test_progress_raising_interrupts_the_read(tmp_path):
    _lake_(fsspec.filesystem('file'), str(tmp_path / 'Block'))

    def progress(desc, done, total):
        if total is not None:
            raise _Stop_()

    with pytest.raises(_Stop_):
        BuildJournal.read('Block', datalake=str(tmp_path), n_workers=1, storage_options={}, progress=progress)


def test_the_listing_reports_each_directory_not_each_level(tmp_path):
    fs = fsspec.filesystem('file')
    for i in range(5):
        _parquet_(fs, str(tmp_path / f'Block/k{i}/.journal/pkg.Block/journal/h{i}/e.parquet'))
    told = []
    BuildJournal.list('Block', datalake=str(tmp_path), n_workers=2, storage_options={},
                     progress=lambda desc, n, total: told.append(n))
    # The root, its five k*, then each level below them five at a time: far more reports than levels.
    assert len(told) > 20 and told[-1] == 5


def test_progress_raising_interrupts_the_listing(tmp_path):
    _lake_(fsspec.filesystem('file'), str(tmp_path / 'Block'))

    def progress(desc, n, total):
        raise _Stop_()

    with pytest.raises(_Stop_):
        BuildJournal.list('Block', datalake=str(tmp_path), n_workers=1, storage_options={}, progress=progress)
