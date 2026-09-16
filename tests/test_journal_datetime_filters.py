"""
Filtering a journal by ``date=`` or ``datetime=``.

dbx writes a timestamp as ``isoformat()`` with ``' '`` and ``':'`` replaced by
``'-'`` (`JOURNAL_DATETIME_FORMAT`), so that it can also be a path component.
pandas cannot parse that unaided: dateutil reads the ``-`` between the hour and
the minute as a date separator and raises.

A block journal is parsed on the way into `Datajournal`, so it reached the
filter already holding datetimes and these filters worked. The exec journal is
not, so it reached the same filter holding the raw strings, where ``date=``
raised ``DateParseError`` and ``datetime=`` compared a string column against a
parsed value and silently matched nothing.
"""

import datetime

import pandas as pd
import pytest

import dbx
from dbx.dataparts import (
    JOURNAL_DATETIME_FORMAT,
    filter_journal_frame,
    write_exec_journal,
)


STAMPS = [
    '2026-09-16T18-41-43.619534',
    '2026-09-16T09-05-00.000001',
    '2026-09-15T23-59-59.999999',
]


@pytest.fixture
def raw_frame():
    """A frame shaped like the exec journal: 'datetime' holds dbx's strings."""
    return pd.DataFrame({'exec': ['a', 'b', 'c'], 'datetime': STAMPS})


@pytest.fixture
def parsed_frame(raw_frame):
    """The same frame shaped like a block journal: 'datetime' already parsed."""
    df = raw_frame.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], format=JOURNAL_DATETIME_FORMAT)
    return df


# date= ---------------------------------------------------------------------

def test_date_filters_unparsed_timestamps(raw_frame):
    """The bug: this raised DateParseError instead of filtering."""
    assert list(filter_journal_frame(raw_frame, date='2026-09-16')['exec']) == ['a', 'b']


def test_date_filters_parsed_timestamps(parsed_frame):
    assert list(filter_journal_frame(parsed_frame, date='2026-09-16')['exec']) == ['a', 'b']


def test_date_accepts_a_list(raw_frame):
    got = filter_journal_frame(raw_frame, date=['2026-09-16', '2026-09-15'])
    assert list(got['exec']) == ['a', 'b', 'c']


def test_date_accepts_a_date_object(raw_frame):
    got = filter_journal_frame(raw_frame, date=datetime.date(2026, 9, 15))
    assert list(got['exec']) == ['c']


def test_date_accepts_a_dbx_timestamp_and_truncates_it(raw_frame):
    """A whole timestamp names the day it falls on, not only the row it equals."""
    got = filter_journal_frame(raw_frame, date='2026-09-16T18-41-43.619534')
    assert list(got['exec']) == ['a', 'b']


def test_date_that_matches_nothing_is_empty(raw_frame):
    assert len(filter_journal_frame(raw_frame, date='2020-01-01')) == 0


# datetime= -----------------------------------------------------------------

def test_datetime_filters_unparsed_timestamps(raw_frame):
    """The same bug, silent: a str column never equalled the parsed value."""
    assert list(filter_journal_frame(raw_frame, datetime=STAMPS[1])['exec']) == ['b']


def test_datetime_filters_parsed_timestamps(parsed_frame):
    assert list(filter_journal_frame(parsed_frame, datetime=STAMPS[1])['exec']) == ['b']


def test_datetime_accepts_a_list_of_dbx_timestamps(raw_frame):
    """The list branch parsed with bare to_datetime, so it raised on dbx's format."""
    got = filter_journal_frame(raw_frame, datetime=[STAMPS[0], STAMPS[2]])
    assert list(got['exec']) == ['a', 'c']


# Timestamps the exact format does not cover --------------------------------

def test_a_whole_second_timestamp_still_matches():
    """isoformat() drops '.%f' when the microsecond is 0."""
    df = pd.DataFrame({'exec': ['a', 'b'], 'datetime': ['2026-09-16T18-41-43', STAMPS[2]]})
    assert list(filter_journal_frame(df, date='2026-09-16')['exec']) == ['a']


def test_a_foreign_timestamp_is_not_dropped():
    """A column carrying an ISO timestamp from elsewhere still filters."""
    df = pd.DataFrame({'exec': ['a', 'b'], 'datetime': ['2026-09-16 18:41:43', STAMPS[2]]})
    assert list(filter_journal_frame(df, date='2026-09-16')['exec']) == ['a']


def test_an_unreadable_timestamp_matches_no_date_rather_than_raising():
    df = pd.DataFrame({'exec': ['a', 'b'], 'datetime': ['not a timestamp', STAMPS[2]]})
    assert list(filter_journal_frame(df, date='2026-09-15')['exec']) == ['b']


# End to end, through the exec journal --------------------------------------

def test_exec_journal_is_filterable_by_date(tmp_path, monkeypatch):
    """What dbx.journal(date=...) does on a real exec journal."""
    monkeypatch.setenv('DBX_URL', str(tmp_path / 'root'))
    write_exec_journal("1 + 1  # a comment naming this row")

    today = datetime.date.today().isoformat()
    assert len(dbx.journal(date=today)) == 1
    assert len(dbx.journal(date='2000-01-01')) == 0
    assert dbx.journal(date=today).iloc[0]['comment'] == 'a comment naming this row'
