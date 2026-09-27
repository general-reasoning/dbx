"""
Constructing a block writes nothing; only build() alters the datalake.

A specialization used to be installed -- redirected, and recorded in a
``.redirection`` marker and a journal entry -- as a block was CONSTRUCTED. So
merely forming a block, or asking whether it was valid, wrote to the lake, and
paid a journal read for it. Construction is declarative now: resolving and
installing are build()'s, which the caller sanctions by calling it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_specializations import v1, v1table, v2  # noqa: E402
from test_specializations import TestOneJournalReadForAWholeTable as _Tables  # noqa: E402 -- not collected twice


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


def _lake(root):
    """Every file under *root*, with its size and modification time."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


def test_constructing_and_asking_writes_nothing(tmp_path):
    v1(tmp_path).build()                                  # an older, narrower build
    before = _lake(tmp_path)
    block = v2(tmp_path)                                  # which it could adopt
    block.valid()
    block.redirected_topics()
    block.specializations()
    block.find_specialization()
    assert block.redirected_topics() == []
    assert _lake(tmp_path) == before


def test_a_table_and_its_tabs_are_formed_and_queried_without_writing(tmp_path, monkeypatch):
    v1table(tmp_path, spec={'n': 3}).build()
    _Tables._grow_the_tab(monkeypatch)
    before = _lake(tmp_path)
    table = v1table(tmp_path, spec={'n': 3})
    for i in range(3):
        table.tab(i).valid()
    table.valid_tabs(parallelization='inline')
    table.find_tab_specializations(parallelization='inline')
    assert _lake(tmp_path) == before


def test_building_is_what_adopts(tmp_path):
    v1(tmp_path).build()
    block = v2(tmp_path)
    assert block.redirected_topics() == []
    block.build()
    assert block.redirected_topics() == ['spectra']
    assert v2(tmp_path).redirected_topics() == ['spectra']     # recorded, for every later one
