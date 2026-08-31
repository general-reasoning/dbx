"""A remote tab is cached where the table says, not where mosaic guesses.

``Stream`` left without a ``local=`` invents ``{tmpdir}/{blake2s(remote)}`` and
then REFUSES to reuse it, so the second open of the same remote tab -- a second
process, a second run, a retry after a crash -- dies with "Could not create a
temporary local directory ... already exists". The table computed a cache
directory and never passed it, so every remote read went to that guessed path.

It cannot be passed to ``StreamingDataset`` either: mosaic takes ``streams=`` or
``remote``/``local``, never both. It belongs on each ``Stream``, one
subdirectory per tab -- which is also what keeps two tabs from sharing a
directory, itself the same collision.

These drive the wiring with stub tabs rather than a real cloud path: what is
under test is which directory each stream is told to use, and no network is
needed to see that.
"""
import os
import pytest
from dataclasses import dataclass

pytest.importorskip("torch", reason="torch is an optional dependency")
pytest.importorskip("streaming", reason="mosaicml-streaming is an optional dependency")

from dbx import datapoints
from dbx.datapoints import SLICETOPIC, DatapointTab, DatapointTable


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Tab(DatapointTab):
    VERSION = 1
    TOPICS = {'numbers': SLICETOPIC}

    @dataclass
    class VAR(DatapointTab.VAR):
        n: int = 2

    def __build__(self):
        with self.slice_writers({'numbers': {'idx': 'int'}}) as writers:
            for i in range(self.var.n):
                writers['numbers'].write({'idx': i})


class Table(DatapointTable):
    VERSION = 1
    TAB = Tab

    @dataclass
    class VAR(DatapointTable.VAR):
        n_tabs_: int = 2

    @property
    def n_tabs(self):
        return self.var.n_tabs_


class RemoteTab:
    """A tab whose slice lives somewhere that has to be downloaded."""

    def __init__(self, hash):
        self.hash = hash

    def path(self, *topicpath):
        return f"abfss://c@a.dfs.core.windows.net/{self.hash}/{'/'.join(topicpath)}"


@pytest.fixture
def table(tmp_path):
    return Table(url=str(tmp_path), spec=dict(n_tabs_=2))


@pytest.fixture
def remote_table(table, monkeypatch):
    """The table above, with its tabs replaced by remote ones."""
    monkeypatch.setattr(type(table), 'tab',
                        lambda self, idx: RemoteTab(f"{'ab'[idx]}" * 32), raising=False)
    return table


@pytest.fixture
def opened(monkeypatch):
    """Records the streams a datastream() call would open, without opening them."""
    calls = []

    class Recorder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(datapoints, 'StreamingDataset', Recorder)
    return calls


class TestARemoteTabIsToldWhereToCache:

    def test_the_stream_gets_a_local_directory(self, table, tmp_path):
        stream = table._tab_stream(RemoteTab('a' * 32), 'numbers', local=str(tmp_path / 'cache'))
        assert stream.local == str(tmp_path / 'cache')

    def test_without_one_it_refuses_rather_than_letting_mosaic_guess(self, table):
        with pytest.raises(ValueError, match='local'):
            table._tab_stream(RemoteTab('a' * 32), 'numbers')

    def test_the_directory_is_created(self, table, tmp_path):
        local = tmp_path / 'cache' / 'deeper'
        table._tab_stream(RemoteTab('a' * 32), 'numbers', local=str(local))
        assert local.is_dir()

    def test_abfss_is_translated_for_mosaic(self, table, tmp_path):
        stream = table._tab_stream(RemoteTab('a' * 32), 'numbers', local=str(tmp_path / 'c'))
        assert stream.remote.startswith('azure-dl://')


class TestALocalTabIsItsOwnCache:

    def test_no_copy_is_arranged_for_it(self, table):
        tab = table.tab(0)
        stream = table._tab_stream(tab, 'numbers')
        assert stream.remote is None
        assert stream.local == tab.path('numbers')

    def test_a_local_argument_is_ignored(self, table, tmp_path):
        tab = table.tab(0)
        stream = table._tab_stream(tab, 'numbers', local=str(tmp_path / 'unused'))
        assert stream.local == tab.path('numbers')
        assert not (tmp_path / 'unused').exists()


class TestDatastreamPassesItsCacheDown:
    """The defect: the directory was computed, created, and then dropped."""

    def test_every_stream_is_cached_under_the_tables_cache_dir(self, remote_table, opened):
        remote_table.datastream('numbers')
        streams = opened[0]['streams']
        assert len(streams) == remote_table.n_tabs
        for stream in streams:
            assert stream.local is not None
            assert stream.local.startswith(remote_table.cacheroot)

    def test_no_stream_lands_on_mosaics_guess(self, remote_table, opened):
        """`{tmpdir}/{blake2s(remote)}` is where mosaic puts a stream it was not
        told about: shared by every process on the box, and refused on reuse."""
        import hashlib
        import tempfile
        remote_table.datastream('numbers')
        for stream in opened[0]['streams']:
            guess = os.path.join(
                tempfile.gettempdir(),
                hashlib.blake2s(stream.remote.encode('utf-8'), digest_size=16).hexdigest(),
            )
            assert stream.local != guess

    def test_tabs_do_not_share_a_directory(self, remote_table, opened):
        remote_table.datastream('numbers')
        locals_ = [s.local for s in opened[0]['streams']]
        assert len(set(locals_)) == len(locals_)

    def test_slices_do_not_share_a_directory(self, remote_table, opened):
        remote_table.datastream('numbers')
        first = {s.local for s in opened[0]['streams']}
        remote_table._slicenames = lambda slices: tuple(slices)   # accept a second name
        remote_table.slice_names = lambda slices: tuple(slices)   # accept a second name
        remote_table.datastream('letters')
        second = {s.local for s in opened[1]['streams']}
        assert not (first & second)

    def test_an_explicit_cache_dir_is_honoured(self, remote_table, opened, tmp_path):
        remote_table.datastream('numbers', cache=str(tmp_path / 'elsewhere'), cache_dir='mine')
        for stream in opened[0]['streams']:
            assert stream.local.startswith(str(tmp_path / 'elsewhere' / 'mine'))

    def test_the_same_table_caches_to_the_same_place_twice(self, remote_table, opened):
        """Stable across calls and runs, so a cache is reused rather than
        rebuilt -- and never with mosaic's refuse-on-reuse semantics."""
        remote_table.datastream('numbers')
        remote_table.datastream('numbers')
        assert [s.local for s in opened[0]['streams']] == [s.local for s in opened[1]['streams']]

    def test_the_dataset_is_not_given_a_local_of_its_own(self, remote_table, opened):
        """mosaic takes streams= or remote/local, never both."""
        remote_table.datastream('numbers')
        assert 'local' not in opened[0] and 'remote' not in opened[0]


class TestLocalTablesStillRead:
    """The path every test in the suite takes: nothing about it changes."""

    def test_a_built_table_reads_back(self, table):
        table.build()
        assert len(table.datastream('numbers')) == 4

    def test_twice_in_a_row(self, table):
        table.build()
        first = len(table.datastream('numbers'))
        assert len(table.datastream('numbers')) == first
