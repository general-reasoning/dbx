"""
Pipeline: a Datablock tree defined flat; PipeJournal: where every creation of one is recorded.
"""
import os
from dataclasses import dataclass

import pytest

import dbx
from dbx.datablocks import DATAFILE, Datablock
from dbx.journals import DataJournal, DataJournalEntry, DataJournalFrame
from dbx.journals import PipeJournal
from dbx.pipelines import Pipeline


class Leaf(Datablock):
    TOPICS = {'x': DATAFILE('x.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        n: int = 1

    def __build__(self):
        with open(self.path('x', ensure_dirpath=True), 'w') as f:
            f.write(str(self.var.n))


class Root(Datablock):
    TOPICS = {'y': DATAFILE('y.txt')}

    @dataclass
    class VAR(Datablock.VAR):
        leaf: Leaf = None
        m: int = 1

    def __build__(self):
        with open(self.path('y', ensure_dirpath=True), 'w') as f:
            f.write(str(self.var.m))


class Flat(Pipeline):
    @dataclass
    class VAR(Pipeline.VAR):
        n: int = 1
        m: int = 1

    def __create__(self):
        return Root(spec={'leaf': Leaf(spec={'n': self.var.n}), 'm': self.var.m})


class Broken(Pipeline):
    def __create__(self):
        return 'not a block'


@pytest.fixture
def lake(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
    return str(tmp_path)


def test_entrypoint_is_what_create_returns(lake):
    p = Flat(spec={'n': 2, 'm': 3})
    root = p.create()
    assert isinstance(root, Root) and root.var.m == 3 and root.var.leaf.var.n == 2
    assert p.entrypoint.hash == root.hash
    assert p.read('entrypoint').hash == root.hash
    assert p.entrypath == root.anchorkeypath


def test_entrypoint_is_synthetic(lake):
    p = Flat()
    assert p.path('entrypoint') is None
    assert p.valid()


def test_create_must_return_a_block(lake):
    with pytest.raises(TypeError, match='not the Datablock'):
        Broken().create()


def test_abstract(lake):
    with pytest.raises(NotImplementedError, match='__create__'):
        Pipeline().create()


def test_construction_writes_nothing(lake):
    Flat()
    assert not os.path.exists(os.path.join(lake, PipeJournal.ANCHOR))


def test_create_is_journaled_under_pipelines(lake):
    p = Flat()
    p.create()
    frame = PipeJournal.read(PipeJournal.ANCHOR, datalake=lake)
    assert len(frame) == 1
    entry = frame.iloc[0]
    assert entry['event'] == 'create'
    assert entry['entrypath'] == p.entrypath
    assert entry['hash'] == p.hash
    assert entry['entry_path'].startswith(os.path.join(lake, PipeJournal.ANCHOR, p.anchorkey, '.journal'))
    # ... and not in the pipeline's own journal.
    assert not os.path.exists(os.path.join(p.anchorkeypath, '.journal'))


def test_pipelines_go_to_the_pipelines_own_datalake(lake, tmp_path_factory):
    elsewhere = str(tmp_path_factory.mktemp('elsewhere'))
    p = Flat(datalake=elsewhere)
    p.create()
    assert os.path.isdir(os.path.join(elsewhere, PipeJournal.ANCHOR, p.anchorkey, '.journal'))
    assert not os.path.exists(os.path.join(lake, PipeJournal.ANCHOR))
    assert list(PipeJournal.collect(datalake=elsewhere)['hash']) == [p.hash]


def test_a_block_journal_is_where_it_was(lake):
    leaf = Leaf()
    leaf.build()
    assert DataJournal.dirpath(leaf).startswith(os.path.join(leaf.anchorkeypath, '.journal'))
    assert len(leaf.datajournal(event='build:end')) == 1
    assert 'entrypath' not in leaf.datajournal().columns


def test_extra_columns_may_not_replace_the_entrys_own(lake):
    with pytest.raises(ValueError, match='hash'):
        DataJournal.write(Leaf(), 'note', columns={'hash': 'x'})


def test_collect_groups_entrypaths_per_pipeline(lake):
    a, b = Flat(spec={'n': 1}), Flat(spec={'n': 2})
    a.create()
    Flat(spec={'n': 1}).create()        # another instance: another entry, same entrypath
    b.create()
    frame = PipeJournal.collect()
    assert isinstance(frame, DataJournalFrame)
    assert len(frame) == 2
    assert 'entrypath' not in frame.columns
    by_hash = {row['hash']: row['entrypaths'] for _, row in frame.iterrows()}
    assert by_hash == {a.hash: [a.entrypath], b.hash: [b.entrypath]}
    assert list(frame['hash']) == [b.hash, a.hash]       # newest first


def test_collect_keeps_every_distinct_entrypath(lake, monkeypatch):
    p = Flat()
    p.create()
    first = p.entrypath
    # A changed __create__: the same pipeline, a new entrypoint.
    monkeypatch.setattr(Flat, '__create__', lambda self: Leaf(spec={'n': 99}))
    Flat().create()
    (row,) = [r for _, r in PipeJournal.collect().iterrows()]
    assert row['entrypaths'] == [Leaf(spec={'n': 99}).anchorkeypath, first]


def test_collect_filters(lake):
    a, b = Flat(spec={'n': 1}), Flat(spec={'n': 2})
    a.create()
    b.create()
    assert list(PipeJournal.collect(hash=a.hash)['hash']) == [a.hash]
    assert list(PipeJournal.collect(entrypath=b.entrypath)['hash']) == [b.hash]
    assert len(PipeJournal.collect(nosuchcolumn='x')) == 0


def test_collect_construct_only(lake):
    built, redirected, unbuilt = Flat(spec={'n': 1}), Flat(spec={'n': 2}), Flat(spec={'n': 3})
    for p in (built, redirected, unbuilt):
        p.create()
    built.entrypoint.build_tree()
    redirected.entrypoint.write_journal_entry(event='UNSAFE_redirect')
    assert len(PipeJournal.collect()) == 3
    frame = PipeJournal.collect(construct_only=True)
    assert set(frame['hash']) == {built.hash, redirected.hash}
    assert all(len(ps) == 1 for ps in frame['entrypaths'])


def test_collect_of_an_empty_lake(lake):
    frame = PipeJournal.collect()
    assert isinstance(frame, DataJournalFrame) and len(frame) == 0
    assert len(PipeJournal.collect(construct_only=True)) == 0


def test_pipelines_are_not_anchors(lake):
    Flat().create()
    assert PipeJournal.ANCHOR not in dbx.anchors(lake)


def test_exported():
    assert dbx.Pipeline is Pipeline and dbx.PipeJournal is PipeJournal
    assert dbx.pipejournal is dbx.journals.pipejournal


def test_pipejournal_is_collect(lake):
    a, b = Flat(spec={'n': 1}), Flat(spec={'n': 2})
    a_entrypath = a.create().anchorkeypath
    b.create()
    # Read only now: entrypath creates, and so re-journals, a pipeline.
    frame = dbx.pipejournal()
    assert list(frame['hash']) == list(PipeJournal.collect()['hash']) == [b.hash, a.hash]
    assert list(dbx.pipejournal(hash=a.hash)['hash']) == [a.hash]
    entry = dbx.pipejournal(iloc=1)
    assert isinstance(entry, DataJournalEntry) and entry['entrypaths'] == [a_entrypath]
    assert dbx.pipejournal(loc=0)['hash'] == b.hash


def test_journal_dispatches_pipelines(lake):
    Flat().create()
    for name in ('pipelines', PipeJournal.ANCHOR):
        assert list(dbx.journal(name)['hash']) == [Flat().hash]
    assert len(dbx.journal('pipelines', construct_only=True)) == 0


def test_constructed_entrypaths_are_on_the_datalakes_filesystem():
    """A bare entrypath -- as the datalake's host recorded it -- is looked for on the datalake, not here.

    Read over ssh, a datalake's entrypaths are paths on the remote host; a
    memory:// datalake stands in for one, its bare paths absent from the
    local disk.
    """
    import fsspec
    import pandas as pd
    from dbx.journals import Datalog, PipeJournal

    fs = fsspec.filesystem('memory')
    built, unbuilt = '/remote-lake/Model/v1', '/remote-lake/Model/v2'
    for path, event in ((built, 'build:end'), (unbuilt, 'build:start')):
        with fs.open(f'{path}/.journal/pkg.Model/journal/h/e.parquet', 'wb') as f:
            pd.DataFrame([{'event': event}]).to_parquet(f)
    found = PipeJournal._constructed_entrypaths_({built, unbuilt}, datalake='memory:///remote-lake',
                                                 storage_options={}, n_workers=2, log=Datalog())
    assert found == {built}
