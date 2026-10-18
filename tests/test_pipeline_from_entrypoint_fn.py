"""@pipeline: an entrypoint function's arguments as a Pipeline's VAR; try_specialize_tree: adopt a tree, build none."""
from dataclasses import dataclass

import pytest

import dbx
from dbx.datablocks import DATAFILE, Datablock
from dbx.pipelines import PipelineFromEntrypointFn, pipeline


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


@pipeline(journal=True)
def leaf(name, *, var_n: int = 1, datalake=None):
    return Leaf(datalake=datalake, spec={'n': var_n}, tag=name)


@pipeline(tag='rooted', journal=True)
def root(name, *, var_n: int = 1, var_m: int = 1, n_workers: int = 1, datalake=None):
    return Root(datalake=datalake, spec={'leaf': leaf(name, var_n=var_n, datalake=datalake).entry, 'm': var_m})


@pipeline(journal=True)
def no_datalake(name, *, var_n: int = 1):
    return Leaf(spec={'n': var_n})


@pytest.fixture
def lake(tmp_path, monkeypatch):
    monkeypatch.setenv('DBX_DATALAKE', str(tmp_path))
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')
    return str(tmp_path)


def test_the_var_is_the_leading_and_var_arguments_and_the_rest_is_operational(lake):
    p = root('a', var_m=3, n_workers=4)
    assert isinstance(p, PipelineFromEntrypointFn) and type(p).__name__ == 'RootPipeline'
    assert (p.var.name, p.var.n, p.var.m) == ('a', 1, 3)
    assert p.operational['n_workers'] == 4
    assert root('a', var_m=3, n_workers=8).hash == p.hash, "operational arguments are no identity"
    assert root('a', var_m=4).hash != p.hash


def test_the_entry_is_the_functions_tree(lake):
    block = root('a', var_n=2, var_m=3).entry
    assert isinstance(block, Root) and block.var.leaf.var.n == 2 and block.var.m == 3


def test_a_block_held_hashes_as_its_quote(lake):
    held = Root(spec={'leaf': leaf('a', var_n=2).entry, 'm': 3})
    quoted = Root(spec={'leaf': dbx.quote(Leaf(spec={'n': 2}, tag='a')), 'm': 3})
    assert held.hash == quoted.hash


def test_the_tag_is_the_functions_qualified_name_unless_given(lake):
    assert leaf('a').tag == f"{__name__}.leaf"
    assert root('a').tag == 'rooted'
    leaf('b').entry
    assert len(dbx.pipejournal(datalake=lake, tag='leaf')) == 1, "a tag filter matches by containment"
    root('a').entry     # creates a leaf pipeline too, within its tree
    assert len(dbx.pipejournal(datalake=lake, tag='rooted')) == 1
    assert len(dbx.pipejournal(datalake=lake, tag='leaf')) == 2


def test_its_quote_evaluates(lake):
    p = root('a', var_m=3)
    assert dbx.databits.eval(p.quote()).hash == p.hash


def test_every_pipeline_takes_a_datalake_and_its_function_only_when_it_does(tmp_path, lake):
    other = str(tmp_path / 'other')
    p = no_datalake('a', datalake=other)
    p.entry
    assert len(dbx.pipejournal(datalake=other)) == 1


def test_a_call_is_cached_by_all_its_arguments(lake):
    assert root('a', var_m=3) is root('a', var_m=3)
    assert root('a', var_m=3, n_workers=4) is not root('a', var_m=3), "operational arguments key the cache too"
    assert root('a', var_m=3) is not root('a', var_m=4)


def test_unhashable_arguments_are_cached_by_value(lake):
    @pipeline
    def columns(name, *, var_columns: dict = None, devices: list = None):
        return Leaf(spec={'n': len(var_columns or {})}, tag=name)

    p = columns('a', var_columns={'signals': ['x', 'y']}, devices=['cuda:0'])
    assert columns('a', var_columns={'signals': ['x', 'y']}, devices=['cuda:0']) is p
    assert columns('a', var_columns={'signals': ['x']}, devices=['cuda:0']) is not p


def test_a_nested_pipeline_is_created_and_journaled_once(lake):
    root('a', var_m=3).entry
    root('a', var_m=4).entry     # another root, the same leaf
    assert len(dbx.pipejournal(datalake=lake, tag='rooted')) == 2
    (row,) = [r for _, r in dbx.PipeJournal.read(dbx.PipeJournal.ANCHOR, datalake=lake).iterrows()
              if r['tag'].endswith('.leaf')]
    assert row['event'] == 'create'


def test_an_omitted_datalake_is_keyed_by_dbx_datalake_at_the_call(lake, tmp_path, monkeypatch):
    p = leaf('a')
    monkeypatch.setenv('DBX_DATALAKE', str(tmp_path / 'elsewhere'))
    assert leaf('a') is not p and leaf('a') is leaf('a')


def test_cache_clear(lake):
    p = leaf('a')
    leaf.cache_clear()
    assert leaf('a') is not p and leaf('a').hash == p.hash


def test_a_pipeline_not_journaled_writes_nothing(lake):
    @pipeline
    def quiet(name, *, var_n: int = 1):
        return Leaf(spec={'n': var_n}, tag=name)

    assert quiet.Pipeline.JOURNAL is False and leaf.Pipeline.JOURNAL is True
    quiet('a').entry.build()
    assert len(dbx.pipejournal(datalake=lake)) == 0


def test_a_var_through_kwargs_is_refused(lake):
    @pipeline
    def passing(name, **operational):
        return Leaf(spec={'n': 1})
    assert passing('a', n_workers=2).operational['n_workers'] == 2
    with pytest.raises(TypeError, match="identity"):
        passing('a', var_n=2)


def test_try_specialize_tree_specializes_what_resolves_and_builds_nothing(lake):
    """A leaf built under its old topic name is adopted; the root, never built, is left not valid."""
    class OldLeaf(Datablock):
        TOPICS = {'old': DATAFILE('x.txt')}

        @dataclass
        class VAR(Datablock.VAR):
            n: int = 1

        def __build__(self):
            with open(self.path('old', ensure_dirpath=True), 'w') as f:
                f.write('old')

    old = OldLeaf(anchor='Grown', spec={'n': 2}).build()
    renamed = Datablock.Specialization(spec={}, version=None, topics={'old': DATAFILE('x.txt')},
                                       redirect_topics={'x': 'old'})
    grown = Leaf(anchor='Grown', spec={'n': 2}, SPECIALIZATIONS=[renamed])
    tree = Root(spec={'leaf': grown, 'm': 1})
    status = tree.try_specialize_tree()
    assert status == {grown.anchorkey: ('valid', []), tree.anchorkey: ('not-valid', [])}
    assert grown.path('x') == old.path('old') and not tree.valid()
