"""A quote writes its nested blocks bare, so nothing in it is escaped, at any depth.

Held as a string literal, a nested block's quote was escaped once more at every
level of nesting -- every backslash doubled -- so a quote twelve levels deep was
98% backslashes, and parsing it took seconds. Written bare, ``'inner':
fqcn(...)``, there is nothing to escape. Quotes recorded the old way must still
evaluate, to the same block.
"""
import pytest
from dataclasses import dataclass

import dbx.databits as databits
from dbx.datablocks import Datablock
from dbx.databits import bare_specline, eval as dbx_eval
from dbx.journals import cite_specline

DEPTH = 10
SPECLINE = "$os.path.join(dbx.getenv('HOME'), 'x, y: z')"


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Link(Datablock):
    """A chain of these, each holding the next: as deep as a real tree."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        inner: object = None
        n: int = 0
        source: str = None

    def __build__(self):
        pass


def chain(tmp_path, depth=DEPTH):
    block = Link(datalake=str(tmp_path), spec=dict(n=0, source=SPECLINE)).set(tag='link0')
    for i in range(1, depth):
        block = Link(datalake=str(tmp_path), spec=dict(inner=block, n=i), n_workers=3).set(tag=f'link{i}')
    return block


def innermost(block):
    while block.var.inner is not None:
        block = block.var.inner
    return block


def old_quote(block, monkeypatch):
    """*block*'s quote as it was rendered before: every nested block a string, escaped."""
    with monkeypatch.context() as m:
        m.setattr(databits, 'bare_specline', lambda value: None)
        return block.quote()


def test_a_deep_quote_has_nothing_escaped(tmp_path, monkeypatch):
    top = chain(tmp_path)
    assert '\\' not in top.quote()
    assert len(old_quote(top, monkeypatch)) > 4 * len(top.quote()), "the old rendering this replaces doubled at every level"


def test_a_deep_quote_evaluates_to_the_same_blocks(tmp_path):
    top = chain(tmp_path)
    back = dbx_eval(top.quote())
    assert back.hash == top.hash and back.n_workers == top.n_workers
    assert innermost(back).hash == innermost(top).hash


def test_a_quote_recorded_escaped_still_evaluates(tmp_path, monkeypatch):
    top = chain(tmp_path)
    recorded = old_quote(top, monkeypatch)
    assert '\\' in recorded
    back = dbx_eval(recorded)
    assert back.hash == top.hash
    assert innermost(back).hash == innermost(top).hash


def test_a_specline_is_written_bare_and_read_back_unexpanded(tmp_path):
    leaf = innermost(chain(tmp_path))
    assert f"'source': {SPECLINE[1:]}" in leaf.quote()
    back = dbx_eval(leaf.quote())
    assert back.spec['source'] == SPECLINE, "a specline is part of the identity unexpanded"
    assert back.hash == leaf.hash


def test_two_evaluations_share_no_spec(tmp_path):
    """The parse of a quote is read once per process; what each block is given is its own."""
    q = chain(tmp_path, depth=3).quote()
    a, b = dbx_eval(q), dbx_eval(q)
    assert a.spec is not b.spec
    a.spec['n'] = -1
    assert b.spec['n'] == 2


def test_cite_specline_lays_out_a_bare_nested_block(tmp_path):
    top = chain(tmp_path, depth=3)
    laid_out = cite_specline(top.quote())
    nested = [line for line in laid_out.splitlines() if line.lstrip().startswith(f"'inner': ${Link.__module__}.Link(")]
    assert len(nested) == 2, laid_out


@pytest.mark.parametrize('value, bare', [
    ("$a.b.C(x=1, spec={'k': 'v'})", "a.b.C(x=1, spec={'k': 'v'})"),
    ("$dbx.getenv('X')", "dbx.getenv('X')"),
    ("$a, b", None),              # a bare tuple: its comma would split the spec
    ("$lambda x: x", None),       # a lambda: its colon would split the entry
    ("$ a.b", None),              # not the text it was, once stripped
    ("$a.b(", None),              # not an expression
    ("@a.b", None),               # not a $ specline
    ("plain", None),
])
def test_bare_specline(value, bare):
    assert bare_specline(value) == bare
