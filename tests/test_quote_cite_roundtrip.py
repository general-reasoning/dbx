"""Round-trip and rendering tests for ``Datablock.quote()`` / ``.cite()``.

The two methods have opposite contracts, and neither was covered:

* ``quote()`` is the **evaluable** form. ``dbx.eval(block.quote())`` must give
  back an equivalent block -- same identity *and* same operational kwargs,
  because ``local`` decides where artifacts are staged.
* ``cite()`` is **presentation only**. It recurses over the object graph and is
  deliberately not evaluable, which is what lets it stay readable at depth.

Every test here corresponds to a failure that actually happened:

* ``quote()`` losing its ``$`` prefix -- ``is_specline()`` went False and
  ``dbx.eval`` silently returned the *string* instead of a block. No exception.
* ``quote(pretty=True)`` running ``pprint.pformat`` over the already-joined
  kwarg string, so its repr became a single positional argument
  ("__init__() takes 1 positional argument but 2 were given").
* nested children inheriting ``pretty=True``, whose newlines the parent's
  ``repr`` escaped to ``\\n``, which ``deslash`` then stripped to a bare ``n``.
* ``tailkwargs`` being dropped from ``quote()``, silently changing ``local``.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock
from dbx.dataparts import eval as dbx_eval


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Leaf(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = 'leaf'
        size: int = 3

    def __build__(self):
        pass


class Mid(Datablock):
    """One level of nesting: a Datablock-valued VAR field."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        leaf: object = None
        seed: int = 42

    def __build__(self):
        pass


class Top(Datablock):
    """Two levels of nesting -- where the escaping used to explode."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        mid: object = None
        epochs: int = 10

    def __build__(self):
        pass


@pytest.fixture
def nested(tmp_path):
    """``Top -> Mid -> Leaf``, each tagged and with an operational kwarg."""
    leaf = Leaf(url=str(tmp_path), spec=dict(label='leaf', size=3)).set(tag='leaf')
    mid = Mid(url=str(tmp_path), spec=dict(leaf=leaf, seed=42),
              local=str(tmp_path / 'local'), n_workers=7).set(tag='mid')
    return Top(url=str(tmp_path), spec=dict(mid=mid, epochs=10),
               local=str(tmp_path / 'local'), n_workers=7).set(tag='top')


# =====================================================================
# quote(): must stay evaluable
# =====================================================================

class TestQuoteRoundTrip:

    @pytest.mark.parametrize('pretty', [False, True])
    def test_quote_is_a_specline(self, nested, pretty):
        """Losing the ``$`` makes dbx.eval return a str, with no error."""
        q = nested.quote(pretty=pretty)
        assert q.startswith('$'), q[:80]
        assert Datablock.is_specline(q)

    @pytest.mark.parametrize('pretty', [False, True])
    def test_quote_evaluates_to_a_datablock(self, nested, pretty):
        back = dbx_eval(nested.quote(pretty=pretty))
        assert isinstance(back, Datablock), f"got {type(back).__name__}"

    @pytest.mark.parametrize('pretty', [False, True])
    def test_quote_preserves_identity(self, nested, pretty):
        back = dbx_eval(nested.quote(pretty=pretty))
        assert back.hash == nested.hash
        assert back.key == nested.key

    @pytest.mark.parametrize('pretty', [False, True])
    def test_quote_preserves_operational_kwargs(self, nested, pretty):
        """`local` is not in the identity hash but decides where artifacts are
        staged, so dropping it desynchronises storage from identity silently."""
        back = dbx_eval(nested.quote(pretty=pretty))
        assert back.local == nested.local
        assert back.n_workers == nested.n_workers

    def test_quote_preserves_the_nested_child(self, nested):
        back = dbx_eval(nested.quote())
        assert back.var.mid.hash == nested.var.mid.hash
        assert back.var.mid.var.leaf.hash == nested.var.mid.var.leaf.hash

    def test_pretty_does_not_emit_a_positional_argument(self, nested):
        """The pformat bug: the whole kwarg list became one quoted positional."""
        q = nested.quote(pretty=True)
        first = q[q.index('(') + 1:].lstrip()
        assert not first.startswith(('"', "'")), (
            f"first argument is a bare string -- positional: {first[:80]}"
        )

    def test_pretty_and_plain_evaluate_equal(self, nested):
        a = dbx_eval(nested.quote(pretty=False))
        b = dbx_eval(nested.quote(pretty=True))
        assert a.hash == b.hash and a.key == b.key
        assert a.local == b.local

    def test_tailkwargs_off_keeps_identity_but_loses_local(self, nested):
        """Documented trade-off, pinned so it cannot become the quote default."""
        back = dbx_eval(nested.quote(tailkwargs=False))
        assert back.hash == nested.hash
        assert back.key == nested.key           # tag is kept
        assert back.local != nested.local       # operational kwargs are not

    def test_quote_defaults_to_faithful(self, nested):
        """quote() is for evaluation, so fidelity is the default."""
        import inspect
        assert inspect.signature(Datablock.quote).parameters['tailkwargs'].default is True

    @pytest.mark.parametrize('pretty', [False, True])
    def test_nested_children_are_single_line(self, nested, pretty):
        """A child rendered multi-line has its newlines escaped by the parent's
        repr to a literal backslash-n, and `deslash` then strips the backslash
        leaving a bare 'n' -- an unevaluable specline.

        Checked over the WHOLE output, not just the line the child starts on:
        with pretty=True the child is spread across several chunk lines, so the
        escape shows up on a *later* line than the one containing "'mid':".
        """
        q = nested.quote(pretty=pretty)
        assert '\\n' not in q, [l for l in q.split('\n') if '\\n' in l][:2]


# =====================================================================
# cite(): presentation only
# =====================================================================

class TestCiteRendering:

    def test_cite_is_multiline_and_indented(self, nested):
        c = nested.cite()
        lines = c.split('\n')
        assert len(lines) > 10
        assert any(l.startswith('    ') for l in lines)

    def test_cite_has_no_escaping_at_any_depth(self, nested):
        """The point of recursing over objects instead of over quote() strings:
        a child-inside-a-child used to arrive double-escaped."""
        assert '\\' not in nested.cite()

    def test_cite_renders_children_as_blocks_not_strings(self, nested):
        """A child must appear as an unquoted nested call, not a quoted
        specline -- that is what keeps deep levels readable."""
        c = nested.cite()
        mid_line = next(l for l in c.split('\n') if "'mid':" in l)
        after = mid_line.split("'mid':", 1)[1].strip()
        assert after.startswith('$'), after[:60]
        assert not after.startswith(("'", '"')), after[:60]

    def test_cite_indents_each_level_further(self, nested):
        c = nested.cite()
        depth = {}
        for name in ("'mid':", "'leaf':"):
            line = next(l for l in c.split('\n') if name in l)
            depth[name] = len(line) - len(line.lstrip())
        assert depth["'leaf':"] > depth["'mid':"]

    def test_cite_omits_operational_tailkwargs_by_default(self, nested):
        c = nested.cite()
        assert 'n_workers' not in c
        assert "tag='top'" in c          # CITE_KEEP_TAILKWARGS

    def test_cite_tailkwargs_true_includes_them(self, nested):
        assert 'n_workers=7' in nested.cite(tailkwargs=True)

    def test_cite_defaults_to_brief(self, nested):
        import inspect
        assert inspect.signature(Datablock.cite).parameters['tailkwargs'].default is False

    def test_cite_is_shorter_than_faithful_cite(self, nested):
        assert len(nested.cite()) < len(nested.cite(tailkwargs=True))

    def test_deslash_is_harmless_for_cite(self, nested):
        """cite() emits no escapes, so deslash has nothing to corrupt -- which
        is why deslash=2 is safe here and not in quote()."""
        assert nested.cite(deslash=0) == nested.cite(deslash=2)


# =====================================================================
# helpers used by quote(pretty=True)
# =====================================================================

class TestSplitTopLevel:

    def test_splits_at_top_level_only(self):
        got = Datablock._split_top_level("a=1, b={'x': 2, 'y': 3}, c=4")
        assert got == ["a=1, ", "b={'x': 2, 'y': 3}, ", "c=4"]

    def test_respects_quotes(self):
        got = Datablock._split_top_level("a='x, y', b=2")
        assert got == ["a='x, y', ", "b=2"]

    def test_respects_backslash_escapes(self):
        got = Datablock._split_top_level(r"a='x\', y', b=2")
        assert len(got) == 2, got

    def test_respects_brackets(self):
        got = Datablock._split_top_level("a=[1, 2, 3], b=2")
        assert got == ["a=[1, 2, 3], ", "b=2"]

    def test_rejoins_verbatim(self):
        text = "a=1, b={'x': 2}, c='p, q', d=[3, 4]"
        assert ''.join(Datablock._split_top_level(text)) == text


class TestCiteChunks:

    def test_chunks_concatenate_to_the_original(self, nested):
        """Correctness is structural: the chunks are repr'd and concatenated
        verbatim, so a bad break point costs readability, never meaning."""
        specline = nested.var.mid.quote()
        rendered = nested._cite_chunks(specline, '    ')
        assert eval(rendered) == specline

    def test_chunks_are_indented(self, nested):
        rendered = nested._cite_chunks(nested.var.mid.quote(), '    ')
        assert rendered.startswith('(\n')
        assert '\n    ' in rendered


# =====================================================================
# cite() as a recorded field: Bid, the journal, and DatajournalEntry
# =====================================================================

class Solo(Datablock):
    """Actually writes its topic, so build() gets past the validity check."""
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = 'solo'

    def __build__(self):
        with open(self.path('output', ensure_dirpath=True), 'w') as f:
            f.write('data')


class TestCiteInJournal:

    @pytest.fixture
    def built(self, tmp_path):
        b = Solo(url=str(tmp_path), spec=dict(label='solo')).set(tag='solo')
        b.build()
        return b

    def test_build_writes_cite_txt(self, built):
        entry = built.journal(iloc=-1)
        assert entry.block.cite() is not None, "journal has no cite column"
        assert '-cite-' in entry.block.cite() and entry.block.cite().endswith('.txt')
        assert entry.read('cite') == built.cite()

    def test_cite_is_absent_not_fatal_on_an_older_journal(self):
        """Journals written before `cite` existed have no such column.

        `cite` is an explicit property returning .get(), so it degrades to None
        instead of raising AttributeError through pandas' attribute fallback --
        which is what a bare column lookup would do.
        """
        import pandas as pd
        from dbx.datablocks import DatajournalEntry
        entry = DatajournalEntry(pd.Series({'hash': 'abc', 'anchor': 'a.B'}))
        assert entry.block.cite() is None
        assert entry.read('cite') is None
