"""Recursive, sparse ``diffnorm``: descend into nested blocks instead of blobbing.

A norm is flat text, so a nested block arrives inside the parent's spec dict as
one long string. The old comparison stopped at the top-level kwargs, so a single
changed leaf three levels down came back as::

    {'spec': (<2820-char string>, <2864-char string>)}

-- two near-identical blobs to eyeball. The real-world case that prompted this
was an ``IJEPAsaurUSStill`` whose only difference from a recorded build was that
``num_workers`` and ``prefetch_factor`` had since been removed from VAR.

Structuring has to be conservative in one specific way: values that merely
*look* parenthesised (a tuple like ``'(0.75, 1.5)'``) must stay leaves, or a
diff would report ``{}`` for them and hide a real change.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import ABSENT, Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Leaf(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        label: str = 'leaf'
        ratio: object = None

    def __build__(self):
        pass


class Mid(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        leaf: object = None
        seed: int = 42

    def __build__(self):
        pass


class Top(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class VAR(Datablock.VAR):
        mid: object = None
        epochs: int = 10

    def __build__(self):
        pass


def _tree(tmp_path, *, label='leaf', ratio=(0.75, 1.5), seed=42, epochs=10):
    leaf = Leaf(url=str(tmp_path), spec=dict(label=label, ratio=ratio))
    mid = Mid(url=str(tmp_path), spec=dict(leaf=leaf, seed=seed))
    return Top(url=str(tmp_path), spec=dict(mid=mid, epochs=epochs))


class TestRecursiveDescent:

    def test_leaf_change_three_levels_down_is_a_short_path(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED')
        diff = b.diffsubsig(a.subsignature())
        assert diff == {'spec': {'mid': {'spec': {'leaf': {'spec': {
            'label': ('CHANGED', 'leaf')}}}}}}

    def test_only_the_differing_leaf_appears(self, tmp_path):
        """Sparse: siblings that match must not be carried along."""
        diff = _tree(tmp_path, seed=7).diffsubsig(_tree(tmp_path).subsignature())
        assert diff == {'spec': {'mid': {'spec': {'seed': (7, 42)}}}}
        assert 'leaf' not in diff['spec']['mid']['spec']

    def test_two_changes_at_different_depths(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED', epochs=99)
        diff = b.diffsubsig(a.subsignature())
        assert diff['spec']['epochs'] == (99, 10)
        assert diff['spec']['mid']['spec']['leaf']['spec']['label'] == (
            'CHANGED', 'leaf')

    def test_flat_mode_keeps_the_whole_subtree(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED')
        flat = b.diffsubsig(a.subsignature(), recursive=False)
        assert set(flat) == {'spec'}
        self_side, other_side = flat['spec']
        assert not isinstance(self_side, tuple), "flat mode descended anyway"
        assert len(repr(self_side)) > 100, "the whole subtree should be one value"
        # raw=True is the un-deserialised form: one long string, as rendered.
        raw_self = b.diffsubsig(a.subsignature(), recursive=False, raw=True)['spec'][0]
        assert isinstance(raw_self, str) and len(raw_self) > 100

    def test_no_difference_is_empty(self, tmp_path):
        a = _tree(tmp_path)
        assert a.diffsubsig(a.subsignature()) == {}

    def test_url_difference_stays_at_the_top(self, tmp_path):
        class LegacyTop(Top):
            LEGACY_NORM = True
        a = LegacyTop(url=str(tmp_path))
        b = LegacyTop(url=str(tmp_path / 'elsewhere'))
        diff = b.diffsubsig(a.subsignature())
        assert 'url' in diff


class TestTupleValuesStayLeaves:
    """``'(0.75, 1.5)'`` is parenthesised but is NOT a nested block."""

    def test_tuple_change_is_reported(self, tmp_path):
        a = _tree(tmp_path, ratio=(0.75, 1.5))
        b = _tree(tmp_path, ratio=(0.5, 2.0))
        diff = b.diffsubsig(a.subsignature())
        leafdiff = diff['spec']['mid']['spec']['leaf']['spec']
        assert 'ratio' in leafdiff
        # Evaluated back into real tuples, not left as text.
        assert leafdiff['ratio'] == ((0.5, 2.0), (0.75, 1.5))

    def test_structure_normval_leaves_a_tuple_alone(self):
        assert Datablock._structure_subsignatureval("'(0.75, 1.5)'") == "'(0.75, 1.5)'"
        assert Datablock._structure_subsignatureval('(0.75, 1.5)') == '(0.75, 1.5)'

    def test_structure_normval_expands_a_real_norm(self):
        got = Datablock._structure_subsignatureval("(url=/tmp/x, spec={'a': '1'})")
        assert got == {'url': '/tmp/x', 'spec': {'a': "'1'"}}

    def test_structure_normval_unwraps_one_layer_of_quoting(self):
        """A child norm is stored as a string VALUE in the parent's spec."""
        got = Datablock._structure_subsignatureval('"(url=/tmp/x, spec={})"')
        assert got == {'url': '/tmp/x', 'spec': '{}'}

    def test_structure_normval_passes_through_plain_scalars(self):
        for text in ("'42'", '42', 'None', "'a, b'"):
            assert Datablock._structure_subsignatureval(text) == text


class TestDeslash:

    def test_deslash_strips_escapes_from_reported_values(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED')
        raw = b.diffsubsig(a.subsignature(), recursive=False, raw=True)['spec'][0]
        assert '\\' in raw, "expected the flat form to carry escapes"
        clean = b.diffsubsig(a.subsignature(), recursive=False, raw=True,
                           deslash=True)['spec'][0]
        assert '\\' not in clean

    def test_deslash_does_not_break_parsing(self, tmp_path):
        """Deslashing must happen on output, not before the parse.

        Stripping backslashes first would destroy the ``\\'`` escapes that mark
        where a nested norm's own quoted values begin and end, so the structure
        would collapse and the diff would come back at the wrong depth.
        """
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED')
        assert (b.diffsubsig(a.subsignature(), deslash=True) ==
                b.diffsubsig(a.subsignature(), deslash=False))


class TestReport:

    def test_report_is_one_path_per_difference(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED', epochs=99)
        text = b.diffsubsig(a.subsignature(), report=True)
        assert 'spec.epochs' in text
        assert 'spec.mid.spec.leaf.spec.label' in text
        assert text.count('self :') == 2

    def test_report_says_so_when_identical(self, tmp_path):
        a = _tree(tmp_path)
        assert a.diffsubsig(a.subsignature(), report=True) == 'no differences'

    def test_report_truncates_long_values_but_the_dict_does_not(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='x' * 400)
        text = b.diffsubsig(a.subsignature(), report=True, maxlen=40)
        assert '(+' in text and 'chars)' in text
        full = b.diffsubsig(a.subsignature())['spec']['mid']['spec']['leaf']['spec']['label'][0]
        assert full == 'x' * 400

    def test_maxlen_none_disables_truncation(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='x' * 400)
        text = b.diffsubsig(a.subsignature(), report=True, maxlen=None)
        assert 'chars)' not in text


class TestJournalFilters:
    """``dict(event='build:end', iloc=0)`` used to silently drop the event."""

    class Solo(Datablock):
        TOPICS = {'output': 'output.txt'}

        @dataclass
        class VAR(Datablock.VAR):
            x: int = 1

        def __build__(self):
            with open(self.path('output', ensure_dirpath=True), 'w') as f:
                f.write('data')

    def test_extra_keys_filter_the_journal(self, tmp_path):
        """The exact bug: iloc=0 alone is the newest entry of ANY event.

        For a block whose artifact was copied in rather than built, the newest
        entry is an ``UNSAFE_copy_from:END`` at the CURRENT identity, so the
        diff came back empty and the real difference against the recorded build
        stayed hidden.
        """
        a = self.Solo(url=str(tmp_path), spec={'x': 1})
        a.build()
        # A second INSTANCE: one journal file per instance, so writing from `a`
        # again would overwrite its build:end rather than add a row.
        self.Solo(url=str(tmp_path), spec={'x': 1}).write_journal_entry(
            event='UNSAFE_copy_from:END')

        filtered = a._journal_entry({'event': 'build:end', 'iloc': 0})
        unfiltered = a._journal_entry({'iloc': 0})
        assert filtered.get('event') == 'build:end'
        assert unfiltered.get('event') == 'UNSAFE_copy_from:END'

    def test_filtered_selector_reaches_the_right_norm(self, tmp_path):
        a = self.Solo(url=str(tmp_path), spec={'x': 1})
        a.build()
        b = self.Solo(url=str(tmp_path), spec={'x': 2})
        assert b.diffsubsig(journal={'event': 'build:end', 'iloc': 0}) == {
            'spec': {'x': (2, 1)}}

    def test_a_filter_matching_nothing_is_visible(self, tmp_path):
        a = self.Solo(url=str(tmp_path), spec={'x': 1})
        a.build()
        with pytest.raises(Exception):
            a.diffsubsig(journal={'event': 'no:such:event', 'iloc': 0})

    def test_entry_path_rejects_extra_filters(self, tmp_path):
        a = self.Solo(url=str(tmp_path), spec={'x': 1})
        a.build()
        entry_path = a.journal()['entry_path'].iloc[-1]
        with pytest.raises(ValueError, match='entry_path'):
            a.diffsubsig(journal={'entry_path': entry_path, 'event': 'build:end'})


class TestSplitTopLevelItems:

    def test_drops_the_separator(self):
        assert Datablock._split_top_level_items("a, b, c") == ['a', ' b', ' c']

    def test_respects_nesting_and_quotes(self):
        got = Datablock._split_top_level_items("a=[1, 2], b='p, q', c={'k': 1}")
        assert got == ["a=[1, 2]", " b='p, q'", " c={'k': 1}"]

    def test_parse_dictstr_keys_are_unquoted(self):
        assert Datablock._parse_dictstr("{'a': 1, 'b': '2'}") == {'a': '1', 'b': "'2'"}

    def test_parse_dictstr_rejects_a_non_dict(self):
        assert Datablock._parse_dictstr("(a=1)") == {}
        assert Datablock._parse_dictstr("{1, 2}") == {}


class TestTypedLeaves:
    """A norm is flat text, but the text records the type -- so recover it.

    Reading ``('15.0', "'15.0'")`` off a diff, you cannot tell a float from a
    string: both sides are strings, one of which happens to contain quotes.
    Evaluated, the same pair reads ``(15.0, '15.0')`` -- which says the two sides
    were rendered by different LEGACY_NORM settings, not that the value changed.
    """

    @dataclass
    class _C(Datablock.VAR):
        ori_extent: float = 15.0
        n: object = 128
        flag: bool = True
        nothing: object = None
        ratio: object = (0.75, 1.5)
        label: str = 'abc'

    def _blocks(self):
        class L(Datablock):
            LEGACY_NORM = True
            TOPICS = {'o': 'o.txt'}
            VAR = TestTypedLeaves._C
            def __build__(self): pass
        class M(Datablock):
            TOPICS = {'o': 'o.txt'}
            VAR = TestTypedLeaves._C
            def __build__(self): pass
        return M(url='/tmp/dbx-typed'), L(url='/tmp/dbx-typed')

    def test_types_survive_into_the_diff(self):
        modern, legacy = self._blocks()
        diff = modern.diffsubsig(legacy.subsignature())['spec']
        assert diff['n'] == (128, '128')
        assert diff['ori_extent'] == (15.0, '15.0')
        assert diff['nothing'] == (None, 'None')
        assert diff['ratio'] == ((0.75, 1.5), '(0.75, 1.5)')

    def test_self_side_types_are_the_real_python_types(self):
        modern, legacy = self._blocks()
        diff = modern.diffsubsig(legacy.subsignature())['spec']
        assert isinstance(diff['n'][0], int)
        assert isinstance(diff['ori_extent'][0], float)
        assert diff['nothing'][0] is None
        assert isinstance(diff['ratio'][0], tuple)
        # ... and the legacy side is genuinely a string, which is the finding.
        assert all(isinstance(diff[k][1], str)
                   for k in ('n', 'ori_extent', 'nothing', 'ratio'))

    def test_raw_gives_the_source_text_back(self):
        modern, legacy = self._blocks()
        diff = modern.diffsubsig(legacy.subsignature(), raw=True)['spec']
        assert diff['n'] == ('128', "'128'")
        assert diff['ori_extent'] == ('15.0', "'15.0'")

    def test_a_non_literal_leaf_is_left_alone(self):
        """Urls, object reprs and speclines are not Python literals."""
        assert Datablock._literal('abfss://c@a.net/x') == 'abfss://c@a.net/x'
        assert Datablock._literal('<Foo object at 0x7f00>') == '<Foo object at 0x7f00>'
        assert Datablock._literal('$pkg.mod.Cls(a=1)') == '$pkg.mod.Cls(a=1)'
        assert Datablock._literal('2026-07-19 22:34:17') == '2026-07-19 22:34:17'

    def test_literal_round_trips_the_scalars(self):
        assert Datablock._literal('128') == 128
        assert Datablock._literal('15.0') == 15.0
        assert Datablock._literal('None') is None
        assert Datablock._literal('True') is True
        assert Datablock._literal("'15.0'") == '15.0'
        assert Datablock._literal('(0.75, 1.5)') == (0.75, 1.5)
        assert Datablock._literal("['a', 'b']") == ['a', 'b']


class TestTypingNeverHidesADifference:
    """Detection compares the TEXT; only the reporting is evaluated."""

    @dataclass
    class _C(Datablock.VAR):
        n: object = 1

    def _cls(self):
        class M(Datablock):
            TOPICS = {'o': 'o.txt'}
            VAR = TestTypingNeverHidesADifference._C
            def __build__(self): pass
        return M

    def test_int_versus_float_is_still_reported(self):
        """``1 == 1.0`` in Python, so evaluating first would drop this entirely."""
        M = self._cls()
        diff = M(url='/tmp/dbx-typed', spec=dict(n=1)).diffsubsig(
            M(url='/tmp/dbx-typed', spec=dict(n=1.0)).subsignature())
        assert diff == {'spec': {'n': ('1', '1.0')}}

    def test_quoted_versus_bare_is_still_visible(self):
        """The two sides evaluate to the same str, so the bytes are reported."""
        class L(Datablock):
            LEGACY_NORM = True
            TOPICS = {'output': 'output.txt'}
            def __build__(self): pass

        norm_quoted = "(url='/tmp/dbx-typed', spec={})"
        diff = L(url='/tmp/dbx-typed').diffsubsig(norm_quoted, legacy=True)
        assert diff['url'] == ('/tmp/dbx-typed', "'/tmp/dbx-typed'")


class TestAbsentIsNotNone:
    """A missing key and a key whose value is None are different findings."""

    def test_absent_marker_is_distinct_from_a_none_value(self, tmp_path):
        @dataclass
        class CA(Datablock.VAR):
            kept: object = None
        @dataclass
        class CB(CA):
            added: object = None

        class A(Datablock):
            TOPICS = {'o': 'o.txt'}
            VAR = CA
            def __build__(self): pass
        class B(A):
            VAR = CB

        diff = B(url=str(tmp_path)).diffsubsig(A(url=str(tmp_path)).subsignature())
        self_val, other_val = diff['spec']['added']
        assert self_val is None, "a real None value"
        assert other_val is ABSENT, "the key did not exist on the other side"
        assert repr(other_val) == '<absent>'
        assert not other_val

    def test_absent_survives_raw(self, tmp_path):
        @dataclass
        class CA(Datablock.VAR):
            kept: int = 1
        @dataclass
        class CB(CA):
            added: int = 2

        class A(Datablock):
            TOPICS = {'o': 'o.txt'}
            VAR = CA
            def __build__(self): pass
        class B(A):
            VAR = CB

        diff = B(url=str(tmp_path)).diffsubsig(A(url=str(tmp_path)).subsignature(), raw=True)
        assert diff['spec']['added'] == ('2', ABSENT)


class TestReportShowsTypes:

    def test_report_distinguishes_a_float_from_its_string(self):
        @dataclass
        class C(Datablock.VAR):
            ori_extent: float = 15.0

        class M(Datablock):
            TOPICS = {'o': 'o.txt'}
            VAR = C
            def __build__(self): pass
        class L(M):
            LEGACY_NORM = True

        text = M(url='/tmp/dbx-typed').diffsubsig(
            L(url='/tmp/dbx-typed').subsignature(), report=True)
        assert 'self : 15.0' in text
        assert "other: '15.0'" in text
