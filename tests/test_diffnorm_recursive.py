"""Recursive, sparse ``diffnorm``: descend into nested blocks instead of blobbing.

A norm is flat text, so a nested block arrives inside the parent's spec dict as
one long string. The old comparison stopped at the top-level kwargs, so a single
changed leaf three levels down came back as::

    {'spec': (<2820-char string>, <2864-char string>)}

-- two near-identical blobs to eyeball. The real-world case that prompted this
was an ``IJEPAsaurUSStill`` whose only difference from a recorded build was that
``num_workers`` and ``prefetch_factor`` had since been removed from CONFIG.

Structuring has to be conservative in one specific way: values that merely
*look* parenthesised (a tuple like ``'(0.75, 1.5)'``) must stay leaves, or a
diff would report ``{}`` for them and hide a real change.
"""
import pytest
from dataclasses import dataclass

from dbx.datablocks import Datablock


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


class Leaf(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        label: str = 'leaf'
        ratio: object = None

    def __build__(self):
        pass


class Mid(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        leaf: object = None
        seed: int = 42

    def __build__(self):
        pass


class Top(Datablock):
    TOPICS = {'output': 'output.txt'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
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
        diff = b.diffnorm(a.norm())
        assert diff == {'spec': {'mid': {'spec': {'leaf': {'spec': {
            'label': ("'CHANGED'", "'leaf'")}}}}}}

    def test_only_the_differing_leaf_appears(self, tmp_path):
        """Sparse: siblings that match must not be carried along."""
        diff = _tree(tmp_path, seed=7).diffnorm(_tree(tmp_path).norm())
        assert diff == {'spec': {'mid': {'spec': {'seed': ('7', '42')}}}}
        assert 'leaf' not in diff['spec']['mid']['spec']

    def test_two_changes_at_different_depths(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED', epochs=99)
        diff = b.diffnorm(a.norm())
        assert diff['spec']['epochs'] == ('99', '10')
        assert diff['spec']['mid']['spec']['leaf']['spec']['label'] == (
            "'CHANGED'", "'leaf'")

    def test_flat_mode_keeps_the_whole_subtree(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED')
        flat = b.diffnorm(a.norm(), recursive=False)
        assert set(flat) == {'spec'}
        self_side, other_side = flat['spec']
        assert isinstance(self_side, str) and len(self_side) > 200

    def test_no_difference_is_empty(self, tmp_path):
        a = _tree(tmp_path)
        assert a.diffnorm(a.norm()) == {}

    def test_url_difference_stays_at_the_top(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path / 'elsewhere')
        diff = b.diffnorm(a.norm())
        assert 'url' in diff


class TestTupleValuesStayLeaves:
    """``'(0.75, 1.5)'`` is parenthesised but is NOT a nested block."""

    def test_tuple_change_is_reported(self, tmp_path):
        a = _tree(tmp_path, ratio=(0.75, 1.5))
        b = _tree(tmp_path, ratio=(0.5, 2.0))
        diff = b.diffnorm(a.norm())
        leafdiff = diff['spec']['mid']['spec']['leaf']['spec']
        assert 'ratio' in leafdiff
        self_val, other_val = leafdiff['ratio']
        assert '0.5' in self_val and '0.75' in other_val

    def test_structure_normval_leaves_a_tuple_alone(self):
        assert Datablock._structure_normval("'(0.75, 1.5)'") == "'(0.75, 1.5)'"
        assert Datablock._structure_normval('(0.75, 1.5)') == '(0.75, 1.5)'

    def test_structure_normval_expands_a_real_norm(self):
        got = Datablock._structure_normval("(url=/tmp/x, spec={'a': '1'})")
        assert got == {'url': '/tmp/x', 'spec': {'a': "'1'"}}

    def test_structure_normval_unwraps_one_layer_of_quoting(self):
        """A child norm is stored as a string VALUE in the parent's spec."""
        got = Datablock._structure_normval('"(url=/tmp/x, spec={})"')
        assert got == {'url': '/tmp/x', 'spec': '{}'}

    def test_structure_normval_passes_through_plain_scalars(self):
        for text in ("'42'", '42', 'None', "'a, b'"):
            assert Datablock._structure_normval(text) == text


class TestDeslash:

    def test_deslash_strips_escapes_from_reported_values(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED')
        raw = b.diffnorm(a.norm(), recursive=False)['spec'][0]
        assert '\\' in raw, "expected the flat form to carry escapes"
        clean = b.diffnorm(a.norm(), recursive=False, deslash=True)['spec'][0]
        assert '\\' not in clean

    def test_deslash_does_not_break_parsing(self, tmp_path):
        """Deslashing must happen on output, not before the parse.

        Stripping backslashes first would destroy the ``\\'`` escapes that mark
        where a nested norm's own quoted values begin and end, so the structure
        would collapse and the diff would come back at the wrong depth.
        """
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED')
        assert (b.diffnorm(a.norm(), deslash=True) ==
                b.diffnorm(a.norm(), deslash=False))


class TestReport:

    def test_report_is_one_path_per_difference(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='CHANGED', epochs=99)
        text = b.diffnorm(a.norm(), report=True)
        assert 'spec.epochs' in text
        assert 'spec.mid.spec.leaf.spec.label' in text
        assert text.count('self :') == 2

    def test_report_says_so_when_identical(self, tmp_path):
        a = _tree(tmp_path)
        assert a.diffnorm(a.norm(), report=True) == 'no differences'

    def test_report_truncates_long_values_but_the_dict_does_not(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='x' * 400)
        text = b.diffnorm(a.norm(), report=True, maxlen=40)
        assert '(+' in text and 'chars)' in text
        full = b.diffnorm(a.norm())['spec']['mid']['spec']['leaf']['spec']['label'][0]
        assert len(full) > 400

    def test_maxlen_none_disables_truncation(self, tmp_path):
        a = _tree(tmp_path)
        b = _tree(tmp_path, label='x' * 400)
        text = b.diffnorm(a.norm(), report=True, maxlen=None)
        assert 'chars)' not in text


class TestJournalFilters:
    """``dict(event='build:end', iloc=0)`` used to silently drop the event."""

    class Solo(Datablock):
        TOPICS = {'output': 'output.txt'}

        @dataclass
        class CONFIG(Datablock.CONFIG):
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
        assert b.diffnorm(journal={'event': 'build:end', 'iloc': 0}) == {
            'spec': {'x': ('2', '1')}}

    def test_a_filter_matching_nothing_is_visible(self, tmp_path):
        a = self.Solo(url=str(tmp_path), spec={'x': 1})
        a.build()
        with pytest.raises(Exception):
            a.diffnorm(journal={'event': 'no:such:event', 'iloc': 0})

    def test_entry_path_rejects_extra_filters(self, tmp_path):
        a = self.Solo(url=str(tmp_path), spec={'x': 1})
        a.build()
        entry_path = a.journal()['entry_path'].iloc[-1]
        with pytest.raises(ValueError, match='entry_path'):
            a.diffnorm(journal={'entry_path': entry_path, 'event': 'build:end'})


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
