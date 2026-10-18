"""Datablock.walk_tree, and what walks with it: update_tree and touch_tree."""
import pytest

from dbx.databits import quote
from dbx.journals import CONSTRUCTED_EVENTS
from test_build_tree_deep import Leaf, Mid, Root, _make_tree, _make_tree_with_exemptions


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv('DBX_DIRTY_REPO_OK', '1')


def _labels(walk):
    return [(path, block.var.label.strip("'")) for path, block in walk]


def test_a_tree_is_walked_bottom_up_with_its_paths(tmp_path):
    root, mid, leaf = _make_tree(str(tmp_path))
    assert _labels(root.walk_tree()) == [(('child', 'dep'), 'L'), (('child',), 'M'), ((), 'R')]
    assert _labels(root.walk_tree(exclude_self=True)) == [(('child', 'dep'), 'L'), (('child',), 'M')]


def test_a_skipped_subtree_is_not_walked_and_is_reported(tmp_path):
    root = _make_tree_with_exemptions(str(tmp_path))[0]
    skipped = []
    assert _labels(root.walk_tree(skip_callback=skipped.append)) == [((), 'R')]
    assert skipped == [('child',)]
    assert len(list(root.walk_tree(None))) == 3, "skips=None: every block"


def test_a_block_reached_twice_is_walked_once(tmp_path):
    """Root -> N -> M -> L, and N's own label aside, M reached again is not walked again."""
    url = str(tmp_path)
    leaf = Leaf(datalake=url, spec=dict(label="'L'"))
    mid = Mid(datalake=url, spec=dict(label="'M'", dep=quote(leaf)))
    upper = Mid(datalake=url, spec=dict(label="'N'", dep=quote(mid)))
    root = Root(datalake=url, spec=dict(label="'R'", child=quote(upper)))
    assert [label for _, label in _labels(root.walk_tree())] == ['L', 'M', 'N', 'R']
    seen = set()
    walk = root.walk_tree()
    for _, block in walk:
        assert block.anchorkey not in seen
        seen.add(block.anchorkey)


@pytest.mark.parametrize('method, event', [('update_tree', 'update'), ('touch_tree', 'touch')])
def test_update_and_touch_tree_journal_every_block_under_one_tree(tmp_path, method, event):
    root, mid, leaf = _make_tree(str(tmp_path))
    assert getattr(root, method)('re-recorded', inline=True) is root
    for block in (leaf, mid, root):
        entries = block.buildjournal(event=event)
        assert len(entries) == 1 and entries.iloc[0]['note'] == 're-recorded'
        assert entries.iloc[0]['tree'] == root.tree, "found together, as one run's"
    assert f'^{event}$' in CONSTRUCTED_EVENTS


def _touched(block):
    """Whether *block* has a touch entry -- a block with no journal at all has none."""
    try:
        df = block.buildjournal()
    except FileNotFoundError:
        return False
    return bool(len(df)) and bool((df['event'] == 'touch').any())


def test_touch_tree_skips_what_build_tree_skips(tmp_path):
    root, mid, leaf = _make_tree_with_exemptions(str(tmp_path))
    root.touch_tree()
    assert _touched(root) and not _touched(mid) and not _touched(leaf)


def test_touch_tree_can_leave_the_top_out(tmp_path):
    root, mid, leaf = _make_tree(str(tmp_path))
    root.touch_tree(exclude_self=True)
    assert _touched(mid) and _touched(leaf) and not _touched(root)
