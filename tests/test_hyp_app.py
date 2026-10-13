"""The hyp window's busy spinner: on while data are loaded or prepared, off once they are."""

import os
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from hyp.app import Hyp, Worker


@pytest.fixture
def app():
    return Hyp(engine=None, worker=Worker())


def settle(app, until, timeout=5.0):
    """Drain the app's events until *until()* holds, as its loop would."""
    deadline = time.monotonic() + timeout
    while not until():
        assert time.monotonic() < deadline, "timed out"
        app.drain(time.monotonic())
        time.sleep(0.01)


def test_idle_is_not_busy(app):
    assert not app.busy()


def test_a_read_is_busy_until_its_result_is_handed_back(app):
    release, got = threading.Event(), []
    app.run_async(lambda: release.wait() and 42, got.append)
    assert app.busy()
    app.drain(time.monotonic())
    assert app.busy() and not got
    release.set()
    settle(app, lambda: not app.busy())
    assert got == [42]


def test_a_failed_read_stops_being_busy(app):
    def fail():
        raise OSError("gone")

    app.run_async(fail, lambda _: None)
    settle(app, lambda: not app.busy())
    assert app.error == "OSError: gone"


def test_an_engine_job_is_busy(app):
    release = threading.Event()
    app.worker.submit(lambda progress: release.wait())
    assert app.busy()
    release.set()
    settle(app, lambda: not app.busy())


def test_the_busy_frame_draws(app):
    app.worker.pending = 1
    app.draw(time.monotonic())
    app.worker.pending = 0
    app.draw(time.monotonic())


class Named(Hyp):
    """A Hyp whose points have names and datetimes, as HypDbxPipelines' do."""

    NAMES = {0: "gigapath clip", 1: None, 2: "a name far too long to leave the datetime any room at all"}

    def __point_label__(self, i):
        return "gigapath_deep_feature_clip"

    def __point_name__(self, i):
        return self.NAMES[i]

    def __point_version__(self, i):
        return {0: "3"}.get(i)

    def __point_datetime__(self, i):
        return "2026-10-12 15:58"


@pytest.fixture
def named():
    app = Named(engine=None, worker=Worker())
    app.settings = {"names": True, "versions": True, **app.settings}
    return app


def test_the_second_line_is_the_name_then_the_datetime(named):
    assert named.sublabel(0) == "gigapath clip · v3 · 2026-10-12 15:58"
    assert named.sublabel(1) == "2026-10-12 15:58"
    long = named.sublabel(2)
    assert long.endswith("… · 2026-10-12 15:58") and len(long) < 52


def test_names_versions_and_datetimes_toggle_apart(named):
    named._toggle_setting_("names")
    assert named.sublabel(0) == "v3 · 2026-10-12 15:58"
    named._toggle_setting_("datetimes")
    assert named.sublabel(0) == "v3"
    named._toggle_setting_("versions")
    assert named.sublabel(0) is None
    named._toggle_setting_("names")
    assert named.sublabel(0) == "gigapath clip"


def test_names_is_a_setting_only_where_points_have_names(app, named):
    assert "names" not in app.settings
    assert [v for _, v in named._settings_items_()] == ["names", "versions", "datetimes", "line numbers"]


def test_the_tooltip_is_the_label_in_full(named):
    assert named.__point_tooltip__(0) == "gigapath_deep_feature_clip\ngigapath clip · v3 · 2026-10-12 15:58"
    assert named.__point_tooltip__(2) == f"gigapath_deep_feature_clip\n{Named.NAMES[2]} · 2026-10-12 15:58"
    for setting in ("names", "versions", "datetimes"):
        named._toggle_setting_(setting)
    assert named.__point_tooltip__(0) == "gigapath_deep_feature_clip"


def test_a_two_line_tooltip_draws(named):
    named.tooltip(named.__point_tooltip__(0), (100, 100))


NESTED = """anchor   pkg.TrainPipeline

── spec ──
{
    'lr': 0.1,
    'data': {
        'table': 'T',
        'cols': {
            'a': 1,
            'b': 2
        }
    },
    'name': 'clip'
}"""


def test_a_double_click_off_the_disk_resets_the_view(app):
    import pygame

    w, h = app.screen.get_size()
    off = (5, 5)  # a corner: off the disk
    assert abs(app.to_disk(off)) > 1
    app.view.k = 3.0

    def click(pos, now):
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos), now)

    click(off, 10.0)
    assert app.view.k == 3.0, "one click is not a reset"
    click(off, 10.2)
    assert app.view.k == 1.0
    app.view.k = 3.0
    click(off, 20.0)
    click(off, 21.0)
    assert app.view.k == 3.0, "too slow to be a double click"
    click((w - 5, h - 40), 21.2)
    assert app.view.k == 3.0, "too far apart to be a double click"


def test_a_double_click_on_the_disk_does_not_reset(app):
    import pygame

    w, h = app.screen.get_size()
    on = (w // 2, (h - 24) // 2)
    app.view.k = 3.0
    for t in (10.0, 10.2):
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=on), t)
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=on), t)
    assert app.view.k == 3.0


@pytest.fixture
def shown(app):
    """A viewer over the app, drawn once, so that it knows where its text is."""
    from hyp.app import TextViewer

    v = TextViewer("t", "alpha beta\ngamma delta\nepsilon")
    app.push_viewer(v)
    app.draw(0)
    return v


def at(v, line, col):
    """The screen position of *line*, *col* of the text *v* shows."""
    bx, top, lh, cw, _, _ = v._body
    return (bx + col * cw, top + line * lh + lh // 2)


def mouse(v, kind, pos):
    import pygame

    return v.handle(pygame.event.Event(kind, button=1, pos=pos, rel=(0, 0), buttons=(1, 0, 0)))


def test_dragging_selects_the_text_under_it(shown):
    import pygame

    mouse(shown, pygame.MOUSEBUTTONDOWN, at(shown, 0, 6))
    mouse(shown, pygame.MOUSEMOTION, at(shown, 1, 5))
    mouse(shown, pygame.MOUSEBUTTONUP, at(shown, 1, 5))
    assert shown.selected_text() == "beta\ngamma"


def test_a_click_clears_the_selection(shown):
    import pygame

    mouse(shown, pygame.MOUSEBUTTONDOWN, at(shown, 0, 0))
    mouse(shown, pygame.MOUSEMOTION, at(shown, 0, 5))
    mouse(shown, pygame.MOUSEBUTTONUP, at(shown, 0, 5))
    assert shown.selected_text() == "alpha"
    mouse(shown, pygame.MOUSEBUTTONDOWN, at(shown, 2, 1))
    mouse(shown, pygame.MOUSEBUTTONUP, at(shown, 2, 1))
    assert shown.selection() is None


def test_copy_takes_the_selection_or_everything(shown, monkeypatch):
    import pygame
    from hyp import app as app_mod

    copied = []
    monkeypatch.setattr(app_mod, "_copy_", lambda text: copied.append(text) or True)
    shown.copy(now=0)
    assert copied == ["alpha beta\ngamma delta\nepsilon"] and "all 30 characters" in shown.note[0]
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_META)
    shown.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
    assert shown.selected_text() == shown.text
    mouse(shown, pygame.MOUSEBUTTONDOWN, at(shown, 1, 0))
    mouse(shown, pygame.MOUSEMOTION, at(shown, 1, 5))
    mouse(shown, pygame.MOUSEBUTTONUP, at(shown, 1, 5))
    shown.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
    assert copied[-1] == "gamma" and "5 characters" in shown.note[0]


def test_escape_clears_a_selection_before_it_closes(shown, monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_META)
    shown.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    assert shown.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)) is None
    assert shown.selection() is None
    assert shown.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)) == "close"


class Clicked(Hyp):
    """A Hyp recording what its click hooks were asked."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def __click_point__(self, i, now):
        self.calls.append(("click", i))

    def __context_point__(self, i, pos):
        self.calls.append(("context", i, pos))


def press_release(app, pos, button=1):
    import pygame

    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos), 1.0)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=button, pos=pos), 1.0)


def test_clicks_on_a_point_go_to_its_hooks():
    app = Clicked(engine=None, worker=Worker())
    w, h = app.screen.get_size()
    on = (w // 2, (h - 24) // 2)
    app.hover = 7
    press_release(app, on)
    app.hover = 7
    press_release(app, on, button=3)
    assert app.calls == [("click", 7), ("context", 7, on)]


def test_a_popup_takes_the_next_click(app):
    import pygame

    picked = []
    app.open_popup([("first", 1), ("second", 2)], picked.append, (100, 100))
    app.draw(0)
    row = app.popup.list.y + 4 + app.popup.row_h + app.popup.row_h // 2
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(app.popup.list.x + 10, row)), 1.0)
    assert picked == [2] and app.popup is None

    app.open_popup([("first", 1)], picked.append, (100, 100))
    app.draw(0)
    k = app.view.k
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(600, 600)), 1.0)
    assert picked == [2] and app.popup is None and app.press is None, "a click elsewhere only closes it"
    assert app.view.k == k

    app.open_popup(None, picked.append, (100, 100), empty="none")
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE), 1.0)
    assert app.popup is None


def test_a_popup_near_the_bottom_opens_upwards(app):
    w, h = app.screen.get_size()
    app.open_popup([(f"item {i}", i) for i in range(10)], lambda v: None, (100, h - 20))
    app.draw(0)
    assert app.popup.list.bottom <= h - 20


def test_a_line_with_deeper_lines_after_it_folds():
    from hyp.app import TextViewer

    v = TextViewer("t", NESTED)
    folds = [i for i in range(len(v.paras)) if v.foldable(i)]
    assert [v.paras[i] for i in folds] == ["{", "    'data': {", "        'cols': {"]
    v.toggle(folds[1])  # 'data'
    assert v.folded()[3:] == ["{", "    'lr': 0.1,", "    'data': {  … 5 lines", "    },", "    'name': 'clip'", "}"]
    v.toggle(folds[1])
    assert v.folded() == NESTED.split("\n")


def test_the_fold_menu_and_the_keys_fold_a_level_at_once():
    import pygame
    from hyp.app import TextViewer

    v = TextViewer("t", NESTED)
    v.fold_menu.select(1)
    assert [v.paras[i] for i in sorted(v.collapsed)] == ["    'data': {", "        'cols': {"]
    assert "    'data': {  … 5 lines" in v.folded()
    v.fold_menu.select(None)
    assert not v.collapsed
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
    assert [v.paras[i] for i in sorted(v.collapsed)] == ["        'cols': {"]
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0))
    assert not v.collapsed


def test_clicking_a_triangle_toggles_its_fold(app):
    import pygame
    from hyp.app import TextViewer

    v = TextViewer("t", NESTED)
    app.push_viewer(v)
    app.draw(0)
    tx, top, lh, cw, rows, gx = v._body
    data = v.paras.index("    'data': {")
    row = next(r for r, (i, first, _) in enumerate(v._wrapped[3]) if i == data and first)
    triangle = (tx - 2 * cw + cw // 2, top + (row - v.top) * lh + lh // 2)
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=triangle))
    assert v.collapsed == {data} and v.selection() is None
    app.draw(0)
    v.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=triangle))
    assert v.collapsed == set()


def test_line_numbers_are_of_the_texts_own_lines(app):
    from hyp.app import TextViewer

    v = TextViewer("t", NESTED)
    v.toggle(v.paras.index("    'data': {"))
    app.push_viewer(v)
    app.settings["line numbers"] = True
    app.draw(0)
    numbered = [i + 1 for i, first, _ in v._wrapped[3] if first]
    assert numbered[:7] == [1, 2, 3, 4, 5, 6, 12], "the folded lines' numbers are skipped, not reused"
    tx_numbered = v._body[0]
    app.settings["line numbers"] = False
    app.draw(0)
    assert v._body[0] < tx_numbered, "without numbers, the text starts further left"


def test_the_reload_button_reloads(app):
    import pygame

    jobs, hooked, snapshots = [], [], []
    app.worker.submit = jobs.append
    app.__reload__ = lambda: hooked.append(True)
    app.engine = SimpleNamespace(reload=lambda progress: None, snapshot=lambda: "SNAP")
    app.draw(0)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=app.reload_button.button.center), 1.0)
    assert hooked == [True] and len(jobs) == 1

    app.worker.events.put = lambda ev: snapshots.append(ev)
    jobs[0](lambda msg: None)
    assert snapshots[0] == ("snapshot", ("SNAP", None))


from types import SimpleNamespace  # noqa: E402
