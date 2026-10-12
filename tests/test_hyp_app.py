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

    def __point_datetime__(self, i):
        return "2026-10-12 15:58"


@pytest.fixture
def named():
    app = Named(engine=None, worker=Worker())
    app.settings = {"names": True, **app.settings}
    return app


def test_the_second_line_is_the_name_then_the_datetime(named):
    assert named.sublabel(0) == "gigapath clip · 2026-10-12 15:58"
    assert named.sublabel(1) == "2026-10-12 15:58"
    long = named.sublabel(2)
    assert long.endswith("… · 2026-10-12 15:58") and len(long) < 52


def test_names_and_datetimes_toggle_apart(named):
    named._toggle_setting_("names")
    assert named.sublabel(0) == "2026-10-12 15:58"
    named._toggle_setting_("datetimes")
    assert named.sublabel(0) is None
    named._toggle_setting_("names")
    assert named.sublabel(0) == "gigapath clip"


def test_names_is_a_setting_only_where_points_have_names(app, named):
    assert "names" not in app.settings
    assert [v for _, v in named._settings_items_()] == ["names", "datetimes"]


def test_the_tooltip_is_the_label_in_full(named):
    assert named.__point_tooltip__(0) == "gigapath_deep_feature_clip\ngigapath clip · 2026-10-12 15:58"
    assert named.__point_tooltip__(2) == f"gigapath_deep_feature_clip\n{Named.NAMES[2]} · 2026-10-12 15:58"
    named._toggle_setting_("names")
    named._toggle_setting_("datetimes")
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


def test_fold_collapses_each_run_of_deeper_lines():
    from hyp.app import TextViewer

    v = TextViewer("t", NESTED)
    assert v.folded() == NESTED.split("\n")
    v.fold = 1
    assert v.folded()[3:] == ["{", "    'lr': 0.1,", "    'data': {", "        … 5 lines", "    },", "    'name': 'clip'", "}"]
    v.fold = 2
    assert "            … 2 lines" in v.folded() and "        'table': 'T'," in v.folded()
    v.fold = None
    assert v.lines(200) == NESTED.split("\n")


def test_digit_keys_fold_and_zero_unfolds():
    import pygame
    from hyp.app import TextViewer

    v = TextViewer("t", NESTED)
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
    assert v.fold == 2
    v.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0))
    assert v.fold is None


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
