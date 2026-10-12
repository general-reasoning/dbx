"""The pygame UI: a navigable Poincare disk of embedded texts.

    left click     slide the clicked point (or the nearest datum) to the center
    left drag      move the plane rigidly, keeping the grabbed point under the cursor
    scroll         hyperbolic zoom about the center
    right click    on a datum: show its stored text; elsewhere: type a text to
                   embed, store and place
    hover          one-line summary, and geodesics to embedding-space neighbors
    Home           slide back to the corpus center and reset the zoom
"""

import argparse
import math
import os
import queue
import re
import shutil
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pygame

from . import poincare as P
from .engine import Engine
from .models import EMBED_MODEL, SUMMARY_MODEL, DeepSeekEmbedder, Ollama, OllamaEmbedder, Summarizer
from .store import Store

# Animation time per unit of hyperbolic distance travelled.
SECONDS_PER_UNIT = 0.35
# A click this close to the rim is clamped to it: anything further out is
# many units away, and the slide would take seconds.
MAX_CLICK_RADIUS = 0.995
RELAYOUT_SECONDS = 0.8
ZOOM_STEP = 1.12
ZOOM_RANGE = (0.25, 8.0)
HOVER_PX = 9
# A press that moves further than this before release is a drag, not a click.
DRAG_PX = 4
# Dragging is clamped inside this radius: near the rim one pixel spans a huge
# hyperbolic distance, and the plane would lurch.
DRAG_RADIUS = 0.97
LABEL_RADIUS = 0.72
MAX_LABELS = 48

BG = (17, 20, 24)
DISK = (26, 31, 39)
RIM = (90, 100, 120)
GRID = (44, 51, 63)
LABEL = (205, 212, 225)
TIP_BG = (245, 240, 225)
TIP_FG = (30, 30, 30)
EDGE = (250, 200, 90)
STATUS = (150, 160, 175)


def make_embedder(kind, model, ollama):
    if kind == "deepseek":
        return DeepSeekEmbedder(model) if model else DeepSeekEmbedder()
    return OllamaEmbedder(ollama, model or EMBED_MODEL)


def ease(s):
    return 0.5 - 0.5 * math.cos(math.pi * s)


class Worker:
    """One background thread running engine jobs in order, reporting through a queue."""

    def __init__(self):
        self.events = queue.Queue()
        self.jobs = queue.Queue()
        self.pending = 0
        threading.Thread(target=self._loop_, daemon=True).start()

    def submit(self, fn):
        self.pending += 1
        self.jobs.put(fn)

    def status(self, msg):
        self.events.put(("status", msg))

    def _loop_(self):
        while True:
            fn = self.jobs.get()
            try:
                fn(self.status)
            except Exception as e:  # shown in the status line; the UI keeps running
                self.events.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                self.events.put(("finished", None))


class View:
    """Data coordinates -> display coordinates: a disk isometry, then a hyperbolic zoom."""

    def __init__(self):
        self.M = P.Mobius()
        self.k = 1.0
        self.anim = None  # (start isometry, target point, start time, duration)

    def isometry(self, now):
        if self.anim is None:
            return self.M
        M0, p, t0, dur = self.anim
        s = 1.0 if dur <= 0 else min(1.0, (now - t0) / dur)
        # Translating by successive points of the geodesic from 0 to p slides
        # p along it to the center; the rest of the plane moves rigidly along.
        M = M0.then(P.Mobius.translation(P.toward(p, ease(s))))
        if s >= 1.0:
            self.M, self.anim = M, None
        return M

    def center_on(self, p, now):
        """Slide p (in view coordinates, before the zoom) to the center."""
        self.M = self.isometry(now)  # start from wherever a running slide got to
        r = abs(p)
        self.anim = None
        if r < 1e-9:
            return
        if r > MAX_CLICK_RADIUS:
            p = p / r * MAX_CLICK_RADIUS
        self.anim = (self.M, p, now, SECONDS_PER_UNIT * float(P.distance0(p)))

    def drag(self, a, b):
        """Move the plane rigidly so that a (view coordinates) lands on b."""
        self.M = self.M.then(P.Mobius.transvection(a, b))

    def zoom_by(self, factor):
        self.k = float(np.clip(self.k * factor, *ZOOM_RANGE))


def wrap(font, text, width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = word if not line else line + " " + word
            if font.size(trial)[0] <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


class TextBox:
    WIDTH = 440

    def __init__(self, pos):
        self.pos = pos
        self.text = ""
        pygame.key.start_text_input()

    def close(self):
        pygame.key.stop_text_input()

    def handle(self, ev):
        """Returns "submit", "cancel" or None."""
        if ev.type == pygame.TEXTINPUT:
            self.text += ev.text
        elif ev.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            if ev.key == pygame.K_ESCAPE:
                return "cancel"
            if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if mods & pygame.KMOD_SHIFT:
                    self.text += "\n"
                else:
                    return "submit"
            elif ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif ev.key == pygame.K_v and mods & (pygame.KMOD_META | pygame.KMOD_CTRL):
                try:
                    self.text += pygame.scrap.get_text() or ""
                except Exception:
                    pass
        return None

    def draw(self, surf, font):
        pad = 10
        lines = wrap(font, self.text + ("|" if time.time() % 1 < 0.5 else " "), self.WIDTH - 2 * pad)
        lh = font.get_linesize()
        hint = font.render("Enter: embed   Shift+Enter: newline   Esc: cancel", True, STATUS)
        h = 2 * pad + lh * max(1, len(lines)) + lh + 4
        x = min(self.pos[0], surf.get_width() - self.WIDTH - 8)
        y = min(self.pos[1], surf.get_height() - h - 8)
        rect = pygame.Rect(x, y, self.WIDTH, h)
        pygame.draw.rect(surf, (32, 38, 48), rect, border_radius=6)
        pygame.draw.rect(surf, EDGE, rect, 1, border_radius=6)
        for i, ln in enumerate(lines):
            surf.blit(font.render(ln, True, LABEL), (x + pad, y + pad + i * lh))
        surf.blit(hint, (x + pad, y + h - pad - lh + 2))
        pygame.key.set_text_input_rect(rect)


class Dropdown:
    """A button in a viewer's header that opens a list to pick one item from.

    *items* is a list of ``(text, value)``, or a function returning one, asked
    each time the list opens -- None from it while the items are still being
    read. Picking an item calls ``on_select(value)``.
    """

    ROWS = 16

    def __init__(self, label, items, on_select, empty="none", value=None, match=None):
        self.label = label
        self.items = items
        self.on_select = on_select
        self.empty = empty
        self.value = value  # the item shown now
        # Finds the item shown now among the items, while value is not set:
        # they may be read after the window opened on it.
        self.match = match
        self.open = False
        self.top = 0
        self.button = pygame.Rect(0, 0, 0, 0)
        self.list = pygame.Rect(0, 0, 0, 0)
        self.row_h = 18

    def current(self):
        return self.items() if callable(self.items) else self.items

    def select(self, value):
        self.value = value
        self.on_select(value)

    def step(self, d):
        """Select the item *d* places from the one shown, wrapping around."""
        values = [v for _, v in self.current() or []]
        if not values:
            return
        if self.value in values:
            i = values.index(self.value)
        else:
            i = next((j for j, v in enumerate(values) if self.match and self.match(v)), -1 if d > 0 else 0)
        self.select(values[(i + d) % len(values)])

    def handle(self, ev):
        """Whether *ev* was this dropdown's to handle."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.button.collidepoint(ev.pos):
                self.open, self.top = not self.open, 0
                return True
            if not self.open:
                return False
            self.open = False
            items = self.current() or []
            if self.list.collidepoint(ev.pos):
                i = self.top + (ev.pos[1] - self.list.y - 4) // self.row_h
                if 0 <= i < len(items):
                    self.select(items[i][1])
            return True
        if ev.type == pygame.MOUSEWHEEL and self.open:
            dy = getattr(ev, "precise_y", ev.y)
            n = len(self.current() or [])
            self.top = max(0, min(max(0, n - self.ROWS), self.top - (round(dy) or (1 if dy > 0 else -1))))
            return True
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE and self.open:
            self.open = False
            return True
        return False

    def draw_button(self, surf, font, x, y, color):
        img = font.render(self.label, True, color)
        h = img.get_height()
        self.button = pygame.Rect(x - 4, y - 2, img.get_width() + 22, h + 4)
        pygame.draw.rect(surf, (36, 42, 54), self.button, border_radius=4)
        surf.blit(img, (x, y))
        # The font has no ▾: draw one.
        tx, ty = x + img.get_width() + 6, y + h // 2 - 2
        pygame.draw.polygon(surf, color, [(tx, ty), (tx + 8, ty), (tx + 4, ty + 5)])
        return self.button.width

    def draw_list(self, surf, font):
        if not self.open:
            return
        items = self.current()
        texts = [t for t, _ in items] if items else ["loading…" if items is None else self.empty]
        self.row_h = font.get_linesize() + 2
        shown = texts[self.top : self.top + self.ROWS]
        width = max(self.button.width, max(font.size(t)[0] for t in shown) + 24)
        self.list = pygame.Rect(self.button.x, self.button.bottom + 2, width, self.row_h * len(shown) + 8)
        self.list.right = min(self.list.right, surf.get_width() - 4)
        pygame.draw.rect(surf, (30, 35, 45), self.list, border_radius=4)
        pygame.draw.rect(surf, EDGE, self.list, 1, border_radius=4)
        mx, my = pygame.mouse.get_pos()
        for i, t in enumerate(shown):
            row = pygame.Rect(self.list.x + 1, self.list.y + 4 + i * self.row_h, self.list.width - 2, self.row_h)
            if items and row.collidepoint(mx, my):
                pygame.draw.rect(surf, (52, 60, 76), row)
            surf.blit(font.render(t, True, LABEL if items else STATUS), (row.x + 10, row.y + 1))
        if items and len(items) > self.ROWS:
            more = font.render(f"{self.top + 1}–{self.top + len(shown)} of {len(items)}", True, STATUS)
            surf.blit(more, (self.list.right - more.get_width() - 8, self.list.bottom - self.row_h))


class TextViewer:
    """A read-only, scrollable view of one stored text, with optional header menus.

    Without *menus* the header is *title*; with them, it is their buttons --
    the first usually standing in for the title. *inset* shrinks the window,
    so that one opened over another shows the one beneath around its edges.
    """

    def __init__(self, title, text, menus=(), inset=0):
        self.title = title
        self.menus = list(menus)
        self.inset = inset
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.close = pygame.Rect(0, 0, 0, 0)
        self.set_text(text)

    def set_text(self, text):
        self.text = text.replace("\t", "    ")
        self.top = 0  # first visible line
        self._wrapped = (None, [])

    def lines(self, cols):
        if self._wrapped[0] != cols:
            out = []
            for para in self.text.split("\n"):
                # Continuation lines hang under the value of a "column  value"
                # row, or under the indentation of a yaml line. The value is
                # wrapped in the width beside its column, so that a long
                # path or hash fills that space rather than leaving the
                # column's name alone on a line.
                m = re.match(r"^\S+ {2,}|^ *", para)
                head = m[0][: cols // 2]
                body = para[len(m[0]) :]
                wrapped = textwrap.wrap(body, cols - len(head), replace_whitespace=False, drop_whitespace=False, break_on_hyphens=False) or [""]
                out += [(head if i == 0 else " " * len(head)) + w for i, w in enumerate(wrapped)]
            self._wrapped = (cols, out)
        return self._wrapped[1]

    def scroll(self, n):
        self.top = max(0, self.top + n)

    def handle(self, ev):
        """Returns "close" or None."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if any(m.handle(ev) for m in self.menus):
            return None
        if ev.type == pygame.MOUSEWHEEL:
            dy = getattr(ev, "precise_y", ev.y)
            self.scroll(-round(3 * dy) or (-1 if dy > 0 else 1))
        elif ev.type == pygame.KEYDOWN:
            page = max(1, self.rect.height // 18 - 4)
            step = {pygame.K_UP: -1, pygame.K_DOWN: 1, pygame.K_PAGEUP: -page, pygame.K_PAGEDOWN: page, pygame.K_SPACE: page}
            if ev.key == pygame.K_ESCAPE:
                return "close"
            if ev.key in (pygame.K_LEFT, pygame.K_RIGHT) and self.menus:
                self.menus[0].step(1 if ev.key == pygame.K_RIGHT else -1)
            elif ev.key in step:
                self.scroll(step[ev.key])
            elif ev.key == pygame.K_HOME:
                self.top = 0
            elif ev.key == pygame.K_c and pygame.key.get_mods() & (pygame.KMOD_META | pygame.KMOD_CTRL):
                try:
                    pygame.scrap.put_text(self.text)
                except Exception:
                    pass
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and not self.rect.collidepoint(ev.pos):
            return "close"
        return None

    def draw(self, surf, font, mono):
        w, h = surf.get_size()
        self.rect = pygame.Rect(0, 0, int(w * 0.8) - 2 * self.inset, int(h * 0.82) - 2 * self.inset)
        self.rect.center = (w // 2, (h - 24) // 2)
        x, y, rw, rh = self.rect
        pad = 14
        pygame.draw.rect(surf, (24, 28, 36), self.rect, border_radius=8)
        pygame.draw.rect(surf, EDGE, self.rect, 1, border_radius=8)
        self.close = pygame.Rect(x + rw - 34, y + 8, 24, 24)
        hot = self.close.collidepoint(pygame.mouse.get_pos())
        if hot:
            pygame.draw.rect(surf, (52, 60, 76), self.close, border_radius=4)
        c = self.close.inflate(-12, -12)
        for a, b in ((c.topleft, c.bottomright), (c.topright, c.bottomleft)):
            pygame.draw.line(surf, EDGE if hot else LABEL, a, b, 2)
        if self.menus:
            bx = x + pad
            for m in self.menus:
                bx += m.draw_button(surf, font, bx, y + pad, EDGE) + 12
        else:
            surf.blit(font.render(self.title, True, EDGE), (x + pad, y + pad))

        lh = mono.get_linesize()
        body_top = y + pad + font.get_linesize() + 10
        rows = max(1, (y + rh - pad - lh - 6 - body_top) // lh)
        lines = self.lines(max(20, (rw - 2 * pad) // mono.size("M")[0]))
        self.top = min(self.top, max(0, len(lines) - rows))
        for i, ln in enumerate(lines[self.top : self.top + rows]):
            surf.blit(mono.render(ln, True, LABEL), (x + pad, body_top + i * lh))
        where = f"lines {self.top + 1}–{min(len(lines), self.top + rows)} of {len(lines)}"
        cycle = "   Left / Right: previous / next" if self.menus else ""
        hint = font.render(f"{where}   scroll / Up Down / PgUp PgDn: move{cycle}   Cmd+C: copy   Esc: close", True, STATUS)
        surf.blit(hint, (x + pad, y + rh - pad - lh + 2))
        for m in self.menus:
            m.draw_list(surf, font)


def _grid_():
    """Polylines in data coordinates: diameters, and circles of hyperbolic radius 1..6."""
    d = np.linspace(-9, 9, 240)
    lines = [np.tanh(d / 2) * np.exp(1j * a) for a in np.linspace(0, np.pi, 12, endpoint=False)]
    ring = np.exp(1j * np.linspace(0, 2 * np.pi, 181))
    lines += [math.tanh(r / 2) * ring for r in range(1, 7)]
    return lines


class Hyp:
    """The disk, its navigation, and the hooks a specialization overrides.

    A specialization -- `dbxpipelines.HypDbxPipelines` -- changes where the
    corpus comes from (`__source__`, `__default_data__`), what opening a point
    does (`__open_point__`), and how a point is labeled (`__point_label__`,
    `__point_datetime__`, `__point_tooltip__`). The hooks are the only methods
    a specialization is meant to override; everything else is shared.
    """

    title = "hyp"

    #: Where corpora are kept unless --data says otherwise: one directory each.
    HOME = os.path.join(os.path.expanduser("~"), ".hyp")

    # 1. Hooks ------------------------------------------------------------------

    @classmethod
    def __default_data__(cls, args):
        return os.path.join(cls.HOME, "seeds")

    @classmethod
    def __source__(cls, args):
        """The corpus source the `Engine` syncs from, or None for the seed labels."""
        return None

    def __open_point__(self, i):
        """Right click on point *i*: show its stored text."""
        self.viewers = [TextViewer(self.snap.summaries[i], self.snap.texts[i])]

    def __point_label__(self, i):
        """The label drawn beside point *i*."""
        return self.snap.summaries[i]

    def __point_datetime__(self, i):
        """When point *i* is from, as text, or None: here, when it was added."""
        added = self.snap.points[i].get("added") if self.snap.points else None
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(added)) if added else None

    def __point_tooltip__(self, i):
        """The tooltip of point *i*: its summary, and its datetime when they are shown."""
        dt = self.__point_datetime__(i) if self.settings["datetimes"] else None
        return f"{self.snap.summaries[i]} · {dt}" if dt else self.snap.summaries[i]

    def push_viewer(self, viewer):
        """Open *viewer* over the ones open, inset so that they show around it."""
        viewer.inset = 28 * len(self.viewers)
        self.viewers.append(viewer)

    def run_async(self, fn, then):
        """Call *fn* on a thread of its own, then *then* with its result on the UI's.

        For reads a viewer waits on -- they must not queue behind the engine's
        jobs -- and for errors to reach the status line rather than the void.
        """

        def run():
            try:
                result = fn()
            except Exception as e:  # shown in the status line
                self.worker.events.put(("error", f"{type(e).__name__}: {e}"))
                return
            self.worker.events.put(("call", lambda: then(result)))

        threading.Thread(target=run, daemon=True).start()

    def __init__(self, engine, worker, args=None):
        self.args = args
        self.engine = engine
        self.worker = worker
        self.view = View()
        self.snap = None
        self.tween = None  # (from positions, to positions, start time)
        self.status = "starting"
        self.error = None
        self.box = None
        self.viewers = []  # open text windows, the last on top
        self.hover = None
        self.press = None  # where the left button went down, while it is down
        self.dragging = False
        self.last = None
        self.labels = {}
        self.settings = {"datetimes": True}
        self.settings_menu = Dropdown("settings", self._settings_items_, self._toggle_setting_)
        self.grid = _grid_()

        pygame.init()
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((1100, 1000), pygame.RESIZABLE)
        try:
            pygame.scrap.init()
        except Exception:
            pass
        pygame.key.stop_text_input()
        self.font = pygame.font.SysFont("helveticaneue,helvetica,arial", 15)
        self.small = pygame.font.SysFont("helveticaneue,helvetica,arial", 12)
        self.tiny = pygame.font.SysFont("helveticaneue,helvetica,arial", 10)
        self.mono = pygame.font.SysFont("menlo,monaco,couriernew", 13)
        self.clock = pygame.time.Clock()

    # -- geometry -----------------------------------------------------------

    def frame(self):
        w, h = self.screen.get_size()
        return w / 2, (h - 24) / 2, 0.47 * min(w, h - 24)

    def to_screen(self, z):
        cx, cy, R = self.frame()
        return cx + R * z.real, cy - R * z.imag

    def to_disk(self, pos):
        cx, cy, R = self.frame()
        return complex((pos[0] - cx) / R, -(pos[1] - cy) / R)

    def view_point(self, pos, limit):
        """Screen pixel -> view coordinates (before the zoom), clamped inside limit."""
        z = self.to_disk(pos)
        if abs(z) > limit:
            z = z / abs(z) * limit
        return complex(P.unzoom(np.array(z), self.view.k))

    def positions(self, now):
        if self.tween is None:
            return self.snap.positions
        a, b, t0 = self.tween
        s = min(1.0, (now - t0) / RELAYOUT_SECONDS)
        if s >= 1.0:
            self.tween = None
            return b
        return P.geodesic(a, b, ease(s))

    def display(self, data, M):
        return P.zoom(M(data), self.view.k)

    # -- events ---------------------------------------------------------------

    def on_snapshot(self, snap, focus, now):
        if self.snap is not None and len(self.snap.positions):
            old = self.positions(now)
            start = snap.positions.copy()
            start[: len(old)] = old
            self.tween = (start, snap.positions, now)
        else:
            # First layout: open on the corpus center.
            self.view.M = P.Mobius()
        self.snap = snap
        if focus is not None:
            M = self.view.isometry(now)
            self.view.center_on(M(snap.positions[focus]), now)

    def submit(self, text):
        engine = self.engine

        def job(progress):
            i = engine.add(text, progress)
            self.worker.events.put(("snapshot", (engine.snapshot(), i)))
            progress(f"added: {engine.store.points[i]['summary']}")

        self.worker.submit(job)

    def handle(self, ev, now):
        if self.settings_menu.handle(ev):
            return
        if self.viewers:
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN) and self.viewers[-1].handle(ev) == "close":
                self.viewers.pop()
            return
        if self.box is not None and ev.type in (pygame.TEXTINPUT, pygame.KEYDOWN):
            action = self.box.handle(ev)
            if action == "submit" and self.box.text.strip():
                self.submit(self.box.text)
            if action in ("submit", "cancel"):
                self.box.close()
                self.box = None
            return
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.box is not None:
                self.box.close()
                self.box = None
                return
            if abs(self.to_disk(ev.pos)) < 1:
                self.press = ev.pos
                self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.press is not None:
            if not self.dragging:
                dx, dy = ev.pos[0] - self.press[0], ev.pos[1] - self.press[1]
                if dx * dx + dy * dy < DRAG_PX**2:
                    return
                self.dragging = True
                self.last = self.press
                self.view.M = self.view.isometry(now)  # a drag interrupts a slide
                self.view.anim = None
            self.view.drag(self.view_point(self.last, DRAG_RADIUS), self.view_point(ev.pos, DRAG_RADIUS))
            self.last = ev.pos
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.press is not None:
            if not self.dragging:
                z = self.to_disk(self.press)
                if self.hover is not None:
                    z = self.display(self.positions(now)[self.hover], self.view.isometry(now))
                self.view.center_on(complex(P.unzoom(np.array(z), self.view.k)), now)
            self.press = None
            self.dragging = False
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            if self.box is not None:
                self.box.close()
                self.box = None
            if self.hover is not None:
                self.__open_point__(self.hover)
            else:
                self.box = TextBox(ev.pos)
        elif ev.type == pygame.MOUSEWHEEL:
            self.view.zoom_by(ZOOM_STEP ** getattr(ev, "precise_y", ev.y))
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_HOME:
            self.view.k = 1.0
            self.view.center_on(self.view.isometry(now)(0j), now)

    def drain(self, now):
        """Handle what the worker and async reads sent; whether there was any."""
        got = False
        while True:
            try:
                kind, payload = self.worker.events.get_nowait()
            except queue.Empty:
                return got
            got = True
            if kind == "status":
                self.status, self.error = payload, None
            elif kind == "error":
                self.error = payload
            elif kind == "snapshot":
                self.on_snapshot(*payload, now)
            elif kind == "call":
                payload()
            elif kind == "finished":
                self.worker.pending -= 1

    # -- drawing --------------------------------------------------------------

    def label(self, i, text, when=None):
        """Point *i*'s label: *text*, and below it, dimmer, *when*."""
        key = (i, text, when)
        if key not in self.labels:
            short = text if len(text) <= 32 else text[:31] + "…"
            lines = [self.small.render(short, True, LABEL)]
            if when:
                lines.append(self.tiny.render(when, True, STATUS))
            img = pygame.Surface((max(l.get_width() for l in lines), sum(l.get_height() for l in lines)), pygame.SRCALPHA)
            y = 0
            for line in lines:
                img.blit(line, (0, y))
                y += line.get_height()
            self.labels[key] = img
        return self.labels[key]

    def _settings_items_(self):
        return [(f"[{'x' if on else ' '}]  {name}", name) for name, on in self.settings.items()]

    def _toggle_setting_(self, name):
        self.settings[name] = not self.settings[name]

    def polyline(self, data, M, color, width=1):
        pts = np.asarray(self.to_screen(self.display(data, M))).T
        if len(pts) > 1:
            pygame.draw.aalines(self.screen, color, False, pts) if width == 1 else pygame.draw.lines(self.screen, color, False, pts, width)

    def draw(self, now):
        surf = self.screen
        surf.fill(BG)
        cx, cy, R = self.frame()
        pygame.draw.circle(surf, DISK, (cx, cy), R)
        M = self.view.isometry(now)
        for line in self.grid:
            self.polyline(line, M, GRID)
        pygame.draw.circle(surf, RIM, (cx, cy), R, 2)

        self.hover = None
        if self.snap is not None and len(self.snap.positions):
            data = self.positions(now)
            z = self.display(data, M)
            sx, sy = self.to_screen(z)
            r2 = np.abs(z) ** 2

            mx, my = pygame.mouse.get_pos()
            if self.box is None and not self.viewers:
                d2 = (sx - mx) ** 2 + (sy - my) ** 2
                i = int(np.argmin(d2))
                if d2[i] <= HOVER_PX**2:
                    self.hover = i

            if self.hover is not None:
                t = np.linspace(0, 1, 40)
                for j in self.snap.neighbors[self.hover]:
                    self.polyline(P.geodesic(data[self.hover], data[j], t), M, EDGE)

            # Far points first, so the near ones draw over them; size follows
            # the conformal factor, so distance reads as shrinkage.
            for i in np.argsort(-r2):
                ang = math.degrees(math.atan2(data[i].imag, data[i].real)) % 360
                col = pygame.Color(0)
                col.hsva = (ang, 45 + 40 * min(1.0, abs(data[i]) * 1.5), 95, 100)
                rad = 1.5 + 6.0 * (1.0 - r2[i])
                pygame.draw.circle(surf, col, (sx[i], sy[i]), rad)
                if i == self.hover:
                    pygame.draw.circle(surf, EDGE, (sx[i], sy[i]), rad + 3, 2)

            placed = []
            for i in np.argsort(r2)[:MAX_LABELS]:
                if r2[i] > LABEL_RADIUS**2:
                    break
                img = self.label(i, self.__point_label__(i), self.__point_datetime__(i) if self.settings["datetimes"] else None)
                rect = img.get_rect(midleft=(sx[i] + 8, sy[i]))
                if rect.collidelist(placed) < 0:
                    img.set_alpha(int(255 * (1.0 - r2[i] / LABEL_RADIUS**2) ** 0.5))
                    surf.blit(img, rect)
                    placed.append(rect.inflate(6, 2))

            if self.hover is not None:
                self.tooltip(self.__point_tooltip__(self.hover), (mx, my))

        if self.box is not None:
            self.box.draw(surf, self.font)
        w = surf.get_width()
        self.settings_menu.draw_button(surf, self.small, w - self.small.size("settings")[0] - 30, 10, STATUS)
        for viewer in self.viewers:
            viewer.draw(surf, self.font, self.mono)
        self.settings_menu.draw_list(surf, self.small)
        self.statusline()
        pygame.display.flip()

    def tooltip(self, text, pos):
        img = self.font.render(text, True, TIP_FG)
        rect = img.get_rect(topleft=(pos[0] + 14, pos[1] + 14)).inflate(16, 10)
        rect.clamp_ip(self.screen.get_rect())
        pygame.draw.rect(self.screen, TIP_BG, rect, border_radius=4)
        self.screen.blit(img, img.get_rect(center=rect.center))

    def statusline(self):
        w, h = self.screen.get_size()
        busy = "⋯ " if self.worker.pending else ""
        n = len(self.snap.texts) if self.snap else 0
        msg = f"error: {self.error}" if self.error else f"{busy}{self.status}"
        left = self.small.render(f"{n} texts · zoom {self.view.k:.2f} · {msg}", True, (240, 120, 110) if self.error else STATUS)
        right = self.small.render("click: center · drag: move · scroll: zoom · right-click: open / add · Home: reset", True, STATUS)
        self.screen.blit(left, (10, h - 20))
        self.screen.blit(right, (w - right.get_width() - 10, h - 20))

    def raise_window(self):
        # Launched from a terminal, macOS leaves the window behind it.
        try:
            pygame.Window.from_display_module().focus()
        except Exception:
            pass

    def run(self):
        self.raise_window()
        while True:
            now = time.monotonic()
            events = pygame.event.get()
            for ev in events:
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return
                self.handle(ev, now)
            news = self.drain(now)
            # Redraw only while something can change on screen; an idle frame
            # at 60 fps costs half a core for nothing.
            moving = self.view.anim is not None or self.tween is not None
            if events or news or moving or self.worker.pending or self.box is not None:
                self.draw(now)
            self.clock.tick(60 if moving else 30)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hyp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="corpus directory (default: one per source under ~/.hyp)")
    ap.add_argument("--dbx.pipejournal", dest="pipejournal", action="store_true", help="the pipelines of a dbx datalake, from dbx.pipejournal(), one point each")
    ap.add_argument("--dbx.datalake", dest="datalake", metavar="URL", help="the datalake dbx.pipejournal() reads, e.g. ssh://host:/path/to/datalake, host as named in ~/.ssh/config (default: dbx's)")
    ap.add_argument("--n-workers", type=int, default=16, help="threads reading a remote source, e.g. the dbx journal and its side files (default: 16)")
    ap.add_argument("--embedder", choices=["ollama", "deepseek"], default="ollama", help="ollama: an Ollama embedding model; deepseek: pooled DeepSeek hidden states via transformers")
    ap.add_argument("--embed-model", default=None, help=f"default: {EMBED_MODEL} for ollama")
    ap.add_argument("--summary-model", default=SUMMARY_MODEL)
    ap.add_argument("--reset", action="store_true", help="discard the stored corpus and start over")
    ap.add_argument("--seed", type=int, default=0, help="UMAP random state")
    args = ap.parse_args(argv)

    if args.datalake and not args.pipejournal:
        ap.error("--dbx.datalake needs --dbx.pipejournal")
    cls = Hyp
    if args.pipejournal:
        from .dbxpipelines import HypDbxPipelines as cls
    if args.data is None:
        args.data = cls.__default_data__(args)
    if args.reset and Path(args.data).exists():
        shutil.rmtree(args.data)
    ollama = Ollama()
    engine = Engine(
        Store(args.data),
        make_embedder(args.embedder, args.embed_model, ollama),
        Summarizer(ollama, args.summary_model),
        seed=args.seed,
        source=cls.__source__(args),
    )
    worker = Worker()
    app = cls(engine, worker, args)

    def start(progress):
        ollama.ensure_server(progress)
        engine.start(progress)
        worker.events.put(("snapshot", (engine.snapshot(), None)))
        progress("ready")

    worker.submit(start)
    app.run()


if __name__ == "__main__":
    main()
