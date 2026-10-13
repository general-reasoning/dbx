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
import sys
import textwrap
import threading
import time
import traceback

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
# The longest slide to a point picked from the points menu, however far it is.
GOTO_SECONDS = 1.5
# The most points the points menu lists, and the longest item it shows.
POINTS_MENU_LIMIT = 2000
POINTS_MENU_CHARS = 140
RELAYOUT_SECONDS = 0.8
ZOOM_STEP = 1.12
ZOOM_RANGE = (0.25, 8.0)
HOVER_PX = 9
# A press that moves further than this before release is a drag, not a click.
DRAG_PX = 4
# Two left clicks off the disk this close, in time and place, reset the view.
DOUBLE_CLICK_S = 0.4
DOUBLE_CLICK_PX = 6
# Seconds an error stays on the status line -- its traceback stays in the terminal.
ERROR_S = 15
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
DIMMED = (85, 92, 105)  # a menu shown but not yet usable
DISK_SEE_THROUGH = 150  # the disk's alpha, out of 255, over layers behind it


def make_embedder(kind, model, ollama):
    if kind == "deepseek":
        return DeepSeekEmbedder(model) if model else DeepSeekEmbedder()
    return OllamaEmbedder(ollama, model or EMBED_MODEL)


def ease(s):
    return 0.5 - 0.5 * math.cos(math.pi * s)


def _double_click_(last, pos, now):
    """``(whether, last')``: whether a click at *pos* is the second of a double click after *last* -- ``(pos, time)`` or None -- and what to keep for the next."""
    if last is not None:
        (x, y), t = last
        if now - t <= DOUBLE_CLICK_S and (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= DOUBLE_CLICK_PX**2:
            return True, None  # a third click starts another
    return False, (pos, now)


def _failed_(e) -> str:
    """*e*, raised on a background thread: its traceback to the terminal, whole; a line for the status line, returned."""
    traceback.print_exception(e, file=sys.stderr)
    sys.stderr.flush()
    return f"{type(e).__name__}: {e}"


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
                self.events.put(("error", _failed_(e)))
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

    def center_on(self, p, now, exact=False):
        """Slide p (in view coordinates, before the zoom) to the center.

        A click near the rim goes only so far toward it; *exact* goes all the
        way, however far, in at most GOTO_SECONDS.
        """
        self.M = self.isometry(now)  # start from wherever a running slide got to
        r = abs(p)
        self.anim = None
        if r < 1e-9:
            return
        if r > MAX_CLICK_RADIUS and not exact:
            p = p / r * MAX_CLICK_RADIUS
        seconds = SECONDS_PER_UNIT * float(P.distance0(p))
        self.anim = (self.M, p, now, min(seconds, GOTO_SECONDS) if exact else seconds)

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
    read. Picking an item calls ``on_select(value)``. Shown only while
    *visible()*, and usable only while *enabled()*: shown dimmed, and opening
    on no click, until then.
    """

    ROWS = 16

    def __init__(self, label, items, on_select, empty="none", value=None, match=None, visible=lambda: True,
                 enabled=lambda: True):
        self.label = label
        self.visible = visible
        self.enabled = enabled
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
        if not self.visible():
            self.open = False
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.button.collidepoint(ev.pos):
                self.open, self.top = (not self.open and self.enabled()), 0
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

    def width(self, font):
        """The button's width, as drawn with *font*."""
        return font.size(self.label)[0] + 22

    def draw_button(self, surf, font, x, y, color, max_width=None):
        """Draw the button at (*x*, *y*), its label shortened with … to fit *max_width*; returns its width."""
        label = self.label
        if not self.enabled():
            color = DIMMED
        if max_width is not None:
            while len(label) > 1 and font.size(label)[0] + 26 > max_width:
                label = label[:-2] + "…"
        img = font.render(label, True, color)
        h = img.get_height()
        self.button = pygame.Rect(x - 4, y - 2, img.get_width() + 22, h + 4)
        pygame.draw.rect(surf, (36, 42, 54), self.button, border_radius=4)
        surf.blit(img, (x, y))
        # The font has no ▾: draw one.
        tx, ty = x + img.get_width() + 6, y + h // 2 - 2
        pygame.draw.polygon(surf, color, [(tx, ty), (tx + 8, ty), (tx + 4, ty + 5)])
        return self.button.width

    def draw_list(self, surf, font):
        if not self.open or not self.visible() or not self.enabled():
            return
        items = self.current()
        texts = [t for t, _ in items] if items else ["loading…" if items is None else self.empty]
        self.row_h = font.get_linesize() + 2
        shown = texts[self.top : self.top + self.ROWS]
        width = max(self.button.width, max(font.size(t)[0] for t in shown) + 24)
        self.list = pygame.Rect(self.button.x, self.button.bottom + 2, width, self.row_h * len(shown) + 8)
        self.list.right = min(self.list.right, surf.get_width() - 4)
        if self.list.bottom > surf.get_height() - 4:
            self.list.bottom = self.button.top - 2  # no room below: open upwards
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


class Button:
    """A button in a viewer's header, beside its menus: clicking it calls *on_click*.

    Shown only while *visible()* -- when what it opens is there to open.
    """

    def __init__(self, label, on_click, visible=lambda: True):
        self.label = label
        self.on_click = on_click
        self.visible = visible
        self.button = pygame.Rect(0, 0, 0, 0)

    def handle(self, ev):
        """Whether *ev* was this button's to handle."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.visible() and self.button.collidepoint(ev.pos):
            self.on_click()
            return True
        return False

    def width(self, font):
        return font.size(self.label)[0] + 8

    def draw_button(self, surf, font, x, y, color, max_width=None):
        img = font.render(self.label, True, color)
        self.button = pygame.Rect(x - 4, y - 2, img.get_width() + 8, img.get_height() + 4)
        hot = self.button.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surf, (52, 60, 76) if hot else (36, 42, 54), self.button, border_radius=4)
        surf.blit(img, (x, y))
        return self.button.width

    def draw_list(self, surf, font):
        pass


class TextViewer:
    """A read-only, scrollable view of one stored text, with optional header menus.

    Without *menus* the header is *title*; with them, it is their buttons --
    the first usually standing in for the title. *inset* shrinks the window,
    so that one opened over another shows the one beneath around its edges.

    A line whose next line is indented deeper has a triangle in the gutter:
    clicking it folds the lines indented deeper that follow -- into ``… N
    lines`` -- and unfolds them. The ``fold`` menu at the right of the title
    row (and the keys 0-9) folds every such line at a level at once. With
    *numbers*, the gutter numbers the text's own lines. Text is selected by
    dragging over it, every line by Cmd+A, and copied by Cmd+C -- the
    selection, or all of it when there is none.
    """

    #: Spaces per level of nesting, for the fold menu: as `dbxpipelines.dict_lines` lays a dict out.
    INDENT = 4
    #: The levels the fold menu offers; the keys reach 1-9.
    FOLD_LEVELS = 6
    #: Seconds a note -- "copied ..." -- stays in the hint line.
    NOTE_S = 2.5

    def __init__(self, title, text, menus=(), inset=0):
        self.title = title
        self.menus = list(menus)
        self.inset = inset
        # The title's color, as a function returning one -- or None for the usual.
        self.title_color = lambda: None
        self.over = None  # the viewer this one was opened over: it opens below that one's header
        self.fold_menu = Dropdown("fold", [("unfold all", None)] + [(f"fold past level {n}", n) for n in range(1, self.FOLD_LEVELS + 1)],
                                  self.set_fold)
        self.header_bottom = 0  # where the header ends, as last drawn
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.close = pygame.Rect(0, 0, 0, 0)
        self.note = None  # (text, time): shown in the hint line for NOTE_S
        self._body = None  # (text x, top, line height, char width, rows, gutter x): where the text was last drawn
        self.set_text(text)

    def set_text(self, text):
        self.text = text.replace("\t", "    ")
        self.paras = self.text.split("\n")  # the text's own lines
        self.collapsed = set()  # indices of the lines folded
        self.top = 0  # first visible row
        self._wrapped = (None, None, [], [])
        self.clear_selection()

    # -- folding --------------------------------------------------------------

    @staticmethod
    def _indent_(line):
        return len(line) - len(line.lstrip(" "))

    def foldable(self, i):
        """Whether line *i* has a fold: its next line is indented deeper."""
        nxt = self.paras[i + 1] if i + 1 < len(self.paras) else ""
        return bool(nxt.strip()) and self._indent_(nxt) > self._indent_(self.paras[i])

    def toggle(self, i):
        """Fold line *i*'s deeper lines, or unfold them."""
        self.collapsed ^= {i}
        self.clear_selection()

    def set_fold(self, level):
        """Fold every line at *level* levels of indentation or deeper that has a fold; None unfolds them all."""
        self.collapsed = set() if level is None else {
            i for i in range(len(self.paras)) if self.foldable(i) and self._indent_(self.paras[i]) >= self.INDENT * level}
        self.top = 0
        self.clear_selection()

    def shown(self):
        """``[(line, hidden)]``: the lines shown, by index, each with how many deeper lines its fold hides."""
        out, i, n = [], 0, len(self.paras)
        while i < n:
            if i in self.collapsed and self.foldable(i):
                depth, j = self._indent_(self.paras[i]), i + 1
                while j < n and self.paras[j].strip() and self._indent_(self.paras[j]) > depth:
                    j += 1
                out.append((i, j - i - 1))
                i = j
            else:
                out.append((i, 0))
                i += 1
        return out

    def folded(self):
        """The lines shown, as text: a folded one ends with ``… N lines``."""
        return [self.paras[i] + (f"  … {k} line{'s' if k > 1 else ''}" if k else "") for i, k in self.shown()]

    def lines(self, cols):
        """The rows shown, wrapped to *cols*; `_rows_` says which line each is of."""
        key = (cols, frozenset(self.collapsed))
        if self._wrapped[:2] != key:
            if self._wrapped[0] is not None:
                self.clear_selection()  # its rows and columns are of the old layout
            out, meta = [], []
            for i, hidden in self.shown():
                para = self.paras[i]
                # Continuation lines hang under the value of a "column  value"
                # row, or under the indentation of a yaml line. The value is
                # wrapped in the width beside its column, so that a long
                # path or hash fills that space rather than leaving the
                # column's name alone on a line.
                m = re.match(r"^\S+ {2,}|^ *", para)
                head = m[0][: cols // 2]
                body = para[len(m[0]) :]
                wrapped = textwrap.wrap(body, cols - len(head), replace_whitespace=False, drop_whitespace=False, break_on_hyphens=False) or [""]
                for k, w in enumerate(wrapped):
                    out.append((head if k == 0 else " " * len(head)) + w)
                    # (line, first row of it, lines its fold hides -- on its last row)
                    meta.append((i, k == 0, hidden if k == len(wrapped) - 1 else 0))
            self._wrapped = (cols, frozenset(self.collapsed), out, meta)
        return self._wrapped[2]

    def scroll(self, n):
        self.top = max(0, self.top + n)

    # -- selection ------------------------------------------------------------

    def clear_selection(self):
        self.anchor = self.head = None  # (row, column) in the rows shown
        self.selecting = False

    def selection(self):
        """The selected span, ``((row, col), (row, col))`` start first; None when nothing is selected."""
        if self.anchor is None or self.head is None or self.anchor == self.head:
            return None
        return tuple(sorted((self.anchor, self.head)))

    def selected_text(self):
        """The text selected, as shown -- folded and wrapped -- or None."""
        span = self.selection()
        if span is None:
            return None
        (r0, c0), (r1, c1) = span
        lines = self._wrapped[2]
        if r0 == r1:
            return lines[r0][c0:c1]
        return "\n".join([lines[r0][c0:]] + lines[r0 + 1 : r1] + [lines[r1][:c1]])

    def copy(self, now=None):
        """Put the selection -- or, with none, the whole text -- on the clipboard, and say so."""
        text = self.selected_text()
        what = f"copied {len(text):,} characters" if text is not None else f"copied all {len(self.text):,} characters"
        if text is None:
            text = self.text
        self.note = (what if _copy_(text) else "could not copy: no clipboard", now if now is not None else time.monotonic())

    # -- events and drawing ---------------------------------------------------

    def handle(self, ev):
        """Returns "close" or None."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if self.fold_menu.handle(ev) or any(m.handle(ev) for m in self.menus):
            return None
        mods = pygame.key.get_mods() & (pygame.KMOD_META | pygame.KMOD_CTRL)
        if ev.type == pygame.MOUSEWHEEL:
            dy = getattr(ev, "precise_y", ev.y)
            self.scroll(-round(3 * dy) or (-1 if dy > 0 else 1))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self._in_gutter_(ev.pos):
            line = self._fold_at_(ev.pos)
            if line is not None:
                self.toggle(line)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self._in_body_(ev.pos):
            self.anchor = self.head = self._at_(ev.pos)
            self.selecting = True
        elif ev.type == pygame.MOUSEMOTION and self.selecting:
            self.head = self._at_(ev.pos, scroll=True)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.selecting:
            self.selecting = False
            if self.anchor == self.head:
                self.clear_selection()  # a click, not a drag
        elif ev.type == pygame.KEYDOWN:
            page = max(1, self.rect.height // 18 - 4)
            step = {pygame.K_UP: -1, pygame.K_DOWN: 1, pygame.K_PAGEUP: -page, pygame.K_PAGEDOWN: page, pygame.K_SPACE: page}
            if ev.key == pygame.K_ESCAPE:
                if self.selection() is not None:
                    self.clear_selection()
                    return None
                return "close"
            if ev.key == pygame.K_c and mods:
                self.copy()
            elif ev.key == pygame.K_a and mods:
                lines = self._wrapped[2]
                if lines:
                    self.anchor, self.head = (0, 0), (len(lines) - 1, len(lines[-1]))
            elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT) and self.menus:
                self.menus[0].step(1 if ev.key == pygame.K_RIGHT else -1)
            elif ev.key in step:
                self.scroll(step[ev.key])
            elif ev.key == pygame.K_HOME:
                self.top = 0
            elif pygame.K_0 <= ev.key <= pygame.K_9 and not mods:
                self.set_fold((ev.key - pygame.K_0) or None)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and not self.rect.collidepoint(ev.pos):
            return "close"
        return None

    def draw(self, surf, font, mono, numbers=False):
        """Draw the window; with *numbers*, the text's lines are numbered in the gutter."""
        w, h = surf.get_size()
        self.rect = pygame.Rect(0, 0, int(w * 0.8) - 2 * self.inset, int(h * 0.82) - 2 * self.inset)
        self.rect.center = (w // 2, (h - 24) // 2)
        if self.over is not None:
            # Below the header of the one it was opened over, which stays in view.
            bottom = self.rect.bottom
            self.rect.top = self.over.header_bottom + 6
            self.rect.height = max(160, bottom - self.rect.top)
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
        # The header: the title -- the first menu, if any -- has the top row to
        # itself, as wide as it needs up to the fold menu by the close button;
        # the other menus and buttons sit in a row underneath.
        fold_x = self.close.x - 12 - (font.size(self.fold_menu.label)[0] + 22)
        self.fold_menu.draw_button(surf, font, fold_x, y + pad, STATUS)
        title_w = fold_x - 12 - (x + pad)
        row_h = font.get_linesize() + 10
        bottom = y + pad + font.get_linesize()
        title_color = self.title_color() or EDGE
        if self.menus:
            self.menus[0].draw_button(surf, font, x + pad, y + pad, title_color, max_width=title_w)
            rest = [m for m in self.menus[1:] if getattr(m, "visible", lambda: True)()]
            bx, by = x + pad, y + pad + row_h
            for m in rest:
                if bx > x + pad and bx + m.width(font) > x + rw - pad:
                    bx, by = x + pad, by + row_h  # a row full: the next one
                bx += m.draw_button(surf, font, bx, by, EDGE) + 12
            if rest:
                bottom = by + font.get_linesize()
        else:
            surf.blit(font.render(_fit_(font, self.title, title_w), True, title_color), (x + pad, y + pad))
        self.header_bottom = bottom + 4

        # The gutter: line numbers, if asked for, then the fold triangles.
        lh, cw = mono.get_linesize(), mono.size("M")[0]
        digits = len(str(len(self.paras))) if numbers else 0
        gutter_x = x + pad
        tri_x = gutter_x + (digits + 1) * cw if numbers else gutter_x
        text_x = tri_x + 2 * cw
        body_top = bottom + 10
        rows = max(1, (y + rh - pad - lh - 6 - body_top) // lh)
        lines = self.lines(max(20, (x + rw - pad - text_x) // cw))
        meta = self._wrapped[3]
        self.top = min(self.top, max(0, len(lines) - rows))
        self._body = (text_x, body_top, lh, cw, rows, gutter_x)
        span = self.selection()
        mouse = pygame.mouse.get_pos()
        for k, ln in enumerate(lines[self.top : self.top + rows]):
            r, ty = self.top + k, body_top + k * lh
            i, first, hidden = meta[r]
            if first and numbers:
                surf.blit(mono.render(str(i + 1).rjust(digits), True, GRID if not hidden else STATUS), (gutter_x, ty))
            if first and self.foldable(i):
                cx, cy, t = tri_x + cw // 2, ty + lh // 2, max(3, cw // 2)
                color = EDGE if pygame.Rect(tri_x - 2, ty, 2 * cw, lh).collidepoint(mouse) else STATUS
                if i in self.collapsed:
                    pygame.draw.polygon(surf, color, [(cx - t // 2, cy - t), (cx - t // 2, cy + t), (cx + t, cy)])  # ▸
                else:
                    pygame.draw.polygon(surf, color, [(cx - t, cy - t // 2), (cx + t, cy - t // 2), (cx, cy + t)])  # ▾
            if span is not None and span[0][0] <= r <= span[1][0]:
                c0 = span[0][1] if r == span[0][0] else 0
                c1 = span[1][1] if r == span[1][0] else len(ln) + 1  # a line break selected shows as a cell
                pygame.draw.rect(surf, (60, 74, 104), (text_x + c0 * cw, ty, max(cw // 2, (c1 - c0) * cw), lh))
            surf.blit(mono.render(ln, True, LABEL), (text_x, ty))
            if hidden:
                surf.blit(mono.render(f"  … {hidden} line{'s' if hidden > 1 else ''}", True, STATUS), (text_x + len(ln) * cw, ty))
        where = f"lines {self.top + 1}–{min(len(lines), self.top + rows)} of {len(lines)}"
        cycle = "   Left / Right: previous / next" if self.menus else ""
        copy = "Cmd+C: copy selection" if span is not None else "drag: select   Cmd+A: all   Cmd+C: copy"
        note = self.note[0] if self.note and time.monotonic() - self.note[1] < self.NOTE_S else None
        hint = f"{where}   {note}" if note else f"{where}   {copy}{cycle}   Esc: close"
        surf.blit(font.render(_fit_(font, hint, rw - 2 * pad), True, EDGE if note else STATUS), (x + pad, y + rh - pad - lh + 2))
        for m in self.menus:
            m.draw_list(surf, font)
        self.fold_menu.draw_list(surf, font)

    # 4. Helpers -------------------------------------------------------------

    def _in_body_(self, pos):
        if self._body is None:
            return False
        tx, top, lh, _, rows, _ = self._body
        return self.rect.collidepoint(pos) and pos[0] >= tx - 4 and top <= pos[1] < top + rows * lh

    def _in_gutter_(self, pos):
        if self._body is None:
            return False
        tx, top, lh, _, rows, gx = self._body
        return gx <= pos[0] < tx - 4 and top <= pos[1] < top + rows * lh

    def _fold_at_(self, pos):
        """The line whose fold triangle is in the gutter at *pos*, or None."""
        _, top, lh, _, _, _ = self._body
        r = self.top + (pos[1] - top) // lh
        meta = self._wrapped[3]
        if 0 <= r < len(meta):
            i, first, _ = meta[r]
            if first and self.foldable(i):
                return i
        return None

    def _at_(self, pos, scroll=False):
        """The (row, column) of the text at screen *pos*; past the top or bottom, with *scroll*, the text scrolls a row."""
        tx, top, lh, cw, rows, _ = self._body
        lines = self._wrapped[2]
        if not lines:
            return (0, 0)
        if scroll and pos[1] < top:
            self.scroll(-1)
        elif scroll and pos[1] >= top + rows * lh:
            self.top = min(self.top + 1, max(0, len(lines) - rows))
        r = min(len(lines) - 1, max(0, self.top + (pos[1] - top) // lh))
        c = min(len(lines[r]), max(0, round((pos[0] - tx) / cw)))
        return (r, c)


def _fit_(font, text, width):
    """*text*, shortened with … until *font* renders it within *width*."""
    if font.size(text)[0] <= width:
        return text
    while len(text) > 1 and font.size(text + "…")[0] > width:
        text = text[:-1]
    return text + "…"


def _copy_(text) -> bool:
    """Put *text* on the clipboard: SDL's, through pygame, else macOS's pbcopy. Whether it got there."""
    try:
        pygame.scrap.put_text(text)
        return True
    except Exception:
        pass
    if sys.platform == "darwin":
        import subprocess

        try:
            subprocess.run(["pbcopy"], input=text, text=True, check=True, timeout=5)
            return True
        except Exception:
            pass
    return False


class DiskViewer:
    """A window with a Poincaré disk of its own: points laid out from some data, colored by label.

    Navigated as the main disk is -- click to center, drag to move, scroll to
    zoom, Home or a double click off the disk to go back -- and closed as a text window is. Until
    :meth:`set_points` gives it points it shows *note*: what it is waiting on.
    Labels are colored by how many points have them, the most first; past
    `PALETTE`, the rest share one gray, as "other".
    """

    PALETTE = [(230, 120, 100), (110, 175, 235), (140, 205, 120), (235, 190, 90), (190, 140, 230),
               (90, 205, 195), (235, 140, 190), (175, 175, 110), (120, 140, 230), (220, 160, 120)]
    OTHER = (120, 125, 135)

    def __init__(self, title, note="laying out …"):
        self.title = title
        self.note = note
        self.menus = []
        self.inset = 0
        self.over = None
        self.header_bottom = 0
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.close = pygame.Rect(0, 0, 0, 0)
        self.view = View()
        self.positions = np.zeros(0, dtype=complex)
        self.labels = None
        self.tips = []
        self.colors = []
        self.legend = []  # [(color, label, count)]
        self.press = None
        self.dragging = False
        self.last = None
        self.hover = None
        self.last_off_disk = None  # (pos, time) of the last left click off the disk: half a double click
        self._frame = (0, 0, 1)
        self.grid = _grid_()

    def set_note(self, note):
        self.note = note

    def set_points(self, positions, labels=None, tips=None):
        """Show *positions* -- complex, inside the unit disk -- each colored by its label, *tips* shown on hover."""
        from collections import Counter

        self.positions = np.asarray(positions, dtype=complex)
        self.labels = None if labels is None else [str(l) for l in labels]
        self.tips = list(tips) if tips is not None else [l or "" for l in (self.labels or [""] * len(self.positions))]
        self.note = None
        if self.labels is None:
            self.colors, self.legend = [], []
            return
        counts = Counter(self.labels).most_common()
        named = {label: self.PALETTE[i] for i, (label, _) in enumerate(counts[: len(self.PALETTE)])}
        self.colors = [named.get(l, self.OTHER) for l in self.labels]
        self.legend = [(named[l], l, n) for l, n in counts[: len(self.PALETTE)]]
        rest = sum(n for _, n in counts[len(self.PALETTE) :])
        if rest:
            self.legend.append((self.OTHER, f"other ({len(counts) - len(self.PALETTE)} labels)", rest))

    def animating(self):
        return self.view.anim is not None

    # -- geometry -------------------------------------------------------------

    def to_screen(self, z):
        cx, cy, R = self._frame
        return cx + R * z.real, cy - R * z.imag

    def to_disk(self, pos):
        cx, cy, R = self._frame
        return complex((pos[0] - cx) / R, -(pos[1] - cy) / R)

    def display(self, data, M):
        return P.zoom(M(data), self.view.k)

    def view_point(self, pos, limit):
        z = self.to_disk(pos)
        if abs(z) > limit:
            z = z / abs(z) * limit
        return complex(P.unzoom(np.array(z), self.view.k))

    # -- events and drawing ---------------------------------------------------

    def handle(self, ev):
        """Returns "close" or None."""
        now = time.monotonic()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            return "close"
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and not self.rect.collidepoint(ev.pos):
            return "close"
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_HOME:
            self.reset_view(now)
        elif ev.type == pygame.MOUSEWHEEL:
            self.view.zoom_by(ZOOM_STEP ** getattr(ev, "precise_y", ev.y))
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and abs(self.to_disk(ev.pos)) < 1:
            self.press, self.dragging = ev.pos, False
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            double, self.last_off_disk = _double_click_(self.last_off_disk, ev.pos, now)
            if double:
                self.reset_view(now)
        elif ev.type == pygame.MOUSEMOTION and self.press is not None:
            if not self.dragging:
                dx, dy = ev.pos[0] - self.press[0], ev.pos[1] - self.press[1]
                if dx * dx + dy * dy < DRAG_PX**2:
                    return None
                self.dragging, self.last = True, self.press
                self.view.M, self.view.anim = self.view.isometry(now), None
            self.view.drag(self.view_point(self.last, DRAG_RADIUS), self.view_point(ev.pos, DRAG_RADIUS))
            self.last = ev.pos
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.press is not None:
            if not self.dragging:
                if self.hover is not None:
                    z = self.display(self.positions[self.hover], self.view.isometry(now))
                else:
                    z = self.to_disk(self.press)
                self.view.center_on(complex(P.unzoom(np.array(z), self.view.k)), now)
            self.press, self.dragging = None, False
        return None

    def reset_view(self, now):
        """Back to the original view: zoom 1, the center sliding back to the middle."""
        self.view.k = 1.0
        self.view.center_on(self.view.isometry(now)(0j), now)

    def draw(self, surf, font, mono, numbers=False):
        now = time.monotonic()
        w, h = surf.get_size()
        self.rect = pygame.Rect(0, 0, int(w * 0.8) - 2 * self.inset, int(h * 0.82) - 2 * self.inset)
        self.rect.center = (w // 2, (h - 24) // 2)
        if self.over is not None:
            bottom = self.rect.bottom
            self.rect.top = self.over.header_bottom + 6
            self.rect.height = max(200, bottom - self.rect.top)
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
        surf.blit(font.render(_fit_(font, self.title, self.close.x - 12 - (x + pad)), True, EDGE), (x + pad, y + pad))
        self.header_bottom = y + pad + font.get_linesize() + 4

        # The disk on the left of the body, the legend on the right.
        top, bottom = self.header_bottom + 6, y + rh - pad - font.get_linesize() - 4
        legend_w = min(260, rw // 3) if self.legend else 0
        body_w = rw - 2 * pad - legend_w
        R = max(10, 0.47 * min(body_w, bottom - top))
        cx, cy = x + pad + body_w / 2, (top + bottom) / 2
        self._frame = (cx, cy, R)
        pygame.draw.circle(surf, DISK, (cx, cy), R)
        M = self.view.isometry(now)
        for line in self.grid:
            pts = np.asarray(self.to_screen(self.display(line, M))).T
            if len(pts) > 1:
                pygame.draw.aalines(surf, GRID, False, pts)
        pygame.draw.circle(surf, RIM, (cx, cy), R, 2)

        self.hover = None
        if self.note:
            img = font.render(self.note, True, STATUS)
            surf.blit(img, img.get_rect(center=(cx, cy)))
        elif len(self.positions):
            z = self.display(self.positions, M)
            sx, sy = self.to_screen(z)
            r2 = np.abs(z) ** 2
            mx, my = pygame.mouse.get_pos()
            if self.rect.collidepoint(mx, my):
                d2 = (sx - mx) ** 2 + (sy - my) ** 2
                i = int(np.argmin(d2))
                if d2[i] <= HOVER_PX**2:
                    self.hover = i
            for i in np.argsort(-r2):
                if self.colors:
                    col = self.colors[i]
                else:
                    ang = math.degrees(math.atan2(self.positions[i].imag, self.positions[i].real)) % 360
                    col = pygame.Color(0)
                    col.hsva = (ang, 60, 95, 100)
                rad = 1.2 + 4.0 * (1.0 - r2[i])
                pygame.draw.circle(surf, col, (sx[i], sy[i]), rad)
                if i == self.hover:
                    pygame.draw.circle(surf, EDGE, (sx[i], sy[i]), rad + 3, 2)
        ly = top
        for color, label, count in self.legend:
            lx = x + rw - pad - legend_w + 8
            pygame.draw.circle(surf, color, (lx + 5, ly + font.get_linesize() // 2), 5)
            text = _fit_(font, f"{label}  {count:,}", legend_w - 24)
            surf.blit(font.render(text, True, LABEL), (lx + 16, ly))
            ly += font.get_linesize() + 2
        n = len(self.positions)
        hint = f"{n:,} points   click: center · drag: move · scroll: zoom · double-click outside or Home: reset · Esc: close"
        surf.blit(font.render(_fit_(font, hint, rw - 2 * pad), True, STATUS), (x + pad, y + rh - pad - font.get_linesize() + 2))
        if self.hover is not None and self.tips:
            tip = font.render(_fit_(font, self.tips[self.hover], 500), True, TIP_FG)
            r = tip.get_rect(topleft=(pygame.mouse.get_pos()[0] + 14, pygame.mouse.get_pos()[1] + 14)).inflate(16, 10)
            r.clamp_ip(surf.get_rect())
            pygame.draw.rect(surf, TIP_BG, r, border_radius=4)
            surf.blit(tip, tip.get_rect(center=r.center))


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
    corpus comes from and how it is embedded (`__source__`, `__embedder__`, `__default_data__`,
    `__storage_options__`), what opening a point
    does (`__open_point__`), what clicking one does (`__click_point__`,
    `__context_point__`), and how a point is labeled (`__point_label__`,
    `__point_name__`, `__point_version__`, `__point_datetime__`,
    `__point_tooltip__`). The hooks are the only methods
    a specialization is meant to override; everything else is shared.
    """

    title = "hyp"

    #: The controls, at the right of the status line.
    HINT = "click: center · drag: move · scroll: zoom · right-click: open / add · Home or double-click outside: reset"

    #: Where corpora are kept unless --data says otherwise: one directory each.
    HOME = os.path.join(os.path.expanduser("~"), ".hyp")

    # 1. Hooks ------------------------------------------------------------------

    @classmethod
    def __default_data__(cls, args):
        return os.path.join(cls.HOME, "seeds")

    @classmethod
    def __storage_options__(cls, args):
        """The fsspec storage options the corpus directory is opened with."""
        return None

    @classmethod
    def __embedder__(cls, embedder):
        """The embedder the corpus is embedded with, made from the one --embedder names: here, that one."""
        return embedder

    @classmethod
    def __source__(cls, args):
        """The corpus source the `Engine` syncs from, or None for the seed labels."""
        return None

    def __reload__(self):
        """What reload does besides reloading the corpus: here, nothing."""

    def __click_point__(self, i, now):
        """Left click on point *i*: slide it to the center."""
        z = self.display(self.positions(now)[i], self.view.isometry(now))
        self.view.center_on(complex(P.unzoom(np.array(z), self.view.k)), now)

    def __context_point__(self, i, pos):
        """Right click on point *i*, at screen *pos*: open it."""
        self.__open_point__(i)

    def __open_point__(self, i):
        """Right click on point *i*: show its stored text."""
        viewer = TextViewer(self.snap.summaries[i], self.snap.texts[i])
        viewer.title_color = self.title_color(i)
        self.viewers = [viewer]

    def __point_label__(self, i):
        """The label drawn beside point *i*."""
        return self.snap.summaries[i]

    def __point_name__(self, i):
        """A name for point *i*, as text, drawn under its label before its datetime; or None: here, none."""
        return None

    def __point_version__(self, i):
        """A version of point *i*, as text, drawn under its label after its name, as ``v<version>``; or None: here, none."""
        return None

    def __point_datetime__(self, i):
        """When point *i* is from, as text, or None: here, when it was added."""
        added = self.snap.points[i].get("added") if self.snap.points else None
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(added)) if added else None

    def __point_tooltip__(self, i):
        """The tooltip of point *i*: its label, both lines, as the settings show them -- in full, where the label is cut short."""
        sub = self.sublabel(i, full=True)
        return f"{self.__point_label__(i)}\n{sub}" if sub else self.__point_label__(i)

    def __animating__(self):
        """What of the specialization's layers asks to be redrawn: anything with an ``animating()`` that says so."""
        return []

    def __draw_layer__(self, layer, now):
        """Draw the specialization's layer *layer*: here, there are none."""

    def __layer_covers__(self, layer, pos):
        """Whether *layer* -- not the disk -- has something at screen *pos*: here, none does."""
        return False

    def __layer_event__(self, layer, ev, now):
        """Handle *ev*, which is *layer*'s -- the frontmost at the mouse, not the disk; whether it was handled."""
        return False

    def layer_at(self, pos):
        """The frontmost layer with something at *pos*: the disk, inside its circle or where no other layer is."""
        for layer in reversed(self.layers):
            if layer == "disk":
                if abs(self.to_disk(pos)) < 1:
                    return "disk"
            elif self.__layer_covers__(layer, pos):
                return layer
        return "disk"

    def bring_to_front(self, layer):
        self.layers.remove(layer)
        self.layers.append(layer)

    def send_to_back(self, layer):
        self.layers.remove(layer)
        self.layers.insert(0, layer)

    def rotate_layers(self):
        """The back layer to the front: Tab."""
        self.layers.append(self.layers.pop(0))

    def _layers_items_(self):
        """The layers menu: front first, each with what can be done with it."""
        items = []
        for layer in reversed(self.layers):
            if layer != self.layers[-1]:
                items.append((f"{layer}: bring to front", ("front", layer)))
            if layer != self.layers[0]:
                items.append((f"{layer}: send to back", ("back", layer)))
        return items

    def _arrange_(self, pick):
        how, layer = pick
        (self.bring_to_front if how == "front" else self.send_to_back)(layer)

    def open_popup(self, items, on_select, pos, empty="none"):
        """Open a menu of *items* on the disk at *pos* -- as a Dropdown takes them -- calling *on_select* with the one picked."""
        self.popup = Dropdown("", items, on_select, empty=empty)
        self.popup.button = pygame.Rect(pos[0], pos[1], 0, 0)
        self.popup.open = True

    def push_viewer(self, viewer):
        """Open *viewer* over the ones open: inset so that they show around it, and below the header of the top one."""
        viewer.inset = 28 * len(self.viewers)
        viewer.over = self.viewers[-1] if self.viewers else None
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
                self.worker.events.put(("error", _failed_(e)))
            else:
                self.worker.events.put(("call", lambda: then(result)))
            finally:
                self.worker.events.put(("read", None))

        self.reading += 1
        self.error = None  # the last error was the last thing asked's: this is another
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
        self.error_at = 0.0
        self.reading = 0  # run_async reads under way: with the worker's jobs, what the spinner shows
        self.box = None
        self.viewers = []  # open text windows, the last on top
        self.hover = None
        self.press = None  # where the left button went down, while it is down
        self.last_off_disk = None  # (pos, time) of the last left click off the disk: half a double click
        self.dragging = False
        self.last = None
        self.labels = {}
        self.settings = {"datetimes": True, "line numbers": False, "title colors": True}
        self.settings_menu = Dropdown("settings", self._settings_items_, self._toggle_setting_)
        self.layers_menu = Dropdown("layers", self._layers_items_, self._arrange_)
        self.points_menu = Dropdown("points", self._points_items_, self.go_to, empty="no points")
        self._points_ = (None, None, [])  # (snapshot, settings, items): what the points menu last listed
        self.popup = None  # a menu opened on the disk, at the mouse: see open_popup
        self.reload_button = Button("reload", self.reload)
        # Back to front. A specialization adds layers of its own: see __draw_layer__.
        self.layers = ["disk"]
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

    def reload(self):
        """Read the corpus's source again, re-embed what changed, lay it all out afresh, and redraw it."""
        engine = self.engine

        def job(progress):
            engine.reload(progress)
            self.worker.events.put(("snapshot", (engine.snapshot(), None)))
            progress("reloaded")

        self.__reload__()
        self.worker.submit(job)

    def submit(self, text):
        engine = self.engine

        def job(progress):
            i = engine.add(text, progress)
            self.worker.events.put(("snapshot", (engine.snapshot(), i)))
            progress(f"added: {engine.store.points[i]['summary']}")

        self.worker.submit(job)

    def handle(self, ev, now):
        if self.settings_menu.handle(ev) or (not self.viewers and (self.reload_button.handle(ev) or self.points_menu.handle(ev))):
            return
        if len(self.layers) > 1 and not self.viewers and self.layers_menu.handle(ev):
            return
        if self.popup is not None and ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEWHEEL, pygame.KEYDOWN):
            # The menu's, whatever it is: a click picks an item or, elsewhere, only closes it.
            self.popup.handle(ev)
            if not self.popup.open:
                self.popup = None
            return
        if self.viewers:
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP) and self.viewers[-1].handle(ev) == "close":
                self.viewers.pop()
            return
        if len(self.layers) > 1:
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_TAB and self.box is None:
                self.rotate_layers()
                return
            pos = getattr(ev, "pos", None) or pygame.mouse.get_pos()
            layer = self.layer_at(pos) if self.press is None else "disk"  # a drag of the disk stays the disk's
            if layer != "disk" and self.__layer_event__(layer, ev, now):
                return
            if self.__layer_event__(None, ev, now):  # a layer's drag, or its keys: wherever the mouse is
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
            elif self.double_click_off_disk(ev.pos, now):
                self.reset_view(now)
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
                if self.hover is not None:
                    self.__click_point__(self.hover, now)
                else:
                    z = self.to_disk(self.press)
                    self.view.center_on(complex(P.unzoom(np.array(z), self.view.k)), now)
            self.press = None
            self.dragging = False
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            if self.box is not None:
                self.box.close()
                self.box = None
            if self.hover is not None:
                self.__context_point__(self.hover, ev.pos)
            else:
                self.box = TextBox(ev.pos)
        elif ev.type == pygame.MOUSEWHEEL:
            self.view.zoom_by(ZOOM_STEP ** getattr(ev, "precise_y", ev.y))
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_HOME:
            self.reset_view(now)

    def reset_view(self, now):
        """Back to the original view: zoom 1, the corpus center sliding back to the middle."""
        self.view.k = 1.0
        self.view.center_on(self.view.isometry(now)(0j), now)

    def double_click_off_disk(self, pos, now):
        """Whether a left click at *pos*, off the disk, is the second of a double click there."""
        double, self.last_off_disk = _double_click_(self.last_off_disk, pos, now)
        return double

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
                self.error, self.error_at = payload, now
            elif kind == "snapshot":
                self.on_snapshot(*payload, now)
            elif kind == "call":
                payload()
            elif kind == "finished":
                self.worker.pending -= 1
            elif kind == "read":
                self.reading -= 1

    # -- drawing --------------------------------------------------------------

    def label(self, i, text, sub=None):
        """Point *i*'s label: *text*, and below it, dimmer, *sub* -- its `sublabel`."""
        key = (i, text, sub)
        if key not in self.labels:
            short = text if len(text) <= 32 else text[:31] + "…"
            lines = [self.small.render(short, True, LABEL)]
            if sub:
                sub = sub if len(sub) <= 48 else sub[:47] + "…"
                lines.append(self.tiny.render(sub, True, STATUS))
            img = pygame.Surface((max(l.get_width() for l in lines), sum(l.get_height() for l in lines)), pygame.SRCALPHA)
            y = 0
            for line in lines:
                img.blit(line, (0, y))
                y += line.get_height()
            self.labels[key] = img
        return self.labels[key]

    def sublabel(self, i, full=False):
        """The second line of point *i*'s label: its name, ``v<version>`` and datetime, each as the settings say; None for none.

        A long name is cut short, so that the rest stays readable, unless *full*.
        """
        name = self.__point_name__(i) if self.settings.get("names") else None
        if name and len(name) > 30 and not full:
            name = name[:29] + "…"  # so that a long one leaves the rest readable
        version = self.__point_version__(i) if self.settings.get("versions") else None
        parts = [name, f"v{version}" if version else None, self.__point_datetime__(i) if self.settings.get("datetimes") else None]
        return " · ".join(p for p in parts if p) or None

    def point_color(self, z):
        """A point's color, by where it lies on the disk -- *z*, in data coordinates: its angle the hue, its radius the saturation."""
        col = pygame.Color(0)
        col.hsva = (math.degrees(math.atan2(z.imag, z.real)) % 360, 45 + 40 * min(1.0, abs(z) * 1.5), 95, 100)
        return col

    def title_color(self, i):
        """For a window of point *i*: a function giving its title the point's color, while the settings say to; None otherwise."""
        snap = self.snap
        return lambda: (self.point_color(snap.positions[i]) if self.settings.get("title colors") and i < len(snap.positions)
                        else None)

    def _points_items_(self):
        """Every point, by its label and second line in full, alphabetically: at most POINTS_MENU_LIMIT of them."""
        if self.snap is None:
            return []
        known = (self.snap, tuple(self.settings.items()))
        if self._points_[:2] != known:
            def text(i):
                sub = self.sublabel(i, full=True)
                t = f"{self.__point_label__(i)}  ·  {sub}" if sub else str(self.__point_label__(i))
                return t if len(t) <= POINTS_MENU_CHARS else t[: POINTS_MENU_CHARS - 1] + "…"

            items = sorted(((text(i), i) for i in range(len(self.snap.positions))), key=lambda t: t[0].lower())
            if len(items) > POINTS_MENU_LIMIT:
                items = items[:POINTS_MENU_LIMIT] + [(f"… and {len(items) - POINTS_MENU_LIMIT:,} more, not listed", None)]
            self._points_ = (*known, items)
        return self._points_[2]

    def go_to(self, i, now=None):
        """Slide point *i* to the center of the disk, however far it is; a pick of the points menu."""
        if i is None or self.snap is None or not 0 <= i < len(self.snap.positions):
            return
        now = time.monotonic() if now is None else now
        self.view.center_on(self.view.isometry(now)(self.positions(now)[i]), now, exact=True)

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
        # The layers, back to front: the disk, and a specialization's own --
        # the disk see-through over any behind it.
        for layer in self.layers:
            if layer == "disk":
                self.draw_disk(now, see_through=layer != self.layers[0])
            else:
                self.__draw_layer__(layer, now)
        self.draw_chrome(now)

    def draw_disk(self, now, see_through=False):
        """The disk, its points and their labels: opaque, or see-through over the layers behind it."""
        surf = self.screen
        cx, cy, R = self.frame()
        if see_through:
            veil = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            pygame.draw.circle(veil, (*DISK, DISK_SEE_THROUGH), (cx, cy), R)
            surf.blit(veil, (0, 0))
        else:
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
            if self.box is None and not self.viewers and self.popup is None and self.layer_at((mx, my)) == "disk":
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
                col = self.point_color(data[i])
                rad = 1.5 + 6.0 * (1.0 - r2[i])
                pygame.draw.circle(surf, col, (sx[i], sy[i]), rad)
                if i == self.hover:
                    pygame.draw.circle(surf, EDGE, (sx[i], sy[i]), rad + 3, 2)

            placed = []
            for i in np.argsort(r2)[:MAX_LABELS]:
                if r2[i] > LABEL_RADIUS**2:
                    break
                img = self.label(i, self.__point_label__(i), self.sublabel(i))
                rect = img.get_rect(midleft=(sx[i] + 8, sy[i]))
                if rect.collidelist(placed) < 0:
                    img.set_alpha(int(255 * (1.0 - r2[i] / LABEL_RADIUS**2) ** 0.5))
                    surf.blit(img, rect)
                    placed.append(rect.inflate(6, 2))

            if self.hover is not None:
                self.tooltip(self.__point_tooltip__(self.hover), (mx, my))

    def draw_chrome(self, now):
        """What is over every layer: the text box, the top bar's menus, the text windows, the status line."""
        surf = self.screen
        if self.box is not None:
            self.box.draw(surf, self.font)
        w = surf.get_width()
        sx = w - self.small.size("settings")[0] - 30
        self.settings_menu.draw_button(surf, self.small, sx, 10, STATUS)
        rx = sx - self.small.size("reload")[0] - 22
        self.reload_button.draw_button(surf, self.small, rx, 10, STATUS)
        # points, on the right with the rest: its list opens leftward from the window's edge.
        self.points_menu.draw_button(surf, self.small, rx - self.points_menu.width(self.small) - 14, 10, STATUS)
        if len(self.layers) > 1:
            self.layers_menu.draw_button(surf, self.small, 14, 10, STATUS)
        for viewer in self.viewers:
            viewer.draw(surf, self.font, self.mono, numbers=self.settings.get("line numbers"))
        self.settings_menu.draw_list(surf, self.small)
        if not self.viewers:
            self.points_menu.draw_list(surf, self.small)
            self.layers_menu.draw_list(surf, self.small)
        if self.popup is not None:
            self.popup.draw_list(surf, self.small)
        self.statusline(now)
        pygame.display.flip()

    def tooltip(self, text, pos):
        """*text* in a box by *pos*: its first line as a label's, any after it smaller, as a label's second line."""
        first, *rest = text.split("\n")
        lines = [self.font.render(first, True, TIP_FG)] + [self.small.render(t, True, (90, 90, 90)) for t in rest]
        w, h = max(l.get_width() for l in lines), sum(l.get_height() for l in lines)
        rect = pygame.Rect(pos[0] + 14, pos[1] + 14, w, h).inflate(16, 10)
        rect.clamp_ip(self.screen.get_rect())
        pygame.draw.rect(self.screen, TIP_BG, rect, border_radius=4)
        y = rect.y + 5
        for line in lines:
            self.screen.blit(line, (rect.x + 8, y))
            y += line.get_height()

    def busy(self):
        """Whether data are being loaded or prepared: an engine job, or a read a viewer waits on."""
        return bool(self.worker.pending or self.reading)

    def spinner(self, center, now, radius=7, dots=10):
        """A wheel of *dots* at *center*, the brightest one going round once a second."""
        head = int(now * dots) % dots
        for j in range(dots):
            a = 0.2 + 0.8 * (1.0 - ((head - j) % dots) / dots)  # 1 at the head, fading behind it to 0.2
            color = tuple(int(b + (s - b) * a) for b, s in zip(BG, LABEL))
            angle = 2 * math.pi * j / dots - math.pi / 2
            pygame.draw.circle(self.screen, color, (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)), 2)

    def statusline(self, now):
        w, h = self.screen.get_size()
        n = len(self.snap.texts) if self.snap else 0
        if self.error and now - self.error_at > ERROR_S:
            self.error = None
        msg = f"error: {self.error}" if self.error else self.status
        left = self.small.render(f"{n} texts · zoom {self.view.k:.2f} · {msg}", True, (240, 120, 110) if self.error else STATUS)
        right = self.small.render(self.HINT, True, STATUS)
        x = 10
        if self.busy():
            self.spinner((x + 8, h - 20 + left.get_height() // 2), now)
            x += 24
        self.screen.blit(left, (x, h - 20))
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
            moving = (self.view.anim is not None or self.tween is not None
                      or any(getattr(v, "animating", lambda: False)() for v in [*self.viewers, *self.__animating__()]))
            if events or news or moving or self.busy() or self.box is not None:
                self.draw(now)
            self.clock.tick(60 if moving else 30)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hyp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="corpus directory, local or an fsspec url (default: ~/.hyp/seeds; with --pipejournal, {DBX_DATALAKE}/.hyp)")
    ap.add_argument("--pipejournal", action="store_true", help="the pipelines of dbx's datalake (DBX_DATALAKE), from dbx.pipejournal(), one point each")
    ap.add_argument("--extra-construction-events", "--ece", nargs="+", default=[], metavar="EVENT",
                    help="with --pipejournal: events that also make an entry block count as constructed -- each alone, or "
                         "followed by = or : and the rest: `--ece ckpt` takes a block that has checkpointed (ckpt=<tag>)")
    ap.add_argument("--docs", action="store_true", help="the documents -- markdowns and pdfs -- under --dir, one point each")
    ap.add_argument("--dir", default=None, help="with --docs: the directory scanned, local or an fsspec url (default: the current one)")
    ap.add_argument("--remote-python", default=None, help="with --pipejournal: the python, on the datalake's host, that read runs trueinst() in -- with dbx, ray and the project (default: $HYP_REMOTE_PYTHON, else the one each entry records its build ran in, else python3)")
    ap.add_argument("--n-workers", type=int, default=16, help="threads reading a remote source, e.g. the dbx journal and its side files (default: 16)")
    ap.add_argument("--embedder", choices=["ollama", "deepseek"], default="ollama", help="ollama: an Ollama embedding model; deepseek: pooled DeepSeek hidden states via transformers")
    ap.add_argument("--embed-model", default=None, help=f"default: {EMBED_MODEL} for ollama")
    ap.add_argument("--summary-model", default=SUMMARY_MODEL)
    ap.add_argument("--reset", action="store_true", help="discard the stored corpus and start over")
    ap.add_argument("--seed", type=int, default=0, help="UMAP random state")
    args = ap.parse_args(argv)

    cls = Hyp
    if args.pipejournal and args.docs:
        ap.error("--pipejournal and --docs are two corpora: one at a time")
    if args.extra_construction_events and not args.pipejournal:
        ap.error("--extra-construction-events is for --pipejournal's entry blocks")
    if args.dir is not None and not args.docs:
        ap.error("--dir is the directory --docs scans")
    if args.pipejournal:
        from .dbxpipelines import HypDbxPipelines as cls
    elif args.docs:
        from .docs import HypDocs as cls
    if args.data is None:
        args.data = cls.__default_data__(args)
    store = Store(args.data, cls.__storage_options__(args))
    if args.reset:
        store.clear()
    ollama = Ollama()
    engine = Engine(
        store,
        cls.__embedder__(make_embedder(args.embedder, args.embed_model, ollama)),
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
