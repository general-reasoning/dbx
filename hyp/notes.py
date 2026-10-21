"""A markdown written in hyp --docs: begun from what a right click on empty disk typed, edited in a window of its own, saved as a document.

A right click on empty disk offers **Markdown**: what is typed in its box,
on Enter, becomes a new markdown in the directory scanned, ``<first line>.md``,
and opens in the middle -- the size of a Claude session's window -- as text
to go on writing in (`NoteWindow`): Enter starts a new line there. It is
saved, and the disk brought up to date with it, as it begins, at Cmd+S, and
as its window closes; its point is then any document's.
"""

import posixpath
import re

import pygame

from .docview import DIM, FONTS, TEXT, AreaEdit, DocWindow, Message

#: A name's longest, before its ``.md``.
NAME_CHARS = 60


class Note:
    """A markdown being written: its text, and where it is saved -- named, at the first save, by its first line, in *root*."""

    def __init__(self, root, text="", storage_options=None):
        self.root = str(root).rstrip("/") or "/"
        self.storage_options = storage_options or {}
        self.text = text
        self.saved = None  # the text as last saved; None before the first save
        self.url = None  # where it is saved; None before the first save

    def dirty(self):
        """Whether the text is not as saved: unsaved, or changed since."""
        return self.text != self.saved

    def save(self):
        """Write the text -- named the first time, by its first line, not over any file there -- and its url."""
        import fsspec

        fs, root = fsspec.url_to_fs(self.root, **self.storage_options)
        if self.url is None:
            stem = _stem_(self.text)
            name, n = f"{stem}.md", 2
            while fs.exists(posixpath.join(root, name)):
                name, n = f"{stem}-{n}.md", n + 1
            self.url = posixpath.join(self.root, name)
        fs.makedirs(root, exist_ok=True)
        with fsspec.open(self.url, "w", **self.storage_options) as f:
            f.write(self.text)
        self.saved = self.text
        return self.url

    def name(self):
        """Its file's name, or what it will be named: by its first line."""
        return posixpath.basename(self.url) if self.url else f"{_stem_(self.text)}.md"


class NoteWindow(DocWindow):
    """A markdown's text, edited: an `AreaEdit` filling the window, its whole text scrolled, zoomed by Cmd and + or -.

    Enter starts a new line. Cmd+S saves -- *on_save(window)*, for its owner
    to save and embed -- and Esc or × close it, its owner saving what is not
    saved. The wheel scrolls; a click places the cursor, a drag selects.
    """

    def __init__(self, note, on_save):
        super().__init__("Markdown", "")
        self.note, self.on_save = note, on_save
        self.edit = AreaEdit(note.text)
        self.doc = Message("")  # nothing rendered: the text is the edit's, drawn over the body
        self._selecting_ = False

    def sync(self):
        """The note's text as the window's: what is saved."""
        self.note.text = self.edit.text

    def dirty(self):
        self.sync()
        return self.note.dirty()

    def handle(self, ev):
        """Returns "close" or None."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.body.collidepoint(ev.pos):
            self.edit.press(ev.pos, shift=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
            self._selecting_ = True
            return None
        if ev.type == pygame.MOUSEMOTION and self._selecting_:
            self.edit.drag(ev.pos)
            return None
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._selecting_ = False
            return None
        if ev.type == pygame.MOUSEWHEEL:
            self.edit.scroll(-round(getattr(ev, "precise_y", ev.y) * 3) or (-1 if ev.y > 0 else 1))
            return None
        if ev.type == pygame.TEXTINPUT:
            self.edit.handle(ev)
            return None
        if ev.type == pygame.KEYDOWN:
            k, mods = ev.key, pygame.key.get_mods() | getattr(ev, "mod", 0)
            cmd = mods & (pygame.KMOD_META | pygame.KMOD_CTRL)
            if k == pygame.K_ESCAPE:
                return "close"
            if k == pygame.K_s and cmd:
                self.sync()
                self.on_save(self)
                return None
            if cmd and k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS,
                             pygame.K_0, pygame.K_KP0):
                return super().handle(ev)  # zoomed, as any document's window
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.edit._insert_("\n")  # a new line: there is nothing to submit
                return None
            if k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                self.edit.scroll(-10 if k == pygame.K_PAGEUP else 10)
                return None
            self.edit.handle(ev)
        return None

    def draw(self, surf, rect, font, small):
        self.sync()
        self.title = f"Markdown · {self.note.name()}"
        state = "unsaved" if self.note.dirty() else "saved"
        # The state and the keys first: a long path is cut short, not they.
        self.sub = f"{state} · Cmd+S: save · Esc: close and save · {self.note.url or posixpath.join(self.note.root, self.note.name())}"
        super().draw(surf, rect, font, small)
        body = self.body
        if body.width < 40 or body.height < 20:
            return
        f = FONTS.get(15 * self.zoom)
        pad = 10
        old = surf.get_clip()
        surf.set_clip(body.clip(old) if old else body)
        most = max(1, (body.height - 2 * pad) // f.get_linesize())
        self.edit.draw(surf, f, body.x + pad, body.y + pad, body.width - 2 * pad - 8, most, TEXT)
        surf.set_clip(old)
        if not self.edit.text:
            surf.blit(small.render("write here …", True, DIM), (body.x + pad + 4, body.y + pad))


def _stem_(text):
    """A file's name for *text*: its first line, lower-cased, words joined by hyphens; ``note`` for none."""
    line = (text.strip().split("\n", 1)[0] if text.strip() else "").lstrip("#").strip()
    stem = re.sub(r"[^\w\s-]", "", line).strip().lower()
    stem = re.sub(r"[\s_-]+", "-", stem)[:NAME_CHARS].strip("-")  # a run of spaces and hyphens, one hyphen
    return stem or "note"
