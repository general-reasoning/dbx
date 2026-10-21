"""A markdown written in hyp --docs: begun from what a right click on empty disk typed, edited in a window of its own, saved as a document.

A right click on empty disk offers **Markdown**: what is typed in its box,
on Enter, becomes a new markdown in the directory scanned, ``<first line>.md``,
and opens in the middle -- the size of a Claude session's window -- as text
to go on writing in (`NoteWindow`): Enter starts a new line there. It is
saved, and the disk brought up to date with it, as it begins, at Cmd+S, and
as its window closes; its point is then any document's.

Its title is its file's name, edited in place as a Claude session's is;
keeping it renames the file. Its **preview** button splits the window: the
markdown rendered above -- mermaid diagrams too -- as it is written below.
"""

import posixpath
import re

import fsspec
import pygame

from .docview import BAR, DIM, EDGE, FONTS, TEXT, AreaEdit, DocWindow, EditableTitle, MarkdownDoc, Message

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
        self._named_ = None  # a name given before the first save, its stem: taken then, not the first line's

    def dirty(self):
        """Whether the text is not as saved: unsaved, or changed since."""
        return self.text != self.saved

    def save(self):
        """Write the text -- named the first time, by the name given or its first line, not over any file there -- and its url."""
        fs, root = fsspec.url_to_fs(self.root, **self.storage_options)
        if self.url is None:
            stem = self._named_ or _stem_(self.text)
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
        """Its file's name, or what it will be named."""
        return f"{self.stem()}.md"

    def stem(self):
        """Its file's name less its ``.md``: as saved, as given, or by its first line."""
        if self.url:
            return posixpath.basename(self.url)[: -len(".md")]
        return self._named_ or _stem_(self.text)

    def rename(self, stem):
        """Name it *stem*, ``.md`` after: the file moved, where it is saved; the name taken at its first save, where not.

        Whether it changed: a blank name changes nothing. A name another file
        has is refused -- `FileExistsError` -- not written over.
        """
        stem = re.sub(r"\.md$", "", stem.strip().replace("/", "-")).strip()
        if not stem or stem == self.stem():
            return False
        fs, root = fsspec.url_to_fs(self.root, **self.storage_options)
        if fs.exists(posixpath.join(root, f"{stem}.md")):
            raise FileExistsError(f"{stem}.md is taken: another name")
        if self.url is None:
            self._named_ = stem
            return True
        old = fsspec.url_to_fs(self.url, **self.storage_options)[1]
        fs.mv(old, posixpath.join(posixpath.dirname(old), f"{stem}.md"))
        self.url = posixpath.join(posixpath.dirname(self.url), f"{stem}.md")
        return True


class NoteWindow(EditableTitle, DocWindow):
    """A markdown's text, edited: an `AreaEdit` filling the window -- or, previewed, its lower pane -- zoomed by Cmd and + or -.

    Enter starts a new line. Cmd+S saves -- *on_save(window)*, for its owner
    to save and embed -- and Esc or × close it, its owner saving what is not
    saved. The wheel scrolls the pane under the mouse; a click places the
    cursor, a drag selects. Its title, its file's name, is edited in place
    (`EditableTitle`): kept, the file renamed -- *on_rename(window)*, for its
    owner to file. The **preview** button, at the title row's right, splits
    it: the markdown rendered above, as it is written, and the bar between
    dragged to share the window between them.
    """

    TITLE_PREFIX = "Markdown · "
    SPLIT_RANGE = (0.15, 0.85)  # of the body, the preview's share, at the least and the most
    BAR_PX = 6  # the bar between the preview and the text: what a drag on it moves

    def __init__(self, note, on_save, on_rename=lambda window: None):
        super().__init__("Markdown", "")
        self.note, self.on_save, self.on_rename = note, on_save, on_rename
        self.edit = AreaEdit(note.text)
        self.doc = Message("")  # nothing rendered in the body: the text is the edit's, and the preview its own
        self._init_title_()
        self._editing_ = False  # the mouse pressed in the text, and not yet let go: a drag selects
        self.previewed = False  # split: the markdown rendered above the text
        self.split = 0.5  # the preview's share of the body, while previewed
        self._dragging_bar_ = False
        self.preview_button = pygame.Rect(0, 0, 0, 0)
        self.text_rect = pygame.Rect(0, 0, 0, 0)  # where the text is, as last drawn
        self.preview_rect = pygame.Rect(0, 0, 0, 0)  # where the preview is, as last drawn; empty while not previewed
        self._preview_ = (None, None)  # (the text it is of, its MarkdownDoc): made again only as the text changes
        self.psy = 0  # the preview scrolled, in pixels
        self.problem = None  # what went wrong, as said in the window: a name taken, say

    def sync(self):
        """The note's text as the window's: what is saved."""
        self.note.text = self.edit.text

    def dirty(self):
        self.sync()
        return self.note.dirty()

    def toggle_preview(self):
        self.previewed = not self.previewed

    def handle(self, ev):
        """Returns "close" or None."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.preview_button.collidepoint(ev.pos):
            self.keep_title()
            self.toggle_preview()
            return None
        if self._title_event_(ev):
            return None
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self._bar_().collidepoint(ev.pos):
            self._dragging_bar_ = True
            return None
        if ev.type == pygame.MOUSEMOTION and self._dragging_bar_:
            lo, hi = self.SPLIT_RANGE
            self.split = min(hi, max(lo, (ev.pos[1] - self.body.y) / max(1, self.body.height)))
            return None
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.text_rect.collidepoint(ev.pos):
            self.edit.press(ev.pos, shift=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
            self._editing_ = True
            return None
        if ev.type == pygame.MOUSEMOTION and self._editing_:
            self.edit.drag(ev.pos)
            return None
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._editing_ = self._dragging_bar_ = False
            return None
        if ev.type == pygame.MOUSEWHEEL:
            rows = -round(getattr(ev, "precise_y", ev.y) * 3) or (-1 if ev.y > 0 else 1)
            if self.preview_rect.collidepoint(pygame.mouse.get_pos()):
                self.psy = max(0, self.psy + rows * 16)
            else:
                self.edit.scroll(rows)
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
        label = small.render("preview" if not self.previewed else "hide preview", True, TEXT)
        self.header_reserve = label.get_width() + 22
        self.title = "" if self.naming is not None else f"{self.TITLE_PREFIX}{self.note.name()}"
        state = "unsaved" if self.note.dirty() else "saved"
        keys = "Enter: keep this name · Esc: put it back" if self.naming is not None else "Cmd+S: save · Esc: close and save"
        where = self.note.url or posixpath.join(self.note.root, self.note.name())
        # The state and the keys first: a long path is cut short, not they.
        self.sub = " · ".join(p for p in (self.problem, state, keys, where) if p)
        super().draw(surf, rect, font, small)
        self._draw_title_(surf, font, small)
        zoom_w = small.size(f"{self.zoom:.0%}")[0]
        self.preview_button = label.get_rect(topright=(self.close.x - zoom_w - 22, self.rect.y + 12)).inflate(10, 4)
        hot = self.preview_button.collidepoint(pygame.mouse.get_pos())
        pygame.draw.rect(surf, (52, 60, 76) if hot or self.previewed else (36, 42, 54), self.preview_button, border_radius=4)
        surf.blit(label, label.get_rect(center=self.preview_button.center))
        body = self.body
        if body.width < 40 or body.height < 20:
            return
        old = surf.get_clip()
        if self.previewed:
            cut = body.y + round(body.height * self.split)
            self.preview_rect = pygame.Rect(body.x, body.y, body.width, cut - body.y - self.BAR_PX // 2)
            self.text_rect = pygame.Rect(body.x, cut + self.BAR_PX // 2, body.width, body.bottom - cut - self.BAR_PX // 2)
            self._draw_preview_(surf, old)
            bar = self._bar_()
            pygame.draw.rect(surf, EDGE if self._dragging_bar_ or bar.collidepoint(pygame.mouse.get_pos()) else BAR, bar)
        else:
            self.preview_rect, self.text_rect = pygame.Rect(0, 0, 0, 0), body
        f = FONTS.get(15 * self.zoom)
        pad, r = 10, self.text_rect
        surf.set_clip(r.clip(old) if old else r)
        most = max(1, (r.height - 2 * pad) // f.get_linesize())
        self.edit.draw(surf, f, r.x + pad, r.y + pad, r.width - 2 * pad - 8, most, TEXT)
        if not self.edit.text:
            surf.blit(small.render("write here …", True, DIM), (r.x + pad + 4, r.y + pad))
        surf.set_clip(old)

    def animating(self):
        """A mermaid diagram rendering in the preview: drawn again till it is in."""
        doc = self._preview_[1]
        return bool(self.previewed and doc is not None and doc.pending())

    # 4. Helpers ----------------------------------------------------------------

    def _title_text_(self):
        return self.note.stem()

    def _title_kept_(self, text):
        try:
            changed = self.note.rename(text)
        except (FileExistsError, OSError) as e:
            self.problem = str(e)
            return
        self.problem = None
        if changed:
            self.on_rename(self)  # for the owner to file: saved under the new name, and the disk brought up to date

    def _bar_(self):
        """The bar between the preview and the text, as last drawn; none while not previewed."""
        if not self.previewed or not self.preview_rect.height:
            return pygame.Rect(0, 0, 0, 0)
        return pygame.Rect(self.body.x, self.preview_rect.bottom, self.body.width, self.BAR_PX)

    def _draw_preview_(self, surf, old):
        """The markdown, rendered as the text now is -- laid out again only once it has changed -- in the upper pane, scrolled."""
        text, doc = self._preview_
        if text != self.edit.text:
            doc = MarkdownDoc(self.edit.text)
            self._preview_ = (self.edit.text, doc)
        r = self.preview_rect
        laid = doc.layout(r.width - 10, round(self.zoom, 4))
        self.psy = min(self.psy, max(0, laid.height - r.height))
        surf.set_clip(r.clip(old) if old else r)
        doc.draw(surf, r.x, r.y - self.psy, r)
        surf.set_clip(old)


def _stem_(text):
    """A file's name for *text*: its first line, lower-cased, words joined by hyphens; ``note`` for none."""
    line = (text.strip().split("\n", 1)[0] if text.strip() else "").lstrip("#").strip()
    stem = re.sub(r"[^\w\s-]", "", line).strip().lower()
    stem = re.sub(r"[\s_-]+", "-", stem)[:NAME_CHARS].strip("-")  # a run of spaces and hyphens, one hyphen
    return stem or "note"
