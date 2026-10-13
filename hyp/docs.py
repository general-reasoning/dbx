"""hyp --docs: the documents of a directory -- markdowns and pdfs -- on the disk.

``hyp --docs`` scans the current directory, ``hyp --docs --dir URL`` another,
local or any fsspec url (``ssh://host/path``, ``gs://bucket/path``, ...): every
``.md``, ``.markdown`` and ``.pdf`` under it, but for hidden directories and
the usual heaps (`SKIP_DIRS`). Each is embedded by its name and the start of
its text, and labeled by its name, its modification time beneath.

Left click opens a document in the center, filling the window; right click
places it in a quarter or a half instead, in place of what is there. The
disk, the placed windows ("tiles") and the center window are three layers,
in an order the layers menu -- or Tab, which brings the back one to the front
-- changes: what is under the mouse in the front one gets the clicks.

The corpus is kept under ~/.hyp/docs/, one directory per scanned directory:
nothing is written to the one scanned.
"""

import hashlib
import os
import posixpath
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pygame

from .app import Hyp
from .docview import DocWindow, MarkdownDoc, Message, PdfDoc, pdf_text

KINDS = {".md": "markdown", ".markdown": "markdown", ".pdf": "pdf"}
# Directories never scanned, besides hidden ones.
SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "venv", "env", "build", "dist"}
MAX_DOCS = 5_000
MAX_DEPTH = 16
TEXT_CHARS = 8_000  # of a document's text, kept with its point
EMBED_CHARS = 4_000  # of it, embedded
PDF_TEXT_BYTES = 64 << 20  # a larger pdf is embedded by its name alone

# Where a right click can put a document: (x, y, width, height), as fractions of the window.
SLOTS = {
    "top left": (0, 0, 0.5, 0.5), "top right": (0.5, 0, 0.5, 0.5),
    "bottom left": (0, 0.5, 0.5, 0.5), "bottom right": (0.5, 0.5, 0.5, 0.5),
    "left half": (0, 0, 0.5, 1), "right half": (0.5, 0, 0.5, 1),
    "top half": (0, 0, 1, 0.5), "bottom half": (0, 0.5, 1, 0.5),
}
GAP = 6
# The placement picker's icons, as the screen has them: quarters in its corners, halves at its edges, the center in its middle.
PICKER = (("top left", "top half", "top right"),
          ("left half", "center", "right half"),
          ("bottom left", "bottom half", "bottom right"))


class TilePicker:
    """Where to place a document, as a grid of little windows -- each showing, filled, the part of the window it takes.

    A popup as `Hyp.open_popup` opens one: *open* until a click picks an icon
    or lands elsewhere, or Esc. The slot under the mouse is named beneath.
    """

    ICON = (36, 26)
    PAD = 6

    def __init__(self, on_select, pos):
        self.on_select = on_select
        self.open = True
        iw, ih = self.ICON
        self.rect = pygame.Rect(pos[0], pos[1], 3 * iw + 4 * self.PAD, 3 * ih + 4 * self.PAD + 18)
        self.icons = {}  # slot -> its rect, as last drawn

    def _icons_(self):
        iw, ih = self.ICON
        return {slot: pygame.Rect(self.rect.x + self.PAD + c * (iw + self.PAD), self.rect.y + self.PAD + r * (ih + self.PAD), iw, ih)
                for r, row in enumerate(PICKER) for c, slot in enumerate(row)}

    def slot_at(self, pos):
        return next((slot for slot, r in self._icons_().items() if r.collidepoint(pos)), None)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            slot = self.slot_at(ev.pos) if ev.button == 1 else None
            self.open = False
            if slot is not None:
                self.on_select(slot)
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.open = False
        return True

    def draw_list(self, surf, font):
        from .docview import BAR, DIM, EDGE, TEXT

        self.rect.clamp_ip(surf.get_rect())
        pygame.draw.rect(surf, (30, 35, 45), self.rect, border_radius=6)
        pygame.draw.rect(surf, EDGE, self.rect, 1, border_radius=6)
        hover = self.slot_at(pygame.mouse.get_pos())
        for slot, r in self._icons_().items():
            hot = slot == hover
            pygame.draw.rect(surf, (52, 60, 76) if hot else (24, 28, 36), r, border_radius=3)
            if slot == "center":
                part = r.inflate(-8, -8)
            else:
                fx, fy, fw, fh = SLOTS[slot]
                inner = r.inflate(-4, -4)
                part = pygame.Rect(inner.x + round(fx * inner.width), inner.y + round(fy * inner.height),
                                   round(fw * inner.width), round(fh * inner.height)).inflate(-2, -2)
            pygame.draw.rect(surf, EDGE if hot else TEXT, part, border_radius=2)
            pygame.draw.rect(surf, EDGE if hot else BAR, r, 1, border_radius=3)
        label = font.render(hover or "place it …", True, TEXT if hover else DIM)
        surf.blit(label, label.get_rect(midbottom=(self.rect.centerx, self.rect.bottom - 4)))


@dataclass(frozen=True)
class Doc:
    key: str  # its url: what a re-sync matches by
    text: str  # the start of its text
    embed: str  # its name, then the start of its text
    summary: str  # its name: the label
    extra: dict = field(default_factory=dict)  # path, kind, mtime, size: stored with the point


def _mtime_(info):
    """*info*'s modification time, as epoch seconds, from whichever field its filesystem names it by; None for none."""
    from datetime import datetime

    for k in ("mtime", "LastModified", "last_modified", "updated", "modified", "created"):
        v = info.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, datetime):
            return v.timestamp()
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def _is_local_(fs):
    protocol = fs.protocol if isinstance(fs.protocol, (tuple, list)) else (fs.protocol,)
    return bool({"file", "local"} & set(protocol))


def canonical(root):
    """*root* as one name for it: a local path made absolute, a url as it is."""
    return root if "://" in str(root) else os.path.abspath(os.path.expanduser(str(root)))


def find_docs(root, storage_options=None, limit=MAX_DOCS):
    """``(fs, [info], cut)``: the documents under *root*, their fsspec infos, in path order; *cut* if past *limit*.

    *fs* knows the urls of what it found: ``fs.url_of(name)`` -- see `url_of`.
    """
    import fsspec

    fs, top = fsspec.url_to_fs(canonical(root), **(storage_options or {}))
    fs.hyp_prefix = _prefix_(canonical(root), top)
    found, seen, stack = [], set(), [(top, 0)]
    while stack:
        d, depth = stack.pop()
        if d in seen or depth > MAX_DEPTH:
            continue
        seen.add(d if not _is_local_(fs) else os.path.realpath(d))
        try:
            entries = fs.ls(d, detail=True)
        except (OSError, PermissionError):
            continue
        subdirs = []
        for e in sorted(entries, key=lambda e: e["name"]):
            name = e["name"].rstrip("/")
            base = posixpath.basename(name)
            if base.startswith("."):
                continue
            if e.get("type") == "directory":
                if base not in SKIP_DIRS:
                    subdirs.append((name, depth + 1))
            elif posixpath.splitext(base)[1].lower() in KINDS:
                found.append(e)
                if len(found) >= limit:
                    return fs, sorted(found, key=lambda e: e["name"]), True
        stack.extend(reversed(subdirs))
    return fs, sorted(found, key=lambda e: e["name"]), False


def _prefix_(url, top):
    """What precedes *top*, the path fsspec made of *url*, in *url*: ``ssh://host`` of ``ssh://host/a/b``, ``gs://`` of ``gs://b/a``.

    fsspec's own ``unstrip_protocol`` drops an ssh url's host, which a file's
    url must keep to be opened again.
    """
    if "://" not in url:
        return ""
    url, tail = url.rstrip("/"), top.strip("/")
    if not tail or not url.endswith(tail):
        return ""  # not a plain suffix: fsspec's own urls, then
    head = url[: -len(tail)]
    return head if head.endswith("://") else head.rstrip("/")


def url_of(fs, name):
    """A file's name as a path here, or as a url for any other filesystem -- its host kept."""
    if _is_local_(fs):
        return name
    prefix = getattr(fs, "hyp_prefix", None)
    if not prefix:
        return fs.unstrip_protocol(name)
    return prefix + name if prefix.endswith("://") or name.startswith("/") else f"{prefix}/{name}"


def read_text(fs, info, kind):
    """The text of the document *info* names: a markdown's, or a pdf's first pages'."""
    if kind == "markdown":
        return fs.cat_file(info["name"]).decode("utf-8", "replace")
    if (info.get("size") or 0) > PDF_TEXT_BYTES:
        return ""
    return pdf_text(fs.cat_file(info["name"]))


def documents(root, progress=print, n_workers=16, cache=None, storage_options=None):
    """The documents under *root*, as the engine takes them, read on *n_workers* threads.

    *cache* -- ``{url: {"mtime", "size", "text"}}`` -- holds what was read
    before: a document whose time and size are unchanged is not read again.
    It is updated in place.
    """
    progress(f"scanning {root}")
    fs, infos, cut = find_docs(root, storage_options)
    if cut:
        progress(f"more than {MAX_DOCS:,} documents under {root}: the first {MAX_DOCS:,}, by path")
    cache = {} if cache is None else cache
    progress(f"reading {len(infos):,} documents")

    def one(info):
        url, base = url_of(fs, info["name"]), posixpath.basename(info["name"].rstrip("/"))
        kind = KINDS[posixpath.splitext(base)[1].lower()]
        mtime, size = _mtime_(info), info.get("size")
        known = cache.get(url)
        if known and known.get("mtime") == mtime and known.get("size") == size:
            text = known.get("text", "")
        else:
            try:
                text = read_text(fs, info, kind)[:TEXT_CHARS]
            except Exception as e:  # unreadable: placed by its name, saying why
                text = f"({type(e).__name__}: {e})"
            cache[url] = {"mtime": mtime, "size": size, "text": text}
        return Doc(key=url, text=text, embed=f"{base}\n\n{text[:EMBED_CHARS]}", summary=base,
                   extra={"path": url, "kind": kind, "mtime": mtime, "size": size})

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(one, infos))


def load(url, kind, storage_options=None):
    """The document at *url*, to show: a `MarkdownDoc` or a `PdfDoc`; a `Message` saying why, where it cannot be read."""
    import fsspec

    try:
        with fsspec.open(url, "rb", **(storage_options or {})) as f:
            data = f.read()
        return PdfDoc(data) if kind == "pdf" else MarkdownDoc(data.decode("utf-8", "replace"))
    except Exception as e:
        traceback.print_exc()
        return Message(f"could not read {url}: {type(e).__name__}: {e}")


def _overlap_(a, b):
    ax, ay, aw, ah = SLOTS[a]
    bx, by, bw, bh = SLOTS[b]
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class HypDocs(Hyp):
    """One point per document of a directory; a document opens in the center, or in a quarter or half of the window."""

    title = "hyp: docs"
    HINT = "click: open · right-click: place · drag: move · scroll: zoom · Tab: next layer · double-click outside: reset"
    LAYERS = ("disk", "tiles", "center")

    # 1. Hooks ------------------------------------------------------------------

    @classmethod
    def root(cls, args):
        return canonical(getattr(args, "dir", None) or os.getcwd())

    @classmethod
    def __default_data__(cls, args):
        return os.path.join(cls.HOME, "docs", hashlib.sha1(cls.root(args).encode()).hexdigest()[:12])

    @classmethod
    def __source__(cls, args):
        import json

        import fsspec

        root, n_workers = cls.root(args), getattr(args, "n_workers", 16)
        index = posixpath.join(str(args.data), "docs.json")

        def source(progress):
            fs, path = fsspec.url_to_fs(index)
            try:
                cache = json.loads(fs.cat_file(path))
            except (FileNotFoundError, ValueError):
                cache = {}
            docs = documents(root, progress, n_workers=n_workers, cache=cache)
            with fs.open(path, "w") as f:
                json.dump(cache, f)
            return docs

        return source

    def __init__(self, engine, worker, args=None):
        super().__init__(engine, worker, args)
        self.layers = list(self.LAYERS)
        self.tiles = {}  # slot -> DocWindow
        self.center = None
        self.focus = None  # the window keys go to, when the mouse is over none: the one last clicked
        self.capture = None  # the window a drag began in

    def __click_point__(self, i, now):
        """Left click on a document: open it in the center."""
        self.place(i, "center")

    def __context_point__(self, i, pos):
        """Right click on a document -- on the disk, or in the points menu: where to place it, as a `TilePicker` at the mouse."""
        self.popup = TilePicker(lambda slot: self.place(i, slot), pos)

    def __point_datetime__(self, i):
        mtime = self.snap.points[i].get("mtime") if self.snap.points else None
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else None

    def __animating__(self):
        return self.windows()

    # 2. Placing ----------------------------------------------------------------

    def windows(self):
        return [*self.tiles.values(), *([self.center] if self.center else [])]

    def place(self, i, slot):
        """Open document *i* at *slot* -- "center", or one of SLOTS -- in place of any window there, its layer brought to the front."""
        point = self.snap.points[i]
        url, kind = point.get("path", self.snap.summaries[i]), point.get("kind", "markdown")
        when = self.__point_datetime__(i)
        window = DocWindow(self.snap.summaries[i], " · ".join(p for p in (when, url) if p))
        window.title_color = self.title_color(i)
        if slot == "center":
            self.center = window
            self.bring_to_front("center")
        else:
            for other in [s for s in self.tiles if _overlap_(s, slot)]:
                del self.tiles[other]
            self.tiles[slot] = window
            self.bring_to_front("tiles")
        self.focus = window
        self.run_async(lambda: load(url, kind), window.set_doc)

    def area(self):
        """Where windows go: between the top bar and the status line."""
        w, h = self.screen.get_size()
        return pygame.Rect(8, 36, w - 16, h - 36 - 28)

    def slot_rect(self, slot):
        a = self.area()
        if slot == "center":
            return a
        fx, fy, fw, fh = SLOTS[slot]
        r = pygame.Rect(a.x + round(fx * a.width), a.y + round(fy * a.height), round(fw * a.width), round(fh * a.height))
        return r.inflate(-GAP, -GAP)

    def _close_(self, window):
        if window is self.center:
            self.center = None
        self.tiles = {s: w for s, w in self.tiles.items() if w is not window}
        if self.focus is window:
            self.focus = None
        if self.capture is window:
            self.capture = None

    def _window_at_(self, layer, pos):
        if layer == "center":
            return self.center if self.center and self.center.rect.collidepoint(pos) else None
        return next((w for w in self.tiles.values() if w.rect.collidepoint(pos)), None)

    # 3. Layers -------------------------------------------------------------------

    def __draw_layer__(self, layer, now):
        if layer == "center":
            if self.center:
                self.center.draw(self.screen, self.slot_rect("center"), self.font, self.small)
        else:
            for slot, window in self.tiles.items():
                window.draw(self.screen, self.slot_rect(slot), self.font, self.small)

    def __layer_covers__(self, layer, pos):
        return self._window_at_(layer, pos) is not None

    def __layer_event__(self, layer, ev, now):
        if layer is None:
            # Not over a window of the front layer at the mouse: a drag one began, or the keys of the last one clicked.
            if self.capture is not None and ev.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONUP):
                window = self.capture
                if ev.type == pygame.MOUSEBUTTONUP:
                    self.capture = None
                window.handle(ev)
                return True
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self.focus = None  # a click on the disk: its keys are the disk's again
                return False
            if self.focus is not None and ev.type == pygame.KEYDOWN and ev.key not in (pygame.K_HOME,):
                if self.focus.handle(ev) == "close":
                    self._close_(self.focus)
                return True
            return False
        window = self._window_at_(layer, getattr(ev, "pos", None) or pygame.mouse.get_pos())
        if window is None:
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.focus = window
            if ev.button == 1:
                self.capture = window
        if ev.type == pygame.MOUSEMOTION and self.capture is None:
            return True
        if window.handle(ev) == "close":
            self._close_(window)
        return True
