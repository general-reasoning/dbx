"""Documents in a window: markdown laid out with its diagrams, pdfs as their pages.

A `DocWindow` shows one document -- a `MarkdownDoc` or a `PdfDoc` -- scrolled
both ways and zoomed by + and -. A document lays itself out for a width and a
zoom (`layout`) and draws what of it is in view (`draw`); the window keeps the
scroll, and asks again when its width or the zoom changes.

Mermaid diagrams are rendered by the mermaid CLI, ``mmdc`` -- or the one
HYP_MMDC names -- here, off the UI thread, and kept under ~/.hyp/mermaid; with
no ``mmdc``, a diagram is shown as its source. Nothing is sent anywhere.
"""

import bisect
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field

import pygame

from .fonts import Font

SANS = "helveticaneue,helvetica,arial"
MONO = "menlo,monaco,couriernew"
BASE_PT = 15  # body text at zoom 1
ZOOM_STEP = 1.15
ZOOM_RANGE = (0.3, 6.0)
SCROLL_PX = 48  # per wheel notch or arrow key
MARKDOWN_LIMIT = 400_000  # characters laid out of a markdown; the rest is cut

TEXT = (205, 212, 225)
HEAD = (236, 239, 245)
DIM = (150, 160, 175)
LINK = (110, 175, 235)
CODE_BG = (32, 37, 48)
RULE = (60, 68, 84)
QUOTE = (80, 92, 115)
EDGE = (250, 200, 90)
WINDOW_BG = (24, 28, 36)
BAR = (70, 80, 100)

# pdfium is not thread-safe: every call into it, from any thread, holds this.
PDFIUM_LOCK = threading.RLock()


class Fonts:
    """pygame fonts by size and style, made once each."""

    def __init__(self):
        self._made = {}

    def get(self, size, bold=False, italic=False, mono=False):
        key = (max(6, round(size)), bold, italic, mono)
        if key not in self._made:
            self._made[key] = Font(MONO if mono else SANS, key[0], bold=bold, italic=italic)  # what it lacks, from a fallback
        return self._made[key]


FONTS = Fonts()


# -- markdown ----------------------------------------------------------------


@dataclass
class Block:
    """One block of a markdown: a heading, paragraph, code, mermaid diagram, rule or table."""

    kind: str
    spans: list = field(default_factory=list)  # [(text, (bold, italic, code, link))]
    text: str = ""
    level: int = 0
    indent: int = 0  # list nesting
    quote: int = 0  # blockquote nesting
    prefix: str = None  # a list item's bullet or number, hung before its first block
    rows: list = field(default_factory=list)  # a table's rows of cells' spans, the header first


def spans_of(inline):
    """An inline token's text as ``[(text, (bold, italic, code, link))]``."""
    out, bold, italic, link = [], 0, 0, 0
    for c in inline.children or []:
        style = (bold > 0, italic > 0, False, link > 0)
        if c.type in ("text", "html_inline"):
            out.append((c.content, style))
        elif c.type == "softbreak":
            out.append((" ", style))
        elif c.type == "hardbreak":
            out.append(("\n", style))
        elif c.type == "code_inline":
            out.append((c.content, (bold > 0, italic > 0, True, link > 0)))
        elif c.type == "image":
            out.append((f"[{c.content or 'image'}]", (False, True, False, False)))
        elif c.type == "strong_open":
            bold += 1
        elif c.type == "strong_close":
            bold -= 1
        elif c.type == "em_open":
            italic += 1
        elif c.type == "em_close":
            italic -= 1
        elif c.type == "link_open":
            link += 1
        elif c.type == "link_close":
            link -= 1
    return out


def front_matter(text):
    """``(fields, body)``: *text*'s yaml front matter -- a dict, or None where it has none -- and the markdown after it."""
    import yaml

    m = re.match(r"---[ \t]*\n(.*?\n)?---[ \t]*(\n|$)", text, re.S)
    if not m:
        return None, text
    try:
        fields = yaml.safe_load(m.group(1) or "")
    except yaml.YAMLError:
        return None, text
    return (fields if isinstance(fields, dict) else None), text[m.end():]


def parse_markdown(text):
    """*text*'s blocks, in order: commonmark, with tables and strikethrough, by markdown-it; its front matter left out."""
    from markdown_it import MarkdownIt

    fields, body = front_matter(text)
    text = body if fields is not None else text
    tokens = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"]).parse(text)
    blocks, lists, quote, prefix = [], [], 0, None

    def hung():
        nonlocal prefix
        p, prefix = prefix, None
        return p

    i = 0
    while i < len(tokens):
        t = tokens[i]
        where = dict(indent=len(lists), quote=quote)
        if t.type == "heading_open":
            blocks.append(Block("heading", spans_of(tokens[i + 1]), level=int(t.tag[1]), **where))
            i += 3
            continue
        if t.type == "paragraph_open":
            blocks.append(Block("para", spans_of(tokens[i + 1]), prefix=hung(), **where))
            i += 3
            continue
        if t.type in ("bullet_list_open", "ordered_list_open"):
            lists.append([t.type == "ordered_list_open", int(t.attrGet("start") or 1)])
        elif t.type in ("bullet_list_close", "ordered_list_close"):
            lists.pop()
        elif t.type == "list_item_open":
            ordered, n = lists[-1]
            prefix = f"{n}." if ordered else "•–·"[(len(lists) - 1) % 3]
            if ordered:
                lists[-1][1] += 1
        elif t.type == "blockquote_open":
            quote += 1
        elif t.type == "blockquote_close":
            quote -= 1
        elif t.type in ("fence", "code_block"):
            lang = (t.info or "").strip().split()[0].lower() if (t.info or "").strip() else ""
            blocks.append(Block("mermaid" if lang == "mermaid" else "code", text=t.content.rstrip("\n"),
                                prefix=hung(), **where))
        elif t.type == "html_block":
            blocks.append(Block("code", text=t.content.rstrip("\n"), prefix=hung(), **where))
        elif t.type == "hr":
            blocks.append(Block("rule", **where))
        elif t.type == "table_open":
            rows = []
            i += 1
            while i < len(tokens) and tokens[i].type != "table_close":
                if tokens[i].type == "tr_open":
                    rows.append([])
                elif tokens[i].type == "inline" and rows:
                    rows[-1].append(spans_of(tokens[i]))
                i += 1
            blocks.append(Block("table", rows=rows, **where))
        i += 1
    return blocks


@dataclass(frozen=True)
class Run:
    """A piece of text as drawn: where, in what font, and of which block -- what a selection is made of."""

    x: int
    y: int
    w: int
    h: int
    text: str
    font: object
    block: int  # the markdown block it is of: runs of one block on lines of their own are one paragraph
    hard: bool  # its block's lines are lines -- code, a table -- not a paragraph wrapped


class Laid:
    """What a document draws, as laid out: items sorted by top, each ``(top, bottom, draw)``; and its text, as `Run`s in reading order."""

    def __init__(self):
        self.items = []
        self.width = 0
        self.height = 0
        self.runs = []  # the text drawn, in reading order
        self.block, self.hard = 0, False  # what the runs added now are of: set as each block is laid out

    def add(self, top, bottom, fn):
        self.items.append((top, bottom, fn))
        self.width = max(self.width, getattr(fn, "right", 0))

    def surf(self, surface, x, y):
        def draw(dst, ox, oy):
            dst.blit(surface, (ox + x, oy + y))

        draw.right = x + surface.get_width()
        self.add(y, y + surface.get_height(), draw)

    def rect(self, x, y, w, h, color, width=0, radius=0):
        def draw(dst, ox, oy):
            pygame.draw.rect(dst, color, (ox + x, oy + y, w, h), width, border_radius=radius)

        draw.right = x + w
        self.add(y, y + h, draw)

    def run(self, x, y, img, text, font):
        """Record *text*, drawn as *img* at (*x*, *y*) with *font*, as a `Run` of the block being laid out."""
        self.runs.append(Run(x, y, img.get_width(), img.get_height(), text, font, self.block, self.hard))

    def text_at(self, x, y, near=False):
        """The text position -- ``(run, character)`` -- at content (*x*, *y*): on a run, or with *near* the nearest; None off them all."""
        if not self.runs:
            return None
        on = [i for i, r in enumerate(self.runs) if r.y <= y < r.y + r.h and r.x - 4 <= x <= r.x + r.w + 4]
        if on:
            i = on[0]
        elif near:
            i = min(range(len(self.runs)), key=lambda k: (_gap_(y, self.runs[k].y, self.runs[k].y + self.runs[k].h),
                                                           _gap_(x, self.runs[k].x, self.runs[k].x + self.runs[k].w)))
        else:
            return None
        r = self.runs[i]
        edges = [r.font.size(r.text[:c])[0] for c in range(len(r.text) + 1)]
        return i, min(range(len(edges)), key=lambda c: abs(r.x + edges[c] - x))

    def text_between(self, a, b):
        """The text from position *a* to *b*, as it reads: a paragraph's wrapped lines joined by a space, code's kept, blocks apart."""
        (i, c), (j, d) = sorted((a, b))
        out = []
        for k in range(i, j + 1):
            r = self.runs[k]
            piece = r.text[c if k == i else 0: d if k == j else len(r.text)]
            if k > i:
                prev = self.runs[k - 1]
                if r.block != prev.block:
                    out.append("\n\n")
                elif r.y != prev.y:
                    out.append("\n" if r.hard else " ")
                elif r.x > prev.x + prev.w + 1:
                    out.append("\t" if r.hard else " ")
            out.append(piece)
        return "".join(out)

    def finish(self):
        self.items.sort(key=lambda t: t[0])
        self._tops = [t[0] for t in self.items]
        # The tallest item: what starts this far above the view may still reach into it.
        self._reach = max((b - t for t, b, _ in self.items), default=0)

    def draw(self, dst, ox, oy, clip):
        lo = bisect.bisect_left(self._tops, clip.top - oy - self._reach)
        hi = bisect.bisect_right(self._tops, clip.bottom - oy)
        for top, bottom, fn in self.items[lo:hi]:
            if oy + bottom >= clip.top:
                fn(dst, ox, oy)


def flow(laid, spans, x0, y, width, size, bold=False, color=TEXT):
    """Lay *spans* out as wrapped lines from (*x0*, *y*) within *width*; the y below them.

    Runs of a style are rendered whole, a line at a time. A word wider than
    *width* -- a long url, say -- is not broken: it runs past, to be scrolled to.
    """
    line, x, line_h = [], x0, FONTS.get(size, bold).get_linesize()

    def flush():
        nonlocal line, x, y, line_h
        for rx, text, font, col, code in line:
            img = font.render(text, True, col)
            dy = (line_h - img.get_height()) // 2
            if code:
                laid.rect(rx - 2, y + dy - 1, img.get_width() + 4, img.get_height() + 2, CODE_BG, radius=3)
            laid.surf(img, rx, y + dy)
            laid.run(rx, y + dy, img, text, font)
        y += line_h
        line, x, line_h = [], x0, FONTS.get(size, bold).get_linesize()

    space = False
    for text, (b, it, code, link) in spans:
        font = FONTS.get(size * (0.9 if code else 1.0), bold or b, it, mono=code)
        col = LINK if link else color
        for piece in re.findall(r"\n|[^\S\n]+|[^\s]+", text):
            if piece == "\n":
                flush()
                space = False
                continue
            if piece.isspace():
                space = bool(line)
                continue
            gap = font.size(" ")[0] if space else 0
            w = font.size(piece)[0]
            if line and x + gap + w > x0 + width:
                flush()
                gap = 0
            last = line[-1] if line else None
            if last and last[2] is font and last[3] == col and last[4] == code:
                line[-1] = (last[0], last[1] + (" " if gap else "") + piece, font, col, code)
            else:
                line.append((x + gap, piece, font, col, code))
            x += gap + w
            line_h = max(line_h, font.get_linesize())
            space = False
    if line:
        flush()
    return y


class MarkdownDoc:
    """A markdown, laid out with its headings, lists, quotes, code, tables and mermaid diagrams."""

    def __init__(self, text, mermaid=None):
        cut = len(text) > MARKDOWN_LIMIT
        self.blocks = parse_markdown(text[:MARKDOWN_LIMIT])
        if cut:
            self.blocks.append(Block("para", [(f"… cut short: the first {MARKDOWN_LIMIT:,} of {len(text):,} characters", (False, True, False, False))]))
        self.mermaid = mermaid if mermaid is not None else Mermaid.shared()
        self.version = 0  # bumped when a diagram arrives: lay out again
        self._laid = None

    def pending(self):
        """Whether a diagram is being rendered, or one has arrived since the last layout: whether to keep redrawing."""
        stale = self._laid is not None and self._laid[0][2] != self.version
        return stale or any(self.mermaid.state(b.text) == "pending" for b in self.blocks if b.kind == "mermaid")

    def _diagram_ready_(self):
        self.version += 1

    def layout(self, width, zoom):
        key = (width, zoom, self.version)
        if self._laid is not None and self._laid[0] == key:
            return self._laid[1]
        laid, size = Laid(), BASE_PT * zoom
        m = round(16 * zoom)
        y = m
        for n, b in enumerate(self.blocks):
            laid.block, laid.hard = n, b.kind in ("code", "mermaid", "table")
            x0 = m + round(24 * zoom) * b.indent + round(16 * zoom) * b.quote
            avail = max(80, width - x0 - m)
            top = y
            if b.prefix:
                img = FONTS.get(size).render(b.prefix, True, DIM)
                laid.surf(img, x0 - img.get_width() - round(6 * zoom), y)
            if b.kind == "heading":
                hsize = size * {1: 2.0, 2: 1.6, 3: 1.3, 4: 1.15}.get(b.level, 1.0)
                y += round(hsize * 0.6) if n else 0
                y = flow(laid, b.spans, x0, y, avail, hsize, bold=True, color=HEAD)
                if b.level <= 2:
                    laid.rect(x0, y + 2, avail, 1, RULE)
                    y += 4
                y += round(size * 0.4)
            elif b.kind == "para":
                y = flow(laid, b.spans, x0, y, avail, size) + round(size * 0.6)
            elif b.kind == "code":
                y = self._code_(laid, b.text, x0, y, avail, zoom) + round(size * 0.6)
            elif b.kind == "mermaid":
                y = self._mermaid_(laid, b.text, x0, y, avail, zoom) + round(size * 0.6)
            elif b.kind == "rule":
                laid.rect(x0, y + round(size * 0.5), avail, 1, RULE)
                y += round(size * 1.2)
            elif b.kind == "table":
                y = self._table_(laid, b.rows, x0, y, size) + round(size * 0.6)
            for q in range(b.quote):
                laid.rect(m + round(16 * zoom) * q + round(4 * zoom), top, max(2, round(3 * zoom)), y - top - round(size * 0.4), QUOTE)
        laid.height = y + m
        laid.finish()
        self._laid = (key, laid)
        return laid

    def _code_(self, laid, text, x0, y, avail, zoom, note=None):
        font, pad = FONTS.get(BASE_PT * 0.9 * zoom, mono=True), round(8 * zoom)
        lines = text.split("\n")
        if note:
            lines = [None] + lines
        imgs = [FONTS.get(BASE_PT * 0.85 * zoom, italic=True).render(note, True, DIM) if ln is None else font.render(ln.expandtabs(4) or " ", True, TEXT)
                for ln in lines]
        h = sum(i.get_height() for i in imgs) + 2 * pad
        laid.rect(x0, y, max(avail, max(i.get_width() for i in imgs) + 2 * pad), h, CODE_BG, radius=4)
        ty = y + pad
        for ln, img in zip(lines, imgs):
            laid.surf(img, x0 + pad, ty)
            if ln is not None:
                laid.run(x0 + pad, ty, img, ln.expandtabs(4), font)
            ty += img.get_height()
        return y + h

    def _mermaid_(self, laid, source, x0, y, avail, zoom):
        state = self.mermaid.state(source)
        if state is None:
            state = self.mermaid.render(source, self._diagram_ready_)
        if state == "done":
            img = self.mermaid.image(source)
            if img is not None:
                s = 0.5 * zoom  # rendered at twice the size, for a sharp zoom
                img = pygame.transform.smoothscale(img, (max(1, round(img.get_width() * s)), max(1, round(img.get_height() * s))))
                laid.surf(img, x0, y)
                return y + img.get_height()
        note = {"pending": "mermaid: rendering the diagram …",
                "unavailable": "mermaid: install the mermaid CLI to see this as a diagram -- npm install -g @mermaid-js/mermaid-cli"
                }.get(state, f"mermaid: could not render: {self.mermaid.error(source)}")
        return self._code_(laid, source, x0, y, avail, zoom, note=note)

    def _table_(self, laid, rows, x0, y, size):
        if not rows:
            return y
        pad = round(size * 0.5)
        cells = [[(spans, i == 0) for spans in row] for i, row in enumerate(rows)]
        ncol = max(len(r) for r in cells)

        def text_of(spans):
            return "".join(t for t, _ in spans).replace("\n", " ")

        def img(spans, head):
            return FONTS.get(size, bold=head).render(text_of(spans) or " ", True, HEAD if head else TEXT)

        imgs = [[img(s, h) for s, h in row] + [None] * (ncol - len(row)) for row in cells]
        texts = [[(text_of(s), FONTS.get(size, bold=h)) for s, h in row] + [None] * (ncol - len(row)) for row in cells]
        widths = [max((r[c].get_width() for r in imgs if r[c] is not None), default=0) + 2 * pad for c in range(ncol)]
        row_h = FONTS.get(size, bold=True).get_linesize() + pad
        for r, row in enumerate(imgs):
            x = x0
            if r == 0:
                laid.rect(x0, y, sum(widths), row_h, CODE_BG)
            for c, cell in enumerate(row):
                laid.rect(x, y, widths[c], row_h, RULE, width=1)
                if cell is not None:
                    laid.surf(cell, x + pad, y + pad // 2)
                    laid.run(x + pad, y + pad // 2, cell, *texts[r][c])
                x += widths[c]
            y += row_h
        return y

    def draw(self, dst, ox, oy, clip):
        self._laid[1].draw(dst, ox, oy, clip)


# -- mermaid -----------------------------------------------------------------


class Mermaid:
    """Mermaid diagrams as images: rendered by ``mmdc`` off the UI thread, kept by their source's hash."""

    CACHE = os.path.join(os.path.expanduser("~"), ".hyp", "mermaid")
    TIMEOUT_S = 90
    _shared = None

    def __init__(self, command=None, cache=None):
        self.command = command if command is not None else (os.environ.get("HYP_MMDC") or shutil.which("mmdc"))
        self.cache = cache or self.CACHE
        self._state = {}  # sha -> "pending" | "done" | ("error", message)
        self._images = {}
        self._lock = threading.Lock()

    @classmethod
    def shared(cls):
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    @staticmethod
    def _sha_(source):
        return hashlib.sha1(source.encode()).hexdigest()

    def _png_(self, sha):
        return os.path.join(self.cache, f"{sha}.png")

    def state(self, source):
        """"done", "pending", "unavailable", "error", or None for a diagram not yet asked for."""
        sha = self._sha_(source)
        st = self._state.get(sha)
        if st is None and os.path.exists(self._png_(sha)):
            self._state[sha] = st = "done"
        if st is None and not self.command:
            return "unavailable"
        return st[0] if isinstance(st, tuple) else st

    def error(self, source):
        st = self._state.get(self._sha_(source))
        return st[1] if isinstance(st, tuple) else ""

    def image(self, source):
        sha = self._sha_(source)
        if sha not in self._images:
            try:
                self._images[sha] = pygame.image.load(self._png_(sha))
            except (pygame.error, FileNotFoundError):
                return None
        return self._images[sha]

    def render(self, source, done):
        """Start rendering *source*, calling *done()* when it is rendered or has failed; its state now."""
        if not self.command:
            return "unavailable"
        sha = self._sha_(source)
        with self._lock:
            if self._state.get(sha) is not None:
                return self.state(source)
            self._state[sha] = "pending"

        def run():
            os.makedirs(self.cache, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                src, out = os.path.join(tmp, "d.mmd"), os.path.join(tmp, "d.png")
                with open(src, "w") as f:
                    f.write(source)
                try:
                    r = subprocess.run([self.command, "-i", src, "-o", out, "-t", "dark", "-b", "transparent", "-s", "2", "-q"],
                                       capture_output=True, text=True, timeout=self.TIMEOUT_S)
                    if r.returncode == 0 and os.path.exists(out):
                        shutil.move(out, self._png_(sha))
                        self._state[sha] = "done"
                    else:
                        msg = (r.stderr or r.stdout or f"exit {r.returncode}").strip().splitlines()
                        self._state[sha] = ("error", msg[-1] if msg else f"exit {r.returncode}")
                except (OSError, subprocess.TimeoutExpired) as e:
                    self._state[sha] = ("error", f"{type(e).__name__}: {e}")
            done()

        threading.Thread(target=run, daemon=True).start()
        return "pending"


# -- pdf ---------------------------------------------------------------------


def pdf_text(data, pages=30):
    """The text of the first *pages* pages of the pdf *data*, by pdfium."""
    import pypdfium2 as pdfium

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(data)
        try:
            out = []
            for i in range(min(len(pdf), pages)):
                page = pdf[i]
                tp = page.get_textpage()
                out.append(tp.get_text_range())
                tp.close()
                page.close()
            return "\n".join(out)
        finally:
            pdf.close()


class PdfDoc:
    """A pdf's pages, one under another: as wide as the window at zoom 1, rendered by pdfium as they come into view."""

    GAP = 12
    KEEP = 16  # rendered pages kept

    def __init__(self, data):
        import pypdfium2 as pdfium

        self._data = data  # pdfium reads from it for as long as the document is open
        with PDFIUM_LOCK:
            self.pdf = pdfium.PdfDocument(data)
            self.sizes = [self.pdf.get_page_size(i) for i in range(len(self.pdf))]
        self.version = 0
        self._rendered = {}  # (page, scale) -> surface, the most recent last
        self._key = None

    def pending(self):
        return False

    def layout(self, width, zoom):
        widest = max((w for w, _ in self.sizes), default=1)
        self.scale = zoom * max(50, width - 2 * self.GAP) / widest
        self.tops, y = [], self.GAP
        for _, h in self.sizes:
            self.tops.append(y)
            y += round(h * self.scale) + self.GAP
        laid = Laid()
        laid.width, laid.height = round(widest * self.scale) + 2 * self.GAP, y
        return laid

    def _page_(self, i):
        key = (i, round(self.scale, 4))
        if key in self._rendered:
            self._rendered[key] = self._rendered.pop(key)  # the most recent last
            return self._rendered[key]
        with PDFIUM_LOCK:
            page = self.pdf[i]
            pil = page.render(scale=self.scale).to_pil().convert("RGB")
            page.close()
        img = pygame.image.frombytes(pil.tobytes(), pil.size, "RGB")
        self._rendered[key] = img
        while len(self._rendered) > self.KEEP:
            self._rendered.pop(next(iter(self._rendered)))
        return img

    def draw(self, dst, ox, oy, clip):
        for i, top in enumerate(self.tops):
            h = round(self.sizes[i][1] * self.scale)
            if oy + top + h < clip.top:
                continue
            if oy + top > clip.bottom:
                break
            dst.blit(self._page_(i), (ox + self.GAP, oy + top))


class Message:
    """A note in place of a document: what is being read, or why it could not be."""

    def __init__(self, text):
        self.text = text
        self.version = 0

    def pending(self):
        return False

    def layout(self, width, zoom):
        self._laid = Laid()
        y = flow(self._laid, [(self.text, (False, True, False, False))], 16, 16, max(80, width - 32), BASE_PT * zoom, color=DIM)
        self._laid.height = y + 16
        self._laid.finish()
        return self._laid

    def draw(self, dst, ox, oy, clip):
        self._laid.draw(dst, ox, oy, clip)


# -- the window --------------------------------------------------------------


SELECTION = (60, 90, 140)  # the highlight beneath selected text being edited


class TextEdit:
    """Text being edited, with a cursor, a selection and undo: what `LineEdit` and `AreaEdit` draw.

    Typing and pasting (Cmd+V) insert at the cursor, in place of what is
    selected. Left and Right move the cursor, Option with them a word, Cmd
    with them -- and Home and End -- to either end, and Up and Down a line;
    Shift with any of them selects as it goes, and Cmd+A selects all.
    Backspace and Delete remove the selection, or else before and after the
    cursor, Option+Backspace a word; Cmd+C copies the selection and Cmd+X cuts
    it. Cmd+Z -- or Ctrl+Z -- undoes, a run of typing at once, and Cmd+Shift+Z
    or Ctrl+Y redoes. With the mouse: a press places the cursor, a drag
    selects, Shift+click extends the selection, a double click selects a word
    (`press`, `drag`, at a character position: the subclass says which).
    Enter submits, Esc cancels: `handle` says which.
    """

    DOUBLE_CLICK_S = 0.4
    UNDO_LIMIT = 200

    def __init__(self, text=""):
        self.text = text
        self.cursor = len(text)
        self.anchor = None  # where the selection began, the cursor its other end; None for none
        self._undo_, self._redo_ = [], []  # (text, cursor) before each edit, and those undone
        self._typing_ = False  # the last edit was typing: the next typed joins it, one undo
        self._last_press_ = (0.0, -1)  # (when, where): what tells a double click

    def selection(self):
        """The selection, as (start, end), start < end; None for none."""
        if self.anchor is None or self.anchor == self.cursor:
            return None
        return min(self.anchor, self.cursor), max(self.anchor, self.cursor)

    def set_text(self, text):
        """Put *text* in, the cursor at its end, with nothing to undo."""
        self.text, self.cursor, self.anchor = text, len(text), None
        self._undo_, self._redo_, self._typing_ = [], [], False

    def handle(self, ev):
        """Returns "submit", "cancel" or None."""
        if ev.type == pygame.TEXTINPUT:
            self._insert_(ev.text, typing=True)
            return None
        if ev.type != pygame.KEYDOWN:
            return None
        mods = pygame.key.get_mods() | getattr(ev, "mod", 0)
        meta, ctrl = mods & pygame.KMOD_META, mods & pygame.KMOD_CTRL
        cmd, alt, shift = meta or ctrl, mods & pygame.KMOD_ALT, mods & pygame.KMOD_SHIFT
        k, t, c = ev.key, self.text, self.cursor
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return "submit"
        if k == pygame.K_ESCAPE:
            return "cancel"
        if k == pygame.K_z and cmd:
            self._redo_step_() if shift else self._undo_step_()
            return None
        if k == pygame.K_y and ctrl:
            self._redo_step_()
            return None
        moves = {pygame.K_LEFT: lambda: 0 if cmd else self._word_start_(c) if alt else max(0, c - 1),
                 pygame.K_RIGHT: lambda: len(t) if cmd else self._word_end_(c) if alt else min(len(t), c + 1),
                 pygame.K_HOME: lambda: 0, pygame.K_END: lambda: len(t),
                 pygame.K_UP: lambda: 0 if cmd else self._vertical_(c, -1),
                 pygame.K_DOWN: lambda: len(t) if cmd else self._vertical_(c, +1)}
        if k in moves:
            self._typing_ = False
            sel = self.selection()
            if shift:
                self.anchor = c if self.anchor is None else self.anchor
                self.cursor = moves[k]()
            elif sel is not None and not (cmd or alt) and k in (pygame.K_LEFT, pygame.K_RIGHT):
                self.cursor, self.anchor = sel[0] if k == pygame.K_LEFT else sel[1], None  # collapsed to its edge
            else:
                self.cursor, self.anchor = moves[k](), None
        elif k == pygame.K_a and cmd:
            self.anchor, self.cursor = 0, len(t)
        elif k in (pygame.K_BACKSPACE, pygame.K_DELETE) and self.selection() is not None:
            self._edit_(*self._without_selection_())
        elif k == pygame.K_BACKSPACE:
            start = 0 if cmd else self._word_start_(c) if alt else max(0, c - 1)
            if start < c:
                self._edit_(t[:start] + t[c:], start)
        elif k == pygame.K_DELETE:
            if c < len(t):
                self._edit_(t[:c] + t[c + 1:], c)
        elif k in (pygame.K_c, pygame.K_x) and cmd and self.selection() is not None:
            from .app import _copy_

            a, b = self.selection()
            _copy_(t[a:b])
            if k == pygame.K_x:
                self._edit_(*self._without_selection_())
        elif k == pygame.K_v and cmd:
            try:
                self._insert_(self._pasted_(pygame.scrap.get_text() or ""))
            except Exception:
                pass
        return None

    def press(self, at, shift=False, now=None):
        """The mouse pressed at character position *at*: the cursor placed there -- the selection extended to it, with *shift*; a word selected, pressed twice."""
        self._typing_ = False
        now = time.monotonic() if now is None else now
        when, where = self._last_press_
        self._last_press_ = (now, at)
        if not shift and at == where and now - when < self.DOUBLE_CLICK_S:
            self.anchor, self.cursor = self._word_at_(at)
            return
        if shift:
            self.anchor = self.cursor if self.anchor is None else self.anchor
        else:
            self.anchor = at  # where a drag from here selects from
        self.cursor = at

    def drag(self, at):
        """The mouse dragged to character position *at*, pressed: the selection from where it was pressed to there."""
        self.cursor = at

    def _vertical_(self, c, step):
        """Where Up (*step* -1) or Down (+1) takes the cursor from *c*: here, one line, the ends."""
        return 0 if step < 0 else len(self.text)

    def _pasted_(self, text):
        """What pasted *text* inserts: here, one line of it."""
        return " ".join(text.split())

    def _edit_(self, text, cursor, typing=False):
        """Make the text *text*, the cursor at *cursor*: undoable -- a run of typing as one step."""
        if not (typing and self._typing_):
            self._undo_.append((self.text, self.cursor))
            del self._undo_[: -self.UNDO_LIMIT]
        self._redo_.clear()
        self.text, self.cursor, self.anchor, self._typing_ = text, cursor, None, typing

    def _undo_step_(self):
        if self._undo_:
            self._redo_.append((self.text, self.cursor))
            self.text, self.cursor = self._undo_.pop()
            self.anchor, self._typing_ = None, False

    def _redo_step_(self):
        if self._redo_:
            self._undo_.append((self.text, self.cursor))
            self.text, self.cursor = self._redo_.pop()
            self.anchor, self._typing_ = None, False

    def _without_selection_(self):
        a, b = self.selection()
        return self.text[:a] + self.text[b:], a

    def _insert_(self, text, typing=False):
        if self.selection() is not None:
            base, at = self._without_selection_()
            typing = False  # replacing a selection is a step of its own
        else:
            base, at = self.text, self.cursor
        self._edit_(base[:at] + text + base[at:], at + len(text), typing=typing)

    def _word_at_(self, c):
        """The word *c* is in or at the edge of, as (start, end): grown from *c* out to the spaces either side."""
        t, a, b = self.text, c, c
        while a > 0 and not t[a - 1].isspace():
            a -= 1
        while b < len(t) and not t[b].isspace():
            b += 1
        return a, b

    def _word_start_(self, c):
        t = self.text
        while c > 0 and t[c - 1].isspace():
            c -= 1
        while c > 0 and not t[c - 1].isspace():
            c -= 1
        return c

    def _word_end_(self, c):
        t = self.text
        while c < len(t) and t[c].isspace():
            c += 1
        while c < len(t) and not t[c].isspace():
            c += 1
        return c


class LineEdit(TextEdit):
    """One line of `TextEdit`: what a window's title is while it is renamed. `draw` scrolls it, where it is wider than its room, so the cursor shows."""

    def __init__(self, text=""):
        super().__init__(text)
        self._laid_ = (0, "", 0)  # (x, what precedes the text as drawn, the first character drawn): what a click is placed by

    def press(self, font, x, shift=False, now=None):
        """The mouse pressed at screen *x*, on the text as last drawn: `TextEdit.press`."""
        super().press(self._index_at_(font, x), shift, now)

    def drag(self, font, x):
        super().drag(self._index_at_(font, x))

    def draw(self, surf, font, x, y, room, color, prefix=""):
        """Draw *prefix*, then the text, from (*x*, *y*) in *room* pixels, its selection and the cursor; scrolled so the cursor shows. The rect drawn."""
        t, c = self.text, self.cursor
        first = 0
        while first < c and font.size(prefix + ("…" if first else "") + t[first:c])[0] > room - 4:
            first += 1  # scrolled right, a character at a time, till the cursor is in the room
        lead = prefix + ("…" if first else "")
        sel = self.selection()
        lh = font.get_linesize()
        if sel is not None:
            a, b = max(sel[0], first), sel[1]
            xa, xb = x + font.size(lead + t[first:a])[0], min(x + room, x + font.size(lead + t[first:b])[0])
            if xb > xa:
                pygame.draw.rect(surf, SELECTION, pygame.Rect(xa, y, xb - xa, lh))
        shown = _fit_(font, lead + t[first:], room)
        rect = surf.blit(font.render(shown, True, color), (x, y))
        cx = x + font.size(lead + t[first:c])[0]
        if sel is None and time.time() % 1 < 0.6:
            pygame.draw.line(surf, color, (cx, y + 2), (cx, y + lh - 2), 2)
        self._laid_ = (x, lead, first)
        return rect.union(pygame.Rect(x, y, room, lh))

    def _index_at_(self, font, x):
        """The character boundary nearest screen *x*, on the text as last drawn."""
        x0, lead, first = self._laid_
        t = self.text
        edges = [x0 + font.size(lead + t[first:i])[0] for i in range(first, len(t) + 1)]
        return first + min(range(len(edges)), key=lambda j: abs(edges[j] - x))


class AreaEdit(TextEdit):
    """`TextEdit` of several lines, wrapped to a width: what a Claude session's box holds.

    Shift+Enter starts a new line; pasted text keeps its lines. `rows` wraps
    it -- a long word broken where it must be -- and `draw` shows at most
    so many rows, scrolled to keep the cursor among them.
    """

    def __init__(self, text=""):
        super().__init__(text)
        self._rows_key_, self._rows_ = None, []
        self._laid_ = None  # (x, y, line height, the first row shown, the font): what a click is placed by

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) \
                and (pygame.key.get_mods() | getattr(ev, "mod", 0)) & pygame.KMOD_SHIFT:
            self._insert_("\n")
            return None
        return super().handle(ev)

    def rows(self, font, width):
        """The text wrapped to *width*: its rows as (start, end) of the text, every character in one but the newlines between."""
        key = (self.text, width, id(font))
        if self._rows_key_ == key:
            return self._rows_
        rows, o = [], 0
        for para in self.text.split("\n"):
            s = 0
            while True:
                if font.size(para[s:])[0] <= width:
                    rows.append((o + s, o + len(para)))
                    break
                e = s + 1
                while e < len(para) and font.size(para[s:e + 1])[0] <= width:
                    e += 1
                space = para.rfind(" ", s, e)
                if space > s:
                    e = space + 1  # the space ends this row
                rows.append((o + s, o + e))
                s = e
            o += len(para) + 1
        self._rows_key_, self._rows_ = key, rows
        return rows

    def row_of(self, c, rows):
        """The row the cursor at *c* is on: a row wrapped where *c* is, the next one."""
        for i, (a, b) in enumerate(rows):
            if c < b or (c == b and (i + 1 == len(rows) or rows[i + 1][0] > b)):
                return i
        return len(rows) - 1

    def draw(self, surf, font, x, y, width, most, color):
        """Draw at most *most* rows from (*x*, *y*), *width* wide, its selection and the cursor; scrolled so the cursor shows."""
        rows, lh, t = self.rows(font, width), font.get_linesize(), self.text
        here = self.row_of(self.cursor, rows)
        first = 0
        if self._laid_ is not None:
            first = self._laid_[3]
        first = min(max(first, here - most + 1), here, max(0, len(rows) - most))
        sel = self.selection()
        for n, (a, b) in enumerate(rows[first: first + most]):
            ry = y + n * lh
            if sel is not None and sel[0] <= b and sel[1] >= a:
                xa = x + font.size(t[a: max(a, sel[0])])[0]
                xb = x + font.size(t[a: min(b, sel[1])])[0]
                if xb > xa:
                    pygame.draw.rect(surf, SELECTION, pygame.Rect(xa, ry, xb - xa, lh))
            surf.blit(font.render(t[a:b], True, color), (x, ry))
        if sel is None and time.time() % 1 < 0.6:
            a, _ = rows[here]
            cx, cy = x + font.size(t[a: self.cursor])[0], y + (here - first) * lh
            pygame.draw.line(surf, color, (cx, cy + 2), (cx, cy + lh - 2), 2)
        self._laid_ = (x, y, lh, first, font)

    def index_at(self, pos):
        """The character boundary nearest screen *pos*, on the rows as last drawn."""
        if self._laid_ is None:
            return self.cursor
        x, y, lh, first, font = self._laid_
        rows = self._rows_
        r = min(max(first + (pos[1] - y) // lh, 0), len(rows) - 1)
        return self._nearest_in_row_(rows[r], pos[0] - x, font)

    def press(self, pos, shift=False, now=None):
        super().press(self.index_at(pos), shift, now)

    def drag(self, pos):
        super().drag(self.index_at(pos))

    def _vertical_(self, c, step):
        if self._laid_ is None:
            return super()._vertical_(c, step)
        font, rows = self._laid_[4], self._rows_
        r = self.row_of(c, rows)
        if not 0 <= r + step < len(rows):
            return 0 if step < 0 else len(self.text)
        x = font.size(self.text[rows[r][0]: c])[0]
        return self._nearest_in_row_(rows[r + step], x, font)

    def _nearest_in_row_(self, row, x, font):
        a, b = row
        edges = [font.size(self.text[a:i])[0] for i in range(a, b + 1)]
        return a + min(range(len(edges)), key=lambda j: abs(edges[j] - x))

    def _pasted_(self, text):
        return text.replace("\r\n", "\n")


class DocWindow:
    """A document's window: its title and modification time above it, scrolled both ways, zoomed by + and -.

    Where it is -- *rect* -- its owner says, at each draw. Wheel and arrows
    scroll (shift and the wheel, or a sideways swipe, scroll across); + and -
    zoom, 0 back to 1; Home and End go to the ends; Esc or × closes it. A
    drag that starts on text selects it -- scrolling, dragged past the edge
    -- Shift+click extends the selection, a double click selects a word,
    Cmd+A all of it, and Cmd+C copies it; a drag that starts anywhere else
    pans.
    """

    DOUBLE_CLICK_S = 0.4
    SELECTION = (70, 110, 170, 110)  # the highlight over selected text, drawn over it: RGBA

    def __init__(self, title, sub=""):
        self.title = title
        self.sub = sub
        self.doc = Message("reading …")
        self.zoom = 1.0
        self.sx = self.sy = 0
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.body = pygame.Rect(0, 0, 0, 0)
        self.close = pygame.Rect(0, 0, 0, 0)
        self.title_rect = pygame.Rect(0, 0, 0, 0)  # the title's, as last drawn
        self.title_color = lambda: None
        self.press = None
        self._size = (1, 1)  # the content's, as last laid out
        self._laid_ = None  # the layout as last drawn: what a press is placed on
        self.anchor = self.cursor = None  # the selection's two ends, as (run, character) of the layout; None for none
        self.selecting = False  # pressed on text and not yet let go: a drag selects
        self._last_press_ = (0.0, None)  # (when, where): what tells a double click

    def set_doc(self, doc):
        self.doc, self.sx, self.sy = doc, 0, 0
        self.anchor = self.cursor = None

    def selection_text(self):
        """The text selected, as it reads; "" for none."""
        if self.anchor is None or self.cursor is None or self.anchor == self.cursor or self._laid_ is None:
            return ""
        return self._laid_.text_between(self.anchor, self.cursor)

    def animating(self):
        return self.doc.pending()

    def _zoom_(self, factor):
        old = self.zoom
        self.zoom = min(max(self.zoom * factor, ZOOM_RANGE[0]), ZOOM_RANGE[1]) if factor else 1.0
        r = self.zoom / old
        self.sx, self.sy = self.sx * r, self.sy * r  # the same place stays in view, near enough

    def _clamp_(self):
        w, h = self._size
        self.sx = min(max(0, self.sx), max(0, w - self.body.width))
        self.sy = min(max(0, self.sy), max(0, h - self.body.height))

    def handle(self, ev):
        """Returns "close" or None."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if ev.type == pygame.KEYDOWN:
            k, shift = ev.key, bool(ev.mod & pygame.KMOD_SHIFT)
            cmd = (ev.mod | pygame.key.get_mods()) & (pygame.KMOD_META | pygame.KMOD_CTRL)
            if k == pygame.K_c and cmd:
                from .app import _copy_

                text = self.selection_text()
                if text:
                    _copy_(text)
                return None
            if k == pygame.K_a and cmd:
                runs = self._laid_.runs if self._laid_ is not None else []
                if runs:
                    self.anchor, self.cursor = (0, 0), (len(runs) - 1, len(runs[-1].text))
                return None
            if k == pygame.K_ESCAPE:
                return "close"
            if k in (pygame.K_PLUS, pygame.K_KP_PLUS) or (k == pygame.K_EQUALS):
                self._zoom_(ZOOM_STEP)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._zoom_(1 / ZOOM_STEP)
            elif k in (pygame.K_0, pygame.K_KP0) and not shift:
                self._zoom_(0)
            elif k == pygame.K_DOWN:
                self.sy += SCROLL_PX
            elif k == pygame.K_UP:
                self.sy -= SCROLL_PX
            elif k == pygame.K_RIGHT:
                self.sx += SCROLL_PX
            elif k == pygame.K_LEFT:
                self.sx -= SCROLL_PX
            elif k in (pygame.K_PAGEDOWN, pygame.K_SPACE):
                self.sy += self.body.height * 0.9
            elif k == pygame.K_PAGEUP:
                self.sy -= self.body.height * 0.9
            elif k == pygame.K_HOME:
                self.sx = self.sy = 0
            elif k == pygame.K_END:
                self.sy = self._size[1]
            self._clamp_()
        elif ev.type == pygame.MOUSEWHEEL:
            dx, dy = getattr(ev, "precise_x", ev.x), getattr(ev, "precise_y", ev.y)
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                dx, dy = dy, 0
            self.sx += dx * SCROLL_PX
            self.sy -= dy * SCROLL_PX
            self._clamp_()
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.body.collidepoint(ev.pos):
            if not self._press_text_(ev.pos):
                self.anchor = self.cursor = None
                self.press = (ev.pos, self.sx, self.sy)
        elif ev.type == pygame.MOUSEMOTION and self.selecting:
            self._drag_text_(ev.pos)
        elif ev.type == pygame.MOUSEMOTION and self.press is not None:
            (px, py), sx, sy = self.press
            self.sx, self.sy = sx - (ev.pos[0] - px), sy - (ev.pos[1] - py)
            self._clamp_()
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.press = None
            self.selecting = False
        return None

    def draw(self, surf, rect, font, small):
        self.rect = pygame.Rect(rect)
        x, y, w, h = self.rect
        pad = 12
        pygame.draw.rect(surf, WINDOW_BG, self.rect, border_radius=8)
        pygame.draw.rect(surf, EDGE if self.press else BAR, self.rect, 1, border_radius=8)
        self.close = pygame.Rect(x + w - 30, y + 6, 22, 22)
        hot = self.close.collidepoint(pygame.mouse.get_pos())
        if hot:
            pygame.draw.rect(surf, (52, 60, 76), self.close, border_radius=4)
        c = self.close.inflate(-10, -10)
        for a, b in ((c.topleft, c.bottomright), (c.topright, c.bottomleft)):
            pygame.draw.line(surf, EDGE if hot else TEXT, a, b, 2)
        zoom = small.render(f"{self.zoom:.0%}", True, DIM)
        surf.blit(zoom, (self.close.x - zoom.get_width() - 10, y + pad))
        room = self.close.x - zoom.get_width() - 24 - (x + pad)
        title = font.render(_fit_(font, self.title, room), True, self.title_color() or EDGE)
        self.title_rect = surf.blit(title, (x + pad, y + pad - 2))  # where a click finds it
        top = y + pad + font.get_linesize()
        for line in self.sub.split("\n") if self.sub else ():  # a line each
            surf.blit(small.render(_fit_(small, line, w - 2 * pad), True, DIM), (x + pad, top))
            top += small.get_linesize()
        self.body = pygame.Rect(x + 1, top + 6, w - 2, y + h - 1 - (top + 6))
        if self.body.width < 40 or self.body.height < 20:
            return
        laid = self.doc.layout(self.body.width - 10, round(self.zoom, 4))
        self._size = (laid.width, laid.height)
        self._clamp_()
        old = surf.get_clip()
        surf.set_clip(self.body.clip(old) if old else self.body)
        self.doc.draw(surf, self.body.x - round(self.sx), self.body.y - round(self.sy), self.body)
        self._laid_ = laid
        self._highlight_(surf, laid)
        surf.set_clip(old)
        self._scrollbars_(surf)

    def _content_(self, pos):
        """Screen *pos* in the content's coordinates, as laid out."""
        return pos[0] - self.body.x + self.sx, pos[1] - self.body.y + self.sy

    def _press_text_(self, pos):
        """A press at *pos*: on text, the selection begun there -- or extended, with Shift; a word, pressed twice. Whether it was on text."""
        laid = self._laid_
        if laid is None or not laid.runs:
            return False
        shift = pygame.key.get_mods() & pygame.KMOD_SHIFT
        at = laid.text_at(*self._content_(pos), near=bool(shift and self.anchor is not None))
        if at is None:
            return False
        now = time.monotonic()
        when, where = self._last_press_
        self._last_press_ = (now, at)
        if shift and self.anchor is not None:
            self.cursor = at
        elif where == at and now - when < self.DOUBLE_CLICK_S:
            r, c = at
            text, a, b = laid.runs[r].text, c, c
            while a > 0 and not text[a - 1].isspace():
                a -= 1
            while b < len(text) and not text[b].isspace():
                b += 1
            self.anchor, self.cursor = (r, a), (r, b)
        else:
            self.anchor = self.cursor = at
        self.selecting = True
        return True

    def _drag_text_(self, pos):
        """The mouse dragged to *pos*, pressed on text: the selection to the text nearest it -- scrolled, dragged past the body's edge."""
        if pos[1] < self.body.top:
            self.sy -= SCROLL_PX
        elif pos[1] > self.body.bottom:
            self.sy += SCROLL_PX
        self._clamp_()
        if self._laid_ is not None:
            self.cursor = self._laid_.text_at(*self._content_(pos), near=True)

    def _highlight_(self, surf, laid):
        """Shade the selected text, run by run, over what is drawn."""
        if self.anchor is None or self.cursor is None or self.anchor == self.cursor:
            return
        (i, c), (j, d) = sorted((self.anchor, self.cursor))
        ox, oy = self.body.x - round(self.sx), self.body.y - round(self.sy)
        for k in range(i, min(j, len(laid.runs) - 1) + 1):
            r = laid.runs[k]
            a = r.font.size(r.text[:c])[0] if k == i else 0
            b = r.font.size(r.text[:d])[0] if k == j else r.w
            if b > a:
                shade = pygame.Surface((b - a, r.h), pygame.SRCALPHA)
                shade.fill(self.SELECTION)
                surf.blit(shade, (ox + r.x + a, oy + r.y))

    def _scrollbars_(self, surf):
        (cw, ch), b = self._size, self.body
        if ch > b.height:
            th = max(20, b.height * b.height // ch)
            ty = b.y + (b.height - th) * self.sy / max(1, ch - b.height)
            pygame.draw.rect(surf, BAR, (b.right - 6, ty, 4, th), border_radius=2)
        if cw > b.width:
            tw = max(20, b.width * b.width // cw)
            tx = b.x + (b.width - tw) * self.sx / max(1, cw - b.width)
            pygame.draw.rect(surf, BAR, (tx, b.bottom - 6, tw, 4), border_radius=2)


def _gap_(v, lo, hi):
    """How far *v* is outside [*lo*, *hi*): 0 inside."""
    return lo - v if v < lo else v - hi + 1 if v >= hi else 0


def _fit_(font, text, width):
    """*text*, cut with … to fit *width* as drawn with *font*."""
    if font.size(text)[0] <= width:
        return text
    while len(text) > 1 and font.size(text + "…")[0] > width:
        text = text[:-1]
    return text + "…"
