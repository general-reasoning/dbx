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
from dataclasses import dataclass, field

import pygame

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
            self._made[key] = pygame.font.SysFont(MONO if mono else SANS, key[0], bold=bold, italic=italic)
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


class Laid:
    """What a document draws, as laid out: items sorted by top, each ``(top, bottom, draw)``."""

    def __init__(self):
        self.items = []
        self.width = 0
        self.height = 0

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
        for img in imgs:
            laid.surf(img, x0 + pad, ty)
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

        def img(spans, head):
            text = "".join(t for t, _ in spans).replace("\n", " ")
            return FONTS.get(size, bold=head).render(text or " ", True, HEAD if head else TEXT)

        imgs = [[img(s, h) for s, h in row] + [None] * (ncol - len(row)) for row in cells]
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


class DocWindow:
    """A document's window: its title and modification time above it, scrolled both ways, zoomed by + and -.

    Where it is -- *rect* -- its owner says, at each draw. Wheel and arrows
    scroll (shift and the wheel, or a sideways swipe, scroll across); a drag
    pans; + and - zoom, 0 back to 1; Home and End go to the ends; Esc or ×
    closes it.
    """

    def __init__(self, title, sub=""):
        self.title = title
        self.sub = sub
        self.doc = Message("reading …")
        self.zoom = 1.0
        self.sx = self.sy = 0
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.body = pygame.Rect(0, 0, 0, 0)
        self.close = pygame.Rect(0, 0, 0, 0)
        self.title_color = lambda: None
        self.press = None
        self._size = (1, 1)  # the content's, as last laid out

    def set_doc(self, doc):
        self.doc, self.sx, self.sy = doc, 0, 0

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
            self.press = (ev.pos, self.sx, self.sy)
        elif ev.type == pygame.MOUSEMOTION and self.press is not None:
            (px, py), sx, sy = self.press
            self.sx, self.sy = sx - (ev.pos[0] - px), sy - (ev.pos[1] - py)
            self._clamp_()
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.press = None
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
        surf.blit(title, (x + pad, y + pad - 2))
        top = y + pad + font.get_linesize()
        if self.sub:
            surf.blit(small.render(_fit_(small, self.sub, w - 2 * pad), True, DIM), (x + pad, top))
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
        surf.set_clip(old)
        self._scrollbars_(surf)

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


def _fit_(font, text, width):
    """*text*, cut with … to fit *width* as drawn with *font*."""
    if font.size(text)[0] <= width:
        return text
    while len(text) > 1 and font.size(text + "…")[0] > width:
        text = text[:-1]
    return text + "…"
