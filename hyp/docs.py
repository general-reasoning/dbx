"""hyp --docs: the documents of a directory -- markdowns and pdfs -- on the disk.

`hyp --docs` scans the current directory, `hyp --docs --dir URL` another,
local or any fsspec url (`ssh://host/path`, `gs://bucket/path`, ...): every
`.md`, `.markdown` and `.pdf` under it, but for hidden directories and
the usual heaps (`SKIP_DIRS`). Each is embedded by its name and the start of
its text, and labeled by its name, its modification time beneath.

Left click opens a document in the center, filling the window; right click
places it in a quarter or a half instead, in place of what is there. The
disk, the placed windows ("tiles") and the center window are three layers,
in an order the layers menu -- or Tab, which brings the back one to the front
-- changes: what is under the mouse in the front one gets the clicks.

A right click on empty disk opens a Claude session in a window of its own
(`hyp.claude`); each turn saves the session as a markdown in the directory
scanned, or one under it (`--claude-dir`). That is all hyp writes there:
the corpus is kept under ~/.hyp/docs/, one directory per scanned directory.
A markdown's front matter is no part of what is embedded of it.

A conversation is shown turn by turn: every turn is a point of its own,
embedded by what was asked and answered in it, labeled by a few words
Claude names its question by, and joined by a line to the turn before it --
a branch's first turn to the turn it goes on from -- so a conversation is
drawn as the tree it is, and laid out by its links (`layout.LINK_PULL`). A
click on a turn opens the conversation up to it, the box beneath: what is
sent from there goes on after it, a new branch unless it is its branch's
last (`claude.Session.branch`).
"""

import hashlib
import json
import os
import posixpath
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import fsspec
import pygame

from . import claude, notes
from .app import Hyp, TextBox
from .docview import BAR, DIM, EDGE, TEXT, DocWindow, MarkdownDoc, Message, PdfDoc, front_matter, pdf_text

KINDS = {".md": "markdown", ".markdown": "markdown", ".pdf": "pdf"}
# Directories never scanned, besides hidden ones.
SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "venv", "env", "build", "dist"}
MAX_DOCS = 5_000
MAX_DEPTH = 16
TEXT_CHARS = 8_000  # of a document's text, kept with its point
EMBED_CHARS = 4_000  # of it, embedded
SESSION_CHARS = 256_000  # of a Claude session's transcript, kept and embedded: all of it, but for the longest
CHUNK_CHARS = 4_000  # a longer text is embedded a piece this long at a time (`ChunkedEmbedder`)
PDF_TEXT_BYTES = 64 << 20  # a larger pdf is embedded by its name alone
TURN = "claude-turn"  # a point's kind: a turn of a Claude conversation, a point of its own
QUESTION_CHARS = 600  # of a turn's question, given Claude to name it by
WORDS_CHARS = 40  # of the words Claude names a question by: longer, they are cut short
FIRST_WORDS = 5  # of a question, standing in for the words Claude has not named it by
WORDS_BATCH = 40  # questions Claude is asked to name at once: every conversation's, at a first start, may be hundreds
ASKED_CHARS = 160  # of a turn's question, in its point's tooltip
# What Claude is asked, to name the questions of a session's turns: every one not named before, at once.
WORDS = """Each numbered message below was sent by a user to Claude, in one conversation. Name what each asks or says in two to five words: a heading, as a table of contents would have it, not a sentence.

Reply with exactly {n} lines, the k-th naming the k-th message: no numbers, no quotes, nothing else.

{questions}"""

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
    With *on_claude*, a row beneath the icons asks Claude about the document
    instead: a click on it calls `on_claude()`.
    """

    ICON = (36, 26)
    PAD = 6
    CLAUDE = "ask Claude about it"
    ROW = 20  # the Claude row's height

    def __init__(self, on_select, pos, on_claude=None):
        self.on_select = on_select
        self.on_claude = on_claude
        self.open = True
        iw, ih = self.ICON
        self.rect = pygame.Rect(pos[0], pos[1], 3 * iw + 4 * self.PAD, 3 * ih + 4 * self.PAD + 18 + (self.ROW if on_claude else 0))
        self.icons = {}  # slot -> its rect, as last drawn

    def _icons_(self):
        iw, ih = self.ICON
        return {slot: pygame.Rect(self.rect.x + self.PAD + c * (iw + self.PAD), self.rect.y + self.PAD + r * (ih + self.PAD), iw, ih)
                for r, row in enumerate(PICKER) for c, slot in enumerate(row)}

    def slot_at(self, pos):
        return next((slot for slot, r in self._icons_().items() if r.collidepoint(pos)), None)

    def claude_row(self):
        """The Claude row's rect, beneath the icons and their label; empty without *on_claude*."""
        if self.on_claude is None:
            return pygame.Rect(0, 0, 0, 0)
        return pygame.Rect(self.rect.x + 2, self.rect.bottom - self.ROW - 2, self.rect.width - 4, self.ROW)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            slot = self.slot_at(ev.pos) if ev.button == 1 else None
            claude_hit = ev.button == 1 and self.claude_row().collidepoint(ev.pos)
            self.open = False
            if slot is not None:
                self.on_select(slot)
            elif claude_hit:
                self.on_claude()
        elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.open = False
        return True

    def draw_list(self, surf, font):
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
        below = self.rect.bottom - 4 - (self.ROW if self.on_claude else 0)
        surf.blit(label, label.get_rect(midbottom=(self.rect.centerx, below)))
        row = self.claude_row()
        if row.width:
            hot = row.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(surf, (52, 60, 76) if hot else (24, 28, 36), row, border_radius=3)
            text = font.render(self.CLAUDE, True, EDGE if hot else TEXT)
            surf.blit(text, text.get_rect(center=row.center))


@dataclass(frozen=True)
class Doc:
    key: str  # its url: what a re-sync matches by
    text: str  # the start of its text
    embed: str  # its name, then the start of its text
    summary: str  # its name: the label
    extra: dict = field(default_factory=dict)  # path, kind, mtime, size: stored with the point


def _mtime_(info):
    """*info*'s modification time, as epoch seconds, from whichever field its filesystem names it by; None for none."""
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
    """`(fs, [info], cut)`: the documents under *root*, their fsspec infos, in path order; *cut* if past *limit*.

    *fs* knows the urls of what it found: `fs.url_of(name)` -- see `url_of`.
    """
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
    """What precedes *top*, the path fsspec made of *url*, in *url*: `ssh://host` of `ssh://host/a/b`, `gs://` of `gs://b/a`.

    fsspec's own `unstrip_protocol` drops an ssh url's host, which a file's
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


def documents(root, progress=print, n_workers=16, cache=None, storage_options=None, names=None):
    """The documents under *root*, as the engine takes them, read on *n_workers* threads: a Claude conversation's, a turn each.

    *cache* -- `{url: {"mtime", "size", "text"}}` -- holds what was read
    before: a document whose time and size are unchanged is not read again.
    It is updated in place. *names* -- `{digest: words}`, `HypDocs._words_`'s
    -- are the words Claude named questions by: a turn not named yet is
    labeled by its question's first words.
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
                text = read_text(fs, info, kind)
            except Exception as e:  # unreadable: placed by its name, saying why
                text = f"({type(e).__name__}: {e})"
            fields = front_matter(text)[0] if kind == "markdown" and text.startswith("---") else None
            text = text[: SESSION_CHARS if claude.is_session(fields) else TEXT_CHARS]
            cache[url] = {"mtime": mtime, "size": size, "text": text}
        extra, summary = {"path": url, "kind": kind, "mtime": mtime, "size": size}, base
        fields, body = front_matter(text) if kind == "markdown" and text.startswith("---") else (None, text)
        body = body.strip() if fields is not None else text  # the front matter is no part of what a document says
        embed = f"{base}\n\n{body[:EMBED_CHARS]}"
        if claude.is_session(fields):
            return _turn_docs_(url, fields, body, names or {})
        return [Doc(key=url, text=text[:TEXT_CHARS], embed=embed, summary=summary, extra=extra)]

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        found = [doc for docs in ex.map(one, infos) for doc in docs]
    _link_branches_(found)
    return found


def _turn_docs_(url, fields, body, names):
    """A saved Claude conversation's own turns -- a branch's, after those it goes on from -- as the engine takes them, a point each.

    Each is keyed `<url>#<n>`, its own turns counted from 0; embedded by what
    was asked and answered in it, alone -- the transcript's markup and the
    title are every turn's, and would pull them all together; labeled by the
    words Claude named its question by, or its first words till then, and a
    conversation's first turn by its title. A turn's "link" is the turn
    before it; a branch's first turn's, the turn it goes on from, found once
    every file is read (`_link_branches_`).
    """
    said, title = claude.turns(body), fields.get("title") or posixpath.basename(url)
    above = claude.parent_url(url, fields)
    out = []
    for n, (asked, answered) in enumerate(said):
        words = names.get(question_digest(asked))
        extra = {"path": url, "kind": TURN, "turn": n, "session": fields.get("session"), "title": title,
                 "words": words or _first_words_(asked), "named": words is not None, "asked": asked[: 2 * ASKED_CHARS]}
        if n:
            extra["link"] = f"{url}#{n - 1}"
        elif above is not None:
            extra.update(above=above, fork=int(fields.get("fork") or 0))
        if n == len(said) - 1:
            extra["updated"] = fields.get("updated")  # its branch's last turn: dated by its last write
        label = f"Claude · {title}" if n == 0 and above is None else extra["words"]
        out.append(Doc(key=f"{url}#{n}", text=f"## You\n\n{asked}\n\n## Claude\n\n{answered}"[:TEXT_CHARS],
                       embed=f"{asked}\n\n{answered}"[:SESSION_CHARS], summary=label, extra=extra))
    return out


def _link_branches_(docs):
    """Each branch's first turn linked to the turn it goes on from: its parent's turn `fork`, counted in the parent's whole path.

    A parent's own turns follow the `fork` turns it keeps of its own
    parent's, so the turn is its own turn `fork - its fork - 1`. A parent not
    found, or without that turn, leaves the branch unlinked.
    """
    turns, forks = {}, {}
    for doc in docs:
        if doc.extra.get("kind") == TURN:
            url = doc.extra["path"]
            turns[url] = turns.get(url, 0) + 1
            if doc.extra["turn"] == 0:
                forks[url] = doc.extra.get("fork", 0) if doc.extra.get("above") else 0
    for doc in docs:
        above = doc.extra.get("above")
        if above and above in turns:
            n = doc.extra["fork"] - forks[above] - 1
            if 0 <= n < turns[above]:
                doc.extra["link"] = f"{above}#{n}"


def question_digest(question):
    """A question's key in words.json: the words Claude named it by are kept by it."""
    return hashlib.sha1(question.encode()).hexdigest()[:20]


def read_names(data):
    """The words Claude named questions by, as the corpus at *data* keeps them in its words.json: {} without one."""
    if not data:
        return {}
    fs, path = fsspec.url_to_fs(posixpath.join(str(data), "words.json"))
    try:
        names = json.loads(fs.cat_file(path))
    except (FileNotFoundError, ValueError):
        return {}
    return names if isinstance(names, dict) else {}


class ChunkedEmbedder:
    """An embedder for texts longer than a model reads well: each embedded a piece at a time, the pieces' unit vectors averaged.

    A text is its first line -- a document's name -- and the rest. A rest
    of `CHUNK_CHARS` or less is embedded as it is, by *base*, exactly as
    without this wrapper; that is every document but a Claude session's
    transcript, embedded whole. A longer one is cut into pieces of about
    that length, at paragraph breaks where it can be, each embedded with the
    name before it. Their unit vectors are averaged, and scaled to their mean
    length: the text lies where all of it points, not where its start does.

    Its name is *base*'s: for a text in one piece the vector is *base*'s
    own, so a corpus embedded before it is not embedded again.
    """

    def __init__(self, base, chunk=CHUNK_CHARS):
        self.base, self.chunk = base, chunk
        self.name = base.name

    def prepare(self, progress=print):
        self.base.prepare(progress)

    def __call__(self, texts):
        import numpy as np

        pieces = [self.pieces(t) for t in texts]
        flat = [p for ps in pieces for p in ps]
        if len(flat) == len(texts):
            return self.base(texts)
        vecs = np.asarray(self.base(flat), dtype=np.float32)
        out, k = [], 0
        for ps in pieces:
            v = vecs[k : k + len(ps)]
            k += len(ps)
            norms = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            mean = (v / norms).mean(axis=0)
            out.append(mean / max(float(np.linalg.norm(mean)), 1e-12) * float(norms.mean()))
        return np.stack(out).astype(np.float32)

    def pieces(self, text):
        """*text* as embedded: itself, or its name before each piece of the rest."""
        head, _, rest = text.partition("\n")
        rest = rest.strip()
        if len(rest) <= self.chunk:
            return [text]
        out = []
        while rest:
            cut = len(rest) if len(rest) <= self.chunk else (rest.rfind("\n\n", self.chunk // 2, self.chunk) + 1 or self.chunk)
            out.append(f"{head}\n\n{rest[:cut].strip()}")
            rest = rest[cut:].strip()
        return out


def load(url, kind, storage_options=None):
    """The document at *url*, to show: a `MarkdownDoc` or a `PdfDoc`; a `Message` saying why, where it cannot be read."""
    try:
        with fsspec.open(url, "rb", **(storage_options or {})) as f:
            data = f.read()
        return PdfDoc(data) if kind == "pdf" else MarkdownDoc(data.decode("utf-8", "replace"))
    except Exception as e:
        traceback.print_exc()
        return Message(f"could not read {url}: {type(e).__name__}: {e}")


def _named_lines_(reply, n):
    """Claude's *reply*, as the *n* names it was asked for: a line each, numbering, bullets and quotes taken off; None if it is not *n* lines."""
    lines = [re.sub(r"^(\d+[.)]|[-*•])\s*", "", ln.strip()).strip().strip("\"'`*").strip() for ln in reply.splitlines() if ln.strip()]
    if len(lines) != n or not all(lines):
        return None
    return [ln if len(ln) <= WORDS_CHARS else ln[: WORDS_CHARS - 1] + "…" for ln in lines]


def _first_words_(text):
    """*text*'s first FIRST_WORDS words, what a question is named by where Claude has not named it."""
    words = text.split()
    if not words:
        return "(nothing asked)"
    return " ".join(words[:FIRST_WORDS]) + (" …" if len(words) > FIRST_WORDS else "")


def _overlap_(a, b):
    ax, ay, aw, ah = SLOTS[a]
    bx, by, bw, bh = SLOTS[b]
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class HypDocs(Hyp):
    """One point per document of a directory; a document opens in the center, or in a quarter or half of the window."""

    title = "hyp: docs"
    BOX_HINT = "Enter: ask Claude, in a new session   Shift+Enter: newline   Esc: cancel"
    # What a right click on empty disk offers, and what its box then says it does on Enter.
    NEW = {"claude": ("Claude: ask it, in a new session", BOX_HINT),
           "markdown": ("Markdown: a new document, to write", "Enter: a new markdown, to go on writing   Shift+Enter: newline   Esc: cancel")}
    HINT = ("click: open, or go on from a turn · right-click: place · drag: move · scroll: zoom · Tab: next layer"
            " · double-click outside: reset")
    LAYERS = ("disk", "tiles", "center")
    # The settings kept with the corpus, in its settings.json, as they are turned on and off: and what each is till then.
    KEPT = {"web access": False, "hand off to the cloud": True}
    CLAUDE_AREA = 1 / 3  # of the window's, a Claude session's window takes, in its middle -- and a markdown being written
    TYPED = (claude.ClaudeWindow, notes.NoteWindow)  # windows typed into: text input on while one is open

    # 1. Hooks ------------------------------------------------------------------

    @classmethod
    def root(cls, args):
        return canonical(getattr(args, "dir", None) or os.getcwd())

    @classmethod
    def claude_dir(cls, args):
        """Where Claude sessions are saved, under `root`: --claude-dir, or "" for the root itself."""
        rel = posixpath.normpath(getattr(args, "claude_dir", None) or ".")
        return "" if rel == "." else rel

    @classmethod
    def __default_data__(cls, args):
        return os.path.join(cls.HOME, "docs", hashlib.sha1(cls.root(args).encode()).hexdigest()[:12])

    @classmethod
    def __embedder__(cls, embedder):
        """A Claude session is embedded whole: `ChunkedEmbedder`, over the embedder --embedder names."""
        return ChunkedEmbedder(embedder)

    @classmethod
    def __source__(cls, args):
        root, n_workers = cls.root(args), getattr(args, "n_workers", 16)
        index = posixpath.join(str(args.data), "docs.json")

        def source(progress):
            fs, path = fsspec.url_to_fs(index)
            try:
                cache = json.loads(fs.cat_file(path))
            except (FileNotFoundError, ValueError):
                cache = {}
            docs = documents(root, progress, n_workers=n_workers, cache=cache, names=read_names(args.data))
            with fs.open(path, "w") as f:
                json.dump(cache, f)
            return docs

        return source

    def __init__(self, engine, worker, args=None):
        super().__init__(engine, worker, args)
        self.layers = list(self.LAYERS)
        self.tiles = {}  # slot -> DocWindow
        self.center = None
        self.expanded = False  # the center's Claude or markdown window fills the area, as a document's does, rather than a third
        self.box_kind = "claude"  # what the box a right click on empty disk opened makes, on Enter: of NEW
        self.focus = None  # the window keys go to, when the mouse is over none: the one last clicked
        self.capture = None  # the window a drag began in
        self.controls = []  # a Claude session is opened by a right click: on empty disk, or on a document
        self.claude_account = None  # who Claude Code runs as, as last read: when a session's window opens, and after each turn
        self.handoffs = {}  # session id -> its Handoff, while a session closed here goes on on the phone
        # Every session not open here is on remote -- handed off as hyp starts, and as its window closes -- and one
        # opened here is taken back.
        self.settings = {**self.settings, "sessions on remote": True}
        # Whether Claude may search the web and fetch pages, besides reading the documents; whether a session handed off goes
        # to the cloud -- the Mac free to sleep -- or to the phone by Remote Control, the Mac running it: kept (KEPT).
        kept = self._kept_settings_()
        self.settings.update({name: bool(kept.get(name, default)) for name, default in self.KEPT.items()})
        self._remoted_ = False  # the sessions put on remote as hyp started: once
        self.names = None  # a question's digest -> the words Claude named it by, as words.json keeps them: read once
        self._naming_ = False  # Claude is naming turns: one naming at a time (`_name_turns_`)
        self._unnamed_ = set()  # questions Claude could not name: not asked again till hyp starts again

    def __click_point__(self, i, now):
        """Left click on a document: open it in the center; on a turn of a Claude conversation, the conversation up to it, to go on from it.

        Its branch's last turn: the branch, to go on with. Any other: a new
        branch going on after it (`claude.Session.branch`), kept once
        something is sent from it.
        """
        point = self.snap.points[i] if self.snap.points else {}
        if point.get("kind") != TURN:
            self.place(i, "center")
            return
        args, n = self.args, point.get("turn", 0)

        def at_turn():
            session = claude.Session.load(point["path"], self.root(args), self.claude_dir(args))
            return session.branch(len(session.upstream()) + n + 1)

        self.run_async(at_turn, self.open_claude)

    def __context_point__(self, i, pos):
        """Right click on a document -- on the disk, or in the points menu: where to place it, or ask Claude about it, as a `TilePicker` at the mouse."""
        self.popup = TilePicker(lambda slot: self.place(i, slot), pos, on_claude=lambda: self.ask_about(i))

    def __setting_changed__(self, name):
        """A setting of KEPT toggled: kept with the corpus. Web access reaches the sessions open, from their next turn on."""
        if name not in self.KEPT:
            return
        on = self.settings[name]
        data = getattr(self.args, "data", None)
        if data:
            fs, path = fsspec.url_to_fs(posixpath.join(str(data), "settings.json"))
            with fs.open(path, "w") as f:
                json.dump({**self._kept_settings_(), name: on}, f)
        if name == "web access":
            for window in self.windows():
                if isinstance(window, claude.ClaudeWindow):
                    window.session.web = on
            self.status = ("web access on: Claude may search the web and fetch pages, from the next turn" if on
                           else "web access off: Claude reads the documents only, from the next turn")
        else:
            self.status = ("a session handed off goes to the cloud: it goes on there, the Mac free to sleep" if on
                           else "a session handed off goes to the phone by Remote Control: the Mac runs it, and must stay awake")

    def __wants_text__(self):
        """Typed into: a Claude session's window, whenever one is open -- its box, or its title -- besides what the disk's are."""
        return super().__wants_text__() or any(isinstance(w, self.TYPED) for w in self.windows())

    def __context_empty__(self, pos):
        """A right click on empty disk: Claude or Markdown, at *pos*; either then a box to type in, at the same place."""
        self.open_popup([(label, kind) for kind, (label, _) in self.NEW.items()], lambda kind: self._type_new_(kind, pos), pos)

    def __submit_text__(self, text):
        """A text typed on empty disk: asked of Claude, as the first turn of a new session; or a new markdown's beginning."""
        if self.box_kind == "markdown":
            self.open_note(text.strip() + "\n")
            return
        window = self._new_claude_()
        self._send_claude_(window, text.strip())

    def __point_datetime__(self, i):
        """When a document was last changed; a branch's last turn, when it was saved: its front matter's `updated`; any other turn, none."""
        point = self.snap.points[i] if self.snap.points else {}
        if point.get("kind") == TURN and not point.get("updated"):
            return None
        updated = point.get("updated")
        if updated:
            return str(updated).replace("T", " ")[:16]
        mtime = point.get("mtime")
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else None

    def __point_tooltip__(self, i):
        """A turn's: its label, its conversation's title and which turn of it, and the beginning of its question; any other point's, as every point's."""
        point = self.snap.points[i] if self.snap.points else {}
        if point.get("kind") != TURN:
            return super().__point_tooltip__(i)
        asked = " ".join(point.get("asked", "").split())
        asked = asked if len(asked) <= ASKED_CHARS else asked[: ASKED_CHARS - 1] + "…"
        which = f"Claude · {point.get('title')} · its turn {point.get('turn', 0) + 1}" + (" -- a branch" if point.get("above") else "")
        return "\n".join(t for t in (self.snap.summaries[i], which, self.sublabel(i, full=True), asked) if t)

    def __animating__(self):
        return self.windows()

    # 2. Placing ----------------------------------------------------------------

    def windows(self):
        return [*self.tiles.values(), *([self.center] if self.center else [])]

    def open_claude(self, session=None):
        """A Claude session's window, in the middle: *session*, to go on with, or a new one; in place of the center's window.

        One at a time: without *session*, the one open, wherever it is; with
        one, any other open is closed -- saved -- first.
        """
        open_ = self._claude_window_()
        if session is None and open_ is not None:
            window = open_
            self.bring_to_front("center" if window is self.center else "tiles")
        else:
            args = self.args
            session = session or claude.Session(self.root(args), self.claude_dir(args))
            session.web = self.settings.get("web access", False)
            window = claude.ClaudeWindow(session, self._send_claude_, self._rename_claude_, self._hand_off_, self._take_back_)
            self._reclaim_(window)
            window.account = self.claude_account  # as last read, until read again: an account switched to since is shown
            self.run_async(claude.account, lambda said: self._account_(window, said))
            for other in {id(w): w for w in (open_, self.center) if w is not None}.values():
                self._close_(other)
            self.center, self.expanded = window, False
            self.bring_to_front("center")
        window._render_(follow=True)  # at the end: the box goes on from there
        self.focus = window

    def open_note(self, text):
        """A new markdown, begun with *text*: saved at once, in the directory scanned, and opened in the middle to go on writing."""
        args = self.args
        # As sessions are, with fsspec's own options; a name kept, the file renamed and the disk brought up to date.
        window = notes.NoteWindow(notes.Note(self.root(args), text), self._save_note_, lambda w: self._file_(w.note))
        if self.center is not None:
            self._close_(self.center)
        self.center, self.expanded = window, False
        self.bring_to_front("center")
        self.focus = window
        self._file_(window.note)

    def move(self, window, slot):
        """Put the open *window* at *slot* -- "center", filling the area, or one of SLOTS -- in place of the windows it would overlap there."""
        self.tiles = {s: w for s, w in self.tiles.items() if w is not window}
        if window is self.center:
            self.center = None
        if slot == "center":
            if self.center is not None:
                self._close_(self.center)
            self.center, self.expanded = window, True
            self.bring_to_front("center")
        else:
            for other in [s for s in self.tiles if _overlap_(s, slot)]:
                self._close_(self.tiles[other])
            self.tiles[slot] = window
            self.bring_to_front("tiles")
        self.focus = window

    def ask_about(self, i):
        """A new Claude session about document *i*: its path, relative to the directory scanned, typed in the box, for the question to follow."""
        path = self.snap.points[i].get("path") or self.snap.summaries[i]
        root = self.root(self.args)
        rel = posixpath.relpath(path, root) if str(path).startswith(root.rstrip("/") + "/") else path
        self._new_claude_().text = f"About {rel}: "

    def place(self, i, slot):
        """Open document *i* at *slot* -- "center", or one of SLOTS -- in place of any window there, its layer brought to the front."""
        point = self.snap.points[i]
        url, kind = point.get("path"), point.get("kind", "markdown")
        when = self.__point_datetime__(i)
        window = DocWindow(self.snap.summaries[i], " · ".join(p for p in (when, url) if p))
        window.title_color = self.title_color(i)
        if slot == "center":
            if self.center is not None:
                self._close_(self.center)  # a Claude session is saved, not dropped
            self.center, self.expanded = window, False
            self.bring_to_front("center")
        else:
            for other in [s for s in self.tiles if _overlap_(s, slot)]:
                self._close_(self.tiles[other])  # a Claude session there is saved, not dropped
            self.tiles[slot] = window
            self.bring_to_front("tiles")
        self.focus = window
        if url is None:  # no file: a text added to the store as a point of its own, as hyp did with a typed one
            window.set_doc(MarkdownDoc(self.snap.texts[i]))
            return
        turn = point.get("turn") if kind == TURN else None

        def show(doc):
            window.set_doc(doc)
            if turn is not None:
                window.pin_heading("You", turn)  # a turn's session, shown at the turn

        self.run_async(lambda: load(url, kind), show)

    def area(self):
        """Where windows go: between the top bar and the status line."""
        w, h = self.screen.get_size()
        return pygame.Rect(8, 36, w - 16, h - 36 - 28)

    def slot_rect(self, slot):
        a = self.area()
        if slot == "center":
            if isinstance(self.center, self.TYPED) and not self.expanded:
                f = self.CLAUDE_AREA ** 0.5  # each side: the area, a third
                return pygame.Rect(0, 0, round(a.width * f), round(a.height * f)).move_to(center=a.center)
            return a
        fx, fy, fw, fh = SLOTS[slot]
        r = pygame.Rect(a.x + round(fx * a.width), a.y + round(fy * a.height), round(fw * a.width), round(fh * a.height))
        return r.inflate(-GAP, -GAP)

    def _send_claude_(self, window, text):
        """Send *text* in *window*'s session, on a thread, the reply shown as it comes."""
        window.begin(text)
        window.session.web = self.settings.get("web access", False)
        put = self.worker.events.put

        def turn():
            """None once the reply is in; what went wrong, where the turn failed -- the text then put back to send again."""
            try:
                window.session.send(text, lambda t: put(("call", lambda: window.delta(t))), lambda: window.cancelled)
            except claude.Failed as e:
                return str(e)
            except Exception as e:  # unforeseen: in the window, and in the status line with its traceback
                said = f"{type(e).__name__}: {e}"  # now: *e* is gone once the except is left, before the call is made
                put(("call", lambda: window.finish(said)))
                raise
            return None

        def then(note):
            window.finish(note)
            # Read again after every turn: what Claude Code ran as then is whose the quota the turn said is.
            self.run_async(claude.account, lambda said: self._account_(window, said, turn=True))
            if note is None:
                self._file_(window.session)

        self.run_async(turn, then)

    def _new_claude_(self):
        """A new session's window, in the middle -- in place of any open, which is saved -- and the window."""
        args = self.args
        self.open_claude(claude.Session(self.root(args), self.claude_dir(args)))
        return self._claude_window_()

    def _claude_window_(self):
        """The Claude session's window open, wherever it is -- the center or a tile -- or None."""
        return next((w for w in self.windows() if isinstance(w, claude.ClaudeWindow)), None)

    def _hand_off_(self, window):
        """Hand *window*'s session off: to the cloud (`claude.CloudHandoff`), or, the setting off, to the phone by Remote Control
        (`claude.Handoff`), its QR code in the window either way."""
        if not window.session.resumable:
            window.note = "nothing to hand off yet: send a first message, then hand it off"
            return
        # Saved before it goes: the file holds everything said here, written now, not once a job gets to it.
        window.session.save()
        self._file_(window.session)  # embedded, in the background
        window.note = None
        window.session.web = self.settings.get("web access", False)
        if self.settings.get("hand off to the cloud"):
            window.handoff = claude.CloudHandoff(window.session)
            window.handoff.start()  # pushed, and begun, on a thread of its own
            return
        window.handoff = claude.Handoff(window.session)
        before = window.session.id
        window.handoff.start()
        if window.session.id != before:
            window.session.save()  # a fresh copy of it handed off: its id, the copy's, in the markdown
        # Read again as it goes: whose claude.ai it goes to, said on its panel, is who Claude Code is logged in as now.
        self.run_async(claude.account, lambda said: self._account_(window, said))

    def _take_back_(self, window, then=None):
        """Bring *window*'s session back: from the cloud (`CloudHandoff.take_back`), or from the phone, Claude Code there ended;
        the turns made there added, and filed. What stops it is said in the window, the session left where it is."""
        handoff = window.handoff
        cloud = getattr(handoff, "cloud", False)

        def back():
            try:
                if cloud:
                    return handoff.take_back(), None
                handoff.stop()
                return window.session.take_back(handoff.started), None
            except claude.Failed as e:
                return None, str(e)

        def done(result):
            n, problem = result
            if problem is not None:
                window.note = f"not taken back: {problem}"
                return
            window.handoff = None
            window.note = f"{n} turn{'s' * (n != 1)} from the {'cloud' if cloud else 'phone'}" if n else None
            window._render_(follow=True)
            self._file_(window.session)  # its turns, or that it is back: in its file
            if then is not None:
                then()

        if cloud:
            window.note = "taking it back from the cloud …"
        self.run_async(back, done)

    def __snapshot_arrived__(self):
        """As the points come in: the turns Claude has not named yet, named (`_name_turns_`); and the first time, every
        Claude conversation among them put on remote, but one open here."""
        self._name_turns_()
        if self._remoted_ or not self.settings.get("sessions on remote") or self.snap is None:
            return
        if self.settings.get("hand off to the cloud"):
            return  # a cloud session each, at every start: handed off as their windows close, or by **remote**
        self._remoted_ = True
        urls = list(dict.fromkeys(p["path"] for p in self.snap.points if p.get("kind") == TURN and p.get("path")))
        root, subdir, web = self.root(self.args), self.claude_dir(self.args), self.settings.get("web access", False)

        def open_here(session):
            return any(isinstance(w, claude.ClaudeWindow) and w.session.md_url == session.md_url for w in self.windows())

        def job():
            handed = []
            for url in urls:  # one at a time: each is a Claude Code started, and its conversation copied where it must be
                try:
                    session = claude.Session.load(url, root, subdir)
                except (OSError, ValueError):
                    continue
                if open_here(session) or session.id in self.handoffs:
                    continue
                claude.carry_over(session.id)
                if claude.conversation_path(session.id) is None:
                    continue  # no record of it to resume: it stays here
                before = session.id
                session.web = web
                handoff = claude.Handoff(session)
                handoff.start()
                if session.id != before:
                    session.save()  # a fresh copy of it gone: its id in the markdown, for when it is taken back
                handed.append(handoff)
            return handed

        def done(handed):
            for handoff in handed:
                if open_here(handoff.session):  # opened here meanwhile: it stays here
                    handoff.stop()
                else:
                    self.handoffs[handoff.session.id] = handoff
            n = len(self.handoffs)
            self.status = f"on remote: {n} Claude session{'s' * (n != 1)} -- open one here to take it back"

        self.run_async(job, done)

    def _reclaim_(self, window):
        """A session reopened here: taken back from the phone, if it is there; else what was said elsewhere since it was saved, added."""
        handoff = self.handoffs.pop(window.session.id, None)
        if handoff is None and window.session.cloud:
            handoff = claude.CloudHandoff(window.session)  # in the cloud since hyp last ran: as its file says
        if handoff is not None:
            window.handoff = handoff
            self._take_back_(window)
            return
        since = window.session.saved_at()
        if since is None:
            return

        def done(n):
            if n:
                window.note = f"{n} turn{'s' * (n != 1)} from elsewhere, since it was saved"
                window._render_(follow=True)
                self._file_(window.session)

        self.run_async(lambda: window.session.take_back(since), done)

    def _type_new_(self, kind, pos):
        """The box of a right click on empty disk, for *kind* -- "claude" or "markdown" -- at *pos*."""
        self.box_kind = kind
        self.box = TextBox(pos, self.NEW[kind][1])

    def _save_note_(self, window):
        if window.note.dirty():
            self._file_(window.note)

    def _rename_claude_(self, window):
        """A session's title kept: filed at once -- its point relabeled -- unless a turn is in flight, which files it as it ends."""
        if not window.busy() and window.session.dirty():
            self._file_(window.session)

    def _account_(self, window, said, turn=False):
        """*said*, `claude.account`'s answer, as *window*'s account -- and, read just after a turn, as whose its quota is."""
        self.claude_account = window.account = said or "account: unknown (claude auth status said nothing)"
        if turn and window.session.plan:
            window.session.plan_account = window.account

    def _file_(self, session):
        """Save *session* and bring the disk up to date with it -- its new turn embedded, the layout nudged -- in the background, the window usable meanwhile."""
        engine, put = self.engine, self.worker.events.put

        def job():
            url = session.save()
            if engine is not None:
                engine.refresh(self.worker.status)  # a turn added moves points; a title given only relabels: shown anew either way
                put(("snapshot", (engine.snapshot(), None)))
            return url

        self.run_async(job, lambda url: setattr(self, "status", f"saved and embedded {posixpath.basename(url)}"))

    def _name_turns_(self):
        """The turns on the disk Claude has not named yet, named -- WORDS_BATCH at a time, in the background -- and the disk relabeled.

        One naming at a time; questions it could not name are not asked again
        till hyp starts again, and why is said in the status line.
        """
        if self._naming_ or self.snap is None:
            return
        todo = list(dict.fromkeys(p["asked"] for p in self.snap.points
                                  if p.get("kind") == TURN and not p.get("named") and p.get("asked") not in self._unnamed_))
        if not todo:
            return
        self._naming_ = True
        engine, put, status = self.engine, self.worker.events.put, self.worker.status

        def job():
            status(f"Claude is naming {len(todo)} turn{'s' * (len(todo) != 1)}")
            try:
                self._words_(todo, ask=True)
            except claude.Failed as e:
                self._unnamed_.update(todo)
                return f"turns named by their first words: Claude could not name them: {e}"
            if engine is not None:
                engine.refresh(status)  # relabeled: nothing moves, but the disk is shown anew
                put(("snapshot", (engine.snapshot(), None)))
            return f"{len(todo)} turn{'s' * (len(todo) != 1)} named"

        def done(said):
            self._naming_ = False
            self.status = said

        self.run_async(job, done)

    def _words_(self, questions, ask):
        """The few words naming each of *questions*, and how many are not Claude's.

        Claude's are kept, by each question's digest, in the corpus's
        words.json. With *ask*, those it has not named are asked of it,
        WORDS_BATCH at a time, and kept as each batch is named; a question it
        has not named is named by its first words. Raises `claude.Failed`
        where Claude cannot say.
        """
        names = self._names_()
        digests = [question_digest(q) for q in questions]
        todo = list(dict.fromkeys(n for n, d in enumerate(digests) if d not in names))
        if todo and ask:
            for b in range(0, len(todo), WORDS_BATCH):
                batch = todo[b : b + WORDS_BATCH]
                listed = "\n".join(f"{m + 1}. {' '.join(questions[n].split())[:QUESTION_CHARS]}" for m, n in enumerate(batch))
                reply = claude.ask(WORDS.format(n=len(batch), questions=listed))
                said = _named_lines_(reply, len(batch))
                if said is None:
                    raise claude.Failed(f"its answer was not {len(batch)} lines: {reply[:200]}")
                names.update({digests[n]: w for n, w in zip(batch, said)})
                self._keep_names_()
            todo = []
        return [names.get(d) or _first_words_(q) for d, q in zip(digests, questions)], len(todo)

    def _names_(self):
        """`names`, read from the corpus's words.json the first time: {} without one."""
        if self.names is None:
            self.names = read_names(getattr(self.args, "data", None))
        return self.names

    def _keep_names_(self):
        data = getattr(self.args, "data", None)
        if data:
            fs, path = fsspec.url_to_fs(posixpath.join(str(data), "words.json"))
            with fs.open(path, "w") as f:
                json.dump(self.names, f)

    def _kept_settings_(self):
        """The settings kept with the corpus, in its settings.json: {} without one."""
        data = getattr(self.args, "data", None)
        if not data:
            return {}
        fs, path = fsspec.url_to_fs(posixpath.join(str(data), "settings.json"))
        try:
            kept = json.loads(fs.cat_file(path))
        except (FileNotFoundError, ValueError):
            return {}
        return kept if isinstance(kept, dict) else {}

    def _close_(self, window):
        if isinstance(window, notes.NoteWindow) and window.dirty():
            self._file_(window.note)  # what is not saved, saved as it closes
        if isinstance(window, claude.ClaudeWindow):
            handing = window.handoff is None and self.settings.get("sessions on remote") and window.session.resumable \
                and not window.busy()
            window.close_input()
            if handing:
                self._hand_off_(window)  # saved first, there and then
            elif window.session.dirty():  # a turn not yet filed: one cancelled as it was closed has none
                self._file_(window.session)
            if window.handoff is not None and window.handoff.problem is None:
                self.handoffs[window.session.id] = window.handoff  # goes on there, the window gone
                where = "in the cloud" if getattr(window.handoff, "cloud", False) else "on the phone"
                self.status = f"{where}: {window.session.title()} -- reopen it here to take it back"
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
        if ev.type == pygame.MOUSEBUTTONDOWN:
            for editing in [w for w in self.windows() if getattr(w, "naming", None) is not None]:
                if not editing.rect.collidepoint(ev.pos):
                    editing.keep_title()  # a click off its window -- on the disk, or another window -- keeps the title too
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.capture = None  # a press before this one has ended, whether or not its release came
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
            typed = ev.type == pygame.TEXTINPUT and isinstance(self.focus, self.TYPED)  # typed with the mouse away
            if self.focus is not None and (typed or ev.type == pygame.KEYDOWN and ev.key not in (pygame.K_HOME,)):
                if self.focus.handle(ev) == "close":
                    self._close_(self.focus)
                return True
            return False
        window = self._window_at_(layer, getattr(ev, "pos", None) or pygame.mouse.get_pos())
        if window is None:
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3 and window.header().collidepoint(ev.pos):
            # A right click on its title bar: where to put it, as a document's right click places one.
            self.popup = TilePicker(lambda slot: self.move(window, slot), ev.pos)
            self.focus = window
            return True
        if ev.type == pygame.MOUSEBUTTONDOWN:
            self.focus = window
            if ev.button == 1:
                self.capture = window
        if ev.type == pygame.MOUSEMOTION and self.capture is None:
            return True
        if ev.type == pygame.MOUSEBUTTONUP:
            self.capture = None  # released over a window: its drag is over too, not only one released off them
        if window.handle(ev) == "close":
            self._close_(window)
        return True
