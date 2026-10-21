"""A Claude session in hyp --markdowns: a conversation with Claude Code, saved as a markdown of the directory scanned.

The **Claude** button opens a window in the middle of the disk -- a third
of the window's area -- with the conversation above and a box to type in
below. Each turn runs the Claude Code CLI, ``claude -p``, once, under the
user's own Claude login: a subscription covers it, and no API key or
credits are needed. Claude Code keeps the conversation itself, by session
id, so a turn after the first is ``claude -p --resume <id>``.

Claude runs in the directory hyp scans (``--dir``), with read-only tools
(`TOOLS`): it can read, list and search the documents on the disk, and
change nothing. A ``--dir`` that is a url is not a directory Claude Code can
work in, so there it has no tools at all.

Every turn writes the session into ``--dir``, or a directory under it
(``--claude-dir``), as one markdown, ``claude-session-<started>.md``: the
transcript, with front matter saying what it is -- ``hyp: claude-session``
and the session's id. The disk is brought up to date with it at once: the
whole transcript embedded, the layout nudged, so the session moves among the
documents as the conversation goes. Its point opens the session again, to go
on with: the transcript is shown, and the next turn resumes it by its id.
"""

import glob
import json
import os
import posixpath
import shutil
import subprocess
import time
import uuid

import pygame

from .docview import DIM, EDGE, TEXT, DocWindow, MarkdownDoc, front_matter

KIND = "claude-session"
PREFIX = "claude-session-"
# What Claude may do in --dir: read, list, search. Nothing that writes, runs or fetches.
TOOLS = ("Read", "Glob", "Grep")
SYSTEM = (
    "You are in a session opened from hyp, an app that lays out the markdown and pdf documents under {root} "
    "as points on a hyperbolic disk. {tools} When the user closes this session, its transcript is saved there as "
    "a markdown document of its own and embedded with the others; it can be reopened and continued later. "
    "Answer in markdown."
)
SYSTEM_TOOLS = "You may read, list and search those documents, to answer from them; you cannot change them."
SYSTEM_NO_TOOLS = "You cannot see the documents unless the user pastes them in."
# The models a session's window switches among, by Claude Code's names for them; None is Claude Code's own default.
MODELS = (None, "opus", "sonnet", "haiku")


class Failed(Exception):
    """A turn that did not come back: what to tell the user, as they would say it."""


class Session:
    """One conversation: Claude Code's session id, its transcript, and where it is saved.

    *root* is the directory hyp scans, local or any fsspec url; *subdir* a
    relative directory under it to save in, or "" for *root* itself. A
    session loaded from its markdown has *prior*, the transcript as saved,
    and is *resumable*: Claude Code has it, by its id.
    """

    def __init__(self, root, subdir="", storage_options=None, command=None, id=None, started=None, title=None,
                 prior="", resumable=False, md_url=None, choice=None, usage=None):
        self.root, self.subdir = str(root).rstrip("/") or "/", subdir
        self.storage_options = storage_options or {}
        self.command = command
        self.id = id or str(uuid.uuid4())
        self.started = started or time.strftime("%Y-%m-%dT%H:%M:%S")
        self._title_ = title
        self.prior = prior
        self.resumable = resumable
        self.turns = []  # this time's: (text you sent, Claude's reply)
        self.saved = 0  # how many of them the file has
        self.renamed = False  # a title given since the file was last written
        self.model = None  # the model the last turn ran on, as Claude Code says
        self.choice = choice  # the model asked for, of MODELS: None, Claude Code's default
        # Over every turn, saved ones too: tokens in (cached or not) and out, and what they would cost at API rates.
        self.usage = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, **(usage or {})}
        self.plan = None  # the plan's windows, as the last turn left them: {"five_hour": {"utilization", "resetsAt"}, ...}
        self.overage = False  # whether the last turn ran on overage, past the plan's windows
        stamp = self.started.replace("-", "").replace(":", "").replace("T", "-")
        self.md_url = md_url or posixpath.join(self.root, *([subdir] if subdir else []), f"{PREFIX}{stamp}.md")

    # 2. Declared API ---------------------------------------------------------

    @classmethod
    def load(cls, md_url, root, subdir="", storage_options=None, command=None):
        """The session saved at *md_url*, to go on with."""
        fields, body = front_matter(_read_(md_url, storage_options))
        if not is_session(fields):
            raise ValueError(f"{md_url} is not a {KIND}")
        usage = {k: fields[k] for k in ("tokens_in", "tokens_out", "cost_usd") if isinstance(fields.get(k), (int, float))}
        session = cls(root, subdir, storage_options, command, id=fields["session"], started=fields.get("started"),
                      title=fields.get("title"), prior=body.strip(), resumable=True, md_url=md_url,
                      choice=fields.get("choice") if fields.get("choice") in MODELS else None, usage=usage)
        session.model = fields.get("model")
        return session

    def send(self, text, on_text=lambda t: None, cancelled=lambda: False):
        """Send *text* as the next turn, streaming the reply to *on_text*; the reply, or None if *cancelled*.

        The turn joins the transcript only once its reply is whole. One that
        fails raises `Failed`, saying why.
        """
        args = self._args_()
        try:
            proc = subprocess.Popen(args, cwd=self._cwd_(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        except OSError as e:
            raise Failed(f"could not run Claude Code ({args[0]}): {e}") from None
        proc.stdin.write(text)
        proc.stdin.close()
        reply, result = [], None
        try:
            for line in proc.stdout:
                if cancelled():
                    proc.kill()
                    return None
                event = _event_(line)
                if event is None:
                    continue
                if event.get("type") == "system" and event.get("subtype") == "init":
                    self.resumable = True  # Claude Code has the session now, whatever comes of the turn
                    self.model = event.get("model") or self.model
                elif event.get("type") == "rate_limit_event":
                    info = event.get("rate_limit_info") or {}
                    self.plan = info.get("unifiedWindows") or self.plan
                    self.overage = bool(info.get("isUsingOverage"))
                elif event.get("type") == "result":
                    result = event
                    self._count_(event)
                else:
                    piece = _piece_(event, bool(reply))
                    if piece:
                        reply.append(piece)
                        on_text(piece)
        finally:
            proc.wait()
            err = proc.stderr.read().strip()
        if result is None or result.get("is_error") or result.get("subtype") not in (None, "success"):
            said = (result or {}).get("result") or err or f"Claude Code exited with {proc.returncode}"
            raise Failed(explain(said))
        answer = "".join(reply).strip() or str(result.get("result") or "").strip()
        self.turns.append((text, answer))
        return answer

    def save(self):
        """Write the transcript; its url."""
        import fsspec

        fs, path = fsspec.url_to_fs(self.md_url, **self.storage_options)
        fs.makedirs(posixpath.dirname(path), exist_ok=True)
        n = len(self.turns)
        with fs.open(path, "w") as f:
            f.write(self.markdown())
        self.saved = max(self.saved, n)
        self.renamed = False
        return self.md_url

    def dirty(self):
        """Whether a turn, or a title given, is not yet in the file; a session with no turn has no file to write."""
        return len(self.turns) > self.saved or (self.renamed and bool(self.prior or self.turns))

    def rename(self, title):
        """Title the session *title* -- its point's label -- from the next save; a blank one puts back the default, its first line."""
        title = " ".join(str(title).split())
        if title != (self._title_ or ""):
            self._title_ = title or None
            self.renamed = True

    def markdown(self, partial=None):
        """The transcript: front matter, what was saved before, this time's turns -- and *partial*, a reply still coming, last."""
        fields = {"hyp": KIND, "session": self.id, "backend": "claude-code", "started": self.started,
                  "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "title": self.title()}
        if self.model:
            fields["model"] = self.model
        if self.choice:
            fields["choice"] = self.choice
        fields.update(tokens_in=self.usage["tokens_in"], tokens_out=self.usage["tokens_out"], cost_usd=round(self.usage["cost_usd"], 4))
        lines = ["---", *(f"{k}: {json.dumps(v)}" for k, v in fields.items()), "---", ""]
        heading = f"# Claude · {self.title()}"
        prior = self.prior
        if prior.startswith("# Claude · "):
            prior = heading + prior[len(prior.split("\n", 1)[0]):]  # its heading, retitled as the session is
        lines += [prior, ""] if prior else [heading, ""]
        for said, answer in self.turns:
            lines += ["## You", "", said.strip(), "", "## Claude", "", answer, ""]
        if partial is not None:
            lines += ["## Claude", "", partial or "…", ""]
        return "\n".join(lines)

    def title(self):
        """The title it was saved with; else the first line of its first turn; else the time it began."""
        if self._title_:
            return self._title_
        line = (self.turns[0][0] if self.turns else "").strip().split("\n", 1)[0]
        return (line[:77] + "…" if len(line) > 80 else line) or self.started.replace("T", " ")

    def switch(self):
        """Ask for the next of MODELS from the next turn on: Claude Code resumes the session on it."""
        self.choice = MODELS[(MODELS.index(self.choice) + 1) % len(MODELS)]

    def usage_line(self):
        """The session's usage, as its window shows it: the model, tokens in and out, and the cost at API rates."""
        u = self.usage
        said = [self.model or "model: Claude Code's default",
                f"{_tokens_(u['tokens_in'])} in · {_tokens_(u['tokens_out'])} out",
                f"≈${u['cost_usd']:.2f} at API rates"]
        return " · ".join(said)

    def quota_line(self, now=None):
        """What is left of the plan's windows, as the last turn said, and when each resets; or that no turn has said yet."""
        if not self.plan:
            return "quota: known after a turn"
        names = {"five_hour": "5 h", "seven_day": "7 d"}
        said = [f"{names.get(k, k)} {max(0.0, 1 - float(w.get('utilization') or 0)):.0%} left"
                + (f" (resets {_when_(w['resetsAt'], now)})" if w.get("resetsAt") else "")
                for k, w in self.plan.items() if isinstance(w, dict)]
        return "quota: " + " · ".join(said) + (" · on overage" if self.overage else "")

    # 4. Helpers --------------------------------------------------------------

    def _count_(self, result):
        """Add a turn's *result* to the session's usage."""
        u = result.get("usage") or {}
        self.usage["tokens_in"] += sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        self.usage["tokens_out"] += int(u.get("output_tokens") or 0)
        self.usage["cost_usd"] += float(result.get("total_cost_usd") or 0)

    def _local_(self):
        return "://" not in self.root

    def _cwd_(self):
        # Claude Code keeps a session by the directory it ran in: always the same one, so --resume finds it.
        return self.root if self._local_() else os.path.expanduser("~")

    def _args_(self):
        tools = ",".join(TOOLS) if self._local_() else ""
        system = SYSTEM.format(root=self.root, tools=SYSTEM_TOOLS if tools else SYSTEM_NO_TOOLS)
        return [self.command or claude_command(), "-p", "--output-format", "stream-json", "--include-partial-messages",
                "--verbose", f"--tools={tools}", f"--allowedTools={tools}", "--permission-mode", "dontAsk",
                "--append-system-prompt", system, *(("--model", self.choice) if self.choice else ()),
                *(("--resume", self.id) if self.resumable else ("--session-id", self.id))]


def ask(prompt, timeout=180, command=None):
    """One question to Claude Code, and its answer's text: no tools, no session kept -- what an arrow's line is asked with."""
    args = [command or claude_command(), "-p", "--output-format", "json", "--tools=", "--no-session-persistence",
            "--permission-mode", "dontAsk"]
    try:
        done = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, cwd=os.path.expanduser("~"))
    except OSError as e:
        raise Failed(f"could not run Claude Code ({args[0]}): {e}") from None
    except subprocess.TimeoutExpired:
        raise Failed(f"Claude Code took more than {timeout} s") from None
    result = _event_(done.stdout.strip().splitlines()[-1]) if done.stdout.strip() else None
    if result is None or result.get("is_error") or done.returncode:
        raise Failed(explain((result or {}).get("result") or done.stderr or f"Claude Code exited with {done.returncode}"))
    return str(result.get("result") or "")


def account(command=None, timeout=20):
    """The account Claude Code is logged in as, as the window shows it -- email, organization, plan -- from ``claude auth status``; None if not."""
    try:
        done = subprocess.run([command or claude_command(), "auth", "status", "--json"], capture_output=True, text=True,
                              timeout=timeout, cwd=os.path.expanduser("~"))
    except (OSError, subprocess.TimeoutExpired, Failed):
        return None
    status = _event_(done.stdout.strip()) if done.stdout.strip() else None
    if not status or not status.get("loggedIn"):
        return "not logged in: run `claude` once in a terminal and log in" if status else None
    plan = status.get("subscriptionType")
    return " · ".join(str(p) for p in (status.get("email"), status.get("orgName"), plan and f"{plan} plan") if p) or None


def _tokens_(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else str(n)


def _when_(epoch, now=None):
    """A reset time, as the window says it: the hour where it is today, else the weekday and the hour."""
    at, today = time.localtime(float(epoch)), time.localtime(now if now is not None else time.time())
    same_day = (at.tm_year, at.tm_yday) == (today.tm_year, today.tm_yday)
    return time.strftime("%H:%M" if same_day else "%a %H:%M", at)


def claude_command():
    """The Claude Code CLI to run: $HYP_CLAUDE, else ``claude`` on the PATH, else the one VS Code's extension carries."""
    named = os.environ.get("HYP_CLAUDE") or shutil.which("claude")
    if named:
        return named
    bundled = sorted(glob.glob(os.path.expanduser("~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude")),
                     key=os.path.getmtime)
    if bundled:
        return bundled[-1]
    raise Failed("Claude Code is not installed: install its CLI (claude.com/claude-code), or point HYP_CLAUDE at it")


def explain(said):
    """What Claude Code said of a turn that failed, as the window says it: what to do, where that is known."""
    said = str(said).strip()
    low = said.lower()
    if "not logged in" in low or "/login" in low or "authentication" in low:
        return "Claude Code is not logged in: run `claude` once in a terminal and log in, then send again"
    if "no conversation found" in low:
        return "Claude Code has no record of this session here: it keeps them per machine and directory"
    return said.splitlines()[-1] if said else "Claude Code failed, saying nothing"


class ClaudeWindow(DocWindow):
    """A session's window: the transcript above, scrolled as a document's; below it, what is being typed.

    Enter sends, Shift+Enter starts a new line, Cmd+V pastes; PageUp and
    PageDown, Home and End, and the wheel scroll the transcript; Esc or ×
    closes the window, and its owner saves the session. Its title is the
    session's: a click edits it, Enter keeps it -- *on_rename(window)*, for
    its owner to file -- and Esc puts it back. Beneath the title, the account
    Claude Code is logged in as (*account*, once its owner has read it), the
    session's usage, and what is left of the plan's quota; at the right of
    the box, the model, which a click switches, from the next turn on.
    """

    INPUT_LINES = 8  # the box grows to this many lines, then scrolls

    def __init__(self, session, on_send, on_rename=lambda window: None):
        super().__init__("Claude", posixpath.basename(session.md_url))
        self.session, self.on_send, self.on_rename = session, on_send, on_rename
        self.naming = None  # the title being typed, while it is edited
        self.account = None  # who Claude Code runs as, as `account` says; None until read
        self.text = ""
        self.sending = None  # what was sent, while its reply is awaited
        self.partial = None  # the reply coming in, while one is
        self.cancelled = False
        self.note = None  # what went wrong with the last turn
        self.input = pygame.Rect(0, 0, 0, 0)
        self.chip = pygame.Rect(0, 0, 0, 0)  # the model's, as last drawn
        self._render_()
        pygame.key.start_text_input()

    def busy(self):
        return self.partial is not None

    def begin(self, text):
        self.sending, self.partial, self.note = text, "", None
        self._render_(follow=True)

    def delta(self, text):
        if self.partial is not None:
            self.partial += text
            self._render_(follow=True)

    def finish(self, note=None):
        """The turn is over; *note*, if it failed, saying why -- and what was sent is put back, to send again."""
        if note is not None and self.sending and not self.text:
            self.text = self.sending
        self.sending, self.partial, self.note = None, None, note
        self._render_(follow=True)

    def close_input(self):
        self.cancelled = True
        pygame.key.stop_text_input()

    def animating(self):
        return self.busy() or self.doc.pending()

    def handle(self, ev):
        """Returns "close" or None."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.chip.collidepoint(ev.pos):
            if not self.busy():
                self.session.switch()
            return None
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.title_rect.collidepoint(ev.pos):
            if self.naming is None:
                self.naming = self.session.title()
            return None
        if self.naming is not None and ev.type in (pygame.TEXTINPUT, pygame.KEYDOWN):
            return self._name_(ev)
        if ev.type == pygame.TEXTINPUT:
            self.text += ev.text
            return None
        if ev.type == pygame.KEYDOWN:
            k, mods = ev.key, pygame.key.get_mods()
            if k == pygame.K_ESCAPE:
                return "close"
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if mods & pygame.KMOD_SHIFT:
                    self.text += "\n"
                elif self.text.strip() and not self.busy():
                    text, self.text = self.text.strip(), ""
                    self.on_send(self, text)
            elif k == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif k == pygame.K_v and mods & (pygame.KMOD_META | pygame.KMOD_CTRL):
                try:
                    self.text += (pygame.scrap.get_text() or "").replace("\r\n", "\n")
                except Exception:
                    pass
            elif k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN, pygame.K_HOME, pygame.K_END):
                return super().handle(ev)
            return None
        return super().handle(ev)

    def draw(self, surf, rect, font, small):
        from .app import wrap

        rect = pygame.Rect(rect)
        pad, lh = 10, font.get_linesize()
        lines = wrap(font, self.text + "|", rect.width - 2 * pad - 2) or ["|"]
        shown = lines[-self.INPUT_LINES:]
        if self.naming is not None:
            hint = "Enter: keep this title · Esc: put it back"
        elif self.busy():
            hint = "Claude is answering …"
        else:
            hint = "Enter: send · Shift+Enter: new line · click the title: rename · Esc: close and save"
        box_h = 2 * pad + lh * len(shown) + small.get_linesize() + 4
        self.title = f"{self.naming}|" if self.naming is not None else f"Claude · {self.session.title()}"
        self.sub = "\n".join([f"{posixpath.basename(self.session.md_url)} · {self.account or 'account: reading …'}",
                              f"{self.session.usage_line()} · {self.session.quota_line()}"])
        super().draw(surf, pygame.Rect(rect.x, rect.y, rect.width, rect.height - box_h), font, small)
        self.rect = rect  # the box's too: a click on it is the window's, not the disk's
        self.input = pygame.Rect(rect.x, rect.bottom - box_h + 4, rect.width, box_h - 4)
        pygame.draw.rect(surf, (32, 38, 48), self.input, border_radius=8)
        pygame.draw.rect(surf, EDGE, self.input, 1, border_radius=8)
        for i, ln in enumerate(shown):
            surf.blit(font.render(ln, True, DIM if self.busy() else TEXT), (self.input.x + pad, self.input.y + pad + i * lh))
        chip = small.render(f"model: {self.session.choice or 'default'} ▾", True, DIM if self.busy() else TEXT)
        bottom = self.input.bottom - pad - small.get_linesize() + 4
        self.chip = chip.get_rect(topright=(self.input.right - pad, bottom)).inflate(8, 2)
        hot = self.chip.collidepoint(pygame.mouse.get_pos()) and not self.busy()
        pygame.draw.rect(surf, (52, 60, 76) if hot else (40, 46, 58), self.chip, border_radius=4)
        surf.blit(chip, chip.get_rect(center=self.chip.center))
        note = small.render(self.note or hint, True, (240, 120, 110) if self.note else DIM)
        surf.blit(note, (self.input.x + pad, bottom))
        pygame.key.set_text_input_rect(self.input)

    def _name_(self, ev):
        """A key or text while the title is edited: typed into it; Enter keeps it, Esc puts it back. Never closes the window."""
        if ev.type == pygame.TEXTINPUT:
            self.naming += ev.text
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.session.rename(self.naming)
            self.naming = None
            self.on_rename(self)
        elif ev.key == pygame.K_ESCAPE:
            self.naming = None
        elif ev.key == pygame.K_BACKSPACE:
            self.naming = self.naming[:-1]
        elif ev.key == pygame.K_v and pygame.key.get_mods() & (pygame.KMOD_META | pygame.KMOD_CTRL):
            try:
                self.naming += " ".join((pygame.scrap.get_text() or "").split())
            except Exception:
                pass
        return None

    def _render_(self, follow=False):
        at_end = self.sy >= self._size[1] - self.body.height - 4
        text = self.session.markdown()
        if self.sending is not None:
            text += f"\n## You\n\n{self.sending}\n\n## Claude\n\n{self.partial or '…'}\n"
        self.doc = MarkdownDoc(_body_(text))
        if follow or at_end:
            self.sy = 10**9  # clamped to the end at the next draw


def is_session(fields):
    """Whether front matter *fields* say their markdown is a saved Claude session."""
    return isinstance(fields, dict) and fields.get("hyp") == KIND and bool(fields.get("session"))


def _event_(line):
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _piece_(event, after):
    """What of Claude Code's stream *event* the reply shows: text as it is written; a note of each tool Claude uses."""
    if event.get("type") != "stream_event":
        return None
    e = event.get("event") or {}
    if e.get("type") == "content_block_delta" and (e.get("delta") or {}).get("type") == "text_delta":
        return e["delta"].get("text") or None
    if e.get("type") == "content_block_start":
        block = e.get("content_block") or {}
        if block.get("type") == "tool_use":
            return f"\n\n*{block.get('name', 'a tool')} …*\n\n"
        if block.get("type") == "text" and after:
            return "\n\n"  # a new paragraph of the reply, after a tool's
    return None


def _body_(markdown):
    _, body = front_matter(markdown)
    return body


def _read_(url, storage_options=None):
    import fsspec

    with fsspec.open(url, "r", **(storage_options or {})) as f:
        return f.read()
