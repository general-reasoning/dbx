"""A Claude session in hyp --docs: a conversation with Claude Code, saved as a markdown of the directory scanned.

The **Claude** button opens a window in the middle of the disk -- a third
of the window's area -- with the conversation above and a box to type in
below. Each turn runs the Claude Code CLI, `claude -p`, once, under the
user's own Claude login: a subscription covers it, and no API key or
credits are needed. Claude Code keeps the conversation itself, by session
id, so a turn after the first is `claude -p --resume <id>`.

Claude runs in the directory hyp scans (`--dir`), with read-only tools
(`TOOLS`): it can read, list and search the documents on the disk, and
change nothing. A `--dir` that is a url is not a directory Claude Code can
work in, so there it has none of them. The web is not reached unless the
session's `web` says so -- hyp's "web access" setting -- and then Claude may
search it and fetch pages too (`WEB_TOOLS`).

Every turn writes the session into `--dir`, or a directory under it
(`--claude-dir`), as one markdown, `claude-session-<started>-<id>.md`: the
transcript, with front matter saying what it is -- `hyp: claude-session`
and the session's id. The disk is brought up to date with it at once: the
whole transcript embedded, the layout nudged, so the session moves among the
documents as the conversation goes. Its point opens the session again, to go
on with: the transcript is shown, and the next turn resumes it by its id.
"""

import glob
import json
import os
import posixpath
import pty
import re
import select
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime

import pygame
import segno

from .docview import DIM, EDGE, TEXT, AreaEdit, DocWindow, EditableTitle, MarkdownDoc, fit, front_matter

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
# What Claude may do besides, while the session's `web` is on: search the web, and read the pages it finds.
WEB_TOOLS = ("WebSearch", "WebFetch")
SYSTEM_WEB = "You may also search the web and fetch web pages, to answer from them."
SYSTEM_NO_WEB = "You cannot reach the web."
# hyp's own messages to Claude begin so -- a cloud handoff's opening, a take-back's -- and are no turns of the conversation.
HYP_MARK = "[hyp] "
# What Claude Code records as the user's when a turn is cut short: no turn of its own, but its turn's, which goes on after it.
INTERRUPTED = "[Request interrupted"
MAX_BRANCH_DEPTH = 256  # branches of branches, read up to a conversation's first: past this, a parent named is taken to be lost
# Where Claude Code keeps its login and conversations, unless CLAUDE_CONFIG_DIR says elsewhere.
DEFAULT_CONFIG = os.path.expanduser("~/.claude")
# The models a session's window switches among, by Claude Code's names for them; None is Claude Code's own default.
MODELS = (None, "opus", "sonnet", "haiku")
# The session's url on claude.ai, as a session handed off to a phone shows it: what its QR code is of.
SESSION_URL = re.compile(r"https://claude\.ai/code/session_[A-Za-z0-9]+")
CLOUD_SESSION = re.compile(r"https://claude\.ai/code/(session_[A-Za-z0-9]+)")
# In --dir: the GitHub repository its notes are pushed to, `owner/name`, for handing a conversation off to the cloud.
GITHUB_FILE = ".hyp-github"
# What a cloud session is first told: the conversation so far is in files of the repository it is given.
CLOUD_PROMPT = (
    "This conversation began in hyp, on a Mac, and goes on here. Its turns so far are in files of this repository: "
    "{files}. Read them first, in that order, as the conversation so far: in each, a `## You` section is what the user "
    "said, and the `## Claude` section after it what you answered. Change none of them. Then say in a line that you have "
    "the conversation, and wait for the user's next message."
)
TAKE_BACK = "The conversation goes on on the Mac now. Reply only: OK."  # typed as a teleported one starts: so it is recorded here
# What a cloud session is first told: the conversation so far, every turn as it was said -- so the phone shows it whole.
CLOUD_OPENING = (
    "This conversation began in hyp, on a Mac, and goes on here. Here is every turn of it so far, in order, as it was "
    "said:\n\n{turns}\n\nTake it as this conversation so far -- what Claude said, you said. Reply only: Ready. Then wait "
    "for the user's next message."
)
# And handed off again, of the turns made on the Mac since: written out, as they were said.
CATCH_UP_TURNS = (
    "Meanwhile the conversation went on in hyp, on the Mac, after what was said here. Here are its turns since, in order, "
    "as they were said:\n\n{turns}\n\nTake them as said here too. Reply only: Ready. Then wait for the user's next message."
)
CLOUD_TEXT_CHARS = 200_000  # of a conversation written out in a message: past this, the cloud is pointed to its files instead
# Past CLOUD_TEXT_CHARS, what a cloud session is told, handed off again, of the turns made on the Mac since: in the files.
CATCH_UP = (
    "Meanwhile the conversation went on in hyp, on the Mac, after what was said here: its turns {first} to {last} are in "
    "files of this repository, now pushed -- {files}. Read them, as said here too, change none of them, and wait for the "
    "user's next message."
)
# What a pseudo-terminal's Claude Code is not told: that it runs inside another, whose transcripts it would then not save.
CHILD_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_PID")
# Remote Control's own consent, asked as /remote-control first connects -- spaces gone: accepted, as what was asked for.
CONSENT = "enableremotecontrolopens"
# What Claude Code asks only once, interactively -- spaces gone, as its screen draws them -- and what to do instead.
FIRST_RUN = {
    "choosethetextstyle": "Claude Code has not been set up to run interactively",
    "doyoutrustthefiles": "Claude Code does not yet trust this directory",
}


class Failed(Exception):
    """A turn that did not come back: what to tell the user, as they would say it."""


class Session:
    """One conversation -- a branch of one -- Claude Code's session id, its transcript, and where it is saved.

    *root* is the directory hyp scans, local or any fsspec url; *subdir* a
    relative directory under it to save in, or "" for *root* itself. A
    session loaded from its markdown has *prior*, the transcript as saved,
    and is *resumable*: Claude Code has it, by its id.

    A conversation branches: a branch goes on from a turn of another, its
    *above*, keeping the first *fork* turns of that one's (`path`) and
    adding its own. Its file holds only its own turns, and names its parent's
    file and *fork* in its front matter: the turns before them are assembled
    from its parents, as it is loaded (`upstream`). Claude Code's side is
    the same: a branch's first turn forks the conversation it was made
    from, resumed at the last message of its turn *fork* (`branch`).
    """

    def __init__(self, root, subdir="", storage_options=None, command=None, id=None, started=None, title=None,
                 prior="", resumable=False, md_url=None, choice=None, usage=None, updated=None, above=None, fork=0,
                 source=None):
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
        self.updated = updated  # when the file was last written, as its front matter says: what was said elsewhere since is new
        self.model = None  # the model the last turn ran on, as Claude Code says
        self.choice = choice  # the model asked for, of MODELS: None, Claude Code's default
        # Over every turn, saved ones too: tokens in (cached or not) and out, and what they would cost at API rates.
        self.usage = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, **(usage or {})}
        self.plan = None  # the plan's windows, as the last turn left them: {"five_hour": {"utilization", "resetsAt"}, ...}
        self.overage = False  # whether the last turn ran on overage, past the plan's windows
        self.web = False  # whether Claude may search the web and fetch pages (WEB_TOOLS), from the next turn on: its owner says
        self.plan_at = None  # when a turn last said what is left of the plan's windows: the time the quota is as of
        self.plan_account = None  # who Claude Code ran as then, as its owner read it just after: whose the quota is
        self.above, self.fork = above, fork  # the branch this one goes on from, and how many of its turns it keeps
        self.lost = None  # the parent's file, where its front matter names one that cannot be read: its turns are not shown
        # The session whose conversation a new branch's first turn forks: one holding its first *fork* turns. The session,
        # not its id: handed off meanwhile, it may be given another (`fresh_copy`), and the old one's record go.
        self.source = source
        # Which Claude Code conversations hold its path's turns (`held`): `id` those after its first *start*; *earlier*,
        # [(id, start, upto)], those before -- each turns start+1..upto -- where it was taken back from the cloud, which
        # went on from what it read of them, not from a conversation that holds them.
        self.start, self.earlier = 0, []
        # Its cloud session, `session_…`, kept after it is taken back, so that it goes on there again (`CloudHandoff`): *away*
        # while it is there; *cloud_seen*, how many of the cloud's turns are here; *cloud_upto*, how many of its own turns
        # the cloud has -- the turns it was given, the last time they met.
        self.cloud, self.away, self.cloud_seen, self.cloud_upto = None, False, 0, 0
        stamp = self.started.replace("-", "").replace(":", "").replace("T", "-")
        # By its start and its id: two sessions begun in one second are two files, not one written over.
        name = f"{PREFIX}{stamp}-{self.id[:8]}.md"
        self.md_url = md_url or posixpath.join(self.root, *([subdir] if subdir else []), name)

    # 2. Declared API ---------------------------------------------------------

    @classmethod
    def load(cls, md_url, root, subdir="", storage_options=None, command=None, _below_=()):
        """The session saved at *md_url*, to go on with -- a branch, with the branches it goes on from, loaded too."""
        fields, body = front_matter(_read_(md_url, storage_options))
        if not is_session(fields):
            raise ValueError(f"{md_url} is not a {KIND}")
        usage = {k: fields[k] for k in ("tokens_in", "tokens_out", "cost_usd") if isinstance(fields.get(k), (int, float))}
        session = cls(root, subdir, storage_options, command, id=fields["session"], started=fields.get("started"),
                      title=fields.get("title"), prior=body.strip(), resumable=True, md_url=md_url,
                      choice=fields.get("choice") if fields.get("choice") in MODELS else None, usage=usage,
                      updated=fields.get("updated"))
        session.model = fields.get("model")
        session.start = int(fields.get("start") or 0)
        session.earlier = [tuple(e) for e in fields.get("earlier") or () if isinstance(e, list) and len(e) == 3]
        session.cloud = fields.get("cloud") or None
        session.away = bool(fields.get("away", session.cloud is not None))  # before kept: only while away
        session.cloud_seen, session.cloud_upto = int(fields.get("cloud_seen") or 0), int(fields.get("cloud_upto") or 0)
        parent = parent_url(md_url, fields)
        if parent is not None:
            session.fork = int(fields.get("fork") or 0)
            try:
                if parent in _below_ or len(_below_) > MAX_BRANCH_DEPTH:
                    raise ValueError(f"{parent}: a branch of its own branches")
                session.above = cls.load(parent, root, subdir, storage_options, command, (*_below_, md_url))
            except (OSError, ValueError):
                session.lost = parent
        return session

    def branch(self, k):
        """A new branch going on after turn *k* of `path` (1-based): its upstream the first *k* turns, and nothing yet of its own.

        Its parent is the branch whose own turns hold turn *k* -- this one, or
        one it goes on from -- so the tree is drawn as it grew. Claude Code's
        side is forked from this conversation, which holds every turn of its
        `path`, at its first turn (`send`). Turn *k* the last of that parent's
        is no branch: the parent itself, to go on with.
        """
        owner = self
        while owner.above is not None and k <= owner.fork:
            owner = owner.above
        if k >= len(owner.path()):
            return owner
        source = self if self.resumable else owner
        new = Session(self.root, self.subdir, self.storage_options, self.command, choice=self.choice, above=owner, fork=k,
                      source=source)
        # Its conversation is forked from whichever of the source's holds turn k: what that holds not, it holds not either.
        start = source.start if k > source.start else next((s for _, s, u in source.earlier if s < k <= u), 0)
        new.start, new.earlier = start, [e for e in source.earlier if e[2] <= start]
        return new

    def held(self, k):
        """`(conversation, message)`: where turn *k* of `path` (1-based) ends, in the Claude Code conversation that holds it; None where none here does."""
        conversation, start = self.holder(k)
        if conversation is None:
            return None
        carry_over(conversation)
        at = anchor(conversation, k - start)
        return (conversation, at) if at is not None else None

    def holder(self, k):
        """`(conversation, start)`: the Claude Code conversation holding turn *k* of `path`, and the turns before its own; (None, 0) for none."""
        if k > self.start:
            return self.id, self.start
        return next(((e, s) for e, s, u in reversed(self.earlier) if s < k <= u), (None, 0))

    def upstream(self):
        """The turns before its own, `(asked, answered)` each: the first `fork` of its parent's `path`; none for a conversation's first branch."""
        return self.above.path()[: self.fork] if self.above is not None else []

    def own(self):
        """Its own turns, `(asked, answered)` each: those saved, then this time's."""
        return [*turns(self.prior), *self.turns]

    def path(self):
        """Every turn of the conversation, to its own last: its `upstream`, then its `own`."""
        return [*self.upstream(), *self.own()]

    def send(self, text, on_text=lambda t: None, cancelled=lambda: False):
        """Send *text* as the next turn, streaming the reply to *on_text*; the reply, or None if *cancelled*.

        The turn joins the transcript only once its reply is whole. One that
        fails raises `Failed`, saying why.
        """
        if self.resumable:
            carry_over(self.id)  # begun under another login, Claude Code's: resumed under this one
        held = None
        if not self.resumable and self.source is not None:
            held = self.source.held(self.fork)
            if held is None:
                raise Failed(f"Claude Code has no record here of the conversation to branch from ({self.source.title()}), or "
                             f"not of its turn {self.fork}: it keeps them per machine and directory")
        args = self._args_(held)
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
                    self.plan_at = time.time()
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
        self.updated = time.strftime("%Y-%m-%dT%H:%M:%S")
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
        parent = self.above.md_url if self.above is not None else self.lost
        if parent is not None:  # a branch: its parent's file, relative to its own where it can be, and its turns kept
            fields.update(parent=_relative_(parent, self.md_url), fork=self.fork)
        if self.start or self.earlier:  # taken back from the cloud: which conversations hold which turns
            fields.update(start=self.start, earlier=[list(e) for e in self.earlier])
        if self.cloud:
            fields.update(cloud=self.cloud, away=self.away, cloud_seen=self.cloud_seen, cloud_upto=self.cloud_upto)
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
        """The title it was given or saved with; else the first line of its first own turn; else, a branch, its parent's, as a branch's; else the time it began."""
        if self.given_title():
            return self.given_title()
        return f"{self.above.title()} ⑂ {self.fork}" if self.above is not None else self.started.replace("T", " ")

    def given_title(self):
        """The title it was given or saved with, else the first line of its first turn; "" while it has neither -- its start time is no title."""
        if self._title_:
            return self._title_
        line = (self.turns[0][0] if self.turns else "").strip().split("\n", 1)[0]
        return line[:77] + "…" if len(line) > 80 else line

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
        """What is left of the plan's windows, as the last turn said -- and when it said so -- and when each resets; or that no turn has said yet."""
        if not self.plan:
            return "quota: known after a turn"
        names = {"five_hour": "5 h", "seven_day": "7 d"}
        said = [f"{names.get(k, k)} {max(0.0, 1 - float(w.get('utilization') or 0)):.0%} left"
                + (f" (resets {_when_(w['resetsAt'], now)})" if w.get("resetsAt") else "")
                for k, w in self.plan.items() if isinstance(w, dict)]
        as_of = f" (as of {_when_(self.plan_at, now)})" if self.plan_at else ""
        return f"quota{as_of}: " + " · ".join(said) + (" · on overage" if self.overage else "")

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

    def take_back(self, since):
        """The turns made in this conversation elsewhere since *since* -- a time -- added to this time's: how many.

        Read from Claude Code's own record of the conversation, its
        `<id>.jsonl`: each prompt typed, and what Claude answered in text.
        What Claude Code adds besides -- its commands' output, its notes,
        tool results -- is left out.
        """
        path = conversation_path(self.id)
        if path is None:
            return 0
        added = exchanges(path, since)
        self.model = _model_(path) or self.model
        self.turns += added
        return len(added)

    def saved_at(self):
        """When the file was last written, as time.time() -- its front matter's `updated`, local time -- or None for never."""
        try:
            return time.mktime(time.strptime(str(self.updated), "%Y-%m-%dT%H:%M:%S")) if self.updated else None
        except ValueError:
            return None

    def handoff_args(self):
        """The command Claude Code is run with to take the session elsewhere: the conversation resumed, interactively.

        Remote Control is not asked for here, but typed in once it is up
        (`Handoff`): started with the session, it shows the phone only what
        comes after, and the conversation there would begin empty.
        """
        tools, system = self._tools_()
        return [self.command or claude_command(), "--resume", self.id,
                f"--tools={tools}", f"--allowedTools={tools}", "--permission-mode", "dontAsk",
                "--append-system-prompt", system, *(("--model", self.choice) if self.choice else ())]

    def _tools_(self):
        """`(tools, system)`: the tools Claude Code is given, comma-separated, and the system prompt, saying what they are.

        The documents' (TOOLS) where they are local; the web's (WEB_TOOLS) with `web`.
        """
        tools = [*(TOOLS if self._local_() else ()), *(WEB_TOOLS if self.web else ())]
        said = [SYSTEM_TOOLS if self._local_() else SYSTEM_NO_TOOLS, SYSTEM_WEB if self.web else SYSTEM_NO_WEB]
        return ",".join(tools), SYSTEM.format(root=self.root, tools=" ".join(said))

    def _args_(self, held=None):
        """The command a turn runs: the conversation resumed by its id, or begun with it -- or, *held* `(conversation,
        message)` where a branch's upstream ends, that conversation forked there, under its id."""
        tools, system = self._tools_()
        if self.resumable:
            which = ("--resume", self.id)
        elif held is not None:
            which = ("--resume", held[0], "--resume-session-at", held[1], "--fork-session", "--session-id", self.id)
        else:
            which = ("--session-id", self.id)
        return [self.command or claude_command(), "-p", "--output-format", "stream-json", "--include-partial-messages",
                "--verbose", f"--tools={tools}", f"--allowedTools={tools}", "--permission-mode", "dontAsk",
                "--append-system-prompt", system, *(("--model", self.choice) if self.choice else ()), *which]


class Handoff:
    """A session handed off to a phone: Claude Code run interactively, with Remote Control, in a pseudo-terminal of hyp's own.

    hyp drives a session through `claude -p`, a run a turn, and Remote
    Control needs a session that stays: so to hand one off, the conversation
    is resumed interactively -- `claude --resume <id>` -- in a terminal no
    one sees, `/remote-control` typed at its prompt, so that the phone is
    given the conversation whole, and its screen read for the session's url
    (`url`), shown as a QR code to open it on a phone. It goes on there, and
    `stop` ends it; `Session.take_back` brings in what was said meanwhile.
    Claude Code asks some things once, interactively -- a theme, whether to
    trust the directory -- and those it asks here are not answered: `problem`
    says to run it once in a terminal. Remote Control's own consent is: it is
    what was asked for.
    """

    SCREEN_CHARS = 20_000  # of the screen, the last, kept to read
    PROMPT_S = 1.5  # how long, at the least, before /remote-control is typed in
    QUIET_S = 1.0  # and how long the screen must have said nothing new: the resumed conversation laid out first
    CONNECT_S = 30  # how long Remote Control has to say its url, before the handoff is given up

    def __init__(self, session, name=None):
        self.session = session
        self.url = None  # the session's on claude.ai, once Claude Code has said it
        self.problem = None  # why it is not handed off, where it is not
        self.started = None  # when it was, as time.time(): what was said since is the phone's
        self.pid = self.fd = None
        self.name = name or f"hyp · {session.title()}"  # the session's, on claude.ai and the phone: as its point is labeled
        self._screen_ = ""
        self._typed_ = False  # /remote-control typed in, at its prompt
        self._said_ = 0.0  # when the screen last showed something new
        self._accepted_ = False  # Remote Control's consent accepted, where it was asked

    def start(self):
        """Run Claude Code with the session, in the background; its url is read off its screen as it comes."""
        carry_over(self.session.id)
        # Handed off before -- here, or under another login -- it would reconnect to that claude.ai session, which shows
        # the phone only what came after it began: a fresh copy has none, and so begins with the whole conversation.
        self.session.id = fresh_copy(self.session.id) or self.session.id
        args = self.session.handoff_args()
        self.started = time.time()
        pid, fd = pty.fork()
        if pid == 0:  # the child: Claude Code, where the session runs
            try:
                os.chdir(self.session._cwd_())
                os.environ["TERM"] = "xterm-256color"
                os.execv(args[0], args)
            finally:
                os._exit(127)
        self.pid, self.fd = pid, fd
        threading.Thread(target=self._read_, daemon=True).start()

    def alive(self):
        return self.pid is not None

    def stop(self, timeout=10):
        """End Claude Code there -- the session gone offline on the phone -- waiting for it to have written what it holds."""
        if self.pid is None:
            return
        pid = self.pid
        # Its terminal is read all the while: unread, Claude Code blocks writing to it, and ends at no signal.
        for sig, seconds in ((signal.SIGTERM, timeout), (signal.SIGKILL, 2)):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            if self._ended_(pid, seconds):
                break
        self.pid = None
        try:
            os.close(self.fd)
        except OSError:
            pass

    def qr(self):
        """The url's QR code, as rows of 0 and 1 -- dark -- quiet zone included; None before the url is."""
        if self.url is None:
            return None
        return [list(row) for row in segno.make(self.url, error="m").matrix_iter(border=2)]

    # 4. Helpers ----------------------------------------------------------------

    def _ended_(self, pid, seconds):
        """Whether *pid* ended within *seconds*, reaped: its terminal drained meanwhile, so that it is never blocked on it."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                if os.waitpid(pid, os.WNOHANG) != (0, 0):
                    return True
            except ChildProcessError:
                return True
            try:
                if select.select([self.fd], [], [], 0.1)[0]:
                    os.read(self.fd, 65536)
            except OSError:
                time.sleep(0.1)
        return False

    def _type_(self, text):
        """Type *text* at Claude Code's prompt, and Enter."""
        try:
            os.write(self.fd, text.encode())
            time.sleep(0.3)
            os.write(self.fd, b"\r")
        except OSError:
            pass

    def _read_(self):
        """Read its screen till it ends: the url as it appears, or what it asks that is not answered here.

        At its prompt, /remote-control is typed in, and Remote Control's
        consent, where it is asked, accepted.
        """
        while self.pid is not None:
            if self.url is None and time.time() - self.started > self.CONNECT_S:
                tail = " ".join(self._screen_.split())[-160:]
                self.stop()
                self.problem = f"Remote Control did not connect in {self.CONNECT_S} s{': ' + tail if tail else ''}"
                return
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.5)
                chunk = os.read(self.fd, 65536) if ready else b""
            except OSError:
                break
            if ready and not chunk:
                break
            if chunk:
                self._said_ = time.time()
            self._screen_ = (self._screen_ + _plain_(chunk.decode("utf-8", "replace")))[-self.SCREEN_CHARS:]
            if self.url is None and (m := SESSION_URL.search(self._screen_)):
                self.url = m[0]
            squashed = re.sub(r"\s+", "", self._screen_).lower()
            if self.url is None and "apiusagebilling" in squashed:
                self.stop()
                self.problem = ("Claude Code is logged in for API usage billing, not a claude.ai plan, and Remote Control "
                                f"needs one: {login_hint()}, with a claude.ai account")
                return
            for asked, said in FIRST_RUN.items():
                if self.url is None and asked in squashed:
                    self.stop()  # stopped first: a problem said is one with nothing left running
                    self.problem = f"{said}: run `claude --remote-control` once in a terminal, in {self.session._cwd_()}, and answer it"
                    return
            settled = time.time() - self.started > self.PROMPT_S and time.time() - self._said_ > self.QUIET_S
            if self.url is None and not self._typed_ and "❯" in self._screen_ and settled:
                # Typed in, not passed as --remote-control: so it carries the conversation, history and all, to the phone.
                self._type_(f"/remote-control {self.name}")
                self._typed_ = True
            if self.url is None and self._typed_ and not self._accepted_ and CONSENT in squashed:
                self._type_("")  # its first choice, Enable Remote Control: what hyp was asked to do
                self._accepted_ = True
        if self.url is None and self.problem is None:
            tail = " ".join(self._screen_.split())[-200:]
            self.problem = f"Claude Code ended before it connected{': ' + tail if tail else ''}"


class CloudHandoff:
    """A conversation handed off to Claude Code in the cloud: it goes on there, the Mac free to sleep, and comes back.

    The cloud can only be given files, not a conversation: so the
    conversation's files -- its own, and its parents' -- are committed and
    pushed to the GitHub repository --dir's `.hyp-github` names (`repository`),
    and a cloud session begun on it -- `claude --cloud` -- told to read them as
    the conversation so far (CLOUD_PROMPT). Its url, read off Claude Code's
    screen, is shown as a QR code, to open it in the Claude app, and kept in
    the session's front matter (`Session.cloud`), so a restart does not lose
    it. `take_back` brings it back: Claude Code's `--teleport`, the turns
    made there added. hyp's own messages there begin HYP_MARK, and are no
    turns.
    """

    cloud = True  # where it went, as the window says it
    SCREEN_CHARS = 20_000  # of the screen, the last, kept to read
    CONNECT_S = 90  # how long `claude --cloud` has to say its session's url
    TELEPORT_S = 240  # how long a take-back has, all told
    QUIET_S = 2.0  # how long the screen must have said nothing new, before something is typed at it

    def __init__(self, session, name=None):
        self.session = session
        self.name = name or session.title()  # the cloud session's, on claude.ai and the phone: as the turn's point is labeled
        self.url = f"https://claude.ai/code/{session.cloud}" if session.cloud else None
        self.problem = None  # why it is not handed off, where it is not
        self.started = time.time() if session.cloud else None
        self.pid = self.fd = None
        self._screen_ = ""
        self._said_ = 0.0  # when its screen last said something new
        self._trusted_ = False  # Claude Code's question, whether to trust --dir, answered
        # One teleport at a time: two at once -- a take-back while one is under way -- would share its terminal, and the
        # second ended the first's. A second waits for the first, and finds the session back.
        self._lock_ = threading.Lock()

    # 2. Declared API -----------------------------------------------------------

    def start(self):
        """Push the conversation's files and begin its cloud session, on a thread of its own: its url, or why not, as it comes."""
        self.started = time.time()
        threading.Thread(target=self.begin, daemon=True).start()

    def begin(self, push=True):
        """Push the conversation's files -- unless *push* is false, for files pushed already -- and begin its cloud session, here: its url, or why not.

        A conversation with a file of its own has the cloud session's id kept
        in it; a new branch, which has none till something is said in it, is
        its owner's to keep.
        """
        s = self.session
        self.started = self.started or time.time()
        try:
            top, remote, branch = repository(s.root)
            files = _conversation_files_(s, top)
            if push:
                push_files(s.root, [f for f, _ in files], f"hyp: {s.title()}, to the cloud")
            if s.cloud is not None and self._resume_(files):
                return
        except Failed as e:
            self.problem = str(e)
            return
        self._spawn_([claude_command(), "-n", self.name, "--cloud", self._opening_(files)])
        deadline = time.time() + self.CONNECT_S
        while self.pid is not None and time.time() < deadline:
            self._drain_(0.5)
            if (m := CLOUD_SESSION.search(self._screen_)) is not None:
                self.stop()  # the cloud session goes on: what began it here is done
                s.cloud, self.url = m[1], m[0]
                s.away, s.cloud_seen, s.cloud_upto = True, 0, len(s.path())
                if s.resumable:
                    s.save()  # where it went, in its file: a restart of hyp takes it back from there
                return
        tail = _said_(self._screen_)
        self.stop()
        self.problem = f"Claude Code did not begin a cloud session{': ' + tail if tail else ''}"

    def alive(self):
        return self.session.away

    def stop(self, timeout=5):
        """End the Claude Code running here for it, if one is: the cloud session goes on."""
        if self.pid is None:
            return
        pid, self.pid = self.pid, None
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    if os.waitpid(pid, os.WNOHANG) != (0, 0):
                        break
                except ChildProcessError:
                    break
                self._drain_(0.1)
            else:
                continue
            break
        try:
            os.close(self.fd)
        except OSError:
            pass

    def qr(self):
        """The url's QR code, as rows of 0 and 1 -- dark -- quiet zone included; None before the url is."""
        if self.url is None:
            return None
        return [list(row) for row in segno.make(self.url, error="m").matrix_iter(border=2)]

    def take_back(self):
        """Bring the conversation back from the cloud: `(how many turns came, branch)`, as `sync`."""
        return self.sync()

    def sync(self):
        """Bring home what was said in the point's cloud session since it was last here: `(how many turns, branch)`.

        The cloud session is kept, to go on in again. If the Mac said nothing
        meanwhile, the turns made there are the session's own, and *branch* is
        None. If both did, the conversation forked where the cloud last met the
        Mac: the cloud's turns come as a branch going on from there -- saved,
        the cloud session its own now, as it was -- and the session goes on
        here without one. Raises `Failed`, saying why, where it cannot be
        brought back.
        """
        s = self.session
        if not self._lock_.acquire(blocking=False):
            with self._lock_:  # brought back meanwhile, by the one under way
                return 0, None
        try:
            record, turns = self._fetch_()
        finally:
            self._lock_.release()
        came = turns[s.cloud_seen :]
        if came and s.resumable and len(s.path()) > s.cloud_upto:
            branch = s.branch(s.cloud_upto)
            _adopt_(branch, record, came)
            branch.cloud, branch.away, branch.cloud_seen = s.cloud, s.away, len(turns)
            branch.cloud_upto = len(branch.path())
            branch.save()
            s.cloud, s.away, s.cloud_seen, s.cloud_upto = None, False, 0, 0
            return len(came), branch
        if came:
            _adopt_(s, record, came)
        s.away, s.cloud_seen, s.cloud_upto = False, len(turns), len(s.path())
        return len(came), None

    # 4. Helpers ----------------------------------------------------------------

    def _resume_(self, files):
        """Go on in the point's own cloud session, rather than begin one: whether it could, its files pushed already.

        What was said there since it was here comes home first (`sync`). Then,
        if the Mac went on meanwhile, the cloud is told of its turns in a
        message (CATCH_UP); if both did, the cloud session is the branch's
        that its turns came as, and a new one is to be begun for this.
        """
        s = self.session
        known, last = s.cloud_upto, len(s.path())  # the Mac's turns the cloud has, and those it has
        came, branch = self.sync()
        if branch is not None:
            return False
        if not came and last > known:  # the Mac went on, the cloud not: told of the Mac's turns, as they were said
            turns = _written_out_(s.path()[known:last], known)
            if len(turns) <= CLOUD_TEXT_CHARS:
                self._post_(HYP_MARK + CATCH_UP_TURNS.format(turns=turns))
            else:
                said = ", then ".join(f"{f} ({how})" for f, how in files)
                self._post_(HYP_MARK + CATCH_UP.format(first=known + 1, last=last, files=said))
        s.away, s.cloud_upto, self.url = True, len(s.path()), f"https://claude.ai/code/{s.cloud}"
        if s.resumable:
            s.save()
        return True

    def _opening_(self, files):
        """What the cloud session is first told: the conversation written out, every turn as it was said -- what the phone
        shows of it -- or, past CLOUD_TEXT_CHARS, where its files are, to read (CLOUD_PROMPT)."""
        turns = _written_out_(self.session.path())
        if len(turns) <= CLOUD_TEXT_CHARS:
            return HYP_MARK + CLOUD_OPENING.format(turns=turns)
        return HYP_MARK + CLOUD_PROMPT.format(files=", then ".join(f"{f} ({how})" for f, how in files))

    def _post_(self, message):
        """Send *message* to the point's cloud session, as its next: `claude -p --cloud <id>`, which does not wait for an answer."""
        env = {k: v for k, v in os.environ.items() if k not in CHILD_MARKERS}
        try:
            done = subprocess.run([claude_command(), "-p", "--cloud", self.session.cloud, "--output-format", "json", message],
                                  cwd=self.session._cwd_(), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                  timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Failed(f"could not tell the cloud session what was said on the Mac: {e}") from None
        said = _event_(done.stdout.strip().splitlines()[-1]) if done.stdout.strip() else None
        if done.returncode or not (said or {}).get("ok"):
            raise Failed(f"the cloud session was not told what was said on the Mac: {(done.stderr or done.stdout).strip()[-200:]}")

    def _fetch_(self):
        """Teleport the point's cloud session here: `(record, turns)` -- Claude Code's record of it here, and every turn made there.

        Claude Code's `--teleport` checks out the branch the cloud pushed its
        changes to, if any: so first what is not committed is set aside (`git
        stash`), and after, the branch the notes were on is checked out again,
        the cloud's merged into it and pushed, and what was set aside put back.
        TAKE_BACK is typed at it, so that Claude Code records it here.
        """
        s = self.session
        top, remote, branch = repository(s.root)
        began = time.time()
        stashed = bool(_git_(top, "status", "--porcelain"))
        if stashed:
            _git_(top, "stash", "push", "--include-untracked", "-m", f"hyp: from the cloud, {s.title()}")
        merged = None
        try:
            self._teleport_()
            here = _git_(top, "branch", "--show-current")
            if here and here != branch:  # the cloud's branch, checked out: its changes merged into the notes'
                _git_(top, "checkout", branch)
                try:
                    _git_(top, "merge", "--no-edit", here)
                    merged = here
                except Failed as e:
                    _git_(top, "merge", "--abort")
                    raise Failed(f"what the cloud changed, on {here}, conflicts with {branch}: merge it by hand ({e})") from None
        finally:
            if stashed:
                try:
                    _git_(top, "stash", "pop")
                except Failed as e:
                    raise Failed(f"what was not committed is kept in git's stash, not put back: {e}") from None
        if merged:
            _git_(top, "push", remote, branch)
        record = _teleported_(s.cloud, began)
        return record, exchanges(record) if record is not None else []

    def _teleport_(self):
        """Run `claude --teleport` on the session's cloud session, type TAKE_BACK once it is loaded, then /exit; raises `Failed` where it does not load."""
        self._spawn_([claude_command(), "--teleport", self.session.cloud])
        deadline, typed, exited = time.time() + self.TELEPORT_S, None, False
        try:
            while self.pid is not None and time.time() < deadline:
                said = self._drain_(0.5)
                squashed = re.sub(r"\s+", "", self._screen_).lower()
                quiet = not said and time.time() - self._said_ > self.QUIET_S
                if typed is None and "sessionresumed" in squashed and quiet:
                    self._type_(HYP_MARK + TAKE_BACK)
                    typed = len(squashed)
                elif typed is not None and not exited and len(squashed) > typed + 40 and quiet:
                    self._type_("/exit")
                    exited = True
                if self.pid is not None and os.waitpid(self.pid, os.WNOHANG) != (0, 0):
                    self.pid = None
            if typed is None:
                tail = _said_(self._screen_)
                raise Failed(f"Claude Code could not bring the cloud session back{': ' + tail if tail else ''}")
        finally:
            self.stop()

    def _spawn_(self, args):
        self._screen_, self._said_, self._trusted_ = "", time.time(), False
        pid, fd = pty.fork()
        if pid == 0:  # the child: Claude Code, in --dir
            try:
                os.chdir(self.session._cwd_())
                for name in CHILD_MARKERS:
                    os.environ.pop(name, None)
                os.environ["TERM"] = "xterm-256color"
                os.execv(args[0], args)
            finally:
                os._exit(127)
        self.pid, self.fd = pid, fd

    def _drain_(self, seconds):
        """Read what its screen says for up to *seconds*, answering the trust question; whether it said anything."""
        try:
            ready, _, _ = select.select([self.fd], [], [], seconds)
            chunk = os.read(self.fd, 65536) if ready else b""
        except OSError:
            return False
        if not chunk:
            return False
        self._said_ = time.time()
        self._screen_ = (self._screen_ + _plain_(chunk.decode("utf-8", "replace")))[-self.SCREEN_CHARS:]
        if not self._trusted_ and "itrustthisfolder" in re.sub(r"\s+", "", self._screen_[-2000:]).lower():
            self._trusted_ = True  # hyp's own --dir, which its sessions run in already: trusted
            time.sleep(0.5)
            os.write(self.fd, b"\x1b[B")
            time.sleep(0.3)
            os.write(self.fd, b"\r")
        return True

    def _type_(self, text):
        try:
            os.write(self.fd, text.encode())
            time.sleep(0.5)
            os.write(self.fd, b"\r")
        except OSError:
            pass


def github_repo(root):
    """The GitHub repository --dir *root*'s notes are pushed to, as its GITHUB_FILE names it: `owner/name`; None without one."""
    try:
        with open(os.path.join(root, GITHUB_FILE), encoding="utf-8") as f:
            words = f.read().split()
    except OSError:
        return None
    said = words[0] if words else ""
    m = re.search(r"(?:github\.com[:/])?([\w.-]+/[\w.-]+?)(?:\.git)?/?$", said)
    return m[1] if m else None


def repository(root):
    """`(top, remote, branch)` of --dir *root*'s git repository: its top directory, the remote GITHUB_FILE names, the branch checked out.

    Raises `Failed`, saying what to do, where there is no such thing.
    """
    if "://" in str(root):
        raise Failed("the cloud is given a git repository, and --dir is a url: hand off to the phone instead")
    repo = github_repo(root)
    if repo is None:
        raise Failed(f"to hand off to the cloud, write the GitHub repository {root}'s notes go to -- owner/name -- into "
                     f"{os.path.join(root, GITHUB_FILE)}")
    try:
        top = _git_(root, "rev-parse", "--show-toplevel")
    except Failed:
        raise Failed(f"{root} is not a git repository: git init there, git remote add origin git@github.com:{repo}.git, "
                     "commit and push") from None
    found = re.compile(rf"github\.com[:/]{re.escape(repo)}(\.git)?/?$", re.I)
    try:  # each remote's url as configured, not as git's url.<base>.insteadOf would rewrite it
        urls = _git_(top, "config", "--get-regexp", r"^remote\..*\.url$").splitlines()
    except Failed:
        urls = []
    remote = next((line.split()[0][len("remote."):-len(".url")] for line in urls
                   if len(line.split()) >= 2 and found.search(line.split()[1])), None)
    if remote is None:
        raise Failed(f"no remote of {top} is github.com/{repo}: git remote add origin git@github.com:{repo}.git")
    branch = _git_(top, "branch", "--show-current")
    if not branch:
        raise Failed(f"{top} has no branch checked out: check one out, to push the notes to")
    own = os.path.realpath(config_dir())
    if own.startswith(os.path.realpath(top) + os.sep):
        # A take-back sets aside what is not committed -- which would be Claude Code's own login and records, while it runs
        # from them -- and a handoff could push them: so they must be no part of the repository.
        try:
            _git_(top, "check-ignore", "-q", own)
        except Failed:
            raise Failed(f"Claude Code's own directory, {own}, is in the repository and not ignored: add "
                         f"{os.path.relpath(own, top)}/ to {os.path.join(top, '.gitignore')}") from None
    return top, remote, branch


def _git_(cwd, *args):
    """git *args* run in *cwd*: what it printed; raises `Failed` with what it said, where it fails."""
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as e:
        raise Failed(f"could not run git: {e}") from None
    if done.returncode:
        said = (done.stderr or done.stdout).strip().splitlines()
        raise Failed(f"git {args[0]}: {said[-1] if said else f'exited with {done.returncode}'}")
    return done.stdout.strip()


def push_files(root, paths, message):
    """Commit *paths* -- of --dir *root*'s repository, relative to its top -- with *message*, if any changed, and push the branch."""
    top, remote, branch = repository(root)
    _git_(top, "add", "--", *paths)
    if _git_(top, "diff", "--cached", "--name-only", "--", *paths):
        _git_(top, "commit", "-m", message, "--", *paths)
    _git_(top, "push", remote, branch)


def _conversation_files_(session, top):
    """The files a conversation's turns are in, oldest first, relative to *top*, each with how many of its turns are the conversation's: `[(path, how)]`.

    A new branch, with no turn of its own yet, has no file: its parent's
    are the conversation's, up to the turn it goes on from.
    """
    chain, s = [], session
    while s is not None:
        chain.append(s)
        s = s.above
    chain.reverse()
    stop = None
    if not (session.resumable or session.prior or session.turns):
        chain, stop = chain[:-1], session.fork
    out = []
    for n, s in enumerate(chain):
        upto = chain[n + 1].fork if n + 1 < len(chain) else stop
        if upto is None:
            how = "all of its turns"
        else:
            keep = upto - s.fork  # its own turns, up to where the next goes on from it
            how = f"its first {keep} turn{'s' * (keep != 1)}"
        out.append((os.path.relpath(s.md_url, top), how))
    return out


def _said_(screen, chars=160):
    """What Claude Code's *screen* last said, to say why something failed: its last lines of words, its status bar and
    prompt -- usage, effort, rules -- left out, which would read as the reason."""
    lines = [" ".join(line.split()) for line in screen.splitlines()]
    words = [line for line in lines if re.search(r"[A-Za-z]{3}", line) and not re.match(r"^[─━╭╰│❯>⏵◐◑◒◓·\s]", line)
             and not re.search(r"/effort|shift\+tab|% (left|used)|usage|tokens", line, re.I)]
    return " · ".join(words)[-chars:]


def _written_out_(said, before=0):
    """Turns *said*, `(asked, answered)` each, written out as a message shows them: numbered from *before* + 1, who said what."""
    return "\n\n---\n\n".join(f"**You** (turn {before + n + 1}):\n\n{asked.strip()}\n\n**Claude:**\n\n{answered.strip()}"
                                for n, (asked, answered) in enumerate(said))


def _adopt_(session, record, came):
    """*session* goes on with the turns *came*, made in the cloud: in Claude Code's record of it here, at *record*, from now on.

    The turns before them are held by the conversation it was in -- or, a
    new branch, the one it was to fork (`Session.earlier`).
    """
    s, before = session, len(session.path())
    if s.resumable:
        s.earlier.append((s.id, s.start, before))
    else:  # a new branch, made in the cloud: the turns before its own held by the conversation it was to fork
        holder, start = s.source.holder(before) if s.source is not None else (None, 0)
        s.earlier = [e for e in s.source.earlier if e[2] <= start] if s.source is not None else []
        if holder is not None:
            s.earlier.append((holder, start, before))
        s.source = None
    s.id, s.start, s.resumable = posixpath.splitext(os.path.basename(record))[0], before, True
    s.turns += came
    s.model = _model_(record) or s.model


def _teleported_(cloud, since):
    """Claude Code's record here of the cloud session *cloud*, as `--teleport` wrote it since *since*: the newest; None for none."""
    found = [p for p in glob.glob(os.path.join(config_dir(), "projects", "*", "*.jsonl")) if os.path.getmtime(p) >= since]
    found = [p for p in found if cloud in open(p, encoding="utf-8", errors="replace").read()]
    return max(found, key=os.path.getmtime) if found else None


def config_dir():
    """Where Claude Code keeps its login and its conversations, for what hyp runs: `$CLAUDE_CONFIG_DIR`, else its own default."""
    return os.environ.get("CLAUDE_CONFIG_DIR") or DEFAULT_CONFIG


def conversation_path(session_id, config=None):
    """Where Claude Code keeps the conversation *session_id*, in *config* or `config_dir`: `projects/<directory>/<id>.jsonl`; None if nowhere."""
    found = glob.glob(os.path.join(config or config_dir(), "projects", "*", f"{session_id}.jsonl"))
    return max(found, key=os.path.getmtime) if found else None


def login_hint():
    """How to log in the Claude Code hyp runs, as the window says it: in its own configuration directory, where it has one."""
    if config_dir() == DEFAULT_CONFIG:
        return "run `claude auth login` in a terminal"
    return f"run `hyp --docs --claude-login` -- or `CLAUDE_CONFIG_DIR={config_dir()} claude auth login` -- in a terminal"


def fresh_copy(session_id):
    """The conversation copied as a new one, without its record of any claude.ai session -- the original removed: its id.

    Left out too: what keeps its history from a claude.ai session, written
    as it was resumed under another login than it began in; and what of it
    ran under another account's organization is given this one's -- Claude
    Code shows a claude.ai session only what ran under its own. The user,
    whose both accounts are, asked for it whole. None, nothing
    done, where it has neither: never handed off, it gets a claude.ai session
    with all of it, as it is. In `config_dir`; the
    original is removed only there, never from Claude Code's own default.
    """
    path = conversation_path(session_id)
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    events = [_event_(line) for line in lines]
    org = organization()
    foreign = lambda e: org is not None and _org_(e) not in (None, org)
    if not any(_remote_(e) or foreign(e) for e in events) and not _broken_(events):
        return None
    copy = str(uuid.uuid4())
    # The records left out are links in the chain a conversation is read back by, from its newest record to its first:
    # what pointed to one points past it, to its parent. Left as they were, the chain broke there, and a resume -- Claude,
    # and the phone -- had only what came after. Where it already broke, its link is the record before it in the file.
    skipped = {e["uuid"]: e.get("parentUuid") for e in events if _remote_(e) and e.get("uuid")}
    kept, previous, out = set(), None, []
    for line, event in zip(lines, events):
        if _remote_(event):
            continue
        changed = False
        if event and event.get("parentUuid"):
            parent = event["parentUuid"]
            while parent in skipped:
                parent = skipped[parent]
            if parent is not None and parent not in kept:
                parent = previous
            changed = parent != event["parentUuid"]
            event["parentUuid"] = parent
        if foreign(event):  # run under another account's organization: the one it goes on in now, so the phone is shown it
            event["attachment"]["organizationUuid"] = org
            changed = True
        if changed:
            line = json.dumps(event, ensure_ascii=False) + "\n"
        if event and event.get("uuid"):
            kept.add(event["uuid"])
            if not event.get("isSidechain"):
                previous = event["uuid"]
        out.append(line.replace(session_id, copy))
    with open(os.path.join(os.path.dirname(path), f"{copy}.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(out)
    if config_dir() != DEFAULT_CONFIG:
        os.remove(path)
        shutil.rmtree(path[: -len(".jsonl")], ignore_errors=True)
    return copy


def carry_over(session_id):
    """The conversation *session_id* copied into `config_dir`, from Claude Code's default one, where only that has it: whether it was.

    A session begun under the default login and resumed under hyp's own
    (`--claude-config-dir`) is found there by `--resume` only once its record is.
    """
    if config_dir() == DEFAULT_CONFIG or conversation_path(session_id) is not None:
        return False
    source = conversation_path(session_id, DEFAULT_CONFIG)
    if source is None:
        return False
    target = os.path.join(config_dir(), "projects", os.path.basename(os.path.dirname(source)))
    os.makedirs(target, exist_ok=True)
    shutil.copy2(source, target)
    return True


def organization(command=None):
    """The organization Claude Code is logged in under, its id, from `claude auth status`; None if it says none."""
    try:
        done = subprocess.run([command or claude_command(), "auth", "status", "--json"], capture_output=True, text=True,
                              timeout=20, cwd=os.path.expanduser("~"))
    except (OSError, subprocess.TimeoutExpired, Failed):
        return None
    status = _event_(done.stdout.strip()) if done.stdout.strip() else None
    return (status or {}).get("orgId")


def _org_(event):
    """The organization a record of a conversation says it ran under -- a `credential_org` attachment's -- or None."""
    attachment = (event or {}).get("attachment") if isinstance((event or {}).get("attachment"), dict) else {}
    return attachment.get("organizationUuid") if attachment.get("type") == "credential_org" else None


def _broken_(events):
    """Whether a conversation's chain -- each record's parent, back to its first -- names a parent it does not have."""
    known = {e["uuid"] for e in events if e and e.get("uuid")}
    return any(e and e.get("parentUuid") and e["parentUuid"] not in known for e in events)


def _remote_(event):
    """Whether a record of a conversation is of its claude.ai session -- what a resume reconnects to -- or keeps its history
    from one: `history-suppression`, written as it is restored under another account than it began in, which the user,
    whose both accounts are, asked to have shown all the same."""
    if not event:
        return False
    attachment = event.get("attachment") if isinstance(event.get("attachment"), dict) else {}
    return (event.get("type") in ("bridge-session", "history-suppression") or event.get("subtype") == "bridge_status"
            or attachment.get("type") == "remote_session_change")


def _at_(event):
    """When a record of the conversation was written, as time.time(); 0 where it says not."""
    try:
        return datetime.fromisoformat(str(event.get("timestamp")).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def anchor(session_id, k):
    """The uuid of the last message of turn *k* (1-based) of conversation *session_id*, as Claude Code records it: where a branch after it is resumed. None for none.

    Turns are counted by the prompts typed, as `Session.take_back` reads
    them; turn 0 is before the first.
    """
    path = conversation_path(session_id)
    if path is None:
        return None
    count, last, hyps = 0, None, False
    for e in _records_(path):
        if e["type"] == "user" and _prompt_(e):
            if count == k:
                return last
            count, last, hyps = count + 1, None, False
        elif e["type"] == "user" and _hyps_(e):
            hyps = True  # hyp's own message: what answers it is no turn's
        elif e["type"] == "assistant" and count == k and not hyps:
            last = e.get("uuid") or last
    return last if count == k else None


def exchanges(path, since=None):
    """The turns Claude Code's record at *path* holds, `(asked, answered)` each, in order -- those since *since*, a time, if given.

    Each prompt typed, and what Claude answered in text; what Claude Code
    adds besides -- its commands' output, its notes, tool results -- and
    hyp's own messages, with what answered them, are left out.
    """
    out, prompt, reply = [], None, []
    for e in _records_(path):
        if since is not None and _at_(e) < since:
            continue
        if e["type"] == "user" and (_prompt_(e) or _hyps_(e)):
            if prompt is not None:
                out.append((prompt, "\n\n".join(reply)))
            prompt = _text_(e.get("message", {}).get("content")).strip() if _prompt_(e) else None
            reply = []
        elif e["type"] == "assistant" and prompt is not None:
            text = _text_(e.get("message", {}).get("content")).strip()
            if text:
                reply.append(text)
    if prompt is not None:
        out.append((prompt, "\n\n".join(reply)))
    return out


def _records_(path):
    """The user's and Claude's records of Claude Code's record at *path*, in order: none of its notes.

    A message that came while Claude was answering is recorded as a queued
    command, not a user's: it is given as one, a prompt of its own.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            e = _event_(line)
            if not e:
                continue
            queued = e.get("attachment") if e.get("type") == "attachment" else None
            if isinstance(queued, dict) and queued.get("type") == "queued_command" and queued.get("prompt"):
                yield {"type": "user", "uuid": e.get("uuid"), "timestamp": e.get("timestamp"),
                       "message": {"role": "user", "content": str(queued["prompt"])}}
            elif e.get("type") in ("user", "assistant") and not e.get("isMeta"):
                yield e


def _model_(path):
    """The model Claude last answered on, in Claude Code's record at *path*; None for none."""
    models = [e.get("message", {}).get("model") for e in _records_(path) if e["type"] == "assistant"]
    return next((m for m in reversed(models) if m), None)


def parent_url(md_url, fields):
    """The file of the branch the session saved at *md_url* goes on from, as its front matter *fields* name it; None for a first branch."""
    parent = fields.get("parent") if isinstance(fields, dict) else None
    if not parent:
        return None
    parent = str(parent)
    if "://" in parent or parent.startswith("/"):
        return parent
    head, sep, path = md_url.rpartition("://")  # a url's scheme and host kept: normpath would fold its "//"
    return f"{head}{sep}{posixpath.normpath(posixpath.join(posixpath.dirname(path), parent))}"


def _relative_(url, start):
    """*url* relative to *start*'s directory, where both are on one filesystem; else *url* as it is."""
    head_a, _, path_a = url.rpartition("://")
    head_b, _, path_b = start.rpartition("://")
    if head_a != head_b:
        return url
    if head_a and path_a.split("/", 1)[0] != path_b.split("/", 1)[0]:
        return url  # another host, or bucket
    return posixpath.relpath(path_a, posixpath.dirname(path_b))


def _prompt_(event):
    """Whether Claude Code's record *event*, a user's, is a prompt typed: not a tool's result, a command's output, a reminder, a summary, a turn cut short, or hyp's own."""
    text = _text_(event.get("message", {}).get("content"))
    return bool(text.strip()) and not text.lstrip().startswith(("<", HYP_MARK, INTERRUPTED)) and not event.get("isCompactSummary")


def _hyps_(event):
    """Whether Claude Code's record *event*, a user's, is hyp's own message: a cloud handoff's opening, a take-back's."""
    return _text_(event.get("message", {}).get("content")).lstrip().startswith(HYP_MARK)


def _text_(content):
    """A message's text: as it is, or its text blocks joined."""
    if isinstance(content, str):
        return content
    return "\n\n".join(c.get("text", "") for c in content or () if isinstance(c, dict) and c.get("type") == "text")


def _plain_(text):
    """A terminal's output as text: its escape sequences and carriage returns gone."""
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[()][0-9A-Za-z]|\x1b[=>]|\r", "", text)


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
    """The account Claude Code is logged in as, as the window shows it -- email, organization, plan -- from `claude auth status`; None if not."""
    try:
        done = subprocess.run([command or claude_command(), "auth", "status", "--json"], capture_output=True, text=True,
                              timeout=timeout, cwd=os.path.expanduser("~"))
    except (OSError, subprocess.TimeoutExpired, Failed):
        return None
    status = _event_(done.stdout.strip()) if done.stdout.strip() else None
    if not status or not status.get("loggedIn"):
        return f"not logged in: {login_hint()}" if status else None
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
    """The Claude Code CLI to run: $HYP_CLAUDE, else `claude` on the PATH, else the one VS Code's extension carries."""
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
        return f"Claude Code is not logged in: {login_hint()}, then send again"
    if "no conversation found" in low:
        return "Claude Code has no record of this session here: it keeps them per machine and directory"
    return said.splitlines()[-1] if said else "Claude Code failed, saying nothing"


class ClaudeWindow(EditableTitle, DocWindow):
    """A session's window: the transcript above, scrolled as a document's; below it, what is being typed.

    Enter sends, Shift+Enter starts a new line, Cmd+V pastes; PageUp and
    PageDown, Home and End, and the wheel scroll the transcript; Esc or ×
    closes the window, and its owner saves the session. Its title is the
    session's: a click edits it, Enter keeps it -- *on_rename(window)*, for
    its owner to file -- and Esc puts it back. Beneath the title, the account
    Claude Code is logged in as (*account*, once its owner has read it), the
    session's usage, and what is left of the plan's quota; at the right of
    the box, the model, which a click switches, from the next turn on.

    **remote**, at the title row's right, hands the session off --
    *on_hand_off(window)*, for its owner to start a `Handoff` -- and while it
    is, its QR code covers the transcript, to open it on a phone, nothing is
    sent from the box, and the button, **take back**, brings it back --
    *on_take_back(window)*, the turns made there added.

    The transcript is the conversation's whole `path`, a branch's upstream
    turns too, a line marking where its own begin: a branch is opened from
    a turn's point on the disk, by its owner (`Session.branch`).
    """

    INPUT_LINES = 8  # the box grows to this many lines, then scrolls
    TITLE_PREFIX = "Claude · "

    def __init__(self, session, on_send, on_rename=lambda window: None, on_hand_off=None, on_take_back=None):
        super().__init__("Claude", posixpath.basename(session.md_url))
        self.session, self.on_send, self.on_rename = session, on_send, on_rename
        self.on_hand_off, self.on_take_back = on_hand_off, on_take_back
        self.handoff = None  # the Handoff, while the session is elsewhere
        self.handoff_lines = []  # what its panel says beneath the QR code, as last drawn
        self._shown_ = (None, None)  # the handoff's (url, problem), as its panel last drew them
        self.remote_button = pygame.Rect(0, 0, 0, 0)
        self.edit = AreaEdit()  # what is typed in the box
        self._box_selecting_ = False  # the mouse pressed in the box, and not yet let go: a drag selects
        self._init_title_()
        self.account = None  # who Claude Code runs as, as `account` last said; None until read
        self.sending = None  # what was sent, while its reply is awaited
        self.partial = None  # the reply coming in, while one is
        self.cancelled = False
        self.note = None  # what went wrong with the last turn
        self.input = pygame.Rect(0, 0, 0, 0)
        self.chip = pygame.Rect(0, 0, 0, 0)  # the model's, as last drawn
        self._render_()

    @property
    def text(self):
        """What is typed in the box."""
        return self.edit.text

    @text.setter
    def text(self, value):
        self.edit.set_text(value)

    def busy(self):
        return self.partial is not None

    def begin(self, text):
        self.sending, self.partial, self.note = text, "", None
        self.pinned = None  # the reply followed, not a turn kept in view
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

    def quota_line(self):
        """The session's quota line, and whose it is where the account has changed since: a turn shows this one's."""
        s = self.session
        line = s.quota_line()
        if s.plan and s.plan_account and self.account and s.plan_account != self.account:
            line += f" -- {s.plan_account.split(' · ')[0]}'s; a turn shows this account's"
        return line

    def _title_text_(self):
        return self.session.given_title()  # its start time, shown till it has a title, is no title to edit

    def _title_kept_(self, text):
        self.session.rename(text)
        self.on_rename(self)  # for the owner to file

    def close_input(self):
        self.cancelled = True

    def animating(self):
        # A handoff's url, or why it failed, comes on a thread of its own: drawn again till what is shown is what it says --
        # else the QR code waited, drawn only at the next event, a mouse moved.
        h = self.handoff
        unshown = h is not None and (h.url is None and h.problem is None or (h.url, h.problem) != self._shown_)
        return self.busy() or self.doc.pending() or unshown

    def handle(self, ev):
        """Returns "close" or None."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.close.collidepoint(ev.pos):
            return "close"
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.chip.collidepoint(ev.pos):
            if not self.busy():
                self.session.switch()
            return None
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.remote_button.collidepoint(ev.pos):
            self.keep_title()
            if self.handoff is not None and self.on_take_back is not None:
                self.on_take_back(self)
            elif self.handoff is None and self.on_hand_off is not None and not self.busy():
                self.on_hand_off(self)
            return None
        if self._title_event_(ev):  # a click off the title, while edited, keeps it and goes on: into the box, say
            return None
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self._box_selecting_ = False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.input.collidepoint(ev.pos):
            self.edit.press(ev.pos, shift=bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
            self._box_selecting_ = True
            return None
        if self._box_selecting_ and ev.type == pygame.MOUSEMOTION:
            self.edit.drag(ev.pos)
            return None
        if ev.type == pygame.TEXTINPUT:
            self.edit.handle(ev)
            return None
        if ev.type == pygame.KEYDOWN:
            k, mods = ev.key, pygame.key.get_mods() | getattr(ev, "mod", 0)
            cmd = mods & (pygame.KMOD_META | pygame.KMOD_CTRL)
            if k == pygame.K_ESCAPE:
                return "close"
            if k in (pygame.K_RETURN, pygame.K_KP_ENTER) and not mods & pygame.KMOD_SHIFT:
                if self.handoff is not None:
                    self.note = "the session is on the phone: take it back to go on here"
                elif self.text.strip() and not self.busy():
                    text = self.text.strip()
                    self.edit.set_text("")
                    self.on_send(self, text)
                return None
            if k in (pygame.K_PAGEUP, pygame.K_PAGEDOWN, pygame.K_HOME, pygame.K_END):
                return super().handle(ev)  # the transcript scrolled
            if cmd and k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_MINUS, pygame.K_KP_MINUS,
                             pygame.K_0, pygame.K_KP0):
                return super().handle(ev)  # Cmd and + or -, or 0: the transcript zoomed, as any document's window
            if k == pygame.K_c and cmd and self.edit.selection() is None:
                return super().handle(ev)  # the transcript's selection copied, to paste into the box, say
            self.edit.handle(ev)
            return None
        return super().handle(ev)

    def draw(self, surf, rect, font, small):

        rect = pygame.Rect(rect)
        pad, lh = 10, font.get_linesize()
        width = rect.width - 2 * pad - 4
        shown = min(len(self.edit.rows(font, width)), self.INPUT_LINES)
        if self.naming is not None:
            hint = "Enter: keep this title · Esc: put it back"
        elif self.handoff is not None:
            where = "in the cloud" if getattr(self.handoff, "cloud", False) else "on the phone"
            hint = f"{where} · take back: go on here, with what was said there"
        elif self.busy():
            hint = "Claude is answering …"
        elif self.session.above is not None and not self.session.own():
            hint = f"Enter: send -- a new branch, after turn {self.session.fork} · Esc: close, leaving no branch"
        else:
            hint = "Enter: send · Shift+Enter: new line · click the title: rename · Esc: close and save"
        box_h = 2 * pad + lh * shown + small.get_linesize() + 4
        self.title = "" if self.naming is not None else f"Claude · {self.session.title()}"  # being edited: drawn below
        remote = small.render("take back" if self.handoff is not None else "remote", True, TEXT)
        self.header_reserve = remote.get_width() + 22
        # The account first: what the session runs on, and where a session handed off goes.
        account = f"Claude Code as {self.account}" if self.account else "Claude Code's account: reading …"
        web = " · web access on" if self.session.web else ""  # what Claude may reach, said where whose it is
        self.sub = "\n".join([f"{account}{web} · {posixpath.basename(self.session.md_url)}",
                              f"{self.session.usage_line()} · {self.quota_line()}"])
        super().draw(surf, pygame.Rect(rect.x, rect.y, rect.width, rect.height - box_h), font, small)
        self._draw_title_(surf, font, small)
        zoom_w = small.size(f"{self.zoom:.0%}")[0]
        self.remote_button = remote.get_rect(topright=(self.close.x - zoom_w - 22, self.rect.y + 12)).inflate(10, 4)
        hot = self.remote_button.collidepoint(pygame.mouse.get_pos()) and (self.handoff is not None or not self.busy())
        pygame.draw.rect(surf, (52, 60, 76) if hot or self.handoff else (36, 42, 54), self.remote_button, border_radius=4)
        surf.blit(remote, remote.get_rect(center=self.remote_button.center))
        if self.handoff is not None:
            self._draw_handoff_(surf, font, small)
        self.rect = rect  # the box's too: a click on it is the window's, not the disk's
        self.input = pygame.Rect(rect.x, rect.bottom - box_h + 4, rect.width, box_h - 4)
        pygame.draw.rect(surf, (32, 38, 48), self.input, border_radius=8)
        pygame.draw.rect(surf, EDGE, self.input, 1, border_radius=8)
        self.edit.draw(surf, font, self.input.x + pad, self.input.y + pad, width, self.INPUT_LINES, DIM if self.busy() else TEXT)
        chip = small.render(f"model: {self.session.choice or 'default'} ▾", True, DIM if self.busy() else TEXT)
        bottom = self.input.bottom - pad - small.get_linesize() + 4
        self.chip = chip.get_rect(topright=(self.input.right - pad, bottom)).inflate(8, 2)
        hot = self.chip.collidepoint(pygame.mouse.get_pos()) and not self.busy()
        pygame.draw.rect(surf, (52, 60, 76) if hot else (40, 46, 58), self.chip, border_radius=4)
        surf.blit(chip, chip.get_rect(center=self.chip.center))
        room = self.chip.x - (self.input.x + pad) - 12  # beside the model's chip: a long note cut short, not drawn under it
        note = small.render(fit(small, self.note or hint, room), True, (240, 120, 110) if self.note else DIM)
        surf.blit(note, (self.input.x + pad, bottom))
        pygame.key.set_text_input_rect(self.input)

    def _draw_handoff_(self, surf, font, small):
        """Over the transcript, while the session is elsewhere: its QR code and url, or why it is not there."""
        h, b = self.handoff, self.body
        self._shown_ = (h.url, h.problem)
        pygame.draw.rect(surf, (24, 28, 36), b)
        qr = h.qr()
        if qr is not None:
            n = len(qr)
            cell = max(2, min(b.height - 4 * font.get_linesize(), b.width - 40) // n)
            side = cell * n
            x0, y0 = b.centerx - side // 2, b.y + 14
            pygame.draw.rect(surf, (255, 255, 255), (x0, y0, side, side))
            for r, row in enumerate(qr):
                for c, dark in enumerate(row):
                    if dark:
                        pygame.draw.rect(surf, (0, 0, 0), (x0 + c * cell, y0 + r * cell, cell, cell))
            top = y0 + side + 10
            # Whose claude.ai it is on: a phone signed into another account cannot open it.
            whose = (self.account or "").split(" · ")[0] or "the account Claude Code is logged in as"
            where = ("in the cloud: scan it with the phone -- the Claude app, or claude.ai/code -- and go on there, the Mac free to sleep"
                     if getattr(h, "cloud", False) else
                     "scan with the phone: the Claude app, or claude.ai/code -- the session goes on there")
            lines = [(font, h.url, TEXT), (small, where, DIM),
                     (small, f"on {whose}'s claude.ai: sign the phone into it, or switch Claude Code with `claude auth login`", DIM)]
        else:
            top = b.y + 20
            waiting = ("handing off: pushing the conversation, and beginning a cloud session …" if getattr(h, "cloud", False)
                       else "handing off: connecting to Remote Control …")
            lines = [(small, h.problem, (240, 120, 110))] if h.problem else [(font, waiting, DIM)]
        self.handoff_lines = [text for _, text, _ in lines]  # as drawn
        for f, text, color in lines:
            img = f.render(fit(f, text, b.width - 20), True, color)
            surf.blit(img, img.get_rect(midtop=(b.centerx, top)))
            top += f.get_linesize() + 4

    def _render_(self, follow=False):
        at_end = self.sy >= self._size[1] - self.body.height - 4
        self.anchor = self.cursor = None  # laid out again: a selection of the old layout would point anywhere
        self.doc = MarkdownDoc(self._transcript_())
        if follow or at_end:
            self.sy = 10**9  # clamped to the end at the next draw

    def _transcript_(self):
        """What the window shows: the conversation's whole `path` -- its upstream, then a line where its own begin -- and the turn being sent."""
        s = self.session
        lines = [f"# Claude · {s.title()}", ""]
        if s.lost:
            lines += [f"*A branch of {s.lost}, which cannot be read: the turns before this branch's are not shown.*", ""]
        for asked, answered in s.upstream():
            lines += ["## You", "", asked.strip(), "", "## Claude", "", answered, ""]
        if s.above is not None:
            mine = "this branch" if s.own() or self.sending is not None else "a new branch -- ask below"
            lines += ["---", "", f"*From here, {mine}: going on from {s.above.title()}, after its turn {s.fork}.*", ""]
        for asked, answered in s.own():
            lines += ["## You", "", asked.strip(), "", "## Claude", "", answered, ""]
        if self.sending is not None:
            lines += ["## You", "", self.sending, "", "## Claude", "", self.partial or "…", ""]
        return "\n".join(lines)



def said(body):
    """A saved transcript's *body* as what was said: its markup -- the `# Claude · <title>` heading, the `## You` and `## Claude` lines -- left out."""
    body = re.sub(r"^# Claude · .*$", "", body, flags=re.M)
    body = re.sub(r"^## (You|Claude)[ \t]*$", "", body, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def turns(body):
    """A saved transcript's *body* as its turns: `(asked, answered)` each, in order, as its `## You` and `## Claude` lines part them.

    What precedes the first `## You` -- the `# Claude · <title>` heading -- is
    no turn's, and a reply in more than one `## Claude` part is one answer. A
    `## You` inside fenced code is the code's: the window's markdown does not
    make a heading of it either, and counts its turns as this does. A turn
    cut short -- `[Request interrupted by user]`, as Claude Code records it
    -- is no turn: what comes after it is its turn's, as `exchanges` reads it.
    """
    out, part, lines, fence = [], None, [], None

    def close():
        text = "\n".join(lines).strip()
        if part == "You" and not (text.startswith(INTERRUPTED) and out):
            out.append([text, ""])
        elif out:  # Claude's, or a turn cut short, whose turn goes on: one answer
            out[-1][1] = f"{out[-1][1]}\n\n{'' if part == 'You' else text}".strip()

    for line in body.splitlines():
        mark = line.strip()[:3]
        if mark in ("```", "~~~"):
            fence = None if fence == mark else fence or mark
        heading = None if fence else re.match(r"^## (You|Claude)[ \t]*$", line)
        if heading:
            close()
            part, lines = heading.group(1), []
        else:
            lines.append(line)
    close()
    return [tuple(t) for t in out]


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
