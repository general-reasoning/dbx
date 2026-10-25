"""hyp --docs' Claude sessions: run through a fake Claude Code CLI, saved in the directory scanned, found again, resumed."""

import json
import re
import os
import stat
import sys
import time
from types import SimpleNamespace as NS

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from hyp import claude, docs, docview
from hyp.app import Worker
from hyp.engine import Engine
from hyp.store import Store

from test_hyp_engine import EchoSummarizer, HashEmbedder, quiet

# A stand-in for `claude -p ... --output-format stream-json`: each run's argv, cwd and stdin logged; its reply
# FAKE_REPLY (words streamed), or FAKE_FAIL's message as a failed result, or FAKE_STDERR on stderr with no result.
FAKE = r'''#!{python}
import json, os, sys
if sys.argv[1:3] == ["auth", "login"]:  # logged, as a run is, with the login's directory: what hyp ran it in
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "config": os.environ.get("CLAUDE_CONFIG_DIR")}}) + "\n")
    sys.exit(0)
if sys.argv[1:3] == ["auth", "status"]:  # FAKE_ACCOUNT, as `claude auth status --json` says it; read before stdin, which it never gets
    print(os.environ.get("FAKE_ACCOUNT", json.dumps({{"loggedIn": True, "email": "me@example.com", "orgName": "Example Org",
                                                      "orgId": "org-now", "subscriptionType": "team"}})))
    sys.exit(0)
if "--resume" in sys.argv and "-p" not in sys.argv:  # a session handed off: interactive, /remote-control typed at its prompt
    import time
    from datetime import datetime, timezone
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": ""}}) + "\n")
    if os.environ.get("FAKE_SILENT"):
        print("Claude Code, saying nothing of Remote Control", flush=True)
        time.sleep(30)
    if os.environ.get("FAKE_API_BILLING"):
        print("Claude Code  Opus 5.5 \u00b7 API Usage Billing", flush=True)
        time.sleep(30)
    if os.environ.get("FAKE_FIRST_RUN"):
        print("Let's get started. Choose the text style that looks best with your terminal", flush=True)
        time.sleep(30)
    print("Claude Code  resumed\n\u276f ", flush=True)
    typed = sys.stdin.readline().strip()
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write(json.dumps({{"typed": typed}}) + "\n")
    if os.environ.get("FAKE_CONSENT"):
        print("Remote Control  1. Enable Remote Control Opens a secure connection to claude.ai.  2. Never mind", flush=True)
        with open(os.environ["FAKE_LOG"], "a") as f:
            f.write(json.dumps({{"consent": sys.stdin.readline().strip() == ""}}) + "\n")
    sid = sys.argv[sys.argv.index("--resume") + 1]
    home = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "projects", "here")
    os.makedirs(home, exist_ok=True)
    now = lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with open(os.path.join(home, sid + ".jsonl"), "a") as f:  # what is said on the phone, as Claude Code records it
        for said, answer in json.loads(os.environ.get("FAKE_PHONE", "[]")):
            f.write(json.dumps({{"type": "user", "timestamp": now(), "message": {{"role": "user", "content": said}}}}) + "\n")
            f.write(json.dumps({{"type": "user", "timestamp": now(), "isMeta": True, "message": {{"role": "user", "content": "a note"}}}}) + "\n")
            f.write(json.dumps({{"type": "assistant", "timestamp": now(),
                                 "message": {{"role": "assistant", "model": "claude-sonnet-5-5", "content": [{{"type": "text", "text": answer}}]}}}}) + "\n")
    print("\x1b[1m /rc connecting\u2026\x1b[0m https://claude.ai/code/session_01FAKEurl\u2714", flush=True)
    while True:
        time.sleep(0.2)
text = sys.stdin.read()
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": text}}) + "\n")
if os.environ.get("FAKE_STDERR"):
    sys.stderr.write(os.environ["FAKE_STDERR"])
    sys.exit(1)
def put(e):
    print(json.dumps(e), flush=True)
sid = sys.argv[sys.argv.index("--resume" if "--resume" in sys.argv else "--session-id") + 1]
model = {{"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}}.get(sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "", "claude-opus-5-5")
put({{"type": "system", "subtype": "init", "session_id": sid, "model": model}})
put({{"type": "rate_limit_event", "rate_limit_info": {{"unifiedWindows": {{"five_hour": {{"utilization": 0.71}}, "seven_day": {{"utilization": 0.65}}}}}}}})
if os.environ.get("FAKE_FAIL"):
    put({{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": os.environ["FAKE_FAIL"]}})
    sys.exit(1)
put({{"type": "stream_event", "event": {{"type": "content_block_start", "content_block": {{"type": "text", "text": ""}}}}}})
for w in os.environ.get("FAKE_REPLY", "Hello there.").split(" "):
    put({{"type": "stream_event", "event": {{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": w + " "}}}}}})
if os.environ.get("FAKE_TOOL"):
    put({{"type": "stream_event", "event": {{"type": "content_block_start", "content_block": {{"type": "tool_use", "name": "Read"}}}}}})
    put({{"type": "stream_event", "event": {{"type": "content_block_start", "content_block": {{"type": "text", "text": ""}}}}}})
    put({{"type": "stream_event", "event": {{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": "After reading."}}}}}})
put({{"type": "result", "subtype": "success", "is_error": False, "result": "(final)", "session_id": sid, "total_cost_usd": 0.25,
     "usage": {{"input_tokens": 10, "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 2000, "output_tokens": 300}}}})
'''


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """The fake CLI, as HYP_CLAUDE; ``cli.runs()`` what each run was given."""
    path = tmp_path / "bin" / "claude"
    path.parent.mkdir()
    path.write_text(FAKE.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "runs.jsonl"
    monkeypatch.setenv("HYP_CLAUDE", str(path))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))  # where the conversations handed off are recorded
    for v in ("FAKE_REPLY", "FAKE_FAIL", "FAKE_STDERR", "FAKE_TOOL", "FAKE_ACCOUNT", "FAKE_PHONE", "FAKE_FIRST_RUN",
              "FAKE_API_BILLING", "FAKE_SILENT", "FAKE_CONSENT"):
        monkeypatch.delenv(v, raising=False)
    return NS(runs=lambda: [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else [])


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("# Alpha\n")
    return root


def test_a_first_turn_starts_the_session_by_its_id_in_the_directory_with_read_only_tools(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_REPLY", "Hello there, from Claude.")
    s = claude.Session(str(tree))
    got = []
    assert s.send("Hi\nsecond line", got.append) == "Hello there, from Claude."
    assert "".join(got).strip() == "Hello there, from Claude."
    (run,) = cli.runs()
    argv = run["argv"]
    assert run["cwd"] == os.path.realpath(tree) and run["stdin"] == "Hi\nsecond line"
    assert argv[argv.index("--session-id") + 1] == s.id and "--resume" not in argv
    assert "--tools=Read,Glob,Grep" in argv and "--allowedTools=Read,Glob,Grep" in argv
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk" and argv[argv.index("--output-format") + 1] == "stream-json"
    assert str(tree) in argv[argv.index("--append-system-prompt") + 1]
    assert s.resumable and s.model == "claude-opus-5-5" and s.title() == "Hi"
    s.send("Again")
    argv = cli.runs()[1]["argv"]
    assert argv[argv.index("--resume") + 1] == s.id and "--session-id" not in argv, "the next turn resumes it"


def test_a_tool_claude_uses_is_noted_in_the_reply(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_TOOL", "1")
    reply = claude.Session(str(tree)).send("read a.md")
    assert "*Read …*" in reply and reply.endswith("After reading.")


def test_a_session_is_saved_as_one_markdown_and_resumed_from_it(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_REPLY", "First answer.")
    s = claude.Session(str(tree), "sessions")
    s.send("First question")
    url = s.save()
    assert os.path.dirname(url) == str(tree / "sessions") and os.listdir(tree / "sessions") == [os.path.basename(url)]
    fields, body = docview.front_matter(open(url).read())
    assert claude.is_session(fields) and fields["session"] == s.id and fields["title"] == "First question"
    assert fields["model"] == "claude-opus-5-5" and "## You" in body and "First answer." in body

    monkeypatch.setenv("FAKE_REPLY", "Second answer.")
    again = claude.Session.load(url, str(tree), "sessions")
    assert again.id == s.id and again.resumable and again.md_url == url and "First answer." in again.prior
    again.send("Second question")
    argv = cli.runs()[-1]["argv"]
    assert argv[argv.index("--resume") + 1] == s.id
    again.save()
    text = open(url).read()
    assert text.count("First answer.") == 1 and "Second answer." in text and docview.front_matter(text)[0]["title"] == "First question"


def test_a_failed_or_cancelled_turn_is_not_kept(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "Not logged in · Please run /login")
    s = claude.Session(str(tree))
    with pytest.raises(claude.Failed, match="not logged in"):
        s.send("something")
    assert s.turns == []
    monkeypatch.delenv("FAKE_FAIL")
    monkeypatch.setenv("FAKE_STDERR", "Error: No conversation found with session ID: x")
    with pytest.raises(claude.Failed, match="no record of this session"):
        s.send("again")
    monkeypatch.delenv("FAKE_STDERR")
    assert claude.Session(str(tree)).send("x", cancelled=lambda: True) is None


def test_no_cli_says_how_to_get_one(tree, monkeypatch):
    monkeypatch.setenv("HYP_CLAUDE", str(tree / "no-such-claude"))
    with pytest.raises(claude.Failed, match="could not run Claude Code"):
        claude.Session(str(tree)).send("x")


def test_a_url_directory_gives_claude_no_tools():
    s = claude.Session("ssh://host/notes", command="claude")
    argv = s._args_()
    assert "--tools=" in argv and s._cwd_() == os.path.expanduser("~")
    assert "cannot see the documents" in argv[argv.index("--append-system-prompt") + 1]


def test_front_matter_is_read_and_not_drawn():
    fields, body = docview.front_matter("---\nhyp: claude-session\nsession: abc\n---\n\n# Title\n")
    assert fields == {"hyp": "claude-session", "session": "abc"} and body.strip() == "# Title"
    assert [b.kind for b in docview.parse_markdown("---\na: 1\n---\n# T\n")] == ["heading"]
    assert docview.front_matter("# no front matter\n---\n") == (None, "# no front matter\n---\n")


@pytest.fixture
def saved(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_REPLY", "An answer.")
    s = claude.Session(str(tree))
    s.send("What is alpha?")
    return tree, s.save()


def test_a_saved_session_is_a_document_labeled_as_one(saved):
    tree, url = saved
    found = {d.key: d for d in docs.documents(str(tree), progress=lambda *a: None)}
    assert set(found) == {str(tree / "a.md"), url}
    d = found[url]
    assert d.extra["kind"] == claude.KIND and d.summary == "Claude · What is alpha?" and "An answer." in d.embed


@pytest.fixture
def app(saved, monkeypatch):
    from hyp.app import Worker

    tree, url = saved
    a = docs.HypDocs(engine=None, worker=Worker(), args=NS(dir=str(tree), claude_dir="sessions"))
    a.run_async = lambda fn, then: then(fn())
    a.refreshed = 0
    a.settings["sessions on remote"] = False  # on, by default: the tests of handing off turn it on

    def refresh(progress):
        a.refreshed += 1
        return True

    a.engine = NS(refresh=refresh, snapshot=lambda: a.snap, shards={}, shattered={})
    found = docs.documents(str(tree), progress=lambda *a: None)
    a.snap = NS(positions=np.array([0.1 + 0.1j, -0.4 + 0.2j]), summaries=[d.summary for d in found],
                texts=[d.text for d in found], neighbors=np.array([[1], [0]]), points=[dict(d.extra) for d in found])
    monkeypatch.setenv("FAKE_REPLY", "Reply one.")
    return a


def _type_(window, text):
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text=text))
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))


def test_the_claude_button_opens_a_window_a_third_of_the_area_in_the_middle(app):
    app.open_claude()
    window = app.center
    assert isinstance(window, claude.ClaudeWindow) and app.layers[-1] == "center" and app.focus is window
    app.draw(0)
    a, r = app.area(), window.rect
    assert r.center == a.center and r.width * r.height == pytest.approx(a.width * a.height / 3, rel=0.02)
    app.open_claude()
    assert app.center is window, "one at a time"


def test_every_turn_is_saved_into_the_claude_dir_and_embedded_at_once(app, saved):
    tree, _ = saved
    app.open_claude()
    window = app.center
    _type_(window, "Hello Claude")
    app.drain(0)  # the reply's deltas, as the loop hands them over
    assert window.session.turns == [("Hello Claude", "Reply one.")] and not window.busy()
    (saved_md,) = os.listdir(tree / "sessions")
    assert "Reply one." in (tree / "sessions" / saved_md).read_text() and app.refreshed == 1, "before it is closed"
    _type_(window, "Once more")
    app.drain(0)
    assert (tree / "sessions" / saved_md).read_text().count("Reply one.") == 2 and app.refreshed == 2
    app.draw(0)
    app._close_(window)
    assert os.listdir(tree / "sessions") == [saved_md] and app.refreshed == 2, "closed: nothing left to file"


def test_a_window_closed_unused_saves_nothing(app, saved):
    tree, _ = saved
    app.open_claude()
    app._close_(app.center)
    assert not (tree / "sessions").exists() and app.refreshed == 0


def test_a_click_on_a_saved_session_resumes_it_where_it_was_saved(app, saved, cli):
    _, url = saved
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.__click_point__(i, 0.0)
    window = app.center
    assert isinstance(window, claude.ClaudeWindow) and window.session.md_url == url and window.session.resumable
    _type_(window, "And beta?")
    app._close_(window)
    text = open(url).read()
    assert "What is alpha?" in text and "And beta?" in text and "Reply one." in text
    assert "--resume" in cli.runs()[-1]["argv"]


def test_what_is_typed_reaches_the_session_wherever_the_mouse_is(app, monkeypatch):
    app.open_claude()
    app.draw(0)
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (2, app.screen.get_height() // 2))  # off the window
    app.handle(pygame.event.Event(pygame.TEXTINPUT, text="typed away"), 0)
    assert app.center.text == "typed away"


def test_a_turn_that_fails_says_why_and_gives_the_text_back(app, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "Not logged in · Please run /login")
    app.run_async = docs.HypDocs.run_async.__get__(app)  # on its thread, as in the app
    app.open_claude()
    window = app.center
    _type_(window, "Hello")
    deadline = time.monotonic() + 10
    while (app.busy() or window.busy()) and time.monotonic() < deadline:
        app.drain(0)
        time.sleep(0.01)
    assert not window.busy() and "not logged in" in window.note and window.text == "Hello" and window.session.turns == []


@pytest.mark.parametrize("rel, ok", [("sessions", True), ("notes/claude", True), ("./s", True),
                                     ("/abs", False), ("../up", False), (".claude", False), ("node_modules/x", False)])
def test_claude_dir_must_be_a_scanned_directory_under_dir(rel, ok):
    from hyp import app as app_mod

    assert (app_mod._unscanned_(rel) is None) == ok
    if not ok:
        with pytest.raises(SystemExit):
            app_mod.main(["--docs", "--claude-dir", rel])
    with pytest.raises(SystemExit):
        app_mod.main(["--claude-dir", "sessions"])  # with --docs only


def test_a_session_is_embedded_whole_and_no_document_by_its_front_matter(tree):
    long = "x " * 6000
    (tree / "fm.md").write_text("---\ntitle: noise\nid: 1234\n---\n\n# Body\n")
    s = claude.Session(str(tree))
    s.turns = [("question", long + "the end of it")]
    url = s.save()
    found = {d.key: d for d in docs.documents(str(tree), progress=lambda *a: None)}
    assert found[str(tree / "fm.md")].embed == "fm.md\n\n# Body" , "front matter is not embedded"
    session = found[url]
    assert session.embed.endswith("the end of it") and s.id not in session.embed and "hyp:" not in session.embed
    assert len(session.text) <= docs.TEXT_CHARS, "what the point keeps to show is as any document's"


class CountingEmbedder:
    name = "fake:count"

    def __init__(self):
        self.seen = []

    def prepare(self, progress=print):
        pass

    def __call__(self, texts):
        self.seen += list(texts)
        return np.array([[float(len(t)), 1.0 + t.count("b")] for t in texts], dtype=np.float32)


def test_a_long_text_is_embedded_a_piece_at_a_time_and_a_short_one_as_it_was():
    base = CountingEmbedder()
    emb = docs.ChunkedEmbedder(base, chunk=100)
    short = "name.md\n\n" + "a" * 100
    assert np.array_equal(emb([short]), base([short])), "one piece: the base's own vector"
    assert emb.name == base.name, "so a corpus is not embedded again"
    long = "name.md\n\n" + ("a" * 80 + "\n\n") + ("b" * 80 + "\n\n") + "c" * 30
    pieces = emb.pieces(long)
    assert len(pieces) == 3 and all(p.startswith("name.md\n\n") for p in pieces) and pieces[1].endswith("b" * 80)
    base.seen.clear()
    (v,) = emb([long])
    assert base.seen == pieces
    units = [u / np.linalg.norm(u) for u in base(pieces)]
    mean = np.mean(units, axis=0)
    assert np.allclose(v / np.linalg.norm(v), mean / np.linalg.norm(mean), atol=1e-6)


def test_a_refresh_embeds_what_changed_and_nudges_the_layout_from_where_it_was(tmp_path, monkeypatch):
    from hyp import layout
    from hyp.engine import Engine
    from hyp.store import Store

    from test_hyp_engine import EchoSummarizer, quiet

    texts = {f"k{i}": f"text {i}" for i in range(6)}
    source = lambda progress: [docs.Doc(key=k, text=t, embed=t, summary=k) for k, t in texts.items()]
    rng = np.random.default_rng(0)
    embedder = type("Random", (), {"name": "fake:rand", "prepare": lambda self, progress=print: None,
                                   "__call__": lambda self, ts: rng.normal(size=(len(ts), 8)).astype(np.float32)})()
    engine = Engine(Store(tmp_path), embedder, EchoSummarizer(), source=source)
    engine.start(quiet)
    before = engine.store.layout.copy()
    seen, real = {}, layout.compute
    monkeypatch.setattr(layout, "compute", lambda e, previous=None, seed=0: (seen.__setitem__("previous", previous), real(e, previous=previous, seed=seed))[1])
    assert engine.refresh(quiet) is False and "previous" not in seen, "nothing new: nothing laid out"
    texts["k6"] = "a new one"
    assert engine.refresh(quiet) is True
    assert np.array_equal(seen["previous"], before) and len(engine.store.layout) == 7


def test_a_sessions_usage_adds_up_turn_by_turn_and_is_kept_with_it(cli, tree):
    s = claude.Session(str(tree))
    s.send("one")
    s.send("two")
    assert s.usage == {"tokens_in": 6020, "tokens_out": 600, "cost_usd": 0.5}
    line = s.usage_line()
    assert line == "claude-opus-5-5 · 6.0k in · 600 out · ≈$0.50 at API rates"
    assert re.fullmatch(r"quota \(as of \d\d:\d\d\): 5 h 29% left · 7 d 35% left", s.quota_line())
    again = claude.Session.load(s.save(), str(tree))
    assert again.usage == s.usage and again.model == "claude-opus-5-5"
    again.send("three")
    assert again.usage["tokens_in"] == 9030, "counted on from where it was saved"


def test_the_model_is_switched_from_the_next_turn_on_and_kept_with_the_session(cli, tree):
    s = claude.Session(str(tree))
    s.send("one")
    assert "--model" not in cli.runs()[-1]["argv"], "Claude Code's default, unless one is asked for"
    assert [s.switch() or s.choice for _ in claude.MODELS] == ["opus", "sonnet", "haiku", None]
    s.switch(), s.switch(), s.switch()
    s.send("two")
    argv = cli.runs()[-1]["argv"]
    assert argv[argv.index("--model") + 1] == "haiku" and "--resume" in argv and s.model == "claude-haiku-5-5"
    again = claude.Session.load(s.save(), str(tree))
    assert again.choice == "haiku"


def test_a_click_on_the_model_switches_it_and_the_usage_is_shown(app):
    app.open_claude()
    window = app.center
    app.draw(0)
    assert "in · " in window.sub and "API rates" in window.sub
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.chip.center))
    assert window.session.choice == "opus"


def test_the_quota_says_what_is_left_and_when_it_resets():
    s = claude.Session("/tmp")
    assert s.quota_line() == "quota: known after a turn"
    now = time.mktime((2026, 10, 21, 9, 0, 0, 0, 0, -1))
    s.plan = {"five_hour": {"utilization": 0.04, "resetsAt": now + 3600},
              "seven_day": {"utilization": 0.41, "resetsAt": now + 3 * 86400}}
    assert s.quota_line(now) == "quota: 5 h 96% left (resets 10:00) · 7 d 59% left (resets Sat 09:00)"
    s.overage = True
    assert s.quota_line(now).endswith(" · on overage")


def test_the_account_is_who_claude_code_is_logged_in_as(cli, monkeypatch):
    assert claude.account() == "me@example.com · Example Org · team plan"
    monkeypatch.setenv("FAKE_ACCOUNT", json.dumps({"loggedIn": False}))
    assert claude.account().startswith("not logged in")
    monkeypatch.setenv("FAKE_ACCOUNT", "not json")
    assert claude.account() is None


def test_the_account_is_read_as_a_window_opens_and_after_each_turn(app, cli, monkeypatch):
    reads = []
    monkeypatch.setattr(claude, "account", lambda real=claude.account: reads.append(1) or real())
    app.open_claude()
    window = app.center
    app.draw(0)
    assert window.sub.split("\n")[0].startswith("Claude Code as me@example.com · Example Org · team plan · "), "the account first"
    assert "quota: known after a turn" in window.sub.split("\n")[1]
    _type_(window, "Hello")
    app.drain(0)
    app.draw(0)
    assert re.search(r"quota \(as of \d\d:\d\d\): 5 h 29% left · 7 d 35% left$", window.sub)
    assert window.session.plan_account == window.account and len(reads) == 2, "opened, and after the turn"
    app._close_(window)
    app.open_claude()
    assert app.center.account == window.account and len(reads) == 3, "read again for the next window"


def test_an_account_switched_to_is_shown_and_the_quota_said_to_be_the_old_ones_till_a_turn(app, cli, monkeypatch):
    app.open_claude()
    window = app.center
    _type_(window, "Hello")
    app.drain(0)
    monkeypatch.setenv("FAKE_ACCOUNT", json.dumps({"loggedIn": True, "email": "other@example.com", "orgName": "Other",
                                                   "subscriptionType": "max"}))
    app._close_(window)
    app.open_claude(window.session)  # the same session, the turn's quota with it
    again = app.center
    app.draw(0)
    head, usage = again.sub.split("\n")
    assert head.startswith("Claude Code as other@example.com · Other · max plan"), "the account switched to"
    assert usage.endswith("-- me@example.com's; a turn shows this account's")
    _type_(again, "Again")
    app.drain(0)
    app.draw(0)
    assert again.session.plan_account == again.account and "a turn shows" not in again.sub


def _click_title_(window):
    """A click at the title's end -- where the cursor goes, now that a click places it -- and its release."""
    pos = (window.title_rect.right - 1, window.title_rect.centery)
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos))


def test_a_sessions_title_is_edited_in_its_window_and_relabels_its_point(app, saved):
    tree, url = saved
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.__click_point__(i, 0.0)
    window = app.center
    app.draw(0)
    assert window.title == "Claude · What is alpha?"
    _click_title_(window)
    assert window.naming.text == "What is alpha?" and window.naming.cursor == len("What is alpha?")
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE, mod=pygame.KMOD_META, unicode=""))
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="Alpha, explained"))
    assert window.text == "", "the title takes what is typed, not the box"
    refreshed = app.refreshed
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))
    assert window.naming is None and app.refreshed == refreshed + 1, "filed at once"
    fields, body = docview.front_matter(open(url).read())
    assert fields["title"] == "Alpha, explained" and body.lstrip().startswith("# Claude · Alpha, explained")
    found = {d.key: d for d in docs.documents(str(tree), progress=lambda *a: None)}
    assert found[url].summary == "Claude · Alpha, explained" and found[url].extra["updated"] == fields["updated"]


def test_escape_puts_the_title_back_and_keeps_the_window_open(app):
    app.open_claude()
    window = app.center
    app.draw(0)
    _click_title_(window)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text=" and more"))
    assert window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="")) is None
    assert window.naming is None and window.session._title_ is None and app.center is window


def test_a_title_given_before_any_turn_waits_for_the_first(app, saved):
    tree, _ = saved
    app.open_claude()
    window = app.center
    app.draw(0)
    _click_title_(window)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text=" named"))
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))
    assert not (tree / "sessions").exists(), "no turn: no file to write"
    _type_(window, "Hello")
    app.drain(0)
    (md,) = os.listdir(tree / "sessions")
    assert docview.front_matter((tree / "sessions" / md).read_text())[0]["title"] == "named"


def test_a_sessions_point_is_dated_by_its_last_turn(app, saved):
    _, url = saved
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.snap.points[i]["updated"] = "2026-10-20T08:15:42"
    assert app.__point_datetime__(i) == "2026-10-20 08:15"


def _right_click_empty_disk_(app, kind="claude"):
    """A right click on empty disk -- off any window, off any point -- and *kind* picked from its menu: the box it opens."""
    app.draw(0)
    app.hover = None
    w = app.center
    pos = (w.rect.left - 30, w.rect.centery) if w is not None else app.area().center
    assert app.layer_at(pos) == "disk"
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=pos), 0)
    assert [v for _, v in app.popup.current()] == ["claude", "markdown"], "Claude or Markdown"
    app.popup.select(kind)
    app.popup = None
    return app.box


def test_a_text_typed_on_empty_disk_is_asked_of_claude_in_a_new_session(app, saved):
    tree, _ = saved
    box = _right_click_empty_disk_(app)
    assert box is not None and "ask Claude" in box.hint
    app.handle(pygame.event.Event(pygame.TEXTINPUT, text="What is in here?"), 0)
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"), 0)
    app.drain(0)
    window = app.center
    assert app.box is None and isinstance(window, claude.ClaudeWindow)
    assert window.session.turns == [("What is in here?", "Reply one.")], "its first turn, not a point of its own"
    first = window.session
    app.__submit_text__("Another thing")
    app.drain(0)
    assert app.center.session is not first and app.center.session.turns[0][0] == "Another thing", "a new session each time"
    assert len(os.listdir(tree / "sessions")) == 2


def test_a_right_click_on_a_document_can_ask_claude_about_it(app, saved):
    from hyp.docs import TilePicker

    tree, _ = saved
    i = next(k for k, p in enumerate(app.snap.points) if p["path"].endswith("a.md"))
    app.__context_point__(i, (100, 100))
    picker = app.popup
    assert isinstance(picker, TilePicker)
    app.draw(0)
    picker.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=picker.claude_row().center))
    window = app.center
    assert not picker.open and isinstance(window, claude.ClaudeWindow) and window.text == "About a.md: "
    assert window.session.turns == [], "the question is the user's to finish and send"


def test_a_point_with_no_file_shows_its_text_and_reads_nothing(app, monkeypatch):
    monkeypatch.setattr(docs, "load", lambda *a: pytest.fail("no file to load"))
    app.snap.points[0] = {"key": None}
    app.snap.texts[0] = "Yep"
    app.place(0, "center")
    assert isinstance(app.center.doc, docview.MarkdownDoc)


def _key_(window, key, mod=0):
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod, unicode=""))


def test_a_title_is_edited_anywhere_in_it_by_its_cursor(app, monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_claude()
    window = app.center
    window.session.rename("alpha gamma")
    app.draw(0)
    assert window.title == "Claude · alpha gamma"
    _click_title_(window)
    app.draw(0)
    e = window.naming
    _key_(window, pygame.K_LEFT, pygame.KMOD_ALT)  # a word back: before "gamma"
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="beta "))
    assert e.text == "alpha beta gamma" and e.cursor == len("alpha beta ")
    _key_(window, pygame.K_HOME)
    _key_(window, pygame.K_DELETE)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="A"))
    _key_(window, pygame.K_RIGHT, pygame.KMOD_META)
    assert e.text == "Alpha beta gamma" and e.cursor == len(e.text)
    _key_(window, pygame.K_LEFT)
    _key_(window, pygame.K_BACKSPACE)
    assert e.text == "Alpha beta gama" and e.cursor == len("Alpha beta gam")
    font = window._font_
    x0, lead, first = e._laid_
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x0 + font.size(lead + "Alpha")[0], window.title_rect.centery)))
    assert e.cursor == len("Alpha"), "a click places the cursor"
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"))
    assert window.naming is None and window.session.title() == "Alpha beta gama"


def test_a_new_session_is_titled_from_nothing_not_its_start_time(app):
    app.open_claude()
    window = app.center
    app.draw(0)
    assert window.title.startswith("Claude · 20"), "shown by its start time until it has a title"
    _click_title_(window)
    assert window.naming.text == ""


def test_a_long_title_scrolls_to_keep_the_cursor_in_view():
    from hyp.docview import LineEdit

    pygame.font.init()
    font, surf = pygame.font.Font(None, 18), pygame.Surface((400, 40))
    e = LineEdit("word " * 40)
    e.draw(surf, font, 0, 0, 200, (255, 255, 255), prefix="Claude · ")
    x0, lead, first = e._laid_
    assert first > 0 and lead.endswith("…") and font.size(lead + e.text[first:e.cursor])[0] <= 200
    e.cursor = 0
    e.draw(surf, font, 0, 0, 200, (255, 255, 255), prefix="Claude · ")
    assert e._laid_[2] == 0, "back at the start, unscrolled"


def test_a_click_in_the_box_keeps_the_title_and_typing_goes_there(app):
    app.open_claude()
    window = app.center
    window.session.rename("old")
    app.draw(0)
    _click_title_(window)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text=" and new"))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.input.center))
    assert window.naming is None and window.session.title() == "old and new", "kept, not lost"
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="a question"))
    assert window.text == "a question" and window.session.title() == "old and new"


def test_a_click_on_the_disk_keeps_the_title_too(app):
    app.open_claude()
    window = app.center
    window.session.rename("old")
    app.draw(0)
    _click_title_(window)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="er"))
    off = (window.rect.left - 5, window.rect.centery)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=off), 0)
    assert window.naming is None and window.session.title() == "older"


def test_one_click_closes_a_window_after_a_press_whose_release_never_came(app):
    i = next(k for k, p in enumerate(app.snap.points) if p["path"].endswith("a.md"))
    app.place(i, "center")
    app.draw(0)
    window = app.center
    app.press = (5, 5)  # the disk's: let go outside the window, its release never came
    app.capture = window
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.close.center), 0)
    assert app.center is None, "the first click, not the second"


def test_the_activating_click_is_passed_through_on_macos(app):
    assert os.environ.get("SDL_MOUSE_FOCUS_CLICKTHROUGH") == "1"


@pytest.fixture
def typing(monkeypatch):
    """Whether pygame's text input is on, as the app last switched it: its start and stop recorded."""
    state = NS(on=False)
    monkeypatch.setattr(pygame.key, "start_text_input", lambda: setattr(state, "on", True))
    monkeypatch.setattr(pygame.key, "stop_text_input", lambda: setattr(state, "on", False))
    return state


def test_typing_reaches_a_session_that_replaced_another_its_title_then_its_box(app, typing):
    app._typing_ = False
    app.open_claude()
    app._sync_text_input_()
    assert typing.on, "a session's window is typed into"
    first = app.center
    _right_click_empty_disk_(app)
    app.handle(pygame.event.Event(pygame.TEXTINPUT, text="first question"), 0)
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"), 0)
    app.drain(0)
    app._sync_text_input_()
    window = app.center
    assert app.box is None and window is not first and window.session.turns, "a new session, in place of the first"
    assert typing.on, "still on: the old window closing and the box closing no longer turn it off"
    app.draw(0)
    _click_title_(window)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="Named"))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.input.center))
    app._sync_text_input_()
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="next"))
    assert typing.on and window.text == "next" and window.session.title().endswith("Named")
    app._close_(window)
    app._sync_text_input_()
    assert not typing.on, "nothing typed into: off"



def test_a_title_is_selected_by_drag_shift_and_double_click_and_the_selection_replaced(app, monkeypatch):
    from hyp import app as app_mod

    copied = []
    monkeypatch.setattr(app_mod, "_copy_", lambda text: copied.append(text) or True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_claude()
    window = app.center
    window.session.rename("alpha beta gamma")
    app.draw(0)
    _click_title_(window)
    app.draw(0)
    e, font = window.naming, window._font_
    x0, lead, first = e._laid_
    at = lambda i: (x0 + font.size(lead + e.text[first:i])[0], window.title_rect.centery)
    # a drag over "beta", then typed over
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=at(6)))
    window.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=at(10), rel=(1, 0), buttons=(1, 0, 0)))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=at(10)))
    assert e.selection() == (6, 10)
    app.draw(0)  # the selection drawn, highlighted
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="BETA"))
    assert e.text == "alpha BETA gamma" and e.selection() is None
    # Shift+Left twice, then Backspace
    for _ in range(2):
        _key_(window, pygame.K_LEFT, pygame.KMOD_SHIFT)
    assert e.selection() == (8, 10)
    _key_(window, pygame.K_BACKSPACE)
    assert e.text == "alpha BE gamma" and e.cursor == 8
    # a double click selects a word; Cmd+X cuts it
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=at(12)))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=at(12)))
    assert e.selection() == (9, 14)
    _key_(window, pygame.K_x, pygame.KMOD_META)
    assert copied == ["gamma"] and e.text == "alpha BE "
    # Cmd+A, Cmd+C; Left collapses to the start
    _key_(window, pygame.K_a, pygame.KMOD_META)
    _key_(window, pygame.K_c, pygame.KMOD_META)
    assert copied[-1] == "alpha BE " and e.selection() == (0, 9)
    _key_(window, pygame.K_LEFT)
    assert e.selection() is None and e.cursor == 0


def test_a_first_click_on_the_title_places_the_cursor_where_it_lands(app):
    app.open_claude()
    window = app.center
    window.session.rename("alpha beta")
    app.draw(0)
    font = window._font_
    x = window.title_rect.x + font.size("Claude · alph")[0]
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(x, window.title_rect.centery)))
    assert window.naming.cursor == 4


def test_a_layouts_text_reads_back_as_written():
    from hyp.docview import MarkdownDoc

    pygame.font.init()
    doc = MarkdownDoc("# Title\n\nA paragraph long enough to wrap onto a second line here.\n\n```\nx = 1\ny = 2\n```\n")
    laid = doc.layout(220, 1.0)
    runs = laid.runs
    assert len({r.y for r in runs if r.block == 1}) > 1, "the paragraph wrapped"
    text = laid.text_between((0, 0), (len(runs) - 1, len(runs[-1].text)))
    assert text == "Title\n\nA paragraph long enough to wrap onto a second line here.\n\nx = 1\ny = 2"


def test_the_transcript_is_selected_by_a_drag_copied_and_pasted_into_the_box(app, monkeypatch):
    from hyp import app as app_mod

    clipboard = []
    monkeypatch.setattr(app_mod, "_copy_", lambda text: clipboard.append(text) or True)
    monkeypatch.setattr(pygame.scrap, "get_text", lambda: clipboard[-1] if clipboard else "")
    app.open_claude()
    window = app.center
    _type_(window, "Hello")
    app.drain(0)
    app.draw(0)
    laid = window._laid_
    k = next(i for i, r in enumerate(laid.runs) if "Reply one." in r.text)
    r = laid.runs[k]
    ox, oy = window.body.x - window.sx, window.body.y - window.sy
    at = lambda c: (ox + r.x + r.font.size(r.text[:c])[0], oy + r.y + r.h // 2)
    start = r.text.index("Reply one.")
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=at(start)))
    window.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=at(start + len("Reply one")), rel=(1, 0), buttons=(1, 0, 0)))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=at(start + len("Reply one"))))
    assert window.selection_text() == "Reply one" and window.press is None, "selected, not panned"
    app.draw(0)  # highlighted
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, mod=pygame.KMOD_META, unicode=""))
    assert clipboard == ["Reply one"]
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_META)
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v, mod=pygame.KMOD_META, unicode=""))
    assert window.text == "Reply one", "fed back to Claude, in the box"


def test_a_drag_off_the_text_still_pans(app):
    i = next(k for k, p in enumerate(app.snap.points) if p["path"].endswith("a.md"))
    app.place(i, "center")
    window = app.center
    window.set_doc(docview.MarkdownDoc("# Alpha\n"))
    app.draw(0)
    empty = (window.body.right - 20, window.body.bottom - 20)
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=empty))
    assert window.press is not None and not window.selecting


def test_a_session_is_embedded_by_its_title_and_what_was_said_not_its_file_or_markup(saved):
    tree, url = saved
    d = {d.key: d for d in docs.documents(str(tree), progress=lambda *a: None)}[url]
    assert d.embed.split("\n", 1)[0] == "What is alpha?", "its title first: what each piece is embedded under"
    assert "claude-session-" not in d.embed and "## You" not in d.embed and "# Claude ·" not in d.embed
    assert "What is alpha?" in d.embed and "An answer." in d.embed


def test_said_leaves_out_the_markup_and_keeps_what_was_said():
    body = "# Claude · Topic\n\n## You\n\nA question?\n\n## Claude\n\nAn answer, with a ## heading of its own.\n\n## Claude's view\n"
    assert claude.said(body) == "A question?\n\nAn answer, with a ## heading of its own.\n\n## Claude's view"


def test_reembed_embeds_all_afresh_and_drops_what_the_source_no_longer_gives(tmp_path):
    from hyp.engine import Engine
    from hyp.store import Store

    from test_hyp_engine import EchoSummarizer

    texts = {f"k{i}": f"text {i}" for i in range(6)}
    source = lambda progress: [docs.Doc(key=k, text=t, embed=t, summary=k) for k, t in texts.items()]
    calls = []
    rng = np.random.default_rng(0)

    def embed(ts):
        calls.append(list(ts))
        return rng.normal(size=(len(ts), 8)).astype(np.float32)

    embedder = type("Random", (), {"name": "fake:rand", "prepare": lambda self, progress=print: None,
                                   "__call__": lambda self, ts: embed(ts)})()
    engine = Engine(Store(tmp_path), embedder, EchoSummarizer(), source=source)
    engine.start(lambda *a, **k: None)
    engine.store.add(["Yep"], ["Yep"], np.ones((1, 8), dtype=np.float32))  # a typed text: no file, no key
    engine.store.notes = {"n": {"line": "kept"}}
    before = engine.store.embeddings.copy()
    calls.clear()
    engine.reembed(lambda *a, **k: None)
    s = engine.store
    assert [p["key"] for p in s.points] == list(texts) and len(s.embeddings) == 6, "the typed text is gone"
    assert sorted(t for c in calls for t in c) == sorted(texts.values()), "every entry embedded again, not only the changed"
    assert not np.allclose(s.embeddings, before[:6]) and len(s.layout) == 6 and s.notes == {"n": {"line": "kept"}}
    again = Store(tmp_path)
    again.load()
    assert len(again.points) == 6, "saved"


def test_there_is_no_claude_button_a_right_click_opens_a_session(app):
    assert not any(getattr(c, "label", "") == "Claude" for c in app.controls)
    assert app.reembed_button.label == "re-embed"


def test_a_few_points_are_placed_by_their_distances_not_on_a_ring():
    from hyp import layout

    near = np.array([1.0, 0.1, 0.0])
    e = np.stack([near, near + [0.0, 0.15, 0.0], [0.0, 0.2, 1.0]])  # two close, one apart
    w = layout.compute(e)
    d = lambda a, b: abs(w[a] - w[b])
    assert d(0, 1) < 0.5 * min(d(0, 2), d(1, 2)), "the two that mean alike, together"
    assert max(abs(w)) == pytest.approx(layout.FEW_RADIUS)
    again = layout.compute(e, previous=w * np.exp(1j))
    assert np.allclose(again, w * np.exp(1j)), "relaid, it keeps its turn"


def _press_(window, pos, kind=pygame.MOUSEBUTTONDOWN):
    window.handle(pygame.event.Event(kind, button=1, pos=pos) if kind != pygame.MOUSEMOTION
                  else pygame.event.Event(kind, pos=pos, rel=(1, 0), buttons=(1, 0, 0)))


def test_the_box_selects_by_the_mouse_and_undoes(app, monkeypatch):
    from hyp import app as app_mod

    clipboard = []
    monkeypatch.setattr(app_mod, "_copy_", lambda text: clipboard.append(text) or True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_claude()
    window = app.center
    for word in ("alpha ", "beta ", "gamma"):
        window.handle(pygame.event.Event(pygame.TEXTINPUT, text=word))
    app.draw(0)
    e = window.edit
    x, y, lh, first, font = e._laid_
    at = lambda i: (x + font.size(e.text[:i])[0], y + lh // 2)
    _press_(window, at(6))
    _press_(window, at(10), pygame.MOUSEMOTION)
    _press_(window, at(10), pygame.MOUSEBUTTONUP)
    assert e.selection() == (6, 10) and window.press is None, "selected in the box, not the transcript panned"
    _key_(window, pygame.K_c, pygame.KMOD_META)
    assert clipboard == ["beta"], "the box's selection, not the transcript's"
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="BETA"))
    assert window.text == "alpha BETA gamma"
    _key_(window, pygame.K_z, pygame.KMOD_CTRL)
    assert window.text == "alpha beta gamma" and e.selection() is None, "Ctrl+Z: the replacement undone"
    _key_(window, pygame.K_z, pygame.KMOD_META)
    assert window.text == "", "Cmd+Z: the typing undone, one run at a time"
    _key_(window, pygame.K_z, pygame.KMOD_META | pygame.KMOD_SHIFT)
    _key_(window, pygame.K_y, pygame.KMOD_CTRL)
    assert window.text == "alpha BETA gamma", "redone, both"


def test_the_box_takes_lines_and_moves_between_them(app, monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_claude()
    window = app.center
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="first"))
    _key_(window, pygame.K_RETURN, pygame.KMOD_SHIFT)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="second"))
    app.draw(0)
    e = window.edit
    assert window.text == "first\nsecond" and window.session.turns == [], "Shift+Enter: a new line, nothing sent"
    _key_(window, pygame.K_UP)
    assert e.cursor == len("first"), "up a line, as far along as it will go"
    _key_(window, pygame.K_DOWN, pygame.KMOD_SHIFT | pygame.KMOD_META)
    assert e.selection() == (len("first"), len("first\nsecond")), "Cmd+Shift+Down: selected to the end"
    monkeypatch.setattr(pygame.scrap, "get_text", lambda: "pasted\r\nlines")
    _key_(window, pygame.K_v, pygame.KMOD_META)
    assert window.text == "firstpasted\nlines", "pasted over the selection, its lines kept"
    _key_(window, pygame.K_RETURN)
    app.drain(0)
    assert window.session.turns[0][0] == "firstpasted\nlines" and window.text == ""


def test_a_long_line_wraps_in_the_box_and_a_click_lands_on_its_row(app, monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_claude()
    window = app.center
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="word " * 60))
    app.draw(0)
    e = window.edit
    x, y, lh, first, font = e._laid_
    rows = e._rows_
    assert len(rows) > 2 and all(font.size(window.text[a:b].rstrip())[0] <= window.input.width for a, b in rows)
    r = first + 1
    _press_(window, (x + 1, y + lh + lh // 2))
    assert e.cursor == rows[r][0], "a click at the second row shown: its start"


def test_the_title_undoes_too(app, monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_claude()
    window = app.center
    window.session.rename("old")
    app.draw(0)
    _click_title_(window)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text=" and new"))
    _key_(window, pygame.K_z, pygame.KMOD_META)
    assert window.naming.text == "old"


def test_symbols_a_font_lacks_are_drawn_from_a_fallback_not_as_boxes():
    from hyp import fonts

    pygame.font.init()
    f = fonts.Font("helveticaneue,helvetica,arial", 15)
    text = "⊙: ∧-like intro, ∨-like elim (Díaz-Caro–Dowek)"
    runs = f._runs_(text)
    drawn_by = {ch: font for font, piece in runs for ch in piece}
    for ch in "⊙∧∨":
        assert fonts._drawn_(drawn_by[ch], ch) != fonts._drawn_(drawn_by[ch], fonts.NO_GLYPH), f"{ch} drawn, not boxed"
    assert drawn_by["D"] is f.font and drawn_by["í"] is f.font, "what the font has, it draws itself"
    img = f.render(text, True, (255, 255, 255))
    assert img.get_size()[0] == f.size(text)[0], "measured as drawn: the cursor and selections land on it"
    assert f.render("plain", True, (255, 255, 255)).get_size() == f.font.render("plain", True, (255, 255, 255)).get_size()


def test_cmd_plus_and_minus_zoom_a_sessions_transcript_not_type_into_its_box(app):
    app.open_claude()
    window = app.center
    z = window.zoom
    _key_(window, pygame.K_EQUALS, pygame.KMOD_META)
    assert window.zoom > z and window.text == ""
    _key_(window, pygame.K_MINUS, pygame.KMOD_META)
    _key_(window, pygame.K_MINUS, pygame.KMOD_META)
    assert window.zoom < z
    _key_(window, pygame.K_0, pygame.KMOD_META)
    assert window.zoom == 1.0


def _right_click_title_(app, window):
    app.draw(0)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(window.rect.centerx, window.rect.y + 12)), 0)
    return app.popup


def test_a_right_click_on_a_title_bar_moves_the_window_and_the_center_expands_it(app):
    from hyp.docs import TilePicker

    app.open_claude()
    window = app.center
    picker = _right_click_title_(app, window)
    assert isinstance(picker, TilePicker) and picker.on_claude is None
    picker.on_select("right half")
    app.popup = None
    assert app.center is None and app.tiles == {"right half": window}
    app._sync_text_input_()
    app.draw(0)
    assert window.rect.left >= app.area().centerx - 10, "drawn in the right half"
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="still typed into"))
    assert window.text == "still typed into" and app.__wants_text__()
    app.open_claude()
    assert app.tiles == {"right half": window} and app.focus is window, "the one open, where it is: not another"
    _right_click_title_(app, window).on_select("center")
    app.popup = None
    app.draw(0)
    assert app.center is window and window.rect.size == app.area().size, "expanded: the whole area"


def test_a_documents_window_moves_too_and_what_it_covers_closes(app):
    i = next(k for k, p in enumerate(app.snap.points) if p["path"].endswith("a.md"))
    app.place(i, "top left")
    doc = app.tiles["top left"]
    app.place(i, "bottom right")
    other = app.tiles["bottom right"]
    _right_click_title_(app, doc).on_select("bottom half")
    app.popup = None
    assert app.tiles == {"bottom half": doc} and other not in app.windows()


def test_a_right_click_in_a_windows_body_is_no_title_bars(app):
    app.open_claude()
    window = app.center
    app.draw(0)
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=window.body.center), 0)
    assert app.popup is None


def test_markdown_from_empty_disk_opens_in_the_middle_to_go_on_writing_and_is_saved(app, saved):
    from hyp import notes

    tree, _ = saved
    box = _right_click_empty_disk_(app, "markdown")
    assert "markdown" in box.hint
    app.handle(pygame.event.Event(pygame.TEXTINPUT, text="My notes: on QBism"), 0)
    app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"), 0)
    window = app.center
    assert app.box is None and isinstance(window, notes.NoteWindow)
    md = tree / "my-notes-on-qbism.md"
    assert md.read_text() == "My notes: on QBism\n" and app.refreshed == 1, "saved at once, and embedded"
    app._sync_text_input_()
    assert app.__wants_text__()
    app.draw(0)
    a, r = app.area(), window.rect
    assert r.center == a.center and r.width * r.height == pytest.approx(a.width * a.height / 3, rel=0.02), "a Claude session's size"
    for ev in (pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0, unicode="\r"),
               pygame.event.Event(pygame.TEXTINPUT, text="A second line.")):
        window.handle(ev)
    assert window.edit.text == "My notes: on QBism\n\nA second line.", "Enter: a new line, nothing submitted"
    app.draw(0)
    assert "unsaved" in window.sub
    window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=pygame.KMOD_META, unicode=""))
    assert md.read_text() == "My notes: on QBism\n\nA second line." and app.refreshed == 2
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text=" More."))
    app._close_(window)
    assert md.read_text().endswith("A second line. More."), "closed: what was unsaved, saved"
    app.open_note("My notes: on QBism\n")
    assert (tree / "my-notes-on-qbism-2.md").exists(), "a name taken: not written over"


def test_a_markdown_being_written_expands_from_its_title_bar(app):
    app.open_note("Draft\n")
    window = app.center
    _right_click_title_(app, window).on_select("center")
    app.popup = None
    app.draw(0)
    assert app.center is window and window.rect.size == app.area().size


def test_a_notes_name_is_its_first_line(tmp_path):
    from hyp import notes

    assert notes._stem_("# QM: TT -- Formalisation (draft)\nbody") == "qm-tt-formalisation-draft"
    assert notes._stem_("   \n") == "note"
    n = notes.Note(str(tmp_path), "")
    assert n.save().endswith("/note.md") and not n.dirty()


def test_a_markdowns_title_is_its_files_name_edited_in_place_and_kept_renames_it(app, saved, monkeypatch):
    tree, _ = saved
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_note("Draft notes\n")
    window = app.center
    old = tree / "draft-notes.md"
    assert old.exists()
    app.draw(0)
    assert window.title == "Markdown · draft-notes.md"
    _click_title_(window)
    assert window.naming.text == "draft-notes"
    _key_(window, pygame.K_a, pygame.KMOD_META)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="wrong"))
    _key_(window, pygame.K_z, pygame.KMOD_META)
    assert window.naming.text == "draft-notes", "undone"
    _key_(window, pygame.K_a, pygame.KMOD_META)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="draft-QBism"))
    refreshed = app.refreshed
    _key_(window, pygame.K_RETURN)
    assert window.naming is None and not old.exists() and (tree / "draft-QBism.md").read_text() == "Draft notes\n"
    assert app.refreshed == refreshed + 1, "filed under its new name"
    (tree / "taken.md").write_text("someone else's")
    app.draw(0)
    _click_title_(window)
    _key_(window, pygame.K_a, pygame.KMOD_META)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="taken"))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.body.center))  # a click off it keeps it
    assert (tree / "taken.md").read_text() == "someone else's" and "taken" in window.problem
    assert (tree / "draft-QBism.md").exists(), "refused: not written over, not moved"


def test_a_markdown_named_before_its_first_save_takes_the_name(tmp_path):
    from hyp import notes

    n = notes.Note(str(tmp_path), "First line\n")
    assert n.rename("chosen") and n.name() == "chosen.md"
    assert n.save().endswith("/chosen.md")


def test_a_markdowns_text_selects_undoes_and_copies_as_the_claude_box_does(app, monkeypatch):
    from hyp import app as app_mod

    clipboard = []
    monkeypatch.setattr(app_mod, "_copy_", lambda text: clipboard.append(text) or True)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_note("alpha beta gamma\n")
    window = app.center
    app.draw(0)
    e = window.edit
    x, y, lh, first, font = e._laid_
    at = lambda i: (x + font.size(e.text[:i])[0], y + lh // 2)
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=at(6)))
    window.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=at(10), rel=(1, 0), buttons=(1, 0, 0)))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=at(10)))
    _key_(window, pygame.K_c, pygame.KMOD_META)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="BETA"))
    assert clipboard == ["beta"] and e.text == "alpha BETA gamma\n"
    _key_(window, pygame.K_z, pygame.KMOD_CTRL)
    assert e.text == "alpha beta gamma\n"


def test_the_preview_splits_the_window_renders_as_written_and_its_bar_moves(app, monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.open_note("# Heading\n\nSome *text*.\n")
    window = app.center
    app.draw(0)
    assert window.preview_rect.height == 0 and window.text_rect == window.body
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.preview_button.center))
    app.draw(0)
    p, t = window.preview_rect, window.text_rect
    assert window.previewed and p.bottom < t.top and p.height == pytest.approx(t.height, abs=8), "split in half"
    doc = window._preview_[1]
    assert [r.text for r in doc.layout(p.width - 10, 1.0).runs][:1] == ["Heading"], "rendered: the heading a heading"
    window.edit.cursor = len(window.edit.text)
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="More."))
    app.draw(0)
    assert window._preview_[1] is not doc and "More." in window._preview_[0], "as it is written"
    bar = window._bar_()
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=bar.center))
    window.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(bar.centerx, window.body.y + window.body.height // 4), rel=(0, -1), buttons=(1, 0, 0)))
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=bar.center))
    app.draw(0)
    assert window.split == pytest.approx(0.25, abs=0.02) and window.preview_rect.height < window.text_rect.height
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: window.preview_rect.center)
    window.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, precise_y=-1.0))
    assert window.edit.top == 0, "the wheel over the preview scrolls the preview, not the text"
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.preview_button.center))
    assert not window.previewed


def test_the_preview_renders_mermaid(app, monkeypatch):
    from hyp import docview

    asked = []
    monkeypatch.setattr(docview.Mermaid, "state", lambda self, source: asked.append(source) or "pending")
    app.open_note("A diagram:\n\n```mermaid\ngraph TD; A-->B\n```\n")
    window = app.center
    window.toggle_preview()
    app.draw(0)
    assert "graph TD; A-->B" in asked and window.animating(), "drawn again till the diagram is in"


def _until_(cond, timeout=10):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.05)
    return cond()


PHONE = json.dumps([["And on the phone?", "Answered there."]])


def test_a_handoff_runs_the_session_with_remote_control_reads_its_url_and_the_phones_turns_come_back(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_PHONE", PHONE)
    s = claude.Session(str(tree))
    s.send("Here first")
    h = claude.Handoff(s)
    h.start()
    assert _until_(lambda: h.url is not None), h.problem
    assert h.url == "https://claude.ai/code/session_01FAKEurl" and h.problem is None
    runs = cli.runs()
    argv = next(r["argv"] for r in runs if "argv" in r and r["argv"][:2] == ["--resume", s.id] and "-p" not in r["argv"])
    assert "--remote-control" not in argv, "resumed, interactive: Remote Control asked for at its prompt, not with it"
    assert "--fork-session" not in argv, "resumed as itself: the phone is given its history"
    assert {"typed": "/remote-control hyp · Here first"} in runs, "so the phone is given the conversation, history and all"
    qr = h.qr()
    assert len(qr) == len(qr[0]) > 20 and any(any(row) for row in qr)
    h.stop()
    assert not h.alive()
    assert s.take_back(h.started) == 1, "one turn there: its note left out"
    assert s.turns[-1] == ("And on the phone?", "Answered there.") and s.model == "claude-sonnet-5-5"


def test_what_claude_code_asks_only_once_is_said_not_answered(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_FIRST_RUN", "1")
    s = claude.Session(str(tree))
    s.send("Here first")
    h = claude.Handoff(s)
    h.start()
    assert _until_(lambda: h.problem is not None)
    assert "run `claude --remote-control` once in a terminal" in h.problem and h.url is None and not h.alive()


def test_the_remote_button_saves_hands_off_shows_the_qr_and_take_back_writes_the_phones_turns_into_the_md(app, saved, monkeypatch):
    _, url = saved
    monkeypatch.setenv("FAKE_PHONE", PHONE)
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.__click_point__(i, 0.0)
    window = app.center
    _type_(window, "One more here")
    app.drain(0)
    window.session.turns.append(("typed, not yet saved", "nor this"))  # unsaved, as the button is clicked
    app.draw(0)
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.remote_button.center))
    assert "typed, not yet saved" in open(url).read(), "saved before it goes"
    assert _until_(lambda: window.handoff.url is not None)
    app.draw(0)  # the QR code over the transcript
    assert any("on me@example.com's claude.ai" in line for line in window.handoff_lines), "whose account it is on"
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text="not now"))
    _key_(window, pygame.K_RETURN)
    assert "on the phone" in window.note and window.session.turns[-1][0] != "not now"
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.remote_button.center))
    assert window.handoff is None and "1 turn from the phone" in window.note
    assert "And on the phone?" in open(url).read() and "Answered there." in open(url).read(), "the .md brought up to date"


def test_a_session_closed_goes_to_the_phone_and_reopened_comes_back(app, saved, monkeypatch):
    _, url = saved
    monkeypatch.setenv("FAKE_PHONE", PHONE)
    app.settings["sessions on remote"] = True
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.__click_point__(i, 0.0)
    window = app.center
    _type_(window, "Last word here")
    app.drain(0)
    app._close_(window)
    handoff = app.handoffs[window.session.id]
    assert "Last word here" in open(url).read(), "saved before it went"
    assert _until_(lambda: handoff.url is not None) and "on the phone" in app.status
    app.__click_point__(i, 0.0)
    again = app.center
    assert again.handoff is None and window.session.id not in app.handoffs and not handoff.alive()
    assert again.session.turns[-1] == ("And on the phone?", "Answered there.")
    assert "Answered there." in open(url).read()


def test_a_session_with_no_turn_is_closed_not_handed_off(app):
    app.settings["sessions on remote"] = True
    app.open_claude()
    app._close_(app.center)
    assert app.handoffs == {}


def test_reopened_what_was_said_elsewhere_since_its_save_comes_in_without_a_handoff(app, saved, tmp_path):
    _, url = saved
    sid = docview.front_matter(open(url).read())[0]["session"]
    home = tmp_path / "claude-config" / "projects" / "here"
    home.mkdir(parents=True)
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 5))  # after the save
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600))
    (home / f"{sid}.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"type": "user", "timestamp": old, "message": {"role": "user", "content": "What is alpha?"}},
        {"type": "user", "timestamp": at, "message": {"role": "user", "content": "From a terminal, later"}},
        {"type": "assistant", "timestamp": at, "message": {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]}},
    ]) + "\n")
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.__click_point__(i, 0.0)
    window = app.center
    assert window.session.turns == [("From a terminal, later", "Noted.")], "only what is newer than the save"
    assert "From a terminal, later" in open(url).read()



def test_a_conversation_begun_under_the_default_login_is_carried_over_to_hyps_own(tmp_path, monkeypatch):
    default, own = tmp_path / "default", tmp_path / "own"
    (default / "projects" / "here").mkdir(parents=True)
    (default / "projects" / "here" / "abc.jsonl").write_text('{"type": "user"}\n')
    monkeypatch.setattr(claude, "DEFAULT_CONFIG", str(default))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(own))
    assert claude.carry_over("abc") and (own / "projects" / "here" / "abc.jsonl").exists()
    assert not claude.carry_over("abc"), "there already: not again"
    assert "CLAUDE_CONFIG_DIR=" + str(own) in claude.login_hint()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(default))
    assert not claude.carry_over("abc") and claude.login_hint() == "run `claude auth login` in a terminal"



def test_an_api_billing_login_is_said_to_need_a_claude_ai_one(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_API_BILLING", "1")
    s = claude.Session(str(tree))
    s.send("Here first")
    h = claude.Handoff(s)
    h.start()
    assert _until_(lambda: h.problem is not None)
    assert "API usage billing" in h.problem and "claude auth login" in h.problem and not h.alive()


def test_a_handoff_that_never_connects_is_given_up_saying_what_the_screen_said(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_SILENT", "1")
    monkeypatch.setattr(claude.Handoff, "CONNECT_S", 1)
    s = claude.Session(str(tree))
    s.send("Here first")
    h = claude.Handoff(s)
    h.start()
    assert _until_(lambda: h.problem is not None)
    assert "did not connect in 1 s" in h.problem and "saying nothing of Remote Control" in h.problem and not h.alive()



def test_claude_login_logs_in_hyps_own_claude_code_with_a_claude_ai_account(cli, tmp_path, monkeypatch, capsys):
    from hyp import app as app_mod

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "")  # what main sets, put back after
    monkeypatch.delenv("HYP_CLAUDE_CONFIG_DIR", raising=False)
    notes = tmp_path / "notes"
    notes.mkdir()
    with pytest.raises(SystemExit) as done:
        app_mod.main(["--docs", "--dir", str(notes), "--claude-login", "me@example.com"])
    assert done.value.code == 0
    (login,) = [r for r in cli.runs() if r["argv"][:2] == ["auth", "login"]]
    assert login["argv"] == ["auth", "login", "--claudeai", "--email", "me@example.com"], "a claude.ai account, never the Console's"
    assert login["config"] == str(notes / ".claude") and login["cwd"] == os.path.realpath(notes)
    assert "me@example.com · Example Org · team plan" in capsys.readouterr().out, "the login said, as it now is"
    with pytest.raises(SystemExit) as done:
        app_mod.main(["--claude-login"])
    assert done.value.code == 2, "nothing to log in: no --docs, no --claude-config-dir"



def test_remote_controls_consent_where_asked_is_accepted_as_what_was_asked_for(cli, tree, monkeypatch):
    monkeypatch.setenv("FAKE_CONSENT", "1")
    s = claude.Session(str(tree))
    s.send("Here first")
    h = claude.Handoff(s)
    h.start()
    assert _until_(lambda: h.url is not None), h.problem
    assert {"consent": True} in cli.runs() and h.problem is None
    h.stop()




def test_a_conversation_handed_off_before_goes_as_a_fresh_copy_so_the_phone_has_all_of_it(app, saved, tmp_path):
    _, url = saved
    sid = docview.front_matter(open(url).read())[0]["session"]
    home = tmp_path / "claude-config" / "projects" / "here"
    home.mkdir(parents=True, exist_ok=True)
    old = home / f"{sid}.jsonl"
    old.write_text("\n".join(json.dumps(e) for e in [
        {"type": "user", "sessionId": sid, "message": {"role": "user", "content": "What is alpha?"}},
        {"type": "bridge-session", "sessionId": sid, "bridgeSessionId": "cse_01OLD"},
        {"type": "system", "subtype": "bridge_status", "sessionId": sid, "url": "https://claude.ai/code/session_01OLD"},
        {"type": "history-suppression", "sessionId": sid, "cause": "restored_owner_mismatch"},
    ]) + "\n")
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.__click_point__(i, 0.0)
    window = app.center
    app.draw(0)
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.remote_button.center))
    copy = window.session.id
    assert copy != sid and not old.exists(), "a fresh copy, the old one gone: one copy still"
    text = (home / f"{copy}.jsonl").read_text()
    assert "What is alpha?" in text and "01OLD" not in text and sid not in text, "all of it, its own, no claude.ai session"
    assert "history-suppression" not in text, "its history not kept from the phone"
    assert docview.front_matter(open(url).read())[0]["session"] == copy, "the markdown follows it"
    assert _until_(lambda: window.handoff.url is not None)
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=window.remote_button.center))


def test_a_conversation_never_handed_off_goes_as_it_is(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "projects" / "here").mkdir(parents=True)
    (tmp_path / "projects" / "here" / "abc.jsonl").write_text(json.dumps({"type": "user", "sessionId": "abc"}) + "\n")
    assert claude.fresh_copy("abc") is None and (tmp_path / "projects" / "here" / "abc.jsonl").exists()



def test_a_handoffs_qr_code_is_drawn_as_its_url_comes_not_at_the_next_event(app):
    app.open_claude()
    window = app.center
    window.handoff = NS(url=None, problem=None, qr=lambda: None)
    app.draw(0)
    assert window.animating(), "waiting: drawn again"
    window.handoff.url = "https://claude.ai/code/session_01X"  # come on its thread, between frames
    window.handoff.qr = lambda: [[1, 0], [0, 1]]
    assert window.animating(), "come, and not yet shown: drawn again, with no event to ask for it"
    app.draw(0)
    assert not window.animating(), "shown: nothing more to draw"



def test_a_conversation_run_under_another_organization_goes_as_a_copy_under_this_one(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    home = tmp_path / "projects" / "here"
    home.mkdir(parents=True)
    rows = [{"type": "user", "uuid": "u1", "sessionId": "abc", "message": {"role": "user", "content": "Begun at work"}},
            {"type": "attachment", "uuid": "a1", "parentUuid": "u1", "sessionId": "abc",
             "attachment": {"type": "credential_org", "organizationUuid": "org-before"}},
            {"type": "assistant", "uuid": "r1", "parentUuid": "a1", "sessionId": "abc", "message": {"content": [{"type": "text", "text": "ok"}]}}]
    (home / "abc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    copy = claude.fresh_copy("abc")
    out = [json.loads(line) for line in (home / f"{copy}.jsonl").read_text().splitlines()]
    assert [r["uuid"] for r in out] == ["u1", "a1", "r1"], "every record kept: none of them the claude.ai session's"
    assert out[1]["attachment"]["organizationUuid"] == "org-now" and all(r["sessionId"] == copy for r in out)
    assert claude.fresh_copy(copy) is None, "under this organization already, never handed off: as it is"



def _chain_(rows):
    """The uuids read back from the newest record, as Claude Code reads a conversation."""
    by = {r["uuid"]: r for r in rows if r.get("uuid")}
    r, out = [r for r in rows if r.get("uuid")][-1], []
    while r is not None:
        out.append(r["uuid"])
        r = by.get(r.get("parentUuid"))
    return out[::-1]


def test_a_copy_keeps_the_conversations_chain_whole_where_records_are_left_out(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    home = tmp_path / "projects" / "here"
    home.mkdir(parents=True)
    rows = [{"type": "user", "uuid": "u1", "parentUuid": None, "sessionId": "abc", "message": {"role": "user", "content": "first"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": "abc", "message": {"content": []}},
            {"type": "system", "subtype": "bridge_status", "uuid": "b1", "parentUuid": "a1", "sessionId": "abc", "url": "x"},
            {"type": "user", "uuid": "u2", "parentUuid": "b1", "sessionId": "abc", "message": {"role": "user", "content": "second"}},
            {"type": "assistant", "uuid": "a2", "parentUuid": "u2", "sessionId": "abc", "message": {"content": []}}]
    (home / "abc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    copy = claude.fresh_copy("abc")
    out = [json.loads(line) for line in (home / f"{copy}.jsonl").read_text().splitlines()]
    assert _chain_(out) == ["u1", "a1", "u2", "a2"], "read back whole: the record left out passed over, not a break"


def test_a_conversation_already_broken_is_mended_in_its_copy(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    home = tmp_path / "projects" / "here"
    home.mkdir(parents=True)
    rows = [{"type": "user", "uuid": "u1", "parentUuid": None, "sessionId": "abc", "message": {"role": "user", "content": "first"}},
            {"type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": "abc", "message": {"content": []}},
            {"type": "user", "uuid": "u2", "parentUuid": "gone", "sessionId": "abc", "message": {"role": "user", "content": "second"}},
            {"type": "assistant", "uuid": "a2", "parentUuid": "u2", "sessionId": "abc", "message": {"content": []}}]
    (home / "abc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    copy = claude.fresh_copy("abc")
    assert copy is not None, "broken: copied, to mend it"
    out = [json.loads(line) for line in (home / f"{copy}.jsonl").read_text().splitlines()]
    assert _chain_(out) == ["u1", "a1", "u2", "a2"]



def test_every_session_goes_on_remote_as_hyp_starts_but_one_open_here(app, saved, tmp_path, monkeypatch):
    tree, url = saved
    monkeypatch.setenv("FAKE_PHONE", PHONE)
    sid = docview.front_matter(open(url).read())[0]["session"]
    home = tmp_path / "claude-config" / "projects" / "here"
    home.mkdir(parents=True, exist_ok=True)
    (home / f"{sid}.jsonl").write_text(json.dumps({"type": "user", "uuid": "u1", "sessionId": sid,
                                                   "message": {"role": "user", "content": "What is alpha?"}}) + "\n")
    app.settings["sessions on remote"] = True
    app.__snapshot_arrived__()
    assert list(app.handoffs) == [sid] and "on remote: 1 Claude session" in app.status
    handoff = app.handoffs[sid]
    assert _until_(lambda: handoff.url is not None)
    app.__snapshot_arrived__()
    assert list(app.handoffs) == [sid], "once, as hyp starts: not at every snapshot"
    i = next(k for k, p in enumerate(app.snap.points) if p["kind"] == claude.KIND)
    app.__click_point__(i, 0.0)
    window = app.center
    assert app.handoffs == {} and not handoff.alive(), "opened here: taken back"
    assert window.session.turns[-1] == ("And on the phone?", "Answered there.")
    app._close_(window)
    assert list(app.handoffs) == [window.session.id], "closed: on remote again"
    for h in app.handoffs.values():
        h.stop()


def test_sessions_stay_here_with_the_setting_off(app):
    app.settings["sessions on remote"] = False
    app.__snapshot_arrived__()
    assert app.handoffs == {}


# -- a session's turns, as points of their own ---------------------------------------------------------------------


def test_a_transcript_is_parted_into_its_turns():
    body = "\n".join(["# Claude · t", "", "## You", "", "one?", "", "## Claude", "", "first", "", "## Claude", "", "more",
                      "", "## You", "", "```", "## You", "```", "", "## Claude", "", "second", ""])
    assert claude.turns(body) == [("one?", "first\n\nmore"), ("```\n## You\n```", "second")]
    assert claude.turns("# Claude · t\n\nnothing said") == []


@pytest.fixture
def turns(cli, tree, tmp_path, monkeypatch):
    """hyp --docs on a real engine, over a saved session of three turns: Claude's naming of their questions, faked and recorded."""
    s = claude.Session(str(tree))
    for q in ("What is alpha?", "How does beta relate to it?", "And gamma, finally?"):
        monkeypatch.setenv("FAKE_REPLY", " ".join([f"About {q.split()[-1]}"] * 400))  # long: the transcript scrolls
        s.send(q)
    url = s.save()
    asked = []

    def ask(prompt, timeout=180, command=None):
        asked.append(prompt)
        n = int(re.search(r"exactly (\d+) lines", prompt).group(1))
        return "\n".join(f"{k + 1}. Topic {len(asked)}.{k + 1}" for k in range(n))

    monkeypatch.setattr(claude, "ask", ask)
    data = tmp_path / "data"
    args = NS(dir=str(tree), claude_dir="", data=str(data))
    embedder = HashEmbedder()
    engine = Engine(Store(str(data)), embedder, EchoSummarizer(), source=docs.HypDocs.__source__(args))
    engine.start(quiet)
    a = docs.HypDocs(engine=engine, worker=Worker(), args=args)
    a.run_async = lambda fn, then: then(fn())
    a.settings["sessions on remote"] = False
    a.on_snapshot(engine.snapshot(), None, 0.0)
    session = next(n for n, p in enumerate(a.snap.points) if p.get("kind") == claude.KIND)
    return NS(app=a, url=url, asked=asked, embedder=embedder, data=data, session=session, n=len(a.snap.points))


def _shards_(app):
    return [(k, p) for k, p in enumerate(app.snap.points) if p.get("kind") == docs.SHARD]


def test_cmd_right_click_on_a_session_shows_its_turns_about_it_named_by_claude(turns, monkeypatch):
    a = turns.app
    whole = a.snap.positions.copy()
    a.hover = turns.session
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_META)
    cx, cy, _ = a.frame()
    a.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(int(cx), int(cy))), 0.0)
    a.drain(0.0)

    shards = _shards_(a)
    assert [p["turn"] for _, p in shards] == [0, 1, 2] and a.sharded() == {turns.url}
    assert [a.snap.summaries[k] for k, _ in shards] == [f"Claude · What is alpha? · Topic 1.{n}" for n in (1, 2, 3)]
    assert len(turns.asked) == 1 and all(q in turns.asked[0] for q in ("What is alpha?", "beta relate", "gamma, finally"))
    assert sorted(json.loads((turns.data / "words.json").read_text()).values()) == ["Topic 1.1", "Topic 1.2", "Topic 1.3"]
    assert np.array_equal(a.snap.positions[: turns.n], whole), "the rest of the disk stays where it was"
    k = shards[0][0]
    label = a.__point_label__(k)
    assert label == "Claude · What is al… · Topic 1.1" and len(label) == 32, "the session's label cut short, the turn's words kept"
    assert "What is alpha?" in a.__point_tooltip__(k)

    a.__context_empty__((5, 5))
    a.drain(0.0)
    assert not _shards_(a) and len(a.snap.points) == turns.n and a.popup is None, "folded back, rather than the menu"
    seen = len(turns.embedder.seen)
    a.__cmd_context_point__(turns.session, (0, 0))
    a.drain(0.0)
    assert len(_shards_(a)) == 3 and len(turns.asked) == 1 and len(turns.embedder.seen) == seen, "named and embedded once"
    a.__cmd_context_point__(_shards_(a)[1][0], (0, 0))
    a.drain(0.0)
    assert not _shards_(a), "Cmd+right click on a turn folds its session's back too"


def test_a_click_on_a_turn_opens_its_session_at_that_turn(turns):
    a = turns.app
    a.__cmd_context_point__(turns.session, (0, 0))
    a.drain(0.0)
    k = _shards_(a)[1][0]
    a.__click_point__(k, 0.0)
    window = a._claude_window_()
    a.draw(0.0)
    run = next(r for r in window._laid_.runs if r.block == window.pinned)
    assert run.text == "You" and window.sy > 0 and run.y - window.sy == 12, "the second turn's You, at the top"
    window.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1, precise_x=0.0, precise_y=-1.0))
    assert window.pinned is None, "scrolled: the reader's again"


def test_a_session_filed_with_its_turns_shown_shows_the_new_one_too(turns, monkeypatch):
    a = turns.app
    a.__cmd_context_point__(turns.session, (0, 0))
    a.drain(0.0)
    a.__click_point__(turns.session, 0.0)
    monkeypatch.setenv("FAKE_REPLY", "Delta it is.")
    _type_(a._claude_window_(), "And delta?")
    a.drain(0.0)
    assert [p["words"] for _, p in _shards_(a)] == ["Topic 1.1", "Topic 1.2", "Topic 1.3", "Topic 2.1"]
    assert "And delta?" in turns.asked[1] and "What is alpha?" not in turns.asked[1], "only the new question asked"


def test_turns_claude_cannot_name_are_named_by_their_first_words(turns, monkeypatch):
    def down(prompt, timeout=180, command=None):
        raise claude.Failed("Claude Code is not logged in")

    monkeypatch.setattr(claude, "ask", down)
    a = turns.app
    a.__cmd_context_point__(turns.session, (0, 0))
    a.drain(0.0)
    assert [p["words"] for _, p in _shards_(a)] == ["What is alpha?", "How does beta relate to …", "And gamma, finally?"]
    assert "Claude could not name them: Claude Code is not logged in" in a.status
    assert not (turns.data / "words.json").exists()


def _cmd_right_click_(app, monkeypatch, pos=None):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_META)
    cx, cy, _ = app.frame()
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=pos or (int(cx), int(cy))), 0.0)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.drain(0.0)


def test_cmd_right_click_on_empty_disk_shatters_every_session_and_again_makes_them_whole(turns, monkeypatch):
    monkeypatch.setattr(docs, "WORDS_BATCH", 2)
    a = turns.app
    a.hover = None
    _cmd_right_click_(a, monkeypatch)
    shards = _shards_(a)
    assert [p["turn"] for _, p in shards] == [0, 1, 2] and all(p["shattered"] for _, p in shards) and a.shattered()
    assert not any(p.get("kind") == claude.KIND for p in a.snap.points), "the session gone: its turns in its place"
    assert shards[1][1]["embed"].startswith("How does beta relate to it?"), "embedded alone: not by its session's title"
    assert [p["words"] for _, p in shards] == ["Topic 1.1", "Topic 1.2", "Topic 2.1"], "named two at a time"
    assert not a.sharded(), "none ringed about a session"

    a.__context_empty__((5, 5))
    assert a.popup is not None, "a plain right click on empty disk: the menu, as ever"
    a.popup = None
    a.__cmd_context_point__(shards[0][0], (100, 100))
    assert isinstance(a.popup, docs.TilePicker), "Cmd+right click on a turn shattered: placed, as a right click"
    a.popup = None
    a.hover = None
    _cmd_right_click_(a, monkeypatch)
    assert not _shards_(a) and len(a.snap.points) == turns.n and not a.shattered(), "whole again"


def test_a_session_filed_while_shattered_is_shattered_again_its_new_turn_among_them(turns, monkeypatch):
    a = turns.app
    a.hover = None
    _cmd_right_click_(a, monkeypatch)
    a.__click_point__(_shards_(a)[0][0], 0.0)
    monkeypatch.setenv("FAKE_REPLY", "Delta it is.")
    _type_(a._claude_window_(), "And delta?")
    a.drain(0.0)
    assert [p["words"] for _, p in _shards_(a)] == ["Topic 1.1", "Topic 1.2", "Topic 1.3", "Topic 2.1"]
    assert a.shattered() and not any(p.get("kind") == claude.KIND for p in a.snap.points)
