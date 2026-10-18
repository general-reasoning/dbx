"""hyp --markdowns' Claude sessions: run through a fake Claude Code CLI, saved in the directory scanned, found again, resumed."""

import json
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

# A stand-in for `claude -p ... --output-format stream-json`: each run's argv, cwd and stdin logged; its reply
# FAKE_REPLY (words streamed), or FAKE_FAIL's message as a failed result, or FAKE_STDERR on stderr with no result.
FAKE = r'''#!{python}
import json, os, sys
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
    for v in ("FAKE_REPLY", "FAKE_FAIL", "FAKE_STDERR", "FAKE_TOOL"):
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

    def refresh(progress):
        a.refreshed += 1
        return True

    a.engine = NS(refresh=refresh, snapshot=lambda: a.snap)
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
            app_mod.main(["--markdowns", "--claude-dir", rel])
    with pytest.raises(SystemExit):
        app_mod.main(["--claude-dir", "sessions"])  # with --markdowns only


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
    assert line == "claude-opus-5-5 · 6.0k in · 600 out · ≈$0.50 at API rates · plan 5 h 71%, 7 d 65%"
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
