"""Arrows between points: the direction of their embeddings' difference, read where else it runs in the corpus, then named."""

import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hyp import arrows
from hyp.engine import Engine
from hyp.store import Store

from test_hyp_engine import EchoSummarizer, quiet

PIPELINES = ("probe", "clip", "fold", "collator")
SPLITS = ("TEST", "TRAIN")


class GridEmbedder:
    """"<pipeline> <split>" as a pipeline's direction plus a split's, and a little noise: every TEST -> TRAIN is one move."""

    name = "fake:grid"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def prepare(self, progress=print):
        pass

    def __call__(self, texts):
        out = self.rng.normal(scale=0.02, size=(len(texts), len(PIPELINES) + len(SPLITS)))
        for row, t in zip(out, texts):
            p, s = t.split(" ")[:2]
            row[PIPELINES.index(p)] += 1.0
            row[len(PIPELINES) + SPLITS.index(s)] += 0.6
        return out.astype(np.float32)


def grid():
    texts = [f"{p} {s}" for p in PIPELINES for s in SPLITS]
    return texts, arrows.unit_rows(GridEmbedder()(texts))


def test_an_arrow_is_read_by_the_same_move_elsewhere_and_by_its_axis():
    texts, units = grid()
    i, j = texts.index("probe TEST"), texts.index("probe TRAIN")
    pairs, moves = arrows.neighbor_moves(units, [[k for k in range(len(texts)) if k != a] for a in range(len(texts))])
    ev = arrows.evidence(units, i, j, pairs, moves, texts)
    analogies = {(a, b) for a, b, _ in ev.analogies}
    assert {("clip TEST", "clip TRAIN"), ("fold TEST", "fold TRAIN"), ("collator TEST", "collator TRAIN")} <= analogies
    assert [c > 0.9 for a, b, c in ev.analogies] == [True, True, True, False, False], "one pipeline's move, then a lesser one"
    assert ("probe TRAIN", "probe TEST") not in analogies and ("probe TEST", "probe TRAIN") not in analogies, "not itself"
    assert all(x.endswith("TEST") for x in ev.low[:4]) and all(x.endswith("TRAIN") for x in ev.high[:4])
    same, cos = ev.same()
    assert same >= 3 and cos > 0.9
    assert arrows.evidence(units, i, i, pairs, moves, texts) is None, "no direction between a point and itself"


def test_claude_is_asked_to_name_the_direction_from_the_evidence_and_answers_a_line_an_arrow():
    ev = arrows.Evidence(analogies=[("clip TEST", "clip TRAIN", 0.97)], low=["probe TEST"], high=["fold TRAIN"])
    p = arrows.prompt([("probe TEST", "probe TRAIN", ev), ("a", "b", arrows.Evidence(low=["x"], high=["y"]))])
    assert '1. from "probe TEST" to "probe TRAIN"' in p and '"clip TEST" -> "clip TRAIN" (cos 0.97)' in p
    assert 'its axis runs from: "probe TEST"' in p and 'toward: "fold TRAIN"' in p
    assert "the same move elsewhere: none" in p and "Do not describe the two endpoints" in p and "JSON array" in p
    assert arrows.parse('Here: ["one", "  two   words "]', 2) == ["one", "two words"]
    assert arrows.parse('["one"]', 2) is None and arrows.parse("no json", 1) is None


@pytest.fixture
def corpus(tmp_path):
    texts = [f"{p} {s}" for p in PIPELINES for s in SPLITS] + ["probe TEST again"]
    emb = GridEmbedder()
    store = Store(tmp_path)
    store.reset(emb.name)
    store.add(texts, texts, emb(texts))
    store.embeddings[-1] = store.embeddings[0]  # "probe TEST" run again: one embedding
    store.save()
    return tmp_path, texts, emb


def test_an_arrow_is_named_once_and_kept_and_one_between_equal_embeddings_not_asked(corpus):
    tmp_path, texts, emb = corpus
    engine = Engine(Store(tmp_path), emb, EchoSummarizer())
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return '["Test split toward train split"]'

    i, j, twin = texts.index("probe TEST"), texts.index("probe TRAIN"), len(texts) - 1
    notes = engine.arrow_notes(i, [j, twin], ask, quiet)
    assert notes[0] == {"line": "Test split toward train split", "same": notes[0]["same"], "cos": notes[0]["cos"]}
    assert notes[0]["same"] >= 3 and notes[0]["cos"] > 0.9
    assert notes[1]["line"] == "the same embedding" and len(asked) == 1
    assert '"clip TEST" -> "clip TRAIN"' in asked[0], "asked from the evidence"
    assert engine.arrow_notes(i, [j, twin], ask, quiet)[0]["line"] == "Test split toward train split" and len(asked) == 1
    again = Engine(Store(tmp_path), emb, EchoSummarizer())
    assert again.arrow_notes(i, [j], lambda p: asked.append(p), quiet)[0]["line"] == "Test split toward train split"
    assert len(asked) == 1, "and kept on disk"


def test_a_corpus_grown_reads_its_arrows_again(corpus):
    tmp_path, texts, emb = corpus
    store = Store(tmp_path)
    engine = Engine(store, emb, EchoSummarizer())
    asked = []
    ask = lambda p: asked.append(p) or '["x"]'
    engine.arrow_notes(0, [1], ask, quiet)
    store.add(["fold TRAIN more"], ["fold TRAIN more"], emb(["fold TRAIN more"]))
    engine.arrow_notes(0, [1], ask, quiet)
    assert len(asked) == 2, "the evidence is the whole corpus's"


def _app_(tmp_path, monkeypatch, texts):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from hyp.app import Hyp, Worker

    emb = GridEmbedder()
    store = Store(tmp_path)
    store.reset(emb.name)
    store.add(texts, texts, emb(texts))
    app = Hyp(engine=Engine(store, emb, EchoSummarizer()), worker=Worker())
    app.snap = NS(positions=np.array([0j, 0.4 + 0.2j, -0.3 - 0.4j][: len(texts)]), summaries=texts, texts=texts,
                  neighbors=np.array([[k for k in range(len(texts)) if k != a][:2] for a in range(len(texts))]),
                  points=[{} for _ in texts])
    app.draw(0)
    cx, cy, _ = app.frame()
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (cx, cy))
    return app


def test_shift_over_a_point_has_its_arrows_named_and_draws_them(tmp_path, monkeypatch):
    import pygame

    from hyp import claude

    asked = []
    monkeypatch.setattr(claude, "ask", lambda prompt: asked.append(prompt) or '["toward training", "toward folds"]')
    app = _app_(tmp_path, monkeypatch, ["probe TEST", "probe TRAIN", "fold TEST"])
    app.run_async = lambda fn, then: then(fn())
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.draw(0)
    assert app.hover == 0 and not app.arrows, "no Shift: nothing read"
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_SHIFT)
    app.draw(0)
    ((key, got),) = app.arrows.items()
    assert key == (id(app.snap), 0) and [n["line"] for n in got] == ["toward training", "toward folds"]
    assert len(asked) == 1, "both arrows in one question"
    app.draw(0)  # drawn from what was named: nothing asked again
    assert len(asked) == 1


def test_a_reading_that_failed_says_why_and_is_not_asked_again_at_once(tmp_path, monkeypatch):
    import pygame

    from hyp import claude

    asked = []

    def fail(prompt):
        asked.append(prompt)
        raise claude.Failed("Claude Code is not logged in")

    monkeypatch.setattr(claude, "ask", fail)
    app = _app_(tmp_path, monkeypatch, ["probe TEST", "probe TRAIN"])

    def run_async(fn, then):
        try:
            then(fn())
        except Exception:
            pass

    app.run_async = run_async
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_SHIFT)
    app.draw(0)
    app.drain(0)
    ((_, got),) = app.arrows.items()
    assert isinstance(got, tuple) and "not logged in" in got[0]
    app.draw(0)
    assert len(asked) == 1, "not every frame"


def test_what_a_naming_rests_on_is_said_beneath_it():
    from hyp.app import _arrow_evidence_

    assert _arrow_evidence_({"line": "x", "same": 5, "cos": 0.97}) == "the same move in 5 other pairs · cos up to 0.97"
    assert _arrow_evidence_({"line": "x", "same": 1, "cos": 0.71}) == "the same move in 1 other pair · cos up to 0.71"
    assert _arrow_evidence_({"line": "x", "same": 0, "cos": 0.55}) == "no other pair moves this way (nearest cos 0.55)"
    assert _arrow_evidence_({"line": "the same embedding", "same": 0, "cos": 1.0}) == ""
