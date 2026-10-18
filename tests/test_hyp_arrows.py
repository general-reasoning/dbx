"""Arrows between points: their embeddings' difference, decoded against the corpus's phrases."""

import re

import numpy as np

from hyp import arrows
from hyp.engine import Engine
from hyp.store import Store

from test_hyp_engine import EchoSummarizer, quiet


class BagEmbedder:
    """Each word a dimension of its own: an arrow's right reading is known."""

    name = "fake:bag"
    WORDS = ("model", "probe", "lr", "0.1", "0.001", "fold", "3", "5", "tablets", "64")

    def __init__(self):
        self.calls = 0

    def prepare(self, progress=print):
        pass

    def __call__(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), len(self.WORDS)), dtype=np.float32)
        for row, t in zip(out, texts):
            for w in re.findall(r"[\w.]+", t.lower()):
                if w in self.WORDS:
                    row[self.WORDS.index(w)] += 1.0
        return out


def test_a_texts_phrases_are_its_lines_unquoted_and_one_in_every_text_is_no_phrase():
    texts = ["model probe\n{'lr': 0.1,\n 'fold': 3}", "model probe\n{'lr': 0.001,\n 'fold': 3}", "model probe\n{'lr': 0.1}"]
    got = arrows.phrases(texts)
    assert "model probe" not in got, "in every text: no direction"
    assert got[:2] == ["fold: 3", "lr: 0.1"], "the ones in the most texts first"
    assert "lr: 0.001" in got
    assert all(not set("'{}") & set(p) for p in got)


def test_a_long_line_is_cut_at_its_sentences():
    long = "First sentence about the probe. " + "Second about the folds; " + "x" * 130
    got = arrows.phrases([long, "other"])
    assert "First sentence about the probe." in got and "Second about the folds;".rstrip(";") in got
    assert all(len(p) <= arrows.PHRASE_CHARS for p in got)


def test_an_arrow_reads_as_what_its_end_has_and_its_start_does_not():
    emb = BagEmbedder()
    phrases = ["lr 0.001", "lr 0.1", "fold 5", "fold 3", "model"]
    here, there = emb(["model lr 0.1 fold 3"])[0], emb(["model lr 0.001 fold 3"])[0]
    toward, away = arrows.decode(here, there, emb(phrases), k=1)
    assert phrases[toward[0][0]] == "lr 0.001" and toward[0][1] > 0
    assert phrases[away[0][0]] == "lr 0.1" and away[0][1] < 0
    assert arrows.decode(here, here, emb(phrases)) == ([], []), "no arrow between a point and itself"


def test_the_engine_decodes_a_points_arrows_and_embeds_the_phrases_once(tmp_path):
    emb = BagEmbedder()
    texts = ["model probe\nlr: 0.1\nfold: 3", "model probe\nlr: 0.001\nfold: 3", "model probe\nlr: 0.1\nfold: 5\ntablets: 64"]
    store = Store(tmp_path)
    store.reset(emb.name)
    store.add(texts, texts, emb(texts))
    store.save()
    engine = Engine(store, emb, EchoSummarizer())

    (to_1, to_2) = engine.arrows(0, [1, 2], quiet)
    assert to_1[0][0][0] == "lr: 0.001" and to_1[1][0][0] == "lr: 0.1"
    assert {p for p, _ in to_2[0]} & {"fold: 5", "tablets: 64"}
    calls = emb.calls
    engine.arrows(0, [1, 2], quiet)
    assert emb.calls == calls, "the phrases are embedded once"

    kept = Store(tmp_path)
    assert kept.phrases == store.phrases and kept.phrase_embeddings.shape == store.phrase_embeddings.shape
    kept.embedder = "another"
    kept.save_phrases()
    store.save()  # the corpus says fake:bag again; its phrases, another
    assert Store(tmp_path).phrases == [], "phrases embedded with another model are dropped"


def test_shift_over_a_point_decodes_its_arrows_and_draws_them(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace as NS

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from hyp.app import Hyp, Worker

    emb = BagEmbedder()
    texts = ["model probe\nlr: 0.1", "model probe\nlr: 0.001", "model probe\nfold: 5"]
    store = Store(tmp_path)
    store.reset(emb.name)
    store.add(texts, texts, emb(texts))
    app = Hyp(engine=Engine(store, emb, EchoSummarizer()), worker=Worker())
    app.run_async = lambda fn, then: then(fn())
    app.snap = NS(positions=np.array([0j, 0.4 + 0.2j, -0.3 - 0.4j]), summaries=texts, texts=texts,
                  neighbors=np.array([[1, 2], [0, 2], [0, 1]]), points=[{} for _ in texts])
    app.draw(0)
    cx, cy, _ = app.frame()
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (cx, cy))
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    app.draw(0)
    assert app.hover == 0 and not app.arrows, "no Shift: nothing decoded"
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_SHIFT)
    app.draw(0)
    ((key, got),) = app.arrows.items()
    assert key == (id(app.snap), 0) and got[0][0][0][0] == "lr: 0.001"
    app.draw(0)  # drawn from what was decoded: nothing asked again
    assert len(app.arrows) == 1
