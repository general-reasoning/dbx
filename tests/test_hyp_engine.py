"""The engine, store and layout, with a fake embedder: no Ollama needed."""

import hashlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hyp import layout, poincare, seeds
from hyp.engine import Engine
from hyp.store import Store


def knn_agreement(neighbors, topics):
    """The fraction of each point's neighbors sharing its topic, averaged."""
    topics = np.asarray(topics)
    return float((topics[neighbors] == topics[:, None]).mean())


def layout_neighbors(positions, k):
    d = poincare.distance(positions[:, None], positions[None, :])
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


class TopicEmbedder:
    """Topic one-hot plus noise: an embedder whose right answer is known."""

    name = "fake:topics"

    def __init__(self):
        self.topics = {t: i for i, t in enumerate(seeds.SEEDS)}
        self.lookup = {text: topic for topic, items in seeds.SEEDS.items() for text in items}
        self.rng = np.random.default_rng(0)

    def prepare(self, progress=print):
        pass

    def __call__(self, texts):
        out = self.rng.normal(scale=0.2, size=(len(texts), len(self.topics) + 4))
        for row, t in zip(out, texts):
            row[self.topics.get(self.lookup.get(t), 0)] += 1.0
        return out.astype(np.float32)


class EchoSummarizer:
    def prepare(self, progress=print):
        pass

    def __call__(self, text):
        return text[:40]


def quiet(msg, done=None, total=None):
    pass


def test_start_seeds_and_lays_out_on_the_disk(tmp_path):
    engine = Engine(Store(tmp_path), TopicEmbedder(), EchoSummarizer())
    engine.start(quiet)
    snap = engine.snapshot()
    texts, topics = seeds.texts_and_topics()
    assert snap.texts == tuple(texts)
    assert np.all(np.abs(snap.positions) < 1)
    # Layout neighbors should mostly share a topic when the embedding says so.
    assert knn_agreement(layout_neighbors(snap.positions, 5), topics) > 0.6


def test_add_persists_and_keeps_the_map_in_place(tmp_path):
    engine = Engine(Store(tmp_path), TopicEmbedder(), EchoSummarizer())
    engine.start(quiet)
    before = engine.snapshot().positions
    text = "a brand new text about nothing in particular"
    i = engine.add(text, quiet)
    assert i == len(before)

    reloaded = Store(tmp_path)
    assert len(reloaded) == len(before) + 1
    assert reloaded.points[-1]["summary"] == EchoSummarizer()(text)
    assert reloaded.layout.shape == (len(before) + 1,)
    # Seeded from the previous layout and aligned to it, the old points
    # should mostly stay near where they were.
    moved = np.abs(reloaded.layout[: len(before)] - before)
    assert np.median(moved) < 0.25


def test_nearest_excludes_self():
    emb = np.eye(4) + 0.01
    nn = layout.nearest(emb, 2)
    assert nn.shape == (4, 2)
    assert not np.any(nn == np.arange(4)[:, None])


def test_store_over_an_fsspec_url():
    """A corpus kept at a url -- a datalake's .hyp -- saves, reloads and clears as one on local disk does."""
    import fsspec

    url = "memory://hyp-store-test/.hyp"
    store = Store(url)
    store.reset("fake")
    store.add(["a", "b"], ["A", "B"], np.eye(2, dtype=np.float32))
    store.save()
    store.save()  # over the files the first save wrote
    again = Store(url)
    assert again.embedder == "fake" and [p["text"] for p in again.points] == ["a", "b"]
    assert np.array_equal(again.embeddings, np.eye(2, dtype=np.float32))
    again.clear()
    assert len(Store(url)) == 0
    fsspec.filesystem("memory").rm("/hyp-store-test", recursive=True)



def test_points_the_source_no_longer_gives_are_kept_but_not_shown(tmp_path):
    """A filter narrowed, a file gone: its embedding stays, so widening again embeds nothing; what is shown is laid out by itself."""
    from hyp.engine import Engine
    from hyp.store import Store

    texts = {k: f"text {k} " * 3 for k in "abcdefgh"}
    given = list(texts)
    source = lambda progress: [NS(key=k, text=texts[k], embed=texts[k], summary=k, extra={}) for k in given]  # noqa: E731

    class Counting:
        name = "fake"
        embedded = 0

        def prepare(self, progress=print):
            pass

        def __call__(self, batch):
            Counting.embedded += len(batch)
            return np.random.default_rng(len(batch)).normal(size=(len(batch), 8)).astype(np.float32)

    engine = Engine(Store(str(tmp_path)), Counting(), EchoSummarizer(), source=source)
    engine.start(quiet)
    assert len(engine.snapshot().positions) == 8 and Counting.embedded == 8
    given = list("abcde")
    engine.reload(quiet)
    snap = engine.snapshot()
    assert sorted(snap.summaries) == list("abcde") and len(snap.positions) == 5 and snap.neighbors.shape[0] == 5
    assert len(engine.store) == 8, "kept"
    given = list(texts)
    engine.reload(quiet)
    assert len(engine.snapshot().positions) == 8 and Counting.embedded == 8, "nothing embedded again"


def test_an_engine_switches_corpora(tmp_path):
    from hyp.engine import Engine
    from hyp.store import Store

    one = lambda progress: [NS(key=f"a{i}", text=f"alpha {i}", embed=f"alpha {i}", summary=f"a{i}", extra={}) for i in range(6)]  # noqa: E731
    two = lambda progress: [NS(key=f"b{i}", text=f"beta {i}", embed=f"beta {i}", summary=f"b{i}", extra={}) for i in range(7)]  # noqa: E731
    engine = Engine(Store(str(tmp_path / "one")), TopicEmbedder(), EchoSummarizer(), source=one)
    engine.start(quiet)
    engine.switch(Store(str(tmp_path / "two")), two)
    engine.start(quiet)
    assert len(engine.snapshot().positions) == 7 and engine.snapshot().summaries[0] == "b0"
    assert len(Store(str(tmp_path / "one"))) == 6, "the first corpus, kept where it was"


def test_a_sync_interrupted_while_embedding_leaves_the_store_as_it_was(tmp_path):
    """The store changes only once everything is embedded: an interrupted job's next sync sees the same news."""
    class Stop(Exception):
        pass

    version = {"a": "one", "b": "one"}
    source = lambda progress: [NS(key=k, text=v, embed=v, summary=k, extra={}) for k, v in version.items()]  # noqa: E731
    engine = Engine(Store(tmp_path), TopicEmbedder(), EchoSummarizer(), source=source)
    engine.start(quiet)
    version.update(a="two", c="new")

    def stop_embedding(msg, done=None, total=None):
        if msg.startswith("embedding"):
            raise Stop()

    with pytest.raises(Stop):
        engine.reload(stop_embedding)
    assert [p["embed"] for p in engine.store.points] == ["one", "one"]
    told = []
    engine.reload(lambda msg, done=None, total=None: told.append((msg, done, total)))
    assert ("embedding 1 new and 1 changed entries", 2, 2) in told


class HashEmbedder:
    """A vector per text, the same every time: what it was asked to embed, recorded."""

    name = "fake:hash"

    def __init__(self, dim=16):
        self.dim, self.seen = dim, []

    def prepare(self, progress=print):
        pass

    def __call__(self, texts):
        self.seen += list(texts)
        seed = lambda t: int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)  # noqa: E731
        return np.stack([np.random.default_rng(seed(t)).normal(size=self.dim) for t in texts]).astype(np.float32)


def test_linked_points_are_laid_out_within_reach_of_each_other_their_embeddings_untouched(tmp_path):
    rng = np.random.default_rng(3)
    e = rng.normal(size=(40, 16)).astype(np.float32)
    pairs = [(0, 1), (2, 3), (4, 5), (10, 0)]  # 10 grew from 0, which grew from 1
    d = layout.distances(e, pairs)
    plain = layout.distances(e)
    assert np.isclose(d[0, 1], plain[0, 1] * (1 - layout.LINK_PULL)) and d[0, 2] == plain[0, 2]
    w = layout.compute(e, seed=0, links=pairs)
    nn = poincare.distance(w[:, None], w[None, :])
    np.fill_diagonal(nn, np.inf)
    reach = layout.LINK_REACH * float(np.median(nn.min(axis=1)))
    assert all(poincare.distance(w[j], w[i]) <= reach * 1.05 for i, j in pairs)
    few = layout.compute(e[:4], links=[(0, 1)])
    assert poincare.distance(few[0], few[1]) < poincare.distance(*layout.compute(e[:4])[:2]), "a few: scaled by the shortened distance"

    links = {"k3": "k7"}
    source = lambda progress: [NS(key=f"k{i}", text=f"t{i}", embed=f"text {i}", summary=f"k{i}",  # noqa: E731
                                  extra={"link": links[f"k{i}"]} if f"k{i}" in links else {}) for i in range(12)]
    engine = Engine(Store(tmp_path), HashEmbedder(), EchoSummarizer(), source=source)
    engine.start(quiet)
    snap = engine.snapshot()
    assert snap.points[3]["link"] == "k7"
    links["k5"] = "k7"
    assert engine.refresh(quiet), "a link made: laid out again, though nothing was embedded"
    assert engine.store.points[5]["link"] == "k7" and not engine.refresh(quiet)


def test_the_points_shown_are_laid_out_from_where_they_were_not_afresh(tmp_path, monkeypatch):
    keys, gone = [f"k{i}" for i in range(10)], set()
    source = lambda progress: [NS(key=k, text=k, embed=f"text {k}", summary=k, extra={}) for k in keys if k not in gone]  # noqa: E731
    engine = Engine(Store(tmp_path), HashEmbedder(), EchoSummarizer(), source=source)
    engine.start(quiet)
    before = dict(zip(keys, engine.snapshot().positions))
    seen, real = [], layout.compute
    monkeypatch.setattr(layout, "compute", lambda e, previous=None, seed=0, links=(): (seen.append(previous), real(e, previous=previous, seed=seed, links=links))[1])
    gone.add("k0")  # a point the source no longer gives: kept, not shown
    engine.refresh(quiet)
    (previous,) = seen
    assert np.array_equal(previous, [before[k] for k in keys[1:]]), "from where each was"
    keys.append("k10")
    engine.refresh(quiet)
    assert len(seen[-1]) == 9 and len(engine.snapshot().positions) == 10, "the new one beside its nearest, the rest from where they were"
