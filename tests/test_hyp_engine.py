"""The engine, store and layout, with a fake embedder: no Ollama needed."""

import hashlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hyp import layout, poincare, seeds
from hyp.engine import SHARD_INNER, SHARD_RADIUS, Engine
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


def test_a_points_shards_are_ringed_about_it_and_fold_back_leaving_the_map_as_it_was(tmp_path):
    texts = {f"k{i}": f"text {i}" for i in range(8)}
    source = lambda progress: [NS(key=k, text=t, embed=t, summary=k, extra={}) for k, t in texts.items()]  # noqa: E731
    embedder = HashEmbedder()
    engine = Engine(Store(tmp_path), embedder, EchoSummarizer(), source=source)
    engine.start(quiet)
    before = engine.snapshot()
    shards = [NS(key=f"k3#{n}", text=f"turn {n}", embed=f"k3\n\nturn {n}", summary=f"k3 · {n}", extra={"turn": n}) for n in range(4)]
    engine.shard("k3", shards, quiet)

    snap = engine.snapshot()
    assert len(snap.positions) == 12 and np.array_equal(snap.positions[:8], before.positions), "the map does not move"
    assert snap.summaries[8:] == ("k3 · 0", "k3 · 1", "k3 · 2", "k3 · 3")
    assert [p["origin"] for p in snap.points[8:]] == ["k3"] * 4 and [p["turn"] for p in snap.points[8:]] == [0, 1, 2, 3]
    # Ringed about their point, wherever it is: as far from it as about the origin -- distances, an isometry keeps.
    d = poincare.distance(snap.positions[3], snap.positions[8:])
    assert d.min() >= poincare.distance0(SHARD_INNER) - 1e-9
    assert d.max() == pytest.approx(poincare.distance0(SHARD_RADIUS))
    assert all(3 in row for row in snap.neighbors[8:]), "a shard's nearest: its point, among them"

    seen = len(embedder.seen)
    engine.shard("k3", shards[:3] + [NS(key="k3#9", text="new", embed="k3\n\nnew", summary="k3 · new", extra={})], quiet)
    assert embedder.seen[seen:] == ["k3\n\nnew"], "sharded again: only what is new embedded"
    assert engine.unshard("k3") and not engine.unshard()
    after = engine.snapshot()
    assert len(after.positions) == 8 and np.array_equal(after.positions, before.positions)
    with pytest.raises(KeyError):
        engine.shard("nowhere", shards, quiet)


def test_about_rings_points_between_two_radii():
    rng = np.random.default_rng(1)
    for n in (1, 2, 7):
        w = layout.about(rng.normal(size=(n, 8)), 0.45, 0.1)
        assert len(w) == n and np.all(np.abs(w) >= 0.1 - 1e-9) and np.abs(w).max() <= 0.45 + 1e-9


def test_points_shattered_lie_on_their_own_in_their_place_and_are_put_back(tmp_path, monkeypatch):
    texts = {f"k{i}": f"text {i}" for i in range(8)}
    source = lambda progress: [NS(key=k, text=t, embed=t, summary=k, extra={}) for k, t in texts.items()]  # noqa: E731
    engine = Engine(Store(tmp_path), HashEmbedder(), EchoSummarizer(), source=source)
    engine.start(quiet)
    before = engine.snapshot()
    engine.shard("k1", [NS(key="k1#0", text="t", embed="k1\n\nt", summary="k1 · t", extra={})], quiet)
    groups = {"k3": [NS(key=f"k3#{n}", text=f"turn {n}", embed=e, summary=f"k3 · {n}", extra={}) for n, e in enumerate(["text 6", "x", "y"])],
              "k5": [NS(key=f"k5#{n}", text=f"turn {n}", embed=f"z{n}", summary=f"k5 · {n}", extra={}) for n in range(2)]}
    laid, real = [], layout.compute
    monkeypatch.setattr(layout, "compute", lambda e, previous=None, seed=0: (laid.append(len(e)), real(e, previous=previous, seed=seed))[1])
    engine.shatter(groups, quiet)

    snap = engine.snapshot()
    keys = [p["key"] for p in snap.points]
    assert keys == ["k0", "k1", "k2", "k4", "k6", "k7", "k3#0", "k3#1", "k3#2", "k5#0", "k5#1"], "in their points' place; ringed shards gone"
    assert [p.get("origin") for p in snap.points[6:]] == ["k3"] * 3 + ["k5"] * 2
    _, embedded, _ = engine._rows_()
    assert np.allclose(embedded[6], HashEmbedder()(["text 6"])[0]), "its own embedding, not its point's mixed in"
    assert keys.index("k6") in snap.neighbors[6] and laid == [11], "laid out once, where what it says puts it: by k6"
    engine.shatter({k: [NS(**{**vars(e), "summary": "renamed"}) for e in g] for k, g in groups.items()}, quiet)
    assert engine.snapshot().summaries[6] == "renamed" and laid == [11], "relabeled: not laid out again"
    assert np.array_equal(engine.snapshot().positions, snap.positions)
    assert engine.unshatter() and not engine.unshatter()
    assert np.array_equal(engine.snapshot().positions, before.positions), "whole again: the disk as it was"
