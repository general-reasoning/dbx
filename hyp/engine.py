"""The engine: embed, summarize, store and lay out texts.

The UI talks to it from a worker thread; a lock serializes mutations, and
snapshot() hands the UI an immutable copy to draw.
"""

import threading
from dataclasses import dataclass

import numpy as np

from . import arrows, layout, seeds

NEIGHBORS = 5


@dataclass(frozen=True)
class Snapshot:
    texts: tuple
    summaries: tuple
    positions: np.ndarray  # complex disk positions, data coordinates
    neighbors: np.ndarray  # (n, k) nearest neighbors in embedding space
    points: tuple = ()  # each point's stored fields, for a specialization's use


class Engine:
    def __init__(self, store, embedder, summarizer, seed=0, source=None):
        """*source*, if given, replaces the seed corpus: a callable taking a
        progress callback and returning objects with ``key``, ``text``,
        ``embed``, ``summary`` and ``extra`` (more fields to store), such as
        `dbxpipelines.Entry`. Each start adds the ones whose key is not stored
        yet, and updates the ones that changed.
        """
        self.store = store
        self.source = source
        self.embedder = embedder
        self.summarizer = summarizer
        self.seed = seed
        self.lock = threading.Lock()
        # The keys the source gave at its last sync -- what is shown -- or None
        # for every point stored. A point the source no longer gives (a filter
        # narrowed it, its file went) is kept, embedding and all, but not shown.
        self.keys = None
        self._shown_ = None  # (indices, layout) of the points shown, where not all are

    def switch(self, store, source):
        """Another corpus: *store*, synced from *source*. `start` brings it up to date."""
        with self.lock:
            self.store, self.source = store, source
            self.keys, self._shown_ = None, None

    def start(self, progress=print):
        """Make the models available and bring the store up to date."""
        self.embedder.prepare(progress)
        self.summarizer.prepare(progress)
        with self.lock:
            s = self.store
            if s.embedder != self.embedder.name and len(s):
                # Vectors from different models are not comparable: re-embed
                # what is there with the current one.
                progress(f"re-embedding {len(s)} texts with {self.embedder.name}")
                points = s.points
                s.reset(self.embedder.name)
                s.add(
                    [p["text"] for p in points],
                    [p["summary"] for p in points],
                    self.embedder([p.get("embed", p["text"]) for p in points]),
                    [{k: v for k, v in p.items() if k not in ("text", "summary", "added")} for p in points],
                )
            if not len(s):
                s.reset(self.embedder.name)
            previous = s.layout
            moved = False
            if self.source is not None:
                moved = self._sync_(progress)
            elif not len(s):
                texts, _ = seeds.texts_and_topics()
                progress(f"embedding {len(texts)} seed texts")
                s.add(texts, list(texts), self.embedder(texts))
            if self._shown_indices_() is not None:
                self._lay_out_shown_(progress)
            elif s.layout is None or len(s.layout) != len(s) or moved:
                progress("computing hyperbolic UMAP layout")
                keep = previous if previous is not None and len(previous) <= len(s) else None
                s.layout = layout.compute(s.embeddings, previous=keep, seed=self.seed)
            s.save()

    def reload(self, progress=print):
        """Read the source again -- new entries embedded, changed ones re-embedded -- and lay everything out afresh."""
        with self.lock:
            s = self.store
            if self.source is not None:
                self._sync_(progress)
            if self._shown_indices_() is not None:
                self._lay_out_shown_(progress)
            else:
                progress(f"computing hyperbolic UMAP layout of {len(s)} texts")
                s.layout = layout.compute(s.embeddings, seed=self.seed)
            s.save()

    def arrows(self, i, js, progress=print):
        """The arrows from point *i* to points *js* -- indices into the snapshot -- each decoded: `arrows.decode`'s (toward, away), phrases in place of indices.

        The phrases are the stored points' embed texts cut up (`arrows.phrases`),
        embedded once and kept in the store. Those not yet embedded are embedded
        here, outside the lock -- the first Shift over a corpus takes a while --
        and *progress* hears how far it has got.
        """
        with self.lock:
            s = self.store
            rows = self._shown_[0] if self._shown_ is not None else np.arange(len(s))
            here, there = s.embeddings[rows[i]], [s.embeddings[rows[j]] for j in js]
            wanted = arrows.phrases([p.get("embed", p["text"]) for p in s.points])
            known = set(s.phrases)
            name = s.embedder
        todo = [p for p in wanted if p not in known]
        if todo:
            msg = f"arrows: embedding {len(todo)} phrases"
            batches = []
            for k in range(0, len(todo), 64):
                progress(msg, k, len(todo))
                batches.append(np.asarray(self.embedder(todo[k : k + 64]), dtype=np.float32))
            progress(f"arrows: {len(todo)} phrases embedded")
            with self.lock:
                s = self.store
                if s.embedder == name:  # not re-embedded, nor another corpus, while these were made
                    vecs = np.concatenate(batches)
                    s.phrase_embeddings = vecs if not len(s.phrases) else np.concatenate([s.phrase_embeddings, vecs])
                    s.phrases = s.phrases + todo
                    s.save_phrases()
        with self.lock:
            s = self.store
            index = {p: k for k, p in enumerate(s.phrases)}
            keep = [index[p] for p in wanted if p in index]
            names, vecs = [s.phrases[k] for k in keep], s.phrase_embeddings[keep] if keep else np.zeros((0, 0))

        def named(pairs):
            return [(names[k], score) for k, score in pairs]

        return [tuple(named(side) for side in arrows.decode(here, v, vecs)) for v in there]

    def _shown_indices_(self):
        """The indices of the points shown, where the source gives fewer than are stored; None where it gives them all."""
        if self.keys is None:
            return None
        shown = [i for i, p in enumerate(self.store.points) if p.get("key") in self.keys]
        return None if len(shown) == len(self.store) else np.array(shown, dtype=int)

    def _lay_out_shown_(self, progress):
        """Lay out the points shown, by themselves: kept apart from the stored layout, which is of them all."""
        shown = self._shown_indices_()
        progress(f"computing hyperbolic UMAP layout of {len(shown)} of {len(self.store)} texts")
        self._shown_ = (shown, layout.compute(self.store.embeddings[shown], seed=self.seed) if len(shown) else np.zeros(0, dtype=complex))

    def _sync_(self, progress):
        """Add the source's new entries and update changed ones; True if any embedding changed.

        The store is changed only once everything is embedded, so a job
        interrupted on the way -- its *progress* raising -- leaves it as it was.
        """
        s = self.store
        index = {p.get("key"): i for i, p in enumerate(s.points)}
        entries = self.source(progress)
        new = [e for e in entries if e.key not in index]
        changed = [e for e in entries if e.key in index and s.points[index[e.key]].get("embed") != e.embed]
        todo = new + changed
        if todo:
            msg = f"embedding {len(new)} new and {len(changed)} changed entries"
            batches = []
            for i in range(0, len(todo), 64):
                progress(msg, i, len(todo))
                batches.append(self.embedder([e.embed for e in todo[i : i + 64]]))
            progress(msg, len(todo), len(todo))
            vecs = np.concatenate(batches)
        self.keys, self._shown_ = {e.key for e in entries}, None
        for e in entries:
            if e.key in index:
                # Text and summary may change without the embedding changing:
                # a pipeline run again keeps its spec but gains a datetime.
                s.points[index[e.key]].update(text=e.text, summary=e.summary, embed=e.embed, **e.extra)
        if not todo:
            progress("source: nothing new")
            return False
        for e, v in zip(changed, vecs[len(new) :]):
            s.embeddings[index[e.key]] = v
        if new:
            s.add([e.text for e in new], [e.summary for e in new], vecs[: len(new)], [{"key": e.key, "embed": e.embed, **e.extra} for e in new])
        return True

    def add(self, text, progress=print):
        """Embed, summarize, store and place one text; returns its index."""
        text = text.strip()
        if not text:
            raise ValueError("empty text")
        progress("summarizing")
        summary = self.summarizer(text)
        progress("embedding")
        vec = self.embedder([text])
        with self.lock:
            s = self.store
            previous = s.layout
            s.add([text], [summary], vec)
            progress("updating layout")
            s.layout = layout.compute(s.embeddings, previous=previous, seed=self.seed)
            s.save()
            return len(s) - 1

    def snapshot(self):
        with self.lock:
            s = self.store
            if self._shown_ is not None:
                shown, positions = self._shown_
                points = [s.points[i] for i in shown]
                return Snapshot(
                    texts=tuple(p["text"] for p in points),
                    summaries=tuple(p["summary"] for p in points),
                    positions=np.array(positions, dtype=complex),
                    neighbors=layout.nearest(s.embeddings[shown], NEIGHBORS) if len(shown) else np.zeros((0, 0), dtype=int),
                    points=tuple(dict(p) for p in points),
                )
            return Snapshot(
                texts=tuple(p["text"] for p in s.points),
                summaries=tuple(p["summary"] for p in s.points),
                positions=np.array(s.layout, dtype=complex),
                neighbors=layout.nearest(s.embeddings, NEIGHBORS),
                points=tuple(dict(p) for p in s.points),
            )
