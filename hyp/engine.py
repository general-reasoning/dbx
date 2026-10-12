"""The engine: embed, summarize, store and lay out texts.

The UI talks to it from a worker thread; a lock serializes mutations, and
snapshot() hands the UI an immutable copy to draw.
"""

import threading
from dataclasses import dataclass

import numpy as np

from . import layout, seeds

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
            if s.layout is None or len(s.layout) != len(s) or moved:
                progress("computing hyperbolic UMAP layout")
                keep = previous if previous is not None and len(previous) <= len(s) else None
                s.layout = layout.compute(s.embeddings, previous=keep, seed=self.seed)
            s.save()

    def _sync_(self, progress):
        """Add the source's new entries and update changed ones; True if any embedding changed."""
        s = self.store
        index = {p.get("key"): i for i, p in enumerate(s.points)}
        entries = self.source(progress)
        new = [e for e in entries if e.key not in index]
        changed = [e for e in entries if e.key in index and s.points[index[e.key]].get("embed") != e.embed]
        for e in entries:
            if e.key in index:
                # Text and summary may change without the embedding changing:
                # a pipeline run again keeps its spec but gains a datetime.
                s.points[index[e.key]].update(text=e.text, summary=e.summary, embed=e.embed, **e.extra)
        if not new and not changed:
            progress("source: nothing new")
            return False
        progress(f"embedding {len(new)} new and {len(changed)} changed entries")
        todo = new + changed
        vecs = np.concatenate([self.embedder([e.embed for e in todo[i : i + 64]]) for i in range(0, len(todo), 64)])
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
            return Snapshot(
                texts=tuple(p["text"] for p in s.points),
                summaries=tuple(p["summary"] for p in s.points),
                positions=np.array(s.layout, dtype=complex),
                neighbors=layout.nearest(s.embeddings, NEIGHBORS),
                points=tuple(dict(p) for p in s.points),
            )
