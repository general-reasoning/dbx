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


def _label_(point):
    """A point as the arrows name it: its label -- a pipeline's, with its VAR.name -- or its summary."""
    return " | ".join(str(x) for x in (point.get("label") or point.get("summary", ""), point.get("name")) if x)


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

    def refresh(self, progress=print):
        """Read the source again and embed what is new or changed -- the layout nudged from where it was, not laid out afresh; whether anything moved.

        What a Claude session's every turn calls: the disk follows the
        conversation without reshuffling at each step, as `reload` would.
        """
        with self.lock:
            s = self.store
            if self.source is None:
                return False
            previous = s.layout
            moved = self._sync_(progress)
            if self._shown_indices_() is not None:
                self._lay_out_shown_(progress)
            elif moved or s.layout is None or len(s.layout) != len(s):
                progress("updating the layout")
                keep = previous if previous is not None and len(previous) <= len(s) else None
                s.layout = layout.compute(s.embeddings, previous=keep, seed=self.seed)
            s.save()
            return moved

    def arrow_notes(self, i, js, ask, progress=print):
        """What each arrow from point *i* to points *js* -- indices into the snapshot -- means: ``{"line", "same", "cos"}``.

        An arrow is read where else its direction runs in the corpus
        (`arrows.evidence`): the other neighbor pairs making the same move,
        and the ends of its axis. *ask*, a function from a prompt to its
        answer's text (Claude), names each direction in a line: every arrow
        not named before in one question, at most once a pair -- the notes
        are kept in the store, by the two embed texts and the corpus's size,
        as the evidence is the whole corpus's. An arrow between two equal
        embeddings is not asked about.
        """
        import hashlib

        with self.lock:
            s = self.store
            rows = self._shown_[0] if self._shown_ is not None else np.arange(len(s))
            units = arrows.unit_rows(s.embeddings[rows])
            points = [s.points[r] for r in rows]
            embeds = [p.get("embed", p["text"]) for p in points]
            keys = [hashlib.sha1(f"{len(rows)}\0{embeds[i]}\0{embeds[j]}".encode()).hexdigest()[:20] for j in js]
            notes = [s.notes.get(k) for k in keys]
            name = s.embedder
        labels = [_label_(p) for p in points]
        todo, asked = [], []
        if any(n is None for n in notes):
            pairs, moves = arrows.neighbor_moves(units, layout.nearest(units, NEIGHBORS))
            for n, j in enumerate(js):
                if notes[n] is not None:
                    continue
                ev = arrows.evidence(units, i, j, pairs, moves, labels)
                if ev is None:
                    notes[n] = {"line": "the same embedding", "same": 0, "cos": 1.0}
                else:
                    todo.append(n)
                    asked.append((labels[i], labels[j], ev))
        if todo:
            progress(f"arrows: Claude is reading {len(todo)} arrow{'s' if len(todo) > 1 else ''}")
            reply = ask(arrows.prompt(asked))
            said = arrows.parse(reply, len(todo))
            if said is None:
                raise ValueError(f"Claude's answer was not {len(todo)} lines: {reply[:200]}")
            for n, line, (_, _, ev) in zip(todo, said, asked):
                same, cos = ev.same()
                notes[n] = {"line": line, "same": same, "cos": round(cos, 3)}
            with self.lock:
                s = self.store
                if s.embedder == name:  # not re-embedded, nor another corpus, while Claude read
                    s.notes.update({keys[n]: notes[n] for n in todo})
                    s.save_notes()
            progress("arrows: read")
        return notes

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
