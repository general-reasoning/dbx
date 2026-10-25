"""The engine: embed, summarize, store and lay out texts.

The UI talks to it from a worker thread; a lock serializes mutations, and
snapshot() hands the UI an immutable copy to draw.
"""

import hashlib
import threading
from dataclasses import dataclass

import numpy as np

from . import arrows, layout, poincare, seeds

NEIGHBORS = 5
# A point's shards are ringed about it (`Engine.shard`): the farthest this far from it, as if it were at the origin, and
# none nearer than SHARD_INNER -- their own neighborhood, wherever on the disk the point lies.
SHARD_RADIUS = 0.45
SHARD_INNER = 0.1
# How much of a shard's embedding is its point's: as much as its own, so its nearest are its point and its siblings.
SHARD_PULL = 1.0
SHARD_BATCH = 8  # shards embedded at a time: a session's few turns still move the bar


@dataclass(frozen=True)
class Snapshot:
    texts: tuple
    summaries: tuple
    positions: np.ndarray  # complex disk positions, data coordinates
    neighbors: np.ndarray  # (n, k) nearest neighbors in embedding space
    points: tuple = ()  # each point's stored fields, for a specialization's use


@dataclass(frozen=True)
class Shards:
    """A point's shards, as `Engine.shard` or `Engine.shatter` made them: their stored fields, their embeddings, and -- ringed about the point -- their places about it, were it at the origin.

    Each one's fields have "origin": its point's key, what it slides out of.
    """

    points: tuple
    embeddings: np.ndarray
    offsets: np.ndarray = None  # None for shards laid out on their own, `shatter`'s


def _links_(points):
    """The rows each point is joined to by its "link" -- another's key: what it grew from, a branch's parent -- `[(i, j)]`, laid out near each other."""
    row = {p.get("key"): n for n, p in enumerate(points) if p.get("key") is not None}
    return [(n, row[p["link"]]) for n, p in enumerate(points) if p.get("link") in row and row[p["link"]] != n]


def _shard_points_(key, entries):
    """Shards *entries*, as their point's -- *key*'s -- stored fields: a point's, with "origin", the key."""
    return tuple({"text": e.text, "summary": e.summary, "key": e.key, "embed": e.embed, **e.extra, "origin": key} for e in entries)


def _label_(point):
    """A point as the arrows name it: its label -- a pipeline's, with its VAR.name -- or its summary."""
    return " | ".join(str(x) for x in (point.get("label") or point.get("summary", ""), point.get("name")) if x)


class Engine:
    def __init__(self, store, embedder, summarizer, seed=0, source=None):
        """*source*, if given, replaces the seed corpus: a callable taking a
        progress callback and returning objects with `key`, `text`,
        `embed`, `summary` and `extra` (more fields to store), such as
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
        self.shards = {}  # a point's key -> its Shards: shown after the points, about it, while it is shown
        self.shattered = {}  # a point's key -> its Shards: shown in its place, laid out on their own among the rest
        self._shattered_at_ = None  # {key: position} of the rows shown while shattered: laid out again as they change
        self._shard_vectors_ = {}  # a shard's embed text -> its own embedding: a point sharded again embeds only what is new

    def switch(self, store, source):
        """Another corpus: *store*, synced from *source*. `start` brings it up to date."""
        with self.lock:
            self.store, self.source = store, source
            self.keys, self._shown_ = None, None
            self.shards, self.shattered, self._shattered_at_ = {}, {}, None

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
                s.layout = layout.compute(s.embeddings, previous=keep, seed=self.seed, links=_links_(s.points))
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
                s.layout = layout.compute(s.embeddings, seed=self.seed, links=_links_(s.points))
            s.save()

    def reembed(self, progress=print):
        """Embed the whole corpus afresh and lay it out anew: from the source where there is one, else the texts stored.

        Points the source no longer gives are dropped -- a text typed in, say,
        where every point is a file. Everything is embedded before the store
        changes, so a job interrupted on the way leaves it as it was. The
        arrows' lines are kept: they are read by the embed texts, which a
        re-embedding by the same model does not change.
        """
        with self.lock:
            s = self.store
            if self.source is not None:
                entries = self.source(progress)
                rows = [(e.text, e.summary, {"key": e.key, "embed": e.embed, **e.extra}, e.embed) for e in entries]
            else:
                rows = [(p["text"], p["summary"], {k: v for k, v in p.items() if k not in ("text", "summary", "added")},
                         p.get("embed", p["text"])) for p in s.points]
            vecs = self._embed_([r[3] for r in rows], progress, f"re-embedding {len(rows)} texts with {self.embedder.name}")
            notes = s.notes
            s.reset(self.embedder.name)
            s.notes = notes
            if rows:
                s.add([r[0] for r in rows], [r[1] for r in rows], vecs, [r[2] for r in rows])
            self.keys = {e.key for e in entries} if self.source is not None else None
            self._shown_ = None
            progress(f"computing hyperbolic UMAP layout of {len(s)} texts")
            s.layout = layout.compute(s.embeddings, seed=self.seed, links=_links_(s.points)) if len(s) else None
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
                s.layout = layout.compute(s.embeddings, previous=keep, seed=self.seed, links=_links_(s.points))
            s.save()
            return moved

    def arrow_notes(self, i, js, ask, progress=print):
        """What each arrow from point *i* to points *js* -- indices into the snapshot -- means: `{"line", "same", "cos"}`.

        An arrow is read where else its direction runs in the corpus
        (`arrows.evidence`): the other neighbor pairs making the same move,
        and the ends of its axis. *ask*, a function from a prompt to its
        answer's text (Claude), names each direction in a line: every arrow
        not named before in one question, at most once a pair -- the notes
        are kept in the store, by the two embed texts and the corpus's size,
        as the evidence is the whole corpus's. An arrow between two equal
        embeddings is not asked about.
        """

        with self.lock:
            s = self.store
            points, embeddings, _ = self._rows_()
            units = arrows.unit_rows(embeddings)
            embeds = [p.get("embed", p["text"]) for p in points]
            keys = [hashlib.sha1(f"{len(points)}\0{embeds[i]}\0{embeds[j]}".encode()).hexdigest()[:20] for j in js]
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

    def shard(self, key, entries, progress=print):
        """Show point *key* with its shards: *entries*, as a source gives them, each embedded and placed about the point.

        A shard's embedding is its own and its point's averaged (SHARD_PULL):
        its nearest are its point and its siblings, as the hover's arcs and
        the arrows have them. Its place is its point's neighborhood: ringed
        about it, by classical scaling of what the shards themselves say
        (`layout.about`), and carried wherever the point is laid out. The map
        does not move for them, and is as it was once they go (`unshard`).
        Sharded again -- relabeled, or a turn added -- only what is new is
        embedded.
        """
        with self.lock:
            s = self.store
            i = next((n for n, p in enumerate(s.points) if p.get("key") == key), None)
            if i is None:
                raise KeyError(f"no point {key!r} to shard")
            own = self._vectors_(entries, progress)
            if not entries:
                self.shards.pop(key, None)
                return
            whole = arrows.unit_rows(s.embeddings[i : i + 1])
            mixed = arrows.unit_rows(arrows.unit_rows(own) + SHARD_PULL * whole) * np.linalg.norm(own, axis=1, keepdims=True)
            self.shards[key] = Shards(points=_shard_points_(key, entries), embeddings=mixed.astype(np.float32),
                                      offsets=layout.about(own, SHARD_RADIUS, SHARD_INNER))

    def unshard(self, key=None):
        """Fold point *key*'s shards back into it -- every point's, without *key*; whether any were shown."""
        with self.lock:
            if key is None:
                had, self.shards = bool(self.shards), {}
                return had
            return self.shards.pop(key, None) is not None

    def shatter(self, groups, progress=print):
        """Show each point of *groups* -- `{key: entries}`, as a source gives them -- as its shards instead, on their own among the rest.

        Not anchored to it, as `shard`'s are: a shard's embedding is its own,
        and the points shattered are gone from the disk, which is laid out
        again with their shards in their place -- UMAP seeded with where
        everything was, each shard beside its nearest (`layout.compute`). So
        a shard lies where what it says puts it, wherever its point was.
        Shown till `unshatter`; shards ringed about a point (`shard`) go.
        Shattered again -- relabeled, or a turn added -- only what is new is
        embedded, and the disk is laid out again only if what it shows has
        changed, from where it was.
        """
        with self.lock:
            flat = [e for entries in groups.values() for e in entries]
            vecs, k, shattered = self._vectors_(flat, progress), 0, {}
            for key, entries in groups.items():
                if entries:
                    shattered[key] = Shards(points=_shard_points_(key, entries), embeddings=vecs[k : k + len(entries)])
                k += len(entries)
            was = self.shards, self.shattered, self._shattered_at_
            self.shards, self.shattered = {}, shattered
            try:
                self._rows_(progress)  # laid out now, on the thread called on, not by the next snapshot
            except BaseException:
                self.shards, self.shattered, self._shattered_at_ = was  # interrupted: as it was
                raise

    def unshatter(self):
        """Put the points shattered back, whole, the disk as it is without them; whether any were shattered."""
        with self.lock:
            had = bool(self.shattered)
            self.shattered, self._shattered_at_ = {}, None
            return had

    def _rows_(self, progress=print):
        """What is shown, row for row as a snapshot has it: `(points, embeddings, positions)`.

        The points, then each sharded one's shards, ringed about it; or,
        while points are shattered, the rest of them, then the shards of
        those shattered, laid out together (`_shattered_rows_`).
        """
        s = self.store
        if self._shown_ is not None:
            shown, positions = self._shown_
            points, embeddings = [s.points[i] for i in shown], s.embeddings[shown]
        else:
            points, embeddings, positions = list(s.points), s.embeddings, s.layout
        positions = np.array(positions, dtype=complex)
        if self.shattered:
            return self._shattered_rows_(points, embeddings, positions, progress)
        row = {p.get("key"): n for n, p in enumerate(points)}
        more = [(row[key], shards) for key, shards in self.shards.items() if row.get(key) is not None]
        if not more:
            return points, embeddings, positions
        points = points + [dict(p) for _, shards in more for p in shards.points]
        embeddings = np.concatenate([embeddings, *(shards.embeddings for _, shards in more)])
        positions = np.concatenate([positions, *(poincare.translate(shards.offsets, -positions[n]) for n, shards in more)])
        return points, embeddings, positions

    def _shattered_rows_(self, points, embeddings, positions, progress):
        """The rows shown while points are shattered: those not shattered, then the shards of those shown, laid out together.

        The layout is kept by key, and computed again only when the keys
        shown change: seeded with where each row was -- in the last such
        layout, or else on the disk -- those with a place first, the rest
        beside their nearest (`layout.compute`).
        """
        whole = [n for n, p in enumerate(points) if p.get("key") not in self.shattered]
        present = {p.get("key") for p in points}
        parts = [shards for key, shards in self.shattered.items() if key in present]
        rows = [points[n] for n in whole] + [dict(p) for shards in parts for p in shards.points]
        embedded = np.concatenate([embeddings[whole], *(shards.embeddings for shards in parts)])
        keys = [p.get("key") for p in rows]
        at = self._shattered_at_ or {}
        if set(keys) != set(at) or len(at) != len(keys):
            was = {**{points[n].get("key"): positions[n] for n in whole}, **at}
            order = sorted(range(len(rows)), key=lambda n: keys[n] not in was)  # those with a place first, in order
            m = sum(k in was for k in keys)
            progress(f"computing hyperbolic UMAP layout of {len(rows)} texts, {len(rows) - len(whole)} of them shards")
            laid = layout.compute(embedded[order], previous=np.array([was[keys[n]] for n in order[:m]], dtype=complex) if m else None,
                                  seed=self.seed, links=_links_([rows[n] for n in order]))
            self._shattered_at_ = at = {keys[n]: z for n, z in zip(order, laid)}
        return rows, embedded, np.array([at[k] for k in keys], dtype=complex)

    def _vectors_(self, entries, progress):
        """The embeddings of shards *entries*, a row each: those embedded before, kept by their embed texts; the rest embedded now."""
        new = list(dict.fromkeys(e.embed for e in entries if e.embed not in self._shard_vectors_))
        if new:
            self._shard_vectors_.update(zip(new, self._embed_(new, progress, f"embedding {len(new)} shards", batch=SHARD_BATCH)))
        if not entries:
            return np.zeros((0, self.store.embeddings.shape[1] if self.store.embeddings.ndim == 2 else 0), dtype=np.float32)
        return np.stack([self._shard_vectors_[e.embed] for e in entries]).astype(np.float32)

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
        points = [self.store.points[i] for i in shown]
        self._shown_ = (shown, layout.compute(self.store.embeddings[shown], seed=self.seed, links=_links_(points)) if len(shown)
                        else np.zeros(0, dtype=complex))

    def _sync_(self, progress):
        """Add the source's new entries and update changed ones; True if any embedding changed, or any link.

        The store is changed only once everything is embedded, so a job
        interrupted on the way -- its *progress* raising -- leaves it as it was.
        """
        s = self.store
        index = {p.get("key"): i for i, p in enumerate(s.points)}
        entries = self.source(progress)
        new = [e for e in entries if e.key not in index]
        changed = [e for e in entries if e.key in index and s.points[index[e.key]].get("embed") != e.embed]
        # A link made or gone moves points too, though no embedding changes: they are laid out by it.
        relinked = any(s.points[index[e.key]].get("link") != e.extra.get("link") for e in entries if e.key in index)
        todo = new + changed
        if todo:
            vecs = self._embed_([e.embed for e in todo], progress, f"embedding {len(new)} new and {len(changed)} changed entries")
        self.keys, self._shown_ = {e.key for e in entries}, None
        for e in entries:
            if e.key in index:
                # Text and summary may change without the embedding changing:
                # a pipeline run again keeps its spec but gains a datetime.
                s.points[index[e.key]].update(text=e.text, summary=e.summary, embed=e.embed, **e.extra)
        if not todo:
            progress("source: nothing new" if not relinked else "source: links changed")
            return relinked
        for e, v in zip(changed, vecs[len(new) :]):
            s.embeddings[index[e.key]] = v
        if new:
            s.add([e.text for e in new], [e.summary for e in new], vecs[: len(new)], [{"key": e.key, "embed": e.embed, **e.extra} for e in new])
        return True

    def _embed_(self, texts, progress, msg, batch=64):
        """*texts* embedded, *batch* at a time, saying *msg* with the count done: one row each."""
        batches = []
        for i in range(0, len(texts), batch):
            progress(msg, i, len(texts))
            batches.append(self.embedder(texts[i : i + batch]))
        progress(msg, len(texts), len(texts))
        return np.concatenate(batches) if batches else np.zeros((0, 0), dtype=np.float32)

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
            s.layout = layout.compute(s.embeddings, previous=previous, seed=self.seed, links=_links_(s.points))
            s.save()
            return len(s) - 1

    def snapshot(self):
        with self.lock:
            points, embeddings, positions = self._rows_()
            return Snapshot(
                texts=tuple(p["text"] for p in points),
                summaries=tuple(p["summary"] for p in points),
                positions=positions,
                neighbors=layout.nearest(embeddings, NEIGHBORS) if len(points) else np.zeros((0, 0), dtype=int),
                points=tuple(dict(p) for p in points),
            )
