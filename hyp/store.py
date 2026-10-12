"""The on-disk corpus: texts, their summaries, embeddings and disk positions.

    <dir>/meta.json        {"embedder": name}
    <dir>/points.jsonl     one {"text", "summary", "added"} per line, with
                           "key" and "embed" for points read from a source
    <dir>/embeddings.npy   (n, d) float32
    <dir>/layout.npy       (n,) complex128 Poincare disk positions

Every write replaces whole files atomically, so a crash leaves the previous
state rather than a torn one. The corpus is small enough for that.
"""

import json
import os
import time
from pathlib import Path

import numpy as np


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = None
        self.points = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.layout = None
        self.load()

    def __len__(self):
        return len(self.points)

    def _path_(self, name):
        return self.root / name

    def load(self):
        meta = self._path_("meta.json")
        if meta.exists():
            self.embedder = json.loads(meta.read_text()).get("embedder")
        pts = self._path_("points.jsonl")
        if pts.exists():
            self.points = [json.loads(ln) for ln in pts.read_text().splitlines() if ln.strip()]
        emb = self._path_("embeddings.npy")
        if emb.exists():
            self.embeddings = np.load(emb)
        lay = self._path_("layout.npy")
        if lay.exists():
            self.layout = np.load(lay)
        # Embeddings without their points, or the reverse, are unusable; a
        # layout that does not cover every point is recomputed.
        n = min(len(self.points), len(self.embeddings))
        self.points, self.embeddings = self.points[:n], self.embeddings[:n]
        if self.layout is not None and len(self.layout) != n:
            self.layout = None

    def _write_(self, name, write):
        tmp = self._path_(name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, self._path_(name))

    def save(self):
        self._write_("meta.json", lambda f: f.write(json.dumps({"embedder": self.embedder}).encode()))
        self._write_("points.jsonl", lambda f: f.write("".join(json.dumps(p, default=str) + "\n" for p in self.points).encode()))
        self._write_("embeddings.npy", lambda f: np.save(f, self.embeddings))
        if self.layout is not None:
            self._write_("layout.npy", lambda f: np.save(f, self.layout))

    def reset(self, embedder):
        self.embedder = embedder
        self.points = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.layout = None

    def add(self, texts, summaries, embeddings, extras=None):
        """Append points; *extras*, one dict per point, adds fields to each."""
        now = time.time()
        extras = extras or [{}] * len(texts)
        self.points += [{"text": t, "summary": s, "added": now, **x} for t, s, x in zip(texts, summaries, extras)]
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = embeddings if self.embeddings.size == 0 else np.concatenate([self.embeddings, embeddings])
