"""The stored corpus: texts, their summaries, embeddings and disk positions.

Under *root*, a local directory or any fsspec url -- a datalake's
``{datalake}/.hyp`` over ssh, say:

    <dir>/meta.json        {"embedder": name}
    <dir>/points.jsonl     one {"text", "summary", "added"} per line, with
                           "key" and "embed" for points read from a source
    <dir>/embeddings.npy   (n, d) float32
    <dir>/layout.npy       (n,) complex128 Poincare disk positions
    <dir>/notes.json       {"embedder": name, "notes": {pair: note}}: what
                           each arrow means, as Claude named it, asked once

Every write goes to a temporary file that is then moved over the old one, so
a crash leaves the previous state rather than a torn one. The corpus is small
enough for that.
"""

import io
import json
import posixpath
import time

import fsspec
import numpy as np


class Store:
    def __init__(self, root, storage_options=None):
        self.root = str(root)
        self.fs, self._dir = fsspec.url_to_fs(self.root, **(storage_options or {}))
        self.fs.makedirs(self._dir, exist_ok=True)
        self.embedder = None
        self.points = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.layout = None
        self.notes = {}
        self.load()

    def __len__(self):
        return len(self.points)

    def _path_(self, name):
        return posixpath.join(self._dir, name)

    def _read_(self, name):
        """The bytes of file *name*, or None where there is none."""
        try:
            with self.fs.open(self._path_(name), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def load(self):
        meta = self._read_("meta.json")
        if meta is not None:
            self.embedder = json.loads(meta).get("embedder")
        pts = self._read_("points.jsonl")
        if pts is not None:
            self.points = [json.loads(ln) for ln in pts.decode().splitlines() if ln.strip()]
        emb = self._read_("embeddings.npy")
        if emb is not None:
            self.embeddings = np.load(io.BytesIO(emb))
        lay = self._read_("layout.npy")
        if lay is not None:
            self.layout = np.load(io.BytesIO(lay))
        # Embeddings without their points, or the reverse, are unusable; a
        # layout that does not cover every point is recomputed.
        n = min(len(self.points), len(self.embeddings))
        self.points, self.embeddings = self.points[:n], self.embeddings[:n]
        if self.layout is not None and len(self.layout) != n:
            self.layout = None
        notes = self._read_("notes.json")
        if notes is not None:
            notes = json.loads(notes)
            if notes.get("embedder") == self.embedder:  # read from another model's arrows, they say nothing of these
                self.notes = dict(notes.get("notes", {}))

    def _write_(self, name, write):
        tmp, path = self._path_(name + ".tmp"), self._path_(name)
        with self.fs.open(tmp, "wb") as f:
            write(f)
        try:
            self.fs.mv(tmp, path)
        except OSError:
            # SFTP's rename will not replace a file that exists.
            self.fs.rm(path)
            self.fs.mv(tmp, path)

    def clear(self):
        """Remove everything stored under *root*: what --reset does."""
        if self.fs.exists(self._dir):
            self.fs.rm(self._dir, recursive=True)
        self.fs.makedirs(self._dir, exist_ok=True)
        self.reset(None)

    def save(self):
        self._write_("meta.json", lambda f: f.write(json.dumps({"embedder": self.embedder}).encode()))
        self._write_("points.jsonl", lambda f: f.write("".join(json.dumps(p, default=str) + "\n" for p in self.points).encode()))
        self._write_("embeddings.npy", lambda f: np.save(f, self.embeddings))
        if self.layout is not None:
            self._write_("layout.npy", lambda f: np.save(f, self.layout))

    def save_notes(self):
        """Write the arrows' lines."""
        self._write_("notes.json", lambda f: f.write(json.dumps({"embedder": self.embedder, "notes": self.notes}).encode()))

    def reset(self, embedder):
        self.embedder = embedder
        self.points = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.layout = None
        self.notes = {}

    def add(self, texts, summaries, embeddings, extras=None):
        """Append points; *extras*, one dict per point, adds fields to each."""
        now = time.time()
        extras = extras or [{}] * len(texts)
        self.points += [{"text": t, "summary": s, "added": now, **x} for t, s, x in zip(texts, summaries, extras)]
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = embeddings if self.embeddings.size == 0 else np.concatenate([self.embeddings, embeddings])
