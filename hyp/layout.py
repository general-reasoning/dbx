"""Hyperbolic UMAP: embeddings to Poincare disk positions."""

import threading
import warnings

import numpy as np

from . import poincare


def _normalized_(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def nearest(embeddings, k):
    """Indices of each row's k nearest neighbors by cosine, itself excluded."""
    e = _normalized_(np.asarray(embeddings, dtype=np.float32))
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, len(e) - 1)
    if k <= 0:
        return np.zeros((len(e), 0), dtype=int)
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(-sim, idx, axis=1).argsort(axis=1)
    return np.take_along_axis(idx, order, axis=1)


def _align_(new, old):
    """The rotation, and whether to reflect first, best matching new to old about 0.

    UMAP's output is only defined up to isometry, so without this every
    relayout would spin the map around.
    """
    best = (False, 1.0)
    best_err = np.inf
    for reflect in (False, True):
        cand = np.conj(new) if reflect else new
        rot = np.sum(np.conj(cand) * old)
        rot = rot / abs(rot) if abs(rot) > 0 else 1.0
        err = np.sum(np.abs(cand * rot - old) ** 2)
        if err < best_err:
            best, best_err = (reflect, rot), err
    return best


# One layout at a time: UMAP's numba kernels run parallel, and numba's default
# threading layer aborts the process -- uncatchably -- when two threads launch
# them at once, as a slice disk laid out during a reload would.
_LOCK_ = threading.Lock()


def compute(embeddings, previous=None, seed=0):
    """Disk positions for embeddings, centered on their hyperbolic barycenter; one call at a time."""
    with _LOCK_:
        return _compute_(embeddings, previous, seed)


def _compute_(embeddings, previous=None, seed=0):
    """Disk positions for embeddings, centered on their hyperbolic barycenter.

    previous, if given, are positions for the first len(previous) rows: they
    seed the optimization, and new rows start beside their nearest old
    neighbor, so adding a point nudges the map instead of reshuffling it.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=complex)
    if n < 5:
        # Too few points for a neighbor graph: a small ring.
        return 0.3 * np.exp(2j * np.pi * np.arange(n) / n) if n > 1 else np.zeros(1, dtype=complex)

    import umap

    init = "spectral" if n >= 32 else "random"
    if previous is not None and 0 < len(previous) <= n:
        m = len(previous)
        xy = poincare.to_hyperboloid(previous)
        if m < n:
            nn = nearest(emb, 1)[m:, 0]
            # A new point's nearest neighbor may itself be new; fall back to
            # the origin for those rather than chase chains.
            base = np.where(nn[:, None] < m, xy[np.minimum(nn, m - 1)], 0.0)
            rng = np.random.default_rng(seed + n)
            xy = np.concatenate([xy, base + rng.normal(scale=0.05, size=base.shape)])
        init = xy

    reducer = umap.UMAP(
        n_neighbors=min(15, n - 1),
        metric="cosine",
        output_metric="hyperboloid",
        init=init,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy = reducer.fit_transform(emb)

    w = poincare.from_hyperboloid(xy)
    w = poincare.translate(w, poincare.center_of_mass(w))
    if previous is not None and len(previous):
        reflect, rot = _align_(w[: len(previous)], np.asarray(previous))
        w = (np.conj(w) if reflect else w) * rot
    return w
