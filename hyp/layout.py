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


#: The radius the farthest of a few points is placed at, scaled by `_scaled_`: as far out as the ring they were on.
FEW_RADIUS = 0.3


def _scaled_(embeddings):
    """A few points on the disk, placed by classical multidimensional scaling of their cosine distances.

    The plane positions whose distances best match theirs, centered on the
    origin and scaled so that the farthest is at FEW_RADIUS: their distances
    relative to one another are theirs, and near the origin the disk's
    distances are the plane's, near enough. One point is at the origin.
    """
    e = _normalized_(np.asarray(embeddings, dtype=np.float64))
    n = len(e)
    if n < 2:
        return np.zeros(n, dtype=complex)
    d2 = np.maximum(0.0, 1.0 - e @ e.T) ** 2
    j = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(-0.5 * j @ d2 @ j)
    top = np.argsort(vals)[::-1][:2]
    xy = vecs[:, top] * np.sqrt(np.maximum(vals[top], 0.0))
    w = xy[:, 0] + 1j * (xy[:, 1] if xy.shape[1] > 1 else 0.0)
    w = w - w.mean()
    far = float(np.abs(w).max())
    return w * (FEW_RADIUS / far) if far > 1e-12 else np.zeros(n, dtype=complex)


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
        # Too few points for a neighbor graph -- and UMAP's would not do: it scales each point's distances by its
        # nearest neighbor's, so three points come out equidistant whatever they mean. Placed by their cosine
        # distances themselves instead; they were once a ring, the same for every corpus.
        w = _scaled_(embeddings)
        if previous is not None and len(previous) == n:
            reflect, rot = _align_(w, np.asarray(previous))
            w = (np.conj(w) if reflect else w) * rot
        return w

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
