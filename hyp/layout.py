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


# The radius the farthest of a few points is placed at, by `scaled`: as far out as the ring they were on.
FEW_RADIUS = 0.3
# How much of the distance between two points a link takes away, as the layout measures it -- not their embeddings,
# which say what they say: a branch of a conversation laid out by its parent, whatever it goes on to.
LINK_PULL = 0.7
# The farthest a point is laid out from the one it is linked to, in the layout's typical distance to a nearest neighbor:
# UMAP draws them together, but one neighbor among fifteen does not hold them, so a point laid out farther is moved in,
# along the geodesic (`_reach_`). Typical: the median, over at most REACH_SAMPLE points.
LINK_REACH = 1.5
REACH_SAMPLE = 500


def distances(embeddings, links=()):
    """Their cosine distances, a square, as the layout measures them: shortened by LINK_PULL between two points *links* -- `[(i, j)]` -- joins."""
    e = _normalized_(np.asarray(embeddings, dtype=np.float32))
    d = np.maximum(0.0, 1.0 - e @ e.T)
    for i, j in links:
        d[i, j] = d[j, i] = d[i, j] * (1.0 - LINK_PULL)
    np.fill_diagonal(d, 0.0)
    return d


def scaled(embeddings, radius=FEW_RADIUS, links=()):
    """A few points on the disk, placed by classical multidimensional scaling of their cosine distances.

    The plane positions whose distances best match theirs, centered on the
    origin and scaled so that the farthest is at *radius*: their distances
    relative to one another are theirs, and near the origin the disk's
    distances are the plane's, near enough. One point is at the origin.
    """
    n = len(embeddings)
    if n < 2:
        return np.zeros(n, dtype=complex)
    d2 = distances(embeddings, links).astype(np.float64) ** 2
    j = np.eye(n) - 1.0 / n
    vals, vecs = np.linalg.eigh(-0.5 * j @ d2 @ j)
    top = np.argsort(vals)[::-1][:2]
    xy = vecs[:, top] * np.sqrt(np.maximum(vals[top], 0.0))
    w = xy[:, 0] + 1j * (xy[:, 1] if xy.shape[1] > 1 else 0.0)
    w = w - w.mean()
    far = float(np.abs(w).max())
    return w * (radius / far) if far > 1e-12 else np.zeros(n, dtype=complex)


def _linked_knn_(embeddings, k, links):
    """UMAP's neighbor graph, given it: each row's *k* nearest -- itself first -- by `distances`, a linked row among them however far.

    `(indices, distances, None)`: no search index, as no new point is ever
    placed by transform.
    """
    d = distances(embeddings, links)
    np.fill_diagonal(d, -1.0)  # itself first, whatever else is at no distance
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    for i, j in links:
        for a, b in ((i, j), (j, i)):
            if b not in idx[a]:
                idx[a, -1] = b  # in place of its farthest
    dist = np.take_along_axis(d, idx, axis=1)
    order = np.argsort(dist, axis=1, kind="stable")
    idx, dist = np.take_along_axis(idx, order, axis=1), np.take_along_axis(dist, order, axis=1)
    return idx.astype(np.int64), np.maximum(dist, 0.0).astype(np.float32), None


# One layout at a time: UMAP's numba kernels run parallel, and numba's default
# threading layer aborts the process -- uncatchably -- when two threads launch
# them at once, as a slice disk laid out during a reload would.
_LOCK_ = threading.Lock()


def compute(embeddings, previous=None, seed=0, links=()):
    """Disk positions for embeddings, centered on their hyperbolic barycenter; one call at a time."""
    with _LOCK_:
        return _compute_(embeddings, previous, seed, links)


def _compute_(embeddings, previous=None, seed=0, links=()):
    """Disk positions for embeddings, centered on their hyperbolic barycenter.

    previous, if given, are positions for the first len(previous) rows: they
    seed the optimization, and new rows start beside their nearest old
    neighbor -- or the old one a link joins them to -- so adding a point
    nudges the map instead of reshuffling it.

    *links*, `[(i, j)]` -- row *i* grew from row *j*, a branch from its
    parent -- join rows the layout puts near each other: their distance
    shortened (`distances`), each among the other's neighbors in UMAP's
    graph, which is given it (`_linked_knn_`), and *i* at most LINK_REACH
    from *j* (`_reach_`).
    """
    links = [(int(i), int(j)) for i, j in links if i != j]
    emb = np.asarray(embeddings, dtype=np.float32)
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=complex)
    if n < 5:
        # Too few points for a neighbor graph -- and UMAP's would not do: it scales each point's distances by its
        # nearest neighbor's, so three points come out equidistant whatever they mean. Placed by their cosine
        # distances themselves instead; they were once a ring, the same for every corpus.
        w = scaled(embeddings, links=links)
        if previous is not None and len(previous) == n:
            reflect, rot = _align_(w, np.asarray(previous))
            w = (np.conj(w) if reflect else w) * rot
        return _reach_(w, links)

    import umap

    init = "spectral" if n >= 32 else "random"
    if previous is not None and 0 < len(previous) <= n:
        m = len(previous)
        xy = poincare.to_hyperboloid(previous)
        if m < n:
            nn = nearest(emb, 1)[m:, 0]
            joined = {i: j for a, b in links for i, j in ((a, b), (b, a)) if i >= m and j < m}
            nn = np.array([joined.get(m + r, nn[r]) for r in range(len(nn))], dtype=int)  # a new point linked: by its link
            # A new point's nearest neighbor may itself be new; fall back to
            # the origin for those rather than chase chains.
            base = np.where(nn[:, None] < m, xy[np.minimum(nn, m - 1)], 0.0)
            rng = np.random.default_rng(seed + n)
            xy = np.concatenate([xy, base + rng.normal(scale=0.05, size=base.shape)])
        init = xy

    k = min(15, n - 1)
    reducer = umap.UMAP(
        n_neighbors=k,
        metric="cosine",
        output_metric="hyperboloid",
        init=init,
        random_state=seed,
        precomputed_knn=_linked_knn_(emb, k, links) if links else (None, None, None),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy = reducer.fit_transform(emb)

    w = poincare.from_hyperboloid(xy)
    w = poincare.translate(w, poincare.center_of_mass(w))
    if previous is not None and len(previous):
        reflect, rot = _align_(w[: len(previous)], np.asarray(previous))
        w = (np.conj(w) if reflect else w) * rot
    return _reach_(w, links)


def _reach_(w, links):
    """*w*, each row *i* of *links* `(i, j)` moved in to row *j*'s reach where it lies farther, along the geodesic between them.

    The reach is LINK_REACH times the median distance from a point to its
    nearest, over the layout. A row is moved after the row it grew from, so
    a branch of a branch follows its parent where that was moved.
    """
    if not links or len(w) < 2:
        return w
    w, above = np.array(w, dtype=complex), dict(links)
    some = np.random.default_rng(0).choice(len(w), min(len(w), REACH_SAMPLE), replace=False)
    d = poincare.distance(w[some][:, None], w[None, :])
    d[np.arange(len(some)), some] = np.inf
    reach = LINK_REACH * float(np.median(d.min(axis=1)))

    def depth(i):
        seen = set()
        while i in above and i not in seen:
            seen.add(i)
            i = above[i]
        return len(seen)

    for i in sorted(above, key=depth):
        j = above[i]
        far = float(poincare.distance(w[j], w[i]))
        if far > reach:
            w[i] = complex(poincare.geodesic(w[j], w[i], np.array(reach / far)))
    return w
