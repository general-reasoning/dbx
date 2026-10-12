"""Poincare disk geometry on complex numbers.

Points of the disk are complex numbers with |z| < 1. Everything here
vectorizes over numpy arrays of them.
"""

import numpy as np

# Points this close to the boundary are infinitely far away for any practical
# purpose; clamping keeps arctanh finite.
MAX_RADIUS = 1.0 - 1e-9


def _clamp_(r):
    return np.minimum(r, MAX_RADIUS)


def translate(z, a):
    """The disk isometry taking a to 0 along the geodesic through them."""
    return (z - a) / (1.0 - np.conj(a) * z)


def toward(p, t):
    """The point a fraction t of the hyperbolic way from 0 to p."""
    r = abs(p)
    if r == 0:
        return 0j
    return np.tanh(t * np.arctanh(_clamp_(r))) * p / r


def distance0(z):
    """Hyperbolic distance from 0 (curvature -1)."""
    return 2.0 * np.arctanh(_clamp_(np.abs(z)))


def distance(a, b):
    return distance0(translate(b, a))


def geodesic(a, b, t):
    """The point a fraction t of the hyperbolic way from a to b.

    Translate a to 0, where the geodesic is a diameter, walk out along it, and
    translate back.
    """
    b0 = translate(b, a)
    r = np.abs(b0)
    safe = np.where(r > 0, r, 1.0)
    scale = np.where(r > 0, np.tanh(t * np.arctanh(_clamp_(r))) / safe, 0.0)
    return translate(b0 * scale, -a)


def zoom(z, k):
    """Hyperbolic zoom about 0: a point at distance d is drawn at distance k*d.

    Not an isometry, but it magnifies the center while keeping the whole
    plane inside the disk, which is the point of a hyperbolic view.
    """
    r = np.abs(z)
    safe = np.where(r > 0, r, 1.0)
    scale = np.where(r > 0, np.tanh(k * np.arctanh(_clamp_(r))) / safe, 0.0)
    return z * scale


def unzoom(z, k):
    return zoom(z, 1.0 / k)


class Mobius:
    """An orientation-preserving disk isometry z -> (a z + b) / (conj(b) z + conj(a)).

    Kept as its matrix [[a, b], [conj b, conj a]] so that views compose
    without accumulating anything but a renormalizable scale.
    """

    def __init__(self, a=1 + 0j, b=0j):
        det = abs(a) ** 2 - abs(b) ** 2
        s = np.sqrt(det)
        self.a = complex(a) / s
        self.b = complex(b) / s

    def __call__(self, z):
        return (self.a * z + self.b) / (np.conj(self.b) * z + np.conj(self.a))

    def then(self, other):
        """other after self."""
        a = other.a * self.a + other.b * np.conj(self.b)
        b = other.a * self.b + other.b * np.conj(self.a)
        return Mobius(a, b)

    def inverse(self):
        return Mobius(np.conj(self.a), -self.b)

    @staticmethod
    def translation(p):
        """The isometry taking p to 0, as translate(., p)."""
        return Mobius(1 + 0j, -complex(p))

    @staticmethod
    def transvection(a, b):
        """The translation along the geodesic through a and b taking a to b.

        Unlike translation(a) then translation(-b), which also takes a to b,
        it adds no rotation: a drag built from these does not spin the plane.
        """
        b0 = complex(translate(b, a))
        return Mobius.translation(a).then(Mobius.translation(-b0)).then(Mobius.translation(-a))

    def __repr__(self):
        return f"Mobius(a={self.a:.4g}, b={self.b:.4g})"


def from_hyperboloid(xy):
    """UMAP's hyperboloid output (the free x, y coordinates) to the disk."""
    xy = np.asarray(xy, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    z = np.sqrt(1.0 + x * x + y * y)
    return (x + 1j * y) / (1.0 + z)


def to_hyperboloid(w):
    """The disk back to hyperboloid x, y coordinates."""
    w = np.asarray(w, dtype=complex)
    v = 2.0 * w / (1.0 - np.abs(w) ** 2)
    return np.stack([v.real, v.imag], axis=1)


def center_of_mass(w):
    """The hyperbolic barycenter of disk points: the hyperboloid mean, renormalized."""
    w = np.asarray(w, dtype=complex)
    if len(w) == 0:
        return 0j
    xy = to_hyperboloid(w)
    t = np.sqrt(1.0 + (xy**2).sum(axis=1))
    mx, my, mt = xy[:, 0].mean(), xy[:, 1].mean(), t.mean()
    norm = np.sqrt(mt * mt - mx * mx - my * my)
    mx, my, mt = mx / norm, my / norm, mt / norm
    return complex(mx, my) / (1.0 + mt)
