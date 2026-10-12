import numpy as np
import pytest

from hyp import poincare as P

rng = np.random.default_rng(0)


def random_disk(n):
    return np.sqrt(rng.uniform(0, 0.95**2, n)) * np.exp(2j * np.pi * rng.uniform(size=n))


def test_translate_takes_a_to_zero_and_preserves_distance():
    a, z, w = random_disk(3)
    assert abs(P.translate(a, a)) < 1e-12
    assert P.distance(P.translate(z, a), P.translate(w, a)) == pytest.approx(P.distance(z, w))


def test_mobius_matches_translate_and_composes():
    a, b = random_disk(2)
    z = random_disk(20)
    Ta, Tb = P.Mobius.translation(a), P.Mobius.translation(b)
    assert np.allclose(Ta(z), P.translate(z, a))
    assert np.allclose(Ta.then(Tb)(z), Tb(Ta(z)))
    assert np.allclose(Ta.inverse()(Ta(z)), z)
    assert np.all(np.abs(Ta.then(Tb)(z)) < 1)


def test_transvection_takes_a_to_b_and_slides_their_geodesic_along_itself():
    a, b = random_disk(2)
    T = P.Mobius.transvection(a, b)
    assert T(a) == pytest.approx(b)
    # Points of the geodesic through a and b stay on it (no rotation): moved
    # to the origin with a, it is a diameter, so the ratio stays real.
    direction = P.translate(b, a)
    for t in (0.2, 0.5, 0.9):
        q = T(P.geodesic(a, b, t))
        assert (P.translate(q, a) / direction).imag == pytest.approx(0, abs=1e-9)


def test_sliding_toward_p_moves_p_along_its_diameter():
    # The center-on-click slide: at fraction s, p has covered s of its distance to 0.
    p = 0.8 * np.exp(0.7j)
    for s in (0.0, 0.25, 0.5, 1.0):
        q = P.Mobius.translation(P.toward(p, s))(p)
        assert P.distance0(q) == pytest.approx((1 - s) * P.distance0(p), abs=1e-9)
        if abs(q) > 1e-9:
            assert np.angle(q) == pytest.approx(np.angle(p))


def test_geodesic_endpoints_and_constant_speed():
    a, b = random_disk(2)
    assert P.geodesic(a, b, 0.0) == pytest.approx(a)
    assert P.geodesic(a, b, 1.0) == pytest.approx(b)
    m = P.geodesic(a, b, 0.3)
    assert P.distance(a, m) == pytest.approx(0.3 * P.distance(a, b))


def test_zoom_scales_distance_from_center_and_inverts():
    z = random_disk(10)
    assert np.allclose(P.distance0(P.zoom(z, 2.0)), 2.0 * P.distance0(z))
    assert np.allclose(P.unzoom(P.zoom(z, 3.0), 3.0), z)


def test_hyperboloid_round_trip_and_center_of_mass():
    w = random_disk(10)
    assert np.allclose(P.from_hyperboloid(P.to_hyperboloid(w)), w)
    c = P.center_of_mass(w)
    assert abs(P.center_of_mass(P.translate(w, c))) < 1e-9
