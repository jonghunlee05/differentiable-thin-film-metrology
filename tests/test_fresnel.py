"""Fresnel coefficients at a single interface.

Spec §4.2.
Normal-incidence limit by DTFM-006; Brewster-angle null by DTFM-007.
"""

import numpy as np
import pytest

from src.fresnel import cos_theta_t, fresnel_r, reflectance

# (incident, transmitted). Air→glass, air→silicon-ish, and glass→air so the
# n_i > n_j direction is covered too.
INDEX_PAIRS = [(1.0, 1.46), (1.0, 3.88), (1.46, 1.0), (1.33, 1.46)]


@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_normal_incidence_matches_closed_form(n_i, n_j):
    """At θ=0 the coefficients collapse to (n_i − n_j)/(n_i + n_j).

    r_p carries the opposite sign under the §4.2 convention; see the module
    docstring. Asserting the sign explicitly is the point — a flipped r_p would
    otherwise survive here and put a spurious sign in every multilayer stack.
    """
    expected = (n_i - n_j) / (n_i + n_j)
    r_s, r_p = fresnel_r(n_i, n_j, 0.0)

    assert r_s.imag == pytest.approx(0.0, abs=1e-15)
    assert r_s.real == pytest.approx(expected, rel=1e-12)
    assert r_p.real == pytest.approx(-expected, rel=1e-12)


@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_normal_incidence_reflectance_is_polarisation_independent(n_i, n_j):
    """R_s and R_p must agree at θ=0 — the sign difference is not observable."""
    expected = ((n_i - n_j) / (n_i + n_j)) ** 2
    R_s, R_p = reflectance(n_i, n_j, 0.0)

    assert R_s == pytest.approx(expected, rel=1e-12)
    assert R_p == pytest.approx(expected, rel=1e-12)


def test_inputs_broadcast():
    """A swept angle and a wavelength-dependent index evaluate in one call."""
    theta = np.linspace(0.0, 1.4, 40)
    n_j = np.linspace(1.45, 1.47, 40)
    r_s, r_p = fresnel_r(1.0, n_j, theta)

    assert r_s.shape == theta.shape
    assert r_p.shape == theta.shape


def test_reflectance_is_bounded():
    """Passive interface: no angle may reflect more than the incident power."""
    theta = np.linspace(0.0, np.pi / 2 - 1e-9, 500)
    for n_i, n_j in INDEX_PAIRS:
        R_s, R_p = reflectance(n_i, n_j, theta)
        assert np.all(R_s >= 0.0) and np.all(R_s <= 1.0 + 1e-12)
        assert np.all(R_p >= 0.0) and np.all(R_p <= 1.0 + 1e-12)


def test_grazing_incidence_reflects_everything():
    """R → 1 as θ → 90°, for either polarisation."""
    theta = np.pi / 2 - 1e-9
    for n_i, n_j in INDEX_PAIRS:
        R_s, R_p = reflectance(n_i, n_j, theta)
        assert R_s == pytest.approx(1.0, abs=1e-6)
        assert R_p == pytest.approx(1.0, abs=1e-6)


def test_total_internal_reflection_is_unit_magnitude_not_nan():
    """Past the critical angle cosθ_t is imaginary, so |r| = 1 exactly.

    A real-valued sqrt would return nan here instead, silently and mid-array.
    """
    n_i, n_j = 1.46, 1.0
    theta_c = np.arcsin(n_j / n_i)
    theta = np.linspace(theta_c + 1e-6, np.pi / 2 - 1e-9, 200)

    cos_j = cos_theta_t(n_i, n_j, theta)
    assert np.all(np.isfinite(cos_j))
    assert np.all(np.abs(cos_j.real) < 1e-7), "cosθ_t should be purely imaginary beyond θ_c"

    R_s, R_p = reflectance(n_i, n_j, theta)
    assert np.allclose(R_s, 1.0, atol=1e-12)
    assert np.allclose(R_p, 1.0, atol=1e-12)


def test_no_nan_anywhere_in_a_full_sweep():
    """Guards the failure mode the emath.sqrt choice exists to prevent."""
    theta = np.linspace(0.0, np.pi / 2 - 1e-9, 1000)
    for n_i, n_j in INDEX_PAIRS:
        r_s, r_p = fresnel_r(n_i, n_j, theta)
        assert np.all(np.isfinite(r_s)) and np.all(np.isfinite(r_p))
