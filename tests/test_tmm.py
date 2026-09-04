"""Transfer-matrix forward model.

Spec §4.2-§4.3.
Torch/numpy parity at a single interface by DTFM-008; quarter-wave null and
agreement with the reference `tmm` package by DTFM-015 and DTFM-016.
"""

import numpy as np
import pytest
import torch

from src import fresnel as np_fresnel
from src import tmm_torch as pt

# Spanning weak and strong contrast, and both directions across the interface
# so total internal reflection is exercised.
INDEX_PAIRS = [(1.0, 1.46), (1.0, 3.88), (1.46, 1.0), (1.33, 1.46), (2.0, 1.38)]

# Right up to grazing, and dense enough to cross every critical angle above.
THETA = np.linspace(0.0, np.pi / 2 - 1e-9, 401)


# --- torch/numpy parity, DTFM-008 -------------------------------------------
#
# The numpy module is the reference. These assert the port reproduces it to
# float64 precision, so a future change to either has the other to fail against.


@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_reflection_coefficients_match_numpy(n_i, n_j):
    ref_s, ref_p = np_fresnel.fresnel_r(n_i, n_j, THETA)
    got_s, got_p = pt.fresnel_r(n_i, n_j, torch.tensor(THETA))

    assert np.allclose(got_s.numpy(), ref_s, rtol=0, atol=1e-13)
    assert np.allclose(got_p.numpy(), ref_p, rtol=0, atol=1e-13)


@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_reflectance_matches_numpy(n_i, n_j):
    ref_s, ref_p = np_fresnel.reflectance(n_i, n_j, THETA)
    got_s, got_p = pt.reflectance(n_i, n_j, torch.tensor(THETA))

    assert np.allclose(got_s.numpy(), ref_s, rtol=0, atol=1e-13)
    assert np.allclose(got_p.numpy(), ref_p, rtol=0, atol=1e-13)


def test_matches_numpy_across_an_index_and_angle_grid():
    """A 2-D sweep, not just the parametrised pairs — the AC asks for a grid."""
    n_j = np.linspace(1.2, 4.0, 29).reshape(-1, 1)
    theta = THETA.reshape(1, -1)

    ref_s, ref_p = np_fresnel.fresnel_r(1.0, n_j, theta)
    got_s, got_p = pt.fresnel_r(1.0, torch.tensor(n_j), torch.tensor(theta))

    assert got_s.shape == ref_s.shape
    assert np.allclose(got_s.numpy(), ref_s, rtol=0, atol=1e-13)
    assert np.allclose(got_p.numpy(), ref_p, rtol=0, atol=1e-13)


def test_transmitted_cosine_matches_numpy_including_total_internal_reflection():
    """The branch choice must agree with numpy.emath.sqrt, not merely be finite.

    Past the critical angle the cosine is imaginary and the sign of that
    imaginary part is a branch decision. Getting it backwards still yields
    |r| = 1, so reflectance alone would not catch it — compare the cosine.
    """
    n_i, n_j = 1.46, 1.0
    theta = np.linspace(np.arcsin(n_j / n_i) + 1e-9, np.pi / 2 - 1e-9, 200)

    ref = np_fresnel.cos_theta_t(n_i, n_j, theta)
    got = pt.cos_theta_t(n_i, n_j, torch.tensor(theta)).numpy()

    assert np.all(np.isfinite(got))
    assert np.allclose(got, ref, rtol=0, atol=1e-13)
    assert np.all(got.imag >= 0.0), "wrong branch: evanescent decay became gain"


def test_brewster_null_survives_the_port():
    """The physics claim from DTFM-007, re-asserted against the torch path."""
    for n_i, n_j in INDEX_PAIRS:
        theta_b = torch.tensor(np.arctan2(n_j, n_i))
        _, r_p = pt.fresnel_r(n_i, n_j, theta_b)
        assert r_p.abs().item() == pytest.approx(0.0, abs=1e-14)


# --- differentiability, DTFM-008 --------------------------------------------
#
# Numerical agreement is only half the point of the port. Values matching while
# gradients fail to flow would satisfy a parity test and be useless downstream.


def test_gradient_flows_to_angle():
    theta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    R_s, _ = pt.reflectance(1.0, 1.46, theta)
    R_s.backward()

    assert theta.grad is not None
    assert torch.isfinite(theta.grad)
    assert theta.grad.item() != 0.0


def test_gradient_flows_to_refractive_index():
    n_j = torch.tensor(1.46, dtype=torch.float64, requires_grad=True)
    _, R_p = pt.reflectance(1.0, n_j, 0.4)
    R_p.backward()

    assert n_j.grad is not None
    assert torch.isfinite(n_j.grad)
    assert n_j.grad.item() != 0.0


def test_gradient_flows_through_a_batched_sweep():
    """Autograd must survive the broadcasting the dataset generator will use."""
    theta = torch.linspace(0.05, 1.4, 64, dtype=torch.float64, requires_grad=True)
    R_s, _ = pt.reflectance(1.0, 1.46, theta)
    R_s.sum().backward()

    assert theta.grad.shape == theta.shape
    assert torch.all(torch.isfinite(theta.grad))
    assert torch.any(theta.grad != 0.0)


def test_reflectance_is_real_valued():
    """R must come back real, not complex with a zero imaginary part — a complex
    R would silently poison any loss built on it.
    """
    R_s, R_p = pt.reflectance(1.0, 1.46, torch.tensor(THETA))
    assert not R_s.is_complex()
    assert not R_p.is_complex()
    assert R_s.dtype == torch.float64
