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


# --- absorbing media and the branch cut, DTFM-010 ---------------------------
#
# Spec §4.2 flags the branch of this square root as the single most common bug
# in a from-scratch implementation, because the wrong root produces gain rather
# than absorption and nothing raises. These tests assert the physical
# consequence — a wave that decays — rather than the formula that produces it.

# n + ik with k >= 0. Weak absorbers through to a metal-like index.
ABSORBING = [1.5 + 0.05j, 3.88 + 0.02j, 4.0 + 1.5j, 1.46 + 0.5j, 0.15 + 3.5j]
# Incident media, including absorbing ones: every interior interface of a stack
# containing an absorbing film has one.
INCIDENT = [1.0, 1.46, 1.5 + 0.1j, 3.0 + 0.8j]

ABSORBING_ANGLES = np.linspace(0.0, np.pi / 2 - 1e-6, 60)


@pytest.mark.parametrize("n_j", ABSORBING)
@pytest.mark.parametrize("n_i", INCIDENT)
def test_absorbing_media_attenuate_and_never_amplify(n_i, n_j):
    """The acceptance criterion, stated as physics.

    A wave crossing into the transmitted medium carries
    exp(i·(2π/λ)·ñ_j·d·cosθ_j). Its magnitude after a distance d is
    exp(−(2π/λ)·d·Im(ñ_j cosθ_j)). Requiring that to be ≤ 1 for every positive
    d is exactly requiring Im(ñ_j cosθ_j) ≥ 0.
    """
    theta = torch.tensor(ABSORBING_ANGLES)
    cos_j = pt.cos_theta_t(n_i, n_j, theta)

    decay_rate = (pt.as_complex(n_j) * cos_j).imag
    assert torch.all(decay_rate >= -1e-12), "wrong root: the medium is amplifying"

    # The same statement as an amplitude, over a range of thicknesses.
    for d_over_lambda in (0.1, 1.0, 10.0):
        amplitude = torch.exp(-2 * np.pi * d_over_lambda * decay_rate)
        assert torch.all(amplitude <= 1.0 + 1e-12)


@pytest.mark.parametrize("n_j", ABSORBING)
@pytest.mark.parametrize("n_i", INCIDENT)
def test_reflectance_from_an_absorbing_medium_is_bounded(n_i, n_j):
    """A passive interface cannot reflect more than it receives.

    Only checked for a transparent incident medium: with an absorbing one, R is
    a ratio of fields inside a lossy material and is not bounded by one.
    """
    if np.imag(n_i) != 0.0:
        pytest.skip("R is not a power ratio when the incident medium absorbs")

    R_s, R_p = pt.reflectance(n_i, n_j, torch.tensor(ABSORBING_ANGLES))
    assert torch.all(R_s >= 0.0) and torch.all(R_s <= 1.0 + 1e-12)
    assert torch.all(R_p >= 0.0) and torch.all(R_p <= 1.0 + 1e-12)


@pytest.mark.parametrize("n_j", ABSORBING)
@pytest.mark.parametrize("n_i", INCIDENT)
def test_absorbing_torch_matches_absorbing_numpy(n_i, n_j):
    """Parity from DTFM-008, re-established now that both handle complex indices."""
    ref = np_fresnel.cos_theta_t(n_i, n_j, ABSORBING_ANGLES)
    got = pt.cos_theta_t(n_i, n_j, torch.tensor(ABSORBING_ANGLES)).numpy()
    assert np.allclose(got, ref, rtol=0, atol=1e-13)

    # Relative, unlike the transparent case. Between two nearly equal absorbing
    # indices the Fresnel denominator nearly cancels and |r| can reach ~70, so an
    # absolute bound would be asserting far below the round-off of the values
    # themselves rather than testing agreement.
    ref_s, ref_p = np_fresnel.fresnel_r(n_i, n_j, ABSORBING_ANGLES)
    got_s, got_p = pt.fresnel_r(n_i, n_j, torch.tensor(ABSORBING_ANGLES))
    assert np.allclose(got_s.numpy(), ref_s, rtol=1e-12, atol=1e-14)
    assert np.allclose(got_p.numpy(), ref_p, rtol=1e-12, atol=1e-14)


def test_the_principal_root_really_is_wrong_often_enough_to_matter():
    """Guards the guard.

    If the naive torch.sqrt happened to be correct everywhere, the selection
    above would be dead code and could be deleted without any test noticing.
    It is not: the principal root is unphysical across a large fraction of
    absorbing cases, so this asserts the guard is load-bearing.
    """
    unphysical = 0
    total = 0
    for n_i in INCIDENT:
        for n_j in ABSORBING:
            n_jc = pt.as_complex(n_j)
            sin_t = (pt.as_complex(n_i) / n_jc) * torch.sin(
                pt.as_complex(torch.tensor(ABSORBING_ANGLES))
            )
            naive = torch.sqrt(1.0 - sin_t**2)
            unphysical += int((~pt.is_forward(n_jc, naive)).sum())
            total += naive.numel()

    assert unphysical > 0.1 * total, "branch guard appears to be dead code"


def test_transparent_media_are_unaffected_by_the_guard():
    """A regression guard on everything built before absorption existed."""
    for n_i, n_j in INDEX_PAIRS:
        ref_s, ref_p = np_fresnel.fresnel_r(n_i, n_j, THETA)
        got_s, got_p = pt.fresnel_r(n_i, n_j, torch.tensor(THETA))
        assert np.allclose(got_s.numpy(), ref_s, rtol=0, atol=1e-13)
        assert np.allclose(got_p.numpy(), ref_p, rtol=0, atol=1e-13)


def test_a_gain_medium_is_rejected_rather_than_silently_computed():
    """Im(ñ) < 0 is an emitting medium, outside the physics this models.

    Better a loud failure than a plausible spectrum from an impossible material.
    """
    with pytest.raises(AssertionError, match="gain medium"):
        pt.cos_theta_t(1.0, 1.5 - 0.2j, torch.tensor([0.4]))

    with pytest.raises(AssertionError, match="incident"):
        pt.cos_theta_t(1.5 - 0.2j, 1.46, torch.tensor([0.4]))


def test_gradients_flow_through_the_branch_selection():
    """torch.where must not sever autograd — §5.3's Jacobian depends on it."""
    theta = torch.linspace(0.1, 1.4, 32, dtype=torch.float64, requires_grad=True)
    R_s, _ = pt.reflectance(1.0, 4.0 + 1.5j, theta)
    R_s.sum().backward()

    assert torch.all(torch.isfinite(theta.grad))
    assert torch.any(theta.grad != 0.0)
