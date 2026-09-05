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


# --- interface and layer matrices, DTFM-011 ---------------------------------
#
# Spec §4.3. These are the building blocks the stack product is made of, so a
# convention error here propagates into every spectrum the project ever
# produces. The tests below pin the conventions rather than the arithmetic.

WAVELENGTH = 550.0  # nm; only the ratio d/λ enters
POLARISATIONS = ["s", "p"]


@pytest.mark.parametrize("pol", POLARISATIONS)
@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_interface_matrix_reproduces_the_fresnel_coefficient(n_i, n_j, pol):
    """The strongest available check on the interface matrix.

    §4.3 extracts r from a stack as M[1,0]/M[0,0]. For a bare interface the
    stack *is* I_ij, so that ratio must return the §4.2 coefficient exactly.
    This ties the matrix convention to the already-validated coefficients
    rather than trusting the transcription.
    """
    theta = torch.tensor(THETA)
    M = pt.interface_matrix(n_i, n_j, theta, pol)
    r_expected = pt.fresnel_r(n_i, n_j, theta)[POLARISATIONS.index(pol)]

    assert torch.allclose(M[..., 1, 0] / M[..., 0, 0], r_expected, atol=1e-13)


@pytest.mark.parametrize("pol", POLARISATIONS)
@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_interface_matrix_shape_and_structure(n_i, n_j, pol):
    """Shape (..., 2, 2), symmetric off-diagonal, equal diagonal — the form
    (1/t)·[[1, r], [r, 1]] demands all three.
    """
    M = pt.interface_matrix(n_i, n_j, torch.tensor(THETA), pol)

    assert M.shape == (THETA.size, 2, 2)
    assert torch.allclose(M[..., 0, 1], M[..., 1, 0], atol=1e-14)
    assert torch.allclose(M[..., 0, 0], M[..., 1, 1], atol=1e-14)


@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_stokes_relations_between_r_and_t(n_i, n_j):
    """t is new code; r is already trusted. The Stokes relations tie them.

    Under the §4.2 sign convention, t_s = 1 + r_s exactly, while the p relation
    carries the index ratio: t_p = (n_i/n_j)·(1 + r_p). Getting the p convention
    wrong is the classic thin-film error, and it would show up here.
    """
    theta = torch.tensor(THETA)
    r_s, r_p = pt.fresnel_r(n_i, n_j, theta)
    t_s, t_p = pt.fresnel_t(n_i, n_j, theta)

    assert torch.allclose(t_s, 1.0 + r_s, atol=1e-13)
    assert torch.allclose(t_p, (n_i / n_j) * (1.0 + r_p), atol=1e-13)


@pytest.mark.parametrize("pol", POLARISATIONS)
def test_zero_thickness_layer_is_the_identity(pol):
    """The acceptance criterion. A layer of no thickness must do nothing.

    Sensitive to a stray factor in δ — a wrong 2π, or λ and d transposed — since
    any of those would leave a residue at d = 0 only if the expression is
    malformed, and would otherwise show up as a wrong period later, much harder
    to attribute.
    """
    cos_theta = pt.cos_theta_t(1.0, 1.46, torch.tensor(THETA))
    L = pt.layer_matrix(1.46, 0.0, WAVELENGTH, cos_theta)

    identity = torch.eye(2, dtype=L.dtype).expand_as(L)
    assert torch.allclose(L, identity, atol=1e-15)


def test_layer_matrix_is_diagonal_with_unit_determinant():
    """det L = exp(−iδ)·exp(+iδ) = 1 for every δ, real or complex.

    A determinant drifting from one would mean the two diagonal entries are no
    longer reciprocal, i.e. the propagation is creating or destroying amplitude.
    """
    cos_theta = pt.cos_theta_t(1.0, 1.46, torch.tensor(THETA))
    L = pt.layer_matrix(1.46, 120.0, WAVELENGTH, cos_theta)

    assert torch.allclose(L[..., 0, 1], torch.zeros_like(L[..., 0, 1]), atol=1e-15)
    assert torch.allclose(L[..., 1, 0], torch.zeros_like(L[..., 1, 0]), atol=1e-15)
    assert torch.allclose(torch.linalg.det(L), torch.ones_like(L[..., 0, 0]), atol=1e-12)


def test_layer_phase_is_periodic_in_thickness():
    """R is quasi-periodic in n·d/λ — spec §5.2(b), the fringe-order ambiguity.

    Adding a half-wave of optical thickness advances δ by exactly π, so the
    layer matrix returns to itself after a full wave. This is the mechanism
    behind the degeneracy the whole project is built to characterise, and it is
    worth pinning here at its source.
    """
    n, cos_theta = 1.46, pt.cos_theta_t(1.0, 1.46, torch.tensor(0.0))
    d = WAVELENGTH / (2.0 * n)  # half-wave of optical thickness

    L0 = pt.layer_matrix(n, 0.0, WAVELENGTH, cos_theta)
    L_half = pt.layer_matrix(n, d, WAVELENGTH, cos_theta)
    L_full = pt.layer_matrix(n, 2 * d, WAVELENGTH, cos_theta)

    assert torch.allclose(L_half, -L0, atol=1e-12)
    assert torch.allclose(L_full, L0, atol=1e-12)


def test_absorbing_layer_gives_a_positive_imaginary_phase():
    """The DTFM-010 branch choice, seen through the layer phase.

    Absorption lives in Im(δ) = (2π/λ)·d·Im(ñ cosθ). The branch selection is
    what makes this positive; the other root would flip its sign and the layer
    would amplify.
    """
    n = 1.5 + 0.05j
    cos_theta = pt.cos_theta_t(1.0, n, torch.tensor(0.3))

    for d in (50.0, 200.0, 1000.0):
        assert pt.layer_phase(n, d, WAVELENGTH, cos_theta).imag.item() > 0.0

    assert pt.layer_phase(n, 0.0, WAVELENGTH, cos_theta).imag.item() == pytest.approx(0.0)


def test_absorbing_layer_matrix_decays_in_one_entry_and_grows_in_the_other():
    """A transfer matrix is not a propagator, and the difference matters here.

    It maps the fields at the far side of the layer back to the near side, so it
    *un*-propagates: the entry that decays physically appears inverted. With
    Im(δ) > 0, |L[0,0]| = exp(+Im δ) > 1 and |L[1,1]| = exp(−Im δ) < 1, and the
    two are exact reciprocals so det L = 1.

    This is worth asserting explicitly because |L[0,0]| > 1 looks alarming — it
    reads as gain — and a future reader checking for absorption would naturally
    look at the first entry and conclude the DTFM-010 branch fix was wrong.
    """
    n = 1.5 + 0.05j
    cos_theta = pt.cos_theta_t(1.0, n, torch.tensor(0.3))
    thicknesses = (0.0, 50.0, 200.0, 1000.0)

    mats = [pt.layer_matrix(n, d, WAVELENGTH, cos_theta) for d in thicknesses]
    decaying = [m[1, 1].abs().item() for m in mats]
    growing = [m[0, 0].abs().item() for m in mats]

    assert decaying[0] == pytest.approx(1.0)
    assert all(b < a for a, b in zip(decaying, decaying[1:], strict=False))
    assert decaying[-1] < 0.6

    for dec, gro in zip(decaying, growing, strict=True):
        assert dec * gro == pytest.approx(1.0, rel=1e-12)


def test_matrices_are_differentiable():
    """§7.3's reconstruction loss backpropagates through these."""
    d = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
    cos_theta = pt.cos_theta_t(1.0, 1.46, torch.tensor(0.3))

    L = pt.layer_matrix(1.46, d, WAVELENGTH, cos_theta)
    L[0, 0].abs().backward()

    assert torch.isfinite(d.grad) and d.grad.item() != 0.0

    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    M = pt.interface_matrix(1.0, 1.46, theta, "s")
    M[1, 0].abs().backward()

    assert torch.isfinite(theta.grad) and theta.grad.item() != 0.0


def test_polarisation_argument_is_validated():
    """A typo must fail loudly rather than silently selecting s."""
    with pytest.raises(ValueError, match="must be 's' or 'p'"):
        pt.interface_matrix(1.0, 1.46, torch.tensor(0.3), "S")


# --- the full stack, DTFM-012 -----------------------------------------------
#
# Spec §4.3. M = I_01·L_1·I_12·…·I_{N−1,N}, then r = M[1,0]/M[0,0].
#
# This is the forward model everything downstream depends on: the dataset
# generator of §7.1, the classical fit of §6, and the reconstruction loss of
# §7.3 all call it. The tests below assert physical properties — limits,
# periodicities, conservation — rather than reproducing the arithmetic.

SPECTRUM = torch.linspace(400.0, 800.0, 51)


@pytest.mark.parametrize("pol", POLARISATIONS)
@pytest.mark.parametrize(("n_i", "n_j"), INDEX_PAIRS)
def test_stack_with_no_layers_is_the_bare_interface(n_i, n_j, pol):
    """The limit that ties the stack to the already-validated §4.2 result."""
    theta = torch.tensor(THETA)
    got = pt.stack_reflectance(WAVELENGTH, [], [n_i, n_j], theta, pol)
    expected = pt.reflectance(n_i, n_j, theta)[POLARISATIONS.index(pol)]

    assert torch.allclose(got, expected, atol=1e-13)


@pytest.mark.parametrize("pol", POLARISATIONS)
def test_zero_thickness_layer_vanishes_from_the_stack(pol):
    """A layer of no thickness must be indistinguishable from no layer.

    Stronger than DTFM-011's identity check: that showed L = I in isolation,
    this shows the surrounding interface matrices then collapse correctly too,
    which is a statement about the whole product rather than one factor.
    """
    for angle in (0.0, 0.4, 1.2):
        with_layer = pt.stack_reflectance(WAVELENGTH, [0.0], [1.0, 2.3, 3.88], angle, pol)
        without = pt.stack_reflectance(WAVELENGTH, [], [1.0, 3.88], angle, pol)
        assert torch.allclose(with_layer, without, atol=1e-12)


@pytest.mark.parametrize("n_layers", [1, 2, 3, 5, 8])
def test_arbitrary_layer_count(n_layers):
    """The acceptance criterion: no fixed stack depth anywhere."""
    indices = [1.0] + [1.46 if k % 2 else 2.3 for k in range(n_layers)] + [3.88]
    thicknesses = [80.0 + 20.0 * k for k in range(n_layers)]

    R = pt.stack_reflectance(SPECTRUM, thicknesses, indices, 0.0, "s")

    assert R.shape == SPECTRUM.shape
    assert torch.all(torch.isfinite(R))
    assert torch.all((R >= 0.0) & (R <= 1.0))


def test_s_and_p_coincide_at_normal_incidence():
    """The acceptance criterion, and a real constraint on the p convention.

    At θ=0 there is no plane of incidence to distinguish the polarisations, so
    the two must give identical reflectance. The §4.2 sign convention makes r_p
    differ from r_s by a sign here, which |r|² removes — a convention error in
    the stack product would not survive this.
    """
    indices = [1.0, 1.46, 2.3, 3.88]
    thicknesses = [120.0, 65.0]

    R_s = pt.stack_reflectance(SPECTRUM, thicknesses, indices, 0.0, "s")
    R_p = pt.stack_reflectance(SPECTRUM, thicknesses, indices, 0.0, "p")

    assert torch.allclose(R_s, R_p, atol=1e-13)


def test_s_and_p_differ_away_from_normal_incidence():
    """Guards the previous test: if the two paths were accidentally identical,
    that test would pass for the wrong reason.
    """
    R_s = pt.stack_reflectance(SPECTRUM, [120.0], [1.0, 1.46, 3.88], 0.9, "s")
    R_p = pt.stack_reflectance(SPECTRUM, [120.0], [1.0, 1.46, 3.88], 0.9, "p")

    assert not torch.allclose(R_s, R_p, atol=1e-3)


@pytest.mark.parametrize("pol", POLARISATIONS)
def test_reflectance_is_periodic_in_optical_thickness(pol):
    """The fringe-order ambiguity of §5.2(b), at full-stack level.

    Two thicknesses differing by λ/(2n cosθ) produce identical spectra. This is
    the degeneracy the project exists to characterise: the measurement cannot
    distinguish them, so any estimator claiming to have done so is claiming
    information the physics does not contain.
    """
    n, theta = 1.46, 0.0
    period = WAVELENGTH / (2.0 * n * np.cos(theta))

    for d in (40.0, 137.0, 260.0):
        base = pt.stack_reflectance(WAVELENGTH, [d], [1.0, n, 3.88], theta, pol)
        for k in (1, 2, 5):
            shifted = pt.stack_reflectance(
                WAVELENGTH, [d + k * period], [1.0, n, 3.88], theta, pol
            )
            assert torch.allclose(base, shifted, atol=1e-10)


def test_a_thick_absorbing_layer_hides_the_substrate():
    """A physical degeneracy worth pinning: you cannot see through an opaque film.

    As absorption accumulates, what lies beneath stops affecting the spectrum.
    Two stacks differing only in substrate become numerically identical — an
    unmeasurable parameter, of exactly the kind §10's failure atlas catalogues.
    """
    n_abs = 1.5 + 0.3j
    previous = None

    for d in (100.0, 500.0, 2000.0):
        on_silicon = pt.stack_reflectance(WAVELENGTH, [d], [1.0, n_abs, 3.88], 0.0, "s")
        on_air = pt.stack_reflectance(WAVELENGTH, [d], [1.0, n_abs, 1.0], 0.0, "s")
        difference = (on_silicon - on_air).abs().item()

        if previous is not None:
            assert difference < previous
        previous = difference

    assert previous < 1e-6


@pytest.mark.parametrize("pol", POLARISATIONS)
def test_passive_stacks_never_reflect_more_than_they_receive(pol):
    """Energy conservation across absorbing multilayers — the broadest single
    check that no sign or branch error has crept into the product.
    """
    stacks = [
        ([120.0], [1.0, 1.46, 3.88]),
        ([120.0, 65.0], [1.0, 1.46, 2.3, 3.88]),
        ([80.0, 40.0, 200.0], [1.0, 1.5 + 0.05j, 2.3, 4.0 + 1.5j, 3.88]),
        ([30.0, 300.0], [1.0, 0.15 + 3.5j, 1.46, 1.0]),
    ]
    for thicknesses, indices in stacks:
        for angle in (0.0, 0.5, 1.3):
            R = pt.stack_reflectance(SPECTRUM, thicknesses, indices, angle, pol)
            assert torch.all(R >= 0.0)
            assert torch.all(R <= 1.0 + 1e-12)


def test_stack_produces_fringes_rather_than_a_flat_spectrum():
    """The whole point of the measurement. A forward model returning a smooth
    curve would satisfy every bound above and encode no thickness at all.
    """
    R = pt.stack_reflectance(SPECTRUM, [500.0], [1.0, 1.46, 3.88], 0.0, "s")
    turning_points = ((R[1:-1] - R[:-2]) * (R[2:] - R[1:-1]) < 0).sum().item()

    assert turning_points >= 2, "no interference structure — the layer phase is not acting"
    assert (R.max() - R.min()).item() > 0.1


def test_gradients_flow_through_the_whole_stack():
    """§7.3's reconstruction loss differentiates the forward model end to end.

    ∂R/∂d against finite differences is DTFM-017's check; this asserts only that
    the path exists and is finite.
    """
    d = torch.tensor([120.0, 65.0], dtype=torch.float64, requires_grad=True)
    R = pt.stack_reflectance(SPECTRUM, [d[0], d[1]], [1.0, 1.46, 2.3, 3.88], 0.3, "s")
    R.sum().backward()

    assert torch.all(torch.isfinite(d.grad))
    assert torch.all(d.grad != 0.0)


def test_layer_count_mismatch_is_rejected():
    """Ambient and substrate are semi-infinite and take no thickness. Getting
    that off by one silently would shift every layer's phase.
    """
    with pytest.raises(ValueError, match="expected 1"):
        pt.stack_reflectance(WAVELENGTH, [100.0, 200.0], [1.0, 1.46, 3.88], 0.0, "s")

    with pytest.raises(ValueError, match="at least an ambient"):
        pt.stack_reflectance(WAVELENGTH, [], [1.0], 0.0, "s")


# --- batching, DTFM-013 -----------------------------------------------------
#
# Spec §7.1. The training loop samples a batch of parameter vectors and needs
# one spectrum each. The forward model is called once per training step forever,
# so a python loop here would make the simulator the bottleneck rather than the
# network — and §11 calls moving cost out of the inference loop the entire
# industrial argument for the project.


def test_batched_spectra_match_a_loop_over_single_calls():
    """The acceptance criterion. Broadcasting must not change any answer."""
    wavelengths = SPECTRUM
    thicknesses = torch.tensor([[120.0, 65.0], [300.0, 40.0], [55.0, 210.0], [480.0, 90.0]])
    indices = [1.0, 1.46, 2.3, 3.88]

    for pol in POLARISATIONS:
        for angle in (0.0, 0.7):
            batched = pt.spectra(wavelengths, thicknesses, indices, angle, pol)
            looped = torch.stack([
                pt.stack_reflectance(wavelengths, [row[0], row[1]], indices, angle, pol)
                for row in thicknesses
            ])
            assert batched.shape == (thicknesses.shape[0], wavelengths.numel())
            assert torch.allclose(batched, looped, atol=1e-14)


def test_batching_handles_a_dispersive_index():
    """§4.4 supplies n(λ), so an index may be a spectrum rather than a scalar."""
    wavelengths = SPECTRUM
    n_film = 1.45 + 0.02 * (wavelengths - 600.0) / 200.0
    thicknesses = torch.tensor([[120.0], [340.0]])

    batched = pt.spectra(wavelengths, thicknesses, [1.0, n_film, 3.88])
    looped = torch.stack([
        pt.stack_reflectance(wavelengths, [row[0]], [1.0, n_film, 3.88]) for row in thicknesses
    ])

    assert torch.allclose(batched, looped, atol=1e-14)


def test_batching_handles_a_per_sample_index():
    """The index is itself a fitted parameter (§5.1), so it varies across a batch."""
    wavelengths = SPECTRUM
    thicknesses = torch.tensor([[120.0], [120.0], [120.0]])
    n_film = torch.tensor([1.40, 1.46, 1.52]).unsqueeze(-1).expand(3, wavelengths.numel())

    batched = pt.spectra(wavelengths, thicknesses, [1.0, n_film, 3.88])

    assert batched.shape == (3, wavelengths.numel())
    # Different indices must give different spectra, or the parameter is being ignored.
    assert not torch.allclose(batched[0], batched[1], atol=1e-6)
    assert not torch.allclose(batched[1], batched[2], atol=1e-6)


def test_a_single_film_may_be_passed_unbatched():
    one = pt.spectra(SPECTRUM, torch.tensor([120.0, 65.0]), [1.0, 1.46, 2.3, 3.88])
    assert one.shape == (1, SPECTRUM.numel())


def test_batched_gradients_match_per_sample_gradients():
    """The acceptance criterion's third clause, stated strictly.

    §7.3 backpropagates the reconstruction loss through a batch. It is not
    enough that gradients are finite: each row's gradient must equal what that
    film would have got alone, or the batch axis is leaking between samples.
    """
    thicknesses = torch.tensor([[120.0], [340.0], [55.0]], dtype=torch.float64,
                               requires_grad=True)
    pt.spectra(SPECTRUM, thicknesses, [1.0, 1.46, 3.88]).sum().backward()

    for i, d in enumerate([120.0, 340.0, 55.0]):
        single = torch.tensor(d, dtype=torch.float64, requires_grad=True)
        pt.stack_reflectance(SPECTRUM, [single], [1.0, 1.46, 3.88]).sum().backward()
        assert thicknesses.grad[i, 0].item() == pytest.approx(single.grad.item(), rel=1e-12)


def test_batch_shape_errors_are_rejected():
    """Mismatched axes broadcast into a plausible wrong answer rather than
    raising, so the shapes are checked rather than trusted.
    """
    with pytest.raises(ValueError, match="expected 4"):
        pt.spectra(SPECTRUM, torch.tensor([[120.0, 65.0]]), [1.0, 1.46, 3.88])

    with pytest.raises(ValueError, match="wavelengths must be 1-D"):
        pt.spectra(SPECTRUM.reshape(1, -1), torch.tensor([[120.0]]), [1.0, 1.46, 3.88])

    with pytest.raises(ValueError, match=r"thicknesses must be"):
        pt.spectra(SPECTRUM, torch.zeros(2, 2, 2), [1.0, 1.46, 3.88])
