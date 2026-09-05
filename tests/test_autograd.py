"""Autograd derivatives against central finite differences.

Spec §5.3, §12 Week 0.
Single-interface derivatives by DTFM-009; the same check through a multilayer
stack, including ∂R/∂d, by DTFM-017.

Why this matters beyond week 0: §5.3 builds the Jacobian `J[i,k] = ∂f(λ_i)/∂θ_k`
from autograd rather than finite differences, and everything downstream — Fisher
information, the Cramér–Rao bound, the correlation between thickness and index —
is computed from that `J`. A wrong derivative would not raise; it would produce
a plausible, wrong error bar. This is the test that makes autograd trustworthy
enough to build those on.

Note on scope: the ticket's acceptance criteria named ∂R/∂n *and* ∂R/∂d. There
is no thickness in the model yet — `d` enters only through the layer phase
δ = (2π/λ)·ñ·d·cosθ of §4.3, which arrives with the transfer matrices. ∂R/∂d is
therefore checked in DTFM-017, against the full stack, and the board records
that. Asserting it here would mean inventing a thickness the forward model does
not have.
"""

import numpy as np
import pytest
import torch

from src import tmm_torch as pt

# Away from θ=0 and grazing, where several derivatives vanish or diverge and a
# comparison would be uninformative. Both index orderings included.
#
# All angles are kept *below* the critical angle where n_i > n_j. Past it the
# interface is in total internal reflection, R ≡ 1 identically, and every
# derivative is exactly zero — a relative comparison of two numbers that are
# both ~1e-16 tests nothing. That regime is asserted separately below.
CASES = [
    (1.0, 1.46, 0.30),
    (1.0, 1.46, 0.90),
    (1.0, 3.88, 0.60),
    (1.46, 1.0, 0.25),  # θ_c = 0.744
    (1.33, 1.46, 0.75),
    (2.00, 1.38, 0.50),  # θ_c = 0.761
]

# Step for float64 central differences. The truncation error falls as h² while
# round-off grows as ~eps/h, so the total is minimised near eps^(1/3) ≈ 6e-6.
H = 1e-6


def _autograd(f, x0: float) -> float:
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    f(x).backward()
    return x.grad.item()


def _central_difference(f, x0: float, h: float = H) -> float:
    with torch.no_grad():
        hi = f(torch.tensor(x0 + h, dtype=torch.float64)).item()
        lo = f(torch.tensor(x0 - h, dtype=torch.float64)).item()
    return (hi - lo) / (2.0 * h)


@pytest.mark.parametrize(("n_i", "n_j", "theta"), CASES)
def test_dR_dn_matches_finite_differences(n_i, n_j, theta):
    """∂R/∂n for both polarisations — the derivative §5.3's Jacobian is built from."""
    for pol in (0, 1):
        def f(n, _pol=pol, _n_i=n_i, _theta=theta):
            return pt.reflectance(_n_i, n, _theta)[_pol]

        assert _autograd(f, n_j) == pytest.approx(_central_difference(f, n_j), rel=1e-7)


@pytest.mark.parametrize(("n_i", "n_j", "theta"), CASES)
def test_dR_dtheta_matches_finite_differences(n_i, n_j, theta):
    for pol in (0, 1):
        def f(t, _pol=pol, _n_i=n_i, _n_j=n_j):
            return pt.reflectance(_n_i, _n_j, t)[_pol]

        assert _autograd(f, theta) == pytest.approx(_central_difference(f, theta), rel=1e-7)


@pytest.mark.parametrize(("n_i", "n_j", "theta"), CASES)
def test_gradient_wrt_incident_index_matches_finite_differences(n_i, n_j, theta):
    """The incident medium is a parameter too, and is fitted in practice."""
    for pol in (0, 1):
        def f(n, _pol=pol, _n_j=n_j, _theta=theta):
            return pt.reflectance(n, _n_j, _theta)[_pol]

        assert _autograd(f, n_i) == pytest.approx(_central_difference(f, n_i), rel=1e-7)


@pytest.mark.parametrize(("n_i", "n_j", "theta"), CASES)
def test_finite_difference_error_falls_as_h_squared(n_i, n_j, theta):
    """Agreement at one step size can be luck; the convergence order cannot.

    Central differences are O(h²), so halving h should cut the discrepancy by
    about four. A derivative that were merely close would not show this slope.
    """
    def f(n, _n_i=n_i, _theta=theta):
        return pt.reflectance(_n_i, n, _theta)[0]

    exact = _autograd(f, n_j)
    coarse = abs(_central_difference(f, n_j, 1e-3) - exact)
    fine = abs(_central_difference(f, n_j, 5e-4) - exact)

    # Ratio ≈ 4 for a second-order scheme. Loose bounds: the point is the slope,
    # not a precise constant, and a first-order error would give ≈ 2.
    assert 3.0 < coarse / fine < 5.0


@pytest.mark.parametrize(("n_i", "n_j"), [(1.0, 1.46), (1.0, 3.88), (1.46, 1.0)])
def test_dRp_dtheta_vanishes_at_brewster(n_i, n_j):
    """A gradient check with physics attached, not just numerics.

    R_p is minimised at θ_B, so its derivative there is zero. Autograd has to
    reproduce that, and it is a null — unreachable by a scale error in the
    derivative.
    """
    def f(t, _n_i=n_i, _n_j=n_j):
        return pt.reflectance(_n_i, _n_j, t)[1]

    theta_b = float(np.arctan2(n_j, n_i))
    assert _autograd(f, theta_b) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("n_i", "n_j"), [(1.46, 1.0), (2.00, 1.38)])
def test_derivatives_vanish_under_total_internal_reflection(n_i, n_j):
    """Past θ_c the interface reflects everything at every angle, so R ≡ 1 and
    ∂R/∂θ and ∂R/∂n are identically zero.

    Worth asserting rather than merely excluding from the comparison cases: a
    formulation that leaked a nonzero gradient here would be reporting
    sensitivity to a parameter the measurement genuinely cannot see, which is
    exactly the kind of false information §5.3's covariance would inherit.
    """
    theta_c = float(np.arcsin(n_j / n_i))

    for theta in np.linspace(theta_c + 0.02, np.pi / 2 - 1e-6, 5):
        for pol in (0, 1):
            def f_theta(t, _pol=pol, _n_i=n_i, _n_j=n_j):
                return pt.reflectance(_n_i, _n_j, t)[_pol]

            def f_n(n, _pol=pol, _n_i=n_i, _theta=float(theta)):
                return pt.reflectance(_n_i, n, _theta)[_pol]

            assert _autograd(f_theta, float(theta)) == pytest.approx(0.0, abs=1e-12)
            assert _autograd(f_n, n_j) == pytest.approx(0.0, abs=1e-12)


def test_second_derivative_is_available():
    """§5.3's Fisher information needs JᵀJ; a Hessian is one differentiation
    further and is cheap to confirm now rather than discover missing at week 7.
    """
    n = torch.tensor(1.46, dtype=torch.float64, requires_grad=True)
    R_s, _ = pt.reflectance(1.0, n, 0.5)
    (first,) = torch.autograd.grad(R_s, n, create_graph=True)
    (second,) = torch.autograd.grad(first, n)

    assert torch.isfinite(second)
    assert second.item() != 0.0


def test_batched_jacobian_matches_per_element_gradients():
    """The Jacobian §5.3 needs is over a spectrum, so the batched path is the
    one that will actually be used. It must agree with scalar differentiation.
    """
    thetas = np.linspace(0.2, 1.3, 12)

    batched = torch.tensor(thetas, dtype=torch.float64, requires_grad=True)
    pt.reflectance(1.0, 1.46, batched)[0].sum().backward()

    for i, theta in enumerate(thetas):
        def f(t):
            return pt.reflectance(1.0, 1.46, t)[0]

        assert batched.grad[i].item() == pytest.approx(_autograd(f, float(theta)), rel=1e-12)


# --- gradients through the full stack, DTFM-017 -----------------------------
#
# Spec §5.3. This completes the check DTFM-009 could only half-do: thickness did
# not exist then, so ∂R/∂d was deferred here.
#
# It matters more than a numerical formality. §5.3 builds J[i,k] = ∂f(λ_i)/∂θ_k
# from autograd, and the Fisher information, the Cramér–Rao bound and the
# thickness–index correlation of §5.2(a) are all computed from that J. A wrong
# derivative would not raise — it would produce a confident, wrong error bar,
# which §15 calls a disqualifying instinct in this field.

STACKS_FOR_GRADIENTS = [
    ([120.0], [1.0, 1.46, 3.88]),
    ([500.0], [1.0, 1.46, 3.88]),
    ([120.0, 65.0], [1.0, 1.38, 2.30, 1.52]),
    ([80.0, 40.0, 200.0], [1.0, 1.5 + 0.05j, 2.3, 4.0 + 1.5j, 3.88]),
]
GRADIENT_WAVELENGTH = 633.0


def _scalar_R(thicknesses, indices, wavelength=GRADIENT_WAVELENGTH, theta=0.2, pol="s"):
    return pt.stack_reflectance(torch.tensor(wavelength), thicknesses, indices, theta, pol)


@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize(("thicknesses", "indices"), STACKS_FOR_GRADIENTS)
def test_dR_dd_matches_finite_differences(thicknesses, indices, pol):
    """∂R/∂d — the derivative DTFM-009 could not check, for every layer."""
    for k in range(len(thicknesses)):
        def f(d, _k=k, _t=thicknesses, _n=indices, _p=pol):
            layers = [*_t]
            layers[_k] = d
            return _scalar_R(layers, _n, pol=_p)

        assert _autograd(f, thicknesses[k]) == pytest.approx(
            _central_difference(f, thicknesses[k], 1e-4), rel=1e-6
        )


@pytest.mark.parametrize(("thicknesses", "indices"), STACKS_FOR_GRADIENTS)
def test_dR_dd_error_falls_as_h_squared(thicknesses, indices):
    """Agreement at one step size can be luck; a second-order slope cannot."""
    def f(d, _t=thicknesses, _n=indices):
        return _scalar_R([d, *_t[1:]], _n)

    exact = _autograd(f, thicknesses[0])
    coarse = abs(_central_difference(f, thicknesses[0], 2e-2) - exact)
    fine = abs(_central_difference(f, thicknesses[0], 1e-2) - exact)

    assert 3.0 < coarse / fine < 5.0


def test_dR_dn_of_a_buried_layer_matches_finite_differences():
    """The index of an interior layer is a fitted parameter too (§5.1)."""
    thicknesses = [120.0, 65.0]

    def f(n):
        return _scalar_R(thicknesses, [1.0, 1.38, n, 1.52])

    assert _autograd(f, 2.30) == pytest.approx(_central_difference(f, 2.30, 1e-6), rel=1e-7)


def test_gradients_flow_to_cauchy_dispersion_coefficients():
    """The AC names dispersion parameters, which do not exist yet.

    §4.4's Cauchy model n(λ) = A + B/λ² arrives at DTFM-020. Its coefficients
    reach R only through a wavelength-dependent index, so the chain rule path is
    what actually needs checking — and it can be checked now by writing the two
    terms inline. When `src/dispersion.py` lands, it plugs into this same path.
    """
    wavelengths = torch.linspace(450.0, 750.0, 25, dtype=torch.float64)

    def R_of(A, B):
        n = A + B / wavelengths**2
        return pt.stack_reflectance(wavelengths, [420.0], [1.0, n, 3.88], 0.2, "s").sum()

    A0, B0 = 1.45, 3.5e3
    for i, x0 in enumerate((A0, B0)):
        p = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        R_of(*(p, torch.tensor(B0, dtype=torch.float64)) if i == 0 else
              (torch.tensor(A0, dtype=torch.float64), p)).backward()

        h = 1e-6 * max(abs(x0), 1.0)
        with torch.no_grad():
            args_hi = (x0 + h, B0) if i == 0 else (A0, B0 + h)
            args_lo = (x0 - h, B0) if i == 0 else (A0, B0 - h)
            fd = (R_of(*map(lambda v: torch.tensor(v, dtype=torch.float64), args_hi)).item()
                  - R_of(*map(lambda v: torch.tensor(v, dtype=torch.float64), args_lo)).item()
                  ) / (2 * h)

        assert p.grad.item() == pytest.approx(fd, rel=1e-6)


def test_the_spectral_jacobian_matches_finite_differences_column_by_column():
    """J[i,k] = ∂R(λ_i)/∂θ_k — the object §5.3 actually consumes.

    Checked as a whole rather than one derivative at a time: the Jacobian is
    what Fisher information is built from, and an error in a single column would
    corrupt one parameter's error bar while leaving the others plausible.
    """
    wavelengths = torch.linspace(450.0, 750.0, 24, dtype=torch.float64)
    theta_true = torch.tensor([320.0, 1.46], dtype=torch.float64)

    def spectrum(theta):
        return pt.stack_reflectance(wavelengths, [theta[0]], [1.0, theta[1], 3.88], 0.2, "s")

    J = torch.autograd.functional.jacobian(spectrum, theta_true)
    assert J.shape == (wavelengths.numel(), 2)
    assert torch.all(torch.isfinite(J))

    for k, h in enumerate((1e-4, 1e-7)):
        step = torch.zeros_like(theta_true)
        step[k] = h
        with torch.no_grad():
            fd = (spectrum(theta_true + step) - spectrum(theta_true - step)) / (2 * h)
        assert torch.allclose(J[:, k], fd, rtol=1e-5, atol=1e-9)


def test_the_jacobian_gives_a_usable_fisher_matrix():
    """A forward-looking check, not a numerical one.

    §5.3 forms F = JᵀJ/σ² and inverts it. If J were rank-deficient — because two
    parameters entered identically, say — the inversion would fail or explode,
    and that would be discovered at week 7 rather than here.
    """
    wavelengths = torch.linspace(450.0, 750.0, 200, dtype=torch.float64)
    theta = torch.tensor([320.0, 1.46], dtype=torch.float64)

    def spectrum(t):
        return pt.stack_reflectance(wavelengths, [t[0]], [1.0, t[1], 3.88], 0.2, "s")

    J = torch.autograd.functional.jacobian(spectrum, theta)
    fisher = J.T @ J

    assert torch.linalg.matrix_rank(fisher).item() == 2
    assert torch.all(torch.isfinite(torch.linalg.inv(fisher)))


def test_dR_dd_vanishes_at_the_quarter_wave_null():
    """A gradient check with physics attached.

    An ideal AR coating drives R to zero, and R ≥ 0, so the design thickness is
    a minimum and the derivative there must vanish. A null again — unreachable
    by a scale error in the derivative.
    """
    n_sub = 3.88
    n_film = float(np.sqrt(n_sub))
    d_design = 550.0 / (4.0 * n_film)

    def f(d):
        return pt.stack_reflectance(torch.tensor(550.0), [d], [1.0, n_film, n_sub], 0.0, "s")

    assert _autograd(f, d_design) == pytest.approx(0.0, abs=1e-12)


def test_dR_dd_is_periodic_like_R_itself():
    """§5.2(b) again: if R repeats every λ/(2n), so must its slope.

    Worth asserting on the derivative specifically — it is the derivative, not R,
    that the fitting of §6 follows downhill, so the ambiguity is inherited by
    every gradient-based estimator in the project.
    """
    n, lam = 1.46, 550.0
    period = lam / (2.0 * n)

    def f(d):
        return pt.stack_reflectance(torch.tensor(lam), [d], [1.0, n, 3.88], 0.0, "s")

    base = _autograd(f, 137.0)
    for k in (1, 2, 5):
        assert _autograd(f, 137.0 + k * period) == pytest.approx(base, rel=1e-9)
