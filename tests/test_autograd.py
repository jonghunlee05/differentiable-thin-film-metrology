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
