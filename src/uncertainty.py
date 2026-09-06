"""Identifiability: Fisher information, the Cramér-Rao bound, and ρ(n, d).

Spec §5.3.
Implemented by DTFM-034.

The question this module exists to answer is not "how precise was that fit". It
is the one before it:

    When a measurement cannot pin down the thickness, is that because the
    algorithm is weak, or because the light does not carry the information?

Those demand opposite responses. The first means write better code. The second
means no algorithm will ever do better and the *instrument* is what has to
change. §5.3 puts it plainly: "If your estimator sits near the bound, the
algorithm is fine and the *measurement* is the limit — a conclusion worth
stating out loud."

Relationship to DTFM-033
------------------------
``baseline.covariance_from_jacobian`` computes ``σ²(JᵀJ)⁻¹`` at the point a fit
*stopped*. The Cramér-Rao bound is the same algebra at the point the truth
actually *is*. The formula is shared; what differs is what it means.

- The fit covariance is a property of one estimate. It answers "how tightly did
  this data pin this answer, given the basin I landed in".
- The bound is a property of the **measurement design** — wavelengths, angle,
  observable, noise — and holds for *any* unbiased estimator anyone might ever
  write, including a neural network. It exists before an estimator does.

So this module deliberately does not import a fitter. It takes true parameters
and a measurement, and reports what that measurement can and cannot know.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from src import baseline as bl
from src import dispersion as dp
from src import generate as gen
from src import tmm_torch as pt

__all__ = [
    "Identifiability",
    "bandwidth_matrix",
    "observable_with_bandwidth",
    "cramer_rao_bound",
    "efficiency",
    "fisher_information",
    "identifiability",
    "sweep_thickness",
]

#: Parameter order, matching :data:`baseline.PARAMETER_NAMES`.
PARAMETER_NAMES = bl.PARAMETER_NAMES


def bandwidth_matrix(wavelengths_nm: NDArray, fwhm_nm: float) -> torch.Tensor | None:
    """The spectrometer's slit function as a matrix, for differentiating through.

    ``noise.apply_spectrometer_bandwidth`` builds exactly these weights and
    applies them once; the bound needs them inside an autograd graph, so they are
    built once here and reused across the sweep. Same Gaussian, same row
    normalisation — ``test_uncertainty.py`` asserts the two agree.

    Returns ``None`` for zero width, which the caller reads as "no smoothing"
    rather than as an identity multiply.
    """
    if fwhm_nm < 0.0:
        raise ValueError(f"FWHM must be non-negative, got {fwhm_nm}")
    if fwhm_nm == 0.0:
        return None

    grid = np.asarray(wavelengths_nm, dtype=float)
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    kernel = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / sigma) ** 2)
    return torch.as_tensor(kernel / kernel.sum(axis=1, keepdims=True))


def observable_with_bandwidth(
    parameters,
    wavelengths_nm: NDArray,
    measurement: gen.Measurement,
    substrate: torch.Tensor,
    smoothing: torch.Tensor | None,
) -> torch.Tensor:
    """The measured quantity as a real instrument records it, differentiably.

    **Smoothing is applied to the complex ratio ρ = r_p/r_s, not to Ψ and Δ.**
    That is not a refinement; it is the only correct choice, and
    ``generate.py`` already makes it. A spectroscopic ellipsometer averages the
    polarisation *state* across its slit width. Δ is an angle on ``(−π, π]``, so
    averaging it directly across the wrap turns +179° and −179° into 0° — a value
    no film produces — at exactly the wavelengths where a thick film's phase is
    changing fastest.

    Without smoothing this is :func:`baseline.forward_observable` unchanged, so
    the two paths cannot drift apart for the zero-bandwidth case.
    """
    if smoothing is None:
        return bl.forward_observable(parameters, wavelengths_nm, measurement, substrate)

    thickness, cauchy_a, cauchy_b = parameters
    grid = torch.as_tensor(np.asarray(wavelengths_nm, dtype=float))
    index = dp.cauchy_n((cauchy_a, cauchy_b, 0.0), grid)
    indices = [1.0, index, substrate]

    if measurement.observable != "ellipsometry":
        clean = pt.stack_reflectance(grid, [thickness], indices, measurement.angle_rad, "s")
        return smoothing @ clean

    ratio = pt.stack_rho(grid, [thickness], indices, measurement.angle_rad)
    real = smoothing @ ratio.real
    imaginary = smoothing @ ratio.imag
    psi = torch.atan(torch.sqrt(real**2 + imaginary**2))
    delta = torch.atan2(imaginary, real)
    return torch.cat([psi, delta])


def fisher_information(jacobian: NDArray, sigma: float) -> NDArray[np.float64]:
    """``F = JᵀJ / σ²`` — §5.3's Fisher information matrix.

    For Gaussian noise of known width this is the whole story: the curvature of
    the log-likelihood, and therefore how sharply the data distinguishes one
    parameter vector from a neighbouring one. Large ``F`` means the measurement
    reacts strongly to a change in the parameters, which is what makes them
    recoverable.
    """
    jacobian = np.atleast_2d(np.asarray(jacobian, dtype=float))
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return jacobian.T @ jacobian / float(sigma) ** 2


def cramer_rao_bound(
    jacobian: NDArray, sigma: float, *, rcond: float = 1e-12
) -> tuple[NDArray[np.float64], int]:
    """``C = F⁻¹`` — the floor on the covariance of any unbiased estimator.

    Returns the covariance and the numerical rank of ``F``.

    **Not a pseudo-inverse, and the difference is not academic.** The obvious
    implementation is ``np.linalg.pinv(F, rcond=...)``, which was what this was.
    It is wrong here in a way that produces a confident nonsense number rather
    than an error.

    ``pinv`` handles a near-singular direction by *dropping* it, which sets that
    direction's variance to **zero**. So at a thickness where ``F`` happens to
    tip past the cutoff, the bound comes back as ``3.5e-10 nm`` — reading as a
    measurement of extraordinary precision at exactly the point where the
    measurement has stopped being able to distinguish the parameters at all.
    Measured on the real sweep, at 1844 nm:

        1836 nm   rank 3   CRB 3.2e-02 nm
        1844 nm   rank 2   CRB 3.5e-10 nm     <- 8 orders too good
        1860 nm   rank 3   CRB 3.4e-02 nm

    The physically correct answer for a rank-deficient ``F`` is an **infinite**
    bound: that combination of parameters is not identifiable and no estimator
    can pin it down. Zero is the opposite of that.

    So the eigenvalues are floored rather than truncated. A direction the data
    barely constrains gets a huge variance, which errs towards saying "this
    measurement cannot tell you" — conservative, and in the right direction.
    ``F`` is symmetric positive semi-definite, so ``eigh`` is both cheaper and
    better conditioned than a general inverse.
    """
    fisher = fisher_information(jacobian, sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)

    largest = float(eigenvalues.max())
    floor = largest * rcond
    rank = int(np.sum(eigenvalues > floor))
    covariance = (eigenvectors * (1.0 / np.maximum(eigenvalues, floor))) @ eigenvectors.T
    return covariance, rank


@dataclass
class Identifiability:
    """What a measurement design can and cannot determine, before any fitting.

    Reported together on purpose. A bound quoted without its correlation hides
    the most common failure in this field — two parameters each nominally
    well-determined, but only in combination.
    """

    parameters: NDArray[np.float64]
    fisher: NDArray[np.float64] = field(repr=False)
    covariance: NDArray[np.float64] = field(repr=False)
    sigma: float
    rank: int = 3
    names: tuple[str, ...] = PARAMETER_NAMES

    @property
    def identifiable(self) -> bool:
        """Whether ``F`` has full rank — i.e. whether every parameter is
        separately determinable at all.

        ``False`` means at least one direction in parameter space is invisible to
        this measurement, and the corresponding bound is a floored placeholder
        rather than a number to quote. §5.2's degeneracies are precisely the
        conditions that drive this false.
        """
        return self.rank == len(self.names)

    @property
    def bounds(self) -> NDArray[np.float64]:
        """One-sigma Cramér-Rao floor on each parameter."""
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def thickness_bound_nm(self) -> float:
        return float(self.bounds[0])

    @property
    def relative_thickness_bound(self) -> float:
        """σ_d / d — the fair way to compare a 20 nm film against a 2000 nm one.

        §7.1 chose a log-uniform prior on exactly this reasoning: an absolute
        error in nanometres flatters thick films, because 1 nm of 1500 is a
        different achievement from 1 nm of 30.
        """
        return float(self.bounds[0] / self.parameters[0])

    @property
    def correlation(self) -> NDArray[np.float64]:
        """§5.3's ``ρ_jk = C_jk / √(C_jj C_kk)``."""
        scale = np.sqrt(np.clip(np.diag(self.covariance), 1e-300, None))
        return self.covariance / np.outer(scale, scale)

    @property
    def thickness_index_correlation(self) -> float:
        """ρ(d, A) — the number §5.3 singles out.

        "ρ between thickness and index will frequently exceed 0.99. Reporting
        that alongside the fitted value is what separates a metrologist from
        someone who called ``curve_fit``."
        """
        return float(self.correlation[0, 1])

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.fisher))

    def summary(self) -> str:
        widest = max(len(n) for n in self.names)
        lines = [f"  {'parameter':<{widest}}  {'value':>12}  {'CRB (1 sigma)':>14}"]
        for name, value, bound in zip(self.names, self.parameters, self.bounds, strict=True):
            lines.append(f"  {name:<{widest}}  {value:12.6g}  {bound:14.6g}")
        lines.append(f"\n  rho(d, index) = {self.thickness_index_correlation:+.4f}")
        lines.append(f"  cond(F)       = {self.condition_number:.3e}")
        return "\n".join(lines)


def identifiability(
    parameters: NDArray | list[float],
    wavelengths_nm: NDArray,
    *,
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
    substrate: torch.Tensor | None = None,
    sigma: float | None = None,
    bandwidth_fwhm_nm: float = 0.0,
    smoothing: torch.Tensor | None = None,
) -> Identifiability:
    """Everything §5.3 asks for, at one point in parameter space.

    ``bandwidth_fwhm_nm`` **defaults to zero, and that is a known inconsistency
    rather than a choice.** Every other module applies the 3 nm slit function from
    ``configs/default.yaml``, so this bound currently describes a sharper
    instrument than the one being simulated.

    The machinery to include it is here and tested — :func:`bandwidth_matrix`
    reproduces ``noise.apply_spectrometer_bandwidth`` exactly, and
    :func:`observable_with_bandwidth` smooths the complex ratio the way
    ``generate.py`` does. It is not switched on because turning it on makes the
    predicted precision **better**, by 45% at 420 nm, and blurring a measurement
    must not improve it. Six explanations were tested and none survived: it is
    not ρ being pulled towards zero (|ρ| moves by 0.08%), and the observable
    itself barely changes while its derivative w.r.t. thickness rises 1.6x.

    Enabling this before understanding it would ship a number that looks like
    physics and is not. See ``Implementation-Notes.md`` §25 and DTFM-069.

    ``sigma`` defaults to the measurement's own instrument figure — the
    ellipsometer noise for Ψ/Δ, and 1e-3 in reflectance, which is 0.1% and about
    what a good reflectometer achieves. Both are stated rather than fitted,
    because a *bound* has to come from the instrument: estimating σ from
    residuals would make the floor depend on the estimator it is supposed to
    judge.
    """
    measurement = measurement or gen.Measurement()
    prior = prior or gen.Prior()
    parameters = np.asarray(parameters, dtype=float)

    if substrate is None:
        n, k = dp.load_nk(prior.substrate, wavelengths_nm)
        substrate = torch.tensor(n + 1j * k)
    if sigma is None:
        sigma = (
            measurement.ellipsometer_sigma_rad if measurement.observable == "ellipsometry" else 1e-3
        )

    if smoothing is None and bandwidth_fwhm_nm:
        smoothing = bandwidth_matrix(wavelengths_nm, bandwidth_fwhm_nm)

    theta = torch.as_tensor(parameters).clone()
    jacobian = torch.autograd.functional.jacobian(
        lambda values: observable_with_bandwidth(
            values, wavelengths_nm, measurement, substrate, smoothing
        ),
        theta,
    ).numpy()
    covariance, rank = cramer_rao_bound(jacobian, sigma)
    return Identifiability(
        parameters=parameters,
        fisher=fisher_information(jacobian, sigma),
        covariance=covariance,
        sigma=float(sigma),
        rank=rank,
    )


def efficiency(observed_sigma: float, bound_sigma: float) -> float:
    """How close an estimator gets to the floor, as a variance ratio in ``(0, 1]``.

    ``1.0`` means the estimator extracts everything the measurement contains —
    no algorithm can improve on it, and the only way forward is a better
    instrument. That is §5.3's "conclusion worth stating out loud", and it is
    the number that makes a later claim about the network meaningful: a network
    at 0.9 efficiency is near-optimal, and the same absolute error at 0.05
    efficiency means most of the available information is being thrown away.

    Values slightly above 1 are possible and are not a triumph — the bound
    constrains *unbiased* estimators, and a bounded fit clipped at the edge of
    its prior is biased. Values far above 1 mean the comparison is invalid, not
    that physics was beaten.
    """
    if bound_sigma <= 0.0 or observed_sigma <= 0.0:
        raise ValueError("both sigmas must be positive")
    return float((bound_sigma / observed_sigma) ** 2)


def sweep_thickness(
    thicknesses_nm: NDArray | list[float],
    wavelengths_nm: NDArray,
    *,
    cauchy: tuple[float, float] = (1.46, 0.004),
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
    sigma: float | None = None,
    bandwidth_fwhm_nm: float = 0.0,
) -> dict[str, NDArray[np.float64]]:
    """The bound across the prior's thickness range, in one pass.

    A single bound is a number; the sweep is where the physics shows. §5.2's
    three degeneracies each predict a *shape* here rather than a value, and a
    curve either has that shape or it does not.

    The substrate is loaded once and reused, which matters: ``dp.load_nk``
    interpolates tabulated data, and re-doing it per point would put a slow
    lookup inside the loop for no benefit.
    """
    measurement = measurement or gen.Measurement()
    prior = prior or gen.Prior()
    thicknesses_nm = np.asarray(thicknesses_nm, dtype=float)

    n, k = dp.load_nk(prior.substrate, wavelengths_nm)
    substrate = torch.tensor(n + 1j * k)
    smoothing = bandwidth_matrix(wavelengths_nm, bandwidth_fwhm_nm)

    absolute = np.empty(thicknesses_nm.size)
    relative = np.empty(thicknesses_nm.size)
    rho = np.empty(thicknesses_nm.size)
    condition = np.empty(thicknesses_nm.size)
    rank = np.empty(thicknesses_nm.size, dtype=int)

    for i, thickness in enumerate(thicknesses_nm):
        result = identifiability(
            [thickness, cauchy[0], cauchy[1]],
            wavelengths_nm,
            measurement=measurement,
            prior=prior,
            substrate=substrate,
            sigma=sigma,
            smoothing=smoothing,
        )
        absolute[i] = result.thickness_bound_nm
        relative[i] = result.relative_thickness_bound
        rho[i] = result.thickness_index_correlation
        condition[i] = result.condition_number
        rank[i] = result.rank

    return {
        "thickness_nm": thicknesses_nm,
        "bound_nm": absolute,
        "relative_bound": relative,
        "correlation": rho,
        "condition": condition,
        "rank": rank,
    }
