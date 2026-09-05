"""Classical inversion: Levenberg-Marquardt and autograd gradient descent.

Spec §6.
Levenberg-Marquardt by DTFM-030; autograd descent by DTFM-031; multi-start and
the convergence landscape by DTFM-032; fit covariance by DTFM-033.

§6 opens by insisting this exists before any network is trained: "Without it, a
learned model's accuracy number is meaningless." Levenberg-Marquardt has been
inverting spectra since the 1960s and does it well — it is simply slow, and the
project's industrial argument (§11) is about removing that cost, not about
beating the method.

So this module is the thing everything later is measured against, which makes
its *honesty* matter more than its speed. Every fit records what it did:
iterations, wall-clock, whether it converged, and what it converged to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import least_squares

from src import dispersion as dp
from src import generate as gen
from src import tmm_torch as pt

__all__ = ["FitResult", "forward_observable", "fit_least_squares", "wrapped_residual"]

#: Order of the fitted parameter vector, matching :attr:`gen.Batch.targets`.
PARAMETER_NAMES = ("thickness_nm", "cauchy_a", "cauchy_b")


@dataclass
class FitResult:
    """One converged fit, with everything §6 says to record.

    §15 is blunt that a fit without an error bar is a disqualifying instinct in
    this field. The covariance arrives at DTFM-033; what is here already is the
    rest of the record, so that a result can never be quoted without also being
    able to say how it was reached.
    """

    parameters: NDArray[np.float64]
    truth: NDArray[np.float64] | None
    success: bool
    iterations: int
    function_evaluations: int
    wall_clock_s: float
    cost: float
    residual_rms: float
    message: str
    initial: NDArray[np.float64] = field(repr=False)

    @property
    def error(self) -> NDArray[np.float64] | None:
        """Signed error against the truth, when the truth is known."""
        return None if self.truth is None else self.parameters - self.truth

    @property
    def thickness_error_nm(self) -> float | None:
        return None if self.truth is None else float(self.parameters[0] - self.truth[0])


def forward_observable(
    parameters,
    wavelengths_nm: NDArray,
    measurement: gen.Measurement,
    substrate: torch.Tensor,
) -> torch.Tensor:
    """The forward model as the fitter sees it: θ in, measured quantity out.

    Returns reflectance, or ``(Ψ, Δ)`` concatenated, matching whatever
    :class:`gen.Measurement` selects — so the fitter never needs to know which
    observable it is working with.
    """
    thickness, cauchy_a, cauchy_b = parameters
    grid = torch.as_tensor(np.asarray(wavelengths_nm, dtype=float))
    index = dp.cauchy_n((cauchy_a, cauchy_b, 0.0), grid)
    indices = [1.0, index, substrate]
    angle = measurement.angle_rad

    if measurement.observable == "ellipsometry":
        psi, delta = pt.stack_psi_delta(grid, [thickness], indices, angle)
        return torch.cat([psi, delta])
    return pt.stack_reflectance(grid, [thickness], indices, angle, "s")


def wrapped_residual(model: NDArray, observed: NDArray, observable: str) -> NDArray:
    """Model minus observation, with Δ wrapped onto the circle.

    This is the one place ellipsometry needs different arithmetic from
    reflectance, and getting it wrong is expensive rather than merely untidy.

    Δ lives on ``(−π, π]``. A model at ``+179°`` and an observation at ``−179°``
    are two degrees apart, but subtracting gives ``358°`` — a residual 180 times
    too large, at exactly the wavelengths where the fit is most informative.
    Least squares would then bend the whole solution to chase a discontinuity
    that is not there. Wrapping the *residual*, rather than the values, fixes it.
    """
    difference = np.asarray(model, dtype=float) - np.asarray(observed, dtype=float)
    if observable != "ellipsometry":
        return difference

    half = difference.size // 2
    delta_part = (difference[half:] + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([difference[:half], delta_part])


def fit_least_squares(
    observed: NDArray,
    wavelengths_nm: NDArray,
    initial: NDArray | list[float],
    *,
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
    truth: NDArray | None = None,
    max_iterations: int = 200,
) -> FitResult:
    """Recover ``θ = (d, A, B)`` from one measured spectrum, by §6's method.

    Trust-region reflective rather than plain Levenberg-Marquardt, because the
    problem is bounded: thickness cannot be negative and the dispersion
    coefficients have a physical range. §6 allows either and names the reason —
    ``scipy``'s ``lm`` does not accept bounds at all, and an unbounded fit here
    wanders into negative thicknesses where the forward model is meaningless.

    Bounds default to the prior's support, which keeps the fit inside the region
    the training data will also come from. That is deliberate: §10 compares the
    two estimators, and giving the classical fit a wider search space than the
    network's prior would make the comparison unfair in the classical method's
    favour.
    """
    measurement = measurement or gen.Measurement()
    prior = prior or gen.Prior()
    observed = np.asarray(observed, dtype=float)

    substrate_n, substrate_k = dp.load_nk(prior.substrate, wavelengths_nm)
    substrate = torch.tensor(substrate_n + 1j * substrate_k)

    lower = np.array([prior.thickness_nm[0], prior.cauchy_a[0], prior.cauchy_b[0]])
    upper = np.array([prior.thickness_nm[1], prior.cauchy_a[1], prior.cauchy_b[1]])
    start = np.clip(np.asarray(initial, dtype=float), lower, upper)

    def residual(parameters: NDArray) -> NDArray:
        with torch.no_grad():
            model = forward_observable(
                parameters, wavelengths_nm, measurement, substrate
            ).numpy()
        return wrapped_residual(model, observed, measurement.observable)

    started = time.perf_counter()
    solution = least_squares(
        residual, start, bounds=(lower, upper), method="trf", max_nfev=max_iterations
    )
    elapsed = time.perf_counter() - started

    return FitResult(
        parameters=np.asarray(solution.x, dtype=float),
        truth=None if truth is None else np.asarray(truth, dtype=float),
        success=bool(solution.success),
        iterations=int(getattr(solution, "njev", 0) or 0),
        function_evaluations=int(solution.nfev),
        wall_clock_s=elapsed,
        cost=float(solution.cost),
        residual_rms=float(np.sqrt(np.mean(solution.fun**2))),
        message=str(solution.message),
        initial=start,
    )
