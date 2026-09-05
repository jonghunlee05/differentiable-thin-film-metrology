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

__all__ = [
    "FitResult",
    "fit_autograd",
    "fit_least_squares",
    "forward_observable",
    "torch_residual",
    "wrapped_residual",
]

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
            model = forward_observable(parameters, wavelengths_nm, measurement, substrate).numpy()
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


def torch_residual(model: torch.Tensor, observed: torch.Tensor, observable: str) -> torch.Tensor:
    """:func:`wrapped_residual`, in torch, differentiably.

    The same arithmetic exists twice because the two fitters need different
    things: ``scipy`` wants a numpy array and cannot accept a tensor, while
    autograd needs the graph and cannot accept an array. Duplicated logic drifts,
    so ``test_baseline.py`` asserts the two agree to machine precision on the
    same inputs — including across the ±π seam, which is the only place they
    could plausibly differ.

    ``torch.remainder`` follows Python's sign convention rather than C's, so it
    matches numpy's ``%``, and it is differentiable with unit gradient away from
    the seam — which is what wrapping a residual needs.
    """
    difference = model - observed
    if observable != "ellipsometry":
        return difference

    half = difference.numel() // 2
    delta_part = torch.remainder(difference[half:] + np.pi, 2 * np.pi) - np.pi
    return torch.cat([difference[:half], delta_part])


def fit_autograd(
    observed: NDArray,
    wavelengths_nm: NDArray,
    initial: NDArray | list[float],
    *,
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
    truth: NDArray | None = None,
    optimiser: str = "lbfgs",
    max_iterations: int = 200,
    learning_rate: float | None = None,
    learning_rate_decay: float = 0.999,
    tolerance: float = 1e-14,
) -> FitResult:
    """Recover ``θ = (d, A, B)`` by descending through the simulator itself.

    §6's second baseline. Where :func:`fit_least_squares` treats the forward
    model as a black box and rebuilds its Jacobian by nudging each parameter in
    turn, this asks the model for the exact gradient it already knows how to
    give — one backward pass for all three derivatives, at any accuracy the
    forward model has.

    Two decisions carry the fit.

    **Descent happens in normalised coordinates.** ``θ`` spans 20-2000 nm in
    thickness and 0.002-0.012 in Cauchy ``B``, so ``∂L/∂d`` and ``∂L/∂B`` are
    orders of magnitude apart and a single step size is either negligible for one
    or divergent for another. Mapping the prior's box onto the unit cube makes
    one step size meaningful for all three. This is not cosmetic: it is why the
    fit converges at all, and ``test_baseline.py`` measures the spread it exists
    to remove.

    **Bounds are enforced by projection** back into the unit cube after each
    step. That cube is the prior's support, and the same box
    :func:`fit_least_squares` is given — §10 compares the two estimators, so
    neither may search a region the other cannot.

    ``optimiser`` selects which descent, and the choice is the interesting part:

    ``"lbfgs"``
        Quasi-Newton with a strong-Wolfe line search. This is the fair
        head-to-head against Levenberg-Marquardt, because both build curvature
        and both choose their own step length. It reaches the same answers to
        about a nanometre in a billion.
    ``"adam"``
        The first-order optimiser §7.3 will train the network with. Included
        because that comparison is the one that matters later — and because it
        does *not* match: Adam's step size is set by its learning rate rather
        than by the gradient, so it settles at an accuracy of roughly its own
        step and no further. Decaying the rate lowers that floor without
        removing it.
    """
    measurement = measurement or gen.Measurement()
    prior = prior or gen.Prior()
    if optimiser not in ("lbfgs", "adam"):
        raise ValueError(f"optimiser must be 'lbfgs' or 'adam', got {optimiser!r}")

    substrate_n, substrate_k = dp.load_nk(prior.substrate, wavelengths_nm)
    substrate = torch.tensor(substrate_n + 1j * substrate_k)
    observed_t = torch.as_tensor(np.asarray(observed, dtype=float))

    lower = np.array([prior.thickness_nm[0], prior.cauchy_a[0], prior.cauchy_b[0]])
    upper = np.array([prior.thickness_nm[1], prior.cauchy_a[1], prior.cauchy_b[1]])
    span = torch.as_tensor(upper - lower)
    floor = torch.as_tensor(lower)

    start = np.clip(np.asarray(initial, dtype=float), lower, upper)
    unit = torch.as_tensor((start - lower) / (upper - lower)).clone().requires_grad_(True)

    def loss_and_grad() -> torch.Tensor:
        """One forward pass, one backward pass, with the box respected."""
        with torch.no_grad():
            unit.clamp_(0.0, 1.0)
        if unit.grad is not None:
            unit.grad = None
        model = forward_observable(floor + unit * span, wavelengths_nm, measurement, substrate)
        loss = 0.5 * (torch_residual(model, observed_t, measurement.observable) ** 2).sum()
        loss.backward()
        return loss

    evaluations = 0
    started = time.perf_counter()

    if optimiser == "lbfgs":
        engine = torch.optim.LBFGS(
            [unit],
            lr=1.0 if learning_rate is None else learning_rate,
            max_iter=max_iterations,
            history_size=50,
            tolerance_grad=1e-16,
            tolerance_change=1e-18,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal evaluations
            evaluations += 1
            return loss_and_grad()

        engine.step(closure)
        steps = engine.state_dict()["state"][0]["n_iter"]
        converged = steps < max_iterations
    else:
        engine = torch.optim.Adam([unit], lr=0.005 if learning_rate is None else learning_rate)
        schedule = torch.optim.lr_scheduler.ExponentialLR(engine, gamma=learning_rate_decay)
        previous = float("inf")
        converged = False
        steps = 0
        while steps < max_iterations and not converged:
            steps += 1
            current = float(loss_and_grad().detach())
            engine.step()
            schedule.step()
            evaluations += 1
            converged = abs(previous - current) <= tolerance * max(1.0, current)
            previous = current

    elapsed = time.perf_counter() - started

    with torch.no_grad():
        unit.clamp_(0.0, 1.0)
        parameters = (floor + unit * span).detach()
        final = torch_residual(
            forward_observable(parameters, wavelengths_nm, measurement, substrate),
            observed_t,
            measurement.observable,
        ).numpy()

    return FitResult(
        parameters=parameters.numpy(),
        truth=None if truth is None else np.asarray(truth, dtype=float),
        success=bool(converged),
        iterations=int(steps),
        function_evaluations=evaluations,
        wall_clock_s=elapsed,
        cost=float(0.5 * np.sum(final**2)),
        residual_rms=float(np.sqrt(np.mean(final**2))),
        message=(
            f"{optimiser}: converged after {steps} iterations"
            if converged
            else f"{optimiser}: stopped at the iteration limit ({max_iterations})"
        ),
        initial=start,
    )
