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
    "covariance_from_jacobian",
    "model_jacobian",
    "MultiStartResult",
    "fit_multi_start",
    "multi_start_grid",
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
    covariance: NDArray[np.float64] | None = field(default=None, repr=False)
    noise_sigma: float | None = None

    @property
    def standard_errors(self) -> NDArray[np.float64] | None:
        """One-sigma uncertainty on each parameter — §15's non-negotiable.

        **Local, and that word is load-bearing.** These come from the curvature of
        the cost at the point the fit stopped, so they describe the width of the
        basin it landed in. They know nothing about a rival basin 583 nm away. A
        wrong-fringe fit reports a small, confident error bar; what exposes it is
        the residual, not this. ``test_baseline.py`` measures exactly that, because
        an error bar whose limits are not stated is worse than none.
        """
        if self.covariance is None:
            return None
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def thickness_sigma_nm(self) -> float | None:
        errors = self.standard_errors
        return None if errors is None else float(errors[0])

    @property
    def correlation(self) -> NDArray[np.float64] | None:
        """§5.3's ρ, which the spec calls the thing separating a metrologist from
        someone who called ``curve_fit``. ρ(d, n) here routinely exceeds 0.99.
        """
        if self.covariance is None:
            return None
        scale = np.sqrt(np.clip(np.diag(self.covariance), 1e-300, None))
        return self.covariance / np.outer(scale, scale)

    @property
    def condition_number(self) -> float | None:
        """How close the fit is to being unable to separate its parameters.

        §5.2's degeneracies show up here before they show up in an error bar. A
        large value means the covariance was obtained by inverting a nearly
        singular matrix, and the individual standard errors are then far less
        trustworthy than their size suggests.
        """
        if self.covariance is None:
            return None
        return float(np.linalg.cond(self.covariance))

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


def model_jacobian(
    parameters,
    wavelengths_nm: NDArray,
    measurement: gen.Measurement,
    substrate: torch.Tensor,
) -> NDArray[np.float64]:
    """``J[i, k] = ∂f(λ_i) / ∂θ_k`` at ``parameters``, from autograd.

    §5.3's Jacobian, taken exactly rather than by finite differences. ``scipy``
    returns its own approximate ``jac`` at the solution and it would have been
    less code to use it, but DTFM-031 established the model can hand over the
    exact derivative — and an error bar built on a differencing step size is an
    error bar with an arbitrary constant in it.

    The Δ wrap does not enter. Wrapping shifts the residual by a multiple of 2π,
    which is locally constant, so the derivative of the wrapped residual equals
    the derivative of the model everywhere except on the seam itself.
    """
    theta = torch.as_tensor(np.asarray(parameters, dtype=float)).clone()

    def observable(values: torch.Tensor) -> torch.Tensor:
        return forward_observable(values, wavelengths_nm, measurement, substrate)

    return torch.autograd.functional.jacobian(observable, theta).numpy()


def covariance_from_jacobian(
    jacobian: NDArray,
    residual: NDArray,
    *,
    sigma: float | None = None,
    rcond: float = 1e-12,
) -> tuple[NDArray[np.float64], float]:
    """§5.3's ``C = σ²(JᵀJ)⁻¹``, and the noise scale it was computed with.

    Two ways to get ``σ``, and the choice matters:

    ``sigma`` given
        The instrument's own noise figure — what a metrologist has from the tool
        specification. Believed as stated.
    ``sigma`` omitted
        Estimated from the fit itself as ``s² = RSS / (m − n)``. Honest when the
        instrument figure is unknown, and badly optimistic when the model is
        wrong: a systematic error the model cannot represent inflates the
        residual, so ``s`` absorbs it and the error bar grows to cover a bias it
        should have been flagging. DTFM-035's model selection is the tool for
        that; this docstring is the warning until then.

    Inversion is by pseudo-inverse with an explicit cutoff rather than
    ``np.linalg.inv``. §5.2's degeneracies make ``JᵀJ`` nearly singular in the
    regimes this project is written about, and a plain inverse there returns
    enormous finite numbers with no indication that anything happened. The
    cutoff at least makes the rank deficiency explicit — and
    :attr:`FitResult.condition_number` reports what was inverted.
    """
    jacobian = np.atleast_2d(np.asarray(jacobian, dtype=float))
    residual = np.asarray(residual, dtype=float)
    points, parameters = jacobian.shape

    if sigma is None:
        degrees_of_freedom = max(points - parameters, 1)
        sigma = float(np.sqrt(np.sum(residual**2) / degrees_of_freedom))

    normal = jacobian.T @ jacobian
    return float(sigma) ** 2 * np.linalg.pinv(normal, rcond=rcond), float(sigma)


def fit_least_squares(
    observed: NDArray,
    wavelengths_nm: NDArray,
    initial: NDArray | list[float],
    *,
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
    truth: NDArray | None = None,
    max_iterations: int = 200,
    sigma: float | None = None,
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

    covariance, noise = covariance_from_jacobian(
        model_jacobian(solution.x, wavelengths_nm, measurement, substrate),
        solution.fun,
        sigma=sigma,
    )

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
        covariance=covariance,
        noise_sigma=noise,
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
    sigma: float | None = None,
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

    covariance, noise = covariance_from_jacobian(
        model_jacobian(parameters.numpy(), wavelengths_nm, measurement, substrate),
        final,
        sigma=sigma,
    )

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
        covariance=covariance,
        noise_sigma=noise,
    )


@dataclass
class MultiStartResult:
    """Every convergence point from one multi-start sweep, not just the winner.

    §6: "Handle multimodality with multi-start from many initial guesses; record
    where each converges. That landscape *is* the fringe-order ambiguity made
    visible." (The spec's name for it is inexact — see
    ``Implementation-Notes.md`` §18 — but the instruction is right.)

    So the losing fits are kept. They are the evidence that the ambiguity is
    real, they are what the landscape figure is drawn from, and DTFM-033 needs
    them to say whether an error bar computed at the winner means anything when
    a competing minimum sits 600 nm away.
    """

    fits: list[FitResult]
    starts: NDArray[np.float64]
    best_index: int
    truth: NDArray[np.float64] | None
    wall_clock_s: float

    @property
    def best(self) -> FitResult:
        """The lowest-cost fit — what a practitioner would actually report."""
        return self.fits[self.best_index]

    @property
    def parameters(self) -> NDArray[np.float64]:
        return self.best.parameters

    @property
    def thickness_error_nm(self) -> float | None:
        return self.best.thickness_error_nm

    @property
    def covariance(self) -> NDArray[np.float64] | None:
        return self.best.covariance

    @property
    def standard_errors(self) -> NDArray[np.float64] | None:
        """The winner's error bar — with the same locality caveat, sharpened.

        Multi-start is the one place in this project that *knows* the rival
        minima exist, because it visited them. The covariance still describes only
        the basin it settled in, so a fit can carry a 0.006 nm error bar while
        another minimum sits 583 nm away with a cost 1e26 higher. Both numbers are
        correct and they answer different questions: the error bar says how
        precisely the data pins the answer *given* the basin, and
        :attr:`costs` says how sure the basin choice was.
        """
        return self.best.standard_errors

    @property
    def thickness_sigma_nm(self) -> float | None:
        return self.best.thickness_sigma_nm

    @property
    def correlation(self) -> NDArray[np.float64] | None:
        return self.best.correlation

    @property
    def condition_number(self) -> float | None:
        return self.best.condition_number

    @property
    def costs(self) -> NDArray[np.float64]:
        return np.array([fit.cost for fit in self.fits])

    @property
    def thicknesses(self) -> NDArray[np.float64]:
        """Where each start converged to, in the order the starts were given."""
        return np.array([fit.parameters[0] for fit in self.fits])

    def basins(self, tolerance_nm: float = 1.0) -> tuple[NDArray, NDArray]:
        """Distinct convergence points, and how many starts fell into each.

        Clustering is deliberately crude — sort, then split wherever consecutive
        convergence points differ by more than ``tolerance_nm``. The minima here
        are separated by hundreds of nanometres and each is reached to about a
        nanometre in a billion, so anything more elaborate would be answering a
        question the data does not pose.
        """
        ordered = np.sort(self.thicknesses)
        if ordered.size == 0:
            return np.empty(0), np.empty(0, dtype=int)

        edges = np.flatnonzero(np.diff(ordered) > tolerance_nm) + 1
        groups = np.split(ordered, edges)
        return (
            np.array([group.mean() for group in groups]),
            np.array([group.size for group in groups]),
        )

    @property
    def success_fraction(self) -> float | None:
        """Fraction of starts that reached the *right* answer, when truth is known.

        The number §10 actually wants. A single fit reports ``success`` when the
        optimiser is satisfied, which DTFM-030 showed says nothing about being
        correct. This says how much of the prior a cold start can be drawn from
        and still land on the truth — which is a property of the problem, not of
        the optimiser's mood.
        """
        if self.truth is None:
            return None
        errors = np.abs(self.thicknesses - self.truth[0])
        return float(np.mean(errors < 1.0))


def multi_start_grid(
    count: int = 20, prior: gen.Prior | None = None, *, spacing: str = "uniform"
) -> NDArray[np.float64]:
    """Thickness starting points spanning the prior's support.

    **Spaced uniformly in thickness, not the way the prior samples**, and that
    is the substance of this function.

    The prior is log-uniform for a reason §7.1 sets out: each decade of thickness
    deserves equal weight because the thin regime is where the problem is hard.
    Seeding the *search* the same way is the obvious move and it is wrong. What a
    start has to do is land inside the **basin of attraction** of the true
    minimum, and those basins are spread evenly in thickness rather than in its
    logarithm. Measured, by sweeping starts 10 nm apart and recording which reach
    the truth:

    | film | widest contiguous run of starts that reach it |
    |---|---|
    | 65 nm | 160 nm |
    | 150 nm | 230 nm |
    | **420 nm** | **110 nm** |
    | 900 nm | 250 nm |
    | 1500 nm | 250 nm |
    | 1900 nm | 240 nm |

    A log grid puts half its starts below 250 nm and leaves a 470 nm gap above
    1500 nm — four times the narrowest basin, so whole regions are entered by no
    start at all:

    | start spacing | starts | films recovered |
    |---|---|---|
    | log-uniform | 12 | 5 of 7 |
    | log-uniform | 24 | 6 of 7 |
    | **uniform** | **12** | **7 of 7** |

    Doubling a log grid did not fix what changing its spacing fixed for free.
    ``spacing`` is kept as an argument so that comparison stays runnable rather
    than being a claim in a docstring.

    **Where the default comes from.** Two lines of evidence, and they agree.
    Coverage requires the spacing to be no wider than the narrowest basin, 110 nm,
    giving ``count ≥ 1980 × 0.96 / 110 ≈ 18``. Measured recovery on 40 films drawn
    from the prior:

    | starts | spacing | films recovered |
    |---|---|---|
    | 8 | 238 nm | 29 of 40 |
    | 12 | 158 nm | 40 of 40 |
    | 20 | 95 nm | 40 of 40 |
    | 32 | 59 nm | 40 of 40 |

    12 recovers everything but sits exactly on the observed boundary — 8 fails,
    and every one of its failures is a film thinner than its first start. 20 puts
    the spacing inside the narrowest basin with margin, at 20 fits per site. That
    is about a second, which is also what §1 quotes for production inversion; the
    honest cost of a correct classical answer was never one fit.

    This default was originally 12, justified by an argument that the minima sit
    one fringe apart (265 nm) so 165 nm spacing could not miss one. That argument
    is false — the minima are 4 nm apart (``Implementation-Notes.md`` §18 and §19)
    — and the number it produced happened to work. It has been re-derived from the
    basin measurement above, which is the quantity that actually governs coverage.

    Endpoints are pulled inward — a start on a bound sits where the projection
    clamps, and the fit can stall there instead of descending.
    """
    prior = prior or gen.Prior()
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if spacing not in ("uniform", "log-uniform"):
        raise ValueError(f"spacing must be 'uniform' or 'log-uniform', got {spacing!r}")

    low, high = prior.thickness_nm
    edge = 0.02
    fractions = (np.arange(count) + 0.5) / count if count > 1 else np.array([0.5])
    fractions = edge + fractions * (1.0 - 2.0 * edge)

    if spacing == "log-uniform":
        return np.exp(np.log(low) + fractions * (np.log(high) - np.log(low)))
    return low + fractions * (high - low)


def fit_multi_start(
    observed: NDArray,
    wavelengths_nm: NDArray,
    *,
    starts: NDArray | list[float] | None = None,
    count: int = 20,
    spacing: str = "uniform",
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
    truth: NDArray | None = None,
    method: str = "least_squares",
    **fit_options,
) -> MultiStartResult:
    """§6's answer to multimodality: run the fit from many starts, keep them all.

    DTFM-030 and DTFM-031 both established that a single cold-started fit reports
    ``success`` while being hundreds of nanometres wrong, and that exact gradients
    do not help because the multimodality lives in the cost surface rather than in
    the Jacobian. Only a different *search* can address it, and this is that
    search — the cheapest one that works, and the one §6 names.

    The winner is chosen by cost, which is the only criterion available when the
    truth is not known. That is worth stating plainly: multi-start does not
    *resolve* the ambiguity, it *surveys* it, and then bets on the deepest basin
    found. On noiseless data the right basin is a trillion times deeper than any
    rival, so the bet is safe. Under noise the margin shrinks, and how far it
    shrinks is the question DTFM-033's covariance and §8's calibration exist to
    answer.

    Cost is the point, not an aside. §11's amortisation argument is that
    classical inversion is expensive per site; a single fit takes tens of
    milliseconds, and this multiplies that by ``count``. The wall clock recorded
    here is the honest number for what a correct classical answer costs.

    The bet on the deepest basin holds up under realistic noise, which is worth
    stating because it was assumed rather than checked for a while. At an
    ellipsometer sigma of 1e-3 rad the recovered thickness is still good to about
    0.005 nm, and it takes roughly 30x that noise before the wrong basin is ever
    chosen.

    Dispersion starts are fixed at the prior's midpoint while thickness is swept.
    Thickness is what is ambiguous, and measurably so: freezing the dispersion at
    the truth leaves 17-33 local minima and does not move the hit rate at all
    (``Implementation-Notes.md`` §18). Sweeping three parameters would raise the
    cost by the cube for a landscape that is one-dimensional in the direction that
    matters.
    """
    prior = prior or gen.Prior()
    fitters = {"least_squares": fit_least_squares, "autograd": fit_autograd}
    if method not in fitters:
        raise ValueError(f"method must be one of {sorted(fitters)}, got {method!r}")
    fitter = fitters[method]

    if starts is None:
        starts = multi_start_grid(count, prior, spacing=spacing)
    starts = np.asarray(starts, dtype=float)

    middle_a = 0.5 * (prior.cauchy_a[0] + prior.cauchy_a[1])
    middle_b = 0.5 * (prior.cauchy_b[0] + prior.cauchy_b[1])

    began = time.perf_counter()
    fits = [
        fitter(
            observed,
            wavelengths_nm,
            [thickness, middle_a, middle_b],
            measurement=measurement,
            prior=prior,
            truth=truth,
            **fit_options,
        )
        for thickness in starts
    ]
    elapsed = time.perf_counter() - began

    return MultiStartResult(
        fits=fits,
        starts=starts,
        best_index=int(np.argmin([fit.cost for fit in fits])),
        truth=None if truth is None else np.asarray(truth, dtype=float),
        wall_clock_s=elapsed,
    )
