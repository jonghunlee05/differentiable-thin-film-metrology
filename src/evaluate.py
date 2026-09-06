"""The evaluation protocol — §10's table, computed the same way for every method.

Spec §10.
Implemented by DTFM-036 for the classical baselines; the learned model joins the
same table at DTFM-044.

§10 opens with the sentence this module exists to enforce:

    Every claim gets a number, computed the same way for every method.

So the estimator is an argument, not a branch. Anything that maps a spectrum to
three parameters can be scored here, and the comparison is then structural rather
than a matter of trusting that two separate scripts measured the same thing.

Stratification is not a refinement
----------------------------------
§10 asks for error "stratified by regime (thin / mid / thick, low / high SNR)",
and that stratification is what makes the headline number honest. Thick films are
easy: the fringes are dense and the thickness falls out. Thin films are the
regime §5.2(c) is about, where the signal departs from a bare interface only
quadratically and the fit returns a confident meaningless number.

A single RMSE over a log-uniform prior therefore reports mostly how many easy
films happened to be drawn. Two methods can share an RMSE while one is uniformly
mediocre and the other is excellent above 300 nm and useless below it. §7.1 chose
a log-uniform prior precisely so the thin regime is well represented; reporting a
single average would throw that choice away at the last step.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from src import baseline as bl
from src import dispersion as dp
from src import generate as gen

__all__ = [
    "Case",
    "Report",
    "REGIMES",
    "SNR_LEVELS",
    "evaluate",
    "make_cases",
    "regime_of",
]

#: §10's thickness regimes. The boundaries are the physics, not round numbers:
#: 100 nm is roughly where §5.2(c)'s quadratic insensitivity stops dominating,
#: and 700 nm is where DTFM-034 measured ρ(d, n) locking above 0.99 so that only
#: the optical path n·d is really being determined.
REGIMES: dict[str, tuple[float, float]] = {
    "thin": (0.0, 100.0),
    "mid": (100.0, 700.0),
    "thick": (700.0, np.inf),
}

#: §10's two noise levels, in radians on Ψ and Δ. ``high`` is the instrument
#: figure from ``configs/default.yaml``, matched to a real ellipsometer; ``low``
#: is ten times worse, which DTFM-032 measured as still recoverable and DTFM-028
#: measured as the point where ellipsometry's advantage starts to erode.
SNR_LEVELS: dict[str, float] = {"high": 1e-3, "low": 1e-2}


def regime_of(thickness_nm: float) -> str:
    """Which of §10's three thickness regimes a film falls in."""
    for name, (low, high) in REGIMES.items():
        if low <= thickness_nm < high:
            return name
    raise ValueError(f"no regime covers {thickness_nm} nm")


@dataclass
class Case:
    """One film, one noise level — the unit every method is scored on.

    Held as data rather than regenerated per method so that both baselines see
    **the same noise draw**, not merely the same distribution. §10 compares
    methods, and a comparison in which each estimator got its own random numbers
    measures the noise as much as the method.
    """

    truth: NDArray[np.float64]
    observed: NDArray[np.float64]
    wavelengths_nm: NDArray[np.float64]
    sigma: float
    snr: str

    @property
    def thickness_nm(self) -> float:
        return float(self.truth[0])

    @property
    def regime(self) -> str:
        return regime_of(self.thickness_nm)


@dataclass
class Report:
    """Scored results for one method, sliceable the way §10 asks.

    Keeps every case rather than only the summary. §10's failure atlas (DTFM-055)
    needs the individual failures, and a summary that cannot be reopened is a
    summary that has to be trusted.
    """

    method: str
    cases: list[Case] = field(repr=False)
    estimates: NDArray[np.float64] = field(repr=False)
    seconds: NDArray[np.float64] = field(repr=False)
    converged: NDArray[np.bool_] = field(repr=False)
    residuals: NDArray[np.float64] = field(repr=False)

    @property
    def truth(self) -> NDArray[np.float64]:
        return np.array([case.thickness_nm for case in self.cases])

    @property
    def errors(self) -> NDArray[np.float64]:
        return self.estimates - self.truth

    def _mask(self, regime: str | None, snr: str | None) -> NDArray[np.bool_]:
        keep = np.ones(len(self.cases), dtype=bool)
        if regime is not None:
            keep &= np.array([case.regime == regime for case in self.cases])
        if snr is not None:
            keep &= np.array([case.snr == snr for case in self.cases])
        return keep

    def metrics(self, *, regime: str | None = None, snr: str | None = None) -> dict:
        """§10's row: RMSE, median absolute error, wall clock, failure rate.

        ``failure_rate`` is deliberately **not** the optimiser's own success flag.
        DTFM-030 established that a fit reports ``success`` while being 583 nm
        wrong, so counting that flag would report roughly zero failures for a
        method that is wrong most of the time from a cold start. A failure here
        means the answer is wrong by more than 1 nm — a fact about the estimate,
        not about the optimiser's mood.

        ``convergence_flag_rate`` keeps the optimiser's own opinion alongside it,
        because the *gap* between the two columns is one of §10's findings rather
        than an inconsistency to tidy away.
        """
        keep = self._mask(regime, snr)
        count = int(keep.sum())
        if count == 0:
            return {"n": 0}

        errors = self.errors[keep]
        return {
            "n": count,
            "rmse_nm": float(np.sqrt(np.mean(errors**2))),
            "median_abs_nm": float(np.median(np.abs(errors))),
            "p95_abs_nm": float(np.percentile(np.abs(errors), 95)),
            "seconds": float(np.median(self.seconds[keep])),
            "failure_rate": float(np.mean(np.abs(errors) > 1.0)),
            "convergence_flag_rate": float(np.mean(~self.converged[keep])),
            "median_residual": float(np.median(self.residuals[keep])),
        }

    def table(self) -> list[dict]:
        """Every regime x SNR cell, plus the overall row §10 warns not to trust alone."""
        rows = [{"regime": "all", "snr": "all", **self.metrics()}]
        for snr in SNR_LEVELS:
            for regime in REGIMES:
                rows.append({"regime": regime, "snr": snr, **self.metrics(regime=regime, snr=snr)})
        return [row for row in rows if row.get("n", 0) > 0]


def make_cases(
    count: int = 60,
    *,
    seed: int = 0,
    wavelengths_nm: NDArray | None = None,
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
) -> list[Case]:
    """Draw films from the prior and observe each at both noise levels.

    Every film appears once per SNR level rather than being assigned one at
    random, so the low-noise and high-noise columns are the *same* films. That
    turns "error grows with noise" from a statement about two samples into a
    paired comparison on one.

    The prior is §7.1's, unmodified. §10's whole argument depends on the test
    distribution matching the training distribution the network will later see —
    scoring the baseline on a different draw would make the eventual comparison
    meaningless.
    """
    measurement = measurement or gen.Measurement()
    prior = prior or gen.Prior()
    wavelengths_nm = (
        np.linspace(400.0, 800.0, 200) if wavelengths_nm is None else np.asarray(wavelengths_nm)
    )

    rng = np.random.default_rng(seed)
    sampled = gen.sample_parameters(prior, count, rng)
    n, k = dp.load_nk(prior.substrate, wavelengths_nm)
    substrate = torch.tensor(n + 1j * k)

    cases: list[Case] = []
    for i in range(count):
        truth = np.array(
            [sampled["thickness_nm"][i], sampled["cauchy_a"][i], sampled["cauchy_b"][i]]
        )
        with torch.no_grad():
            clean = bl.forward_observable(truth, wavelengths_nm, measurement, substrate).numpy()
        for snr, sigma in SNR_LEVELS.items():
            cases.append(
                Case(
                    truth=truth,
                    observed=clean + rng.normal(0.0, sigma, clean.shape),
                    wavelengths_nm=wavelengths_nm,
                    sigma=sigma,
                    snr=snr,
                )
            )
    return cases


def evaluate(
    cases: list[Case],
    estimator: Callable[[Case], tuple[NDArray, bool, float]] | None = None,
    *,
    method: str = "multi_start",
    measurement: gen.Measurement | None = None,
    prior: gen.Prior | None = None,
    **fit_options,
) -> Report:
    """Score one estimator over every case, timing each inversion.

    ``estimator`` takes a :class:`Case` and returns ``(parameters, converged,
    residual_rms)``. That signature is the whole point of this module: the
    network at DTFM-044 satisfies it as easily as a fitter does, so it enters
    §10's table without a second scoring path being written for it.

    Timing is per case and recorded as the median rather than the mean. A single
    pathological fit that runs to the iteration limit would drag a mean and
    misrepresent what an inversion costs; §11's amortisation argument needs the
    typical cost, and the tail is visible in the failure columns anyway.
    """
    measurement = measurement or gen.Measurement()
    prior = prior or gen.Prior()

    if estimator is None:
        fitters = {
            "multi_start": lambda case: bl.fit_multi_start(
                case.observed,
                case.wavelengths_nm,
                measurement=measurement,
                prior=prior,
                sigma=case.sigma,
                **fit_options,
            ),
            # Started at the *whole* truth, dispersion included. An earlier
            # version started at the true thickness with the index pinned at
            # 1.46, which is neither warm nor cold: DTFM-034 measured ρ(d, n)
            # above 0.99 past 700 nm, so an index off by 0.07 is a thickness off
            # by hundreds of nanometres. That produced a 70% failure rate in the
            # thick regime for a method supposedly handed the answer, and the
            # number was a property of the start rather than of the method.
            # Kept as "warm_thickness_only" because the effect is worth showing.
            "least_squares": lambda case: bl.fit_least_squares(
                case.observed,
                case.wavelengths_nm,
                case.truth,
                measurement=measurement,
                prior=prior,
                sigma=case.sigma,
                **fit_options,
            ),
            "autograd": lambda case: bl.fit_autograd(
                case.observed,
                case.wavelengths_nm,
                case.truth,
                measurement=measurement,
                prior=prior,
                sigma=case.sigma,
                **fit_options,
            ),
            # One search, knowing nothing — the row that justifies multi-start.
            # Without it the table shows multi-start is good but never shows why
            # anyone would pay 20x for it. The start is the prior's geometric
            # midpoint, sqrt(20 x 2000) = 200 nm, which is the least informed
            # single guess available under a log-uniform prior.
            "cold_single": lambda case: bl.fit_least_squares(
                case.observed,
                case.wavelengths_nm,
                [float(np.sqrt(prior.thickness_nm[0] * prior.thickness_nm[1])), 1.475, 0.007],
                measurement=measurement,
                prior=prior,
                sigma=case.sigma,
                **fit_options,
            ),
            "warm_thickness_only": lambda case: bl.fit_least_squares(
                case.observed,
                case.wavelengths_nm,
                [case.thickness_nm, 1.46, 0.004],
                measurement=measurement,
                prior=prior,
                sigma=case.sigma,
                **fit_options,
            ),
        }
        if method not in fitters:
            raise ValueError(f"method must be one of {sorted(fitters)}, got {method!r}")
        chosen = fitters[method]

        def estimator(case: Case):  # noqa: F811 - deliberate default
            result = chosen(case)
            # A single fit carries its own flags; a multi-start sweep carries the
            # winner's. Both expose .parameters, which is why they share a scorer.
            single = result.best if isinstance(result, bl.MultiStartResult) else result
            return result.parameters, bool(single.success), float(single.residual_rms)

    estimates = np.empty(len(cases))
    seconds = np.empty(len(cases))
    converged = np.empty(len(cases), dtype=bool)
    residuals = np.empty(len(cases))

    for i, case in enumerate(cases):
        began = time.perf_counter()
        parameters, ok, residual = estimator(case)
        seconds[i] = time.perf_counter() - began
        estimates[i] = float(np.asarray(parameters)[0])
        converged[i] = ok
        residuals[i] = residual

    return Report(
        method=method,
        cases=cases,
        estimates=estimates,
        seconds=seconds,
        converged=converged,
        residuals=residuals,
    )
