"""Does σ̂ mean what it says? — DTFM-048 and DTFM-049, spec §8.2.

A network that reports an uncertainty is making a falsifiable claim: that the
truth lies within ``±σ̂`` about 68% of the time, and within ``±1.96σ̂`` about 95%.
This module checks that claim at every level at once and reduces the answer to
one number.

**Calibration alone is not a virtue.** A model that reports σ̂ = 2000 nm of
thickness on every film is perfectly calibrated at every level above about 90% and
completely useless. The counterweight is *sharpness* — how small the stated
uncertainties are — and the two are only meaningful together. §8.2 asks for the
reliability diagram; :func:`sharpness` is reported beside it so the diagram cannot
be gamed by hedging.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

#: Nominal levels for the reliability diagram. Dense in the middle where most
#: films sit, and reaching 0.99 because the tail is where an over-confident model
#: does its damage: being wrong at 99% nominal is a wafer scrapped on a number
#: that claimed to be certain.
LEVELS: NDArray[np.float64] = np.array(
    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.683, 0.75, 0.8, 0.9, 0.95, 0.99]
)


def _z(level: float | NDArray) -> NDArray[np.float64]:
    """Half-width of the central interval holding ``level`` of a standard normal.

    ``1σ`` is the 68.3% interval, not the 68%: the conventional shorthand rounds,
    and rounding here would put a permanent 0.3% bias into every coverage number.
    """
    return stats.norm.ppf((1.0 + np.asarray(level, dtype=float)) / 2.0)


def coverage(error: NDArray, sigma: NDArray, level: float = 0.683) -> float:
    """Fraction of films whose true value fell inside the stated interval.

    ``error`` is signed or absolute — only its magnitude is used, because the
    interval is symmetric. A well-calibrated model returns ``level`` back.
    """
    error = np.abs(np.asarray(error, dtype=float))
    sigma = np.asarray(sigma, dtype=float)
    if error.shape != sigma.shape:
        raise ValueError(f"one sigma per error: got {error.shape} and {sigma.shape}")
    if np.any(sigma <= 0):
        raise ValueError("a standard deviation must be positive")
    return float(np.mean(error <= _z(level) * sigma))


def reliability(
    error: NDArray, sigma: NDArray, levels: NDArray | None = None
) -> dict[str, NDArray[np.float64]]:
    """Nominal against empirical coverage — §8.2's diagram, as numbers.

    A perfectly calibrated model traces the diagonal. **Above** the diagonal is
    over-coverage: the intervals are wider than the errors need, which is
    conservative and wasteful. **Below** is over-confidence, which is the
    dangerous direction — a stated 95% interval that actually holds 60% of the
    time will let a wafer through on a number nobody should have trusted.
    """
    levels = LEVELS if levels is None else np.asarray(levels, dtype=float)
    empirical = np.array([coverage(error, sigma, level) for level in levels])
    return {"nominal": levels, "empirical": empirical, "gap": empirical - levels}


def expected_calibration_error(
    error: NDArray, sigma: NDArray, levels: NDArray | None = None
) -> float:
    """§8.2's one number: mean distance from the diagonal — DTFM-049.

    Averaged over levels rather than weighted by anything, so it says what it
    looks like: *on average, stated confidence is wrong by this much*. An ECE of
    0.05 means a nominal 68% interval is really holding about 63% or 73%.

    It is a **signed-blind** summary — over- and under-confidence both add to it —
    so it should be read with :func:`reliability`, which shows the direction, and
    with :func:`sharpness`, which shows whether the calibration was bought by
    hedging.
    """
    return float(np.mean(np.abs(reliability(error, sigma, levels)["gap"])))


def sharpness(sigma: NDArray) -> float:
    """Median stated uncertainty — the counterweight to calibration.

    Reported alongside every calibration number because calibration on its own is
    trivially gameable: a model that says σ̂ = 2000 nm of thickness everywhere is
    perfectly calibrated at high levels and worth nothing. Between two calibrated
    models the sharper one is better; between a calibrated model and a sharper
    miscalibrated one, the choice is a judgement the numbers cannot make.
    """
    return float(np.median(np.asarray(sigma, dtype=float)))


def summary(error: NDArray, sigma: NDArray, levels: NDArray | None = None) -> dict:
    """Everything §8.2 asks for about one model, in one dictionary."""
    curve = reliability(error, sigma, levels)
    return {
        "coverage_68": coverage(error, sigma, 0.683),
        "coverage_95": coverage(error, sigma, 0.95),
        "ece": expected_calibration_error(error, sigma, levels),
        "sharpness_nm": sharpness(sigma),
        "median_error_nm": float(np.median(np.abs(error))),
        "worst_gap": float(curve["gap"][np.argmax(np.abs(curve["gap"]))]),
        "nominal": curve["nominal"].tolist(),
        "empirical": curve["empirical"].tolist(),
    }
