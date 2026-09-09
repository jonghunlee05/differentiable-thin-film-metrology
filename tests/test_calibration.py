"""Coverage, reliability and ECE — DTFM-048 and DTFM-049, spec §8.2.

A model reporting σ̂ makes a falsifiable claim. These check the checker: that a
perfectly calibrated model scores as calibrated, that both failure directions are
caught, and that the summary cannot be gamed by hedging.
"""

import numpy as np
import pytest

from src import calibration as cal


@pytest.fixture(scope="module")
def gaussian():
    """Errors drawn from exactly the stated sigma — a perfectly calibrated model."""
    rng = np.random.default_rng(0)
    sigma = np.full(40_000, 2.0)
    return rng.normal(0.0, 2.0, sigma.shape), sigma


def test_a_perfectly_calibrated_model_scores_as_calibrated(gaussian):
    """The null case. If this drifts, every other number here is meaningless."""
    error, sigma = gaussian

    assert cal.coverage(error, sigma, 0.683) == pytest.approx(0.683, abs=0.01)
    assert cal.coverage(error, sigma, 0.95) == pytest.approx(0.95, abs=0.01)
    assert cal.expected_calibration_error(error, sigma) < 0.01


def test_one_sigma_means_68_3_percent_not_68(gaussian):
    """The conventional shorthand rounds, and rounding here would bake a
    permanent 0.3% bias into every coverage number this project reports.
    """
    error, sigma = gaussian

    assert cal.coverage(error, sigma, 0.683) > cal.coverage(error, sigma, 0.68)


def test_overconfidence_and_hedging_are_both_caught(gaussian):
    """ECE is signed-blind on purpose — both directions are miscalibration — so
    the reliability curve is what says which. Over-confidence is the dangerous
    one: a stated 95% interval holding 68% lets a wafer through on a number
    nobody should have trusted.
    """
    error, sigma = gaussian

    confident = cal.summary(error, sigma / 2)
    hedging = cal.summary(error, sigma * 3)

    assert confident["coverage_68"] < 0.5, "halved sigma must under-cover"
    assert hedging["coverage_68"] > 0.95, "tripled sigma must over-cover"
    assert confident["ece"] > 0.1 and hedging["ece"] > 0.1
    assert cal.reliability(error, sigma / 2)["gap"].mean() < 0 < (
        cal.reliability(error, sigma * 3)["gap"].mean()
    ), "the sign of the gap is what distinguishes them"


def test_calibration_alone_can_be_bought_by_hedging(gaussian):
    """Which is why sharpness is reported beside it.

    A model that says σ̂ = 2000 nm of thickness on every film is perfectly
    calibrated at every level above about 90% and worth nothing. Between two
    calibrated models the sharper one is better, and a calibration number with no
    sharpness beside it cannot say that.
    """
    error, sigma = gaussian

    honest = cal.summary(error, sigma)
    hedged = cal.summary(error, sigma * 3)

    assert hedged["coverage_95"] >= honest["coverage_95"], "hedging looks good at 95%"
    assert hedged["sharpness_nm"] > honest["sharpness_nm"] * 2, "and sharpness exposes it"


def test_a_mismatched_sigma_is_refused():
    """One sigma per error. Broadcasting a scalar would silently score every film
    against the same interval, which is the opposite of heteroscedastic.
    """
    with pytest.raises(ValueError, match="one sigma per error"):
        cal.coverage(np.zeros(10), np.ones(4))


def test_a_zero_sigma_is_refused():
    """A zero uncertainty is a claim of perfect knowledge, and dividing by it
    would report either 0% or 100% coverage depending on floating-point luck.
    """
    with pytest.raises(ValueError, match="must be positive"):
        cal.coverage(np.zeros(4), np.zeros(4))


def test_the_reliability_curve_is_monotone(gaussian):
    """Coverage cannot fall as the interval widens. Not a property of the model —
    a property of arithmetic — so a violation means the checker is broken.
    """
    error, sigma = gaussian

    empirical = cal.reliability(error, sigma)["empirical"]

    assert np.all(np.diff(empirical) >= -1e-12)
