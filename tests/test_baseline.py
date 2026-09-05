"""Classical inversion — the baseline every later number is measured against.

Spec §6.
Levenberg-Marquardt by DTFM-030.

§6: "This must exist before any network is trained. Without it, a learned model's
accuracy number is meaningless."
"""

import numpy as np
import pytest
import torch

from src import baseline as bl
from src import dispersion as dp
from src import generate as gen

WAVELENGTHS = np.linspace(400.0, 800.0, 200)
TRUTH = np.array([420.0, 1.46, 0.004])


@pytest.fixture(scope="module")
def substrate() -> torch.Tensor:
    n, k = dp.load_nk("Si", WAVELENGTHS)
    return torch.tensor(n + 1j * k)


def _observe(truth, measurement, substrate) -> np.ndarray:
    return bl.forward_observable(truth, WAVELENGTHS, measurement, substrate).numpy()


# --- the acceptance criterion ------------------------------------------------


@pytest.mark.parametrize("observable", ["ellipsometry", "reflectance"])
def test_it_recovers_known_parameters_from_noiseless_data(observable, substrate):
    """The acceptance criterion. Started near the answer, it must find it exactly."""
    measurement = gen.Measurement(observable=observable)
    observed = _observe(TRUTH, measurement, substrate)

    result = bl.fit_least_squares(
        observed, WAVELENGTHS, [400.0, 1.45, 0.005], measurement=measurement, truth=TRUTH
    )

    assert result.success
    assert abs(result.thickness_error_nm) < 1e-6
    assert np.allclose(result.parameters, TRUTH, rtol=1e-6)


def test_every_fit_records_what_it_did(substrate):
    """§6 lists what to record for every case. §15 calls a fit without an error
    bar a disqualifying instinct — the covariance arrives at DTFM-033, but the
    rest of the record has to be here so a number can never be quoted without
    being able to say how it was reached.
    """
    measurement = gen.Measurement()
    result = bl.fit_least_squares(
        _observe(TRUTH, measurement, substrate), WAVELENGTHS, [400.0, 1.45, 0.005],
        measurement=measurement, truth=TRUTH,
    )

    assert result.parameters.shape == (3,)
    assert result.error is not None and result.error.shape == (3,)
    assert result.function_evaluations > 0
    assert result.wall_clock_s > 0.0
    assert isinstance(result.success, bool)
    assert result.message
    assert result.residual_rms >= 0.0


def test_the_fit_is_slow_enough_for_the_project_to_have_a_point(substrate):
    """§11's amortisation argument needs a number, and this is where it comes from.

    A single fit takes tens of milliseconds. §1 quotes roughly a second per site
    in production, on harder problems; either way it is far above the
    microseconds a forward pass costs. Asserting a loose upper bound only —
    the point is that it is measured per fit, not that it hits a target.
    """
    measurement = gen.Measurement()
    result = bl.fit_least_squares(
        _observe(TRUTH, measurement, substrate), WAVELENGTHS, [400.0, 1.45, 0.005],
        measurement=measurement,
    )

    assert 1e-4 < result.wall_clock_s < 30.0


# --- the failure mode §10 predicts -------------------------------------------


@pytest.mark.parametrize("true_thickness", [65.0, 900.0, 1500.0])
def test_a_cold_start_converges_to_the_wrong_fringe(true_thickness, substrate):
    """§5.2(b)'s fringe-order ambiguity, on the first estimator built.

    R is quasi-periodic in n·d, so the cost surface has many minima. Started at
    300 nm, the fit walks into whichever basin it lands in and stops. It reports
    ``success`` while being hundreds of nanometres wrong.

    This is not a defect to fix here — §6 prescribes multi-start for it, at
    DTFM-032. It is the behaviour the baseline genuinely has, and pretending
    otherwise would make every later comparison dishonest.
    """
    measurement = gen.Measurement()
    truth = np.array([true_thickness, 1.46, 0.004])
    result = bl.fit_least_squares(
        _observe(truth, measurement, substrate), WAVELENGTHS, [300.0, 1.47, 0.005],
        measurement=measurement, truth=truth,
    )

    assert result.success, "the optimiser is satisfied"
    assert abs(result.thickness_error_nm) > 10.0, "and wrong"


def test_the_classical_method_fails_loudly(substrate):
    """§10's central prediction, and the reason the baseline matters.

    "The expected and most valuable finding is that the classical method fails
    *loudly* and the network fails *silently*."

    Measured here: a correct fit leaves a residual of ~1e-12, a wrong one ~1e0.
    Twelve orders of magnitude. Anyone looking at the residual knows immediately
    that the answer is not to be trusted — which is exactly the property §8 asks
    whether a network's uncertainty head can match.
    """
    measurement = gen.Measurement()

    good = bl.fit_least_squares(
        _observe(TRUTH, measurement, substrate), WAVELENGTHS, [400.0, 1.45, 0.005],
        measurement=measurement, truth=TRUTH,
    )
    wrong_truth = np.array([1500.0, 1.44, 0.003])
    bad = bl.fit_least_squares(
        _observe(wrong_truth, measurement, substrate), WAVELENGTHS, [300.0, 1.47, 0.005],
        measurement=measurement, truth=wrong_truth,
    )

    assert abs(good.thickness_error_nm) < 1e-6
    assert abs(bad.thickness_error_nm) > 100.0
    assert bad.residual_rms > 1e6 * good.residual_rms


# --- the delta wrap ----------------------------------------------------------


def test_the_residual_wraps_delta_onto_the_circle():
    """The one place ellipsometry needs different arithmetic, and it is not tidy-up.

    Δ lives on (−π, π]. A model at +179° against an observation at −179° is two
    degrees apart, but subtracting gives 358° — a residual 180 times too large,
    at exactly the wavelengths where the fit is most informative. Least squares
    would bend the whole solution to chase a discontinuity that is not there.
    """
    model = np.array([0.5, 0.5, np.radians(179.0), np.radians(-179.0)])
    observed = np.array([0.5, 0.5, np.radians(-179.0), np.radians(179.0)])

    naive = model - observed
    wrapped = bl.wrapped_residual(model, observed, "ellipsometry")

    assert abs(np.degrees(naive[2])) == pytest.approx(358.0, abs=0.1)
    assert abs(np.degrees(wrapped[2])) == pytest.approx(2.0, abs=0.1)
    assert np.allclose(wrapped[:2], 0.0)


def test_reflectance_residuals_are_left_alone():
    model = np.array([0.4, 0.6])
    observed = np.array([0.1, 0.9])

    assert np.allclose(bl.wrapped_residual(model, observed, "reflectance"), [0.3, -0.3])


def test_wrapping_matters_for_a_real_fit(substrate):
    """Not a synthetic edge case: a thick film sweeps Δ through its wrap many
    times across the band, so an unwrapped residual would corrupt real fits.
    """
    measurement = gen.Measurement()
    truth = np.array([1400.0, 1.46, 0.004])
    observed = _observe(truth, measurement, substrate)

    half = observed.size // 2
    delta = observed[half:]
    wraps = int(np.sum(np.abs(np.diff(delta)) > np.pi))

    assert wraps > 0, "this film should cross the wrap"

    result = bl.fit_least_squares(
        observed, WAVELENGTHS, [1390.0, 1.46, 0.004], measurement=measurement, truth=truth
    )
    assert abs(result.thickness_error_nm) < 1e-3


# --- bounds and behaviour under noise ----------------------------------------


def test_the_fit_stays_inside_the_prior_support(substrate):
    """Bounds default to the prior, deliberately.

    §10 compares the classical fit against the network. Giving the fit a wider
    search space than the network's prior would tilt that comparison in the
    classical method's favour without saying so.
    """
    prior = gen.Prior()
    measurement = gen.Measurement()
    result = bl.fit_least_squares(
        _observe(TRUTH, measurement, substrate), WAVELENGTHS, [10.0, 1.0, 0.0],
        measurement=measurement, prior=prior,
    )

    assert prior.thickness_nm[0] <= result.parameters[0] <= prior.thickness_nm[1]
    assert prior.cauchy_a[0] <= result.parameters[1] <= prior.cauchy_a[1]


def test_noise_degrades_the_answer_without_breaking_it(substrate):
    """With realistic instrument noise the fit should still land close, and its
    error should grow with the noise rather than jump around.
    """
    measurement = gen.Measurement()
    clean = _observe(TRUTH, measurement, substrate)
    rng = np.random.default_rng(0)

    errors = []
    for sigma in (1e-4, 1e-3, 1e-2):
        noisy = clean + rng.normal(0.0, sigma, clean.shape)
        result = bl.fit_least_squares(
            noisy, WAVELENGTHS, [415.0, 1.46, 0.004], measurement=measurement, truth=TRUTH
        )
        errors.append(abs(result.thickness_error_nm))

    assert errors[0] < 0.5
    assert errors[-1] > errors[0]


def test_the_fit_is_reproducible(substrate):
    """§15. A deterministic optimiser on deterministic data must repeat exactly."""
    measurement = gen.Measurement()
    observed = _observe(TRUTH, measurement, substrate)

    first = bl.fit_least_squares(observed, WAVELENGTHS, [400.0, 1.45, 0.005],
                                 measurement=measurement)
    second = bl.fit_least_squares(observed, WAVELENGTHS, [400.0, 1.45, 0.005],
                                  measurement=measurement)

    assert np.array_equal(first.parameters, second.parameters)
    assert first.function_evaluations == second.function_evaluations
