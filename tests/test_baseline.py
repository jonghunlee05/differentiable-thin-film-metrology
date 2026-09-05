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
        _observe(TRUTH, measurement, substrate),
        WAVELENGTHS,
        [400.0, 1.45, 0.005],
        measurement=measurement,
        truth=TRUTH,
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
        _observe(TRUTH, measurement, substrate),
        WAVELENGTHS,
        [400.0, 1.45, 0.005],
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
        _observe(truth, measurement, substrate),
        WAVELENGTHS,
        [300.0, 1.47, 0.005],
        measurement=measurement,
        truth=truth,
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
        _observe(TRUTH, measurement, substrate),
        WAVELENGTHS,
        [400.0, 1.45, 0.005],
        measurement=measurement,
        truth=TRUTH,
    )
    wrong_truth = np.array([1500.0, 1.44, 0.003])
    bad = bl.fit_least_squares(
        _observe(wrong_truth, measurement, substrate),
        WAVELENGTHS,
        [300.0, 1.47, 0.005],
        measurement=measurement,
        truth=wrong_truth,
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
        _observe(TRUTH, measurement, substrate),
        WAVELENGTHS,
        [10.0, 1.0, 0.0],
        measurement=measurement,
        prior=prior,
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

    first = bl.fit_least_squares(
        observed, WAVELENGTHS, [400.0, 1.45, 0.005], measurement=measurement
    )
    second = bl.fit_least_squares(
        observed, WAVELENGTHS, [400.0, 1.45, 0.005], measurement=measurement
    )

    assert np.array_equal(first.parameters, second.parameters)
    assert first.function_evaluations == second.function_evaluations


# --- DTFM-031: the second baseline, descending through the simulator ---------


def test_the_torch_residual_matches_the_numpy_one():
    """The same wrap arithmetic exists twice, so it is pinned twice.

    ``scipy`` cannot take a tensor and autograd cannot take an array, so
    :func:`bl.wrapped_residual` and :func:`bl.torch_residual` are separate
    implementations of one rule. Duplicated logic drifts silently — which is
    §10's whole theme — so they are checked against each other, including at the
    ±π seam where a sign convention could plausibly differ.
    """
    rng = np.random.default_rng(7)
    model = np.concatenate([rng.uniform(0.0, 1.5, 50), rng.uniform(-np.pi, np.pi, 50)])
    observed = np.concatenate([rng.uniform(0.0, 1.5, 50), rng.uniform(-np.pi, np.pi, 50)])
    seam = np.array([0.5, np.radians(179.0), 0.5, np.radians(-179.0)])
    seam_observed = np.array([0.5, np.radians(-179.0), 0.5, np.radians(179.0)])

    for a, b in ((model, observed), (seam, seam_observed)):
        for observable in ("ellipsometry", "reflectance"):
            numpy_side = bl.wrapped_residual(a, b, observable)
            torch_side = bl.torch_residual(
                torch.as_tensor(a), torch.as_tensor(b), observable
            ).numpy()
            assert np.array_equal(numpy_side, torch_side), observable


def test_the_gradient_is_exact_rather_than_finite_differenced(substrate):
    """The whole reason this baseline exists at all.

    Levenberg-Marquardt has to rebuild the Jacobian by nudging each parameter and
    re-running the model — three extra forward passes per step here, and one per
    parameter in general. The differentiable model already knows its own
    derivative, so one backward pass gives all three at once, exactly.

    "Exactly" is the claim under test: autograd's gradient must match central
    differences to the accuracy central differences themselves can manage.
    """
    measurement = gen.Measurement()
    observed = torch.as_tensor(_observe(TRUTH, measurement, substrate))
    theta = torch.tensor([430.0, 1.45, 0.005], requires_grad=True)

    def loss_at(values) -> torch.Tensor:
        model = bl.forward_observable(values, WAVELENGTHS, measurement, substrate)
        return 0.5 * (bl.torch_residual(model, observed, "ellipsometry") ** 2).sum()

    loss_at(theta).backward()
    analytic = theta.grad.numpy().copy()

    steps = np.array([1e-4, 1e-7, 1e-9])
    numeric = np.empty(3)
    with torch.no_grad():
        for i, step in enumerate(steps):
            shift = torch.zeros(3, dtype=torch.float64)
            shift[i] = step
            base = theta.detach()
            numeric[i] = float(loss_at(base + shift) - loss_at(base - shift)) / (2 * step)

    assert np.allclose(analytic, numeric, rtol=1e-5)


def test_the_gradient_components_span_orders_of_magnitude(substrate):
    """Why the descent runs in normalised coordinates, stated as a measurement.

    ``d`` is in nanometres and Cauchy ``B`` is a number near 0.004, so the two
    partial derivatives are not remotely comparable in size. A single step size
    applied to raw ``θ`` is therefore either negligible for one parameter or
    divergent for another — mapping the prior's box onto the unit cube is what
    makes one learning rate meaningful for all three, and is load-bearing rather
    than tidy-up.
    """
    measurement = gen.Measurement()
    observed = torch.as_tensor(_observe(TRUTH, measurement, substrate))
    theta = torch.tensor([430.0, 1.45, 0.005], requires_grad=True)

    model = bl.forward_observable(theta, WAVELENGTHS, measurement, substrate)
    (0.5 * (bl.torch_residual(model, observed, "ellipsometry") ** 2).sum()).backward()

    magnitudes = np.abs(theta.grad.numpy())
    assert magnitudes.max() / magnitudes.min() > 1e3


@pytest.mark.parametrize("observable", ["ellipsometry", "reflectance"])
def test_autograd_descent_recovers_known_parameters(observable, substrate):
    """DTFM-031's acceptance criterion, matching DTFM-030's.

    Started near the answer, descending through the simulator must find it —
    to about a nanometre in a billion, which is the same standard the
    Levenberg-Marquardt baseline is held to.
    """
    measurement = gen.Measurement(observable=observable)
    observed = _observe(TRUTH, measurement, substrate)

    result = bl.fit_autograd(
        observed, WAVELENGTHS, [400.0, 1.45, 0.005], measurement=measurement, truth=TRUTH
    )

    assert result.success
    assert abs(result.thickness_error_nm) < 1e-6
    assert np.allclose(result.parameters, TRUTH, rtol=1e-6)


def test_autograd_records_what_it_did(substrate):
    """§6's record, kept identically by both baselines.

    They share :class:`bl.FitResult` on purpose: DTFM-036 tabulates them side by
    side, and two estimators reporting different fields could not be compared
    without someone deciding, later and informally, what the missing ones meant.
    """
    measurement = gen.Measurement()
    result = bl.fit_autograd(
        _observe(TRUTH, measurement, substrate),
        WAVELENGTHS,
        [400.0, 1.45, 0.005],
        measurement=measurement,
        truth=TRUTH,
    )

    assert result.parameters.shape == (3,)
    assert result.error is not None and result.error.shape == (3,)
    assert result.iterations > 0
    assert result.function_evaluations >= result.iterations
    assert result.wall_clock_s > 0.0
    assert isinstance(result.success, bool)
    assert result.message
    assert result.residual_rms >= 0.0


def test_the_two_baselines_agree_where_both_converge(substrate):
    """§6 asks for the two compared, and this is the comparison's premise.

    Given the same data and the same warm start they must land on the same
    parameters. They share nothing but the forward model — one rebuilds the
    Jacobian from finite differences in ``scipy``, the other reads it off the
    autograd graph — so agreement to six significant figures is evidence that
    both are solving the stated problem rather than each solving its own.
    """
    measurement = gen.Measurement()
    observed = _observe(TRUTH, measurement, substrate)
    start = [400.0, 1.45, 0.005]

    classical = bl.fit_least_squares(
        observed, WAVELENGTHS, start, measurement=measurement, truth=TRUTH
    )
    descent = bl.fit_autograd(observed, WAVELENGTHS, start, measurement=measurement, truth=TRUTH)

    assert np.allclose(classical.parameters, descent.parameters, rtol=1e-6)


@pytest.mark.parametrize("true_thickness", [65.0, 900.0, 1500.0])
def test_exact_gradients_do_not_fix_the_fringe_ambiguity(true_thickness, substrate):
    """The finding this ticket was worth running for.

    It would be easy to assume the classical fit misses the right fringe because
    its finite-differenced Jacobian is approximate. It does not. Handed the exact
    derivative, the descent walks into a wrong basin from the same cold start and
    reports the same kind of confident wrong number — at 900 nm the two baselines
    even agree with each other, both landing near 316 nm.

    §5.2(b)'s ambiguity is a property of the *cost surface*, not of how the
    gradient was obtained, so no improvement in gradient quality can remove it.
    Only a different search can, which is why §6 prescribes multi-start and why
    DTFM-032 exists.
    """
    measurement = gen.Measurement()
    truth = np.array([true_thickness, 1.46, 0.004])
    observed = _observe(truth, measurement, substrate)
    start = [300.0, 1.47, 0.005]

    classical = bl.fit_least_squares(
        observed, WAVELENGTHS, start, measurement=measurement, truth=truth
    )
    descent = bl.fit_autograd(observed, WAVELENGTHS, start, measurement=measurement, truth=truth)

    assert abs(classical.thickness_error_nm) > 10.0
    assert abs(descent.thickness_error_nm) > 10.0, "exact gradients do not rescue it"
    assert descent.residual_rms > 1e3 * 1e-9, "and it says so, loudly"


def test_adam_settles_at_its_step_size_rather_than_at_the_answer(substrate):
    """Why the default is a line search, stated as a measured limitation.

    Adam's step length is set by its learning rate, not by the gradient, so near
    a minimum it does not shrink its stride — it circles at a radius of roughly
    one step. Decaying the rate lowers that radius but never removes it. From the
    same warm start where L-BFGS reaches ~1e-9 nm, Adam stops around 0.1-3 nm
    even after 2000 iterations, and correctly reports that it never converged.

    This matters beyond this module: Adam is what §7.3 trains the network with.
    A floor set by the optimiser rather than by the data is exactly the kind of
    error that would later be misread as the network's own accuracy limit.
    """
    measurement = gen.Measurement()
    observed = _observe(TRUTH, measurement, substrate)
    start = [440.0, 1.45, 0.005]

    line_search = bl.fit_autograd(
        observed, WAVELENGTHS, start, measurement=measurement, truth=TRUTH
    )
    first_order = bl.fit_autograd(
        observed,
        WAVELENGTHS,
        start,
        measurement=measurement,
        truth=TRUTH,
        optimiser="adam",
        max_iterations=600,
    )

    assert abs(line_search.thickness_error_nm) < 1e-6
    assert abs(first_order.thickness_error_nm) > 1e-3, "a floor, not convergence"
    assert abs(first_order.thickness_error_nm) < 20.0, "but the right basin"
    assert not first_order.success


def test_descent_stays_inside_the_prior_support(substrate):
    """Projection after each step, for the same reason DTFM-030 takes bounds:
    §10's comparison is unfair if either estimator may search where the other
    cannot.
    """
    prior = gen.Prior()
    measurement = gen.Measurement()

    for optimiser in ("lbfgs", "adam"):
        result = bl.fit_autograd(
            _observe(TRUTH, measurement, substrate),
            WAVELENGTHS,
            [10.0, 1.0, 0.0],
            measurement=measurement,
            prior=prior,
            optimiser=optimiser,
            max_iterations=50,
        )
        assert prior.thickness_nm[0] <= result.parameters[0] <= prior.thickness_nm[1]
        assert prior.cauchy_a[0] <= result.parameters[1] <= prior.cauchy_a[1]
        assert prior.cauchy_b[0] <= result.parameters[2] <= prior.cauchy_b[1]


def test_autograd_descent_is_reproducible(substrate):
    """§15. No sampling anywhere in this fitter, so it must repeat bit for bit."""
    measurement = gen.Measurement()
    observed = _observe(TRUTH, measurement, substrate)

    for optimiser in ("lbfgs", "adam"):
        first = bl.fit_autograd(
            observed,
            WAVELENGTHS,
            [400.0, 1.45, 0.005],
            measurement=measurement,
            optimiser=optimiser,
            max_iterations=60,
        )
        second = bl.fit_autograd(
            observed,
            WAVELENGTHS,
            [400.0, 1.45, 0.005],
            measurement=measurement,
            optimiser=optimiser,
            max_iterations=60,
        )
        assert np.array_equal(first.parameters, second.parameters), optimiser
        assert first.function_evaluations == second.function_evaluations


def test_an_unknown_optimiser_is_refused():
    with pytest.raises(ValueError, match="lbfgs"):
        bl.fit_autograd(np.zeros(400), WAVELENGTHS, [400.0, 1.45, 0.005], optimiser="sgd")
