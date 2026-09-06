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
def test_a_cold_start_converges_to_a_wrong_minimum(true_thickness, substrate):
    """The failure the whole baseline exists to expose, on the first estimator built.

    The cost surface has many local minima. Started at 300 nm, the fit walks into
    whichever basin it lands in and stops. It reports ``success`` while being
    hundreds of nanometres wrong.

    Originally written as "converges to the wrong *fringe*", citing §5.2(b). That
    label was checked later and does not hold — the minima are 4 nm apart where a
    fringe is 265 nm, and this test's own landing point at 316.96 nm is 53 nm from
    the nearest predicted alias. The behaviour is real and the numbers are
    unchanged; only the name was wrong. See ``Implementation-Notes.md`` §18 and
    ``test_broadband_data_replaces_few_deep_minima_with_many_shallow_ones``.

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
def test_exact_gradients_do_not_fix_the_multimodality(true_thickness, substrate):
    """The finding this ticket was worth running for.

    It would be easy to assume the classical fit misses the right minimum because
    its finite-differenced Jacobian is approximate. It does not. Handed the exact
    derivative, the descent walks into a wrong basin from the same cold start and
    reports the same kind of confident wrong number — at 900 nm the two baselines
    even agree with each other, both landing near 316 nm.

    The multimodality is a property of the *cost surface*, not of how the gradient
    was obtained, so no improvement in gradient quality can remove it. Only a
    different search can, which is why §6 prescribes multi-start and why DTFM-032
    exists.
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


# --- DTFM-032: multi-start, and the landscape it surveys ---------------------


def test_the_start_grid_spans_the_prior_without_sitting_on_its_bounds():
    """Endpoints are pulled inward deliberately: a start on a bound sits exactly
    where the projection clamps, and the fit can stall there rather than descend.
    """
    prior = gen.Prior()
    grid = bl.multi_start_grid(12, prior)

    assert grid.size == 12
    assert np.all(np.diff(grid) > 0)
    assert prior.thickness_nm[0] < grid[0] < grid[-1] < prior.thickness_nm[1]
    assert grid[0] - prior.thickness_nm[0] > 1.0
    assert prior.thickness_nm[1] - grid[-1] > 1.0


def test_the_start_grid_is_uniform_in_thickness_not_in_the_prior_spacing():
    """The substance of ``multi_start_grid``, and it is counter-intuitive.

    The prior is log-uniform because §7.1 weights each thickness decade equally.
    Seeding the *search* the same way is the obvious move and it is wrong: what a
    start must do is land inside the **basin of attraction** of the true minimum,
    and those are spread evenly in thickness rather than in its logarithm.

    The threshold below is the narrowest basin actually measured — 110 nm, on a
    420 nm film — not a fringe period. Both grids here carry the same number of
    starts, so this compares spacing alone: the uniform grid fits inside the
    narrowest basin, the log grid leaves a hole twice that wide. An earlier
    version asserted against 268 nm on the theory that minima sit one fringe apart. They do not;
    they are 4 nm apart (``Implementation-Notes.md`` §18), and asserting against
    that number would have passed while meaning nothing.
    """
    prior = gen.Prior()
    assert prior.spacing == "log-uniform", "the premise of this test"

    narrowest_basin_nm = 110.0
    uniform = bl.multi_start_grid(prior=prior)
    logarithmic = bl.multi_start_grid(prior=prior, spacing="log-uniform")

    assert np.max(np.diff(uniform)) < narrowest_basin_nm, (
        "no measured basin fits between two starts"
    )
    assert np.max(np.diff(logarithmic)) > 2 * narrowest_basin_nm, (
        "while the log grid, at the same cost, leaves a gap two basins wide"
    )


def test_the_default_start_count_is_the_one_that_was_measured():
    """§15's habit, applied to a default rather than to a result.

    The default was 12, justified by an argument later shown to be false. It is
    now 20, from two agreeing lines of evidence: the narrowest measured basin is
    110 nm, requiring ``count ≥ 1980 × 0.96 / 110 ≈ 18``; and on 40 films drawn
    from the prior, 8 starts recover 29 of 40 while 12 and above recover all of
    them. 12 works but sits exactly on the observed boundary, so the default
    carries margin rather than sitting at the edge of what was tested.
    """
    prior = gen.Prior()
    grid = bl.multi_start_grid(prior=prior)

    assert grid.size == 20
    assert np.max(np.diff(grid)) < 110.0, "inside the narrowest basin"
    assert bl.multi_start_grid(8, prior).max() > 65.0, (
        "and the count that failed did so because its first start sat above the "
        "thin films it missed — every one of them under 85 nm"
    )


@pytest.mark.parametrize("true_thickness", [30.0, 65.0, 420.0, 900.0, 1500.0])
def test_multi_start_recovers_films_a_single_cold_start_cannot(true_thickness, substrate):
    """DTFM-032's acceptance criterion, and the repair of DTFM-030's failure.

    A single fit cold-started at 300 nm reports ``success`` while being hundreds
    of nanometres wrong, and DTFM-031 showed exact gradients do not help because
    the ambiguity is in the cost surface. Surveying it does help: twelve starts
    across the prior, keep the lowest cost.
    """
    measurement = gen.Measurement()
    truth = np.array([true_thickness, 1.46, 0.004])
    observed = _observe(truth, measurement, substrate)

    result = bl.fit_multi_start(observed, WAVELENGTHS, measurement=measurement, truth=truth)

    assert abs(result.thickness_error_nm) < 1e-3
    assert np.allclose(result.parameters, truth, rtol=1e-5)


def test_uniform_starts_beat_log_starts_at_the_same_cost(substrate):
    """The measurement behind the default, not an appeal to the docstring.

    Same number of fits, same films, same everything but the spacing of the
    starting guesses. Changing the spacing recovers films that doubling a log
    grid does not.
    """
    measurement = gen.Measurement()
    films = (30.0, 65.0, 150.0, 420.0, 900.0, 1500.0, 1900.0)

    recovered = {}
    for spacing in ("uniform", "log-uniform"):
        hits = 0
        for thickness in films:
            truth = np.array([thickness, 1.46, 0.004])
            result = bl.fit_multi_start(
                _observe(truth, measurement, substrate),
                WAVELENGTHS,
                count=12,
                spacing=spacing,
                measurement=measurement,
                truth=truth,
            )
            hits += abs(result.thickness_error_nm) < 1e-3
        recovered[spacing] = hits

    assert recovered["uniform"] == len(films)
    assert recovered["log-uniform"] < len(films)


def test_the_losing_fits_are_kept(substrate):
    """§6: "record where each converges. That landscape *is* the fringe-order
    ambiguity made visible."

    The losers are the evidence. They are what the figure is drawn from, and
    DTFM-033 needs them to say whether an error bar at the winner means anything
    with a rival minimum hundreds of nanometres away.
    """
    measurement = gen.Measurement()
    truth = np.array([900.0, 1.46, 0.004])

    result = bl.fit_multi_start(
        _observe(truth, measurement, substrate),
        WAVELENGTHS,
        count=12,
        measurement=measurement,
        truth=truth,
    )
    centres, counts = result.basins()

    assert len(result.fits) == 12
    assert result.starts.size == 12
    assert result.thicknesses.size == 12
    assert centres.size > 1, "the ambiguity is real, and this records it"
    assert counts.sum() == 12
    assert result.wall_clock_s > 0.0


def test_multi_start_wins_on_depth_rather_than_on_odds(substrate):
    """Why taking the lowest cost is safe even though most starts fail.

    Only about one start in eight reaches the truth. That would be a weak
    procedure if the winner were hard to identify — but on noiseless data the
    true basin is roughly 1e26 times deeper than the best rival, so the choice is
    not a close call. The margin, not the hit rate, is what makes this work, and
    it is the margin that noise will erode. §8's calibration and DTFM-033's
    covariance are about how far.
    """
    measurement = gen.Measurement()
    truth = np.array([900.0, 1.46, 0.004])

    result = bl.fit_multi_start(
        _observe(truth, measurement, substrate),
        WAVELENGTHS,
        starts=np.linspace(20.0, 2000.0, 41),
        measurement=measurement,
        truth=truth,
    )

    assert result.success_fraction < 0.5, "most starts do not find it"
    landed = result.thicknesses
    runner_up = np.min(result.costs[np.abs(landed - truth[0]) > 1.0])

    assert abs(result.thickness_error_nm) < 1e-3, "yet the answer is right"
    assert runner_up > 1e12 * max(result.best.cost, 1e-300), "because it is unmistakable"


def test_multi_start_costs_what_it_looks_like_it_costs(substrate):
    """§11's amortisation argument, made concrete.

    A correct classical answer is not one fit, it is ``count`` fits. This is the
    honest per-site number the network is asked to beat, and it is an order of
    magnitude above the single-fit figure DTFM-030 recorded.
    """
    measurement = gen.Measurement()
    truth = np.array([900.0, 1.46, 0.004])
    observed = _observe(truth, measurement, substrate)

    single = bl.fit_least_squares(
        observed, WAVELENGTHS, [880.0, 1.46, 0.004], measurement=measurement
    )
    sweep = bl.fit_multi_start(observed, WAVELENGTHS, count=12, measurement=measurement)

    assert sweep.wall_clock_s > 4.0 * single.wall_clock_s
    assert sum(fit.function_evaluations for fit in sweep.fits) > single.function_evaluations


def test_multi_start_drives_either_baseline(substrate):
    """The survey is a search strategy, not a property of one optimiser — so both
    DTFM-030's and DTFM-031's fitters plug into it, and both should land on the
    same film.

    ``count`` stays at the default rather than being trimmed to make the test
    quick: six starts sit 317 nm apart, wider than the widest measured basin, and
    both methods then miss the film. That is the grid derivation asserting itself
    rather than a flaky test.
    """
    measurement = gen.Measurement()
    truth = np.array([420.0, 1.46, 0.004])
    observed = _observe(truth, measurement, substrate)

    classical = bl.fit_multi_start(observed, WAVELENGTHS, measurement=measurement, truth=truth)
    descent = bl.fit_multi_start(
        observed, WAVELENGTHS, measurement=measurement, truth=truth, method="autograd"
    )

    assert abs(classical.thickness_error_nm) < 1e-3
    assert abs(descent.thickness_error_nm) < 1e-3


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="count"):
        bl.multi_start_grid(0)
    with pytest.raises(ValueError, match="spacing"):
        bl.multi_start_grid(4, spacing="geometric")
    with pytest.raises(ValueError, match="method"):
        bl.fit_multi_start(np.zeros(400), WAVELENGTHS, count=2, method="newton")


def test_broadband_data_replaces_few_deep_minima_with_many_shallow_ones(substrate):
    """What the cold-start failures of DTFM-030 to 032 actually are.

    All three tickets called them §5.2(b) fringe-order ambiguity. That is the
    *single-wavelength* picture: R is periodic in ``n·d``, so aliases sit one
    fringe apart, ``λ / (2 n cos θ_t) ≈ 265 nm`` here. Measured against the real
    cost surface, it does not hold — the minima are 4 nm apart, not 265, and the
    fit's landing point at 316.96 nm is 53 nm from the nearest predicted alias.

    The cause is broadband. Each wavelength has its own fringe period (177 nm at
    400 nm, 353 nm at 800 nm), and summing 200 of them produces a beat landscape
    far denser than any one of them. Narrow the band or thin the sampling and the
    textbook picture returns:

    | band | points | minima | median spacing |
    |---|---|---|---|
    | 599-601 nm | 3 | 15 | 132 nm |
    | 550-650 nm | 50 | 53 | 6.4 nm |
    | 400-800 nm | 200 | 279 | 4.2 nm |

    This is not a defect. It is the bargain broadband makes, and it cuts both
    ways: the false minima get *shallow* as they get dense, which is why the true
    minimum ends up ~1e26 deeper and why multi-start works on depth. §5.2(b)'s
    ambiguity has not been removed by using a spectrum — it has been traded for a
    rougher surface with an unmistakable global answer.
    """
    grid = np.linspace(20.0, 2000.0, 4000)
    truth = np.array([900.0, 1.46, 0.004])

    def minima_of(measurement) -> np.ndarray:
        observed = _observe(truth, measurement, substrate)
        cost = np.empty(grid.size)
        with torch.no_grad():
            for i, thickness in enumerate(grid):
                model = bl.forward_observable(
                    [thickness, truth[1], truth[2]], WAVELENGTHS, measurement, substrate
                ).numpy()
                residual = bl.wrapped_residual(model, observed, measurement.observable)
                cost[i] = 0.5 * np.sum(residual**2)
        interior = (cost[1:-1] < cost[:-2]) & (cost[1:-1] < cost[2:])
        return grid[1:-1][interior]

    single_fringe_nm = 265.0
    broadband = minima_of(gen.Measurement())
    reflectance = minima_of(gen.Measurement(observable="reflectance"))

    assert np.median(np.diff(broadband)) < 0.2 * single_fringe_nm, (
        "the minima are far denser than one per fringe"
    )
    assert broadband.size > 5 * reflectance.size, (
        "and the phase observable is much rougher than the intensity one"
    )


def test_picking_the_deepest_basin_survives_realistic_noise(substrate):
    """The claim multi-start rests on, checked rather than assumed.

    Choosing the lowest-cost fit is only sound while the true basin is clearly
    the deepest, and noise fills basins in. That much was stated in DTFM-032
    without being measured, which is the habit
    ``Implementation-Notes.md`` §19 is about. Measured:

    | ellipsometer σ (rad) | films correct | median error |
    |---|---|---|
    | 0 | 5 of 5 | 0.0000 nm |
    | 1e-3 (realistic) | 5 of 5 | 0.005 nm |
    | 1e-2 | 5 of 5 | 0.09 nm |
    | 3e-2 | 4 of 5 | 0.20 nm |

    The margin is far more robust than the original wording implied — it takes
    roughly 30x a real instrument's noise before a wrong basin is ever chosen. The
    error still grows with noise, which is what DTFM-033's covariance has to
    report honestly.
    """
    measurement = gen.Measurement()
    rng = np.random.default_rng(0)

    errors = {}
    for sigma in (1e-3, 1e-2):
        worst = 0.0
        for thickness in (65.0, 420.0, 1500.0):
            truth = np.array([thickness, 1.46, 0.004])
            observed = _observe(truth, measurement, substrate)
            noisy = observed + rng.normal(0.0, sigma, observed.shape)
            result = bl.fit_multi_start(
                noisy, WAVELENGTHS, count=12, measurement=measurement, truth=truth
            )
            worst = max(worst, abs(result.thickness_error_nm))
        errors[sigma] = worst

    assert errors[1e-3] < 0.1, "at realistic noise the right basin is still chosen"
    assert errors[1e-2] < 2.0, "and it degrades gracefully rather than jumping basins"
    assert errors[1e-2] > errors[1e-3], "error grows with noise, as it must"


# --- DTFM-033: the error bar §15 calls non-negotiable -------------------------


def test_no_fit_is_returned_without_an_error_bar(substrate):
    """DTFM-033's acceptance criterion, and §15's blunt one.

    "Never report a fit without an error bar. In this field that is a
    disqualifying instinct rather than an oversight."

    So it is not an opt-in argument. Every path that produces a
    :class:`bl.FitResult` produces the covariance with it, and multi-start
    forwards the winner's.
    """
    measurement = gen.Measurement()
    observed = _observe(TRUTH, measurement, substrate)
    start = [415.0, 1.46, 0.004]

    results = [
        bl.fit_least_squares(observed, WAVELENGTHS, start, measurement=measurement),
        bl.fit_autograd(observed, WAVELENGTHS, start, measurement=measurement),
        bl.fit_multi_start(observed, WAVELENGTHS, count=6, measurement=measurement),
    ]

    for result in results:
        assert result.covariance is not None
        assert result.covariance.shape == (3, 3)
        assert result.standard_errors is not None
        assert np.all(result.standard_errors > 0)
        assert np.all(np.isfinite(result.standard_errors))
        assert result.thickness_sigma_nm > 0
        assert result.correlation is not None
        assert result.condition_number is not None


def test_the_jacobian_is_the_exact_one(substrate):
    """§5.3's ``J[i,k] = ∂f(λ_i)/∂θ_k``, taken from autograd rather than differences.

    ``scipy`` returns its own approximate ``jac`` at the solution and using it
    would have been less code. It is built from a differencing step, so the error
    bar would carry an arbitrary constant chosen by the optimiser's internals.
    DTFM-031 established the model can hand over the exact derivative.
    """
    measurement = gen.Measurement()
    theta = np.array([420.0, 1.46, 0.004])
    jacobian = bl.model_jacobian(theta, WAVELENGTHS, measurement, substrate)

    assert jacobian.shape == (2 * WAVELENGTHS.size, 3)

    steps = np.array([1e-5, 1e-8, 1e-10])
    for k, step in enumerate(steps):
        shift = np.zeros(3)
        shift[k] = step
        with torch.no_grad():
            ahead = bl.forward_observable(
                theta + shift, WAVELENGTHS, measurement, substrate
            ).numpy()
            behind = bl.forward_observable(
                theta - shift, WAVELENGTHS, measurement, substrate
            ).numpy()
        numeric = (ahead - behind) / (2 * step)
        assert np.allclose(jacobian[:, k], numeric, rtol=1e-4, atol=1e-8), k


def test_the_error_bar_scales_with_the_noise_it_was_given(substrate):
    """``C = σ²(JᵀJ)⁻¹`` — so σ_θ is linear in σ, exactly.

    Trivial arithmetic, worth pinning because it is the property that makes the
    error bar mean something physical: quote a tool's noise figure and the
    uncertainty it implies follows, rather than being a number the optimiser
    happened to produce.
    """
    measurement = gen.Measurement()
    observed = _observe(TRUTH, measurement, substrate)
    start = [415.0, 1.46, 0.004]

    single = bl.fit_least_squares(observed, WAVELENGTHS, start, measurement=measurement, sigma=1e-3)
    double = bl.fit_least_squares(observed, WAVELENGTHS, start, measurement=measurement, sigma=2e-3)

    assert double.thickness_sigma_nm == pytest.approx(2.0 * single.thickness_sigma_nm, rel=1e-9)


def test_more_wavelengths_buy_precision_at_the_usual_rate(substrate):
    """σ_θ ∝ 1/√m — but only once the grid resolves the Jacobian, which is the
    part worth writing down.

    ``σ²(JᵀJ)⁻¹`` makes the rate exact for independent samples, so the law should
    hold everywhere. It does not, at low sampling. Measured over 400-800 nm with
    ``σ√m`` constant if the law holds:

    | points | σ_d (nm) | σ√m |
    |---|---|---|
    | 50 | 0.0144 | 0.102 |
    | 100 | 0.0054 | 0.054 |
    | 200 | 0.0057 | 0.080 |
    | 400 | 0.0022 | 0.044 |
    | 800 | 0.0017 | 0.048 |
    | 3200 | 0.0008 | 0.046 |

    Note 100 → 200: **more wavelengths made the predicted precision worse.** That
    cannot happen under the law. ``JᵀJ`` is a Riemann sum over a Jacobian that
    oscillates across the band, and at coarse sampling the sum's error oscillates
    with the grid rather than shrinking with it. Past a few hundred points it
    settles and the rate is clean — 800 → 3200 gives 2.07 against a predicted 2.

    Checked and *not* the tabulated substrate being re-interpolated onto each
    grid: replacing Si with a constant index leaves the same wobble.

    The practical reading is that a denser spectrometer buys the boring
    statistical rate, so the interesting levers are elsewhere — which is what
    DTFM-028 found by changing the *observable* and getting a factor of 111.
    """
    measurement = gen.Measurement()
    sigmas = {}
    for count in (800, 3200):
        grid = np.linspace(400.0, 800.0, count)
        n, k = dp.load_nk("Si", grid)
        local = torch.tensor(n + 1j * k)
        jacobian = bl.model_jacobian(TRUTH, grid, measurement, local)
        covariance, _ = bl.covariance_from_jacobian(
            jacobian, np.zeros(jacobian.shape[0]), sigma=1e-3
        )
        sigmas[count] = float(np.sqrt(covariance[0, 0]))

    assert sigmas[800] / sigmas[3200] == pytest.approx(2.0, rel=0.1)


def test_the_estimated_noise_recovers_the_noise_that_was_added(substrate):
    """When the instrument figure is unknown, ``s² = RSS/(m−n)`` stands in.

    It is only trustworthy while the model can represent the data. A systematic
    the model cannot fit inflates the residual, ``s`` absorbs it, and the error
    bar grows to *cover* a bias it should have been reporting — which is why
    DTFM-035's model selection exists and why ``sigma`` can be supplied instead.
    """
    measurement = gen.Measurement()
    rng = np.random.default_rng(3)
    added = 2e-3
    observed = _observe(TRUTH, measurement, substrate)
    noisy = observed + rng.normal(0.0, added, observed.shape)

    result = bl.fit_least_squares(noisy, WAVELENGTHS, [415.0, 1.46, 0.004], measurement=measurement)

    assert result.noise_sigma == pytest.approx(added, rel=0.15)


def test_a_wrong_fringe_fit_reports_a_confident_error_bar(substrate):
    """The limitation, stated as a test rather than as a caveat.

    The covariance is the curvature of the cost at the point the fit stopped, so
    it describes the width of *that* basin. It knows nothing about a rival 583 nm
    away. A cold-started fit therefore reports a small, tight error bar around a
    badly wrong answer — and it is right to, because the question it answers is
    "how precisely does the data pin the answer given this basin".

    What exposes the error is the residual, which is orders of magnitude larger.
    §10's thesis is that the classical method fails loudly; this test locates
    exactly *where* the loudness lives, and it is not in the error bar. Anyone
    quoting σ without also quoting the residual has the wrong instinct.
    """
    measurement = gen.Measurement()
    truth = np.array([900.0, 1.46, 0.004])
    observed = _observe(truth, measurement, substrate)

    good = bl.fit_least_squares(
        observed,
        WAVELENGTHS,
        [890.0, 1.46, 0.004],
        measurement=measurement,
        truth=truth,
        sigma=1e-3,
    )
    wrong = bl.fit_least_squares(
        observed,
        WAVELENGTHS,
        [300.0, 1.47, 0.005],
        measurement=measurement,
        truth=truth,
        sigma=1e-3,
    )

    assert abs(wrong.thickness_error_nm) > 100.0, "badly wrong"
    assert wrong.thickness_sigma_nm < 10.0, "and quietly confident about it"
    assert abs(wrong.thickness_error_nm) > 10.0 * wrong.thickness_sigma_nm, (
        "the error bar does not cover the error"
    )
    assert wrong.residual_rms > 100.0 * good.residual_rms, (
        "the residual is what gives it away, not the covariance"
    )


def test_the_error_bar_matches_the_observed_scatter(substrate):
    """The test that decides whether any of this means anything.

    ``C = σ²(JᵀJ)⁻¹`` is a linearisation. It is only an error bar if refitting
    noisy repeats actually scatters by that much. Measured over 200 repeats per
    film at an ellipsometer σ of 1e-3 rad:

    | film | predicted σ_d | observed σ_d | ratio |
    |---|---|---|---|
    | 65 nm | 0.01136 | 0.01110 | 0.98 |
    | 150 nm | 0.00069 | 0.00064 | 0.93 |
    | 420 nm | 0.00569 | 0.00551 | 0.97 |
    | 900 nm | 0.02490 | 0.02668 | 1.07 |
    | 1500 nm | 0.02486 | 0.02455 | 0.99 |

    Agreement to within 7% across a 23-fold range of thickness. The Gaussian
    approximation is doing its job here, which is a statement about *this*
    measurement being well-conditioned near the truth — not a general licence.
    The same number is meaningless once the fit is in the wrong basin, which
    ``test_a_wrong_fringe_fit_reports_a_confident_error_bar`` shows.
    """
    measurement = gen.Measurement()
    truth = np.array([420.0, 1.46, 0.004])
    sigma = 1e-3
    clean = _observe(truth, measurement, substrate)
    rng = np.random.default_rng(1)

    predicted = bl.fit_least_squares(
        clean, WAVELENGTHS, truth, measurement=measurement, sigma=sigma
    ).thickness_sigma_nm
    recovered = [
        bl.fit_least_squares(
            clean + rng.normal(0.0, sigma, clean.shape),
            WAVELENGTHS,
            truth,
            measurement=measurement,
            sigma=sigma,
        ).parameters[0]
        for _ in range(80)
    ]
    observed = float(np.std(recovered, ddof=1))

    assert observed / predicted == pytest.approx(1.0, abs=0.3)


@pytest.mark.parametrize(
    ("thickness", "floor", "ceiling"),
    [(150.0, 0.0, 0.4), (900.0, 0.95, 1.0), (1500.0, 0.95, 1.0)],
)
def test_thickness_and_index_correlate_more_as_the_film_thickens(
    thickness, floor, ceiling, substrate
):
    """§5.3's ρ, and a correction to what the spec predicts about it.

    §5.3: "ρ between thickness and index will frequently exceed 0.99. Reporting
    that alongside the fitted value is what separates a metrologist from someone
    who called ``curve_fit``."

    True for thick films, and false for thin ones. Measured at σ = 1e-3:

    | film | ρ(d, A) | condition number |
    |---|---|---|
    | 65 nm | −0.599 | 4.7e5 |
    | **150 nm** | **−0.086** | 1.1e6 |
    | 420 nm | −0.746 | 9.0e8 |
    | 900 nm | −0.987 | 9.4e9 |
    | 1500 nm | −0.996 | 8.2e10 |

    The near-zero at 150 nm is the anomaly ``Implementation-Notes.md`` parked as
    an open question under DTFM-026, now reproduced with tooling built for it
    rather than an ad-hoc script. There is a thickness where ``d`` and ``n``
    become almost independently determined, and it sits in the range this project
    is written about. DTFM-034 locates it properly; what this test fixes is that
    the effect is real and not an artefact of the earlier script.
    """
    measurement = gen.Measurement()
    truth = np.array([thickness, 1.46, 0.004])
    result = bl.fit_least_squares(
        _observe(truth, measurement, substrate),
        WAVELENGTHS,
        truth,
        measurement=measurement,
        sigma=1e-3,
    )

    assert floor <= abs(float(result.correlation[0, 1])) <= ceiling


def test_the_condition_number_grows_with_thickness(substrate):
    """Why the inverse is a pseudo-inverse with an explicit cutoff.

    ``JᵀJ`` runs from 1e5 to 1e11 over the prior's range. At the top of that,
    ``np.linalg.inv`` returns large finite numbers with nothing raised, and the
    resulting error bar looks like a measurement rather than like numerical
    noise. §5.2's degeneracies are the reason, and this is where they show up
    first — before they show up in a wrong answer.
    """
    measurement = gen.Measurement()
    conditions = []
    for thickness in (65.0, 900.0, 1500.0):
        truth = np.array([thickness, 1.46, 0.004])
        result = bl.fit_least_squares(
            _observe(truth, measurement, substrate),
            WAVELENGTHS,
            truth,
            measurement=measurement,
            sigma=1e-3,
        )
        conditions.append(result.condition_number)

    assert conditions[0] < conditions[1] < conditions[2]
    assert conditions[-1] > 1e9, "and the thick end is genuinely ill-conditioned"
