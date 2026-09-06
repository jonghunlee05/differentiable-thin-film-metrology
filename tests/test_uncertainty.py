"""Identifiability — what the measurement can know, before any estimator exists.

Spec §5.3.
Implemented by DTFM-034.

§5.3: "The CRB is the theoretical floor on the variance of any unbiased
estimator for this measurement design. If your estimator sits near the bound,
the algorithm is fine and the *measurement* is the limit — a conclusion worth
stating out loud."
"""

import numpy as np
import pytest
import torch

from src import baseline as bl
from src import dispersion as dp
from src import generate as gen
from src import uncertainty as un

WAVELENGTHS = np.linspace(400.0, 800.0, 200)
TRUTH = np.array([420.0, 1.46, 0.004])


@pytest.fixture(scope="module")
def substrate() -> torch.Tensor:
    n, k = dp.load_nk("Si", WAVELENGTHS)
    return torch.tensor(n + 1j * k)


# --- the algebra --------------------------------------------------------------


def test_the_fisher_matrix_is_symmetric_and_positive_semidefinite(substrate):
    """``F = JᵀJ/σ²`` is a Gram matrix, so this is structural rather than empirical.

    Worth asserting anyway: it is the cheapest possible detector for a Jacobian
    that has been transposed, mis-shaped, or silently broadcast — none of which
    would raise, and all of which would produce a plausible wrong bound.
    """
    result = un.identifiability(TRUTH, WAVELENGTHS, substrate=substrate)

    assert result.fisher.shape == (3, 3)
    assert np.allclose(result.fisher, result.fisher.T, rtol=0, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(result.fisher) > 0)


def test_the_bound_scales_with_the_instrument_noise(substrate):
    """``F ∝ 1/σ²`` and ``C ∝ σ²``, so the floor is linear in σ.

    This is what makes the bound an *instrument* statement: quote a tool's noise
    figure and the best achievable precision follows, with no estimator involved.
    """
    single = un.identifiability(TRUTH, WAVELENGTHS, substrate=substrate, sigma=1e-3)
    double = un.identifiability(TRUTH, WAVELENGTHS, substrate=substrate, sigma=2e-3)

    assert double.thickness_bound_nm == pytest.approx(2.0 * single.thickness_bound_nm, rel=1e-9)
    assert np.allclose(double.fisher, single.fisher / 4.0, rtol=1e-9)


def test_the_bound_and_the_fit_covariance_are_the_same_algebra(substrate):
    """DTFM-033 and DTFM-034 must agree where they overlap, or one is wrong.

    ``baseline.covariance_from_jacobian`` evaluates ``σ²(JᵀJ)⁻¹`` where a fit
    stopped; this evaluates it where the truth is. On noiseless data a converged
    fit *is* at the truth, so the two must coincide — and the two modules compute
    it through different code paths.

    They mean different things, which is why both exist. The fit covariance is a
    property of one estimate; the bound is a property of the measurement design
    and applies to estimators nobody has written yet, including the network.
    """
    measurement = gen.Measurement()
    observed = bl.forward_observable(TRUTH, WAVELENGTHS, measurement, substrate).numpy()
    fit = bl.fit_least_squares(observed, WAVELENGTHS, TRUTH, measurement=measurement, sigma=1e-3)
    bound = un.identifiability(TRUTH, WAVELENGTHS, substrate=substrate, sigma=1e-3)

    assert fit.thickness_sigma_nm == pytest.approx(bound.thickness_bound_nm, rel=1e-6)


def test_a_zero_or_negative_noise_is_refused():
    with pytest.raises(ValueError, match="sigma"):
        un.fisher_information(np.eye(3), 0.0)
    with pytest.raises(ValueError, match="sigma"):
        un.fisher_information(np.eye(3), -1e-3)


def test_efficiency_is_a_variance_ratio():
    assert un.efficiency(2.0, 2.0) == pytest.approx(1.0)
    assert un.efficiency(4.0, 2.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        un.efficiency(1.0, 0.0)


# --- the conclusion §5.3 asks for ---------------------------------------------


def test_the_classical_fit_sits_on_the_bound(substrate):
    """§5.3's "conclusion worth stating out loud", measured.

    If the estimator reaches the floor then no algorithm can beat it, and the
    only way to a better number is a better instrument. Measured by refitting
    noisy repeats and comparing the observed scatter against the bound.

    This is also what makes any later claim about the network meaningful. "The
    network reaches 0.03 nm" says nothing on its own; "the network reaches 0.9
    efficiency where the classical fit reaches 1.0" is a comparison.
    """
    measurement = gen.Measurement()
    sigma = 1e-3
    clean = bl.forward_observable(TRUTH, WAVELENGTHS, measurement, substrate).numpy()
    rng = np.random.default_rng(11)

    bound = un.identifiability(TRUTH, WAVELENGTHS, substrate=substrate, sigma=sigma)
    recovered = [
        bl.fit_least_squares(
            clean + rng.normal(0.0, sigma, clean.shape),
            WAVELENGTHS,
            TRUTH,
            measurement=measurement,
            sigma=sigma,
        ).parameters[0]
        for _ in range(80)
    ]
    observed = float(np.std(recovered, ddof=1))

    assert un.efficiency(observed, bound.thickness_bound_nm) > 0.6, (
        "the classical fit extracts most of the available information"
    )
    assert un.efficiency(observed, bound.thickness_bound_nm) < 1.6, (
        "and does not beat a bound it cannot beat"
    )


def test_ellipsometry_carries_more_information_than_reflectance(substrate):
    """§3's question with "a real number attached", now from the bound itself.

    §3 asks to "quantify with Fisher information *how much* the degeneracy
    shrinks moving from reflectance-only to full ellipsometry". DTFM-028 answered
    it with a bespoke script to make a scheduling decision; this is the same
    question asked by the module built for it.

    Compared at equal σ, so the difference is the observable rather than an
    instrument being handed a better noise figure.
    """
    sigma = 1e-3
    ell = un.identifiability(
        TRUTH, WAVELENGTHS, measurement=gen.Measurement(), substrate=substrate, sigma=sigma
    )
    ref = un.identifiability(
        TRUTH,
        WAVELENGTHS,
        measurement=gen.Measurement(observable="reflectance"),
        substrate=substrate,
        sigma=sigma,
    )

    assert ell.thickness_bound_nm < ref.thickness_bound_nm
    assert ref.thickness_bound_nm / ell.thickness_bound_nm > 5.0


def test_thick_films_lock_thickness_and_index_together(substrate):
    """§5.3's ρ prediction, which holds at the thick end and only there.

    "ρ between thickness and index will frequently exceed 0.99." For films past
    roughly 700 nm that is exactly right — the phase term dominates, only the
    optical path ``n·d`` is really being measured, and the two parameters trade
    off almost perfectly.

    At the thin end it is false, by an order of magnitude. That is
    ``test_the_correlation_oscillates_and_crosses_zero``.
    """
    for thickness in (900.0, 1500.0, 2000.0):
        result = un.identifiability(
            [thickness, 1.46, 0.004], WAVELENGTHS, substrate=substrate, sigma=1e-3
        )
        assert abs(result.thickness_index_correlation) > 0.98, thickness


def test_the_sweep_reports_every_column_it_promises():
    thicknesses = np.array([50.0, 200.0, 800.0])
    swept = un.sweep_thickness(thicknesses, WAVELENGTHS)

    for key in ("thickness_nm", "bound_nm", "relative_bound", "correlation", "condition"):
        assert key in swept
        assert swept[key].shape == thicknesses.shape
        assert np.all(np.isfinite(swept[key]))
    assert np.allclose(swept["relative_bound"], swept["bound_nm"] / thicknesses, rtol=1e-12)


def test_the_correlation_oscillates_and_crosses_zero(substrate):
    """§5.3's ρ claim is false at the thin end, and the failure has structure.

    §5.3 says ρ "will frequently exceed 0.99". Swept across the prior it does —
    above roughly 700 nm. Below 200 nm it *oscillates and crosses zero*, which
    the spec does not anticipate and which means thickness and index are, at
    particular thicknesses, almost independently determined.

    This is the anomaly parked as an open question since DTFM-026. It is real and
    reproducible. It is **not explained** — see ``Implementation-Notes.md`` §22
    for a quarter-wave hypothesis that predicted the band dependence perfectly
    and then failed on the angle dependence. The test pins the observation, not a
    mechanism.
    """
    thickness = np.arange(20.0, 300.0, 2.0)
    rho = un.sweep_thickness(thickness, WAVELENGTHS)["correlation"]
    crossings = thickness[np.flatnonzero(np.sign(rho[:-1]) != np.sign(rho[1:]))]

    assert crossings.size >= 2, "ρ changes sign more than once below 300 nm"
    assert np.min(np.abs(rho)) < 0.1, "and passes close to zero"
    assert np.max(np.abs(rho)) > 0.9, "while still reaching near-total degeneracy nearby"


def test_blue_light_suits_thin_films_and_red_light_suits_thick_ones():
    """A measurement-design result the bound gives away for free.

    Compared at equal noise and equal sample count, so only the band differs:

    | band | 100 nm film | 420 nm film |
    |---|---|---|
    | blue 400-600 | **2.0e-5** | 9.1e-5 |
    | red 600-800 | 1.1e-4 | **7.1e-6** |

    The winner swaps. Short wavelengths resolve thin films better and long ones
    resolve thick films better, which is what matching fringe spacing to
    thickness should do.

    The project keeps a fixed 400-800 nm band regardless, and that is deliberate:
    every result so far is measured on it, and tuning the band per film would
    flatter every estimator including the network. A fixed design that is
    good-everywhere and optimal-nowhere is the fairer benchmark.
    """
    blue = np.linspace(400.0, 600.0, 200)
    red = np.linspace(600.0, 800.0, 200)

    def bound(grid, thickness) -> float:
        return un.identifiability(
            [thickness, 1.46, 0.004], grid, sigma=1e-3
        ).relative_thickness_bound

    assert bound(blue, 100.0) < bound(red, 100.0), "blue wins on a thin film"
    assert bound(red, 420.0) < bound(blue, 420.0), "and red wins on a thicker one"


def test_a_rank_deficient_measurement_does_not_report_a_tiny_bound(substrate):
    """The regression for a real defect, and it is the project's own theme.

    ``cramer_rao_bound`` originally used ``np.linalg.pinv``. A pseudo-inverse
    handles a near-singular direction by *dropping* it, which sets that
    direction's variance to zero — so at thicknesses where ``F`` tips past the
    cutoff, the bound came back **eight orders of magnitude too good**:

        1836 nm   rank 3   CRB 3.2e-02 nm
        1844 nm   rank 2   CRB 3.5e-10 nm     <- the bug
        1860 nm   rank 3   CRB 3.4e-02 nm

    3.5e-10 nm is a precision far below any physical meaning, reported at exactly
    the thickness where the measurement had *stopped* being able to separate the
    parameters. Nothing raised. It surfaced only because a figure's summary line
    quoted a best-case number that was obviously absurd.

    The correct bound for a rank-deficient ``F`` is **infinite** — that direction
    is not identifiable. Flooring the eigenvalues instead of truncating them errs
    towards a large variance, which is the safe direction.

    Asserted against the neighbours rather than an absolute threshold: the bound
    must vary smoothly with thickness, because the physics does.
    """
    grid = np.array([1836.0, 1844.0, 1860.0])
    swept = un.sweep_thickness(grid, WAVELENGTHS)
    bounds = swept["bound_nm"]

    assert np.all(bounds > 1e-4), "no bound is absurdly, unphysically small"
    assert bounds[1] / bounds[0] > 0.1, "and the middle point is not orders below its neighbour"
    assert bounds[1] / bounds[2] < 10.0

    deficient = un.identifiability([1844.0, 1.46, 0.004], WAVELENGTHS, substrate=substrate)
    healthy = un.identifiability([1860.0, 1.46, 0.004], WAVELENGTHS, substrate=substrate)
    assert not deficient.identifiable, "and the rank deficiency is reported rather than hidden"
    assert healthy.identifiable


def test_the_bound_is_smooth_across_the_whole_prior():
    """The general form of the check above, and the one that would have caught it.

    A bound built from a pseudo-inverse fails at whichever thicknesses happen to
    tip past the cutoff, which is unpredictable in advance — so the useful test is
    not "1844 nm is fine" but "no thickness anywhere produces a bound wildly out
    of line with its neighbours". The underlying physics has no discontinuities,
    so neither should the floor derived from it.
    """
    grid = np.arange(1000.0, 2001.0, 4.0)
    bounds = un.sweep_thickness(grid, WAVELENGTHS)["bound_nm"]
    jumps = bounds[1:] / bounds[:-1]

    assert np.all(np.isfinite(bounds))
    assert bounds.min() > 1e-4, f"smallest bound {bounds.min():.2e} nm is unphysical"
    assert jumps.max() < 20.0 and jumps.min() > 0.05, "no order-of-magnitude cliffs"


def test_the_bound_oscillates_with_the_wavelength_grid_not_with_the_film():
    """A wobble that looks like physics and is arithmetic — DTFM-034.

    Sweeping the bound across thickness produces a visibly ragged curve above
    ~500 nm. It survives every attempt to explain it away as an error:

    - not round-off — two mathematically different inversions agree to 1e-12
    - not undersampling — at 0.5 nm resolution the swings are still 32%
    - not the material models — a constant film index on a constant substrate
      still swings 50%

    It is the **discrete wavelength grid**. As a film thickens its fringes drift
    along the wavelength axis, and each time one crosses a sample point the
    information total twitches. Measured:

    | wavelengths | grid step | oscillation period | swing |
    |---|---|---|---|
    | 100 | 4.04 nm | 4.11 nm | 78% |
    | 200 | 2.01 nm | 3.19 nm | 51% |
    | 400 | 1.00 nm | 1.59 nm | 28% |
    | 800 | 0.50 nm | 0.80 nm | 12% |

    Three doublings, three halvings of both period and amplitude.

    **Not a defect, and deliberately not smoothed away.** A real spectrometer
    also samples at fixed pixel wavelengths, so a real instrument's information
    content genuinely does twitch as fringes drift across its detector. What the
    table does say is that this project's 200 points sit at the coarse end of the
    "hundreds" a real tool uses, so the effect here is larger than it would be on
    real hardware.

    This also closes ``Implementation-Notes.md`` §21, which recorded ``σ ∝ 1/√m``
    misbehaving below ~400 points and attributed it to "a Riemann sum over an
    oscillating Jacobian" without pinning it down. Same effect, now measured.
    """
    thickness = np.arange(900.0, 916.0, 0.25)

    def period_of(points: int) -> float:
        grid = np.linspace(400.0, 800.0, points)
        bounds = un.sweep_thickness(thickness, grid)["bound_nm"]
        peaks = thickness[1:-1][(bounds[1:-1] > bounds[:-2]) & (bounds[1:-1] > bounds[2:])]
        return float(np.mean(np.diff(peaks)))

    coarse, fine = period_of(200), period_of(400)

    assert coarse / fine == pytest.approx(2.0, rel=0.35), (
        "halving the grid step halves the oscillation — it tracks the grid, not the film"
    )


def test_the_smoothing_matrix_matches_the_one_the_rest_of_the_project_uses(substrate):
    """``bandwidth_matrix`` exists so the slit function can sit inside an autograd
    graph. It is the same Gaussian ``noise.apply_spectrometer_bandwidth`` builds,
    and duplicated numerics drift, so the two are checked against each other.

    ``observable_with_bandwidth`` must also be *exactly*
    ``baseline.forward_observable`` when the width is zero — otherwise the two
    paths could diverge silently for the case the bound actually ships with.
    """
    from src import noise as nz

    grid = np.linspace(400.0, 800.0, 60)
    matrix = un.bandwidth_matrix(grid, 3.0)
    values = np.sin(grid / 7.0)

    assert un.bandwidth_matrix(grid, 0.0) is None
    assert np.allclose(
        nz.apply_spectrometer_bandwidth(grid, values, 3.0),
        (matrix @ torch.as_tensor(values)).numpy(),
        rtol=0,
        atol=1e-15,
    )

    measurement = gen.Measurement()
    assert np.array_equal(
        un.observable_with_bandwidth(TRUTH, WAVELENGTHS, measurement, substrate, None).numpy(),
        bl.forward_observable(TRUTH, WAVELENGTHS, measurement, substrate).numpy(),
    )


def test_the_bound_is_optimistic_against_a_real_instrument(substrate):
    """A stated limitation, pinned so it cannot quietly stop being true.

    Real spectroscopic ellipsometers claim thickness repeatability "better than
    0.1 nm". This project's Cramér-Rao floor for a 420 nm film is ±0.006 nm —
    about **16x more optimistic**.

    Not a contradiction. The bound is photon-noise limited: it assumes the only
    error is detector noise. A tool's repeatability also carries stage
    positioning, temperature drift, and above all the fact that no optical model
    describes a real film exactly. §10's out-of-distribution work is about that
    last term.

    The test exists because the gap is the honest caveat on every efficiency
    number this module produces, and a caveat in prose decays.
    """
    bound = un.identifiability(TRUTH, WAVELENGTHS, substrate=substrate, sigma=1e-3)
    vendor_repeatability_nm = 0.1

    assert bound.thickness_bound_nm < vendor_repeatability_nm, (
        "the photon-limited floor is well below what real tools achieve"
    )
    assert vendor_repeatability_nm / bound.thickness_bound_nm > 10.0, (
        "by more than an order of magnitude, which is the caveat worth quoting"
    )
