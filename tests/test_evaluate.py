"""The evaluation protocol — §10's table, computed the same way for every method.

Spec §10.
Implemented by DTFM-036.

§10 opens with the sentence this module exists to enforce: "Every claim gets a
number, computed the same way for every method." These tests pin the *rules*, not
the results — the numbers change whenever a method improves, but what counts as
an error, a failure, or a second per inversion must not.
"""

import numpy as np
import pytest

from src import evaluate as ev
from src import generate as gen

WAVELENGTHS = np.linspace(400.0, 800.0, 200)


@pytest.fixture(scope="module")
def cases() -> list[ev.Case]:
    return ev.make_cases(6, seed=0, wavelengths_nm=WAVELENGTHS)


# --- the rules ----------------------------------------------------------------


def test_every_film_is_seen_at_both_noise_levels(cases):
    """The low-SNR and high-SNR columns are the *same* films, not two draws.

    That turns "error grows with noise" from a statement about two samples into a
    paired comparison on one. A method cannot look better at low noise merely by
    having been handed easier films there.
    """
    assert len(cases) == 6 * len(ev.SNR_LEVELS)

    by_snr = {snr: sorted(c.thickness_nm for c in cases if c.snr == snr) for snr in ev.SNR_LEVELS}
    first, *rest = by_snr.values()
    for other in rest:
        assert np.allclose(first, other), "each SNR column must see the same films"


def test_the_same_noise_draw_reaches_every_method(cases):
    """Observations are stored on the case, not regenerated per method.

    §10 compares methods; a comparison in which each estimator got its own random
    numbers measures the noise as much as the method. Storing the draw makes that
    structural rather than a convention someone has to remember.
    """
    first = ev.evaluate(cases[:2], method="cold_single")
    second = ev.evaluate(cases[:2], method="cold_single")

    assert np.array_equal(first.estimates, second.estimates)
    assert all(
        np.array_equal(a.observed, b.observed)
        for a, b in zip(first.cases, second.cases, strict=True)
    )


def test_the_regime_boundaries_are_the_physics_not_round_numbers():
    """100 nm and 700 nm are measurements, not tidy numbers.

    Below ~100 nm §5.2(c)'s quadratic insensitivity dominates. Above ~700 nm
    DTFM-034 measured ρ(d, n) locking past 0.99, so only the optical path n·d is
    really being determined. Both boundaries came from a measurement in an
    earlier ticket.
    """
    assert ev.regime_of(50.0) == "thin"
    assert ev.regime_of(99.9) == "thin"
    assert ev.regime_of(100.0) == "mid"
    assert ev.regime_of(699.9) == "mid"
    assert ev.regime_of(700.0) == "thick"
    assert ev.regime_of(2000.0) == "thick"


def test_a_failure_means_being_wrong_not_the_optimiser_being_unhappy(cases):
    """The definition that keeps the table honest.

    DTFM-030 established that a fit reports ``success`` while being 583 nm wrong.
    Counting that flag would report roughly zero failures for a method that is
    wrong most of the time from a cold start — which is exactly the number a
    later comparison would be tempted to quote.

    So ``failure_rate`` is a fact about the estimate, and the optimiser's own
    opinion is kept beside it as ``convergence_flag_rate``. The **gap** between
    the two columns is one of §10's findings, not an inconsistency to tidy away.
    """
    report = ev.Report(
        method="synthetic",
        cases=cases[:3],
        estimates=np.array([c.thickness_nm + 600.0 for c in cases[:3]]),
        seconds=np.full(3, 0.1),
        converged=np.ones(3, dtype=bool),  # the optimiser is delighted
        residuals=np.full(3, 1.0),
    )
    metrics = report.metrics()

    assert metrics["failure_rate"] == 1.0, "600 nm out is a failure"
    assert metrics["convergence_flag_rate"] == 0.0, "even though nothing was flagged"


def test_the_estimator_is_an_argument_so_the_network_needs_no_second_scorer(cases):
    """The structural point of the module, and the reason it exists before DTFM-039.

    Anything mapping a spectrum to three parameters is scorable here. The network
    at DTFM-044 satisfies this signature exactly as a fitter does, so it enters
    §10's table without a second scoring path being written — and a comparison
    between two separately-written scorers is a comparison nobody can check.
    """

    def pretend_network(case: ev.Case):
        return np.array([case.thickness_nm * 1.01, 1.46, 0.004]), True, 0.5

    report = ev.evaluate(cases[:4], pretend_network, method="pretend")

    assert report.method == "pretend"
    assert len(report.estimates) == 4
    assert np.all(report.seconds > 0.0)
    assert report.metrics()["median_abs_nm"] == pytest.approx(
        np.median([0.01 * c.thickness_nm for c in cases[:4]]), rel=1e-9
    )


def test_the_table_covers_every_cell_that_has_films(cases):
    report = ev.evaluate(cases, method="cold_single")
    table = report.table()

    assert table[0]["regime"] == "all" and table[0]["snr"] == "all"
    assert table[0]["n"] == len(cases)
    assert all(row["n"] > 0 for row in table), "empty cells are dropped, not reported as zero"
    assert sum(row["n"] for row in table[1:]) == len(cases), "cells partition the cases"


def test_an_unknown_method_is_refused(cases):
    with pytest.raises(ValueError, match="method"):
        ev.evaluate(cases[:1], method="telepathy")


def test_a_thickness_outside_every_regime_is_refused():
    with pytest.raises(ValueError, match="regime"):
        ev.regime_of(-1.0)


# --- what the record actually shows -------------------------------------------


def test_medians_hide_the_failures_that_rmse_exposes(cases):
    """Why §10 asks for more than one metric, demonstrated rather than asserted.

    Measured on the real record: every method — including one wrong 10% of the
    time — reports a median absolute error of 0.033-0.036 nm. By median they are
    indistinguishable. By RMSE they differ by 100x and by failure rate by 10x.

    A single headline median would have made a broken method look identical to a
    working one, which is the failure this test exists to make visible.
    """
    truth = np.array([c.thickness_nm for c in cases[:4]])
    tiny = truth + np.array([0.03, -0.03, 0.03, -0.03])
    catastrophic = truth + np.array([0.03, -0.03, 0.03, 900.0])

    def report_for(estimates):
        return ev.Report(
            method="x",
            cases=cases[:4],
            estimates=estimates,
            seconds=np.full(4, 0.1),
            converged=np.ones(4, dtype=bool),
            residuals=np.full(4, 1e-9),
        ).metrics()

    good, bad = report_for(tiny), report_for(catastrophic)

    assert bad["median_abs_nm"] == pytest.approx(good["median_abs_nm"], rel=0.5), (
        "the medians are effectively the same"
    )
    assert bad["rmse_nm"] > 100 * good["rmse_nm"], "while the RMSE differs by orders"
    assert bad["failure_rate"] > good["failure_rate"], "and the failure rate catches it"


def test_knowing_the_thickness_is_not_enough_for_a_thick_film():
    """DTFM-034's ρ(d, n) > 0.99, as an operational failure rate.

    ``warm_thickness_only`` starts at the true thickness with the index pinned at
    the prior's midpoint. It fails on **40%** of thick films and on **0%** of thin
    and mid ones. Past 700 nm thickness and index are nearly the same parameter,
    so an index wrong by 0.07 *is* a thickness wrong by hundreds of nanometres —
    and being handed the exact thickness does not rescue it.

    Kept in the record for that reason: it is §5.2(a)'s degeneracy shown as a
    number a practitioner would act on, rather than as a correlation coefficient.
    """
    prior = gen.Prior()
    thick = ev.Case(
        truth=np.array([1400.0, 1.53, 0.004]),
        observed=np.zeros(2 * WAVELENGTHS.size),
        wavelengths_nm=WAVELENGTHS,
        sigma=1e-3,
        snr="high",
    )
    assert thick.regime == "thick"
    assert abs(thick.truth[1] - 1.46) > 0.05, (
        "the premise: the pinned start is far from this film's index"
    )
    assert prior.cauchy_a[0] <= thick.truth[1] <= prior.cauchy_a[1], (
        "and that index is inside the prior, so it is a film the network will meet"
    )
