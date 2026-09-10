"""Where the physics predicts failure — DTFM-053, spec §5.2.

§5.2's three degeneracies each predict a *shape* in the identifiability analysis
rather than a value. These check the shapes are there, so that a later claim
about a threshold rests on a curve that behaves as the physics says it must.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from src import generate as gen
from src import uncertainty as un

WAVELENGTHS = np.linspace(400.0, 800.0, 200)

_spec = importlib.util.spec_from_file_location(
    "degeneracy_probe",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "degeneracy_probe.py",
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


@pytest.fixture(scope="module")
def sweep():
    grid = np.geomspace(1.0, 2000.0, 160)
    return grid, un.sweep_thickness(grid, WAVELENGTHS, prior=gen.Prior())


def test_thickness_and_index_are_almost_perfectly_degenerate_in_the_thin_limit(sweep):
    """§5.2(a) and (c) are the same phenomenon at the thin end.

    At a single wavelength the measurement constrains only the product ``n·d``.
    Dispersion breaks that partially — but for ``d ≪ λ`` there is barely any
    fringe structure to disperse, so the correlation returns to −1 and the two
    parameters stop being separable at all.
    """
    grid, curve = sweep
    rho = np.abs(curve["correlation"])

    assert rho[0] > 0.999, "at 1 nm of thickness the two parameters are one parameter"
    assert rho[np.argmin(np.abs(grid - 150.0))] < 0.9, "and are separable by 150 nm"


def test_the_bound_becomes_a_large_fraction_of_the_film(sweep):
    """§5.2(c) in the form that matters to someone depositing a film.

    An absolute bound of 0.4 nm of thickness sounds tolerable until the film is
    1 nm thick. The relative bound is the honest statement, and it is what makes
    the thin regime a *failure* rather than merely a harder case.
    """
    grid, curve = sweep
    relative = curve["relative_bound"]

    assert relative[0] > 0.1, "the bound at 1 nm exceeds 10% of the film"
    assert relative[np.argmin(np.abs(grid - 100.0))] < 0.001, "and is negligible by 100 nm"


def test_the_threshold_is_interpolated_not_snapped_to_the_grid():
    """The threshold is this ticket's headline number, so quoting a grid point
    would report the sampling density as a physical constant.
    """
    grid = np.array([10.0, 20.0, 30.0])
    curve = np.array([1.0, 0.8, 0.6])

    crossing = probe.threshold(grid, curve, 0.9)

    assert 10.0 < crossing < 20.0
    assert crossing == pytest.approx(15.0)


def test_a_curve_that_never_crosses_reports_nan():
    """Rather than the first or last grid point, either of which would read as a
    located threshold that was never found.
    """
    grid = np.array([10.0, 20.0, 30.0])

    assert np.isnan(probe.threshold(grid, np.array([1.0, 1.0, 1.0]), 0.5))


def test_the_degeneracy_threshold_sits_near_the_priors_floor(sweep):
    """The finding this ticket exists to record.

    |ρ| falls below 0.99 at about 21 nm of thickness — d/λ ≈ 0.035 at band
    centre — and DTFM-026's prior starts at 20 nm. The sampling floor and the
    physical threshold coincide to within a nanometre, so every film the network
    was trained on sits on the identifiable side of it, and everything below is
    both out of distribution *and* physically ambiguous.
    """
    grid, curve = sweep

    crossing = probe.threshold(grid, np.abs(curve["correlation"]), 0.99)

    assert 15.0 < crossing < 30.0, (
        f"expected the threshold near the 20 nm prior floor, got {crossing}"
    )
