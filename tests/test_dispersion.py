"""Real optical constants from the vendored refractiveindex.info data.

Spec §4.4.
Loader by DTFM-019; the fitted dispersion models by DTFM-020 - DTFM-022.

These tests check the loaded numbers against values published independently of
the database — a table that silently changed, or a unit error between microns
and nanometres, would otherwise produce a plausible material and no complaint.
"""

import numpy as np
import pytest
import torch

from src import dispersion as dp
from src import tmm_torch as pt

HENE = 632.8  # nm, the wavelength most optical constants are quoted at


def test_all_four_materials_of_the_spec_are_available():
    """§4.4 names SiO₂, Si₃N₄, TiO₂ and Si specifically."""
    assert dp.available_materials() == ["Si", "Si3N4", "SiO2", "TiO2"]


@pytest.mark.parametrize(
    ("material", "n_expected", "k_expected", "tolerance"),
    [
        # Fused silica at the He-Ne line is 1.4570 in every optics reference.
        ("SiO2", 1.4570, 0.0, 1e-3),
        # Aspnes & Studna's silicon: n = 3.882, k = 0.019 at 632.8 nm.
        ("Si", 3.882, 0.019, 5e-3),
        # Luke's stoichiometric Si₃N₄ film, ~2.04.
        ("Si3N4", 2.040, 0.0, 5e-3),
    ],
)
def test_values_match_the_published_literature(material, n_expected, k_expected, tolerance):
    """Anchors the loader to numbers quoted outside the database itself.

    A micron/nanometre mix-up, an off-by-one column, or an upstream revision
    would all still produce smooth plausible curves. Only a comparison against
    an externally known value catches them.
    """
    n, k = dp.load_nk(material, [HENE])
    assert n[0] == pytest.approx(n_expected, abs=tolerance)
    assert k[0] == pytest.approx(k_expected, abs=tolerance)


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2", "Si"])
def test_k_is_never_negative(material):
    """The n + ik convention, which src.tmm_torch requires.

    Some sources tabulate n − ik. Ingesting under the wrong sign would give a
    gain medium — a spectrum that looks entirely reasonable from a material that
    cannot exist. `_require_passive` rejects it downstream; this catches it at
    the source, where the diagnosis is obvious.
    """
    low, high = dp.material_range_nm(material)
    wavelengths = np.linspace(low, min(high, 2000.0), 400)
    _, k = dp.load_nk(material, wavelengths)

    assert np.all(k >= 0.0)


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2", "Si"])
def test_the_index_is_physically_plausible(material):
    """n > 1 for every condensed material, and nothing absurdly large."""
    low, high = dp.material_range_nm(material)
    wavelengths = np.linspace(max(low, 300.0), min(high, 800.0), 200)
    n, _ = dp.load_nk(material, wavelengths)

    assert np.all(n > 1.0)
    assert np.all(n < 10.0)


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2"])
def test_normal_dispersion_across_the_visible(material):
    """n falls with increasing wavelength below the absorption edge.

    §4.4 notes this is why a Cauchy model is appropriate there and wrong above
    it. It is also the effect that makes spectroscopic measurement break the
    thickness-index degeneracy of §5.2(a) at all: without dispersion, a spectrum
    would constrain only the product n·d.
    """
    n, _ = dp.load_nk(material, np.linspace(420.0, 780.0, 60))
    assert np.all(np.diff(n) < 0.0)


def test_silicon_absorbs_across_the_visible():
    """k > 0 for silicon, which is what makes DTFM-010's branch guard matter.

    Absorption also rises towards the blue: silicon's band gap is in the
    infrared, so shorter wavelengths are absorbed harder.
    """
    n, k = dp.load_nk("Si", [450.0, 550.0, 632.8])

    assert np.all(k > 0.0)
    assert k[0] > k[1] > k[2]


def test_titania_is_transparent_in_the_visible_and_absorbing_in_the_uv():
    """TiO₂'s band gap sits near 385 nm, so the absorption edge is inside the
    dataset — an internal consistency check on the tabulated k column.
    """
    _, k_visible = dp.load_nk("TiO2", [500.0, 600.0, 700.0])
    _, k_uv = dp.load_nk("TiO2", [250.0, 300.0])

    assert np.all(k_visible < 1e-3)
    assert np.all(k_uv > 1e-3)


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2", "Si"])
def test_extrapolation_is_refused(material):
    """The honest response outside the measured range.

    An extrapolated dispersion curve is an invented material. It would return a
    smooth, reasonable-looking number with no warning — the exact failure mode
    §10 calls dangerous, since it gets acted upon. §4.4 asks for models fitted to
    real data, so outside that data the loader declines.
    """
    low, high = dp.material_range_nm(material)

    with pytest.raises(ValueError, match="outside"):
        dp.load_nk(material, [low - 1.0])
    with pytest.raises(ValueError, match="outside"):
        dp.load_nk(material, [high + 1.0])
    with pytest.raises(ValueError, match="outside"):
        dp.load_nk(material, [low + 1.0, high + 50.0])


def test_the_valid_range_boundaries_themselves_are_accepted():
    """Refusing the endpoints would make the stated range a lie."""
    for material in dp.available_materials():
        low, high = dp.material_range_nm(material)
        n, k = dp.load_nk(material, [low, high])
        assert np.all(np.isfinite(n)) and np.all(np.isfinite(k))


def test_an_unknown_material_names_what_is_available():
    with pytest.raises(KeyError, match="SiO2"):
        dp.load_nk("Unobtainium", [550.0])


def test_shapes_follow_the_request():
    wavelengths = np.linspace(400.0, 800.0, 37)
    n, k = dp.load_nk("SiO2", wavelengths)

    assert n.shape == wavelengths.shape == k.shape
    assert n.dtype == np.float64


def test_real_constants_drive_the_forward_model():
    """The point of the ticket: a spectrum from materials that actually exist.

    Until now every index in the project was chosen by hand. This is the first
    stack built from measured data, and it is also a units check — a
    micron/nanometre error in the loader would move the fringes far enough to be
    obvious here.
    """
    wavelengths = np.linspace(450.0, 800.0, 201)
    n_film, k_film = dp.load_nk("SiO2", wavelengths)
    n_sub, k_sub = dp.load_nk("Si", wavelengths)

    film = torch.tensor(n_film + 1j * k_film)
    substrate = torch.tensor(n_sub + 1j * k_sub)
    R = pt.stack_reflectance(torch.tensor(wavelengths), [500.0], [1.0, film, substrate], 0.0, "s")

    assert R.shape == (wavelengths.size,)
    assert torch.all((R >= 0.0) & (R <= 1.0))

    # Real SiO₂ on real silicon must show interference, not a smooth curve.
    turning_points = ((R[1:-1] - R[:-2]) * (R[2:] - R[1:-1]) < 0).sum().item()
    assert turning_points >= 2
    assert (R.max() - R.min()).item() > 0.1


def test_dispersion_changes_the_spectrum_enough_to_matter():
    """Justifies §4.4 existing at all.

    If treating n as constant gave the same spectrum, dispersion would be a
    detail rather than a chapter. It does not: over the visible, SiO₂'s index
    varies enough to shift the fringes measurably, and that shift is what breaks
    the thickness-index degeneracy of §5.2(a).
    """
    wavelengths = np.linspace(450.0, 800.0, 201)
    n_film, _ = dp.load_nk("SiO2", wavelengths)
    n_sub, k_sub = dp.load_nk("Si", wavelengths)
    substrate = torch.tensor(n_sub + 1j * k_sub)

    dispersive = pt.stack_reflectance(
        torch.tensor(wavelengths), [1500.0], [1.0, torch.tensor(n_film), substrate], 0.0, "s"
    )
    constant = pt.stack_reflectance(
        torch.tensor(wavelengths), [1500.0], [1.0, float(n_film.mean()), substrate], 0.0, "s"
    )

    assert (dispersive - constant).abs().max().item() > 0.01


# --- Cauchy dispersion, DTFM-020 -------------------------------------------
#
# Spec §4.4. The model is fitted to the measured data of DTFM-019 rather than
# given invented coefficients, and the fit is only meaningful where the material
# is transparent — the validity condition §4.4 calls interview-defensible
# physics, and which the tests below establish quantitatively rather than assert.

TRANSPARENT = ["SiO2", "Si3N4"]


@pytest.mark.parametrize("material", TRANSPARENT)
def test_cauchy_fits_transparent_materials_closely(material):
    """The acceptance criterion: residuals reported, and small.

    1e-4 in refractive index is far below what any spectrometer resolves, so a
    Cauchy model is a faithful stand-in for the tabulated data over the visible.
    """
    fit = dp.fit_cauchy(material)

    assert fit.rms_residual < 2e-4
    assert fit.max_residual < 1e-3
    assert fit.range_nm == (400.0, 800.0)


@pytest.mark.parametrize("material", TRANSPARENT)
def test_the_third_term_earns_its_place(material):
    """C/λ⁴ must actually improve the fit, or it is a free parameter for nothing.

    §6 asks the same question formally with AIC/BIC at DTFM-035 — whether the
    data justifies an extra parameter. This is the cheap version of it.
    """
    two = dp.fit_cauchy(material, terms=2)
    three = dp.fit_cauchy(material, terms=3)

    assert three.rms_residual < two.rms_residual
    assert two.c == 0.0


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2"])
def test_coefficients_describe_normal_dispersion(material):
    """A > 1 and B > 0 — anything else is not a transparent dielectric.

    B > 0 is what makes n fall with wavelength, which §5.2(a) relies on: without
    dispersion a spectrum constrains only the product n·d, and thickness and
    index would be perfectly degenerate.
    """
    fit = dp.fit_cauchy(material)

    assert fit.a > 1.0
    assert fit.b > 0.0


@pytest.mark.parametrize("material", TRANSPARENT)
def test_the_fitted_model_reproduces_the_data_it_was_fitted_to(material):
    wavelengths = np.linspace(420.0, 780.0, 73)
    fit = dp.fit_cauchy(material, wavelengths)
    n_data, _ = dp.load_nk(material, wavelengths)

    assert np.allclose(dp.cauchy_n(fit, wavelengths), n_data, atol=1e-3)


def test_absorbing_materials_are_refused_by_default():
    """Cauchy has no imaginary part, so fitting an absorber is a wrong model
    rather than a poor fit — silicon's k reaches 0.386 across the visible.
    """
    with pytest.raises(ValueError, match="below the absorption edge"):
        dp.fit_cauchy("Si")

    forced = dp.fit_cauchy("Si", allow_absorbing=True)
    assert forced.rms_residual > 100 * dp.fit_cauchy("SiO2").rms_residual


def test_fit_quality_degrades_approaching_the_absorption_edge():
    """The headline physics of §4.4, measured rather than asserted.

    Cauchy is the leading terms of a Sellmeier form expanded far from resonance.
    Approach the absorption edge and that expansion diverges, so the residual
    must grow monotonically as the fit range closes on it — while the fitted
    curve still *looks* like a dispersion curve, which is why the residual is
    the thing to watch rather than the plot.

    TiO₂ is the clean case: its edge sits at ~395 nm, inside the measured range.
    """
    starts = [700.0, 600.0, 500.0, 450.0, 420.0, 405.0]
    residuals = [
        dp.fit_cauchy("TiO2", np.linspace(start, 800.0, 200)).rms_residual for start in starts
    ]

    assert all(b > a for a, b in zip(residuals, residuals[1:], strict=False))
    assert residuals[-1] > 500 * residuals[0]


def test_titania_far_from_its_edge_fits_as_well_as_a_transparent_material():
    """The converse of the test above: the model is not simply bad for TiO₂.

    Given room from the absorption edge it fits to 2e-6 — better than SiO₂ over
    the visible. The limitation is the edge, not the material.
    """
    fit = dp.fit_cauchy("TiO2", np.linspace(700.0, 800.0, 200))
    assert fit.rms_residual < 1e-5


def test_the_model_is_differentiable_in_its_coefficients():
    """§7.1 samples dispersion coefficients as parameters and §7.3
    backpropagates through them, so the model must survive autograd.
    """
    fit = dp.fit_cauchy("SiO2")
    coefficients = torch.tensor([fit.a, fit.b, fit.c], dtype=torch.float64, requires_grad=True)
    wavelengths = torch.linspace(450.0, 750.0, 50, dtype=torch.float64)

    dp.cauchy_n(coefficients, wavelengths).sum().backward()

    assert torch.all(torch.isfinite(coefficients.grad))
    assert torch.all(coefficients.grad > 0.0)


def test_a_cauchy_film_drives_the_forward_model():
    """End to end: fitted coefficients through the stack, differentiably.

    This is the shape §7.1 will use — a film described by dispersion parameters
    rather than a fixed index, with gradients reaching those parameters.
    """
    fit = dp.fit_cauchy("SiO2")
    coefficients = torch.tensor([fit.a, fit.b, fit.c], dtype=torch.float64, requires_grad=True)
    wavelengths = torch.linspace(450.0, 800.0, 120, dtype=torch.float64)

    n_film = dp.cauchy_n(coefficients, wavelengths)
    n_sub, k_sub = dp.load_nk("Si", wavelengths.numpy())
    substrate = torch.tensor(n_sub + 1j * k_sub)

    R = pt.stack_reflectance(wavelengths, [500.0], [1.0, n_film, substrate], 0.0, "s")
    R.sum().backward()

    assert torch.all((R >= 0.0) & (R <= 1.0))
    assert torch.all(torch.isfinite(coefficients.grad))
    assert torch.any(coefficients.grad != 0.0)


def test_invalid_term_count_is_rejected():
    with pytest.raises(ValueError, match="terms must be 2 or 3"):
        dp.fit_cauchy("SiO2", terms=4)
