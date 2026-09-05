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


# --- Sellmeier dispersion, DTFM-021 ----------------------------------------
#
# Spec §4.4: "Transparent materials, wider range, physically better behaved."
# Each of those three claims is tested below rather than taken on trust.


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2"])
def test_sellmeier_beats_cauchy_within_the_fitted_range(material):
    """"Physically better behaved", measured.

    Cauchy is Sellmeier expanded away from resonance and truncated, so it cannot
    do better than the form it approximates given comparable freedom.
    """
    sellmeier = dp.fit_sellmeier(material, oscillators=2)
    cauchy = dp.fit_cauchy(material, terms=3)

    assert sellmeier.rms_residual < cauchy.rms_residual


@pytest.mark.parametrize("material", ["SiO2", "Si3N4"])
def test_sellmeier_extrapolates_better_than_cauchy(material):
    """"Wider range", and the sharper form of the claim.

    Both models are fitted on 500-700 nm, then asked about wavelengths they
    never saw. Keeping the pole rather than expanding it away is what lets
    Sellmeier stay honest outside the fitted window — and this is the property
    that matters, since §7.1's prior will sample films the fit never covered.
    """
    fit_range = np.linspace(500.0, 700.0, 100)
    low, high = dp.material_range_nm(material)
    wide = np.linspace(max(low, 250.0), min(high, 2500.0), 400)
    n_true, _ = dp.load_nk(material, wide)

    cauchy_error = np.abs(dp.cauchy_n(dp.fit_cauchy(material, fit_range), wide) - n_true).max()
    sellmeier_error = np.abs(
        dp.sellmeier_n(dp.fit_sellmeier(material, fit_range, oscillators=2), wide) - n_true
    ).max()

    assert sellmeier_error < cauchy_error / 3.0


def test_the_fitted_resonance_matches_the_published_one():
    """The coefficients mean something, unlike Cauchy's.

    Luke's Si₃N₄ Sellmeier coefficient is C₁ = 0.1353406 µm² — a resonance at
    135.3 nm. Fitting only 400-800 nm data, where that pole is far outside the
    window and never directly observed, recovers it. A curve-fitting exercise
    would not.
    """
    fit = dp.fit_sellmeier("Si3N4", oscillators=2)
    ultraviolet = min(fit.resonances_nm)

    assert ultraviolet == pytest.approx(135.3, rel=0.05)


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2"])
def test_poles_stay_outside_the_fitted_window(material):
    """A pole inside the data is a division by zero mid-range.

    The model would blow up between two measured points rather than describe
    anything, so each oscillator is confined to one side of the window.
    """
    fit = dp.fit_sellmeier(material, oscillators=2)
    low, high = fit.range_nm

    for resonance in fit.resonances_nm:
        assert resonance < low or resonance > high


def test_titania_needs_two_ultraviolet_poles_not_the_textbook_uv_ir_pair():
    """A regression guard on the multi-start, and the physics behind it.

    The obvious starting guess — one UV electronic resonance, one IR lattice
    vibration — is right for SiO₂ and Si₃N₄ and wrong for TiO₂, whose band gap
    sits at 385 nm and which has no useful infrared pole over the visible.
    Started from that guess alone the optimiser drives the second oscillator's
    strength to zero and returns a fit 86× worse, reporting nothing amiss.

    §6 prescribes multi-start for precisely this failure. Without it this
    assertion fails at 4.3e-03.
    """
    fit = dp.fit_sellmeier("TiO2", oscillators=2)

    assert fit.rms_residual < 1e-4
    assert all(b > 0.0 for b in fit.b), "an oscillator collapsed to zero strength"
    assert all(r < 400.0 for r in fit.resonances_nm)


@pytest.mark.parametrize("material", ["SiO2", "Si3N4"])
def test_a_second_oscillator_improves_the_fit(material):
    one = dp.fit_sellmeier(material, oscillators=1)
    two = dp.fit_sellmeier(material, oscillators=2)

    assert two.rms_residual < one.rms_residual


def test_sellmeier_refuses_absorbing_materials():
    """Its poles mark where absorption is without describing it — a real index
    with no k. §4.4 sends absorbers to a Lorentz oscillator (DTFM-022).
    """
    with pytest.raises(ValueError, match="transparent"):
        dp.fit_sellmeier("Si")


@pytest.mark.parametrize("material", ["SiO2", "Si3N4", "TiO2"])
def test_the_fitted_model_reproduces_its_data(material):
    wavelengths = np.linspace(420.0, 780.0, 73)
    fit = dp.fit_sellmeier(material, wavelengths, oscillators=2)
    n_data, _ = dp.load_nk(material, wavelengths)

    assert np.allclose(dp.sellmeier_n(fit, wavelengths), n_data, atol=1e-3)
    assert np.all(dp.sellmeier_n(fit, wavelengths) > 1.0)


def test_sellmeier_is_differentiable_in_its_coefficients():
    """§7.1 samples dispersion coefficients; §7.3 backpropagates through them."""
    fit = dp.fit_sellmeier("SiO2", oscillators=2)
    b = torch.tensor(fit.b, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(fit.c, dtype=torch.float64, requires_grad=True)
    wavelengths = torch.linspace(450.0, 750.0, 50, dtype=torch.float64)

    dp.sellmeier_n((b, c), wavelengths).sum().backward()

    assert torch.all(torch.isfinite(b.grad)) and torch.any(b.grad != 0.0)
    assert torch.all(torch.isfinite(c.grad)) and torch.any(c.grad != 0.0)


def test_at_least_one_oscillator_is_required():
    with pytest.raises(ValueError, match="at least one oscillator"):
        dp.fit_sellmeier("SiO2", oscillators=0)


# --- Lorentz oscillators, DTFM-022 -----------------------------------------
#
# Spec §4.4 sends absorbing materials here, and it is the only one of the three
# models with an imaginary part at all. Cauchy has none; Sellmeier's poles mark
# where absorption is without describing it, because a bare pole has no width.
# Γₖ is that width, and it is the whole difference.

SI_TRANSPARENT_WINDOW = np.linspace(400.0, 800.0, 200)  # below silicon's E₁ point


def test_lorentz_fits_an_absorbing_material_the_others_refuse():
    """The reason the model exists.

    `fit_cauchy("Si")` and `fit_sellmeier("Si")` both raise — neither has an
    imaginary part. Lorentz describes the same data to 0.05% in n.
    """
    fit = dp.fit_lorentz("Si", SI_TRANSPARENT_WINDOW, oscillators=2)
    n_data, _ = dp.load_nk("Si", SI_TRANSPARENT_WINDOW)

    assert fit.rms_residual_n < 0.01
    assert fit.rms_residual_n / n_data.mean() < 1e-3
    assert fit.rms_residual_k < 0.02


def test_the_fitted_bands_are_silicons_known_critical_points():
    """E₁ at 3.4 eV and E₂ at 4.25 eV are textbook silicon band structure.

    Fitting the dielectric function over 300-800 nm recovers both. As with
    Sellmeier's resonance in DTFM-021, coefficients that land on independently
    known physics are evidence the model is describing the material rather than
    interpolating the numbers.
    """
    fit = dp.fit_lorentz("Si", np.linspace(300.0, 800.0, 200), oscillators=2)

    assert fit.energies[0] == pytest.approx(3.4, abs=0.3)
    assert fit.energies[1] == pytest.approx(4.25, abs=0.4)
    assert all(g > 0.0 for g in fit.widths)


def test_the_model_reproduces_absorption_not_just_the_index():
    """k is the point. A model matching n while getting k wrong would satisfy a
    naive residual and be useless for an absorbing stack.
    """
    fit = dp.fit_lorentz("Si", SI_TRANSPARENT_WINDOW, oscillators=2)
    _, k_data = dp.load_nk("Si", SI_TRANSPARENT_WINDOW)
    _, k_model = dp.lorentz_nk(fit, SI_TRANSPARENT_WINDOW)

    assert np.all(k_model > 0.0)
    assert np.corrcoef(k_model, k_data)[0, 1] > 0.99


@pytest.mark.parametrize("material", ["Si", "TiO2"])
def test_the_model_is_passive_everywhere(material):
    """Im(ñ) >= 0, or the material amplifies light.

    Guaranteed by the sign of the iΓE term and the principal square root — the
    same branch question as DTFM-010 in another guise. `src.tmm_torch` rejects a
    violation downstream; catching it here keeps the diagnosis local.
    """
    fit = dp.fit_lorentz(material, oscillators=2)
    wavelengths = np.linspace(*fit.range_nm, 500)
    _, k = dp.lorentz_nk(fit, wavelengths)

    assert np.all(k >= 0.0)
    assert np.all(np.isfinite(k))


def test_fit_degrades_above_silicons_critical_points():
    """The validity boundary, measured — as DTFM-020 did for Cauchy's edge.

    A Lorentz oscillator is a broad, symmetric absorption band. Crystalline
    silicon's E₁ and E₂ are sharp van Hove singularities in the joint density of
    states, which a sum of Lorentzians cannot reproduce; the literature reaches
    for Tauc-Lorentz or critical-point parabolic band models there.

    Below E₁ the fit is good to 0.05% in n. Extend the window past it and the
    error grows by two orders of magnitude — while still looking like a
    dispersion curve, which is why the residual is reported.
    """
    below = dp.fit_lorentz("Si", np.linspace(500.0, 800.0, 200), oscillators=2)
    across = dp.fit_lorentz("Si", np.linspace(250.0, 800.0, 200), oscillators=2)

    assert below.rms_residual_n < 0.01
    assert across.rms_residual_n > 50 * below.rms_residual_n


def test_kramers_kronig_consistency():
    """The note §4.4 asks for, checked numerically rather than asserted.

    A Lorentz oscillator is a causal response function, so its real and
    imaginary parts are not independent: either determines the other through

        ε₁(E) − ε_∞ = (2/π) P ∫₀^∞ E' ε₂(E') / (E'² − E²) dE'

    This test recovers ε₁ from ε₂ alone and compares. It matters beyond
    elegance: it is why fitting n and k *jointly* against complex ε is the right
    move, and why a model fitted only to n cannot be trusted for k. It is also
    the physics behind §4.4's remark that Kramers-Kronig constrains n and k to
    be consistent with each other.

    The principal value is taken by subtraction — with g(E') = E'ε₂(E'), the
    singular part cancels because P∫₀^∞ dE'/(E'²−E²) = 0 exactly — leaving a
    removable integrand. Residual error is numerical (finite grid and finite
    upper limit), not physical.
    """
    coefficients = (1.0, (12.0,), (3.4,), (0.6,))
    grid = np.linspace(1e-6, 400.0, 400_001)

    def epsilon_2(energies):
        return dp.lorentz_epsilon(coefficients, dp.HC_EV_NM / energies).imag

    g = grid * epsilon_2(grid)

    def kk_epsilon_1(energy: float) -> float:
        g_at = energy * epsilon_2(np.array([energy]))[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = (g - g_at) / (grid**2 - energy**2)
        bad = ~np.isfinite(integrand)
        integrand[bad] = np.interp(grid[bad], grid[~bad], integrand[~bad])
        return 1.0 + (2.0 / np.pi) * np.trapezoid(integrand, grid)

    for energy in (1.5, 2.0, 2.5, 4.0, 5.0):
        direct = dp.lorentz_epsilon(coefficients, dp.HC_EV_NM / energy).real
        assert kk_epsilon_1(energy) == pytest.approx(direct, rel=0.05)


def test_lorentz_is_differentiable_in_its_coefficients():
    """§7.1 samples dispersion parameters; §7.3 backpropagates through them."""
    fit = dp.fit_lorentz("Si", SI_TRANSPARENT_WINDOW, oscillators=2)
    amplitudes = torch.tensor(fit.amplitudes, dtype=torch.float64, requires_grad=True)
    wavelengths = torch.linspace(450.0, 750.0, 40, dtype=torch.float64)

    n, k = dp.lorentz_nk(
        (fit.eps_inf, amplitudes, torch.tensor(fit.energies), torch.tensor(fit.widths)),
        wavelengths,
    )
    (n.sum() + k.sum()).backward()

    assert torch.all(torch.isfinite(amplitudes.grad))
    assert torch.any(amplitudes.grad != 0.0)


def test_a_lorentz_film_drives_the_forward_model():
    """An absorbing film described by fitted parameters, through the stack.

    This is the combination DTFM-010's branch guard exists for: a complex index
    from a fitted model, inside a multilayer, with gradients flowing back to the
    oscillator strengths.
    """
    fit = dp.fit_lorentz("Si", SI_TRANSPARENT_WINDOW, oscillators=2)
    amplitudes = torch.tensor(fit.amplitudes, dtype=torch.float64, requires_grad=True)
    wavelengths = torch.linspace(500.0, 800.0, 120, dtype=torch.float64)

    n, k = dp.lorentz_nk(
        (fit.eps_inf, amplitudes, torch.tensor(fit.energies), torch.tensor(fit.widths)),
        wavelengths,
    )
    R = pt.stack_reflectance(wavelengths, [80.0], [1.0, n + 1j * k, 1.46], 0.0, "s")
    R.sum().backward()

    assert torch.all((R >= 0.0) & (R <= 1.0))
    assert torch.all(torch.isfinite(amplitudes.grad))
    assert torch.any(amplitudes.grad != 0.0)


def test_at_least_one_lorentz_oscillator_is_required():
    with pytest.raises(ValueError, match="at least one oscillator"):
        dp.fit_lorentz("Si", oscillators=0)
