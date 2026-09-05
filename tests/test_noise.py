"""Non-idealities that change the stack itself.

Spec §4.5.
Roughness and the interfacial layer by DTFM-023.

Both effects are applied *before* the forward model, because both add layers.
The remaining §4.5 effects act on the computed spectrum instead and arrive at
DTFM-024 and DTFM-025.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import least_squares

from src import noise
from src import tmm_torch as pt

WAVELENGTHS = torch.linspace(450.0, 800.0, 300, dtype=torch.float64)
FILM, SUBSTRATE = 1.46, 3.88
BASE_THICKNESS = 420.0


def _spectrum(thicknesses, indices):
    return pt.stack_reflectance(WAVELENGTHS, thicknesses, indices, 0.0, "s")


# --- Bruggeman effective medium --------------------------------------------


@pytest.mark.parametrize(("fraction", "expected"), [(1.0, 1.46), (0.0, 1.0)])
def test_bruggeman_is_exact_in_the_pure_limits(fraction, expected):
    """A mixture of one thing is that thing. Exactly, not approximately."""
    assert float(np.real(noise.effective_medium_index(1.46, 1.0, fraction))) == pytest.approx(
        expected, abs=1e-12
    )


def test_the_mixture_lies_between_its_components_and_is_monotonic():
    fractions = np.linspace(0.0, 1.0, 21)
    n = np.real(noise.effective_medium_index(1.46, 1.0, fractions))

    assert np.all((n >= 1.0) & (n <= 1.46))
    assert np.all(np.diff(n) > 0.0)


def test_bruggeman_is_symmetric_in_its_two_components():
    """Neither component is host and neither is inclusion — that symmetry is
    why §4.5 names Bruggeman rather than a Maxwell-Garnett form, and it is the
    right picture for a rough surface where film and void interpenetrate.
    """
    forward = noise.effective_medium_index(1.46, 1.0, 0.3)
    reversed_ = noise.effective_medium_index(1.0, 1.46, 0.7)

    assert complex(forward) == pytest.approx(complex(reversed_), abs=1e-12)


def test_mixing_happens_in_epsilon_not_in_the_index():
    """Averaging n directly would be the wrong quantity averaged.

    ε is what responds linearly to the field. The difference is not academic —
    at 50/50 the two answers differ by 0.01 in index, which is far more than the
    fit tolerances elsewhere in this project.
    """
    bruggeman = float(np.real(noise.effective_medium_index(1.46, 1.0, 0.5)))
    naive_index_average = 0.5 * 1.46 + 0.5 * 1.0

    assert abs(bruggeman - naive_index_average) > 0.005


def test_a_mixture_of_absorbing_components_stays_passive():
    """Im(ε) ≥ 0 after the root choice, or the mixture amplifies light.

    The same branch decision as DTFM-010, in a third guise: the quadratic has
    two roots and only one describes a material that absorbs.
    """
    epsilon = noise.bruggeman_epsilon((3.88 + 0.02j) ** 2, 1.0, np.linspace(0.0, 1.0, 50))

    assert np.all(np.imag(epsilon) >= 0.0)
    assert np.all(np.imag(np.sqrt(epsilon)) >= 0.0)


def test_fractions_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        noise.bruggeman_epsilon(2.0, 1.0, 1.5)


# --- surface roughness ------------------------------------------------------


def test_roughness_is_switchable_without_a_separate_code_path():
    """The acceptance criterion's "toggleable"."""
    thicknesses, indices = noise.add_surface_roughness(
        [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], 0.0
    )

    assert thicknesses == [BASE_THICKNESS]
    assert indices == [1.0, FILM, SUBSTRATE]


def test_roughness_adds_one_layer_at_the_top_of_the_stack():
    thicknesses, indices = noise.add_surface_roughness(
        [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], 12.0
    )

    assert thicknesses == [12.0, BASE_THICKNESS]
    assert len(indices) == 4
    assert indices[0] == 1.0 and indices[-1] == SUBSTRATE
    # the new layer is a film/void mixture, so it sits between the two
    assert 1.0 < float(np.real(indices[1])) < FILM


def test_roughness_damps_fringe_amplitude():
    """§4.5's stated signature, measured.

    Reflection from a graded boundary is spread over a range of path lengths
    rather than one, so the interference is less complete. This is the effect
    §10's failure atlas must distinguish from spot non-uniformity (DTFM-024),
    which damps fringes too.
    """
    amplitudes = []
    for roughness in (0.0, 5.0, 10.0, 20.0, 40.0):
        thicknesses, indices = noise.add_surface_roughness(
            [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], roughness
        )
        spectrum = _spectrum(thicknesses, indices)
        amplitudes.append((spectrum.max() - spectrum.min()).item())

    assert all(b < a for a, b in zip(amplitudes, amplitudes[1:], strict=False))
    assert amplitudes[-1] < 0.9 * amplitudes[0]


def test_the_void_fraction_is_configurable():
    """The acceptance criterion's "configurable". More void, lower index."""
    _, mostly_film = noise.add_surface_roughness(
        [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], 10.0, void_fraction=0.2
    )
    _, mostly_void = noise.add_surface_roughness(
        [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], 10.0, void_fraction=0.8
    )

    assert float(np.real(mostly_void[1])) < float(np.real(mostly_film[1]))


def test_negative_roughness_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        noise.add_surface_roughness([BASE_THICKNESS], [1.0, FILM, SUBSTRATE], -1.0)


# --- interfacial layer ------------------------------------------------------


def test_the_interfacial_layer_is_switchable():
    thicknesses, indices = noise.add_interfacial_layer(
        [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], 0.0
    )

    assert thicknesses == [BASE_THICKNESS]
    assert indices == [1.0, FILM, SUBSTRATE]


def test_the_interfacial_layer_goes_between_film_and_substrate():
    thicknesses, indices = noise.add_interfacial_layer(
        [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], 3.0
    )

    assert thicknesses == [BASE_THICKNESS, 3.0]
    assert indices[0] == 1.0 and indices[1] == FILM and indices[-1] == SUBSTRATE
    # default is an interdiffused mixture, so its index sits between the two
    assert FILM < float(np.real(indices[2])) < SUBSTRATE


def test_an_explicit_interfacial_index_is_honoured():
    """A chemically distinct phase — the thin oxide between a nitride and its
    silicon substrate, say — rather than an interdiffused mixture.
    """
    _, indices = noise.add_interfacial_layer(
        [BASE_THICKNESS], [1.0, 2.02, SUBSTRATE], 2.0, index=1.46
    )

    assert indices[2] == 1.46


def test_an_unmodelled_interfacial_layer_biases_the_recovered_thickness():
    """§4.5's "biases thickness if unmodelled", quantified.

    The observed spectrum comes from a stack that has an interfacial layer; the
    fit uses a model that does not know it exists. The recovered thickness
    absorbs the difference.

    The dangerous part is the residual. At 2 nm of interface the thickness is
    wrong by 1.3 nm — well outside any fab tolerance — while the fit residual is
    4.5e-04, which looks like a good fit. Nothing announces the error, which is
    §10's point about failures that get acted upon, appearing here for the first
    time in the inverse direction.
    """
    def observed_with(interface_nm: float):
        thicknesses, indices = noise.add_interfacial_layer(
            [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], interface_nm
        )
        return _spectrum(thicknesses, indices).numpy()

    def fit_thickness(target: np.ndarray) -> tuple[float, float]:
        def residual(parameters):
            return _spectrum([parameters[0]], [1.0, FILM, SUBSTRATE]).numpy() - target

        solution = least_squares(residual, [BASE_THICKNESS], bounds=([50.0], [900.0]))
        return float(solution.x[0]), float(np.sqrt(np.mean(solution.fun**2)))

    unbiased, _ = fit_thickness(observed_with(0.0))
    assert unbiased == pytest.approx(BASE_THICKNESS, abs=1e-3)

    biases = []
    for interface_nm in (1.0, 2.0, 5.0, 10.0):
        recovered, residual = fit_thickness(observed_with(interface_nm))
        biases.append(recovered - BASE_THICKNESS)
        assert residual < 2e-2, "the fit still looks acceptable while being wrong"

    assert all(b > 0.0 for b in biases), "the bias has a consistent sign"
    assert all(a < b for a, b in zip(biases, biases[1:], strict=False)), "and grows with it"
    assert biases[1] == pytest.approx(1.3, abs=0.3)
    assert biases[-1] > 5.0


def test_negative_interfacial_thickness_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        noise.add_interfacial_layer([BASE_THICKNESS], [1.0, FILM, SUBSTRATE], -1.0)


# --- both together ----------------------------------------------------------


def test_both_effects_compose_into_a_four_layer_stack():
    """A realistic post-deposition stack: rough top, film, interface, substrate."""
    thicknesses, indices = noise.add_surface_roughness(
        [BASE_THICKNESS], [1.0, FILM, SUBSTRATE], 8.0
    )
    thicknesses, indices = noise.add_interfacial_layer(thicknesses, indices, 2.0)

    assert len(thicknesses) == 3
    assert len(indices) == 5

    spectrum = _spectrum(thicknesses, indices)
    assert torch.all((spectrum >= 0.0) & (spectrum <= 1.0))


def test_the_modified_stack_still_differentiates():
    """§7.3 backpropagates through whatever the generator produced, defects and
    all — the reconstruction loss does not get a clean stack.
    """
    thickness = torch.tensor(BASE_THICKNESS, dtype=torch.float64, requires_grad=True)
    thicknesses, indices = noise.add_surface_roughness([thickness], [1.0, FILM, SUBSTRATE], 8.0)
    thicknesses, indices = noise.add_interfacial_layer(thicknesses, indices, 2.0)

    _spectrum(thicknesses, indices).sum().backward()

    assert torch.isfinite(thickness.grad) and thickness.grad.item() != 0.0


# --- effects on the measured spectrum, DTFM-024 -----------------------------
#
# Spec §4.5. These three act on the spectrum rather than the stack, so they are
# applied after the forward model.

BLUR_WAVELENGTHS = np.linspace(450.0, 800.0, 400)
THICK_FILM = 900.0  # closely spaced fringes, where blurring actually bites


def _clean(thickness_nm: float) -> np.ndarray:
    return pt.stack_reflectance(
        torch.tensor(BLUR_WAVELENGTHS), [float(thickness_nm)], [1.0, FILM, SUBSTRATE], 0.0, "s"
    ).numpy()


def _amplitude(spectrum) -> float:
    return float(np.max(spectrum) - np.min(spectrum))


# --- spot non-uniformity ----------------------------------------------------


def test_thickness_averaging_is_switchable():
    exact = _clean(THICK_FILM)
    averaged = noise.average_over_thickness(_clean, THICK_FILM, 0.0)

    assert np.array_equal(exact, averaged)


def test_thickness_spread_damps_fringes_monotonically():
    """§4.5's signature. Fringes are periodic in d, so a spread in d blurs them."""
    amplitudes = [
        _amplitude(noise.average_over_thickness(_clean, THICK_FILM, sigma, samples=41))
        for sigma in (0.0, 5.0, 10.0, 20.0, 40.0)
    ]

    assert all(b < a for a, b in zip(amplitudes, amplitudes[1:], strict=False))
    assert amplitudes[-1] < 0.7 * amplitudes[0]


def test_intensities_are_averaged_not_amplitudes():
    """Light from different parts of the spot is mutually incoherent.

    Averaging the complex amplitude instead would describe a single film of the
    mean thickness and produce no damping at all — the effect would vanish while
    the code still looked like it was doing something.
    """
    averaged = noise.average_over_thickness(_clean, THICK_FILM, 30.0, samples=41)
    single_film_at_mean = _clean(THICK_FILM)

    assert _amplitude(averaged) < 0.8 * _amplitude(single_film_at_mean)


def test_averaging_keeps_the_spectrum_physical():
    averaged = noise.average_over_thickness(_clean, THICK_FILM, 30.0, samples=41)

    assert np.all((averaged >= 0.0) & (averaged <= 1.0))


def test_averaging_rejects_bad_arguments():
    with pytest.raises(ValueError, match="non-negative"):
        noise.average_over_thickness(_clean, THICK_FILM, -1.0)
    with pytest.raises(ValueError, match="odd"):
        noise.average_over_thickness(_clean, THICK_FILM, 5.0, samples=20)


# --- finite spectrometer bandwidth ------------------------------------------


def test_bandwidth_is_switchable():
    spectrum = _clean(THICK_FILM)
    assert np.array_equal(
        spectrum, noise.apply_spectrometer_bandwidth(BLUR_WAVELENGTHS, spectrum, 0.0)
    )


def test_bandwidth_damps_fringes_monotonically():
    spectrum = _clean(THICK_FILM)
    amplitudes = [
        _amplitude(noise.apply_spectrometer_bandwidth(BLUR_WAVELENGTHS, spectrum, fwhm))
        for fwhm in (0.0, 4.0, 8.0, 16.0, 32.0)
    ]

    assert all(b < a for a, b in zip(amplitudes, amplitudes[1:], strict=False))


def test_bandwidth_conserves_the_mean_level():
    """Blurring redistributes but does not create or destroy — a normalised
    kernel must leave the average essentially untouched.
    """
    spectrum = _clean(THICK_FILM)
    blurred = noise.apply_spectrometer_bandwidth(BLUR_WAVELENGTHS, spectrum, 10.0)

    assert blurred.mean() == pytest.approx(spectrum.mean(), abs=2e-3)


def test_bandwidth_does_not_wrap_around_the_band_edges():
    """Explicit per-point weights rather than an FFT, deliberately.

    A circular convolution would fold the red end of the spectrum onto the blue
    one. The ends of a band are exactly where a fit is most sensitive, so an
    artefact there is expensive. A monotonic ramp must stay monotonic.
    """
    ramp = np.linspace(0.1, 0.9, BLUR_WAVELENGTHS.size)
    blurred = noise.apply_spectrometer_bandwidth(BLUR_WAVELENGTHS, ramp, 20.0)

    assert np.all(np.diff(blurred) > 0.0)
    assert blurred[0] == pytest.approx(ramp[0], abs=0.05)
    assert blurred[-1] == pytest.approx(ramp[-1], abs=0.05)


# --- the shared signature, which is the point of the ticket -----------------


def test_bandwidth_and_spot_non_uniformity_are_hard_to_tell_apart():
    """§4.5: "Same signature as above — and that ambiguity is itself interesting."

    Quantified: a single instrument bandwidth can reproduce the spectrum of a
    20 nm thickness spread to within 2.3e-02, where the difference between
    either and the undamped spectrum is 4.2e-02. So the two effects resemble
    each other roughly twice as well as either resembles the truth.

    They are not identical, and the reason is worth stating: a thickness spread
    blurs in d, which maps to a wavelength blur that grows as λ²/2nd, while an
    instrument bandwidth is constant in wavelength. That difference is the only
    handle a fit has for separating them.
    """
    clean = _clean(THICK_FILM)
    spread = noise.average_over_thickness(_clean, THICK_FILM, 20.0, samples=61)

    best = min(
        (np.abs(noise.apply_spectrometer_bandwidth(BLUR_WAVELENGTHS, clean, f) - spread).max(), f)
        for f in np.linspace(1.0, 40.0, 40)
    )
    mismatch, _ = best
    difference_from_clean = np.abs(clean - spread).max()

    assert mismatch < difference_from_clean
    assert mismatch > 0.0, "the two effects are not exactly degenerate"


def test_separating_them_needs_spectral_range():
    """The actionable form of that ambiguity.

    Over a narrow band the two are indistinguishable; the handle above only
    appears once the band is wide enough for λ² to vary appreciably. Measured
    here as the ratio of how well the two effects match each other to how well
    either matches the clean spectrum — 1.0 means indistinguishable.

    This bears directly on §10's failure atlas: whether a defect is separable is
    a property of the measurement design, not only of the estimator.
    """
    def separability(low: float, high: float) -> float:
        wavelengths = np.linspace(low, high, 300)
        tensor = torch.tensor(wavelengths)

        def forward(thickness):
            return pt.stack_reflectance(
                tensor, [float(thickness)], [1.0, FILM, SUBSTRATE], 0.0, "s"
            ).numpy()

        clean = forward(THICK_FILM)
        spread = noise.average_over_thickness(forward, THICK_FILM, 20.0, samples=41)
        mismatch = min(
            np.abs(noise.apply_spectrometer_bandwidth(wavelengths, clean, f) - spread).max()
            for f in np.linspace(1.0, 40.0, 40)
        )
        return float(np.abs(clean - spread).max() / mismatch)

    narrow = separability(575.0, 625.0)
    wide = separability(400.0, 1000.0)

    assert narrow < 1.6, "a narrow band should barely separate them"
    assert wide > 1.5 * narrow, "a wider band should separate them better"


# --- backside reflection ----------------------------------------------------


def test_backside_reflection_is_switchable():
    spectrum = _clean(THICK_FILM)
    assert np.array_equal(
        spectrum, noise.add_backside_reflection(spectrum, SUBSTRATE, transmittance=0.0)
    )


def test_backside_reflection_adds_an_offset_rather_than_fringes():
    """§4.5's signature. The wafer is far thicker than the coherence length, so
    the rear reflection adds in intensity and carries no phase — it raises the
    curve without changing its shape.
    """
    spectrum = _clean(THICK_FILM)
    with_back = noise.add_backside_reflection(spectrum, 1.52)

    assert np.all(with_back >= spectrum)
    assert with_back.mean() > spectrum.mean() + 1e-3
    # the fringe structure is preserved, not reshaped
    assert np.corrcoef(with_back, spectrum)[0, 1] > 0.98


def test_backside_reflection_rejects_impossible_transmittance():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        noise.add_backside_reflection(_clean(THICK_FILM), 1.52, transmittance=1.5)


# --- instrument errors, detector noise, and corrupt(), DTFM-025 -------------
#
# Spec §4.5 and §7.1. The last three effects, plus the pipeline that assembles
# all seven into one measured spectrum.


def _forward(wavelengths, thickness):
    return pt.stack_reflectance(
        torch.tensor(np.asarray(wavelengths, dtype=float)),
        [float(thickness)],
        [1.0, FILM, SUBSTRATE],
        0.0,
        "s",
    ).numpy()


# --- wavelength calibration -------------------------------------------------


def test_wavelength_calibration_is_identity_when_uncalibrated():
    assert np.array_equal(
        BLUR_WAVELENGTHS, noise.apply_wavelength_calibration(BLUR_WAVELENGTHS)
    )


def test_wavelength_calibration_applies_scale_and_offset():
    shifted = noise.apply_wavelength_calibration(np.array([500.0]), 0.01, 2.0)
    assert shifted[0] == pytest.approx(500.0 * 1.01 + 2.0)


def test_a_calibration_error_biases_thickness_and_does_not_average_away():
    """§4.5's "systematic thickness bias", quantified.

    Fringe positions carry the thickness, so mislabelling the wavelength axis
    mislabels the thickness. Unlike detector noise this is the *same* error on
    every repeat — no amount of averaging removes it, which is what makes a
    calibration drift more dangerous than a noisy detector.
    """
    truth = 420.0
    observed = _forward(noise.apply_wavelength_calibration(BLUR_WAVELENGTHS, 0.002, 0.0), truth)

    def residual(parameters):
        return _forward(BLUR_WAVELENGTHS, parameters[0]) - observed

    recovered = float(least_squares(residual, [truth], bounds=([50.0], [900.0])).x[0])

    assert abs(recovered - truth) > 0.5, "a 0.2% wavelength error should move the answer"
    assert abs(recovered - truth) < 5.0


# --- baseline drift ---------------------------------------------------------


def test_baseline_drift_is_identity_by_default():
    spectrum = _clean(THICK_FILM)
    assert np.array_equal(spectrum, noise.apply_baseline_drift(spectrum))


def test_baseline_drift_scales_and_shifts():
    spectrum = _clean(THICK_FILM)
    drifted = noise.apply_baseline_drift(spectrum, 1.05, 0.01)

    assert np.allclose(drifted, 1.05 * spectrum + 0.01)


def test_baseline_drift_correlates_with_the_index():
    """§4.5: "correlates with dispersion parameters", in the §5.3 sense.

    My first attempt asked whether some index could *mimic* a gain change. None
    can — even with thickness free the best fit is no better than doing nothing.
    That was the wrong test: correlation is a statement about the Jacobian, not
    about one parameter impersonating another.

    Taken properly, the effect is severe. With thickness, index, gain and offset
    all free, every off-diagonal correlation exceeds 0.99 in magnitude: the
    measurement cannot separate a brighter lamp from a denser film.
    """
    wavelengths = torch.linspace(450.0, 800.0, 300, dtype=torch.float64)
    theta = torch.tensor([900.0, 1.46, 1.0, 0.0], dtype=torch.float64)

    def model(parameters):
        spectrum = pt.stack_reflectance(
            wavelengths, [parameters[0]], [1.0, parameters[1], SUBSTRATE], 0.0, "s"
        )
        return parameters[2] * spectrum + parameters[3]

    jacobian = torch.autograd.functional.jacobian(model, theta)
    covariance = torch.linalg.inv(jacobian.T @ jacobian)
    scale = torch.sqrt(torch.diag(covariance))
    correlation = covariance / (scale[:, None] * scale[None, :])

    assert abs(correlation[1, 2].item()) > 0.99, "index vs gain"
    assert abs(correlation[1, 3].item()) > 0.99, "index vs offset"


def test_fitting_the_drift_wrecks_the_conditioning():
    """The cost of the previous test, and why §6 asks about model complexity.

    Adding gain and offset as free parameters raises the condition number of
    JᵀJ by a factor of ~550. The extra parameters do not buy accuracy; they buy
    a nearly singular problem and wider error bars on everything.

    §6's DTFM-035 asks this formally with AIC/BIC — "is my model too complex for
    the information my measurement contains". This is that question with a
    number attached, met before the ticket that asks it.
    """
    wavelengths = torch.linspace(450.0, 800.0, 300, dtype=torch.float64)

    def condition(with_drift: bool) -> float:
        count = 4 if with_drift else 2
        theta = torch.tensor([900.0, 1.46, 1.0, 0.0][:count], dtype=torch.float64)

        def model(parameters):
            spectrum = pt.stack_reflectance(
                wavelengths, [parameters[0]], [1.0, parameters[1], SUBSTRATE], 0.0, "s"
            )
            return parameters[2] * spectrum + parameters[3] if with_drift else spectrum

        jacobian = torch.autograd.functional.jacobian(model, theta)
        return torch.linalg.cond(jacobian.T @ jacobian).item()

    plain, with_drift = condition(False), condition(True)

    assert with_drift > 100 * plain
    assert with_drift < 1e13, "still invertible in float64, so the correlations mean something"


def test_non_positive_gain_is_rejected():
    with pytest.raises(ValueError, match="gain must be positive"):
        noise.apply_baseline_drift(_clean(THICK_FILM), 0.0)


# --- detector noise ---------------------------------------------------------


def test_noise_is_reproducible_from_its_seed():
    """§15: fixed seed, fixed output. An explicit Generator, never global state."""
    spectrum = _clean(THICK_FILM)
    first = noise.add_detector_noise(spectrum, np.random.default_rng(7))
    second = noise.add_detector_noise(spectrum, np.random.default_rng(7))
    different = noise.add_detector_noise(spectrum, np.random.default_rng(8))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_shot_noise_scales_with_the_square_root_of_the_signal():
    """§7.1's noise model, and it matters for §8.

    Shot noise is Poisson in photon count, so σ ∝ sqrt(R): bright parts of a
    spectrum are noisier in absolute terms. The aleatoric uncertainty the network
    is asked to predict is therefore *not* uniform across a spectrum, and a model
    assuming constant noise would misstate it in both directions.
    """
    rng = np.random.default_rng(0)
    bright = np.full(60_000, 0.36)
    dim = np.full(60_000, 0.04)

    spread_bright = (noise.add_detector_noise(bright, rng, read_sigma=0.0) - bright).std()
    spread_dim = (noise.add_detector_noise(dim, rng, read_sigma=0.0) - dim).std()

    assert spread_bright / spread_dim == pytest.approx(np.sqrt(0.36 / 0.04), rel=0.05)


def test_read_noise_dominates_where_the_signal_is_small():
    """Which is exactly where an anti-reflection null or a fringe minimum sits."""
    rng = np.random.default_rng(1)
    dark = np.full(60_000, 1e-4)
    spread = (noise.add_detector_noise(dark, rng) - dark).std()

    assert spread == pytest.approx(5e-4, rel=0.1)


def test_negative_noise_scales_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        noise.add_detector_noise(_clean(THICK_FILM), np.random.default_rng(0), shot_scale=-1.0)


# --- the corrupt() pipeline -------------------------------------------------


def test_the_default_configuration_activates_at_least_three_effects():
    """The acceptance criterion. §4.5 asks for three; the default runs four."""
    active = noise.Corruption().active

    assert len(active) >= 3
    assert "detector noise" in active
    assert "bandwidth" in active


def test_every_effect_can_be_switched_off():
    pristine = noise.Corruption(
        roughness_nm=0.0, bandwidth_fwhm_nm=0.0, shot_scale=0.0, read_sigma=0.0
    )

    assert pristine.active == ()
    observed = noise.corrupt(BLUR_WAVELENGTHS, _forward, 420.0, pristine)
    assert np.allclose(observed, _forward(BLUR_WAVELENGTHS, 420.0), atol=1e-12)


def test_the_stack_defects_are_applied_by_the_config_not_by_corrupt():
    """Roughness and the interfacial layer change what the film *is*, so they
    cannot be applied to a finished spectrum. Keeping them on the same object
    stops `active` claiming an effect that nothing applies.
    """
    configuration = noise.Corruption(roughness_nm=5.0, interfacial_nm=2.0)
    thicknesses, indices = configuration.stack_with_defects([420.0], [1.0, FILM, SUBSTRATE])

    assert len(thicknesses) == 3
    assert "roughness" in configuration.active
    assert "interfacial layer" in configuration.active


def test_corrupt_is_reproducible_and_stays_physical():
    configuration = noise.Corruption(
        spot_sigma_nm=8.0, wavelength_scale=1e-3, baseline_gain=1.02, baseline_offset=2e-3
    )
    first = noise.corrupt(
        BLUR_WAVELENGTHS, _forward, 420.0, configuration, np.random.default_rng(3)
    )
    second = noise.corrupt(
        BLUR_WAVELENGTHS, _forward, 420.0, configuration, np.random.default_rng(3)
    )

    assert np.array_equal(first, second)
    assert first.shape == BLUR_WAVELENGTHS.shape
    assert np.all(np.isfinite(first))


def test_noise_is_applied_last_so_the_instrument_cannot_smooth_it():
    """Order matters, and getting it wrong understates the uncertainty §8 models.

    Blurring after adding noise would let the spectrometer average away noise the
    detector has not generated yet. The observable consequence is that the
    point-to-point scatter survives at close to its full size even with a wide
    slit — which is what a real instrument shows.
    """
    wide_slit = noise.Corruption(roughness_nm=0.0, bandwidth_fwhm_nm=20.0, read_sigma=2e-3)
    observed = noise.corrupt(BLUR_WAVELENGTHS, _forward, 420.0, wide_slit, np.random.default_rng(5))

    smooth = noise.Corruption(
        roughness_nm=0.0, bandwidth_fwhm_nm=20.0, shot_scale=0.0, read_sigma=0.0
    )
    reference = noise.corrupt(BLUR_WAVELENGTHS, _forward, 420.0, smooth)

    scatter = np.std(np.diff(observed - reference))
    assert scatter > 1.5e-3, "noise was smoothed away, so it was applied too early"
