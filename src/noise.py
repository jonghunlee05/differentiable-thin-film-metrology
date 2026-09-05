"""Non-idealities: roughness, interfacial layer, bandwidth, drift, noise.

Spec §4.5.
Roughness and the interfacial layer by DTFM-023; spot non-uniformity, finite
bandwidth and backside reflection by DTFM-024; wavelength calibration, baseline
drift and detector noise by DTFM-025.

Why these exist at all. A perfect spectrum is easy to invert and proves nothing:
every method looks excellent on data with no defects in it. §4.5 lists seven real
effects, and the ones here are the two that change the *stack itself* rather than
the measured curve — they add layers, so they are applied before the forward
model rather than after it.

Both effects are also degeneracies waiting to happen, which is the real reason
they matter to this project. Roughness damps fringe amplitude, and so does spot
non-uniformity (§4.5, DTFM-024). An unmodelled interfacial layer biases the
recovered thickness, and so does a wavelength calibration error. §10's failure
atlas is largely a catalogue of which of these the estimator can tell apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Corruption",
    "add_backside_reflection",
    "add_detector_noise",
    "add_ellipsometer_noise",
    "add_interfacial_layer",
    "add_surface_roughness",
    "apply_spectrometer_bandwidth",
    "apply_baseline_drift",
    "apply_wavelength_calibration",
    "average_over_thickness",
    "corrupt",
    "bruggeman_epsilon",
    "effective_medium_index",
]


def bruggeman_epsilon(epsilon_a, epsilon_b, fraction_a):
    """Bruggeman effective-medium dielectric function of a two-component mix.

    Solves the self-consistency condition of §4.5

    ``f_a (ε_a − ε)/(ε_a + 2ε) + f_b (ε_b − ε)/(ε_b + 2ε) = 0``

    which rearranges to the quadratic ``2ε² − Bε − ε_a ε_b = 0`` with
    ``B = (2f_a − f_b)ε_a + (2f_b − f_a)ε_b``. Taking the root with
    ``Im(ε) ≥ 0`` keeps the mixture passive — the same branch decision as
    DTFM-010, in a third guise.

    Bruggeman rather than a simple average because it is *symmetric* in its two
    components: neither is the host and neither the inclusion. That is the right
    picture for a rough surface, which is film and void interpenetrating at
    comparable fractions, and it is why §4.5 names it specifically.

    Parameters
    ----------
    epsilon_a, epsilon_b : complex dielectric functions of the two components.
    fraction_a : volume fraction of the first, in ``[0, 1]``.
    """
    fraction_a = np.asarray(fraction_a, dtype=float)
    if np.any(fraction_a < 0.0) or np.any(fraction_a > 1.0):
        raise ValueError("fraction_a must lie in [0, 1]")
    fraction_b = 1.0 - fraction_a

    b = (2.0 * fraction_a - fraction_b) * epsilon_a + (2.0 * fraction_b - fraction_a) * epsilon_b
    discriminant = np.sqrt(b**2 + 8.0 * epsilon_a * epsilon_b + 0j)

    root_plus = (b + discriminant) / 4.0
    root_minus = (b - discriminant) / 4.0
    return np.where(np.imag(root_plus) >= 0.0, root_plus, root_minus)


def effective_medium_index(index_a, index_b, fraction_a):
    """Bruggeman mixture expressed as a refractive index rather than ``ε``.

    Mixing happens in the dielectric function, not the index: ``ε`` is what
    responds linearly to the field, so averaging ``n`` directly would be the
    wrong quantity averaged and would give a different, unphysical answer.
    """
    epsilon = bruggeman_epsilon(np.asarray(index_a) ** 2, np.asarray(index_b) ** 2, fraction_a)
    return np.sqrt(epsilon)


def add_surface_roughness(
    thicknesses: Sequence,
    indices: Sequence,
    roughness_nm: float,
    *,
    void_fraction: float = 0.5,
    ambient_index=None,
) -> tuple[list, list]:
    """Model surface roughness as a Bruggeman layer on top of the stack (§4.5).

    A rough surface is not a sharp boundary between film and air but a region
    that is partly each. Optically it behaves as a thin layer of the mixture,
    and the standard treatment gives it the roughness height as its thickness
    and equal parts film and void.

    The observable consequence is **damped fringe amplitude**: light reflecting
    from a graded boundary is spread over a range of path lengths rather than a
    single one, so the interference is less complete. §4.5 lists that signature,
    and DTFM-024's spot non-uniformity produces a similar one — which is the
    ambiguity §10's failure atlas has to catalogue.

    Passing ``roughness_nm = 0`` returns the stack unchanged, so the effect is
    switchable without a separate code path.
    """
    if roughness_nm < 0.0:
        raise ValueError(f"roughness must be non-negative, got {roughness_nm}")
    if not 0.0 <= void_fraction <= 1.0:
        raise ValueError(f"void_fraction must lie in [0, 1], got {void_fraction}")
    if len(indices) < 2:
        raise ValueError("need at least an ambient and a substrate to roughen")
    if roughness_nm == 0.0:
        return list(thicknesses), list(indices)

    ambient = indices[0] if ambient_index is None else ambient_index
    top_film = indices[1]
    mixed = effective_medium_index(top_film, ambient, 1.0 - void_fraction)

    return [roughness_nm, *thicknesses], [indices[0], mixed, *indices[1:]]


def add_interfacial_layer(
    thicknesses: Sequence,
    indices: Sequence,
    thickness_nm: float,
    *,
    index=None,
    mix_fraction: float = 0.5,
) -> tuple[list, list]:
    """Insert a thin layer between the last film and the substrate (§4.5).

    Deposition rarely produces an atomically abrupt boundary: there is usually a
    few nanometres of interdiffused or chemically distinct material. If it is not
    modelled, the fit absorbs it into the film thickness — §4.5's "biases
    thickness if unmodelled", which DTFM-023's tests quantify.

    ``index`` defaults to a Bruggeman mixture of the film and substrate, which is
    the physically motivated default for an interdiffused region. Give it
    explicitly for a chemically distinct interfacial phase, such as the thin
    oxide that grows between a nitride and its silicon substrate.

    Passing ``thickness_nm = 0`` returns the stack unchanged.
    """
    if thickness_nm < 0.0:
        raise ValueError(f"interfacial thickness must be non-negative, got {thickness_nm}")
    if len(indices) < 2:
        raise ValueError("need at least an ambient and a substrate")
    if thickness_nm == 0.0:
        return list(thicknesses), list(indices)

    film = indices[-2]
    substrate = indices[-1]
    layer_index = effective_medium_index(film, substrate, mix_fraction) if index is None else index

    return [*thicknesses, thickness_nm], [*indices[:-1], layer_index, indices[-1]]


# --- effects on the measured spectrum, DTFM-024 -----------------------------
#
# Unlike roughness and the interfacial layer, these three do not change the
# stack. They act on the spectrum the stack produces, so they are applied after
# the forward model rather than before it.


def average_over_thickness(
    forward,
    thickness_nm: float,
    sigma_nm: float,
    *,
    samples: int = 21,
):
    """Average reflectance over a spread of thicknesses within the spot (§4.5).

    A measurement spot is tens of microns across and the film is not perfectly
    uniform beneath it, so the detector sees a *sum of spectra* from slightly
    different thicknesses rather than one spectrum. Averaging ``R`` over
    ``d ~ N(d₀, s²)`` is the standard treatment.

    Averaging intensities, not amplitudes: light from different parts of the
    spot is mutually incoherent, so the powers add. Averaging the complex
    amplitude instead would model a single film of the mean thickness and miss
    the effect entirely.

    The signature is **damped high-order fringes**: fringes are periodic in
    ``d``, so a spread in ``d`` blurs them, and blurs the closely-spaced ones
    hardest. §4.5 notes finite spectrometer bandwidth produces the same
    signature — see :func:`apply_spectrometer_bandwidth` and the discussion in
    :mod:`tests.test_noise`.

    Parameters
    ----------
    forward : callable taking a thickness and returning a spectrum.
    thickness_nm : the mean thickness ``d₀``.
    sigma_nm : the standard deviation ``s``. Zero returns ``forward(d₀)``
        unchanged, so the effect is switchable.
    samples : quadrature points across the distribution. Odd, so the mean is
        always evaluated.
    """
    if sigma_nm < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma_nm}")
    if samples < 3 or samples % 2 == 0:
        raise ValueError(f"samples must be odd and at least 3, got {samples}")
    if sigma_nm == 0.0:
        return forward(thickness_nm)

    # Gauss-Hermite would need fewer points, but a plain truncated grid keeps
    # the weights inspectable and the ±4σ tail contributes below 1e-4.
    offsets = np.linspace(-4.0, 4.0, samples)
    weights = np.exp(-0.5 * offsets**2)
    weights = weights / weights.sum()

    total = None
    for offset, weight in zip(offsets, weights, strict=True):
        contribution = forward(thickness_nm + offset * sigma_nm) * weight
        total = contribution if total is None else total + contribution
    return total


def apply_spectrometer_bandwidth(wavelengths_nm, reflectance, fwhm_nm: float):
    """Convolve the spectrum with the instrument's slit function (§4.5).

    No spectrometer resolves a single wavelength: each detector element collects
    a band, so the recorded value at ``λ`` is an average of nearby ones. Modelled
    as a Gaussian of the stated full width at half maximum.

    Like spot non-uniformity this **damps high-order fringes** — closely spaced
    fringes are averaged away while widely spaced ones survive. That the two
    effects share a signature is §4.5's point, and it is a degeneracy in the
    *noise model* sitting on top of the degeneracy in the physics.

    Uses explicit normalised weights per output point rather than an FFT, so the
    result is correct at the ends of the band instead of wrapping around them —
    and a spectrum's edges are exactly where a fit is most sensitive.
    """
    if fwhm_nm < 0.0:
        raise ValueError(f"FWHM must be non-negative, got {fwhm_nm}")

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    values = np.asarray(reflectance, dtype=float)
    if fwhm_nm == 0.0:
        return values

    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    separation = wavelengths_nm[:, None] - wavelengths_nm[None, :]
    kernel = np.exp(-0.5 * (separation / sigma) ** 2)
    kernel = kernel / kernel.sum(axis=1, keepdims=True)
    return kernel @ values


def add_backside_reflection(
    reflectance,
    substrate_index,
    *,
    transmittance=None,
):
    """Add the incoherent reflection from the wafer's rear face (§4.5).

    A wafer is polished on both sides and is hundreds of microns thick — far more
    than the coherence length of a broadband source — so light reaching the back
    surface reflects and returns *without* a fixed phase relationship to the
    front reflection. It adds in intensity rather than amplitude, which is why it
    appears as a near-constant **offset** rather than as extra fringes.

    Approximated as one bounce off the substrate/air boundary, attenuated by the
    round trip through the front surface. The substrate must be transparent
    enough for light to reach the back at all; silicon in the visible absorbs it
    entirely, which is itself the reason the effect is usually seen in the
    infrared or on glass rather than on a silicon wafer in the visible.

    Set ``transmittance = 0`` to switch the effect off.
    """
    values = np.asarray(reflectance, dtype=float)
    index = np.asarray(substrate_index)

    back_face = np.abs((index - 1.0) / (index + 1.0)) ** 2
    if transmittance is None:
        transmittance = 1.0 - values
    transmittance = np.asarray(transmittance, dtype=float)

    if np.any(transmittance < 0.0) or np.any(transmittance > 1.0):
        raise ValueError("transmittance must lie in [0, 1]")

    return values + transmittance**2 * back_face


# --- instrument errors and detector noise, DTFM-025 -------------------------


def apply_wavelength_calibration(recorded_nm, scale: float = 0.0, offset_nm: float = 0.0):
    """True wavelength behind each recorded one: ``λ → λ(1+α) + β`` (§4.5).

    A spectrometer reports the wavelength it *believes* it measured. If its
    calibration has drifted, the light at the channel labelled 550 nm was really
    at a slightly different wavelength, so the spectrum should be evaluated
    there instead.

    §4.5 gives the consequence as a **systematic thickness bias**, and the
    mechanism is direct: fringe positions carry the thickness, so mislabelling
    the wavelength axis mislabels the thickness. Unlike detector noise this does
    not average away with repeated measurements — it is the same error every
    time, which is what makes it dangerous.
    """
    return np.asarray(recorded_nm, dtype=float) * (1.0 + scale) + offset_nm


def apply_baseline_drift(reflectance, gain: float = 1.0, offset: float = 0.0):
    """Multiplicative and additive drift in the measured level: ``R → aR + b``.

    Source intensity drifts, the reference measurement ages, stray light adds a
    floor. §4.5 notes this **correlates with the dispersion parameters**, which
    is the reason it matters here rather than being a nuisance: raising the whole
    curve looks a little like raising the index, so a fit can trade one against
    the other. Another entry for §10's atlas.
    """
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    return gain * np.asarray(reflectance, dtype=float) + offset


def add_detector_noise(
    reflectance,
    rng: np.random.Generator,
    *,
    shot_scale: float = 3e-3,
    read_sigma: float = 5e-4,
):
    """Shot noise plus additive detector noise, per §7.1.

    Shot noise is Poisson in photon count, so its standard deviation scales as
    ``sqrt(R)`` — the bright parts of a spectrum are noisier in absolute terms
    and quieter in relative terms. Read noise is independent of signal and
    dominates where R is small, which is exactly where an anti-reflection null
    or a fringe minimum sits.

    That split matters for §8: the aleatoric uncertainty the network is asked to
    predict is *not* uniform across a spectrum, and a model assuming constant
    noise would misstate it in both directions.

    Takes an explicit ``Generator`` rather than using global state, so a run is
    reproducible from its seed alone (§15).
    """
    if shot_scale < 0.0 or read_sigma < 0.0:
        raise ValueError("noise scales must be non-negative")

    values = np.asarray(reflectance, dtype=float)
    sigma = np.sqrt(shot_scale**2 * np.clip(values, 0.0, None) + read_sigma**2)
    return values + rng.normal(0.0, sigma)


@dataclass(frozen=True)
class Corruption:
    """Everything §4.5 can do to a measurement, in one configurable object.

    Defaults describe a decent instrument looking at a real film: a finite slit
    width, a little surface roughness, and both noise terms. That is four of the
    seven effects active, above the three §4.5 asks for. The remaining three —
    an interfacial layer, a calibration error, baseline drift — default to off
    because they are *faults* rather than facts of life, and the failure atlas
    of §10 wants to switch them on deliberately.
    """

    roughness_nm: float = 1.0
    interfacial_nm: float = 0.0
    spot_sigma_nm: float = 0.0
    bandwidth_fwhm_nm: float = 3.0
    wavelength_scale: float = 0.0
    wavelength_offset_nm: float = 0.0
    baseline_gain: float = 1.0
    baseline_offset: float = 0.0
    shot_scale: float = 3e-3
    read_sigma: float = 5e-4
    backside_substrate_index: float | None = None

    def stack_with_defects(self, thicknesses: Sequence, indices: Sequence) -> tuple[list, list]:
        """Apply the two defects that change the stack (DTFM-023).

        These belong here rather than in :func:`corrupt` because they alter what
        the film *is*, not what the instrument records — so they must be applied
        before the forward model runs. Keeping them on the same object means one
        description of the whole measurement, and means :attr:`active` cannot
        claim an effect that nothing applies.
        """
        thicknesses, indices = add_surface_roughness(thicknesses, indices, self.roughness_nm)
        return add_interfacial_layer(thicknesses, indices, self.interfacial_nm)

    @property
    def active(self) -> tuple[str, ...]:
        """Which effects are switched on — useful as a failure-atlas row (§10).

        The first two are applied by :meth:`stack_with_defects`, the rest by
        :func:`corrupt`.
        """
        names = []
        if self.roughness_nm:
            names.append("roughness")
        if self.interfacial_nm:
            names.append("interfacial layer")
        if self.spot_sigma_nm:
            names.append("spot non-uniformity")
        if self.bandwidth_fwhm_nm:
            names.append("bandwidth")
        if self.wavelength_scale or self.wavelength_offset_nm:
            names.append("wavelength calibration")
        if self.baseline_gain != 1.0 or self.baseline_offset:
            names.append("baseline drift")
        if self.shot_scale or self.read_sigma:
            names.append("detector noise")
        if self.backside_substrate_index is not None:
            names.append("backside reflection")
        return tuple(names)


def corrupt(
    wavelengths_nm,
    forward,
    thickness_nm: float,
    corruption: Corruption | None = None,
    rng: np.random.Generator | None = None,
):
    """Turn an ideal spectrum into a measured one — §7.1's ``corrupt``.

    ``forward(wavelengths_nm, thickness_nm)`` must return a clean reflectance,
    built from a stack that has already been through
    :meth:`Corruption.stack_with_defects` — roughness and the interfacial layer
    change what the film is, so they cannot be applied to a finished spectrum.

    Order is not arbitrary — it follows the light:

    1. **wavelength calibration** — decides which wavelengths were really seen,
       so it must come before the spectrum is evaluated at all
    2. **spot non-uniformity** — the detector sums over the illuminated area
    3. **bandwidth** — the instrument then averages neighbouring wavelengths
    4. **backside reflection** — added incoherently at the detector
    5. **baseline drift** — the instrument's own gain and offset
    6. **detector noise** — last, because nothing downstream of the detector
       smooths it

    Applying noise before the blur would let the instrument average away noise it
    has not generated yet, which understates the uncertainty §8 has to model.
    """
    corruption = corruption or Corruption()
    rng = rng or np.random.default_rng()

    recorded = np.asarray(wavelengths_nm, dtype=float)
    true_wavelengths = apply_wavelength_calibration(
        recorded, corruption.wavelength_scale, corruption.wavelength_offset_nm
    )

    spectrum = average_over_thickness(
        lambda d: np.asarray(forward(true_wavelengths, d), dtype=float),
        thickness_nm,
        corruption.spot_sigma_nm,
    )
    spectrum = apply_spectrometer_bandwidth(recorded, spectrum, corruption.bandwidth_fwhm_nm)

    if corruption.backside_substrate_index is not None:
        spectrum = add_backside_reflection(spectrum, corruption.backside_substrate_index)

    spectrum = apply_baseline_drift(
        spectrum, corruption.baseline_gain, corruption.baseline_offset
    )
    return add_detector_noise(
        spectrum, rng, shot_scale=corruption.shot_scale, read_sigma=corruption.read_sigma
    )


def add_ellipsometer_noise(
    psi,
    delta,
    rng: np.random.Generator,
    *,
    sigma_rad: float = 1e-3,
):
    """Angular noise on ``(Ψ, Δ)`` — the ellipsometric analogue of §4.5's detector noise.

    Different in kind from :func:`add_detector_noise`, not merely in scale.
    Reflectance noise is dominated by photon statistics and scales as ``sqrt(R)``.
    An ellipsometer measures a *polarisation state*, and its uncertainty is
    angular and roughly signal-independent — set by the analyser's angular
    resolution and the polariser extinction ratio rather than by photon count.

    The default 1e-3 rad is 0.057°, a good but unexceptional instrument.
    DTFM-028 measured the whole ellipsometry advantage as conditional on this
    number: at 0.57° the technique is *worse* than reflectometry.

    Baseline drift is deliberately absent. ``ρ = r_p/r_s`` is a ratio, so a
    source that drifts in intensity scales both and cancels — which removes the
    §4.5 term DTFM-025 measured as correlating above 0.99 with refractive index.
    That cancellation is a real advantage of the technique and should not be
    modelled away by adding a gain term here out of symmetry with reflectance.
    """
    if sigma_rad < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma_rad}")

    psi = np.asarray(psi, dtype=float)
    delta = np.asarray(delta, dtype=float)
    noisy_psi = psi + rng.normal(0.0, sigma_rad, psi.shape)
    noisy_delta = delta + rng.normal(0.0, sigma_rad, delta.shape)

    # Ψ is an amplitude ratio in [0, π/2] and Δ wraps on the circle. Noise must
    # respect both, or a sample near a boundary becomes physically impossible.
    noisy_psi = np.clip(noisy_psi, 0.0, np.pi / 2)
    noisy_delta = (noisy_delta + np.pi) % (2 * np.pi) - np.pi
    return noisy_psi, noisy_delta
