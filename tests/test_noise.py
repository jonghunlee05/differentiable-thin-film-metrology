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
