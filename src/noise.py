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

import numpy as np

__all__ = [
    "add_interfacial_layer",
    "add_surface_roughness",
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
