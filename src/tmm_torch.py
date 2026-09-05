"""Differentiable transfer-matrix forward model.

Spec §4.2-§4.3.
Single-interface Fresnel in torch by DTFM-008; branch-cut guard for absorbing
media by DTFM-010; interface and layer matrices by DTFM-011; the full stack
by DTFM-012.

This is the model the project actually uses. `src.fresnel` is its numpy
counterpart and exists to be disagreed with: DTFM-008 asserts the two produce
the same numbers, so a later refactor here has something independent to fail
against. Conventions — including the `r_p` sign discussed in `src.fresnel` —
are shared deliberately; a mismatch between the two would put a spurious sign
in every stack built on top.

Everything is computed in complex128. Thin-film phases are complex by nature
and the transfer matrices in §4.3 are complex throughout, so carrying complex
dtype from the single interface upward avoids a cast in the middle of the
stack product where a dropped imaginary part would be hard to see.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = [
    "as_complex",
    "cos_theta_t",
    "fresnel_r",
    "fresnel_t",
    "interface_matrix",
    "is_forward",
    "layer_matrix",
    "layer_phase",
    "reflectance",
    "stack_cosines",
    "stack_matrix",
    "stack_r",
    "spectra",
    "stack_reflectance",
]

_CDTYPE = torch.complex128


def as_complex(x: torch.Tensor | float | int) -> torch.Tensor:
    """Promote a real scalar or tensor to complex128, preserving autograd.

    A real tensor is promoted with an exactly ``+0.0`` imaginary part. That sign
    matters: ``sqrt`` of a negative real takes the branch chosen by the sign of
    the zero imaginary component, so a ``-0.0`` here would silently flip the
    root under total internal reflection.
    """
    if not isinstance(x, torch.Tensor):
        # A python complex must be built as complex directly: torch.tensor
        # rejects it under a float dtype rather than coercing.
        dtype = _CDTYPE if isinstance(x, complex) else torch.float64
        x = torch.tensor(x, dtype=dtype)
    if x.is_complex():
        return x.to(_CDTYPE)
    return torch.complex(x.to(torch.float64), torch.zeros_like(x, dtype=torch.float64))


def is_forward(n_j: torch.Tensor, cos_j: torch.Tensor) -> torch.Tensor:
    """Whether ``cos_j`` is the root describing a wave that decays as it travels.

    The transmitted wave carries ``exp(i·(2π/λ)·ñ_j·d·cosθ_j)``. Its magnitude is
    ``exp(−(2π/λ)·d·Im(ñ_j cosθ_j))``, so the wave decays exactly when
    ``Im(ñ_j cosθ_j) ≥ 0`` and grows when that is negative. Absorbing media must
    absorb; the other root is a medium that emits light it was never given.
    """
    return (n_j * cos_j).imag >= 0.0


def cos_theta_t(
    n_i: torch.Tensor | float,
    n_j: torch.Tensor | float,
    theta_i: torch.Tensor | float,
) -> torch.Tensor:
    """Cosine of the transmitted angle, from Snell's law ``n_i sinθ_i = n_j sinθ_j``.

    Complex throughout, so total internal reflection gives an imaginary cosine
    and ``|r| = 1`` rather than a ``nan``. Mirrors ``src.fresnel.cos_theta_t``.

    Branch selection
    ----------------
    Spec §4.2 warns that this square root has two roots and that taking the
    wrong one produces gain instead of absorption — silently, since nothing
    raises. ``torch.sqrt`` returns the principal root, which is unphysical for a
    substantial fraction of absorbing cases, so the physical root is selected
    explicitly here and the choice is then asserted rather than trusted.

    §4.2 states the condition as ``Im(cosθ_j) ≥ 0``. That is correct whenever the
    *incident* medium is transparent, but not in general: with an absorbing
    incident medium it still admits gain, which is the case that arises at every
    interior interface of a stack containing an absorbing film. The condition
    used is ``Im(ñ_j cosθ_j) ≥ 0`` — see :func:`is_forward` — which is the one
    that actually makes the transmitted wave decay, and which agrees with the
    reference `tmm` package wherever that package will answer.
    """
    n_i, n_j = as_complex(n_i), as_complex(n_j)
    theta_i = as_complex(theta_i)

    _require_passive(n_i, "incident")
    _require_passive(n_j, "transmitted")

    sin_theta_t = (n_i / n_j) * torch.sin(theta_i)
    cos_j = torch.sqrt(1.0 - sin_theta_t**2)

    return torch.where(is_forward(n_j, cos_j), cos_j, -cos_j)


def _require_passive(n: torch.Tensor, which: str) -> None:
    """Reject ``Im(ñ) < 0`` — a medium that emits light it was never given.

    This is where the branch-cut condition is actually asserted. Selecting the
    root by :func:`is_forward` makes ``Im(ñ_j cosθ_j) >= 0`` true by
    construction, so re-checking it afterwards could never fail and would be
    reassurance rather than a test. The assumption that genuinely needs
    guarding is the one on the *input*: every medium here is passive.

    It earns its place. Optical-constant tables are published under both the
    ``n + ik`` and ``n − ik`` conventions, and §4.4 has this project loading real
    data from refractiveindex.info. Ingesting it under the wrong sign would
    otherwise produce a perfectly plausible spectrum from an impossible
    material, with nothing anywhere raising.
    """
    if bool(torch.any(n.imag < -1e-15)):
        raise AssertionError(
            f"the {which} medium has Im(n) < 0, which describes a gain medium "
            "amplifying the light passing through it. Passive media require "
            "Im(n) >= 0 — check whether the optical constants were tabulated "
            "under the n - ik convention rather than n + ik."
        )


def fresnel_r(
    n_i: torch.Tensor | float,
    n_j: torch.Tensor | float,
    theta_i: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Amplitude reflection coefficients ``(r_s, r_p)`` at one interface, §4.2.

    Differentiable with respect to any argument carrying ``requires_grad``.
    Inputs broadcast against each other.
    """
    n_i, n_j = as_complex(n_i), as_complex(n_j)
    cos_i = torch.cos(as_complex(theta_i))
    cos_j = cos_theta_t(n_i, n_j, theta_i)
    return fresnel_r_from_cosines(n_i, n_j, cos_i, cos_j)


def fresnel_r_from_cosines(
    n_i: torch.Tensor, n_j: torch.Tensor, cos_i: torch.Tensor, cos_j: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(r_s, r_p)`` from cosines already in hand, §4.2.

    Inside a stack the cosine in each medium is obtained once from the Snell
    invariant and reused. Recovering an angle with ``arccos`` and taking its
    cosine again would be a round trip through a multivalued inverse — the
    branch hazard of DTFM-010 in a second guise — so the cosines are passed
    directly and the angle never reconstructed.
    """
    r_s = (n_i * cos_i - n_j * cos_j) / (n_i * cos_i + n_j * cos_j)
    r_p = (n_j * cos_i - n_i * cos_j) / (n_j * cos_i + n_i * cos_j)
    return r_s, r_p


def reflectance(
    n_i: torch.Tensor | float,
    n_j: torch.Tensor | float,
    theta_i: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflected intensity fraction ``(R_s, R_p) = (|r_s|², |r_p|²)``.

    Uses ``abs()**2`` rather than ``r.real**2 + r.imag**2`` so the gradient
    stays defined; both are real-valued and differentiable.
    """
    r_s, r_p = fresnel_r(n_i, n_j, theta_i)
    return r_s.abs() ** 2, r_p.abs() ** 2


def fresnel_t(
    n_i: torch.Tensor | float,
    n_j: torch.Tensor | float,
    theta_i: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Amplitude transmission coefficients ``(t_s, t_p)`` at one interface.

    Needed because the §4.3 interface matrix carries a factor ``1/t_ij``. Written
    with the same denominators as :func:`fresnel_r`, so the two obey the Stokes
    relations that :mod:`tests.test_tmm` asserts::

        t_s = 1 + r_s
        t_p = (n_i / n_j) · (1 + r_p)

    The p relation carries the index ratio because of the §4.2 sign convention
    for ``r_p``; see the discussion in :mod:`src.fresnel`.
    """
    n_i, n_j = as_complex(n_i), as_complex(n_j)
    cos_i = torch.cos(as_complex(theta_i))
    cos_j = cos_theta_t(n_i, n_j, theta_i)
    return fresnel_t_from_cosines(n_i, n_j, cos_i, cos_j)


def fresnel_t_from_cosines(
    n_i: torch.Tensor, n_j: torch.Tensor, cos_i: torch.Tensor, cos_j: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(t_s, t_p)`` from cosines already in hand. See :func:`fresnel_r_from_cosines`."""
    t_s = (2.0 * n_i * cos_i) / (n_i * cos_i + n_j * cos_j)
    t_p = (2.0 * n_i * cos_i) / (n_j * cos_i + n_i * cos_j)
    return t_s, t_p


def _stack_2x2(m00: torch.Tensor, m01: torch.Tensor,
               m10: torch.Tensor, m11: torch.Tensor) -> torch.Tensor:
    """Assemble a batch of 2x2 matrices with shape ``(..., 2, 2)``."""
    row0 = torch.stack((m00, m01), dim=-1)
    row1 = torch.stack((m10, m11), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def interface_matrix(
    n_i: torch.Tensor | float,
    n_j: torch.Tensor | float,
    theta_i: torch.Tensor | float,
    polarisation: str,
) -> torch.Tensor:
    """The §4.3 interface matrix ``I_ij = (1/t_ij)·[[1, r_ij], [r_ij, 1]]``.

    Returns shape ``(..., 2, 2)``, broadcasting over wavelength and any swept
    parameter, so a whole spectrum is one call.

    Parameters
    ----------
    polarisation : ``"s"`` or ``"p"``. The two are independent until they are
        combined into an intensity, and §4.3 requires both.
    """
    n_i, n_j = as_complex(n_i), as_complex(n_j)
    cos_i = torch.cos(as_complex(theta_i))
    cos_j = cos_theta_t(n_i, n_j, theta_i)
    return interface_matrix_from_cosines(n_i, n_j, cos_i, cos_j, polarisation)


def interface_matrix_from_cosines(
    n_i: torch.Tensor,
    n_j: torch.Tensor,
    cos_i: torch.Tensor,
    cos_j: torch.Tensor,
    polarisation: str,
) -> torch.Tensor:
    """``I_ij`` from cosines already in hand. See :func:`fresnel_r_from_cosines`."""
    r_s, r_p = fresnel_r_from_cosines(n_i, n_j, cos_i, cos_j)
    t_s, t_p = fresnel_t_from_cosines(n_i, n_j, cos_i, cos_j)
    r, t = _select_polarisation(polarisation, (r_s, r_p), (t_s, t_p))

    ones = torch.ones_like(r)
    return _stack_2x2(ones, r, r, ones) / t.unsqueeze(-1).unsqueeze(-1)


def layer_phase(
    n: torch.Tensor | float,
    thickness: torch.Tensor | float,
    wavelength: torch.Tensor | float,
    cos_theta: torch.Tensor,
) -> torch.Tensor:
    """Phase accumulated crossing a layer, ``δ = (2π/λ)·ñ·d·cosθ`` (§4.3).

    ``thickness`` and ``wavelength`` must share units; only their ratio enters.
    For an absorbing layer ``δ`` has a positive imaginary part, which is what
    makes ``exp(iδ)`` a decay — the branch selection in :func:`cos_theta_t` is
    what guarantees that sign.
    """
    n = as_complex(n)
    two_pi_over_lambda = 2.0 * torch.pi / as_complex(wavelength)
    return two_pi_over_lambda * n * as_complex(thickness) * cos_theta


def layer_matrix(
    n: torch.Tensor | float,
    thickness: torch.Tensor | float,
    wavelength: torch.Tensor | float,
    cos_theta: torch.Tensor,
) -> torch.Tensor:
    """The §4.3 propagation matrix ``L_j = diag(exp(−iδ), exp(+iδ))``.

    Returns shape ``(..., 2, 2)``. A zero thickness gives the identity, which is
    the check that the phase convention has not picked up a stray factor.

    A transfer matrix maps the fields at the far side of the layer back to the
    near side, so it un-propagates rather than propagates. For an absorbing
    layer that inverts which entry shrinks: ``|L[0,0]| = exp(+Im δ) > 1`` while
    ``|L[1,1]| = exp(−Im δ) < 1``, with the two exact reciprocals and
    ``det L = 1``. The entry above one is not gain — reading it as such is the
    obvious way to mistakenly conclude the branch selection in
    :func:`cos_theta_t` is broken.
    """
    delta = layer_phase(n, thickness, wavelength, cos_theta)
    zero = torch.zeros_like(delta)
    return _stack_2x2(torch.exp(-1j * delta), zero, zero, torch.exp(1j * delta))


def _select_polarisation(
    polarisation: str,
    s_and_p: tuple[torch.Tensor, torch.Tensor],
    other_s_and_p: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if polarisation == "s":
        return s_and_p[0], other_s_and_p[0]
    if polarisation == "p":
        return s_and_p[1], other_s_and_p[1]
    raise ValueError(f"polarisation must be 's' or 'p', got {polarisation!r}")


def stack_cosines(
    indices: Sequence[torch.Tensor | float],
    theta_0: torch.Tensor | float,
) -> list[torch.Tensor]:
    """Cosine of the propagation angle in every medium of a stack.

    Snell's law is a conservation law across the whole stack: ``ñ sinθ`` takes
    the same value in every layer. So each cosine follows from the *ambient*
    medium alone, in one step, rather than by chaining interface to interface —
    which would accumulate both round-off and branch decisions.
    """
    n_0 = as_complex(indices[0])
    theta_0 = as_complex(theta_0)
    cos_0 = torch.cos(theta_0)
    return [cos_0] + [cos_theta_t(n_0, n, theta_0) for n in indices[1:]]


def stack_matrix(
    wavelength: torch.Tensor | float,
    thicknesses: Sequence[torch.Tensor | float],
    indices: Sequence[torch.Tensor | float],
    theta_0: torch.Tensor | float,
    polarisation: str,
) -> torch.Tensor:
    """The §4.3 characteristic matrix of a stack.

    ``M = I_01 · L_1 · I_12 · L_2 · … · I_{N−1,N}``

    Parameters
    ----------
    wavelength : in the same units as ``thicknesses``; only the ratio enters.
    thicknesses : the ``N − 1`` finite layers, ambient and substrate excluded —
        both are semi-infinite and have no thickness to accumulate phase over.
    indices : ``N + 1`` refractive indices, ambient first and substrate last.
        Absorbing media are ``n + ik`` with ``k >= 0``.
    theta_0 : angle of incidence in the ambient medium, radians.
    polarisation : ``"s"`` or ``"p"``.

    Returns shape ``(..., 2, 2)``, broadcasting over wavelength and over any
    swept parameter.
    """
    if len(indices) < 2:
        raise ValueError("a stack needs at least an ambient and a substrate medium")
    if len(thicknesses) != len(indices) - 2:
        raise ValueError(
            f"got {len(thicknesses)} thicknesses for {len(indices)} media; expected "
            f"{len(indices) - 2}. Ambient and substrate are semi-infinite and take no "
            "thickness."
        )

    n = [as_complex(x) for x in indices]
    cos = stack_cosines(indices, theta_0)

    matrix = interface_matrix_from_cosines(n[0], n[1], cos[0], cos[1], polarisation)
    for j in range(1, len(n) - 1):
        layer = layer_matrix(n[j], thicknesses[j - 1], wavelength, cos[j])
        boundary = interface_matrix_from_cosines(n[j], n[j + 1], cos[j], cos[j + 1], polarisation)
        matrix = matrix @ layer @ boundary
    return matrix


def stack_r(
    wavelength: torch.Tensor | float,
    thicknesses: Sequence[torch.Tensor | float],
    indices: Sequence[torch.Tensor | float],
    theta_0: torch.Tensor | float = 0.0,
    polarisation: str = "s",
) -> torch.Tensor:
    """Amplitude reflection coefficient of a stack, ``r = M[1,0] / M[0,0]`` (§4.3)."""
    matrix = stack_matrix(wavelength, thicknesses, indices, theta_0, polarisation)
    return matrix[..., 1, 0] / matrix[..., 0, 0]


def stack_reflectance(
    wavelength: torch.Tensor | float,
    thicknesses: Sequence[torch.Tensor | float],
    indices: Sequence[torch.Tensor | float],
    theta_0: torch.Tensor | float = 0.0,
    polarisation: str = "s",
) -> torch.Tensor:
    """Reflectance ``R = |r|²`` of a stack — the measured quantity of §1.

    This is the forward model: film parameters in, spectrum out. Everything the
    project does downstream calls it — the dataset generator of §7.1, the
    classical fit of §6, and the reconstruction loss of §7.3, which is why it
    must stay differentiable.
    """
    return stack_r(wavelength, thicknesses, indices, theta_0, polarisation).abs() ** 2


def spectra(
    wavelengths: torch.Tensor,
    thicknesses: torch.Tensor,
    indices: Sequence[torch.Tensor | float],
    theta_0: torch.Tensor | float = 0.0,
    polarisation: str = "s",
) -> torch.Tensor:
    """Reflectance spectra for a whole batch of films at once.

    The training-loop entry point. §7.1 samples a batch of parameter vectors and
    needs one spectrum per sample; computing them in a python loop would make the
    simulator, not the network, the bottleneck.

    ``stack_reflectance`` already broadcasts, so this adds no physics — it exists
    to own the reshaping. Getting a batch axis to line up against a wavelength
    axis by hand is a silent-failure risk: mismatched shapes broadcast into a
    plausible result of the wrong meaning rather than raising, and the resulting
    training data would be wrong in a way no test downstream would attribute here.

    Parameters
    ----------
    wavelengths : shape ``(W,)``.
    thicknesses : shape ``(B, L)`` — one row per film, one column per layer.
        A 1-D tensor of shape ``(L,)`` is treated as a single film.
    indices : ``L + 2`` entries, ambient first and substrate last. Each may be a
        scalar, a spectrum of shape ``(W,)`` for a dispersive material (§4.4), or
        shape ``(B, W)`` when the index itself is a fitted parameter.
    theta_0 : angle of incidence in the ambient medium, radians.

    Returns
    -------
    Reflectance of shape ``(B, W)``.
    """
    if thicknesses.ndim == 1:
        thicknesses = thicknesses.unsqueeze(0)
    if thicknesses.ndim != 2:
        raise ValueError(
            f"thicknesses must be (B, L) or (L,), got shape {tuple(thicknesses.shape)}"
        )
    if wavelengths.ndim != 1:
        raise ValueError(f"wavelengths must be 1-D of shape (W,), got {tuple(wavelengths.shape)}")

    n_layers = thicknesses.shape[1]
    if len(indices) != n_layers + 2:
        raise ValueError(
            f"got {len(indices)} indices for {n_layers} layers; expected {n_layers + 2} "
            "(ambient, each layer, substrate)."
        )

    # Batch on axis 0, wavelength on axis 1. Layer thicknesses do not vary with
    # wavelength, and an index that is already (B, W) is left alone.
    per_layer = [thicknesses[:, k : k + 1] for k in range(n_layers)]
    lam = wavelengths.reshape(1, -1)
    broadcast_indices = [
        n if isinstance(n, torch.Tensor) and n.ndim == 2 else _as_row(n) for n in indices
    ]

    return stack_reflectance(lam, per_layer, broadcast_indices, theta_0, polarisation)


def _as_row(n: torch.Tensor | float) -> torch.Tensor | float:
    """Reshape a per-wavelength index to ``(1, W)``; leave scalars untouched."""
    if isinstance(n, torch.Tensor) and n.ndim == 1:
        return n.reshape(1, -1)
    return n
