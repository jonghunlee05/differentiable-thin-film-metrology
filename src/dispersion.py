"""Cauchy, Sellmeier and Lorentz dispersion; real n,k loaders.

Spec §4.4.
Loader for vendored refractiveindex.info data by DTFM-019; the dispersion models
fitted to it by DTFM-020 - DTFM-022.

Real optical constants matter here beyond realism. Every refractive index in the
project so far has been a number chosen by hand, and §4.4 is explicit: *fit the
models to that data rather than inventing coefficients*. An invented index makes
a spectrum that looks entirely plausible and describes no material that exists.

See ``data/refractiveindex/PROVENANCE.md`` for sources, citations and licence.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import yaml
from numpy.typing import NDArray

__all__ = [
    "MATERIALS",
    "CauchyFit",
    "available_materials",
    "cauchy_n",
    "fit_cauchy",
    "load_nk",
    "material_range_nm",
]

_DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "refractiveindex"


@dataclass(frozen=True)
class _Material:
    path: str
    citation: str


#: The four materials of §4.4. Each entry is the standard reference for that
#: material; the reasoning for each choice is in PROVENANCE.md.
MATERIALS: dict[str, _Material] = {
    "SiO2": _Material("SiO2/Malitson.yml", "Malitson, J. Opt. Soc. Am. 55, 1205 (1965)"),
    "Si3N4": _Material("Si3N4/Luke.yml", "Luke et al., Opt. Lett. 40, 4823 (2015)"),
    "TiO2": _Material("TiO2/Siefke.yml", "Siefke et al., Adv. Opt. Mater. 4, 1780 (2016)"),
    "Si": _Material("Si/Aspnes.yml", "Aspnes and Studna, Phys. Rev. B 27, 985 (1983)"),
}


def available_materials() -> list[str]:
    """Names accepted by :func:`load_nk`."""
    return sorted(MATERIALS)


def _read_dataset(material: str) -> dict:
    if material not in MATERIALS:
        raise KeyError(
            f"unknown material {material!r}; available: {', '.join(available_materials())}"
        )
    path = _DATA_ROOT / MATERIALS[material].path
    if not path.exists():
        raise FileNotFoundError(
            f"vendored optical constants missing at {path}. Restore them with "
            "scripts/fetch_optical_constants.py — see data/refractiveindex/PROVENANCE.md."
        )
    with path.open() as handle:
        document = yaml.safe_load(handle)

    entries = [d for d in document["DATA"] if d["type"] in ("formula 1", "tabulated nk")]
    if not entries:
        types = [d["type"] for d in document["DATA"]]
        raise NotImplementedError(
            f"{material}: no supported dispersion entry, found {types}. Only Sellmeier "
            "('formula 1') and tabulated n,k are read; adding another means adding its "
            "formula, not guessing at it."
        )
    return entries[0]


def material_range_nm(material: str) -> tuple[float, float]:
    """Wavelengths, in nm, over which the published data is valid."""
    entry = _read_dataset(material)
    if entry["type"] == "formula 1":
        low, high = (float(x) for x in str(entry["wavelength_range"]).split())
    else:
        table = _parse_tabulated(entry["data"])
        low, high = float(table[0, 0]), float(table[-1, 0])
    return low * 1e3, high * 1e3


def _parse_tabulated(block: str) -> NDArray[np.float64]:
    """Rows of ``wavelength_um n k`` into an (N, 3) array, ascending in wavelength."""
    rows = [line.split() for line in block.strip().splitlines() if line.strip()]
    table = np.array(rows, dtype=float)
    return table[np.argsort(table[:, 0])]


def _sellmeier_formula_1(coefficients: list[float], wavelength_um: NDArray) -> NDArray:
    """refractiveindex.info "formula 1": ``n² − 1 = c₀ + Σ Bᵢλ²/(λ² − Cᵢ²)``.

    The database's own published coefficients, evaluated as given. This is
    reading the file, not modelling — fitting *this project's* Sellmeier model to
    the resulting data is DTFM-021, and keeping the two separate is what lets
    that ticket's fit residuals mean anything.
    """
    c0, pairs = coefficients[0], coefficients[1:]
    n_squared = np.full_like(wavelength_um, 1.0 + c0)
    for b, c in zip(pairs[0::2], pairs[1::2], strict=True):
        n_squared = n_squared + b * wavelength_um**2 / (wavelength_um**2 - c**2)
    return np.sqrt(n_squared)


def load_nk(
    material: str, wavelengths_nm: NDArray | list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Refractive index and extinction coefficient on a requested grid.

    Parameters
    ----------
    material : one of :func:`available_materials`.
    wavelengths_nm : wavelengths in nanometres.

    Returns
    -------
    ``(n, k)`` as float64 arrays, in the ``ñ = n + ik`` convention with ``k >= 0``
    — the same convention ``src.tmm_torch`` requires, and the one these files use.

    Raises
    ------
    ValueError
        If any requested wavelength lies outside the published validity range.
        Extrapolating a dispersion curve invents a material: the number returned
        would look reasonable, carry no warning, and describe nothing real. §4.4
        asks for models fitted to data, so the honest response outside the data
        is to refuse. Use :func:`material_range_nm` to check first.
    """
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    low_nm, high_nm = material_range_nm(material)

    outside = (wavelengths_nm < low_nm) | (wavelengths_nm > high_nm)
    if np.any(outside):
        offenders = wavelengths_nm[outside]
        raise ValueError(
            f"{material} data covers {low_nm:.1f}-{high_nm:.1f} nm; "
            f"{offenders.size} requested wavelength(s) fall outside, from "
            f"{offenders.min():.1f} to {offenders.max():.1f} nm. Extrapolating a "
            "dispersion curve invents optical constants rather than measuring them."
        )

    entry = _read_dataset(material)
    wavelength_um = wavelengths_nm * 1e-3

    if entry["type"] == "formula 1":
        coefficients = [float(x) for x in str(entry["coefficients"]).split()]
        n = _sellmeier_formula_1(coefficients, wavelength_um)
        # A Sellmeier form describes a transparent material by construction: it
        # has no imaginary part to report, which is why §4.4 sends absorbing
        # materials to a Lorentz oscillator instead (DTFM-022).
        k = np.zeros_like(n)
    else:
        table = _parse_tabulated(entry["data"])
        n = np.interp(wavelength_um, table[:, 0], table[:, 1])
        k = np.interp(wavelength_um, table[:, 0], table[:, 2])

    return np.asarray(n, dtype=float), np.asarray(k, dtype=float)


# --- Cauchy dispersion, DTFM-020 -------------------------------------------


@dataclass(frozen=True)
class CauchyFit:
    """A Cauchy model fitted to measured data, with the residuals that justify it.

    Coefficients follow the usual convention with **wavelength in micrometres**,
    so ``A`` is dimensionless, ``B`` is µm² and ``C`` is µm⁴ — the form the
    coefficients are quoted in throughout the literature, which makes them
    comparable to published values rather than only to themselves.
    """

    a: float
    b: float
    c: float
    rms_residual: float
    max_residual: float
    range_nm: tuple[float, float]
    max_k_in_range: float

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"n(λ) = {self.a:.5f} + {self.b:.5f}/λ² + {self.c:.5f}/λ⁴  (λ in µm), "
            f"rms {self.rms_residual:.2e} over {self.range_nm[0]:.0f}-{self.range_nm[1]:.0f} nm"
        )


def cauchy_n(coefficients, wavelengths_nm):
    """Evaluate ``n(λ) = A + B/λ² + C/λ⁴``.

    Parameters
    ----------
    coefficients : a :class:`CauchyFit`, or any ``(A, B, C)`` sequence — including
        torch tensors carrying ``requires_grad``.
    wavelengths_nm : numpy array or torch tensor.

    Plain arithmetic throughout, so this stays differentiable when handed
    tensors. §7.1 samples dispersion coefficients as parameters and §7.3
    backpropagates the reconstruction loss through them, so the model has to
    survive autograd rather than only produce numbers.
    """
    if isinstance(coefficients, CauchyFit):
        a, b, c = coefficients.a, coefficients.b, coefficients.c
    else:
        a, b, c = coefficients

    wavelength_um = wavelengths_nm * 1e-3
    inverse_squared = 1.0 / wavelength_um**2
    return a + b * inverse_squared + c * inverse_squared**2


def fit_cauchy(
    material: str,
    wavelengths_nm: NDArray | list[float] | None = None,
    *,
    terms: int = 3,
    allow_absorbing: bool = False,
) -> CauchyFit:
    """Fit §4.4's Cauchy model to the measured data for ``material``.

    Cauchy is linear in its coefficients once written in ``x = 1/λ²``, so this is
    an exact linear least-squares solve rather than an iterative fit — there is
    no starting guess to get wrong and no local minimum to fall into.

    Parameters
    ----------
    material : one of :func:`available_materials`.
    wavelengths_nm : grid to fit over. Defaults to 400-800 nm, clipped to the
        material's published range.
    terms : 2 fits ``A + B/λ²``; 3 adds the ``C/λ⁴`` term.
    allow_absorbing : see below.

    Raises
    ------
    ValueError
        If the range contains appreciable absorption and ``allow_absorbing`` is
        left False.

    Notes
    -----
    **Why absorption is refused by default.** Cauchy is an empirical expansion in
    even powers of ``1/λ``, valid only where the material is transparent — below
    the absorption edge, in §4.4's phrasing. Physically it is the first terms of
    a Sellmeier form far from any resonance: with the nearest absorption band
    remote, ``λ²/(λ² − C²)`` expands in ``1/λ²`` and the series converges
    quickly. Approach the edge and that expansion diverges, so the fit degrades
    smoothly and misleadingly — the residuals grow, but the curve still looks
    like a dispersion curve.

    It also has no imaginary part at all, so it cannot represent ``k``. Fitting
    it across an absorption edge produces a real index for a material that is
    absorbing there, which is not a poor fit but a wrong model. §4.4 sends those
    materials to a Lorentz oscillator instead (DTFM-022).

    The flag exists so that failure can be *demonstrated* deliberately, which the
    tests do — not as a convenience for fitting absorbing materials.
    """
    if terms not in (2, 3):
        raise ValueError(f"terms must be 2 or 3, got {terms}")

    low_nm, high_nm = material_range_nm(material)
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(max(low_nm, 400.0), min(high_nm, 800.0), 200)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    n, k = load_nk(material, wavelengths_nm)
    max_k = float(np.max(k))
    if max_k > 1e-3 and not allow_absorbing:
        raise ValueError(
            f"{material} absorbs over {wavelengths_nm.min():.0f}-{wavelengths_nm.max():.0f} nm "
            f"(max k = {max_k:.3g}). Cauchy is valid only below the absorption edge and has no "
            "imaginary part to represent k with, so fitting here would give a real index for an "
            "absorbing material — a wrong model rather than a poor fit. Use a Lorentz oscillator "
            "(§4.4), restrict the range, or pass allow_absorbing=True to demonstrate the failure."
        )

    wavelength_um = wavelengths_nm * 1e-3
    inverse_squared = 1.0 / wavelength_um**2
    columns = [np.ones_like(inverse_squared), inverse_squared]
    if terms == 3:
        columns.append(inverse_squared**2)

    coefficients, *_ = np.linalg.lstsq(np.column_stack(columns), n, rcond=None)
    padded = list(coefficients) + [0.0] * (3 - len(coefficients))

    residual = np.column_stack(columns) @ coefficients - n
    return CauchyFit(
        a=float(padded[0]),
        b=float(padded[1]),
        c=float(padded[2]),
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        max_residual=float(np.max(np.abs(residual))),
        range_nm=(float(wavelengths_nm.min()), float(wavelengths_nm.max())),
        max_k_in_range=max_k,
    )
