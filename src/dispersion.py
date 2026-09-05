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

__all__ = ["MATERIALS", "available_materials", "load_nk", "material_range_nm"]

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
