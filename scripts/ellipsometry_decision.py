"""The Fisher-information evidence behind DTFM-028.

Spec §3 asks for exactly this: "quantify with Fisher information *how much* the
degeneracy shrinks moving from reflectance-only to full ellipsometry. That is an
information-content question with a real number attached."

    python scripts/ellipsometry_decision.py

Committed so the decision can be re-examined rather than taken on trust. It is
also the ticket's own audit trail: the second table below is what turned a
"handful of lines" extension into a scope change.
"""

from __future__ import annotations

import numpy as np
import torch

from src import dispersion as dp
from src import tmm_torch as pt

WAVELENGTHS = torch.linspace(400.0, 800.0, 200, dtype=torch.float64)
NOMINAL_ANGLE = np.radians(70.0)
REFLECTANCE_NOISE = 1e-3  # 0.1% in R, a good reflectometer
ELLIPSOMETER_NOISE = 1e-3  # radians, about 0.057 deg


def _substrate() -> torch.Tensor:
    n, k = dp.load_nk("Si", WAVELENGTHS.numpy())
    return torch.tensor(n + 1j * k)


def _film_index(cauchy_a: torch.Tensor) -> torch.Tensor:
    return dp.cauchy_n(
        (cauchy_a, torch.tensor(0.004, dtype=torch.float64),
         torch.tensor(0.0, dtype=torch.float64)),
        WAVELENGTHS,
    )


def reflectance_model(theta: torch.Tensor, angle: float) -> torch.Tensor:
    substrate = _substrate()
    return pt.stack_reflectance(
        WAVELENGTHS, [theta[0]], [1.0, _film_index(theta[1]), substrate], angle, "s"
    )


def ellipsometry_model(theta: torch.Tensor, angle: float) -> torch.Tensor:
    """``tan(Ψ)·e^{iΔ} = r_p / r_s``, returned as the stacked pair (Ψ, Δ)."""
    substrate = _substrate()
    indices = [1.0, _film_index(theta[1]), substrate]
    r_s = pt.stack_r(WAVELENGTHS, [theta[0]], indices, angle, "s")
    r_p = pt.stack_r(WAVELENGTHS, [theta[0]], indices, angle, "p")
    ratio = r_p / r_s
    return torch.cat([torch.atan(ratio.abs()), torch.atan2(ratio.imag, ratio.real)])


def cramer_rao(model, thickness_nm: float, sigma: float, angle: float) -> tuple[float, float]:
    """Return ``(σ_d, ρ(d, n))`` from ``C = (JᵀJ/σ²)⁻¹``, per §5.3."""
    theta = torch.tensor([float(thickness_nm), 1.46], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(lambda t: model(t, angle), theta)
    covariance = torch.linalg.inv(jacobian.T @ jacobian / sigma**2)
    correlation = covariance[0, 1] / torch.sqrt(covariance[0, 0] * covariance[1, 1])
    return float(torch.sqrt(covariance[0, 0])), float(correlation)


def main() -> int:
    print("  DTFM-028 — is ellipsometry worth adopting?\n")
    print(f"  At {np.degrees(NOMINAL_ANGLE):.0f} deg incidence. Reflectance noise "
          f"{REFLECTANCE_NOISE:.0e}, ellipsometer noise "
          f"{np.degrees(ELLIPSOMETER_NOISE):.3f} deg.\n")
    print(f"  {'film':>7}  {'R only: sigma_d':>16}{'rho':>9}  "
          f"{'ellipso: sigma_d':>17}{'rho':>9}  {'gain':>7}")

    for thickness in (25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0):
        sigma_r, rho_r = cramer_rao(reflectance_model, thickness, REFLECTANCE_NOISE, NOMINAL_ANGLE)
        sigma_e, rho_e = cramer_rao(
            ellipsometry_model, thickness, ELLIPSOMETER_NOISE, NOMINAL_ANGLE
        )
        print(f"  {thickness:6.0f}n  {sigma_r:15.4f}n{rho_r:9.4f}  "
              f"{sigma_e:16.4f}n{rho_e:9.4f}  {sigma_r / sigma_e:6.1f}x")

    print("\n  Why the angle had to change: r_p/r_s is constant at normal incidence,")
    print("  where s and p coincide, so ellipsometry carries no information there.\n")
    print(f"  {'angle':>9}  {'gain at 200 nm':>16}")
    for degrees in (0, 5, 10, 20, 30, 45, 60, 70, 75):
        if degrees == 0:
            print(f"  {degrees:8d}d  {'no information':>16}")
            continue
        angle = np.radians(degrees)
        sigma_r, _ = cramer_rao(reflectance_model, 200.0, REFLECTANCE_NOISE, angle)
        sigma_e, _ = cramer_rao(ellipsometry_model, 200.0, ELLIPSOMETER_NOISE, angle)
        print(f"  {degrees:8d}d  {sigma_r / sigma_e:15.1f}x")

    print("\n  And the gain is conditional on the instrument, not free:\n")
    print(f"  {'ellipsometer noise':>22}  {'gain at 200 nm':>16}")
    for sigma, label in ((1e-4, "0.006 deg"), (1e-3, "0.057 deg"),
                         (1e-2, "0.57 deg"), (3e-2, "1.7 deg")):
        sigma_r, _ = cramer_rao(reflectance_model, 200.0, REFLECTANCE_NOISE, NOMINAL_ANGLE)
        sigma_e, _ = cramer_rao(ellipsometry_model, 200.0, sigma, NOMINAL_ANGLE)
        verdict = "" if sigma_r / sigma_e > 1 else "   <- worse than reflectance"
        print(f"  {label:>22}  {sigma_r / sigma_e:15.1f}x{verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
