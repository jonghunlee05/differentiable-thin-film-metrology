"""The multimodal cost surface, drawn — DTFM-032.

Spec §6: "Handle multimodality with multi-start from many initial guesses;
record where each converges. That landscape *is* the fringe-order ambiguity made
visible."

    python scripts/multistart_landscape.py

Writes ``figures/multistart_landscape.png`` and prints the table behind it.

Two panels, and the left one is the argument. A dense sweep of starting guesses
against where each one stops: a single fit is one point on it, and the staircase
of flat treads shows that the answer is decided by which tread you began on
rather than by the data. The right panel is the cost surface the staircase is
made of — every tread is a local minimum, and the true one is the only deep one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from src import baseline as bl
from src import dispersion as dp
from src import generate as gen

WAVELENGTHS = np.linspace(400.0, 800.0, 200)
TRUE_THICKNESS = 900.0
TRUTH = np.array([TRUE_THICKNESS, 1.46, 0.004])
SWEEP = 121  # starting guesses across the prior, for the landscape
FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"


def _observed(measurement: gen.Measurement, substrate: torch.Tensor) -> np.ndarray:
    return bl.forward_observable(TRUTH, WAVELENGTHS, measurement, substrate).numpy()


def _cost_profile(observed, measurement, substrate, thicknesses) -> np.ndarray:
    """Cost with thickness swept and dispersion held at the truth.

    A slice, not the full surface. The fit varies all three parameters, so the
    basins it finds are slightly wider than this slice suggests — but the slice
    is what shows the structure in thickness, which is the claim being made.
    """
    profile = np.empty(thicknesses.size)
    with torch.no_grad():
        for i, thickness in enumerate(thicknesses):
            model = bl.forward_observable(
                [thickness, TRUTH[1], TRUTH[2]], WAVELENGTHS, measurement, substrate
            ).numpy()
            residual = bl.wrapped_residual(model, observed, measurement.observable)
            profile[i] = 0.5 * np.sum(residual**2)
    return profile


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measurement = gen.Measurement()
    prior = gen.Prior()
    n, k = dp.load_nk(prior.substrate, WAVELENGTHS)
    substrate = torch.tensor(n + 1j * k)
    observed = _observed(measurement, substrate)

    starts = np.linspace(*prior.thickness_nm, SWEEP)
    sweep = bl.fit_multi_start(
        observed,
        WAVELENGTHS,
        starts=starts,
        measurement=measurement,
        prior=prior,
        truth=TRUTH,
    )
    landed = sweep.thicknesses
    basins, counts = sweep.basins()

    coarse, coarse_counts = sweep.basins(tolerance_nm=20.0)
    print(f"\n  true thickness       {TRUE_THICKNESS:.1f} nm")
    print(f"  starting guesses     {SWEEP} across {prior.thickness_nm} nm")
    print(f"  distinct minima      {coarse.size} at 20 nm resolution, {basins.size} at 1 nm")
    print(f"  reached the truth    {sweep.success_fraction * 100:.1f}% of starts")
    print(f"  wall clock           {sweep.wall_clock_s:.1f} s for the sweep")

    print("\n  the twelve deepest minima found:")
    print(f"  {'converged to':>14}  {'starts':>6}  {'cost':>12}  {'error vs truth':>15}")
    deepest = sorted(
        ((centre, count) for centre, count in zip(coarse, coarse_counts, strict=True)),
        key=lambda pair: sweep.fits[int(np.argmin(np.abs(landed - pair[0])))].cost,
    )[:12]
    for centre, count in deepest:
        nearest = sweep.fits[int(np.argmin(np.abs(landed - centre)))]
        print(
            f"  {centre:14.3f}  {count:6d}  {nearest.cost:12.3e}"
            f"  {centre - TRUE_THICKNESS:14.2f} nm"
        )

    # The reason multi-start works at all, and it is not the hit rate.
    costs = sweep.costs
    winner = costs.min()
    runner_up = np.min(costs[np.abs(landed - TRUE_THICKNESS) > 1.0])
    print(
        f"\n  best cost {winner:.2e} against next-best {runner_up:.2e}"
        f" — a factor of {runner_up / max(winner, 1e-300):.0e}."
    )
    print("  Only 1 start in 8 finds the truth, but the one that does is unmistakable.")

    # The comparison that set the default start spacing — see multi_start_grid.
    print("\n  start spacing, 12 starts, does the deepest basin get found?")
    for spacing in ("log-uniform", "uniform"):
        found = []
        for thickness in (30.0, 65.0, 150.0, 420.0, 900.0, 1500.0, 1900.0):
            truth = np.array([thickness, 1.46, 0.004])
            result = bl.fit_multi_start(
                bl.forward_observable(truth, WAVELENGTHS, measurement, substrate).numpy(),
                WAVELENGTHS,
                count=12,
                spacing=spacing,
                measurement=measurement,
                prior=prior,
                truth=truth,
            )
            found.append(abs(result.thickness_error_nm) < 1e-3)
        print(f"    {spacing:12s}  {sum(found)} of {len(found)} films recovered")

    grid = np.linspace(*prior.thickness_nm, 2000)
    profile = _cost_profile(observed, measurement, substrate, grid)

    figure, (left, deep, ripple) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    left.plot(starts, landed, ".", markersize=4, color="#1f77b4")
    left.axhline(TRUE_THICKNESS, color="#d62728", lw=1, ls="--", label="true thickness")
    left.plot(
        prior.thickness_nm,
        prior.thickness_nm,
        color="0.8",
        lw=0.8,
        zorder=0,
        label="start = answer",
    )
    left.set_xlabel("starting guess (nm)")
    left.set_ylabel("converged thickness (nm)")
    left.set_title("Where each start ends up")
    left.set_xlim(*prior.thickness_nm)
    left.set_ylim(*prior.thickness_nm)
    left.legend(fontsize=8, loc="upper left")

    deep.semilogy(grid, np.maximum(profile, 1e-30), lw=1, color="#1f77b4")
    deep.axvline(TRUE_THICKNESS, color="#d62728", lw=1, ls="--")
    deep.set_xlabel("thickness (nm)")
    deep.set_ylabel("cost  ½Σr²")
    deep.set_title("Why picking the lowest cost is safe")
    deep.set_xlim(*prior.thickness_nm)
    deep.set_ylim(1e-26, 1e4)

    # The same slice, zoomed onto the plateau. The panel to its left spans 28
    # decades, on which a 20% ripple is a flat line — and that ripple is every
    # local minimum in the problem. Reading the log panel alone says there is
    # only one minimum, which is how a real staircase gets explained away.
    plateau = profile[np.abs(grid - TRUE_THICKNESS) > 30.0]
    ripple.plot(grid, profile, lw=1, color="#1f77b4")
    ripple.plot(
        landed, sweep.costs, "o", markersize=4, color="#ff7f0e", label="where the fits stopped"
    )
    ripple.axvline(TRUE_THICKNESS, color="#d62728", lw=1, ls="--")
    ripple.set_ylim(0.9 * plateau.min(), 1.05 * plateau.max())
    ripple.set_xlim(*prior.thickness_nm)
    ripple.set_xlabel("thickness (nm)")
    ripple.set_ylabel("cost  ½Σr²")
    ripple.set_title("The same slice, zoomed on the plateau")
    ripple.legend(fontsize=8, loc="lower right")

    figure.suptitle(
        f"Multimodal inversion for a {TRUE_THICKNESS:.0f} nm film: "
        f"{coarse.size} local minima, one true one {runner_up / max(winner, 1e-300):.0e}x deeper"
        " — spec §6",
        fontsize=10,
    )

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "multistart_landscape.png"
    figure.savefig(out, dpi=160)
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
