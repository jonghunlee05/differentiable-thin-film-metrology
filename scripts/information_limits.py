"""Where this measurement is strong, and where it stops working — DTFM-034.

Spec §5.3. Writes ``figures/information_limits.png`` and prints the tables.

    python scripts/information_limits.py

Three panels, each answering a question that a single number cannot.

**Left — the floor.** The Cramér-Rao bound on thickness across the prior's whole
range, as a fraction of the film's own thickness. Absolute nanometres flatter
thick films: 1 nm of 1500 is a different achievement from 1 nm of 30, and §7.1
chose a log-uniform prior on exactly that reasoning.

**Middle — ρ(d, n).** §5.3 predicts this "will frequently exceed 0.99". True past
roughly 700 nm and false below 200, where it oscillates and crosses zero. That
disagreement is the point of the panel.

**A note on the raggedness.** The curves are visibly ragged above ~500 nm. That
is the discrete wavelength grid, not a rendering fault and not physics: fringes
drift across fixed sample points as the film thickens, and both the period and the
amplitude of the wobble halve every time the point count doubles
(``Implementation-Notes.md`` §24). A real spectrometer samples at fixed pixels too,
so the effect is real — it is simply larger here, because 200 points is at the
coarse end of what a real instrument uses.

**Right — which colours to use.** The bound depends on the band, and not
uniformly: blue light resolves thin films better and red light resolves thick
films better. Fixed 400-800 nm is used everywhere else in the project and is
optimal nowhere.
"""

from __future__ import annotations

import pathlib

import numpy as np

from src import generate as gen
from src import uncertainty as un

FULL = np.linspace(400.0, 800.0, 200)
FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"

BANDS = {
    "blue 400-600": (400.0, 600.0),
    "full 400-800": (400.0, 800.0),
    "red 600-800": (600.0, 800.0),
}


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prior = gen.Prior()
    thickness = np.concatenate([np.arange(20.0, 300.0, 2.0), np.arange(300.0, 2001.0, 8.0)])
    swept = un.sweep_thickness(thickness, FULL)

    rho = swept["correlation"]
    crossings = thickness[np.flatnonzero(np.sign(rho[:-1]) != np.sign(rho[1:]))]
    print(f"\n  prior range            {prior.thickness_nm} nm")
    print(
        f"  best relative bound    {swept['relative_bound'].min():.2e}"
        f" at {thickness[swept['relative_bound'].argmin()]:.0f} nm"
    )
    print(
        f"  worst relative bound   {swept['relative_bound'].max():.2e}"
        f" at {thickness[swept['relative_bound'].argmax()]:.0f} nm"
    )
    print(f"  rho -> {rho[-1]:+.4f} at the thick end (spec §5.3 predicts > 0.99)")
    print(f"  rho crosses zero at    {np.round(crossings, 0)} nm — spec §5.3 does not predict this")

    print("\n  Which band to use, by film thickness (relative CRB, lower is better):")
    films = (30.0, 100.0, 300.0, 900.0, 1500.0)
    print(f"  {'band':>14} " + "".join(f"{f'{d:.0f} nm':>12}" for d in films))
    curves = {}
    for name, (low, high) in BANDS.items():
        grid = np.linspace(low, high, 200)
        row = [un.identifiability([d, 1.46, 0.004], grid).relative_thickness_bound for d in films]
        print(f"  {name:>14} " + "".join(f"{v:12.2e}" for v in row))
        coarse = np.concatenate([np.arange(20.0, 300.0, 6.0), np.arange(300.0, 2001.0, 25.0)])
        curves[name] = (coarse, un.sweep_thickness(coarse, grid)["relative_bound"])

    figure, (left, mid, right) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    left.loglog(thickness, swept["relative_bound"], lw=1.4, color="#1f77b4")
    left.set_xlabel("film thickness (nm)")
    left.set_ylabel("Cramér-Rao bound  σ_d / d")
    left.set_title("The floor: best possible precision")
    left.grid(True, which="both", alpha=0.18)

    mid.semilogx(thickness, rho, lw=1.4, color="#1f77b4")
    mid.axhline(0.0, color="0.6", lw=0.8)
    mid.axhline(-0.99, color="#d62728", lw=1, ls="--", label="§5.3 predicts |ρ| > 0.99")
    for crossing in crossings:
        mid.plot(crossing, 0.0, "o", markersize=4, color="#ff7f0e")
    mid.set_xlabel("film thickness (nm)")
    mid.set_ylabel("ρ (thickness, index)")
    mid.set_title("Where d and n become inseparable")
    mid.set_ylim(-1.05, 0.35)
    mid.legend(fontsize=8, loc="lower right")
    mid.grid(True, which="both", alpha=0.18)

    colours = {"blue 400-600": "#3b5bdb", "full 400-800": "#495057", "red 600-800": "#c92a2a"}
    for name, (grid_d, values) in curves.items():
        right.loglog(grid_d, values, lw=1.4, color=colours[name], label=name)
    right.set_xlabel("film thickness (nm)")
    right.set_ylabel("Cramér-Rao bound  σ_d / d")
    right.set_title("Blue for thin films, red for thick")
    right.legend(fontsize=8)
    right.grid(True, which="both", alpha=0.18)

    figure.suptitle(
        "What this measurement can and cannot determine — spec §5.3, DTFM-034\n"
        "raggedness above 500 nm is the discrete wavelength grid, not noise "
        "(Implementation-Notes §24)",
        fontsize=9,
    )
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "information_limits.png"
    figure.savefig(out, dpi=160)
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
