"""Probing where the physics predicts failure — DTFM-053, §5.2 and §10.

    python scripts/degeneracy_probe.py --members runs/members.npz

§5.2 names three degeneracies and calls them the intellectual core of the
project. They are not vague difficulties; each predicts a *shape* in the
identifiability analysis, and a curve either has that shape or it does not.

  (a) thickness-index correlation -- at one wavelength only the product n*d is
      constrained. Dispersion breaks it partially, and how much is computable.
  (b) fringe-order ambiguity -- a film of thickness d and one of
      d + lambda/(2 n cos theta) give near-identical spectra over a finite band.
  (c) thin-film insensitivity -- for d << lambda the derivative goes to zero. The
      fit still returns a number and the number is meaningless. §5.2 calls
      locating that threshold quantitatively one of the project's best figures.

For each regime this records what the classical method does, what the network
does, and -- the question only answerable now that sigma is calibrated --
**whether the uncertainty head noticed**.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import dataset as ds  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import generate as gen  # noqa: E402
from src import uncertainty as un  # noqa: E402

BAND_CENTRE_NM = 600.0


def threshold(thickness: np.ndarray, curve: np.ndarray, level: float) -> float:
    """The thickness at which ``curve`` first falls below ``level``.

    Interpolated rather than snapped to a grid point, because the answer is the
    headline number of this ticket and quoting a grid spacing as a physical
    threshold would be an artefact of how densely it was sampled.
    """
    below = np.where(curve < level)[0]
    if not len(below) or below[0] == 0:
        return float("nan")
    i = below[0]
    return float(np.interp(level, [curve[i], curve[i - 1]], [thickness[i], thickness[i - 1]]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", help="npz cache of member predictions")
    parser.add_argument("--classical-points", type=int, default=18,
                        help="thicknesses to fit classically; ~16 s each")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--out", default="figures/degeneracy.png")
    parser.add_argument("--json", dest="json_out", default="runs/degeneracy.json")
    args = parser.parse_args()

    prior = gen.Prior()
    wavelengths = np.linspace(400.0, 800.0, 200)

    # --- (a) and (c): the identifiability sweep, from physics alone ----------
    grid = np.geomspace(1.0, 2000.0, 260)
    sweep = un.sweep_thickness(grid, wavelengths, prior=prior)
    rho = np.abs(sweep["correlation"])
    relative = sweep["relative_bound"]

    thresholds = {
        "rho_below_0.999_nm": threshold(grid, rho, 0.999),
        "rho_below_0.99_nm": threshold(grid, rho, 0.99),
        "rho_below_0.95_nm": threshold(grid, rho, 0.95),
        "relative_crb_below_10pct_nm": threshold(grid, relative, 0.10),
        "relative_crb_below_1pct_nm": threshold(grid, relative, 0.01),
        "relative_crb_below_0.1pct_nm": threshold(grid, relative, 0.001),
    }
    thresholds["d_over_lambda_at_rho_0.99"] = thresholds["rho_below_0.99_nm"] / BAND_CENTRE_NM
    print("  (c) the d << lambda threshold")
    for key, value in thresholds.items():
        print(f"      {key:32} {value:9.4f}")
    print(f"      the prior floor is {prior.thickness_nm[0]:.0f} nm, "
          f"d/lambda = {prior.thickness_nm[0] / BAND_CENTRE_NM:.4f}")

    report: dict = {"thresholds": thresholds}

    # --- what each method does, by band -------------------------------------
    bands = [(20, 30), (30, 45), (45, 70), (70, 110), (110, 200),
             (200, 400), (400, 800), (800, 1500), (1500, 2000)]
    if args.members:
        cache = np.load(args.members)
        truth, mu, va = cache["truth"], cache["mu"], cache["va"]
        batch = ds.sample_batch(len(truth), wavelengths, np.random.default_rng(999),
                                prior=prior)
        bound = un.bound_per_film(batch.targets.numpy(), wavelengths, prior=prior)
        mean = mu.mean(axis=0)
        aleatoric = va.mean(axis=0)
        epistemic = np.clip((mu**2).mean(axis=0) - mean**2, 0.0, None)
        sigma = np.sqrt(aleatoric + epistemic)
        error = np.abs(mean - truth)

        rows = []
        print(f"\n  did the uncertainty head notice?\n"
              f"      {'band':>16} {'n':>5} {'CRB':>9} {'error':>9} {'sigma':>9}")
        for low, high in bands:
            keep = (truth >= low) & (truth < high)
            if keep.sum() < 5:
                continue
            rows.append({"low": low, "high": high, "n": int(keep.sum()),
                         "crb": float(np.median(bound[keep])),
                         "error": float(np.median(error[keep])),
                         "sigma": float(np.median(sigma[keep]))})
            print(f"      {low:6d}-{high:<9d} {keep.sum():5d} {rows[-1]['crb']:9.4f} "
                  f"{rows[-1]['error']:9.4f} {rows[-1]['sigma']:9.4f}")
        report["bands"] = rows

        thin = (truth >= 20) & (truth < 45)
        mid = (truth >= 110) & (truth < 400)
        response = {
            "crb_ratio": float(np.median(bound[thin]) / np.median(bound[mid])),
            "error_ratio": float(np.median(error[thin]) / np.median(error[mid])),
            "sigma_ratio": float(np.median(sigma[thin]) / np.median(sigma[mid])),
        }
        response["under_reaction"] = response["error_ratio"] / response["sigma_ratio"]
        report["thin_response"] = response
        print("\n      moving from the 110-400 band into 20-45:")
        print(f"      the bound rises {response['crb_ratio']:.1f}x, "
              f"the error rises {response['error_ratio']:.1f}x, "
              f"sigma rises {response['sigma_ratio']:.1f}x")
        print(f"      -> the head noticed, and under-reacted by "
              f"{response['under_reaction']:.1f}x")

    # --- the classical arm, which is the expensive one ----------------------
    if not args.skip_classical:
        probe = np.geomspace(20.0, 2000.0, args.classical_points)
        print(f"\n  classical fits at {len(probe)} thicknesses (~16 s each)")
        cases = []
        rng = np.random.default_rng(11)
        n_sub, k_sub = un.dp.load_nk(prior.substrate, wavelengths)
        substrate = un.torch.tensor(n_sub + 1j * k_sub)
        for value in probe:
            truth_row = np.array([value, 1.46, 0.004])
            with un.torch.no_grad():
                clean = un.bl.forward_observable(
                    truth_row, wavelengths, gen.Measurement(), substrate
                ).numpy()
            cases.append(ev.Case(
                truth=truth_row,
                observed=clean + rng.normal(0.0, ev.SNR_LEVELS["high"], clean.shape),
                wavelengths_nm=wavelengths, sigma=ev.SNR_LEVELS["high"], snr="high",
            ))
        fitted = ev.evaluate(cases, method="multi_start", prior=prior)
        report["classical"] = [
            {"thickness_nm": float(t), "error_nm": float(abs(e)), "seconds": float(s)}
            for t, e, s in zip(fitted.truth, fitted.errors, fitted.seconds, strict=True)
        ]
        print(f"      median error {np.median(np.abs(fitted.errors)):.4f} nm of thickness")

    # --- the figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax = axes[0]
    ax.semilogx(grid, rho, lw=1.8, color="#a33232", label="|rho(thickness, index)|")
    for level, style in ((0.99, "--"), (0.95, ":")):
        ax.axhline(level, color="0.5", ls=style, lw=1)
        ax.annotate(f"{level}", (grid[0], level), fontsize=7, va="bottom", color="0.4")
    ax.axvline(thresholds["rho_below_0.99_nm"], color="#3b3aa6", lw=1.4)
    ax.axvspan(grid[0], thresholds["rho_below_0.99_nm"], color="#a33232", alpha=0.07)
    ax.axvline(prior.thickness_nm[0], color="#0f7050", ls="-.", lw=1.4,
               label=f"prior floor ({prior.thickness_nm[0]:.0f} nm)")
    ax.set(xlabel="thickness (nm)", ylabel="|correlation|", ylim=(0, 1.02),
           title=f"(a)+(c) degeneracy ends at {thresholds['rho_below_0.99_nm']:.0f} nm")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    ax.loglog(grid, 100 * relative, lw=1.8, color="#b0621a", label="Cramér–Rao bound")
    if args.members:
        centres = [np.sqrt(r["low"] * r["high"]) for r in report["bands"]]
        ax.loglog(centres, [100 * r["error"] / c for r, c in zip(report["bands"], centres,
                  strict=True)], "o-", ms=4, lw=1.2, color="#3b3aa6", label="network error")
        ax.loglog(centres, [100 * r["sigma"] / c for r, c in zip(report["bands"], centres,
                  strict=True)], "s--", ms=4, lw=1.2, color="#9694ee", label="network sigma-hat")
    ax.axvline(thresholds["rho_below_0.99_nm"], color="0.5", lw=1)
    ax.set(xlabel="thickness (nm)", ylabel="percent of thickness",
           title="(c) what the thin regime costs")
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n  figure -> {out}")

    js = pathlib.Path(args.json_out)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(report, indent=2))
    print(f"  numbers -> {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
