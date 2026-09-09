"""The headline figure — DTFM-051, spec §12 Week 7 and §16.

    python scripts/calibration_figure.py --members runs/members.npz

One figure carrying the project's result, good enough to open the README with.

The claim it makes is narrower than "the network works", and deliberately so:

    The network is a hundred times less precise than a classical fit, states an
    uncertainty that is honest about exactly that, and is the better estimator
    when the classical model is wrong.

Each panel carries one part of that, and the fourth carries what is still broken.
A headline figure that showed only the good half would be advertising.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import calibration as cal  # noqa: E402
from src import dataset as ds  # noqa: E402
from src import generate as gen  # noqa: E402
from src import uncertainty as un  # noqa: E402

#: DTFM-044, one seed, 80 cases per test set. Quoted rather than recomputed: the
#: classical arm costs ~10 s per film and this figure should redraw in seconds.
BENCHMARK = {
    "classical_smooth": (0.025, 10.31),
    "learned_smooth": (0.856, 0.0012),
    "classical_rough": (1.014, 10.13),
    "learned_rough": (0.736, 0.0011),
}
INK, MUTED = "#161923", "#5b6273"
NET, CLASSICAL, FLOOR = "#3b3aa6", "#0f7050", "#a33232"


def combine(mu: np.ndarray, va: np.ndarray, count: int):
    index = np.arange(count)
    mean = mu[index].mean(axis=0)
    aleatoric = va[index].mean(axis=0)
    epistemic = np.clip((mu[index] ** 2).mean(axis=0) - mean**2, 0.0, None)
    return mean, np.sqrt(aleatoric + epistemic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", required=True)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--out", default="figures/calibration_headline.png")
    args = parser.parse_args()

    prior = gen.Prior()
    wavelengths = np.linspace(400.0, 800.0, 200)
    data = np.load(args.members)
    truth, mu, va = data["truth"], data["mu"], data["va"]
    batch = ds.sample_batch(len(truth), wavelengths, np.random.default_rng(args.seed),
                            prior=prior)
    bound = un.bound_per_film(batch.targets.numpy(), wavelengths, prior=prior)
    ok = np.isfinite(bound)

    one_mean, one_sigma = combine(mu, va, 1)
    all_mean, _ = combine(mu, va, mu.shape[0])
    one_error = one_mean - truth
    all_error = all_mean - truth

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.facecolor": "white", "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))

    # --- A: the uncertainty is honest ---------------------------------------
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED, label="perfectly calibrated")
    for count, colour, style in ((1, NET, "-"), (mu.shape[0], "#9694ee", "--")):
        mean, sigma = combine(mu, va, count)
        curve = cal.reliability(mean - truth, sigma)
        ece = cal.expected_calibration_error(mean - truth, sigma)
        ax.plot(curve["nominal"], curve["empirical"], style, marker="o", ms=3.5, lw=1.6,
                color=colour, label=f"{count} model{'s' if count > 1 else ''}  ECE {ece:.3f}")
    ax.set(xlabel="stated confidence", ylabel="how often it was right",
           xlim=(0, 1), ylim=(0, 1), title="A · the error bars are honest")
    ax.legend(fontsize=7.5, loc="lower right")

    # --- B: and far from the floor -------------------------------------------
    ax = axes[0, 1]
    ax.loglog(bound[ok], one_sigma[ok], ".", ms=2, alpha=0.25, color=NET,
              label="stated uncertainty")
    ax.loglog(bound[ok], np.abs(one_error)[ok], ".", ms=2, alpha=0.25, color="#b0621a",
              label="actual error")
    edge = np.array([bound[ok].min(), bound[ok].max()])
    ax.loglog(edge, edge, ls="--", lw=1.4, color=FLOOR, label="Cramér–Rao floor")
    ax.axhline(0.034, color=CLASSICAL, lw=1.4, ls=":", label="classical fit (0.034)")
    efficiency = float(np.median((bound[ok] / np.abs(all_error)[ok]) ** 2))
    ax.set(xlabel="Cramér–Rao bound (nm of thickness)", ylabel="nm of thickness",
           title=f"B · but it uses {100 * efficiency:.2f}% of the information")
    ax.legend(fontsize=7.5, loc="upper left", markerscale=4)

    # --- C: the trade it actually wins ---------------------------------------
    # A slope chart, not a scatter. The finding is that the two lines CROSS: the
    # classical fit is 34x better when its model is right and 1.4x worse when it
    # is not, while the network barely notices. A scatter of four points puts
    # that in two crowded corners and hides the one thing worth seeing.
    ax = axes[1, 0]
    positions = [0, 1]
    for key, colour, label in (
        ("classical", CLASSICAL, "classical fit  ·  10 s per film"),
        ("learned", NET, "network  ·  0.001 s per film"),
    ):
        values = [BENCHMARK[f"{key}_smooth"][0], BENCHMARK[f"{key}_rough"][0]]
        ax.plot(positions, values, "o-", ms=9, lw=2.4, color=colour, label=label,
                markeredgecolor="white", markeredgewidth=1.4, zorder=3)
        # Offset outward horizontally, not vertically: the two right-hand values
        # are 0.736 and 1.014, close enough on a log axis that stacked labels
        # overwrite each other.
        for x, value in zip(positions, values, strict=True):
            outward = -14 if x == 0 else 14
            ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points",
                        xytext=(outward, 0), ha="right" if x == 0 else "left",
                        va="center", fontsize=8.5, color=colour)
    ax.set_xticks(positions)
    ax.set_xticklabels(["smooth films\n(the fitter's model is exact)",
                        "rough films\n(0-4 nm of roughness it cannot fit)"], fontsize=8)
    ax.set(yscale="log", xlim=(-0.55, 1.55), ylabel="median error (nm of thickness)",
           title="C · when the model is wrong, the lines cross")
    ax.legend(fontsize=7.5, loc="center left")

    # --- D: what is still broken ---------------------------------------------
    ax = axes[1, 1]
    ax.scatter(truth, all_error, s=3, alpha=0.28, color=MUTED, edgecolor="none")
    ax.axhline(0.0, color=MUTED, lw=0.8)
    ceiling = prior.thickness_nm[1]
    ax.axvspan(1800, ceiling, color=FLOOR, alpha=0.08)
    near = truth > 1800
    ax.scatter(truth[near], all_error[near], s=8, color=FLOOR, zorder=3,
               label=f"within 200 nm of the prior ceiling\nmedian bias "
                     f"{np.median(all_error[near]):+.1f} nm of thickness")
    ax.set(xscale="log", xlabel="true thickness (nm)",
           ylabel="signed error (nm of thickness)",
           title="D · the failure is at the edge of the prior")
    ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        "A hundred times less precise than the classical fit — and honest about it",
        fontsize=12.5, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"  figure -> {out}")
    print(f"  A  ECE {cal.expected_calibration_error(one_error, one_sigma):.4f} for one model")
    classical_efficiency = float(np.median((bound[ok] / 0.034) ** 2))
    print(f"  B  efficiency {efficiency:.5f}; classical {classical_efficiency:.3f}")
    print(f"  D  edge bias {np.median(all_error[near]):+.2f} nm of thickness on "
          f"{near.sum()} films; elsewhere {np.median(all_error[~near]):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
