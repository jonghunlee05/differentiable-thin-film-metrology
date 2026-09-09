"""Reliability diagram and ECE for the ensemble — DTFM-048/049, §8.2.

    python scripts/calibration_report.py --members runs/members.npz

Takes a cache of per-member predictions (``truth``, ``mu``, ``va``) and reports
§8.2's numbers for every ensemble size, plus the figure.

Kept separate from the training code because calibration is measured *after* the
fact, on predictions, and needs no model in memory. That also means it can be
re-run on any saved set of members without a GPU or a checkpoint loader.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import calibration as cal  # noqa: E402

SIZES = (1, 2, 3, 5, 10, 15, 20)
TRIALS = 30


def ensemble(mu: np.ndarray, va: np.ndarray, index: np.ndarray):
    """§8.1's decomposition for one subset of members."""
    mean = mu[index].mean(axis=0)
    aleatoric = va[index].mean(axis=0)
    epistemic = np.clip((mu[index] ** 2).mean(axis=0) - mean**2, 0.0, None)
    return mean, np.sqrt(aleatoric + epistemic), np.sqrt(aleatoric), np.sqrt(epistemic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", required=True, help="npz with truth, mu, va")
    parser.add_argument("--out", default="figures/calibration.png")
    parser.add_argument("--json", dest="json_out", default="runs/calibration.json")
    args = parser.parse_args()

    data = np.load(args.members)
    truth, mu, va = data["truth"], data["mu"], data["va"]
    total = mu.shape[0]
    rng = np.random.default_rng(0)

    rows = []
    for size in [s for s in SIZES if s <= total]:
        # Several random subsets per size, so the curve carries a spread rather
        # than depending on which members happened to be chosen first.
        trials = 1 if size == total else TRIALS
        picks = [rng.choice(total, size, replace=False) for _ in range(trials)]
        summaries = []
        for index in picks:
            mean, sigma, aleatoric, epistemic = ensemble(mu, va, index)
            summary = cal.summary(mean - truth, sigma)
            summary["aleatoric_nm"] = float(np.median(aleatoric))
            summary["epistemic_nm"] = float(np.median(epistemic))
            summaries.append(summary)
        row = {"M": size}
        for key in ("coverage_68", "coverage_95", "ece", "sharpness_nm",
                    "median_error_nm", "aleatoric_nm", "epistemic_nm"):
            values = [s[key] for s in summaries]
            row[key] = float(np.mean(values))
            row[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["nominal"] = summaries[0]["nominal"]
        row["empirical"] = np.mean([s["empirical"] for s in summaries], axis=0).tolist()
        rows.append(row)

    print(f"  {total} members\n")
    print(f"  {'M':>3} {'median nm':>10} {'sharpness':>10} {'cov 68%':>9} "
          f"{'cov 95%':>9} {'ECE':>7}")
    for row in rows:
        print(f"  {row['M']:3d} {row['median_error_nm']:10.3f} {row['sharpness_nm']:10.3f} "
              f"{row['coverage_68']:9.3f} {row['coverage_95']:9.3f} {row['ece']:7.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1, label="perfect")
    for row, colour in zip(rows, plt.cm.viridis(np.linspace(0.15, 0.9, len(rows))),
                           strict=True):
        ax.plot(row["nominal"], row["empirical"], "o-", ms=3, lw=1.2, color=colour,
                label=f"M={row['M']}")
    ax.set(xlabel="nominal coverage", ylabel="empirical coverage",
           title="reliability", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    sizes = [r["M"] for r in rows]
    ax.errorbar(sizes, [r["ece"] for r in rows], yerr=[r["ece_sd"] for r in rows],
                fmt="o-", ms=4, lw=1.2, color="tab:red", label="ECE")
    ax.set(xlabel="ensemble members", ylabel="expected calibration error",
           title="calibration against ensemble size", xscale="log")
    twin = ax.twinx()
    twin.errorbar(sizes, [r["median_error_nm"] for r in rows],
                  yerr=[r["median_error_nm_sd"] for r in rows],
                  fmt="s--", ms=4, lw=1.2, color="tab:blue", label="median error")
    twin.set_ylabel("median error (nm of thickness)")
    twin.grid(False)
    lines = ax.get_lines()[:1] + twin.get_lines()[:1]
    ax.legend(lines, ["ECE (left)", "median error (right)"], fontsize=7, loc="center right")

    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n  figure -> {out}")

    js = pathlib.Path(args.json_out)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(rows, indent=2))
    print(f"  numbers -> {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
