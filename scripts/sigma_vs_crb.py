"""Claimed uncertainty against the physical floor — DTFM-050, §8.2 and §10.

    python scripts/sigma_vs_crb.py --members runs/members.npz

§8.2 calls this the strongest single result the project can produce, and it is
only available because the forward model is in hand: without a differentiable
simulator there is no Fisher information, and without that there is no floor to
compare against.

Two questions, and the second is the one that matters:

  1. Does the network ever claim a precision the measurement cannot support?
     A sigma below the Cramer-Rao bound is not literally impossible -- the bound
     constrains *unbiased* estimators and a prior-bounded network is biased -- but
     it means either bias is being traded for variance, or the model is simply
     over-confident. Either is worth flagging.

  2. How much of the available information does it actually extract? That is the
     efficiency, and it is where a fast method has to justify itself.
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
from src import generate as gen  # noqa: E402
from src import uncertainty as un  # noqa: E402

CLASSICAL_NM = 0.034  # DTFM-036, smooth films


def combine(mu: np.ndarray, va: np.ndarray, count: int):
    index = np.arange(count)
    mean = mu[index].mean(axis=0)
    aleatoric = va[index].mean(axis=0)
    epistemic = np.clip((mu[index] ** 2).mean(axis=0) - mean**2, 0.0, None)
    return mean, np.sqrt(aleatoric + epistemic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", required=True)
    parser.add_argument("--seed", type=int, default=999, help="the eval batch's seed")
    parser.add_argument("--out", default="figures/sigma_vs_crb.png")
    parser.add_argument("--json", dest="json_out", default="runs/sigma_vs_crb.json")
    args = parser.parse_args()

    prior = gen.Prior()
    wavelengths = np.linspace(400.0, 800.0, 200)
    data = np.load(args.members)
    truth, mu, va = data["truth"], data["mu"], data["va"]

    # The films must be the same ones the members were scored on, or the bound
    # belongs to a different measurement than the sigma it is compared against.
    batch = ds.sample_batch(len(truth), wavelengths, np.random.default_rng(args.seed),
                            prior=prior)
    if not np.allclose(batch.targets[:, 0].numpy(), truth):
        raise SystemExit("the eval batch does not match the cached members — check --seed")

    bound = un.bound_per_film(batch.targets.numpy(), wavelengths, prior=prior)
    ok = np.isfinite(bound)
    print(f"  Cramer-Rao bound: {ok.sum()}/{len(bound)} films, "
          f"median {np.median(bound[ok]):.4f} nm of thickness")

    rows = []
    for count in (1, 5, 20):
        if count > mu.shape[0]:
            continue
        mean, sigma = combine(mu, va, count)
        error = np.abs(mean - truth)
        ratio = sigma[ok] / bound[ok]
        rows.append({
            "M": count,
            "sigma_over_crb_median": float(np.median(ratio)),
            "sigma_over_crb_p95": float(np.percentile(ratio, 95)),
            "below_crb_fraction": float(np.mean(ratio < 1.0)),
            "error_over_crb_median": float(np.median(error[ok] / bound[ok])),
            "efficiency_median": float(np.median((bound[ok] / error[ok]) ** 2)),
            "median_error_nm": float(np.median(error)),
        })

    print(f"\n  {'M':>3} {'sigma/CRB':>11} {'below CRB':>11} {'error/CRB':>11} "
          f"{'efficiency':>12}")
    for row in rows:
        print(f"  {row['M']:3d} {row['sigma_over_crb_median']:10.1f}x "
              f"{100 * row['below_crb_fraction']:10.2f}% "
              f"{row['error_over_crb_median']:10.1f}x {row['efficiency_median']:12.5f}")
    classical = {"error_over_crb": CLASSICAL_NM / float(np.median(bound[ok])),
                 "efficiency": float(np.median((bound[ok] / CLASSICAL_NM) ** 2))}
    print(f"\n  classical fit: {classical['error_over_crb']:.1f}x the bound, "
          f"efficiency {classical['efficiency']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    mean, sigma = combine(mu, va, min(20, mu.shape[0]))
    ax = axes[0]
    ax.loglog(bound[ok], sigma[ok], ".", ms=2, alpha=0.3, color="tab:blue",
              label="claimed sigma-hat")
    ax.loglog(bound[ok], np.abs(mean - truth)[ok], ".", ms=2, alpha=0.3,
              color="tab:orange", label="actual |error|")
    edge = [bound[ok].min(), bound[ok].max()]
    ax.loglog(edge, edge, "k--", lw=1, label="the bound itself")
    ax.set(xlabel="Cramer-Rao bound (nm of thickness)",
           ylabel="nm of thickness", title="nothing may sit below the dashed line")
    ax.legend(fontsize=7, markerscale=4)

    ax = axes[1]
    order = np.argsort(truth[ok])
    ax.semilogy(truth[ok][order], (bound[ok] / np.abs(mean - truth)[ok])[order] ** 2,
                lw=0.8, color="tab:green", label="network")
    ax.axhline(classical["efficiency"], color="tab:purple", ls="--", lw=1.2,
               label=f"classical fit ({classical['efficiency']:.2f})")
    ax.axhline(1.0, color="0.4", ls=":", lw=1, label="the floor (efficiency 1)")
    ax.set(xlabel="thickness (nm)", ylabel="efficiency  (CRB / error)^2",
           title="how much of the information is used")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n  figure -> {out}")

    js = pathlib.Path(args.json_out)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps({"rows": rows, "classical": classical}, indent=2))
    print(f"  numbers -> {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
