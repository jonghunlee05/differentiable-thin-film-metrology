"""Train the MLP inverse model — DTFM-039, spec §7.2.

    python scripts/train_mlp.py --steps 8000

Reports against the classical baseline and against the trivial baseline the AC
names. Deliberately minimal: config files, checkpointing and resume are DTFM-043,
and putting a half-formed version of them here would have to be undone.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
from datetime import UTC, datetime

import numpy as np
import torch
from torch import nn

from src import dataset as ds
from src import generate as gen
from src import models

WAVELENGTHS = np.linspace(400.0, 800.0, 200)

#: Every run appends one line here, so tuning is a record rather than a memory.
#: Gitignored — it is generated output, and §15 keeps generated data out of the
#: repository. The file is the input to ``scripts/run_history.py``.
HISTORY = pathlib.Path(__file__).resolve().parent.parent / "runs" / "history.jsonl"


def _commit() -> str:
    """Which code produced this result. Without it a run log is a list of numbers
    with no way to reproduce any of them — §15's requirement, applied to runs.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def record(settings: dict, results: dict) -> None:
    """Append one run to the history."""
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        **settings,
        **results,
    }
    with HISTORY.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(
        f"\n  appended to {HISTORY.relative_to(HISTORY.parent.parent)}"
        f"  ({sum(1 for _ in HISTORY.open())} runs recorded)"
    )


def evaluate(model, prior, films: int = 2000, seed: int = 999) -> dict:
    """Error on films the network has never seen, by regime.

    §10 stratifies rather than reporting one number, for the reason DTFM-036
    measured: every method that mostly works reports the same median, so a single
    figure cannot tell a working estimator from one that fails on 10% of films.
    """
    model.eval()
    batch = ds.sample_batch(films, WAVELENGTHS, np.random.default_rng(seed), prior=prior)
    with torch.no_grad():
        predicted = model(batch.observed.float())
    outside = float(model.scale_theta.outside_prior(predicted).float().mean())
    error = (predicted[:, 0] - batch.targets[:, 0]).numpy()
    truth = batch.targets[:, 0].numpy()

    report = {
        "median_nm": float(np.median(np.abs(error))),
        "rmse_nm": float(np.sqrt(np.mean(error**2))),
        "wrong_over_1nm": float(np.mean(np.abs(error) > 1.0)),
        "outside_prior": outside,
    }
    for name, low, high in (("thin", 0, 100), ("mid", 100, 700), ("thick", 700, np.inf)):
        mask = (truth >= low) & (truth < high)
        if mask.any():
            report[f"median_{name}"] = float(np.median(np.abs(error[mask])))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="widen the output sigmoid beyond the prior by this fraction",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    prior = gen.Prior()
    model = models.build_model(
        {"width": args.width, "depth": args.depth, "output_margin": args.margin}, prior=prior
    )
    scaler = model.scale_theta
    print(
        f"  {model.parameter_count:,} parameters  "
        f"(width {args.width}, depth {args.depth}, lr {args.lr}, batch {args.batch},"
        f" margin {args.margin})"
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.steps)
    rng = np.random.default_rng(args.seed)

    began = time.perf_counter()
    generating = 0.0
    # The loss curve, not just its last value. Whether a run has converged is the
    # first question asked of any result, and one number cannot answer it — run 1
    # ended at 0.0051 still falling, which only the curve reveals.
    # NOTE: the loss is in *cube* units, and `--margin` changes the size of the
    # cube. At margin 0.2 the cube is 1.40x wider, so the same physical error
    # gives a loss 1.96x smaller — losses from runs with different margins are
    # NOT comparable. Compare `median_nm` instead; nanometres do not move.
    # See Implementation-Notes.md §28.
    curve: list[tuple[int, float]] = []
    for step in range(1, args.steps + 1):
        mark = time.perf_counter()
        batch = ds.sample_batch(args.batch, WAVELENGTHS, rng, prior=prior)
        generating += time.perf_counter() - mark

        # Loss on the unit cube, not in nanometres. Summed in physical units the
        # thickness term would swamp the dispersion coefficients by five orders
        # of magnitude and they would go unlearned — DTFM-031's finding, applied
        # to a loss rather than to an optimiser step.
        model.train()
        optimiser.zero_grad()
        predicted = scaler.encode(model(batch.observed.float()))
        loss = nn.functional.mse_loss(predicted, scaler.encode(batch.targets).float())
        loss.backward()
        optimiser.step()
        schedule.step()

        if step % max(args.steps // 100, 1) == 0:
            curve.append((step, float(loss.item())))
        if step % max(args.steps // 8, 1) == 0:
            print(
                f"  step {step:6d}   loss {loss.item():.6f}   {time.perf_counter() - began:6.1f}s",
                flush=True,
            )

    total = time.perf_counter() - began
    print(
        f"\n  trained in {total:.1f}s  "
        f"({100 * generating / total:.0f}% generating data, "
        f"{100 * (total - generating) / total:.0f}% learning)"
    )

    # Inference timing, properly: warm up, then average. A single timed call
    # measures dispatch overhead and reported 11 ms in an earlier scratch run —
    # a thousandfold error on the network's whole selling point.
    sample = ds.sample_batch(1, WAVELENGTHS, np.random.default_rng(1), prior=prior)
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            model(sample.observed.float())
        mark = time.perf_counter()
        for _ in range(500):
            model(sample.observed.float())
        per_film = (time.perf_counter() - mark) / 500

    report = evaluate(model, prior)
    print("\n  2000 unseen films:")
    print(f"    median |error|     {report['median_nm']:9.3f} nm")
    print(f"    RMSE               {report['rmse_nm']:9.3f} nm")
    print(f"    wrong by >1 nm     {100 * report['wrong_over_1nm']:8.1f}%")
    print(
        f"    thin / mid / thick {report.get('median_thin', float('nan')):7.3f} /"
        f" {report.get('median_mid', float('nan')):.3f} /"
        f" {report.get('median_thick', float('nan')):.3f} nm"
    )
    print(f"    inference          {per_film * 1e6:9.1f} microseconds per film")
    print(
        f"    outside the prior  {100 * report['outside_prior']:8.2f}%"
        f"   (physically impossible; 0% is guaranteed only at margin=0)"
    )

    trivial = np.median(
        np.abs(
            np.exp(np.random.default_rng(0).uniform(np.log(20.0), np.log(2000.0), 2000))
            - np.exp(0.5 * (np.log(20.0) + np.log(2000.0)))
        )
    )
    print(f"\n  the AC's bar — 'always guess the average': {trivial:.1f} nm median")
    print(
        f"  this model: {report['median_nm']:.3f} nm  ->  "
        f"{'BEATS IT' if report['median_nm'] < trivial else 'DOES NOT BEAT IT'}"
        f" by {trivial / max(report['median_nm'], 1e-9):.0f}x"
    )
    print("\n  classical baseline for context: 0.034 nm median, 1% wrong, 8.44 s/film")

    record(
        {
            "steps": args.steps,
            "batch": args.batch,
            "width": args.width,
            "depth": args.depth,
            "lr": args.lr,
            "seed": args.seed,
            "margin": args.margin,
            "parameters": model.parameter_count,
            "architecture": "mlp",
        },
        {
            **report,
            "final_loss": float(loss.item()),
            "loss_curve": curve,
            "train_seconds": total,
            "inference_us": per_film * 1e6,
            "trivial_baseline_nm": float(trivial),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
