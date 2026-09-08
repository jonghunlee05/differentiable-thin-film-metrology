"""Benchmark the learned inverse model against the classical fit — DTFM-044, §10.

    python scripts/benchmark.py --run runs/cnn-c64k7-d3-s96000-nll-seed0 --cases 60

Scores both methods on the *same* films with the *same* noise draws, through the
same `evaluate` module, and prints §10's regime x SNR table for each.

TWO TEST SETS, NOT ONE. §4.5 gives every training film 0-4 nm of thickness of
surface roughness; the cases DTFM-036 scored the classical baseline on have none,
because they are built from the three fitted parameters alone. That gap is not
small — roughness moves Delta by up to 2.3 rad against an instrument sigma of
0.001 rad — and it cuts both ways:

  smooth films  the fitter's model is exactly right, and the network is being
                asked about films unlike the ones it learned from
  rough films   the network is in its training distribution, and the fitter is
                misspecified: it has three parameters and roughness is not one

Reporting one of those alone would flatter whichever method it favoured. §3a
also forbids claiming novelty for the speed result, so the table states the
per-film cost and leaves the amortisation argument to the reader.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import yaml

from src import evaluate as ev
from src import generate as gen
from src import models
from src import training as tr


def load_model(run_dir: pathlib.Path, prior: gen.Prior):
    """Rebuild a trained network from its run directory.

    The config travels with the weights (DTFM-043), so the architecture does not
    have to be guessed or passed in — which is exactly the reproducibility
    property §15 asks for, used rather than merely asserted.
    """
    config = tr.RunConfig(**yaml.safe_load((run_dir / "config.yaml").read_text()))
    model = models.build_model(config.model_settings(), prior=prior)
    model.load_state_dict(tr.Checkpoint.load(run_dir / "checkpoint.pt").model)
    model.eval()
    return model, config


def summarise(per_seed: list[list[dict]]) -> list[dict]:
    """Mean and spread of each table cell across seeds.

    The cells line up because every seed was scored on the same cases in the same
    order, so row *i* is the same regime and SNR in every table.
    """
    summary = []
    for cells in zip(*per_seed, strict=True):
        row = {"regime": cells[0]["regime"], "snr": cells[0]["snr"], "n": cells[0]["n"]}
        for field in ("median_abs_nm", "p95_abs_nm", "failure_rate", "seconds"):
            values = np.array([c[field] for c in cells], dtype=float)
            row[field] = float(values.mean())
            row[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def show_spread(title: str, rows: list[dict], seeds: int) -> None:
    print(f"\n  {title}  ({seeds} seed{'s' if seeds != 1 else ''})")
    print(
        f"    {'regime':>7} {'snr':>5} {'n':>4} {'median':>17} {'p95':>9} "
        f"{'>1nm':>15} {'s/film':>9}"
    )
    for row in rows:
        spread = f"±{row['median_abs_nm_std']:.3f}" if seeds > 1 else ""
        fail = f"{100 * row['failure_rate']:5.1f}%"
        if seeds > 1:
            fail += f" ±{100 * row['failure_rate_std']:4.1f}"
        print(
            f"    {row['regime']:>7} {row['snr']:>5} {row['n']:4d} "
            f"{row['median_abs_nm']:10.3f} {spread:>6} {row['p95_abs_nm']:9.3f} "
            f"{fail:>15} {row['seconds']:9.4f}"
        )


def show(title: str, report: ev.Report) -> list[dict]:
    rows = report.table()
    print(f"\n  {title}")
    print(
        f"    {'regime':>7} {'snr':>5} {'n':>4} {'median':>9} {'p95':>9} "
        f"{'>1nm':>7} {'flagged':>8} {'s/film':>9}"
    )
    for row in rows:
        print(
            f"    {row['regime']:>7} {row['snr']:>5} {row['n']:4d} "
            f"{row['median_abs_nm']:9.3f} {row['p95_abs_nm']:9.3f} "
            f"{100 * row['failure_rate']:6.1f}% {100 * row['convergence_flag_rate']:7.1f}% "
            f"{row['seconds']:9.4f}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        nargs="+",
        help="one or more run directories, each containing checkpoint.pt. Several seeds of "
        "the same configuration are scored on the same cases and reported with a spread, "
        "because a single seed's number is a draw rather than a result.",
    )
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-classical", action="store_true", help="network only, for a quick look"
    )
    parser.add_argument("--out", default="runs/benchmark.json")
    args = parser.parse_args()

    prior = gen.Prior()
    loaded = [load_model(pathlib.Path(d), prior) for d in args.run]
    models_, configs = zip(*loaded, strict=True)
    print(f"  network: {configs[0].name}  ({models_[0].parameter_count:,} parameters)")
    print(f"  seeds:   {len(models_)}")

    results: dict[str, list[dict]] = {}
    for label, roughness in (("smooth films (DTFM-036's test set)", False),
                             ("rough films (the training distribution)", True)):
        cases = ev.make_cases(args.cases, seed=args.seed, prior=prior, roughness=roughness)
        print(f"\n{'=' * 78}\n  {label} — {len(cases)} cases\n{'=' * 78}")

        started = time.perf_counter()
        # Each seed is scored on the identical cases, then summarised. Reporting
        # one seed would repeat the mistake DTFM-070 spent 138 runs correcting:
        # a single number here is a draw from a distribution, not a result.
        per_seed = [
            ev.evaluate(cases, ev.network_estimator(m, prior=prior), prior=prior).table()
            for m in models_
        ]
        key = f"learned_{'rough' if roughness else 'smooth'}"
        results[key] = summarise(per_seed)
        show_spread("learned", results[key], len(models_))
        print(f"    ({time.perf_counter() - started:.1f}s for {len(models_)} seeds)")

        if not args.skip_classical:
            started = time.perf_counter()
            classical = ev.evaluate(cases, method="multi_start", prior=prior)
            results[f"classical_{'rough' if roughness else 'smooth'}"] = show(
                "classical (multi-start Levenberg-Marquardt)", classical
            )
            print(f"    ({time.perf_counter() - started:.1f}s for all {len(cases)} cases)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "run": configs[0].name,
                "seeds": [c.seed for c in configs],
                "cases": args.cases,
                "case_seed": args.seed,
                "tables": results,
            },
            indent=2,
        )
    )
    print(f"\n  written to {out}")
    print("\n  §3a: the speed figure is amortised training cost paid once. No novelty is")
    print("  claimed for it, and it is stated rather than argued from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
