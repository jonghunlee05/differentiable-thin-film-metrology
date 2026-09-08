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
    parser.add_argument("--run", required=True, help="a run directory containing checkpoint.pt")
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-classical", action="store_true", help="network only, for a quick look"
    )
    parser.add_argument("--out", default="runs/benchmark.json")
    args = parser.parse_args()

    prior = gen.Prior()
    run_dir = pathlib.Path(args.run)
    model, config = load_model(run_dir, prior)
    print(f"  network: {config.name}  ({model.parameter_count:,} parameters)")

    results: dict[str, list[dict]] = {}
    for label, roughness in (("smooth films (DTFM-036's test set)", False),
                             ("rough films (the training distribution)", True)):
        cases = ev.make_cases(args.cases, seed=args.seed, prior=prior, roughness=roughness)
        print(f"\n{'=' * 78}\n  {label} — {len(cases)} cases\n{'=' * 78}")

        started = time.perf_counter()
        learned = ev.evaluate(cases, ev.network_estimator(model, prior=prior), prior=prior)
        learned.method = "learned"
        results[f"learned_{'rough' if roughness else 'smooth'}"] = show("learned", learned)
        print(f"    ({time.perf_counter() - started:.1f}s for all {len(cases)} cases)")

        if not args.skip_classical:
            started = time.perf_counter()
            classical = ev.evaluate(cases, method="multi_start", prior=prior)
            results[f"classical_{'rough' if roughness else 'smooth'}"] = show(
                "classical (multi-start Levenberg-Marquardt)", classical
            )
            print(f"    ({time.perf_counter() - started:.1f}s for all {len(cases)} cases)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": config.name, "cases": args.cases,
                               "seed": args.seed, "tables": results}, indent=2))
    print(f"\n  written to {out}")
    print("\n  §3a: the speed figure is amortised training cost paid once. No novelty is")
    print("  claimed for it, and it is stated rather than argued from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
