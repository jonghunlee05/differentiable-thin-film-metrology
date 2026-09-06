"""The baseline performance record — DTFM-036, spec §6 and §10.

    python scripts/baseline_record.py

Writes ``figures/baseline_record.png`` and prints §10's table for every method.

This is the benchmark the learned model is judged against, so it is deliberately
unflattering: the same films, the same noise draws, stratified by regime, with
the failure rate defined as *being wrong* rather than as the optimiser being
dissatisfied.

Three methods, and the third is the one that matters:

``least_squares`` and ``autograd``
    Started at the **whole** truth, dispersion included. No real measurement gets
    that; they are an upper bound on what local optimisation achieves when handed
    the answer, not a usable method.
``warm_thickness_only``
    Started at the true thickness with the index pinned at the prior's midpoint —
    warm in one parameter, cold in another. Included because it fails hard in the
    thick regime, and that failure is DTFM-034's ρ(d, n) > 0.99 made operational:
    past 700 nm an index wrong by 0.07 *is* a thickness wrong by hundreds of nm.
``multi_start``
    Twenty starts spanning the prior, knowing nothing. This is the honest
    classical number, and it is roughly twenty times slower for exactly that
    reason. §11's amortisation argument is about this column.
"""

from __future__ import annotations

import pathlib

import numpy as np

from src import evaluate as ev

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"
#: Labels rather than bare function names, because two of these rows are handed
#: the answer and a reader skimming the table must not be able to mistake them
#: for the baseline. Only ``cold start, 20 searches`` is a usable method.
METHODS = {
    "least_squares": "GIVEN THE ANSWER (LM)",
    "autograd": "GIVEN THE ANSWER (autograd)",
    "warm_thickness_only": "given thickness, not index",
    "cold_single": "cold start, 1 search",
    "multi_start": "cold start, 20 searches  <- THE BASELINE",
}
FILMS = 40


def _print_table(report: ev.Report) -> None:
    print(f"\n  {report.method}")
    header = (
        f"  {'regime':>7} {'snr':>5} {'n':>4} {'RMSE nm':>10} {'median nm':>11}"
        f" {'p95 nm':>10} {'sec/fit':>9} {'wrong>1nm':>10} {'flagged':>8}"
    )
    print(header)
    for row in report.table():
        print(
            f"  {row['regime']:>7} {row['snr']:>5} {row['n']:4d} {row['rmse_nm']:10.4f}"
            f" {row['median_abs_nm']:11.5f} {row['p95_abs_nm']:10.4f} {row['seconds']:9.3f}"
            f" {100 * row['failure_rate']:9.0f}% {100 * row['convergence_flag_rate']:7.0f}%"
        )


def main() -> int:
    import sys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = ev.make_cases(FILMS, seed=0)
    print(f"\n  {FILMS} films from the prior, each observed at both noise levels")
    print(f"  {len(cases)} cases total")

    reports = {}
    for method, label in METHODS.items():
        print(f"  running {method} ...", flush=True)
        reports[method] = ev.evaluate(cases, method=method)
        reports[method].method = label
        _print_table(reports[method])
        sys.stdout.flush()

    print("\n  The comparison §10 asks for, overall:")
    print(f"  {'method':>32} {'RMSE nm':>10} {'median nm':>11} {'sec/fit':>9} {'wrong>1nm':>10}")
    for report in reports.values():
        row = report.metrics()
        print(
            f"  {report.method:>32} {row['rmse_nm']:10.4f} {row['median_abs_nm']:11.5f}"
            f" {row['seconds']:9.3f} {100 * row['failure_rate']:9.0f}%"
        )

    single = reports["cold_single"].metrics()
    swept = reports["multi_start"].metrics()
    print(
        f"\n  One cold search is wrong {100 * single['failure_rate']:.0f}% of the time."
        f" Twenty of them: {100 * swept['failure_rate']:.0f}%."
        f" That is what the {swept['seconds'] / single['seconds']:.0f}x wall clock buys."
    )

    honest = reports["multi_start"]
    warm = reports["least_squares"]
    print(
        f"\n  multi-start costs {honest.metrics()['seconds'] / warm.metrics()['seconds']:.0f}x"
        " a warm-started fit — the price of not being told the answer."
    )

    figure, (parity, byregime, timing) = plt.subplots(
        1, 3, figsize=(15, 4.4), constrained_layout=True
    )

    colours = {
        "least_squares": "#c3c7d2",
        "autograd": "#8f95a5",
        "warm_thickness_only": "#b0721a",
        "cold_single": "#a33232",
        "multi_start": "#3b3aa6",
    }
    for method, report in reports.items():
        parity.loglog(
            report.truth,
            np.maximum(report.estimates, 1e-2),
            ".",
            markersize=5,
            color=colours[method],
            alpha=0.75,
            label=report.method,
        )
    limits = (15.0, 2500.0)
    parity.plot(limits, limits, color="0.6", lw=1, zorder=0)
    parity.set_xlim(*limits)
    parity.set_ylim(*limits)
    parity.set_xlabel("true thickness (nm)")
    parity.set_ylabel("recovered thickness (nm)")
    parity.set_title("Parity — on the line is correct")
    parity.legend(fontsize=8, loc="upper left")
    parity.grid(True, which="both", alpha=0.16)

    width = 0.16
    positions = np.arange(len(ev.REGIMES))
    for offset, (method, report) in enumerate(reports.items()):
        values = [
            report.metrics(regime=regime).get("median_abs_nm", np.nan) for regime in ev.REGIMES
        ]
        byregime.bar(
            positions + (offset - 2.0) * width,
            values,
            width,
            color=colours[method],
            label=report.method,
        )
    byregime.set_yscale("log")
    byregime.set_xticks(positions)
    byregime.set_xticklabels(
        [f"{r}\n{ev.REGIMES[r][0]:.0f}-{ev.REGIMES[r][1]:.0f} nm" for r in ev.REGIMES]
    )
    byregime.set_ylabel("median |error| (nm)")
    byregime.set_title("Stratified — the headline hides this")
    byregime.legend(fontsize=8)
    byregime.grid(True, axis="y", which="both", alpha=0.16)

    for method, report in reports.items():
        timing.loglog(
            report.seconds,
            np.maximum(np.abs(report.errors), 1e-6),
            ".",
            markersize=5,
            color=colours[method],
            alpha=0.75,
            label=report.method,
        )
    timing.set_xlabel("seconds per inversion")
    timing.set_ylabel("|error| (nm)")
    timing.set_title("What accuracy costs")
    timing.legend(fontsize=8)
    timing.grid(True, which="both", alpha=0.16)

    figure.suptitle(
        "Classical baseline — the record the network is measured against · spec §6, §10",
        fontsize=10,
    )
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "baseline_record.png"
    figure.savefig(out, dpi=160)
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
