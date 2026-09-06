"""Compare every training run — DTFM-039, for tuning as a record rather than a memory.

    python scripts/run_history.py            # print the table
    python scripts/run_history.py --json     # emit for the comparison page

Reads ``runs/history.jsonl``, one line per run, appended by
``scripts/train_mlp.py``. The file is gitignored: it is generated output, and §15
keeps generated data out of the repository.

The point is to make a change's *effect* visible. Without a log, tuning is a
sequence of impressions — "that felt better" — and there is no way to tell which
change caused which result, or to notice a change that helped one regime while
quietly ruining another.
"""

from __future__ import annotations

import argparse
import json
import pathlib

HISTORY = pathlib.Path(__file__).resolve().parent.parent / "runs" / "history.jsonl"
CLASSICAL = {"median_nm": 0.034, "wrong_over_1nm": 0.01, "seconds": 8.44}


def load() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON for the page")
    args = parser.parse_args()

    runs = load()
    if not runs:
        print(f"  no runs yet — {HISTORY} is empty")
        return 0

    if args.json:
        print(json.dumps({"runs": runs, "classical": CLASSICAL}))
        return 0

    print(f"\n  {len(runs)} run(s) recorded\n")
    header = (
        f"  {'#':>2} {'width':>6} {'depth':>6} {'lr':>8} {'batch':>6} {'steps':>7}"
        f" {'median nm':>10} {'wrong>1nm':>10} {'thin':>8} {'thick':>8} {'us/film':>8}"
    )
    print(header)
    best = min(runs, key=lambda r: r["median_nm"])
    for i, run in enumerate(runs, 1):
        mark = " <-- best" if run is best else ""
        print(
            f"  {i:2d} {run['width']:6d} {run['depth']:6d} {run['lr']:8.1e}"
            f" {run['batch']:6d} {run['steps']:7d} {run['median_nm']:10.3f}"
            f" {100 * run['wrong_over_1nm']:9.1f}% {run.get('median_thin', float('nan')):8.2f}"
            f" {run.get('median_thick', float('nan')):8.2f} {run['inference_us']:8.1f}{mark}"
        )

    print(
        f"\n  classical baseline: {CLASSICAL['median_nm']} nm median,"
        f" {100 * CLASSICAL['wrong_over_1nm']:.0f}% wrong, {CLASSICAL['seconds']} s/film"
    )
    gap = best["median_nm"] / CLASSICAL["median_nm"]
    speed = CLASSICAL["seconds"] * 1e6 / best["inference_us"]
    print(f"  best run is {gap:.0f}x worse on accuracy and {speed:,.0f}x faster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
