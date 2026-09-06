"""Merge a downloaded sweep into the run history — DTFM-043.

    gh run download 12345678 --dir /tmp/sweep
    python scripts/collect_runs.py /tmp/sweep

A parallel sweep returns one artifact per run, so results arrive as a directory
of files rather than as lines in the log the dashboard reads. This is the join.

Idempotent: a run already present — matched on its directory name — is skipped,
so re-running after a partial download does not duplicate rows.
"""

from __future__ import annotations

import argparse
import sys

from src import training as tr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="where the artifacts were downloaded")
    args = parser.parse_args()

    added = tr.collect(args.directory)
    total = sum(1 for _ in tr.HISTORY.open()) if tr.HISTORY.exists() else 0
    print(f"  added {added} run(s); {total} now in {tr.HISTORY.name}")
    if added == 0:
        print("  (nothing new — already collected, or no result.json files found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
