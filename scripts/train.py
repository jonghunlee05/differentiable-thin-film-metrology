"""Train from a config — DTFM-043, spec §15.

    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/sweep.yaml --sweep
    python scripts/train.py --config configs/train.yaml --only 2   # one sweep entry

Supersedes ``scripts/train_mlp.py``, which took hyperparameters on a command line
and discarded the trained weights. §15 asks that the run behind a reported number
be reproducible from its config; a shell command that has scrolled away is not a
config, and a network that no longer exists cannot be re-examined.

``--only`` exists for the parallel matrix: each GitHub job runs one index of the
same sweep file, so eight runs happen on eight machines at once rather than in
sequence on one. See ``.github/workflows/sweep.yml``.
"""

from __future__ import annotations

import argparse
import sys

from src import training as tr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--sweep", action="store_true", help="run every entry in the sweep")
    parser.add_argument("--only", type=int, help="run one entry of the sweep, by index")
    parser.add_argument("--list", action="store_true", help="print the sweep and stop")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    runs = tr.expand_sweep(tr.load_config(args.config))

    if args.list:
        for index, run in enumerate(runs):
            print(f"  {index}  {run.name}")
        return 0

    if args.only is not None:
        if not 0 <= args.only < len(runs):
            print(f"  --only {args.only} out of range; {len(runs)} runs in {args.config}")
            return 1
        runs = [runs[args.only]]
    elif not args.sweep:
        runs = runs[:1]

    for index, run in enumerate(runs, 1):
        print(f"\n  [{index}/{len(runs)}] {run.name}", flush=True)
        record = tr.train(run, resume=not args.no_resume)
        print(
            f"    median {record['median_nm']:.3f} nm   "
            f"wrong>1nm {100 * record['wrong_over_1nm']:.1f}%   "
            f"{record['inference_us']:.1f} us/film   -> {record['run_dir']}"
        )

    print("\n  classical baseline: 0.034 nm median, 1% wrong, 8.44 s/film")
    return 0


if __name__ == "__main__":
    sys.exit(main())
