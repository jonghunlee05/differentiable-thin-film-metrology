"""Re-download the vendored refractiveindex.info optical constants.

Spec §4.4. See ``data/refractiveindex/PROVENANCE.md`` for sources and licence.

Deliberately *not* run automatically, and not wired into any test. The pinned
copies in ``data/refractiveindex/`` are what every result in this project was
computed against; a silent upstream revision would change published numbers with
no commit recording it. Run this only to deliberately refresh, then inspect the
diff and re-run the suite before committing.

    python scripts/fetch_optical_constants.py            # report differences only
    python scripts/fetch_optical_constants.py --write    # overwrite the vendored files
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

BASE = (
    "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database"
    "/master/database/data/main"
)

#: material -> path within the upstream repository. Keep in step with
#: src.dispersion.MATERIALS and with PROVENANCE.md.
DATASETS = {
    "SiO2": "SiO2/nk/Malitson.yml",
    "Si3N4": "Si3N4/nk/Luke.yml",
    "TiO2": "TiO2/nk/Siefke.yml",
    "Si": "Si/nk/Aspnes.yml",
}

DATA_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "refractiveindex"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="overwrite the vendored files rather than comparing"
    )
    args = parser.parse_args()

    changed = 0
    for material, remote in DATASETS.items():
        url = f"{BASE}/{remote}"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            upstream = response.read()

        local = DATA_ROOT / material / pathlib.Path(remote).name
        current = local.read_bytes() if local.exists() else None

        if current == upstream:
            print(f"  unchanged  {material:6s} {local.relative_to(DATA_ROOT)}")
            continue

        changed += 1
        state = "missing" if current is None else "DIFFERS"
        print(f"  {state:9s}  {material:6s} {local.relative_to(DATA_ROOT)}")
        if args.write:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(upstream)
            print(f"             -> wrote {len(upstream)} bytes")

    if changed and not args.write:
        print(f"\n  {changed} file(s) differ from upstream. Re-run with --write to update,")
        print("  then check the diff and re-run the test suite: published numbers may move.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
