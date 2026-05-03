#!/usr/bin/env python3
"""Regenerate perspective_mask_index.json from the car-type-review-app reviews DB.

Usage:
    python scripts/generate_mask_index.py --db /path/to/reviews.sqlite3

Output:
    perspective_mask_index.json in the repo root (next to this script's parent).
    Maps run_id -> relative path to perspective_mask.png under perspective_masks/.
    Runs with no subtype label or no matching mask file are omitted (they will
    fail at dispatch time rather than silently use a wrong mask).
"""

import argparse
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MASKS_DIR = REPO / "perspective_masks"
OUT = REPO / "perspective_mask_index.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to reviews.sqlite3",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.db))
    rows = conn.execute(
        "SELECT run_id, vehicle, subtype_label FROM reviews"
    ).fetchall()
    conn.close()

    index: dict[str, str] = {}
    skipped_no_subtype = 0
    skipped_no_file = 0

    for run_id, vehicle, subtype_label in rows:
        if not subtype_label:
            skipped_no_subtype += 1
            continue
        candidate = MASKS_DIR / vehicle / subtype_label / "perspective_mask.png"
        if not candidate.exists():
            skipped_no_file += 1
            continue
        index[run_id] = f"perspective_masks/{vehicle}/{subtype_label}/perspective_mask.png"

    OUT.write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"Written {len(index)} entries to {OUT}")
    if skipped_no_subtype:
        print(f"Skipped {skipped_no_subtype} runs with no subtype label")
    if skipped_no_file:
        print(f"Skipped {skipped_no_file} runs whose mask file is missing from perspective_masks/")


if __name__ == "__main__":
    main()
