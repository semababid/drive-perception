#!/usr/bin/env python
"""Convert the downloaded KITTI labels to YOLO format under data/processed/kitti-det.

Run this after downloading data and before the train/val split:

    python scripts/convert_labels.py            # build once
    python scripts/convert_labels.py --force    # rebuild from scratch
"""

from __future__ import annotations

import argparse

from drive_perception.data import convert
from drive_perception.paths import DETECT_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="rebuild even if a set exists")
    args = p.parse_args()

    r = convert.convert(force=args.force)
    print(f"frames processed : {r.frames}")
    print(f"boxes written    : {r.boxes_written}")
    print(f"dropped (class)  : {r.dropped_other_class}")
    print(f"dropped (degen)  : {r.dropped_degenerate}")
    print(f"output           : {DETECT_DIR}")


if __name__ == "__main__":
    main()
