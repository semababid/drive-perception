#!/usr/bin/env python
"""Fetch KITTI tracking sequences for the tracker evaluation.

Tracking needs consecutive frames rather than the shuffled stills the detector trains
on, so this pulls whole sequences. Only the ones asked for are transferred, out of the
15 GB archive.

    python scripts/download_tracking.py                          # sequences 0000-0002
    python scripts/download_tracking.py --sequences 0000 0011    # pick your own
"""

from __future__ import annotations

import argparse

from drive_perception.data import download
from drive_perception.paths import TRACKING

DEFAULT_SEQUENCES = ["0000", "0001", "0002"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        metavar="ID",
        help=f"sequence ids to fetch (default: {' '.join(DEFAULT_SEQUENCES)})",
    )
    p.add_argument("--force", action="store_true", help="re-download the label archive")
    args = p.parse_args()

    counts = download.download_tracking(args.sequences, force=args.force)
    print(f"total {sum(counts.values())} frames under {TRACKING}")


if __name__ == "__main__":
    main()
