#!/usr/bin/env python
"""Download KITTI 2D object-detection data (left color images + training labels).

The default is a small subset so a fresh clone can start training in minutes; the
subset is pulled straight from the remote archive with HTTP range requests, so it
costs a few hundred MB rather than the full 12 GB. Pass --full for everything.

    python scripts/download_kitti.py                 # first 300 frames
    python scripts/download_kitti.py --subset 1000   # first 1000 frames
    python scripts/download_kitti.py --full          # all 7481 frames (~12 GB)
"""

from __future__ import annotations

import argparse

from drive_perception.data import download


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--subset",
        type=int,
        default=300,
        metavar="N",
        help="download only the first N training frames (default: 300)",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="download the entire training split (~12 GB)",
    )
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = p.parse_args()

    if args.full:
        download.download_full(force=args.force)
    else:
        download.download_subset(args.subset, force=args.force)


if __name__ == "__main__":
    main()
