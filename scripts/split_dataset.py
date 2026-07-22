#!/usr/bin/env python
"""Split the converted KITTI set into train/val and write the Ultralytics data.yaml.

Run after convert_labels.py. The split is seeded, so it is reproducible:

    python scripts/split_dataset.py
    python scripts/split_dataset.py --val-frac 0.15 --seed 7
"""

from __future__ import annotations

import argparse

from drive_perception.data import split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-frac", type=float, default=0.2, help="validation fraction")
    p.add_argument("--seed", type=int, default=42, help="shuffle seed")
    args = p.parse_args()

    r = split.build(val_frac=args.val_frac, seed=args.seed)
    print(f"train frames: {r['train_frames']}    val frames: {r['val_frames']}")
    print("class     train   val")
    for name in r["train_boxes"]:
        print(f"  {name:9s} {r['train_boxes'][name]:5d} {r['val_boxes'][name]:5d}")
    if 0 in r["val_boxes"].values():
        print("warning: a class has no boxes in val; consider a different seed")
    print(f"data.yaml written to {r['yaml']}")


if __name__ == "__main__":
    main()
