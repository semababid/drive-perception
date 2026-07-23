#!/usr/bin/env python
"""Track through a KITTI sequence and report how the ids behaved.

Saves a few annotated frames so the result can be eyeballed. Writing the whole clip out
as a video is the next step.

    python scripts/track_sequence.py data/tracking/training/image_02/0000
    python scripts/track_sequence.py <dir> --weights models/yolo11n_kitti.pt --tracker botsort
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from drive_perception.config import load_config
from drive_perception.paths import OUTPUTS
from drive_perception.pipeline import run_sequence
from drive_perception.tracker import Tracker
from drive_perception.viz import save


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="sequence directory or video file")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--tracker", default=cfg.tracker, choices=["bytetrack", "botsort"])
    p.add_argument("--conf", type=float, default=cfg.detect.conf)
    p.add_argument("--limit", type=int, default=None, help="stop after N frames")
    p.add_argument("--save-every", type=int, default=50, help="save every Nth frame")
    p.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="only track these class ids. For COCO weights, 0 2 1 are person, car, "
        "bicycle; leave unset for the fine-tuned model, which only has our three",
    )
    args = p.parse_args()

    tracker = Tracker(
        weights=args.weights,
        tracker=args.tracker,
        conf=args.conf,
        iou=cfg.detect.iou,
        imgsz=cfg.detect.imgsz,
        classes=args.classes,
    )
    out_dir = OUTPUTS / f"track_{Path(args.source).name}"

    seen_ids: set[int] = set()
    per_frame: list[int] = []
    for result in tqdm(
        run_sequence(tracker, Path(args.source), limit=args.limit),
        desc="tracking",
        unit="frame",
    ):
        seen_ids.update(t.track_id for t in result.tracks)
        per_frame.append(len(result.tracks))
        if result.index % args.save_every == 0:
            save(result.annotated, out_dir / f"{result.index:06d}.jpg")

    frames = len(per_frame)
    print(f"frames tracked   : {frames}")
    print(f"unique track ids : {len(seen_ids)}")
    print(f"objects per frame: {sum(per_frame) / frames:.1f}" if frames else "no frames")
    print(f"sample frames    : {out_dir}")


if __name__ == "__main__":
    main()
