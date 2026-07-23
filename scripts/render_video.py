#!/usr/bin/env python
"""Render a tracked sequence to an annotated video.

KITTI tracking clips were captured at 10 Hz, which is the default playback rate here.

    python scripts/render_video.py data/tracking/training/image_02/0000
    python scripts/render_video.py <dir> --weights models/yolo11n_kitti.pt --out demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from drive_perception.config import load_config
from drive_perception.paths import OUTPUTS
from drive_perception.pipeline import render_video
from drive_perception.tracker import Tracker


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="sequence directory or video file")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--tracker", default=cfg.tracker, choices=["bytetrack", "botsort"])
    p.add_argument("--conf", type=float, default=cfg.detect.conf)
    p.add_argument("--fps", type=float, default=10.0, help="playback rate of the output")
    p.add_argument("--limit", type=int, default=None, help="stop after N frames")
    p.add_argument("--out", default=None, help="output video path")
    p.add_argument("--no-trails", action="store_true", help="draw boxes only")
    p.add_argument("--classes", nargs="+", type=int, default=None, metavar="ID")
    args = p.parse_args()

    tracker = Tracker(
        weights=args.weights,
        tracker=args.tracker,
        conf=args.conf,
        iou=cfg.detect.iou,
        imgsz=cfg.detect.imgsz,
        classes=args.classes,
    )
    out = Path(args.out) if args.out else OUTPUTS / f"track_{Path(args.source).name}.mp4"

    stats = render_video(
        tracker,
        Path(args.source),
        out,
        fps=args.fps,
        limit=args.limit,
        with_trails=not args.no_trails,
    )
    print(f"frames written : {stats['frames']}")
    print(f"unique ids     : {stats['unique_ids']}")
    print(f"processing rate: {stats['process_fps']} FPS")
    print(f"video          : {stats['path']}")


if __name__ == "__main__":
    main()
