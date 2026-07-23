#!/usr/bin/env python
"""Run the detector on one image and print what it found.

Uses COCO-pretrained yolo11n by default, so it works before any KITTI training. Drawing
the boxes comes in the visualization step; this one just proves the wrapper runs.

    python scripts/detect.py data/raw/kitti/training/image_2/000000.png
    python scripts/detect.py <image> --weights runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse

from drive_perception.detector import Detector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="path to an image")
    p.add_argument("--weights", default="yolo11n.pt", help="model weights")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = p.parse_args()

    det = Detector(weights=args.weights, conf=args.conf)
    detections = det.predict(args.image)
    print(f"{len(detections)} detections:")
    for d in detections:
        x1, y1, x2, y2 = (round(v, 1) for v in d.box)
        print(f"  {d.cls_name:12s} {d.score:.2f}  [{x1}, {y1}, {x2}, {y2}]")


if __name__ == "__main__":
    main()
