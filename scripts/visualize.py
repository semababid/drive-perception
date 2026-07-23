#!/usr/bin/env python
"""Detect on an image and write an annotated copy to outputs/.

    python scripts/visualize.py data/raw/kitti/training/image_2/000008.png
    python scripts/visualize.py <image> --weights runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from drive_perception.detector import Detector
from drive_perception.paths import OUTPUTS
from drive_perception.viz import draw_detections, save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="path to an image")
    p.add_argument("--weights", default="yolo11n.pt", help="model weights")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--out", default=None, help="output path (defaults to outputs/)")
    args = p.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")

    detections = Detector(weights=args.weights, conf=args.conf).predict(args.image)
    annotated = draw_detections(image, detections)

    out = Path(args.out) if args.out else OUTPUTS / f"{Path(args.image).stem}_detected.jpg"
    save(annotated, out)
    print(f"{len(detections)} detections drawn, written to {out}")


if __name__ == "__main__":
    main()
