#!/usr/bin/env python
"""Check that the ONNX export agrees with PyTorch, and record the result.

Every latency number later in the project is only meaningful if the faster backend is
computing the same thing as the slower one. This is what establishes that.

    python scripts/check_parity.py
    python scripts/check_parity.py --images 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drive_perception.config import load_config
from drive_perception.export import (
    describe_onnx,
    detection_parity,
    onnx_path_for,
    raw_parity,
)
from drive_perception.paths import DETECT_DIR, REPORTS

# FP32 on both sides should agree to far better than this. The threshold is loose
# enough to absorb kernel-level differences and tight enough that a real divergence,
# such as a wrongly fused layer, fails immediately.
RAW_TOLERANCE = 1e-3


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="models/yolo11n_kitti.pt")
    p.add_argument("--images", type=int, default=10, help="validation frames to compare")
    args = p.parse_args()

    onnx_path = onnx_path_for(args.weights)
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} missing. Run scripts/export_onnx.py first.")

    # Take the probe shape from the exported graph rather than from config. If the
    # two disagree, that disagreement is the bug worth surfacing, not something to
    # paper over by feeding each backend a different input.
    graph = describe_onnx(onnx_path)
    _, _, height, width = graph["inputs"]["images"]
    print(f"onnx input shape : {height} by {width}")
    raw = raw_parity(args.weights, onnx_path, imgsz=(height, width))
    print(f"raw output shape : {raw['shape']}")
    print(f"max abs diff     : {raw['max_abs_diff']:.3e}")
    print(f"mean abs diff    : {raw['mean_abs_diff']:.3e}")
    passed = raw["max_abs_diff"] < RAW_TOLERANCE
    print(f"within {RAW_TOLERANCE:.0e}      : {'yes' if passed else 'NO'}")

    frames = [
        Path(line)
        for line in (DETECT_DIR / "val.txt").read_text().splitlines()
        if line.strip()
    ][: args.images]
    det = detection_parity(
        args.weights, onnx_path, frames, conf=cfg.detect.conf, imgsz=(height, width)
    )
    print()
    print(f"images compared      : {det['images']}")
    print(f"same detection count : {det['same_detection_count']}/{det['images']}")
    print(f"max box difference   : {det['max_box_diff_px']} px")
    print(f"max score difference : {det['max_score_diff']}")
    print(f"class mismatches     : {det['class_mismatches']}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "parity.json").write_text(
        json.dumps(
            {"tolerance": RAW_TOLERANCE, "raw": raw, "detections": det, "passed": passed},
            indent=2,
        )
    )
    print(f"\nwritten to {REPORTS / 'parity.json'}")
    if not passed:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()
