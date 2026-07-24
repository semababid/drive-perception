#!/usr/bin/env python
"""Export the fine-tuned detector to ONNX and print the resulting graph metadata.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights models/yolo11s_kitti.pt --dynamic
"""

from __future__ import annotations

import argparse

from drive_perception.config import load_config
from drive_perception.export import DEFAULT_OPSET, describe_onnx, export_onnx


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="models/yolo11n_kitti.pt")
    p.add_argument("--imgsz", type=int, default=cfg.detect.imgsz)
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="allow a variable batch size, at a cost in TensorRT build and run time",
    )
    args = p.parse_args()

    path = export_onnx(
        args.weights, imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic
    )
    info = describe_onnx(path)
    print(f"written : {info['path']}")
    print(f"size    : {info['size_mb']} MB")
    print(f"opset   : {info['opset']}  (ir version {info['ir_version']})")
    for name, shape in info["inputs"].items():
        print(f"input   : {name} {shape}")
    for name, shape in info["outputs"].items():
        print(f"output  : {name} {shape}")


if __name__ == "__main__":
    main()
