#!/usr/bin/env python
"""Benchmark inference latency across the local backends and write reports/latency_*.

Four backends are timed on the same frames at the same resolution, so the only thing
that varies is where and how the model runs:

  torch-mps    the training framework on the Apple GPU, the unoptimised reference
  torch-cpu    the same framework on the CPU, to show what the GPU is worth
  onnx-cpu     the exported graph under ONNX Runtime on the CPU, no torch involved
  onnx-coreml  the same graph handed to CoreML, which reaches the Neural Engine

The TensorRT numbers come later from the Colab notebook and are merged in at the final
results step, since this machine has no NVIDIA GPU.

    python scripts/benchmark.py
    python scripts/benchmark.py --weights models/yolo11s_kitti.pt --frames 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drive_perception.benchmark import measure
from drive_perception.config import load_config
from drive_perception.detector import Detector
from drive_perception.export import KITTI_IMGSZ, onnx_path_for
from drive_perception.onnx_detector import OnnxDetector
from drive_perception.paths import DETECT_DIR, REPORTS


def build_backends(weights: str, onnx_path: Path, conf: float, iou: float) -> list:
    """Construct the backends to time, skipping any whose provider is unavailable."""
    import onnxruntime as ort

    backends = [
        ("torch-mps", Detector(weights, conf, iou, imgsz=KITTI_IMGSZ, device="mps")),
        ("torch-cpu", Detector(weights, conf, iou, imgsz=KITTI_IMGSZ, device="cpu")),
        ("onnx-cpu", OnnxDetector(onnx_path, conf, iou, ["CPUExecutionProvider"])),
    ]
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        backends.append(
            (
                "onnx-coreml",
                OnnxDetector(
                    onnx_path, conf, iou, ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                ),
            )
        )
    return backends


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="models/yolo11n_kitti.pt")
    p.add_argument("--frames", type=int, default=100, help="frames to time per backend")
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    onnx_path = onnx_path_for(args.weights)
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} missing. Run scripts/export_onnx.py first.")

    frames = [
        Path(line)
        for line in (DETECT_DIR / "val.txt").read_text().splitlines()
        if line.strip()
    ][: args.frames]
    frame_strings = [str(f) for f in frames]

    rows = []
    for name, detector in build_backends(args.weights, onnx_path, cfg.detect.conf, cfg.detect.iou):
        stats = measure(detector.predict, frame_strings, name, warmup=args.warmup)
        rows.append(stats)
        print(
            f"{stats.backend:12s} median {stats.median_ms:7.2f} ms  "
            f"p90 {stats.p90_ms:7.2f} ms  {stats.fps:6.1f} FPS"
        )

    fastest = max(rows, key=lambda r: r.fps)
    slowest = min(rows, key=lambda r: r.fps)
    speedup = round(fastest.fps / slowest.fps, 2) if slowest.fps else 0.0
    print(f"\nfastest: {fastest.backend} at {fastest.fps} FPS ({speedup}x over {slowest.backend})")

    model = Path(args.weights).stem
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"latency_{model}.json").write_text(
        json.dumps(
            {
                "model": model,
                "frames": len(frames),
                "imgsz": list(KITTI_IMGSZ),
                "results": [r.as_dict() for r in rows],
            },
            indent=2,
        )
    )
    print(f"written to {REPORTS / f'latency_{model}.json'}")


if __name__ == "__main__":
    main()
