#!/usr/bin/env python
"""Assemble every measured number into reports/results.md.

Accuracy and latency come from separate runs, so this reads them back and lays them out
together. It draws a hard line between the two machines involved: the CPU, MPS and CoreML
figures are from an Apple M-series laptop, and the TensorRT figures are from a Colab
Tesla T4. Those are different pieces of hardware, so the tables are grouped by machine
rather than merged into one ranking that would compare a Mac against a datacentre GPU.
"""

from __future__ import annotations

import json

from drive_perception.data.convert import CLASS_NAMES
from drive_perception.paths import REPORTS

MODELS = ["yolo11n_kitti", "yolo11s_kitti"]


def _load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        raise SystemExit(f"{path} missing; run the step that produces it first")
    return json.loads(path.read_text())


def accuracy_section() -> list[str]:
    acc = {m: _load(f"accuracy_{m}.json") for m in MODELS}
    lines = [
        "## Detection accuracy",
        "",
        "Fine-tuned on the KITTI training split, scored on all 1496 validation frames at "
        "IoU 0.5 with the project's own evaluator, which applies KITTI's ignore rules.",
        "",
        "| model | " + " | ".join(CLASS_NAMES) + " | mAP50 |",
        "|---|" + "---|" * (len(CLASS_NAMES) + 1),
    ]
    for m in MODELS:
        overall = acc[m]["results"][0]
        ap = overall["per_class_ap"]
        cells = " | ".join(f"{ap[c]:.3f}" for c in CLASS_NAMES)
        lines.append(f"| {m.replace('_kitti', '')} | {cells} | **{overall['mAP']:.3f}** |")

    lines += [
        "",
        "Per difficulty tier (mAP50):",
        "",
        "| model | easy | moderate | hard |",
        "|---|---|---|---|",
    ]
    for m in MODELS:
        by_tier = {r["tier"]: r["mAP"] for r in acc[m]["results"]}
        lines.append(
            f"| {m.replace('_kitti', '')} | {by_tier['easy']:.3f} "
            f"| {by_tier['moderate']:.3f} | {by_tier['hard']:.3f} |"
        )
    return lines


def latency_section() -> list[str]:
    local = {m: _load(f"latency_{m}.json") for m in MODELS}
    trt = _load("tensorrt_results.json")

    def local_fps(model: str, backend: str) -> str:
        for row in local[model]["results"]:
            if row["backend"] == backend:
                return f"{row['fps']:.1f}"
        return "n/a"

    lines = [
        "## Inference latency",
        "",
        "End to end per frame (preprocess, inference, postprocess), median over 100 frames "
        "at the deployed 224x640 input. The two groups are different machines and are not "
        "directly comparable across the line; each shows the best a given target reaches.",
        "",
        "Apple M-series laptop:",
        "",
        "| backend | yolo11n FPS | yolo11s FPS |",
        "|---|---|---|",
    ]
    for backend in ("torch-cpu", "torch-mps", "onnx-cpu", "onnx-coreml"):
        lines.append(
            f"| {backend} | {local_fps('yolo11n_kitti', backend)} "
            f"| {local_fps('yolo11s_kitti', backend)} |"
        )

    lines += [
        "",
        f"NVIDIA {trt['gpu']} (Colab):",
        "",
        "| backend | yolo11n FPS | yolo11s FPS |",
        "|---|---|---|",
    ]
    for prec in ("fp32", "fp16"):
        n = trt["models"]["yolo11n_kitti"][prec]["fps"]
        s = trt["models"]["yolo11s_kitti"][prec]["fps"]
        lines.append(f"| tensorrt-{prec} | {n:.1f} | {s:.1f} |")
    return lines


def int8_section() -> list[str]:
    d = _load("int8_results.json")
    lines = [
        "## INT8: measured, and not adopted",
        "",
        f"Quantised to INT8 on the {d['gpu']}, calibrated on training frames and scored on "
        f"{d['eval_frames']} held-out frames, at a square {d['imgsz']} input (an engine can "
        "only be validated on square images). The point of this table is the change from "
        "FP16, measured the same way for both.",
        "",
        "| model | precision | mAP50 | median ms | FPS |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        for prec in ("fp16", "int8"):
            s = d["models"][m][prec]
            lines.append(
                f"| {m.replace('_kitti', '')} | {prec} | {s['mAP50']:.3f} "
                f"| {s['median_ms']:.2f} | {s['fps']:.1f} |"
            )
        fp16, int8 = d["models"][m]["fp16"], d["models"][m]["int8"]
        dmap = int8["mAP50"] - fp16["mAP50"]
        speed = fp16["median_ms"] / int8["median_ms"]
        lines.append(
            f"| | *int8 vs fp16* | *{dmap:+.3f}* | | *{speed:.2f}x* |"
        )

    lines += [
        "",
        "On this hardware INT8 was not worth it. It cost accuracy on both models and "
        "returned no speed on the small one and less speed on the larger one. These "
        "detectors are small enough that per-frame time is spent on letterboxing and "
        "non-maximum suppression rather than the matrix multiplies INT8 accelerates, so "
        "there is little for the INT8 path to win back. FP16 is the setting to ship.",
    ]
    return lines


def main() -> None:
    parts = [
        "# Results",
        "",
        "Two detectors carried through the whole project: yolo11n as the fast edge target "
        "and yolo11s as the accuracy anchor. Both are fine-tuned on KITTI, then measured "
        "for accuracy and for latency across every backend the project can reach.",
        "",
        *accuracy_section(),
        "",
        *latency_section(),
        "",
        "## What the two models trade",
        "",
        "yolo11s buys about four points of mAP50 over yolo11n. On the CoreML target that "
        "costs roughly 15 percent of the frame rate, and both models still clear 100 FPS, "
        "so the accuracy model is close to free here. On the T4 the larger model is "
        "actually the faster of the two at FP16, because the tiny model leaves the tensor "
        "cores idle. Which model to ship depends on the target, and both are viable.",
        "",
        *int8_section(),
        "",
        "## The short version",
        "",
        "- Fine-tuning lifted mAP50 from 0.394 zero-shot to 0.858 for yolo11n and 0.883 "
        "for yolo11s, most of it on the pedestrian and cyclist classes.",
        "- The exported ONNX matches PyTorch to sub-pixel box differences, so every "
        "latency number is a like-for-like comparison of the same detections.",
        "- CoreML gives the biggest local win, roughly doubling the ONNX CPU frame rate.",
        "- TensorRT FP16 is the fastest overall and the setting to deploy on NVIDIA; INT8 "
        "did not pay for itself at this model size.",
    ]
    out = REPORTS / "results.md"
    out.write_text("\n".join(parts) + "\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
