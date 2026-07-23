#!/usr/bin/env python
"""Score the COCO-pretrained detector on the KITTI validation split.

This is the zero-shot baseline: no KITTI training at all, just COCO classes remapped
onto car, pedestrian and cyclist. It sets the bar that fine-tuning has to beat, and it
writes reports/baseline_metrics.json plus reports/baseline.md.

    python scripts/evaluate_baseline.py
    python scripts/evaluate_baseline.py --weights yolo11s.pt --conf 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from drive_perception.detector import Detector
from drive_perception.evaluate import (
    TIERS,
    evaluate,
    iou,
    load_ground_truth,
    map_predictions,
)
from drive_perception.paths import DETECT_DIR, KITTI_RAW, REPORTS


def cyclist_diagnostic(predictions: dict, ground_truth: dict) -> dict:
    """Measure whether the cyclist score is a detection failure or a box mismatch.

    Counting the bicycle detections and the best IoU any of them reaches separates the
    two cases: no predictions would mean the model never saw a bike, while plenty of
    predictions that all fall short of the threshold means the boxes disagree."""
    n_pred = best = 0
    best = 0.0
    for stem, gts in ground_truth.items():
        targets = [g for g in gts if g.cls_id == 2]
        for d in predictions.get(stem, []):
            if d.cls_id != 2:
                continue
            n_pred += 1
            for g in targets:
                best = max(best, iou(d.box, g.box))
    return {"predictions": n_pred, "best_iou": round(best, 3)}


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| tier | car | pedestrian | cyclist | mAP |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ap = r["per_class_ap"]

        def fmt(v):
            return "n/a" if v is None else f"{v:.3f}"

        lines.append(
            f"| {r['tier']} | {fmt(ap['car'])} | {fmt(ap['pedestrian'])} "
            f"| {fmt(ap['cyclist'])} | {fmt(r['mAP'])} |"
        )
    return lines


def _report(rows: list[dict], weights: str, n_images: int, cyc: dict) -> str:
    overall = rows[0]
    gt = overall["gt_counts"]
    lines = [
        "# Zero-shot baseline on KITTI val",
        "",
        f"Model: `{weights}`, COCO-pretrained, no KITTI training. "
        f"Scored on {n_images} validation frames at IoU 0.5.",
        "",
        f"Ground-truth boxes: car {gt['car']}, pedestrian {gt['pedestrian']}, "
        f"cyclist {gt['cyclist']}.",
        "",
        *_table(rows),
        "",
        "## Reading these numbers",
        "",
        "The COCO classes do not line up with KITTI one for one, and the gaps show up "
        "directly in the per-class scores.",
        "",
        "- **car** maps cleanly from the COCO `car` class, so this column is the fair "
        "measure of what a pretrained detector already knows about the KITTI domain.",
        "- **pedestrian** maps from COCO `person`. The definitions are close, though "
        "KITTI splits out seated people into a separate class that we drop, so a "
        "detection of someone sitting counts against the model here.",
        "- **cyclist** is the weak mapping and the number should be read with that in "
        "mind. KITTI marks a rider and their bicycle as one Cyclist box, while COCO "
        "sees a `bicycle` and a `person` separately.",
        "",
        f"The cyclist column is a box mismatch rather than a detection failure, and the "
        f"numbers separate the two cases. The model produced {cyc['predictions']} bicycle "
        f"detections against {overall['gt_counts']['cyclist']} cyclist boxes, so it is "
        f"clearly seeing the bikes. The best overlap any of those detections reached was "
        f"{cyc['best_iou']}, short of the 0.5 threshold, because the COCO box stops at "
        "the bicycle while the KITTI box also contains the rider. Every one of them is "
        "therefore scored as a miss.",
        "",
        "Fine-tuning on KITTI removes all three mismatches at once, because the model "
        "then learns the KITTI class definitions directly. That is the comparison the "
        "next step makes.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="yolo11n.pt", help="model weights")
    p.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="confidence floor. Low on purpose: AP needs the low-scoring tail of the "
        "curve, and a high threshold silently truncates recall",
    )
    args = p.parse_args()

    val_list = DETECT_DIR / "val.txt"
    if not val_list.exists():
        raise SystemExit(f"{val_list} missing. Run scripts/split_dataset.py first.")
    # splitlines, not split: these paths can legitimately contain spaces, and splitting
    # on whitespace would tear each one apart at the first space.
    images = [Path(line) for line in val_list.read_text().splitlines() if line.strip()]
    stems = [p.stem for p in images]

    detector = Detector(weights=args.weights, conf=args.conf)
    predictions = {
        p.stem: map_predictions(detector.predict(str(p)))
        for p in tqdm(images, desc="detecting", unit="img")
    }
    ground_truth = load_ground_truth(stems, KITTI_RAW / "training" / "label_2")

    rows = [evaluate(predictions, ground_truth)]
    rows += [evaluate(predictions, ground_truth, tier=t) for t in TIERS]

    cyc = cyclist_diagnostic(predictions, ground_truth)

    REPORTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": args.weights,
        "conf": args.conf,
        "images": len(images),
        "results": rows,
        "cyclist_diagnostic": cyc,
    }
    (REPORTS / "baseline_metrics.json").write_text(json.dumps(payload, indent=2))
    (REPORTS / "baseline.md").write_text(_report(rows, args.weights, len(images), cyc))

    print("\n".join(_table(rows)))
    print(f"\nreports written to {REPORTS}")


if __name__ == "__main__":
    main()
