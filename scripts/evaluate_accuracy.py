#!/usr/bin/env python
"""Score a fine-tuned detector on the full KITTI validation split and record its mAP.

Both detector sizes go through this same code so their accuracy is comparable. It writes
reports/accuracy_<model>.json with the overall and per-tier per-class AP, which the final
results table reads alongside the latency numbers.

    python scripts/evaluate_accuracy.py --weights models/yolo11n_kitti.pt
    python scripts/evaluate_accuracy.py --weights models/yolo11s_kitti.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from drive_perception.detector import Detector
from drive_perception.evaluate import TIERS, evaluate, load_ground_truth, map_predictions
from drive_perception.export import KITTI_IMGSZ
from drive_perception.paths import DETECT_DIR, KITTI_RAW, REPORTS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="models/yolo11n_kitti.pt")
    p.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="low on purpose: AP needs the low-scoring tail of the curve",
    )
    args = p.parse_args()

    images = [
        Path(line)
        for line in (DETECT_DIR / "val.txt").read_text().splitlines()
        if line.strip()
    ]
    ground_truth = load_ground_truth(
        [p.stem for p in images], KITTI_RAW / "training" / "label_2"
    )

    detector = Detector(weights=args.weights, conf=args.conf, imgsz=KITTI_IMGSZ)
    predictions = {
        p.stem: map_predictions(detector.predict(str(p)))
        for p in tqdm(images, desc=Path(args.weights).stem, unit="img")
    }

    rows = [evaluate(predictions, ground_truth)]
    rows += [evaluate(predictions, ground_truth, tier=t) for t in TIERS]

    model = Path(args.weights).stem
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"accuracy_{model}.json"
    out.write_text(
        json.dumps({"model": model, "images": len(images), "results": rows}, indent=2)
    )
    overall = rows[0]
    print(f"{model}: mAP50={overall['mAP']}  per-class={overall['per_class_ap']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
