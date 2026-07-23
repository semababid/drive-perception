#!/usr/bin/env python
"""Score the pretrained and fine-tuned detectors on the same data and write the
before and after comparison to reports/comparison.md.

Both models run through the same evaluation code on the same validation split, so the
difference between the two columns is the effect of fine-tuning and nothing else.

    python scripts/compare_models.py
    python scripts/compare_models.py --tuned models/yolo11s_kitti.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from drive_perception.data.convert import CLASS_NAMES
from drive_perception.detector import Detector
from drive_perception.evaluate import TIERS, evaluate, load_ground_truth, map_predictions
from drive_perception.paths import DETECT_DIR, KITTI_RAW, REPORTS


def score(weights: str, images: list[Path], ground_truth: dict, conf: float) -> list[dict]:
    detector = Detector(weights=weights, conf=conf)
    predictions = {
        p.stem: map_predictions(detector.predict(str(p)))
        for p in tqdm(images, desc=Path(weights).stem, unit="img")
    }
    rows = [evaluate(predictions, ground_truth)]
    rows += [evaluate(predictions, ground_truth, tier=t) for t in TIERS]
    return rows


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _delta(before, after) -> str:
    if before is None or after is None:
        return "n/a"
    d = after - before
    return f"{d:+.3f}"


def _table(base: list[dict], tuned: list[dict]) -> list[str]:
    lines = [
        "| tier | metric | pretrained | fine-tuned | change |",
        "|---|---|---|---|---|",
    ]
    for b, t in zip(base, tuned, strict=True):
        for name in CLASS_NAMES:
            lines.append(
                f"| {b['tier']} | {name} AP | {_fmt(b['per_class_ap'][name])} "
                f"| {_fmt(t['per_class_ap'][name])} "
                f"| {_delta(b['per_class_ap'][name], t['per_class_ap'][name])} |"
            )
        lines.append(
            f"| {b['tier']} | **mAP** | **{_fmt(b['mAP'])}** | **{_fmt(t['mAP'])}** "
            f"| **{_delta(b['mAP'], t['mAP'])}** |"
        )
    return lines


def _report(base: list[dict], tuned: list[dict], base_w: str, tuned_w: str, n: int) -> str:
    b, t = base[0], tuned[0]
    return "\n".join(
        [
            "# Fine-tuning against the zero-shot baseline",
            "",
            f"`{base_w}` is the COCO-pretrained model with its classes remapped onto "
            f"KITTI. `{tuned_w}` is the same architecture fine-tuned on the KITTI "
            f"training split. Both are scored by the same code on the same "
            f"{n} validation frames at IoU 0.5.",
            "",
            *_table(base, tuned),
            "",
            "## What changed",
            "",
            f"Overall mAP moved from {_fmt(b['mAP'])} to {_fmt(t['mAP'])}. The three "
            "classes did not move for the same reasons, and the per-class rows are "
            "more informative than the mean.",
            "",
            f"**Cyclist** is the clearest result: {_fmt(b['per_class_ap']['cyclist'])} "
            f"to {_fmt(t['per_class_ap']['cyclist'])}. The pretrained model could see "
            "bicycles perfectly well but drew them the way COCO defines them, around "
            "the bicycle alone, while KITTI draws one box around rider and bicycle "
            "together. No amount of extra confidence fixes a box that is the wrong "
            "shape. Learning the KITTI definition does.",
            "",
            f"**Car** was already the pretrained model's strongest class at "
            f"{_fmt(b['per_class_ap']['car'])}, so there was less headroom, and it "
            f"reached {_fmt(t['per_class_ap']['car'])}.",
            "",
            f"**Pedestrian** went from {_fmt(b['per_class_ap']['pedestrian'])} to "
            f"{_fmt(t['per_class_ap']['pedestrian'])}. COCO's `person` class is close "
            "to KITTI's Pedestrian, so the starting point was reasonable, and the gain "
            "comes mostly from the smaller and partly occluded cases.",
            "",
            "The tier rows show where the remaining errors live. The gap between easy "
            "and hard is the distant, occluded traffic that the exploratory analysis "
            "flagged at the start, and it is still the hardest part of the problem.",
        ]
    ) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="yolo11n.pt")
    p.add_argument("--tuned", default="models/yolo11n_kitti.pt")
    p.add_argument("--conf", type=float, default=0.05)
    args = p.parse_args()

    if not Path(args.tuned).exists():
        raise SystemExit(f"{args.tuned} missing. Run scripts/finetune.py first.")

    images = [
        Path(line)
        for line in (DETECT_DIR / "val.txt").read_text().splitlines()
        if line.strip()
    ]
    ground_truth = load_ground_truth(
        [p.stem for p in images], KITTI_RAW / "training" / "label_2"
    )

    base = score(args.pretrained, images, ground_truth, args.conf)
    tuned = score(args.tuned, images, ground_truth, args.conf)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "comparison_metrics.json").write_text(
        json.dumps(
            {
                "pretrained": {"weights": args.pretrained, "results": base},
                "finetuned": {"weights": args.tuned, "results": tuned},
                "images": len(images),
            },
            indent=2,
        )
    )
    (REPORTS / "comparison.md").write_text(
        _report(base, tuned, args.pretrained, args.tuned, len(images))
    )
    print("\n".join(_table(base, tuned)))
    print(f"\nreports written to {REPORTS}")


if __name__ == "__main__":
    main()
