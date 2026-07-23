#!/usr/bin/env python
"""Score the tracker on KITTI tracking sequences and write reports/tracking.md.

Runs the tracker over each sequence, matches its output against the ground truth frame
by frame, and reports MOTA, IDF1 and identity switches per class.

    python scripts/evaluate_tracking.py
    python scripts/evaluate_tracking.py --weights models/yolo11n_kitti.pt --tracker botsort
"""

from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from drive_perception.config import load_config
from drive_perception.data.convert import CLASS_NAMES
from drive_perception.paths import REPORTS, TRACKING
from drive_perception.pipeline import run_sequence
from drive_perception.track_eval import (
    build_accumulator,
    map_track_classes,
    parse_tracking_gt,
    summarize,
)
from drive_perception.tracker import Tracker


def _table(results: dict) -> list[str]:
    lines = [
        "| class | MOTA | IDF1 | ID switches | FP | misses | frag | objects |",
        "|---|---|---|---|---|---|---|---|",
    ]
    def fmt(v):
        return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else v)

    for name, m in results.items():
        lines.append(
            f"| {name} | {fmt(m['mota'])} | {fmt(m['idf1'])} | {m['num_switches']} "
            f"| {m['num_false_positives']} | {m['num_misses']} "
            f"| {m['num_fragmentations']} | {m['num_objects']} |"
        )
    return lines


def _report(results: dict, weights: str, tracker: str, sequences: list[str]) -> str:
    return "\n".join(
        [
            "# Tracking metrics on KITTI",
            "",
            f"Detector `{weights}`, tracker `{tracker}`, sequences "
            f"{', '.join(sequences)}, matched at IoU 0.5.",
            "",
            *_table(results),
            "",
            "## How to read these",
            "",
            "MOTA combines misses, false positives and identity switches into a single "
            "number, so it drops for any of the three. IDF1 asks a narrower question: "
            "how consistently was one real object given one identity. A tracker that "
            "finds everything but keeps renaming it scores well on MOTA and badly on "
            "IDF1, which is why both appear here.",
            "",
            "Identity switches are reported raw because they are the cost a driving "
            "stack actually feels. Every switch is a moment where whatever was "
            "following that object lost its history and started over.",
            "",
            "Ground-truth boxes marked DontCare, and the neighbouring classes the "
            "benchmark treats as ambiguous (Van against car, seated people against "
            "pedestrian), are excluded rather than counted as errors.",
        ]
    ) + "\n"


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--tracker", default=cfg.tracker, choices=["bytetrack", "botsort"])
    p.add_argument("--conf", type=float, default=cfg.detect.conf)
    p.add_argument("--sequences", nargs="+", default=None, help="sequence ids to score")
    p.add_argument("--classes", nargs="+", type=int, default=None, metavar="ID")
    p.add_argument("--out", default="tracking", help="report basename under reports/")
    args = p.parse_args()

    image_root = TRACKING / "training" / "image_02"
    label_root = TRACKING / "training" / "label_02"
    if not image_root.exists():
        raise SystemExit("no tracking data. Run scripts/download_tracking.py first.")
    sequences = args.sequences or sorted(d.name for d in image_root.iterdir() if d.is_dir())

    tracker = Tracker(
        weights=args.weights,
        tracker=args.tracker,
        conf=args.conf,
        iou=cfg.detect.iou,
        imgsz=cfg.detect.imgsz,
        classes=args.classes,
    )

    # One accumulator per class, fed by every sequence. Frame ids are offset per clip so
    # the last frame of one sequence is never mistaken for the neighbour of the first
    # frame of the next, which would invent identity switches that never happened.
    accumulators: dict = {}
    offset = 0
    for seq in sequences:
        predictions: dict[int, list] = {}
        for result in tqdm(
            run_sequence(tracker, image_root / seq, with_trails=False),
            desc=f"seq {seq}",
            unit="frame",
        ):
            predictions[result.index] = map_track_classes(result.tracks)
        ground_truth = parse_tracking_gt((label_root / f"{seq}.txt").read_text())
        for name in CLASS_NAMES:
            accumulators[name] = build_accumulator(
                predictions,
                ground_truth,
                name,
                acc=accumulators.get(name),
                frame_offset=offset,
            )
        offset += len(predictions) + 1_000

    results = summarize(accumulators)
    REPORTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": args.weights,
        "tracker": args.tracker,
        "sequences": sequences,
        "results": results,
    }
    (REPORTS / f"{args.out}_metrics.json").write_text(json.dumps(payload, indent=2))
    (REPORTS / f"{args.out}.md").write_text(
        _report(results, args.weights, args.tracker, sequences)
    )
    print("\n".join(_table(results)))
    print(f"\nreports written to {REPORTS}")


if __name__ == "__main__":
    main()
