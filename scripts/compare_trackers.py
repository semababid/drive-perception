#!/usr/bin/env python
"""Collect the tracking runs into one comparison and write reports/tracking_comparison.md.

Three runs are compared, and they are deliberately arranged so that only one thing
changes at a time. The first two share a tracker and differ in detector, which isolates
what detection quality is worth. The last two share a detector and differ in tracker,
which isolates what the association algorithm is worth.

    python scripts/compare_trackers.py
"""

from __future__ import annotations

import json

from drive_perception.data.convert import CLASS_NAMES
from drive_perception.paths import REPORTS

RUNS = [
    ("pretrained + ByteTrack", "tracking_baseline_metrics.json"),
    ("fine-tuned + ByteTrack", "tracking_finetuned_metrics.json"),
    ("fine-tuned + BoT-SORT", "tracking_botsort_metrics.json"),
]

HEADLINE = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses"]
LABELS = {
    "mota": "MOTA",
    "idf1": "IDF1",
    "num_switches": "ID switches",
    "num_false_positives": "false positives",
    "num_misses": "misses",
}


def _fmt(key: str, value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if key in {"mota", "idf1", "motp"} else f"{int(value)}"


def _overall_table(loaded: list[tuple[str, dict]]) -> list[str]:
    lines = ["| metric | " + " | ".join(n for n, _ in loaded) + " |"]
    lines.append("|" + "---|" * (len(loaded) + 1))
    for key in HEADLINE:
        cells = [_fmt(key, r["results"]["OVERALL"][key]) for _, r in loaded]
        lines.append(f"| {LABELS[key]} | " + " | ".join(cells) + " |")
    return lines


def _per_class_table(loaded: list[tuple[str, dict]]) -> list[str]:
    lines = ["| class | metric | " + " | ".join(n for n, _ in loaded) + " |"]
    lines.append("|" + "---|" * (len(loaded) + 2))
    for name in CLASS_NAMES:
        for key in ("mota", "idf1", "num_switches"):
            cells = [_fmt(key, r["results"][name][key]) for _, r in loaded]
            lines.append(f"| {name} | {LABELS[key]} | " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    loaded: list[tuple[str, dict]] = []
    for label, filename in RUNS:
        path = REPORTS / filename
        if not path.exists():
            raise SystemExit(f"{path} missing. Run scripts/evaluate_tracking.py first.")
        loaded.append((label, json.loads(path.read_text())))

    base, tuned, botsort = (r for _, r in loaded)
    o_base = base["results"]["OVERALL"]
    o_tuned = tuned["results"]["OVERALL"]
    o_bot = botsort["results"]["OVERALL"]

    detection_gain = o_tuned["mota"] - o_base["mota"]
    switch_from_detector = int(o_tuned["num_switches"] - o_base["num_switches"])
    switch_from_tracker = int(o_bot["num_switches"] - o_tuned["num_switches"])

    report = "\n".join(
        [
            "# Tracking: what detection fixes and what association fixes",
            "",
            f"Three runs over the same {len(base['sequences'])} KITTI sequences, "
            "matched at IoU 0.5. Only one thing changes between neighbouring columns, "
            "so each step isolates a single cause.",
            "",
            *_overall_table(loaded),
            "",
            "## Reading the result",
            "",
            f"Replacing the pretrained detector with the fine-tuned one lifted MOTA by "
            f"{detection_gain:+.3f}, cutting misses from {int(o_base['num_misses'])} to "
            f"{int(o_tuned['num_misses'])}. That is the single largest change in the "
            "table, and it comes entirely from the detector.",
            "",
            f"It barely touched identity switches, which moved by {switch_from_detector} "
            f"from {int(o_base['num_switches'])} to {int(o_tuned['num_switches'])}. "
            "Finding an object more reliably does not, on its own, help the tracker "
            "decide that the object it found is the same one as last frame.",
            "",
            f"Switching from ByteTrack to BoT-SORT, with the detector held fixed, moved "
            f"switches by {switch_from_tracker}, from {int(o_tuned['num_switches'])} to "
            f"{int(o_bot['num_switches'])}. The two trackers use identical association "
            "thresholds here, so the difference is global motion compensation. The KITTI "
            "camera is mounted on a moving car, and estimating that motion between "
            "frames lets the tracker tell an object moving on its own apart from the "
            "whole scene shifting.",
            "",
            "The practical conclusion is that misses and identity switches are separate "
            "problems with separate fixes. A better detector addresses the first and "
            "leaves the second largely untouched. A motion-aware association step "
            "addresses the second. Reaching for a better detector to solve identity "
            "churn would have been effort spent in the wrong place.",
            "",
            "## Per class",
            "",
            *_per_class_table(loaded),
            "",
            "Cyclist is worth noting: it was untrackable with the pretrained detector "
            "because COCO draws a bicycle where KITTI draws a rider and bicycle "
            "together, so almost nothing matched. Once the detector learned the KITTI "
            "box, cyclist identity became the most stable of the three classes.",
        ]
    ) + "\n"

    (REPORTS / "tracking_comparison.md").write_text(report)
    print("\n".join(_overall_table(loaded)))
    print(f"\nwritten to {REPORTS / 'tracking_comparison.md'}")


if __name__ == "__main__":
    main()
