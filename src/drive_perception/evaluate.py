"""Detection metrics: IoU matching, per-class average precision, and mAP.

This is written out rather than delegated to `model.val()` for two reasons. The
zero-shot baseline uses an 80-class COCO model against a 3-class dataset, which the
built-in validator cannot score directly. And the baseline and the fine-tuned model
have to be compared on identical maths, which is only guaranteed if both run through
the same code.

AP uses all-point interpolation over the precision-recall curve, the VOC 2010 and
later convention, at a single IoU threshold of 0.5.

Difficulty tiers follow KITTI. When a tier is selected, ground-truth boxes outside it
become ignore regions rather than disappearing: a detection landing on one is discarded
instead of being counted as a false positive. Dropping them outright would punish the
model for finding objects the benchmark chose not to score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data.convert import CLASS_ID, CLASS_NAMES
from .data.stats import Obj
from .detector import Detection

# COCO names mapped onto the KITTI classes. `car` and `person` line up cleanly. The
# cyclist row is the weak one: KITTI labels a rider and their bike as a single Cyclist
# box, while COCO sees a `bicycle` and a `person` as two separate things, so the
# bicycle box covers only part of the KITTI box and often misses the IoU threshold.
COCO_TO_KITTI: dict[str, str] = {
    "car": "car",
    "person": "pedestrian",
    "bicycle": "cyclist",
}

TIERS = ("easy", "moderate", "hard")


@dataclass(frozen=True)
class GTBox:
    box: tuple[float, float, float, float]
    cls_id: int
    tier: str  # easy, moderate, hard, or ignored


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def average_precision(scores: list[float], is_tp: list[bool], n_gt: int) -> float:
    """AP from a flat list of scored detections, using all-point interpolation."""
    if n_gt == 0:
        return float("nan")  # undefined, not zero: there was nothing to find
    if not scores:
        return 0.0

    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.asarray(is_tp, dtype=bool)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)

    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Pad the curve, then make precision monotonically decreasing before integrating.
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def parse_kitti_gt(label_text: str) -> list[GTBox]:
    """Read raw KITTI label text into GT boxes carrying their difficulty tier.

    The raw labels are used rather than the converted YOLO ones because only these
    still hold the truncation and occlusion columns the tiers are defined from."""
    out: list[GTBox] = []
    for line in label_text.splitlines():
        f = line.split()
        if len(f) < 8 or f[0] not in CLASS_ID:
            continue
        left, top, right, bottom = (float(x) for x in f[4:8])
        obj = Obj(
            cls=f[0],
            truncated=float(f[1]),
            occluded=int(float(f[2])),
            height=bottom - top,
            width=right - left,
        )
        out.append(GTBox((left, top, right, bottom), CLASS_ID[f[0]], obj.tier()))
    return out


def load_ground_truth(stems: Sequence[str], label_dir: Path) -> dict[str, list[GTBox]]:
    return {s: parse_kitti_gt((label_dir / f"{s}.txt").read_text()) for s in stems}


def map_predictions(dets: Sequence[Detection]) -> list[Detection]:
    """Relabel detections as KITTI classes, dropping everything unmapped.

    Handles both models the project uses. A COCO-pretrained model needs its names
    translated, while the fine-tuned model already emits KITTI names and must pass
    through untouched. Translating only the COCO names would silently discard every
    pedestrian and cyclist the fine-tuned model found and report an AP of zero for
    both, which looks like a broken model rather than a broken mapping."""
    out: list[Detection] = []
    for d in dets:
        name = d.cls_name.lower()
        kitti = COCO_TO_KITTI.get(name, name)
        if kitti not in CLASS_NAMES:
            continue
        out.append(
            Detection(
                box=d.box,
                score=d.score,
                cls_id=CLASS_NAMES.index(kitti),
                cls_name=kitti,
            )
        )
    return out


def _match_one_image(
    dets: Sequence[Detection],
    gts: Sequence[GTBox],
    cls_id: int,
    tier: str | None,
    iou_thr: float,
) -> tuple[list[float], list[bool], int]:
    """Greedy highest-score-first matching for a single class in a single image."""
    scored = sorted((d for d in dets if d.cls_id == cls_id), key=lambda d: -d.score)
    same_class = [g for g in gts if g.cls_id == cls_id]
    if tier is None:
        targets = [g for g in same_class if g.tier != "ignored"]
        ignored = [g for g in same_class if g.tier == "ignored"]
    else:
        targets = [g for g in same_class if g.tier == tier]
        ignored = [g for g in same_class if g.tier != tier]

    taken = [False] * len(targets)
    scores: list[float] = []
    flags: list[bool] = []
    for d in scored:
        best_i, best_iou = -1, iou_thr
        for i, g in enumerate(targets):
            if taken[i]:
                continue
            v = iou(d.box, g.box)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            taken[best_i] = True
            scores.append(d.score)
            flags.append(True)
            continue
        # No target matched. If it landed on an ignore region, drop it silently.
        if any(iou(d.box, g.box) >= iou_thr for g in ignored):
            continue
        scores.append(d.score)
        flags.append(False)
    return scores, flags, len(targets)


def evaluate(
    predictions: dict[str, list[Detection]],
    ground_truth: dict[str, list[GTBox]],
    tier: str | None = None,
    iou_thr: float = 0.5,
) -> dict:
    """Per-class AP and mAP over a set of frames, optionally restricted to one tier."""
    per_class: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cls_id, name in enumerate(CLASS_NAMES):
        scores: list[float] = []
        flags: list[bool] = []
        n_gt = 0
        for stem, gts in ground_truth.items():
            s, f, n = _match_one_image(
                predictions.get(stem, []), gts, cls_id, tier, iou_thr
            )
            scores += s
            flags += f
            n_gt += n
        per_class[name] = average_precision(scores, flags, n_gt)
        counts[name] = n_gt

    scored = [v for v in per_class.values() if not np.isnan(v)]
    return {
        "tier": tier or "all",
        "iou_threshold": iou_thr,
        "per_class_ap": {k: (None if np.isnan(v) else round(v, 4)) for k, v in per_class.items()},
        "gt_counts": counts,
        "mAP": round(float(np.mean(scored)), 4) if scored else None,
    }
