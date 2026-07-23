"""Tracking metrics: MOTA, IDF1 and identity switches.

Detection accuracy says nothing about whether a tracker keeps calling the same car the
same car. MOTA folds misses, false positives and identity switches into one number,
IDF1 measures how consistently identities are preserved, and the raw switch count says
how often the tracker changed its mind. A tracker can post a decent MOTA while churning
identities, so all three get reported rather than just the headline.

Two KITTI rules are applied so the numbers are not unfairly harsh:

DontCare regions mark areas the benchmark refuses to score. A prediction landing in one
is dropped rather than counted as a false positive, matched on the fraction of the
prediction covered by the region rather than IoU, because those regions are large and
IoU would almost never fire.

Neighbouring classes are ignored the way the official benchmark ignores them. A Van
scored against Car, or a seated person scored against Pedestrian, is neither a hit nor a
miss: the class boundary is genuinely ambiguous, so it is excluded instead of punished.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# motmetrics 1.4.0 still calls np.asfarray, which NumPy dropped in 2.0. Restoring the
# one-line equivalent keeps the library working. Pinning NumPy back instead would drag
# the whole torch and ultralytics stack down to match, which is a far larger change
# than the single removed alias actually warrants.
if not hasattr(np, "asfarray"):  # pragma: no cover
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402  (must follow the shim above)

from .data.convert import CLASS_NAMES  # noqa: E402
from .evaluate import COCO_TO_KITTI, iou  # noqa: E402
from .tracker import Track  # noqa: E402

# KITTI tracking labels use these names for the classes we score.
TRACK_CLASSES = {"Car": "car", "Pedestrian": "pedestrian", "Cyclist": "cyclist"}

# Classes too close to ours to score against, excluded rather than counted wrong.
NEIGHBOUR_IGNORE: dict[str, set[str]] = {
    "car": {"Van"},
    "pedestrian": {"Person_sitting"},
    "cyclist": set(),
}

IGNORE_CLASS = "DontCare"

METRICS = [
    "mota",
    "idf1",
    "motp",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "num_fragmentations",
    "mostly_tracked",
    "mostly_lost",
    "num_objects",
]


@dataclass(frozen=True)
class GTTrack:
    frame: int
    track_id: int
    raw_cls: str  # the KITTI name, e.g. Car or DontCare
    box: tuple[float, float, float, float]


def parse_tracking_gt(text: str) -> list[GTTrack]:
    """Read a KITTI tracking label file.

    These rows carry two extra leading columns compared with the detection labels:
    the frame index and the track id, so the box sits at columns 6 through 9."""
    out: list[GTTrack] = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 10:
            continue
        left, top, right, bottom = (float(x) for x in f[6:10])
        out.append(
            GTTrack(
                frame=int(f[0]),
                track_id=int(f[1]),
                raw_cls=f[2],
                box=(left, top, right, bottom),
            )
        )
    return out


def to_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    """motmetrics wants rectangles as x, y, width, height."""
    return (box[0], box[1], box[2] - box[0], box[3] - box[1])


def covered_fraction(box: Sequence[float], region: Sequence[float]) -> float:
    """How much of `box` falls inside `region`. Used for ignore regions, where IoU
    would understate the overlap because the region is much larger than the box."""
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / area if area > 0 else 0.0


def map_track_classes(tracks: Sequence[Track]) -> list[Track]:
    """Relabel COCO track classes as KITTI ones, keeping ids intact.

    The detection helper returns plain Detections and would throw the track id away,
    which is the one field that matters here."""
    out: list[Track] = []
    for t in tracks:
        kitti = COCO_TO_KITTI.get(t.cls_name.lower(), None)
        name = kitti if kitti is not None else t.cls_name.lower()
        if name not in CLASS_NAMES:
            continue
        out.append(
            Track(
                box=t.box,
                score=t.score,
                cls_id=CLASS_NAMES.index(name),
                cls_name=name,
                track_id=t.track_id,
            )
        )
    return out


def _keep_hypotheses(
    hyps: Sequence[Track],
    targets: Sequence[GTTrack],
    ignore_regions: Sequence[Sequence[float]],
    iou_thr: float,
    ignore_overlap: float,
) -> list[Track]:
    """Drop predictions that only overlap an ignore region and no scorable object."""
    keep: list[Track] = []
    for h in hyps:
        if any(iou(h.box, g.box) >= iou_thr for g in targets):
            keep.append(h)
            continue
        if any(covered_fraction(h.box, r) >= ignore_overlap for r in ignore_regions):
            continue
        keep.append(h)
    return keep


def build_accumulator(
    predictions: dict[int, list[Track]],
    ground_truth: Sequence[GTTrack],
    cls_name: str,
    iou_thr: float = 0.5,
    ignore_overlap: float = 0.5,
    acc: mm.MOTAccumulator | None = None,
    frame_offset: int = 0,
) -> mm.MOTAccumulator:
    """Accumulate frame-by-frame matches for one class.

    Passing an existing accumulator and a frame offset lets several sequences feed one
    set of totals. The offset keeps frame ids from different clips apart, which matters
    because identity continuity is measured across consecutive frame ids."""
    raw_wanted = next(k for k, v in TRACK_CLASSES.items() if v == cls_name)
    ignore_names = NEIGHBOUR_IGNORE[cls_name] | {IGNORE_CLASS}

    by_frame: dict[int, list[GTTrack]] = {}
    ignores: dict[int, list[tuple]] = {}
    for g in ground_truth:
        if g.raw_cls == raw_wanted:
            by_frame.setdefault(g.frame, []).append(g)
        elif g.raw_cls in ignore_names:
            ignores.setdefault(g.frame, []).append(g.box)

    acc = acc if acc is not None else mm.MOTAccumulator(auto_id=False)
    for frame in sorted(set(by_frame) | set(predictions)):
        targets = by_frame.get(frame, [])
        hyps = [t for t in predictions.get(frame, []) if t.cls_name == cls_name]
        hyps = _keep_hypotheses(
            hyps, targets, ignores.get(frame, []), iou_thr, ignore_overlap
        )
        # motmetrics treats distance as 1 - IoU, so the cutoff is 1 - the IoU we want.
        distances = mm.distances.iou_matrix(
            [to_xywh(g.box) for g in targets],
            [to_xywh(h.box) for h in hyps],
            max_iou=1.0 - iou_thr,
        )
        acc.update(
            [g.track_id for g in targets],
            [h.track_id for h in hyps],
            distances,
            frameid=frame + frame_offset,
        )
    return acc


def summarize(accumulators: dict[str, mm.MOTAccumulator]) -> dict:
    """Compute the metric table for a set of per-class accumulators."""
    if not accumulators:
        return {}
    host = mm.metrics.create()
    frame = host.compute_many(
        list(accumulators.values()),
        names=list(accumulators),
        metrics=METRICS,
        generate_overall=True,
    )
    out: dict[str, dict] = {}
    for name, row in frame.iterrows():
        out[str(name)] = {
            k: (None if row[k] != row[k] else round(float(row[k]), 4)) for k in METRICS
        }
    return out
