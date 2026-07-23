"""Multi-object tracking on top of the detector.

Detection alone answers "what is in this frame". Tracking answers "is that the same car
as last frame", which is what a driving stack actually needs: counting objects,
measuring how they move, and reacting to one that keeps approaching.

`Track` extends `Detection` with a persistent id, so anything that already accepts
detections, the visualization above all, accepts tracks unchanged.

ByteTrack is the default. It associates high-confidence boxes first and then makes a
second pass over the low-confidence ones, which is what keeps an id alive through the
partial occlusions that fill a street scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .detector import Detection

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Track(Detection):
    track_id: int


def build_tracks(
    xyxy: list,
    scores: list,
    cls_ids: list,
    track_ids: list,
    names: dict[int, str],
) -> list[Track]:
    """Map raw tracker output to Track objects, sorted by score. Pure, so the mapping
    is testable without loading a model."""
    tracks = [
        Track(
            box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            score=float(s),
            cls_id=int(c),
            cls_name=names[int(c)],
            track_id=int(t),
        )
        for b, s, c, t in zip(xyxy, scores, cls_ids, track_ids, strict=True)
    ]
    tracks.sort(key=lambda t: t.score, reverse=True)
    return tracks


def track_label(track: Track, show_score: bool = True) -> str:
    """Label text for a tracked object. The id leads because when watching a clip the
    question is whether an id stayed stable, not what the confidence was."""
    head = f"#{track.track_id} {track.cls_name}"
    return f"{head} {track.score:.2f}" if show_score else head


class Tracker:
    """Detector plus a tracking algorithm, driven one frame at a time.

    State carries across `update` calls, which is what lets ids persist. Call `reset`
    between clips, otherwise ids continue climbing from the previous sequence and the
    identity metrics for the new one are meaningless."""

    def __init__(
        self,
        weights: str | Path = "yolo11n.pt",
        tracker: str = "bytetrack",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        if tracker not in {"bytetrack", "botsort"}:
            raise ValueError(f"tracker must be bytetrack or botsort, got {tracker!r}")
        self.model = YOLO(str(weights))
        self.names: dict[int, str] = self.model.names
        self.tracker_cfg = f"{tracker}.yaml"
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def update(self, frame: str | Path | np.ndarray) -> list[Track]:
        """Feed the next frame and get back the currently tracked objects."""
        result = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        boxes = result.boxes
        # id is None on the first frames, or whenever nothing was confidently tracked.
        if boxes is None or boxes.id is None:
            return []
        return build_tracks(
            xyxy=boxes.xyxy.tolist(),
            scores=boxes.conf.tolist(),
            cls_ids=boxes.cls.tolist(),
            track_ids=boxes.id.tolist(),
            names=self.names,
        )

    def reset(self) -> None:
        """Drop all track state so the next clip starts numbering from scratch."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            tracker.reset()
