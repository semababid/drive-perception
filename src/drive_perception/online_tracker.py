"""A small IoU-based online tracker, with no framework behind it.

The project already has a strong tracker for offline evaluation: ByteTrack and BoT-SORT
through Ultralytics, compared in the tracking reports. That path pulls in PyTorch, which
is the wrong dependency to ship inside a prediction service.

This tracker exists for the service instead. It associates the current detections to the
live tracks by intersection over union, greedily and highest overlap first, opens a new
id for anything unmatched, and keeps a lost track alive for a few frames in case it comes
back. There is no motion model, so it is weaker than ByteTrack under fast motion, but it
runs on the same detections with only numpy behind it, which is what a lightweight ONNX
service wants.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .detector import Detection
from .geometry import iou
from .tracker import Track


@dataclass
class _LiveTrack:
    box: tuple[float, float, float, float]
    cls_id: int
    cls_name: str
    score: float
    age: int = 0  # frames since this track was last matched


@dataclass
class SimpleTracker:
    iou_threshold: float = 0.3
    max_age: int = 30  # frames a lost track survives before it is dropped
    _tracks: dict[int, _LiveTrack] = field(default_factory=dict)
    _next_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Advance one frame and return the tracks visible in it (matched or new)."""
        # Candidate matches, only within the same class, above the overlap threshold.
        candidates = []
        for tid, track in self._tracks.items():
            for di, det in enumerate(detections):
                if det.cls_id != track.cls_id:
                    continue
                overlap = iou(track.box, det.box)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, tid, di))
        candidates.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        result: list[Track] = []

        for _, tid, di in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            det = detections[di]
            self._tracks[tid] = _LiveTrack(det.box, det.cls_id, det.cls_name, det.score, 0)
            result.append(self._as_track(tid, det))

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _LiveTrack(det.box, det.cls_id, det.cls_name, det.score, 0)
            result.append(self._as_track(tid, det))

        # Age the tracks that went unseen this frame, and drop the ones long gone.
        for tid in list(self._tracks):
            if tid not in matched_tracks and tid not in matched_dets:
                self._tracks[tid].age += 1
                if self._tracks[tid].age > self.max_age:
                    del self._tracks[tid]

        result.sort(key=lambda t: t.score, reverse=True)
        return result

    @staticmethod
    def _as_track(tid: int, det: Detection) -> Track:
        return Track(
            box=det.box,
            score=det.score,
            cls_id=det.cls_id,
            cls_name=det.cls_name,
            track_id=tid,
        )
