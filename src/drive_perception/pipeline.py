"""The end to end pipeline: read frames, track through them, draw the result.

This is the piece that turns a detector and a tracker into something you can watch. It
reads either a directory of ordered frames, which is how KITTI ships its sequences, or
a video file, and yields one annotated frame at a time rather than building a list, so
a long clip does not have to fit in memory at once.

Trails are anchored at the bottom centre of each box. For objects on a road that point
sits where the object meets the ground, so the trail follows its path along the road
instead of sliding around as the box grows on approach.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .tracker import Track, Tracker, track_label
from .viz import draw_detections, draw_hud, draw_trails

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class Trails:
    """Recent positions per track id, with stale ids dropped.

    Without pruning this grows for the whole clip and starts drawing lines for objects
    that left the scene long ago."""

    maxlen: int = 30
    forget_after: int = 15  # frames an id can go unseen before it is dropped
    points: dict[int, deque] = field(default_factory=dict)
    meta: dict[int, tuple[int, str]] = field(default_factory=dict)
    last_seen: dict[int, int] = field(default_factory=dict)

    def update(self, tracks: list[Track], frame_index: int) -> None:
        for t in tracks:
            x1, _, x2, y2 = t.box
            anchor = ((x1 + x2) / 2, y2)  # ground contact point
            self.points.setdefault(t.track_id, deque(maxlen=self.maxlen)).append(anchor)
            self.meta[t.track_id] = (t.cls_id, t.cls_name)
            self.last_seen[t.track_id] = frame_index
        self._prune(frame_index)

    def _prune(self, frame_index: int) -> None:
        stale = [
            tid
            for tid, seen in self.last_seen.items()
            if frame_index - seen > self.forget_after
        ]
        for tid in stale:
            self.points.pop(tid, None)
            self.meta.pop(tid, None)
            self.last_seen.pop(tid, None)

    def drawable(self) -> list[tuple[list[tuple[float, float]], int, str]]:
        return [
            (list(pts), *self.meta[tid])
            for tid, pts in self.points.items()
            if tid in self.meta
        ]


@dataclass
class FrameResult:
    index: int
    tracks: list[Track]
    annotated: np.ndarray


def sequence_frames(directory: Path) -> list[Path]:
    """Ordered image paths for a KITTI sequence directory."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def iter_frames(source: Path) -> Iterator[np.ndarray]:
    """Yield frames from a directory of images or from a video file."""
    source = Path(source)
    if source.is_dir():
        for path in sequence_frames(source):
            image = cv2.imread(str(path))
            if image is not None:
                yield image
        return

    capture = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def annotate(
    image: np.ndarray,
    tracks: list[Track],
    trails: Trails | None = None,
    show_score: bool = True,
) -> np.ndarray:
    """Draw trails first, then boxes on top, so labels are never hidden by a line."""
    out = draw_trails(image, trails.drawable()) if trails is not None else image
    return draw_detections(
        out, tracks, labels=[track_label(t, show_score) for t in tracks]
    )


def run_sequence(
    tracker: Tracker,
    source: Path,
    with_trails: bool = True,
    limit: int | None = None,
) -> Iterator[FrameResult]:
    """Track through a clip, yielding each annotated frame as it is produced.

    The tracker is reset first so ids start at one for this clip rather than continuing
    from whatever was tracked before."""
    tracker.reset()
    trails = Trails() if with_trails else None
    for index, frame in enumerate(iter_frames(source)):
        if limit is not None and index >= limit:
            return
        tracks = tracker.update(frame)
        if trails is not None:
            trails.update(tracks, index)
        yield FrameResult(index, tracks, annotate(frame, tracks, trails))


def write_video(
    frames: Iterable[np.ndarray],
    out_path: Path,
    fps: float = 10.0,
    codec: str = "mp4v",
) -> dict:
    """Write frames to a video file, sized from the first frame.

    OpenCV silently discards any frame whose size differs from the writer's, producing
    a short video and no error at all, so mismatched frames are resized rather than
    dropped. KITTI sequences vary in width between clips, which makes this a real case
    and not a hypothetical one."""
    writer = None
    size: tuple[int, int] | None = None
    written = resized = 0
    try:
        for frame in frames:
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*codec), fps, size
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open a video writer for {out_path}")
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
                resized += 1
            writer.write(frame)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return {"frames": written, "resized": resized, "size": size, "path": out_path}


def render_video(
    tracker: Tracker,
    source: Path,
    out_path: Path,
    fps: float = 10.0,
    limit: int | None = None,
    with_trails: bool = True,
    hud: bool = True,
    codec: str = "mp4v",
) -> dict:
    """Track through a clip and write the annotated result to a video file.

    The HUD reports the measured processing rate rather than the playback rate, since
    the question a reviewer asks of a perception demo is how fast the model actually
    ran, not how fast the file plays back."""
    ids: set[int] = set()
    started = time.perf_counter()

    def annotated_frames() -> Iterator[np.ndarray]:
        for result in run_sequence(tracker, source, with_trails, limit):
            ids.update(t.track_id for t in result.tracks)
            frame = result.annotated
            if hud:
                elapsed = time.perf_counter() - started
                rate = (result.index + 1) / elapsed if elapsed > 0 else 0.0
                frame = draw_hud(
                    frame,
                    [
                        f"frame {result.index + 1}",
                        f"tracking {len(result.tracks)} objects",
                        f"{rate:.1f} FPS",
                    ],
                )
            yield frame

    stats = write_video(annotated_frames(), out_path, fps=fps, codec=codec)
    elapsed = time.perf_counter() - started
    stats["unique_ids"] = len(ids)
    stats["process_fps"] = round(stats["frames"] / elapsed, 1) if elapsed else 0.0
    return stats
