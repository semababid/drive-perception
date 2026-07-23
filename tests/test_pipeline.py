"""Pipeline tests. Trail bookkeeping and frame reading are pure enough to test without
a model, and the pruning rule is the part that would otherwise leak memory unnoticed."""

import cv2
import numpy as np

from drive_perception.pipeline import Trails, annotate, iter_frames, sequence_frames
from drive_perception.tracker import Track


def _track(tid=1, box=(10, 10, 30, 40), cls_id=0, name="car"):
    return Track(box=box, score=0.9, cls_id=cls_id, cls_name=name, track_id=tid)


def test_trail_follows_the_ground_contact_point():
    t = Trails()
    t.update([_track(box=(10, 0, 30, 40))], frame_index=0)
    # bottom centre of the box: x = (10+30)/2, y = 40
    assert list(t.points[1]) == [(20.0, 40.0)]


def test_trail_accumulates_and_is_capped():
    t = Trails(maxlen=3)
    for i in range(5):
        t.update([_track(box=(i, 0, i + 10, 20))], frame_index=i)
    assert len(t.points[1]) == 3  # oldest points dropped


def test_stale_tracks_are_forgotten():
    t = Trails(forget_after=2)
    t.update([_track(tid=1)], frame_index=0)
    t.update([_track(tid=2)], frame_index=1)
    # Track 1 has not been seen for 3 frames by now, so it should be gone.
    t.update([_track(tid=2)], frame_index=4)
    assert 1 not in t.points
    assert 2 in t.points


def test_drawable_pairs_points_with_class():
    t = Trails()
    t.update([_track(tid=5, cls_id=2, name="cyclist")], frame_index=0)
    (points, cls_id, cls_name) = t.drawable()[0]
    assert cls_id == 2 and cls_name == "cyclist" and len(points) == 1


def test_annotate_returns_a_new_frame():
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    trails = Trails()
    trails.update([_track()], frame_index=0)
    out = annotate(img, [_track()], trails)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_iter_frames_reads_a_directory_in_order(tmp_path):
    for i in (2, 0, 1):  # written out of order on purpose
        cv2.imwrite(str(tmp_path / f"{i:06d}.png"), np.full((8, 8, 3), i * 10, np.uint8))
    frames = list(iter_frames(tmp_path))
    assert len(frames) == 3
    # Sorted by filename, so the pixel values come back in ascending order.
    assert [int(f[0, 0, 0]) for f in frames] == [0, 10, 20]


def test_sequence_frames_ignores_non_images(tmp_path):
    cv2.imwrite(str(tmp_path / "000000.png"), np.zeros((4, 4, 3), np.uint8))
    (tmp_path / "notes.txt").write_text("not a frame")
    assert [p.name for p in sequence_frames(tmp_path)] == ["000000.png"]
