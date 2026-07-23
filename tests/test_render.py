"""Video writing tests. These write real files to a temp directory and read them back,
because the failure this guards against is silent: OpenCV drops mismatched frames
without raising, so only a round trip proves the frames actually landed."""

import cv2
import numpy as np

from drive_perception.pipeline import write_video
from drive_perception.viz import draw_hud


def _frames(n, w=64, h=48):
    return [np.full((h, w, 3), i * 5 % 255, dtype=np.uint8) for i in range(n)]


def test_video_round_trip_keeps_every_frame(tmp_path):
    out = tmp_path / "clip.mp4"
    stats = write_video(_frames(12), out, fps=10)
    assert stats["frames"] == 12
    assert out.exists() and out.stat().st_size > 0

    capture = cv2.VideoCapture(str(out))
    try:
        counted = 0
        while capture.read()[0]:
            counted += 1
    finally:
        capture.release()
    assert counted == 12


def test_mismatched_frames_are_resized_not_dropped(tmp_path):
    # The second frame is a different size. It must still reach the file.
    frames = [np.zeros((48, 64, 3), np.uint8), np.zeros((30, 40, 3), np.uint8)]
    stats = write_video(frames, tmp_path / "mixed.mp4", fps=5)
    assert stats["frames"] == 2
    assert stats["resized"] == 1
    assert stats["size"] == (64, 48)


def test_empty_input_writes_nothing(tmp_path):
    stats = write_video([], tmp_path / "empty.mp4", fps=10)
    assert stats["frames"] == 0
    assert stats["size"] is None


def test_hud_draws_without_changing_shape():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = draw_hud(img, ["frame 1", "tracking 3 objects", "24.0 FPS"])
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_hud_with_no_lines_is_a_noop():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    assert np.array_equal(draw_hud(img, []), img)
