"""Tracker tests. The mapping and the Track/Detection relationship are checked here;
anything needing a real model is left to the pipeline step."""

import pytest

from drive_perception.detector import Detection
from drive_perception.tracker import Track, build_tracks, track_label
from drive_perception.viz import draw_detections

NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def test_build_tracks_maps_ids_and_sorts():
    tracks = build_tracks(
        xyxy=[[0, 0, 10, 10], [5, 5, 20, 20]],
        scores=[0.4, 0.9],
        cls_ids=[0, 1],
        track_ids=[7, 3],
        names=NAMES,
    )
    assert [t.score for t in tracks] == [0.9, 0.4]
    assert [t.track_id for t in tracks] == [3, 7]
    assert tracks[0].cls_name == "pedestrian"


def test_track_is_a_detection():
    # The visualization is typed against Detection, so Track must genuinely be one.
    t = Track(box=(0, 0, 1, 1), score=0.5, cls_id=0, cls_name="car", track_id=2)
    assert isinstance(t, Detection)


def test_tracks_render_through_the_detection_drawing_path():
    import numpy as np

    tracks = build_tracks([[1, 1, 9, 9]], [0.8], [0], [4], NAMES)
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    out = draw_detections(img, tracks, labels=[track_label(t) for t in tracks])
    assert out.shape == img.shape


def test_track_label_leads_with_the_id():
    t = Track(box=(0, 0, 1, 1), score=0.91, cls_id=0, cls_name="car", track_id=7)
    assert track_label(t) == "#7 car 0.91"
    assert track_label(t, show_score=False) == "#7 car"


def test_ragged_tracker_output_raises():
    with pytest.raises(ValueError):
        build_tracks([[0, 0, 1, 1]], [0.5], [0], [1, 2], NAMES)
