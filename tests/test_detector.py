"""Detector tests. The pure box-mapping is checked fast; a real inference run is marked
slow because it downloads weights and pulls in torch."""

import pytest

from drive_perception.detector import Detection, build_detections

NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def test_build_detections_maps_and_sorts_by_score():
    dets = build_detections(
        xyxy=[[0, 0, 10, 10], [5, 5, 20, 20]],
        scores=[0.4, 0.9],
        cls_ids=[0, 2],
        names=NAMES,
    )
    # Sorted high-to-low, so the 0.9 cyclist comes first.
    assert [d.score for d in dets] == [0.9, 0.4]
    assert dets[0].cls_name == "cyclist"
    assert dets[0].box == (5.0, 5.0, 20.0, 20.0)
    assert isinstance(dets[0], Detection)


def test_build_detections_empty():
    assert build_detections([], [], [], NAMES) == []


def test_build_detections_length_mismatch_raises():
    # strict zip guards against a backend returning ragged outputs.
    with pytest.raises(ValueError):
        build_detections(xyxy=[[0, 0, 1, 1]], scores=[0.5, 0.6], cls_ids=[0], names=NAMES)


@pytest.mark.slow
def test_real_inference_runs():
    # Downloads yolo11n.pt and runs on a synthetic image; proves the wrapper end to end.
    import numpy as np

    from drive_perception.detector import Detector

    det = Detector(weights="yolo11n.pt")
    out = det.predict(np.zeros((640, 640, 3), dtype="uint8"))
    assert isinstance(out, list)  # a blank image yields zero or more detections
