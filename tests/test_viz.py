"""Visualization tests. These draw on small arrays, so they stay fast and need no model."""

import numpy as np
import pytest

from drive_perception.detector import Detection
from drive_perception.viz import PALETTE, color_for, draw_detections, format_label


def _det(cls_id=0, name="car", score=0.9, box=(10, 10, 50, 40)):
    return Detection(box=box, score=score, cls_id=cls_id, cls_name=name)


def test_colors_are_distinct_per_class_and_cycle():
    assert len({color_for(i) for i in range(3)}) == 3
    assert color_for(len(PALETTE)) == color_for(0)


def test_color_follows_class_name_across_models():
    # COCO calls a car id 2, our fine-tuned model calls it id 0. Same colour either way,
    # otherwise the same object changes colour between the two models.
    assert color_for(2, "car") == color_for(0, "car")
    # COCO synonyms share the colour of the KITTI class they stand for.
    assert color_for(0, "person") == color_for(1, "pedestrian")
    assert color_for(0, "bicycle") == color_for(2, "cyclist")


def test_unknown_class_name_falls_back_to_id():
    assert color_for(1, "traffic light") == PALETTE[1]


def test_format_label():
    assert format_label(_det()) == "car 0.90"
    assert format_label(_det(), show_score=False) == "car"


def test_draw_does_not_mutate_input_and_keeps_shape():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    before = img.copy()
    out = draw_detections(img, [_det()])
    assert out.shape == img.shape
    assert np.array_equal(img, before), "input image was modified"
    assert not np.array_equal(out, before), "nothing was drawn"


def test_box_outside_frame_is_clamped():
    # A box running past the edges must not raise; it should clamp and still draw.
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    out = draw_detections(img, [_det(box=(-30, -30, 500, 500))])
    assert out.shape == img.shape


def test_custom_labels_override_text():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # This is the path the tracker uses to show persistent ids.
    out = draw_detections(img, [_det()], labels=["#7 car 0.90"])
    assert out.shape == img.shape


def test_label_count_mismatch_raises():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        draw_detections(img, [_det(), _det()], labels=["only one"])
