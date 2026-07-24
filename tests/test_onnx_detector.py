"""Tests for the standalone ONNX inference path. Letterboxing, suppression and box
decoding are pure array work, so all of this runs without a model or a session."""

import numpy as np
import pytest

from drive_perception.onnx_detector import PAD_VALUE, decode, letterbox, nms


def test_letterbox_preserves_aspect_ratio():
    # A 1242x375 driving frame into a 224x640 box: width fills, height is padded.
    image = np.zeros((375, 1242, 3), np.uint8)
    out, scale, (pad_x, pad_y) = letterbox(image, (224, 640))
    assert out.shape == (224, 640, 3)
    assert scale == pytest.approx(640 / 1242, rel=1e-6)
    assert pad_x == 0 and pad_y > 0


def test_letterbox_pads_with_the_expected_grey():
    image = np.full((10, 100, 3), 255, np.uint8)
    out, _, (_, pad_y) = letterbox(image, (64, 64))
    assert out[0, 0].tolist() == [PAD_VALUE] * 3  # padded band
    assert out[pad_y + 1, 32].tolist() == [255] * 3  # actual content


def test_letterbox_of_a_matching_ratio_adds_no_padding():
    out, scale, pad = letterbox(np.zeros((100, 200, 3), np.uint8), (50, 100))
    assert pad == (0, 0)
    assert scale == pytest.approx(0.5)
    assert out.shape == (50, 100, 3)


def test_nms_keeps_the_best_of_two_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.6, 0.9], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [1]  # the 0.9 box wins


def test_nms_keeps_boxes_that_do_not_overlap():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.6, 0.9], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.5)) == [0, 1]


def test_nms_on_empty_input():
    assert nms(np.empty((0, 4), np.float32), np.empty((0,), np.float32), 0.5) == []


def _single_prediction(cx, cy, w, h, class_scores):
    # Graph layout is (1, 4 + classes, anchors); one anchor here.
    row = np.array([cx, cy, w, h, *class_scores], dtype=np.float32)
    return row[None, :, None]


def test_decode_maps_a_box_back_to_original_coordinates():
    # No scaling and no padding, so the decoded box should match the input exactly.
    out = _single_prediction(50, 50, 20, 10, [0.9, 0.0, 0.0])
    boxes, scores, ids = decode(out, conf=0.25, iou=0.7, scale=1.0, pad=(0, 0), shape=(100, 100))
    assert boxes.shape == (1, 4)
    assert boxes[0].tolist() == [40.0, 45.0, 60.0, 55.0]  # centre and size to corners
    assert scores[0] == pytest.approx(0.9)
    assert ids[0] == 0


def test_decode_undoes_scale_and_padding():
    # Box at x=100 in padded space, 10px left pad, half scale -> x=180 originally.
    out = _single_prediction(100, 60, 20, 20, [0.0, 0.8, 0.0])
    boxes, _, ids = decode(
        out, conf=0.25, iou=0.7, scale=0.5, pad=(10, 5), shape=(1000, 1000)
    )
    assert boxes[0][0] == pytest.approx((100 - 10 - 10) / 0.5)
    assert ids[0] == 1


def test_decode_drops_low_confidence():
    out = _single_prediction(50, 50, 10, 10, [0.1, 0.05, 0.0])
    boxes, scores, ids = decode(out, conf=0.25, iou=0.7, scale=1.0, pad=(0, 0), shape=(100, 100))
    assert len(boxes) == 0 and len(scores) == 0 and len(ids) == 0


def test_decode_clips_boxes_to_the_frame():
    # A box hanging off the top left must be clipped, not left negative.
    out = _single_prediction(5, 5, 40, 40, [0.9, 0.0, 0.0])
    boxes, _, _ = decode(out, conf=0.25, iou=0.7, scale=1.0, pad=(0, 0), shape=(100, 100))
    assert boxes[0][0] >= 0.0 and boxes[0][1] >= 0.0


def test_decode_does_not_suppress_across_classes():
    # Two overlapping boxes of different classes must both survive, because a cyclist
    # and the car behind it can legitimately occupy the same pixels.
    a = np.array([50, 50, 20, 20, 0.9, 0.0, 0.0], dtype=np.float32)
    b = np.array([51, 51, 20, 20, 0.0, 0.85, 0.0], dtype=np.float32)
    out = np.stack([a, b], axis=1)[None]
    boxes, _, ids = decode(out, conf=0.25, iou=0.5, scale=1.0, pad=(0, 0), shape=(200, 200))
    assert sorted(ids.tolist()) == [0, 1]
