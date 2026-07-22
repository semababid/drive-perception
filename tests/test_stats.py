"""Fast checks for the KITTI difficulty-tier logic — the one piece of the EDA that
encodes the benchmark protocol and must not drift."""

from drive_perception.data.stats import Obj


def test_easy_object():
    # Tall, fully visible, barely truncated -> easy.
    assert Obj("Car", truncated=0.0, occluded=0, height=60, width=90).tier() == "easy"


def test_moderate_needs_partial_occlusion_bump():
    # 30 px tall and partly occluded fails easy but meets moderate.
    assert Obj("Car", truncated=0.2, occluded=1, height=30, width=40).tier() == "moderate"


def test_hard_object():
    assert Obj("Pedestrian", truncated=0.4, occluded=2, height=28, width=12).tier() == "hard"


def test_tiny_box_is_ignored():
    # Below 25 px KITTI ignores the object entirely, whatever its occlusion.
    assert Obj("Cyclist", truncated=0.0, occluded=0, height=18, width=10).tier() == "ignored"
