"""Evaluation tests. The AP maths is the easiest thing here to get subtly wrong, so it
is checked against cases that can be worked out by hand."""

import math

from drive_perception.detector import Detection
from drive_perception.evaluate import (
    GTBox,
    average_precision,
    evaluate,
    iou,
    map_predictions,
    parse_kitti_gt,
)


def _det(box, score=0.9, cls_id=0, name="car"):
    return Detection(box=box, score=score, cls_id=cls_id, cls_name=name)


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes offset by 5 in x: intersection 50, union 150.
    assert math.isclose(iou((0, 0, 10, 10), (5, 0, 15, 10)), 50 / 150)


def test_ap_perfect_detector_is_one():
    assert average_precision([0.9, 0.8], [True, True], n_gt=2) == 1.0


def test_ap_with_no_predictions_is_zero():
    assert average_precision([], [], n_gt=5) == 0.0


def test_ap_undefined_when_no_ground_truth():
    assert math.isnan(average_precision([0.9], [False], n_gt=0))


def test_ap_half_recall():
    # One TP out of two GT, no FP: precision stays 1 up to recall 0.5, so AP = 0.5.
    assert math.isclose(average_precision([0.9], [True], n_gt=2), 0.5)


def test_ap_ranks_confident_false_positive_worse():
    # Same TP/FP counts, but the FP outranking the TP must score lower.
    good = average_precision([0.9, 0.1], [True, False], n_gt=1)
    bad = average_precision([0.9, 0.1], [False, True], n_gt=1)
    assert good > bad


def test_parse_kitti_gt_keeps_classes_and_tiers():
    text = (
        "Car 0.00 0 -1.5 100.0 100.0 200.0 200.0 1 1 1 1 1 1 1\n"
        "DontCare -1 -1 -10 0.0 0.0 5.0 5.0 -1 -1 -1 -1 -1 -1 -1\n"
    )
    gts = parse_kitti_gt(text)
    assert len(gts) == 1  # DontCare dropped
    assert gts[0].cls_id == 0
    assert gts[0].tier == "easy"  # 100px tall, unoccluded, untruncated


def test_map_predictions_translates_coco_names():
    mapped = map_predictions(
        [
            _det((0, 0, 1, 1), name="car"),
            _det((0, 0, 1, 1), name="person"),
            _det((0, 0, 1, 1), name="bicycle"),
            _det((0, 0, 1, 1), name="traffic light"),
        ]
    )
    assert [d.cls_name for d in mapped] == ["car", "pedestrian", "cyclist"]
    assert [d.cls_id for d in mapped] == [0, 1, 2]


def test_map_predictions_passes_through_finetuned_names():
    # The fine-tuned model already emits KITTI names. Dropping them here would report
    # an AP of zero for pedestrian and cyclist and look like a broken model.
    mapped = map_predictions(
        [
            _det((0, 0, 1, 1), name="car"),
            _det((0, 0, 1, 1), name="pedestrian"),
            _det((0, 0, 1, 1), name="cyclist"),
        ]
    )
    assert [d.cls_name for d in mapped] == ["car", "pedestrian", "cyclist"]
    assert [d.cls_id for d in mapped] == [0, 1, 2]


def test_evaluate_perfect_match():
    gt = {"a": [GTBox((0, 0, 10, 10), 0, "easy")]}
    preds = {"a": [_det((0, 0, 10, 10))]}
    r = evaluate(preds, gt)
    assert r["per_class_ap"]["car"] == 1.0
    assert r["gt_counts"]["car"] == 1


def test_detection_on_ignored_box_is_not_a_false_positive():
    # An 'ignored' GT is out of scope for scoring. Finding it must not be punished, so
    # car AP stays undefined (no scorable GT) rather than dropping to zero.
    gt = {"a": [GTBox((0, 0, 10, 10), 0, "ignored")]}
    preds = {"a": [_det((0, 0, 10, 10))]}
    r = evaluate(preds, gt)
    assert r["per_class_ap"]["car"] is None


def test_tier_filter_scopes_the_ground_truth():
    gt = {
        "a": [
            GTBox((0, 0, 10, 10), 0, "easy"),
            GTBox((50, 50, 60, 60), 0, "hard"),
        ]
    }
    preds = {"a": [_det((0, 0, 10, 10))]}
    easy = evaluate(preds, gt, tier="easy")
    assert easy["gt_counts"]["car"] == 1
    assert easy["per_class_ap"]["car"] == 1.0
    # Against the hard tier only, that same detection matches nothing scorable.
    hard = evaluate(preds, gt, tier="hard")
    assert hard["gt_counts"]["car"] == 1
    assert hard["per_class_ap"]["car"] == 0.0
