"""Tracking metric tests, built on cases whose answers are known in advance: perfect
tracking, a single swap, a miss, and an ignored region."""

from drive_perception.track_eval import (
    GTTrack,
    build_accumulator,
    covered_fraction,
    map_track_classes,
    parse_tracking_gt,
    summarize,
    to_xywh,
)
from drive_perception.tracker import Track

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (100.0, 100.0, 110.0, 110.0)


def _track(tid, box, name="car", cls_id=0):
    return Track(box=box, score=0.9, cls_id=cls_id, cls_name=name, track_id=tid)


def _gt(frame, tid, box, raw="Car"):
    return GTTrack(frame=frame, track_id=tid, raw_cls=raw, box=box)


def test_to_xywh():
    assert to_xywh((10, 20, 40, 60)) == (10, 20, 30, 40)


def test_covered_fraction_is_relative_to_the_box():
    # The box sits entirely inside a much larger region.
    assert covered_fraction((10, 10, 20, 20), (0, 0, 100, 100)) == 1.0
    assert covered_fraction((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_parse_tracking_gt_reads_frame_and_id():
    text = (
        "0 0 Van 0 0 -1.79 296.74 161.75 455.22 292.37 2.0 1.8 4.4 -4.5 1.8 13.4 -2.1\n"
        "0 -1 DontCare -1 -1 -10 219.31 188.49 245.50 218.56 -1000 -1000 -1000 -10 -1 -1 -1\n"
    )
    gts = parse_tracking_gt(text)
    assert len(gts) == 2
    assert gts[0].frame == 0 and gts[0].track_id == 0 and gts[0].raw_cls == "Van"
    assert gts[0].box == (296.74, 161.75, 455.22, 292.37)
    assert gts[1].raw_cls == "DontCare"


def test_perfect_tracking_scores_one():
    gt = [_gt(0, 1, BOX_A), _gt(1, 1, BOX_A)]
    preds = {0: [_track(5, BOX_A)], 1: [_track(5, BOX_A)]}
    result = summarize({"car": build_accumulator(preds, gt, "car")})["car"]
    assert result["mota"] == 1.0
    assert result["idf1"] == 1.0
    assert result["num_switches"] == 0


def test_identity_switch_is_counted():
    # Same object across two frames, but the tracker renames it on the second.
    gt = [_gt(0, 1, BOX_A), _gt(1, 1, BOX_A)]
    preds = {0: [_track(5, BOX_A)], 1: [_track(6, BOX_A)]}
    result = summarize({"car": build_accumulator(preds, gt, "car")})["car"]
    assert result["num_switches"] == 1
    assert result["mota"] < 1.0


def test_missed_object_lowers_mota():
    gt = [_gt(0, 1, BOX_A), _gt(1, 1, BOX_A)]
    preds = {0: [_track(5, BOX_A)], 1: []}
    result = summarize({"car": build_accumulator(preds, gt, "car")})["car"]
    assert result["num_misses"] == 1
    assert result["mota"] == 0.5  # one miss out of two objects


def test_prediction_inside_dontcare_is_not_a_false_positive():
    gt = [_gt(0, 1, BOX_A), _gt(0, -1, (95.0, 95.0, 200.0, 200.0), raw="DontCare")]
    # Second prediction sits inside the DontCare region and must be discarded.
    preds = {0: [_track(5, BOX_A), _track(6, BOX_B)]}
    result = summarize({"car": build_accumulator(preds, gt, "car")})["car"]
    assert result["num_false_positives"] == 0
    assert result["mota"] == 1.0


def test_van_is_ignored_when_scoring_car():
    # A Van detected as a car is neither right nor wrong, so it must not be an FP.
    gt = [_gt(0, 1, BOX_A), _gt(0, 2, BOX_B, raw="Van")]
    preds = {0: [_track(5, BOX_A), _track(6, BOX_B)]}
    result = summarize({"car": build_accumulator(preds, gt, "car")})["car"]
    assert result["num_false_positives"] == 0


def test_map_track_classes_keeps_ids_and_drops_unmapped():
    mapped = map_track_classes(
        [
            _track(1, BOX_A, name="person"),
            _track(2, BOX_A, name="bicycle"),
            _track(3, BOX_A, name="traffic light"),
        ]
    )
    assert [t.cls_name for t in mapped] == ["pedestrian", "cyclist"]
    assert [t.track_id for t in mapped] == [1, 2]
