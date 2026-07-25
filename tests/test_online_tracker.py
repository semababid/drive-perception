"""Tests for the lightweight service tracker. Its whole job is keeping ids stable across
frames, so the cases here are id continuity, birth, death, and class separation."""

from drive_perception.detector import Detection
from drive_perception.online_tracker import SimpleTracker


def _det(box, cls_id=0, name="car", score=0.9):
    return Detection(box=box, score=score, cls_id=cls_id, cls_name=name)


def test_new_detection_gets_an_id():
    t = SimpleTracker()
    tracks = t.update([_det((0, 0, 10, 10))])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_same_object_keeps_its_id_across_frames():
    t = SimpleTracker()
    first = t.update([_det((0, 0, 10, 10))])
    # A slightly moved box with high overlap is the same object.
    second = t.update([_det((1, 1, 11, 11))])
    assert first[0].track_id == second[0].track_id == 1


def test_disjoint_box_gets_a_new_id():
    t = SimpleTracker()
    t.update([_det((0, 0, 10, 10))])
    tracks = t.update([_det((100, 100, 110, 110))])
    assert tracks[0].track_id == 2


def test_two_objects_get_separate_ids():
    t = SimpleTracker()
    tracks = t.update([_det((0, 0, 10, 10)), _det((50, 50, 60, 60))])
    assert {tr.track_id for tr in tracks} == {1, 2}


def test_same_location_different_class_is_not_matched():
    t = SimpleTracker()
    t.update([_det((0, 0, 10, 10), cls_id=0, name="car")])
    # A cyclist appearing where the car was must not inherit the car's id.
    tracks = t.update([_det((0, 0, 10, 10), cls_id=2, name="cyclist")])
    assert tracks[0].track_id == 2
    assert tracks[0].cls_name == "cyclist"


def test_lost_track_is_dropped_after_max_age():
    t = SimpleTracker(max_age=2)
    t.update([_det((0, 0, 10, 10))])  # id 1 born
    for _ in range(3):
        t.update([])  # three empty frames, past max_age
    # Track 1 is gone, so the object reappearing is a fresh id.
    tracks = t.update([_det((0, 0, 10, 10))])
    assert tracks[0].track_id == 2


def test_reset_clears_ids():
    t = SimpleTracker()
    t.update([_det((0, 0, 10, 10))])
    t.reset()
    tracks = t.update([_det((0, 0, 10, 10))])
    assert tracks[0].track_id == 1
