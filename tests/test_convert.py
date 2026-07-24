"""Fast checks for the KITTI to YOLO box conversion. The geometry is pure and worth
locking down: an off-by-one in normalization silently poisons every label."""

from drive_perception.data.convert import CLASS_ID, to_yolo_bbox


def _line(cls, left, top, right, bottom):
    # A KITTI line is 15 columns; only 0 and 4..7 matter here, rest can be zeros.
    return [cls, "0", "0", "0", str(left), str(top), str(right), str(bottom)] + ["0"] * 7


def test_center_and_size_are_normalized():
    # 200x200 box at (100,200)-(300,400) in a 1000x500 image.
    row = to_yolo_bbox(_line("Car", 100, 200, 300, 400), w=1000, h=500)
    assert row == "0 0.200000 0.600000 0.200000 0.400000"


def test_every_class_view_comes_from_one_definition():
    # These were four separate copies once. If they ever diverge again, the detection
    # and tracking evaluations start scoring different things without any error.
    from drive_perception.data.convert import CLASS_NAMES, KITTI_CLASSES
    from drive_perception.data.stats import KEEP
    from drive_perception.track_eval import TRACK_CLASSES

    assert KEEP is KITTI_CLASSES
    assert TRACK_CLASSES is KITTI_CLASSES
    assert CLASS_NAMES == list(KITTI_CLASSES.values())
    assert CLASS_ID == {raw: i for i, raw in enumerate(KITTI_CLASSES)}


def test_class_ids_match_the_three_kept_classes():
    assert CLASS_ID == {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    assert to_yolo_bbox(_line("Pedestrian", 0, 0, 50, 100), 500, 500).startswith("1 ")
    assert to_yolo_bbox(_line("Cyclist", 0, 0, 50, 100), 500, 500).startswith("2 ")


def test_box_is_clipped_to_image_bounds():
    # Left and top spill negative; they must clip to 0 before normalizing.
    row = to_yolo_bbox(_line("Car", -20, -20, 100, 100), w=200, h=200)
    # clipped box is (0,0)-(100,100): center (50,50)/200 = 0.25, size 100/200 = 0.5
    assert row == "0 0.250000 0.250000 0.500000 0.500000"


def test_degenerate_box_is_dropped():
    # Zero-width after clipping -> None so the caller can count it as degenerate.
    assert to_yolo_bbox(_line("Car", 100, 50, 100, 150), w=500, h=500) is None
