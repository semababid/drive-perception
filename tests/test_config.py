"""Phase A smoke test: the config loads and carries both detector sizes. It gives CI
something real to run and guards against the YAML drifting out of shape later."""

from drive_perception.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.detect.imgsz > 0
    assert cfg.tracker in {"bytetrack", "botsort"}


def test_both_detectors_present():
    cfg = load_config()
    # The whole benchmark story depends on carrying an edge model and an accuracy
    # anchor through the pipeline, so both must be set and distinct.
    assert cfg.models.edge == "yolo11n"
    assert cfg.models.accuracy == "yolo11s"
    assert cfg.model_names == ["yolo11n", "yolo11s"]
