"""Export tests. The real export needs torch and is marked slow; the path and metadata
handling are checked here against a hand-built ONNX graph, so no model is required."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from drive_perception.export import (
    DEFAULT_OPSET,
    KITTI_IMGSZ,
    describe_onnx,
    export_onnx,
    onnx_path_for,
    preprocess,
)


def test_default_export_shape_matches_driving_frames():
    # Square exports pad a 3.3:1 driving frame with blank bars, which changes the
    # apparent scale of every object and costs almost three times the pixels.
    height, width = KITTI_IMGSZ
    assert width > height
    assert height % 32 == 0 and width % 32 == 0  # stride-aligned


def test_preprocess_accepts_a_square_size():
    out = preprocess(np.zeros((375, 1242, 3), np.uint8), 640)
    assert out.shape == (1, 3, 640, 640)


def test_preprocess_accepts_a_rectangle_as_height_width():
    out = preprocess(np.zeros((375, 1242, 3), np.uint8), (224, 640))
    assert out.shape == (1, 3, 224, 640)


def test_preprocess_normalises_to_unit_range():
    out = preprocess(np.full((10, 10, 3), 255, np.uint8), (32, 32))
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_onnx_path_sits_beside_the_checkpoint():
    assert onnx_path_for("models/yolo11n_kitti.pt").name == "yolo11n_kitti.onnx"
    assert onnx_path_for("a/b/c.pt").parent.as_posix() == "a/b"


def test_opset_is_pinned_low_enough_for_tensorrt():
    # TensorRT trails ONNX Runtime on opset support. Raising this without checking the
    # target runtime is how an export starts failing only at deployment time.
    assert DEFAULT_OPSET <= 17


def test_export_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_onnx(tmp_path / "nope.pt")


def _tiny_model(path, opset=17):
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 7, 8400])
    node = helper.make_node("Identity", ["images"], ["output0"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=9
    )
    onnx.save(model, str(path))


def test_describe_reads_shapes_and_opset(tmp_path):
    path = tmp_path / "m.onnx"
    _tiny_model(path)
    info = describe_onnx(path)
    assert info["opset"] == 17
    assert info["inputs"]["images"] == [1, 3, 640, 640]
    assert info["outputs"]["output0"] == [1, 7, 8400]
    assert info["size_mb"] >= 0


def test_describe_flags_dynamic_axes(tmp_path):
    # A dynamic dimension comes back as its symbolic name rather than a number, which
    # is what makes an accidentally dynamic export visible before TensorRT sees it.
    path = tmp_path / "dyn.onnx"
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [None, 3, 640, 640])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [None, 7, 8400])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])], "g", [x], [y]
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9),
        str(path),
    )
    info = describe_onnx(path)
    assert info["inputs"]["images"][0] != 1  # not a fixed batch of one
