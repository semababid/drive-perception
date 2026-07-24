"""Export the trained detector to ONNX.

ONNX is the step that takes the model out of PyTorch. Once the graph is in ONNX it can
run under ONNX Runtime on any machine, or be compiled by TensorRT into an engine for a
specific GPU, without the training framework being installed at all.

Two choices here are deliberate rather than defaults.

The opset is pinned. Exporters happily emit the newest opset they know, and TensorRT
consistently supports fewer opsets than ONNX Runtime does, so an unpinned export is the
classic failure that works on the development machine and is rejected by the edge
runtime much later. Pinning it means the file that passes the parity check is the same
file the benchmark compiles.

Shapes are static by default. A fixed input lets TensorRT specialise the whole graph,
which is both faster to build and faster to run. Dynamic shapes stay available for cases
that genuinely need a variable batch, at a cost in both.
"""

from __future__ import annotations

from pathlib import Path

# TensorRT is the constraint on this number, not ONNX Runtime. Opset 17 is broadly
# supported by both, which keeps one exported file usable across every backend the
# benchmark measures.
DEFAULT_OPSET = 17


def onnx_path_for(weights: str | Path) -> Path:
    """Where the exporter writes the ONNX file for a given checkpoint."""
    return Path(weights).with_suffix(".onnx")


# KITTI frames are roughly 1242 by 375, an aspect ratio near 3.3 to 1. PyTorch inference
# letterboxes to that shape and pads only up to a stride multiple, so the network
# actually sees 224 by 640. Exporting a square 640 by 640 input instead pads the frame
# with large empty bars, changes the apparent size of every object, and costs 2.9 times
# the pixels for the privilege. Exporting at the shape the model is really used at keeps
# the runtime honest and the compute down.
KITTI_IMGSZ = (224, 640)  # height, width


def export_onnx(
    weights: str | Path,
    imgsz: int | tuple[int, int] = KITTI_IMGSZ,
    opset: int = DEFAULT_OPSET,
    dynamic: bool = False,
    simplify: bool = True,
) -> Path:
    """Export a checkpoint to ONNX and return the written path.

    `imgsz` accepts a single number for a square input or a (height, width) pair. The
    pair is the right choice for driving footage, which is far wider than it is tall."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing. Run scripts/finetune.py first.")

    YOLO(str(weights)).export(
        format="onnx",
        imgsz=list(imgsz) if isinstance(imgsz, tuple) else imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
    )
    out = onnx_path_for(weights)
    if not out.exists():
        raise RuntimeError(f"export reported success but {out} is missing")
    return out


def preprocess(image_bgr, imgsz: int | tuple[int, int] = 640):
    """Turn a BGR image into the exact tensor both backends expect.

    Both runtimes must be fed byte-identical input, otherwise a parity failure could
    just as easily be a preprocessing difference, and the comparison proves nothing."""
    import cv2
    import numpy as np

    height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    resized = cv2.resize(image_bgr, (width, height))
    chw = resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    tensor = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    return tensor[None]  # add the batch axis


def raw_parity(
    weights: str | Path,
    onnx_path: str | Path,
    imgsz: int | tuple[int, int] = KITTI_IMGSZ,
    seed: int = 0,
) -> dict:
    """Compare the raw tensors PyTorch and ONNX Runtime produce for one input.

    This is the strict check. It runs before non-maximum suppression, so nothing is
    rounded or discarded and a small numerical drift cannot hide behind a threshold."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    batch = preprocess(image, (height, width))

    module = YOLO(str(weights)).model.float().eval()
    with torch.no_grad():
        torch_out = module(torch.from_numpy(batch))
    if isinstance(torch_out, list | tuple):
        torch_out = torch_out[0]
    torch_array = torch_out.cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_array = session.run(None, {session.get_inputs()[0].name: batch})[0]

    if torch_array.shape != onnx_array.shape:
        raise AssertionError(
            f"shape mismatch: torch {torch_array.shape} vs onnx {onnx_array.shape}"
        )
    diff = np.abs(torch_array - onnx_array)
    return {
        "shape": tuple(torch_array.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def detection_parity(
    weights: str | Path,
    onnx_path: str | Path,
    images: list,
    conf: float = 0.25,
    imgsz: int | tuple[int, int] = KITTI_IMGSZ,
) -> dict:
    """Compare the detections each backend reports on real images.

    The raw check can pass while the detections still differ, because a tiny score
    difference either side of the confidence threshold changes how many boxes survive.
    This is the version a user would notice."""
    from .detector import Detector

    torch_det = Detector(weights=weights, conf=conf, imgsz=imgsz)
    onnx_det = Detector(weights=onnx_path, conf=conf, imgsz=imgsz)

    counts_match = 0
    worst_box = worst_score = 0.0
    class_mismatches = 0
    for image in images:
        a = torch_det.predict(str(image))
        b = onnx_det.predict(str(image))
        if len(a) != len(b):
            continue
        counts_match += 1
        for da, db in zip(a, b, strict=True):
            worst_box = max(worst_box, max(abs(x - y) for x, y in zip(da.box, db.box, strict=True)))
            worst_score = max(worst_score, abs(da.score - db.score))
            class_mismatches += int(da.cls_id != db.cls_id)
    return {
        "images": len(images),
        "same_detection_count": counts_match,
        "max_box_diff_px": round(worst_box, 4),
        "max_score_diff": round(worst_score, 6),
        "class_mismatches": class_mismatches,
    }


def describe_onnx(path: str | Path) -> dict:
    """Read back the graph metadata worth checking after an export.

    The input and output shapes are the part that matters. A wrong input shape or an
    unexpectedly dynamic axis will not fail here, it will fail much later inside the
    runtime with a far less obvious message."""
    import onnx

    path = Path(path)
    model = onnx.load(str(path))

    def shape_of(value) -> list:
        dims = []
        for d in value.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
        return dims

    return {
        "path": path,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "opset": max(i.version for i in model.opset_import),
        "ir_version": model.ir_version,
        "inputs": {v.name: shape_of(v) for v in model.graph.input},
        "outputs": {v.name: shape_of(v) for v in model.graph.output},
    }
