"""A detector that runs the exported ONNX graph directly, without PyTorch.

Ultralytics can load an ONNX file, but it pins the session to the CPU provider and
still expects torch to be installed. Neither suits a deployment target. This module
owns the whole path instead: session creation, letterboxing, decoding and non-maximum
suppression, using only onnxruntime, numpy and OpenCV.

That independence is the point. It makes the accelerated providers reachable, which is
what the latency comparison needs, and it is the shape a real edge service takes, where
shipping a training framework to run inference would be absurd.

Correctness is not assumed. The parity check compares this path against PyTorch on real
frames, and the class names are read out of the graph rather than hardcoded, so a model
trained on different classes cannot be silently mislabelled.
"""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection, build_detections

PAD_VALUE = 114  # the grey Ultralytics letterboxes with, matched so scores line up


def letterbox(
    image: np.ndarray, target: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize into `target` preserving aspect ratio, padding the remainder.

    Returns the padded image, the scale applied, and the padding added on the left and
    top. Both are needed afterwards: predictions come back in padded coordinates and
    have to be walked back to the original frame."""
    height, width = target
    src_h, src_w = image.shape[:2]
    scale = min(height / src_h, width / src_w)
    new_w, new_h = round(src_w * scale), round(src_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (width - new_w) // 2, (height - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression over xyxy boxes, highest score first."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[best], x1[rest])
        iy1 = np.maximum(y1[best], y1[rest])
        ix2 = np.minimum(x2[best], x2[rest])
        iy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou < iou_threshold]
    return keep


def decode(
    output: np.ndarray,
    conf: float,
    iou: float,
    scale: float,
    pad: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the raw graph output into boxes in original image coordinates.

    The graph emits one column per anchor, shaped (1, 4 + classes, anchors), with box
    centres and sizes in the padded image. There is no objectness column in this model
    family: the class score is the confidence."""
    predictions = output[0].T  # (anchors, 4 + classes)
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    keep = scores >= conf
    if not keep.any():
        empty = np.empty((0,), dtype=np.float32)
        return np.empty((0, 4), dtype=np.float32), empty, empty.astype(int)

    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    cx, cy, w, h = boxes_cxcywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Undo the letterbox: remove the padding, then the scaling.
    pad_x, pad_y = pad
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale

    src_h, src_w = shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, src_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, src_h)

    # Suppress per class by shifting each class into its own coordinate band, so two
    # different classes overlapping the same object never suppress one another.
    offsets = class_ids.astype(np.float32) * (max(src_h, src_w) + 1.0)
    kept = nms(boxes + offsets[:, None], scores, iou)
    return boxes[kept], scores[kept], class_ids[kept]


class OnnxDetector:
    """Runs the exported graph under ONNX Runtime and returns the usual Detections."""

    def __init__(
        self,
        onnx_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.7,
        providers: list[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"{onnx_path} missing. Run scripts/export_onnx.py first.")

        available = ort.get_available_providers()
        requested = providers or ["CPUExecutionProvider"]
        missing = [p for p in requested if p not in available]
        if missing:
            raise ValueError(f"providers not available: {missing}. Have: {available}")

        self.session = ort.InferenceSession(str(onnx_path), providers=requested)
        self.input_name = self.session.get_inputs()[0].name
        _, _, height, width = self.session.get_inputs()[0].shape
        self.imgsz = (int(height), int(width))
        self.conf = conf
        self.iou = iou

        # Ultralytics stores the class names in the graph metadata. Reading them keeps
        # this backend honest for any model, rather than assuming the KITTI three.
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = (
            {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
            if "names" in meta
            else {}
        )

    @property
    def backend(self) -> str:
        return self.session.get_providers()[0]

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        image = cv2.imread(str(source)) if not isinstance(source, np.ndarray) else source
        if image is None:
            raise FileNotFoundError(f"could not read image: {source}")

        padded, scale, pad = letterbox(image, self.imgsz)
        chw = padded[:, :, ::-1].transpose(2, 0, 1)
        batch = np.ascontiguousarray(chw, dtype=np.float32)[None] / 255.0

        output = self.session.run(None, {self.input_name: batch})[0]
        boxes, scores, class_ids = decode(
            output, self.conf, self.iou, scale, pad, image.shape[:2]
        )
        return build_detections(
            xyxy=boxes.tolist(),
            scores=scores.tolist(),
            cls_ids=class_ids.tolist(),
            names=self.names,
        )
