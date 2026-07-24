"""A thin wrapper around YOLO11 that hands back plain `Detection` objects instead of
framework-specific result tensors.

Callers depend on this interface, not on Ultralytics. That is deliberate: when the
ONNX runtime backend arrives later, it produces the same `Detection` list from the same
`predict` call, and nothing downstream (visualization, tracking, the API) has to change.

The heavy torch and ultralytics imports are loaded lazily inside `__init__`, so the
data-pipeline tests and the box-mapping test below run without pulling in torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float
    cls_id: int
    cls_name: str


def build_detections(
    xyxy: list,
    scores: list,
    cls_ids: list,
    names: dict[int, str],
) -> list[Detection]:
    """Map raw model outputs to Detection objects, sorted by score (high to low).

    Pure and backend-independent on purpose: both the torch path and the future ONNX
    path funnel their outputs through here, and it is the part worth unit-testing."""
    dets = [
        Detection(
            box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            score=float(s),
            cls_id=int(c),
            cls_name=names[int(c)],
        )
        for b, s, c in zip(xyxy, scores, cls_ids, strict=True)
    ]
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets


class Detector:
    """YOLO11 detector. Point it at COCO-pretrained weights (`yolo11n.pt`) for a zero-shot
    baseline, or at the fine-tuned KITTI weights once those exist."""

    def __init__(
        self,
        weights: str | Path = "yolo11n.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | tuple[int, int] = 640,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names: dict[int, str] = self.model.names
        self.conf = conf
        self.iou = iou
        # A pair is passed straight through. Driving frames are far wider than they
        # are tall, and forcing them square wastes most of the input on blank padding.
        self.imgsz = list(imgsz) if isinstance(imgsz, tuple) else imgsz
        # None lets Ultralytics pick. The benchmark sets it explicitly to compare the
        # same model on cpu against mps, which is not a comparison auto-select allows.
        self.device = device

    def predict(self, source: str | Path | np.ndarray) -> list[Detection]:
        """Run detection on one image (path, array, or PIL image) and return Detections."""
        result = self.model.predict(
            source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        return build_detections(
            xyxy=boxes.xyxy.tolist(),
            scores=boxes.conf.tolist(),
            cls_ids=boxes.cls.tolist(),
            names=self.names,
        )
