"""FastAPI service that runs the detector over HTTP.

The service loads the exported ONNX model through OnnxDetector, so it runs without
PyTorch or Ultralytics installed. That is the point of having built a standalone ONNX
path: the thing that serves predictions is small and has no training framework in it.

The detector is provided through a dependency rather than imported at module load, so a
test can swap in a stub and exercise the endpoints without a model file, and the real
model is loaded once and reused across requests.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from drive_perception.onnx_detector import OnnxDetector
from drive_perception.viz import draw_detections

DEFAULT_WEIGHTS = "models/yolo11n_kitti.onnx"


class DetectionOut(BaseModel):
    box: list[float]  # x1, y1, x2, y2 in pixels
    score: float
    cls_id: int
    cls_name: str


class DetectResponse(BaseModel):
    count: int
    image_width: int
    image_height: int
    detections: list[DetectionOut]


@lru_cache
def get_detector() -> OnnxDetector:
    """Load the ONNX detector once. The weights path can be overridden with the
    DRIVE_WEIGHTS environment variable, which the container image uses."""
    path = Path(os.environ.get("DRIVE_WEIGHTS", DEFAULT_WEIGHTS))
    return OnnxDetector(path, conf=0.25, iou=0.7)


app = FastAPI(
    title="drive-perception",
    summary="Real-time driving-scene object detection.",
    version="1.0",
)


@app.get("/")
def root() -> dict:
    """Service description, without touching the model."""
    return {
        "service": "drive-perception",
        "endpoints": {
            "POST /detect": "image file -> detections as JSON",
            "POST /detect/annotated": "image file -> annotated JPEG",
        },
    }


def _decode(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return image


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    detector: OnnxDetector = Depends(get_detector),
) -> DetectResponse:
    """Detect objects in one uploaded image and return them as JSON."""
    image = _decode(await file.read())
    detections = detector.predict(image)
    height, width = image.shape[:2]
    return DetectResponse(
        count=len(detections),
        image_width=width,
        image_height=height,
        detections=[
            DetectionOut(
                box=list(d.box), score=d.score, cls_id=d.cls_id, cls_name=d.cls_name
            )
            for d in detections
        ],
    )


@app.post("/detect/annotated")
async def detect_annotated(
    file: UploadFile = File(...),
    detector: OnnxDetector = Depends(get_detector),
) -> Response:
    """Detect objects and return the image with the boxes drawn on it."""
    image = _decode(await file.read())
    annotated = draw_detections(image, detector.predict(image))
    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode result")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
