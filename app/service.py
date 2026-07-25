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
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from drive_perception.online_tracker import SimpleTracker
from drive_perception.onnx_detector import OnnxDetector
from drive_perception.viz import draw_detections

DEFAULT_WEIGHTS = "models/yolo11n_kitti.onnx"

# Upload ceilings, so a huge or empty body is refused before any decode work. A single
# frame is small; a clip is allowed more room but is still bounded.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_TRACK_FRAMES = 300  # a clip longer than this is truncated rather than run unbounded


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


class FrameTracks(BaseModel):
    frame: int
    tracks: list[dict]


class TrackResponse(BaseModel):
    frames: int
    unique_tracks: int
    per_frame: list[FrameTracks]


class HealthResponse(BaseModel):
    status: str
    classes: dict[int, str]
    input_height: int
    input_width: int


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


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413, detail=f"file too large; limit is {max_bytes // 1024 // 1024} MB"
        )
    return data


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return image


@app.get("/health", response_model=HealthResponse)
def health(detector: OnnxDetector = Depends(get_detector)) -> HealthResponse:
    """Report that the model loaded and what it detects."""
    height, width = detector.imgsz
    return HealthResponse(
        status="ok", classes=detector.names, input_height=height, input_width=width
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    detector: OnnxDetector = Depends(get_detector),
) -> DetectResponse:
    """Detect objects in one uploaded image and return them as JSON."""
    image = _decode(await _read_limited(file, MAX_IMAGE_BYTES))
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
    image = _decode(await _read_limited(file, MAX_IMAGE_BYTES))
    annotated = draw_detections(image, detector.predict(image))
    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode result")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/track", response_model=TrackResponse)
async def track(
    file: UploadFile = File(...),
    detector: OnnxDetector = Depends(get_detector),
) -> TrackResponse:
    """Detect and track objects through an uploaded video, returning per-frame tracks.

    The clip is written to a temporary file because OpenCV reads video from a path, not
    from memory, and the file is removed once the frames have been read."""
    data = await _read_limited(file, MAX_VIDEO_BYTES)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        capture = cv2.VideoCapture(tmp.name)
        tracker = SimpleTracker()
        per_frame: list[FrameTracks] = []
        seen: set[int] = set()
        index = 0
        while index < MAX_TRACK_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            tracks = tracker.update(detector.predict(frame))
            seen.update(t.track_id for t in tracks)
            per_frame.append(
                FrameTracks(
                    frame=index,
                    tracks=[
                        {
                            "track_id": t.track_id,
                            "cls_name": t.cls_name,
                            "score": round(t.score, 3),
                            "box": [round(v, 1) for v in t.box],
                        }
                        for t in tracks
                    ],
                )
            )
            index += 1
        capture.release()
    finally:
        os.unlink(tmp.name)

    if index == 0:
        raise HTTPException(status_code=400, detail="could not read any frames from the video")
    return TrackResponse(frames=index, unique_tracks=len(seen), per_frame=per_frame)
