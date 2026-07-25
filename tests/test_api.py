"""API tests. A stub detector is injected so the endpoints run without a model file or
onnxruntime, which keeps them in the fast suite. The wiring, image decoding, error
handling and response shape are what get checked here."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.service import app, get_detector
from drive_perception.detector import Detection


class StubDetector:
    names = {0: "car", 1: "pedestrian", 2: "cyclist"}

    def predict(self, image):
        # Two fixed detections, independent of the pixels, so assertions are stable.
        return [
            Detection(box=(10.0, 20.0, 30.0, 40.0), score=0.9, cls_id=0, cls_name="car"),
            Detection(box=(5.0, 5.0, 15.0, 25.0), score=0.5, cls_id=2, cls_name="cyclist"),
        ]


@pytest.fixture
def client():
    app.dependency_overrides[get_detector] = lambda: StubDetector()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _png_bytes(w=64, h=48):
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    return buf.tobytes()


def test_root_lists_endpoints_without_loading_a_model():
    # No dependency override here, proving root never touches the detector.
    resp = TestClient(app).get("/")
    assert resp.status_code == 200
    assert "/detect" in str(resp.json()["endpoints"])


def test_detect_returns_json_detections(client):
    resp = client.post("/detect", files={"file": ("frame.png", _png_bytes(64, 48), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["image_width"] == 64 and body["image_height"] == 48
    assert body["detections"][0]["cls_name"] == "car"
    assert body["detections"][0]["box"] == [10.0, 20.0, 30.0, 40.0]


def test_detect_rejects_a_non_image(client):
    resp = client.post("/detect", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_detect_rejects_empty_body(client):
    resp = client.post("/detect", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_detect_requires_a_file(client):
    # No file field at all is a validation error from FastAPI.
    assert client.post("/detect").status_code == 422


def test_annotated_returns_a_jpeg(client):
    resp = client.post(
        "/detect/annotated", files={"file": ("frame.png", _png_bytes(80, 60), "image/png")}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    # Round-trips as a decodable image of the same size.
    decoded = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (60, 80)
