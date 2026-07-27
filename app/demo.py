"""Streamlit demo for the driving-scene detector.

The demo runs the same ONNX model and the same lightweight tracker as the service, so
what a viewer sees here is what the deployed API produces. Nothing about the model is
re-implemented for the browser; this file is the interface layer over the pieces the
rest of the project already built and tested.

Run it with:

    streamlit run app/demo.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from drive_perception.online_tracker import SimpleTracker
from drive_perception.onnx_detector import OnnxDetector
from drive_perception.pipeline import Trails, annotate
from drive_perception.viz import draw_detections, draw_hud

DEFAULT_WEIGHTS = os.environ.get("DRIVE_WEIGHTS", "models/yolo11n_kitti.onnx")
MAX_FRAMES = 300


@st.cache_resource
def load_detector(conf: float) -> OnnxDetector:
    """Load the model once per confidence setting and reuse it across reruns."""
    return OnnxDetector(DEFAULT_WEIGHTS, conf=conf, iou=0.7)


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr[:, :, ::-1]


def run_image(detector: OnnxDetector) -> None:
    upload = st.file_uploader("Image", type=["png", "jpg", "jpeg"], key="image")
    if upload is None:
        st.info("Upload a street scene to see the detections.")
        return

    image = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not read that file as an image.")
        return

    start = time.perf_counter()
    detections = detector.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    annotated = draw_detections(image, detections)
    st.image(_to_rgb(annotated), caption=f"{len(detections)} detections", use_container_width=True)

    counts: dict[str, int] = {}
    for d in detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    cols = st.columns(len(counts) + 1) if counts else st.columns(1)
    cols[0].metric("inference", f"{elapsed_ms:.0f} ms")
    for col, (name, n) in zip(cols[1:], counts.items(), strict=False):
        col.metric(name, n)


def run_video(detector: OnnxDetector) -> None:
    upload = st.file_uploader("Video", type=["mp4", "mov", "avi"], key="video")
    if upload is None:
        st.info("Upload a short clip to watch the tracker keep ids across frames.")
        return

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(upload.getvalue())
    tmp.close()

    frame_slot = st.empty()
    fps_slot, count_slot, id_slot = st.columns(3)
    tracker = SimpleTracker()
    trails = Trails()
    capture = cv2.VideoCapture(tmp.name)

    try:
        seen: set[int] = set()
        start = time.perf_counter()
        index = 0
        while index < MAX_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            tracks = tracker.update(detector.predict(frame))
            trails.update(tracks, index)
            seen.update(t.track_id for t in tracks)

            rate = (index + 1) / (time.perf_counter() - start)
            drawn = annotate(frame, tracks, trails)
            drawn = draw_hud(drawn, [f"frame {index + 1}", f"{rate:.1f} FPS"])
            frame_slot.image(_to_rgb(drawn), use_container_width=True)
            fps_slot.metric("FPS", f"{rate:.1f}")
            count_slot.metric("tracking now", len(tracks))
            id_slot.metric("unique ids", len(seen))
            index += 1
    finally:
        capture.release()
        os.unlink(tmp.name)

    st.success(f"Processed {index} frames, {len(seen)} unique tracks.")


def main() -> None:
    st.set_page_config(page_title="drive-perception", page_icon="🚗", layout="wide")
    st.title("drive-perception")
    st.caption(
        "Real-time driving-scene detection and tracking, running the exported ONNX model."
    )

    if not Path(DEFAULT_WEIGHTS).exists():
        st.error(f"Model not found at {DEFAULT_WEIGHTS}. Run scripts/export_onnx.py first.")
        return

    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    detector = load_detector(conf)
    st.sidebar.write("Classes:", ", ".join(detector.names.values()))
    st.sidebar.write("Input size:", "x".join(str(v) for v in detector.imgsz))

    image_tab, video_tab = st.tabs(["Image", "Video"])
    with image_tab:
        run_image(detector)
    with video_tab:
        run_video(detector)


if __name__ == "__main__":
    main()
