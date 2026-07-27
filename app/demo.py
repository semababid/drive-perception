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
SAMPLE_IMAGE = Path(__file__).parent / "samples" / "sample_street.jpg"
MAX_FRAMES = 300

# Box colours from the visualisation palette, reused in the UI so the legend chips match
# what gets drawn on the frames.
CLASS_COLORS = {"car": "#3296ff", "pedestrian": "#ffaa28", "cyclist": "#c83cc8"}

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1080px;}
h1 {font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.1rem;}
.subtitle {color: #9aa4b2; font-size: 1.02rem; margin-bottom: 1.1rem;}
.chips {display: flex; gap: 8px; flex-wrap: wrap; margin: 0.2rem 0 1.4rem;}
.chip {background: #171b26; border: 1px solid #262c3a; border-radius: 999px;
       padding: 5px 13px; font-size: 0.82rem; color: #c7cedb;}
.chip b {color: #e6e9ef;}
.dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px;}
[data-testid="stMetric"] {background: #141824; border: 1px solid #242b3a;
       border-radius: 14px; padding: 14px 18px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
[data-testid="stImage"] img {border-radius: 12px; border: 1px solid #242b3a;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
</style>
"""


@st.cache_resource
def load_detector(conf: float) -> OnnxDetector:
    """Load the model once per confidence setting and reuse it across reruns."""
    return OnnxDetector(DEFAULT_WEIGHTS, conf=conf, iou=0.7)


def _to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr[:, :, ::-1]


def _chip(label: str, value: str, color: str | None = None) -> str:
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ""
    return f'<span class="chip">{dot}{label} <b>{value}</b></span>'


def header(detector: OnnxDetector) -> None:
    st.markdown("# drive-perception")
    st.markdown(
        '<div class="subtitle">Real-time driving-scene detection and tracking, '
        "running the exported ONNX model.</div>",
        unsafe_allow_html=True,
    )
    chips = [
        _chip("backend", "ONNX Runtime"),
        _chip("input", "x".join(str(v) for v in detector.imgsz)),
        *[_chip("", name, CLASS_COLORS.get(name)) for name in detector.names.values()],
    ]
    st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)


def run_image(detector: OnnxDetector) -> None:
    left, right = st.columns([3, 1])
    upload = left.file_uploader(
        "Upload a street scene", type=["png", "jpg", "jpeg"], key="image"
    )
    right.write("")
    right.write("")
    use_sample = right.button(
        "Try an example", use_container_width=True, disabled=not SAMPLE_IMAGE.exists()
    )

    if upload is not None:
        st.session_state.image_bytes = upload.getvalue()
    elif use_sample:
        st.session_state.image_bytes = SAMPLE_IMAGE.read_bytes()

    data = st.session_state.get("image_bytes")
    if not data:
        st.info("Upload a street scene, or click Try an example.")
        return

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Could not read that file as an image.")
        return

    start = time.perf_counter()
    detections = detector.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    st.image(_to_rgb(draw_detections(image, detections)), use_container_width=True)

    counts: dict[str, int] = {}
    for d in detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    metrics = st.columns(4)
    metrics[0].metric("detections", len(detections))
    metrics[1].metric("inference", f"{elapsed_ms:.0f} ms")
    metrics[2].metric("throughput", f"{1000 / elapsed_ms:.0f} FPS" if elapsed_ms else "n/a")
    metrics[3].metric("classes seen", len(counts))


def run_video(detector: OnnxDetector) -> None:
    upload = st.file_uploader("Upload a short clip", type=["mp4", "mov", "avi"], key="video")
    if upload is None:
        st.info("Upload a short clip to watch the tracker hold ids across frames.")
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
            drawn = draw_hud(annotate(frame, tracks, trails), [f"frame {index + 1}"])
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
    st.set_page_config(page_title="drive-perception", page_icon="🚗", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    if not Path(DEFAULT_WEIGHTS).exists():
        st.error(f"Model not found at {DEFAULT_WEIGHTS}. Run scripts/export_onnx.py first.")
        return

    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    st.sidebar.caption("Lower shows more, at the cost of false positives.")
    detector = load_detector(conf)

    header(detector)
    image_tab, video_tab = st.tabs(["  Image  ", "  Video  "])
    with image_tab:
        run_image(detector)
    with video_tab:
        run_video(detector)


if __name__ == "__main__":
    main()
