# drive-perception

Real-time multi-object detection and tracking for driving scenes. A YOLO11 detector
feeds a ByteTrack tracker to keep stable object IDs across frames on the KITTI
benchmark. The detector is then exported to ONNX and timed on three inference backends
(CPU, CoreML, and TensorRT), so the speed and accuracy trade-off is measured instead of
guessed.

Two detector sizes run through the whole project. `yolo11n` is the fast edge target and
`yolo11s` is the accuracy anchor. Measuring both gives an accuracy-versus-latency curve
rather than a single data point.

> **Status:** early build. [docs/ROADMAP.md](docs/ROADMAP.md) tracks what is done and
> what comes next.

## Why this project

Driving-scene perception covers detection, tracking, and the deployment work needed to
run a model fast on limited hardware. That combination sits at the core of most
automotive and robotics computer-vision roles. This repo walks the full path: a model
that trains, exports, quantises, reports honest numbers, serves over an API, and runs in
a live demo.

## Planned stack

- Detector: YOLO11 (`n` and `s`), fine-tuned on KITTI car, pedestrian, and cyclist
- Tracking: ByteTrack, with BoT-SORT as a comparison
- Dataset: KITTI 2D object detection and multi-object tracking benchmarks
- Runtime: ONNX Runtime for portability, TensorRT FP16 and INT8 for the edge numbers
- Serving: FastAPI and Docker
- Demo: Streamlit with a live FPS counter

## Metrics reported

- Detection: mAP@0.5 and per-class AP for car, pedestrian, and cyclist
- Tracking: MOTA, IDF1, and ID-switches
- Deployment: latency, FPS, and model size on every backend

## Development

```bash
uv pip install -e ".[dev]"
ruff check src tests scripts
pytest -m "not slow" -q
```

## License

MIT. See [LICENSE](LICENSE).
