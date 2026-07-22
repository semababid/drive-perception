# drive-perception

Real-time multi-object detection and tracking for driving scenes. A YOLO11 detector
feeds a ByteTrack tracker to produce stable object IDs across frames on the KITTI
benchmark, then the detector is exported to ONNX and measured across inference
backends — CPU, CoreML and TensorRT — so the speed/accuracy trade-off is explicit
rather than assumed.

The project carries two detector sizes end to end: `yolo11n` as the fast edge target
and `yolo11s` as the accuracy anchor. Reporting both turns the benchmark into a curve
a perception engineer can actually read.

> **Status:** early build. The roadmap in [docs/ROADMAP.md](docs/ROADMAP.md) tracks
> what is done and what is next.

## Why this project

Driving-scene perception — detection, tracking and the deployment work to run it fast
on constrained hardware — is the core of automotive and robotics computer-vision work.
This repo is built to show the full path: not just a model that trains, but one that
exports, quantises, benchmarks honestly, serves over an API and runs in a live demo.

## Planned stack

- **Detector:** YOLO11 (`n` + `s`), fine-tuned on KITTI car / pedestrian / cyclist
- **Tracking:** ByteTrack, with BoT-SORT as a comparison
- **Dataset:** KITTI 2D object detection + multi-object tracking benchmarks
- **Runtime:** ONNX Runtime (portable), TensorRT FP16/INT8 (edge, benchmarked on Colab)
- **Serving:** FastAPI + Docker
- **Demo:** Streamlit with a live FPS counter

## Metrics reported

- Detection: mAP@0.5 and per-class AP (car / pedestrian / cyclist)
- Tracking: MOTA, IDF1, ID-switches
- Deployment: latency / FPS and model size across every backend

## Development

```bash
uv pip install -e ".[dev]"
ruff check src tests scripts
pytest -m "not slow" -q
```

## License

MIT — see [LICENSE](LICENSE).
