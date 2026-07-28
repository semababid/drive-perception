# Build roadmap

**Status: complete through #32 (v1.0).** Phases A to G are done: the data pipeline,
detection and fine-tuning, tracking, the deployment and benchmark work, the service and
container, the demo, and the docs. Phase H (depth and bird's-eye view) is optional and
not started.

The project ships as a sequence of small, self-contained commits grouped into eight
phases. Each numbered item is one commit. The milestone checkpoints mark points where
the project is already worth showing even if the rest is unfinished.

The detector backbone is YOLO11. Two sizes run through the whole pipeline: `yolo11n` is
the fast edge and deploy target, and `yolo11s` is the accuracy anchor. The benchmark
reports both, so the results read as an accuracy-versus-latency curve rather than a
single point.

## Phase A: Foundation
1. `chore: git init, gitignore, MIT license`
2. `chore: add pyproject and dependencies`
3. `chore: ruff and pytest tooling config`
4. `chore: folder skeleton and config loader`
5. `ci: lint and smoke-test workflow`

## Phase B: Data
6. `feat: KITTI download script with subset flag`
7. `feat: KITTI EDA with difficulty tiers and box-size analysis`
8. `feat: KITTI to YOLO label converter`
9. `feat: train/val split and data.yaml`

## Phase C: Detection
10. `feat: detector wrapper with pretrained inference`
11. `feat: detection visualization module`
12. `feat: baseline mAP evaluation on KITTI val`
13. `feat: fine-tune detector on 3 KITTI classes`
14. `feat: post-finetune eval and before/after report`

## Phase D: Tracking
15. `feat: download tracking sequences subset`
16. `feat: ByteTrack tracker wrapper`
17. `feat: detect-track-annotate pipeline`
18. `feat: render annotated tracking video`
19. `feat: tracking metrics for MOTA, IDF1, ID-switches`
20. `feat: BoT-SORT comparison row`

**Milestone 1 (after #20):** a complete tracked-detection pipeline with metrics.

## Phase E: Optimization and runtime
21. `feat: ONNX export with pinned opset`
22. `test: torch-vs-ONNX parity check`
23. `feat: ONNXRuntime backend in detector`
24. `feat: CPU/CoreML latency benchmark`
25. `feat: TensorRT FP16 benchmark notebook (Colab T4)`
26. `feat: INT8 quantization with calibration set`
27. `docs: final latency/accuracy/size table and charts`

**Milestone 2 (after #27):** the deployment and edge story.

## Phase F: Serving
28. `feat: FastAPI /detect endpoint`
29. `feat: FastAPI /track endpoint with validation and health check`
30. `feat: Dockerfile and docker-compose`

## Phase G: Demo and docs
31. `feat: Streamlit demo with live FPS and controls`
32. `docs: design.md, README, results, limitations`

**Milestone 3 (after #32):** polished, hire-ready v1.0.

## Phase H: Stretch (optional)
33. `feat: monocular depth estimation (Depth Anything)`
34. `feat: bird's-eye-view overlay`
