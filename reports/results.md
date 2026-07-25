# Results

Two detectors carried through the whole project: yolo11n as the fast edge target and yolo11s as the accuracy anchor. Both are fine-tuned on KITTI, then measured for accuracy and for latency across every backend the project can reach.

## Detection accuracy

Fine-tuned on the KITTI training split, scored on all 1496 validation frames at IoU 0.5 with the project's own evaluator, which applies KITTI's ignore rules.

| model | car | pedestrian | cyclist | mAP50 |
|---|---|---|---|---|
| yolo11n | 0.960 | 0.768 | 0.846 | **0.858** |
| yolo11s | 0.968 | 0.811 | 0.871 | **0.883** |

Per difficulty tier (mAP50):

| model | easy | moderate | hard |
|---|---|---|---|
| yolo11n | 0.955 | 0.748 | 0.481 |
| yolo11s | 0.964 | 0.809 | 0.507 |

## Inference latency

End to end per frame (preprocess, inference, postprocess), median over 100 frames at the deployed 224x640 input. The two groups are different machines and are not directly comparable across the line; each shows the best a given target reaches.

Apple M-series laptop:

| backend | yolo11n FPS | yolo11s FPS |
|---|---|---|
| torch-cpu | 58.1 | 38.3 |
| torch-mps | 64.4 | 60.1 |
| onnx-cpu | 65.6 | 31.3 |
| onnx-coreml | 127.4 | 109.7 |

NVIDIA Tesla T4 (Colab):

| backend | yolo11n FPS | yolo11s FPS |
|---|---|---|
| tensorrt-fp32 | 171.4 | 165.5 |
| tensorrt-fp16 | 217.4 | 227.1 |

## What the two models trade

yolo11s buys about four points of mAP50 over yolo11n. On the CoreML target that costs roughly 15 percent of the frame rate, and both models still clear 100 FPS, so the accuracy model is close to free here. On the T4 the larger model is actually the faster of the two at FP16, because the tiny model leaves the tensor cores idle. Which model to ship depends on the target, and both are viable.

## INT8: measured, and not adopted

Quantised to INT8 on the Tesla T4, calibrated on training frames and scored on 128 held-out frames, at a square 640 input (an engine can only be validated on square images). The point of this table is the change from FP16, measured the same way for both.

| model | precision | mAP50 | median ms | FPS |
|---|---|---|---|---|
| yolo11n | fp16 | 0.846 | 5.98 | 167.1 |
| yolo11n | int8 | 0.805 | 5.98 | 167.2 |
| | *int8 vs fp16* | *-0.041* | | *1.00x* |
| yolo11s | fp16 | 0.879 | 6.66 | 150.3 |
| yolo11s | int8 | 0.862 | 7.31 | 136.7 |
| | *int8 vs fp16* | *-0.017* | | *0.91x* |

On this hardware INT8 was not worth it. It cost accuracy on both models and returned no speed on the small one and less speed on the larger one. These detectors are small enough that per-frame time is spent on letterboxing and non-maximum suppression rather than the matrix multiplies INT8 accelerates, so there is little for the INT8 path to win back. FP16 is the setting to ship.

## The short version

- Fine-tuning lifted mAP50 from 0.394 zero-shot to 0.858 for yolo11n and 0.883 for yolo11s, most of it on the pedestrian and cyclist classes.
- The exported ONNX matches PyTorch to sub-pixel box differences, so every latency number is a like-for-like comparison of the same detections.
- CoreML gives the biggest local win, roughly doubling the ONNX CPU frame rate.
- TensorRT FP16 is the fastest overall and the setting to deploy on NVIDIA; INT8 did not pay for itself at this model size.
