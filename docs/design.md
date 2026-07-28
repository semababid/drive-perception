# Design notes

This document records the decisions behind drive-perception and why they were made. The
README covers what the project does; this covers why it is built the way it is.

## The problem

Detect and track cars, pedestrians and cyclists in driving footage, then run the result
fast enough and small enough to deploy. KITTI is the dataset because it is the standard
driving benchmark, it comes from a car-mounted camera in real traffic, and its
Easy/Moderate/Hard difficulty protocol gives a principled way to report where a model
fails rather than hiding it behind a single average.

## Pipeline

```
KITTI frames
   -> label conversion (KITTI boxes to normalised YOLO)
   -> train/val split
   -> fine-tune YOLO11 (car, pedestrian, cyclist)
   -> evaluate (mAP with KITTI ignore rules)
   -> export to ONNX
   -> benchmark (CPU, CoreML, TensorRT FP32/FP16/INT8)
   -> serve (FastAPI + Docker) and demo (Streamlit)
```

Detection and tracking are separate concerns. Detection answers what is in a frame;
tracking answers whether the thing in this frame is the same one as last frame, which is
what a driving stack actually needs for counting and for reacting to an object over time.

## Decisions

### Two detector sizes, carried the whole way

`yolo11n` is the fast edge target and `yolo11s` is the accuracy anchor. Reporting both
turns the benchmark into an accuracy-versus-latency curve rather than a single point,
which is the shape of question a deployment actually asks: how much accuracy does the
faster model give up, and is that trade worth it on the target hardware.

### Rectangular export, not square

KITTI frames are about 1242 by 375, a ratio near 3.3 to 1. PyTorch runs inference on a
letterboxed 224 by 640 tensor, but the naive ONNX export defaults to a square 640 by 640,
which pads the frame with large empty bars, changes the apparent size of every object,
and costs 2.9 times the pixels. The parity check caught this: raw tensors matched, but
the detections disagreed on half the images. Exporting at the real aspect ratio fixed the
correctness and cut the compute at the same time.

### A serving path with no training framework

The service and the demo run the exported ONNX graph through a detector written from
scratch: letterboxing, the forward pass through onnxruntime, decoding and non-maximum
suppression, with only numpy and OpenCV behind it. This is more code than calling
Ultralytics, but it is the difference between a 587 MB image and a multi-gigabyte one,
and it is what a real edge service looks like. Shipping a training framework to run
inference would be the wrong dependency.

### Two trackers for two jobs

The offline evaluation uses ByteTrack and BoT-SORT through Ultralytics, because the point
there is to measure tracking quality properly, including a comparison between the two.
The online service uses a small IoU tracker with no motion model, because the point there
is to serve predictions without pulling PyTorch into the image. They are different tools
for different jobs, and keeping them separate is deliberate.

### Metrics computed in-house

Detection AP and the tracking metrics are implemented in the project rather than taken
from a library. The zero-shot baseline uses an 80-class COCO model against three KITTI
classes, which the built-in validator cannot score, and the baseline and fine-tuned model
have to be compared on identical maths. The AP implementation was cross-checked against
Ultralytics' own on the same data and agreed to within 0.008 mAP, which is the confidence
that the numbers are right.

## Evaluation methodology

- **Detection:** all-point-interpolation AP at IoU 0.5, per class, broken down by KITTI's
  Easy/Moderate/Hard tiers. Ground-truth boxes outside the selected tier become ignore
  regions rather than disappearing, so finding an object the benchmark chose not to score
  is neither rewarded nor punished.
- **Tracking:** MOTA, IDF1 and raw identity switches via motmetrics, with KITTI's DontCare
  regions and neighbouring classes (a Van scored against Car, a seated person against
  Pedestrian) excluded rather than counted as errors.
- **Latency:** end to end per frame, median over 100 frames, so preprocessing and NMS are
  included and a single slow frame does not skew the number.
- **INT8 accuracy:** measured on a held-out set disjoint from the calibration set, because
  calibrating and scoring on the same images would tune the quantisation to the test set.

## Deployment targets

Latency is reported on two machines and never merged into one ranking. The CPU, MPS and
CoreML numbers are from an Apple laptop; the TensorRT numbers are from a Colab T4. CoreML
is the best local option and roughly doubles the ONNX CPU rate. TensorRT FP16 is the
fastest overall and the setting to ship on NVIDIA. INT8 was measured and dropped: at this
model size there is no matmul-bound work for it to accelerate.

## Limitations and future work

- Hard-tier accuracy is the weak point: distant, occluded traffic. Higher input
  resolution or a small-object-focused augmentation strategy would be the next thing to
  try.
- The online IoU tracker fragments identities under fast motion. A motion model would
  help, at the cost of the dependency-free design.
- KITTI is one city in daylight. Night, rain and other domains are untested, and a real
  deployment would need far more varied data.
- The INT8 result is specific to a small model on a T4. A larger model or an
  INT8-optimised accelerator could change the conclusion, which is exactly why it is
  reported as a measurement rather than a rule.
