# Tracking metrics on KITTI

Detector `yolo11n.pt`, tracker `bytetrack`, sequences 0000, 0001, 0002, matched at IoU 0.5.

| class | MOTA | IDF1 | ID switches | FP | misses | frag | objects |
|---|---|---|---|---|---|---|---|
| car | 0.459 | 0.542 | 83.0 | 270.0 | 1787.0 | 100.0 | 3956.0 |
| pedestrian | -0.277 | 0.380 | 2.0 | 212.0 | 187.0 | 7.0 | 314.0 |
| cyclist | -0.441 | 0.083 | 0.0 | 116.0 | 214.0 | 0.0 | 229.0 |
| OVERALL | 0.362 | 0.506 | 85.0 | 598.0 | 2188.0 | 107.0 | 4499.0 |

## How to read these

MOTA combines misses, false positives and identity switches into a single number, so it drops for any of the three. IDF1 asks a narrower question: how consistently was one real object given one identity. A tracker that finds everything but keeps renaming it scores well on MOTA and badly on IDF1, which is why both appear here.

Identity switches are reported raw because they are the cost a driving stack actually feels. Every switch is a moment where whatever was following that object lost its history and started over.

Ground-truth boxes marked DontCare, and the neighbouring classes the benchmark treats as ambiguous (Van against car, seated people against pedestrian), are excluded rather than counted as errors.
