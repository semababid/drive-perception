# Tracking metrics on KITTI

Detector `models/yolo11n_kitti.pt`, tracker `bytetrack`, sequences 0000, 0001, 0002, matched at IoU 0.5.

| class | MOTA | IDF1 | ID switches | FP | misses | frag | objects |
|---|---|---|---|---|---|---|---|
| car | 0.782 | 0.720 | 70.0 | 201.0 | 590.0 | 70.0 | 3956.0 |
| pedestrian | 0.369 | 0.543 | 11.0 | 63.0 | 124.0 | 16.0 | 314.0 |
| cyclist | 0.467 | 0.688 | 2.0 | 39.0 | 81.0 | 4.0 | 229.0 |
| OVERALL | 0.738 | 0.706 | 83.0 | 303.0 | 795.0 | 90.0 | 4499.0 |

## How to read these

MOTA combines misses, false positives and identity switches into a single number, so it drops for any of the three. IDF1 asks a narrower question: how consistently was one real object given one identity. A tracker that finds everything but keeps renaming it scores well on MOTA and badly on IDF1, which is why both appear here.

Identity switches are reported raw because they are the cost a driving stack actually feels. Every switch is a moment where whatever was following that object lost its history and started over.

Ground-truth boxes marked DontCare, and the neighbouring classes the benchmark treats as ambiguous (Van against car, seated people against pedestrian), are excluded rather than counted as errors.
