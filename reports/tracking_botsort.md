# Tracking metrics on KITTI

Detector `models/yolo11n_kitti.pt`, tracker `botsort`, sequences 0000, 0001, 0002, matched at IoU 0.5.

| class | MOTA | IDF1 | ID switches | FP | misses | frag | objects |
|---|---|---|---|---|---|---|---|
| car | 0.798 | 0.737 | 49.0 | 215.0 | 534.0 | 64.0 | 3956.0 |
| pedestrian | 0.510 | 0.585 | 11.0 | 60.0 | 83.0 | 17.0 | 314.0 |
| cyclist | 0.459 | 0.689 | 1.0 | 40.0 | 83.0 | 3.0 | 229.0 |
| OVERALL | 0.761 | 0.724 | 61.0 | 315.0 | 700.0 | 84.0 | 4499.0 |

## How to read these

MOTA combines misses, false positives and identity switches into a single number, so it drops for any of the three. IDF1 asks a narrower question: how consistently was one real object given one identity. A tracker that finds everything but keeps renaming it scores well on MOTA and badly on IDF1, which is why both appear here.

Identity switches are reported raw because they are the cost a driving stack actually feels. Every switch is a moment where whatever was following that object lost its history and started over.

Ground-truth boxes marked DontCare, and the neighbouring classes the benchmark treats as ambiguous (Van against car, seated people against pedestrian), are excluded rather than counted as errors.
