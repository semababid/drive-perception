# Zero-shot baseline on KITTI val

Model: `yolo11n.pt`, COCO-pretrained, no KITTI training. Scored on 1496 validation frames at IoU 0.5.

Ground-truth boxes: car 4365, pedestrian 799, cyclist 223.

| tier | car | pedestrian | cyclist | mAP |
|---|---|---|---|---|
| all | 0.728 | 0.444 | 0.010 | 0.394 |
| easy | 0.890 | 0.607 | 0.008 | 0.502 |
| moderate | 0.666 | 0.154 | 0.002 | 0.274 |
| hard | 0.268 | 0.058 | 0.012 | 0.113 |

## Reading these numbers

The COCO classes do not line up with KITTI one for one, and the gaps show up directly in the per-class scores.

- **car** maps cleanly from the COCO `car` class, so this column is the fair measure of what a pretrained detector already knows about the KITTI domain.
- **pedestrian** maps from COCO `person`. The definitions are close, though KITTI splits out seated people into a separate class that we drop, so a detection of someone sitting counts against the model here.
- **cyclist** is the weak mapping and the number should be read with that in mind. KITTI marks a rider and their bicycle as one Cyclist box, while COCO sees a `bicycle` and a `person` separately.

The cyclist column is a box mismatch rather than a detection failure, and the numbers separate the two cases. The model produced 2133 bicycle detections against 223 cyclist boxes, so it is clearly seeing the bikes. The best overlap any of those detections reached was 0.78, short of the 0.5 threshold, because the COCO box stops at the bicycle while the KITTI box also contains the rider. Every one of them is therefore scored as a miss.

Fine-tuning on KITTI removes all three mismatches at once, because the model then learns the KITTI class definitions directly. That is the comparison the next step makes.
