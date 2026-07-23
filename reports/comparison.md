# Fine-tuning against the zero-shot baseline

`yolo11n.pt` is the COCO-pretrained model with its classes remapped onto KITTI. `models/yolo11n_kitti.pt` is the same architecture fine-tuned on the KITTI training split. Both are scored by the same code on the same 1496 validation frames at IoU 0.5.

| tier | metric | pretrained | fine-tuned | change |
|---|---|---|---|---|
| all | car AP | 0.728 | 0.960 | +0.231 |
| all | pedestrian AP | 0.444 | 0.768 | +0.324 |
| all | cyclist AP | 0.010 | 0.846 | +0.836 |
| all | **mAP** | **0.394** | **0.858** | **+0.464** |
| easy | car AP | 0.890 | 0.994 | +0.104 |
| easy | pedestrian AP | 0.607 | 0.928 | +0.321 |
| easy | cyclist AP | 0.008 | 0.942 | +0.934 |
| easy | **mAP** | **0.502** | **0.955** | **+0.453** |
| moderate | car AP | 0.666 | 0.969 | +0.303 |
| moderate | pedestrian AP | 0.154 | 0.596 | +0.442 |
| moderate | cyclist AP | 0.002 | 0.679 | +0.677 |
| moderate | **mAP** | **0.274** | **0.748** | **+0.474** |
| hard | car AP | 0.268 | 0.792 | +0.524 |
| hard | pedestrian AP | 0.058 | 0.345 | +0.288 |
| hard | cyclist AP | 0.012 | 0.306 | +0.295 |
| hard | **mAP** | **0.113** | **0.481** | **+0.369** |

## What changed

Overall mAP moved from 0.394 to 0.858. The three classes did not move for the same reasons, and the per-class rows are more informative than the mean.

**Cyclist** is the clearest result: 0.010 to 0.846. The pretrained model could see bicycles perfectly well but drew them the way COCO defines them, around the bicycle alone, while KITTI draws one box around rider and bicycle together. No amount of extra confidence fixes a box that is the wrong shape. Learning the KITTI definition does.

**Car** was already the pretrained model's strongest class at 0.728, so there was less headroom, and it reached 0.960.

**Pedestrian** went from 0.444 to 0.768. COCO's `person` class is close to KITTI's Pedestrian, so the starting point was reasonable, and the gain comes mostly from the smaller and partly occluded cases.

The tier rows show where the remaining errors live. The gap between easy and hard is the distant, occluded traffic that the exploratory analysis flagged at the start, and it is still the hardest part of the problem.
