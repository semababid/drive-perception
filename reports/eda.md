# KITTI subset: exploratory analysis

- Frames analysed: **7481**
- Objects total: **51865** (kept for training: **34856**)
- Mean kept objects per frame: **4.66**
- Boxes under 25 px tall: **4287** (**12%** of kept objects)

## Per-class

| class | count | easy | moderate | hard | ignored | median height px | aspect w/h |
|---|---|---|---|---|---|---|---|
| car | 28742 | 5971 | 9739 | 6006 | 7026 | 47.0 | 1.72 |
| pedestrian | 4487 | 2310 | 1259 | 707 | 211 | 87.9 | 0.4 |
| cyclist | 1627 | 654 | 444 | 96 | 433 | 60.1 | 0.69 |

## What this means for the model

- **Severe class imbalance:** Car:Pedestrian:Cyclist is roughly 28742:4487:1627. Cyclist is the scarce class and the one to watch. Its AP should be reported on its own rather than buried in a mean.
- **Small, distant objects cause most of the difficulty, unevenly by class:** 12% of kept objects fall below KITTI's 25 px floor. It hits **cyclist** hardest (27% ignored) and **car** next (24%); the remaining class sits mostly close and large. Keeping input resolution high matters most for those distant cases.
- **Aspect ratios separate the classes cleanly** (car w/h ≈ 1.72, pedestrian ≈ 0.4), a sanity check that the boxes are well-formed and the three classes are visually distinct.
- **Difficulty tiers are uneven per class**, so a single mAP number would hide where the model actually fails. We report Easy/Moderate/Hard separately, as KITTI does.
