# KITTI subset: exploratory analysis

- Frames analysed: **300**
- Objects total: **2142** (kept for training: **1306**)
- Mean kept objects per frame: **4.35**
- Boxes under 25 px tall: **168** (**13%** of kept objects)

## Per-class

| class | count | easy | moderate | hard | ignored | median height px | aspect w/h |
|---|---|---|---|---|---|---|---|
| car | 1094 | 235 | 387 | 210 | 262 | 45.8 | 1.71 |
| pedestrian | 141 | 84 | 32 | 17 | 8 | 76.4 | 0.4 |
| cyclist | 71 | 34 | 15 | 4 | 18 | 71.7 | 0.7 |

## What this means for the model

- **Severe class imbalance:** Car:Pedestrian:Cyclist is roughly 1094:141:71. Cyclist is the scarce class and the one to watch. Its AP should be reported on its own rather than buried in a mean.
- **Small, distant objects cause most of the difficulty, unevenly by class:** 13% of kept objects fall below KITTI's 25 px floor. It hits **cyclist** hardest (25% ignored) and **car** next (24%); the remaining class sits mostly close and large. Keeping input resolution high matters most for those distant cases.
- **Aspect ratios separate the classes cleanly** (car w/h ≈ 1.71, pedestrian ≈ 0.4), a sanity check that the boxes are well-formed and the three classes are visually distinct.
- **Difficulty tiers are uneven per class**, so a single mAP number would hide where the model actually fails. We report Easy/Moderate/Hard separately, as KITTI does.
