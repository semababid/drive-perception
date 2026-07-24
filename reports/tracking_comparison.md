# Tracking: what detection fixes and what association fixes

Three runs over the same 3 KITTI sequences, matched at IoU 0.5. Only one thing changes between neighbouring columns, so each step isolates a single cause.

| metric | pretrained + ByteTrack | fine-tuned + ByteTrack | fine-tuned + BoT-SORT |
|---|---|---|---|
| MOTA | 0.362 | 0.738 | 0.761 |
| IDF1 | 0.506 | 0.706 | 0.724 |
| ID switches | 85 | 83 | 61 |
| false positives | 598 | 303 | 315 |
| misses | 2188 | 795 | 700 |

## Reading the result

Replacing the pretrained detector with the fine-tuned one lifted MOTA by +0.376, cutting misses from 2188 to 795. That is the single largest change in the table, and it comes entirely from the detector.

It barely touched identity switches, which moved by -2 from 85 to 83. Finding an object more reliably does not, on its own, help the tracker decide that the object it found is the same one as last frame.

Switching from ByteTrack to BoT-SORT, with the detector held fixed, moved switches by -22, from 83 to 61. The two trackers use identical association thresholds here, so the difference is global motion compensation. The KITTI camera is mounted on a moving car, and estimating that motion between frames lets the tracker tell an object moving on its own apart from the whole scene shifting.

The practical conclusion is that misses and identity switches are separate problems with separate fixes. A better detector addresses the first and leaves the second largely untouched. A motion-aware association step addresses the second. Reaching for a better detector to solve identity churn would have been effort spent in the wrong place.

## Per class

| class | metric | pretrained + ByteTrack | fine-tuned + ByteTrack | fine-tuned + BoT-SORT |
|---|---|---|---|---|
| car | MOTA | 0.459 | 0.782 | 0.798 |
| car | IDF1 | 0.542 | 0.720 | 0.737 |
| car | ID switches | 83 | 70 | 49 |
| pedestrian | MOTA | -0.277 | 0.369 | 0.510 |
| pedestrian | IDF1 | 0.380 | 0.543 | 0.585 |
| pedestrian | ID switches | 2 | 11 | 11 |
| cyclist | MOTA | -0.441 | 0.467 | 0.459 |
| cyclist | IDF1 | 0.083 | 0.688 | 0.689 |
| cyclist | ID switches | 0 | 2 | 1 |

Cyclist is worth noting: it was untrackable with the pretrained detector because COCO draws a bicycle where KITTI draws a rider and bicycle together, so almost nothing matched. Once the detector learned the KITTI box, cyclist identity became the most stable of the three classes.
