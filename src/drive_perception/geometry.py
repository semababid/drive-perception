"""Box geometry with no dependencies beyond the standard library.

`iou` lives here rather than in the evaluation module because the online tracker needs
it and nothing else from there. Keeping it separate means the prediction service, which
uses the tracker, does not pull in matplotlib and the rest of the evaluation and plotting
stack just to compute an overlap.
"""

from __future__ import annotations

from collections.abc import Sequence


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
