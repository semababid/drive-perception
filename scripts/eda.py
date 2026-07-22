#!/usr/bin/env python
"""Profile the downloaded KITTI labels and write the analysis to reports/.

Produces reports/dataset_stats.json (raw numbers), reports/eda.md (readable summary
with modeling takeaways) and reports/plots/*.png. Run it after downloading data:

    python scripts/eda.py
"""

from __future__ import annotations

from drive_perception.data import stats
from drive_perception.paths import REPORTS


def main() -> None:
    summary = stats.run()
    print(f"frames: {summary['frames']}  kept objects: {summary['objects_kept']}")
    for name, d in summary["per_class"].items():
        t = d["tiers"]
        print(
            f"  {name:11s} n={d['count']:4d}  "
            f"easy={t['easy']:3d} mod={t['moderate']:3d} hard={t['hard']:3d}  "
            f"median_h={d['median_box_height_px']}px"
        )
    print(f"reports written to {REPORTS}")


if __name__ == "__main__":
    main()
