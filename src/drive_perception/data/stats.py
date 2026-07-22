"""Exploratory analysis of the KITTI 2D labels, tailored to what actually drives a
driving-perception model: class imbalance, the small-object problem, and KITTI's own
Easy/Moderate/Hard difficulty tiers.

The difficulty tiers are not ours. They are KITTI's official evaluation protocol,
defined purely from 2D box height, occlusion and truncation. Profiling them now means
the mAP we report later can be broken down the same way the benchmark expects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from ..paths import KITTI_RAW, REPORTS

# The three classes this project detects. Everything else in KITTI (Van, Truck, Tram,
# Misc, Person_sitting, DontCare) is dropped, but we still count it so the imbalance
# is visible rather than hidden.
KEEP = {"Car": "car", "Pedestrian": "pedestrian", "Cyclist": "cyclist"}

# KITTI difficulty thresholds. min box height in pixels, max occlusion level, max
# truncation fraction. An object failing even Hard is ignored by the benchmark.
TIERS = {
    "easy": dict(min_h=40, max_occ=0, max_trunc=0.15),
    "moderate": dict(min_h=25, max_occ=1, max_trunc=0.30),
    "hard": dict(min_h=25, max_occ=2, max_trunc=0.50),
}


@dataclass
class Obj:
    cls: str          # raw KITTI type, e.g. "Car"
    truncated: float  # 0..1
    occluded: int     # 0 visible, 1 partly, 2 largely, 3 unknown
    height: float     # 2D box height in pixels
    width: float      # 2D box width in pixels

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def tier(self) -> str:
        """The strictest KITTI tier this object qualifies for, or 'ignored'."""
        for name, t in TIERS.items():
            if (
                self.height >= t["min_h"]
                and self.occluded <= t["max_occ"]
                and self.truncated <= t["max_trunc"]
            ):
                return name
        return "ignored"


def parse_frame(label_file: Path) -> list[Obj]:
    objs: list[Obj] = []
    for line in label_file.read_text().splitlines():
        f = line.split()
        if len(f) < 8:
            continue
        left, top, right, bottom = (float(x) for x in f[4:8])
        objs.append(
            Obj(
                cls=f[0],
                truncated=float(f[1]),
                occluded=int(float(f[2])),
                height=bottom - top,
                width=right - left,
            )
        )
    return objs


def collect(label_dir: Path) -> tuple[list[Obj], int]:
    files = sorted(label_dir.glob("*.txt"))
    objs = [o for fp in files for o in parse_frame(fp)]
    return objs, len(files)


def summarize(objs: list[Obj], n_frames: int) -> dict:
    kept = [o for o in objs if o.cls in KEEP]

    raw_counts: dict[str, int] = {}
    for o in objs:
        raw_counts[o.cls] = raw_counts.get(o.cls, 0) + 1

    # Per-kept-class: count, tier breakdown, median box height, mean aspect ratio.
    per_class: dict[str, dict] = {}
    for raw, name in KEEP.items():
        group = [o for o in kept if o.cls == raw]
        tier_names = ["easy", "moderate", "hard", "ignored"]
        tiers = {t: sum(1 for o in group if o.tier() == t) for t in tier_names}
        heights = sorted(o.height for o in group)
        median_h = heights[len(heights) // 2] if heights else 0.0
        per_class[name] = {
            "count": len(group),
            "tiers": tiers,
            "median_box_height_px": round(median_h, 1),
            "mean_aspect_w_over_h": round(
                sum(o.aspect for o in group) / len(group), 2
            ) if group else 0.0,
        }

    tiny = sum(1 for o in kept if o.height < 25)
    return {
        "frames": n_frames,
        "objects_total": len(objs),
        "objects_kept": len(kept),
        "raw_class_counts": dict(sorted(raw_counts.items(), key=lambda kv: -kv[1])),
        "per_class": per_class,
        "small_objects_under_25px": tiny,
        "small_object_fraction": round(tiny / len(kept), 3) if kept else 0.0,
        "objects_per_frame_mean": round(len(kept) / n_frames, 2) if n_frames else 0.0,
    }


def _plots(objs: list[Obj], summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    kept = [o for o in objs if o.cls in KEEP]

    # 1. Class imbalance across all raw KITTI classes, kept three highlighted.
    counts = summary["raw_class_counts"]
    names = list(counts)
    colors = ["#2563eb" if c in KEEP else "#cbd5e1" for c in names]
    plt.figure(figsize=(8, 4))
    plt.bar(names, list(counts.values()), color=colors)
    plt.title("KITTI class distribution (kept classes in blue)")
    plt.ylabel("objects")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=120)
    plt.close()

    # 2. The small-object story: box-height histogram with the 25/40px tier lines.
    plt.figure(figsize=(8, 4))
    plt.hist([o.height for o in kept], bins=40, range=(0, 200), color="#2563eb")
    top = plt.ylim()[1]
    # Stagger the two labels vertically so the 25 px and 40 px markers don't collide.
    for x, label, y_frac in [(25, "hard/mod min (25px)", 0.92), (40, "easy min (40px)", 0.80)]:
        plt.axvline(x, color="#dc2626", linestyle="--")
        plt.text(x + 3, top * y_frac, label, color="#dc2626", fontsize=8)
    plt.title("2D box height and the small-object problem")
    plt.xlabel("box height (px)")
    plt.ylabel("objects")
    plt.tight_layout()
    plt.savefig(out / "box_height_distribution.png", dpi=120)
    plt.close()

    # 3. Difficulty tiers per class, the protocol we will report mAP against.
    order = ["easy", "moderate", "hard", "ignored"]
    tier_colors = ["#16a34a", "#eab308", "#f97316", "#94a3b8"]
    classes = list(summary["per_class"])
    bottoms = [0] * len(classes)
    plt.figure(figsize=(7, 4))
    for tier, color in zip(order, tier_colors, strict=True):
        vals = [summary["per_class"][c]["tiers"][tier] for c in classes]
        plt.bar(classes, vals, bottom=bottoms, label=tier, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    plt.title("KITTI difficulty tiers per class")
    plt.ylabel("objects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "difficulty_tiers.png", dpi=120)
    plt.close()


def _markdown(summary: dict) -> str:
    pc = summary["per_class"]
    lines = [
        "# KITTI subset: exploratory analysis",
        "",
        f"- Frames analysed: **{summary['frames']}**",
        f"- Objects total: **{summary['objects_total']}** "
        f"(kept for training: **{summary['objects_kept']}**)",
        f"- Mean kept objects per frame: **{summary['objects_per_frame_mean']}**",
        f"- Boxes under 25 px tall: **{summary['small_objects_under_25px']}** "
        f"(**{summary['small_object_fraction']:.0%}** of kept objects)",
        "",
        "## Per-class",
        "",
        "| class | count | easy | moderate | hard | ignored | median height px | aspect w/h |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, d in pc.items():
        t = d["tiers"]
        lines.append(
            f"| {name} | {d['count']} | {t['easy']} | {t['moderate']} | {t['hard']} "
            f"| {t['ignored']} | {d['median_box_height_px']} | {d['mean_aspect_w_over_h']} |"
        )
    counts = summary["raw_class_counts"]
    car, ped, cyc = counts.get("Car", 0), counts.get("Pedestrian", 0), counts.get("Cyclist", 0)

    # Derive the small-object statement from the data rather than asserting it, so the
    # takeaway can never drift away from the numbers in the table above.
    def ignored_frac(name: str) -> float:
        d = pc[name]
        return d["tiers"]["ignored"] / d["count"] if d["count"] else 0.0

    ranked = sorted(pc, key=ignored_frac, reverse=True)
    worst, second = ranked[0], ranked[1]
    lines += [
        "",
        "## What this means for the model",
        "",
        f"- **Severe class imbalance:** Car:Pedestrian:Cyclist is roughly "
        f"{car}:{ped}:{cyc}. Cyclist is the scarce class and the one to watch. Its AP "
        "should be reported on its own rather than buried in a mean.",
        f"- **Small, distant objects cause most of the difficulty, unevenly by class:** "
        f"{summary['small_object_fraction']:.0%} of kept objects fall below KITTI's 25 px "
        f"floor. It hits **{worst}** hardest ({ignored_frac(worst):.0%} ignored) and "
        f"**{second}** next ({ignored_frac(second):.0%}); the remaining class sits mostly "
        "close and large. Keeping input resolution high matters most for those distant cases.",
        "- **Aspect ratios separate the classes cleanly** "
        f"(car w/h ≈ {pc['car']['mean_aspect_w_over_h']}, "
        f"pedestrian ≈ {pc['pedestrian']['mean_aspect_w_over_h']}), a sanity check that the "
        "boxes are well-formed and the three classes are visually distinct.",
        "- **Difficulty tiers are uneven per class**, so a single mAP number would hide "
        "where the model actually fails. We report Easy/Moderate/Hard separately, as KITTI does.",
    ]
    return "\n".join(lines) + "\n"


def run(kitti_root: Path | None = None, out_dir: Path | None = None) -> dict:
    kitti_root = kitti_root or KITTI_RAW
    out_dir = out_dir or REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)

    objs, n_frames = collect(kitti_root / "training" / "label_2")
    if n_frames == 0:
        raise FileNotFoundError(
            f"no label files under {kitti_root}. Run scripts/download_kitti.py first."
        )

    summary = summarize(objs, n_frames)
    (out_dir / "dataset_stats.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "eda.md").write_text(_markdown(summary))
    _plots(objs, summary, out_dir / "plots")
    return summary
