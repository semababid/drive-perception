"""Convert KITTI 2D labels into the normalized YOLO format Ultralytics reads.

KITTI gives one text file per frame, with absolute pixel boxes and nine object types.
We keep the three that matter for driving perception (Car, Pedestrian, Cyclist), map
them to class ids 0/1/2, and rewrite each box as `cls x_center y_center w h`, all
normalized to the image size. Frames vary in width and height, so every box is scaled
by the dimensions of its own image rather than a fixed constant.

Images are symlinked into an `images/` directory next to `labels/` instead of copied,
so the working set stays at its download size instead of doubling. Ultralytics locates
a label by swapping `images` for `labels` in the path, which is why the two live side
by side under the same parent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..paths import DETECT_DIR, KITTI_RAW

# The single definition of which KITTI classes this project detects, mapping the name
# used in the raw labels to the name used everywhere else. Everything downstream, the
# label converter, the exploratory analysis, the detection metrics and the tracking
# metrics, derives from this one dict. They were four separate copies once, which meant
# adding a class required four coordinated edits and a missed one would have left the
# detection and tracking evaluations quietly scoring different things.
# Anything absent here (Van, Truck, Tram, Misc, Person_sitting, DontCare) is dropped.
KITTI_CLASSES = {"Car": "car", "Pedestrian": "pedestrian", "Cyclist": "cyclist"}

CLASS_NAMES = list(KITTI_CLASSES.values())  # index == class id
CLASS_ID = {raw: index for index, raw in enumerate(KITTI_CLASSES)}


@dataclass
class ConversionResult:
    frames: int
    boxes_written: int
    dropped_other_class: int  # objects of a type we do not train on
    dropped_degenerate: int   # boxes that clip to nothing


def to_yolo_bbox(fields: list[str], w: int, h: int) -> str | None:
    """Turn one KITTI object line (already a kept class) into a YOLO row, or return
    None if the box clips to a degenerate sliver. `fields` is the whitespace-split line;
    columns 4..7 are left, top, right, bottom in pixels."""
    left, top, right, bottom = (float(x) for x in fields[4:8])
    # KITTI boxes occasionally spill a pixel past the frame; clip before normalizing.
    left = min(max(left, 0.0), w)
    right = min(max(right, 0.0), w)
    top = min(max(top, 0.0), h)
    bottom = min(max(bottom, 0.0), h)
    bw, bh = right - left, bottom - top
    if bw <= 1.0 or bh <= 1.0:
        return None
    cid = CLASS_ID[fields[0]]
    xc = (left + right) / 2 / w
    yc = (top + bottom) / 2 / h
    return f"{cid} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}"


def _link(src: Path, dst: Path) -> None:
    """Symlink src to dst, falling back to a copy if the filesystem refuses symlinks."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def convert(
    kitti_root: Path | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> ConversionResult:
    kitti_root = kitti_root or KITTI_RAW
    out_dir = out_dir or DETECT_DIR
    img_src = kitti_root / "training" / "image_2"
    lbl_src = kitti_root / "training" / "label_2"
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"

    label_files = sorted(lbl_src.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(
            f"no labels under {lbl_src}. Run scripts/download_kitti.py first."
        )
    if lbl_out.exists() and any(lbl_out.iterdir()) and not force:
        raise FileExistsError(
            f"{out_dir} already has a converted set. Pass force=True to rebuild."
        )

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written = dropped_other = dropped_degenerate = 0
    for lbl in label_files:
        stem = lbl.stem
        image = img_src / f"{stem}.png"
        if not image.exists():
            continue
        with Image.open(image) as im:
            w, h = im.size

        rows: list[str] = []
        for line in lbl.read_text().splitlines():
            fields = line.split()
            if len(fields) < 8:
                continue
            if fields[0] not in CLASS_ID:
                dropped_other += 1
                continue
            row = to_yolo_bbox(fields, w, h)
            if row is None:
                dropped_degenerate += 1
                continue
            rows.append(row)

        # An empty file is written on purpose: a frame with no kept object is a valid
        # background image and helps the detector learn what not to fire on.
        (lbl_out / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        _link(image, img_out / f"{stem}.png")
        written += len(rows)

    return ConversionResult(
        frames=len(label_files),
        boxes_written=written,
        dropped_other_class=dropped_other,
        dropped_degenerate=dropped_degenerate,
    )
