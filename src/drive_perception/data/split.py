"""Split the converted KITTI set into train and val, and write the Ultralytics
`data.yaml`.

The split is a fixed-seed 80/20 over frames, written as `train.txt` and `val.txt`
lists of image paths rather than by moving files, so the images stay in one place and
nothing is duplicated. Ultralytics finds each label by swapping `images` for `labels`
in the absolute image path, which is why the lists hold resolved absolute paths.

Because Cyclist is the scarce class, the split also reports per-class box counts for
each side. A validation set with no cyclists would make its AP meaningless, so it is
worth seeing the balance rather than trusting the shuffle.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import yaml

from ..paths import DETECT_DIR, DETECT_YAML
from .convert import CLASS_NAMES


def split_items(items: list, val_frac: float, seed: int) -> tuple[list, list]:
    """Deterministically shuffle and split a list into (train, val). Kept pure so the
    split logic can be tested without any files on disk."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_frac)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return sorted(train), sorted(val)


def _class_counts(image_paths: list[Path], labels_dir: Path) -> dict[str, int]:
    counts = dict.fromkeys(CLASS_NAMES, 0)
    for img in image_paths:
        label = labels_dir / f"{img.stem}.txt"
        for line in label.read_text().splitlines():
            if line:
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return counts


def _write_yaml(root: Path, out: Path) -> None:
    data = {
        "path": str(root.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False))


def build(
    out_dir: Path | None = None,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    out_dir = out_dir or DETECT_DIR
    images = sorted((out_dir / "images").glob("*.png"))
    if not images:
        raise FileNotFoundError(
            f"no images under {out_dir / 'images'}. Run scripts/convert_labels.py first."
        )
    labels_dir = out_dir / "labels"

    train, val = split_items(images, val_frac, seed)
    # abspath, not resolve: the list must keep the processed images/ path so Ultralytics
    # can swap images/ for labels/. resolve() would follow the symlink back to the raw
    # image_2 tree, where no converted labels exist.
    (out_dir / "train.txt").write_text("\n".join(os.path.abspath(p) for p in train) + "\n")
    (out_dir / "val.txt").write_text("\n".join(os.path.abspath(p) for p in val) + "\n")
    _write_yaml(out_dir, DETECT_YAML)

    return {
        "train_frames": len(train),
        "val_frames": len(val),
        "train_boxes": _class_counts(train, labels_dir),
        "val_boxes": _class_counts(val, labels_dir),
        "yaml": DETECT_YAML,
    }
