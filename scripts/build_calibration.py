#!/usr/bin/env python
"""Build a small calibration and evaluation bundle for INT8 export on Colab.

TensorRT learns the INT8 activation ranges from a set of calibration images. The images
it calibrates on must not be the images accuracy is measured on, or the quantisation is
tuned to the test set and the reported mAP flatters itself. So calibration frames come
from the training split and the evaluation frames come from the held-out validation
split, and the two never overlap.

Images are resized to the letterbox content size and saved as JPEG. That is the pixels
the model actually sees at inference once a full frame is letterboxed to 224 by 640, so
nothing meaningful is lost, and it turns a 200 MB upload into a few megabytes. YOLO
labels are normalised, so they stay correct through the resize.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import yaml

from drive_perception.data.convert import CLASS_NAMES
from drive_perception.paths import DATA, DETECT_DIR


def _stems(list_file: Path, n: int) -> list[str]:
    lines = [line for line in list_file.read_text().splitlines() if line.strip()]
    return [Path(line).stem for line in lines[:n]]


def _resize_to_fit(image, max_h: int, max_w: int):
    h, w = image.shape[:2]
    scale = min(max_h / h, max_w / w)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _copy_split(stems: list[str], split: str, root: Path, imgsz: tuple[int, int]) -> int:
    img_out = root / "images" / split
    lbl_out = root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    written = 0
    for stem in stems:
        src_img = DETECT_DIR / "images" / f"{stem}.png"
        src_lbl = DETECT_DIR / "labels" / f"{stem}.txt"
        if not src_img.exists() or not src_lbl.exists():
            continue
        image = cv2.imread(str(src_img))
        resized = _resize_to_fit(image, imgsz[0], imgsz[1])
        cv2.imwrite(str(img_out / f"{stem}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")
        written += 1
    return written


def build(n_calib: int, n_eval: int, imgsz: tuple[int, int]) -> Path:
    root = DATA / "int8_calib"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    calib_stems = _stems(DETECT_DIR / "train.txt", n_calib)
    eval_stems = _stems(DETECT_DIR / "val.txt", n_eval)
    if set(calib_stems) & set(eval_stems):
        raise RuntimeError("calibration and evaluation frames overlap")

    n_c = _copy_split(calib_stems, "train", root, imgsz)
    n_e = _copy_split(eval_stems, "val", root, imgsz)

    # Two yamls, so calibration can never see the evaluation frames. Ultralytics decides
    # internally which split it calibrates on, so the calibration yaml points both splits
    # at the calibration images; whatever split it reads, it reads those. Accuracy is
    # measured through the separate eval yaml, whose val split is the held-out frames.
    # `path` is a placeholder the notebook rewrites to the absolute unzip location.
    names = dict(enumerate(CLASS_NAMES))
    (root / "calib.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "val": "images/train", "names": names},
            sort_keys=False,
        )
    )
    (root / "eval.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/val", "val": "images/val", "names": names},
            sort_keys=False,
        )
    )

    archive = DATA / "int8_calib.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root.parent))

    size_mb = archive.stat().st_size / 1e6
    print(f"calibration frames: {n_c}  (from train split)")
    print(f"evaluation frames : {n_e}  (from val split, held out)")
    print(f"bundle            : {archive}  ({size_mb:.1f} MB)")
    return archive


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib", type=int, default=128, help="calibration frames")
    p.add_argument("--eval", type=int, default=128, help="held-out evaluation frames")
    p.add_argument("--imgsz", type=int, nargs=2, default=[224, 640], metavar=("H", "W"))
    args = p.parse_args()
    build(args.calib, args.eval, tuple(args.imgsz))


if __name__ == "__main__":
    main()
