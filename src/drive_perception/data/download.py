"""Fetch the KITTI 2D object-detection data.

KITTI ships the left color images (`image_2`) as one ~12 GB archive and the training
labels as a tiny separate one. Both live on the public avg-kitti S3 mirror, so no
registration or login is needed.

For a first pass you rarely want all 7,481 training frames. `download_subset` pulls
just the first N straight out of the remote image archive using HTTP range requests,
so a working set is a few hundred megabytes instead of twelve gigabytes. Only the
`training` split is fetched. The `testing` split has no public labels, so it is
useless for the evaluation this project reports.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from ..paths import KITTI_RAW, RAW, ensure_dirs

MIRROR = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
IMAGES_URL = f"{MIRROR}/data_object_image_2.zip"
LABELS_URL = f"{MIRROR}/data_object_label_2.zip"

# Paths inside the archives. KITTI numbers frames 000000..007480; a label file and an
# image file that share a stem describe the same frame.
IMAGE_PREFIX = "training/image_2/"
LABEL_PREFIX = "training/label_2/"


def _stream_download(url: str, dest: Path, force: bool) -> Path:
    """Download a whole archive with a progress bar, skipping if already present."""
    if dest.exists() and not force:
        print(f"  cached  {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310  (fixed, trusted mirror)
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(dest, "wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    return dest


def _extract_prefix(archive: Path, prefix: str, ids: set[str] | None = None) -> int:
    """Extract members under `prefix` into the KITTI tree. If `ids` is given, keep only
    files whose stem is in it, so the labels stay aligned with a fetched image subset."""
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            if ids is not None and Path(name).stem not in ids:
                continue
            zf.extract(name, KITTI_RAW)
            count += 1
    return count


def _summary() -> tuple[int, int]:
    images = list((KITTI_RAW / IMAGE_PREFIX).glob("*.png"))
    labels = list((KITTI_RAW / LABEL_PREFIX).glob("*.txt"))
    print(f"  ready   {len(images)} images, {len(labels)} labels under {KITTI_RAW}")
    return len(images), len(labels)


def download_subset(n: int, force: bool = False) -> tuple[int, int]:
    """Fetch the first `n` training frames and their labels, a small and fast working set."""
    ensure_dirs()
    KITTI_RAW.mkdir(parents=True, exist_ok=True)

    # Range-request only the image files we want out of the 12 GB remote archive.
    from remotezip import RemoteZip

    print(f"[1/2] fetching {n} training images via range requests")
    with RemoteZip(IMAGES_URL) as rz:
        names = sorted(
            m for m in rz.namelist() if m.startswith(IMAGE_PREFIX) and m.endswith(".png")
        )[:n]
        for name in tqdm(names, unit="img"):
            rz.extract(name, KITTI_RAW)
    ids = {Path(name).stem for name in names}

    # The label archive is only a few MB, so grab it whole and keep the matching files.
    print("[2/2] fetching labels for those frames")
    label_zip = _stream_download(LABELS_URL, RAW / "data_object_label_2.zip", force)
    _extract_prefix(label_zip, LABEL_PREFIX, ids=ids)
    return _summary()


def download_full(force: bool = False) -> tuple[int, int]:
    """Fetch the entire training split, all 7,481 frames and labels (~12 GB)."""
    ensure_dirs()
    KITTI_RAW.mkdir(parents=True, exist_ok=True)

    print("[1/2] images (~12 GB)")
    image_zip = _stream_download(IMAGES_URL, RAW / "data_object_image_2.zip", force)
    _extract_prefix(image_zip, IMAGE_PREFIX)

    print("[2/2] labels")
    label_zip = _stream_download(LABELS_URL, RAW / "data_object_label_2.zip", force)
    _extract_prefix(label_zip, LABEL_PREFIX)
    return _summary()
