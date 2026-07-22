"""Every path in the project resolves from here, so scripts behave the same whether
they are run from the repo root, from a notebook, or from inside Docker."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"              # KITTI downloads land here untouched
KITTI_RAW = RAW / "kitti"       # extracted KITTI tree: training/image_2, training/label_2
INTERIM = DATA / "interim"     # extracted / half-processed working files
PROCESSED = DATA / "processed" # YOLO-format detection dataset
TRACKING = DATA / "tracking"   # tracking sequences used for MOTA / IDF1
GOLDEN = DATA / "golden"       # tiny checked-in set for regression tests

CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"

# Reports are checked in. They are the evidence behind the numbers in the README,
# small enough that a reader can audit the claims without rebuilding the dataset.
REPORTS = ROOT / "reports"

DETECT_DIR = PROCESSED / "kitti-det"
DETECT_YAML = DETECT_DIR / "kitti-det.yaml"


def ensure_dirs() -> None:
    for d in (RAW, INTERIM, PROCESSED, TRACKING, GOLDEN, MODELS, OUTPUTS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
