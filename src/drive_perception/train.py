"""Fine-tune YOLO11 on the three KITTI classes.

Training starts from the COCO-pretrained checkpoint rather than from scratch. The
baseline showed the pretrained features already transfer well to cars, so the job here
is teaching the model KITTI's class definitions, above all the Cyclist box that COCO
has no equivalent for.

The best checkpoint is copied out of the Ultralytics run directory into models/ under a
stable name, so evaluation and export do not have to guess which run was the good one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .paths import DETECT_YAML, MODELS, ROOT

RUNS = ROOT / "runs"


def pick_device(cuda_available: bool, mps_available: bool) -> str:
    """Choose a training device. Split from the torch lookup so it can be tested."""
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def current_device() -> str:
    import torch

    return pick_device(torch.cuda.is_available(), torch.backends.mps.is_available())


def finetune(
    model: str = "yolo11n",
    data: Path | None = None,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 15,
    device: str | None = None,
    seed: int = 0,
    name: str | None = None,
) -> dict:
    """Fine-tune one model and return where the weights and run directory landed."""
    from ultralytics import YOLO

    data = data or DETECT_YAML
    if not Path(data).exists():
        raise FileNotFoundError(f"{data} missing. Run scripts/split_dataset.py first.")
    device = device or current_device()
    name = name or f"finetune_{model}"

    yolo = YOLO(f"{model}.pt")
    yolo.train(
        data=str(data),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        device=device,
        seed=seed,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        plots=True,
    )

    run_dir = RUNS / name
    best = run_dir / "weights" / "best.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    weights = MODELS / f"{model}_kitti.pt"
    if best.exists():
        shutil.copy2(best, weights)
    return {"weights": weights, "run_dir": run_dir, "device": device}
