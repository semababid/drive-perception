#!/usr/bin/env python
"""Fine-tune a YOLO11 model on the KITTI classes.

Defaults come from configs/default.yaml. The device is picked automatically, which on
Apple Silicon means the MPS GPU rather than the CPU.

    python scripts/finetune.py                      # the edge model, yolo11n
    python scripts/finetune.py --model yolo11s      # the accuracy anchor
    python scripts/finetune.py --epochs 5           # a quick pipeline check
"""

from __future__ import annotations

import argparse

from drive_perception.config import load_config
from drive_perception.train import finetune


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=cfg.models.edge, help="yolo11n or yolo11s")
    p.add_argument("--epochs", type=int, default=cfg.train.epochs)
    p.add_argument("--batch", type=int, default=cfg.train.batch)
    p.add_argument("--imgsz", type=int, default=cfg.detect.imgsz)
    p.add_argument("--patience", type=int, default=cfg.train.patience)
    p.add_argument("--device", default=None, help="cpu, mps or cuda (auto by default)")
    p.add_argument("--name", default=None, help="run name under runs/")
    args = p.parse_args()

    out = finetune(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        name=args.name,
    )
    print(f"device : {out['device']}")
    print(f"run    : {out['run_dir']}")
    print(f"weights: {out['weights']}")


if __name__ == "__main__":
    main()
