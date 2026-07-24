"""Typed access to configs/default.yaml. Loading through this module means a typo in
the YAML fails loudly at startup instead of silently halfway through a training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import CONFIGS


@dataclass(frozen=True)
class Models:
    edge: str      # fast deployment target, e.g. yolo11n
    accuracy: str  # accuracy anchor, e.g. yolo11s


@dataclass(frozen=True)
class Detect:
    imgsz: int
    conf: float
    iou: float


@dataclass(frozen=True)
class Train:
    epochs: int
    batch: int
    patience: int


@dataclass(frozen=True)
class Config:
    models: Models
    detect: Detect
    tracker: str
    train: Train

    @property
    def model_names(self) -> list[str]:
        """Both detectors, in the order the benchmark reports them."""
        return [self.models.edge, self.models.accuracy]


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIGS / "default.yaml"
    raw = yaml.safe_load(path.read_text())

    tracker = raw["tracker"]
    if tracker not in {"bytetrack", "botsort"}:
        raise ValueError(f"tracker must be bytetrack or botsort, got {tracker!r}")

    return Config(
        models=Models(**raw["models"]),
        detect=Detect(**raw["detect"]),
        tracker=tracker,
        train=Train(**raw["train"]),
    )
