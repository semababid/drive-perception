"""Measure end-to-end inference latency across backends.

The number that matters for a perception stack is how long one frame takes from pixels
to detections, so every backend is timed over its whole `predict` call: preprocessing,
inference and postprocessing together. A model-only figure would flatter whichever
backend has the slowest surrounding Python.

Latency is summarised by its median rather than its mean. The first few frames after a
backend warms up are erratic, and a single slow one drags the mean while leaving the
median, which is what a steady stream actually feels, untouched. A warmup pass runs
first for the same reason: the initial call compiles kernels and allocates buffers, and
timing that would measure setup, not inference.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import mean, median


@dataclass(frozen=True)
class LatencyStats:
    backend: str
    frames: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    fps: float

    def as_dict(self) -> dict:
        return asdict(self)


def percentile(values: Sequence[float], q: float) -> float:
    """The q-quantile of already-collected samples, nearest-rank. Small sample sizes
    make interpolation pointless here, so the nearest actual measurement is returned."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def measure(
    predict: Callable[[object], object],
    inputs: Sequence,
    backend: str,
    warmup: int = 10,
) -> LatencyStats:
    """Time `predict` over each input once, after a warmup, and summarise the latency."""
    if not inputs:
        raise ValueError("no inputs to benchmark")

    for _ in range(warmup):
        predict(inputs[0])

    latencies: list[float] = []
    for item in inputs:
        start = time.perf_counter()
        predict(item)
        latencies.append((time.perf_counter() - start) * 1000.0)

    med = median(latencies)
    return LatencyStats(
        backend=backend,
        frames=len(latencies),
        mean_ms=round(mean(latencies), 2),
        median_ms=round(med, 2),
        p90_ms=round(percentile(latencies, 0.9), 2),
        fps=round(1000.0 / med, 1) if med > 0 else 0.0,
    )
