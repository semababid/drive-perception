"""Benchmark tests. The timing itself needs models, but the summary maths is pure and
worth pinning: a wrong percentile or an fps computed from the mean would quietly
misstate every number in the results table."""

import time

import pytest

from drive_perception.benchmark import LatencyStats, measure, percentile


def test_percentile_nearest_rank():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 0.0) == 10
    assert percentile(values, 1.0) == 50
    assert percentile(values, 0.5) == 30


def test_percentile_empty_is_zero():
    assert percentile([], 0.9) == 0.0


def test_percentile_is_order_independent():
    assert percentile([50, 10, 40, 20, 30], 0.9) == 50


def test_measure_reports_frames_and_positive_fps():
    stats = measure(lambda _: time.sleep(0.002), list(range(15)), "fake", warmup=2)
    assert isinstance(stats, LatencyStats)
    assert stats.frames == 15
    assert stats.fps > 0
    # ~2 ms per call, so the median should sit near there, well above zero.
    assert stats.median_ms >= 1.0


def test_measure_fps_matches_median():
    stats = measure(lambda _: time.sleep(0.005), list(range(10)), "fake", warmup=1)
    assert stats.fps == pytest.approx(1000.0 / stats.median_ms, rel=0.01)


def test_measure_requires_inputs():
    with pytest.raises(ValueError):
        measure(lambda _: None, [], "fake")
