"""Training tests. The device choice is the piece worth pinning down, since getting it
wrong silently drops a run onto the CPU and makes training take hours instead of minutes."""

from drive_perception.train import pick_device


def test_cuda_wins_when_present():
    assert pick_device(cuda_available=True, mps_available=True) == "cuda"


def test_mps_used_on_apple_silicon():
    assert pick_device(cuda_available=False, mps_available=True) == "mps"


def test_cpu_is_the_fallback():
    assert pick_device(cuda_available=False, mps_available=False) == "cpu"
