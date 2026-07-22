"""Fast checks for the split logic. It must be deterministic and lossless: every frame
lands on exactly one side, and the same seed always gives the same partition."""

from drive_perception.data.split import split_items


def test_split_is_disjoint_and_complete():
    items = list(range(10))
    train, val = split_items(items, val_frac=0.2, seed=42)
    assert len(val) == 2
    assert len(train) == 8
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == items


def test_split_is_deterministic():
    items = list(range(100))
    a = split_items(items, val_frac=0.2, seed=42)
    b = split_items(items, val_frac=0.2, seed=42)
    assert a == b


def test_seed_changes_the_partition():
    items = list(range(100))
    _, val_a = split_items(items, val_frac=0.2, seed=1)
    _, val_b = split_items(items, val_frac=0.2, seed=2)
    assert val_a != val_b
