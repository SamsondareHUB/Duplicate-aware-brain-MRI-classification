import pandas as pd


def check_oof(a: pd.DataFrame, b: pd.DataFrame, expected_n=3064):
    """Hard assertions: exactly one OOF prediction per image in each arm; identical image sets and labels across arms."""
    for name, d in (("A", a), ("B", b)):
        assert len(d) == expected_n, f"arm {name}: {len(d)} predictions, expected {expected_n}"
        assert d.image_id.is_unique, f"arm {name}: duplicate image ids"
    assert set(a.image_id) == set(b.image_id), "arms cover different images"
    m = a[["image_id", "true_class"]].merge(b[["image_id", "true_class"]], on="image_id", suffixes=("_a", "_b"))
    assert (m.true_class_a == m.true_class_b).all(), "labels differ between arms"
