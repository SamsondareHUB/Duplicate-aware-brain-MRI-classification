"""Prediction-only reproduction of committed results (no raw MRI, no torch)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce_from_predictions as R


def test_committed_predictions_reproduce_every_reported_number():
    assert R.main() == 0
    assert not [c for c in R.LOG if c["status"] == "FAIL"]
    assert len(R.LOG) >= 150
