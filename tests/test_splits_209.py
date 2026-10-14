"""Integrity tests for the PROPOSED 209-group split manifests (no training)."""
import hashlib, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPL = ROOT / "data/splits/proposed_209"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from make_splits_209 import minmove
from validate_patient_ids import normalize_pid


def rd(n): return pd.read_csv(SPL / n, dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)


def test_stem_disjoint_and_counts():
    for n in ("PROPOSED_A209_minmove.csv", "PROPOSED_A209_sgkf_alternative.csv"):
        a = rd(n)
        assert len(a) == 3064 and a.image_id.is_unique and a.stem.nunique() == 209 and a.patient_id.nunique() == 233
        assert (a.groupby("stem").fold.nunique() == 1).all() and (a.groupby("stem").label.nunique() == 1).all()


def test_b209_matches_a209_margins():
    a, b = rd("PROPOSED_A209_minmove.csv"), rd("PROPOSED_B209_imagelevel_seed11.csv")
    assert (pd.crosstab(a.fold, a.label).values == pd.crosstab(b.fold, b.label).values).all()
    assert (b.groupby("patient_id").fold.nunique() > 1).sum() > 100  # B209 is image-level (patients cross folds)


def test_only_spanning_stems_move_and_deterministic():
    o = pd.read_csv(ROOT / "data/splits/patient_disjoint_original.csv", dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
    a = rd("PROPOSED_A209_minmove.csv"); o["stem"] = o.patient_id.map(normalize_pid)
    span = o.groupby("stem").fold.nunique(); moved = o.stem[o.fold != a.fold].unique()
    assert set(moved) <= set(span[span > 1].index) and len(span[span > 1]) == 16
    assert (minmove(o) == a.fold.to_numpy()).all()


def test_historical_splits_untouched_and_manifests_hashed():
    v = json.loads((ROOT / "data/splits/split_validation.json").read_text())["sha256"]
    for n, h in v.items(): assert hashlib.sha256((ROOT / "data/splits" / n).read_bytes()).hexdigest() == h
    pv = json.loads((SPL / "PROPOSED_split_validation_209.json").read_text())
    assert pv["status"].startswith("PROPOSED")
    for n, h in pv["sha256"].items(): assert hashlib.sha256((SPL / n).read_bytes()).hexdigest() == h


def test_dry_run_runs_and_train_refused_without_approval():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_retrain_209.py"), "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "distinct=20" in r.stdout and "no_response=40" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_retrain_209.py"), "--train"], capture_output=True, text=True)
    assert r.returncode != 0
