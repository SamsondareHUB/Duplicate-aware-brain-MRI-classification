import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_patient_ids as V


def test_normalization_rules():
    assert V.normalize_pid("MR024780B") == "MR024780"
    assert V.normalize_pid("MR024780") == "MR024780"          # bare MR id unchanged
    assert V.normalize_pid("100360") == "100360"              # numeric untouched
    assert V.normalize_pid("MR0247800B") == "MR0247800B"      # wrong digit count: not stripped
    assert V.normalize_pid("MR024780BC") == "MR024780BC"      # only ONE trailing capital stripped
    assert V.normalize_pid("MR024780b") == "MR024780b"        # lowercase not matched
    assert V.normalize_pid("ABC1234B") == "ABC1234B"          # non-MR untouched
    assert [V.id_format(x) for x in ("100360", "MR033420", "MR024780D", "X1")] == ["numeric", "MR_no_suffix", "MR_with_letter_suffix", "other"]
    assert V.normalize_pid(V.normalize_pid("MR024780B")) == V.normalize_pid("MR024780B")  # idempotent


def test_cross_fold_and_exposure_detection_on_toy_data():
    s = pd.DataFrame({"image_id": range(1, 7), "patient_id": ["MR000001", "MR000001B", "MR000001B", "MR000002", "MR000002C", "100"],
                      "label": [2] * 6, "fold": [1, 2, 2, 3, 3, 1]})
    s["stem"] = s.patient_id.map(V.normalize_pid)
    g = V.group_table(s, "stem")
    assert bool(g.loc["MR000001", "spans_folds"]) and not bool(g.loc["MR000002", "spans_folds"])
    assert list(V.exposure(s, "stem")) == [True, True, True, False, False, False]
    assert not V.exposure(s, "patient_id").any()


def test_validation_outputs_and_inputs_unchanged():
    watched = sorted((ROOT / "results/multimodel").glob("oof_predictions_*.csv")) + sorted((ROOT / "results/sanity").glob("oof_predictions_*.csv")) \
        + sorted((ROOT / "data/splits").glob("*.csv"))
    h = lambda: [hashlib.sha256(p.read_bytes()).hexdigest() for p in watched]
    before = h(); r = V.main(); assert h() == before and len(watched) == 14
    n = r["normalization"]
    assert (n["unique_raw_pids"], n["unique_normalized_stems"], n["suffix_sharing_groups"], n["affected_raw_pids"]) == (233, 209, 18, 42)
    assert (n["affected_images"], n["stems_spanning_folds"], n["images_in_cross_fold_stems"]) == (811, 16, 734)
    assert n["all_sharing_groups_single_class"]
    ii = r["identity_interpretations"]
    assert ii["raw_pid"]["groups_crossing_folds"] == 0 and ii["stem"]["test_images_potentially_exposed"] == 734
    assert sum(r["formats_raw_pids"]["by_format"].values()) == 233
    # CSV <-> JSON consistency and manifest alignment
    sg = pd.read_csv(ROOT / "results/audit/patient_id/suffix_sharing_groups.csv")
    assert len(sg) == 18 and sg.n_raw_pids.sum() == 42 and sg.n_images.sum() == 811 and sg.spans_folds.sum() == 16
    man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str})
    assert man.patient_id.nunique() == 233 and set(man[man.patient_id.isin(sum((x.split("|") for x in sg.raw_pids), []))].label_name) == {"glioma"}
