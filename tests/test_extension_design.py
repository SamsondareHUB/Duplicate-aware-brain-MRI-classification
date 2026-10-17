"""Lightweight tests for extension split-seed and exposure design logic (no data/torch required beyond committed manifests)."""
import json, sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "experiments/extension"
sys.path.insert(0, str(EXT)); sys.path.insert(0, str(ROOT / "src"))
import make_split_seeds as MS
import make_exposure as ME
from ext_specs import build_queue, get_spec

HIST_A = ROOT / "data/splits/patient_disjoint_original.csv"


@pytest.mark.parametrize("seed", MS.SPLIT_SEEDS)
def test_split_seed_replicates(seed):
    A = pd.read_csv(EXT / f"splits/split_A_seed{seed}.csv", dtype={"patient_id": str})
    B = pd.read_csv(EXT / f"splits/split_B_seed{seed}.csv", dtype={"patient_id": str})
    assert len(A) == 3064 and A.image_id.is_unique and B.image_id.is_unique and A.patient_id.nunique() == 233
    assert (A.groupby("patient_id").fold.nunique() == 1).all()                      # patient-disjoint
    assert (pd.crosstab(A.fold, A.label).values == pd.crosstab(B.fold, B.label).values).all()   # matched margins
    assert (B.groupby("patient_id").fold.nunique() > 1).sum() > 200                  # image-level mixes patients
    assert seed != 11


def test_seeds_regenerate_identically():
    man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
    a, b = MS.build(MS.SPLIT_SEEDS[0], man)
    saved = pd.read_csv(EXT / f"splits/split_A_seed{MS.SPLIT_SEEDS[0]}.csv")
    assert (saved.fold.values == a.fold.values).all()


def test_historical_splits_untouched():
    val = json.loads((ROOT / "data/splits/split_validation.json").read_text())["sha256"]
    import hashlib
    assert hashlib.sha256(HIST_A.read_bytes()).hexdigest() == val["patient_disjoint_original.csv"]


@pytest.mark.parametrize("fold", ME.EXPOSURE_FOLDS)
def test_exposure_design_invariants(fold):
    D = ME.design(fold)
    ft = set(D["fixed_test"].image_id)
    donors = {i for v in D["donors"].values() for i in v}
    assert not (ft & donors)                                          # donors are not test images
    assert ft | donors == set(D["test"].image_id)
    cond = pd.read_csv(EXT / "exposure/conditions.csv")
    for _, r in cond[cond.fold == fold].iterrows():
        tr = set(pd.read_csv(EXT / f"exposure/train_{r.condition}.csv").image_id)
        assert not (tr & ft)                                           # NEVER a fixed-test image in training
        te_pat = set(D["fixed_test"].patient_id)
        man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str}).set_index("image_id")
        overlap = set(man.loc[sorted(tr), "patient_id"]) & te_pat
        if r.kind in ("CLEAN", "CTRL"): assert not overlap            # clean/control stay patient-disjoint
        else: assert len(overlap) == len(D["selected"][r.level])
    for L in ME.LEVELS:                                               # matched size and class counts
        a = cond[(cond.fold == fold) & (cond.level == L) & (cond.kind == "CTRL")].iloc[0]
        b = cond[(cond.fold == fold) & (cond.level == L) & (cond.kind == "SIB")].iloc[0]
        assert a.n_train == b.n_train and a.class_counts == b.class_counts
    assert set(D["selected"][25]) <= set(D["selected"][50]) <= set(D["selected"][75]) <= set(D["selected"][100])
    ctrl_pat = {q for q, _ in D["match"].values()}
    assert not (ctrl_pat & set(D["base"].patient_id))                 # reservoir patients removed from base
    assert D["base"].patient_id.isin(D["test"].patient_id).sum() == 0


def test_queue_is_well_formed():
    q = build_queue()
    assert len(q) == 172 and len(set(q)) == 172
    s = get_spec(q[0]); assert not (set(s["train_ids"]) & set(s["test_ids"]))
    for rid in ("split_densenet121_B_seed303_fold5", "tseed_resnet50_A_ts2_fold1"):
        s = get_spec(rid); assert not (set(s["train_ids"]) & set(s["test_ids"])) and len(s["train_ids"]) + len(s["test_ids"]) == 3064
