"""Frozen new split-seed replicates (233-ID identity). Seeds are pre-specified and published BEFORE any new training:
SPLIT_SEEDS = 101, 202, 303 (distinct from the historical image-level seed 11; the historical patient-disjoint folds come
from the dataset's cvind and have no seed).

Arm A_s: class-stratified patient-grouped 5-fold (sklearn StratifiedGroupKFold, shuffle, random_state=seed) over 233 PIDs.
Arm B_s: image-level folds with fold x class counts equal to A_s exactly (repo make_image_level_folds, same seed).
Output: experiments/extension/splits/split_{A,B}_seed{s}.csv + split_validation_ext.json. Historical splits untouched.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from leakage.splits import make_image_level_folds, fold_class_matrix, patient_overlap_stats

SPLIT_SEEDS = (101, 202, 303)
OUT = ROOT / "experiments/extension/splits"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(seed, man):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold = np.zeros(len(man), int)
    for k, (_, te) in enumerate(sgkf.split(man.image_id, man.label, groups=man.patient_id), start=1):
        fold[te] = k
    a = man[["image_id", "label", "label_name", "patient_id"]].copy(); a["fold"] = fold
    b = a.copy(); b["fold"] = make_image_level_folds(a.image_id, a.label, a.fold, seed)
    return a, b


def main():
    man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
    assert len(man) == 3064 and man.patient_id.nunique() == 233
    OUT.mkdir(parents=True, exist_ok=True)
    val = {"split_seeds": list(SPLIT_SEEDS), "sets": {}}
    hist = pd.read_csv(ROOT / "data/splits/patient_disjoint_original.csv")
    for s in SPLIT_SEEDS:
        a, b = build(s, man)
        pa, pb = OUT / f"split_A_seed{s}.csv", OUT / f"split_B_seed{s}.csv"
        a.to_csv(pa, index=False); b.to_csv(pb, index=False)
        ma, mb = fold_class_matrix(a.label, a.fold), fold_class_matrix(b.label, b.fold)
        st_a, st_b = patient_overlap_stats(a.patient_id, a.fold), patient_overlap_stats(b.patient_id, b.fold)
        checks = {"A_zero_patient_overlap": st_a["patients_in_more_than_one_test_fold"] == 0,
                  "all_3064_once_A": bool(a.image_id.is_unique and len(a) == 3064),
                  "all_3064_once_B": bool(b.image_id.is_unique and len(b) == 3064),
                  "margins_matched": bool((ma.values == mb.values).all()),
                  "A_patients_cross_folds_in_B": st_b["patients_in_more_than_one_test_fold"],
                  "A_differs_from_historical": bool((a.fold.values != hist.sort_values("image_id").fold.values).any())}
        assert checks["A_zero_patient_overlap"] and checks["margins_matched"] and checks["all_3064_once_A"] and checks["all_3064_once_B"]
        val["sets"][str(s)] = {"A_sha256": sha(pa), "B_sha256": sha(pb), "fold_class_counts_A": ma.to_dict(),
                               "fold_sizes": ma.sum(1).to_dict(), "checks": checks}
    (OUT / "split_validation_ext.json").write_text(json.dumps(val, indent=1))
    print(json.dumps({s: v["checks"] | {"sizes": v["fold_sizes"]} for s, v in val["sets"].items()}, indent=1))


if __name__ == "__main__":
    main()
