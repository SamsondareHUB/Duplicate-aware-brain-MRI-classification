"""Generate arm-A / arm-B split manifests, exposure tables and split_validation.json. No training here.
python scripts/make_splits.py [--seed 11]
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.splits import (make_image_level_folds, fold_class_matrix, expected_same_fold, exposure_table,
                            patient_overlap_stats, FOLDS)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "data/splits")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str})
    assert m.readable.all() and len(m) == 3064
    m = m.sort_values("image_id").reset_index(drop=True)
    ids, lab, fa, pid = m.image_id.values, m.label.values.astype(int), m.fold.values.astype(int), m.patient_id.values

    # patient -> exactly one class (needed for the class-stratified patient bootstrap)
    pl = m.groupby("patient_id").label.nunique()
    one_class = bool((pl == 1).all())

    fb = make_image_level_folds(ids, lab, fa, a.seed)  # patient_id NOT passed
    base = pd.DataFrame({"image_id": ids, "label": lab, "label_name": m.label_name, "patient_id": pid})
    A = base.assign(fold=fa); B = base.assign(fold=fb)
    fA, fB = out / "patient_disjoint_original.csv", out / f"imagelevel_seed{a.seed}.csv"
    A.to_csv(fA, index=False); B.to_csv(fB, index=False)

    eA, eB = exposure_table(ids, pid, fa), exposure_table(ids, pid, fb)
    eA.insert(0, "arm", "A_patient_disjoint"); eB.insert(0, "arm", f"B_imagelevel_seed{a.seed}")
    eA.merge(base[["image_id", "label"]], on="image_id").to_csv(out / "exposure_A.csv", index=False)
    eB.merge(base[["image_id", "label"]], on="image_id").to_csv(out / f"exposure_B_seed{a.seed}.csv", index=False)

    MA, MB = fold_class_matrix(lab, fa), fold_class_matrix(lab, fb)
    obs_same = int((fa == fb).sum()); exp_same, exp_frac = expected_same_fold(lab, fa)
    sims = np.array([int((make_image_level_folds(ids, lab, fa, 1000 + k) == fa).sum()) for k in range(2000)])
    # regenerate for determinism check
    fb2 = make_image_level_folds(ids, lab, fa, a.seed)

    def edist(e):
        multi = e[~e.single_slice_patient]
        q = multi.frac_other_train.quantile([0, .05, .25, .5, .75, .95, 1]).round(4)
        return {"n_images": int(len(e)), "n_single_slice_patient_images": int(e.single_slice_patient.sum()),
                "has_same_patient_train_slice_images": int(e.has_same_patient_train_slice.sum()),
                "has_same_patient_train_slice_fraction": round(float(e.has_same_patient_train_slice.mean()), 4),
                "frac_other_train_mean_excl_single": round(float(multi.frac_other_train.mean()), 4) if len(multi) else None,
                "frac_other_train_quantiles_excl_single": {str(k): float(v) for k, v in q.items()},
                "n_other_train_mean": round(float(e.n_other_train.mean()), 3),
                "n_other_train_quantiles": {str(k): float(v) for k, v in e.n_other_train.quantile([0, .25, .5, .75, 1]).items()},
                "n_other_train_max": int(e.n_other_train.max())}

    checks = {
        "A_rows_3064_each_image_once": bool(len(A) == 3064 and A.image_id.is_unique),
        "B_rows_3064_each_image_once": bool(len(B) == 3064 and B.image_id.is_unique),
        "A_B_folds_are_1_to_5": bool(set(fa) == set(FOLDS) and set(fb) == set(FOLDS)),
        "B_fold_class_matrix_equals_A": bool((MA.values == MB.values).all()),
        "B_fold_sizes_equal_A": bool((MA.sum(1).values == MB.sum(1).values).all()),
        "A_equals_manifest_original_folds": bool((A.fold.values == m.fold.values).all()),
        "A_patient_in_multiple_folds_zero": patient_overlap_stats(pid, fa)["patients_in_more_than_one_test_fold"] == 0,
        "A_exposure_all_zero": bool((eA.n_other_train == 0).all()),
        "B_deterministic_regeneration_identical": bool((fb == fb2).all()),
        "each_patient_has_exactly_one_class": one_class,
        "B_differs_from_A": bool(obs_same < len(ids)),
    }
    import numpy, pandas, platform
    V = {"seed": a.seed, "numpy": numpy.__version__, "pandas": pandas.__version__, "python": platform.python_version(),
         "rng": "numpy.random.default_rng(SeedSequence([seed, class_label]))"}
    res = {
        "all_checks_pass": bool(all(checks.values())), "checks": checks, "versions": V,
        "sha256": {fA.name: sha(fA), fB.name: sha(fB)},
        "fold_sizes": {"A": {int(f): int(v) for f, v in MA.sum(1).items()}, "B": {int(f): int(v) for f, v in MB.sum(1).items()}},
        "fold_x_class_A": {int(f): {int(c): int(MA.loc[f, c]) for c in MA.columns} for f in MA.index},
        "fold_x_class_B": {int(f): {int(c): int(MB.loc[f, c]) for c in MB.columns} for f in MB.index},
        "same_fold_A_vs_B": {"observed_images": obs_same, "observed_fraction": round(obs_same / len(ids), 4),
                             "null_expected_images_matched_margins": round(exp_same, 2),
                             "null_expected_fraction_matched_margins": round(exp_frac, 4),
                             "null_simulated_mean_images_2000_seeds_1000_to_2999": round(float(sims.mean()), 2),
                             "null_simulated_sd_images": round(float(sims.std(ddof=1)), 2),
                             "observed_z_vs_simulated_null": round(float((obs_same - sims.mean()) / sims.std(ddof=1)), 2),
                             "note": "informational, not pass/fail"},
        "patient_overlap_A": patient_overlap_stats(pid, fa), "patient_overlap_B": patient_overlap_stats(pid, fb),
        "exposure_A": edist(eA), "exposure_B": edist(eB),
    }
    (out / "split_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)); sys.exit(0 if res["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
