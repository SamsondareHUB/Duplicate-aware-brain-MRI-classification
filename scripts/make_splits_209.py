"""PROPOSED / NOT YET USED FOR TRAINING. Build stem-disjoint (209-group) split manifests. No training, no GPU, no raw MRI.

Identity rule (hypothetical, same as validate_patient_ids.py): ^MR\\d{6}[A-Z]$ loses its trailing capital letter; all else unchanged.
Primary plan  'minmove': start from the frozen original folds; every stem that spans folds is moved WHOLE to the fold already holding
              most of its images (tie -> lowest fold number). Folds of all other stems are untouched.
Alternative   'sgkf'   : sklearn StratifiedGroupKFold(5, shuffle, random_state=11) over stems, for comparison only.
Matched image-level arm 'B209': make_image_level_folds(seed 11) on the minmove fold x class matrix (same function as PR #1).
Outputs -> data/splits/proposed_209/ (never overwrites data/splits/*.csv).
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from leakage.splits import make_image_level_folds, fold_class_matrix, FOLDS, patient_overlap_stats
from validate_patient_ids import normalize_pid

OUT = ROOT / "data/splits/proposed_209"
STATUS = "PROPOSED / NOT YET USED FOR TRAINING"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def minmove(s):
    """Balanced minimal-move: non-spanning stems keep their original fold. Spanning stems (largest first, stem name tiebreak) go to
    the fold they already occupy that minimises sum of squared deviations of fold x class counts from the ORIGINAL fold x class
    matrix (tie -> lowest fold). Deterministic; no RNG."""
    f = s.fold.to_numpy().copy(); lab = s.label.to_numpy()
    nf = s.groupby("stem").fold.transform("nunique").to_numpy()
    target = fold_class_matrix(lab, s.fold.to_numpy()).to_numpy().astype(float)
    cur = fold_class_matrix(lab[nf == 1], f[nf == 1]).to_numpy().astype(float)
    span = s[nf > 1].groupby("stem").agg(n=("image_id", "size"), idx=("image_id", lambda x: list(x.index)))
    for stem in sorted(span.index, key=lambda k: (-span.loc[k, "n"], k)):
        idx = span.loc[stem, "idx"]; c = lab[idx][0] - 1; occ = sorted(set(f[idx]))
        def cost(k):
            t = cur.copy(); t[k - 1, c] += len(idx); return float(((t - target) ** 2).sum())
        best = min(occ, key=lambda k: (cost(k), k))
        cur[best - 1, c] += len(idx); f[idx] = best
    return f


def sgkf(s):
    from sklearn.model_selection import StratifiedGroupKFold
    st = s.groupby("stem").agg(label=("label", "first"), n=("image_id", "size")).reset_index()
    # group-level: stratify on stem class, weight by images via repetition-free approach on image rows
    f = np.zeros(len(s), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=11).split(s, s.label, s.stem)):
        f[te] = k + 1
    return f


def summarize(s, col):
    M = fold_class_matrix(s.label.to_numpy(), s[col].to_numpy())
    return {"fold_sizes": {int(k): int(v) for k, v in M.sum(1).items()},
            "fold_x_class": {int(f): {int(c): int(M.loc[f, c]) for c in M.columns} for f in M.index}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT / "data/splits/patient_disjoint_original.csv", dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
    s["stem"] = s.patient_id.map(normalize_pid)
    s = s.rename(columns={"fold": "fold_original"})
    s["fold_minmove"] = minmove(s.rename(columns={"fold_original": "fold"}))
    s["fold_sgkf"] = sgkf(s)
    ids, lab = s.image_id.to_numpy(), s.label.to_numpy()
    s["fold_B209"] = make_image_level_folds(ids, lab, s.fold_minmove.to_numpy(), 11)
    base = ["image_id", "label", "label_name", "patient_id", "stem"]
    files = {}
    for name, col in (("A209_minmove", "fold_minmove"), ("A209_sgkf_alternative", "fold_sgkf"), ("B209_imagelevel_seed11", "fold_B209")):
        d = s[base].assign(fold=s[col]); p = OUT / f"PROPOSED_{name}.csv"; d.to_csv(p, index=False); files[p.name] = sha(p)
    # mapping tables
    mp = s.groupby("patient_id").agg(stem=("stem", "first"), n_images=("image_id", "size"), label=("label", "first"),
                                     fold_original=("fold_original", "first"), fold_minmove=("fold_minmove", "first")).reset_index()
    mp["merged_with_other_pid"] = mp.groupby("stem").patient_id.transform("size") > 1
    mp["moved_by_minmove"] = mp.fold_original != mp.fold_minmove
    mp.to_csv(OUT / "PROPOSED_pid_to_stem_mapping.csv", index=False)
    moved = s[s.fold_original != s.fold_minmove]
    mv = moved.groupby(["stem", "patient_id"]).agg(n_images=("image_id", "size"), from_fold=("fold_original", "first"),
                                                   to_fold=("fold_minmove", "first")).reset_index()
    mv.to_csv(OUT / "PROPOSED_minmove_moved_pids.csv", index=False)

    def integrity(col):
        g = s.groupby("stem")[col].nunique(); gp = s.groupby("patient_id")[col].nunique()
        return {"stems_in_more_than_one_fold": int((g > 1).sum()), "raw_pids_in_more_than_one_fold": int((gp > 1).sum()),
                "n_stems": int(s.stem.nunique()), "n_raw_pids": int(s.patient_id.nunique()), "rows": int(len(s)),
                "each_image_once": bool(s.image_id.is_unique), "folds": sorted(map(int, s[col].unique())),
                "stem_single_class": bool((s.groupby("stem").label.nunique() == 1).all())}
    res = {"status": STATUS, "rule": "strip one trailing [A-Z] from ^MR\\d{6}[A-Z]$ only",
           "original": summarize(s, "fold_original") | {"stem_integrity": integrity("fold_original")},
           "A209_minmove": summarize(s, "fold_minmove") | {"integrity": integrity("fold_minmove"),
               "n_images_moved": int(len(moved)), "frac_images_moved": round(len(moved) / len(s), 4),
               "n_stems_moved": int(mv.stem.nunique()), "n_raw_pids_moved": int(len(mv)),
               "images_moved_by_class": {str(k): int(v) for k, v in moved.label.value_counts().sort_index().items()}},
           "A209_sgkf_alternative": summarize(s, "fold_sgkf") | {"integrity": integrity("fold_sgkf"),
               "frac_images_same_fold_as_original": round(float((s.fold_sgkf == s.fold_original).mean()), 4),
               "note": "fold numbers arbitrary; not comparable to original folds; fewer-constraints alternative"},
           "B209_imagelevel_seed11": summarize(s, "fold_B209") | {
               "fold_x_class_equals_A209_minmove": bool((fold_class_matrix(lab, s.fold_B209.to_numpy()).values ==
                                                         fold_class_matrix(lab, s.fold_minmove.to_numpy()).values).all()),
               "deterministic": bool((make_image_level_folds(ids, lab, s.fold_minmove.to_numpy(), 11) == s.fold_B209.to_numpy()).all())},
           "sha256": files,
           "frozen_inputs_sha256": {"patient_disjoint_original.csv": sha(ROOT / "data/splits/patient_disjoint_original.csv"),
                                    "imagelevel_seed11.csv": sha(ROOT / "data/splits/imagelevel_seed11.csv")}}
    (OUT / "PROPOSED_split_validation_209.json").write_text(json.dumps(res, indent=1))
    (OUT / "STATUS.txt").write_text(f"{STATUS}\nNo model has been trained on these manifests. See reports/RETRAINING_PLAN_209_GROUPS.md.\n")
    print(json.dumps({k: res[k] for k in ("A209_minmove", "A209_sgkf_alternative")}, indent=None)[:3000])


if __name__ == "__main__":
    main()
