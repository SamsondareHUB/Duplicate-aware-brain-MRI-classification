"""Exploratory sensitivity: do suffix-bearing MR patient IDs (e.g. MR024780B/D/E/G) share a stem that spans folds, and is
accuracy higher for images whose same-stem sibling IDs lie in other (=training) folds under the patient-disjoint protocol A?

Whether same-stem IDs are the same individual is NOT documented by the dataset; this is a provenance sensitivity check,
not a claim of leakage. Prediction-only. Writes results/audit/pid_suffix_sensitivity.json.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
s = pd.read_csv(ROOT / "data/splits/patient_disjoint_original.csv", dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
s["stem"] = np.where(s.patient_id.str.startswith("MR"), s.patient_id.str.replace(r"[A-Za-z]+$", "", regex=True), s.patient_id)
g = s.groupby("stem").agg(n_pid=("patient_id", "nunique"), n_folds=("fold", "nunique"), n_img=("fold", "size"))
multi = g[g.n_pid > 1]
out = {"n_distinct_patient_ids": int(s.patient_id.nunique()), "n_distinct_stems": int(s.stem.nunique()),
       "stems_with_multiple_ids": int(len(multi)), "ids_in_those_stems": int(multi.n_pid.sum()), "images_in_those_stems": int(multi.n_img.sum()),
       "stems_spanning_multiple_folds_under_A": int((multi.n_folds > 1).sum()),
       "images_in_stems_spanning_folds": int(multi[multi.n_folds > 1].n_img.sum())}
# per-image exposure: a different patient ID with the same stem lies in a different (= training) fold
pid_fold = s.groupby(["stem", "patient_id"]).fold.first().reset_index()
exp = []
for st, fo, pi in zip(s.stem, s.fold, s.patient_id):
    o = pid_fold[(pid_fold.stem == st) & (pid_fold.patient_id != pi)]
    exp.append(bool((o.fold != fo).any()))
s["exposed"] = exp
mr_gl = s[s.patient_id.str.startswith("MR")]
out["MR_images"] = int(len(mr_gl)); out["MR_images_exposed_to_sibling_in_other_fold"] = int(mr_gl.exposed.sum())
rng = np.random.default_rng(7); res = {}
stems = s.stem.unique()
for m in MODELS:
    f = ROOT / ("results/sanity/oof_predictions_A.csv" if m == "resnet50" else f"results/multimodel/oof_predictions_{m}_A.csv")
    d = pd.read_csv(f).sort_values("image_id").reset_index(drop=True)
    corr = (d.pred_class.to_numpy() == d.true_class.to_numpy())
    mk = s.patient_id.str.startswith("MR").to_numpy()
    ex = s.exposed.to_numpy()
    a, b = corr[mk & ex], corr[mk & ~ex]
    # stem-cluster bootstrap of difference in accuracy (exposed - unexposed) among MR (all glioma) images
    idx_by_stem = {st: np.where(s.stem.to_numpy() == st)[0] for st in stems if mk[s.stem.to_numpy() == st].any()}
    keys = list(idx_by_stem); diffs = []
    for _ in range(2000):
        pick = rng.integers(0, len(keys), len(keys)); ii = np.concatenate([idx_by_stem[keys[k]] for k in pick])
        e, u = corr[ii][ex[ii]], corr[ii][~ex[ii]]
        if len(e) and len(u): diffs.append(e.mean() - u.mean())
    res[m] = {"acc_exposed": float(a.mean()), "n_exposed": int(len(a)), "acc_unexposed": float(b.mean()), "n_unexposed": int(len(b)),
              "diff": float(a.mean() - b.mean()), "stem_cluster_ci": [float(x) for x in np.percentile(diffs, [2.5, 97.5])]}
out["MR_glioma_accuracy_by_sibling_exposure_protocol_A"] = res
(ROOT / "results/audit/pid_suffix_sensitivity.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
