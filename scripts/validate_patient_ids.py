"""Patient-ID format / suffix-sharing validation and prediction-only sensitivity analysis. No raw MRI, no training.

HYPOTHETICAL normalisation (documented, deliberately narrow): an ID matching ^MR\\d{6}[A-Z]$ loses its single trailing capital letter;
every other ID (numeric IDs, bare MR\\d{6}, anything else) is unchanged. This is a sensitivity device, NOT a claim that suffixed IDs are one person.
Outputs -> results/audit/patient_id/. Saved predictions / splits are only read.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audit/patient_id"
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
CLS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
SUFFIX_RE = re.compile(r"^(MR\d{6})([A-Z])$")


def id_format(pid):
    if re.fullmatch(r"\d+", pid): return "numeric"
    if re.fullmatch(r"MR\d+", pid): return "MR_no_suffix"
    if SUFFIX_RE.match(pid): return "MR_with_letter_suffix"
    return "other"


def normalize_pid(pid):
    m = SUFFIX_RE.match(pid)
    return m.group(1) if m else pid


def load_split(name):
    return pd.read_csv(ROOT / "data/splits" / name, dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)


def group_table(s, key):
    """per-identity table: n_images, classes, folds, spans_folds."""
    g = s.groupby(key).agg(n_images=("image_id", "size"), classes=("label", lambda x: "|".join(CLS[v] for v in sorted(set(x)))),
                           folds=("fold", lambda x: "|".join(map(str, sorted(set(x))))), n_folds=("fold", "nunique"))
    g["spans_folds"] = g.n_folds > 1
    return g


def exposure(s, ident):
    """Image is 'potentially exposed' if another image of the SAME identity group but a DIFFERENT raw PID lies in another fold (=training)."""
    fold, pid, ids = s.fold.to_numpy(), s.patient_id.to_numpy(), s[ident].to_numpy()
    out = np.zeros(len(s), bool)
    for g in np.unique(ids):
        ix = np.where(ids == g)[0]
        for i in ix:
            out[i] = bool(((pid[ix] != pid[i]) & (fold[ix] != fold[i])).any())
    return out


def metrics(cm):
    tp = np.diag(cm); rec = tp / cm.sum(1); prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return rec.mean(), f1.mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A, B = load_split("patient_disjoint_original.csv"), load_split("imagelevel_seed11.csv")
    man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
    assert (man.patient_id == A.patient_id).all() and (man.fold == A.fold).all() and (man.label == A.label).all()
    A["stem"] = A.patient_id.map(normalize_pid); A["fmt"] = A.patient_id.map(id_format)
    res = {"n_image_records": len(A)}
    # --- A. formats
    pids = A.drop_duplicates("patient_id")
    fm = pids.fmt.value_counts().to_dict()
    ex = {f: sorted(pids[pids.fmt == f].patient_id)[:4] for f in fm}
    res["formats_raw_pids"] = {"unique_raw_pids": int(len(pids)), "by_format": fm, "examples": ex,
                               "images_by_format": A.fmt.value_counts().to_dict(),
                               "classes_by_format": {f: A[A.fmt == f].label.map(CLS).value_counts().to_dict() for f in fm}}
    assert (A.groupby("patient_id").label.nunique() == 1).all()
    # --- B/C. stems
    gt = group_table(A, "stem")
    raw_per_stem = A.groupby("stem").patient_id.agg(lambda x: "|".join(sorted(set(x))))
    gt["raw_pids"] = raw_per_stem; gt["n_raw_pids"] = raw_per_stem.str.count(r"\|") + 1
    gt = gt.reset_index()
    sg = gt[gt.n_raw_pids > 1].copy()
    sg["single_class"] = ~sg.classes.str.contains(r"\|")
    sg = sg[["stem", "raw_pids", "n_raw_pids", "n_images", "classes", "folds", "n_folds", "spans_folds"]].sort_values("stem")
    sg.to_csv(OUT / "suffix_sharing_groups.csv", index=False)
    cross = sg[sg.spans_folds]
    res["normalization"] = {"rule": "strip one trailing [A-Z] from ^MR\\d{6}[A-Z]$ only",
                            "unique_raw_pids": int(A.patient_id.nunique()), "unique_normalized_stems": int(A.stem.nunique()),
                            "suffix_sharing_groups": int(len(sg)), "affected_raw_pids": int(sg.n_raw_pids.sum()),
                            "affected_images": int(sg.n_images.sum()),
                            "stems_spanning_folds": int(len(cross)), "images_in_cross_fold_stems": int(cross.n_images.sum()),
                            "stems_not_spanning_folds_among_sharing": int((~sg.spans_folds).sum()),
                            "raw_pids_per_sharing_group_distribution": sg.n_raw_pids.value_counts().sort_index().to_dict(),
                            "all_sharing_groups_single_class": bool(all("|" not in c for c in sg.classes)),
                            "sharing_group_class_images": A[A.stem.isin(sg.stem)].label.map(CLS).value_counts().to_dict(),
                            "mr_suffix_pids_whose_bare_stem_pid_also_exists": int(sum(1 for p in pids.patient_id if SUFFIX_RE.match(p) and SUFFIX_RE.match(p).group(1) in set(pids.patient_id)))}
    # raw PID table
    rp = group_table(A, "patient_id").reset_index().rename(columns={"classes": "class"})
    rp["format"] = rp.patient_id.map(id_format); rp["stem"] = rp.patient_id.map(normalize_pid)
    rp.drop(columns=["n_folds", "spans_folds"]).to_csv(OUT / "raw_pid_table.csv", index=False)
    # --- impact: identity interpretations (protocol A)
    A["exp_raw"] = exposure(A, "patient_id"); A["exp_stem"] = exposure(A, "stem")
    for name, key, e in (("raw_pid", "patient_id", "exp_raw"), ("stem", "stem", "exp_stem")):
        g = group_table(A, key)
        res.setdefault("identity_interpretations", {})[name] = {
            "identity_groups": int(len(g)), "groups_crossing_folds": int(g.spans_folds.sum()),
            "images_in_crossing_groups": int(g[g.spans_folds].n_images.sum()),
            "test_images_potentially_exposed": int(A[e].sum()), "pct_of_3064": round(100 * A[e].mean(), 2),
            "exposed_by_class": A[A[e]].label.map(CLS).value_counts().to_dict(),
            "exposed_by_fold": {int(k): int(v) for k, v in A[A[e]].fold.value_counts().sort_index().items()}}
    # same under the image-level protocol (context): patients spanning folds
    res["protocol_B_context"] = {"raw_pids_spanning_folds": int(B.groupby("patient_id").fold.nunique().gt(1).sum())}
    # --- model-specific performance, affected vs unaffected
    y = A.label.to_numpy() - 1
    ex_s = A.exp_stem.to_numpy(); mr = A.fmt.ne("numeric").to_numpy(); gl = (y == 1)
    perf = {}
    for m in MODELS:
        f = ROOT / ("results/sanity/oof_predictions_A.csv" if m == "resnet50" else f"results/multimodel/oof_predictions_{m}_A.csv")
        d = pd.read_csv(f, dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
        assert (d.patient_id == A.patient_id).all() and (d.true_class == y).all() and (d.test_fold == A.fold).all()
        ok = (d.pred_class.to_numpy() == y)
        sub = {"exposed_by_stem(all glioma)": ok[ex_s], "glioma_not_exposed": ok[gl & ~ex_s], "glioma_MR_not_exposed": ok[gl & mr & ~ex_s],
               "glioma_numeric_ID": ok[gl & ~mr], "all_non_glioma": ok[~gl]}
        perf[m] = {k: {"n": int(len(v)), "accuracy": float(v.mean()), "errors": int((~v).sum())} for k, v in sub.items()}
        d_exposed = ok[ex_s].mean() - ok[gl & ~ex_s].mean()
        perf[m]["diff_exposed_minus_other_glioma"] = float(d_exposed)
    res["performance_protocol_A_affected_vs_unaffected"] = perf
    # --- weak indirect evidence from committed similarity tables (top-200 pairs + 19 cross-patient NCC>=0.90 pairs only)
    ev = {}
    for fn in ("top_similarity_pairs.csv", "cross_patient_high_similarity_pairs.csv"):
        t = pd.read_csv(ROOT / "reports/manifests" / fn, dtype={"patient_a": str, "patient_b": str})
        t = t[t.patient_a != t.patient_b]
        sib = t.patient_a.map(normalize_pid) == t.patient_b.map(normalize_pid)
        ev[fn] = {"cross_pid_pairs": int(len(t)), "same_stem_pairs": int(sib.sum()),
                  "max_ncc_full_same_stem": float(t[sib].ncc_full.max()) if sib.any() else None,
                  "max_ncc_full_cross_pid": float(t.ncc_full.max()) if len(t) else None}
    res["similarity_evidence_committed_tables"] = ev
    # --- descriptive subset: drop the 734 potentially exposed images (changes class mix; NOT a corrected estimate)
    keep = ~ex_s; sub = {}
    for arm in "AB":
        for m in MODELS:
            f = ROOT / ("results/sanity/oof_predictions_%s.csv" % arm if m == "resnet50" else f"results/multimodel/oof_predictions_{m}_{arm}.csv")
            pr = pd.read_csv(f).sort_values("image_id").pred_class.to_numpy()
            cm = np.zeros((3, 3)); np.add.at(cm, (y[keep], pr[keep]), 1)
            ba, f1 = metrics(cm); sub[f"{m}|{arm}"] = {"balanced_accuracy": round(float(ba), 4), "macro_f1": round(float(f1), 4)}
    res["subset_excluding_stem_exposed_images"] = {"n_images": int(keep.sum()), "class_counts": {CLS[k + 1]: int((y[keep] == k).sum()) for k in range(3)}, "metrics": sub,
        "note": "descriptive only; class mix differs from the full data; same images evaluated under A and B"}
    # --- cluster-bootstrap sensitivity: PID units vs stem units (prediction-only; NOT a stem-disjoint retraining)
    def boot(unit, nrep=5000, seed=2026):
        u = A[unit].to_numpy(); names = np.array(sorted(set(u))); code = np.searchsorted(names, u)
        ucls = np.array([np.unique(y[code == j]).tolist() for j in range(len(names))], dtype=object)
        assert all(len(c) == 1 for c in ucls), "unit with multiple classes"
        ucls = np.array([c[0] for c in ucls])
        rng = np.random.default_rng(seed); W = np.zeros((nrep, len(names)))
        for c in range(3):
            idx = np.where(ucls == c)[0]; dr = rng.integers(0, len(idx), (nrep, len(idx)))
            for r in range(nrep): W[r, idx] += np.bincount(dr[r], minlength=len(idx))
        return code, len(names), W
    out = {}
    for unit in ("patient_id", "stem"):
        code, P, W = boot(unit); R = {}
        for arm, sp in (("A", A), ("B", B)):
            for m in MODELS:
                f = ROOT / ("results/sanity/oof_predictions_%s.csv" % arm if m == "resnet50" else f"results/multimodel/oof_predictions_{m}_{arm}.csv")
                pred = pd.read_csv(f).sort_values("image_id").pred_class.to_numpy()
                cm = np.zeros((P, 9)); np.add.at(cm, (code, y * 3 + pred), 1)
                C = (W @ cm).reshape(-1, 3, 3); tp = np.einsum("rii->ri", C); rec = tp / C.sum(2); prec = tp / C.sum(1)
                R[m, arm] = {"ba": rec.mean(1), "f1": (2 * prec * rec / (prec + rec)).mean(1)}
        ci = lambda v: [round(float(x), 4) for x in np.percentile(v, [2.5, 97.5])]
        o = {"units": P, "BA_ci_protocol_A": {m: ci(R[m, "A"]["ba"]) for m in MODELS},
             "protocol_diff_BA_B_minus_A": {m: ci(R[m, "B"]["ba"] - R[m, "A"]["ba"]) for m in MODELS},
             "resnet50_minus_densenet121_BA": {a: ci(R["resnet50", a]["ba"] - R["densenet121", a]["ba"]) for a in "AB"},
             "resnet50_minus_densenet121_F1": {a: ci(R["resnet50", a]["f1"] - R["densenet121", a]["f1"]) for a in "AB"}}
        out[unit] = o
    res["cluster_bootstrap_sensitivity"] = {"raw_pid_units": out["patient_id"], "stem_units": out["stem"],
                                            "note": "stem-clustered bootstrap widens/changes uncertainty only; it is NOT a stem-disjoint retraining and gives no corrected estimate"}
    gt.to_csv(OUT / "stem_table.csv", index=False)
    (OUT / "patient_id_validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("normalization", "identity_interpretations")}, indent=1, default=str)[:3500])
    return res


if __name__ == "__main__":
    main()
