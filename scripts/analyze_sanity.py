"""Pooled out-of-fold analysis of the 10-run sanity experiment (no training here).
python scripts/analyze_sanity.py [--runs results/sanity/runs] [--out results/sanity] [--reps 5000] [--partial]
"""
import argparse, gzip, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage import metrics as M
from leakage.bootstrap import patient_stats, metrics_from_weights, run_bootstrap, stratified_multiplicities
from leakage.oof import check_oof

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
BOOT_SEED = 2026
# ---- decision rule, fixed BEFORE looking at any result ----
CI_MIN_EFFECT_GLOBAL = 0.01   # BA, macro-F1, ECE, Brier: CI excludes 0 and |delta| >= 0.01
CI_MIN_EFFECT_CLASS = 0.02    # per-class recall / F1: CI excludes 0 and |delta| >= 0.02 (6 comparisons -> stricter)
MAGNITUDE_FLAG = 0.02         # point |delta BA| or |delta macro-F1| >= 0.02 flags GO even if CI includes 0


def load_arm(runs, arm, partial):
    fs = sorted(Path(runs).glob(f"{arm}_fold*_predictions.csv"))
    if not partial:
        assert len(fs) == 5, f"arm {arm}: {len(fs)} fold prediction files (expected 5)"
    d = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in fs], ignore_index=True)
    return d.sort_values("image_id").reset_index(drop=True)


def probs(d): return d[["prob_0", "prob_1", "prob_2"]].values


def ci(x): return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results/sanity/runs")); ap.add_argument("--out", default=str(ROOT / "results/sanity"))
    ap.add_argument("--reps", type=int, default=5000); ap.add_argument("--corr-reps", type=int, default=2000)
    ap.add_argument("--partial", action="store_true"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A, B = load_arm(a.runs, "A", a.partial), load_arm(a.runs, "B", a.partial)
    if a.partial:  # dry-run on the images both arms have
        common = sorted(set(A.image_id) & set(B.image_id)); A = A[A.image_id.isin(common)].reset_index(drop=True); B = B[B.image_id.isin(common)].reset_index(drop=True)
        check_oof(A, B, expected_n=len(A))
    else:
        check_oof(A, B, expected_n=3064)
    assert (A.image_id.values == B.image_id.values).all() and (A.patient_id.values == B.patient_id.values).all()
    y = A.true_class.values; pat = A.patient_id.values
    patients = np.array(sorted(set(pat))); code = np.searchsorted(patients, pat); P = len(patients)
    pcls = np.zeros(P, int)
    for p in range(P):
        u = np.unique(y[code == p]); assert len(u) == 1, "patient with multiple classes"; pcls[p] = u[0]
    pA, pB = probs(A), probs(B)
    stA, stB = patient_stats(code, P, y, pA), patient_stats(code, P, y, pB)
    ones = np.ones(P); mA, mB = metrics_from_weights(stA, ones), metrics_from_weights(stB, ones)
    # sanity: patient-stat route == direct route
    cmA, cmB = M.confusion(y, pA.argmax(1)), M.confusion(y, pB.argmax(1))
    assert abs(mA["balanced_accuracy"] - M.balanced_accuracy(cmA)) < 1e-12 and abs(mB["ece"] - M.ece(pB, y)) < 1e-12

    A.to_csv(out / "oof_predictions_A.csv", index=False); B.to_csv(out / "oof_predictions_B.csv", index=False)

    # ---- bootstrap ----
    rows, W = run_bootstrap(stA, stB, pcls, n_rep=a.reps, seed=BOOT_SEED)
    R = pd.DataFrame(rows)
    R.to_csv(out / "bootstrap_replicates.csv.gz", index=False, compression="gzip")
    bs = []
    def add(name, est_a, est_b):
        lo_a, hi_a = ci(R[f"A_{name}"]); lo_b, hi_b = ci(R[f"B_{name}"]); lo_d, hi_d = ci(R[f"delta_{name}"])
        bs.append({"metric": name, "A": est_a, "A_ci_lo": lo_a, "A_ci_hi": hi_a, "B": est_b, "B_ci_lo": lo_b, "B_ci_hi": hi_b,
                   "delta_B_minus_A": est_b - est_a, "delta_ci_lo": lo_d, "delta_ci_hi": hi_d,
                   "ci_excludes_0": bool(lo_d > 0 or hi_d < 0), "boot_frac_delta_gt0": float((R[f"delta_{name}"] > 0).mean())})
    for k in ("balanced_accuracy", "macro_f1", "accuracy", "ece", "brier"): add(k, mA[k], mB[k])
    for c in range(3):
        for k in ("recall", "precision", "f1"): add(f"{k}_c{c}", mA[k][c], mB[k][c])
    BS = pd.DataFrame(bs); BS["class"] = BS.metric.str.extract(r"_c(\d)$")[0].map(lambda v: CLASS_NAMES[int(v)] if pd.notna(v) else "")
    BS.to_csv(out / "bootstrap_summary.csv", index=False)
    summ = BS[BS.metric.isin(["balanced_accuracy", "macro_f1", "accuracy", "ece", "brier"])]
    summ.to_csv(out / "resnet50_summary.csv", index=False)

    # ---- per-class, confusion, calibration, folds ----
    pc = []
    for arm, cm in (("A_patient_disjoint", cmA), ("B_imagelevel_seed11", cmB)):
        pr, rc, f1 = M.prf_from_conf(cm)
        for c in range(3): pc.append({"arm": arm, "class": CLASS_NAMES[c], "support": int(cm[c].sum()), "precision": pr[c], "recall": rc[c], "f1": f1[c]})
    pd.DataFrame(pc).to_csv(out / "per_class_metrics.csv", index=False)
    cr = []
    for arm, cm in (("A_patient_disjoint", cmA), ("B_imagelevel_seed11", cmB)):
        for t in range(3):
            for p_ in range(3): cr.append({"arm": arm, "true": CLASS_NAMES[t], "pred": CLASS_NAMES[p_], "count": int(cm[t, p_])})
    pd.DataFrame(cr).to_csv(out / "confusion_matrices.csv", index=False)
    cal, binrows = [], []
    for arm, pr_, m_ in (("A_patient_disjoint", pA, mA), ("B_imagelevel_seed11", pB, mB)):
        conf = pr_.max(1); acc = (pr_.argmax(1) == y).mean()
        cal.append({"arm": arm, "ece_15bin": m_["ece"], "brier_multiclass": m_["brier"], "mean_confidence": conf.mean(), "accuracy": acc,
                    "mean_confidence_minus_accuracy": conf.mean() - acc, "frac_conf_gt_0.99": float((conf > 0.99).mean()),
                    "mean_nll": float(-np.log(np.clip(pr_[np.arange(len(y)), y], 1e-12, 1)).mean())})
        n, sc, sa = M.ece_bins(pr_, y)
        for b in range(M.NBINS):
            binrows.append({"arm": arm, "bin": b, "lo": b / M.NBINS, "hi": (b + 1) / M.NBINS, "n": int(n[b]),
                            "mean_conf": sc[b] / n[b] if n[b] else np.nan, "accuracy": sa[b] / n[b] if n[b] else np.nan})
    pd.DataFrame(cal).to_csv(out / "calibration_summary.csv", index=False); pd.DataFrame(binrows).to_csv(out / "reliability_bins.csv", index=False)
    fm = []
    for arm, d in (("A_patient_disjoint", A), ("B_imagelevel_seed11", B)):
        for f, g in d.groupby("test_fold"):
            cm = M.confusion(g.true_class.values, g.pred_class.values)
            fm.append({"arm": arm, "fold": int(f), "n": len(g), "accuracy": M.accuracy(cm), "balanced_accuracy": M.balanced_accuracy(cm), "macro_f1": M.macro_f1(cm)})
    pd.DataFrame(fm).to_csv(out / "fold_metrics.csv", index=False)

    # ---- exposure analysis (arm B only; associational, not causal) ----
    Bx = B.copy(); Bx["correct"] = (pB.argmax(1) == y).astype(float); Bx["conf"] = pB.max(1); Bx["brier_i"] = M.brier_per_image(pB, y)
    Bx["A_correct"] = (pA.argmax(1) == y).astype(float); Bx["A_conf"] = pA.max(1); Bx["A_brier_i"] = M.brier_per_image(pA, y)
    Bx["pcode"] = code; Bx["cls"] = y; Bx["n_patient_slices"] = Bx.n_other_total + 1
    def bin_nother(v): return "0" if v == 0 else "1-9" if v <= 9 else "10-13" if v <= 13 else "14-17" if v <= 17 else "18+"
    def bin_frac(v): return "single-slice (NA)" if pd.isna(v) else "0" if v == 0 else "(0,0.65]" if v <= .65 else "(0.65,0.8]" if v <= .8 else "(0.8,0.9]" if v <= .9 else "(0.9,1.0]"
    Bx["bin_n_other_train"] = Bx.n_other_train.map(bin_nother); Bx["bin_frac_other_train"] = Bx.frac_other_train.map(bin_frac)
    Bx["bin_has_same_patient_train_slice"] = Bx.has_same_patient_train_slice.map(lambda v: "True" if v else "False")
    Wf = W.astype(float)
    ex = []
    pops = {"all": np.ones(len(Bx), bool), **{CLASS_NAMES[c]: (y == c) for c in range(3)}}
    for var in ("bin_n_other_train", "bin_frac_other_train", "bin_has_same_patient_train_slice"):
        for popn, pm in pops.items():
            for lvl in sorted(Bx[var].unique(), key=str):
                m = pm & (Bx[var] == lvl).values
                if m.sum() == 0: continue
                cnt = np.bincount(code[m], minlength=P).astype(float); den = Wf @ cnt
                row = {"section": "binned_B_with_A_descriptive", "variable": var.replace("bin_", ""), "population": popn, "level": lvl,
                       "n_images": int(m.sum()), "n_patients": int((cnt > 0).sum())}
                for nm, col in (("correct", "correct"), ("conf", "conf"), ("brier", "brier_i"), ("A_correct", "A_correct"), ("A_conf", "A_conf"), ("A_brier", "A_brier_i")):
                    s = np.bincount(code[m], weights=Bx[col].values[m], minlength=P); est = s.sum() / cnt.sum()
                    with np.errstate(invalid="ignore", divide="ignore"): est_b = (Wf @ s) / den
                    lo, hi = np.nanpercentile(est_b, [2.5, 97.5]); row.update({f"{nm}_mean": est, f"{nm}_ci_lo": lo, f"{nm}_ci_hi": hi})
                ex.append(row)
    # rank correlations (pooled / within class) with patient-cluster bootstrap CI
    rng_reps = min(a.corr_reps, a.reps)
    mult = Wf[:rng_reps][:, code]  # image-level multiplicities per replicate
    multi = ~Bx.single_slice_patient.values
    for popn, pm in pops.items():
        for expo in ("n_other_train", "frac_other_train"):
            sel = pm & (multi if expo == "frac_other_train" else np.ones(len(Bx), bool))
            for outc in ("correct", "conf", "brier_i"):
                x, z = Bx[expo].values[sel], Bx[outc].values[sel]
                est = spearmanr(x, z).statistic; reps = []
                for r in range(rng_reps):
                    idx = np.repeat(np.where(sel)[0], mult[r][sel].astype(int)); reps.append(spearmanr(Bx[expo].values[idx], Bx[outc].values[idx]).statistic)
                lo, hi = np.nanpercentile(reps, [2.5, 97.5])
                ex.append({"section": "spearman_B", "variable": expo, "population": popn, "level": outc, "n_images": int(sel.sum()),
                           "spearman_rho": est, "ci_lo": lo, "ci_hi": hi, "n_boot_reps": rng_reps})
    # collinearity / context
    ns = Bx.groupby("pcode").agg(n=("image_id", "size"), cls=("cls", "first"), pid=("patient_id", "first"))
    ns["mr_prefixed"] = ~ns.pid.str.fullmatch(r"\d+")
    ex.append({"section": "context", "variable": "spearman(n_other_train, n_other_total)", "level": "B", "spearman_rho": spearmanr(Bx.n_other_train, Bx.n_other_total).statistic})
    ex.append({"section": "context", "variable": "frac_other_train", "level": "B_mean_sd_multislice", "mean": float(Bx.frac_other_train.mean()), "sd": float(Bx.frac_other_train.std())})
    for c in range(3):
        g = ns[ns.cls == c]
        ex.append({"section": "context", "variable": "patient_size_and_provenance_by_class", "population": CLASS_NAMES[c], "n_patients": len(g),
                   "mean_slices_per_patient": float(g.n.mean()), "median_slices_per_patient": float(g.n.median()), "frac_patients_MR_prefixed": float(g.mr_prefixed.mean())})
    try:
        import statsmodels.formula.api as smf
        d = Bx[multi].copy(); d["log_n_patient_slices"] = np.log(d.n_patient_slices); d["cls_f"] = d.cls.astype(str)
        for outc in ("correct", "conf", "brier_i"):
            fit = smf.ols(f"{outc} ~ frac_other_train + log_n_patient_slices + C(cls_f)", d).fit(cov_type="cluster", cov_kwds={"groups": d.pcode})
            for term in ("frac_other_train", "log_n_patient_slices"):
                ex.append({"section": "ols_cluster_robust_B_exploratory", "variable": term, "level": outc, "coef": float(fit.params[term]),
                           "ci_lo": float(fit.conf_int().loc[term, 0]), "ci_hi": float(fit.conf_int().loc[term, 1]), "p": float(fit.pvalues[term]), "n_images": int(fit.nobs)})
    except Exception as e:
        ex.append({"section": "ols_cluster_robust_B_exploratory", "level": f"skipped: {e}"})
    pd.DataFrame(ex).to_csv(out / "exposure_analysis.csv", index=False)

    # ---- decision ----
    dec = {"rule": {"ci_min_effect_global": CI_MIN_EFFECT_GLOBAL, "ci_min_effect_per_class": CI_MIN_EFFECT_CLASS,
                    "magnitude_flag_point_delta_BA_or_F1": MAGNITUDE_FLAG, "fixed_before_results": True}, "hits": []}
    for r_ in BS.itertuples():
        glob = r_.metric in ("balanced_accuracy", "macro_f1", "ece", "brier"); cls_ = r_.metric.startswith(("recall_c", "f1_c"))
        thr = CI_MIN_EFFECT_GLOBAL if glob else CI_MIN_EFFECT_CLASS if cls_ else None
        if thr is not None and r_.ci_excludes_0 and abs(r_.delta_B_minus_A) >= thr:
            dec["hits"].append({"metric": r_.metric, "delta": r_.delta_B_minus_A, "ci": [r_.delta_ci_lo, r_.delta_ci_hi]})
    mag = [m for m in ("balanced_accuracy", "macro_f1") if abs(BS.set_index("metric").loc[m, "delta_B_minus_A"]) >= MAGNITUDE_FLAG]
    dec["magnitude_flag_metrics"] = mag
    dec["DECISION"] = "GO_TO_MULTI_MODEL" if (dec["hits"] or mag) else "STOP_AT_NULL_SANITY_RESULT"
    (out / "decision.json").write_text(json.dumps(dec, indent=2))
    print(summ[["metric", "A", "B", "delta_B_minus_A", "delta_ci_lo", "delta_ci_hi"]].round(4).to_string()); print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
