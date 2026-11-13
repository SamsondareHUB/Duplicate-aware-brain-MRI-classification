"""Exposure-response analysis (EXPLORATORY INTERVENTION; SIB arms intentionally create train/test patient overlap).
Population Q = fixed-test eval slices of ELIGIBLE test patients (>=4 slices). E_L = slices of patients selected at level L (nested).
N_L = fixed-test patients not selected at L (never-exposed at that level; includes small-slice ineligible patients).
Contrasts (per model, fold; seeds averaged at the metric level, same bootstrap draws for every arm/seed):
  swap      CTRL_L - CLEAN  on Q   (effect of merely adding L% other-patient images)
  sibling   SIB_L  - CTRL_L on E_L (exposed patients; PRIMARY at L=100 on Q)
  spill     SIB_L  - CTRL_L on N_L (non-exposed patients; training-composition spillover)
  (difference-in-differences = sibling(E_L) - spillover(N_L), computed in the report from these rows)
Bootstrap: 5,000 class-stratified patient resamples, seed 2026. Seeds are NOT resampled (2 seeds only): intervals cover patient
sampling for the fixed trained models, not training randomness; the between-seed range is reported alongside."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import *
from ext_common import _stats, _metric_arrays
from ext_specs import MODELS, EXPO_SEEDS, EXT
from make_exposure import LEVELS, EXPOSURE_FOLDS

NREP = 5000


def rid(model, fold, kind, L, seed):
    return f"expo_{model}_fold{fold}_{kind}_L{L}_s{seed}"


def per_image(d):
    p = d[[f"prob_{k}" for k in range(3)]].to_numpy(); y = d.true_class.to_numpy()
    out = d[["image_id", "patient_id", "true_class"]].copy()
    out["py"] = p[np.arange(len(d)), y]; out["correct"] = (p.argmax(1) == y).astype(float)
    out["brier"] = ((p - np.eye(3)[y]) ** 2).sum(1)
    return out.set_index("image_id")


def metric_diff_boot(frames_x, frames_y, pats, cls):
    """mean over seeds of (metric(X)-metric(Y)) for each bootstrap draw; returns {metric: array[R+1]} (last = point estimate)."""
    W = np.vstack([stratified_multiplicities(cls, NREP, 2026).astype(float), np.ones((1, len(pats)))])
    res = {m: 0.0 for m in ("balanced_accuracy", "macro_f1", "accuracy", "ece", "brier")}
    for fx, fy in zip(frames_x, frames_y):
        ax = _metric_arrays(_stats(fx, pats), W); ay = _metric_arrays(_stats(fy, pats), W)
        for m in res: res[m] = res[m] + (ax[m] - ay[m]) / len(frames_x)
    return res


def summarize(arr, label, **kw):
    boot, est = arr[:-1], arr[-1]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
    return {"contrast": label, "estimate": float(est), "ci_lo": lo, "ci_hi": hi, "boot_p": p, **kw}


def run():
    rows, per_level_rows = [], []
    for model in MODELS:
        for fold in EXPOSURE_FOLDS:
            pat = pd.read_csv(EXT / f"exposure/patients_fold{fold}.csv", dtype={"patient_id": str, "control_patient": str})
            ft = pd.read_csv(EXT / f"exposure/fixed_test_fold{fold}.csv", dtype={"patient_id": str})
            Q_pat = sorted(pat.patient_id)
            seeds = [s for s in EXPO_SEEDS if all(verify(rid(model, fold, "CLEAN", 0, s)) if k == "CLEAN" else verify(rid(model, fold, k, L, s))
                                                   for k, L in [("CLEAN", 0)] + [(k, L) for L in LEVELS for k in ("CTRL", "SIB")])]
            if not seeds: continue
            fr = {(k, L, s): load_run(rid(model, fold, k, L, s)).sort_values("image_id").reset_index(drop=True)
                  for s in seeds for k, L in [("CLEAN", 0)] + [(k, L) for L in LEVELS for k in ("CTRL", "SIB")]}
            base = fr[("CLEAN", 0, seeds[0])]
            cls_of = base.groupby("patient_id").true_class.first()
            # whole fixed-test BA by condition (descriptive)
            for (k, L, s), d in fr.items():
                per_level_rows.append({"model": model, "fold": fold, "seed": s, "kind": k, "level": L, "population": "fixed_test_all", **point_metrics(d)})
            def sub(d, pats): return d[d.patient_id.isin(pats)].reset_index(drop=True)
            def contrast(kx, Lx, ky, Ly, pats, label, level, pop):
                pats = sorted(pats)
                if not pats: return None
                cls = cls_of.loc[pats].to_numpy()
                fx = [sub(fr[(kx, Lx, s)], pats) for s in seeds]; fy = [sub(fr[(ky, Ly, s)], pats) for s in seeds]
                arrs = metric_diff_boot(fx, fy, np.array(pats), cls)
                # patient-level true-class-probability change (more sensitive than BA)
                W = np.vstack([stratified_multiplicities(cls, NREP, 2026).astype(float), np.ones((1, len(pats)))])
                dpy = 0.0
                for a, b in zip(fx, fy):
                    d = per_image(a).py - per_image(b).py
                    dpy = dpy + d.groupby(a.set_index("image_id").patient_id).mean().reindex(pats).to_numpy() / len(seeds)
                arrs["mean_true_class_prob"] = (W @ dpy) / W.sum(1)
                out = []
                for m, arr in arrs.items():
                    out.append(summarize(arr, label, model=model, fold=fold, level=level, population=pop, metric=m, n_patients=len(pats), n_seeds=len(seeds)))
                return out
            Q = [p for p in Q_pat]
            for L in LEVELS:
                E = pat.patient_id[pat[f"selected_L{L}"]].tolist()
                allp = sorted(ft.patient_id.unique()); Nn = [p for p in allp if p not in set(E)]
                for res in (contrast("CTRL", L, "CLEAN", 0, Q, "swap: CTRL-CLEAN", L, "Q_eligible"),
                            contrast("SIB", L, "CTRL", L, E, "sibling: SIB-CTRL", L, "E_exposed"),
                            contrast("SIB", L, "CTRL", L, Q, "sibling: SIB-CTRL", L, "Q_eligible"),
                            contrast("SIB", L, "CTRL", L, Nn, "spillover: SIB-CTRL", L, "N_not_exposed"),
                            contrast("SIB", L, "CLEAN", 0, E, "SIB-CLEAN", L, "E_exposed")):
                    if res: rows += res
    R = pd.DataFrame(rows); P = pd.DataFrame(per_level_rows)
    R.to_csv(EXT / "exposure_contrasts.csv", index=False); P.to_csv(EXT / "exposure_condition_metrics.csv", index=False)
    return R, P


if __name__ == "__main__":
    R, P = run()
    if len(R):
        print(R[R.metric.isin(["balanced_accuracy", "mean_true_class_prob"])].round(4).to_string())
    else: print("no complete (model, fold) exposure matrix yet")
