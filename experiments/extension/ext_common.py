"""Shared loading/metric helpers for the extension analyses (reuses the frozen src/leakage metrics)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments/extension"))
from leakage import metrics as M
from leakage.bootstrap import patient_stats, metrics_from_weights, stratified_multiplicities
from ext_specs import RUNS, build_queue
from ext_runner import verify

KEYS = ("balanced_accuracy", "macro_f1", "accuracy", "ece", "brier")


def load_run(rid):
    d = pd.read_csv(RUNS / f"{rid}_predictions.csv", dtype={"patient_id": str})
    return d


def point_metrics(d):
    y = d.true_class.to_numpy(); p = d[[f"prob_{k}" for k in range(3)]].to_numpy()
    cm = M.confusion(y, p.argmax(1)); prec, rec, f1 = M.prf_from_conf(cm)
    return {"balanced_accuracy": M.balanced_accuracy(cm), "macro_f1": M.macro_f1(cm), "accuracy": M.accuracy(cm),
            "ece": M.ece(p, y), "brier": M.brier(p, y), **{f"recall_c{k}": rec[k] for k in range(3)},
            **{f"f1_c{k}": f1[k] for k in range(3)}, "n": len(d)}


def completed(prefix=""):
    return [r for r in build_queue() if r.startswith(prefix) and verify(r)]


def cluster_boot(d, fn, n_rep=5000, seed=2026):
    """Class-stratified patient-cluster bootstrap of a statistic fn(df) on a prediction frame (patients resampled)."""
    pats = d.patient_id.unique(); cls = d.groupby("patient_id").true_class.first().loc[pats].to_numpy()
    W = stratified_multiplicities(cls, n_rep, seed)
    groups = {p: g for p, g in d.groupby("patient_id")}
    return pats, W, groups


# ---------- vectorised patient-cluster bootstrap over arbitrary named prediction sets (paired by shared multiplicities) ----
def _stats(d, pats):
    codes = pd.Categorical(d.patient_id, categories=pats).codes
    y = d.true_class.to_numpy(); p = d[[f"prob_{k}" for k in range(3)]].to_numpy()
    return patient_stats(codes, len(pats), y, p)


def _metric_arrays(S, W):
    cm = (W @ S["cm"]).reshape(len(W), 3, 3)
    tp = np.einsum("rii->ri", cm); rd = cm.sum(2); pd_ = cm.sum(1)
    rec = np.divide(tp, rd, out=np.zeros_like(tp), where=rd > 0); prec = np.divide(tp, pd_, out=np.zeros_like(tp), where=pd_ > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    n = W @ S["n"]; sc = W @ S["sc"]; sa = W @ S["sa"]
    N = n.sum(1)
    return {"balanced_accuracy": rec.mean(1), "macro_f1": f1.mean(1), "accuracy": tp.sum(1) / cm.sum((1, 2)),
            "ece": np.abs(sa - sc).sum(1) / N, "brier": (W @ S["brier"]) / (W @ S["n_img"]),
            **{f"recall_c{k}": rec[:, k] for k in range(3)}}


def paired_bootstrap(frames, contrasts, n_rep=5000, seed=2026, keys=KEYS):
    """frames: {name: DataFrame(patient_id,true_class,prob_*)} over the SAME patients. contrasts: {label: (nameX, nameY)} -> X - Y.
    Returns (point, ci) DataFrames. Patients resampled with replacement within class, SAME draws for every frame (paired)."""
    first = next(iter(frames.values()))
    pats = np.array(sorted(first.patient_id.unique()))
    cls = first.groupby("patient_id").true_class.first().loc[pats].to_numpy()
    for f in frames.values(): assert set(f.patient_id) == set(pats)
    W = stratified_multiplicities(cls, n_rep, seed).astype(float)
    S = {k: _stats(v, pats) for k, v in frames.items()}
    boot = {k: _metric_arrays(S[k], W) for k in frames}
    pt = {k: _metric_arrays(S[k], np.ones((1, len(pats)))) for k in frames}
    rows = []
    for k in frames:
        for m in keys:
            lo, hi = np.percentile(boot[k][m], [2.5, 97.5])
            rows.append({"item": k, "metric": m, "estimate": float(pt[k][m][0]), "ci_lo": lo, "ci_hi": hi})
    for lab, (x, y) in contrasts.items():
        for m in keys:
            diff = boot[x][m] - boot[y][m]; lo, hi = np.percentile(diff, [2.5, 97.5])
            est = float(pt[x][m][0] - pt[y][m][0])
            p = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
            rows.append({"item": lab, "metric": m, "estimate": est, "ci_lo": lo, "ci_hi": hi, "boot_p_two_sided": p})
    return pd.DataFrame(rows)


def oof_frame(run_ids):
    return pd.concat([load_run(r) for r in run_ids], ignore_index=True).sort_values("image_id").reset_index(drop=True)
