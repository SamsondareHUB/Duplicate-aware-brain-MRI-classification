"""Class-stratified patient-cluster bootstrap with paired arms.

Each replicate resamples patients WITH replacement inside each tumour class (every patient has exactly one class; checked
by the caller) and yields a multiplicity vector over patients. The SAME multiplicity vector is applied to arm A and arm B,
so replicates are paired. Metrics are recomputed from per-patient sufficient statistics (confusion counts, Brier sums, ECE
bin sums) weighted by multiplicity, which is exactly the metric on the resampled image set.
"""
import numpy as np
from . import metrics as M


def stratified_multiplicities(patient_class, n_rep, seed):
    """patient_class: array [P] of class ids. Returns int array [n_rep, P] of draw counts."""
    pc = np.asarray(patient_class)
    rng = np.random.default_rng(seed)
    W = np.zeros((n_rep, len(pc)), dtype=np.int32)
    for c in np.unique(pc):
        idx = np.where(pc == c)[0]
        draws = rng.integers(0, len(idx), size=(n_rep, len(idx)))
        for r in range(n_rep):
            W[r, idx] += np.bincount(draws[r], minlength=len(idx))
    return W


def patient_stats(patient_codes, n_patients, y, probs):
    """Per-patient sufficient statistics. patient_codes: int [N] in 0..P-1."""
    y = np.asarray(y); probs = np.asarray(probs); pred = probs.argmax(1)
    P = n_patients
    cm = np.zeros((P, M.K * M.K))
    np.add.at(cm, (patient_codes, y * M.K + pred), 1)
    br = np.bincount(patient_codes, weights=M.brier_per_image(probs, y), minlength=P)
    conf = probs.max(1); corr = (pred == y).astype(float); b = M.bin_index(conf)
    n = np.zeros((P, M.NBINS)); sc = np.zeros((P, M.NBINS)); sa = np.zeros((P, M.NBINS))
    np.add.at(n, (patient_codes, b), 1); np.add.at(sc, (patient_codes, b), conf); np.add.at(sa, (patient_codes, b), corr)
    return {"cm": cm, "brier": br, "n_img": np.bincount(patient_codes, minlength=P).astype(float), "n": n, "sc": sc, "sa": sa}


def metrics_from_weights(stats, w):
    """w: multiplicity vector [P] (or all-ones). Returns dict of scalar/array metrics."""
    cm = (w @ stats["cm"]).reshape(M.K, M.K)
    prec, rec, f1 = M.prf_from_conf(cm)
    n = w @ stats["n"]; sc = w @ stats["sc"]; sa = w @ stats["sa"]
    return {"balanced_accuracy": float(rec.mean()), "macro_f1": float(f1.mean()), "accuracy": M.accuracy(cm),
            "ece": M.ece_from_bins(n, sc, sa), "brier": float((w @ stats["brier"]) / (w @ stats["n_img"])),
            "recall": rec, "precision": prec, "f1": f1}


def run_bootstrap(stats_a, stats_b, patient_class, n_rep=5000, seed=2026):
    W = stratified_multiplicities(patient_class, n_rep, seed).astype(float)
    rows = []
    for r in range(n_rep):
        a = metrics_from_weights(stats_a, W[r]); b = metrics_from_weights(stats_b, W[r])
        row = {"rep": r}
        for k in ("balanced_accuracy", "macro_f1", "accuracy", "ece", "brier"):
            row[f"A_{k}"] = a[k]; row[f"B_{k}"] = b[k]; row[f"delta_{k}"] = b[k] - a[k]
        for c in range(M.K):
            for k in ("recall", "precision", "f1"):
                row[f"A_{k}_c{c}"] = a[k][c]; row[f"B_{k}_c{c}"] = b[k][c]; row[f"delta_{k}_c{c}"] = b[k][c] - a[k][c]
        rows.append(row)
    return rows, W
