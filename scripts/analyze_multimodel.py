"""Joint four-model analysis from committed out-of-fold predictions.

One class-stratified patient-cluster draw matrix is shared by every model and
both protocols (5000 replicates, seed 2026). PR #1 labels are used throughout:
A=patient-disjoint; B=image-level seed 11.
"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage import metrics as M
from leakage.bootstrap import metrics_from_weights, patient_stats, stratified_multiplicities

OUT = ROOT / "results/multimodel"
FIG = OUT / "figures"
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
ARMS = ("A", "B")
PROTOCOL = {"A": "patient_disjoint", "B": "image_level_seed11"}
CLASS_NAMES = ("meningioma", "glioma", "pituitary")
METRICS = ("balanced_accuracy", "macro_f1", "accuracy", "ece", "brier")
RANK_METRICS = ("balanced_accuracy", "macro_f1")


def ci(a):
    return np.percentile(a, [2.5, 97.5]).tolist()


def load_predictions(model, arm):
    if model == "resnet50":
        p = ROOT / f"results/sanity/oof_predictions_{arm}.csv"
        d = pd.read_csv(p, dtype={"patient_id": str})
        d.insert(0, "model", model)
        d["protocol"] = PROTOCOL[arm]
        d["true_class_name"] = np.array(CLASS_NAMES)[d.true_class.to_numpy()]
        d["pred_class_name"] = np.array(CLASS_NAMES)[d.pred_class.to_numpy()]
        d["confidence"] = d[[f"prob_{k}" for k in range(3)]].max(axis=1)
    else:
        fs = sorted((OUT / "runs").glob(f"{model}_{arm}_fold*_predictions.csv"))
        assert len(fs) == 5, (model, arm, len(fs))
        d = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in fs], ignore_index=True)
    d = d.sort_values("image_id").reset_index(drop=True)
    assert len(d) == 3064 and d.image_id.is_unique, (model, arm, len(d))
    assert (d.model == model).all() and (d.arm == arm).all() and (d.protocol == PROTOCOL[arm]).all()
    split_name = "patient_disjoint_original.csv" if arm == "A" else "imagelevel_seed11.csv"
    split = pd.read_csv(ROOT / "data/splits" / split_name, dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
    assert np.array_equal(d.image_id, split.image_id)
    assert np.array_equal(d.patient_id, split.patient_id)
    assert np.array_equal(d.test_fold, split.fold)
    assert np.array_equal(d.true_class, split.label - 1)
    p = d[[f"prob_{k}" for k in range(3)]].to_numpy(dtype=float)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    assert np.allclose(p.sum(1), 1, atol=1e-6)
    assert np.array_equal(d.pred_class, p.argmax(1))
    assert np.allclose(d.confidence, p.max(1), rtol=0, atol=1e-12)
    assert np.array_equal(d.pred_class_name, np.array(CLASS_NAMES)[p.argmax(1)])
    d.to_csv(OUT / f"oof_predictions_{model}_{arm}.csv", index=False)
    return d, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data, probs = {}, {}
    for model in MODELS:
        for arm in ARMS:
            data[model, arm], probs[model, arm] = load_predictions(model, arm)
    ref = data["resnet50", "A"]
    for key, d in data.items():
        assert np.array_equal(d.image_id, ref.image_id), key
        assert np.array_equal(d.patient_id, ref.patient_id), key
        assert np.array_equal(d.true_class, ref.true_class), key
    assert ref.patient_id.nunique() == 233
    patients = np.array(sorted(ref.patient_id.unique()))
    code = np.searchsorted(patients, ref.patient_id.to_numpy())
    y = ref.true_class.to_numpy(dtype=int)
    pcls = np.zeros(len(patients), dtype=int)
    for j in range(len(patients)):
        unique = np.unique(y[code == j])
        assert len(unique) == 1
        pcls[j] = unique[0]
    W = stratified_multiplicities(pcls, 5000, 2026).astype(float)
    np.savez_compressed(OUT / "bootstrap_patient_draws.npz", patient_id=patients, multiplicities=W.astype(np.int16))
    ones = np.ones(len(patients))
    point, reps = {}, {}
    rows, class_rows, cm_rows, bin_rows, replicate_rows = [], [], [], [], []
    for model, arm in itertools.product(MODELS, ARMS):
        p = probs[model, arm]
        st = patient_stats(code, len(patients), y, p)
        point[model, arm] = metrics_from_weights(st, ones)
        reps[model, arm] = {k: np.zeros(5000) for k in METRICS}
        for k in ("precision", "recall", "f1"):
            reps[model, arm][k] = np.zeros((5000, 3))
        for r in range(5000):
            metric = metrics_from_weights(st, W[r])
            for k in METRICS:
                reps[model, arm][k][r] = metric[k]
            for k in ("precision", "recall", "f1"):
                reps[model, arm][k][r] = metric[k]
        for k in METRICS:
            lo, hi = ci(reps[model, arm][k])
            rows.append({"model": model, "arm": arm, "protocol": PROTOCOL[arm],
                         "metric": k, "estimate": point[model, arm][k], "ci_lo": lo, "ci_hi": hi})
        for c in range(3):
            for k in ("precision", "recall", "f1"):
                lo, hi = ci(reps[model, arm][k][:, c])
                class_rows.append({"model": model, "arm": arm, "protocol": PROTOCOL[arm],
                                   "class": CLASS_NAMES[c], "metric": k, "estimate": point[model, arm][k][c],
                                   "ci_lo": lo, "ci_hi": hi, "support": int((y == c).sum())})
        cm = M.confusion(y, p.argmax(1))
        for c, q in itertools.product(range(3), repeat=2):
            cm_rows.append({"model": model, "arm": arm, "protocol": PROTOCOL[arm],
                            "true_class": CLASS_NAMES[c], "pred_class": CLASS_NAMES[q], "count": int(cm[c, q])})
        n, sc, sa = M.ece_bins(p, y)
        for b in range(M.NBINS):
            bin_rows.append({"model": model, "arm": arm, "protocol": PROTOCOL[arm],
                             "bin": b, "lo": b / M.NBINS, "hi": (b + 1) / M.NBINS,
                             "n": int(n[b]), "mean_confidence": sc[b] / n[b] if n[b] else np.nan,
                             "accuracy": sa[b] / n[b] if n[b] else np.nan})
        for r in range(5000):
            replicate_rows.append({"replicate": r, "model": model, "arm": arm,
                                   **{k: reps[model, arm][k][r] for k in METRICS}})
    pd.DataFrame(rows).to_csv(OUT / "metrics_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(OUT / "confusion_matrices.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(OUT / "reliability_bins.csv", index=False)
    pd.DataFrame(replicate_rows).to_csv(OUT / "bootstrap_replicates.csv.gz", index=False, compression="gzip")
    deltas = []
    for model in MODELS:
        for k in METRICS:
            diff = reps[model, "B"][k] - reps[model, "A"][k]
            lo, hi = ci(diff)
            deltas.append({"model": model, "metric": k,
                           "delta_image_minus_patient": point[model, "B"][k] - point[model, "A"][k],
                           "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(deltas).to_csv(OUT / "protocol_differences.csv", index=False)
    pairs = []
    for arm, k in itertools.product(ARMS, RANK_METRICS):
        for left, right in itertools.combinations(MODELS, 2):
            diff = reps[left, arm][k] - reps[right, arm][k]
            lo, hi = ci(diff)
            pairs.append({"arm": arm, "protocol": PROTOCOL[arm], "metric": k,
                          "model_left": left, "model_right": right,
                          "difference_left_minus_right": point[left, arm][k] - point[right, arm][k],
                          "ci_lo": lo, "ci_hi": hi,
                          "bootstrap_fraction_left_higher": float((diff > 0).mean())})
    pd.DataFrame(pairs).to_csv(OUT / "pairwise_differences.csv", index=False)
    ranking, tops = [], []
    for k in RANK_METRICS:
        scores = {arm: np.array([point[m, arm][k] for m in MODELS]) for arm in ARMS}
        tau = kendalltau(scores["A"], scores["B"]).statistic
        for arm in ARMS:
            ranked = sorted(MODELS, key=lambda m: (-point[m, arm][k], m))
            for place, model in enumerate(ranked, 1):
                ranking.append({"metric": k, "arm": arm, "protocol": PROTOCOL[arm],
                                "rank": place, "model": model, "estimate": point[model, arm][k],
                                "kendall_tau_between_protocols": tau})
            arr = np.stack([reps[m, arm][k] for m in MODELS], axis=1)
            wins = np.isclose(arr, arr.max(axis=1, keepdims=True), atol=1e-12, rtol=0)
            credit = wins / wins.sum(axis=1, keepdims=True)
            for j, model in enumerate(MODELS):
                tops.append({"metric": k, "arm": arm, "protocol": PROTOCOL[arm], "model": model,
                             "bootstrap_top_ranked_frequency": float(credit[:, j].mean()),
                             "tie_policy": "split credit equally among exact tied maxima"})
    pd.DataFrame(ranking).to_csv(OUT / "ranking_summary.csv", index=False)
    pd.DataFrame(tops).to_csv(OUT / "top_ranked_frequencies.csv", index=False)
    fig, axes = plt.subplots(4, 2, figsize=(10, 14), constrained_layout=True)
    bins = pd.DataFrame(bin_rows)
    for i, model in enumerate(MODELS):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            d = bins[(bins.model == model) & (bins.arm == arm) & (bins.n > 0)]
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.plot(d.mean_confidence, d.accuracy, "o-", markersize=4)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy",
                   title=f"{model}: {PROTOCOL[arm]}")
    fig.savefig(FIG / "reliability_diagrams.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    summary = pd.DataFrame(rows)
    x = np.arange(len(MODELS))
    for ax, k in zip(axes, RANK_METRICS):
        for j, arm in enumerate(ARMS):
            d = summary[(summary.metric == k) & (summary.arm == arm)].set_index("model").loc[list(MODELS)]
            xx = x + (j - .5) * .36
            ax.bar(xx, d.estimate, width=.34, label=PROTOCOL[arm])
            ax.errorbar(xx, d.estimate, yerr=[d.estimate-d.ci_lo, d.ci_hi-d.estimate],
                        fmt="none", ecolor="black", capsize=3)
        ax.set_xticks(x, MODELS, rotation=20, ha="right")
        ax.set_ylim(0.7, 1.0)
        ax.set_ylabel(k)
        ax.legend(fontsize=8)
    fig.savefig(FIG / "model_comparison.png", dpi=180)
    plt.close(fig)
    result = {"models": list(MODELS), "n_images_per_model_protocol": 3064,
              "n_patients": len(patients), "bootstrap_replicates": 5000, "bootstrap_seed": 2026,
              "protocol_A": PROTOCOL["A"], "protocol_B": PROTOCOL["B"],
              "new_runs": sum(1 for p in (OUT / "runs").glob("*_run.json"))}
    (OUT / "analysis_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
