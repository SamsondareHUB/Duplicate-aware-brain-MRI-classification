"""209-group analysis pipeline (PROPOSED; no real 209 predictions exist yet). Frozen metrics/bootstrap code from src/leakage is reused unchanged.

Bootstrap: 5,000 class-stratified paired cluster replicates, seed 2026 (identical procedure to scripts/analyze_multimodel.py). DOCUMENTED CHANGE:
the cluster is the normalised identity STEM (209 units) instead of the raw patient ID (233 units). One draw matrix is shared by all models and arms.
Outputs -> results/retrain_209/analysis/ (or --out). Historical files are only read.
  A209 vs B209 : matched (same 3,064 images, same fold x class counts, same clusters)  -> protocol_differences_209.csv
  A (hist) vs A209 : ROBUSTNESS comparison, NOT matched (changed fold assignments, 332 images moved, every train set differs)
  A209 is never compared with historical B as if matched.
  python scripts/analyze_209.py --scenario distinct|repeated|no_response [--runs results/retrain_209] [--n-rep 5000]
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from leakage import metrics as M
from leakage.bootstrap import metrics_from_weights, patient_stats, stratified_multiplicities
import run_retrain_209 as R

METRICS = ("balanced_accuracy", "macro_f1", "accuracy", "ece", "brier")
RANK_METRICS = ("balanced_accuracy", "macro_f1")


def ci(a): return np.percentile(a, [2.5, 97.5]).tolist()


def load_arm(model, arm, runs_dir, spl=R.SPL, allowed_devices=("mps",), n_epochs=10):
    split = pd.read_csv(Path(spl) / R.FILES[arm], dtype={"patient_id": str, "stem": str}).sort_values("image_id").reset_index(drop=True)
    sha = R.sha(Path(spl) / R.FILES[arm]); parts = []
    for f in range(1, 6):
        assert R.fold_state(Path(runs_dir), model, arm, f, split, sha, n_epochs, allowed_devices) == "complete", f"{model} {arm} fold {f} missing"
        parts.append(pd.read_csv(R.paths(Path(runs_dir), model, arm, f)[0], dtype={"patient_id": str, "stem": str}))
    d = pd.concat(parts, ignore_index=True).sort_values("image_id").reset_index(drop=True)
    assert len(d) == 3064 and d.image_id.is_unique and np.array_equal(d.image_id, split.image_id) and np.array_equal(d.test_fold, split.fold)
    return d


def load_hist(model, root=ROOT):
    p = Path(root) / f"results/multimodel/oof_predictions_{model}_A.csv"
    return pd.read_csv(p, dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True) if p.exists() else None


def run_analysis(data, models, arms, out, n_rep=5000, seed=2026, hist=None, unchanged_mask=None):
    """data[(model,arm)] -> prediction frame (with 'stem'). hist[model] -> historical A frame (optional)."""
    out = Path(out); (out / "figures").mkdir(parents=True, exist_ok=True)
    ref = data[models[0], arms[0]]
    for k, d in data.items():
        assert np.array_equal(d.image_id, ref.image_id) and np.array_equal(d.stem, ref.stem) and np.array_equal(d.true_class, ref.true_class), k
    stems = np.array(sorted(ref.stem.unique())); code = np.searchsorted(stems, ref.stem.to_numpy()); y = ref.true_class.to_numpy(int)
    assert len(stems) == 209
    pcls = np.zeros(len(stems), int)
    for j in range(len(stems)):
        u = np.unique(y[code == j]); assert len(u) == 1; pcls[j] = u[0]
    W = stratified_multiplicities(pcls, n_rep, seed).astype(float); ones = np.ones(len(stems))
    P = {k: d[[f"prob_{c}" for c in range(3)]].to_numpy(float) for k, d in data.items()}
    stats = {k: patient_stats(code, len(stems), y, P[k]) for k in data}
    point = {k: metrics_from_weights(stats[k], ones) for k in data}
    reps = {k: [metrics_from_weights(stats[k], W[r]) for r in range(n_rep)] for k in data}
    arr = lambda k, m: np.array([r[m] for r in reps[k]])
    rows, cls_rows, cm_rows, bin_rows = [], [], [], []
    for (m, a) in data:
        for k in METRICS:
            lo, hi = ci(arr((m, a), k)); rows.append(dict(model=m, arm=a, protocol=R.PROTOCOL[a], metric=k, estimate=point[m, a][k], ci_lo=lo, ci_hi=hi))
        for c, k in itertools.product(range(3), ("precision", "recall", "f1")):
            lo, hi = ci(arr((m, a), k)[:, c]); cls_rows.append(dict(model=m, arm=a, protocol=R.PROTOCOL[a], cls=R.CLASS_NAMES[c], metric=k,
                                                                   estimate=point[m, a][k][c], ci_lo=lo, ci_hi=hi, support=int((y == c).sum())))
        cm = M.confusion(y, P[m, a].argmax(1))
        for c, q in itertools.product(range(3), repeat=2): cm_rows.append(dict(model=m, arm=a, true_class=R.CLASS_NAMES[c], pred_class=R.CLASS_NAMES[q], count=int(cm[c, q])))
        n, sc, sa = M.ece_bins(P[m, a], y)
        for b in range(M.NBINS):
            bin_rows.append(dict(model=m, arm=a, bin=b, lo=b / M.NBINS, hi=(b + 1) / M.NBINS, n=int(n[b]),
                                 mean_confidence=sc[b] / n[b] if n[b] else np.nan, accuracy=sa[b] / n[b] if n[b] else np.nan))
    T = lambda n, r: pd.DataFrame(r).to_csv(out / n, index=False)
    T("metrics_summary_209.csv", rows); T("per_class_metrics_209.csv", cls_rows); T("confusion_matrices_209.csv", cm_rows); T("reliability_bins_209.csv", bin_rows)
    if "A209" in arms and "B209" in arms:
        T("protocol_differences_209.csv", [dict(model=m, metric=k, delta_image_minus_patient=point[m, "B209"][k] - point[m, "A209"][k],
                                                  ci_lo=ci(arr((m, "B209"), k) - arr((m, "A209"), k))[0], ci_hi=ci(arr((m, "B209"), k) - arr((m, "A209"), k))[1],
                                                  pairing="MATCHED: same images, same fold x class counts, shared stem draws") for m in models for k in METRICS])
    pairs, ranking, tops = [], [], []
    for a, k in itertools.product(arms, RANK_METRICS):
        for l, r in itertools.combinations(models, 2):
            diff = arr((l, a), k) - arr((r, a), k); lo, hi = ci(diff)
            pairs.append(dict(arm=a, metric=k, model_left=l, model_right=r, difference_left_minus_right=point[l, a][k] - point[r, a][k],
                              ci_lo=lo, ci_hi=hi, bootstrap_fraction_left_higher=float((diff > 0).mean())))
        ranked = sorted(models, key=lambda m: (-point[m, a][k], m))
        for place, m in enumerate(ranked, 1): ranking.append(dict(metric=k, arm=a, rank=place, model=m, estimate=point[m, a][k]))
        A = np.stack([arr((m, a), k) for m in models], axis=1)
        wins = np.isclose(A, A.max(1, keepdims=True), atol=1e-12, rtol=0); cr = wins / wins.sum(1, keepdims=True)
        for j, m in enumerate(models): tops.append(dict(metric=k, arm=a, model=m, bootstrap_top_ranked_frequency=float(cr[:, j].mean()), tie_policy="split credit"))
    T("pairwise_differences_209.csv", pairs); T("ranking_summary_209.csv", ranking); T("top_ranked_frequencies_209.csv", tops)
    if "A209" in arms and "B209" in arms and len(models) > 1:
        from scipy.stats import kendalltau
        T("ranking_protocol_tau_209.csv", [dict(metric=k, kendall_tau_A209_vs_B209=float(kendalltau([point[m, "A209"][k] for m in models], [point[m, "B209"][k] for m in models]).statistic)) for k in RANK_METRICS])
    robust = []
    if hist is not None and "A209" in arms:
        for m in models:
            h = hist[m]
            assert np.array_equal(h.image_id, ref.image_id) and np.array_equal(h.true_class, ref.true_class)
            ph = h[[f"prob_{c}" for c in range(3)]].to_numpy(float); sh = patient_stats(code, len(stems), y, ph)  # historical preds re-clustered by STEM
            for k in METRICS:
                d = np.array([metrics_from_weights(sh, W[r])[k] for r in range(n_rep)]); diff = arr((m, "A209"), k) - d; lo, hi = ci(diff)
                robust.append(dict(model=m, metric=k, A_hist=metrics_from_weights(sh, ones)[k], A209=point[m, "A209"][k], delta_A209_minus_A=point[m, "A209"][k] - metrics_from_weights(sh, ones)[k],
                                   ci_lo=lo, ci_hi=hi, subset="all_3064", pairing="ROBUSTNESS ONLY: same images, changed fold assignment (332 moved), stem-clustered; not a matched experiment"))
            if unchanged_mask is not None:
                msk = unchanged_mask; s1 = patient_stats(code[msk], len(stems), y[msk], P[m, "A209"][msk]); s0 = patient_stats(code[msk], len(stems), y[msk], ph[msk])
                for k in METRICS:
                    diff = np.array([metrics_from_weights(s1, W[r])[k] - metrics_from_weights(s0, W[r])[k] for r in range(n_rep)]); lo, hi = ci(diff)
                    robust.append(dict(model=m, metric=k, A_hist=metrics_from_weights(s0, ones)[k], A209=metrics_from_weights(s1, ones)[k],
                                       delta_A209_minus_A=metrics_from_weights(s1, ones)[k] - metrics_from_weights(s0, ones)[k], ci_lo=lo, ci_hi=hi,
                                       subset=f"images_with_unchanged_test_fold_n{int(msk.sum())}", pairing="ROBUSTNESS ONLY: train sets still differ"))
        T("robustness_original233_vs_A209.csv", robust)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(models), len(arms), figsize=(5 * len(arms), 3.5 * len(models)), squeeze=False, constrained_layout=True)
    bins = pd.DataFrame(bin_rows)
    for i, m in enumerate(models):
        for j, a in enumerate(arms):
            d = bins[(bins.model == m) & (bins.arm == a) & (bins.n > 0)]; ax = axes[i, j]
            ax.plot([0, 1], [0, 1], "k--", lw=1); ax.plot(d.mean_confidence, d.accuracy, "o-", ms=4)
            ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{m}: {R.PROTOCOL[a]}", xlabel="Mean confidence", ylabel="Observed accuracy")
    fig.savefig(out / "figures/reliability_diagrams_209.png", dpi=150); plt.close(fig)
    man = {"status": "PROPOSED PIPELINE", "models": list(models), "arms": list(arms), "n_images": int(len(ref)), "bootstrap_clusters": "normalised identity stems",
           "n_clusters": int(len(stems)), "bootstrap_replicates": n_rep, "bootstrap_seed": seed, "cluster_change_vs_historical": "233 raw patient IDs -> 209 stems",
           "historical_comparison": "robustness only" if hist is not None else "not computed"}
    (out / "analysis_manifest_209.json").write_text(json.dumps(man, indent=2))
    return point, reps, man


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", required=True, choices=tuple(R.SCENARIOS)); ap.add_argument("--runs", default=str(R.OUT))
    ap.add_argument("--out"); ap.add_argument("--n-rep", type=int, default=5000); a = ap.parse_args(argv)
    arms = R.SCENARIOS[a.scenario]; data = {(m, arm): load_arm(m, arm, a.runs) for m in R.MODELS for arm in arms}
    hist = {m: load_hist(m) for m in R.MODELS}
    o = pd.read_csv(ROOT / "data/splits/patient_disjoint_original.csv").sort_values("image_id")
    mask = (o.fold.values == data[R.MODELS[0], "A209"].test_fold.values)
    run_analysis(data, R.MODELS, arms, a.out or Path(a.runs) / "analysis", a.n_rep, 2026, hist if all(v is not None for v in hist.values()) else None, mask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
