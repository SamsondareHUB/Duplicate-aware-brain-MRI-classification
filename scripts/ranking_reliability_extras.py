"""Additional prediction-only diagnostics on model-ranking reliability (no training, no raw MRI).

Uses the committed OOF predictions and the committed paired patient draws (results/multimodel/bootstrap_patient_draws.npz).
Writes results/audit/ranking_extras.json and ranking_extras_*.csv. These are exploratory, post-hoc diagnostics
(NOT pre-registered); they quantify how well the observed ranking change is resolved under a single split seed and a
single training seed.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audit"; OUT.mkdir(exist_ok=True)
MM, SAN = ROOT / "results/multimodel", ROOT / "results/sanity"
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
NB = 15


def load(m, a):
    f = SAN / f"oof_predictions_{a}.csv" if m == "resnet50" else MM / f"oof_predictions_{m}_{a}.csv"
    return pd.read_csv(f, dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)


D = {(m, a): load(m, a) for m in MODELS for a in "AB"}
ref = D["resnet50", "A"]
y = ref.true_class.to_numpy(); pid = ref.patient_id.to_numpy(); fold = {a: D["resnet50", a].test_fold.to_numpy() for a in "AB"}
patients = np.array(sorted(set(pid))); code = np.searchsorted(patients, pid); P = len(patients)
W = np.load(MM / "bootstrap_patient_draws.npz", allow_pickle=True)["multiplicities"].astype(float)
nimg = np.bincount(code, minlength=P).astype(float)
probs = {k: v[[f"prob_{i}" for i in range(3)]].to_numpy(float) for k, v in D.items()}


def per_patient(p, wimg):
    """sufficient stats with per-image weights wimg (1 = slice-weighted)."""
    pred = p.argmax(1); cm = np.zeros((P, 9)); np.add.at(cm, (code, y * 3 + pred), wimg)
    return cm


def metrics_from(cm_rep):
    C = cm_rep.reshape(-1, 3, 3); tp = np.einsum("rii->ri", C); rec = tp / C.sum(2); prec = tp / C.sum(1)
    return {"balanced_accuracy": rec.mean(1), "macro_f1": (2 * prec * rec / (prec + rec)).mean(1), "accuracy": tp.sum(1) / C.sum((1, 2))}


def brier_nll(p):
    br = ((p - np.eye(3)[y]) ** 2).sum(1); nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))
    return np.bincount(code, br, P), np.bincount(code, nll, P)


ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
res, R = {}, {}
for k, p in probs.items():
    M = metrics_from(W @ per_patient(p, np.ones(len(y))))
    b, n = brier_nll(p); M["brier"] = (W @ b) / (W @ nimg); M["nll"] = (W @ n) / (W @ nimg)
    R[k] = M

# 1. difference-in-differences for the top-two architectures + all pairs
did = {}
for met in ("balanced_accuracy", "macro_f1", "accuracy"):
    for a, b in itertools.combinations(MODELS, 2):
        dA = R[a, "A"][met] - R[b, "A"][met]; dB = R[a, "B"][met] - R[b, "B"][met]; inter = dB - dA
        did[f"{met}|{a}-{b}"] = {"diff_patient_disjoint": [float(dA.mean()), *ci(dA)], "diff_imagelevel": [float(dB.mean()), *ci(dB)],
                                 "interaction_B_minus_A": [float(inter.mean()), *ci(inter)],
                                 "ci_A_excl0": bool(ci(dA)[0] > 0 or ci(dA)[1] < 0), "ci_B_excl0": bool(ci(dB)[0] > 0 or ci(dB)[1] < 0),
                                 "interaction_excl0": bool(ci(inter)[0] > 0 or ci(inter)[1] < 0)}
res["pairwise_and_interaction"] = did
res["n_interaction_contrasts_excluding_0"] = {"of": len(did), "n": sum(d["interaction_excl0"] for d in did.values()), "which": [k for k, d in did.items() if d["interaction_excl0"]]}
pd.DataFrame([{"contrast": k, **{kk: vv for kk, vv in d.items()}} for k, d in did.items()]).to_csv(OUT / "ranking_extras_pairwise_interaction.csv", index=False)

# 2. bootstrap rank identity / tau distribution (descriptive)
tau_out = {}
for met in ("balanced_accuracy", "macro_f1"):
    arr = {a: np.stack([R[m, a][met] for m in MODELS], 1) for a in "AB"}
    topA, topB = arr["A"].argmax(1), arr["B"].argmax(1)
    taus = np.array([kendalltau(arr["A"][r], arr["B"][r]).statistic for r in range(len(W))])
    tau_out[met] = {"frac_reps_top_model_differs_between_protocols": float((topA != topB).mean()),
                    "frac_reps_densenet_top_A_and_resnet_top_B": float(((topA == 3) & (topB == 0)).mean()),
                    "frac_reps_resnet_top_A_and_resnet_top_B": float(((topA == 0) & (topB == 0)).mean()),
                    "tau_between_protocols_bootstrap_mean": float(np.nanmean(taus)), "tau_bootstrap_ci": [float(x) for x in np.nanpercentile(taus, [2.5, 97.5])],
                    "tau_point": float(kendalltau([R[m, "A"][met].mean() for m in MODELS], [R[m, "B"][met].mean() for m in MODELS]).statistic),
                    "frac_reps_tau_gt_0": float((taus > 0).mean()), "frac_reps_tau_eq_1": float((taus == 1).mean()),
                    "frac_reps_full_ordering_identical_A_B": float((np.argsort(-arr["A"], 1) == np.argsort(-arr["B"], 1)).all(1).mean())}
res["rank_identity_and_tau"] = tau_out

# 3. metric sensitivity of point rankings
sens = []
for a in "AB":
    pt = {m: {k: v.mean() for k, v in R[m, a].items()} for m in MODELS}
    # point estimates (use exact full-sample values, not bootstrap mean)
    for met in ("balanced_accuracy", "macro_f1", "accuracy", "brier", "nll"):
        full = {}
        for m in MODELS:
            p = probs[m, a]; pred = p.argmax(1)
            from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
            full[m] = {"balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred),
                       "brier": float(((p - np.eye(3)[y]) ** 2).sum(1).mean()), "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())}[met]
        order = sorted(MODELS, key=lambda m: full[m] if met in ("brier", "nll") else -full[m])
        sens.append({"arm": a, "metric": met, "ranking_best_to_worst": order, **{m: full[m] for m in MODELS}})
pd.DataFrame(sens).to_csv(OUT / "ranking_extras_metric_sensitivity.csv", index=False)
res["metric_sensitivity_top_model"] = {f"{s['arm']}|{s['metric']}": s["ranking_best_to_worst"][0] for s in sens}

# 4. patient-weighted (each patient counts equally) vs slice-weighted
pw = []
for a in "AB":
    for m in MODELS:
        p = probs[m, a]; cm = per_patient(p, 1.0 / nimg[code])
        M = metrics_from(cm.sum(0, keepdims=True)); Mb = metrics_from(W @ cm)
        pw.append({"arm": a, "model": m, "pw_balanced_accuracy": float(M["balanced_accuracy"][0]), "pw_accuracy": float(M["accuracy"][0]),
                   "pw_ba_ci_lo": ci(Mb["balanced_accuracy"])[0], "pw_ba_ci_hi": ci(Mb["balanced_accuracy"])[1]})
pwdf = pd.DataFrame(pw); pwdf.to_csv(OUT / "ranking_extras_patient_weighted.csv", index=False)
res["patient_weighted_top_model"] = {a: pwdf[pwdf.arm == a].sort_values("pw_balanced_accuracy", ascending=False).model.tolist() for a in "AB"}
pwd = {}
for a in "AB":
    cmR = per_patient(probs["resnet50", a], 1.0 / nimg[code]); cmD = per_patient(probs["densenet121", a], 1.0 / nimg[code])
    d = metrics_from(W @ cmR)["balanced_accuracy"] - metrics_from(W @ cmD)["balanced_accuracy"]
    pwd[a] = {"resnet50_minus_densenet121_pw_BA": [float(metrics_from(cmR.sum(0, keepdims=True))["balanced_accuracy"][0] - metrics_from(cmD.sum(0, keepdims=True))["balanced_accuracy"][0]), *ci(d)]}
res["patient_weighted_resnet_vs_densenet"] = pwd

# 5. per-fold ResNet50 - DenseNet121 balanced accuracy
pf = []
for a in "AB":
    for f in range(1, 6):
        msk = fold[a] == f; row = {"arm": a, "fold": f, "n": int(msk.sum())}
        from sklearn.metrics import balanced_accuracy_score
        for m in MODELS:
            row[m] = balanced_accuracy_score(y[msk], probs[m, a][msk].argmax(1))
        row["resnet_minus_densenet"] = row["resnet50"] - row["densenet121"]; pf.append(row)
pfd = pd.DataFrame(pf); pfd.to_csv(OUT / "ranking_extras_per_fold_ba.csv", index=False)
res["per_fold_resnet_minus_densenet_sign"] = {a: [int(np.sign(v)) for v in pfd[pfd.arm == a].resnet_minus_densenet] for a in "AB"}

# 6. naive image-iid stratified bootstrap vs patient-cluster bootstrap CI width (resnet50 - densenet121 BA)
rng = np.random.default_rng(31415); nrep = 5000
naive = {}
from sklearn.metrics import balanced_accuracy_score
for a in "AB":
    idx = {c: np.where(y == c)[0] for c in range(3)}; dif = np.zeros(nrep)
    cr = (probs["resnet50", a].argmax(1) == y); cd = (probs["densenet121", a].argmax(1) == y)
    for r in range(nrep):
        s = np.concatenate([rng.choice(idx[c], len(idx[c])) for c in range(3)])
        dif[r] = np.mean([cr[s][y[s] == c].mean() - cd[s][y[s] == c].mean() for c in range(3)])
    cl = R["resnet50", a]["balanced_accuracy"] - R["densenet121", a]["balanced_accuracy"]
    naive[a] = {"naive_image_bootstrap_ci": ci(dif), "patient_cluster_ci": ci(cl), "naive_width": float(np.diff(ci(dif))[0]), "cluster_width": float(np.diff(ci(cl))[0]),
                "naive_excludes_zero": bool(ci(dif)[0] > 0 or ci(dif)[1] < 0), "cluster_excludes_zero": bool(ci(cl)[0] > 0 or ci(cl)[1] < 0)}
res["naive_vs_cluster_resnet_minus_densenet_BA"] = naive

# 7. cluster CI widths A vs B (apparent precision)
ms = pd.read_csv(MM / "metrics_summary.csv"); ms["width"] = ms.ci_hi - ms.ci_lo
res["mean_ci_width_BA_by_arm"] = ms[ms.metric == "balanced_accuracy"].groupby("arm").width.mean().to_dict()
# 8. resolved pairs count
res["n_resolved_pairs_of_6"] = {f"{met}|{a}": sum(1 for (x, z) in itertools.combinations(MODELS, 2)
                                                   if did[f"{met}|{x}-{z}"][f"ci_{a}_excl0"]) for met in ("balanced_accuracy", "macro_f1") for a in "AB"}
(OUT / "ranking_extras.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps({k: res[k] for k in res if k != "pairwise_and_interaction"}, indent=1, default=float))
print(json.dumps({k: v for k, v in did.items() if k.startswith("balanced_accuracy|resnet50-densenet121") or k.startswith("macro_f1|resnet50-densenet121")}, indent=1))
