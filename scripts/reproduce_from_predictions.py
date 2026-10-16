"""Prediction-only independent reproduction of every headline number (no raw MRI, no training, no torch).

Deliberately does NOT import src/leakage/metrics.py or bootstrap.py for the metrics themselves: point metrics use
scikit-learn, ECE/Brier/bootstrap are re-implemented here from scratch. Only the draw generator is imported, solely to
test that the stored bootstrap_patient_draws.npz is what seed 2026 produces.

Source of numerical truth = committed OOF prediction files + frozen split CSVs. Existing summary CSVs/Markdown are the
objects being CHECKED. Nothing in results/ is modified; outputs go to results/audit/.

Usage: python scripts/reproduce_from_predictions.py            (exit code 1 on any FAIL)
"""
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
MM, SAN, OUT = ROOT / "results/multimodel", ROOT / "results/sanity", ROOT / "results/audit"
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
ARMS = {"A": "patient_disjoint_original.csv", "B": "imagelevel_seed11.csv"}  # PR#1 labels: A=patient-disjoint
SHA = {"A": "61bec5aaf9a5f4fce4adf89b186ffd6479eb0f6095ee20b8f60ef44274a3e49e",
       "B": "e8b0442c107396fe66a989b8d448aa0e524c1c554823a711db3cc84beeee74f2"}
NB = 15
LOG = []


def check(name, ok, detail=""):
    LOG.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    if not ok:
        print(f"FAIL  {name}  {detail}")
    return ok


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ece15(p, y):
    conf, corr = p.max(1), (p.argmax(1) == y)
    b = np.clip(np.ceil(conf * NB).astype(int) - 1, 0, NB - 1)
    tot = 0.0
    for k in range(NB):
        m = b == k
        if m.any():
            tot += m.sum() * abs(corr[m].mean() - conf[m].mean())
    return tot / len(y)


def brier(p, y):
    return float(((p - np.eye(3)[y]) ** 2).sum(1).mean())


def load(model, arm):
    f = (SAN / f"oof_predictions_{arm}.csv") if model == "resnet50" else (MM / f"oof_predictions_{model}_{arm}.csv")
    return pd.read_csv(f, dtype={"patient_id": str})


def main():
    OUT.mkdir(exist_ok=True)
    splits = {a: pd.read_csv(ROOT / "data/splits" / fn, dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)
              for a, fn in ARMS.items()}
    for a, fn in ARMS.items():
        check(f"split sha256 {a} ({fn})", sha256(ROOT / "data/splits" / fn) == SHA[a])
        s = splits[a]
        check(f"split {a}: 3064 unique images, 233 patients, folds 1-5",
              len(s) == 3064 and s.image_id.is_unique and s.patient_id.nunique() == 233 and set(s.fold) == {1, 2, 3, 4, 5})
    pf = splits["A"].groupby("patient_id").fold.nunique()
    pb = splits["B"].groupby("patient_id").fold.nunique()
    check("A is patient-disjoint (0 patients in >1 fold)", (pf == 1).sum() == 233)
    n_split_b = int((pb > 1).sum())
    check("B is NOT patient-disjoint (patients spanning >1 fold)", n_split_b > 0, f"{n_split_b} of 233 patients span >=2 folds")
    check("A/B identical fold sizes", (splits["A"].fold.value_counts().sort_index().values
                                       == splits["B"].fold.value_counts().sort_index().values).all())
    check("A/B identical fold x class table",
          (pd.crosstab(splits["A"].fold, splits["A"].label).values == pd.crosstab(splits["B"].fold, splits["B"].label).values).all())
    ref = splits["A"]
    y = ref.label.to_numpy() - 1
    patients = np.array(sorted(ref.patient_id.unique()))
    code = np.searchsorted(patients, ref.patient_id.to_numpy())
    pcls = np.array([np.unique(y[code == j])[0] for j in range(233)])
    check("each patient has exactly one class", all(len(np.unique(y[code == j])) == 1 for j in range(233)))

    P, inv, rows = {}, [], []
    for m, a in itertools.product(MODELS, ARMS):
        d = load(m, a).sort_values("image_id").reset_index(drop=True)
        tag = f"{m}/{a}"
        p = d[[f"prob_{k}" for k in range(3)]].to_numpy(float)
        ok = (len(d) == 3064 and d.image_id.is_unique and np.array_equal(d.image_id, splits[a].image_id)
              and np.array_equal(d.patient_id, splits[a].patient_id) and np.array_equal(d.test_fold, splits[a].fold)
              and np.array_equal(d.true_class, y))
        check(f"{tag}: 3064 OOF rows, ids/patients/folds/labels == split file", ok)
        check(f"{tag}: finite probs in [0,1], rows sum to 1 (max dev {abs(p.sum(1)-1).max():.2e})",
              np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and abs(p.sum(1) - 1).max() < 1e-6)
        check(f"{tag}: pred_class == argmax(prob)", np.array_equal(d.pred_class, p.argmax(1)))
        check(f"{tag}: no duplicated (image) rows", not d.image_id.duplicated().any())
        # per-fold files vs consolidated OOF
        runs = (sorted(SAN.glob(f"runs/{a}_fold*_run.json")) if m == "resnet50"
                else sorted((MM / "runs").glob(f"{m}_{a}_fold*_run.json")))
        check(f"{tag}: 5 run records", len(runs) == 5)
        commits, seeds, eps = set(), set(), set()
        for r in runs:
            j = json.loads(r.read_text())
            commits.add(j["experiment_input_commit"]); seeds.add(j["train_seed"]); eps.add(len(j["epochs"]))
            check(f"{tag} {r.name}: split_sha256 matches {a}", j["split_sha256"] == SHA[a])
            fold_pred = pd.read_csv(str(r).replace("_run.json", "_predictions.csv"), dtype={"patient_id": str})
            tf = j["fold"]
            sub = d[d.test_fold == tf]
            check(f"{tag} fold{tf}: per-fold predictions == consolidated OOF rows and n_test",
                  len(fold_pred) == j["n_test"] == len(sub)
                  and np.allclose(fold_pred.sort_values("image_id")[["prob_0", "prob_1", "prob_2"]].to_numpy(),
                                  sub[["prob_0", "prob_1", "prob_2"]].to_numpy(), atol=1e-12, rtol=0))  # per-fold vs consolidated files differ by CSV float round-trip (~1e-23)
            # train/test disjointness by image in the fold definition
            test_ids = set(splits[a].image_id[splits[a].fold == tf])
            check(f"{tag} fold{tf}: n_train+n_test=3064", j["n_train"] + j["n_test"] == 3064 and j["n_test"] == len(test_ids))
        inv.append({"model": m, "arm": a, "n_run_records": len(runs), "train_seeds": sorted(seeds), "epochs": sorted(eps),
                    "input_commits": sorted(c[:7] for c in commits), "weights": json.loads(runs[0].read_text()).get("weights"),
                    "device": json.loads(runs[0].read_text()).get("device")})
        check(f"{tag}: train_seed 0 and 10 epochs for all folds", seeds == {0} and eps == {10})
        P[m, a] = p
        pred = p.argmax(1)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        rows.append({"model": m, "arm": a, "balanced_accuracy": balanced_accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred),
                     "ece": ece15(p, y), "brier": brier(p, y),
                     **{f"{k}_c{c}": v[c] for k, v in (("precision", pr), ("recall", rc), ("f1", f1)) for c in range(3)},
                     **{f"cm_{i}{j}": confusion_matrix(y, pred, labels=[0, 1, 2])[i, j] for i in range(3) for j in range(3)}})
    pt = pd.DataFrame(rows).set_index(["model", "arm"])
    pt.to_csv(OUT / "independent_point_metrics.csv")
    pd.DataFrame(inv).to_csv(OUT / "run_inventory.csv", index=False)
    check("run inventory: 40 run records (10 ResNet50 + 30 new)", sum(i["n_run_records"] for i in inv) == 40,
          f"total records counted = {sum(i['n_run_records'] for i in inv)}")

    # --- compare with committed summaries --------------------------------------------------------------------
    ms = pd.read_csv(MM / "metrics_summary.csv")
    worst = 0.0
    for _, r in ms.iterrows():
        v = pt.loc[(r.model, r.arm), r.metric]
        worst = max(worst, abs(v - r.estimate))
    check("metrics_summary.csv point estimates == independent recompute", worst < 1e-9, f"max abs diff {worst:.2e}")
    pc = pd.read_csv(MM / "per_class_metrics.csv")
    cn = {"meningioma": 0, "glioma": 1, "pituitary": 2}
    worst = max(abs(pt.loc[(r.model, r.arm), f"{r.metric}_c{cn[r['class']]}"] - r.estimate) for _, r in pc.iterrows())
    check("per_class_metrics.csv == independent recompute", worst < 1e-9, f"max abs diff {worst:.2e}")
    cm = pd.read_csv(MM / "confusion_matrices.csv")
    worst = max(abs(pt.loc[(r.model, r.arm), f"cm_{cn[r.true_class]}{cn[r.pred_class]}"] - r["count"]) for _, r in cm.iterrows())
    check("confusion_matrices.csv == independent recompute", worst == 0, f"max abs diff {worst}")
    pdiff = pd.read_csv(MM / "protocol_differences.csv")
    worst = max(abs((pt.loc[(r.model, "B"), r.metric] - pt.loc[(r.model, "A"), r.metric]) - r.delta_image_minus_patient)
                for _, r in pdiff.iterrows())
    check("protocol_differences.csv point deltas == independent recompute", worst < 1e-9, f"max abs diff {worst:.2e}")
    # published Markdown table (4 dp)
    md = (ROOT / "reports/MULTIMODEL_RESULTS.md").read_text()
    pn = {"Patient-disjoint": "A", "Image-level seed 11": "B"}
    n_md = 0
    bad = []
    for line in md.splitlines():
        mm = re.match(r"\| (resnet50|mobilenet_v2|efficientnet_b0|densenet121) \| (Patient-disjoint|Image-level seed 11) \| (.*)\|$", line)
        if mm and "(" in mm.group(3) and "−" not in line.split("|")[3]:
            vals = [float(x.strip().split(" ")[0]) for x in mm.group(3).split("|") if x.strip()]
            if len(vals) == 5:
                for k, v in zip(("balanced_accuracy", "macro_f1", "accuracy", "ece", "brier"), vals):
                    n_md += 1
                    if abs(round(pt.loc[(mm.group(1), pn[mm.group(2)]), k], 4) - v) > 1e-9:
                        bad.append((mm.group(1), mm.group(2), k, v))
    check("MULTIMODEL_RESULTS.md headline table (40 cells) == independent recompute at 4 dp", n_md == 40 and not bad, f"cells={n_md} mismatches={bad}")

    # --- bootstrap ---------------------------------------------------------------------------------------------
    z = np.load(MM / "bootstrap_patient_draws.npz", allow_pickle=True)
    W = z["multiplicities"].astype(float)
    check("draws: shape (5000,233) and patient order == sorted ids", W.shape == (5000, 233) and np.array_equal(z["patient_id"], patients))
    check("draws: patients are the resampling unit; per-class multiplicity totals == per-class patient counts, every replicate",
          all((W[:, pcls == c].sum(1) == (pcls == c).sum()).all() for c in range(3)))
    sys.path.insert(0, str(ROOT / "src"))
    from leakage.bootstrap import stratified_multiplicities
    check("draws: stored matrix == regeneration with seed 2026", np.array_equal(stratified_multiplicities(pcls, 5000, 2026), W.astype(int)))
    Z = {}

    def stats(p):
        pred = p.argmax(1)
        cmx = np.zeros((233, 9)); np.add.at(cmx, (code, y * 3 + pred), 1)
        b = np.clip(np.ceil(p.max(1) * NB).astype(int) - 1, 0, NB - 1)
        n = np.zeros((233, NB)); sc = np.zeros((233, NB)); sa = np.zeros((233, NB))
        np.add.at(n, (code, b), 1); np.add.at(sc, (code, b), p.max(1)); np.add.at(sa, (code, b), (pred == y))
        br = np.bincount(code, ((p - np.eye(3)[y]) ** 2).sum(1), 233)
        return cmx, n, sc, sa, br

    def boot(p, Wm):
        cmx, n, sc, sa, br = stats(p)
        C = (Wm @ cmx).reshape(-1, 3, 3)
        tp = np.einsum("rii->ri", C); rec = tp / C.sum(2); prec = tp / C.sum(1); f1 = 2 * prec * rec / (prec + rec)
        N, SC, SA = Wm @ n, Wm @ sc, Wm @ sa
        ece = np.abs(SA - SC).sum(1) / N.sum(1)
        return {"balanced_accuracy": rec.mean(1), "macro_f1": f1.mean(1), "accuracy": tp.sum(1) / C.sum((1, 2)),
                "ece": ece, "brier": (Wm @ br) / (Wm @ np.bincount(code, minlength=233))}
    R = {k: boot(P[k], W) for k in P}
    ci = lambda v: np.percentile(v, [2.5, 97.5])
    worst = 0.0
    for _, r in ms.iterrows():
        lo, hi = ci(R[r.model, r.arm][r.metric]); worst = max(worst, abs(lo - r.ci_lo), abs(hi - r.ci_hi))
    check("metrics_summary.csv 95% percentile CIs == independent bootstrap on stored draws", worst < 1e-9, f"max abs diff {worst:.2e}")
    worst = 0.0
    for _, r in pdiff.iterrows():
        lo, hi = ci(R[r.model, "B"][r.metric] - R[r.model, "A"][r.metric]); worst = max(worst, abs(lo - r.ci_lo), abs(hi - r.ci_hi))
    check("protocol_differences.csv paired CIs == independent", worst < 1e-9, f"max abs diff {worst:.2e}")
    pw = pd.read_csv(MM / "pairwise_differences.csv"); worst = 0.0
    for _, r in pw.iterrows():
        dd = R[r.model_left, r.arm][r.metric] - R[r.model_right, r.arm][r.metric]
        lo, hi = ci(dd)
        worst = max(worst, abs(lo - r.ci_lo), abs(hi - r.ci_hi), abs(pt.loc[(r.model_left, r.arm), r.metric] - pt.loc[(r.model_right, r.arm), r.metric] - r.difference_left_minus_right),
                    abs((dd > 0).mean() - r.bootstrap_fraction_left_higher))
    check("pairwise_differences.csv (est, CI, fraction>0; 24 rows) == independent", worst < 1e-9, f"max abs diff {worst:.2e}")
    tf = pd.read_csv(MM / "top_ranked_frequencies.csv"); worst = 0.0
    for (k, a), g in tf.groupby(["metric", "arm"]):
        arr = np.stack([R[m, a][k] for m in MODELS], 1)
        w = np.isclose(arr, arr.max(1, keepdims=True), atol=1e-12, rtol=0); cr = w / w.sum(1, keepdims=True)
        for j, m in enumerate(MODELS):
            worst = max(worst, abs(cr[:, j].mean() - g[g.model == m].bootstrap_top_ranked_frequency.iloc[0]))
    check("top_ranked_frequencies.csv == independent (ties split equally)", worst < 1e-9, f"max abs diff {worst:.2e}")
    rk = pd.read_csv(MM / "ranking_summary.csv"); ok = True
    for k in ("balanced_accuracy", "macro_f1"):
        tau = kendalltau([pt.loc[(m, "A"), k] for m in MODELS], [pt.loc[(m, "B"), k] for m in MODELS]).statistic
        ok &= abs(tau - rk[rk.metric == k].kendall_tau_between_protocols.iloc[0]) < 1e-12
        for a in ARMS:
            order = sorted(MODELS, key=lambda m: -pt.loc[(m, a), k])
            ok &= list(rk[(rk.metric == k) & (rk.arm == a)].sort_values("rank").model) == order
    check("ranking_summary.csv orders and Kendall tau-b == independent", ok)
    # exact ties in point estimates?
    for k in ("balanced_accuracy", "macro_f1"):
        for a in ARMS:
            v = np.array([pt.loc[(m, a), k] for m in MODELS]); check(f"no exact point-estimate ties ({k},{a})", len(set(np.round(v, 12))) == 4)

    # --- fresh, independent bootstrap (different seed, 20000 reps) to test CI stability -------------------------
    rng = np.random.default_rng(987654321)
    nrep = 20000
    W2 = np.zeros((nrep, 233))
    for c in range(3):
        idx = np.where(pcls == c)[0]
        d = rng.integers(0, len(idx), size=(nrep, len(idx)))
        W2[:, idx] = np.apply_along_axis(lambda r: np.bincount(r, minlength=len(idx)), 1, d)
    R2 = {k: boot(P[k], W2) for k in P}
    fresh = []
    for m in MODELS:
        for k in ("balanced_accuracy", "macro_f1"):
            for a in ARMS:
                lo, hi = ci(R2[m, a][k]); fresh.append({"model": m, "arm": a, "metric": k, "ci_lo_fresh": lo, "ci_hi_fresh": hi})
    fr = pd.DataFrame(fresh).merge(ms, on=["model", "arm", "metric"])
    mx = max((fr.ci_lo_fresh - fr.ci_lo).abs().max(), (fr.ci_hi_fresh - fr.ci_hi).abs().max())
    check("fresh 20,000-rep bootstrap (new seed) reproduces published CIs within 0.003", mx < 0.003, f"max |diff| {mx:.4f}")
    fr.to_csv(OUT / "fresh_bootstrap_ci_check.csv", index=False)
    sc = {"bootstrap_seed_fresh": 987654321, "n_rep_fresh": nrep}
    for k in ("balanced_accuracy", "macro_f1"):
        for a in ARMS:
            dd = R2["resnet50", a][k] - R2["densenet121", a][k]
            sc[f"{k}_{a}_resnet50_minus_densenet121"] = {"est": float(pt.loc[("resnet50", a), k] - pt.loc[("densenet121", a), k]),
                                                         "ci": ci(dd).tolist(), "ci_includes_zero": bool(ci(dd)[0] <= 0 <= ci(dd)[1])}
    (OUT / "fresh_bootstrap_resnet_vs_densenet.json").write_text(json.dumps(sc, indent=2))
    for k in ("balanced_accuracy", "macro_f1"):
        for a in ARMS:
            check(f"CLAIM BOUNDARY: ResNet50-DenseNet121 {k} CI includes 0 under {a} (stored draws)",
                  (lambda lo, hi: lo <= 0 <= hi)(*ci(R["resnet50", a][k] - R["densenet121", a][k])))

    (OUT / "reproduction_checks.json").write_text(json.dumps(LOG, indent=1))
    fails = [l for l in LOG if l["status"] == "FAIL"]
    print(f"{len(LOG) - len(fails)}/{len(LOG)} checks PASS; {len(fails)} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
