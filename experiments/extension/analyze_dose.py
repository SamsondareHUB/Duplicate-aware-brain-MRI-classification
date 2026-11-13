"""Dose-response, difference-in-differences and Holm correction for the frozen exposure family (see PROTOCOL_FROZEN.md).
Per (model, fold): bootstrap over ALL fixed-test patients (class-stratified, 5000 draws, seed 2026, shared across arms/levels);
subsets E_L / N_L are formed by zeroing multiplicities. Seeds averaged at the metric level.
  c_L   = BA(SIB_L)-BA(CTRL_L) on E_L            n_L = same on N_L            did_L = c_L - n_L
  slope_sib = OLS slope of c_L on L/100 (L=25..100);  slope_did = OLS slope of did_L on L/100 over L=25,50,75 (N_100 has 3-6 patients)
Pooled slope = mean of the 4 cell slopes per draw. Holm family (frozen): 4 cells x BA(SIB-CTRL, Q) at L=100 + pooled dose-response slope.
The pooled slope reported in the family is slope_sib (slope_did is reported alongside, with its own Holm family)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import *
from ext_common import _stats, _metric_arrays
from ext_specs import MODELS, EXPO_SEEDS, EXT
from make_exposure import LEVELS, EXPOSURE_FOLDS
from analyze_exposure import rid, NREP

def ols(x, Y):  # Y: [R, k]
    xc = x - x.mean(); return (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)

def pv(a): return float(min(1.0, 2 * min((a <= 0).mean(), (a >= 0).mean())))
def ci(a): return np.percentile(a, [2.5, 97.5])

def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * ps[j]); adj[j] = min(1, run)
    return adj

def main(metric="balanced_accuracy"):
    cells, rows = {}, []
    for model in MODELS:
        for fold in EXPOSURE_FOLDS:
            pat = pd.read_csv(EXT / f"exposure/patients_fold{fold}.csv", dtype={"patient_id": str})
            ft = pd.read_csv(EXT / f"exposure/fixed_test_fold{fold}.csv", dtype={"patient_id": str})
            allp = np.array(sorted(ft.patient_id.unique()))
            fr = {(k, L, s): load_run(rid(model, fold, k, L, s)).sort_values("image_id") for s in EXPO_SEEDS for k in ("CTRL", "SIB") for L in LEVELS}
            cls = next(iter(fr.values())).groupby("patient_id").true_class.first().loc[allp].to_numpy()
            W = stratified_multiplicities(cls, NREP, 2026).astype(float); W1 = np.ones((1, len(allp)))
            S = {k: _stats(v, allp) for k, v in fr.items()}
            Q = np.isin(allp, pat.patient_id).astype(float)
            def diff(L, mask):
                out = 0; pt = 0
                for s in EXPO_SEEDS:
                    out = out + (_metric_arrays(S[("SIB", L, s)], W * mask)[metric] - _metric_arrays(S[("CTRL", L, s)], W * mask)[metric]) / len(EXPO_SEEDS)
                    pt = pt + (_metric_arrays(S[("SIB", L, s)], W1 * mask)[metric] - _metric_arrays(S[("CTRL", L, s)], W1 * mask)[metric]) / len(EXPO_SEEDS)
                return np.append(out, pt)
            c, n = [], []
            for L in LEVELS:
                E = np.isin(allp, pat.patient_id[pat[f"selected_L{L}"]]).astype(float)
                c.append(diff(L, E)); n.append(diff(L, 1 - E))
                rows.append({"model": model, "fold": fold, "level": L, "stat": "sib_on_E", "estimate": c[-1][-1], "ci_lo": ci(c[-1][:-1])[0], "ci_hi": ci(c[-1][:-1])[1], "p": pv(c[-1][:-1]), "nE": int(E.sum())})
                d = c[-1] - n[-1]
                rows.append({"model": model, "fold": fold, "level": L, "stat": "DiD_E_minus_N", "estimate": d[-1], "ci_lo": ci(d[:-1])[0], "ci_hi": ci(d[:-1])[1], "p": pv(d[:-1]), "nE": int((1 - E).sum())})
            c = np.array(c).T; n = np.array(n).T; did = c - n   # [R+1, 4]
            x = np.array(LEVELS) / 100.0
            ss = ols(x, c); sd = ols(x[:3], did[:, :3])
            cells[(model, fold)] = {"ss": ss, "sd": sd, "q100": diff(100, Q)}
            for lab, a in (("slope_sib_per_100pct", ss), ("slope_did_25_75_per_100pct", sd)):
                rows.append({"model": model, "fold": fold, "level": "all", "stat": lab, "estimate": a[-1], "ci_lo": ci(a[:-1])[0], "ci_hi": ci(a[:-1])[1], "p": pv(a[:-1])})
    # pooled + Holm
    fam = []
    for (model, fold), v in cells.items():
        a = v["q100"]; fam.append((f"BA SIB-CTRL L=100 Q {model} fold{fold}", a[-1], *ci(a[:-1]), pv(a[:-1])))
    for lab, key in (("pooled slope_sib", "ss"), ("pooled slope_did", "sd")):
        a = np.mean([v[key] for v in cells.values()], axis=0)
        pooled = (lab, a[-1], *ci(a[:-1]), pv(a[:-1]))
        if key == "ss": fam5 = fam + [pooled]
        else: did_row = pooled
    adj = holm([f[4] for f in fam5])
    F = pd.DataFrame([{"hypothesis": f[0], "estimate": f[1], "ci_lo": f[2], "ci_hi": f[3], "p_raw": f[4], "p_holm": h} for f, h in zip(fam5, adj)])
    F.loc[len(F)] = {"hypothesis": did_row[0] + " (alongside, not in family)", "estimate": did_row[1], "ci_lo": did_row[2], "ci_hi": did_row[3], "p_raw": did_row[4], "p_holm": np.nan}
    pd.DataFrame(rows).to_csv(EXT / f"exposure_dose_response_{metric}.csv", index=False)
    F.to_csv(EXT / f"exposure_holm_family_{metric}.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string()); print(F.round(4).to_string())

if __name__ == "__main__": main(*(sys.argv[1:]))
