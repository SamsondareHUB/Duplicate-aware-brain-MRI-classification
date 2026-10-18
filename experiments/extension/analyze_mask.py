"""Tumor-region mechanism analysis from results/extension/mask/*_variants.csv.gz (analysis-time diagnostics only; masks never used in training).
(1) Exposure: per model x fold, CLEAN / CTRL_L100 / SIB_L100, 2-seed mean-probability ensemble, all fixed-test patients (== exposed set at L=100 for eligible patients).
    Reports BA per input variant and SIB-CTRL / SIB-CLEAN differences with paired class-stratified patient-cluster bootstrap (5000, seed 2026).
(2) Split protocol: per model, seed-101 out-of-fold A vs B per variant; B-A per variant.
Variants: full, tumor_only, tumor_dil_only (tumor+7px context), bg_only (tumor+context removed), zero_tight (tumor zeroed), zero_shifted (artifact control hole at non-tumor location)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import *
from ext_specs import MODELS, EXT, RUNS
from analyze_exposure import rid
MASK = ROOT / "results/extension/mask"
VARS = ("full", "tumor_only", "tumor_dil_only", "bg_only", "zero_tight", "zero_shifted")

def vframe(run_id, var):
    m = pd.read_csv(MASK / f"{run_id}_variants.csv.gz"); m = m[m.variant == var].drop(columns=["run_id", "variant", "shift_overlap"])
    base = load_run(run_id)[["image_id", "patient_id", "true_class"]]
    return base.merge(m, on="image_id")

def ens(rids, var):
    fs = [vframe(r, var).sort_values("image_id").reset_index(drop=True) for r in rids]
    o = fs[0].copy()
    for k in range(3): o[f"prob_{k}"] = np.mean([f[f"prob_{k}"].to_numpy() for f in fs], 0)
    return o

def main():
    out = []
    for model in MODELS:
        for fold in (2, 4):
            conds = {"CLEAN": [rid(model, fold, "CLEAN", 0, s) for s in (0, 1)], "CTRL": [rid(model, fold, "CTRL", 100, s) for s in (0, 1)], "SIB": [rid(model, fold, "SIB", 100, s) for s in (0, 1)]}
            frames, contr = {}, {}
            for v in VARS:
                for c, r in conds.items(): frames[f"{c}|{v}"] = ens(r, v)
                contr[f"{v}: SIB-CTRL"] = (f"SIB|{v}", f"CTRL|{v}"); contr[f"{v}: SIB-CLEAN"] = (f"SIB|{v}", f"CLEAN|{v}")
                contr[f"{v}: CTRL-CLEAN"] = (f"CTRL|{v}", f"CLEAN|{v}")
            for c in conds:  # artifact-controlled tumor-specific drop: zero_tight vs zero_shifted
                contr[f"{c}: zero_tight - zero_shifted"] = (f"{c}|zero_tight", f"{c}|zero_shifted")
                contr[f"{c}: tumor_only - bg_only"] = (f"{c}|tumor_only", f"{c}|bg_only")
            R = paired_bootstrap(frames, contr, keys=("balanced_accuracy", "brier"))
            R.insert(0, "fold", fold); R.insert(0, "model", model); R.insert(0, "analysis", "exposure"); out.append(R)
        frames, contr = {}, {}
        for v in VARS:
            for k in "AB":
                frames[f"{k}|{v}"] = pd.concat([ vframe(f"split_{model}_{k}_seed101_fold{f}", v) for f in range(1, 6)]).sort_values("image_id").reset_index(drop=True)
            contr[f"{v}: B-A"] = (f"B|{v}", f"A|{v}")
        R = paired_bootstrap(frames, contr, keys=("balanced_accuracy", "brier"))
        R.insert(0, "fold", "oof5"); R.insert(0, "model", model); R.insert(0, "analysis", "split_seed101"); out.append(R)
    R = pd.concat(out); R.to_csv(EXT / "mask_region_results.csv", index=False)
    pd.set_option("display.width", 220)
    print(R[(R.metric == "balanced_accuracy") & ~R.item.str.contains("CTRL-CLEAN")].round(4).to_string())

if __name__ == "__main__": main()
