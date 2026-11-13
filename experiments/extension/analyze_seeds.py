"""Split-seed and training-seed analyses. Usage: python analyze_seeds.py {split|tseed}
Each replicate = pooled out-of-fold predictions of 5 folds (3,064 images). Historical reference replicate (split: seed 11 /
cvind folds; tseed: training seed 0) is read from the committed PR #1/#2 predictions (not re-run).
Bootstrap = within-replicate patient-cluster resampling (uncertainty over patients for fixed trained models). It is NOT
variation over splits or training seeds; across-replicate spread is reported separately and descriptively."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_common import *
from ext_specs import SPLIT_SEEDS, TRAIN_SEEDS_NEW

HIST = {("resnet50", "A"): "results/sanity/oof_predictions_A.csv", ("resnet50", "B"): "results/sanity/oof_predictions_B.csv",
        **{("densenet121", a): f"results/multimodel/oof_predictions_densenet121_{a}.csv" for a in "AB"}}
def hist(m, a):
    return pd.read_csv(ROOT / HIST[(m, a)], dtype={"patient_id": str}).sort_values("image_id").reset_index(drop=True)


def replicate_frames(kind, tag):
    """returns {(model,arm): frame} or None if incomplete"""
    out = {}
    for m in MODELS:
        for a in "AB":
            if tag == "hist": out[(m, a)] = hist(m, a); continue
            ids = [f"split_{m}_{a}_seed{tag}_fold{k}" if kind == "split" else f"tseed_{m}_{a}_ts{tag}_fold{k}" for k in range(1, 6)]
            if not all(verify(i) for i in ids): return None
            out[(m, a)] = oof_frame(ids)
    return out


MODELS = ("resnet50", "densenet121")


def main(kind):
    tags = ["hist"] + [str(s) for s in (SPLIT_SEEDS if kind == "split" else TRAIN_SEEDS_NEW)]
    res, summ, missing = [], [], []
    for tag in tags:
        fr = replicate_frames(kind, tag)
        if fr is None: missing.append(tag); continue
        label = {"hist": "seed11 (historical)" if kind == "split" else "train seed 0 (historical)"}.get(tag, tag)
        frames = {f"{m}|{a}": f for (m, a), f in fr.items()}
        con = {}
        for m in MODELS: con[f"{m}: B-A"] = (f"{m}|B", f"{m}|A")
        for a in "AB": con[f"resnet50-densenet121 [{a}]"] = (f"resnet50|{a}", f"densenet121|{a}")
        con["interaction: (R-D)[B]-(R-D)[A]"] = None
        con = {k: v for k, v in con.items() if v}
        t = paired_bootstrap(frames, con); t.insert(0, "replicate", label); res.append(t)
        # interaction via explicit rep arithmetic is omitted; derived from point estimates in the summary
    if not res: print("no complete replicate"); return
    R = pd.concat(res, ignore_index=True)
    out = ROOT / "experiments/extension" / f"{kind}_seed_bootstrap_table.csv"; R.to_csv(out, index=False)
    # across-replicate summary of point estimates
    P = R[R.metric.isin(("balanced_accuracy", "macro_f1", "ece", "brier"))]
    wide = P.pivot_table(index=["item", "metric"], columns="replicate", values="estimate")
    wide["mean"] = wide.mean(axis=1); wide["sd"] = wide.iloc[:, :-1].std(axis=1, ddof=1); wide["min"] = wide.iloc[:, :-2].min(axis=1); wide["max"] = wide.iloc[:, :-3].max(axis=1)
    wide.to_csv(ROOT / "experiments/extension" / f"{kind}_seed_summary.csv")
    print("missing replicates:", missing); print(wide.round(4).to_string())
    return R, wide, missing


if __name__ == "__main__":
    main(sys.argv[1])
