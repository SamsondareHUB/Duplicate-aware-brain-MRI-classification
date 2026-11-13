"""Controlled patient-sibling exposure design (EXPLORATORY INTERVENTION; the sibling arms deliberately create train/test
patient overlap and are never a leakage-free evaluation). Built from counts/IDs only; no model outcomes exist at this point.

Pre-specified (frozen before training):
  fixed-test folds (original patient-disjoint split A): EXPOSURE_FOLDS = (2, 4); levels 0/25/50/75/100 % of ELIGIBLE test patients.
  eligible test patient: >= 4 slices.  donor slices d_p = min(floor(n_p/2), 6), chosen by seeded permutation; donors are
  REMOVED from the fixed test set in every arm (a test image is never trained on).  Fixed test = all other fold-f slices.
  level selection: nested, class-stratified seeded order; level L selects round(L * N_class) patients per class.
  base training = images of folds != f, minus reservoir patients.
  reservoir (control sources): one training-fold patient of the same class per eligible test patient, with n >= d_p, same
  ID format (numeric vs MR) preferred, each used once; seeded.  Reservoir patients are removed entirely from base.
  arms per level L>0:  CTRL_L = base + the d_p slices of each selected patient's matched control patient;
                       SIB_L  = base + the d_p donor slices of each selected test patient.   (same n, same class counts)
  CLEAN (level 0) = base.
"""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/extension/exposure"
EXPOSURE_FOLDS = (2, 4)
LEVELS = (25, 50, 75, 100)
MIN_SLICES, DONOR_CAP = 4, 6
SEED_BASE = 7000


def idfmt(pid): return "MR" if str(pid).startswith("MR") else "num"


def design(fold):
    sp = pd.read_csv(ROOT / "data/splits/patient_disjoint_original.csv", dtype={"patient_id": str})
    rng = np.random.default_rng(SEED_BASE + fold)
    test = sp[sp.fold == fold]
    train = sp[sp.fold != fold]
    n_p = test.groupby("patient_id").image_id.size()
    cls = test.groupby("patient_id").label.first()
    elig = n_p[n_p >= MIN_SLICES].index.tolist()
    donors, rows = {}, []
    for p in sorted(elig):
        ids = np.sort(test.image_id[test.patient_id == p].to_numpy())
        d = min(len(ids) // 2, DONOR_CAP)
        donors[p] = sorted(rng.permutation(ids)[:d].tolist())
    donor_ids = {i for v in donors.values() for i in v}
    fixed_test = test[~test.image_id.isin(donor_ids)]
    # nested class-stratified order
    order = {}
    for c in (1, 2, 3):
        pc = sorted(p for p in elig if cls[p] == c)
        order[c] = [pc[i] for i in rng.permutation(len(pc))]
    selected = {L: [p for c in (1, 2, 3) for p in order[c][:int(round(L / 100 * len(order[c])))]] for L in LEVELS}
    # reservoir matching (all eligible patients, 100% level) -- class by class, large d_p first
    tp_n = train.groupby("patient_id").image_id.size(); tp_c = train.groupby("patient_id").label.first()
    used, match = set(), {}
    for c in (1, 2, 3):
        for p in sorted(order[c], key=lambda q: -len(donors[q])):
            d = len(donors[p])
            cands = [q for q in tp_n.index if tp_c[q] == c and tp_n[q] >= d and q not in used]
            same = [q for q in cands if idfmt(q) == idfmt(p)]
            pool = same or cands
            assert pool, f"fold {fold}: no control patient for {p} (class {c}, d={d})"
            q = sorted(pool)[int(rng.integers(len(pool)))]
            used.add(q); match[p] = (q, bool(same))
    ctrl_imgs = {}
    for p, (q, _) in match.items():
        ids = np.sort(train.image_id[train.patient_id == q].to_numpy())
        ctrl_imgs[p] = sorted(rng.permutation(ids)[:len(donors[p])].tolist())
    base = train[~train.patient_id.isin(used)]
    return dict(fold=fold, test=test, fixed_test=fixed_test, donors=donors, ctrl=ctrl_imgs, match=match,
                selected=selected, base=base, elig=elig, cls=cls, order=order)


def train_ids(D, level, kind):
    ids = D["base"].image_id.tolist()
    if level == 0: return sorted(ids)
    key = "donors" if kind == "SIB" else "ctrl"
    add = [i for p in D["selected"][level] for i in D[key][p]]
    return sorted(ids + add)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str}).set_index("image_id")
    summary = {"folds": list(EXPOSURE_FOLDS), "levels": [0, *LEVELS], "min_slices": MIN_SLICES, "donor_cap": DONOR_CAP, "per_fold": {}}
    cond_rows = []
    for f in EXPOSURE_FOLDS:
        D = design(f)
        ft = set(D["fixed_test"].image_id)
        info = {"fixed_test_images": len(ft), "fixed_test_patients": int(D["fixed_test"].patient_id.nunique()),
                "eligible_test_patients": len(D["elig"]), "donor_images_total": sum(len(v) for v in D["donors"].values()),
                "base_train_images": len(D["base"]), "reservoir_patients": len(D["match"]),
                "control_same_idformat": int(sum(s for _, s in D["match"].values())),
                "eligible_by_class": {int(c): len(D["order"][c]) for c in (1, 2, 3)}, "levels": {}}
        D["fixed_test"][["image_id", "patient_id", "label"]].assign(
            exposure_eligible=lambda d: d.patient_id.isin(D["elig"])).to_csv(OUT / f"fixed_test_fold{f}.csv", index=False)
        for L in (0, *LEVELS):
            for kind in (("CLEAN",) if L == 0 else ("CTRL", "SIB")):
                ids = train_ids(D, L, kind)
                assert not (set(ids) & ft), "fixed-test image in training"
                assert len(set(ids)) == len(ids)
                tr = manifest.loc[ids]
                te_p = set(D["fixed_test"].patient_id)
                n_overlap_patients = len(set(tr.patient_id) & te_p)
                name = f"fold{f}_{kind}_L{L}"
                pd.DataFrame({"image_id": ids}).to_csv(OUT / f"train_{name}.csv", index=False)
                cond_rows.append({"condition": name, "fold": f, "kind": kind, "level": L, "n_train": len(ids),
                                  "class_counts": json.dumps(tr.label.value_counts().sort_index().to_dict()),
                                  "train_test_patient_overlap": n_overlap_patients,
                                  "batch_remainder": len(ids) % 32})
                if kind == "CLEAN": assert n_overlap_patients == 0
        info["levels"] = {str(L): len(D["selected"][L]) for L in LEVELS}
        # per-patient design table
        pat = [{"patient_id": p, "class": int(D["cls"][p]), "n_donor": len(D["donors"][p]), "control_patient": D["match"][p][0],
                "control_same_idformat": D["match"][p][1], **{f"selected_L{L}": p in D["selected"][L] for L in LEVELS}} for p in D["elig"]]
        pd.DataFrame(pat).to_csv(OUT / f"patients_fold{f}.csv", index=False)
        summary["per_fold"][str(f)] = info
    cond = pd.DataFrame(cond_rows)
    # CTRL and SIB must be matched in size and class composition
    for (f, L), g in cond[cond.kind != "CLEAN"].groupby(["fold", "level"]):
        a, b = g[g.kind == "CTRL"].iloc[0], g[g.kind == "SIB"].iloc[0]
        assert a.n_train == b.n_train and a.class_counts == b.class_counts, (f, L)
    cond.to_csv(OUT / "conditions.csv", index=False)
    sha = hashlib.sha256("".join(sorted(p.name + hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.csv"))).encode()).hexdigest()
    summary["design_sha256"] = sha
    (OUT / "design_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)); print(cond.to_string())


if __name__ == "__main__":
    main()
