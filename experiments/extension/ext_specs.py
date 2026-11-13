"""Run registry for the extension experiments. A run is fully determined by its run_id; specs are derived, never stored by hand.

Groups (training recipe identical to PR #1/#2 for all):
  expo   expo_{model}_{condition}_s{seed}     fixed-test exposure design (folds 2,4; seeds 0,1)
  split  split_{model}_{A|B}_seed{S}_fold{k}  new split-seed replicates (S in 101,202,303; train seed 0)
  tseed  tseed_{model}_{A|B}_ts{T}_fold{k}    historical splits, new training seeds T in 1,2
Priority order = the order of build_queue().
"""
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "experiments/extension"
RUNS = ROOT / "results/extension/runs"
MODELS = ("resnet50", "densenet121")
SPLIT_SEEDS = (101, 202, 303)
TRAIN_SEEDS_NEW = (1, 2)
EXPO_SEEDS = (0, 1)
HIST = {"A": ROOT / "data/splits/patient_disjoint_original.csv", "B": ROOT / "data/splits/imagelevel_seed11.csv"}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_queue():
    cond = pd.read_csv(EXT / "exposure/conditions.csv")
    q = []
    def expo(seed):
        for m in MODELS:
            for c in cond.condition:
                q.append(f"expo_{m}_{c}_s{seed}")
    def split(S):
        for m in MODELS:
            for arm in "AB":
                for k in range(1, 6): q.append(f"split_{m}_{arm}_seed{S}_fold{k}")
    def tseed(T):
        for m in MODELS:
            for arm in "AB":
                for k in range(1, 6): q.append(f"tseed_{m}_{arm}_ts{T}_fold{k}")
    expo(0); split(101); expo(1); split(202); split(303); tseed(1); tseed(2)
    assert len(q) == len(set(q))
    return q


def get_spec(run_id):
    p = run_id.split("_")
    g = p[0]
    if g == "expo":
        model = p[1]; seed = int(p[-1][1:]); cond = "_".join(p[2:-1])
        row = pd.read_csv(EXT / "exposure/conditions.csv").set_index("condition").loc[cond]
        f = int(row.fold)
        train = pd.read_csv(EXT / f"exposure/train_{cond}.csv").image_id.tolist()
        test = pd.read_csv(EXT / f"exposure/fixed_test_fold{f}.csv").image_id.tolist()
        return dict(run_id=run_id, group=g, model=model, seed=seed, train_ids=train, test_ids=test, fold=f,
                    arm=row.kind, level=int(row.level), split_file=str(EXT / f"exposure/train_{cond}.csv"),
                    split_sha256=sha(EXT / f"exposure/train_{cond}.csv"))
    if g == "split":
        model, arm, S, k = p[1], p[2], int(p[3][4:]), int(p[4][4:])
        f = EXT / f"splits/split_{arm}_seed{S}.csv"; seed = 0
    elif g == "tseed":
        model, arm, seed, k = p[1], p[2], int(p[3][2:]), int(p[4][4:])
        f = HIST[arm]; S = None
    else:
        raise ValueError(run_id)
    sp = pd.read_csv(f, dtype={"patient_id": str})
    return dict(run_id=run_id, group=g, model=model, seed=seed, fold=k, arm=arm, split_seed=S,
                train_ids=sp.image_id[sp.fold != k].tolist(), test_ids=sp.image_id[sp.fold == k].tolist(),
                split_file=str(f), split_sha256=sha(f))
