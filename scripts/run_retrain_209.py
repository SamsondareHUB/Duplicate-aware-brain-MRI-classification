"""209-group retraining runner. PROPOSED / NOT YET APPROVED. Default mode is --dry-run (no torch import, no training).

Training mirrors scripts/run_multimodel.py line for line (same CFG from run_sanity.py, train_seed 0, per-fold re-seeding, final epoch,
no early stopping) but reads data/splits/proposed_209/PROPOSED_*.csv and writes ONLY to results/retrain_209/.
Real training needs BOTH --train and --approved-by "<name, date>". The training branch has not been executed (needs one-fold smoke test
after approval). Arms: A209 = stem-disjoint (minmove), B209 = matched image-level seed 11 on the A209 fold x class matrix.
  python scripts/run_retrain_209.py --dry-run                      # validate splits, print run matrix, write run_matrix_209.csv
  python scripts/run_retrain_209.py --train --approved-by "..." --model densenet121 --arm A209 [--max-new-runs N]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPL = ROOT / "data/splits/proposed_209"
OUT = ROOT / "results/retrain_209"
FILES = {"A209": "PROPOSED_A209_minmove.csv", "B209": "PROPOSED_B209_imagelevel_seed11.csv"}
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
EST_S = {"resnet50": 377, "mobilenet_v2": 169, "efficientnet_b0": 241, "densenet121": 390}  # mean s/fold-run from PR#1/#2 run JSONs (Apple MPS)
GIB = {"resnet50": 6, "mobilenet_v2": 4, "efficientnet_b0": 5, "densenet121": 6}  # rough planning figures, NOT measured


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate():
    v = json.loads((SPL / "PROPOSED_split_validation_209.json").read_text())
    for arm, fn in FILES.items():
        assert sha(SPL / fn) == v["sha256"][fn], f"{fn} changed since validation"
    a = pd.read_csv(SPL / FILES["A209"], dtype={"patient_id": str}); b = pd.read_csv(SPL / FILES["B209"], dtype={"patient_id": str})
    assert len(a) == len(b) == 3064 and a.image_id.is_unique and b.image_id.is_unique and a.stem.nunique() == 209
    assert (a.groupby("stem").fold.nunique() == 1).all(), "A209 stem leaks across folds"
    assert (a.groupby("patient_id").fold.nunique() == 1).all()
    assert (pd.crosstab(a.fold, a.label).values == pd.crosstab(b.fold, b.label).values).all()
    for f, fn in (("data/splits/patient_disjoint_original.csv", "patient_disjoint_original.csv"), ("data/splits/imagelevel_seed11.csv", "imagelevel_seed11.csv")):
        assert sha(ROOT / f) == json.loads((ROOT / "data/splits/split_validation.json").read_text())["sha256"][fn], "historical split modified"
    return a, b


def matrix():
    rows = []
    for m in MODELS:
        for arm in ("A209", "B209"):
            for fold in range(1, 6):
                rows.append(dict(run_id=f"{m}_{arm}_fold{fold}", model=m, arm=arm, fold=fold, train_seed=0, split_file=f"data/splits/proposed_209/{FILES[arm]}",
                                 pred_out=f"results/retrain_209/runs/{m}_{arm}_fold{fold}_predictions.csv",
                                 run_json=f"results/retrain_209/runs/{m}_{arm}_fold{fold}_run.json", est_minutes=round(EST_S[m] / 60, 1),
                                 est_gpu_mem_gib_planning=GIB[m], priority="primary" if arm == "A209" else "matched_control",
                                 status="PROPOSED_NOT_RUN"))
    return pd.DataFrame(rows)


def train(model, arm, max_new):
    import numpy as np, torch, torch.nn.functional as F, torchvision, os, subprocess
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
    from leakage.preprocess import load_all
    from run_sanity import CFG, augment, to_input
    from run_multimodel import make_model, WEIGHTS, CLASS_NAMES
    from safe_train import host_snapshot, unsafe
    assert torch.backends.mps.is_available(), "MPS unavailable: refusing CPU-only training"
    validate(); (OUT / "runs").mkdir(parents=True, exist_ok=True)
    def mk(name):
        if name == "resnet50":
            mo = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1); mo.fc = torch.nn.Linear(mo.fc.in_features, 3); return mo
        return make_model(name)
    split = pd.read_csv(SPL / FILES[arm], dtype={"patient_id": str})
    ids, X = load_all(ROOT / "reports/manifests/image_manifest.csv"); pos = {int(i): k for k, i in enumerate(ids)}; Xt = torch.from_numpy(X)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(); done = 0
    for fold in range(1, 6):
        pp, rp = OUT / f"runs/{model}_{arm}_fold{fold}_predictions.csv", OUT / f"runs/{model}_{arm}_fold{fold}_run.json"
        if pp.exists() or rp.exists(): print("exists, skipping", pp.name); continue
        if done:
            time.sleep(30)
            while (r := unsafe(host_snapshot())): print("waiting:", r); time.sleep(60)
        tr, te = split[split.fold != fold], split[split.fold == fold]
        tri = np.array([pos[int(i)] for i in tr.image_id]); tei = np.array([pos[int(i)] for i in te.image_id])
        ytr = torch.tensor(tr.label.to_numpy() - 1); n = len(tri); assert n % CFG["batch_size"] != 1
        torch.manual_seed(CFG["train_seed"]); np.random.seed(CFG["train_seed"])
        net = mk(model).to("mps"); opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"] * int(np.ceil(n / CFG["batch_size"])), eta_min=0.0)
        gd = torch.Generator().manual_seed(CFG["train_seed"]); ga = torch.Generator().manual_seed(CFG["train_seed"] + 1); t0 = time.time(); log = []
        for ep in range(CFG["epochs"]):
            net.train(); perm = torch.randperm(n, generator=gd); tl = tc = 0.0
            for b in range(0, n, CFG["batch_size"]):
                bi = perm[b:b + CFG["batch_size"]]; xb = augment(Xt[tri[bi.numpy()]], ga); yb = ytr[bi].to("mps")
                lg = net(to_input(xb).to("mps")); loss = F.cross_entropy(lg, yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                tl += loss.item() * len(bi); tc += (lg.argmax(1) == yb).sum().item()
            log.append({"epoch": ep + 1, "train_loss": tl / n, "train_acc": tc / n}); torch.mps.empty_cache()
            print(f"[{model} {arm} f{fold}] ep {ep+1}/{CFG['epochs']} loss={tl/n:.4f}", flush=True)
        net.eval(); outs = []
        with torch.no_grad():
            for b in range(0, len(tei), CFG["eval_batch_size"]): outs.append(net(to_input(Xt[tei[b:b + CFG["eval_batch_size"]]]).to("mps")).float().cpu())
        lg = torch.cat(outs).numpy().astype(np.float64); pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        d = te[["image_id", "patient_id", "stem", "label"]].reset_index(drop=True); d.insert(0, "model", model); d.insert(1, "arm", arm)
        d["test_fold"] = fold; d["true_class"] = d.label - 1; d["true_class_name"] = np.array(CLASS_NAMES)[d.true_class.to_numpy()]
        for k in range(3): d[f"prob_{k}"] = pr[:, k]
        d["pred_class"] = pr.argmax(1); d["pred_class_name"] = np.array(CLASS_NAMES)[d.pred_class.to_numpy()]; d["confidence"] = pr.max(1)
        d["experiment_input_commit"] = commit; d = d.drop(columns=["label"]); d.to_csv(str(pp) + ".tmp", index=False); os.replace(str(pp) + ".tmp", pp)
        rp.write_text(json.dumps({"model": model, "arm": arm, "fold": fold, "n_train": n, "n_test": len(d), "runtime_s": time.time() - t0, "device": "mps",
                                  "train_seed": CFG["train_seed"], "split_sha256": sha(SPL / FILES[arm]), "epochs": log, "cfg": CFG,
                                  "experiment_input_commit": commit, "status": "COMPLETED_209_RETRAIN"}, indent=1))
        done += 1; del net, opt, sched
        if max_new and done >= max_new: return


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--approved-by"); ap.add_argument("--model", choices=MODELS); ap.add_argument("--arm", choices=tuple(FILES)); ap.add_argument("--max-new-runs", type=int)
    a = ap.parse_args()
    if not a.train:
        validate(); m = matrix(); p = SPL / "run_matrix_209.csv"; m.to_csv(p, index=False)
        t = m.groupby("arm").est_minutes.sum() / 60
        print(f"splits OK. runs={len(m)} (A209 {int((m.arm=='A209').sum())}, B209 {int((m.arm=='B209').sum())}); est hours A209={t['A209']:.2f} B209={t['B209']:.2f}; wrote {p.name}")
        return
    if not (a.approved_by and a.model and a.arm): sys.exit("--train requires --approved-by, --model and --arm; refusing.")
    train(a.model, a.arm, a.max_new_runs)


if __name__ == "__main__":
    main()
