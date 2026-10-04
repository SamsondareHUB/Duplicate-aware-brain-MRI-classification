"""Frozen 10-run sanity experiment: ResNet50 (ImageNet V1), arms A/B x 5 folds, training seed 0.
Resumable: completed runs (prediction file present) are skipped. Do not tune anything per arm.
"""
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchvision
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.preprocess import load_all

OUT = ROOT / "results/sanity"; RUNS = OUT / "runs"; RUNS.mkdir(parents=True, exist_ok=True)
CFG = dict(model="torchvision.resnet50", weights="IMAGENET1K_V1", input=224, batch_size=32, epochs=10, optimizer="AdamW",
           lr=1e-4, weight_decay=1e-4, schedule="cosine (per-iteration, T_max=total steps, eta_min=0)",
           loss="cross_entropy (no class weights)", augmentation="hflip p=0.5; rotation uniform(-10,10) deg bilinear fill 0",
           train_seed=0, early_stopping=False, model_selection="none (final epoch)", eval_batch_size=64,
           normalisation="ImageNet mean/std on 3-channel replicate")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SPLITS = {"A": ROOT / "data/splits/patient_disjoint_original.csv", "B": ROOT / "data/splits/imagelevel_seed11.csv"}
EXPO = {"A": ROOT / "data/splits/exposure_A.csv", "B": ROOT / "data/splits/exposure_B_seed11.csv"}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a): return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def to_input(x):  # x [B,224,224] in [0,1] -> [B,3,224,224] normalised
    return (x[:, None].repeat(1, 3, 1, 1) - MEAN) / STD


def augment(x, g):
    flip = torch.rand(len(x), generator=g) < 0.5
    ang = (torch.rand(len(x), generator=g) * 20 - 10).tolist()
    out = torch.empty_like(x)
    for i in range(len(x)):
        im = x[i:i + 1]
        if flip[i]: im = TF.horizontal_flip(im)
        out[i] = TF.rotate(im, ang[i], interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)[0]
    return out


def main():
    dev = "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else "cpu"
    split_sha_expected = json.load(open(ROOT / "data/splits/split_validation.json"))["sha256"]
    assert sha(SPLITS["A"]) == split_sha_expected["patient_disjoint_original.csv"], "split A changed"
    assert sha(SPLITS["B"]) == split_sha_expected["imagelevel_seed11.csv"], "split B changed"
    input_commit = os.environ.get("EXPERIMENT_INPUT_COMMIT", git("rev-parse", "HEAD"))
    env = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
           "numpy": np.__version__, "pandas": pd.__version__, "os": platform.platform(), "machine": platform.machine(),
           "device": dev, "mps_available": bool(torch.backends.mps.is_available()),
           "mps_built": bool(torch.backends.mps.is_built()), "cuda_available": bool(torch.cuda.is_available()),
           "determinism": {"torch.manual_seed": 0, "use_deterministic_algorithms": False,
                           "note": "MPS (if used) does not guarantee bitwise determinism; same backend for all 10 runs"},
           "experiment_input_commit": input_commit, "split_sha256": split_sha_expected, "config": CFG}
    (OUT / "environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=1), flush=True)

    man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv").sort_values("image_id").reset_index(drop=True)
    ids, X = load_all(ROOT / "reports/manifests/image_manifest.csv")
    pos = {int(i): k for k, i in enumerate(ids)}
    Xt = torch.from_numpy(np.asarray(X))
    done = []
    for fold in range(1, 6):
        for arm in ("A", "B"):
            pred_path = RUNS / f"{arm}_fold{fold}_predictions.csv"
            if pred_path.exists():
                continue
            S = pd.read_csv(SPLITS[arm], dtype={"patient_id": str}); E = pd.read_csv(EXPO[arm], dtype={"patient_id": str})
            tr = S[S.fold != fold]; te = S[S.fold == fold]
            tr_idx = np.array([pos[int(i)] for i in tr.image_id]); te_idx = np.array([pos[int(i)] for i in te.image_id])
            ytr = torch.tensor(tr.label.values - 1); n = len(tr_idx)
            assert n % CFG["batch_size"] != 1
            torch.manual_seed(CFG["train_seed"]); np.random.seed(CFG["train_seed"])
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Linear(2048, 3); model = model.to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
            steps = CFG["epochs"] * int(np.ceil(n / CFG["batch_size"]))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
            g_data = torch.Generator().manual_seed(CFG["train_seed"]); g_aug = torch.Generator().manual_seed(CFG["train_seed"] + 1)
            t0 = time.time(); log = []; warns = []
            for ep in range(CFG["epochs"]):
                model.train(); perm = torch.randperm(n, generator=g_data); tl = tc = 0.0
                for b in range(0, n, CFG["batch_size"]):
                    bi = perm[b:b + CFG["batch_size"]]
                    xb = augment(Xt[tr_idx[bi.numpy()]], g_aug); yb = ytr[bi].to(dev)
                    out = model(to_input(xb).to(dev)); loss = F.cross_entropy(out, yb)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                    tl += loss.item() * len(bi); tc += (out.argmax(1) == yb).sum().item()
                log.append({"epoch": ep + 1, "train_loss": tl / n, "train_acc": tc / n, "lr_end": sched.get_last_lr()[0]})
                print(f"[{arm} f{fold}] ep{ep + 1} loss {tl / n:.4f} acc {tc / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
            model.eval(); logits = []
            with torch.no_grad():
                for b in range(0, len(te_idx), CFG["eval_batch_size"]):
                    logits.append(model(to_input(Xt[te_idx[b:b + CFG["eval_batch_size"]]]).to(dev)).float().cpu())
            lg = torch.cat(logits).numpy().astype(np.float64)
            if not np.isfinite(lg).all(): warns.append("non-finite logits")
            pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            df = te[["image_id", "patient_id", "label"]].copy().reset_index(drop=True)
            df.insert(0, "arm", arm); df["test_fold"] = fold; df["true_class"] = df.label - 1
            for k in range(3): df[f"logit_{k}"] = lg[:, k]
            for k in range(3): df[f"prob_{k}"] = pr[:, k]
            df["pred_class"] = pr.argmax(1)
            df = df.merge(E[["image_id", "n_other_total", "n_other_train", "frac_other_train", "single_slice_patient",
                             "has_same_patient_train_slice"]], on="image_id", how="left")
            df["experiment_input_commit"] = input_commit
            df.drop(columns=["label"]).to_csv(pred_path, index=False)
            rt = time.time() - t0
            (RUNS / f"{arm}_fold{fold}_run.json").write_text(json.dumps({
                "arm": arm, "fold": fold, "n_train": int(n), "n_test": int(len(te_idx)), "runtime_s": rt, "device": dev,
                "model": CFG["model"], "weights": CFG["weights"], "train_seed": CFG["train_seed"],
                "split_sha256": sha(SPLITS[arm]), "experiment_input_commit": input_commit, "epochs": log,
                "test_acc_final": float((df.pred_class == df.true_class).mean()), "warnings": warns}, indent=1))
            print(f"[{arm} f{fold}] DONE test acc {(df.pred_class == df.true_class).mean():.4f} runtime {rt:.0f}s", flush=True)
            done.append((arm, fold))
    print("ALL RUNS DONE", done, flush=True)


if __name__ == "__main__":
    main()
