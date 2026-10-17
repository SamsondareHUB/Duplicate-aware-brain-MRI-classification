"""Train + evaluate ONE run (frozen PR #1/#2 recipe). Usage: python ext_runner.py RUN_ID. Writes prediction CSV, run JSON
(atomic, CSV first) and a git-ignored final-epoch checkpoint. Refuses to overwrite a verified run."""
import gc, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, torchvision

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "experiments/extension"))
from leakage.preprocess import load_all
from run_sanity import CFG, augment, to_input
from run_multimodel import make_model, WEIGHTS, CLASS_NAMES
from ext_specs import RUNS, get_spec, sha

CKPT = ROOT / "data/ckpt"
RESNET_WEIGHTS = torchvision.models.ResNet50_Weights.IMAGENET1K_V1


def build_model(name):
    if name == "resnet50":  # identical to run_sanity.py
        m = torchvision.models.resnet50(weights=RESNET_WEIGHTS); m.fc = torch.nn.Linear(m.fc.in_features, 3); return m
    return make_model(name)


def paths(run_id): return RUNS / f"{run_id}_predictions.csv", RUNS / f"{run_id}_run.json"


def verify(run_id, spec=None):
    pp, rp = paths(run_id)
    if not pp.exists() or not rp.exists(): return False
    try:
        spec = spec or get_spec(run_id)
        d = pd.read_csv(pp, dtype={"patient_id": str}); j = json.loads(rp.read_text())
        assert set(d.image_id) == set(spec["test_ids"]) and len(d) == len(spec["test_ids"]) and d.image_id.is_unique
        assert not (set(d.image_id) & set(spec["train_ids"]))
        assert j["run_id"] == run_id and j["model"] == spec["model"] and j["train_seed"] == spec["seed"]
        assert j["device"] == "mps" and j["split_sha256"] == spec["split_sha256"] and len(j["epochs"]) == CFG["epochs"]
        assert j["predictions_sha256"] == sha(pp) and j["n_train"] == len(spec["train_ids"])
        p = d[[f"prob_{k}" for k in range(3)]].to_numpy()
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-6) and np.array_equal(p.argmax(1), d.pred_class.to_numpy())
        return True
    except (AssertionError, KeyError, ValueError, json.JSONDecodeError, OSError):
        return False


def main(run_id):
    spec = get_spec(run_id)
    if verify(run_id, spec): print("already verified", run_id); return 0
    pp, rp = paths(run_id)
    if pp.exists() or rp.exists(): raise SystemExit(f"invalid partial outputs for {run_id}; inspect")
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available(), "MPS unavailable"
    RUNS.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    assert not (set(spec["train_ids"]) & set(spec["test_ids"])), "test image in training"
    man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str}).set_index("image_id")
    ids, X = load_all(ROOT / "reports/manifests/image_manifest.csv")
    pos = {int(i): k for k, i in enumerate(ids)}
    Xt = torch.from_numpy(X)
    tr_idx = np.array([pos[i] for i in spec["train_ids"]]); te_idx = np.array([pos[i] for i in spec["test_ids"]])
    ytr = torch.tensor(man.loc[spec["train_ids"], "label"].to_numpy() - 1)
    n = len(tr_idx); seed = spec["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(spec["model"]).to("mps")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps = CFG["epochs"] * int(np.ceil(n / CFG["batch_size"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    g_data = torch.Generator().manual_seed(seed); g_aug = torch.Generator().manual_seed(seed + 1)
    t0 = time.time(); log = []
    for ep in range(CFG["epochs"]):
        model.train(); perm = torch.randperm(n, generator=g_data); tl = tc = 0.0
        for b in range(0, n, CFG["batch_size"]):
            bi = perm[b:b + CFG["batch_size"]]
            xb = augment(Xt[tr_idx[bi.numpy()]], g_aug); yb = ytr[bi].to("mps")
            logits = model(to_input(xb).to("mps")); loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(bi); tc += (logits.argmax(1) == yb).sum().item()
            del bi, xb, yb, logits, loss
        log.append({"epoch": ep + 1, "train_loss": tl / n, "train_acc": tc / n, "lr_end": sched.get_last_lr()[0]})
        print(f"[{run_id}] epoch {ep+1}/{CFG['epochs']} loss={tl/n:.4f} acc={tc/n:.4f} t={time.time()-t0:.0f}s", flush=True)
        torch.mps.empty_cache()
    model.eval(); outs = []
    with torch.no_grad():
        for b in range(0, len(te_idx), CFG["eval_batch_size"]):
            outs.append(model(to_input(Xt[te_idx[b:b + CFG["eval_batch_size"]]]).to("mps")).float().cpu())
    lg = torch.cat(outs).numpy().astype(np.float64); assert np.isfinite(lg).all()
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    d = man.loc[spec["test_ids"], ["patient_id", "label"]].reset_index()
    d.insert(0, "run_id", run_id)
    d["true_class"] = d.label - 1
    for k in range(3): d[f"prob_{k}"] = pr[:, k]
    for k in range(3): d[f"logit_{k}"] = lg[:, k]
    d["pred_class"] = pr.argmax(1); d["confidence"] = pr.max(1); d = d.drop(columns="label")
    commit = os.environ.get("EXPERIMENT_INPUT_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    torch.save(model.state_dict(), CKPT / f"{run_id}.pt")
    tmp = pp.with_suffix(".tmp"); d.to_csv(tmp, index=False); os.replace(tmp, pp)
    run = {"run_id": run_id, "group": spec["group"], "model": spec["model"], "arm": spec["arm"], "fold": spec["fold"],
           "level": spec.get("level"), "split_seed": spec.get("split_seed"), "train_seed": seed,
           "split_file": str(Path(spec["split_file"]).relative_to(ROOT)), "split_sha256": spec["split_sha256"],
           "n_train": n, "n_test": len(d), "runtime_s": time.time() - t0, "device": "mps", "epochs": log,
           "test_acc_final": float((d.pred_class == d.true_class).mean()), "predictions_sha256": sha(pp),
           "weights": "IMAGENET1K_V1", "config": CFG, "experiment_input_commit": commit, "checkpoint": f"data/ckpt/{run_id}.pt",
           "env": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "numpy": np.__version__, "pandas": pd.__version__, "os": platform.platform()}}
    tmp = rp.with_suffix(".tmp"); tmp.write_text(json.dumps(run, indent=1)); os.replace(tmp, rp)
    assert verify(run_id, spec)
    print(f"[{run_id}] COMPLETE runtime={run['runtime_s']:.0f}s acc={run['test_acc_final']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
