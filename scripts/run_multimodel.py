"""Frozen PR #1 recipe for the three additional architectures.

Run one model at a time. A fold is resumable only after its prediction CSV and
run JSON both pass integrity checks. PR #1's A/B file labels are retained:
A=patient-disjoint, B=image-level seed 11.
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.preprocess import load_all
from run_sanity import CFG, EXPO, SPLITS, augment, sha, to_input
from safe_train import host_snapshot, unsafe

OUT = ROOT / "results" / "multimodel"
RUNS = OUT / "runs"
NAMES = ("mobilenet_v2", "efficientnet_b0", "densenet121")
WEIGHTS = {
    "mobilenet_v2": torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1,
    "efficientnet_b0": torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "densenet121": torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
}
CLASS_NAMES = ("meningioma", "glioma", "pituitary")


def make_model(name):
    if name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(weights=WEIGHTS[name])
        model.classifier[1] = torch.nn.Linear(model.last_channel, 3)
    elif name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=WEIGHTS[name])
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
    elif name == "densenet121":
        model = torchvision.models.densenet121(weights=WEIGHTS[name])
        model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
    else:
        raise ValueError(name)
    return model


def validated_run(pred_path, run_path, split, arm, fold, model_name):
    if not pred_path.exists() or not run_path.exists():
        return False
    try:
        d = pd.read_csv(pred_path, dtype={"patient_id": str})
        j = json.loads(run_path.read_text())
        expected = split.loc[split.fold == fold, "image_id"]
        assert len(d) == len(expected) and d.image_id.is_unique
        assert set(d.image_id) == set(expected)
        assert (d.arm == arm).all() and (d.test_fold == fold).all()
        assert (d.model == model_name).all()
        assert j["model"] == model_name and j["arm"] == arm and j["fold"] == fold
        assert j["device"] == "mps" and j["split_sha256"] == sha(SPLITS[arm])
        assert len(j["epochs"]) == CFG["epochs"]
        p = d[[f"prob_{k}" for k in range(3)]].to_numpy()
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        assert np.allclose(p.sum(1), 1, atol=1e-6)
        assert np.array_equal(p.argmax(1), d.pred_class.to_numpy())
        assert np.allclose(d.confidence.to_numpy(), p.max(1), rtol=0, atol=1e-12)
        assert np.array_equal(d.pred_class_name.to_numpy(), np.array(CLASS_NAMES)[p.argmax(1)])
        return True
    except (AssertionError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_model(name, max_new_runs=None):
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable: refusing CPU-only training")
    RUNS.mkdir(parents=True, exist_ok=True)
    split_validation = json.loads((ROOT / "data/splits/split_validation.json").read_text())
    assert split_validation["all_checks_pass"]
    for arm, filename in (("A", "patient_disjoint_original.csv"), ("B", "imagelevel_seed11.csv")):
        assert sha(SPLITS[arm]) == split_validation["sha256"][filename]
    manifest = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv", dtype={"patient_id": str})
    assert len(manifest) == 3064 and manifest.image_id.is_unique
    assert manifest.patient_id.nunique() == 233 and set(manifest.label) == {1, 2, 3}
    ids, X = load_all(ROOT / "reports/manifests/image_manifest.csv")
    assert len(ids) == 3064 and X.shape == (3064, 224, 224)
    pos = {int(i): k for k, i in enumerate(ids)}
    # Keep the existing read-only preprocessing cache memory-mapped. Indexing
    # materializes only the current batch; no full-dataset RAM copy is needed.
    Xt = torch.from_numpy(X)
    input_commit = os.environ.get("EXPERIMENT_INPUT_COMMIT", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    env = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
           "numpy": np.__version__, "pandas": pd.__version__, "os": platform.platform(),
           "device": "mps", "mps_available": True, "input_commit": input_commit,
           "split_sha256": split_validation["sha256"], "config": CFG,
           "model": name, "weights": str(WEIGHTS[name]),
           "determinism": "MPS is not bitwise deterministic; train seed 0 is reset for each fold"}
    (OUT / f"environment_{name}.json").write_text(json.dumps(env, indent=2))
    new_runs = 0
    completed_this_process = 0
    for fold in range(1, 6):
        for arm in ("A", "B"):
            pred_path = RUNS / f"{name}_{arm}_fold{fold}_predictions.csv"
            run_path = RUNS / f"{name}_{arm}_fold{fold}_run.json"
            split = pd.read_csv(SPLITS[arm], dtype={"patient_id": str})
            if validated_run(pred_path, run_path, split, arm, fold, name):
                print(f"[{name} {arm} f{fold}] verified existing run", flush=True)
                continue
            if pred_path.exists() or run_path.exists():
                raise RuntimeError(f"Incomplete or invalid run output: {pred_path}; inspect before resuming")
            if completed_this_process:
                # Cooling period and hard host check before every next fold.
                deadline = time.monotonic() + 20 * 60
                while True:
                    time.sleep(30)
                    snapshot = host_snapshot()
                    reason = unsafe(snapshot)
                    if snapshot["available_gib"] < 5:
                        reason = reason or "available RAM below 5 GiB pre-run"
                    if snapshot["pressure_free_pct"] is not None and snapshot["pressure_free_pct"] < 20:
                        reason = reason or "memory pressure free below 20% pre-run"
                    print("BETWEEN_RUN_RESOURCE_CHECK", snapshot, "reason=", reason, flush=True)
                    if reason is None:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"Resource pressure did not recover in 20 minutes: {reason}")
            tr, te = split[split.fold != fold], split[split.fold == fold]
            tr_idx = np.array([pos[int(i)] for i in tr.image_id])
            te_idx = np.array([pos[int(i)] for i in te.image_id])
            ytr = torch.tensor(tr.label.to_numpy() - 1)
            n = len(tr_idx)
            assert n % CFG["batch_size"] != 1
            torch.manual_seed(CFG["train_seed"])
            np.random.seed(CFG["train_seed"])
            model = make_model(name).to("mps")
            opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
            steps = CFG["epochs"] * int(np.ceil(n / CFG["batch_size"]))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
            g_data = torch.Generator().manual_seed(CFG["train_seed"])
            g_aug = torch.Generator().manual_seed(CFG["train_seed"] + 1)
            t0 = time.time()
            log = []
            for ep in range(CFG["epochs"]):
                model.train()
                perm = torch.randperm(n, generator=g_data)
                tl = tc = 0.0
                for b in range(0, n, CFG["batch_size"]):
                    bi = perm[b:b + CFG["batch_size"]]
                    xb = augment(Xt[tr_idx[bi.numpy()]], g_aug)
                    yb = ytr[bi].to("mps")
                    logits = model(to_input(xb).to("mps"))
                    loss = F.cross_entropy(logits, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                    tl += loss.item() * len(bi)
                    tc += (logits.argmax(1) == yb).sum().item()
                    if b == 0:
                        print(f"[{name} {arm} f{fold}] first_batch_mps_active_gib="
                              f"{torch.mps.current_allocated_memory()/2**30:.2f} "
                              f"driver_gib={torch.mps.driver_allocated_memory()/2**30:.2f}", flush=True)
                    del bi, xb, yb, logits, loss
                log.append({"epoch": ep + 1, "train_loss": tl / n, "train_acc": tc / n,
                            "lr_end": sched.get_last_lr()[0],
                            "mps_active_gib": torch.mps.current_allocated_memory() / 2**30,
                            "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30})
                print(f"[{name} {arm} f{fold}] epoch {ep+1}/{CFG['epochs']} "
                      f"loss={tl/n:.4f} acc={tc/n:.4f} elapsed={time.time()-t0:.0f}s "
                      f"mps_driver_gib={log[-1]['mps_driver_gib']:.2f}", flush=True)
                # Release only idle MPS allocator blocks; model/optimizer state and
                # the frozen training recipe are unchanged.
                torch.mps.empty_cache()
            model.eval()
            outputs = []
            with torch.no_grad():
                for b in range(0, len(te_idx), CFG["eval_batch_size"]):
                    outputs.append(model(to_input(Xt[te_idx[b:b + CFG["eval_batch_size"]]]).to("mps")).float().cpu())
            lg = torch.cat(outputs).numpy().astype(np.float64)
            assert np.isfinite(lg).all()
            pr = np.exp(lg - lg.max(1, keepdims=True))
            pr /= pr.sum(1, keepdims=True)
            d = te[["image_id", "patient_id", "label"]].copy().reset_index(drop=True)
            d.insert(0, "model", name)
            d.insert(1, "arm", arm)
            d["protocol"] = "patient_disjoint" if arm == "A" else "image_level_seed11"
            d["test_fold"] = fold
            d["true_class"] = d.label - 1
            d["true_class_name"] = np.array(CLASS_NAMES)[d.true_class.to_numpy()]
            for k in range(3):
                d[f"prob_{k}"] = pr[:, k]
            d["pred_class"] = pr.argmax(1)
            d["pred_class_name"] = np.array(CLASS_NAMES)[d.pred_class.to_numpy()]
            d["confidence"] = pr.max(1)
            d["experiment_input_commit"] = input_commit
            d = d.drop(columns=["label"])
            tmp = pred_path.with_suffix(".csv.tmp")
            d.to_csv(tmp, index=False)
            os.replace(tmp, pred_path)
            runtime_s = time.time() - t0
            run = {"model": name, "weights": str(WEIGHTS[name]), "arm": arm, "protocol": d.protocol.iloc[0],
                   "fold": fold, "n_train": n, "n_test": len(d), "runtime_s": runtime_s,
                   "device": "mps", "train_seed": CFG["train_seed"], "split_sha256": sha(SPLITS[arm]),
                   "experiment_input_commit": input_commit, "epochs": log,
                   "test_acc_final": float((d.pred_class == d.true_class).mean())}
            tmp_run = run_path.with_suffix(".json.tmp")
            tmp_run.write_text(json.dumps(run, indent=2))
            os.replace(tmp_run, run_path)
            assert validated_run(pred_path, run_path, split, arm, fold, name)
            print(f"[{name} {arm} f{fold}] COMPLETE runtime={runtime_s:.0f}s test_acc={run['test_acc_final']:.4f}", flush=True)
            new_runs += 1
            completed_this_process += 1
            del model, opt, sched, outputs, lg, pr, d, tr_idx, te_idx, ytr, perm
            gc.collect()
            torch.mps.empty_cache()
            if max_new_runs is not None and new_runs >= max_new_runs:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=NAMES)
    ap.add_argument("--max-new-runs", type=int)
    a = ap.parse_args()
    run_model(a.model, a.max_new_runs)


if __name__ == "__main__":
    main()
