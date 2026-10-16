"""209-group retraining runner. PROPOSED / NOT YET APPROVED / NEVER EXECUTED ON A GPU.

Training body mirrors scripts/run_multimodel.py (PR #2) statement for statement (same variable names so tests/test_runner_parity.py can
verify it textually). CFG, augment, to_input, load_all, make_model and WEIGHTS are IMPORTED from the PR #1/#2 modules, not re-implemented.
Reads data/splits/proposed_209/PROPOSED_*.csv, writes ONLY to results/retrain_209/.

Modes
  default / --dry-run : validate splits, write the two conditional run matrices, import no torch, train nothing.
  --train             : needs --approved-by, --scenario {distinct,repeated,no_response}, --model, --arm. 'distinct' permits only A209 (20 runs);
                        'repeated' and 'no_response' permit A209 and B209 (40 runs).
  --quarantine-partial: moves a partial/corrupt fold's files to results/retrain_209/quarantine/ (never deletes) so it can be re-run.
Use scripts/serial_209.py for monitored, one-fold-per-process execution (the PR #2 safety protocol).
"""
import argparse, contextlib, fcntl, hashlib, json, os, signal, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
SPL = ROOT / "data/splits/proposed_209"
OUT = ROOT / "results/retrain_209"
FILES = {"A209": "PROPOSED_A209_minmove.csv", "B209": "PROPOSED_B209_imagelevel_seed11.csv"}
PROTOCOL = {"A209": "patient_disjoint_209", "B209": "image_level_209_seed11"}
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
CLASS_NAMES = ("meningioma", "glioma", "pituitary")
SCENARIOS = {"distinct": ("A209",), "repeated": ("A209", "B209"), "no_response": ("A209", "B209")}
EST_S = {"resnet50": 377, "mobilenet_v2": 169, "efficientnet_b0": 241, "densenet121": 390}  # mean s per fold-run, PR #1/#2 run JSONs (MPS)
# Mandatory analysis fields common to PR #1 (ResNet50) and PR #2 per-fold prediction files, plus provenance/209 extras.
PR2_COMMON = ["model", "arm", "image_id", "patient_id", "protocol", "test_fold", "true_class", "true_class_name",
              "prob_0", "prob_1", "prob_2", "pred_class", "pred_class_name", "confidence", "experiment_input_commit"]
PR1_RESNET_ONLY = ["logit_0", "logit_1", "logit_2", "n_other_total", "n_other_train", "frac_other_train",
                   "single_slice_patient", "has_same_patient_train_slice"]  # kept for ALL models here (superset); absent from PR #2 files
EXTRA_209 = ["stem", "split_sha256"]
SCHEMA = PR2_COMMON + PR1_RESNET_ONLY + EXTRA_209


class RunIntegrityError(RuntimeError):
    pass


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(spl=SPL):
    v = json.loads((spl / "PROPOSED_split_validation_209.json").read_text())
    for fn in FILES.values():
        assert sha(spl / fn) == v["sha256"][fn], f"{fn} changed since validation"
    a = pd.read_csv(spl / FILES["A209"], dtype={"patient_id": str}); b = pd.read_csv(spl / FILES["B209"], dtype={"patient_id": str})
    assert len(a) == len(b) == 3064 and a.image_id.is_unique and b.image_id.is_unique and a.stem.nunique() == 209
    assert set(a.image_id) == set(b.image_id) and (a.sort_values("image_id").label.values == b.sort_values("image_id").label.values).all()
    assert (a.groupby("stem").fold.nunique() == 1).all(), "A209 stem leaks across folds"
    assert (a.groupby("patient_id").fold.nunique() == 1).all()
    assert (pd.crosstab(a.fold, a.label).values == pd.crosstab(b.fold, b.label).values).all()
    hv = json.loads((ROOT / "data/splits/split_validation.json").read_text())["sha256"]
    for fn in ("patient_disjoint_original.csv", "imagelevel_seed11.csv"):
        assert sha(ROOT / "data/splits" / fn) == hv[fn], "historical split modified"
    return a, b


def matrix(scenario):
    rows = []
    for m in MODELS:
        for arm in SCENARIOS[scenario]:
            for fold in range(1, 6):
                rows.append(dict(run_id=f"{m}_{arm}_fold{fold}", scenario=scenario, model=m, arm=arm, fold=fold, train_seed=0,
                                 split_file=f"data/splits/proposed_209/{FILES[arm]}",
                                 pred_out=f"results/retrain_209/runs/{m}_{arm}_fold{fold}_predictions.csv",
                                 run_json=f"results/retrain_209/runs/{m}_{arm}_fold{fold}_run.json",
                                 est_minutes=round(EST_S[m] / 60, 1), role="primary_stem_disjoint" if arm == "A209" else "matched_image_level_control",
                                 status="PROPOSED_NOT_RUN"))
    return pd.DataFrame(rows)


def paths(out, model, arm, fold):
    r = Path(out) / "runs"
    return r / f"{model}_{arm}_fold{fold}_predictions.csv", r / f"{model}_{arm}_fold{fold}_run.json"


# ---------- integrity (each file checked independently) ----------
def check_pred(p, split, arm, fold, model):
    e = []
    try:
        d = pd.read_csv(p, dtype={"patient_id": str, "stem": str})
    except Exception as x:
        return [f"unreadable prediction CSV: {x}"]
    if list(d.columns) != SCHEMA: e.append("prediction columns differ from SCHEMA")
    if e: return e
    exp = split[split.fold == fold]
    if len(d) != len(exp) or not d.image_id.is_unique or set(d.image_id) != set(exp.image_id): e.append("image_id set != split fold")
    for col, val in (("arm", arm), ("test_fold", fold), ("model", model), ("protocol", PROTOCOL[arm])):
        if not (d[col] == val).all(): e.append(f"{col} != {val}")
    m = exp.set_index("image_id").loc[d.image_id]
    if not (m.patient_id.values == d.patient_id.values).all(): e.append("patient_id mismatch with split")
    if not (m.stem.values == d.stem.values).all(): e.append("stem mismatch with split")
    if not (m.label.values - 1 == d.true_class.values).all(): e.append("true_class mismatch with split")
    pr = d[["prob_0", "prob_1", "prob_2"]].to_numpy(float)
    if not (np.isfinite(pr).all() and np.all((pr >= 0) & (pr <= 1)) and np.allclose(pr.sum(1), 1, atol=1e-6)): e.append("invalid probabilities")
    elif not (np.array_equal(pr.argmax(1), d.pred_class.to_numpy()) and np.allclose(d.confidence.to_numpy(), pr.max(1), rtol=0, atol=1e-12)
              and np.array_equal(d.pred_class_name.to_numpy(), np.array(CLASS_NAMES)[pr.argmax(1)])
              and np.array_equal(d.true_class_name.to_numpy(), np.array(CLASS_NAMES)[d.true_class.to_numpy()])): e.append("pred/confidence/class-name inconsistent")
    if not np.isfinite(d[["logit_0", "logit_1", "logit_2"]].to_numpy(float)).all(): e.append("non-finite logits")
    if d.split_sha256.nunique() != 1: e.append("split_sha256 not constant")
    return e


def check_run(j_path, p_path, split_sha, arm, fold, model, n_train, n_test, n_epochs, allowed_devices):
    try:
        j = json.loads(Path(j_path).read_text())
    except Exception as x:
        return [f"unreadable run JSON: {x}"]
    e = []
    try:
        if not (j["model"] == model and j["arm"] == arm and j["fold"] == fold): e.append("model/arm/fold mismatch")
        if j["device"] not in allowed_devices: e.append(f"device {j['device']} not allowed")
        if j["split_sha256"] != split_sha: e.append("split_sha256 mismatch")
        if len(j["epochs"]) != n_epochs: e.append("epoch count != CFG")
        if j["train_seed"] != 0: e.append("train_seed != 0")
        if j["n_train"] != n_train or j["n_test"] != n_test: e.append("n_train/n_test mismatch")
        if j.get("predictions_sha256") != sha(p_path): e.append("run JSON does not match prediction file hash")
        if not all(np.isfinite(x["train_loss"]) for x in j["epochs"]): e.append("non-finite train loss")
    except KeyError as x:
        e.append(f"missing key {x}")
    return e


def fold_state(out, model, arm, fold, split, split_sha, n_epochs, allowed_devices):
    """'absent' or 'complete'; raises RunIntegrityError for anything partial/corrupt (never skipped silently)."""
    p, j = paths(out, model, arm, fold)
    tmp = [t for t in (p.with_suffix(".csv.tmp"), j.with_suffix(".json.tmp")) if t.exists()]
    if not p.exists() and not j.exists() and not tmp: return "absent"
    if tmp: raise RunIntegrityError(f"leftover temp file(s) {[t.name for t in tmp]}; use --quarantine-partial after inspection")
    if p.exists() != j.exists(): raise RunIntegrityError(f"partial fold: only one of {p.name}/{j.name} exists; use --quarantine-partial after inspection")
    errs = check_pred(p, split, arm, fold, model)
    n_test = int((split.fold == fold).sum())
    errs += check_run(j, p, split_sha, arm, fold, model, len(split) - n_test, n_test, n_epochs, allowed_devices)
    if errs: raise RunIntegrityError(f"{p.name}: {errs}")
    return "complete"


def quarantine(out, model, arm, fold):
    p, j = paths(out, model, arm, fold); q = Path(out) / "quarantine" / time.strftime("%Y%m%dT%H%M%S"); q.mkdir(parents=True, exist_ok=True)
    moved = []
    for f in (p, j, p.with_suffix(".csv.tmp"), j.with_suffix(".json.tmp")):
        if f.exists(): os.replace(f, q / f.name); moved.append(f.name)
    return moved


@contextlib.contextmanager
def uninterruptible():
    """Defer SIGINT/SIGTERM while the two result files are being published so a stop cannot leave a half-written pair."""
    old = {}
    try:
        for s in (signal.SIGINT, signal.SIGTERM): old[s] = signal.signal(s, signal.SIG_IGN)
    except ValueError:  # not main thread (tests)
        old = {}
    try: yield
    finally:
        for s, h in old.items(): signal.signal(s, h)


def build_frame(split, fold, model, arm, logits, commit):
    from leakage.splits import exposure_table
    te = split[split.fold == fold].reset_index(drop=True)
    lg = np.asarray(logits, np.float64); assert lg.shape == (len(te), 3) and np.isfinite(lg).all()
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    ex = exposure_table(split.image_id.values, split.patient_id.values, split.fold.values).set_index("image_id").loc[te.image_id].reset_index(drop=True)
    d = te[["image_id", "patient_id", "label", "stem"]].copy(); d.insert(0, "model", model); d.insert(1, "arm", arm)
    d["protocol"] = PROTOCOL[arm]; d["test_fold"] = fold; d["true_class"] = d.label - 1
    d["true_class_name"] = np.array(CLASS_NAMES)[d.true_class.to_numpy()]
    for k in range(3): d[f"prob_{k}"] = pr[:, k]
    d["pred_class"] = pr.argmax(1); d["pred_class_name"] = np.array(CLASS_NAMES)[d.pred_class.to_numpy()]; d["confidence"] = pr.max(1)
    d["experiment_input_commit"] = commit
    for k in range(3): d[f"logit_{k}"] = lg[:, k]
    for c in ("n_other_total", "n_other_train", "frac_other_train", "single_slice_patient", "has_same_patient_train_slice"): d[c] = ex[c].values
    d["split_sha256"] = sha(SPL / FILES[arm]) if (SPL / FILES[arm]).exists() else ""
    return d[SCHEMA]


def run_folds(model, arm, train_fn, out=OUT, spl=SPL, *, max_new_runs=None, pre_fold=None, cooldown_s=60, sleep=time.sleep,
              commit="unknown", allowed_devices=("mps",), n_epochs=None, log=print, extra_run_fields=None):
    """Serial orchestration with resume safety. train_fn(model, split_df, fold) -> dict(logits, epochs, runtime_s, device)."""
    if n_epochs is None:
        from run_sanity import CFG; n_epochs = CFG["epochs"]
    split = pd.read_csv(Path(spl) / FILES[arm], dtype={"patient_id": str, "stem": str}).sort_values("image_id").reset_index(drop=True)
    split_sha = sha(Path(spl) / FILES[arm]); (Path(out) / "runs").mkdir(parents=True, exist_ok=True); new = 0
    for fold in range(1, 6):
        if fold_state(out, model, arm, fold, split, split_sha, n_epochs, allowed_devices) == "complete":
            log(f"[{model} {arm} f{fold}] verified existing run"); continue
        if new and cooldown_s: sleep(cooldown_s)
        if pre_fold: pre_fold()
        tr = split[split.fold != fold]; res = train_fn(model, split, fold)
        d = build_frame(split, fold, model, arm, res["logits"], commit); d["split_sha256"] = split_sha
        p, j = paths(out, model, arm, fold)
        run = {"model": model, "arm": arm, "protocol": PROTOCOL[arm], "fold": fold, "n_train": int(len(tr)), "n_test": int(len(d)),
               "runtime_s": float(res["runtime_s"]), "device": res["device"], "train_seed": 0, "split_sha256": split_sha,
               "experiment_input_commit": commit, "epochs": res["epochs"], "test_acc_final": float((d.pred_class == d.true_class).mean()),
               **(extra_run_fields or {})}
        with uninterruptible():
            d.to_csv(p.with_suffix(".csv.tmp"), index=False); os.replace(p.with_suffix(".csv.tmp"), p)
            run["predictions_sha256"] = sha(p)
            j.with_suffix(".json.tmp").write_text(json.dumps(run, indent=2)); os.replace(j.with_suffix(".json.tmp"), j)
        if fold_state(out, model, arm, fold, split, split_sha, n_epochs, allowed_devices) != "complete": raise RunIntegrityError("post-write verification failed")
        log(f"[{model} {arm} f{fold}] COMPLETE runtime={run['runtime_s']:.0f}s acc={run['test_acc_final']:.4f}"); new += 1
        if max_new_runs is not None and new >= max_new_runs: return new
    return new


# ---------- real training (PR #2 recipe; never executed in this preparation) ----------
def build_model(name):
    import torch, torchvision
    from run_multimodel import make_model
    if name == "resnet50":  # identical to scripts/run_sanity.py
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(2048, 3)
        return model
    return make_model(name)


def make_real_train_fn():
    import torch, torch.nn.functional as F, torchvision, gc
    from leakage.preprocess import load_all
    from run_sanity import CFG, augment, to_input
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable: refusing CPU-only training")
    if not (ROOT / "data/interim/pre224.npy").exists(): raise RuntimeError("preprocessing cache data/interim/pre224.npy missing: refusing to regenerate silently")
    ids, X = load_all(ROOT / "reports/manifests/image_manifest.csv")
    assert len(ids) == 3064 and X.shape == (3064, 224, 224)
    pos = {int(i): k for k, i in enumerate(ids)}
    Xt = torch.from_numpy(X)

    def train_fn(name, split, fold):
        tr, te = split[split.fold != fold], split[split.fold == fold]
        tr_idx = np.array([pos[int(i)] for i in tr.image_id])
        te_idx = np.array([pos[int(i)] for i in te.image_id])
        ytr = torch.tensor(tr.label.to_numpy() - 1)
        n = len(tr_idx)
        assert n % CFG["batch_size"] != 1
        torch.manual_seed(CFG["train_seed"])
        np.random.seed(CFG["train_seed"])
        model = build_model(name).to("mps")
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
                del bi, xb, yb, logits, loss
            log.append({"epoch": ep + 1, "train_loss": tl / n, "train_acc": tc / n, "lr_end": sched.get_last_lr()[0],
                        "mps_active_gib": torch.mps.current_allocated_memory() / 2**30,
                        "mps_driver_gib": torch.mps.driver_allocated_memory() / 2**30})
            print(f"[{name} f{fold}] epoch {ep+1}/{CFG['epochs']} loss={tl/n:.4f} acc={tc/n:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
            torch.mps.empty_cache()
        model.eval()
        outputs = []
        with torch.no_grad():
            for b in range(0, len(te_idx), CFG["eval_batch_size"]):
                outputs.append(model(to_input(Xt[te_idx[b:b + CFG["eval_batch_size"]]]).to("mps")).float().cpu())
        lg = torch.cat(outputs).numpy().astype(np.float64)
        assert np.isfinite(lg).all()
        res = {"logits": lg, "epochs": log, "runtime_s": time.time() - t0, "device": "mps"}
        del model, opt, sched, outputs
        gc.collect()
        torch.mps.empty_cache()
        return res
    return train_fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--approved-by"); ap.add_argument("--scenario", choices=tuple(SCENARIOS))
    ap.add_argument("--model", choices=MODELS); ap.add_argument("--arm", choices=tuple(FILES)); ap.add_argument("--max-new-runs", type=int)
    ap.add_argument("--quarantine-partial", nargs=3, metavar=("MODEL", "ARM", "FOLD"))
    a = ap.parse_args(argv)
    if a.quarantine_partial:
        m, arm, f = a.quarantine_partial; print("quarantined:", quarantine(OUT, m, arm, int(f))); return 0
    if not a.train:
        validate()
        for sc, fn in (("distinct", "run_matrix_209_distinct_20.csv"), ("repeated", "run_matrix_209_repeated_or_noresponse_40.csv")):
            matrix(sc).to_csv(SPL / fn, index=False)
        print("splits OK; matrices: distinct=%d runs (%.2f h), repeated/no_response=%d runs (%.2f h)" % (
            len(matrix("distinct")), matrix("distinct").est_minutes.sum() / 60, len(matrix("repeated")), matrix("repeated").est_minutes.sum() / 60))
        return 0
    if not (a.approved_by and a.model and a.arm and a.scenario): sys.exit("--train requires --approved-by, --scenario, --model and --arm; refusing.")
    if a.arm not in SCENARIOS[a.scenario]: sys.exit(f"scenario '{a.scenario}' does not include arm {a.arm}; refusing.")
    validate(); OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / "train.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit("Another 209 training process is active (serial execution only)")
    import subprocess
    commit = os.environ.get("EXPERIMENT_INPUT_COMMIT", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    from safe_train import host_snapshot, unsafe

    def pre_fold():
        s = host_snapshot(); r = unsafe(s)
        if s["available_gib"] < 8: r = r or "available RAM below 8 GiB before fold"
        if s["pressure_free_pct"] is not None and s["pressure_free_pct"] < 35: r = r or "memory pressure free below 35% before fold"
        print("PRE_FOLD_RESOURCE_CHECK", s, "reason=", r, flush=True)
        if r: raise RuntimeError(f"unsafe resources, stopping without losing completed folds: {r}")
    from run_sanity import CFG
    (OUT).mkdir(exist_ok=True)
    (OUT / f"environment_{a.model}_{a.arm}.json").write_text(json.dumps({"model": a.model, "arm": a.arm, "scenario": a.scenario, "approved_by": a.approved_by,
                                                                           "config": CFG, "input_commit": commit}, indent=2))
    run_folds(a.model, a.arm, make_real_train_fn(), max_new_runs=a.max_new_runs, pre_fold=pre_fold, commit=commit,
              extra_run_fields={"scenario": a.scenario, "approved_by": a.approved_by, "cfg": CFG})
    return 0


if __name__ == "__main__":
    sys.exit(main())
