"""Aggregate the 10 per-run JSON files into results/sanity/run_manifest.json (no training)."""
import glob, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results/sanity"
runs = []
for f in sorted(glob.glob(str(R / "runs/*_run.json"))):
    r = json.load(open(f)); last = r["epochs"][-1]
    runs.append({k: r[k] for k in ("arm", "fold", "n_train", "n_test", "runtime_s", "device", "model", "weights", "train_seed",
                                   "split_sha256", "experiment_input_commit", "test_acc_final", "warnings")}
                | {"final_epoch_train_loss": last["train_loss"], "final_epoch_train_acc": last["train_acc"], "epochs_logged": len(r["epochs"])})
env = json.load(open(R / "environment.json"))
out = {"total_runs_completed": len(runs), "failed_runs": 10 - len(runs), "total_runtime_min": sum(x["runtime_s"] for x in runs) / 60,
       "experiment_input_commit": env["experiment_input_commit"], "device": env["device"], "runs": runs}
(R / "run_manifest.json").write_text(json.dumps(out, indent=2)); print(len(runs), "runs;", round(out["total_runtime_min"], 1), "min")
