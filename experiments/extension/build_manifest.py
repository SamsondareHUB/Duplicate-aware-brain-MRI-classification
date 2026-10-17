"""EXPERIMENT_MANIFEST.csv: one row per queued run with status COMPLETED (outputs verify) or NOT_STARTED."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ext_specs import build_queue, RUNS, ROOT
from ext_runner import verify
rows = []
for r in build_queue():
    ok = verify(r); d = {}
    if ok: d = json.loads((RUNS / f"{r}_run.json").read_text())
    rows.append({"run_id": r, "group": r.split("_")[0], "status": "COMPLETED" if ok else "NOT_STARTED",
                 "model": d.get("model"), "arm": d.get("arm"), "fold": d.get("fold"), "level": d.get("level"),
                 "split_seed": d.get("split_seed"), "train_seed": d.get("train_seed"), "n_train": d.get("n_train"),
                 "n_test": d.get("n_test"), "runtime_s": d.get("runtime_s") and round(d["runtime_s"], 1),
                 "split_sha256": d.get("split_sha256"), "predictions_sha256": d.get("predictions_sha256"),
                 "input_commit": d.get("experiment_input_commit")})
m = pd.DataFrame(rows); m.to_csv(ROOT / "experiments/extension/EXPERIMENT_MANIFEST.csv", index=False)
print(m.groupby(["group", "status"]).size().unstack(fill_value=0)); print("runtime h:", m.runtime_s.sum() / 3600)
