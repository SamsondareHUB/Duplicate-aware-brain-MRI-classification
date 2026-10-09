"""Run one model's remaining folds in separate, sequential monitored processes."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_multimodel import NAMES, RUNS, SPLITS, validated_run
from safe_train import host_snapshot, unsafe

ROOT = Path(__file__).resolve().parents[1]


def complete_count(model):
    count = 0
    for fold in range(1, 6):
        for arm in ("A", "B"):
            split = pd.read_csv(SPLITS[arm], dtype={"patient_id": str})
            p = RUNS / f"{model}_{arm}_fold{fold}_predictions.csv"
            j = RUNS / f"{model}_{arm}_fold{fold}_run.json"
            if p.exists() or j.exists():
                if not validated_run(p, j, split, arm, fold, model):
                    raise RuntimeError(f"Invalid or incomplete run outputs: {p}, {j}")
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=NAMES)
    a = ap.parse_args()
    baseline = host_snapshot()["swap_gib"]
    print("GLOBAL_SWAP_BASELINE_GIB", baseline, flush=True)
    while True:
        count = complete_count(a.model)
        print("VERIFIED_COMPLETED_FOLDS", a.model, count, flush=True)
        if count == 10:
            print("MODEL_COMPLETE", a.model, flush=True)
            return 0
        # Allow cooling and unrelated interactive work between every fold.
        deadline = time.monotonic() + 20 * 60
        while True:
            time.sleep(60)
            s = host_snapshot()
            reason = unsafe(s, baseline)
            if s["available_gib"] < 8:
                reason = reason or "available RAM below 8 GiB before next fold"
            if s["pressure_free_pct"] is not None and s["pressure_free_pct"] < 35:
                reason = reason or "memory-pressure free below 35% before next fold"
            print("PRE_FOLD_RESOURCE_CHECK", s, "reason=", reason, flush=True)
            if reason is None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Resource headroom did not recover in 20 minutes: {reason}")
        env = os.environ.copy()
        env["MRI_GLOBAL_SWAP_BASELINE_GIB"] = str(baseline)
        rc = subprocess.call([sys.executable, str(ROOT / "scripts/safe_train.py"), a.model,
                              "--max-new-runs", "1"], cwd=ROOT, env=env)
        if rc:
            raise RuntimeError(f"Single-fold process exited {rc}; stopping without automatic retry")
        assert complete_count(a.model) == count + 1


if __name__ == "__main__":
    sys.exit(main())
