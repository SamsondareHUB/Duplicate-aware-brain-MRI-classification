"""Run one model's remaining folds in separate, sequential monitored processes."""
import argparse
import json
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
    ap.add_argument("--cooldown-seconds", type=int, default=60)
    a = ap.parse_args()
    if a.cooldown_seconds < 0:
        raise SystemExit("Cooldown must be nonnegative")
    initial = host_snapshot()
    baseline = min(float(os.environ.get("MRI_GLOBAL_SWAP_BASELINE_GIB", initial["swap_gib"])),
                   initial["swap_gib"])
    print("GLOBAL_SWAP_BASELINE_GIB", baseline, flush=True)
    first_run = True
    while True:
        count = complete_count(a.model)
        print("VERIFIED_COMPLETED_FOLDS", a.model, count, flush=True)
        if count == 10:
            print("MODEL_COMPLETE", a.model, flush=True)
            return 0
        # Allow cooling and unrelated interactive work between every fold.
        deadline = time.monotonic() + 6 * 60 * 60
        earliest = time.monotonic() + (30 if first_run else a.cooldown_seconds)
        while True:
            time.sleep(min(30, max(1, earliest - time.monotonic())))
            s = host_snapshot()
            reason = unsafe(s, baseline)
            min_ram = 12 if a.model == "densenet121" else 8
            if s["available_gib"] < min_ram:
                reason = reason or f"available RAM below {min_ram} GiB before next fold"
            if s["pressure_free_pct"] is not None and s["pressure_free_pct"] < 35:
                reason = reason or "memory-pressure free below 35% before next fold"
            print("PRE_FOLD_RESOURCE_CHECK", s, "reason=", reason, flush=True)
            if reason is None and time.monotonic() >= earliest:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Resource headroom did not recover in 6 hours: {reason}")
        env = os.environ.copy()
        env["MRI_GLOBAL_SWAP_BASELINE_GIB"] = str(baseline)
        rc = subprocess.call([sys.executable, str(ROOT / "scripts/safe_train.py"), a.model,
                              "--max-new-runs", "1"], cwd=ROOT, env=env)
        if rc:
            raise RuntimeError(f"Single-fold process exited {rc}; stopping without automatic retry")
        assert complete_count(a.model) == count + 1
        # The worker (including its temporary caffeinate process) has exited.
        # Publish only the newly validated prediction and run record.
        fold = count // 2 + 1
        arm = "A" if count % 2 == 0 else "B"
        pred = RUNS / f"{a.model}_{arm}_fold{fold}_predictions.csv"
        record = RUNS / f"{a.model}_{arm}_fold{fold}_run.json"
        files = [str(pred.relative_to(ROOT)), str(record.relative_to(ROOT))]
        environment = ROOT / "results/multimodel" / f"environment_{a.model}.json"
        if count == 0:
            files.append(str(environment.relative_to(ROOT)))
        subprocess.run(["git", "add", "--", *files], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", f"Record {a.model} {arm} fold {fold} verified MPS run"],
                       cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "HEAD:melody/multimodel-evaluation-v1"], cwd=ROOT, check=True)
        local_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        remote_line = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/melody/multimodel-evaluation-v1"],
                                              cwd=ROOT, text=True).strip()
        if remote_line.split()[0] != local_head:
            raise RuntimeError("Remote branch does not match locally committed fold")
        runtime = json.loads(record.read_text())["runtime_s"]
        print("FOLD_PUBLISHED", a.model, arm, fold, "runtime_s", round(runtime),
              "remaining_estimate_s", round(runtime * (9 - count)), "sha", local_head, flush=True)
        first_run = False


if __name__ == "__main__":
    sys.exit(main())
