"""Serial, locally monitored MPS execution for one architecture.

Run from the repository root. The training process and its caffeinate process
exit together. Safety thresholds stop the current run without discarding any
previously completed fold outputs; no automatic retry occurs.
"""
import argparse
import fcntl
import os
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/multimodel"
OUT.mkdir(parents=True, exist_ok=True)


def host_snapshot():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(ROOT)
    pressure = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True)
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure.stdout)
    thermal = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True)
    thermal_text = thermal.stdout + thermal.stderr
    warning = "No thermal warning level has been recorded" not in thermal_text
    return {"available_gib": mem.available / 2**30, "swap_gib": swap.used / 2**30,
            "disk_gib": disk.free / 2**30, "pressure_free_pct": int(m.group(1)) if m else None,
            "cpu_pct": psutil.cpu_percent(interval=0.2), "load": os.getloadavg()[0],
            "thermal_warning": warning, "thermal_text": thermal_text.strip()}


def unsafe(s, baseline_swap):
    if s["available_gib"] < 5.0: return "available RAM below 5 GiB"
    if s["swap_gib"] - baseline_swap > 0.5: return "swap grew more than 0.5 GiB"
    if s["disk_gib"] < 10: return "disk free below 10 GiB"
    if s["pressure_free_pct"] is not None and s["pressure_free_pct"] < 25: return "memory pressure free below 25%"
    if s["thermal_warning"]: return "macOS reports a thermal warning"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=("mobilenet_v2", "efficientnet_b0", "densenet121"))
    ap.add_argument("--max-new-runs", type=int)
    a = ap.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable on host; CPU training prohibited")
    lock = open(OUT / "train.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another monitored model-training process is active")
    first = host_snapshot()
    print("PRE_RUN_RESOURCE_CHECK", first, flush=True)
    if first["available_gib"] < 5 or first["disk_gib"] < 10 or first["thermal_warning"]:
        raise SystemExit("Unsafe pre-run resource status; training not started")
    baseline_swap = min(float(os.environ.get("MRI_GLOBAL_SWAP_BASELINE_GIB", first["swap_gib"])),
                        first["swap_gib"])
    pre_reason = unsafe(first, baseline_swap)
    if pre_reason:
        raise SystemExit(f"Unsafe pre-run resource status: {pre_reason}")
    env = os.environ.copy()
    env["TORCH_HOME"] = str(ROOT / "data/torch")
    env["MPLCONFIGDIR"] = str(ROOT / "data/matplotlib")
    env["EXPERIMENT_INPUT_COMMIT"] = "919d6f0fb3ca462983d6df6ec06846e41abd9645"
    env["MRI_BASELINE_SWAP_GIB"] = str(baseline_swap)
    cmd = ["caffeinate", "-i", "/usr/bin/nice", "-n", "10", sys.executable,
           str(ROOT / "scripts/run_multimodel.py"), a.model]
    if a.max_new_runs is not None:
        cmd += ["--max-new-runs", str(a.max_new_runs)]
    log_path = OUT / f"train_{a.model}.log"
    with log_path.open("a", buffering=1) as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                start_new_session=True)
        reason = None
        next_check = time.monotonic() + 15
        while True:
            readable, _, _ = select.select([proc.stdout], [], [], 1)
            if readable:
                line = proc.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    log.write(line)
            if time.monotonic() >= next_check and proc.poll() is None:
                s = host_snapshot()
                print("RESOURCE_CHECK", s, flush=True)
                log.write(f"RESOURCE_CHECK {s}\n")
                reason = unsafe(s, baseline_swap)
                if reason:
                    print("SAFETY_STOP", reason, flush=True)
                    log.write(f"SAFETY_STOP {reason}\n")
                    os.killpg(proc.pid, signal.SIGINT)
                    break
                next_check = time.monotonic() + 15
            if proc.poll() is not None:
                break
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            rc = proc.wait(timeout=10)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        print("TRAIN_EXIT", {"model": a.model, "returncode": rc, "safety_stop": reason}, flush=True)
        return rc if rc else (2 if reason else 0)


if __name__ == "__main__":
    sys.exit(main())
