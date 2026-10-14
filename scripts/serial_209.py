"""Monitored, serial, one-fold-per-process execution of the 209-group runs (PR #2 safety protocol: serial_model.py + safe_train.py).
PROPOSED / NEVER EXECUTED. Requires the same --approved-by/--scenario gate as run_retrain_209.py (passed through).

Safety: single lock; initial and pre-fold preflight (RAM >= 8 GiB, memory pressure free >= 35%, disk, thermal, responsiveness) with a bounded
wait; cooling pause between folds; the child runs under caffeinate + nice in its own session and is stopped with SIGINT (never SIGKILL) only
if a RAM/pressure/thermal/responsiveness limit is crossed; only OUR child process group is ever signalled. Completed folds are kept; the next
invocation re-verifies them (both files) and resumes. No automatic retry, no git push, no shutdown/restart of the machine.
"""
import argparse, fcntl, os, select, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_retrain_209 as R
from safe_train import host_snapshot, unsafe  # PR #2 thresholds, reused unchanged


def preflight(snapshot=host_snapshot, check=unsafe):
    s = snapshot(); r = check(s)
    if s["available_gib"] < 8: r = r or "available RAM below 8 GiB before next fold"
    if s["pressure_free_pct"] is not None and s["pressure_free_pct"] < 35: r = r or "memory-pressure free below 35% before next fold"
    return s, r


def wait_for_headroom(timeout_s, earliest_s=0, snapshot=host_snapshot, check=unsafe, sleep=time.sleep, now=time.monotonic):
    deadline, earliest = now() + timeout_s, now() + earliest_s
    while True:
        s, r = preflight(snapshot, check)
        print("PRE_FOLD_RESOURCE_CHECK", s, "reason=", r, flush=True)
        if r is None and now() >= earliest: return
        if now() >= deadline: raise RuntimeError(f"Resource headroom did not recover in {timeout_s}s: {r}")
        sleep(min(30, max(1, deadline - now())))


def monitored_child(cmd, log_path, snapshot=host_snapshot, check=unsafe, interval=15, env=None):
    """Run cmd in its own session; SIGINT it (only it) on unsafe host state. Returns (returncode, safety_reason)."""
    with open(log_path, "a", buffering=1) as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        reason, nxt = None, time.monotonic() + interval
        while True:
            if select.select([proc.stdout], [], [], 1)[0]:
                line = proc.stdout.readline()
                if line: print(line, end="", flush=True); log.write(line)
            if time.monotonic() >= nxt and proc.poll() is None:
                try: s = snapshot(); log.write(f"RESOURCE_CHECK {s}\n"); reason = check(s)
                except (OSError, subprocess.TimeoutExpired) as e: reason = f"responsiveness check failed: {e}"
                if reason:
                    log.write(f"SAFETY_STOP {reason}\n"); print("SAFETY_STOP", reason, flush=True)
                    os.killpg(proc.pid, signal.SIGINT); break
                nxt = time.monotonic() + interval
            if proc.poll() is not None: break
        try: rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM); rc = proc.wait(timeout=10)
        for line in proc.stdout: print(line, end="", flush=True); log.write(line)
    return rc, reason


def completed(model, arm, split, split_sha):
    from run_sanity import CFG
    return sum(R.fold_state(R.OUT, model, arm, f, split, split_sha, CFG["epochs"], ("mps",)) == "complete" for f in range(1, 6))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=R.MODELS); ap.add_argument("--arm", required=True, choices=tuple(R.FILES))
    ap.add_argument("--scenario", required=True, choices=tuple(R.SCENARIOS)); ap.add_argument("--approved-by", required=True)
    ap.add_argument("--cooldown-seconds", type=int, default=60); ap.add_argument("--preflight-timeout-seconds", type=int, default=120)
    a = ap.parse_args(argv)
    if a.arm not in R.SCENARIOS[a.scenario]: sys.exit(f"scenario '{a.scenario}' does not include arm {a.arm}; refusing.")
    import pandas as pd, torch
    if not torch.backends.mps.is_available(): sys.exit("MPS unavailable; CPU training prohibited")
    R.validate(); R.OUT.mkdir(parents=True, exist_ok=True)
    lock = open(R.OUT / "serial.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit("Another serial 209 process is active")
    split = pd.read_csv(R.SPL / R.FILES[a.arm], dtype={"patient_id": str, "stem": str}).sort_values("image_id").reset_index(drop=True)
    sha = R.sha(R.SPL / R.FILES[a.arm]); first = True
    env = os.environ.copy(); env["TORCH_HOME"] = str(ROOT / "data/torch"); env["MPLCONFIGDIR"] = str(ROOT / "data/matplotlib")
    env["EXPERIMENT_INPUT_COMMIT"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    while True:
        n = completed(a.model, a.arm, split, sha)  # raises RunIntegrityError on partial/corrupt: stop, never skip
        print("VERIFIED_COMPLETED_FOLDS", a.model, a.arm, n, flush=True)
        if n == 5: print("MODEL_ARM_COMPLETE", a.model, a.arm); return 0
        wait_for_headroom(a.preflight_timeout_seconds, 0 if first else a.cooldown_seconds)
        cmd = ["caffeinate", "-i", "/usr/bin/nice", "-n", "10", sys.executable, str(ROOT / "scripts/run_retrain_209.py"), "--train", "--approved-by", a.approved_by,
               "--scenario", a.scenario, "--model", a.model, "--arm", a.arm, "--max-new-runs", "1"]
        rc, reason = monitored_child(cmd, R.OUT / f"train_{a.model}_{a.arm}.log", env=env)
        if rc or reason: print("STOPPED", {"returncode": rc, "safety_stop": reason}, "completed folds kept; re-run to resume"); return rc or 2
        first = False


if __name__ == "__main__":
    sys.exit(main())
