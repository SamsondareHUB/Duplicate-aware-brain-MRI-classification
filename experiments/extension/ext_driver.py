"""Serial, resource-monitored driver for the extension queue. One MPS training process at a time.
Usage: python experiments/extension/ext_driver.py [--only PREFIX] [--max-runs N] [--push-every K]
Resumes by skipping runs whose outputs verify; never overwrites. Stops (no retry) on any resource-safety trigger.
Pre-fold: RAM>=8 GiB, memory-pressure free>=35%. During a run: stop if RAM<5 GiB, pressure free<25%, thermal warning,
disk<10 GiB or a slow (>15 s) responsiveness check. caffeinate is scoped to the training child process only."""
import argparse, fcntl, os, select, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "experiments/extension"))
from safe_train import host_snapshot, unsafe
from ext_specs import build_queue, get_spec, RUNS
from ext_runner import verify, paths

LOGDIR = ROOT / "results/extension"


def preflight(timeout=1800):
    end = time.monotonic() + timeout
    while True:
        s = host_snapshot(); r = unsafe(s)
        if s["available_gib"] < 8: r = r or "available RAM below 8 GiB pre-fold"
        if s["pressure_free_pct"] is not None and s["pressure_free_pct"] < 35: r = r or "memory pressure free below 35% pre-fold"
        if r is None: return s
        if time.monotonic() > end: raise SystemExit(f"UNSAFE_PREFLIGHT: {r}; stopping (resume later)")
        time.sleep(60)


def run_one(run_id, log):
    env = os.environ.copy(); env["TORCH_HOME"] = str(ROOT / "data/torch")
    cmd = ["caffeinate", "-i", "/usr/bin/nice", "-n", "10", sys.executable, str(ROOT / "experiments/extension/ext_runner.py"), run_id]
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    nxt = time.monotonic() + 15; reason = None
    while True:
        if select.select([p.stdout], [], [], 1)[0]:
            line = p.stdout.readline()
            if line: log.write(line); log.flush()
        if p.poll() is not None: break
        if time.monotonic() >= nxt:
            try: reason = unsafe(host_snapshot())
            except Exception as e: reason = f"responsiveness check failed: {e}"
            if reason:
                log.write(f"SAFETY_STOP {reason}\n"); os.killpg(p.pid, signal.SIGINT); break
            nxt = time.monotonic() + 15
    try: rc = p.wait(timeout=60)
    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGTERM); rc = p.wait(timeout=15)
    for line in p.stdout: log.write(line)
    return rc, reason


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--max-runs", type=int, default=10**9)
    ap.add_argument("--push-every", type=int, default=6); ap.add_argument("--cooldown", type=int, default=30)
    a = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    lock = open(LOGDIR / "driver.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("another extension driver is active")
    done = new = 0; pending_files = []
    with open(LOGDIR / "driver.log", "a") as log:
        for rid in build_queue():
            if not rid.startswith(a.only): continue
            if verify(rid): done += 1; continue
            if new >= a.max_runs: break
            s = preflight(); log.write(f"START {rid} {s}\n"); log.flush(); print("START", rid, flush=True)
            rc, reason = run_one(rid, log)
            if rc or not verify(rid):
                log.write(f"FAILED {rid} rc={rc} reason={reason}\n"); print("FAILED", rid, rc, reason, flush=True)
                raise SystemExit(2)
            new += 1; pp, rp = paths(rid); pending_files += [str(pp.relative_to(ROOT)), str(rp.relative_to(ROOT))]
            print("DONE", rid, flush=True)
            subprocess.run(["git", "add", "--", *pending_files], cwd=ROOT, check=True)
            subprocess.run(["git", "commit", "-q", "-m", f"ext run {rid} verified"], cwd=ROOT, check=True); pending_files = []
            if new % a.push_every == 0:
                subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=ROOT)
            time.sleep(a.cooldown)
    print(f"driver finished: previously_verified={done} new={new}", flush=True)


if __name__ == "__main__":
    main()
