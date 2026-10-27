"""Wait until the GPU has been idle for a fixed time, then run the GPU-exclusive jobs.

Idle means memory.used below IDLE_MEM_MIB and utilization below IDLE_UTIL, sampled
every 30 s, for IDLE_MINUTES in a row. Any busy sample resets the clock. Windows
(WDDM) does not report per-process GPU memory, so these two numbers are the only
signal available.

Jobs, in order
  1. encoder weight check of T2/T3/T5 (CPU only, runs immediately)
  2. precision / batch-size agreement on the first 300 test articles
  3. timing of the seven decoding methods, 3 repetitions (run_timing_repeat.py)
While jobs 2 and 3 run, a GPU snapshot is logged every 30 s so a foreign process
that starts mid-run shows up as a jump in memory or utilization.
Everything is logged to logs/gpu_when_idle.log.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = r"E:\anaconda\app\envs\yolov8\python.exe"
BASE = Path(__file__).parent
LOG = BASE / "logs" / "gpu_when_idle.log"
IDLE_MINUTES = 20
RECHECK_MINUTES = 3
MAX_WAIT_HOURS = None  # None: wait indefinitely
HEARTBEAT_MIN = 60
IDLE_MEM_MIB = 400
IDLE_UTIL = 10
POLL_S = 30


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def snapshot():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    parts = [p.strip() for p in r.stdout.strip().split(",")]
    return int(parts[0]), int(parts[1]), parts[2], parts[3]


def wait_idle(minutes, label):
    need = minutes * 60
    start_wait = time.time()
    last_beat = start_wait
    idle_since = None
    while True:
        try:
            mem, util, temp, clk = snapshot()
        except (ValueError, IndexError, OSError) as e:
            log(f"{label}: nvidia-smi query failed ({e!r}), retrying")
            idle_since = None
            time.sleep(POLL_S)
            continue
        busy = mem > IDLE_MEM_MIB or util > IDLE_UTIL
        now = time.time()
        if busy:
            if idle_since is not None:
                log(f"{label}: busy again (mem={mem} MiB util={util}%), idle clock reset")
            idle_since = None
        else:
            idle_since = idle_since or now
            if now - idle_since >= need:
                log(f"{label}: GPU idle for {minutes} min (mem={mem} MiB util={util}% temp={temp} sm={clk} MHz)")
                return True
        if MAX_WAIT_HOURS is not None and now - start_wait > MAX_WAIT_HOURS * 3600:
            log(f"{label}: gave up after {MAX_WAIT_HOURS} h without a long enough idle window")
            return False
        if now - last_beat >= HEARTBEAT_MIN * 60:
            last_beat = now
            log(f"{label}: still waiting, waited {(now - start_wait) / 3600:.1f} h | gpu mem={mem} MiB util={util}%")
        time.sleep(POLL_S)


def run_with_snapshots(cmd, name):
    log(f"start {name}: {' '.join(cmd)}")
    out = open(BASE / "logs" / f"{name}.log", "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=BASE, stdout=out, stderr=subprocess.STDOUT)
    while p.poll() is None:
        try:
            mem, util, temp, clk = snapshot()
            log(f"  {name} running | gpu mem={mem} MiB util={util}% temp={temp} sm={clk} MHz")
        except (ValueError, IndexError, OSError) as e:
            log(f"  {name} running | nvidia-smi query failed ({e!r})")
        time.sleep(POLL_S)
    out.close()
    log(f"done {name} rc={p.returncode}")
    return p.returncode


def main():
    LOG.parent.mkdir(exist_ok=True)
    for tag in ("T2_center_only", "T3_noedge", "T5_no_class_weight"):
        prev = BASE / "logs" / f"encoder_check_{tag}.log"
        if prev.exists() and "changed:" in prev.read_text(encoding="utf-8", errors="ignore"):
            log(f"encoder check for {tag} already done, skipped")
            continue
        rc = run_with_snapshots(
            [PYTHON, "-X", "utf8", "-m", "scripts.check_encoder_weights", f"checkpoints/{tag}/best.pt"],
            f"encoder_check_{tag}")
        if rc != 0:
            log(f"encoder check for {tag} returned {rc}")

    if not wait_idle(IDLE_MINUTES, "before precision check"):
        sys.exit(2)
    rc = run_with_snapshots(
        [PYTHON, "-X", "utf8", "-m", "scripts.verify_precision", "--ckpt", "checkpoints/T1_varlen/best.pt",
         "--split", "test", "--limit", "300", "--out", "results/precision_consistency.json"],
        "precision_check")
    if rc != 0:
        log("precision check failed, continuing to timing")

    if not wait_idle(RECHECK_MINUTES, "before timing"):
        sys.exit(2)
    rc = run_with_snapshots([PYTHON, "run_timing_repeat.py"], "timing_repeat_driver")
    log(f"all done, timing rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
