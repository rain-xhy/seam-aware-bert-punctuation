"""Repeat the T1 test-set timing run N times with a GPU-state snapshot before each run.

Each repetition writes its own result file (results/timing/T1_test_rep{i}.json), so
the per-method partial-resume logic in run_methods does not skip anything.
Only seconds/chars_per_second are meant to be read from these files; F1 is
deterministic and already reported from E4_test.json.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = r"E:\anaconda\app\envs\yolov8\python.exe"
BASE = Path(__file__).parent
OUT_DIR = BASE / "results" / "timing"
LOG = BASE / "logs" / "timing_repeat.log"
REPS = 3
METHODS = ["direct_cut", "serial_yu", "serial_tang", "serial_teacher",
           "serial_cross", "parallel", "tang_parallel"]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gpu_snapshot():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm",
         "--format=csv,noheader"], capture_output=True, text=True)
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True, text=True)
    return q.stdout.strip(), apps.stdout.strip().replace("\n", " | ")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(exist_ok=True)
    for i in range(1, REPS + 1):
        state, apps = gpu_snapshot()
        log(f"rep {i}/{REPS} start | gpu: {state} | compute apps: {apps}")
        out = OUT_DIR / f"T1_test_rep{i}.json"
        cmd = [PYTHON, "-X", "utf8", "-m", "scripts.run_methods",
               "--ckpt", "checkpoints/T1_varlen/best.pt", "--split", "test",
               "--methods", *METHODS, "--batch_size", "4", "--out", str(out)]
        t0 = time.time()
        with open(BASE / "logs" / f"timing_rep{i}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT).returncode
        state_end, apps_end = gpu_snapshot()
        log(f"rep {i}/{REPS} done rc={rc} wall={time.time() - t0:.0f}s | gpu end: {state_end} | apps: {apps_end}")
        if rc != 0:
            log("run failed, stopping", )
            sys.exit(rc)


if __name__ == "__main__":
    main()
