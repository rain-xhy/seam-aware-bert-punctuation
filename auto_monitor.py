#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-monitoring script for training - restarts every hour."""
import time
import subprocess
import sys
from pathlib import Path

LOG_FILE = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\代码\seam-aware-bert-punctuation\logs\T1_varlen_train.log")
MONITOR_LOG = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\代码\seam-aware-bert-punctuation\logs\monitor.log")

def check_status():
    """Check training status from log."""
    if not LOG_FILE.exists():
        return "not_started", None

    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    last_lines = lines[-50:] if len(lines) > 50 else lines

    # Check for errors
    for line in last_lines:
        if 'Traceback' in line or 'Error' in line or 'OOM' in line:
            return "error", ''.join(last_lines[-20:])
        if 'Training finished' in line:
            return "completed", ''.join(last_lines[-10:])

    # Check for progress
    for line in reversed(last_lines):
        if 'Epoch' in line and 'train_loss' in line:
            return "running", line.strip()

    return "unknown", None

def log_message(msg):
    """Log to monitor log file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(MONITOR_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

def main():
    log_message("=== Training Monitor Started ===")
    log_message(f"Watching: {LOG_FILE}")

    check_interval = 300  # 5 minutes
    last_status = None
    error_count = 0

    try:
        while True:
            status, info = check_status()

            # Status changed
            if status != last_status:
                log_message(f"Status changed: {last_status} -> {status}")
                last_status = status

                if status == "error":
                    log_message("❌ ERROR DETECTED!")
                    log_message(f"Error details:\n{info}")
                    error_count += 1

                    if error_count >= 3:
                        log_message("Too many errors, stopping monitor.")
                        break

                elif status == "completed":
                    log_message("✅ Training completed successfully!")
                    break

                elif status == "running" and info:
                    log_message(f"Training progress: {info}")

            # Periodic heartbeat
            if status == "running":
                log_message(f"Heartbeat: Training is running...")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        log_message("Monitor stopped by user.")
    except Exception as e:
        log_message(f"Monitor crashed: {e}")

    log_message("=== Training Monitor Ended ===")

if __name__ == '__main__':
    main()
