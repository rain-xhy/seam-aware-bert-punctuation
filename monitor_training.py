#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Monitor training progress."""
import os
import time
from pathlib import Path

LOG_DIR = Path(r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\代码\seam-aware-bert-punctuation\logs")

def get_latest_log():
    """Find the most recent training log."""
    logs = list(LOG_DIR.glob("T*_train.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)

def parse_progress(log_path):
    """Parse training log for progress."""
    if not log_path or not log_path.exists():
        return None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    info = {
        'log_file': str(log_path),
        'epochs': [],
        'current_epoch': None,
        'best_f1': 0.0,
        'status': 'unknown'
    }

    for line in lines:
        if 'Training' in line and 'epochs' in line:
            # Training 3 epochs, 119510 batches/epoch, grad_accum=8
            if 'batches/epoch' in line:
                parts = line.split(',')
                for part in parts:
                    if 'batches/epoch' in part:
                        info['total_batches'] = int(part.split()[0])

        elif 'Epoch' in line and 'train_loss' in line:
            # Epoch 1/3 | train_loss=0.2159 | val_loss=0.1494 | time=112.8s
            parts = line.split('|')
            epoch_info = {}
            for part in parts:
                part = part.strip()
                if part.startswith('Epoch'):
                    info['current_epoch'] = part.split()[1]
                elif '=' in part:
                    key, val = part.split('=')
                    epoch_info[key.strip()] = val.strip()

            info['epochs'].append(epoch_info)

            # Update best F1
            if 'Val micro-F1' in line:
                f1_str = line.split('Val micro-F1=')[1].split()[0]
                f1 = float(f1_str)
                if f1 > info['best_f1']:
                    info['best_f1'] = f1

        elif 'Training finished' in line:
            info['status'] = 'completed'
        elif 'Traceback' in line or 'Error' in line:
            info['status'] = 'error'

    if info['current_epoch'] and info['status'] == 'unknown':
        info['status'] = 'running'

    return info

def format_time(seconds):
    """Format seconds to human-readable time."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h{m:02d}m{s:02d}s"

def display_progress(info):
    """Display training progress."""
    if not info:
        print("No training log found.")
        return

    print("=" * 60)
    print(f"Training Monitor")
    print("=" * 60)
    print(f"Log: {info['log_file']}")
    print(f"Status: {info['status'].upper()}")

    if info['current_epoch']:
        print(f"Progress: {info['current_epoch']}")

    if info['epochs']:
        print("\nEpoch History:")
        for i, epoch in enumerate(info['epochs'], 1):
            line = f"  Epoch {i}:"
            if 'train_loss' in epoch:
                line += f" train_loss={epoch['train_loss']}"
            if 'val_loss' in epoch:
                line += f" val_loss={epoch['val_loss']}"
            if 'Val micro-F1' in epoch:
                line += f" F1={epoch['Val micro-F1']}"
            if 'time' in epoch:
                line += f" time={epoch['time']}"
            print(line)

    print(f"\nBest F1: {info['best_f1']:.4f}")

    # Estimate remaining time
    if info['epochs'] and info['status'] == 'running':
        avg_time = sum(float(e.get('time', '0s').rstrip('s')) for e in info['epochs']) / len(info['epochs'])
        current_epoch = int(info['current_epoch'].split('/')[0])
        total_epochs = int(info['current_epoch'].split('/')[1])
        remaining_epochs = total_epochs - current_epoch
        estimated_remaining = avg_time * remaining_epochs

        print(f"\nEstimated remaining: {format_time(estimated_remaining)}")

    print("=" * 60)

def watch_mode(interval=60):
    """Watch mode: update every N seconds."""
    print("Watching training progress (Ctrl+C to stop)...\n")
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            log_path = get_latest_log()
            info = parse_progress(log_path)
            display_progress(info)

            if info and info['status'] in ['completed', 'error']:
                print("\nTraining has ended.")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        watch_mode(interval=60)
    else:
        log_path = get_latest_log()
        info = parse_progress(log_path)
        display_progress(info)
