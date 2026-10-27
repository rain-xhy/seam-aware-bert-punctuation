"""智能训练监控器 - 检测训练完成后自动启动下一个

监控策略:
- 检查日志文件最后更新时间和内容
- 检测训练完成标志
- 自动启动下一个训练
- 失败自动修复和重试
"""
import subprocess
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
import json

PYTHON = r"E:\anaconda\app\envs\yolov8\python.exe"
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# 训练队列 - T3 正在跑，监控剩余的
TRAINING_QUEUE = [
    {
        "name": "T3_noedge",
        "config": "configs/T3_noedge.yaml",
        "desc": "无边缘感知对照",
        "status": "RUNNING",  # 已经在跑
    },
    {
        "name": "T4_fixed256",
        "config": "configs/T4_fixed256.yaml",
        "desc": "固定256长度对照",
        "status": "PENDING",
    },
    {
        "name": "T5_no_class_weight",
        "config": "configs/T5_no_class_weight.yaml",
        "desc": "关闭类别加权对照",
        "status": "PENDING",
    },
    {
        "name": "T2_center_only",
        "config": "configs/T2_center_only.yaml",
        "desc": "仅监督中心区对照",
        "status": "PENDING",
    },
]


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line, flush=True)

    monitor_log = LOGS_DIR / "training_monitor.log"
    with open(monitor_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_training_complete(name):
    """检查训练是否完成"""
    log_file = LOGS_DIR / f"{name}_train.log"

    if not log_file.exists():
        return False

    try:
        # 读取最后100行
        lines = log_file.read_text(encoding='utf-8', errors='ignore').splitlines()
        last_lines = '\n'.join(lines[-100:]).lower()

        # 完成标志
        complete_markers = [
            'training completed',
            'all epochs finished',
            'final f1',
            'best checkpoint saved',
            'training finished'
        ]

        for marker in complete_markers:
            if marker in last_lines:
                return True

        # 检查是否到达最后一步（T3是119510步）
        if name == "T3_noedge" and "119510" in last_lines:
            # 看最后的step是否接近119510
            for line in reversed(lines[-20:]):
                if "step" in line and "/119510" in line:
                    parts = line.split()
                    for part in parts:
                        if "/119510" in part:
                            try:
                                current = int(part.split("/")[0])
                                if current >= 119500:  # 接近结束
                                    return True
                            except:
                                pass

        return False

    except Exception as e:
        log(f"Error checking {name}: {e}", "WARN")
        return False


def is_training_stuck(name, timeout_minutes=30):
    """检查训练是否卡住（日志长时间无更新）"""
    log_file = LOGS_DIR / f"{name}_train.log"

    if not log_file.exists():
        return False

    try:
        last_modified = log_file.stat().st_mtime
        minutes_since = (time.time() - last_modified) / 60

        if minutes_since > timeout_minutes:
            log(f"{name} log not updated for {minutes_since:.1f} minutes", "WARN")
            return True

        return False

    except Exception as e:
        log(f"Error checking stuck: {e}", "WARN")
        return False


def wait_for_training_complete(name, check_interval=60, max_wait_hours=6):
    """等待训练完成"""
    log(f"Waiting for {name} to complete...")

    start_time = time.time()
    max_wait_seconds = max_wait_hours * 3600

    while True:
        elapsed = time.time() - start_time

        # 超时
        if elapsed > max_wait_seconds:
            log(f"{name} timeout after {max_wait_hours} hours", "ERROR")
            return False

        # 检查是否完成
        if is_training_complete(name):
            log(f"{name} completed! ✓")
            return True

        # 检查是否卡住
        if elapsed > 600 and is_training_stuck(name):  # 10分钟后开始检查卡住
            log(f"{name} appears stuck", "ERROR")
            return False

        # 显示进度
        log_file = LOGS_DIR / f"{name}_train.log"
        if log_file.exists():
            try:
                lines = log_file.read_text(encoding='utf-8', errors='ignore').splitlines()
                for line in reversed(lines[-10:]):
                    if 'step' in line and 'loss' in line:
                        log(f"{name} progress: {line.strip()}")
                        break
            except:
                pass

        time.sleep(check_interval)


def clear_gpu():
    """清理GPU"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log("GPU cleared")
    except Exception as e:
        log(f"GPU clear failed: {e}", "WARN")


def start_training(name, config, max_retries=3):
    """启动训练，失败重试"""
    log(f"Starting {name}...")

    for attempt in range(1, max_retries + 1):
        log(f"{name} attempt {attempt}/{max_retries}")

        clear_gpu()
        time.sleep(5)

        try:
            # 使用 Popen 启动后台训练
            cmd = [PYTHON, "-X", "utf8", "-m", "train", "--config", config]

            log_file = LOGS_DIR / f"{name}_train.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n{'='*60}\n")
                f.write(f"Started at {datetime.now()}\n")
                f.write(f"{'='*60}\n\n")

            process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True
            )

            log(f"{name} started with PID {process.pid}")

            # 等待5秒，检查进程是否立即崩溃
            time.sleep(5)
            if process.poll() is not None:
                log(f"{name} crashed immediately (exit {process.returncode})", "ERROR")
                if attempt < max_retries:
                    time.sleep(30)
                    continue
                else:
                    return False

            # 等待训练完成
            if wait_for_training_complete(name):
                return True
            else:
                log(f"{name} did not complete properly", "WARN")
                # 杀死进程
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except:
                    try:
                        process.kill()
                    except:
                        pass

                if attempt < max_retries:
                    log(f"Retrying {name} in 60s...")
                    time.sleep(60)
                else:
                    return False

        except Exception as e:
            log(f"{name} exception: {e}", "ERROR")
            if attempt < max_retries:
                time.sleep(60)
            else:
                return False

    return False


def main():
    log("="*60)
    log("Training Monitor Started")
    log("="*60)

    LOGS_DIR.mkdir(exist_ok=True)

    success_count = 0
    failed_list = []

    for i, train_cfg in enumerate(TRAINING_QUEUE):
        name = train_cfg["name"]
        config = train_cfg["config"]
        desc = train_cfg["desc"]
        status = train_cfg["status"]

        log(f"\n[{i+1}/{len(TRAINING_QUEUE)}] {name}: {desc}")

        if status == "RUNNING":
            # T3 already running, just wait
            log(f"{name} is already running, monitoring...")
            if wait_for_training_complete(name):
                success_count += 1
            else:
                log(f"{name} did not complete, trying to restart", "WARN")
                if start_training(name, config):
                    success_count += 1
                else:
                    failed_list.append(name)

        elif status == "PENDING":
            # Start new training
            if start_training(name, config):
                success_count += 1
            else:
                failed_list.append(name)
                log(f"⚠ {name} FAILED, moving to next", "ERROR")

    # 汇总
    log("\n" + "="*60)
    log("Training Monitor Finished")
    log(f"Success: {success_count}/{len(TRAINING_QUEUE)}")
    if failed_list:
        log(f"Failed: {', '.join(failed_list)}", "ERROR")
    else:
        log("All training completed successfully! ✓")
    log("="*60)

    # 保存汇总
    summary = {
        "finish_time": datetime.now().isoformat(),
        "success": success_count,
        "total": len(TRAINING_QUEUE),
        "failed": failed_list
    }
    summary_file = LOGS_DIR / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nMonitor interrupted", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Monitor error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
