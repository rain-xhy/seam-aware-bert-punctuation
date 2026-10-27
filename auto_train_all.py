"""完全自动化训练脚本 - 可靠版

功能:
1. 按顺序训练 T3 → T4 → T5 → T2
2. 日志正确输出到 logs/<name>_train.log
3. 一个完成自动启动下一个
4. 失败自动重试（最多3次）
5. 完成后自动启动评估队列
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import os

PYTHON = r"E:\anaconda\app\envs\yolov8\python.exe"
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"

# 训练队列 - 使用正确的命令行参数
COMMON_ARGS = [
    "--data_dir", "prepared_data_complete",
    "--bert_dir", "bert-base-chinese",
    "--epochs", "3",
    "--batch_size", "1",
    "--grad_accum", "8",
    "--seed", "42"
]

TRAINING_QUEUE = [
    {
        "name": "T3_noedge",
        "desc": "无边缘感知对照",
        "args": COMMON_ARGS + [
            "--out_dir", "checkpoints/T3_noedge",
            "--max_len", "400",
            "--edge_aware_prob", "0"
        ]
    },
    {
        "name": "T4_fixed256",
        "desc": "固定256长度对照",
        "args": COMMON_ARGS + [
            "--out_dir", "checkpoints/T4_fixed256",
            "--min_len", "256",
            "--max_len", "256"
        ]
    },
    {
        "name": "T5_no_class_weight",
        "desc": "关闭类别加权对照",
        "args": COMMON_ARGS + [
            "--out_dir", "checkpoints/T5_no_class_weight",
            "--max_len", "400"
            # 不加 --class_weights 就是关闭
        ]
    },
    {
        "name": "T2_center_only",
        "desc": "仅监督中心区对照",
        "args": COMMON_ARGS + [
            "--out_dir", "checkpoints/T2_center_only",
            "--max_len", "400",
            "--center_only"
        ]
    }
]


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line, flush=True)

    queue_log = LOGS_DIR / "auto_training_queue.log"
    with open(queue_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_for_completion(name, process, log_file, timeout_hours=4):
    """等待训练完成，实时显示进度"""
    log(f"Monitoring {name}...")

    start_time = time.time()
    last_line = ""
    no_update_count = 0

    while True:
        # 检查进程是否还在运行
        ret = process.poll()
        if ret is not None:
            if ret == 0:
                log(f"{name} process completed successfully ✓")
                return True
            else:
                log(f"{name} process failed with exit code {ret}", "ERROR")
                return False

        # 检查超时
        elapsed = time.time() - start_time
        if elapsed > timeout_hours * 3600:
            log(f"{name} timeout after {timeout_hours} hours", "ERROR")
            try:
                process.terminate()
                time.sleep(5)
                process.kill()
            except:
                pass
            return False

        # 读取日志最新进度
        if log_file.exists():
            try:
                lines = log_file.read_text(encoding='utf-8', errors='ignore').splitlines()
                for line in reversed(lines[-10:]):
                    if 'step' in line and 'loss' in line:
                        if line != last_line:
                            print(f"\r{name}: {line.strip()}", end='', flush=True)
                            last_line = line
                            no_update_count = 0
                        break
            except:
                pass

        # 检测卡住（20分钟无更新，考虑验证阶段可能较慢且无日志）
        no_update_count += 1
        if no_update_count > 600:  # 600 * 2s = 20min
            log(f"\n{name} appears stuck (no log update for 20min)", "WARN")
            return False

        time.sleep(2)


def run_training(name, desc, args, max_retries=3):
    """运行单个训练"""
    log(f"\n{'='*60}")
    log(f"Starting {name}: {desc}")
    log(f"{'='*60}")

    log_file = LOGS_DIR / f"{name}_train.log"

    for attempt in range(1, max_retries + 1):
        log(f"Attempt {attempt}/{max_retries}")

        # 清理GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        time.sleep(3)

        # 构建完整命令
        cmd = [PYTHON, "-X", "utf8", "train.py"] + args

        log(f"Command: {' '.join(cmd)}")

        try:
            # 打开日志文件（追加模式）
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Started at {datetime.now()}\n")
                f.write(f"Attempt {attempt}/{max_retries}\n")
                f.write(f"{'='*60}\n\n")
                f.flush()

                # 启动训练进程，stdout和stderr都写入日志
                process = subprocess.Popen(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    bufsize=1  # 行缓冲
                )

            log(f"{name} started with PID {process.pid}")

            # 等待完成
            if wait_for_completion(name, process, log_file):
                print()  # 换行
                log(f"{name} completed successfully! ✓")
                return True
            else:
                print()  # 换行
                log(f"{name} failed on attempt {attempt}", "WARN")

                # 杀死进程（如果还活着）
                try:
                    if process.poll() is None:
                        process.terminate()
                        time.sleep(5)
                        if process.poll() is None:
                            process.kill()
                except:
                    pass

                if attempt < max_retries:
                    wait_time = 30 * attempt
                    log(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        except Exception as e:
            log(f"{name} exception: {e}", "ERROR")
            if attempt < max_retries:
                time.sleep(30)
            else:
                return False

    log(f"{name} failed after {max_retries} attempts", "ERROR")
    return False


def start_eval_queue():
    """启动评估队列"""
    log("\n" + "="*60)
    log("All training completed! Starting evaluation queue...")
    log("="*60)

    try:
        eval_script = BASE_DIR / "run_eval_queue.py"
        if not eval_script.exists():
            log("Evaluation script not found", "WARN")
            return

        cmd = [PYTHON, str(eval_script)]
        process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        log(f"Evaluation queue started with PID {process.pid}")

    except Exception as e:
        log(f"Failed to start evaluation queue: {e}", "ERROR")


def main():
    log("="*60)
    log("AUTOMATED TRAINING QUEUE STARTED")
    log(f"Queue: {' → '.join([t['name'] for t in TRAINING_QUEUE])}")
    log("="*60)

    LOGS_DIR.mkdir(exist_ok=True)

    success_count = 0
    failed_list = []

    for i, train_cfg in enumerate(TRAINING_QUEUE, 1):
        name = train_cfg["name"]
        desc = train_cfg["desc"]
        args = train_cfg["args"]

        log(f"\n[{i}/{len(TRAINING_QUEUE)}] {name}")

        if run_training(name, desc, args):
            success_count += 1
        else:
            failed_list.append(name)
            log(f"⚠ {name} FAILED, continuing to next...", "ERROR")

    # 汇总
    log("\n" + "="*60)
    log("TRAINING QUEUE FINISHED")
    log(f"Success: {success_count}/{len(TRAINING_QUEUE)}")
    if failed_list:
        log(f"Failed: {', '.join(failed_list)}", "ERROR")
    else:
        log("All training completed successfully! ✓✓✓")
    log("="*60)

    # 启动评估
    if success_count > 0:
        start_eval_queue()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nQueue interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Queue fatal error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
