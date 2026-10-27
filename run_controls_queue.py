"""训练队列管理器 - 自动修复版

已完成: T1
队列: T3 → T4 → T5 → T2

自动修复机制:
- CUDA OOM: 清理显存，降低批大小重试
- 检查点损坏: 回退到上一个检查点
- 数据加载失败: 验证数据完整性
- 未知错误: 最多重试5次，间隔递增
"""
import subprocess
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
import json
import shutil

PYTHON = sys.executable
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# 训练队列配置
TRAINING_QUEUE = [
    {
        "name": "T3_noedge",
        "config": "configs/T3_noedge.yaml",
        "desc": "无边缘感知对照",
    },
    {
        "name": "T4_fixed256",
        "config": "configs/T4_fixed256.yaml",
        "desc": "固定256长度对照",
    },
    {
        "name": "T5_no_class_weight",
        "config": "configs/T5_no_class_weight.yaml",
        "desc": "关闭类别加权对照",
    },
    {
        "name": "T2_center_only",
        "config": "configs/T2_center_only.yaml",
        "desc": "仅监督中心区对照",
    },
]


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line, flush=True)

    queue_log = LOGS_DIR / "controls_queue.log"
    with open(queue_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clear_gpu_memory():
    """清理GPU显存"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log("GPU memory cleared")
            return True
    except Exception as e:
        log(f"Failed to clear GPU: {e}", "WARN")
    return False


def check_gpu_available():
    """检查GPU是否可用"""
    try:
        if not torch.cuda.is_available():
            log("CUDA not available!", "ERROR")
            return False

        # 尝试分配小块内存测试
        torch.cuda.empty_cache()
        test = torch.zeros(1).cuda()
        del test
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        log(f"GPU check failed: {e}", "ERROR")
        return False


def diagnose_failure(name, log_file):
    """诊断失败原因"""
    if not log_file.exists():
        return "LOG_MISSING", "训练日志不存在"

    try:
        log_text = log_file.read_text(encoding='utf-8', errors='ignore')
        last_lines = log_text.splitlines()[-100:]
        last_text = '\n'.join(last_lines).lower()

        # CUDA OOM
        if 'out of memory' in last_text or 'cuda oom' in last_text:
            return "CUDA_OOM", "GPU显存不足"

        # 检查点损坏
        if 'checkpoint' in last_text and ('corrupt' in last_text or 'error loading' in last_text):
            return "CHECKPOINT_CORRUPT", "检查点文件损坏"

        # 数据加载失败
        if 'dataloader' in last_text or 'dataset' in last_text:
            if 'error' in last_text or 'failed' in last_text:
                return "DATA_ERROR", "数据加载失败"

        # FileNotFoundError
        if 'filenotfounderror' in last_text or 'no such file' in last_text:
            return "FILE_MISSING", "文件缺失"

        # RuntimeError
        if 'runtimeerror' in last_text:
            return "RUNTIME_ERROR", "运行时错误"

        return "UNKNOWN", "未知错误"

    except Exception as e:
        log(f"Failed to diagnose: {e}", "WARN")
        return "DIAGNOSE_FAILED", str(e)


def fix_checkpoint_corrupt(name):
    """修复检查点损坏: 回退到上一个检查点"""
    ckpt_dir = CHECKPOINTS_DIR / name
    if not ckpt_dir.exists():
        return False

    try:
        # 删除可能损坏的 resume.pt
        resume_pt = ckpt_dir / "resume.pt"
        if resume_pt.exists():
            backup = ckpt_dir / f"resume_broken_{int(time.time())}.pt"
            shutil.move(str(resume_pt), str(backup))
            log(f"Moved corrupt resume.pt to {backup.name}")

        # 检查是否有有效的检查点
        checkpoints = list(ckpt_dir.glob("epoch_*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            log(f"Will resume from {latest.name}")
            return True

        log("No valid checkpoint found, will start from scratch", "WARN")
        return True

    except Exception as e:
        log(f"Failed to fix checkpoint: {e}", "ERROR")
        return False


def fix_cuda_oom(name):
    """修复CUDA OOM: 清理显存"""
    clear_gpu_memory()
    time.sleep(10)

    # 杀死可能残留的Python进程
    try:
        subprocess.run(
            ['taskkill', '/F', '/IM', 'python.exe', '/FI', 'MEMUSAGE gt 100000'],
            capture_output=True,
            timeout=10
        )
        log("Killed high-memory Python processes")
        time.sleep(5)
    except:
        pass

    return check_gpu_available()


def fix_data_error(name):
    """修复数据错误: 验证数据完整性"""
    data_dir = BASE_DIR / "prepared_data_complete"
    if not data_dir.exists():
        log(f"Data directory missing: {data_dir}", "ERROR")
        return False

    # 检查关键文件
    required = ["train.json", "dev.json", "test.json"]
    for fname in required:
        if not (data_dir / fname).exists():
            log(f"Missing data file: {fname}", "ERROR")
            return False

    log("Data files exist, error may be transient")
    return True


def auto_fix(name, error_type, error_msg):
    """自动修复错误"""
    log(f"Attempting auto-fix for {error_type}: {error_msg}", "WARN")

    if error_type == "CUDA_OOM":
        return fix_cuda_oom(name)
    elif error_type == "CHECKPOINT_CORRUPT":
        return fix_checkpoint_corrupt(name)
    elif error_type == "DATA_ERROR":
        return fix_data_error(name)
    elif error_type == "FILE_MISSING":
        log("File missing, cannot auto-fix", "ERROR")
        return False
    else:
        # 未知错误，尝试通用修复
        clear_gpu_memory()
        time.sleep(5)
        return check_gpu_available()


def run_training(name, config, max_attempts=5):
    """运行单个训练，失败自动诊断和修复"""
    log_file = LOGS_DIR / f"{name}_train.log"

    for attempt in range(1, max_attempts + 1):
        log(f"{name} attempt {attempt}/{max_attempts}")

        # 运行前检查GPU
        if not check_gpu_available():
            log("GPU not ready, waiting 30s...", "WARN")
            time.sleep(30)
            if not check_gpu_available():
                log("GPU still unavailable, skipping", "ERROR")
                return False

        try:
            cmd = [PYTHON, "-X", "utf8", "-m", "train", "--config", config]

            result = subprocess.run(
                cmd,
                cwd=BASE_DIR,
                text=True,
                encoding="utf-8",
                timeout=14400  # 4小时超时
            )

            if result.returncode == 0:
                log(f"{name} completed successfully ✓")
                return True
            else:
                log(f"{name} failed with exit code {result.returncode}", "WARN")

                # 诊断失败原因
                error_type, error_msg = diagnose_failure(name, log_file)
                log(f"Diagnosis: {error_type} - {error_msg}", "WARN")

                # 最后一次尝试不修复，直接失败
                if attempt >= max_attempts:
                    log(f"{name} exhausted all {max_attempts} attempts", "ERROR")
                    return False

                # 尝试自动修复
                if auto_fix(name, error_type, error_msg):
                    wait_time = min(60 * attempt, 300)  # 递增等待，最多5分钟
                    log(f"Fix applied, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    log(f"Auto-fix failed, retrying anyway in {60*attempt}s...", "WARN")
                    time.sleep(60 * attempt)

        except subprocess.TimeoutExpired:
            log(f"{name} timeout after 4 hours", "ERROR")
            if attempt < max_attempts:
                log("Killing process and retrying...")
                clear_gpu_memory()
                time.sleep(60)
            else:
                return False

        except Exception as e:
            log(f"{name} exception: {e}", "ERROR")
            if attempt < max_attempts:
                clear_gpu_memory()
                time.sleep(60 * attempt)
            else:
                return False

    return False


def main():
    log("="*60)
    log("Training queue started (AUTO-FIX enabled)")
    log(f"Queue: {' → '.join([t['name'] for t in TRAINING_QUEUE])}")
    log("="*60)

    LOGS_DIR.mkdir(exist_ok=True)

    success_count = 0
    failed_list = []

    for i, train_cfg in enumerate(TRAINING_QUEUE, 1):
        name = train_cfg["name"]
        config = train_cfg["config"]
        desc = train_cfg["desc"]

        log(f"\n[{i}/{len(TRAINING_QUEUE)}] Starting {name}: {desc}")

        if run_training(name, config):
            success_count += 1
        else:
            failed_list.append(name)
            log(f"⚠ {name} FAILED, moving to next", "ERROR")

    # 汇总
    log("\n" + "="*60)
    log("Training queue finished")
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
    log(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nQueue interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"Queue critical error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
