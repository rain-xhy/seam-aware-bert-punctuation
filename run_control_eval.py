"""补充对照模型评测队列
对 T2/T3/T5 在测试集上运行完整的 6 种方法评测
"""
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

PYTHON = r"E:\anaconda\app\envs\yolov8\python.exe"
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

CONTROL_MODELS = [
    {
        "name": "T2_center_only",
        "desc": "仅中心区监督对照",
        "ckpt": "checkpoints/T2_center_only/best.pt"
    },
    {
        "name": "T3_noedge",
        "desc": "无边缘感知对照",
        "ckpt": "checkpoints/T3_noedge/best.pt"
    },
    {
        "name": "T5_no_class_weight",
        "desc": "无类别权重对照",
        "ckpt": "checkpoints/T5_no_class_weight/best.pt"
    }
]

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def run_model_eval(model_info):
    """对一个模型运行测试集评测"""
    name = model_info["name"]
    ckpt = model_info["ckpt"]

    log(f"开始评测 {name}: {model_info['desc']}")

    if not Path(ckpt).exists():
        log(f"检查点不存在: {ckpt}", "ERROR")
        return False

    out_file = RESULTS_DIR / f"{name}_test.json"
    preds_file = RESULTS_DIR / f"{name}_test_preds.npz"
    log_file = LOGS_DIR / f"{name}_test_eval.log"

    cmd = [
        PYTHON, "-X", "utf8", "-m", "scripts.run_methods",
        "--ckpt", ckpt,
        "--split", "test",
        "--methods", "direct_cut", "serial_yu", "serial_tang", "serial_teacher", "serial_cross", "parallel", "tang_parallel",
        "--batch_size", "4",
        "--out", str(out_file),
        "--save_preds", str(preds_file)
    ]

    log(f"命令: {' '.join(cmd[:10])}...")

    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        log(f"{name} 评测成功 ✅")
        return True
    else:
        log(f"{name} 评测失败 ❌ (退出码 {result.returncode})", "ERROR")
        return False

def main():
    log("="*60)
    log("对照模型补充评测")
    log("="*60)

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    success = 0
    failed = []

    for i, model in enumerate(CONTROL_MODELS, 1):
        log(f"\n进度 {i}/{len(CONTROL_MODELS)}")
        if run_model_eval(model):
            success += 1
        else:
            failed.append(model["name"])

    log("\n" + "="*60)
    log(f"评测完成: 成功 {success}/{len(CONTROL_MODELS)}")
    if failed:
        log(f"失败: {', '.join(failed)}", "WARN")
    log("="*60)

if __name__ == "__main__":
    main()
