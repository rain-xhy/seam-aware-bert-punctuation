"""评估实验队列管理器

训练完成后自动按顺序运行所有GPU评估实验
每个实验独占GPU，失败重试，记录日志
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON = r"E:\anaconda\app\envs\yolov8\python.exe"
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# 评估实验列表，按顺序执行
EVAL_EXPERIMENTS = [
    {
        "name": "E4_dev",
        "desc": "6种解码方法验证集对比",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_methods",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--split", "dev",
            "--methods", "direct_cut", "serial_yu", "serial_tang", "serial_teacher", "serial_cross", "parallel", "tang_parallel",
            "--batch_size", "4",  # 降低batch_size避免OOM
            "--out", "results/E4_dev.json",
            "--save_preds", "results/E4_dev_preds.npz"
        ],
        "wait_file": "checkpoints/T1_varlen/best.pt",
    },
    {
        "name": "E4_test",
        "desc": "6种解码方法测试集对比",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_methods",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--split", "test",
            "--methods", "direct_cut", "serial_yu", "serial_tang", "serial_teacher", "serial_cross", "parallel", "tang_parallel",
            "--batch_size", "4",  # 降低batch_size避免OOM
            "--out", "results/E4_test.json",
            "--save_preds", "results/E4_test_preds.npz"
        ],
        "wait_file": "results/E4_dev.json",
    },
    {
        "name": "E3_dev",
        "desc": "窗口/重叠/尾窗扫描+EDEP曲线",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_sweep",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--split", "dev",
            "--windows", "128", "256", "400", "510",
            "--overlaps", "0", "50", "100", "200",
            "--tails", "short", "aligned",
            "--edep",
            "--out", "results/E3_dev.json"
        ],
        "wait_file": "checkpoints/T1_varlen/best.pt",
    },
    {
        "name": "E5_dev",
        "desc": "合并权重对比",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_e5_merge",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--split", "dev",
            "--window", "400",
            "--overlap", "100",
            "--out", "results/E5_dev.json"
        ],
        "wait_file": "checkpoints/T1_varlen/best.pt",
    },
    {
        "name": "E7_dev_full",
        "desc": "补窗与对照完整版",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_e7",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--cal_split", "dev", "--cal_limit", "500",
            "--eval_split", "dev", "--eval_skip", "500",
            "--out", "results/E7_dev_full.json",
            "--save_preds", "results/E7_dev_full_preds.npz"
        ],
        "wait_file": "checkpoints/T1_varlen/best.pt",
    },
    {
        "name": "E8_matrix",
        "desc": "2×4配置矩阵",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_matrix",
            "--models", "T1_varlen=checkpoints/T1_varlen/best.pt", "T4_fixed256=checkpoints/T4_fixed256/best.pt",
            "--split", "dev",
            "--windows", "64", "125", "400", "510",
            "--out", "results/E8_matrix.json"
        ],
        "wait_file": "checkpoints/T4_fixed256/best.pt",
    },
    {
        "name": "fixed_core",
        "desc": "固定核心隔离实验",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_fixed_core",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--split", "dev",
            "--core", "192",
            "--contexts", "0", "14", "31", "63",
            "--out", "results/fixed_core.json"
        ],
        "wait_file": "checkpoints/T1_varlen/best.pt",
    },
    {
        "name": "probe_batch",
        "desc": "批大小拐点探测",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.probe_batch",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--split", "dev",
            "--limit", "100",
            "--max_batch", "32",
            "--out", "results/probe_batch.json"
        ],
        "wait_file": "checkpoints/T1_varlen/best.pt",
    },
    {
        "name": "test_long_text",
        "desc": "10万字长文本测试",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.test_long_text",
            "--ckpt", "checkpoints/T1_varlen/best.pt",
            "--length", "100000"
        ],
        "wait_file": "checkpoints/T1_varlen/best.pt",
    },
    {
        "name": "H1_bootstrap",
        "desc": "H1假设检验1000次bootstrap",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_hypotheses",
            "--split", "dev", "--skip", "500",
            "--serial_npz", "results/E4_dev_preds.npz",
            "--serial_name", "serial_teacher",
            "--b_npz", "results/E4_dev_preds.npz",
            "--b_name", "parallel",
            "--e7_npz", "results/E7_dev_full_preds.npz",
            "--delta", "0.002",
            "--n_boot", "1000",
            "--out", "results/H1_bootstrap.json"
        ],
        "wait_file": "results/E7_dev_full_preds.npz",
    },
    {
        "name": "show_errors",
        "desc": "误差样例展示",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.show_errors",
            "--result", "results/E4_dev.json",
            "--preds", "results/E4_dev_preds.npz",
            "--method", "parallel",
            "--limit", "50"
        ],
        "wait_file": "results/E4_dev_preds.npz",
    },
    {
        "name": "multiwindow_subset",
        "desc": "多窗口子集评测",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.eval_multiwindow_subset",
            "--result", "results/E4_dev.json",
            "--preds", "results/E4_dev_preds.npz",
            "--window", "400",
            "--out", "results/multiwindow_subset.json"
        ],
        "wait_file": "results/E4_dev_preds.npz",
    },
]

# CPU实验，可以和GPU实验并行或串行
CPU_EXPERIMENTS = [
    {
        "name": "baseline_freq",
        "desc": "频率基线",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.run_baseline",
            "--data_dir", "prepared_data_complete",
            "--out", "results/baseline_frequency.json"
        ],
    },
    {
        "name": "verify_data",
        "desc": "数据校验",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.verify_data_hash",
            "prepared_data_complete",
            "--save", "results/data_hashes.json"
        ],
    },
    {
        "name": "recompute_e4",
        "desc": "独立复算E4",
        "cmd": [
            PYTHON, "-X", "utf8", "-m", "scripts.recompute",
            "--result", "results/E4_dev.json",
            "--preds", "results/E4_dev_preds.npz"
        ],
        "wait_file": "results/E4_dev_preds.npz",
    },
]


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


def wait_for_file(filepath, timeout=7200):
    """等待文件出现，超时2小时"""
    path = BASE_DIR / filepath
    start = time.time()
    while not path.exists():
        elapsed = time.time() - start
        if elapsed > timeout:
            log(f"等待文件 {filepath} 超时", "ERROR")
            return False
        if int(elapsed) % 60 == 0 and elapsed > 0:
            log(f"等待 {filepath}，已等待 {int(elapsed/60)} 分钟")
        time.sleep(10)
    log(f"文件 {filepath} 已就绪")
    return True


def run_experiment(exp):
    """运行单个实验，失败重试1次"""
    name = exp["name"]
    desc = exp["desc"]
    cmd = exp["cmd"]
    wait_file = exp.get("wait_file")

    log(f"开始实验 {name}: {desc}")

    # 等待依赖文件
    if wait_file and not wait_for_file(wait_file):
        log(f"实验 {name} 跳过，依赖文件不存在", "WARN")
        return False

    # 记录日志
    log_file = LOGS_DIR / f"{name}_eval.log"

    for attempt in range(2):
        try:
            log(f"实验 {name} 第 {attempt+1} 次尝试")
            with open(log_file, "w", encoding="utf-8") as f:
                result = subprocess.run(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8"
                )

            if result.returncode == 0:
                log(f"实验 {name} 成功完成")
                return True
            else:
                log(f"实验 {name} 失败，退出码 {result.returncode}", "WARN")
                if attempt == 0:
                    log(f"等待 30 秒后重试")
                    time.sleep(30)
        except Exception as e:
            log(f"实验 {name} 异常: {e}", "ERROR")
            if attempt == 0:
                time.sleep(30)

    log(f"实验 {name} 两次尝试均失败", "ERROR")
    return False


def main():
    log("="*60)
    log("评估实验队列启动")
    log("="*60)

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # GPU实验串行执行
    log(f"共 {len(EVAL_EXPERIMENTS)} 个GPU实验待运行")
    success_count = 0
    failed_list = []

    # 已有完整结果的实验可用 --skip 名字1,名字2 跳过
    skip = set()
    if "--skip" in sys.argv:
        skip = set(sys.argv[sys.argv.index("--skip") + 1].split(","))
        log(f"跳过已完成的实验: {sorted(skip)}")

    for i, exp in enumerate(EVAL_EXPERIMENTS, 1):
        log(f"\n进度 {i}/{len(EVAL_EXPERIMENTS)}")
        if exp["name"] in skip:
            log(f"实验 {exp['name']} 已跳过")
            success_count += 1
            continue
        if run_experiment(exp):
            success_count += 1
        else:
            failed_list.append(exp["name"])

    # CPU实验
    log(f"\n共 {len(CPU_EXPERIMENTS)} 个CPU实验")
    for exp in CPU_EXPERIMENTS:
        run_experiment(exp)

    # 汇总
    log("\n" + "="*60)
    log("评估实验队列完成")
    log(f"成功: {success_count}/{len(EVAL_EXPERIMENTS)}")
    if failed_list:
        log(f"失败: {', '.join(failed_list)}", "WARN")
    log("="*60)

    # 生成汇总报告
    summary_file = RESULTS_DIR / "eval_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"评估实验完成时间: {datetime.now()}\n")
        f.write(f"成功: {success_count}/{len(EVAL_EXPERIMENTS)}\n")
        if failed_list:
            f.write(f"失败: {', '.join(failed_list)}\n")
        f.write("\n结果文件:\n")
        for path in sorted(RESULTS_DIR.glob("*.json")):
            f.write(f"  {path.name}\n")

    log(f"汇总报告: {summary_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n用户中断", "WARN")
        sys.exit(1)
    except Exception as e:
        log(f"队列异常: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
