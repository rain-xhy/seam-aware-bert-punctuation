"""评估队列监控脚本
定期检查评估进度，出现问题时报告
"""
import time
import subprocess
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("logs/eval_queue.log")
MONITOR_LOG = Path("logs/eval_monitor.log")

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(MONITOR_LOG, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

def check_status():
    """检查评估队列状态"""
    if not LOG_FILE.exists():
        return "队列未启动"

    # 读取最后几行
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        last_lines = lines[-10:] if len(lines) >= 10 else lines

    last_text = ''.join(last_lines)

    # 检查是否完成
    if "评估实验队列完成" in last_text or "All experiments completed" in last_text:
        return "✅ 已完成"

    # 检查是否有错误
    if "WARN" in last_text or "ERROR" in last_text or "失败" in last_text:
        # 提取错误信息
        error_lines = [line for line in last_lines if "WARN" in line or "ERROR" in line or "失败" in line]
        return f"⚠️ 发现错误:\n" + "\n".join(error_lines)

    # 检查进度
    for line in reversed(last_lines):
        if "进度" in line or "Progress" in line:
            return f"🔄 {line.strip()}"

    return "🔄 运行中..."

def main():
    log("="*60)
    log("评估队列监控启动")
    log("="*60)

    check_interval = 60  # 每60秒检查一次
    max_checks = 1000    # 最多检查1000次（约16小时）

    for i in range(max_checks):
        status = check_status()
        log(f"检查 {i+1}: {status}")

        if "已完成" in status:
            log("🎉 评估队列已完成！")
            break

        if "发现错误" in status:
            log("❌ 发现错误，需要人工介入")
            log(status)
            break

        time.sleep(check_interval)

    log("="*60)
    log("监控结束")
    log("="*60)

if __name__ == "__main__":
    main()
