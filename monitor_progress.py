"""实时监控训练进度"""
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "logs" / "T3_noedge_train.log"
QUEUE_LOG = BASE_DIR / "logs" / "auto_training_queue.log"

def get_last_line():
    if not LOG_FILE.exists():
        return None
    lines = LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return None

def get_queue_status():
    if not QUEUE_LOG.exists():
        return None
    lines = QUEUE_LOG.read_text(encoding='utf-8', errors='ignore').splitlines()
    for line in reversed(lines[-20:]):
        if 'Monitoring' in line or 'failed' in line or 'FINISHED' in line:
            return line.strip()
    return None

print("开始监控（每10秒刷新一次，Ctrl+C 停止）\n")

last_log = None
count = 0

while True:
    count += 1
    now = datetime.now().strftime("%H:%M:%S")
    current_log = get_last_line()
    queue_status = get_queue_status()

    if current_log != last_log:
        print(f"[{now}] 新输出: {current_log}")
        last_log = current_log
    elif count % 6 == 0:  # 每1分钟输出一次状态
        print(f"[{now}] 无变化: {current_log[:80] if current_log else 'N/A'}")

    if count % 3 == 0:  # 每30秒显示队列状态
        print(f"         队列: {queue_status}")

    time.sleep(10)
