"""监控 T3 训练完成并通知
每 30 秒检查一次日志，完成时弹窗提示
"""
import time
from pathlib import Path
import subprocess

LOG_FILE = Path("logs/T3_noedge_train.log")
QUEUE_LOG = Path("logs/controls_queue.log")

def check_t3_done():
    """检查 T3 是否完成"""
    if not LOG_FILE.exists():
        return False

    # 读取最后 50 行
    lines = LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()
    last_lines = '\n'.join(lines[-50:])

    # 检查完成标志
    if 'epoch 3 finished' in last_lines.lower() or \
       'final f1' in last_lines.lower() or \
       'best checkpoint' in last_lines.lower():
        return True

    # 检查队列是否开始 T4
    if QUEUE_LOG.exists():
        queue_lines = QUEUE_LOG.read_text(encoding='utf-8', errors='ignore')
        if 'T4_fixed256 attempt' in queue_lines:
            return True

    return False

def notify():
    """弹窗通知"""
    msg = "T3 训练完成！\n\n可以重启电脑了\n重启后运行：\n1. python run_controls_queue.py\n2. python run_eval_queue.py"

    # Windows 弹窗
    try:
        subprocess.run([
            'msg', '*', msg
        ], shell=True, timeout=5)
    except:
        pass

    # 控制台提示音
    for _ in range(5):
        print('\a')  # 系统提示音
        time.sleep(0.5)

    print("\n" + "="*60)
    print(msg)
    print("="*60)

def main():
    print("开始监控 T3 训练...")
    print("按 Ctrl+C 停止")

    while True:
        if check_t3_done():
            notify()
            break

        # 显示当前进度
        if LOG_FILE.exists():
            lines = LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()
            for line in reversed(lines[-10:]):
                if 'step' in line and '119510' in line:
                    print(f"\r当前: {line.strip()}", end='', flush=True)
                    break

        time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n监控停止")
