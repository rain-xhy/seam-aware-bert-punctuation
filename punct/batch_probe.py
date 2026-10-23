"""Batch size probing for GPU memory optimization."""
import time
import torch
import numpy as np
from typing import Tuple, List


def probe_max_batch_size(
    model,
    sample_input_ids: torch.Tensor,  # [1, seq_len]
    sample_attention_mask: torch.Tensor,  # [1, seq_len]
    device: str = 'cuda',
    max_batch: int = 64,
    safety_factor: float = 0.9
) -> int:
    """探测最大可用batch_size（二分搜索）.

    Args:
        model: BERT模型
        sample_input_ids: 示例输入
        sample_attention_mask: 示例mask
        device: GPU设备
        max_batch: 最大尝试batch
        safety_factor: 安全系数（0.9 = 留10%余量）

    Returns:
        max_safe_batch: 安全的最大batch
    """
    model.eval()
    seq_len = sample_input_ids.shape[1]

    def test_batch(batch_size):
        """测试某个batch_size是否OOM."""
        try:
            # 清空缓存
            torch.cuda.empty_cache()

            # 构造batch
            input_ids = sample_input_ids.repeat(batch_size, 1).to(device)
            attention_mask = sample_attention_mask.repeat(batch_size, 1).to(device)

            # 前向
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # 成功
            torch.cuda.synchronize()
            return True

        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                return False
            else:
                raise e

    # 二分搜索
    left, right = 1, max_batch
    best = 1

    while left <= right:
        mid = (left + right) // 2

        if test_batch(mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1

    # 应用安全系数
    safe_batch = max(1, int(best * safety_factor))

    print(f"Batch probing: max={best}, safe={safe_batch} (seq_len={seq_len})")
    return safe_batch


def find_optimal_batch_size(
    model,
    sample_windows: List,  # List of sample text windows
    encode_fn,  # Function to encode windows
    device: str = 'cuda',
    candidates: List[int] = None
) -> Tuple[int, dict]:
    """找到吞吐量最优的batch_size.

    Args:
        model: BERT模型
        sample_windows: 示例窗口
        encode_fn: 编码函数
        device: GPU设备
        candidates: 候选batch sizes

    Returns:
        optimal_batch: 最优batch
        benchmark_results: {batch_size: {'throughput': ..., 'time': ...}}
    """
    if candidates is None:
        candidates = [1, 2, 4, 8, 16, 32, 64]

    model.eval()
    results = {}

    # 编码示例
    encoded = encode_fn(sample_windows)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    total_tokens = input_ids.shape[0] * input_ids.shape[1]

    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()

            # 预热
            for _ in range(2):
                with torch.no_grad():
                    _ = model(input_ids[:batch_size].to(device),
                             attention_mask=attention_mask[:batch_size].to(device))

            # 计时
            torch.cuda.synchronize()
            start = time.time()

            num_batches = (len(sample_windows) + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(sample_windows))

                with torch.no_grad():
                    _ = model(input_ids[batch_start:batch_end].to(device),
                             attention_mask=attention_mask[batch_start:batch_end].to(device))

            torch.cuda.synchronize()
            elapsed = time.time() - start

            throughput = total_tokens / elapsed

            results[batch_size] = {
                'time': elapsed,
                'throughput': throughput,
                'tokens_per_sec': throughput
            }

            print(f"Batch {batch_size:3d}: {elapsed:.3f}s, {throughput:.0f} tokens/s")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Batch {batch_size:3d}: OOM")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    # 找最优
    if len(results) == 0:
        return 1, results

    optimal_batch = max(results.keys(), key=lambda b: results[b]['throughput'])

    # 检查是否有拐点（增益<5%）
    sorted_batches = sorted(results.keys())
    for i in range(1, len(sorted_batches)):
        b_prev = sorted_batches[i - 1]
        b_curr = sorted_batches[i]

        gain = (results[b_curr]['throughput'] - results[b_prev]['throughput']) / results[b_prev]['throughput']

        if gain < 0.05:  # 增益<5%，认为到拐点了
            optimal_batch = b_prev
            print(f"Found elbow at batch_size={b_prev} (gain to {b_curr} only {gain*100:.1f}%)")
            break

    return optimal_batch, results
