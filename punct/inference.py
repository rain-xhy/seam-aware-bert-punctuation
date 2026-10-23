"""Inference methods for punctuation restoration.

Implements 6 baseline methods + methods B and C from 完善方案.md §4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Window:
    """A single inference window."""
    start: int  # start position in original text
    end: int    # end position in original text
    text: str   # window text (chars)


@dataclass
class InferenceConfig:
    """Configuration for inference methods."""
    window_size: int = 400
    stride: int = 300
    mask_left: int = 0   # for method B: left edge mask
    mask_right: int = 0  # for method B: right edge mask
    short_tail: bool = True  # use short tail window (not align to text end)


def create_windows_direct_cut(text: str, window_size: int = 64) -> List[Window]:
    """Baseline 1: Direct cut, no overlap (老师说的最简单方法)."""
    windows = []
    n = len(text)
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        windows.append(Window(start, end, text[start:end]))
    return windows


def create_windows_sequential(text: str, window_size: int = 400) -> List[Window]:
    """Baseline 2: Sequential sliding window (俞敬松2019风格).

    每次推进由模型预测的句界决定，这里简化为固定步长。
    实际使用时需要在推理循环中动态调整。
    """
    windows = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + window_size, n)
        windows.append(Window(pos, end, text[pos:end]))
        pos = end  # 串行：一次推进整个窗口
    return windows


def create_windows_tang_sequential(text: str, window_size: int = 125, overlap: int = 20) -> List[Window]:
    """Baseline 3: Tang改进的串行（丢最后一句）.

    窗口125字，重叠20字，但串行执行。
    """
    windows = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + window_size, n)
        windows.append(Window(pos, end, text[pos:end]))
        pos += (window_size - overlap)  # 步长 = W - V
    return windows


def create_windows_parallel(
    text: str,
    window_size: int = 400,
    stride: int = 300,
    short_tail: bool = True
) -> List[Window]:
    """Parallel sliding windows (方法B/C的基础).

    Args:
        text: 输入文本
        window_size: 窗口大小W
        stride: 步长S（对称屏蔽后的有效距离）
        short_tail: True=短尾窗（最后窗口不贴齐文末），False=贴齐文末

    Returns:
        窗口列表
    """
    windows = []
    n = len(text)

    if n == 0:
        return windows

    if n <= window_size:
        # 整篇文本一个窗口
        windows.append(Window(0, n, text))
        return windows

    pos = 0
    while pos < n:
        if short_tail and pos > 0 and pos + window_size > n:
            # 短尾窗：从上一步开始，长度不足W
            end = n
        else:
            # 正常窗口或贴齐文末
            end = min(pos + window_size, n)

        windows.append(Window(pos, end, text[pos:end]))

        if end >= n:
            break

        pos += stride

    return windows


def create_windows_tang_A(text: str, window_size: int = 125, overlap: int = 20) -> List[Window]:
    """Tang方法A: 并行窗口，后续需删除每段最后一句."""
    return create_windows_parallel(text, window_size, window_size - overlap, short_tail=False)


# === 合并策略 ===

def merge_overlapping_predictions(
    text: str,
    windows: List[Window],
    logits_list: List[np.ndarray],  # [num_windows, window_len, num_labels]
    mask_left: int = 0,
    mask_right: int = 0,
    merge_strategy: str = 'argmax'  # 'argmax' | 'weighted'
) -> np.ndarray:
    """Merge overlapping window predictions into one prediction per char.

    Args:
        text: 原始文本
        windows: 窗口列表
        logits_list: 每个窗口的logits
        mask_left: 左边缘屏蔽（不对称）
        mask_right: 右边缘屏蔽（不对称）
        merge_strategy: 'argmax'=硬屏蔽后取第一个, 'weighted'=按位置加权

    Returns:
        merged_labels: [text_len] 每个字符的预测标签
    """
    n = len(text)
    num_labels = logits_list[0].shape[-1]

    if merge_strategy == 'argmax':
        # 硬屏蔽+argmax（方法B基础版）
        merged_labels = np.full(n, -1, dtype=np.int32)

        for win, logits in zip(windows, logits_list):
            win_len = len(win.text)
            left_valid = mask_left
            right_valid = win_len - mask_right

            for i in range(left_valid, right_valid):
                global_pos = win.start + i
                if 0 <= global_pos < n and merged_labels[global_pos] == -1:
                    merged_labels[global_pos] = np.argmax(logits[i])

        # 填充未覆盖的位置（shouldn't happen with proper masking）
        merged_labels[merged_labels == -1] = 0
        return merged_labels

    elif merge_strategy == 'weighted':
        # 加权合并（方法B优化版）
        prob_sum = np.zeros((n, num_labels), dtype=np.float32)
        weight_sum = np.zeros(n, dtype=np.float32)

        for win, logits in zip(windows, logits_list):
            win_len = len(win.text)
            probs = softmax(logits, axis=-1)  # [win_len, num_labels]

            # 计算位置权重（高斯或经验）
            weights = compute_position_weights(win_len, mask_left, mask_right)

            for i in range(win_len):
                global_pos = win.start + i
                if 0 <= global_pos < n:
                    prob_sum[global_pos] += probs[i] * weights[i]
                    weight_sum[global_pos] += weights[i]

        # 归一化
        for i in range(n):
            if weight_sum[i] > 0:
                prob_sum[i] /= weight_sum[i]

        merged_labels = np.argmax(prob_sum, axis=-1)
        return merged_labels

    else:
        raise ValueError(f"Unknown merge_strategy: {merge_strategy}")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_position_weights(win_len: int, mask_left: int, mask_right: int) -> np.ndarray:
    """计算窗口内每个位置的权重（中间高，边缘低）.

    返回高斯权重，中心位置权重=1.0，边缘渐降。
    """
    weights = np.ones(win_len, dtype=np.float32)

    # 左边缘衰减
    for i in range(mask_left):
        weights[i] = (i + 1) / (mask_left + 1)

    # 右边缘衰减
    for i in range(win_len - mask_right, win_len):
        weights[i] = (win_len - i) / (mask_right + 1)

    return weights


# === Tang方法A的特殊合并：删最后一句 ===

def merge_tang_A(
    text: str,
    windows: List[Window],
    predicted_texts: List[str]  # 每个窗口已经预测好的带标点文本
) -> str:
    """Tang方法A: 删除每段最后一句，再拼接.

    这里简化：删除每个窗口最后一个句号/问号/感叹号之后的内容。
    """
    segments = []
    for pred_text in predicted_texts:
        # 找最后一个句末标点
        last_period = max(
            pred_text.rfind('。'),
            pred_text.rfind('？'),
            pred_text.rfind('！')
        )
        if last_period > 0:
            segments.append(pred_text[:last_period + 1])
        else:
            # 没有句末标点，全保留
            segments.append(pred_text)

    return ''.join(segments)


# === 方法C: 接缝补窗 ===

def identify_low_confidence_seams(
    merged_probs: np.ndarray,  # [text_len, num_labels] 第一遍合并后的概率
    windows: List[Window],
    mask_left: int,
    mask_right: int,
    threshold: float = 0.1,  # max_prob - second_max_prob < threshold
    seam_margin: int = 8     # 接缝定义：离边缘≤8字
) -> List[int]:
    """识别低置信度的接缝位置.

    Returns:
        positions: 需要补窗的中心位置列表
    """
    n = len(merged_probs)
    seam_positions = []

    # 标记接缝区域
    seam_mask = np.zeros(n, dtype=bool)
    for win in windows:
        # 左接缝
        for i in range(win.start + mask_left, min(win.start + mask_left + seam_margin, win.end)):
            if 0 <= i < n:
                seam_mask[i] = True
        # 右接缝
        for i in range(max(win.end - mask_right - seam_margin, win.start), win.end - mask_right):
            if 0 <= i < n:
                seam_mask[i] = True

    # 在接缝区域找低置信度位置
    for i in range(n):
        if seam_mask[i]:
            max_prob = np.max(merged_probs[i])
            sorted_probs = np.sort(merged_probs[i])[::-1]
            second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

            if max_prob - second_prob < threshold:
                seam_positions.append(i)

    return seam_positions


def create_patch_windows(
    text: str,
    seam_positions: List[int],
    window_size: int = 400,
    sentence_boundaries: Optional[List[int]] = None  # 第一遍预测的句界
) -> List[Window]:
    """创建补窗（句界锚定）.

    Args:
        text: 原文
        seam_positions: 低置信度接缝位置
        window_size: 补窗大小
        sentence_boundaries: 第一遍预测的句界位置（用于锚定）

    Returns:
        补窗列表
    """
    patch_windows = []
    n = len(text)

    for pos in seam_positions:
        # 句界锚定：找最近的句界
        if sentence_boundaries:
            nearest_boundary = min(sentence_boundaries, key=lambda x: abs(x - pos))
            center = nearest_boundary
        else:
            center = pos

        # 以center为中心切窗口
        start = max(0, center - window_size // 2)
        end = min(n, start + window_size)

        # 调整start（避免超出）
        if end - start < window_size and start > 0:
            start = max(0, end - window_size)

        patch_windows.append(Window(start, end, text[start:end]))

    return patch_windows


# === 效率统计 ===

def compute_TER(windows: List[Window], text_len: int) -> float:
    """Token Expansion Ratio: 前向token数 / 文本字数."""
    total_tokens = sum(len(w.text) + 2 for w in windows)  # +2 for [CLS] [SEP]
    return total_tokens / text_len if text_len > 0 else 0.0


def compute_coverage(windows: List[Window], text_len: int) -> float:
    """文本覆盖率（检查是否有字符未被任何窗口覆盖）."""
    covered = np.zeros(text_len, dtype=bool)
    for w in windows:
        covered[w.start:w.end] = True
    return np.mean(covered)


def create_cross_article_windows(
    texts: List[str],
    window_size: int = 400
) -> List[Tuple[int, Window]]:
    """跨文章串行：每步把所有文章的当前窗口组成一个batch.

    Args:
        texts: 多篇文章的文本列表
        window_size: 窗口大小

    Returns:
        [(article_idx, window), ...] 按步组织，同一步的窗口可以batch
    """
    # 每篇文章的当前位置
    positions = [0] * len(texts)
    all_windows = []

    step = 0
    while any(pos < len(texts[i]) for i, pos in enumerate(positions)):
        # 这一步：从每篇文章取一个窗口
        for article_idx, text in enumerate(texts):
            pos = positions[article_idx]
            if pos < len(text):
                end = min(pos + window_size, len(text))
                all_windows.append((article_idx, Window(pos, end, text[pos:end])))
                positions[article_idx] = end
        step += 1

    return all_windows


def merge_on_gpu(
    text_len: int,
    windows: List[Window],
    logits_tensor,  # torch.Tensor [num_windows, max_window_len, num_labels]
    mask_left: int = 0,
    mask_right: int = 0,
    device: str = 'cuda'
) -> np.ndarray:
    """GPU侧合并（使用index_add_，一次回传）.

    Args:
        text_len: 原文长度
        windows: 窗口列表
        logits_tensor: GPU上的logits张量
        mask_left, mask_right: 屏蔽范围
        device: GPU设备

    Returns:
        merged_labels: [text_len] CPU numpy数组
    """
    import torch

    num_labels = logits_tensor.shape[-1]

    # GPU上累加概率
    prob_sum = torch.zeros((text_len, num_labels), dtype=torch.float32, device=device)
    count = torch.zeros(text_len, dtype=torch.float32, device=device)

    # 对每个窗口
    for i, win in enumerate(windows):
        win_len = len(win.text)
        probs = torch.softmax(logits_tensor[i, :win_len, :], dim=-1)  # [win_len, num_labels]

        # 计算有效范围
        left_valid = mask_left
        right_valid = win_len - mask_right

        # 位置权重
        weights = torch.ones(win_len, dtype=torch.float32, device=device)
        for j in range(mask_left):
            weights[j] = (j + 1) / (mask_left + 1)
        for j in range(win_len - mask_right, win_len):
            weights[j] = (win_len - j) / (mask_right + 1)

        # 构造全局索引
        global_indices = torch.arange(win.start, win.start + win_len, dtype=torch.long, device=device)

        # 加权概率
        weighted_probs = probs * weights.unsqueeze(-1)

        # GPU侧累加
        prob_sum.index_add_(0, global_indices, weighted_probs)
        count.index_add_(0, global_indices, weights)

    # 归一化
    count = count.unsqueeze(-1).clamp(min=1e-8)
    prob_sum = prob_sum / count

    # 一次回传CPU
    merged_labels = torch.argmax(prob_sum, dim=-1).cpu().numpy()

    return merged_labels
