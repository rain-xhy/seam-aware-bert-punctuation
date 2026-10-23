"""EDEP (Edge Distance Error Profile) computation.

核心创新点1：量化窗口左右边缘误差的不对称性
"""
import numpy as np
from typing import List, Tuple
from punct.inference import Window


def compute_EDEP(
    text: str,
    gold_labels: np.ndarray,
    windows: List[Window],
    pred_labels_per_window: List[np.ndarray],
    max_distance: int = 50,
    side: str = 'left'  # 'left' or 'right'
) -> Tuple[np.ndarray, np.ndarray]:
    """计算EDEP-L或EDEP-R曲线.

    Args:
        text: 原文
        gold_labels: 真实标签 [text_len]
        windows: 窗口列表
        pred_labels_per_window: 每个窗口的预测 [num_windows][window_len]
        max_distance: 最大距离
        side: 'left'计算EDEP-L, 'right'计算EDEP-R

    Returns:
        distances: [max_distance+1] 距离数组
        error_rates: [max_distance+1] 每个距离的错误率
    """
    # 按距离收集错误
    errors_by_distance = {d: [] for d in range(max_distance + 1)}

    for win_idx, win in enumerate(windows):
        pred = pred_labels_per_window[win_idx]
        win_len = len(win.text)

        for i in range(win_len):
            global_pos = win.start + i

            # 计算到边缘的距离
            if side == 'left':
                distance = i  # 离左边缘的距离
            else:  # right
                distance = win_len - 1 - i  # 离右边缘的距离

            if distance <= max_distance:
                # 记录是否错误
                is_error = (pred[i] != gold_labels[global_pos])
                errors_by_distance[distance].append(int(is_error))

    # 计算每个距离的错误率
    distances = np.arange(max_distance + 1)
    error_rates = np.zeros(max_distance + 1)

    for d in range(max_distance + 1):
        if len(errors_by_distance[d]) > 0:
            error_rates[d] = np.mean(errors_by_distance[d])
        else:
            error_rates[d] = 0.0

    return distances, error_rates


def find_masking_threshold(
    distances: np.ndarray,
    error_rates: np.ndarray,
    baseline_error: float,
    tolerance: float = 0.02
) -> int:
    """找到屏蔽阈值：错误率回到baseline_error ± tolerance的位置.

    Args:
        distances: 距离数组
        error_rates: 错误率数组
        baseline_error: 窗口中部的基线错误率
        tolerance: 容差

    Returns:
        threshold: 建议的屏蔽距离
    """
    for d in range(len(distances)):
        if error_rates[d] <= baseline_error + tolerance:
            return int(distances[d])

    # 如果全程高于baseline，返回最大距离
    return int(distances[-1])


def compute_asymmetric_masking(
    text: str,
    gold_labels: np.ndarray,
    windows: List[Window],
    pred_labels_per_window: List[np.ndarray],
    window_size: int = 400
) -> Tuple[int, int, dict]:
    """计算不对称屏蔽参数 m_l 和 m_r.

    Returns:
        m_l: 左边缘屏蔽距离
        m_r: 右边缘屏蔽距离
        curves: {'EDEP_L': (distances, error_rates), 'EDEP_R': ...}
    """
    # 计算EDEP-L
    dist_l, err_l = compute_EDEP(text, gold_labels, windows, pred_labels_per_window,
                                   max_distance=window_size // 4, side='left')

    # 计算EDEP-R
    dist_r, err_r = compute_EDEP(text, gold_labels, windows, pred_labels_per_window,
                                   max_distance=window_size // 4, side='right')

    # 窗口中部错误率作为baseline（取中间1/3的平均）
    center_errors = []
    for win_idx, win in enumerate(windows):
        pred = pred_labels_per_window[win_idx]
        win_len = len(win.text)
        center_start = win_len // 3
        center_end = 2 * win_len // 3

        for i in range(center_start, center_end):
            global_pos = win.start + i
            if global_pos < len(gold_labels):
                is_error = (pred[i] != gold_labels[global_pos])
                center_errors.append(int(is_error))

    baseline_error = np.mean(center_errors) if center_errors else 0.1

    # 找屏蔽阈值
    m_l = find_masking_threshold(dist_l, err_l, baseline_error)
    m_r = find_masking_threshold(dist_r, err_r, baseline_error)

    curves = {
        'EDEP_L': (dist_l, err_l),
        'EDEP_R': (dist_r, err_r),
        'baseline_error': baseline_error
    }

    return m_l, m_r, curves
