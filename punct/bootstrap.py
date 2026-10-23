"""Bootstrap confidence intervals and non-inferiority testing."""
import numpy as np
from typing import List, Tuple, Dict


def bootstrap_f1_ci(
    gold_per_article: List[np.ndarray],
    pred_per_article: List[np.ndarray],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict:
    """Bootstrap置信区间（按文章配对重采样）.

    Args:
        gold_per_article: [num_articles][article_len]
        pred_per_article: [num_articles][article_len]
        n_bootstrap: Bootstrap次数
        confidence: 置信水平

    Returns:
        {'mean': ..., 'ci_lower': ..., 'ci_upper': ..., 'std': ...}
    """
    from punct.metrics import punctuation_prf

    rng = np.random.RandomState(seed)
    n_articles = len(gold_per_article)

    f1_samples = []

    for _ in range(n_bootstrap):
        # 按文章重采样
        indices = rng.choice(n_articles, size=n_articles, replace=True)

        gold_sample = np.concatenate([gold_per_article[i] for i in indices])
        pred_sample = np.concatenate([pred_per_article[i] for i in indices])

        prf = punctuation_prf(gold_sample, pred_sample, num_labels=8, none_label=0)
        f1_samples.append(prf['micro_f1'])

    f1_samples = np.array(f1_samples)

    alpha = 1 - confidence
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    return {
        'mean': float(np.mean(f1_samples)),
        'std': float(np.std(f1_samples)),
        'ci_lower': float(np.percentile(f1_samples, lower_percentile)),
        'ci_upper': float(np.percentile(f1_samples, upper_percentile)),
        'samples': f1_samples  # 保留用于配对检验
    }


def paired_bootstrap_test(
    gold_per_article: List[np.ndarray],
    pred_A_per_article: List[np.ndarray],
    pred_B_per_article: List[np.ndarray],
    delta: float = 0.002,  # 非劣性容差（0.2个百分点）
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict:
    """配对Bootstrap非劣性检验.

    H0: F1_B - F1_A >= -delta (B不劣于A)

    Args:
        gold_per_article: 真实标签
        pred_A_per_article: 方法A预测
        pred_B_per_article: 方法B预测
        delta: 非劣性容差

    Returns:
        {
            'f1_A': ..., 'f1_B': ..., 'diff': ...,
            'ci_lower': ..., 'ci_upper': ...,
            'non_inferior': bool,  # CI下界 >= -delta
            'superior': bool       # CI下界 > 0
        }
    """
    from punct.metrics import punctuation_prf

    rng = np.random.RandomState(seed)
    n_articles = len(gold_per_article)

    # 原始F1
    gold_all = np.concatenate(gold_per_article)
    pred_A_all = np.concatenate(pred_A_per_article)
    pred_B_all = np.concatenate(pred_B_per_article)

    f1_A = punctuation_prf(gold_all, pred_A_all, num_labels=8, none_label=0)['micro_f1']
    f1_B = punctuation_prf(gold_all, pred_B_all, num_labels=8, none_label=0)['micro_f1']
    diff_observed = f1_B - f1_A

    # Bootstrap差值分布
    diff_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_articles, size=n_articles, replace=True)

        gold_sample = np.concatenate([gold_per_article[i] for i in indices])
        pred_A_sample = np.concatenate([pred_A_per_article[i] for i in indices])
        pred_B_sample = np.concatenate([pred_B_per_article[i] for i in indices])

        f1_A_sample = punctuation_prf(gold_sample, pred_A_sample, num_labels=8, none_label=0)['micro_f1']
        f1_B_sample = punctuation_prf(gold_sample, pred_B_sample, num_labels=8, none_label=0)['micro_f1']

        diff_samples.append(f1_B_sample - f1_A_sample)

    diff_samples = np.array(diff_samples)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(diff_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(diff_samples, 100 * (1 - alpha / 2)))

    return {
        'f1_A': f1_A,
        'f1_B': f1_B,
        'diff': diff_observed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'non_inferior': ci_lower >= -delta,  # B不劣于A
        'superior': ci_lower > 0,             # B优于A
        'delta': delta,
        'confidence': confidence
    }


def compute_per_article_f1(
    gold_per_article: List[np.ndarray],
    pred_per_article: List[np.ndarray]
) -> np.ndarray:
    """计算每篇文章的F1分数.

    Returns:
        f1_per_article: [num_articles]
    """
    from punct.metrics import punctuation_prf

    f1_list = []
    for gold, pred in zip(gold_per_article, pred_per_article):
        if len(gold) > 0:
            prf = punctuation_prf(gold, pred, num_labels=8, none_label=0)
            f1_list.append(prf['micro_f1'])
        else:
            f1_list.append(0.0)

    return np.array(f1_list)
