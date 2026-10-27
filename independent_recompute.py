"""独立复算脚本
从评估结果文件和预测文件重新计算所有指标，验证报告中的数字
参考12号同学的做法
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class IndependentRecomputer:
    """独立复算器 - 不依赖原始评估代码"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.discrepancies = []

    def load_predictions(self, pred_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载预测文件"""
        data = np.load(pred_file, allow_pickle=True)
        preds = data['predictions']  # (n_samples, seq_len)
        labels = data['labels']       # (n_samples, seq_len)
        return preds, labels

    def compute_metrics_from_scratch(self, preds: np.ndarray, labels: np.ndarray,
                                     ignore_index: int = -100) -> Dict[str, float]:
        """从头计算指标（不调用原评估代码）"""
        # 展平并过滤 ignore_index
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()

        mask = labels_flat != ignore_index
        preds_flat = preds_flat[mask]
        labels_flat = labels_flat[mask]

        # 计算混淆矩阵
        num_classes = 8  # 标点符号类别数
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        for pred, label in zip(preds_flat, labels_flat):
            if 0 <= pred < num_classes and 0 <= label < num_classes:
                confusion_matrix[label, pred] += 1

        # 计算 Precision, Recall, F1
        class_metrics = {}
        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(confusion_matrix[i, :].sum())
            }

        # Micro-averaged metrics
        total_tp = np.diag(confusion_matrix).sum()
        total_fp = confusion_matrix.sum() - total_tp
        total_fn = total_fp  # 在多分类中 total_fp = total_fn

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Macro-averaged metrics
        macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])

        # 准确率
        accuracy = total_tp / labels_flat.size if labels_flat.size > 0 else 0.0

        return {
            'accuracy': accuracy,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': confusion_matrix.tolist(),
            'class_metrics': class_metrics,
            'total_samples': int(labels_flat.size)
        }

    def verify_result_file(self, result_file: Path, pred_file: Path = None) -> Dict:
        """验证单个结果文件"""
        print(f"\n{'='*60}")
        print(f"验证: {result_file.name}")
        print(f"{'='*60}")

        # 加载结果文件
        with open(result_file, 'r', encoding='utf-8') as f:
            reported_results = json.load(f)

        verification = {
            'file': result_file.name,
            'status': 'unknown',
            'reported': {},
            'recomputed': {},
            'differences': {}
        }

        # 如果有预测文件，重新计算
        if pred_file and pred_file.exists():
            print(f"  从预测文件重新计算: {pred_file.name}")
            preds, labels = self.load_predictions(pred_file)
            recomputed = self.compute_metrics_from_scratch(preds, labels)

            verification['recomputed'] = recomputed

            # 对比关键指标
            key_metrics = ['micro_f1', 'macro_f1', 'accuracy']
            max_diff = 0.0

            for metric in key_metrics:
                if metric in reported_results:
                    reported_val = reported_results[metric]
                    recomputed_val = recomputed[metric]
                    diff = abs(reported_val - recomputed_val)

                    verification['reported'][metric] = reported_val
                    verification['differences'][metric] = diff

                    max_diff = max(max_diff, diff)

                    status = '✅' if diff < 1e-4 else '⚠️'
                    print(f"  {status} {metric}: reported={reported_val:.4f}, recomputed={recomputed_val:.4f}, diff={diff:.6f}")

            # 判断状态
            if max_diff < 1e-4:
                verification['status'] = 'verified'
                print(f"  ✅ 验证通过 (最大差异: {max_diff:.6f})")
            elif max_diff < 1e-3:
                verification['status'] = 'minor_discrepancy'
                print(f"  ⚠️  轻微差异 (最大差异: {max_diff:.6f})")
                self.discrepancies.append(result_file.name)
            else:
                verification['status'] = 'major_discrepancy'
                print(f"  ❌ 重大差异 (最大差异: {max_diff:.6f})")
                self.discrepancies.append(result_file.name)
        else:
            print(f"  ⚠️  预测文件不存在，无法验证")
            verification['status'] = 'no_predictions'
            verification['reported'] = reported_results

        return verification

    def verify_all_results(self) -> Dict:
        """验证所有结果文件"""
        print("="*80)
        print("独立复算验证")
        print("="*80)

        verifications = []

        # 查找所有结果文件和对应的预测文件
        result_files = list(self.results_dir.glob("E*.json"))

        for result_file in sorted(result_files):
            # 查找对应的预测文件
            pred_file = self.results_dir / f"{result_file.stem}_preds.npz"

            verification = self.verify_result_file(result_file, pred_file)
            verifications.append(verification)

        # 汇总
        print("\n" + "="*80)
        print("验证汇总")
        print("="*80)

        status_counts = defaultdict(int)
        for v in verifications:
            status_counts[v['status']] += 1

        print(f"\n总共验证: {len(verifications)} 个结果文件")
        print(f"  ✅ 验证通过: {status_counts['verified']}")
        print(f"  ⚠️  轻微差异: {status_counts['minor_discrepancy']}")
        print(f"  ❌ 重大差异: {status_counts['major_discrepancy']}")
        print(f"  ⚠️  无预测文件: {status_counts['no_predictions']}")

        if self.discrepancies:
            print(f"\n需要注意的文件:")
            for filename in self.discrepancies:
                print(f"  - {filename}")

        # 保存验证报告
        report = {
            'summary': dict(status_counts),
            'verifications': verifications,
            'discrepancies': self.discrepancies
        }

        report_file = self.results_dir / "verification_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n验证报告已保存: {report_file}")
        print("="*80)

        return report

    def verify_training_results(self):
        """验证训练结果的一致性"""
        print("\n" + "="*80)
        print("验证训练结果")
        print("="*80)

        checkpoints_dir = Path("checkpoints")

        for model_dir in sorted(checkpoints_dir.glob("T*")):
            history_file = model_dir / "history.json"
            if not history_file.exists():
                continue

            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            if not history.get('epochs'):
                continue

            print(f"\n{model_dir.name}:")

            for epoch_data in history['epochs']:
                epoch = epoch_data['epoch']
                train_loss = epoch_data['train_loss']
                val_loss = epoch_data['val_loss']
                micro_f1 = epoch_data['micro_f1']

                # 基本合理性检查
                checks = []
                if 0 <= train_loss <= 10:
                    checks.append('✅')
                else:
                    checks.append('❌')

                if 0 <= val_loss <= 10:
                    checks.append('✅')
                else:
                    checks.append('❌')

                if 0 <= micro_f1 <= 1:
                    checks.append('✅')
                else:
                    checks.append('❌')

                status = '✅' if all(c == '✅' for c in checks) else '⚠️'
                print(f"  {status} Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, F1={micro_f1:.4f}")


if __name__ == "__main__":
    recomputer = IndependentRecomputer()

    # 验证评估结果
    recomputer.verify_all_results()

    # 验证训练结果
    recomputer.verify_training_results()
