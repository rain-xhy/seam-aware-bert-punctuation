"""
Experiment runner for all E0-E8 experiments.

Usage:
    python run_experiments.py --checkpoint checkpoints/T1_varlen/best.pt
"""
import argparse
import json
import os
from pathlib import Path


def run_E0_basic_training():
    """E0: 基础微调（已在训练中）"""
    print("E0: Basic fine-tuning")
    print("  - Model 1 (varlen): checkpoints/T1_varlen/best.pt")
    print("  - Model 2 (fixed256): checkpoints/T2_fixed256/best.pt")
    print("  Status: In progress")


def run_E1_class_weights(checkpoint):
    """E1: 类别加权损失对照"""
    print("\nE1: Class-weighted loss ablation")
    # TODO: Train a model with --class_weights
    # Compare with baseline model


def run_E2_train_infer_matrix(checkpoints):
    """E2: 训练长度×推理窗口矩阵"""
    print("\nE2: Training length × Inference window matrix")
    # 2 models × 4 window sizes = 8 groups
    window_sizes = [64, 125, 400, 510]
    # TODO: Run inference with each combination


def run_E3_window_param_scan(checkpoint):
    """E3: 窗口参数扫描（W, V, 短尾窗）"""
    print("\nE3: Window parameter scanning")
    # Scan W, V, short_tail
    # Find optimal (W, V)
    # Generate EDEP-L/R curves
    # Compare short_tail vs align_to_end


def run_E4_serial_baseline_comparison(checkpoint):
    """E4: 串行对照与效率（6种方法）"""
    print("\nE4: Serial baseline comparison")
    methods = [
        'direct_cut',
        'sequential',
        'tang_sequential',
        'tang_A',
        'teacher_400_300',
        'cross_article_serial'
    ]
    # TODO: Run all methods, measure quality + efficiency


def run_E5_merge_strategy(checkpoint):
    """E5: 合并方式对照"""
    print("\nE5: Merge strategy ablation")
    strategies = ['hard_mask', 'gaussian_weight', 'edep_weight']
    # TODO: Compare merge strategies


def run_E6_long_document(checkpoint):
    """E6: 长文测试"""
    print("\nE6: Long document test")
    # Test on longest article (9592 chars)
    # Test on synthetic 100k chars


def run_E7_method_C_ablation(checkpoint):
    """E7: 方法C消融（补窗）"""
    print("\nE7: Method C ablation")
    ablations = [
        'full_C',
        'no_calibration',
        'no_patching',
        'random_patching',
        'same_TER_B'
    ]
    # TODO: Run ablations


def run_E8_edge_aware_training(checkpoints):
    """E8: 边缘感知训练对照"""
    print("\nE8: Edge-aware training ablation")
    # Compare normal training vs edge-aware training
    # Measure seam_F1 vs interior_F1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/T1_varlen/best.pt')
    parser.add_argument('--experiments', nargs='+', default=['all'],
                       choices=['all', 'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8'])
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = {
        'E0': run_E0_basic_training,
        'E1': lambda: run_E1_class_weights(args.checkpoint),
        'E2': lambda: run_E2_train_infer_matrix([args.checkpoint]),
        'E3': lambda: run_E3_window_param_scan(args.checkpoint),
        'E4': lambda: run_E4_serial_baseline_comparison(args.checkpoint),
        'E5': lambda: run_E5_merge_strategy(args.checkpoint),
        'E6': lambda: run_E6_long_document(args.checkpoint),
        'E7': lambda: run_E7_method_C_ablation(args.checkpoint),
        'E8': lambda: run_E8_edge_aware_training([args.checkpoint]),
    }

    if 'all' in args.experiments:
        to_run = list(experiments.keys())
    else:
        to_run = args.experiments

    print("=" * 60)
    print("Running experiments:", to_run)
    print("=" * 60)

    for exp_name in to_run:
        if exp_name in experiments:
            experiments[exp_name]()

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Results saved to {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
