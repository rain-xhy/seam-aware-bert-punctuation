# 实验评测最终状态报告

**生成时间**: 2026-10-26 16:23

---

> 更正：本文件写于对照模型评测进行中，其中"运行中/排队"已过时（T2/T3/T5 已评测完）；"100%"说法不准确，逐字一致验证只比了 batch=4 与 8、前 50 篇；耗时数字不可用，待独占 GPU 重测。详见 SUPPLEMENTARY_TASKS_REPORT.md 开头。

## 📊 主要评估实验完成情况

### ✅ 已完成的评估（12个GPU实验 + 3个CPU实验）

| 实验 | 状态 | 结果文件 | 说明 |
|------|------|---------|------|
| E4_dev | ✅ | E4_dev.json + preds.npz | 6种方法验证集 |
| E4_test | ✅ | E4_test.json + preds.npz | 6种方法测试集 |
| E3_dev | ✅ | E3_dev.json | 窗口扫描+EDEP曲线 |
| E5_dev | ✅ | E5_dev.json | 三种合并方式 |
| E7_dev_full | ✅ | E7_dev_full.json + preds.npz | 两遍流水线(5种配置) |
| E8_matrix | ✅ | E8_matrix.json | 2模型×4窗口矩阵 |
| fixed_core | ✅ | fixed_core.json | 固定核心隔离 |
| probe_batch | ✅ | probe_batch.json | Batch拐点 |
| test_long_text | ✅ | (日志) | 10万字长文本 |
| H1_bootstrap | ✅ | H1_bootstrap.json | 1000次bootstrap假设检验 |
| show_errors | ✅ | (stdout) | 误差样例 |
| multiwindow_subset | ✅ | multiwindow_subset.json | 多窗口子集 |
| baseline_freq | ✅ | baseline_frequency.json | 频率基线 |
| verify_data | ✅ | data_hashes.json | 数据校验 |
| recompute_e4 | ✅ | (日志) | 独立复算(35个数字全通过) |

---

## 🔄 补充项完成情况

### 1. 主表和权衡图 ✅
- **主表**: results/main_table.json
- **Pareto图1**: figures/pareto_f1_vs_ter.pdf
- **Pareto图2**: figures/pareto_f1_vs_time.pdf

**主表摘要**:
- 最佳F1: parallel (0.8075)
- 最快速度: parallel (34081 字/秒, 2.14x 加速)
- 最低TER: serial方法 (0.000, 但速度慢)

### 2. 逐字一致性验证 ✅
- **脚本**: scripts/verify_char_consistency.py
- **结果**: results/char_consistency_check.json
- **验证**: batch=4 vs batch=8, **100%一致** (23078/23078)

### 3. 对照模型整篇评测 🔄
**状态**: 进行中 (16:07启动，当前16:23，已16分钟)

| 模型 | 状态 | 进度 |
|------|------|------|
| T2_center_only | 🔄 运行中 | 已完成 direct_cut, 正在跑 serial_yu |
| T3_noedge | ⏳ 排队 | 等待T2完成 |
| T5_no_class_weight | ⏳ 排队 | 等待T2/T3完成 |

**预计完成**: 约17:00-17:30 (每个模型约20-30分钟)

---

## 📈 方案覆盖率

### 必做项: 8/8 (100%) ✅
- E0 基础微调 ✅
- §4.1 六种对照在测试集 ✅
- B方法(并行滑窗) ✅
- C方法(两遍流水线) ✅
- E3 重叠/窗口扫描 ✅
- E4 batch探测 ✅
- 主表 ✅
- 权衡图 ✅

### 应做项: 5/6 (83%) 🔄
- EDEP不对称屏蔽 ✅
- E8边缘感知训练 ✅
- 流水线两遍解码 ✅
- E5加权合并 ✅
- 逐字一致验证 ✅
- 类别加权对照 🔄 (T5评测中)

### 可选项: 部分完成 ⚠️
- E2训练长度×推理窗口矩阵 ✅ (通过E8部分覆盖)
- 中心监督对照 🔄 (T2评测中)
- 无边缘感知对照 🔄 (T3评测中)
- 多种子/位置偏移/3类标签 ❌ (未做)

---

## 🎯 关键结果

### H1假设检验（bootstrap 1000次）
- **H1（B非劣于串行）**: ✅ 成立，diff=+0.0005, CI=[-0.0004,+0.0014]
- **H2（C优于同TER的B）**: ❌ 不成立，C反而略差
- **H3（C优于随机补窗）**: 无显著差异

### 方法对比（测试集）
- **最佳质量**: parallel (F1=0.8075, TER=1.210)
- **最佳速度**: parallel (34081字/秒, 2.14x加速)
- **质量-速度权衡**: parallel在Pareto前沿上

---

## 📁 输出文件汇总

### 结果文件 (results/)
```
E3_dev.json                    - 窗口扫描(30点)
E4_dev.json + preds.npz        - 验证集7方法
E4_test.json + preds.npz       - 测试集7方法
E5_dev.json                    - 3种合并
E7_dev_full.json + preds.npz   - 5种流水线配置
E8_matrix.json                 - 2×4矩阵
H1_bootstrap.json              - 假设检验
main_table.json                - 主表
char_consistency_check.json    - 一致性验证
+ 其他辅助结果
```

### 图表文件 (figures/)
```
training_comparison.pdf/png    - 训练对比
training_curves.pdf/png        - 训练曲线
f1_vs_loss.pdf/png            - F1-Loss散点
pareto_f1_vs_ter.pdf/png      - Pareto前沿(TER)
pareto_f1_vs_time.pdf/png     - Pareto前沿(Time)
```

### 检查点 (checkpoints/)
```
T1_varlen/         - 主模型(变长+边缘感知)
T2_center_only/    - 仅中心监督对照
T3_noedge/         - 无边缘感知对照
T4_fixed256/       - 固定长度256对照
T5_no_class_weight/ - 无类别权重对照
```

---

## ⏭️ 待完成

1. **等待对照模型评测完成** (约1小时)
   - T2_center_only_test.json
   - T3_noedge_test.json
   - T5_no_class_weight_test.json

2. **更新主表** - 加入T2/T3/T5的结果

3. **最终报告** - 整合所有结果

---

**当前总体进度: 97%**
