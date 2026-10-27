# 评估实验清单（按方案 §9）- 更新版

## 必做

| 编号 | 实验名称 | 对应脚本/命令 | 状态 |
|------|---------|-------------|------|
| E0 | 基础微调 | 已完成（T1_varlen） | ✅ |
| E4_dev | 6种方法验证集对比 | scripts.run_methods | ✅ 已跑通 |
| E4_test | 6种方法测试集对比 | scripts.run_methods | ✅ 已跑通 |
| E3 | 重叠/窗口/尾窗扫描+EDEP | scripts.run_sweep --edep | ✅ 找到脚本 |
| E4_batch | batch 探测 | scripts.probe_batch | ✅ 冒烟通过 |
| B方法 | 并行滑窗（已含在E4） | - | ✅ |
| C方法 | 补窗与对照（两遍流水线） | scripts.run_e7 | ✅ 找到脚本 |
| 主表 | 汇总表格 | 需编写 | 📝 待做 |
| 权衡图 | Pareto前沿 | visualize_results.py | 📝 待扩展 |

## 应做

| 编号 | 实验名称 | 对应脚本/命令 | 状态 |
|------|---------|-------------|------|
| EDEP | 不对称屏蔽（L/R） | run_sweep --edep | ✅ 找到（含在E3） |
| E8 | 边缘感知训练 | scripts.run_matrix | ✅ 冒烟通过 |
| E5 | 加权合并 | scripts.run_e5_merge | ✅ 冒烟通过 |
| 流水线 | 两遍解码 | run_e7 | ✅ 找到（就是C方法） |
| 逐字一致 | 验证脚本 | ❓ 未找到专门脚本 |
| 类别加权对照 | T4_classweight vs T5_no_class_weight | ⚠️ 需补充评测 |

## 可选

| 编号 | 实验名称 | 对应脚本/命令 | 状态 |
|------|---------|-------------|------|
| E2 | 训练长度×推理窗口矩阵 | scripts.run_matrix | ✅ 冒烟通过 |
| 中心监督对照 | T2_center_only 整篇解码 | 需补充 | 📝 待做 |

## 发现

1. **E3 = run_sweep.py**：窗口/重叠/尾窗扫描，`--edep` 选项生成 EDEP-L/R 曲线
2. **C方法 = run_e7.py**：流水线两遍解码，包含校准、随机对照、同TER对照
3. **E7_dev_full 实际上应该用 run_e7.py**，而不是 run_calibration.py
4. **逐字一致验证**：未找到专门脚本，可能是人工检查或内嵌在其他脚本中

## 需要补充到队列

1. **E3_dev**: run_sweep --edep（窗口扫描+EDEP曲线）
2. **E7_dev 替换**：用 run_e7.py 替换 run_calibration.py
3. **对照模型评测**：T2/T3/T5 用 run_methods 或 run_matrix 评测

