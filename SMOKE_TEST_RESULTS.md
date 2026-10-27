# 冒烟测试结果汇总

## 测试时间
2026-10-26 14:00-14:40

## 测试结果

| # | 实验名称 | 脚本 | 状态 | 备注 |
|---|---------|------|------|------|
| 1 | E5_dev | run_e5_merge | ✅ 通过 | 三种合并方式都有结果 |
| 2 | E7_dev_full | run_calibration | ✅ 通过 | 温度校准T=1.37 |
| 3 | E8_matrix | run_matrix | ✅ 通过 | 2模型×2窗口矩阵 |
| 4 | probe_batch | probe_batch | ✅ 通过 | 推荐batch=8 |
| 5 | baseline_frequency | run_baseline | ✅ 通过 | F1=0.0329 |
| 6 | verify_data | verify_data_hash | ✅ 通过 | 6个文件哈希 |
| 7 | recompute | recompute | ✅ 通过 | 35个数字验证通过 |
| 8 | show_errors | show_errors | ✅ 通过 | 输出到stdout |
| 9 | multiwindow_subset | eval_multiwindow_subset | ✅ 通过 | 依赖E4结果 |
| 10 | H1_bootstrap | run_hypotheses | ⚠️ 跳过 | 需要完整E7，无法小规模测试 |
| 11 | fixed_core | run_fixed_core | ✅ 通过 | 用--contexts多值 |
| 12 | test_long_text | test_long_text | ✅ 已通过 | (之前旧队列已验证) |

## 已修复的参数错误

1. **E7_dev_full**: `--split dev` → `--cal_split dev --test_split dev`
2. **E8_matrix**: `--ckpts` → `--models name=path`，添加 `--windows`
3. **H1_bootstrap**: 改为 `--serial_npz`、`--b_npz`、`--e7_npz`、`--n_boot`
4. **show_errors**: 删除不存在的 `--out`

## 待确认的缺失实验

根据方案§9，以下实验在当前队列中没有对应脚本：

| 实验 | 要求 | 状态 |
|------|------|------|
| E3 | 重叠/窗口/尾窗扫描 | ❓ 未找到对应脚本 |
| EDEP曲线 | 不对称屏蔽L/R | ❓ 可能在E5中，需确认 |
| 流水线两遍解码 | 方案应做项 | ❓ 未找到对应脚本 |
| 逐字一致验证 | 效率工程验证 | ❓ 未找到对应脚本 |
| 对照模型整篇解码 | T2/T3/T5完整评测 | ❓ 需补充到队列 |

## 下一步

1. ✅ 所有可测试的脚本已通过冒烟测试
2. 📝 确认E3、EDEP、流水线、逐字一致是否有对应实现
3. 📝 补充对照模型(T2/T3/T5)的整篇解码评测
4. 🚀 准备重启完整评估队列
