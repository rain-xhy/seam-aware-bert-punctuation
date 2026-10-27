# 对照模型编号对应表

方案 §14 的编号与实际训练的目录不一致。报告统一按"实际目录"写，下表是对应关系。

| 方案 §14 编号 | 方案含义 | 实际目录 | 实际含义 | 状态 |
|---|---|---|---|---|
| T1 | 变长+边缘感知主模型 | checkpoints/T1_varlen | 变长 32–400，edge_aware_prob 默认 0.3，不加权 | ✅ |
| T2 | 固定长度 256 | checkpoints/T4_fixed256 | 固定 256 | ✅ |
| T3 | 不做边缘感知 | checkpoints/T3_noedge | edge_aware_prob=0 | ✅ |
| T4 | 类别加权 | checkpoints/T4_classweight | 约 24000 步崩溃，无检查点 | ❌ 缺失 |
| T5 | 只监督中心区 | checkpoints/T2_center_only | --center_only | ✅ |
| （方案无） | — | checkpoints/T5_no_class_weight | 与 T1 同命令（均无 --class_weights），不是加权对照 | 仅作重复训练参考 |
| （空目录） | — | checkpoints/T2_fixed256 | 空 | — |

注意：实际目录名 T2_center_only 与方案的 T2（固定 256）含义相反；T4_fixed256 与方案的 T4（类别加权）含义不同。报告里写对照时必须带含义，不能只写编号。

T1 与 T5_no_class_weight 的"同配置"判断依据是启动命令（run_full_training.sh 与队列命令），没有核实 T1 训练时 train.py 的版本是否与现在一致。
