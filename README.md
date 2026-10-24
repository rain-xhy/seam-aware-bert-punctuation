# 作业8：接缝感知的并行滑窗BERT断句标点

基于BERT的中文标点预测系统，实现了串行滑窗、并行滑窗、接缝感知二次解码等多种方法的对比实验。

## 环境要求

```bash
pip install -r requirements.txt
```

- Python 3.8+
- PyTorch 2.4.1
- Transformers 4.46.3
- CUDA (可选，推荐4GB+显存)

## 数据准备

```bash
# 下载bert-base-chinese到项目根目录
# 准备数据（按月划分，修正词表）
python prepare_complete_data.py
```

生成的数据在 `prepared_data_complete/` 目录。

## 训练

```bash
python train.py --data_dir prepared_data_complete --bert_dir bert-base-chinese \
    --out_dir checkpoints/T1_varlen --epochs 3 --batch_size 1 --grad_accum 8 \
    --max_len 400 --seed 42 --save_every 4000
# 中断后用同一条命令重启，会从 resume.pt 续训；对照模型的开关见 run_controls3.sh
```

## 实验

### E3: 窗口/重叠/尾部扫描

```bash
python -m scripts.run_sweep --ckpt checkpoints/T1_varlen/best.pt \
    --split dev --windows 128 256 400 510 --overlaps 0 50 100 200 \
    --tails short aligned --edep --out results/E3_dev.json
```

### E4: 方法对比与计时

```bash
python -m scripts.run_methods --ckpt checkpoints/T1_varlen/best.pt \
    --split dev --methods direct_cut serial_teacher serial_cross parallel \
    --out results/E4_dev.json --save_preds results/E4_dev_preds.npz

python -m scripts.run_e4 --ckpt checkpoints/T1_varlen/best.pt \
    --split dev --limit 300 --repeats 5 --out results/E4_timing.json
```

### 独立验证

```bash
python -m scripts.recompute --result results/E4_dev.json \
    --preds results/E4_dev_preds.npz --data_dir prepared_data_complete
```

### Bootstrap显著性检验

```bash
python -m scripts.bootstrap_compare --result results/E4_dev.json \
    --preds results/E4_dev_preds.npz --method_a parallel \
    --method_b serial_teacher --n_bootstrap 10000
```

### 误差分析

```bash
python -m scripts.show_errors --result results/E4_dev.json \
    --preds results/E4_dev_preds.npz --method parallel --limit 20
```

### 长文本测试

```bash
python -m scripts.test_long_text --ckpt checkpoints/T1_varlen/best.pt \
    --length 100000
```

### 数据完整性验证

```bash
python -m scripts.verify_data_hash prepared_data_complete \
    --save data_hashes.json
```

## 代码结构

```
seam-aware-bert-punctuation/
├── punct/                    # 核心模块
│   ├── engine.py            # 批量并行解码引擎
│   ├── runner.py            # 方法包装器与评分
│   ├── two_pass.py          # 二次接缝补窗
│   ├── edep_scan.py         # EDEP-L/R边缘误差扫描
│   ├── metrics.py           # 评估指标
│   ├── train.py             # 训练逻辑
│   ├── char_codec.py        # 字符编解码
│   ├── bootstrap.py         # 统计检验
│   └── ...
├── scripts/                  # 实验脚本
│   ├── run_methods.py       # 方法对比
│   ├── run_sweep.py         # E3窗口扫描
│   ├── run_e4.py            # E4计时实验
│   ├── recompute.py         # 独立验证
│   ├── show_errors.py       # 误差样例
│   ├── bootstrap_compare.py # 显著性检验
│   ├── test_long_text.py    # 长文本测试
│   └── ...
├── tests/                    # 单元测试 (164 tests)
├── prepare_complete_data.py  # 数据准备
└── requirements.txt
```

## 测试

```bash
python -m unittest discover -s tests
```

## 主要创新

1. **接缝感知并行解码**：EDEP-L/R曲线自动确定掩码宽度
2. **二次补窗**：低置信度接缝位置用更长窗口重解码
3. **变长训练**：32-510字符随机裁剪，提升泛化
4. **完整对比**：串行、Tang方法、并行在同一数据上比较
5. **工程优化**：GPU合并、OOM重试、交错计时、批量一致性验证

## 引用

基于Tang et al. (2021) "接缝感知的滑窗方法"改进实现。
