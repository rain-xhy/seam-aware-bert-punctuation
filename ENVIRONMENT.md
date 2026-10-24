# 实验环境配置

## 硬件环境

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **显存**: 4096 MiB
- **CPU**: (自动检测)
- **内存**: (自动检测)

## 软件环境

### Python环境
- **Python版本**: 3.11.x
- **操作系统**: Windows 11

### 深度学习框架
- **PyTorch**: 2.4.1
- **CUDA**: 12.1
- **cuDNN**: (与CUDA版本配套)

### 核心依赖
```
torch==2.4.1
transformers==4.46.3
numpy>=1.24.0,<2.0.0
```

### 可选依赖（可视化与分析）
```
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.5.0
scipy>=1.9.0
```

## 训练配置

### 超参数
- **学习率**: 3e-5
- **优化器**: AdamW (weight_decay=0.01)
- **学习率调度**: 10% warmup + 线性衰减
- **Batch size**: micro-batch 8, 梯度累积2步 (等效16)
- **Epoch数**: 3
- **混合精度**: BF16 autocast
- **梯度裁剪**: max_norm=1.0

### 数据配置
- **变长训练**: 32-510字符随机裁剪
- **边缘感知**: 距边缘<16字符权重0.5
- **数据集**: 人民日报1998年1-6月
  - 训练集: 1-4月
  - 验证集: 5月
  - 测试集: 6月

## 推理配置

### 并行解码
- **默认窗口**: W=400
- **默认重叠**: V=100
- **尾部策略**: short
- **Batch size**: 8 (可自动探测)
- **掩码**: EDEP自动确定

### 二次补窗
- **置信度阈值**: 0.8
- **补窗窗口**: 400
- **随机对照比例**: 0.5

## 复现说明

1. **环境安装**:
   ```bash
   pip install -r requirements.txt
   ```

2. **数据准备**:
   ```bash
   python prepare_complete_data.py
   ```

3. **训练**:
   ```bash
   python train.py --data_dir prepared_data_complete --bert_dir bert-base-chinese \
       --out_dir checkpoints/T1_varlen --epochs 3 --batch_size 1 --grad_accum 8 --max_len 400
   ```

4. **实验**:
   ```bash
   ./run_all_experiments.ps1  # Windows
   bash run_all_experiments.sh  # Linux
   ```

## 版本检测脚本

运行以下脚本自动检测当前环境版本：

```python
import sys
import torch
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

## 注意事项

1. **显存限制**: 4GB显存下，batch_size=8是经过探测的推荐值
2. **OOM重试**: 代码会自动检测OOM并减半batch重试
3. **BF16支持**: 需要Ampere架构及以上GPU（RTX 30系列+）
4. **数据路径**: 修改`prepare_complete_data.py`中的路径以适配本地环境

---

**生成日期**: 2026年10月  
**最后更新**: 实验完成后填入实际版本号
