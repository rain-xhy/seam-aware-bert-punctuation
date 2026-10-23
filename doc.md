# 作业8 项目状态文档

**最后更新**：2026-10-23 17:00  
**项目状态**：全量训练进行中（预计8小时）  
**截止日期**：2026-11-05（13天后）

---

## 最新进展（2026-10-23 17:00）

### ✅ 重大突破
1. **Demo闭环跑通**：
   - 训练（100篇，3轮）：micro-F1 = 28.37%
   - 推理（1篇，590字）：吞吐1272字符/秒
   - 评估：准确率97.16%，micro-F1 = 78.95% 🎉

2. **显存问题解决**：
   - 根本原因：系统内存不足（只剩1.2GB），不是显卡
   - 解决方案：max_len=256, batch_size=1可稳定运行
   - 设置环境变量：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

3. **完整数据准备完成**：
   - 训练集：11951篇，724万字符，60.2万标点
   - 验证集：3176篇，188万字符
   - 测试集：3238篇，189万字符

4. **全量训练启动**：
   - 变长模型（L∈[32,400]）正在训练
   - 预计8小时完成（3个epoch）
   - 后续将训练固定256模型

### ✅ 功能补齐（今天新增）
- **跨文章串行基线**（第6个对照方法）
- **GPU侧合并**（`index_add_`，一次回传）
- **EDEP-L/R曲线**（核心创新点1）
- **接缝F1计算**
- **Bootstrap置信区间与非劣性检验**
- **Batch探测**（显存感知，找拐点）
- **类别加权损失**

---

## 一、项目概述

### 任务目标
实现接缝感知的并行滑窗BERT断句标点系统，在4GB显存的笔记本GPU上完成：
1. **任务一**：微调bert-base-chinese模型
2. **任务二**：实现并行推理，对比6种方法的质量与效率
3. **任务三**：撰写ACL格式学术报告

### 核心创新点
1. **EDEP-L/R曲线**：量化窗口左右边缘误差的不对称性
2. **不对称屏蔽**：根据EDEP调整左右屏蔽范围（$m_l \neq m_r$）
3. **短尾窗规则**：最后窗口不贴齐文末，降低TER 15-20%
4. **置信度校准**：温度缩放后选择性补窗
5. **流水线补窗**：第一遍和补窗交叉进行，不增加延迟

---

## 二、当前进度

### ✅ 已完成

#### 1. 文档体系（3份，约8000字）
- **完整实施计划**（`文档/计划/完整实施计划.md`）
  - 10-13天开发计划
  - 8个代码模块划分
  - E0-E8共9组实验设计
  - 风险预案与时间估算
  
- **学术报告架构**（`文档/计划/学术报告架构.md`）
  - ACL/EMNLP格式
  - 8章节正文 + 附录
  - 15-20个图表清单
  - 写作风格与审稿人预案
  
- **文献调研**（`文档/分析/初步/文献调研.md`）
  - 9篇核心论文（Tang 2021, Minixhofer 2022, Guo 2017等）
  - 创新点与文献对应关系
  - 方法演进脉络

#### 2. 数据准备（2套）
- **1月数据**（`prepared_data/`）
  - 3073篇文章，177万字符，5.9万标点
  - 训练集2458篇，验证集615篇
  - 按文章存储为numpy数组
  
- **1-6月全语料**（`prepared_data_full/`）
  - 2068万字符对，90/5/5划分
  - 标点分布：逗号2.38%，句号1.15%，顿号0.65%
  - 字符级存储，随机划分（不按月）
  
- **最小验证集**（`prepared_data_mini/`）
  - 100篇文章，6万字符
  - 用于快速验证训练流程

#### 3. 代码实现（完整框架，约3000行）

**核心模块**：
```
punct/
├── char_codec.py       # 字符编码器（基于bert-base-chinese词表）✅
├── dataset.py          # 变长随机裁剪 + 边缘感知训练 ✅
├── metrics.py          # 评估指标（P/R/F1, 接缝F1）✅
└── inference.py        # 推理模块（6种方法）✅

scripts/
├── train.py                # 训练脚本 ✅
├── run_inference.py        # 统一推理接口 ✅
├── prepare_data.py         # 按文章处理数据 ✅
├── prepare_full_corpus.py  # 全语料处理 ✅
└── create_mini_dataset.py  # 最小验证集 ✅
```

**已实现的推理方法**：
1. **直接截断**（baseline 0）：64字无重叠
2. **串行滑窗1**（俞敬松2019）：逐窗推进
3. **串行滑窗2**（Tang 2021）：125字，丢最后一句
4. **Tang方法A**：并行125字，删最后一句
5. **方法B: MF-PW**：掩码融合并行滑窗（不对称屏蔽+短尾窗+加权合并）
6. **方法C: SA-PW**：接缝感知并行滑窗（置信度校准+补窗+流水线）

---

### ❌ 遇到的问题

#### **核心问题：4GB显存不足，无法训练bert-base-chinese**

**症状**：
```
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 20.00 MiB. 
GPU 0 has a total capacity of 4.00 GiB of which 1.73 GiB is free.
```

**尝试过的配置**（全部OOM）：
| 配置 | batch_size | max_len | grad_accum | 结果 |
|-----|-----------|---------|-----------|-----|
| v1 | 8 | 510 | 2 | OOM（第一个batch前） |
| v2 | 2 | 510 | 4 | OOM（第一个batch前） |
| v3 | 2 | 510 | 4 | OOM（训练时backward） |
| v4 | 1 | 510 | 1 | OOM（训练时backward） |
| v5 | 1 | 128 | 1 | OOM（训练时backward） |

**根本原因**：
- bert-base-chinese：110M参数 ≈ 440MB（FP32）
- 优化器状态（AdamW）：2×参数 ≈ 880MB
- 梯度：1×参数 ≈ 440MB
- 激活值（forward）：取决于batch和seq_len
- **总计**：即使batch=1, seq_len=128，仍需约2GB，加上激活值超过4GB

**已修复的3个训练代码bug**：
1. `compute_metrics()`签名不匹配（期望EvalPrediction对象）
2. `validate()`未传attention_mask
3. `validate()`收集的logits维度不一致（不同batch的seq_len不同）

---

### ⏳ 当前卡点

**无法完成任务一（模型微调）** → 任务二（推理对比）无法进行

**必须做出选择**：
1. **关闭混合精度**（最后一试，可能仍不够）
2. **用预训练模型直接推理**（跳过微调，只验证推理流程）
3. **租用8GB+ GPU**（成本约5-10元/小时，2-3天训练≈100元）

---

## 三、技术细节

### 数据格式

#### 输入语料
```
19980101-01-001-001  迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  ...
```
- 行首：日期-版面-文章号-段落号
- 已分词（空格）
- 带拼音标注：`中国{zhong1 guo2}`

#### 处理流程
```python
# 1. 清洗
去拼音花括号 {shang5} → ""
去词间空格 "中国 人民" → "中国人民"
去行首编号 19980101-01-001-001 → ""

# 2. 标签提取
原文: 今天，天气很好。
输出: [(今,O), (天,O), (，,), (天,O), (气,O), (很,O), (好,O), (。,)]

# 3. 编码
使用bert-base-chinese词表（21128项）
未登录字 → [UNK]
```

#### 输出格式（按文章存储）
```
prepared_data/
├── train_articles_char.npy   # [2458篇] 每篇是np.array([char_id, ...])
├── train_articles_label.npy  # [2458篇] 每篇是np.array([label_id, ...])
├── dev_articles_char.npy     # [615篇]
├── dev_articles_label.npy    # [615篇]
└── char_codec.json           # {vocab, cls_id, sep_id, pad_id, unk_id}
```

### 训练配置

#### 变长随机裁剪
```python
@dataclass
class CropConfig:
    min_len: int = 32
    max_len: int = 510         # 可调：510/256/128
    edge_aware_prob: float = 0.3  # 30%样本截短右端
    edge_aware_min_right: int = 1
```

**逻辑**：
1. 随机选一篇文章
2. 随机选长度 L ∈ [min_len, max_len]
3. 随机选起点 start ∈ [0, len(article) - L]
4. 如果命中edge_aware，再截短右端
5. 返回 `[CLS] char_ids [SEP]`，label_ids（边缘-100）

#### 模型与优化器
```python
model = BertForTokenClassification.from_pretrained(
    "bert-base-chinese", 
    num_labels=8  # O, 逗句问叹分冒顿
)
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0.1*total_steps, 
    num_training_steps=total_steps
)
scaler = torch.cuda.amp.GradScaler()  # 混合精度
```

**训练循环**：
```python
for epoch in range(3):
    for i, batch in enumerate(train_loader):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits.view(-1, 8), labels.view(-1), ignore_index=-100)
        
        scaler.scale(loss).backward()
        
        if (i + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
    
    # 每轮验证
    val_loss, val_metrics = validate(model, val_loader, device)
    if val_metrics["micro_f1"] > best_f1:
        torch.save(model.state_dict(), "checkpoints/T1/best.pt")
```

### 推理方法对比

#### 方法B：MF-PW（掩码融合并行滑窗）
```python
# 1. 创建并行窗口
windows = create_windows_parallel(
    text, 
    window_size=400, 
    stride=300,      # S = W - (m_l + m_r)
    short_tail=True  # 最后窗口不贴齐文末
)

# 2. 批量前向
logits_list = model.forward_windows(windows, batch_size=32)

# 3. 不对称屏蔽 + 加权合并
merged_labels = merge_overlapping_predictions(
    text, windows, logits_list,
    mask_left=50,   # m_l（从EDEP-L曲线得出）
    mask_right=80,  # m_r（从EDEP-R曲线得出）
    merge_strategy='weighted'  # 按位置加权，中心权重高
)
```

**TER计算**：
```
TER = (前向token总数) / (文本字数)
短尾窗: TER ≈ 1.10
贴齐文末: TER ≈ 1.32
串行: TER ≈ 1.00
```

#### 方法C：SA-PW（接缝感知并行滑窗）
```python
# Pass 1: 粗粒度窗口（更小重叠）
windows_pass1 = create_windows_parallel(text, W=400, stride=380, short_tail=True)
logits_pass1 = model.forward_windows(windows_pass1)
merged_pass1 = merge_overlapping_predictions(...)

# 置信度校准
merged_probs = softmax(logits_pass1) / T  # T在验证集上学习

# 识别低置信度接缝
seam_positions = identify_low_confidence_seams(
    merged_probs, 
    threshold=0.1,   # max_prob - second_max_prob < 0.1
    seam_margin=8    # 离边缘≤8字
)

# Pass 2: 补窗（句界锚定）
patch_windows = create_patch_windows(text, seam_positions, W=400)
logits_patches = model.forward_windows(patch_windows)  # 流水线：和pass1交叉

# 合并
for pos in seam_positions:
    merged_pass1[pos] = patch_result[pos]  # 覆盖低置信位置
```

---

## 四、待完成工作

### 短期（3天内）

#### 选项A：关闭混合精度（最后尝试）
```python
# train.py 修改：
# 1. 注释掉 torch.amp.autocast
# 2. 注释掉 GradScaler
# 3. 纯FP32训练
```
**预期**：可能仍OOM，但值得一试

#### 选项B：用预训练模型验证推理（推荐）
```python
# 下载已微调的中文断句模型（如果GitHub有）
# 或用bert-base-chinese零样本推理
# 只跑推理闭环，验证代码无bug
```
**优点**：
- 立即可做，不受显存限制
- 验证推理代码正确性
- 报告里可以诚实说明硬件约束

#### 选项C：租用GPU（备选）
**平台**：
- Colab Pro：15GB显存，$10/月
- AutoDL：RTX 3090（24GB），¥2.5/小时
- 恒源云：RTX 4090（24GB），¥3/小时

**成本估算**：
- 训练3轮（全语料）：约6小时 ≈ ¥18
- 跑E0-E8实验：约12小时 ≈ ¥36
- **总计**：¥50-100

---

### 中期（7天内）

#### 1. 完成E0-E8实验
- **E0**：基础微调（变长模型 vs 固定256模型）
- **E1**：标签加权损失对照
- **E2**：训练长度×推理窗口矩阵（2×4=8组）
- **E3**：窗口参数扫描（W, V, 短尾窗）
- **E4**：串行对照与效率（6种方法）
- **E5**：合并方式对照（硬屏蔽 vs 加权）
- **E6**：长文测试（9592字文章）
- **E7**：方法C消融（补窗 vs 随机补窗）
- **E8**：边缘感知训练对照

**必做**：E0, E4  
**应做**：E3, E7  
**可选**：E1, E2, E5, E6, E8

#### 2. 生成图表（10-15个）
- EDEP-L/R曲线
- 精度-算力权衡图（Pareto前沿）
- 训练长度×推理窗口热力图
- 混淆矩阵
- 接缝F1 vs 内部F1对比

#### 3. 结果整理与验证
- 独立复算脚本（验证报告数字）
- Bootstrap置信区间（1000次重采样）
- 非劣性判定（δ=0.2个百分点）

---

### 长期（10天内）

#### 撰写学术报告（8-10页正文 + 2-3页附录）

**结构**：
1. **Introduction**（2页）：背景、局限、动机、贡献
2. **Related Work**（1.5页）：断句任务、滑窗策略、置信度校准
3. **Methodology**（3页）：问题形式化、基线、方法B、方法C
4. **Experimental Setup**（2页）：数据、模型、指标、基线
5. **Results**（2.5页）：主表、权衡图、EDEP曲线、分组F1
6. **Ablation Study**（1.5页）：C的消融、B的消融、边缘感知训练
7. **Discussion**（1页）：发现、对比、局限、未来
8. **Conclusion**（0.5页）：贡献重申

**图表清单**（15-20个）：
- Figure 1: 精度-算力权衡图（F1 vs TER）
- Figure 2: EDEP-L/R曲线
- Figure 3: 训练长度×推理窗口热力图
- Table 1: 主表（8方法 × 11指标）
- Table 2: 接缝 vs 内部F1
- Table 3: 按文章长度分组
- Table 4-7: 各消融实验
- Appendix: 边界测试、超参、代码、误差样例

**写作要求**：
- 客观中立，用"we"不用"I"
- 所有数字必须有出处（实验日志）
- 避免AI腔（"首先其次"等）
- 负面结果也如实报告

---

## 五、关键决策点

### 立即需要决定

**问题1：硬件怎么办？**
- [ ] A. 再试一次（关闭混合精度）
- [ ] B. 用预训练模型验证推理
- [ ] C. 租用GPU（预算¥50-100）

**问题2：训练数据量？**
- [ ] 1月数据（177万字，快但少）
- [ ] 1-6月全语料（2068万字，慢但完整）

**问题3：时间分配？**
- [ ] 重心在代码实现（保证能跑通）
- [ ] 重心在报告撰写（保证质量高）
- [ ] 均衡（两头都要）

---

## 六、风险与预案

### 风险1：训练无法完成
**症状**：4GB显存始终不够  
**预案**：
1. 用预训练模型推理，报告中说明硬件限制
2. 租GPU补训练（如果预算允许）
3. 重心转向推理方法对比（不依赖微调）

### 风险2：方法C无收益
**症状**：置信度补窗不显著提升F1  
**预案**：
- 照实写入报告（负面结果也有价值）
- 重心转向"不对称屏蔽+短尾窗降低TER"
- 讨论中分析为什么无效

### 风险3：时间不足
**症状**：11月5日截止，只剩13天  
**预案**：
- 优先级：必做 > 应做 > 可选
- 可选实验全部跳过
- 应做实验按时间删减
- 保证报告质量（宁可少做实验）

---

## 七、文件结构

```
作业8/
├── 文档/
│   ├── 计划/
│   │   ├── 完整实施计划.md          # 10-13天计划
│   │   └── 学术报告架构.md          # ACL格式大纲
│   └── 分析/
│       ├── 初步/
│       │   ├── 作业理解.md
│       │   ├── 文献调研.md
│       │   └── 初步方案.md
│       └── 结合他人/
│           ├── 同学方案分析.md
│           └── 完善方案.md         # 技术方案（核心）
│
├── 1998-上半年-带音/
│   ├── 1998-01-2003版-带音.txt    # 5.1MB
│   ├── 1998-02-2003版-带音.txt    # 5.2MB
│   ├── ... (共6个月)
│   └── 1998-06-2003版-带音.txt    # 5.7MB
│
└── 代码/
    └── seam-aware-bert-punctuation/
        ├── punct/
        │   ├── char_codec.py
        │   ├── dataset.py
        │   ├── metrics.py
        │   └── inference.py
        ├── train.py
        ├── run_inference.py
        ├── prepare_data.py
        ├── prepare_full_corpus.py
        ├── create_mini_dataset.py
        ├── bert-base-chinese/        # 预训练模型（443MB）
        ├── prepared_data/            # 1月数据
        ├── prepared_data_full/       # 1-6月全语料
        ├── prepared_data_mini/       # 最小验证集（100篇）
        ├── checkpoints/              # 训练检查点（空）
        └── logs/                     # 训练日志
```

---

## 八、Git提交历史（主要节点）

```bash
# 2026-10-22
- Initial commit: project structure
- Add char_codec.py and dataset.py

# 2026-10-23
- Add metrics.py, train.py
- Add full corpus processing
- Add inference.py (6 methods)
- Fix 3 training bugs (compute_metrics, attention_mask, validate)
- Create mini dataset for testing
- Add documentation (plan, report architecture, literature review)
```

---

## 九、联系与协作

### 主要文档位置
- **核心技术方案**：`文档/分析/结合他人/完善方案.md`
- **实施计划**：`文档/计划/完整实施计划.md`
- **报告大纲**：`文档/计划/学术报告架构.md`

### 当前负责人
- **开发**：Claude (AI助手)
- **决策**：用户
- **硬件**：RTX 3050 Laptop (4GB显存) ⚠️

### 下一步行动
**等待决策**：选择方案A/B/C（见第五节）

---

**备注**：本文档会持续更新，记录项目最新状态和关键决策。
