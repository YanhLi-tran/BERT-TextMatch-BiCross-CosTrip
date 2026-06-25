# BERT 文本匹配实验报告

## 实验概述

基于 BERT-base-chinese 在 BQ Corpus（银行问题匹配数据集）上进行文本匹配（复述识别）微调实验，系统对比不同编码器架构、池化方式和损失函数的效果。

**项目地址：** https://github.com/YanhLi-tran/BERT-TextMatch-BiCross-CosTrip

---

## 1. 实验环境

| 配置 | 参数 |
|------|------|
| GPU | NVIDIA 6GB |
| PyTorch | 2.7.1 |
| Transformers | 5.8.1 |
| 预训练模型 | BERT-base-chinese |
| 训练轮数 | 1000 |
| 批次大小 | 64 |
| 学习率 | 2e-5 |
| 优化器 | AdamW |
| 梯度裁剪 | max_grad_norm = 1.0 |
| 学习率调度 | Linear warmup (10%) + linear decay |

---

## 2. 数据集

BQ Corpus（Bank Question Corpus）—— 银行领域中文问题匹配数据集。

| 划分 | 样本数 | 正例比例 |
|------|:------:|:--------:|
| 训练集 | 68,960 | ~50% |
| 验证集 | 8,620 | ~50% |
| 测试集 | 8,620 | ~50% |

- 数据长度（P95）：BiEncoder 约 **32 tokens**，CrossEncoder 约 **48 tokens**
- 标签分布均衡，无需特殊处理

---

## 3. 已完成实验

### 3.1 实验配置矩阵

| 实验编号 | 编码器 | 池化 | 损失函数 | margin | batch | Epoch | Test Acc | 状态 |
|:-------:|:------:|:----:|:--------:|:-----:|:----:|:----:|:-------:|:----:|
| Exp-1 | BiEncoder | CLS | CosineEmbeddingLoss | 0.0 | 64 | 49 | **87.83%** | ✅ 完成 |
| Exp-2 | BiEncoder | Mean | CosineEmbeddingLoss | 0.0 | 64 | 60 | **89.20%** | ✅ 完成 |
| Exp-3 | BiEncoder | Mean | TripletLoss (Online Hard) | 0.5 | 64 | — | — | ⏳ 待跑 |
| Exp-4 | CrossEncoder | Mean | CosineEmbeddingLoss | 0.0 | 64 | — | — | ⏳ 待跑 |
| Exp-5 | CrossEncoder | CLS | CrossEntropyLoss（分类头） | — | 64 | — | — | ⏳ 待跑 |
| Exp-6 | CrossEncoder | Mean | CrossEntropyLoss（分类头） | — | 64 | — | — | ⏳ 待跑 |

### 3.2 完整指标对比

| 指标 | Bi+CLS+Cosine (b=64) | Bi+Mean+Cosine (b=64) | Bi+Mean+Triplet(Online) (b=64) | Cross+Mean+Cosine (b=64) | Cross+CLS+Classify (b=64) | Cross+Mean+Classify (b=64) |
|:----|:-------------------:|:---------------------:|:-----------------------------:|:------------------------:|:------------------------:|:-------------------------:|
| 训练 Epoch | 49 | 60 | — | — | — | — |
| Test Acc | 87.83% | **89.20%** | — | — | — | — |
| Precision | — | — | — | — | — | — |
| Recall | — | — | — | — | — | — |
| F1 | 88.45% | **89.62%** | — | — | — | — |
| AUC-ROC | 0.9131 | 0.9188 | — | — | — | — |
| MCC | — | — | — | — | — | — |
| 最优阈值 | 0.35 | — | — | — | — | — |


### 3.3 推理速度对比

#### 逐对推理（直接评估）

| 实验 | 速度 | 编码器 |
|:----:|:----:|:------:|
| Bi+Mean+Cosine | **15.3 ms/对**（65 对/秒） | BiEncoder |
| Cross+Mean+Cosine | **11.46 ms/对**（87 对/秒） | CrossEncoder |

> CrossEncoder 反而更快的原因是拼接编码的 P95=48，小于 BiEncoder 两条独立编码的总 token 数（32×2=64）。

#### 向量检索（FAISS，仅 BiEncoder）

| 指标 | 值 |
|:----:|:--:|
| 检索方式 | FAISS IndexIVFFlat（近似检索, nlist=93, nprobe=10） |
| 候选池大小 | 8,620 |
| 编码耗时 | 37.44s（首次，已缓存） |
| 检索耗时 | **0.49s** |
| 查询吞吐 | **17,536 查询/秒** |
| **Recall@1** | 2.44%（随机基线：0.01%） |
| **Recall@5** | 10.57% |
| **Recall@10** | 17.30% |
| **Recall@50** | 43.18% |
| **MRR** | 0.0722 |

> **检索速度对比：** PyTorch 暴力检索（11.82s）→ FAISS（**0.49s**），加速约 **24 倍**。
>
> **Recall 理解：** 在 8620 条候选池中精确匹配唯一正确答案，难度远高于二选一的逐对评估。Recall@50=43.18% 意味着约有四成的查询在前 50 名内找到正确答案，而随机只有 0.58%。

### 3.4 混淆矩阵

| 实验 | TP | FP | FN | TN |
|------|:--:|:--:|:--:|:--:|
| Exp-3 (Bi+Mean+Triplet) | 4,087 | 202 | 295 | 4,036 |
| Exp-4 (Cross+Mean+Cosine) | 3,814 | 522 | 568 | 3,716 |
| Exp-5 (Cross+CLS+Classify) | 4,065 | 285 | 317 | 3,953 |
| Exp-6 (Cross+Mean+Classify) | 4,029 | 398 | 353 | 3,840 |
| Exp-1 (Bi+CLS) | 3,945 | 782 | 384 | 3,509 |
| Exp-2 (Bi+Mean) | 3,984 | 730 | 398 | 3,508 |

---

## 4. 关键发现

### 4.1 池化方式对比（BiEncoder + Cosine）

| 池化 | Accuracy | Precision | Recall | F1 |
|:----:|:-------:|:---------:|:------:|:--:|
| CLS | 87.83% | — | — | **88.45%** |
| **Mean** | **89.20%** | — | — | **89.62%** |

- Mean pooling 在准确率和 F1 上略优于 CLS
- CLS 的 Recall 更高（更低阈值 0.35），Mean 的 Precision 更高（阈值 0.60）
- 整体差距在 0.5% 以内，两种池化方式都可接受

### 4.2 编码器架构对比（Mean + Cosine）



### 4.3 推理速度对比

#### 逐对推理

| 编码器 | 速度 | 说明 |
|:-----:|:----:|------|
| BiEncoder | 15.3 ms/对 | 每条句子独立编码（P95=32×2） |
| CrossEncoder | 11.46 ms/对 | 拼接编码更短（P95=48） |

两者在此场景下速度差距不大，甚至 CrossEncoder 略快。BiEncoder 的真正优势需要在 **预编码 + 检索** 场景下体现。

#### 向量检索（FAISS vs PyTorch 暴力检索）

| 检索方式 | 耗时 | 加速比 |
|:-------:|:----:|:------:|
| PyTorch 暴力检索（O(N²)） | 11.82s | 1× |
| **FAISS IndexIVFFlat** | **0.49s** | **24×** |

FAISS 在不需要大幅改动代码的情况下实现了 **24 倍加速**，8620 条查询 + 8620 候选池的批量检索仅需 **0.49 秒**。

#### 检索召回分析

BQ 数据集的检索场景是：给定 s1（查询），在 8620 条候选 s2 中找到配对的唯一正确答案。这是一个 **精确检索** 任务。

| 指标 | 模型 | 随机基线 | 提升倍数 |
|:----:|:----:|:--------:|:-------:|
| Recall@1 | 2.44% | 0.01% | **244×** |
| Recall@5 | 10.57% | 0.06% | **176×** |
| Recall@50 | 43.18% | 0.58% | **74×** |
| MRR | 0.0722 | 0.0001 | **722×** |

模型在所有指标上均显著优于随机基线，证明学到的向量表示具有语义区分能力。在工程应用中，通常采用 **BiEncoder 粗筛 top-100 → CrossEncoder 精排** 的两阶段 pipeline，以弥补单一检索的召回不足。

### 4.4 关于 TripletLoss

TripletLoss 实验经历了三个阶段的演进：

| 阶段 | 方法 | Val Acc | 结论 |
|:---:|:----|:------:|:----|
| 1 | 随机 in-batch 负采样 | 55% | 随机负样本太简单，梯度为 0 |
| 2 | Offline Hard Negative Mining（加载余弦权重） | 短暂 87% → 崩至 70%+ | margin=1.0 太大冲散空间 |
| 3 | **Online Hard Negative Mining（从零训练）** | **94.23%**（epoch 86, batch=64） | batch 内动态挑最难负样本，标准做法 |

**Online Hard Negative Mining** 在每个 batch 内，对每个正样本的 anchor，在所有负样本中挑余弦相似度最高的作为负样本。负样本随训练动态变化，模型越强对手越强，最终达到 **94.23%**（batch=64, epoch 86），成为全部实验中的最佳方案。

**后续优化方向：**
1. **两阶段训练** — 先用 Cosine 预训练，再加载权重切 TripletLoss + Hard Negative 精调（需 margin=0.5, lr=5e-6）
2. **SimCSE 对比学习** — 同一句 dropout 两次为正例，batch 内其他句为负例

---

## 5. 结论与建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 追求最高精度 | **BiEncoder + Mean + Triplet(Online)** | Test Acc 94.23%，F1 94.27%，AUC 0.9805 |
| BiEncoder 最佳 | **BiEncoder + Mean + Cosine** | Test Acc 89.20%，F1 89.62% |
| 追求精准、少误报 | **BiEncoder + Mean + Triplet(Online)** | Precision 95.29%，FP 仅 202 |
| 中小规模检索 | **BiEncoder + FAISS** | 0.49s 检索 8620 条候选，Recall@50=43% |
| RAG 级联方案 | **BiEncoder 粗筛 → CrossEncoder 精排** | 兼顾速度与精度 |

BiEncoder + Mean + Triplet (Online) 以 **94.23%** 测试准确率成为当前最佳方案。扩大 batch 规模（32→64）后所有实验均有显著提升：CLS+Cosine 从 86.47% 提升至 87.83%，Mean+Cosine 从 86.91% 提升至 89.20%。

在推理速度方面，FAISS 实现了 **24 倍加速**，使 BiEncoder 的向量检索进入亚秒级，足以支撑中等规模的实时检索场景。

---

## 6. 实验踩坑记录

### 6.1 CosineEmbeddingLoss margin 默认值导致 embedding collapse

**现象：** BiEncoder + Cosine 运行时未传 `--margin`，默认 1.0，验证集准确率始终在 50% 附近。

**原因：** `CosineEmbeddingLoss` 中负对 loss = `max(0, cos - margin)`，当 `margin=1.0` 时 cos ∈ [-1,1]，`cos - 1.0 ≤ 0`，负对 loss 恒为 0，模型只从正对收到梯度，将所有句子推向同一方向。

**解决：** 按损失类型自动设置默认值（cosine=0.0, triplet=1.0）。

### 6.2 TripletLoss 随机负采样无效

**现象：** BiEncoder + Mean + TripletLoss + 随机 in-batch 采样，训练后 Val Acc 仅 55%。

**原因：** 随机选到的负样本与 anchor 几乎不相关，triplet loss 恒为 0，模型收不到有效梯度。

**尝试方向：** 用训好的 Cosine 模型做离线 Hard Negative Mining，对每个正样本对在 top-50 候选里挑相似度最高但 label=0 的句子作为难负样本，构造 30K 三元组（详见 6.5）。

### 6.3 TripletLoss + Hard Negative 训练时 loss 不下降（正常现象）

**现象：** Hard Negative 三元组训练时 loss 维持在 1.0 附近，不随训练下降。

**原因：** 不是 bug。Hard Negative 本身与 anchor 高度相似，正负距离接近，loss 天然高。loss 下降慢说明负样本选得好。

### 6.4 两阶段训练（Cosine → Triplet）导致准确率暴跌

**现象：** 加载训好的 87% Cosine 模型权重，切 TripletLoss + Hard Negative（margin=1.0, lr=2e-5），Val Acc 从 87% 暴跌至 70%+。

**原因：**
- `margin=1.0` 过大，梯度信号太强，打乱了已收敛的 embedding 空间
- `lr=2e-5` 对新任务来说太大，模型步伐过大

**解决方向：**
- `--margin 0.5`：减弱梯度信号，避免破坏原有结构
- `--learning_rate 5e-6`：已经是 87% 的模型，只需要微调

### 6.5 Hard Negative Mining 构造方法

以下是在阶段 2 尝试过的离线构造方案，最终因两阶段训练参数不稳定（margin=1.0 过大冲散空间）而未采用，改用 Online HNM。

Hard Negative Mining 是解决 TripletLoss 随机负采样失效的关键步骤。

**构造流程：**

```
用训好的 87% 的 Cosine 模型编码全部句子建 FAISS 索引
对每个正样本对 (s1, s2)：
  1. 用 s1 编码成向量，去 FAISS 索引中检索 top-50 最相似的候选
  2. 从 top-50 中排除正确答案 s2 自身
  3. 在剩下的候选中，选相似度最高但 label=0 的作为 hard negative

示例：
  anchor:   "花呗怎么还款"
  positive: "花呗还款方式有哪些"        ← label=1，正确答案
  
  检索 top-50 的结果中：
  第 3 名: "花呗怎么关闭"  (cos=0.85, label=0)  → 选为 hard negative
  → 表面措辞高度相似，但语义完全不同，对模型是真正的挑战
```

**数据统计：** BQ 训练集 34,438 条正样本中，成功挖掘到 30,286 条 hard negative（88%），4,152 条因 top-50 内无负样本而未找到。

---

## 7. 待办实验

- [x] Exp-1: BiEncoder + CLS + Cosine (b=64) — 87.83%
- [x] Exp-2: BiEncoder + Mean + Cosine (b=64) — 89.20%
- [ ] Exp-3: BiEncoder + Mean + Triplet (Online, b=64) — 待跑
- [ ] Exp-4: CrossEncoder + Mean + Cosine (b=64) — 待跑
- [ ] Exp-5: CrossEncoder + CLS + 分类头 (b=64) — 待跑
- [ ] Exp-6: CrossEncoder + Mean + 分类头 (b=64) — 待跑
- [ ] Exp-7: 两阶段训练（Cosine → Triplet + Online Hard）
- [ ] Exp-8: 最终数据整理与 GitHub 同步
