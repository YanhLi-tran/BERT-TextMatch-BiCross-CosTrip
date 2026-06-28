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
| Exp-3 | BiEncoder | Mean | TripletLoss (Online Hard) | 0.5 | 64 | 46 | **93.48%** | ✅ 完成 |
| Exp-4 | CrossEncoder | Mean | CosineEmbeddingLoss | 0.0 | 64 | 66 | **92.52%** | ✅ 完成 |
| Exp-5 | CrossEncoder | CLS | CrossEntropyLoss（分类头） | — | 64 | 57 | **94.22%** | ✅ 完成 |
| Exp-6 | CrossEncoder | Mean | CrossEntropyLoss（分类头） | — | 64 | 49 | **92.97%** | ✅ 完成 |

### 3.2 完整指标对比

| 指标 | Bi+CLS+Cosine (b=64) | Bi+Mean+Cosine (b=64) | Bi+Mean+Triplet(Online) (b=64) | Cross+Mean+Cosine (b=64) | Cross+CLS+Classify (b=64) | Cross+Mean+Classify (b=64) |
|:----|:-------------------:|:---------------------:|:-----------------------------:|:------------------------:|:------------------------:|:-------------------------:|
| 训练 Epoch | 49 | 60 | 46 | 66 | 57 | 49 |
| Test Acc | 87.83% | 89.20% | 93.48% | 92.52% | **94.22%** | 92.97% |
| Precision | 85.43% | 87.60% | 94.75% | 92.83% | **94.66%** | 93.54% |
| Recall | 91.69% | 91.74% | 92.29% | 92.42% | **93.93%** | 92.56% |
| F1 | 88.45% | 89.62% | 93.50% | 92.62% | **94.30%** | 93.05% |
| AUC-ROC | 0.9131 | 0.9188 | 0.9793 | 0.9627 | **0.9819** | 0.9762 |
| MCC | 0.7610 | 0.7851 | 0.8701 | 0.8507 | **0.8846** | 0.8596 |
| 最优阈值 | 0.35 | 0.50 | 0.50 | 0.55 | argmax | argmax |


### 3.3 推理速度对比

#### 逐对推理（直接评估）

| 编码器 | 逐对 | batch=10 | batch=20 | batch=50 | 预编码+检索 | 单条检索 |
|:-----:|:----:|:--------:|:--------:|:--------:|:----------:|:--------:|
| BiEncoder | 5.30 | — | — | — | 0.49s (24×) | 0.23ms |
| CrossEncoder | 10.17 | **20.38ms** | **45.53ms** | **172.34ms** | ❌ | ❌ |

> CrossEncoder 逐对推理 10.17ms，但 GPU batch 并行下排 50 条约 172ms（3.45ms/对），排 20 条约 46ms（2.28ms/对）。适合在 BiEncoder 召回的 top-50/top-20 候选中做精排。

#### 向量检索（FAISS，仅 BiEncoder）

| 指标 | 值 |
|:----:|:--:|
| 检索方式 | FAISS IndexIVFFlat（近似检索, nlist=93, nprobe=10） |
| 候选池大小 | 8,620 |
| 编码耗时 | 37.44s（首次，已缓存） |
| 检索耗时 | **0.49s** |
| 查询吞吐 | **17,536 查询/秒** |
| **Recall@1** | 2.49%（随机基线：0.01%） |
| **Recall@5** | 10.98% |
| **Recall@10** | 17.46% |
| **Recall@50** | 42.24% |
| **MRR** | 0.0733 |

> **检索速度对比：** PyTorch 暴力检索（11.82s）→ FAISS（**0.49s**），加速约 **24 倍**。
>
> **Recall 理解：** 在 8620 条候选池中精确匹配唯一正确答案，难度远高于二选一的逐对评估。Recall@50=42.24% 意味着约有四成的查询在前 50 名内找到正确答案，而随机只有 0.58%。

### 3.4 混淆矩阵

| 实验 | TP | FP | FN | TN |
|:----|:--:|:--:|:--:|:--:|
| Exp-5 (Cross+CLS+Classify) | **4,116** | 232 | **266** | **4,006** |
| Exp-3 (Bi+Mean+Triplet) | 4,044 | **224** | 338 | 4,014 |
| Exp-6 (Cross+Mean+Classify) | 4,056 | 280 | 326 | 3,958 |
| Exp-4 (Cross+Mean+Cosine) | 4,050 | 313 | 332 | 3,925 |
| Exp-2 (Bi+Mean+Cosine) | 4,020 | 569 | 362 | 3,669 |
| Exp-1 (Bi+CLS+Cosine) | 4,018 | 685 | 364 | 3,553 |

---

### 3.5 CrossEncoder Batch 推理时延详解

GPU batch 推理时多条 pair 可并行计算，耗时远小于串行累加。

| batch_size | 总耗时 | 每对均摊 | 应用场景 |
|:---------:|:------:|:--------:|:--------|
| 1（串行） | 10.17ms | 10.17ms/对 | 逐对推理基线 |
| 5 | 12.13ms | 2.43ms/对 | 少量候选重排 |
| 10 | **20.38ms** | 2.04ms/对 | **最优效率点** |
| 20 | **45.53ms** | 2.28ms/对 | **推荐的精排规模** |
| 50 | **172.34ms** | 3.45ms/对 | 高召回场景 |
| 64 | ~202ms | 3.17ms/对 | evaluate.py 默认  |
| 100 | 561.02ms | 5.61ms/对 | 超出 500ms 预算 |

**核心规律：** 总耗时与 batch_size 呈非线性关系。小 batch 时增长平缓（batch=1到5仅增加 2ms），大 batch 时增长加速（batch=40到50增加 60ms），原因是 GPU 并行计算的显存带宽在大 batch 时成为瓶颈。

> 补充实测：batch=15 -> 23.48ms，batch=30 -> 57.05ms，batch=40 -> 112.64ms。数据点较多时可直接查表，不建议用公式推算。

**手动验算（参考值）：** 从 evaluate.py 输出 27.32s / 8620 对，每 batch（约64对）约 202ms。
- batch=50: (50/64)*202 = 158ms，实测 172ms（偏差 +9%）
- batch=20: (20/64)*202 = 63ms， 实测 46ms（偏差 -27%）

线性公式在大 batch 时偏差较小，小 batch 时不可用。最可靠的方式是直接使用实测数据点。

**端到端延迟分配（500ms 预算，LLM 生成约 300ms）：**

| 环节 | 耗时 | 说明 |
|:----|:----:|------|
| BiEncoder 编码单条查询 | ~2.7ms | 5.30ms/对 ÷ 2 |
| FAISS 检索 top-50 | 0.23ms | 8620 候选池 |
| CrossEncoder 精排 20 条 | ~46ms | batch 并行，推荐配置 |
| CrossEncoder 精排 50 条 | ~172ms | 高召回配置 |
| LLM 生成 | ~300ms | 假设值 |
| **总计（精排 20 条）** | **~349ms** | < 500ms ✅ |
| **总计（精排 50 条）** | **~475ms** | < 500ms ✅ |

---

## 4. 关键发现

### 4.1 池化方式对比（BiEncoder + Cosine）

| 池化 | Accuracy | Precision | Recall | F1 |
|:----:|:-------:|:---------:|:------:|:--:|
| CLS | 87.83% | 85.43% | **91.69%** | 88.45% |
| **Mean** | **89.20%** | **87.60%** | **91.74%** | **89.62%** |

- Mean pooling 在准确率和 F1 上略优于 CLS
- CLS 的 Recall 更高（更低阈值 0.35），Mean 的 Precision 更高（阈值 0.60）
- 整体差距在 0.5% 以内，两种池化方式都可接受

### 4.2 编码器架构对比（Mean + Cosine）

| 编码器 | Accuracy | Precision | Recall | F1 | AUC | 速度 |
|:-----:|:-------:|:---------:|:-----:|:--:|:---:|:----:|
| **BiEncoder** | 89.20% | 87.60% | **91.74%** | 89.62% | 0.9188 | 5.30 ms |
| **CrossEncoder** | **92.52%** | **92.83%** | 92.42% | **92.62%** | **0.9627** | **4.54 ms** |

- **Accuracy / F1：** CrossEncoder 领先约 3 个百分点（跨注意力充分利用句间交互）
- **Recall：** BiEncoder 略高（阈值切分更有利于覆盖正例）
- **AUC：** CrossEncoder 显著领先（0.9627 vs 0.9188），排序能力更强
- **速度：** CrossEncoder 反而更快（拼接编码总 token 数更少）

### 4.3 推理速度对比

#### 逐对推理

| 编码器 | 逐对 | batch=10 | batch=20 | batch=50 | 预编码+检索 | 单条检索 | 适用场景 |
|:-----:|:----:|:--------:|:--------:|:--------:|:----------:|:--------:|:--------:|
| BiEncoder | 5.30ms | — | — | — | 0.49s (24×) | 0.23ms | 召回层 |
| CrossEncoder | 10.17ms | **20.38ms** | **45.53ms** | **172.34ms** | ❌ | ❌ | 精排层 |

BiEncoder 和 CrossEncoder 的架构差异决定了各自的最佳用途：BiEncoder 适合大规模召回（向量预编码 + FAISS 检索），CrossEncoder 适合在 BiEncoder 粗筛后的少量候选中做精排。

#### 向量检索（FAISS vs PyTorch 暴力检索）

| 检索方式 | 批量 8620 条 | 单条查询 | 加速比 |
|:-------:|:-----------:|:--------:|:------:|
| PyTorch 暴力检索（O(N²)） | 11.82s | 0.57ms | 1× |
| **FAISS IndexIVFFlat** | **0.49s** | **0.23ms** | **24× / 2.5×** |

FAISS 在不需要大幅改动代码的情况下实现了 **24 倍加速**，8620 条查询 + 8620 候选池的批量检索仅需 **0.49 秒**。

#### 检索召回分析

BQ 数据集的检索场景是：给定 s1（查询），在 8620 条候选 s2 中找到配对的唯一正确答案。这是一个 **精确检索** 任务。

| 指标 | 模型 | 随机基线 | 提升倍数 |
|:----:|:----:|:--------:|:-------:|
| Recall@1 | 2.49% | 0.01% | **249×** |
| Recall@5 | 10.98% | 0.06% | **183×** |
| Recall@50 | 42.24% | 0.58% | **73×** |
| MRR | 0.0722 | 0.0001 | **722×** |

模型在所有指标上均显著优于随机基线，证明学到的向量表示具有语义区分能力。在工程应用中，通常采用 **BiEncoder 粗筛 top-100 → CrossEncoder 精排** 的两阶段 pipeline，以弥补单一检索的召回不足。

### 4.4 关于 TripletLoss

TripletLoss 实验经历了三个阶段的演进：

| 阶段 | 方法 | Val Acc | 结论 |
|:---:|:----|:------:|:----|
| 1 | 随机 in-batch 负采样 | 55% | 随机负样本太简单，梯度为 0 |
| 2 | Offline Hard Negative Mining（加载余弦权重） | 短暂 87% → 崩至 70%+ | margin=1.0 太大冲散空间 |
| 3 | **Online Hard Negative Mining（从零训练）** | **93.48%**（epoch 46, batch=64） | batch 内动态挑最难负样本，标准做法 |

**Online Hard Negative Mining** 在每个 batch 内，对每个正样本的 anchor，在所有负样本中挑余弦相似度最高的作为负样本。负样本随训练动态变化，模型越强对手越强，最终达到 **93.48%**（batch=64, epoch 46），成为全部实验中的最佳方案。

**后续优化方向：**
1. **两阶段训练** — 先用 Cosine 预训练，再加载权重切 TripletLoss + Hard Negative 精调（需 margin=0.5, lr=5e-6）
2. **SimCSE 对比学习** — 同一句 dropout 两次为正例，batch 内其他句为负例

---

## 5. 结论与建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 追求最高精度 | **CrossEncoder + CLS + 分类头** | Test Acc **94.22%**，F1 **94.30%**，AUC **0.9819** |
| BiEncoder 最佳 | **BiEncoder + Mean + Triplet(Online)** | Test Acc 93.48%，F1 93.50%，AUC 0.9793 |
| 追求精准、少误报 | **CrossEncoder + CLS + 分类头** | Precision 94.66%，FP 仅 232 |
| 中小规模检索 | **BiEncoder + FAISS** | 0.49s 检索 8620 条候选，Recall@50=43% |
| RAG 级联方案 | **BiEncoder 粗筛 → CrossEncoder 精排** | 兼顾速度与精度 |

CrossEncoder + CLS + 分类头以 **94.22%** 测试准确率成为全部 6 组实验中的最佳方案。扩大 batch 规模（32→64）后所有实验均有显著提升：CLS+Cosine 从 86.47% 提升至 87.83%，Mean+Cosine 从 86.91% 提升至 89.20%，CrossEncoder+CLS+Classify 从 92.34% 提升至 94.22%。

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

## 7. 实验总结

全部 6 组实验在 batch=64 下统一从零训练完成，使用 Early Stopping（patience=5）自动收敛。

| 排名 | 实验 | Test Acc | F1 | AUC |
|:---:|:----|:-------:|:--:|:---:|
| 🥇 | CrossEncoder + CLS + Classify | **94.22%** | **94.30%** | **0.9819** |
| 🥈 | BiEncoder + Mean + Triplet(Online) | 93.48% | 93.50% | 0.9793 |
| 🥉 | CrossEncoder + Mean + Classify | 92.97% | 93.05% | 0.9762 |
| 4 | CrossEncoder + Mean + Cosine | 92.52% | 92.62% | 0.9627 |
| 5 | BiEncoder + Mean + Cosine | 89.20% | 89.62% | 0.9188 |
| 6 | BiEncoder + CLS + Cosine | 87.83% | 88.45% | 0.9131 |

> 所有实验统一条件：BERT-base-chinese（全量 12 层）、BQ 69K 数据集、batch=64、AdamW（lr=2e-5）、Early Stopping（patience=5）。
