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
| 批次大小 | 32 |
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

| 实验编号 | 编码器 | 池化 | 损失函数 | margin | 状态 |
|:-------:|:------:|:----:|:--------:|:-----:|:----:|
| Exp-1 | BiEncoder | CLS | CosineEmbeddingLoss | 0.0 | ✅ 完成 |
| Exp-2 | BiEncoder | Mean | CosineEmbeddingLoss | 0.0 | ✅ 完成 |
| Exp-3 | CrossEncoder | Mean | CosineEmbeddingLoss | 0.0 | ✅ 完成 |
| Exp-4 | BiEncoder | Mean | TripletLoss | 1.0 | ⏳ 待优化 |

### 3.2 完整指标对比

| 指标 | Exp-1 (Bi+CLS+Cosine) | Exp-2 (Bi+Mean+Cosine) | Exp-3 (Cross+Mean+Cosine) |
|------|:--------------------:|:----------------------:|:------------------------:|
| **Test Accuracy** | 86.47% | 86.91% | **87.35%** |
| **Precision** | 83.46% | 84.51% | **87.96%** |
| **Recall** | **91.13%** | 90.92% | 87.04% |
| **F1 Score** | 87.12% | **87.60%** | 87.50% |
| **AUC-ROC** | 0.9122 | 0.9036 | **0.9313** |
| **MCC** | 0.7325 | — | **0.7471** |
| **最优阈值** | 0.35 | 0.60 | 0.60 |

### 3.3 推理速度对比

#### 逐对推理（直接评估）

| 实验 | 速度 | 编码器 |
|:----:|:----:|:------:|
| Exp-2 (Bi+Mean+Cosine) | **15.3 ms/对**（65 对/秒） | BiEncoder |
| Exp-3 (Cross+Mean+Cosine) | **11.46 ms/对**（87 对/秒） | CrossEncoder |

> CrossEncoder 反而更快的原因是拼接编码的 P95=48，小于 BiEncoder 两条独立编码的总 token 数（32×2=64）。

#### 向量检索（FAISS，仅 BiEncoder）

| 指标 | 值 |
|:----:|:--:|
| 检索方式 | FAISS IndexFlatIP（暴力检索） |
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
| Exp-1 (Bi+CLS) | 3,945 | 782 | 384 | 3,509 |
| Exp-2 (Bi+Mean) | 3,984 | 730 | 398 | 3,508 |
| Exp-3 (Cross+Mean) | 3,814 | **522** | 568 | **3,716** |

---

## 4. 关键发现

### 4.1 池化方式对比（BiEncoder + Cosine）

| 池化 | Accuracy | Precision | Recall | F1 |
|:----:|:-------:|:---------:|:------:|:--:|
| CLS | 86.47% | 83.46% | **91.13%** | 87.12% |
| **Mean** | **86.91%** | **84.51%** | 90.92% | **87.60%** |

- Mean pooling 在准确率和 F1 上略优于 CLS
- CLS 的 Recall 更高（更低阈值 0.35），Mean 的 Precision 更高（阈值 0.60）
- 整体差距在 0.5% 以内，两种池化方式都可接受

### 4.2 编码器架构对比（Mean + Cosine）

| 编码器 | Accuracy | Precision | Recall | F1 | AUC | 速度 |
|:-----:|:-------:|:---------:|:-----:|:--:|:---:|:----:|
| **BiEncoder** | 86.91% | 84.51% | **90.92%** | **87.60%** | 0.9036 | 15.3 ms |
| **CrossEncoder** | **87.35%** | **87.96%** | 87.04% | 87.50% | **0.9313** | **11.46 ms** |

- **Accuracy & Precision：** CrossEncoder 领先（跨注意力充分利用句间交互）
- **Recall：** BiEncoder 领先（阈值切分更有利于覆盖正例）
- **AUC：** CrossEncoder 显著领先（0.9313 vs 0.9036），排序能力更强
- **速度：** CrossEncoder 反而更快（拼接编码总 token 数更少）

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
| **FAISS IndexFlatIP** | **0.49s** | **24×** |

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

当前 `BiEncoder + Mean + TripletLoss` 仅达到 55% Val Acc（随机猜测水平），主要原因是 **in-batch 随机负采样**选到的负样本太简单，大部分 triplet loss 为 0，模型无法有效学习。

**待尝试的优化方向：**
1. **Hard Negative Mining** — 用已训好的 Cosine 模型编码全部句子，对每个正样本对找出相似度最高但 label=0 的句子作为难负样本
2. **两阶段训练** — 先用 Cosine 预训练，再加载权重切 TripletLoss + Hard Negative 精调

---

## 5. 结论与建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 追求 F1 平衡 | **BiEncoder + Mean + Cosine** | F1 87.60%，Recall 90.92% |
| 追求精准、少误报 | **CrossEncoder + Mean + Cosine** | Precision 87.96%，FP 最少（522） |
| 追求排序质量 | **CrossEncoder + Mean + Cosine** | AUC 0.9313，显著领先 |
| 大规模检索 | **BiEncoder + FAISS** | 0.49s 检索 8620 条候选，Recall@50=43% |
| 高精度检索 | **BiEncoder 粗筛 → CrossEncoder 精排** | 两阶段 pipeline，兼顾速度与精度 |
| 方案成熟度 | **BiEncoder + Mean + Cosine** | 速度快、效果好、实现简单 |

总体来看，BiEncoder 和 CrossEncoder 在 F1 上差距极小（0.1%），**两者都是有效的文本匹配方案**。CrossEncoder 在排序能力（AUC）上有明显优势，适合作为 reranker；BiEncoder 在召回率和实现简洁性上更优，适合作为 base retriever。

在推理速度方面，FAISS 实现了 **24 倍加速**，使 BiEncoder 的向量检索进入亚秒级，足以支撑中等规模的实时检索场景。

---

## 6. 待办实验

- [ ] Exp-4: BiEncoder + Mean + TripletLoss（Hard Negative Mining 版本）
- [ ] Exp-5: CrossEncoder + CLS + Cosine（补全 CrossEncoder 的池化对比）
- [ ] Exp-6: 两阶段训练（Cosine → Triplet 迁移）
