# 实验命令指南

> 本文档包含所有 4 种编码器×损失函数组合的运行命令，以及参数调优示例。
> 所有命令在 **PowerShell** 中执行，先在 `E:\文本匹配任务` 目录下打开 PowerShell。
>
> 进入目录：`cd E:\文本匹配任务`
>
> 如果遇到执行策略错误，先运行：`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

---

## 1. 基础实验：4 种组合全覆盖

每种组合自动输出到 `outputs/{encoder}_{pooling}_{loss}/`，互不覆盖。

### 1.1 BiEncoder + CosineEmbeddingLoss（最常用，推荐首批）

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ BiEncoder: 两条句子分别编码，产出两个独立句子向量                  │
# │ Cosine:    拉近正样本对、推远负样本对的余弦相似度                  │
# │ CLS:       取 [CLS] token 输出作为句子向量（速度快，效果稳定）     │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder biencoder `
    --pooling cls `
    --loss cosine `
    --margin 0.0
```

### 1.2 BiEncoder + CosineEmbeddingLoss + Mean 池化

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Mean: 对句子所有 token 取均值，比 CLS 更充分地利用整句信息         │
# │ 适用于句子语义分散在多个 token 而非集中在 [CLS] 的场景             │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder biencoder `
    --pooling mean `
    --loss cosine `
    --margin 0.0
```

### 1.3 BiEncoder + CosineEmbeddingLoss + Max 池化

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Max: 取每个维度上的最大值，强调句子中最显著的特征                   │
# │ 适合关键词匹配感较强的任务                                         │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder biencoder `
    --pooling max `
    --loss cosine `
    --margin 0.0
```

### 1.4 BiEncoder + TripletLoss

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Triplet: 通过 in-batch negative sampling 构造三元组                │
# │           负样本只从 label=0 的样本中采样，避免污染                 │
# │ margin:  控制正负样本之间的距离边界，越大要求越严格                │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder biencoder `
    --pooling mean `
    --loss triplet `
    --margin 1.0
```

### 1.5 CrossEncoder + CosineEmbeddingLoss

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ CrossEncoder: 拼接 [CLS] s1 [SEP] s2 [SEP] 一起编码               │
# │               利用 cross-attention 捕捉句间交互                     │
# │ 注意: pooling='cls' 时两句子共享同一 [CLS] 向量                   │
# │       pooling='mean'/'max' 时通过 token_type_ids 分开池化          │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder crossencoder `
    --pooling mean `
    --loss cosine `
    --margin 0.0
```

### 1.6 CrossEncoder + CosineEmbeddingLoss（推荐尝鲜）

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ CrossEncoder 利用跨注意力交互，理论上比 BiEncoder 效果更好          │
# │ poolinig='mean' 通过 token_type_ids 分开池化，获得独立句表示        │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder crossencoder `
    --pooling mean `
    --loss cosine `
    --margin 0.0
```

### 1.7 CrossEncoder + TripletLoss（不推荐）

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ ⚠  CrossEncoder + TripletLoss 下，pooling='cls' 时                │
# │    anchor 和 positive 是同一 [CLS] 向量，loss 会退化               │
# │    代码会在启动时打印警告，建议用 pooling='mean'                   │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder crossencoder `
    --pooling mean `
    --loss triplet `
    --margin 1.0
```

---

## 2. 消融实验：对比不同池化方式

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 在相同 encoder + loss 下，对比三种池化方式的效果                    │
# │ 运行三次，结果分别在独立目录下，最后可用对比图分析                  │
# └─────────────────────────────────────────────────────────────────────┘

# CLS
python main.py --encoder biencoder --pooling cls --loss cosine

# Mean
python main.py --encoder biencoder --pooling mean --loss cosine

# Max
python main.py --encoder biencoder --pooling max --loss cosine
```

---

## 3. 超参数调优

### 3.1 调整学习率

```powershell
# BERT 微调常见学习率: 2e-5（默认）, 3e-5, 5e-5
python main.py --encoder biencoder --pooling cls --loss cosine --learning_rate 3e-5
python main.py --encoder biencoder --pooling cls --loss cosine --learning_rate 5e-5
```

### 3.2 调整批次大小

```powershell
# 批次越大训练越稳定，但显存占用更高
# BiEncoder 显存占用低，可试 64；CrossEncoder 建议 16-32
python main.py --encoder biencoder --pooling cls --loss cosine --batch_size 64
python main.py --encoder biencoder --pooling cls --loss cosine --batch_size 128
```

### 3.3 调整 TripletLoss 的 margin

```powershell
# margin 控制正负样本间的距离边界
# 较大的 margin 要求正负样本区分度更高，但训练更难收敛
python main.py --encoder biencoder --pooling mean --loss triplet --margin 0.5
python main.py --encoder biencoder --pooling mean --loss triplet --margin 1.0
python main.py --encoder biencoder --pooling mean --loss triplet --margin 1.5
```

### 3.4 调整验证频率

```powershell
# eval_every 控制最优模型筛选的频率（默认 50）
# 减小该值能更及时捕捉最优模型，但增加训练耗时
python main.py --encoder biencoder --pooling cls --loss cosine --eval_every 20
```

---

## 4. 快速调试运行

```powershell
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 用少量轮次 + 小批次快速验证代码是否可运行                          │
# └─────────────────────────────────────────────────────────────────────┘
python main.py `
    --encoder biencoder `
    --pooling cls `
    --loss cosine `
    --epochs 10 `
    --batch_size 16 `
    --eval_every 5 `
    --save_every 500    # 每 500 step 保存一次 checkpoint
```

---

## 5. 从 checkpoint 恢复训练（续训）

如果训练中途中断，用 `--resume` 指定最新的 checkpoint 即可续训：

```powershell
# 从最新的 checkpoint 恢复
python main.py `
    --encoder crossencoder `
    --pooling mean `
    --loss cosine `
    --resume outputs/crossencoder_mean_cosine/checkpoint_step_22000.pt
```

续训时会自动恢复以下状态：
- 模型权重
- 优化器状态（学习率、动量等）
- 已完成的 epoch 数（从断点继续，不会从头开始）

所有其他参数（`--batch_size`、`--learning_rate` 等）仍以命令行传入为准，
只有 `--resume` 指定的 checkpoint 路径需要与之前一致。

---

## 6. 输出目录结构

每个实验运行后，在 `outputs/{encoder}_{pooling}_{loss}/` 下生成：

```
outputs/
└── biencoder_cls_cosine/          # 运行配置名
    ├── best_model.pt              # 验证集最优模型权重
    ├── checkpoint_step_200.pt     # 每 200 step (batch) 的 checkpoint
    ├── checkpoint_step_400.pt
    ├── checkpoint_step_600.pt
    ├── checkpoint_step_800.pt
    ├── final_model.pt             # 最后一轮模型
    ├── summary.txt                # 训练摘要（含最优阈值、测试准确率）
    ├── loss_curve.png             # Loss 曲线
    ├── accuracy_curve.png         # Accuracy 曲线
    ├── combined_metrics.png       # 合并图
    ├── eda/                       # 数据探索分析（全局共享）
    │   ├── eda_label_distribution.png
    │   └── eda_length_distribution.png
    └── eval/                      # 独立评估输出（运行 evaluate.py 后生成）
        ├── eval_summary.json
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── pr_curve.png
        └── similarity_distribution.png
```

---

## 6. 对比全部组合

运行以下 6 条命令，然后人工对比各 `summary.txt` 中的 `Test Acc`：

| 命令 | 输出目录 |
|------|----------|
| `python main.py --encoder biencoder --pooling cls --loss cosine` | `biencoder_cls_cosine/` |
| `python main.py --encoder biencoder --pooling mean --loss cosine` | `biencoder_mean_cosine/` |
| `python main.py --encoder biencoder --pooling mean --loss triplet --margin 1.0` | `biencoder_mean_triplet/` |
| `python main.py --encoder crossencoder --pooling mean --loss cosine` | `crossencoder_mean_cosine/` |
| `python main.py --encoder crossencoder --pooling cls --loss cosine` | `crossencoder_cls_cosine/` |
| `python main.py --encoder crossencoder --pooling mean --loss triplet --margin 1.0` | `crossencoder_mean_triplet/` |

---

## 7. 独立评估已训练模型（`evaluate.py`）

训练完成后，用 `evaluate.py` 对任意数据集进行独立评估，输出混淆矩阵、ROC 曲线、PR 曲线、相似度分布等可视化结果。

### 7.1 在验证集上评估（自动搜索最优阈值）

```powershell
python evaluate.py `
    --checkpoint outputs/biencoder_cls_cosine/best_model.pt
```

输出：`outputs/biencoder_cls_cosine/eval/` 目录下生成所有图表。

### 7.2 在测试集上评估

```powershell
# 用验证集找到的最优阈值来评估测试集（推荐方式）
python evaluate.py `
    --checkpoint outputs/biencoder_cls_cosine/best_model.pt `
    --data_split test

# 或者手动指定阈值
python evaluate.py `
    --checkpoint outputs/biencoder_cls_cosine/best_model.pt `
    --data_split test `
    --threshold 0.45
```

### 7.3 在自定义数据文件上评估

```powershell
python evaluate.py `
    --checkpoint outputs/biencoder_cls_cosine/best_model.pt `
    --data_path data/bq_corpus/test.jsonl
```

### 7.4 输出内容

```
outputs/biencoder_cls_cosine/eval/
├── eval_summary.json            # 各项指标的 JSON 摘要
├── confusion_matrix.png         # 混淆矩阵（含 TP/FP/FN/TN 数量和占比）
├── roc_curve.png                # ROC 曲线（含 AUC）
├── pr_curve.png                 # Precision-Recall 曲线（含 AP）
└── similarity_distribution.png  # 正负样本相似度分布对比直方图
```

### 7.5 命令行参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | **必填** | 模型权重路径 |
| `--data_dir` | `data/bq_corpus` | 数据集目录 |
| `--data_split` | `validation` | 数据划分：train / validation / test |
| `--data_path` | None | 自定义文件路径，优先级高于 data_dir+split |
| `--output_dir` | checkpoint所在目录/eval | 图表输出目录 |
| `--threshold` | None | 余弦阈值，None 则自动搜索最优值 |
| `--batch_size` | 64 | 评估批次大小 |

---

## 7.6 BiEncoder vs CrossEncoder 逐对推理速度对比

`evaluate.py` 会自动计时，跑完两个模型的评估后直接比对输出中的 **ms/对** 即可。

### 对比命令

```powershell
# 1. BiEncoder + Mean 推理速度
python evaluate.py `
    --checkpoint outputs/biencoder_mean_cosine/best_model.pt `
    --data_split test

# 2. CrossEncoder + Mean 推理速度
python evaluate.py `
    --checkpoint outputs/crossencoder_mean_cosine/best_model.pt `
    --data_split test
```

### 输出示例

```
[推理速度] 总样本: 8620 对 | 总耗时: 12.34s | 698 对/秒 | 1.43 ms/对
```

### 说明

- 这是 **逐对推理** 速度（每对 (s1, s2) 都要过一遍模型），不是实际线上场景
- BiEncoder 真正的速度优势在于 **预编码 + 向量检索**（候选句离线编码好，查询时只编码 1 条），这个计时不体现那部分优势
- 速度结果会保存在 `eval_summary.json` 的 `infer_speed` 字段中，方便后续对比

### 7.7 预编码 + 向量检索模式（仅 BiEncoder）

模拟真实线上场景：先将所有候选句子预编码存库，查询时只编码 1 条 query，通过 FAISS 向量检索找到最相似的候选。

```powershell
# 首次：编码 8620 对句子并缓存
python evaluate.py --checkpoint outputs/biencoder_mean_cosine/best_model.pt --data_split test --mode retrieval

# 后续：直接加载缓存，秒级出结果
python evaluate.py --checkpoint outputs/biencoder_mean_cosine/best_model.pt --data_split test --mode retrieval
```

输出内容：

```
=============================================
  Retrieval Evaluation Results
=============================================
  Total queries:     8620 (4382 positives)
  Candidate pool:    8620
  Encoding time:     37.44s
  Search time:       0.49s (FAISS)
  Queries/sec:       17536
  ─────────────────────────
  Recall@1            2.44%
  Recall@5            10.57%
  Recall@10           17.30%
  Recall@50           43.18%
  MRR:                0.0722
=============================================
```

输出目录：`outputs/biencoder_mean_cosine/eval_retrieval/`
- `recall_curve.png` — Recall@K 曲线
- `retrieval_summary.json` — 检索指标 JSON
- `embeddings.pt` — 向量缓存（二次加载跳过编码）

> **注意：** 检索模式只在 BiEncoder 下有意义（CrossEncoder 不支持独立编码）。
> **理解 Recall 数字：** 在 8620 条候选池中精确匹配唯一正确答案难度极高（随机猜测 Recall@1=0.01%），模型 Recall@1=2.44% 已是随机的 **244 倍**。实际工程中会配合 CrossEncoder rerank 使用。

---

## 8. 全部训练参数默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--encoder` | `biencoder` | 编码器类型：biencoder / crossencoder |
| `--pooling` | `cls` | 句子向量化方式：cls / mean / max |
| `--loss` | `cosine` | 损失函数：cosine / triplet |
| `--margin` | `cosine=0.0 / triplet=1.0` | 损失函数的 margin（自动根据 loss 类型设置） |
| `--epochs` | `1000` | 总训练轮数 |
| `--batch_size` | `32` | 训练批次大小 |
| `--eval_batch_size` | `64` | 评估批次大小 |
| `--learning_rate` | `2e-5` | 学习率 |
| `--weight_decay` | `0.01` | 权重衰减 |
| `--warmup_ratio` | `0.1` | 学习率预热比例 |
| `--max_grad_norm` | `1.0` | 梯度裁剪范数（0=关闭） |
| `--save_every` | `200` | 每多少步(batch)保存一次 checkpoint |
| `--eval_every` | `1` | 每多少轮在验证集上评估一次 |
| `--vis_interval` | `10` | 每多少轮记录可视化指标 |
| `--max_length` | `None` | 最大序列长度（None 自动用 p95） |
| `--device` | `cuda` | 训练设备 |
| `--seed` | `42` | 随机种子 |
| `--num_workers` | `0` | 数据加载线程数（Windows 建议 0） |
| `--data_dir` | `data/bq_corpus` | 数据集路径 |
| `--bert_model` | `E:\models\bert-base-chinese` | BERT 预训练模型路径 |
| `--output_dir` | `outputs/` | 训练输出目录 |

---

## 9. 性能建议

- **首次运行** 从 `biencoder + cls + cosine` 开始，参数最简、显存最低、收敛稳定
- **追求效果** 尝试 `crossencoder + mean + cosine`，cross-attention 能更好捕捉句间语义
- **TripletLoss** 在有大量正样本对时效果较好（BQ 正负均衡，cosine 可能更稳定）
- **显存不足** 减小 `--batch_size`，或改用 `--encoder biencoder`
