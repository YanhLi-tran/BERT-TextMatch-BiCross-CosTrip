"""
数据工具模块 —— 数据加载、长度分析(p95)、数据集构建
支持 BiEncoder 和 CrossEncoder 两种数据格式
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_jsonl(file_path):
    """
    加载 JSONL 格式的数据文件
    返回: list[dict]，每个 dict 包含 sentence1, sentence2, label
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_p95_length(data, tokenizer, encoder_type="biencoder", max_samples=5000):
    """
    计算训练集序列长度的 p95（第95百分位数）
    根据 encoder_type 采用不同的计算方式：
      - "biencoder": 分别计算两条单句长度，取每条句子各自的 p95，
                     再用这个 p95 作为单条句子的 max_length
      - "crossencoder": 拼接 s1 和 s2 计算总长度 [CLS] s1 [SEP] s2 [SEP]
                        再加 3 个特殊 token
    
    参数:
        data: 数据集，每项含 sentence1 和 sentence2
        tokenizer: 用于分词的 tokenizer
        encoder_type: "biencoder" 或 "crossencoder"
        max_samples: 最多计算的样本数（采样加速）
    """
    # 如果数据量太大则随机采样
    if len(data) > max_samples:
        sampled = random.sample(data, max_samples)
    else:
        sampled = data

    lengths = []
    for item in sampled:
        s1_tokens = tokenizer.tokenize(item["sentence1"])
        s2_tokens = tokenizer.tokenize(item["sentence2"])

        if encoder_type == "biencoder":
            # BiEncoder: 每条句子独立编码 [CLS] s [SEP]，各需 2 个特殊 token
            # 记录两条句子中较长的那个长度
            len1 = len(s1_tokens) + 2  # [CLS] + [SEP]
            len2 = len(s2_tokens) + 2
            lengths.append(max(len1, len2))
        else:
            # CrossEncoder: [CLS] s1 [SEP] s2 [SEP]，共 3 个特殊 token
            lengths.append(len(s1_tokens) + len(s2_tokens) + 3)

    lengths = sorted(lengths)
    p95_index = int(len(lengths) * 0.95)
    p95_value = lengths[p95_index]

    print(f"[数据长度分析 - {encoder_type}] 共采样 {len(sampled)} 条样本")
    print(f"  - 最小长度: {lengths[0]}")
    print(f"  - 最大长度: {lengths[-1]}")
    print(f"  - P50 (中位数): {lengths[int(len(lengths) * 0.50)]}")
    print(f"  - P95: {p95_value}")
    print(f"  - P99: {lengths[int(len(lengths) * 0.99)]}")

    return p95_value


class TextMatchDataset(Dataset):
    """
    文本匹配数据集
    
    根据 encoder_type 决定数据组织形式:
      - "biencoder": 分别编码 sentence1 和 sentence2，返回两个 input_ids 和 attention_mask
      - "crossencoder": 将 [CLS] s1 [SEP] s2 [SEP] 拼接编码，返回单个 input_ids 和 attention_mask
    """

    def __init__(self, data, tokenizer, encoder_type="biencoder", max_length=128, pooling="cls"):
        """
        参数:
            data: list[dict]，包含 sentence1, sentence2, label
            tokenizer: 分词器
            encoder_type: "biencoder" 或 "crossencoder"
            max_length: 最大序列长度
            pooling: 池化方式（仅用于 CrossEncoder 中记录，不影响 tokenize）
        """
        self.data = data
        self.tokenizer = tokenizer
        self.encoder_type = encoder_type
        self.max_length = max_length
        self.pooling = pooling
        # 检查 tokenizer 是否支持 token_type_ids（BERT 支持，RoBERTa 等不支持）
        self.has_token_type_ids = "token_type_ids" in tokenizer.model_input_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        s1, s2, label = item["sentence1"], item["sentence2"], item["label"]

        if self.encoder_type == "biencoder":
            # ---------- BiEncoder: 分别编码两条句子 ----------
            s1_enc = self.tokenizer(
                s1,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            s2_enc = self.tokenizer(
                s2,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            # 去掉 batch 维度
            return {
                "s1_input_ids": s1_enc["input_ids"].squeeze(0),
                "s1_attention_mask": s1_enc["attention_mask"].squeeze(0),
                "s2_input_ids": s2_enc["input_ids"].squeeze(0),
                "s2_attention_mask": s2_enc["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            }

        elif self.encoder_type == "crossencoder":
            # ---------- CrossEncoder: 拼接编码 ----------
            enc = self.tokenizer(
                s1, s2,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc["token_type_ids"].squeeze(0)
                if self.has_token_type_ids else None,
                "label": torch.tensor(label, dtype=torch.long),
            }

    @staticmethod
    def collate_fn(batch, encoder_type="biencoder"):
        """
        自定义 collate 函数，将批次内样本堆叠成 batch
        """
        if encoder_type == "biencoder":
            return {
                "s1_input_ids": torch.stack([b["s1_input_ids"] for b in batch]),
                "s1_attention_mask": torch.stack([b["s1_attention_mask"] for b in batch]),
                "s2_input_ids": torch.stack([b["s2_input_ids"] for b in batch]),
                "s2_attention_mask": torch.stack([b["s2_attention_mask"] for b in batch]),
                "label": torch.stack([b["label"] for b in batch]),
            }
        else:
            result = {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "label": torch.stack([b["label"] for b in batch]),
            }
            if batch[0]["token_type_ids"] is not None:
                result["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
            return result


class TripletDataset(Dataset):
    """
    三元组数据集 —— 用于 TripletLoss 训练
    
    数据格式: JSONL，每条包含 anchor, positive, negative
    输出: 三个句子的 BiEncoder 编码（input_ids + attention_mask）
    """

    def __init__(self, data, tokenizer, max_length=128):
        """
        参数:
            data: list[dict]，每条包含 anchor, positive, negative
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor, positive, negative = item["anchor"], item["positive"], item["negative"]

        # 编码三条句子
        anc_enc = self.tokenizer(anchor, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        pos_enc = self.tokenizer(positive, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        neg_enc = self.tokenizer(negative, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")

        return {
            "anc_input_ids": anc_enc["input_ids"].squeeze(0),
            "anc_attention_mask": anc_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }

    @staticmethod
    def collate_fn(batch):
        """堆叠三元组 batch"""
        return {
            "anc_input_ids": torch.stack([b["anc_input_ids"] for b in batch]),
            "anc_attention_mask": torch.stack([b["anc_attention_mask"] for b in batch]),
            "pos_input_ids": torch.stack([b["pos_input_ids"] for b in batch]),
            "pos_attention_mask": torch.stack([b["pos_attention_mask"] for b in batch]),
            "neg_input_ids": torch.stack([b["neg_input_ids"] for b in batch]),
            "neg_attention_mask": torch.stack([b["neg_attention_mask"] for b in batch]),
        }


def get_dataloaders(args, tokenizer):
    """
    加载数据集并创建 DataLoader
    当指定 --triplet_data 时，训练使用预构建的三元组数据
    自动计算 p95 作为 max_length（若未指定）
    
    返回:
        train_loader, val_loader, test_loader, p95_length
    """
    # 1. 加载原始数据
    use_triplet = args.triplet_data is not None and args.loss == "triplet"

    if use_triplet:
        # 加载预构建三元组作为训练集
        print(f"[数据加载] 使用三元组训练数据: {args.triplet_data}")
        train_data = load_jsonl(args.triplet_data)
        print(f"[数据加载] 三元组训练集: {len(train_data)} 条")
    else:
        train_data = load_jsonl(f"{args.data_dir}/train.jsonl")
        print(f"[数据加载] 训练集: {len(train_data)} 条")

    val_data = load_jsonl(f"{args.data_dir}/validation.jsonl")
    test_data = load_jsonl(f"{args.data_dir}/test.jsonl")

    # 2. 计算 p95 长度（若用户未指定 max_length）
    if args.max_length is None:
        if use_triplet:
            # 三元组数据：对 anchor/positive/negative 都取最长
            sampled = random.sample(train_data, min(5000, len(train_data)))
            lengths = []
            for item in sampled:
                for key in ["anchor", "positive", "negative"]:
                    tokens = tokenizer.tokenize(item[key])
                    lengths.append(len(tokens) + 2)
            lengths = sorted(lengths)
            p95_length = lengths[int(len(lengths) * 0.95)]
            print(f"[数据长度分析 - triplet] 采样 {len(sampled)} 条 | P95={p95_length}")
        else:
            p95_length = compute_p95_length(train_data, tokenizer, encoder_type=args.encoder)
        max_length = p95_length
    else:
        max_length = args.max_length
        p95_length = max_length
        print(f"[数据加载] 使用用户指定的 max_length = {max_length}")

    # 3. 创建 Dataset
    if use_triplet:
        train_dataset = TripletDataset(
            train_data, tokenizer,
            max_length=max_length,
        )
    else:
        train_dataset = TextMatchDataset(
            train_data, tokenizer,
            encoder_type=args.encoder,
            max_length=max_length,
            pooling=args.pooling,
        )
    val_dataset = TextMatchDataset(
        val_data, tokenizer,
        encoder_type=args.encoder,
        max_length=max_length,
        pooling=args.pooling,
    )
    test_dataset = TextMatchDataset(
        test_data, tokenizer,
        encoder_type=args.encoder,
        max_length=max_length,
        pooling=args.pooling,
    )

    # 4. 创建 DataLoader
    train_collate = TripletDataset.collate_fn if use_triplet else \
        (lambda b: TextMatchDataset.collate_fn(b, args.encoder))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: TextMatchDataset.collate_fn(b, args.encoder),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: TextMatchDataset.collate_fn(b, args.encoder),
    )

    return train_loader, val_loader, test_loader, p95_length


def compute_pair_accuracy(sim_scores, labels, threshold=0.5):
    """
    计算配对准确率
    sim_scores: 余弦相似度分数 (batch_size,)
    labels: 真实标签 (batch_size,)  — 1 为正例，0 为负例
    threshold: 判断为正例的相似度阈值
    """
    preds = (sim_scores >= threshold).float()
    correct = (preds == labels.float()).sum().item()
    total = labels.size(0)
    return correct / total, correct, total


def analyze_dataset(data, tokenizer, dataset_name="数据集"):
    """
    数据集探索性分析（EDA）
    统计标签分布、句子长度分布等基本信息
    
    参数:
        data: list[dict]，包含 sentence1, sentence2, label
        tokenizer: 分词器
        dataset_name: 数据集名称（用于打印）
    返回:
        stats: dict，包含各项统计信息
    """
    labels = [item["label"] for item in data]
    n_total = len(data)
    n_pos = sum(labels)  # label=1
    n_neg = n_total - n_pos  # label=0

    # 计算句子长度
    s1_lengths = [len(tokenizer.tokenize(item["sentence1"])) for item in data]
    s2_lengths = [len(tokenizer.tokenize(item["sentence2"])) for item in data]
    all_lengths = sorted(s1_lengths + s2_lengths)

    stats = {
        "dataset": dataset_name,
        "total": n_total,
        "positive": n_pos,
        "negative": n_neg,
        "positive_ratio": n_pos / n_total * 100,
        "s1_len_mean": sum(s1_lengths) / max(n_total, 1),
        "s2_len_mean": sum(s2_lengths) / max(n_total, 1),
        "len_min": all_lengths[0],
        "len_max": all_lengths[-1],
        "len_p50": all_lengths[int(len(all_lengths) * 0.50)],
        "len_p95": all_lengths[int(len(all_lengths) * 0.95)],
        "len_p99": all_lengths[int(len(all_lengths) * 0.99)],
        "s1_lengths": s1_lengths,
        "s2_lengths": s2_lengths,
    }

    print(f"\n{'='*50}")
    print(f"[EDA] {dataset_name} 探索性分析")
    print(f"{'='*50}")
    print(f"  样本总数:     {n_total}")
    print(f"  正例 (label=1): {n_pos} ({n_pos/n_total*100:.1f}%)")
    print(f"  负例 (label=0): {n_neg} ({n_neg/n_total*100:.1f}%)")
    print(f"  句子1平均长度: {stats['s1_len_mean']:.1f} tokens")
    print(f"  句子2平均长度: {stats['s2_len_mean']:.1f} tokens")
    print(f"  单句长度分布: min={stats['len_min']}, "
          f"P50={stats['len_p50']}, P95={stats['len_p95']}, P99={stats['len_p99']}, max={stats['len_max']}")
    print(f"{'='*50}\n")

    return stats
