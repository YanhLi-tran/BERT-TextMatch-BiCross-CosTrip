"""
损失函数模块
提供 CosineEmbeddingLoss 和 TripletLoss 两种损失函数
"""

import torch
import torch.nn as nn


class CosineEmbeddingLoss(nn.Module):
    """
    余弦嵌入损失
    包装 PyTorch 的 nn.CosineEmbeddingLoss
    
    标签转换: 数据中 label=1（正例）→ target=1，label=0（负例）→ target=-1
    margin: 负例的余弦相似度上限（默认 0.0）
    """

    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, emb1, emb2, labels):
        """
        参数:
            emb1: 句子1嵌入 (batch_size, hidden_dim)
            emb2: 句子2嵌入 (batch_size, hidden_dim)
            labels: 原始标签 (batch_size,) — 0 或 1
        返回:
            loss: 标量
        """
        # 将标签从 {0, 1} 转换为 { -1, 1 }
        # label=1（相似）→ target=1，label=0（不相似）→ target=-1
        targets = labels.float() * 2 - 1  # 0→-1, 1→1
        return self.loss_fn(emb1, emb2, targets)


class TripletLoss(nn.Module):
    """
    三元组损失
    包装 PyTorch 的 nn.TripletMarginLoss
    
    需要 (anchor, positive, negative) 三元组
    在训练循环中通过 in-batch negative sampling 构造
    """

    def __init__(self, margin=1.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, reduction=reduction)

    def forward(self, anchor, positive, negative):
        """
        参数:
            anchor:   锚点嵌入 (batch_size, hidden_dim)
            positive: 正例嵌入 (batch_size, hidden_dim)
            negative: 负例嵌入 (batch_size, hidden_dim)
        返回:
            loss: 标量
        """
        return self.loss_fn(anchor, positive, negative)


def get_loss_fn(loss_type, margin):
    """
    根据名称获取损失函数实例
    
    参数:
        loss_type: "cosine" 或 "triplet"
        margin: CosineEmbeddingLoss 或 TripletLoss 的 margin 参数
    返回:
        loss_fn: 损失函数实例
        额外返回: 是否需要 triplet（用于训练循环）
    """
    if loss_type == "cosine":
        return CosineEmbeddingLoss(margin=margin), False
    elif loss_type == "triplet":
        return TripletLoss(margin=margin), True
    else:
        raise ValueError(f"不支持的损失函数: {loss_type}")
