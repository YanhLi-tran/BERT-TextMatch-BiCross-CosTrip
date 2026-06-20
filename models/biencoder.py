"""
BiEncoder（双编码器）模型
- 两条句子分别通过同一个 BERT 编码
- 支持 CLS / Mean / Max 三种池化方式
- 输出两条句子的嵌入向量，用于计算余弦相似度或构建三元组
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BiEncoder(nn.Module):
    """
    双编码器：共享 BERT 编码两条句子，分别产出句子向量
    
    支持的 pooling 方式:
      - "cls": 取 [CLS] token 输出
      - "mean": 对非 padding 位置的 token 输出取均值
      - "max":  对非 padding 位置的 token 输出取最大值
    """

    def __init__(self, model_name, pooling="cls"):
        super().__init__()
        self.pooling = pooling

        # 加载 BERT 模型
        config = AutoConfig.from_pretrained(model_name)
        config.return_dict = True  # 确保输出为命名元组，而非裸 tuple
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_dim = config.hidden_size  # 通常为 768

    def forward(self, input_ids, attention_mask):
        """
        单个句子的前向传播
        
        参数:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        返回:
            sentence_embedding: (batch_size, hidden_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # ---------- 根据 pooling 方式提取句子向量 ----------
        if self.pooling == "cls":
            # CLS 池化：取第一个 token ([CLS]) 的输出
            sentence_emb = last_hidden[:, 0, :]  # (batch_size, hidden_dim)

        elif self.pooling == "mean":
            # 均值池化：对非 padding 位置的 hidden states 取平均
            # attention_mask: (batch_size, seq_len, 1) 以便广播
            mask = attention_mask.unsqueeze(-1).float()
            sentence_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == "max":
            # 最大池化：对非 padding 位置取最大值（将 padding 位置设为 -inf）
            mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            # 将 padding 位置替换为很大的负数，使 max 忽略它们
            last_hidden = last_hidden + (1.0 - mask) * -1e9
            sentence_emb, _ = last_hidden.max(dim=1)

        else:
            raise ValueError(f"不支持的 pooling 方式: {self.pooling}")

        return sentence_emb

    def encode_pair(self, s1_input_ids, s1_attention_mask, s2_input_ids, s2_attention_mask):
        """
        编码一对句子，分别得到两个句子向量
        
        返回:
            emb1, emb2: 各为 (batch_size, hidden_dim)
        """
        emb1 = self.forward(s1_input_ids, s1_attention_mask)
        emb2 = self.forward(s2_input_ids, s2_attention_mask)
        return emb1, emb2

    def compute_cosine_similarity(self, emb1, emb2):
        """
        计算两个句子向量的余弦相似度
        返回: (batch_size,)，取值 [-1, 1]
        """
        # 归一化
        emb1_norm = nn.functional.normalize(emb1, p=2, dim=1)
        emb2_norm = nn.functional.normalize(emb2, p=2, dim=1)
        return (emb1_norm * emb2_norm).sum(dim=1)
