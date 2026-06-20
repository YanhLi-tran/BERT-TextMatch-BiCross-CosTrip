"""
CrossEncoder（交叉编码器）模型
- 两条句子拼接后通过 BERT 编码，利用 cross-attention 捕捉交互信息
- 支持 CLS / Mean / Max 三种池化方式提取句子向量
- 对于 CosineEmbeddingLoss:
    通过 token_type_ids 区分句子A和句子B的 token，分别池化得到两个向量
- 对于 TripletLoss:
    同样提取句子A和句子B的向量，构建三元组
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class CrossEncoder(nn.Module):
    """
    交叉编码器：拼接两条句子输入 BERT，利用跨注意力交互
    
    输入格式: [CLS] sentence1 [SEP] sentence2 [SEP]
    通过 token_type_ids 区分句子 A（0）和句子 B（1）
    
    支持的 pooling 方式:
      - "cls": 取 [CLS] token 输出作为联合表示
      - "mean": 对每句的 token 分别取均值，得到两个句子向量
      - "max":  对每句的 token 分别取最大值，得到两个句子向量
    
    支持分类头模式（classify=True）：
      - 在 [CLS] 上加一个 nn.Linear(768, 2) 做二分类
      - forward 返回 logits (batch_size, 2)
    """

    def __init__(self, model_name, pooling="cls", classify=False):
        super().__init__()
        self.pooling = pooling
        self.classify = classify

        # 加载 BERT 模型
        config = AutoConfig.from_pretrained(model_name)
        config.return_dict = True  # 确保输出为命名元组，而非裸 tuple
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_dim = config.hidden_size  # 通常为 768

        # 分类头（二分类）
        if classify:
            self.classifier = nn.Linear(self.hidden_dim, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播：编码拼接后的句子对
        
        参数:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len) — 0 表示句子 A，1 表示句子 B
        返回:
            classify=True:  logits (batch_size, 2)
            classify=False: emb1, emb2 (each batch_size, hidden_dim)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        cls_vec = last_hidden[:, 0, :]           # (batch_size, hidden_dim)

        # 分类头模式：直接输出 logits
        if self.classify:
            logits = self.classifier(cls_vec)
            return logits

        # 非分类模式：提取句子向量（原逻辑）
        if self.pooling == "cls":
            # CLS 池化：取第一个 token 的输出，作为整个句对表示
            sentence_emb = cls_vec
            return sentence_emb, sentence_emb  # 两个相同

        # ----- 以下需要区分句子 A 和句子 B 的 token -----
        if token_type_ids is None:
            raise ValueError("pooling='mean' 或 'max' 时需要提供 token_type_ids")

        # 句子 A 的 mask: token_type_ids == 0 且 attention_mask == 1
        mask_a = ((token_type_ids == 0) & (attention_mask == 1)).unsqueeze(-1).float()
        # 句子 B 的 mask: token_type_ids == 1 且 attention_mask == 1
        mask_b = ((token_type_ids == 1) & (attention_mask == 1)).unsqueeze(-1).float()

        if self.pooling == "mean":
            emb1 = (last_hidden * mask_a).sum(dim=1) / mask_a.sum(dim=1).clamp(min=1e-9)
            emb2 = (last_hidden * mask_b).sum(dim=1) / mask_b.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == "max":
            hidden_a = last_hidden + (1.0 - mask_a) * -1e9
            hidden_b = last_hidden + (1.0 - mask_b) * -1e9
            emb1, _ = hidden_a.max(dim=1)
            emb2, _ = hidden_b.max(dim=1)

        else:
            raise ValueError(f"不支持的 pooling 方式: {self.pooling}")

        return emb1, emb2

    def compute_cosine_similarity(self, emb1, emb2):
        """
        计算两个句子向量的余弦相似度
        返回: (batch_size,)
        """
        emb1_norm = nn.functional.normalize(emb1, p=2, dim=1)
        emb2_norm = nn.functional.normalize(emb2, p=2, dim=1)
        return (emb1_norm * emb2_norm).sum(dim=1)
