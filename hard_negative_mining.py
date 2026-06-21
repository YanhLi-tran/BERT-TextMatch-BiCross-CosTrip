"""
Hard Negative Mining —— 为 TripletLoss 构建难负样本三元组

流程:
  1. 加载已训好的 BiEncoder + Cosine 模型
  2. 编码全部候选句子（sentence2）建 FAISS 索引
  3. 对每个正样本对 (s1, s2)：
     - 用 s1 编码去索引中检索 top-K
     - 从结果中挑相似度最高但 label=0 的句子作为 hard negative
  4. 保存三元组 (anchor, positive, negative) 到 JSONL 文件

用法:
  python hard_negative_mining.py
"""

import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from models import BiEncoder
from data_utils import load_jsonl, TextMatchDataset, DataLoader

# ====== 配置 ======
CHECKPOINT = r"E:\文本匹配任务\outputs\biencoder_mean_cosine\best_model.pt"
DATA_PATH = r"E:\文本匹配任务\data\bq_corpus\train.jsonl"
OUTPUT_PATH = r"E:\文本匹配任务\data\bq_corpus\triplet_hard.jsonl"
BERT_MODEL = r"E:\models\bert-base-chinese"
BATCH_SIZE = 64
TOP_K = 50          # 在 top-50 中挑 hard negative
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    print("=" * 50)
    print("Hard Negative Mining")
    print("=" * 50)

    # 1. 加载 checkpoint
    print(f"[1/5] 加载模型: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", None)
    pooling = saved_args.pooling if saved_args else "mean"
    print(f"  池化方式: {pooling}")

    model = BiEncoder(model_name=BERT_MODEL, pooling=pooling)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"  模型已加载至 {DEVICE}")

    # 2. 加载 tokenizer 和数据
    print(f"[2/5] 加载数据: {DATA_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    raw_data = load_jsonl(DATA_PATH)
    print(f"  共 {len(raw_data)} 条样本")

    # 统计正负样本
    pos_pairs = [(i, item) for i, item in enumerate(raw_data) if item["label"] == 1]
    neg_pairs = [(i, item) for i, item in enumerate(raw_data) if item["label"] == 0]
    print(f"  正样本: {len(pos_pairs)} 对 | 负样本: {len(neg_pairs)} 对")

    # 3. 编码所有候选句子（sentence2）
    print(f"[3/5] 编码候选句子 ...")
    max_length = saved_args.max_length if saved_args and hasattr(saved_args, "max_length") and saved_args.max_length else 32

    dataset = TextMatchDataset(raw_data, tokenizer, encoder_type="biencoder",
                                max_length=max_length, pooling=pooling)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: TextMatchDataset.collate_fn(b, "biencoder"),
                        num_workers=0)

    all_s2_embs = []
    encode_start = time.time()

    for batch in tqdm(loader, desc="编码 s2"):
        s2_ids = batch["s2_input_ids"].to(DEVICE)
        s2_mask = batch["s2_attention_mask"].to(DEVICE)
        emb = model.forward(s2_ids, s2_mask)
        all_s2_embs.append(emb.cpu())

    s2_embs = torch.cat(all_s2_embs, dim=0)               # (N, 768)
    s2_embs = torch.nn.functional.normalize(s2_embs, p=2, dim=1)
    encode_time = time.time() - encode_start
    print(f"  编码完成: {len(s2_embs)} 条 | 耗时: {encode_time:.1f}s")

    # 4. 建 FAISS 索引
    print(f"[4/5] 建 FAISS 索引 ...")
    import faiss
    index = faiss.IndexFlatIP(s2_embs.size(1))
    index.add(s2_embs.numpy().astype(np.float32))
    dim = s2_embs.size(1)
    nlist = max(1, int(np.sqrt(len(s2_embs))))
    nprobe = max(1, nlist // 10)
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(s2_embs.numpy().astype(np.float32))
    ivf.add(s2_embs.numpy().astype(np.float32))
    ivf.nprobe = nprobe
    print(f"  索引完成: {len(s2_embs)} 条候选, nlist={nlist}, nprobe={nprobe}")

    # 5. 为每个正样本对挖掘 hard negative
    print(f"[5/5] 挖掘 Hard Negative ...")
    labels = torch.tensor([item["label"] for item in raw_data])
    neg_mask = (labels == 0).numpy()

    triplets = []
    no_neg_found = 0

    for idx, item in tqdm(pos_pairs, desc="挖掘"):
        s1_text = item["sentence1"]
        pos_text = item["sentence2"]

        # 编码 query（单条 s1）
        s1_enc = tokenizer(s1_text, return_tensors="pt",
                           truncation=True, padding="max_length",
                           max_length=max_length)
        s1_emb = model.forward(s1_enc["input_ids"].to(DEVICE),
                                s1_enc["attention_mask"].to(DEVICE))
        s1_emb = torch.nn.functional.normalize(s1_emb, p=2, dim=1)

        # FAISS 检索
        D, I = ivf.search(s1_emb.cpu().numpy().astype(np.float32), TOP_K)
        top_indices = I[0]

        # 在 top-K 中找第一个 label=0 的候选
        hard_neg_idx = None
        for cand_idx in top_indices:
            if neg_mask[cand_idx]:
                hard_neg_idx = cand_idx
                break

        if hard_neg_idx is not None:
            neg_text = raw_data[hard_neg_idx]["sentence2"]
            triplets.append({
                "anchor": s1_text,
                "positive": pos_text,
                "negative": neg_text,
                "anchor_idx": idx,
                "positive_idx": idx,
                "negative_idx": int(hard_neg_idx),
            })
        else:
            no_neg_found += 1

    # 6. 保存三元组
    print(f"\n  挖掘完成:")
    print(f"    成功: {len(triplets)} 条")
    print(f"    未找到 hard neg: {no_neg_found} 条")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\n  三元组已保存: {OUTPUT_PATH}")

    # 打印几个例子
    print(f"\n  Hard Negative 示例:")
    for i in range(min(3, len(triplets))):
        t = triplets[i]
        print(f"  ────────────────")
        print(f"  anchor:   {t['anchor']}")
        print(f"  positive: {t['positive']}")
        print(f"  negative: {t['negative']}")


if __name__ == "__main__":
    main()
