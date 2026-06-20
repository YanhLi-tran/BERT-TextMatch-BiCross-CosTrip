"""
独立评估脚本 —— 加载训练好的模型，对任意数据集进行评估并输出可视化结果

用法:
    # 用验证集搜索最优阈值，输出详细指标和图表
    python evaluate.py --checkpoint outputs/biencoder_cls_cosine/best_model.pt

    # 在测试集上评估（用验证集找到的最优阈值）
    python evaluate.py --checkpoint outputs/biencoder_cls_cosine/best_model.pt ^
                       --data_split test

    # 在自定义文件上评估
    python evaluate.py --checkpoint outputs/biencoder_cls_cosine/best_model.pt ^
                       --data_path data/bq_corpus/test.jsonl
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# 设置 matplotlib 非交互后端（必须在 import pyplot 之前）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from data_utils import load_jsonl, TextMatchDataset, DataLoader
from models import BiEncoder, CrossEncoder


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型独立评估与可视化")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径（如 outputs/biencoder_cls_cosine/best_model.pt）")
    parser.add_argument("--data_dir", type=str, default=r"E:\文本匹配任务\data\bq_corpus",
                        help="数据集目录（当 --data_path 未指定时使用）")
    parser.add_argument("--data_split", type=str, default="validation",
                        choices=["train", "validation", "test"],
                        help="数据集划分（train/validation/test），与 --data_dir 配合使用")
    parser.add_argument("--data_path", type=str, default=None,
                        help="自定义数据文件路径（优先级高于 --data_dir + --data_split）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="评估结果输出目录（默认在 checkpoint 所在目录下创建 eval/）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="评估批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    parser.add_argument("--threshold", type=float, default=None,
                        help="余弦相似度阈值（默认 None 则自动在验证集上搜索最优值）")
    parser.add_argument("--mode", type=str, default="direct",
                        choices=["direct", "retrieval"],
                        help="评估模式: direct=逐对推理(默认), retrieval=预编码+向量检索(仅BiEncoder)")
    parser.add_argument("--embedding_cache", type=str, default=None,
                        help="向量缓存路径（retrieval 模式用，存在则加载、不存在则编码后保存）")

    return parser.parse_args()


# ==================== 可视化函数 ====================

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Negative", "Positive"],
           yticklabels=["Negative", "Positive"],
           xlabel="Predicted Label", ylabel="True Label")
    ax.set_title("Confusion Matrix", fontsize=13)

    # 标注数值
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [评估] 混淆矩阵已保存: {save_path}")
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def plot_roc_curve(y_true, y_scores, save_path):
    """绘制 ROC 曲线及计算 AUC"""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
            label="Random (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [评估] ROC 曲线已保存: {save_path} (AUC={roc_auc:.4f})")
    return roc_auc


def plot_pr_curve(y_true, y_scores, save_path):
    """绘制 Precision-Recall 曲线及计算 AP"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#FF5722", lw=2,
            label=f"PR curve (AP = {ap:.4f})")
    # 基线：正样本比例
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [评估] PR 曲线已保存: {save_path} (AP={ap:.4f})")
    return ap


def plot_similarity_distribution(y_true, y_scores, threshold, save_path):
    """
    绘制正负样本的相似度分布直方图
    两条分布的重叠程度直观反映模型区分能力
    """
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-1, 1, 41)

    ax.hist(neg_scores, bins=bins, alpha=0.6, color="#FF5722",
            label=f"Negative (n={len(neg_scores)})", density=True)
    ax.hist(pos_scores, bins=bins, alpha=0.6, color="#4CAF50",
            label=f"Positive (n={len(pos_scores)})", density=True)
    ax.axvline(threshold, color="#2196F3", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Similarity Score Distribution\n"
                 f"Positive vs Negative Pairs", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [评估] 相似度分布图已保存: {save_path}")


# ==================== 指标计算 ====================

def compute_metrics(y_true, y_pred, y_scores=None):
    """计算全面的分类指标"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, matthews_corrcoef)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # 若有分数，补充最佳阈值和 AUC
    if y_scores is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score
        metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
        metrics["ap"] = average_precision_score(y_true, y_scores)

    return metrics


def find_best_threshold(y_true, y_scores):
    """在验证集上搜索最优阈值（最大化 F1）"""
    thresholds = np.arange(0.30, 0.96, 0.05)
    best_th = 0.5
    best_f1 = 0.0
    for th in thresholds:
        preds = (y_scores >= th).astype(int)
        f1 = compute_metrics(y_true, preds)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


def plot_recall_curve(recall_at_k, save_path):
    """
    绘制 Recall@K 曲线
    
    参数:
        recall_at_k: dict, {k: recall_value}
        save_path: 保存路径
    """
    ks = sorted(recall_at_k.keys())
    recalls = [recall_at_k[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, recalls, color="#2196F3", linewidth=2, marker="o", markersize=6)
    ax.fill_between(ks, recalls, alpha=0.15, color="#2196F3")

    for k, r in zip(ks, recalls):
        ax.text(k, r + 0.01, f"{r:.2%}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("K (Top-K)", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_title("Recall@K Curve — Retrieval Evaluation", fontsize=14)
    ax.set_xlim(0, max(ks) + 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [检索] Recall@K 曲线已保存: {save_path}")


@torch.no_grad()
def evaluate_retrieval(model, data_loader, encoder, device, data_path,
                        embedding_cache=None):
    """
    预编码 + 向量检索评估
    
    流程:
      1. 将所有 sentence1 和 sentence2 编码为向量
      2. 对每个 query(s1), 从所有候选(s2)中检索 top-K
      3. 计算 Recall@K / MRR / 推理速度
    
    支持 embedding_cache 持久化：
      - 指定路径存在 → 直接加载缓存的向量
      - 不存在 → 编码后保存到该路径，下次秒级加载
    
    仅适用于 BiEncoder（CrossEncoder 不支持独立编码）
    """
    if encoder != "biencoder":
        print("[警告] 检索模式仅支持 BiEncoder，将使用逐对推理模式")
        return None, None

    from sklearn.metrics import ndcg_score

    all_s1_embs, all_s2_embs, all_labels = [], [], []
    total_pairs = 0
    encode_time = 0.0
    loaded_from_cache = False

    # ----- 阶段1: 编码所有句子（支持缓存） -----
    if embedding_cache is not None and os.path.exists(embedding_cache):
        # 从缓存加载
        cache_start = time.time()
        print(f"  [检索] 加载向量缓存: {embedding_cache}")
        cached = torch.load(embedding_cache, map_location="cpu", weights_only=False)
        s1_embs = cached["s1_embs"]
        s2_embs = cached["s2_embs"]
        labels = cached["labels"]
        total_pairs = s1_embs.size(0)
        encode_time = time.time() - cache_start
        loaded_from_cache = True
        print(f"  [检索] 缓存加载完成: {total_pairs} 对 | 耗时: {encode_time:.2f}s")
    else:
        # 重新编码
        print(f"  [检索] 预编码所有句子 ...")
        encode_start = time.time()

        for batch in tqdm(data_loader, desc="编码", colour="#9C27B0"):
            batch_size = batch["s1_input_ids"].size(0)
            s1_ids = batch["s1_input_ids"].to(device)
            s1_mask = batch["s1_attention_mask"].to(device)
            s2_ids = batch["s2_input_ids"].to(device)
            s2_mask = batch["s2_attention_mask"].to(device)
            labels_b = batch["label"].to(device)

            emb1 = model.forward(s1_ids, s1_mask)
            emb2 = model.forward(s2_ids, s2_mask)

            all_s1_embs.append(emb1.cpu())
            all_s2_embs.append(emb2.cpu())
            all_labels.append(labels_b.cpu())
            total_pairs += batch_size

        encode_time = time.time() - encode_start

        # 拼接
        s1_embs = torch.cat(all_s1_embs, dim=0)       # (N, dim)
        s2_embs = torch.cat(all_s2_embs, dim=0)       # (N, dim)
        labels = torch.cat(all_labels, dim=0)          # (N,)

        # 保存缓存
        if embedding_cache is not None:
            os.makedirs(os.path.dirname(embedding_cache), exist_ok=True)
            torch.save({
                "s1_embs": s1_embs,
                "s2_embs": s2_embs,
                "labels": labels,
            }, embedding_cache)
            print(f"  [检索] 向量缓存已保存: {embedding_cache}")

    cache_label = " (缓存加载)" if loaded_from_cache else ""
    print(f"  [检索] 编码完成{cache_label}: {total_pairs} 对 | 编码耗时: {encode_time:.2f}s | "
          f"{(total_pairs * 2 / max(encode_time, 0.01)):.0f} 句/秒")

    # 归一化
    s1_embs = torch.nn.functional.normalize(s1_embs, p=2, dim=1)
    s2_embs = torch.nn.functional.normalize(s2_embs, p=2, dim=1)

    # ----- 阶段2: 向量检索（FAISS 加速） -----
    print(f"  [检索] 执行向量检索 ...")
    search_start = time.time()

    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False
        print("  [检索] FAISS 未安装，使用 PyTorch 暴力检索（pip install faiss-cpu 可加速）")

    N = s1_embs.size(0)
    ks = [1, 3, 5, 10, 20, 50]
    top_k = max(ks)

    if use_faiss:
        # FAISS 检索（IndexIVFFlat 近似检索，加速 10-100 倍）
        dim = s1_embs.size(1)
        nlist = max(1, int(np.sqrt(N)))
        nprobe = max(1, nlist // 10)

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(s2_embs.numpy().astype(np.float32))
        index.add(s2_embs.numpy().astype(np.float32))
        index.nprobe = nprobe

        s1_np = s1_embs.numpy().astype(np.float32)
        D, I = index.search(s1_np, top_k)
    else:
        # 纯 PyTorch 暴力检索（降级方案）
        sim_matrix = torch.matmul(s1_embs, s2_embs.T)
        # 获取 top-K 索引
        _, top_indices_all = sim_matrix.topk(top_k, dim=1, largest=True)
        I = top_indices_all.cpu().numpy()

    # 对每个 query 计算 Recall@K 和 MRR（向量化加速）
    top_k_idx = I  # (N, top_k)
    labels_np = labels.numpy()

    # 对于 BQ 配对数据集的检索评估：
    #   查询 = s1_i，正确答案 = s2_i（仅当 label_i == 1 时）
    #   检查正确答案的索引 i 是否出现在 top-k 结果中
    pos_mask = (labels_np == 1)

    # 构建正确答案掩码: top_k_idx == i 的位置
    # (N, top_k) 矩阵，第 i 行标记哪些位置命中了正确答案 i
    gt_indices = np.arange(N, dtype=np.int64).reshape(-1, 1)
    is_correct = (top_k_idx == gt_indices) & pos_mask.reshape(-1, 1)
    # is_correct[i, rank] = True 表示 query i 在 rank 处命中了正确答案

    # Recall@K: 对每个 query, top-k 中是否包含正确答案
    recall_at_k = {}
    for k in ks:
        has_match = is_correct[:, :k].sum(axis=1) > 0
        recall_at_k[k] = has_match.sum() / max(pos_mask.sum(), 1)

    # MRR: 仅对有正样本的 query，第一个正确答案位置的倒数均值
    mrr_sum = 0.0
    n_pos_queries = 0
    for i in range(N):
        if not pos_mask[i]:
            continue
        n_pos_queries += 1
        for rank in range(top_k):
            if is_correct[i, rank]:
                mrr_sum += 1.0 / (rank + 1)
                break

    mrr = mrr_sum / max(n_pos_queries, 1)

    search_time = time.time() - search_start

    method = "FAISS-IVFFlat" if use_faiss else "PyTorch(brute)"
    print(f"  [检索] 检索耗时: {search_time:.2f}s ({method}) | "
          f"{(N / search_time):.0f} 查询/秒")

    # ----- 阶段3: 打印结果 -----
    print(f"\n{'='*45}")
    print(f"  Retrieval Evaluation Results")
    print(f"{'='*45}")
    print(f"  Total queries:     {N} ({n_pos_queries} positives)")
    print(f"  Candidate pool:    {N}")
    print(f"  Encoding time:     {encode_time:.2f}s")
    print(f"  Search time:       {search_time:.2f}s")
    print(f"  Queries/sec:       {N / search_time:.0f}")
    print(f"  ─────────────────────────")
    for k in ks:
        print(f"  Recall@{k:<4}         {recall_at_k[k]:.2%}")
    print(f"  MRR:               {mrr:.4f}")
    print("=" * 45)

    # 保存结果
    retrieval_results = {
        "total_queries": N,
        "positive_queries": int(n_pos_queries),
        "candidate_pool_size": N,
        "encoding_time_sec": round(encode_time, 2),
        "search_time_sec": round(search_time, 2),
        "search_method": "FAISS-IVFFlat" if use_faiss else "PyTorch(brute)",
        "queries_per_sec": round(N / search_time, 1),
        "recall_at_k": {str(k): round(v, 4) for k, v in recall_at_k.items()},
        "mrr": round(mrr, 4),
    }

    return recall_at_k, retrieval_results


# ==================== 主流程 ====================

@torch.no_grad()
def evaluate():
    """独立评估主函数"""
    args = get_args()

    # ----- 1. 加载 checkpoint -----
    print(f"[1/6] 加载 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", None)
    run_name = checkpoint.get("run_name", "unknown")

    if saved_args is None:
        # 从 checkpoint 文件名反推
        print("[警告] checkpoint 中未保存 args，从 run_name 推断配置")
        parts = run_name.split("_")
        encoder = parts[0] if len(parts) > 0 else "biencoder"
        pooling = parts[1] if len(parts) > 1 else "cls"
        bert_model = r"E:\models\bert-base-chinese"
    else:
        encoder = saved_args.encoder
        pooling = saved_args.pooling
        bert_model = saved_args.bert_model

    print(f"  编码器: {encoder} | 池化: {pooling} | run: {run_name}")

    # ----- 2. 创建模型并加载权重 -----
    print(f"[2/6] 创建模型 ...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if encoder == "biencoder":
        model = BiEncoder(model_name=bert_model, pooling=pooling)
    else:
        model = CrossEncoder(model_name=bert_model, pooling=pooling)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  模型已加载至 {device}")

    # ----- 3. 加载数据 -----
    print(f"[3/6] 加载数据 ...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    if args.data_path is not None:
        data_path = args.data_path
        print(f"  使用自定义文件: {data_path}")
    else:
        split_map = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
        data_path = os.path.join(args.data_dir, split_map[args.data_split])
        print(f"  使用默认文件: {data_path}")

    raw_data = load_jsonl(data_path)
    print(f"  共 {len(raw_data)} 条样本")

    # 确定 max_length：优先用 checkpoint 里保存的，否则自动计算 p95
    if saved_args and hasattr(saved_args, "max_length") and saved_args.max_length:
        max_length = saved_args.max_length
    else:
        from data_utils import compute_p95_length
        max_length = compute_p95_length(raw_data, tokenizer, encoder_type=encoder)
    print(f"  max_length = {max_length}")

    dataset = TextMatchDataset(raw_data, tokenizer,
                                encoder_type=encoder,
                                max_length=max_length,
                                pooling=pooling)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: TextMatchDataset.collate_fn(b, encoder),
                        num_workers=0)

    # ----- 模式分发：检索模式 vs 直接评估 -----
    if args.mode == "retrieval":
        # 若未指定缓存路径，自动使用 eval_retrieval/embeddings.pt
        if args.output_dir:
            cache_dir = args.output_dir
        else:
            cache_dir = os.path.join(os.path.dirname(args.checkpoint), "eval_retrieval")
        cache_path = args.embedding_cache or os.path.join(cache_dir, "embeddings.pt")

        recall_at_k, ret_results = evaluate_retrieval(
            model, loader, encoder, device, data_path,
            embedding_cache=cache_path,
        )

        # 确定输出目录
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(os.path.dirname(args.checkpoint), "eval_retrieval")
        os.makedirs(output_dir, exist_ok=True)

        # Recall@K 曲线
        if recall_at_k is not None:
            plot_recall_curve(
                recall_at_k,
                save_path=os.path.join(output_dir, "recall_curve.png"),
            )

            # 保存结果
            import json
            ret_path = os.path.join(output_dir, "retrieval_summary.json")
            with open(ret_path, "w", encoding="utf-8") as f:
                json.dump(ret_results, f, indent=2, ensure_ascii=False)
            print(f"  [检索] 结果已保存: {ret_path}")

        print(f"\n{'='*55}")
        print(f"  Retrieval Evaluation Complete")
        print(f"{'='*55}")
        return

    # ----- 4. 直接评估：推理（附带逐对推理计时） -----
    print(f"[4/6] 推理中 ...")
    all_sims, all_labels = [], []
    total_pairs = 0
    infer_start = time.time()

    for batch in tqdm(loader, desc="推理"):
        labels = batch["label"].to(device)
        batch_size = labels.size(0)

        if encoder == "biencoder":
            s1_ids = batch["s1_input_ids"].to(device)
            s1_mask = batch["s1_attention_mask"].to(device)
            s2_ids = batch["s2_input_ids"].to(device)
            s2_mask = batch["s2_attention_mask"].to(device)
            emb1, emb2 = model.encode_pair(s1_ids, s1_mask, s2_ids, s2_mask)
        else:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tids = batch.get("token_type_ids", None)
            if tids is not None:
                tids = tids.to(device)
            emb1, emb2 = model(ids, mask, tids)

        sim = model.compute_cosine_similarity(emb1, emb2)
        all_sims.append(sim.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_pairs += batch_size

    infer_end = time.time()
    infer_time = infer_end - infer_start
    pairs_per_sec = total_pairs / infer_time if infer_time > 0 else 0
    avg_ms_per_pair = infer_time / total_pairs * 1000 if total_pairs > 0 else 0

    print(f"\n  [推理速度] 总样本: {total_pairs} 对 | "
          f"总耗时: {infer_time:.2f}s | "
          f"{pairs_per_sec:.0f} 对/秒 | "
          f"{avg_ms_per_pair:.2f} ms/对")
    print(f"  [推理速度] 编码器类型: {encoder} | 池化方式: {pooling} | "
          f"数据路径: {data_path}")

    y_scores = np.concatenate(all_sims)
    y_true = np.concatenate(all_labels)

    # ----- 5. 确定阈值 -----
    print(f"[5/6] 确定最优阈值 ...")
    if args.threshold is not None:
        threshold = args.threshold
        print(f"  使用命令行指定的阈值: {threshold:.2f}")
    else:
        threshold, best_f1 = find_best_threshold(y_true, y_scores)
        print(f"  在数据上搜索最优阈值: {threshold:.2f} (F1={best_f1:.4f})")

    # ----- 6. 计算指标 & 可视化 -----
    print(f"[6/6] 计算指标并生成可视化 ...")

    # 确定输出目录（必须先定义，后续 plot 函数需要使用）
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(output_dir, exist_ok=True)

    y_pred = (y_scores >= threshold).astype(int)

    # 全面指标
    metrics = compute_metrics(y_true, y_pred, y_scores)
    cm_result = plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    # ROC / PR / 分布图
    roc_auc = plot_roc_curve(y_true, y_scores, save_path=os.path.join(output_dir, "roc_curve.png"))
    ap = plot_pr_curve(y_true, y_scores, save_path=os.path.join(output_dir, "pr_curve.png"))
    plot_similarity_distribution(y_true, y_scores, threshold,
                                  save_path=os.path.join(output_dir, "similarity_distribution.png"))

    # ----- 打印最终报告 -----
    print(f"\n{'='*55}")
    print(f"  评估报告 (阈值 = {threshold:.2f})")
    print(f"{'='*55}")
    print(f"  数据集: {data_path}")
    print(f"  样本数: {len(y_true)} (正例: {int(y_true.sum())}, 负例: {int((1-y_true).sum())})")
    print(f"  ─────────────────────────────")
    print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision): {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"  F1 分数:            {metrics['f1']:.4f}")
    print(f"  MCC:                {metrics['mcc']:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  AUC-ROC:            {metrics.get('auc_roc', 0):.4f}")
    print(f"  AP (PR-AUC):        {metrics.get('ap', 0):.4f}")
    print(f"  ─────────────────────────────")
    print(f"  推理速度:           {avg_ms_per_pair:.2f} ms/对 ({pairs_per_sec:.0f} 对/秒)")
    print(f"  ─────────────────────────────")
    print(f"  TP={cm_result['tp']}  FP={cm_result['fp']}  FN={cm_result['fn']}  TN={cm_result['tn']}")
    print(f"{'='*55}\n")

    # 保存指标到 JSON
    summary = {
        "checkpoint": args.checkpoint,
        "data_path": data_path,
        "encoder": encoder,
        "pooling": pooling,
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": cm_result,
        "infer_speed": {
            "total_pairs": total_pairs,
            "total_time_sec": round(infer_time, 2),
            "pairs_per_sec": round(pairs_per_sec, 1),
            "ms_per_pair": round(avg_ms_per_pair, 2)
        },
    }
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  评估摘要已保存: {summary_path}")
    print(f"  可视化图表已保存至: {output_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    evaluate()
