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

    # ----- 4. 推理：收集相似度分数和标签 -----
    print(f"[4/6] 推理中 ...")
    all_sims, all_labels = [], []

    for batch in tqdm(loader, desc="推理"):
        labels = batch["label"].to(device)

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
    print(f"  TP={cm_result['tp']}  FP={cm_result['fp']}  FN={cm_result['fn']}  TN={cm_result['tn']}")
    print(f"{'='*55}\n")

    # 保存指标到 JSON
    summary = {
        "checkpoint": args.checkpoint,
        "data_path": data_path,
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": cm_result,
    }
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  评估摘要已保存: {summary_path}")
    print(f"  可视化图表已保存至: {output_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    evaluate()
