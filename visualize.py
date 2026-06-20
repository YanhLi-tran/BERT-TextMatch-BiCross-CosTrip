"""
可视化模块 —— 训练指标可视化
输出 Loss / Accuracy 曲线图，保存到对应运行目录
不同参数组合的输出不会互相覆盖（按 run_name 分目录保存）
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免 GUI 依赖
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_metrics(train_metrics, val_metrics, save_dir, run_name):
    """
    绘制训练和验证的 Loss / Accuracy 曲线
    
    参数:
        train_metrics: dict，包含 "loss" 和 "acc" 列表（每个训练 step 或 epoch）
        val_metrics: dict，包含 "loss", "acc", "epoch" 列表（每个验证点）
        save_dir: 图片保存目录
        run_name: 当前运行名称（用于图标题）
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs_trained = list(range(1, len(train_metrics["loss"]) + 1))

    # ---------- 图1: Loss 曲线 ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_trained, train_metrics["loss"], label="Train Loss", color="#2196F3", linewidth=1.5)

    if val_metrics.get("epoch"):
        # 验证集只在某些 epoch 有值，需要对齐
        val_epochs = val_metrics["epoch"]
        ax.plot(val_epochs, val_metrics["loss"], label="Val Loss",
                color="#FF5722", linewidth=1.5, marker="o", markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Loss Curve — {run_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"[可视化] Loss 曲线已保存: {loss_path}")

    # ---------- 图2: Accuracy 曲线 ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_trained, train_metrics["acc"], label="Train Acc", color="#4CAF50", linewidth=1.5)

    if val_metrics.get("epoch"):
        val_epochs = val_metrics["epoch"]
        ax.plot(val_epochs, val_metrics["acc"], label="Val Acc",
                color="#FF9800", linewidth=1.5, marker="o", markersize=4)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Accuracy Curve — {run_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"[可视化] Accuracy 曲线已保存: {acc_path}")

    # ---------- 图3: Loss + Accuracy 合并图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左: Loss
    axes[0].plot(epochs_trained, train_metrics["loss"], label="Train Loss", color="#2196F3")
    if val_metrics.get("epoch"):
        axes[0].plot(val_epochs, val_metrics["loss"], label="Val Loss",
                     color="#FF5722", marker="o", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右: Accuracy
    axes[1].plot(epochs_trained, train_metrics["acc"], label="Train Acc", color="#4CAF50")
    if val_metrics.get("epoch"):
        axes[1].plot(val_epochs, val_metrics["acc"], label="Val Acc",
                     color="#FF9800", marker="o", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    fig.suptitle(f"Training Metrics — {run_name}", fontsize=16, y=1.02)
    fig.tight_layout()
    combined_path = os.path.join(save_dir, "combined_metrics.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 合并图已保存: {combined_path}")

    # ---------- 图4: 学习率曲线 ----------
    # (train_metrics 中不含 lr，此处仅作占位；若需 LR 曲线可在 trainer 中额外记录)
    print(f"[可视化] 所有图表已生成至: {save_dir}")


def plot_final_comparison(all_results, save_dir):
    """
    绘制不同配置的最终验证准确率对比柱状图
    
    参数:
        all_results: list of (run_name, best_val_acc)
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    names = [r[0] for r in all_results]
    accs = [r[1] for r in all_results]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    colors = plt.cm.viridis([max(0.3, min(1.0, a / max(accs + [0.01]))) for a in accs])
    bars = ax.bar(names, accs, color=colors, width=0.6)

    # 在柱子上标注数值
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Best Validation Accuracy", fontsize=12)
    ax.set_title("Comparison of Different Training Configurations", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    path = os.path.join(save_dir, "comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[可视化] 对比图已保存: {path}")


def eda_visualization(train_stats, val_stats, tokenizer, save_dir):
    """
    数据集探索性分析可视化
    
    参数:
        train_stats: analyze_dataset 返回的训练集统计
        val_stats: analyze_dataset 返回的验证集统计
        tokenizer: 分词器（用于获取 vocab 信息，仅占位）
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 图1: 标签分布（饼图 + 柱状图） ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：饼图
    labels_pie = ["Positive (label=1)", "Negative (label=0)"]
    sizes = [train_stats["positive"], train_stats["negative"]]
    colors_pie = ["#4CAF50", "#FF5722"]
    axes[0].pie(sizes, labels=labels_pie, autopct="%1.1f%%",
                colors=colors_pie, startangle=90, textprops={"fontsize": 11})
    axes[0].set_title(f"Label Distribution — {train_stats['dataset']}\n"
                      f"(Total: {train_stats['total']})", fontsize=13)

    # 右：训练集 vs 验证集标签数量对比
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width / 2, [train_stats["positive"], train_stats["negative"]],
                width, label="Train", color=["#66BB6A", "#FF8A65"])
    axes[1].bar(x + width / 2, [val_stats["positive"], val_stats["negative"]],
                width, label="Validation", color=["#81C784", "#FFAB91"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Positive", "Negative"])
    axes[1].set_ylabel("Count")
    axes[1].set_title("Label Counts: Train vs Validation", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(save_dir, "eda_label_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[EDA] 标签分布图已保存: {path}")

    # ---------- 图2: 句子长度分布直方图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    s1_lens = train_stats["s1_lengths"]
    s2_lens = train_stats["s2_lengths"]
    p95_val = train_stats["len_p95"]

    # 左：句子1长度分布
    axes[0].hist(s1_lens, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.5)
    axes[0].axvline(p95_val, color="#FF5722", linestyle="--", linewidth=2,
                    label=f"P95 = {p95_val}")
    axes[0].set_xlabel("Token Length")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Sentence 1 Length Distribution\nMean: {train_stats['s1_len_mean']:.1f}")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 右：句子2长度分布
    axes[1].hist(s2_lens, bins=50, color="#FF9800", alpha=0.7, edgecolor="white", linewidth=0.5)
    axes[1].axvline(p95_val, color="#FF5722", linestyle="--", linewidth=2,
                    label=f"P95 = {p95_val}")
    axes[1].set_xlabel("Token Length")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Sentence 2 Length Distribution\nMean: {train_stats['s2_len_mean']:.1f}")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Token Length Distribution — {train_stats['dataset']}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, "eda_length_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] 长度分布图已保存: {path}")

    print(f"[EDA] 所有 EDA 图已生成至: {save_dir}")

