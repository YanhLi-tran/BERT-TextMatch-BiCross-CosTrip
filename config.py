"""
配置文件 —— 命令行参数解析
支持通过命令行调整所有训练超参数
"""

import argparse


def get_args():
    """解析并返回命令行参数"""
    parser = argparse.ArgumentParser(description="BERT 文本匹配任务微调")

    # -------- 数据与路径参数 --------
    parser.add_argument("--data_dir", type=str, default=r"E:\文本匹配任务\data\bq_corpus",
                        help="数据集目录路径，包含 train.jsonl / validation.jsonl / test.jsonl")
    parser.add_argument("--bert_model", type=str, default=r"E:\models\bert-base-chinese",
                        help="BERT 预训练模型路径")
    parser.add_argument("--output_dir", type=str, default=r"E:\文本匹配任务\outputs",
                        help="训练输出目录（模型保存 + 可视化结果）")

    # -------- 模型结构参数 --------
    parser.add_argument("--encoder", type=str, default="biencoder", choices=["biencoder", "crossencoder"],
                        help="编码器类型：biencoder（双编码器）或 crossencoder（交叉编码器）")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "max"],
                        help="句子向量化方式：cls / mean（均值池化）/ max（最大池化）")
    parser.add_argument("--loss", type=str, default="cosine", choices=["cosine", "triplet"],
                        help="损失函数类型：cosine（余弦嵌入损失）或 triplet（三元组损失）")

    # -------- 训练超参数 --------
    parser.add_argument("--epochs", type=int, default=1000,
                        help="总训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="评估批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减系数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="学习率预热比例")
    parser.add_argument("--margin", type=float, default=None,
                        help="TripletLoss 的 margin / CosineEmbeddingLoss 的 margin（默认: cosine=0.0, triplet=1.0）")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪最大范数（设为 0 则关闭裁剪）")

    # -------- Checkpoint 与验证参数 --------
    parser.add_argument("--save_every", type=int, default=200,
                        help="每多少步(batch)保存一次 checkpoint")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="每多少轮在验证集上评估一次（用于最优模型筛选）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练，传入 checkpoint_step_N.pt 路径")
    parser.add_argument("--max_length", type=int, default=None,
                        help="最大序列长度（默认 None 则自动使用训练集 p95）")

    # -------- 设备与种子 --------
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备：cuda / cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证可复现")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 数据加载线程数（Windows 建议 0）")

    # -------- 可视化参数 --------
    parser.add_argument("--vis_interval", type=int, default=10,
                        help="每隔多少轮记录一次指标用于可视化")

    args = parser.parse_args()

    # 根据损失函数类型设置合理的 margin 默认值（仅当用户未显式传入时生效）
    # CosineEmbeddingLoss: margin=0.0 时负对 loss = max(0, cos)，模型会推开负样本
    # TripletLoss:         margin=1.0 是常用默认值
    if args.margin is None:
        args.margin = 0.0 if args.loss == "cosine" else 1.0

    return args
