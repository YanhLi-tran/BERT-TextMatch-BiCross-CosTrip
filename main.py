"""
主入口 —— BERT 文本匹配微调
组合所有模块，根据命令行参数执行训练流程

用法示例:
    # BiEncoder + CosineEmbeddingLoss + CLS 池化
    python main.py --encoder biencoder --pooling cls --loss cosine

    # BiEncoder + TripletLoss + Mean 池化
    python main.py --encoder biencoder --pooling mean --loss triplet --margin 1.0

    # CrossEncoder + CosineEmbeddingLoss + Max 池化
    python main.py --encoder crossencoder --pooling max --loss cosine

    # CrossEncoder + TripletLoss + CLS 池化
    python main.py --encoder crossencoder --pooling cls --loss triplet --margin 0.8

    # 自定义参数
    python main.py --encoder biencoder --pooling mean --loss cosine --epochs 500 --batch_size 64 --lr 3e-5
"""

import os
import sys
import torch
import numpy as np
import random
from transformers import AutoTokenizer

from config import get_args
from data_utils import get_dataloaders, load_jsonl, analyze_dataset
from models import BiEncoder, CrossEncoder
from trainer import Trainer
from visualize import plot_metrics, eda_visualization


def set_seed(seed):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN 确定性（会降低性能，但确保完全可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(args):
    """
    根据命令行参数创建模型
    
    参数:
        args: 命令行参数
    返回:
        model: BiEncoder 或 CrossEncoder 实例
    """
    if args.encoder == "biencoder":
        model = BiEncoder(
            model_name=args.bert_model,
            pooling=args.pooling,
        )
        print(f"[模型] BiEncoder | 池化方式: {args.pooling}")
    elif args.encoder == "crossencoder":
        model = CrossEncoder(
            model_name=args.bert_model,
            pooling=args.pooling,
        )
        print(f"[模型] CrossEncoder | 池化方式: {args.pooling}")
    else:
        raise ValueError(f"不支持的编码器类型: {args.encoder}")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}")

    return model


def main():
    """主函数"""
    # 1. 解析命令行参数
    args = get_args()
    print(f"\n{'='*60}")
    print(f"BERT 文本匹配微调")
    print(f"{'='*60}")
    print(f"  编码器: {args.encoder}")
    print(f"  池化方式: {args.pooling}")
    print(f"  损失函数: {args.loss} (margin={args.margin})")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"{'='*60}\n")

    # 2. 设置随机种子
    set_seed(args.seed)

    # 3. 加载 tokenizer
    print("[加载] BERT Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # 4. 加载数据（自动计算 p95 长度）
    print("[加载] 训练/验证/测试数据 ...")
    train_loader, val_loader, test_loader, p95_length = get_dataloaders(args, tokenizer)

    # 4.5 数据集探索性分析（EDA）
    raw_train = load_jsonl(f"{args.data_dir}/train.jsonl")
    raw_val = load_jsonl(f"{args.data_dir}/validation.jsonl")
    train_stats = analyze_dataset(raw_train, tokenizer, "训练集")
    val_stats = analyze_dataset(raw_val, tokenizer, "验证集")
    eda_visualization(
        train_stats, val_stats, tokenizer,
        save_dir=os.path.join(args.output_dir, "eda"),
    )

    # 5. 创建模型
    model = create_model(args)

    # 6. 创建训练器
    # run_name: "{encoder}_{pooling}_{loss}"，确保不同参数组合输出不覆盖
    run_name = f"{args.encoder}_{args.pooling}_{args.loss}"
    trainer = Trainer(model, args, run_name=run_name)

    # 6.5 从 checkpoint 恢复训练（如果指定了 --resume）
    if args.resume is not None:
        if os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
            print(f"[恢复] 从 {args.resume} 恢复训练，从 Epoch {trainer.start_epoch} 继续")
        else:
            print(f"[警告] --resume 指定的路径不存在: {args.resume}，从头开始训练")

    # 7. 训练
    train_metrics, val_metrics = trainer.train(train_loader, val_loader)

    # 8. 保存最终模型
    trainer.save_final_model()

    # 9. 测试集评估（使用最优模型 + 最优阈值）
    print("\n[评估] 加载最优模型在测试集上评估 ...")
    best_model_path = os.path.join(trainer.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        val_acc, best_epoch = trainer.load_model_weights(best_model_path)

        # 在验证集上搜索最优余弦相似度阈值
        best_threshold, _ = trainer._find_best_threshold(val_loader)

        # 用最优阈值评估测试集
        test_loss, test_acc = trainer.evaluate(test_loader, desc="测试集",
                                                threshold=best_threshold)
        loss_str = f"{test_loss:.4f}" if args.loss == "cosine" else "N/A"
        print(f"\n{'='*60}")
        print(f"测试集结果 (最优模型 Epoch {best_epoch})")
        print(f"  Val Acc: {val_acc:.4f} | 最优阈值: {best_threshold:.2f}")
        print(f"  Test Loss: {loss_str} | Test Acc: {test_acc:.4f}")
        print(f"{'='*60}\n")
    else:
        print("[警告] 未找到最优模型，跳过测试集评估")
        test_loss, test_acc, best_threshold = 0.0, 0.0, 0.5

    # 10. 可视化训练曲线
    print("\n[可视化] 生成训练曲线 ...")
    plot_metrics(
        train_metrics,
        val_metrics,
        save_dir=trainer.output_dir,
        run_name=run_name,
    )

    # 11. 保存评估结果到文本文件
    summary_path = os.path.join(trainer.output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"运行配置: {run_name}\n")
        f.write(f"编码器: {args.encoder}\n")
        f.write(f"池化方式: {args.pooling}\n")
        f.write(f"损失函数: {args.loss} (margin={args.margin})\n")
        f.write(f"P95 长度: {p95_length}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"最优验证准确率: {trainer.best_val_acc:.4f} (Epoch {trainer.best_epoch})\n")
        f.write(f"最优余弦阈值: {best_threshold:.2f}\n")
        f.write(f"测试集准确率: {test_acc:.4f}\n")
    print(f"[结果] 摘要已保存: {summary_path}")

    print(f"\n[完成] 所有输出已保存到: {trainer.output_dir}")
    return trainer.best_val_acc


if __name__ == "__main__":
    main()
