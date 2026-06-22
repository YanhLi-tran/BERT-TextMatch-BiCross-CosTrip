"""
训练器模块 —— 训练循环、评估、Checkpoint 管理
支持 BiEncoder / CrossEncoder + CosineEmbeddingLoss / TripletLoss 四种组合
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from losses import get_loss_fn
from data_utils import compute_pair_accuracy


class Trainer:
    """
    训练器，封装训练循环、评估、模型保存
    """

    def __init__(self, model, args, run_name=None):
        """
        参数:
            model: 模型实例（BiEncoder 或 CrossEncoder）
            args: 命令行参数
            run_name: 当前运行的标识名（用于输出目录，自动生成）
        """
        self.model = model
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ---------- 损失函数 ----------
        self.loss_fn, self.use_triplet = get_loss_fn(args.loss, args.margin)
        self.use_classify = args.classify

        # 分类头模式使用 CrossEntropyLoss
        if self.use_classify:
            self.ce_loss = nn.CrossEntropyLoss()

        # ---------- 优化器 ----------
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # ---------- 学习率调度器（带预热） ----------
        # 先线性预热 warmup_steps，然后线性衰减到 0
        # 这里使用简易的 LambdaLR
        self.total_steps = None  # 后面 set_total_steps 时设置
        self.scheduler = None

        # ---------- 运行标识 ----------
        if run_name is None:
            run_name = f"{args.encoder}_{args.pooling}_{args.loss}"
        self.run_name = run_name

        # 输出子目录: outputs/{run_name}/
        self.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # 指标记录
        self.train_metrics = defaultdict(list)   # 训练指标
        self.val_metrics = defaultdict(list)     # 验证指标
        self.best_val_acc = 0.0                  # 最优验证准确率
        self.best_epoch = 0                      # 最优模型所在轮次
        self.epochs_no_improve = 0               # Early Stopping 计数器
        self.patience = args.patience            # Early Stopping 耐心值
        self.start_epoch = 0                     # 起始轮次（resume 时使用）
        self._latest_ckpt_path = None            # 最新 checkpoint 路径（用于覆盖）

        # 自动混合精度（AMP）支持
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        mode_str = "分类头" if args.classify else args.loss
        print(f"[训练器初始化] 设备: {self.device}")
        print(f"[训练器初始化] 编码器: {args.encoder} | 池化: {args.pooling} | 模式: {mode_str}")

        # ---------- 组合检查：CrossEncoder + cls + TripletLoss ----------
        if args.encoder == "crossencoder" and args.pooling == "cls" and args.loss == "triplet":
            print("[⚠ 警告] CrossEncoder + pooling='cls' + TripletLoss 组合下，"
                  "anchor 和 positive 均为同一 [CLS] 向量，")
            print("          TripletLoss 退化为仅推开负样本，建议改用 pooling='mean' 获取独立句表示。")
        print(f"[训练器初始化] 输出目录: {self.output_dir}")

    def set_total_steps(self, total_steps):
        """设置总步数并初始化调度器（需要在知道 DataLoader 长度后调用）"""
        self.total_steps = total_steps
        warmup_steps = int(total_steps * self.args.warmup_ratio)

        def lr_lambda(current_step):
            # 预热阶段
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # 衰减阶段
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ================== Triplet 负样本采样 ==================

    def _sample_negatives_for_triplet(self, emb1, emb2, labels):
        """
        In-batch negative sampling: 在 batch 内部为每个正样本对构造负样本
        
        策略: 对于每个 label=1 的 (s1, s2) 对:
          - anchor = emb1 (句子1的嵌入)
          - positive = emb2 (句子2的嵌入)
          - negative = 从 batch 中 **标签为 0 的样本** 中随机选一个 emb2，
                       避免把正样本当作负样本推开
        
        返回:
            anchor, positive, negative: 各为 (n_pos, hidden_dim)
            若 batch 中无正样本或无负样本则返回 None
        """
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        if n_pos == 0 or n_neg == 0:
            return None, None, None

        anchor = emb1[pos_mask]      # (n_pos, hidden_dim)
        positive = emb2[pos_mask]    # (n_pos, hidden_dim)

        # 所有负样本的向量
        neg_embs = emb2[neg_mask]    # (n_neg, hidden_dim)

        if self.args.online_hard:
            # Online Hard Negative Mining:
            # 对每个 anchor，在 batch 内所有负样本中挑余弦相似度最高的
            anchor_norm = torch.nn.functional.normalize(anchor, p=2, dim=1)   # (n_pos, dim)
            neg_embs_norm = torch.nn.functional.normalize(neg_embs, p=2, dim=1)  # (n_neg, dim)
            sim_matrix = torch.matmul(anchor_norm, neg_embs_norm.T)           # (n_pos, n_neg)
            hardest_idx = sim_matrix.argmax(dim=1)                            # (n_pos,)
            negative = neg_embs[hardest_idx]
        else:
            # 随机采样（旧方案）
            neg_indices_all = torch.where(neg_mask)[0]
            pick = torch.randint(0, n_neg, (n_pos,), device=self.device)
            neg_indices = neg_indices_all[pick]
            negative = emb2[neg_indices]

        return anchor, positive, negative

    # ================== 训练步骤 ==================

    def train_step(self, batch):
        """
        单步训练
        
        参数:
            batch: DataLoader 产出的一个 batch
        返回:
            loss: 标量损失值
            acc: 当前 batch 的准确率（仅对 cosine loss 有意义）
        """
        self.model.train()
        labels = batch.get("label", None)
        if labels is not None:
            labels = labels.to(self.device)

        # ----- 预构建三元组模式（TripletDataset） -----
        if "anc_input_ids" in batch:
            anc_ids = batch["anc_input_ids"].to(self.device)
            anc_mask = batch["anc_attention_mask"].to(self.device)
            pos_ids = batch["pos_input_ids"].to(self.device)
            pos_mask = batch["pos_attention_mask"].to(self.device)
            neg_ids = batch["neg_input_ids"].to(self.device)
            neg_mask = batch["neg_attention_mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                anc_emb = self.model.forward(anc_ids, anc_mask)
                pos_emb = self.model.forward(pos_ids, pos_mask)
                neg_emb = self.model.forward(neg_ids, neg_mask)
                # L2 归一化，让欧氏距离与余弦距离等价
                anc_emb = torch.nn.functional.normalize(anc_emb, p=2, dim=1)
                pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
                neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
                loss = self.loss_fn(anc_emb, pos_emb, neg_emb)

            # 反向传播 + 梯度裁剪
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.args.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            return loss.detach(), 0.0

        # ----- 分类头模式（CrossEncoder + classify） -----
        if self.use_classify:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.ce_loss(logits, labels)
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()

            # 反向传播
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.args.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            return loss.detach(), acc

        # ----- 非分类模式：原逻辑 -----
        if self.args.encoder == "biencoder":
            s1_ids = batch["s1_input_ids"].to(self.device)
            s1_mask = batch["s1_attention_mask"].to(self.device)
            s2_ids = batch["s2_input_ids"].to(self.device)
            s2_mask = batch["s2_attention_mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                emb1, emb2 = self.model.encode_pair(s1_ids, s1_mask, s2_ids, s2_mask)

                if self.args.loss == "cosine":
                    loss = self.loss_fn(emb1, emb2, labels)
                else:  # triplet
                    triplet_data = self._sample_negatives_for_triplet(emb1, emb2, labels)
                    if triplet_data[0] is None:
                        # batch 中无正样本，返回 0 损失
                        return torch.tensor(0.0, device=self.device), 0.0
                    anchor, positive, negative = triplet_data
                    # L2 归一化，让欧氏距离与余弦距离等价
                    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
                    positive = torch.nn.functional.normalize(positive, p=2, dim=1)
                    negative = torch.nn.functional.normalize(negative, p=2, dim=1)
                    loss = self.loss_fn(anchor, positive, negative)

                # 计算准确率（cosine loss 时才有意义）
                if self.args.loss == "cosine":
                    sim = self.model.compute_cosine_similarity(emb1, emb2)
                    acc, _, _ = compute_pair_accuracy(sim, labels.float())
                else:
                    acc = 0.0

        else:  # crossencoder
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                emb1, emb2 = self.model(input_ids, attention_mask, token_type_ids)

                if self.args.loss == "cosine":
                    loss = self.loss_fn(emb1, emb2, labels)

                    # 准确率
                    sim = self.model.compute_cosine_similarity(emb1, emb2)
                    acc, _, _ = compute_pair_accuracy(sim, labels.float())
                else:  # triplet
                    triplet_data = self._sample_negatives_for_triplet(emb1, emb2, labels)
                    if triplet_data[0] is None:
                        return torch.tensor(0.0, device=self.device), 0.0
                    anchor, positive, negative = triplet_data
                    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
                    positive = torch.nn.functional.normalize(positive, p=2, dim=1)
                    negative = torch.nn.functional.normalize(negative, p=2, dim=1)
                    loss = self.loss_fn(anchor, positive, negative)
                    acc = 0.0

        # ----- 反向传播 -----
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # AMP 下先 unscale 再裁剪
            if self.args.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.detach(), acc

    # ================== 评估 ==================

    @torch.no_grad()
    def _collect_similarities(self, data_loader):
        """
        收集所有样本的余弦相似度和标签，不经过阈值判断
        用于后续的阈值搜索和准确率计算
        
        返回:
            all_sims:   所有样本的余弦相似度 (n,)
            all_labels: 所有样本的真实标签 (n,)
        """
        self.model.eval()
        all_sims, all_labels = [], []

        collect_pbar = tqdm(data_loader, desc="收集相似度", leave=False,
                            colour="#9C27B0",
                            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt}",
                            ncols=70)
        for batch in collect_pbar:
            labels = batch["label"].to(self.device)

            if self.args.encoder == "biencoder":
                s1_ids = batch["s1_input_ids"].to(self.device)
                s1_mask = batch["s1_attention_mask"].to(self.device)
                s2_ids = batch["s2_input_ids"].to(self.device)
                s2_mask = batch["s2_attention_mask"].to(self.device)
                emb1, emb2 = self.model.encode_pair(s1_ids, s1_mask, s2_ids, s2_mask)
            else:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                emb1, emb2 = self.model(input_ids, attention_mask, token_type_ids)

            sim = self.model.compute_cosine_similarity(emb1, emb2)
            all_sims.append(sim.cpu())
            all_labels.append(labels.cpu())

        return torch.cat(all_sims), torch.cat(all_labels)

    def _find_best_threshold(self, data_loader, thresholds=None):
        """
        在验证集上搜索最优余弦相似度阈值
        
        参数:
            data_loader: 验证集 DataLoader
            thresholds: 候选阈值列表，默认 0.30 ~ 0.95 步长 0.05
        返回:
            best_threshold: 最优阈值
            best_acc: 最优准确率
        """
        if thresholds is None:
            thresholds = [round(t, 2) for t in
                          [i * 0.05 for i in range(6, 20)]]  # 0.30 ~ 0.95

        all_sims, all_labels = self._collect_similarities(data_loader)

        best_threshold = 0.5
        best_acc = 0.0

        for th in thresholds:
            preds = (all_sims >= th).float()
            acc = (preds == all_labels.float()).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_threshold = th

        print(f"[阈值搜索] 最优阈值: {best_threshold:.2f} (准确率: {best_acc:.4f})")
        return best_threshold, best_acc

    @torch.no_grad()
    def evaluate(self, data_loader, desc="评估", threshold=0.5):
        """
        在给定 DataLoader 上评估模型
        
        参数:
            data_loader: DataLoader
            desc: 进度条描述
            threshold: 余弦相似度阈值，默认 0.5
        返回:
            avg_loss: 平均损失（TripletLoss 模式下无意义，返回 0.0）
            avg_acc: 平均准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # TripletLoss 的 loss 依赖随机负采样，评估时无意义，跳过计算
        skip_loss = (self.args.loss == "triplet")

        # 评估进度条：显示实时准确率
        eval_pbar = tqdm(data_loader, desc=desc, leave=False,
                         colour="#FF9800",
                         bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {postfix}",
                         ncols=90)
        for batch in eval_pbar:
            labels = batch["label"].to(self.device)

            # ----- 分类头模式 -----
            if self.use_classify:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.ce_loss(logits, labels)
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                acc = correct / max(total, 1)
                total_loss += loss.item() * total
                total_correct += correct
                total_samples += total
                eval_pbar.set_postfix({"batch_acc": f"{acc:.3f}",
                                       "avg_acc": f"{total_correct/max(total_samples,1):.3f}"})
                continue

            if self.args.encoder == "biencoder":
                s1_ids = batch["s1_input_ids"].to(self.device)
                s1_mask = batch["s1_attention_mask"].to(self.device)
                s2_ids = batch["s2_input_ids"].to(self.device)
                s2_mask = batch["s2_attention_mask"].to(self.device)
                emb1, emb2 = self.model.encode_pair(s1_ids, s1_mask, s2_ids, s2_mask)
                sim = self.model.compute_cosine_similarity(emb1, emb2)
                if not skip_loss:
                    loss = self.loss_fn(emb1, emb2, labels)
                else:
                    loss = torch.tensor(0.0, device=self.device)

            else:  # crossencoder
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                emb1, emb2 = self.model(input_ids, attention_mask, token_type_ids)
                sim = self.model.compute_cosine_similarity(emb1, emb2)
                if not skip_loss:
                    loss = self.loss_fn(emb1, emb2, labels)
                else:
                    loss = torch.tensor(0.0, device=self.device)

            # 统计（使用传入的 threshold）
            total_loss += loss.item() * labels.size(0)
            acc, correct, total = compute_pair_accuracy(sim, labels.float(), threshold=threshold)
            total_correct += correct
            total_samples += total

            # 更新评估进度条：显示当前批次的准确率和累计准确率
            running_acc = total_correct / max(total_samples, 1)
            eval_pbar.set_postfix({
                "batch_acc": f"{acc:.3f}",
                "avg_acc": f"{running_acc:.3f}",
            })

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        return avg_loss, avg_acc

    # ================== 训练循环 ==================

    def train(self, train_loader, val_loader, epochs=None):
        """
        完整训练循环 —— 带 tqdm 可视化进度条（双层：整体进度 + 细粒度 batch 进度）
        
        参数:
            train_loader: 训练 DataLoader
            val_loader: 验证 DataLoader
            epochs: 训练轮数（默认使用 args.epochs）
        """
        if epochs is None:
            epochs = self.args.epochs

        # 设置总步数（用于 LR scheduler）
        total_steps = len(train_loader) * epochs
        self.set_total_steps(total_steps)
        global_step = 0

        print(f"\n{'='*60}")
        print(f"开始训练: {self.run_name}")
        print(f"  总轮数: {epochs} | 每轮步数: {len(train_loader)} | 总步数: {total_steps}")
        if self.patience > 0:
            print(f"  Early Stopping: 连续 {self.patience} 轮 Val Acc 不提升时自动停止")
        print(f"{'='*60}\n")

        # ===== 外层进度条：整体训练进度 =====
        # 显示已完成/总 epoch 数、已用时间/剩余时间、运行速率
        epoch_iterator = trange(
            self.start_epoch, epochs,
            desc="整体进度",
            unit="epoch",
            colour="#2196F3",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for epoch in epoch_iterator:
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            smooth_loss = 0.0  # 平滑 loss（指数滑动平均，过滤 batch 级波动）
            smooth_decay = 0.9

            # ----- 训练一个 epoch -----
            self.model.train()

            # ===== 内层进度条：每个 batch 的细粒度跟踪 =====
            inner_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False,
                colour="#4CAF50",
                bar_format="{desc} |{bar}| {percentage:3.0f}% [{n_fmt}/{total_fmt}] {postfix}",
                ncols=100,
            )

            for batch in inner_pbar:
                loss, acc = self.train_step(batch)
                loss_val = loss.item()

                epoch_loss += loss_val
                epoch_acc += acc
                num_batches += 1
                global_step += 1

                # 平滑 loss（指数滑动平均），过滤单个 batch 的噪声
                if num_batches == 1:
                    smooth_loss = loss_val
                else:
                    smooth_loss = smooth_decay * smooth_loss + (1 - smooth_decay) * loss_val

                # 更新内层进度条：实时 loss、平滑 loss、准确率、学习率
                lr = self.optimizer.param_groups[0]['lr']
                inner_pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "smooth": f"{smooth_loss:.4f}",
                    "acc": f"{acc:.4f}",
                    "lr": f"{lr:.2e}",
                })

                # ----- Checkpoint: 每 save_every step (batch) 保存一次 -----
                if global_step % self.args.save_every == 0:
                    self._save_checkpoint(global_step, epoch + 1)

            avg_train_loss = epoch_loss / max(num_batches, 1)
            avg_train_acc = epoch_acc / max(num_batches, 1)
            epoch_time = time.time() - epoch_start

            # ----- 记录训练指标（每 epoch 记录） -----
            self.train_metrics["loss"].append(avg_train_loss)
            self.train_metrics["acc"].append(avg_train_acc)

            # ----- 更新外层进度条：显示当前 epoch 的摘要 -----
            epoch_iterator.set_postfix({
                "loss": f"{avg_train_loss:.4f}",
                "acc": f"{avg_train_acc:.4f}",
                "time": f"{epoch_time:.1f}s",
            })

            # ----- 验证评估（按 eval_every 间隔，用于最优模型筛选） -----
            do_eval = ((epoch + 1) % self.args.eval_every == 0) or (epoch == epochs - 1)
            if do_eval:
                val_loss, val_acc = self.evaluate(val_loader, desc=f"验证 (Epoch {epoch+1})")

                # 记录验证指标（用于可视化，按 vis_interval 间隔采样保存）
                if (epoch + 1) % self.args.vis_interval == 0 or epoch == epochs - 1:
                    self.val_metrics["loss"].append(val_loss)
                    self.val_metrics["acc"].append(val_acc)
                    self.val_metrics["epoch"].append(epoch + 1)

                # 保存最优模型（每 eval_every 都判断，不错过最优点）
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch + 1
                    self.epochs_no_improve = 0
                    self._save_model("best_model.pt", epoch + 1, val_acc)
                    tqdm.write(f"  ★ 新最优模型! Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
                else:
                    # Early Stopping: 连续 patience 轮不提升则停止
                    if self.patience > 0:
                        self.epochs_no_improve += 1
                        if self.epochs_no_improve >= self.patience:
                            tqdm.write(f"\n  ◆ Early Stopping! 连续 {self.patience} 轮 Val Acc 未提升（当前 {val_acc:.4f}），训练自动结束")
                            tqdm.write(f"  最优模型: Epoch {self.best_epoch} | Val Acc: {self.best_val_acc:.4f}")
                            break

                loss_str = f"{val_loss:.4f}" if self.args.loss == "cosine" else "N/A"
                tqdm.write(f"[Epoch {epoch+1}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                      f"Val Loss: {loss_str} | Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.1f}s")
            else:
                # 不评估时只打印训练指标
                if (epoch + 1) % 10 == 0:
                    tqdm.write(f"[Epoch {epoch+1}/{epochs}] "
                          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                          f"Time: {epoch_time:.1f}s")

        # 训练结束
        print(f"\n{'='*60}")
        print(f"训练完成: {self.run_name}")
        print(f"  最优验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"{'='*60}")

        return self.train_metrics, self.val_metrics

    # ================== 模型保存 ==================

    def _save_model(self, filename, epoch, acc):
        """保存模型权重"""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": acc,
            "args": self.args,
            "run_name": self.run_name,
        }, path)
        return path

    def _save_checkpoint(self, global_step, epoch):
        """保存 checkpoint（按 step 粒度），只保留最新一份"""
        # 删除上一次的 checkpoint
        if self._latest_ckpt_path is not None and os.path.exists(self._latest_ckpt_path):
            os.remove(self._latest_ckpt_path)

        filename = f"checkpoint_step_{global_step}.pt"
        path = self._save_model(filename, epoch, 0.0)
        self._latest_ckpt_path = path
        tqdm.write(f"  [Checkpoint] Step {global_step} (Epoch {epoch}) 已保存: {path}")

    def save_final_model(self):
        """保存最终模型"""
        self._save_model("final_model.pt", self.args.epochs, 0.0)
        print(f"[最终模型] 已保存")

    def load_checkpoint(self, checkpoint_path):
        """加载 checkpoint 恢复训练"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        print(f"[加载 Checkpoint] 从 Epoch {self.start_epoch} 恢复")
        return checkpoint

    def load_model_weights(self, checkpoint_path):
        """
        轻量加载：仅加载模型权重，不修改优化器或起始轮次
        用于测试评估等不需要恢复训练的场景
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        val_acc = checkpoint.get("val_acc", 0.0)
        epoch = checkpoint.get("epoch", 0)
        print(f"[加载模型权重] 来自 Epoch {epoch} (Val Acc: {val_acc:.4f})")
        return val_acc, epoch
