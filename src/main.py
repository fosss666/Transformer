import argparse
import os
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from rouge_score import rouge_scorer
from collections import namedtuple
import time

TempScoredResult = namedtuple("TempScoredResult", ["precision", "recall", "fmeasure"])

from data import get_dataloaders, set_seed
from model import Transformer
from util import count_parameters

# 初始化ROUGE评估器
ROUGE_SCORER = rouge_scorer.RougeScorer(
    rouge_types=["rouge1", "rouge2", "rougeL"],
    use_stemmer=False  # 禁用词干提取
)


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer文本摘要训练")
    parser.add_argument("--config", type=str, default="configs/base.yaml", 
                       help="YAML配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data", help="Gigaword数据集目录")
    parser.add_argument("--block_size", type=int, default=128, help="序列长度")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="初始学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test", action="store_true", help="仅执行测试预测")
    parser.add_argument("--best_model_path", type=str, default="results/results1/best_transformer_model.pth", 
                        help="最优模型保存路径")
    parser.add_argument("--log_path", type=str, default="results/results1/train_log.csv", 
                        help="训练日志保存路径")
    parser.add_argument("--curve_path", type=str, default="results/results1/train_val_curves.png", 
                        help="训练曲线保存路径")
    return parser.parse_args()


def validate(model, val_loader, loss_func, pad_id, sos_id, eos_id, stoi, itos, max_len, device):
    """验证函数"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_rouge1_f = 0.0
    total_rouge2_f = 0.0
    total_rougel_f = 0.0
    sample_count = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            sample_count += batch_size

            # 计算损失
            logits = model(x, y[:, :-1])  # Decoder输入移位，避免EOS泄露
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_flat = y[:, 1:].reshape(-1)
            valid_mask = y_flat != pad_id  # 忽略PADtoken损失

            if valid_mask.sum() > 0:
                logits_valid = logits_flat[valid_mask]
                y_valid = y_flat[valid_mask]
                n_tokens = len(y_valid)
                loss = loss_func(logits_valid, y_valid)
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

            # 生成摘要并计算ROUGE
            y_pred = torch.tensor([[sos_id] + [pad_id]*(max_len-1)], dtype=torch.long, device=device).repeat(batch_size, 1)
            mask_pad_x = model.mask_pad(x, pad_id)
            x_emb = model.embed_x(x)
            x_enc = model.encoder(x_emb, mask_pad_x)

            # Decoder自回归生成
            for i in range(max_len - 1):
                y_curr = y_pred[:, :i+1]
                mask_tril_y = model.mask_tril(y_curr, pad_id)  # 下三角掩码
                y_emb = model.embed_y(y_curr)
                y_dec = model.decoder(x_enc, y_emb, mask_pad_x, mask_tril_y)  # Decoder前向传播
                logits_step = model.fc_out(y_dec)[:, -1, :]
                next_token = logits_step.argmax(dim=-1)
                y_pred[:, i+1] = next_token
                if (next_token == eos_id).all() or (i+1 == max_len - 1):
                    break

            # 解码与ROUGE计算
            for pred_ids, true_ids in zip(y_pred, y):
                # 过滤特殊token（PAD/SOS/EOS）
                pred_text = "".join([itos[id_] for id_ in pred_ids.tolist() if id_ not in [pad_id, sos_id, eos_id]])
                true_text = "".join([itos[id_] for id_ in true_ids[1:].tolist() if id_ not in [pad_id, sos_id, eos_id]])

                if len(pred_text.strip()) > 0 and len(true_text.strip()) > 0:
                    scores = ROUGE_SCORER.score(true_text, pred_text)
                    total_rouge1_f += scores["rouge1"].fmeasure
                    total_rouge2_f += scores["rouge2"].fmeasure
                    total_rougel_f += scores["rougeL"].fmeasure
                else:
                    # 空文本按0分处理，保持指标计算结构一致
                    total_rouge1_f += TempScoredResult(0.0, 0.0, 0.0).fmeasure
                    total_rouge2_f += TempScoredResult(0.0, 0.0, 0.0).fmeasure
                    total_rougel_f += TempScoredResult(0.0, 0.0, 0.0).fmeasure

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 10 else float("inf")
    avg_rouge1_f = total_rouge1_f / sample_count if sample_count > 0 else 0.0
    avg_rouge2_f = total_rouge2_f / sample_count if sample_count > 0 else 0.0
    avg_rougel_f = total_rougel_f / sample_count if sample_count > 0 else 0.0

    return avg_loss, avg_perplexity, avg_rouge1_f, avg_rouge2_f, avg_rougel_f


def predict_text(model, x, stoi, itos, pad_id, sos_id, eos_id, max_len, device):
    """单样本预测"""
    model.eval()
    x = x.unsqueeze(0).to(device)
    y_pred = torch.tensor([[sos_id] + [pad_id]*(max_len-1)], dtype=torch.long, device=device)

    with torch.no_grad():
        mask_pad_x = model.mask_pad(x, pad_id)
        x_emb = model.embed_x(x)
        x_enc = model.encoder(x_emb, mask_pad_x)

        for i in range(max_len - 1):
            y_curr = y_pred[:, :i+1]
            mask_tril_y = model.mask_tril(y_curr, pad_id)
            y_emb = model.embed_y(y_curr)
            y_dec = model.decoder(x_enc, y_emb, mask_pad_x, mask_tril_y)
            logits_step = model.fc_out(y_dec)[:, -1, :]
            next_token = logits_step.argmax(dim=-1)
            y_pred[:, i+1] = next_token
            if next_token.item() == eos_id or (i+1 == max_len - 1):
                break

    # 过滤特殊token，生成可读文本
    pred_text = "".join([itos[id_] for id_ in y_pred.squeeze(0).tolist() if id_ not in [pad_id, sos_id, eos_id]])
    return pred_text


def plot_curves(log_path: str, save_path: str):
    """绘制训练曲线"""
    import pandas as pd
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

    df = pd.read_csv(log_path)

    plt.figure(figsize=(16, 8))
    # 训练/验证损失
    plt.subplot(1, 3, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 验证困惑度
    plt.subplot(1, 3, 2)
    plt.plot(df["epoch"], df["val_perplexity"], label="Val Perplexity", color="#2ca02c", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Perplexity Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 验证ROUGE
    plt.subplot(1, 3, 3)
    plt.plot(df["epoch"], df["val_rouge1_f"], label="Val ROUGE-1", color="#d62728", linewidth=2)
    plt.plot(df["epoch"], df["val_rouge2_f"], label="Val ROUGE-2", color="#9467bd", linewidth=2)
    plt.plot(df["epoch"], df["val_rougel_f"], label="Val ROUGE-L", color="#8c564b", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("ROUGE F1 Score", fontsize=12)
    plt.title("ROUGE Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[main.py] 训练曲线保存至：{save_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    # os.makedirs("results", exist_ok=True) # 为了记录不同结果手动创建了
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[main.py] 初始化完成：设备={device}，随机种子={args.seed}")

    print(f"[main.py] 加载Gigaword数据集（目录：{args.data_dir}）...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.seed
    )
    vocab_size = meta["vocab_size"]
    stoi, itos = meta["stoi"], meta["itos"]
    pad_id, sos_id, eos_id = meta["pad_id"], meta["sos_id"], meta["eos_id"]
    print(f"[main.py] 数据加载完成：vocab_size={vocab_size}，训练批次={len(train_loader)}")

    print(f"[main.py] 初始化Transformer...")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=256,  # 嵌入维度
        n_layers=4,   # 编码器/解码器层数
        n_heads=4,    # 注意力头数 消融2
        d_hidden=1024, # FFN隐藏层维度
        pad_id=pad_id  # PADtoken标识
    ).to(device)
    print(f"[main.py] 模型参数：{count_parameters(model):,} 个")

    # 优化组件AdamW+梯度裁剪+学习率调度
    loss_func = nn.CrossEntropyLoss(ignore_index=pad_id)  # 忽略PAD损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=0.01)  # AdamW优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam优化器
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)  # 每3轮衰减50%的学习率调度
    print(f"[main.py] 优化组件：AdamW + StepLR")

    if not args.test:
        with open(args.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "step", "lr",
                "train_loss",
                "val_loss", "val_perplexity",
                "val_rouge1_f", "val_rouge2_f", "val_rougel_f",
                "epoch_time(s)", "total_elapsed_time(s)"
            ])
        print(f"[main.py] 日志初始化：{args.log_path}")

    # 6. 训练循环
    if not args.test:
        print(f"[main.py] 开始训练（{args.epochs}轮）...")
        global_step = 0
        best_val_loss = float("inf")
        best_model_state = None
        best_optimizer_state = None
        best_epoch = 0
        total_train_start = time.time()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            epoch_train_loss = 0.0
            epoch_train_tokens = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                global_step += 1

                logits = model(x, y[:, :-1])
                logits_flat = logits.reshape(-1, vocab_size)
                y_flat = y[:, 1:].reshape(-1)
                valid_mask = y_flat != pad_id

                if valid_mask.sum() > 0:
                    logits_valid = logits_flat[valid_mask]
                    y_valid = y_flat[valid_mask]
                    n_tokens = len(y_valid)
                    loss = loss_func(logits_valid, y_valid)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 1-19
                    optimizer.step()

                    epoch_train_loss += loss.item() * n_tokens
                    epoch_train_tokens += n_tokens

                if global_step % 200 == 0:
                    avg_batch_loss = epoch_train_loss / epoch_train_tokens if epoch_train_tokens > 0 else 0.0
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"[train-batch] epoch={epoch:2d} | step={global_step:4d} | lr={current_lr:.6f} | loss={avg_batch_loss:.4f}")

            # 每轮验证
            avg_val_loss, avg_val_perplexity, avg_val_rouge1, avg_val_rouge2, avg_val_rougel = validate(
                model=model,
                val_loader=val_loader,
                loss_func=loss_func,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                stoi=stoi,
                itos=itos,
                max_len=args.block_size,
                device=device
            )
            scheduler.step()  # 学习率调度更新

            epoch_time = round(time.time() - epoch_start, 2)
            total_elapsed = round(time.time() - total_train_start, 2)

            # 计算轮次训练损失
            avg_epoch_train_loss = epoch_train_loss / epoch_train_tokens if epoch_train_tokens > 0 else 0.0

            # 内存更新最优参数
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = model.state_dict()
                best_optimizer_state = optimizer.state_dict()

            # 写入日志
            with open(args.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, global_step, optimizer.param_groups[0]["lr"],
                    round(avg_epoch_train_loss, 4),
                    round(avg_val_loss, 4), round(avg_val_perplexity, 4),
                    round(avg_val_rouge1, 4), round(avg_val_rouge2, 4), round(avg_val_rougel, 4),
                    epoch_time, total_elapsed
                ])

            print(f"\n[train-epoch] epoch={epoch:2d}：")
            print(f"  训练损失：{avg_epoch_train_loss:.4f}")
            print(f"  验证指标：loss={avg_val_loss:.4f}，perplexity={avg_val_perplexity:.2f}")
            print(f"  ROUGE指标：rouge1={avg_val_rouge1:.4f}，rouge2={avg_val_rouge2:.4f}，rougel={avg_val_rougel:.4f}")
            print(f"  时间信息：轮次耗时={epoch_time}s，累计耗时={total_elapsed}s\n")

        # 训练结束后保存最优模型+输出总耗时
        total_train_time = round(time.time() - total_train_start, 2)
        if best_model_state is not None:
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": best_optimizer_state,
                "val_loss": best_val_loss,
                "total_train_time(s)": total_train_time,
                "args": args
            }, args.best_model_path)
            print(f"[main.py] 训练结束，最优模型保存至：{args.best_model_path}")
            print(f"[main.py] 总训练时间：{total_train_time}s（{total_train_time/60:.2f}min）")

        plot_curves(args.log_path, args.curve_path)

    print(f"[main.py] 测试预测")
    if args.test:
        # 加载最优模型
        if not os.path.exists(args.best_model_path):
            raise FileNotFoundError(f"最优模型文件不存在：{args.best_model_path}，需要先执行训练")
        checkpoint = torch.load(args.best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[main.py] 加载最优模型：{args.best_model_path}（epoch={checkpoint['epoch']}，训练总耗时={checkpoint['total_train_time(s)']:.2f}s）")
    model.to(device)

    # 测试样本对比
    test_batch = next(iter(test_loader))
    x_test, y_test = test_batch[0].to(device), test_batch[1].to(device)

    for i in range(3):
        x_sample = x_test[i].cpu()
        y_sample = y_test[i].cpu()
        # 解码为可读文本
        input_text = "".join([itos[id_] for id_ in x_sample.tolist() if id_ != pad_id])
        true_text = "".join([itos[id_] for id_ in y_sample.tolist() if id_ not in [pad_id, sos_id, eos_id]])
        pred_text = predict_text(model, x_sample, stoi, itos, pad_id, sos_id, eos_id, args.block_size, device)

        rouge_scores = ROUGE_SCORER.score(true_text, pred_text) if (len(true_text.strip()) > 0 and len(pred_text.strip()) > 0) else {}
        rouge1_f = rouge_scores.get("rouge1", TempScoredResult(0.0, 0.0, 0.0)).fmeasure
        rouge2_f = rouge_scores.get("rouge2", TempScoredResult(0.0, 0.0, 0.0)).fmeasure
        rougel_f = rouge_scores.get("rougeL", TempScoredResult(0.0, 0.0, 0.0)).fmeasure

        print(f"\n=== 测试样本 {i+1} ===")
        print(f"输入文本（原文）：{input_text[:150]}..." if len(input_text) > 150 else f"输入文本：{input_text}")
        print(f"真实文本（参考摘要）：{true_text[:100]}..." if len(true_text) > 100 else f"真实文本：{true_text}")
        print(f"预测文本（模型摘要）：{pred_text[:100]}..." if len(pred_text) > 100 else f"预测文本：{pred_text}")
        print(f"ROUGE分数：rouge1={rouge1_f:.4f}，rouge2={rouge2_f:.4f}，rougel={rougel_f:.4f}")

    print(f"\n[main.py] 所有任务完成over over")


if __name__ == "__main__":
    main()