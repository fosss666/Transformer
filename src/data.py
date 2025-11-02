"""
Gigaword 本地数据集加载脚本（适配本地文件结构）

功能：
- 加载本地已下载的Gigaword数据集（src=原文，tgt=摘要）
- 支持指定数据集根目录（含giga_data/org_data）
- 构建字符级 vocab 并保存
- 提供与模型兼容的DataLoader和元信息
- 支持--test验证数据加载

依赖：torch, numpy, tqdm（可选，用于进度条）
本地文件结构要求（与你的截图一致）：
<data_root>/
  ├─ giga_data/
  └─ org_data/
     ├─ dev.src.txt    （验证集原文）
     ├─ dev.tgt.txt    （验证集摘要）
     ├─ test.src.txt   （测试集原文）
     ├─ test.tgt.txt   （测试集摘要）
     ├─ train.src.txt  （训练集原文）
     ├─ train.tgt.txt  （训练集摘要）
     ├─ train.src.10k  （训练集原文10k子集，优先使用）
     └─ train.tgt.10k  （训练集摘要10k子集，优先使用）
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # 可选，用于显示加载进度

# 本地数据集文件映射（根据你的文件结构定义）
# 优先使用10k子集（数据量小，适合本地训练）
FILE_MAPPING = {
    "train": {
        "src": "train.src",  # 训练集原文
        # "src": "train.src.10k",  # 训练集原文（10k子集）
        "tgt": "train.tgt"   # 训练集摘要
        # "tgt": "train.tgt.10k"   # 训练集摘要（10k子集）
    },
    "val": {
        "src": "dev.src",    # 验证集原文（dev=validation）
        "tgt": "dev.tgt"     # 验证集摘要
    },
    "test": {
        "src": "test.src",   # 测试集原文
        "tgt": "test.tgt"    # 测试集摘要
    }
}
# 数据集根目录下的子文件夹（根据你的截图，文件在org_data或giga_data中）
DATA_SUB_DIRS = ["ggw_data"]  # 自动搜索这两个子目录


def set_seed(seed: int):
    """固定随机种子，确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_data_dir(root_dir: Path) -> Path:
    """自动搜索数据集实际目录（在root_dir下的DATA_SUB_DIRS中）"""
    for sub_dir in DATA_SUB_DIRS:
        candidate = root_dir / sub_dir
        if candidate.exists() and any(candidate.glob("*.src*")):  # 检查是否有src文件
            return candidate
    raise FileNotFoundError(
        f"未在 {root_dir} 下的 {DATA_SUB_DIRS} 中找到数据集文件，请检查路径是否正确"
    )


def load_local_samples(data_root: Path) -> Dict[str, List[Dict]]:
    data_dir = find_data_dir(data_root)
    print(f"[data.py] 从本地目录加载数据：{data_dir}")

    data_splits = {}
    # 定义各数据集的最大样本数（训练集用10k子集，验证集/测试集各取1k）
    MAX_SAMPLES = {
    "train": 200000,  # 20K pairs（训练集）
    "val": 20000,     # 验证集取10%（2千对，比例合理）
    "test": 20000     # 测试集取10%（2千对）
}

    for split, files in FILE_MAPPING.items():
        src_path = data_dir / files["src"]
        tgt_path = data_dir / files["tgt"]

        # 读取文件（保持不变）
        with open(src_path, "r", encoding="utf-8") as f_src, \
             open(tgt_path, "r", encoding="utf-8") as f_tgt:
            src_lines = [line.strip() for line in f_src if line.strip()]
            tgt_lines = [line.strip() for line in f_tgt if line.strip()]
            if len(src_lines) != len(tgt_lines):
                raise ValueError(f"{split} 原文和摘要行数不匹配")

        # 截断样本数（只对val和test生效）
        max_n = MAX_SAMPLES[split]
        if max_n is not None and len(src_lines) > max_n:
            src_lines = src_lines[:max_n]
            tgt_lines = tgt_lines[:max_n]
            print(f"[data.py] {split} 样本截断为 {max_n} 条（原始 {len(src_lines)} 条）")

        # 构建样本列表
        samples = [{"article": src, "summary": tgt} for src, tgt in zip(src_lines, tgt_lines)]
        data_splits[split] = samples
        print(f"[data.py] {split} 样本数：{len(samples)}")

    return data_splits


def build_vocab(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """构建字符级词汇表，包含<PAD>、<SOS>、<EOS>特殊标记"""
    all_chars = set()
    # 遍历所有样本的原文和摘要，收集字符
    for sample in tqdm(samples, desc="构建词汇表"):
        all_chars.update(sample["article"])
        all_chars.update(sample["summary"])

    # 排序字符（确保词汇表顺序固定）
    chars = sorted(list(all_chars))
    # 特殊标记（顺序不可修改，与模型逻辑关联）
    special_tokens = ["<PAD>", "<SOS>", "<EOS>"]
    chars = special_tokens + chars

    # 构建映射表
    stoi = {char: idx for idx, char in enumerate(chars)}  # 字符→ID
    itos = {idx: char for char, idx in stoi.items()}      # ID→字符
    print(f"[data.py] 词汇表构建完成：共 {len(stoi)} 个字符（含 {len(special_tokens)} 个特殊标记）")
    return stoi, itos


def save_vocab(stoi: Dict[str, int], itos: Dict[int, str], vocab_dir: Path):
    """保存词汇表到指定目录"""
    vocab_dir.mkdir(parents=True, exist_ok=True)
    with open(vocab_dir / "stoi.json", "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    with open(vocab_dir / "itos.json", "w", encoding="utf-8") as f:
        json.dump(itos, f, ensure_ascii=False, indent=2)
    print(f"[data.py] 词汇表已保存到：{vocab_dir}")


def load_vocab(vocab_dir: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """加载已保存的词汇表"""
    with open(vocab_dir / "stoi.json", "r", encoding="utf-8") as f:
        stoi = json.load(f)
    with open(vocab_dir / "itos.json", "r", encoding="utf-8") as f:
        itos_raw = json.load(f)
    itos = {int(idx_str): char for idx_str, char in itos_raw.items()}  # 转换键为int
    return stoi, itos


class Seq2SeqDataset(Dataset):
    """
    适配Gigaword的seq2seq数据集：
    - x: 编码后的原文（article），长度=block_size，无SOS/EOS，不足用<PAD>填充
    - y: 编码后的摘要（summary），长度=block_size，带SOS（开头）和EOS（结尾），不足用<PAD>填充
    """
    def __init__(self, samples: List[Dict], stoi: Dict[str, int], block_size: int):
        self.samples = samples
        self.stoi = stoi
        self.block_size = block_size

        # 特殊标记ID（从词汇表获取，避免硬编码）
        self.pad_id = stoi["<PAD>"]
        self.sos_id = stoi["<SOS>"]
        self.eos_id = stoi["<EOS>"]

    def __len__(self) -> int:
        return len(self.samples)

    def _process_sequence(self, text: str, is_target: bool = False) -> torch.Tensor:
        """处理文本为ID序列：截断/填充，目标序列添加SOS/EOS"""
        # 过滤词汇表外的字符（避免KeyError）
        id_list = [self.stoi[char] for char in text if char in self.stoi]

        # 目标序列（摘要）添加SOS和EOS
        if is_target:
            id_list = [self.sos_id] + id_list + [self.eos_id]

        # 截断过长序列
        if len(id_list) > self.block_size:
            id_list = id_list[:self.block_size]
        # 填充不足序列
        else:
            id_list += [self.pad_id] * (self.block_size - len(id_list))

        return torch.tensor(id_list, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回单个样本：(原文编码x, 摘要编码y)"""
        sample = self.samples[idx]
        x = self._process_sequence(sample["article"], is_target=False)  # 原文→x
        y = self._process_sequence(sample["summary"], is_target=True)   # 摘要→y
        return x, y


def prepare_data(data_root: Path, overwrite_vocab: bool = False) -> Tuple[Dict, Dict, Dict]:
    """准备数据：加载本地样本→构建/加载词汇表"""
    # 1. 加载本地数据集样本
    data_splits = load_local_samples(data_root)

    # 2. 构建或加载词汇表
    vocab_dir = data_root / "vocab"  # 词汇表保存目录（在数据根目录下）
    if vocab_dir.exists() and not overwrite_vocab:
        print(f"[data.py] 加载已存在的词汇表：{vocab_dir}")
        stoi, itos = load_vocab(vocab_dir)
    else:
        print(f"[data.py] 基于所有样本构建词汇表...")
        # 合并所有split的样本用于构建词汇表（覆盖所有字符）
        all_samples = data_splits["train"] + data_splits["val"] + data_splits["test"]
        stoi, itos = build_vocab(all_samples)
        # 保存词汇表
        save_vocab(stoi, itos, vocab_dir)

    return data_splits, stoi, itos


def get_dataloaders(
    data_root: str = "data",
    batch_size: int = 32,
    block_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
    overwrite_vocab: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    返回训练/验证/测试DataLoader和元信息，与模型代码兼容
    data_root: 数据集根目录（包含giga_data/org_data的上级目录）
    """
    set_seed(seed)
    data_root = Path(data_root)

    # 准备数据和词汇表
    data_splits, stoi, itos = prepare_data(data_root, overwrite_vocab=overwrite_vocab)

    # 创建数据集
    train_dataset = Seq2SeqDataset(
        samples=data_splits["train"],
        stoi=stoi,
        block_size=block_size
    )
    val_dataset = Seq2SeqDataset(
        samples=data_splits["val"],
        stoi=stoi,
        block_size=block_size
    )
    test_dataset = Seq2SeqDataset(
        samples=data_splits["test"],
        stoi=stoi,
        block_size=block_size
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # 加速GPU传输（可选）
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 元信息（供模型训练代码使用）
    meta = {
        "vocab_size": len(stoi),
        "stoi": stoi,
        "itos": itos,
        "pad_id": stoi["<PAD>"],
        "sos_id": stoi["<SOS>"],
        "eos_id": stoi["<EOS>"],
        "block_size": block_size,
        "train_sample_count": len(data_splits["train"]),
        "val_sample_count": len(data_splits["val"]),
        "test_sample_count": len(data_splits["test"])
    }

    return train_loader, val_loader, test_loader, meta


def main():
    """CLI入口：验证本地数据加载"""
    parser = argparse.ArgumentParser(description="Gigaword本地数据集加载脚本")
    parser.add_argument("--data_root", type=str, default="data", 
                       help="数据集根目录（包含giga_data/org_data的上级目录）")
    parser.add_argument("--overwrite_vocab", action="store_true", 
                       help="强制重新构建词汇表（不影响原始数据）")
    parser.add_argument("--block_size", type=int, default=128, 
                       help="序列最大长度（适配模型输入）")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="测试用batch size")
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子")
    parser.add_argument("--num_workers", type=int, default=0, 
                       help="DataLoader多进程数（Windows建议0）")
    parser.add_argument("--test", action="store_true", 
                       help="测试数据加载：打印batch信息和样本示例")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.test:
        # 测试数据加载流程
        train_loader, val_loader, test_loader, meta = get_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            block_size=args.block_size,
            seed=args.seed,
            num_workers=args.num_workers,
            overwrite_vocab=args.overwrite_vocab
        )

        # 打印基本信息
        print("\n=== 数据加载测试结果 ===")
        print(f"词汇表大小：{meta['vocab_size']}")
        print(f"特殊标记ID：PAD={meta['pad_id']}, SOS={meta['sos_id']}, EOS={meta['eos_id']}")
        print(f"样本数量：train={meta['train_sample_count']}, val={meta['val_sample_count']}, test={meta['test_sample_count']}")

        # 打印第一个batch的形状
        batch = next(iter(train_loader))
        x, y = batch
        print(f"\nBatch 张量形状：x={x.shape}, y={y.shape}（预期：(batch_size, block_size)）")

        # 解码并打印第一个样本（原文+摘要）
        itos = meta["itos"]
        pad_id = meta["pad_id"]
        # 解码原文（x[0]）
        input_ids = x[0].tolist()
        input_text = "".join([itos[idx] for idx in input_ids if idx != pad_id])
        # 解码摘要（y[0]）
        target_ids = y[0].tolist()
        target_text = "".join([itos[idx] for idx in target_ids if idx != pad_id])

        print("\n=== 第一个样本示例 ===")
        print(f"原文（前200字符）：{input_text[:200]}...")
        print(f"摘要（前100字符）：{target_text[:100]}...")

    else:
        # 仅准备数据（加载样本+构建词汇表）
        prepare_data(Path(args.data_root), overwrite_vocab=args.overwrite_vocab)
        print("[data.py] 数据准备完成！可使用 --test 参数验证加载结果")


if __name__ == "__main__":
    main()
