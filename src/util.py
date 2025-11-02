# src/util.py
"""
更通用、稳健的 Transformer 工具模块
- 支持可配置 d_model, n_heads, vocab_size, max_len, dropout
- attention 支持可广播的 mask
- 返回的张量 shape 都是动态的（不再硬编码 50/32 等）
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_bool_mask(mask: Optional[torch.Tensor], device: torch.device):
    """
    将 mask 转为 bool 类型，若为 None 则返回 None。
    mask 的含义：True 表示该位置应被屏蔽（会被替换为 -inf）
    """
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        return mask.bool()
    return mask


def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    Scaled dot-product attention.
    Q: (B, n_heads, L_q, head_dim)
    K: (B, n_heads, L_k, head_dim)
    V: (B, n_heads, L_k, head_dim)
    mask: broadcastable to (B, n_heads, L_q, L_k), bool where True = masked
    returns: (B, n_heads, L_q, head_dim)
    """
    # scores: (B, n_heads, L_q, L_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    head_dim = Q.size(-1)
    scores = scores / math.sqrt(head_dim)

    mask = _ensure_bool_mask(mask, scores.device)
    if mask is not None:
        # masked_fill requires mask broadcastable to scores
        scores = scores.masked_fill(mask, float("-1e9"))

    attn = torch.softmax(scores, dim=-1)  # (B, n_heads, L_q, L_k)
    out = torch.matmul(attn, V)  # (B, n_heads, L_q, head_dim)
    return out


class MultiHead(nn.Module):
    """
    Multi-head attention (with residual inside this module to match your original style).
    参数化：d_model, n_heads, dropout
    输入/输出 shape 约定与模型其余部分保持一致：
      - 接受 Q,K,V: (B, L, d_model)
      - 返回: (B, L, d_model)
    注意：residual 是在 module 内做，基于输入 Q。
    """
    def __init__(self, d_model: int = 32, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # projections
        self.fc_Q = nn.Linear(d_model, d_model)
        self.fc_K = nn.Linear(d_model, d_model)
        self.fc_V = nn.Linear(d_model, d_model)
        self.out_fc = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Q,K,V: (B, L, d_model)
        mask: bool, broadcastable to (B, n_heads, L_q, L_k) or (B, 1, 1, L_k) etc.
        """
        B, L_q, _ = Q.size()
        _, L_k, _ = K.size()

        residual = Q

        # Optionally normalize inputs (Pre-norm style inside this module)
        Qn = self.norm(Q)
        Kn = self.norm(K)
        Vn = self.norm(V)

        # linear projections
        Qp = self.fc_Q(Qn)  # (B, L_q, d_model)
        Kp = self.fc_K(Kn)  # (B, L_k, d_model)
        Vp = self.fc_V(Vn)  # (B, L_k, d_model)

        # reshape for heads: (B, n_heads, L, head_dim)
        Qh = Qp.view(B, L_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Kh = Kp.view(B, L_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Vh = Vp.view(B, L_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # attention: returns (B, n_heads, L_q, head_dim)
        attn_out = attention(Qh, Kh, Vh, mask)

        # combine heads -> (B, L_q, d_model)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)

        out = self.out_fc(attn_out)
        out = self.dropout(out)

        # residual
        return residual + out


class PositionEmbedding(nn.Module):
    """
    词嵌入+位置编码
    """
    def __init__(self, vocab_size: int = 39, d_model: int = 32, max_len: int = 50, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # 词嵌入
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight.data.normal_(0, 0.1)

        # 位置编码！！消融1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer("pe", pe)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: (B, L) token ids
        B, L = x.size()
        assert L <= self.max_len, f"序列长度 L={L} 超过 max_len={self.max_len}"
        emb = self.embed(x)  # (B, L, d_model)
        emb = emb + self.pe[:, :L, :].to(emb.device)# 消融1
        return self.dropout(emb)


class FullyConnectedOutput(nn.Module):
    """
    两层 FFN + residual + LayerNorm（参数化）
    """
    def __init__(self, d_model: int = 32, d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm(x)
        out = self.ff(x)
        out = self.dropout(out)
        return residual + out


# 统计参数量
def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
