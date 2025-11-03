import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ensure_bool_mask(mask: Optional[torch.Tensor], device: torch.device):
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        return mask.bool()
    return mask

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None):
    scores = torch.matmul(Q, K.transpose(-2, -1))
    head_dim = Q.size(-1)
    scores = scores / math.sqrt(head_dim)
    mask = _ensure_bool_mask(mask, scores.device)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-1e9"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out

class MultiHead(nn.Module):
    def __init__(self, d_model: int = 32, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.fc_Q = nn.Linear(d_model, d_model)
        self.fc_K = nn.Linear(d_model, d_model)
        self.fc_V = nn.Linear(d_model, d_model)
        self.out_fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, L_q, _ = Q.size()
        _, L_k, _ = K.size()
        residual = Q
        Qn = self.norm(Q)
        Kn = self.norm(K)
        Vn = self.norm(V)
        Qp = self.fc_Q(Qn)
        Kp = self.fc_K(Kn)
        Vp = self.fc_V(Vn)
        Qh = Qp.view(B, L_q, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Kh = Kp.view(B, L_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Vh = Vp.view(B, L_k, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_out = attention(Qh, Kh, Vh, mask)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        out = self.out_fc(attn_out)
        out = self.dropout(out)
        return residual + out

class PositionEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 39, d_model: int = 32, max_len: int = 50, dropout: float = 0.0, use_pos_emb: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.use_pos_emb = use_pos_emb  # 位置编码开关
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed.weight.data.normal_(0, 0.1)
        # 预计算位置编码
        if self.use_pos_emb:
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
        B, L = x.size()
        assert L <= self.max_len, f"序列长度 L={L} 超过 max_len={self.max_len}"
        emb = self.embed(x)
        # 根据开关决定是否添加位置编码
        if self.use_pos_emb:
            emb = emb + self.pe[:, :L, :].to(emb.device)
        return self.dropout(emb)

class FullyConnectedOutput(nn.Module):
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

def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)