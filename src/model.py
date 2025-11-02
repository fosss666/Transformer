import torch.nn as nn

from mask import mask_pad, mask_tril
from util import MultiHead, PositionEmbedding, FullyConnectedOutput


class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_hidden=512, dropout=0.1):  # 对齐作业表3超参
        super().__init__()
        self.mh = MultiHead(d_model=d_model, n_heads=n_heads, dropout=dropout)  # 传递多头参数
        self.fc = FullyConnectedOutput(d_model=d_model, d_hidden=d_hidden, dropout=dropout)  # 传递FFN参数
        # 消融3
        # self.norm1 = nn.LayerNorm(d_model)  # Pre-LN归一化
        # self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-LN风格：先归一化再通过子层（更稳定） 消融3
        # x_norm = self.norm1(x)
        # sa = self.mh(x_norm, x_norm, x_norm, mask)  # 自注意力
        sa = self.mh(x, x, x, mask)  # 直接用原始x计算注意力（无归一化）
        
        # x = x + self.dropout(sa)  # 残差连接 消融3
        x=self.dropout(sa)

        # 消融3
        # x_norm = self.norm2(x)
        # ff = self.fc(x_norm)  # FFN
        ff=self.fc(x)
        
        # x = x + self.dropout(ff)  # 残差连接 消融3
        x=self.dropout(ff)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers=2, d_model=128, n_heads=4, d_hidden=512, dropout=0.1):  # 对齐作业表3
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, d_hidden=d_hidden, dropout=dropout)
            for _ in range(n_layers)
        ])
        # self.norm = nn.LayerNorm(d_model)  # 最终输出归一化 消融3

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        # return self.norm(x)  # 最终归一化 消融3
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_hidden=512, dropout=0.1):  # 对齐作业表3
        super().__init__()
        self.self_attn = MultiHead(d_model=d_model, n_heads=n_heads, dropout=dropout)  # Decoder自注意力
        self.cross_attn = MultiHead(d_model=d_model, n_heads=n_heads, dropout=dropout)  # 交叉注意力
        self.fc = FullyConnectedOutput(d_model=d_model, d_hidden=d_hidden, dropout=dropout)  # FFN
        # 消融3
        # self.norm1 = nn.LayerNorm(d_model)  # Pre-LN归一化
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, y, mask_pad_x, mask_tril_y):
        # 1. Decoder自注意力（屏蔽未来token）消融3
        # y_norm = self.norm1(y)
        # self_sa = self.self_attn(y_norm, y_norm, y_norm, mask_tril_y)
        # y = y + self.dropout(self_sa)  # 残差连接
        
        self_sa = self.self_attn(y, y, y, mask_tril_y)  # 直接用原始y计算
        y = self.dropout(self_sa)  # 仅保留自注意力输出+dropout

        # 2. 交叉注意力（与Encoder输出交互）消融3
        # y_norm = self.norm2(y)
        # cross = self.cross_attn(y_norm, x_enc, x_enc, mask_pad_x)  # Q=Decoder输出, K/V=Encoder输出
        # y = y + self.dropout(cross)  # 残差连接
        
        cross = self.cross_attn(y, x_enc, x_enc, mask_pad_x)  # 直接用自注意力输出y计算
        y = self.dropout(cross)  # 仅保留交叉注意力输出+dropout

        # 3. FFN
        # y_norm = self.norm3(y)
        # ff = self.fc(y_norm)
        # y = y + self.dropout(ff)  # 残差连接
        
        ff = self.fc(y)  # 直接用交叉注意力输出y计算
        y = self.dropout(ff)  # 仅保留FFN输出+dropout
        return y


class Decoder(nn.Module):
    def __init__(self, n_layers=2, d_model=128, n_heads=4, d_hidden=512, dropout=0.1):  
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, d_hidden=d_hidden, dropout=dropout)
            for _ in range(n_layers)
        ])
        # self.norm = nn.LayerNorm(d_model)  # 最终输出归一化 消融3

    def forward(self, x_enc, y, mask_pad_x, mask_tril_y):
        for layer in self.layers:
            y = layer(x_enc, y, mask_pad_x, mask_tril_y)
        # return self.norm(y)  # 最终归一化 消融3
        return y

 
class Transformer(nn.Module):
    def __init__(
                self, vocab_size,
                d_model=128,
                n_layers=2,
                n_heads=4,
                d_hidden=512,
                dropout=0.1,
                pad_id=0
        ):
        super().__init__()
        self.embed_x = PositionEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout=dropout,
            max_len=512
        )
        self.embed_y = PositionEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout=dropout,
            max_len=512
        )
        self.encoder = Encoder(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_hidden=d_hidden, dropout=dropout
        )
        self.decoder = Decoder(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_hidden=d_hidden, dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_id = pad_id

        self.mask_pad = mask_pad 
        self.mask_tril = mask_tril 
        

    def forward(self, x, y):
        # 生成掩码
        mask_pad_x = mask_pad(x, self.pad_id)    # 输入序列填充掩码
        mask_tril_y = mask_tril(y, self.pad_id)  # Decoder自注意力掩码

        # 嵌入层（token嵌入+位置编码）
        x = self.embed_x(x)  # [batch_size, seq_len, d_model]
        y = self.embed_y(y)  # [batch_size, seq_len, d_model]

        # Encoder编码
        x_enc = self.encoder(x, mask_pad_x)  # [batch_size, seq_len, d_model]

        # Decoder解码
        y_dec = self.decoder(x_enc, y, mask_pad_x, mask_tril_y)  # [batch_size, seq_len, d_model]

        # 输出预测logits
        logits = self.fc_out(y_dec)  # [batch_size, seq_len, vocab_size]
        return logits