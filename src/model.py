import torch.nn as nn
from mask import mask_pad, mask_tril
from util import MultiHead, PositionEmbedding, FullyConnectedOutput

class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_hidden=512, dropout=0.1, use_residual_norm=True):
        super().__init__()
        self.use_residual_norm = use_residual_norm
        self.mh = MultiHead(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.fc = FullyConnectedOutput(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        if self.use_residual_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        if self.use_residual_norm:
            # Pre-LN + 残差连接
            x_norm = self.norm1(x)
            sa = self.mh(x_norm, x_norm, x_norm, mask)
            x = x + self.dropout(sa)
            x_norm = self.norm2(x)
            ff = self.fc(x_norm)
            x = x + self.dropout(ff)
        else:
            # 消融：无残差+无归一化
            sa = self.mh(x, x, x, mask)
            x = self.dropout(sa)
            ff = self.fc(x)
            x = self.dropout(ff)
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers=2, d_model=128, n_heads=4, d_hidden=512, dropout=0.1, use_residual_norm=True):
        super().__init__()
        self.use_residual_norm = use_residual_norm
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, d_hidden=d_hidden, dropout=dropout, use_residual_norm=use_residual_norm)
            for _ in range(n_layers)
        ])
        if self.use_residual_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        if self.use_residual_norm:
            x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_hidden=512, dropout=0.1, use_residual_norm=True):
        super().__init__()
        self.use_residual_norm = use_residual_norm
        self.self_attn = MultiHead(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHead(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.fc = FullyConnectedOutput(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        if self.use_residual_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, y, mask_pad_x, mask_tril_y):
        if self.use_residual_norm:
            # Pre-LN + 残差连接
            y_norm = self.norm1(y)
            self_sa = self.self_attn(y_norm, y_norm, y_norm, mask_tril_y)
            y = y + self.dropout(self_sa)
            y_norm = self.norm2(y)
            cross = self.cross_attn(y_norm, x_enc, x_enc, mask_pad_x)
            y = y + self.dropout(cross)
            y_norm = self.norm3(y)
            ff = self.fc(y_norm)
            y = y + self.dropout(ff)
        else:
            # 消融：无残差+无归一化
            self_sa = self.self_attn(y, y, y, mask_tril_y)
            y = self.dropout(self_sa)
            cross = self.cross_attn(y, x_enc, x_enc, mask_pad_x)
            y = self.dropout(cross)
            ff = self.fc(y)
            y = self.dropout(ff)
        return y

class Decoder(nn.Module):
    def __init__(self, n_layers=2, d_model=128, n_heads=4, d_hidden=512, dropout=0.1, use_residual_norm=True):
        super().__init__()
        self.use_residual_norm = use_residual_norm
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, d_hidden=d_hidden, dropout=dropout, use_residual_norm=use_residual_norm)
            for _ in range(n_layers)
        ])
        if self.use_residual_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x_enc, y, mask_pad_x, mask_tril_y):
        for layer in self.layers:
            y = layer(x_enc, y, mask_pad_x, mask_tril_y)
        if self.use_residual_norm:
            y = self.norm(y)
        return y

class Transformer(nn.Module):
    def __init__(
        self, vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_hidden=512,
        dropout=0.1,
        pad_id=0,
        use_pos_emb=True,  # 控制位置编码
        single_head=False,  # 控制单头注意力
        use_residual_norm=True  # 控制残差+归一化
    ):
        super().__init__()
        self.use_pos_emb = use_pos_emb
        self.single_head = single_head
        self.use_residual_norm = use_residual_norm
        # 单头注意力消融
        self.n_heads = 1 if single_head else n_heads
        
        self.embed_x = PositionEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout=dropout,
            max_len=512,
            use_pos_emb=use_pos_emb
        )
        self.embed_y = PositionEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            dropout=dropout,
            max_len=512,
            use_pos_emb=use_pos_emb
        )
        self.encoder = Encoder(
            n_layers=n_layers, d_model=d_model, n_heads=self.n_heads, d_hidden=d_hidden, 
            dropout=dropout, use_residual_norm=use_residual_norm
        )
        self.decoder = Decoder(
            n_layers=n_layers, d_model=d_model, n_heads=self.n_heads, d_hidden=d_hidden, 
            dropout=dropout, use_residual_norm=use_residual_norm
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_id = pad_id
        self.mask_pad = mask_pad
        self.mask_tril = mask_tril

    def forward(self, x, y):
        mask_pad_x = mask_pad(x, self.pad_id)
        mask_tril_y = mask_tril(y, self.pad_id)
        x = self.embed_x(x)
        y = self.embed_y(y)
        x_enc = self.encoder(x, mask_pad_x)
        y_dec = self.decoder(x_enc, y, mask_pad_x, mask_tril_y)
        logits = self.fc_out(y_dec)
        return logits