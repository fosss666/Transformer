# mask.py 移除错误导入，修改函数参数
import torch

# 1. 修改mask_pad：接收pad_id参数，动态获取序列长度
# mask.py
def mask_pad(data: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    填充掩码：屏蔽<PAD>位置，适配任意查询长度
    input: data (B, L_k) → output: mask (B, 1, 1, L_k)（支持广播到(B, n_heads, L_q, L_k)）
    """
    # 标记PAD位置（True=需屏蔽）
    pad_mask = (data == pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)
    return pad_mask  # 不再扩展为(L_k, L_k)，保留可广播形状

# 2. 修改mask_tril：接收pad_id参数，动态获取序列长度
def mask_tril(data, pad_id):
    # data: [B, L]
    L = data.size(1)
    # 1. 生成未来掩码（下三角掩码，上三角为1表示需屏蔽）
    tril = 1 - torch.tril(torch.ones(1, L, L, dtype=torch.long, device=data.device))  # [1, L, L]
    # 2. 生成padding掩码
    pad_mask = (data == pad_id).unsqueeze(1).long()  # [B, 1, L]
    # 3. 合并掩码（padding或未来位置均需屏蔽，求和后>0即为True）
    mask = pad_mask + tril  # [B, L, L]（广播：[B,1,L] + [1,L,L] → [B,L,L]）
    mask = mask > 0  # 转为布尔型，True表示需屏蔽
    # 调整形状为[B, 1, L, L]，兼容MultiHead的[B, n_heads, L, L]广播需求
    mask = mask.unsqueeze(1)  # [B, 1, L, L]
    return mask