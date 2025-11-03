import torch
def mask_pad(data: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    填充掩码：屏蔽<PAD>位置，适配任意查询长度
    input: data (B, L_k) → output: mask (B, 1, 1, L_k)）
    """
    # 标记PAD位置
    pad_mask = (data == pad_id).unsqueeze(1).unsqueeze(2)
    return pad_mask

# 修改mask_tril，接收pad_id参数，动态获取序列长度
def mask_tril(data, pad_id):
    L = data.size(1)
    # 未来掩码
    tril = 1 - torch.tril(torch.ones(1, L, L, dtype=torch.long, device=data.device))
    # padding掩码
    pad_mask = (data == pad_id).unsqueeze(1).long()
    mask = pad_mask + tril
    mask = mask > 0
    mask = mask.unsqueeze(1)
    return mask