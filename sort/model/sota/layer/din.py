import torch
import torch.nn as nn


class MultiHeadTargetAttention(nn.Module):
    """ 基于多头 Cross-Attention 的目标注意力机制 """

    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super(MultiHeadTargetAttention, self).__init__()
        # batch_first=True 方便直接处理 [Batch, SeqLen, EmbDim] 格式的 Tensor
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, target_item, seq_items, mask):
        """
        :param target_item: 目标候选物品 [B, 1, E]  作为 Query
        :param seq_items:   历史行为序列 [B, L, E]  作为 Key 和 Value
        :param mask:        有效长度掩码 [B, L]，True 表示有效，False 表示 Padding
        """
        # PyTorch MultiheadAttention 的 key_padding_mask 规则：
        # 值为 True 的位置会被当作 padding 忽略掉，所以我们需要对原 mask 取反 (~)
        key_padding_mask = ~mask  # [B, L]
        all_padded = key_padding_mask.all(dim=1)  # [B]
        key_padding_mask[all_padded, 0] = False

        # Cross-Attention 核心计算
        # Query: 目标物品, Key/Value: 历史序列
        attn_output, attn_weights = self.attention(
            query=target_item,
            key=seq_items,
            value=seq_items,
            key_padding_mask=key_padding_mask
        )  # attn_output: [B, 1, E]

        output = self.layer_norm(attn_output)

        # 消除 sequence 的维度，输出聚合成的用户向量
        hist_emb = output.squeeze(1)  # [B, E]
        return hist_emb