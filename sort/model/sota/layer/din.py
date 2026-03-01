import torch
import torch.nn as nn
import math


class MultiHeadTargetAttention(nn.Module):
    """ 移除 Softmax 的多头目标注意力机制 (DIN-style) """

    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super(MultiHeadTargetAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"

        # 线性映射层
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, target_item, seq_items, mask):
        """
        :param target_item: 目标候选物品 [B, 1, E]
        :param seq_items:   历史行为序列 [B, L, E]
        :param mask:        有效长度布尔开关 [B, L]，True 表示有效，False 表示 Padding
        """
        B, L, _ = seq_items.size()

        # 1. 线性变换并拆分为多头
        q = self.q_proj(target_item).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        k = self.k_proj(seq_items).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        v = self.v_proj(seq_items).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]

        # 2. 计算 Attention 原始分数
        # [B, H, 1, D] x [B, H, D, L] -> [B, H, 1, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. 核心改动：用布尔开关代替数值判断，并移除 Softmax 归一化
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # 扩展为 [B, 1, 1, L]

        # 直接使用外部传入的 mask 作为绝对开关：False 的位置强制置为 0，不再参与后续加权
        scores = scores.masked_fill(~mask_expanded, 0.0)

        # 在此处可以加一个 ReLU，过滤掉与目标物品负相关的历史行为
        scores = torch.relu(scores)

        attn_weights = self.dropout(scores)

        # 4. 加权求和 (由于没有 Softmax，这里的加权是强度的绝对累加)
        # [B, H, 1, L] x [B, H, L, D] -> [B, H, 1, D]
        attn_output = torch.matmul(attn_weights, v)

        # 5. 拼接多头并输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.emb_dim)
        output = self.out_proj(attn_output)

        output = self.layer_norm(output)
        hist_emb = output.squeeze(1)  # [B, E]

        return hist_emb


# class MultiHeadTargetAttention(nn.Module):
#     """ 基于多头 Cross-Attention 的目标注意力机制 """
#
#     def __init__(self, emb_dim, num_heads=4, dropout=0.1):
#         super(MultiHeadTargetAttention, self).__init__()
#         # batch_first=True 方便直接处理 [Batch, SeqLen, EmbDim] 格式的 Tensor
#         self.attention = nn.MultiheadAttention(
#             embed_dim=emb_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.layer_norm = nn.LayerNorm(emb_dim)
#
#     def forward(self, target_item, seq_items, mask):
#         """
#         :param target_item: 目标候选物品 [B, 1, E]  作为 Query
#         :param seq_items:   历史行为序列 [B, L, E]  作为 Key 和 Value
#         :param mask:        有效长度掩码 [B, L]，True 表示有效，False 表示 Padding
#         """
#         # PyTorch MultiheadAttention 的 key_padding_mask 规则：
#         # 值为 True 的位置会被当作 padding 忽略掉，所以我们需要对原 mask 取反 (~)
#         key_padding_mask = ~mask  # [B, L]
#         all_padded = key_padding_mask.all(dim=1)  # [B]
#         key_padding_mask[all_padded, 0] = False
#
#         # Cross-Attention 核心计算
#         # Query: 目标物品, Key/Value: 历史序列
#         attn_output, attn_weights = self.attention(
#             query=target_item,
#             key=seq_items,
#             value=seq_items,
#             key_padding_mask=key_padding_mask
#         )  # attn_output: [B, 1, E]
#
#         output = self.layer_norm(attn_output)
#
#         # 消除 sequence 的维度，输出聚合成的用户向量
#         hist_emb = output.squeeze(1)  # [B, E]
#         return hist_emb