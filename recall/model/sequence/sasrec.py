import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, L, H]
        B, L, H = hidden_states.size()

        query_layer = self.query(hidden_states).view(B, L, self.num_heads, self.head_size).transpose(1, 2)
        key_layer = self.key(hidden_states).view(B, L, self.num_heads, self.head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(B, L, self.num_heads, self.head_size).transpose(1, 2)

        # Attention Scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)

        # Apply Mask (causal mask + padding mask)
        # mask is 0 for valid, -1e9 for invalid
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous().view(B, L, H)

        return context_layer


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Linear(hidden_size, hidden_size * 4)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Linear(hidden_size * 4, hidden_size)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(hidden_size, num_heads, dropout_prob)
        self.feed_forward = PointWiseFeedForward(hidden_size, dropout_prob)
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, attention_mask):
        # Sub-layer 1: Self Attention
        normed_hidden_states = self.norm1(hidden_states)
        attention_output = self.attention(normed_hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout(attention_output)

        # Sub-layer 2: Feed Forward
        normed_hidden_states = self.norm2(hidden_states)
        feed_forward_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(feed_forward_output)

        return hidden_states



class SASRecRecallModel(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1):
        super(SASRecRecallModel, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = meta_info['seq_len']

        # --- Embeddings (User Tower) ---
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # 可选：融合其他序列特征 (Category 等)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)

        # --- Transformer Encoder ---
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNorm(embed_dim)

        # --- Item Tower (Simple Embedding) ---
        # 复用 item_emb 和 cat_emb 以减少参数并对齐空间
        # 也可以给 Item Tower 加上 MLP
        self.item_dense = nn.Linear(embed_dim * 2 + 1, embed_dim)  # Item + Cat + Pop

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """生成因果 Mask (Causal Mask) + Padding Mask"""
        # item_seq: [B, L]
        # 1. Padding Mask: padding 的位置设为 1 (需要 mask 掉), 有效位置设为 0
        padding_mask = (item_seq == 0).bool()  # [B, L]

        # 2. Causal Mask: 未来的信息不能看 (下三角矩阵)
        # timeline_mask: [L, L] -> 下三角为 0 (可见), 上三角为 1 (不可见)
        # torch.triu 也可以，这里我们用逻辑：mask[i, j] = 1 if j > i
        batch_size, seq_len = item_seq.size()

        # [L, L]
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=item_seq.device), diagonal=1).bool()

        # 合并 Mask: [B, 1, L, L] 用于 Multi-head Attention
        # 任何一个 mask 为 True 的地方都需要被屏蔽
        extended_mask = padding_mask.unsqueeze(1).unsqueeze(2) | causal_mask.unsqueeze(0).unsqueeze(0)

        # 转换为加法 Mask: Mask 处为 -1e9, 否则为 0
        final_mask = torch.zeros_like(extended_mask, dtype=torch.float)
        final_mask = final_mask.masked_fill(extended_mask, -1e9)

        return final_mask

    def forward_user_tower(self, batch):
        # 1. Inputs
        item_seq = batch['item_id_seq']  # [B, L]
        cat_seq = batch['video_category_seq']  # [B, L]

        # 2. Embeddings
        # Item Embedding
        seqs = self.item_emb(item_seq)  # [B, L, D]
        # 加上 Category Embedding (简单的相加融合，效果通常不错)
        seqs += self.cat_emb(cat_seq)

        # Positional Embedding
        positions = torch.arange(self.seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)
        seqs += self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)

        # 3. Transformer Layers
        # 获取 Mask: 屏蔽 Padding 和未来信息
        mask = self.get_attention_mask(item_seq)

        # 仅在非 Padding 区域进行 LayerNorm (对于 Transformer 来说通常输入已经是 0 可以忽略，但加上 mask 更安全)
        # 但标准的 SASRec 实现中，padding 的 embedding 也是 0，经过计算后可能会有值，所以 mask 很重要

        x = seqs
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)  # [B, L, D]

        # 4. 获取用户向量
        # 只有序列中“最后一个非 Padding 的物品”对应的输出才是我们需要的 User Embedding
        # 假设数据处理时是 Post-Padding (右侧补0) 或 Pre-Padding (左侧补0)
        # 你的 dataset.py 是 Pre-Padding (0, 0, ..., item1, item2) 还是 Post?
        # 看代码 dataset.py 里 `feature_matrix[:, seq_len - i]` 逻辑，且 `np.roll`，看起来是把历史推到了最后。
        # 也就是说，序列的最后一个位置 (index -1) 总是最新的行为（或者是 padding 如果序列全空）。
        # 直接取最后一个位置即可。

        user_embedding = x[:, -1, :]  # [B, D]

        return user_embedding

    def forward_item_tower(self, batch):
        # 与 DSSM 保持一致，或者是简单的 Embedding 查表
        item_id = batch['item_id']
        cat_id = batch['video_category']
        pop_norm = batch['item_pop_norm']  # [B, 1]

        e_item = self.item_emb(item_id)
        e_cat = self.cat_emb(cat_id)

        # 拼接 Dense 特征
        item_vec = torch.cat([e_item, e_cat, pop_norm], dim=1)

        # 简单的线性映射对齐维度
        item_vec = self.item_dense(item_vec)

        return item_vec

    def forward(self, batch):
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # 归一化 (Cosine Similarity 必需)
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb