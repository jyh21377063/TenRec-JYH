import torch
import torch.nn as nn
import torch.nn.functional as F

from sasrec import LayerNorm,TransformerBlock


class MultiInterestExtractor(nn.Module):
    def __init__(self, embed_dim, num_interests, max_len, routing_iters=3):
        super(MultiInterestExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.num_interests = num_interests
        self.max_len = max_len
        self.routing_iters = routing_iters

        # 论文公式 (8): Bilinear Mapping Matrix S
        # 这是一个把行为 embedding 映射到兴趣空间的矩阵
        self.W_s = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, behavior_embeds, mask=None):
        """
        behavior_embeds: [B, L, D] (经过 Transformer 或直接 Embedding 的序列)
        mask: [B, L] (Padding mask, 0为padding, 1为有效)
        """
        B, L, D = behavior_embeds.shape

        # 1. 线性映射 [B, L, D]
        # 对应论文中的 u_hat (或者是公式8中的 S * e_i)
        u_hat = self.W_s(behavior_embeds)

        # 2. 初始化 Routing Logits b_ij [B, num_interests, L]
        # 论文提到使用高斯分布初始化有助于破坏对称性 [cite: 215]
        b = torch.randn(B, self.num_interests, L).to(behavior_embeds.device)

        # 如果有 mask，先把 padding 位置的 logit 设为负无穷，防止 softmax 选中它们
        if mask is not None:
            # mask: [B, L] -> [B, 1, L]
            mask = mask.unsqueeze(1).float()
            b = b.masked_fill(mask == 0, -1e9)

        # 3. Dynamic Routing 迭代循环 [cite: 234]
        # c: [B, K, D] (Interest Capsules)
        c = None
        for i in range(self.routing_iters):
            # Softmax 计算耦合系数 w_ij [cite: 190]
            # [B, K, L]
            w = F.softmax(b, dim=2)

            # 加权求和得到候选胶囊 z_j [cite: 187]
            # [B, K, L] x [B, L, D] -> [B, K, D]
            z = torch.matmul(w, u_hat)

            # Squash 非线性激活 [cite: 193]
            c = self.squash(z)

            # 更新 logits b_ij (除了最后一次迭代) [cite: 242]
            if i < self.routing_iters - 1:
                # [B, K, D] x [B, D, L] -> [B, K, L]
                delta_b = torch.matmul(c, u_hat.transpose(1, 2))
                b = b + delta_b

        return c  # 返回 [B, K, D]

    def squash(self, z):
        # 论文公式 (7): Squash function
        vec_norm_sq = torch.sum(z ** 2, dim=-1, keepdim=True)
        vec_norm = torch.sqrt(vec_norm_sq + 1e-8)
        return (vec_norm_sq / (1 + vec_norm_sq)) * (z / vec_norm)


class LabelAwareAttention(nn.Module):
    def __init__(self, embed_dim, pow_p=1.0):
        super(LabelAwareAttention, self).__init__()
        self.pow_p = pow_p  # 论文中的指数 p，调节注意力分布 [cite: 251]

    def forward(self, user_interests, item_embed):
        """
        user_interests: [B, K, D] (用户的 K 个兴趣胶囊)
        item_embed:     [B, D]    (当前正样本 Item 的 Embedding)
        """
        # 计算 Attention Scores (Dot Product)
        # [B, K, D] * [B, D, 1] -> [B, K, 1]
        attention_logits = torch.matmul(user_interests, item_embed.unsqueeze(-1)).squeeze(-1)

        # 论文公式 (11): Softmax(pow(V^T e_i, p))
        attention_weights = F.softmax(torch.pow(attention_logits, self.pow_p), dim=1)

        # 加权求和
        # [B, 1, K] * [B, K, D] -> [B, 1, D] -> [B, D]
        user_vector = torch.matmul(attention_weights.unsqueeze(1), user_interests).squeeze(1)

        return user_vector


class MINDModel(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=2,
                 dropout=0.1, num_interests=4):
        super(MINDModel, self).__init__()
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

        self.num_interests = num_interests
        self.capsule_net = MultiInterestExtractor(embed_dim, num_interests,
                                                  meta_info['seq_len'])

        # Label-Aware Attention (仅用于训练)
        self.label_attn = LabelAwareAttention(embed_dim, pow_p=1.0)

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
        # 这里的 mask 逻辑要小心，SASRec 是取最后一个 token
        # MIND 是对整个序列聚类，所以要把 padding 去掉
        padding_mask = (batch['item_id_seq'] != 0).long()  # [B, L]

        # 调用胶囊网络提取 K 个兴趣
        # user_interests: [B, K, D]
        user_interests = self.capsule_net(x, mask=padding_mask)

        return user_interests

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
        u_multi_emb = self.forward_user_tower(batch)  # [B, K, D]
        i_emb = self.forward_item_tower(batch)  # [B, D]
        i_emb = F.normalize(i_emb, p=2, dim=1)

        if self.training:
            # === 训练阶段 ===
            # 计算每个兴趣胶囊与 Target Item 的相似度 [B, K]
            att_scores = torch.matmul(u_multi_emb, i_emb.unsqueeze(-1)).squeeze(-1)

            # --- 改进点：使用 Hard Selection ---
            # 选出最匹配的那个兴趣的索引
            hard_indices = torch.argmax(att_scores, dim=1)  # [B]

            # 取出对应的兴趣向量 [B, D]
            # 这种方式强制只有最接近真值的那个兴趣胶囊会被更新，促使其专门化
            u_emb = u_multi_emb[torch.arange(u_multi_emb.size(0)), hard_indices]

            # 如果想保留 Soft 模式，可以设一个参数切换
            # u_emb = self.label_attn(u_multi_emb, i_emb)

            u_emb = F.normalize(u_emb, p=2, dim=1)
            return u_emb, i_emb