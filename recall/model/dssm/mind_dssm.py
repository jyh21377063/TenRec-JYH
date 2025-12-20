import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 基础组件
# ==========================================

class MultiInterestExtractor(nn.Module):
    """
    来自 comirec.py 的胶囊网络核心，用于从序列提取 K 个兴趣
    """

    def __init__(self, embed_dim, num_interests, routing_iters=3):
        super(MultiInterestExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.num_interests = num_interests
        self.routing_iters = routing_iters
        self.W_s = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, behavior_embeds, mask=None):
        B, L, D = behavior_embeds.shape
        u_hat = self.W_s(behavior_embeds)
        b = torch.randn(B, self.num_interests, L).to(behavior_embeds.device)

        if mask is not None:
            mask = mask.unsqueeze(1).float()
            b = b.masked_fill(mask == 0, -1e9)

        c = None
        for i in range(self.routing_iters):
            w = F.softmax(b, dim=2)
            z = torch.matmul(w, u_hat)
            c = self.squash(z)
            if i < self.routing_iters - 1:
                delta_b = torch.matmul(c, u_hat.transpose(1, 2))
                b = b + delta_b
        return c

    def squash(self, z):
        vec_norm_sq = torch.sum(z ** 2, dim=-1, keepdim=True)
        vec_norm = torch.sqrt(vec_norm_sq + 1e-8)
        return (vec_norm_sq / (1 + vec_norm_sq)) * (z / vec_norm)


# 简化的 Transformer Block (为了代码完整性)
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                           dim_feedforward=embed_dim * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        # PyTorch Transformer mask 逻辑: True 代表需要忽略 (Padding)
        # 这里的 mask 输入如果是 1 代表有效，0 代表 padding，需要反转
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


# ==========================================
# 2. 核心：门控融合层 (Gated Feature Fusion)
# ==========================================

class GatedFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(GatedFeatureFusion, self).__init__()

        # 门控网络：计算每个特征向量的权重
        # 输入是所有特征向量的拼接，这里我们假设输入特征都已经投影到了 input_dim
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, feature_list):
        """
        feature_list: List of tensors, each shape [B, D]
        """
        # 1. 堆叠: [B, Num_Features, D]
        # Num_Features = K (interests) + P (profiles) + C (contexts)
        stack_feats = torch.stack(feature_list, dim=1)

        # 2. 计算 Attention Scores (Self-Attention within features)
        # 我们希望网络根据所有特征的上下文，决定谁重要
        # 简单的做法是对每个特征独立打分，复杂做法是 Cross-Attention
        # 这里使用一种 Context-Aware Gating:

        # 全局上下文 (Global Context): 对所有特征取平均，作为一个粗略的 Context
        global_context = torch.mean(stack_feats, dim=1, keepdim=True)  # [B, 1, D]

        # 每个特征结合全局上下文计算权重
        # score = MLP(feat + global_context)
        scores = self.gate_net(stack_feats + global_context)  # [B, N, 1]

        weights = F.softmax(scores, dim=1)  # [B, N, 1]

        # 3. 加权融合
        # [B, N, D] * [B, N, 1] -> Sum -> [B, D]
        weighted_feats = torch.sum(stack_feats * weights, dim=1)

        return self.norm(weighted_feats + self.dropout(weighted_feats))


# ==========================================
# 3. 融合模型：ComiRec + Static + Gating
# ==========================================

class ComiRecGatedFusion(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_interests=4):
        super(ComiRecGatedFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_interests = num_interests
        self.seq_len = meta_info['seq_len']

        # --- Embeddings ---
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.seq_len, embed_dim)

        # 静态特征 Embedding
        self.age_emb = nn.Embedding(120, embed_dim)
        self.gender_emb = nn.Embedding(5, embed_dim)

        # --- 序列建模 (Transformer + Multi-Interest) ---
        self.transformer = TransformerEncoder(embed_dim, num_layers=2, num_heads=2, dropout=0.1)

        self.interest_extractor = MultiInterestExtractor(embed_dim, num_interests=num_interests)

        # --- Item Tower ---
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.enhance_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def forward_user_tower(self, batch):
        # 1. 序列特征处理
        item_seq = batch['item_id_seq']
        cat_seq = batch['video_category_seq']
        mask = (item_seq != 0).long()  # [B, L]

        # Embedding + Positional
        seq_emb = self.item_emb(item_seq) + self.cat_emb(cat_seq)
        positions = torch.arange(self.seq_len, device=item_seq.device).unsqueeze(0)
        seq_emb += self.pos_emb(positions)

        # Transformer Encoding
        transformer_out = self.transformer(seq_emb, mask=mask)  # [B, L, D]

        # Multi-Interest Extraction (MIND核心)
        # interests: [B, K, D]
        interests = self.interest_extractor(transformer_out, mask=mask)  # [B, K, D]

        # 2. 算出静态/画像特征 [B, D]
        e_age = self.age_emb(batch['age'])  # [B, D] (注意确保维度一致)
        e_gender = self.gender_emb(batch['gender'])  # [B, D]

        # 简单的做法：把静态特征相加融合 (或者这里也可以用 Attention 融合)
        profile_vec = e_age + e_gender  # [B, D]

        # 3. 【修正点】利用广播机制注入特征，而不是求和
        # 将 profile_vec 扩展为 [B, 1, D] 以便和 [B, K, D] 进行操作
        profile_vec_expanded = profile_vec.unsqueeze(1).expand(-1, self.num_interests, -1)

        # 方案 A: 直接相加 (残差连接的思想)
        # enhanced_interests = interests + profile_vec_expanded

        # 方案 B (更强): 拼接后过 MLP (保留更多非线性交互)
        # cat([B, K, D], [B, K, D]) -> [B, K, 2D]
        concat_feats = torch.cat([interests, profile_vec_expanded], dim=-1)

        # [B, K, 2D] -> [B, K, D]
        enhanced_interests = self.enhance_mlp(concat_feats)

        # 4. 返回 K 个向量
        return enhanced_interests

    def forward_item_tower(self, batch):
        item_id = batch['item_id']
        cat_id = batch['video_category']
        pop = batch['item_pop_norm']

        vec = torch.cat([self.item_emb(item_id), self.cat_emb(cat_id), pop], dim=1)
        return self.item_mlp(vec)

    def forward(self, batch):
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # Normalize
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb