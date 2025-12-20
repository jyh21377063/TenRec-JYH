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
        B, L, H = hidden_states.size()
        query_layer = self.query(hidden_states).view(B, L, self.num_heads, self.head_size).transpose(1, 2)
        key_layer = self.key(hidden_states).view(B, L, self.num_heads, self.head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(B, L, self.num_heads, self.head_size).transpose(1, 2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)

        # 加上 mask (mask 值为 0 或 -1e9)
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
        normed_hidden_states = self.norm1(hidden_states)
        attention_output = self.attention(normed_hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout(attention_output)

        normed_hidden_states = self.norm2(hidden_states)
        feed_forward_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(feed_forward_output)
        return hidden_states


# ==========================================
# 门控融合层 (Gated Fusion)
# ==========================================

class GatedFeatureFusion(nn.Module):
    """
    实现架构图中的“门控网络动态加权”。
    输入: 多个不同来源的特征向量列表 [vec1, vec2, ...]
    输出: 融合后的特征向量
    """

    def __init__(self, input_dims, output_dim, dropout=0.1):
        super(GatedFeatureFusion, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        # 1. 特征投影：将所有输入特征映射到相同的维度
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            ) for dim in input_dims
        ])

        # 2. 门控网络 (Attention Mechanism)
        # 它可以根据所有特征的拼接，计算出每个特征组的权重
        total_dim = output_dim * len(input_dims)
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, output_dim),  # 压缩信息
            nn.Tanh(),
            nn.Linear(output_dim, len(input_dims)),  # 输出每个特征组的权重 score
        )

        self.dropout = nn.Dropout(dropout)
        self.final_norm = LayerNorm(output_dim)

    def forward(self, feature_list):
        # 1. 对齐维度
        projected_feats = []
        for i, feat in enumerate(feature_list):
            projected_feats.append(self.projectors[i](feat))

        # projected_feats: List of [B, output_dim]

        # 2. 计算门控权重
        # 拼接所有特征作为 Context
        concat_feats = torch.cat(projected_feats, dim=1)  # [B, total_dim]

        # 计算 Attention Scores
        raw_scores = self.gate_net(concat_feats)  # [B, num_features]
        attention_weights = F.softmax(raw_scores, dim=1)  # [B, num_features]

        # 3. 加权融合
        # stack: [B, num_features, output_dim]
        stack_feats = torch.stack(projected_feats, dim=1)

        # weights: [B, num_features, 1]
        weights = attention_weights.unsqueeze(-1)

        # Weighted Sum
        weighted_feats = torch.sum(stack_feats * weights, dim=1)  # [B, output_dim]

        return self.final_norm(weighted_feats + self.dropout(weighted_feats))  # Residual + Norm


# ==========================================
# 3. 进阶双塔模型 (SASRec + Gating + Features)
# ==========================================

class GatingTwoTowerSASRec(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1):
        super(GatingTwoTowerSASRec, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = meta_info['seq_len']

        # ---------------------------------------------------
        # A. Shared Embeddings (ID & Category)
        # ---------------------------------------------------
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)
        self.inter_type_emb = nn.Embedding(meta_info['num_inter_types'], embed_dim, padding_idx=0)
        self.duration_emb = nn.Embedding(meta_info['num_duration_buckets'], embed_dim, padding_idx=0)

        # ---------------------------------------------------
        # B. User Tower Components
        # ---------------------------------------------------

        # B.1 序列建模 (Transformer)
        self.pos_emb = nn.Embedding(self.seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.transformer_norm = LayerNorm(embed_dim)

        # B.2 静态特征 Embeddings (DSSM part)
        # 假设 Age, Gender, InterType, Duration Bucket
        self.age_emb = nn.Embedding(120, embed_dim // 2)
        self.gender_emb = nn.Embedding(5, embed_dim // 2)
        self.inter_type_emb = nn.Embedding(meta_info.get('num_inter_types', 10), embed_dim, padding_idx=0)

        # B.3 门控融合层
        # 输入三部分：Sequence向量, Profile向量, Context向量
        # Seq维度: embed_dim
        # Profile维度: (embed_dim//2 * 2) + 1 (activity)
        # Context维度 (假设 InterType 是 Context): embed_dim
        self.user_fusion = GatedFeatureFusion(
            input_dims=[embed_dim, embed_dim + 1, embed_dim],
            output_dim=embed_dim
        )

        # B.4 最终 MLP
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)  # Output Embedding
        )

        # ---------------------------------------------------
        # C. Item Tower Components
        # ---------------------------------------------------

        # C.1 ID 特征处理 (Item ID + Cat ID)
        # 维度: embed_dim * 2

        # C.2 稠密/其他特征 (Pop, etc)
        # 维度: 1

        # C.3 门控融合
        self.item_fusion = GatedFeatureFusion(
            input_dims=[embed_dim * 2, 1],
            output_dim=embed_dim
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

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
        """生成 Transformer 需要的 Causal Mask"""
        padding_mask = (item_seq == 0).bool()
        batch_size, seq_len = item_seq.size()
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=item_seq.device), diagonal=1).bool()
        extended_mask = padding_mask.unsqueeze(1).unsqueeze(2) | causal_mask.unsqueeze(0).unsqueeze(0)
        final_mask = torch.zeros_like(extended_mask, dtype=torch.float)
        final_mask = final_mask.masked_fill(extended_mask, -1e9)
        return final_mask

    def forward_user_tower(self, batch):
        # ==========================
        # 1. Sequence Feature (SASRec Logic)
        # ==========================
        item_seq = batch['item_id_seq']  # (B, L)
        cat_seq = batch['video_category_seq']
        inter_seq = batch['inter_type_seq']
        dur_seq = batch['duration_bucket_seq']

        # ==========================
        # 2. Embedding Lookup
        # ==========================
        # 此时它们出来的维度都是 (B, L, embed_dim)
        e_item = self.item_emb(item_seq)
        e_cat = self.cat_emb(cat_seq)
        e_inter = self.inter_type_emb(inter_seq)
        e_dur = self.duration_emb(dur_seq)

        # 类似于 BERT 的 input = Token + Segment + Pos
        # 这里我们是 input = Item + Cat + InterType + Duration
        seq_emb = e_item + e_cat + e_inter + e_dur

        # Positional Embedding
        positions = torch.arange(self.seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)
        seq_emb += self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer Blocks
        mask = self.get_attention_mask(item_seq)
        x = seq_emb
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.transformer_norm(x)

        # 取序列最后一个位置的 Embedding 代表“长期兴趣 + 短期意图”
        seq_feature_vec = x[:, -1, :]  # [B, embed_dim]

        # ==========================
        # 2. User Profile Feature (DSSM Logic)
        # ==========================
        age_vec = self.age_emb(batch['age']).squeeze() if batch['age'].dim() > 1 else self.age_emb(batch['age'])
        gen_vec = self.gender_emb(batch['gender']).squeeze() if batch['gender'].dim() > 1 else self.gender_emb(
            batch['gender'])
        act_vec = batch['user_activity_norm']  # [B, 1]

        # 拼接用户静态画像
        profile_feature_vec = torch.cat([age_vec, gen_vec, act_vec], dim=1)  # [B, D_prof]

        # ==========================
        # 3. Context Feature (Interaction Logic)
        # ==========================
        # 这里为了演示，我们取序列中最近一次交互的类型作为 Context，或者如果有单独的 context 字段更好
        # 假设 batch['inter_type_seq'] 是序列，我们取最后一个非0的，这里简化取最后一个
        last_inter_type = batch['inter_type_seq'][:, -1]
        context_feature_vec = self.inter_type_emb(last_inter_type)  # [B, embed_dim]

        # ==========================
        # 4. Gated Fusion (门控融合)
        # ==========================
        # 融合：[序列特征, 画像特征, 上下文特征]
        user_fusion_vec = self.user_fusion([seq_feature_vec, profile_feature_vec, context_feature_vec])

        # 最终映射
        user_output = self.user_mlp(user_fusion_vec)
        return user_output

    def forward_item_tower(self, batch):
        # ==========================
        # 1. ID Features
        # ==========================
        item_id = batch['item_id']
        cat_id = batch['video_category']

        id_vec = torch.cat([self.item_emb(item_id), self.cat_emb(cat_id)], dim=1)  # [B, 2*embed_dim]

        # ==========================
        # 2. Dense/Content Features
        # ==========================
        pop_vec = batch['item_pop_norm']  # [B, 1]
        # 如果有图像/文本 Embedding，可以在这里加入 list

        # ==========================
        # 3. Gated Fusion
        # ==========================
        item_fusion_vec = self.item_fusion([id_vec, pop_vec])

        item_output = self.item_mlp(item_fusion_vec)
        return item_output

    def forward(self, batch):
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # 归一化 (为 InfoNCE / Cosine Loss 做准备)
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb