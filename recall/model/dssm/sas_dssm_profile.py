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

        # Mask 机制：0 表示可见，-1e9 表示屏蔽
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
# 2. 核心模型: ProfilePromptSASRec
# ==========================================

class ProfilePromptSASRec(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1):
        super(ProfilePromptSASRec, self).__init__()
        self.embed_dim = embed_dim
        # 注意：这里我们依然记录原始 seq_len，但在 PosEmb 初始化时会 +1
        self.seq_len = meta_info['seq_len']

        # ---------------------------------------------------
        # A. 序列特征 Embeddings
        # ---------------------------------------------------
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)

        # [关键修改] Positional Embedding 长度 +1，为了容纳 Profile Token
        self.pos_emb = nn.Embedding(self.seq_len + 1, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # ---------------------------------------------------
        # B. 用户画像特征 (Profile Projector)
        # ---------------------------------------------------
        # 假设画像特征维度: Age(dim//2) + Gender(dim//2) + Activity(1)
        self.age_emb = nn.Embedding(120, embed_dim // 2)
        self.gender_emb = nn.Embedding(5, embed_dim // 2)

        # 输入维度计算
        self.profile_input_dim = (embed_dim // 2) + (embed_dim // 2) + 1

        # 映射层：把杂乱的画像特征映射成一个完美的 "Prompt Token"
        self.profile_projector = nn.Sequential(
            nn.Linear(self.profile_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)  # 输出必须是 embed_dim
        )

        # ---------------------------------------------------
        # C. Transformer Encoder
        # ---------------------------------------------------
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.transformer_norm = LayerNorm(embed_dim)

        # ---------------------------------------------------
        # D. Item Tower (保持简单有效)
        # ---------------------------------------------------
        # Item + Cat + Pop(1) -> Output
        self.item_input_dim = embed_dim + embed_dim + 1
        self.item_mlp = nn.Sequential(
            nn.Linear(self.item_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
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
        """
        生成带有 Prompt 的 Causal Mask。
        seq 结构: [Profile, Item1, Item2, ..., ItemN]
        长度: L + 1
        """
        # 1. 基础 Padding Mask (True 代表是 padding，需要被 mask)
        # item_seq: [B, L] -> 0 是 padding
        item_pad_mask = (item_seq == 0).bool()

        # Profile 永远不是 Padding (设为 False)
        batch_size = item_seq.size(0)
        profile_pad_mask = torch.zeros((batch_size, 1), device=item_seq.device, dtype=torch.bool)

        # 拼接 Mask: [B, L+1]
        extended_padding_mask = torch.cat([profile_pad_mask, item_pad_mask], dim=1)

        # 2. Causal Mask (因果掩码)
        # 序列总长 seq_len + 1
        total_len = item_seq.size(1) + 1

        # 生成下三角矩阵的逆（上三角为 1，代表未来，不可见）
        # [L+1, L+1]
        causal_mask = torch.triu(torch.ones((total_len, total_len), device=item_seq.device), diagonal=1).bool()

        # 3. 组合 Mask
        # 维度扩展为 [B, 1, L+1, L+1] 用于 Multi-head Attention
        # 逻辑：只要 padding_mask 是 True 或者 causal_mask 是 True，那个位置就得被遮住
        final_mask = extended_padding_mask.unsqueeze(1).unsqueeze(2) | causal_mask.unsqueeze(0).unsqueeze(0)

        # 转为 float: 遮住的地方为 -1e9，其他为 0
        attention_mask = torch.zeros_like(final_mask, dtype=torch.float)
        attention_mask = attention_mask.masked_fill(final_mask, -1e9)

        return attention_mask

    def forward_user_tower(self, batch):
        # ==========================
        # 1. 准备 Sequence Embedding
        # ==========================
        item_seq = batch['item_id_seq']
        cat_seq = batch['video_category_seq']

        # [B, L, D]
        seq_emb = self.item_emb(item_seq) + self.cat_emb(cat_seq)

        # ==========================
        # 2. 准备 Profile Prompt (前融合)
        # ==========================
        age_vec = self.age_emb(batch['age'])
        if age_vec.dim() == 3: age_vec = age_vec.squeeze(1)

        gen_vec = self.gender_emb(batch['gender'])
        if gen_vec.dim() == 3: gen_vec = gen_vec.squeeze(1)

        act_vec = batch['user_activity_norm']  # [B, 1]

        # 拼接原始画像特征
        raw_profile = torch.cat([age_vec, gen_vec, act_vec], dim=1)

        # 投影为 Token: [B, D]
        profile_token = self.profile_projector(raw_profile)

        # 维度对齐: [B, D] -> [B, 1, D]
        profile_token = profile_token.unsqueeze(1)

        # ==========================
        # 3. 拼接: Profile + Sequence
        # ==========================
        # 结果: [B, L+1, D]
        # 类似于 BERT 的 input: [CLS, tok1, tok2...]，这里是 [Profile, Item1, Item2...]
        input_emb = torch.cat([profile_token, seq_emb], dim=1)

        # ==========================
        # 4. Positional Embedding
        # ==========================
        # 生成 0 到 L 的位置索引
        curr_batch, curr_seq_len, _ = input_emb.size()
        positions = torch.arange(curr_seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand(curr_batch, -1)

        input_emb += self.pos_emb(positions)
        input_emb = self.emb_dropout(input_emb)

        # ==========================
        # 5. Transformer Encoder
        # ==========================
        # 获取 Mask (注意现在序列长了 1)
        mask = self.get_attention_mask(item_seq)

        x = input_emb
        for block in self.transformer_blocks:
            x = block(x, mask)

        x = self.transformer_norm(x)

        # ==========================
        # 6. 获取 User Output
        # ==========================
        # 我们取序列的最后一个位置 (index -1)
        # 此时的 Output 包含了：Profile 的信息 + 所有历史行为的信息
        # 它是基于 Profile 偏好，看完了所有历史行为后，沉淀出的最终 User State
        user_output = x[:, -1, :]

        return user_output

    def forward_item_tower(self, batch):
        item_id = batch['item_id']
        cat_id = batch['video_category']
        pop_vec = batch['item_pop_norm']

        e_item = self.item_emb(item_id)
        e_cat = self.cat_emb(cat_id)

        # Concatenate Features
        item_combined = torch.cat([e_item, e_cat, pop_vec], dim=1)

        # MLP Project
        item_output = self.item_mlp(item_combined)
        return item_output

    def forward(self, batch):
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # 归一化 (Cosine Similarity 必需)
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb