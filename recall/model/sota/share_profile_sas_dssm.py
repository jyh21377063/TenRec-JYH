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


class GatedFusionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusionLayer, self).__init__()

        # 1. 侧边信息的专用 Norm (你提到的必须加的)
        self.side_norm = nn.LayerNorm(embed_dim)

        # 2. 门控网络：输入是 Seq 和 Side 的拼接
        # 它的作用是看一眼当前的“战况”，决定放多少 Side 进来
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim)

        # 初始化技巧：让 Gate 初始值偏小
        # 这样训练初期模型会优先信任 Sequence，避免侧边信息带来的震荡
        nn.init.xavier_normal_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, -1.0)  # bias < 0 导致 Sigmoid 输出 < 0.5

    def forward(self, seq_vec, side_vec):
        """
        seq_vec:  [B, D] -  Transformer 出来的序列向量 (主)
        side_vec: [B, D] - 侧边画像向量 (辅)
        """
        # A. 即使 Side 已经过 Projector，这里再加个 Norm 也是双重保险
        # 确保 Side 和 Seq 在同一个量级 (均值0，方差1)
        side_vec = self.side_norm(side_vec)

        # B. 计算门控值 (Gate Value)
        # 拼接两者，让网络以此判断融合程度
        concat = torch.cat([seq_vec, side_vec], dim=1)
        gate = torch.sigmoid(self.gate_linear(concat))  # [B, D], 值在 0~1 之间

        # C. 注入式融合 (Gated Residual)
        # 关键点：Seq 保持原样，Side 乘以门控后加在 Seq 上
        final_vec = seq_vec + (gate * side_vec)

        return final_vec


class SharedPreFusionSASRec(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super(SharedPreFusionSASRec, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = meta_info['seq_len']

        # ==================================================
        # 1. Shared Embeddings (完全共享)
        # ==================================================
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)

        # User Embedding: 用于序列头部的 Token (Pre-fusion)
        self.user_seq_emb = nn.Embedding(meta_info['num_users'], embed_dim, padding_idx=0)

        # ==================================================
        # 2. Sequence Branch (Transformer)
        # ==================================================
        # 长度 = seq_len + 1 (因为加了 User Token)
        self.pos_emb = nn.Embedding(self.seq_len + 1, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.transformer_norm = LayerNorm(embed_dim)

        # ==================================================
        # 3. Residual Branch (Static Profile)
        # ==================================================
        # 这里专门处理 Transformer 处理不了的“非序列”特征
        self.age_emb = nn.Embedding(120, embed_dim // 2)
        self.gender_emb = nn.Embedding(5, embed_dim // 2)

        # 侧边输入维度: Age + Gender + Activity
        # 如果你特别想强调 User Bias，可以在这里再加一个 User Embedding (不推荐共用 user_seq_emb)
        self.side_input_dim = (embed_dim // 2) + (embed_dim // 2) + 1

        self.side_projector = nn.Sequential(
            nn.Linear(self.side_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 门控
        self.fusion_layer = GatedFusionLayer(embed_dim)

        # 最终融合 Norm
        self.final_user_norm = LayerNorm(embed_dim)

        # ==================================================
        # 4. Item Tower (共享)
        # ==================================================
        self.item_norm = LayerNorm(embed_dim)

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
        # 你的经典 Mask 逻辑：序列长度 + 1 (User Token)
        item_pad_mask = (item_seq == 0).bool()
        batch_size = item_seq.size(0)
        user_token_mask = torch.zeros((batch_size, 1), device=item_seq.device, dtype=torch.bool)
        extended_padding_mask = torch.cat([user_token_mask, item_pad_mask], dim=1)

        total_len = item_seq.size(1) + 1
        causal_mask = torch.triu(torch.ones((total_len, total_len), device=item_seq.device), diagonal=1).bool()

        final_mask = extended_padding_mask.unsqueeze(1).unsqueeze(2) | causal_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = torch.zeros_like(final_mask, dtype=torch.float)
        attention_mask = attention_mask.masked_fill(final_mask, -1e9)
        return attention_mask

    def forward_user_tower(self, batch):
        # -----------------------------------------------
        # Path A: Transformer Sequence (Contextual Interest)
        # -----------------------------------------------
        item_seq = batch['item_id_seq']
        cat_seq = batch['video_category_seq']

        # 1. Items
        seq_emb = self.item_emb(item_seq) + self.cat_emb(cat_seq)

        # 2. User Token (Pre-fusion)
        u_emb = self.user_seq_emb(batch['user_id']).unsqueeze(1)  # [B, 1, D]

        # 3. Concat -> [User, Item1, Item2...]
        input_emb = torch.cat([u_emb, seq_emb], dim=1)

        # 4. Positional
        curr_batch, curr_seq_len, _ = input_emb.size()
        positions = torch.arange(curr_seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand(curr_batch, -1)
        input_emb += self.pos_emb(positions)
        input_emb = self.emb_dropout(input_emb)

        # 5. Encoding
        mask = self.get_attention_mask(item_seq)
        x = input_emb
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.transformer_norm(x)

        # 取最后一个 Vector (包含了 User Token 和 所有 Item 的信息)
        seq_vector = x[:, -1, :]

        # -----------------------------------------------
        # Path B: Residual Side Info (Static Bias)
        # -----------------------------------------------
        age_vec = self.age_emb(batch['age'])
        if age_vec.dim() == 3: age_vec = age_vec.squeeze(1)

        gen_vec = self.gender_emb(batch['gender'])
        if gen_vec.dim() == 3: gen_vec = gen_vec.squeeze(1)

        act_vec = batch['user_activity_norm']

        # 拼接静态特征
        side_feat = torch.cat([age_vec, gen_vec, act_vec], dim=1)

        # 注意：self.side_projector 依然保留，因为它负责把维度对齐
        # 但是原来的 self.side_norm 移到 FusionLayer 里面去做
        raw_side_vec = self.side_projector(side_feat)

        # 使用门控融合
        # seq_vector 是主，raw_side_vec 是辅
        final_vec = self.fusion_layer(seq_vector, raw_side_vec)

        # 最后再做一次 Norm 是好的习惯，保证输出给 Embedding 空间时是规整的
        return self.final_user_norm(final_vec)

    def forward_item_tower(self, batch):
        # 保持不变，使用 Shared Embedding
        target_item = batch['item_id']
        target_cat = batch['video_category']

        i_emb = self.item_emb(target_item)
        c_emb = self.cat_emb(target_cat)

        # Sum Pooling + Norm
        return self.item_norm(i_emb + c_emb)

    def forward(self, batch):
        u_vec = self.forward_user_tower(batch)
        i_vec = self.forward_item_tower(batch)
        return F.normalize(u_vec, p=2, dim=1), F.normalize(i_vec, p=2, dim=1)