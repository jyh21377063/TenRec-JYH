import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 基础组件 (保持不变)
# ==========================================

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


# ==========================================
# 方案 A: 残差融合 SASRec (Residual Fusion)
# ==========================================

class ResidualSASRec(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1):
        super(ResidualSASRec, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = meta_info['seq_len']

        # ---------------------------------------------------
        # 1. 序列特征 (主干网络)
        # ---------------------------------------------------
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.transformer_norm = LayerNorm(embed_dim)

        # ---------------------------------------------------
        # 2. 用户画像特征 (辅助分支)
        # ---------------------------------------------------
        # 维度设为 embed_dim // 2，减少参数量
        self.age_emb = nn.Embedding(120, embed_dim // 2)
        self.gender_emb = nn.Embedding(5, embed_dim // 2)
        # user_activity_norm 是 float (dim=1)

        # 计算辅助特征的总维度: (dim//2) + (dim//2) + 1
        self.side_input_dim = (embed_dim // 2) + (embed_dim // 2) + 1

        # [关键] 侧边特征投影层：将杂乱的画像特征映射到和序列特征一样的维度
        self.side_projector = nn.Sequential(
            nn.Linear(self.side_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)  # 输出维度必须等于 embed_dim 以便相加
        )

        # 融合后的最终 LayerNorm，防止数值波动
        self.final_user_norm = LayerNorm(embed_dim)

        # ---------------------------------------------------
        # 3. Item Tower (保持适度简单)
        # ---------------------------------------------------
        # ID(embed_dim) + Cat(embed_dim) + Pop(1)
        self.item_concat_dim = embed_dim + embed_dim + 1

        self.item_mlp = nn.Sequential(
            nn.Linear(self.item_concat_dim, embed_dim),
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
        padding_mask = (item_seq == 0).bool()
        batch_size, seq_len = item_seq.size()
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=item_seq.device), diagonal=1).bool()
        extended_mask = padding_mask.unsqueeze(1).unsqueeze(2) | causal_mask.unsqueeze(0).unsqueeze(0)
        final_mask = torch.zeros_like(extended_mask, dtype=torch.float)
        final_mask = final_mask.masked_fill(extended_mask, -1e9)
        return final_mask

    def forward_user_tower(self, batch):
        # ==========================
        # Part A: 序列特征 (Golden Signal)
        # ==========================
        item_seq = batch['item_id_seq']
        cat_seq = batch['video_category_seq']

        # Embedding Lookup & Sum
        seq_emb = self.item_emb(item_seq) + self.cat_emb(cat_seq)

        # Positional Embedding
        positions = torch.arange(self.seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)
        seq_emb += self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer Logic
        mask = self.get_attention_mask(item_seq)
        x = seq_emb
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.transformer_norm(x)

        # 取最后一个时间步 (Sequence Vector)
        # [B, embed_dim]
        seq_vector = x[:, -1, :]

        # ==========================
        # Part B: 画像特征 (Residual Signal)
        # ==========================
        age_vec = self.age_emb(batch['age'])
        if age_vec.dim() == 3: age_vec = age_vec.squeeze(1)

        gen_vec = self.gender_emb(batch['gender'])
        if gen_vec.dim() == 3: gen_vec = gen_vec.squeeze(1)

        act_vec = batch['user_activity_norm']  # [B, 1]

        # 拼接画像
        side_feat = torch.cat([age_vec, gen_vec, act_vec], dim=1)  # [B, side_input_dim]

        # 投影画像特征 -> Bias Vector
        side_vector = self.side_projector(side_feat)  # [B, embed_dim]

        # ==========================
        # Part C: 残差融合 (The Solution)
        # ==========================
        # 直接相加：保留了 seq_vector 的几何特性，同时注入了画像偏差
        user_combined = seq_vector + side_vector

        # 最终再过一层 Norm 稳住分布
        user_output = self.final_user_norm(user_combined)

        return user_output

    def forward_item_tower(self, batch):
        # 简单的 MLP 映射
        item_id = batch['item_id']
        cat_id = batch['video_category']
        pop_vec = batch['item_pop_norm']

        e_item = self.item_emb(item_id)
        e_cat = self.cat_emb(cat_id)

        item_combined = torch.cat([e_item, e_cat, pop_vec], dim=1)
        item_output = self.item_mlp(item_combined)

        return item_output

    def forward(self, batch):
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # 归一化 (Cosine Similarity 必需)
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb


# 带有门控的融合方案
class GatedResidualSASRec(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1):
        super(GatedResidualSASRec, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = meta_info['seq_len']

        # ---------------------------------------------------
        # 1. 序列特征 (主干网络)
        # ---------------------------------------------------
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.transformer_norm = LayerNorm(embed_dim)

        # ---------------------------------------------------
        # 2. 用户画像特征 (辅助分支)
        # ---------------------------------------------------
        # 维度设为 embed_dim // 2，减少参数量
        self.age_emb = nn.Embedding(120, embed_dim // 2)
        self.gender_emb = nn.Embedding(5, embed_dim // 2)
        # user_activity_norm 是 float (dim=1)

        # 计算辅助特征的总维度: (dim//2) + (dim//2) + 1
        self.side_input_dim = (embed_dim // 2) + (embed_dim // 2) + 1

        # [关键] 侧边特征投影层：将杂乱的画像特征映射到和序列特征一样的维度
        self.side_projector = nn.Sequential(
            nn.Linear(self.side_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)  # 输出维度必须等于 embed_dim 以便相加
        )

        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # 压缩信息
            nn.Sigmoid()  # 输出 0~1 的门控系数
        )

        # 融合后的最终 LayerNorm，防止数值波动
        self.final_user_norm = LayerNorm(embed_dim)

        # ---------------------------------------------------
        # 3. Item Tower (保持适度简单)
        # ---------------------------------------------------
        # ID(embed_dim) + Cat(embed_dim) + Pop(1)
        self.item_concat_dim = embed_dim + embed_dim + 1

        self.item_mlp = nn.Sequential(
            nn.Linear(self.item_concat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.apply(self._init_weights)
        # 这里的 gate_net[0] 是 Linear, gate_net[1] 是 Sigmoid
        # 我们要找的是产生输出的那个 Linear 层，也就是 gate_net[0]

        # 将 Bias 初始化为 -2.0
        # 效果：Sigmoid(-2.0) ≈ 0.12。意思是训练刚开始时，只放行 12% 的画像特征。
        nn.init.constant_(self.gate_net[0].bias, -2.0)

        # 将 Weight 初始化得很小（接近0）
        # 效果：让 Gate 的初始输出对输入不敏感，稳定在 Bias 决定的 0.12 附近
        nn.init.normal_(self.gate_net[0].weight, mean=0.0, std=0.01)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        padding_mask = (item_seq == 0).bool()
        batch_size, seq_len = item_seq.size()
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=item_seq.device), diagonal=1).bool()
        extended_mask = padding_mask.unsqueeze(1).unsqueeze(2) | causal_mask.unsqueeze(0).unsqueeze(0)
        final_mask = torch.zeros_like(extended_mask, dtype=torch.float)
        final_mask = final_mask.masked_fill(extended_mask, -1e9)
        return final_mask

    def forward_user_tower(self, batch):
        # ==========================
        # Part A: 序列特征 (Golden Signal)
        # ==========================
        item_seq = batch['item_id_seq']
        cat_seq = batch['video_category_seq']

        # Embedding Lookup & Sum
        seq_emb = self.item_emb(item_seq) + self.cat_emb(cat_seq)

        # Positional Embedding
        positions = torch.arange(self.seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)
        seq_emb += self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer Logic
        mask = self.get_attention_mask(item_seq)
        x = seq_emb
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.transformer_norm(x)

        # 取最后一个时间步 (Sequence Vector)
        # [B, embed_dim]
        seq_vector = x[:, -1, :]

        # ==========================
        # Part B: 画像特征 (Residual Signal)
        # ==========================
        age_vec = self.age_emb(batch['age'])
        if age_vec.dim() == 3: age_vec = age_vec.squeeze(1)

        gen_vec = self.gender_emb(batch['gender'])
        if gen_vec.dim() == 3: gen_vec = gen_vec.squeeze(1)

        act_vec = batch['user_activity_norm']  # [B, 1]

        # 拼接画像
        side_feat = torch.cat([age_vec, gen_vec, act_vec], dim=1)  # [B, side_input_dim]

        # 投影画像特征 -> Bias Vector
        side_vector = self.side_projector(side_feat)  # [B, embed_dim]

        # ==========================
        # Part C: 门控残差融合 (The Solution)
        # ==========================
        # 1. 拼接两者作为 Gate 的判断依据
        concat_state = torch.cat([seq_vector, side_vector], dim=1)  # [B, 2*dim]

        # 2. 计算 Alpha (智能水龙头)
        # alpha 接近 1 说明需要画像，接近 0 说明不需要
        alpha = self.gate_net(concat_state)  # [B, embed_dim]

        # 3. 加权融合
        # 你的逻辑：Sequence 是主力(1.0)，Profile 是辅助(alpha)
        user_combined = seq_vector + (alpha * side_vector)

        # 最终 Norm
        user_output = self.final_user_norm(user_combined)
        return user_output

    def forward_item_tower(self, batch):
        # 简单的 MLP 映射
        item_id = batch['item_id']
        cat_id = batch['video_category']
        pop_vec = batch['item_pop_norm']

        e_item = self.item_emb(item_id)
        e_cat = self.cat_emb(cat_id)

        item_combined = torch.cat([e_item, e_cat, pop_vec], dim=1)
        item_output = self.item_mlp(item_combined)

        return item_output

    def forward(self, batch):
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # 归一化 (Cosine Similarity 必需)
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb