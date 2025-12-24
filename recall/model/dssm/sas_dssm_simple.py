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


# ==========================================
# 新版双塔模型 (Concat Fusion)
# ==========================================

class ConcatTwoTowerSASRec(nn.Module):
    def __init__(self, meta_info, embed_dim=64, num_layers=2, num_heads=2, dropout=0.1):
        super(ConcatTwoTowerSASRec, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = meta_info['seq_len']

        # ---------------------------------------------------
        # A. ID & Sequence Embeddings
        # ---------------------------------------------------
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.transformer_norm = LayerNorm(embed_dim)

        # ---------------------------------------------------
        # B. User Side Features (Profile)
        # ---------------------------------------------------
        # 这里的维度你可以自由调整，为了方便，我设为 embed_dim 的一半或相等
        self.age_emb = nn.Embedding(120, embed_dim // 2)
        self.gender_emb = nn.Embedding(5, embed_dim // 2)
        # user_activity_norm 是 float，直接用，不需要 Embedding

        # 计算 User Tower 拼接后的总维度
        # Seq(embed_dim) + Age(embed_dim//2) + Gender(embed_dim//2) + Activity(1)
        self.user_concat_dim = embed_dim + (embed_dim // 2) + (embed_dim // 2) + 1

        # User MLP: 负责把拼接后的长向量压缩回 embed_dim
        self.user_mlp = nn.Sequential(
            nn.Linear(self.user_concat_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # ---------------------------------------------------
        # C. Item Tower Components
        # ---------------------------------------------------
        # ID(embed_dim) + Cat(embed_dim) + Pop(1)
        self.item_concat_dim = embed_dim + embed_dim + 1

        self.item_mlp = nn.Sequential(
            nn.Linear(self.item_concat_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
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
        # 1. Sequence Feature (SASRec)
        # ==========================
        item_seq = batch['item_id_seq']  # (B, L)
        cat_seq = batch['video_category_seq']

        # Embedding Lookup
        e_item = self.item_emb(item_seq)
        e_cat = self.cat_emb(cat_seq)

        # 简单相加融合
        seq_emb = e_item + e_cat

        # Positional Embedding
        positions = torch.arange(self.seq_len, dtype=torch.long, device=item_seq.device)
        positions = positions.unsqueeze(0).expand_as(item_seq)
        seq_emb += self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer
        mask = self.get_attention_mask(item_seq)
        x = seq_emb
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.transformer_norm(x)

        # 取最后一个有效位置 (B, embed_dim)
        seq_feature_vec = x[:, -1, :]

        # ==========================
        # 2. User Profile Feature
        # ==========================
        # 确保维度是 [B, dim]
        age_vec = self.age_emb(batch['age'])
        if age_vec.dim() == 3: age_vec = age_vec.squeeze(1)

        gen_vec = self.gender_emb(batch['gender'])
        if gen_vec.dim() == 3: gen_vec = gen_vec.squeeze(1)

        act_vec = batch['user_activity_norm']  # [B, 1]

        # ==========================
        # 3. Concat Fusion
        # ==========================
        # 没有任何花哨的操作，直接拼起来
        user_combined = torch.cat([seq_feature_vec, age_vec, gen_vec, act_vec], dim=1)

        # MLP 映射
        user_output = self.user_mlp(user_combined)

        return user_output

    def forward_item_tower(self, batch):
        # ==========================
        # 1. Features
        # ==========================
        item_id = batch['item_id']
        cat_id = batch['video_category']
        pop_vec = batch['item_pop_norm']  # [B, 1]

        e_item = self.item_emb(item_id)
        e_cat = self.cat_emb(cat_id)

        # ==========================
        # 2. Concat Fusion
        # ==========================
        item_combined = torch.cat([e_item, e_cat, pop_vec], dim=1)

        # MLP 映射
        item_output = self.item_mlp(item_combined)

        return item_output

    def forward(self, batch):
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # Normalize for Cosine Similarity
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb