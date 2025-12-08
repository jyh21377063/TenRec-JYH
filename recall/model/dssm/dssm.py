import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    def __init__(self, meta_info, embed_dim=64, hidden_dims=[128, 64]):
        super(TwoTowerModel, self).__init__()

        self.meta = meta_info
        self.embed_dim = embed_dim

        # ==========================
        # 1. Embeddings (共享层)
        # ==========================
        # User & Item 都会用到 item_id 和 category，建议共享 Embedding
        self.item_emb = nn.Embedding(meta_info['num_items'], embed_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(meta_info['num_categories'], embed_dim, padding_idx=0)

        # 仅 User Tower 用的序列特征 Embedding
        self.inter_type_emb = nn.Embedding(meta_info['num_inter_types'], embed_dim, padding_idx=0)
        self.duration_emb = nn.Embedding(meta_info['num_duration_buckets'], embed_dim, padding_idx=0)

        # 用户画像 Embedding (Age, Gender) - 假设最大年龄100，性别3
        # 如果这些值很大，建议在 Dataset 里做 Bucket 处理，这里暂时给个大致范围
        self.age_emb = nn.Embedding(120, embed_dim // 2)
        self.gender_emb = nn.Embedding(5, embed_dim // 2)

        # ==========================
        # 2. User Tower Components
        # ==========================
        # User Tower 输入维度计算:
        # 序列池化 (Item + Cat + Inter + Dur) = 4 * embed_dim
        # 稠密特征 (Activity) = 1
        # 画像特征 (Age + Gender) = embed_dim // 2 * 2
        user_input_dim = (4 * embed_dim) + 1 + (2 * (embed_dim // 2))

        self.user_mlp = self._make_mlp(user_input_dim, hidden_dims)

        # ==========================
        # 3. Item Tower Components
        # ==========================
        # Item Tower 输入维度计算:
        # Embedding (Item + Cat) = 2 * embed_dim
        # 稠密特征 (Pop) = 1
        item_input_dim = (2 * embed_dim) + 1

        self.item_mlp = self._make_mlp(item_input_dim, hidden_dims)

        # 初始化权重
        self.apply(self._init_weights)

    def _make_mlp(self, input_dim, hidden_dims):
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = h_dim
        # 最后一层输出 Embedding
        layers.append(nn.Linear(curr_dim, self.embed_dim))
        return nn.Sequential(*layers)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)

    def _masked_mean_pooling(self, seq_emb, mask):
        """
        Args:
            seq_emb: (B, L, D)
            mask: (B, L) 0 for padding, 1 for valid
        Returns:
            pooled: (B, D)
        """
        # 扩展 mask 维度以匹配 embedding: (B, L, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # 求和: 只有非 padding 的位置有值
        sum_emb = torch.sum(seq_emb * mask_expanded, dim=1)

        # 计数: 每个样本有多少个有效 token (加上 epsilon 防止除以 0)
        cnt = torch.sum(mask_expanded, dim=1)
        cnt = torch.clamp(cnt, min=1e-9)

        return sum_emb / cnt

    def forward_user_tower(self, batch):
        # 1. 获取序列数据
        item_seq = batch['item_id_seq']  # (B, L)
        cat_seq = batch['video_category_seq']
        inter_seq = batch['inter_type_seq']
        dur_seq = batch['duration_bucket_seq']

        # 2. 生成 Mask (基于 item_id 是否为 0)
        mask = (item_seq > 0).long()  # (B, L)

        # 3. Embedding Lookup
        e_item = self.item_emb(item_seq)  # (B, L, D)
        e_cat = self.cat_emb(cat_seq)
        e_inter = self.inter_type_emb(inter_seq)
        e_dur = self.duration_emb(dur_seq)

        # 4. 平均池化 (Mean Pooling) -> 这里是 SASRec 的主要对比点
        # 将四个序列 Embedding 拼起来或者分别池化，通常分别池化再拼接效果更稳
        p_item = self._masked_mean_pooling(e_item, mask)  # (B, D)
        p_cat = self._masked_mean_pooling(e_cat, mask)
        p_inter = self._masked_mean_pooling(e_inter, mask)
        p_dur = self._masked_mean_pooling(e_dur, mask)

        # 5. 处理其他特征
        # Dense
        dense_act = batch['user_activity_norm']  # (B, 1)
        # User Profile
        e_age = self.age_emb(batch['age']).squeeze(1) if batch['age'].dim() > 1 else self.age_emb(batch['age'])
        e_gender = self.gender_emb(batch['gender']).squeeze(1) if batch['gender'].dim() > 1 else self.gender_emb(
            batch['gender'])

        # 6. 拼接所有特征
        user_vec = torch.cat([p_item, p_cat, p_inter, p_dur, dense_act, e_age, e_gender], dim=1)

        # 7. MLP 映射
        return self.user_mlp(user_vec)  # (B, embed_dim)

    def forward_item_tower(self, batch):
        # 1. Embedding
        item_id = batch['item_id']
        cat_id = batch['video_category']

        e_item = self.item_emb(item_id)
        e_cat = self.cat_emb(cat_id)

        # 2. Dense
        pop_norm = batch['item_pop_norm']  # (B, 1)

        # 3. 拼接
        item_vec = torch.cat([e_item, e_cat, pop_norm], dim=1)

        # 4. MLP 映射
        return self.item_mlp(item_vec)  # (B, embed_dim)

    def forward(self, batch):
        """
        训练时同时计算两个塔
        """
        u_emb = self.forward_user_tower(batch)
        i_emb = self.forward_item_tower(batch)

        # 归一化 (Cosine Similarity 必须)
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)

        return u_emb, i_emb