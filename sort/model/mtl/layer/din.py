import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, num_features, dim=2, epsilon=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros((num_features,)))

    def forward(self, x):
        # x shape: (Batch, Hidden_Size) or (Batch, Seq_Len, Hidden_Size)
        if x.dim() == 3:
            x_transpose = x.transpose(1, 2)  # (N, C, L)
            x_norm = self.bn(x_transpose)
            x_norm = x_norm.transpose(1, 2)
        else:
            x_norm = self.bn(x)

        p = self.sigmoid(x_norm)

        # 广播 alpha
        if x.dim() == 3:
            alpha = self.alpha.unsqueeze(0).unsqueeze(1)
        else:
            alpha = self.alpha.unsqueeze(0)

        return p * x + (1 - p) * alpha * x


class DINAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, attention_hidden_units=[256, 128]):
        super(DINAttentionLayer, self).__init__()

        input_dim = embedding_dim * 4
        layers = []
        for hidden_unit in attention_hidden_units:
            layers.append(nn.Linear(input_dim, hidden_unit))
            layers.append(Dice(hidden_unit))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_unit

        layers.append(nn.Linear(input_dim, 1))
        self.attn_net = nn.Sequential(*layers)

    def forward(self, query, keys, keys_length):
        """
        query: (Batch, Emb_Dim)
        keys:  (Batch, Seq_Len, Emb_Dim)
        keys_length: (Batch, Seq_Len)  Mask张量，0为padding
        """
        seq_len = keys.size(1)
        query = query.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, D)

        # DIN 特征工程: [Q, K, Q-K, Q*K]
        din_input = torch.cat([
            query,
            keys,
            query - keys,
            query * keys
        ], dim=-1)

        # 计算分数 (B, L, 1)
        scores = self.attn_net(din_input)

        # reshape mask: (B, L) -> (B, L, 1)
        mask = keys_length.unsqueeze(-1)

        # 只保留真实历史记录的 score，padding 部分置 0
        scores = scores * mask

        # (B, L, 1) * (B, L, D) -> sum -> (B, D)
        output = torch.sum(scores * keys, dim=1)

        return output