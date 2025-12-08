import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_emb, key_emb):
        """
        Args:
            query_emb: User Tower output (B, D)
            key_emb: Item Tower output (B, D)
        Returns:
            loss: scalar
        """
        # 1. 再次确保归一化 (虽然模型输出通常已归一化，双重保险)
        query_emb = F.normalize(query_emb, p=2, dim=1)
        key_emb = F.normalize(key_emb, p=2, dim=1)

        # 2. 计算相似度矩阵 (B, B)
        # 每一行代表一个 User，列代表 Batch 内所有的 Item
        # 对角线位置 (i, i) 是正样本，其他位置 (i, j) 是负样本
        logits = torch.matmul(query_emb, key_emb.T) / self.temperature

        # 3. 生成标签
        # 只要数据是对齐输入的，Label 就是 [0, 1, 2, ... B-1]
        batch_size = query_emb.shape[0]
        labels = torch.arange(batch_size).to(query_emb.device)

        # 4. 计算 Cross Entropy
        loss = self.criterion(logits, labels)
        return loss


class MixedInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, hard_neg_weight=1.0):
        """
        Args:
            temperature: 温度系数
            hard_neg_weight: 困难负样本的权重 (相当于复制了多少遍)
        """
        super(MixedInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.hard_neg_weight = hard_neg_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_emb, pos_item_emb, hard_neg_emb=None):
        """
        Args:
            query_emb: (B, D) User embedding
            pos_item_emb: (B, D) Positive Item embedding
            hard_neg_emb: (B, D) or (B, K, D) Hard Negative embedding (Optional)
        """
        batch_size = query_emb.shape[0]

        # 1. 归一化 (防止外部忘了归一化)
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_item_emb = F.normalize(pos_item_emb, p=2, dim=1)

        # 2. 计算 In-Batch Logits (B, B)
        # 每一行 i 的正样本在 column i
        in_batch_logits = torch.matmul(query_emb, pos_item_emb.T) / self.temperature

        if hard_neg_emb is not None:
            # 处理 Hard Negatives
            hard_neg_emb = F.normalize(hard_neg_emb, p=2, dim=-1)

            # 如果输入是 (B, D)，先变成 (B, 1, D) 以便通用处理
            if hard_neg_emb.dim() == 2:
                hard_neg_emb = hard_neg_emb.unsqueeze(1)  # (B, 1, D)

            # 计算点积: (B, 1, D) * (B, 1, D) -> sum -> (B, 1)
            # 或者如果有 K 个负样本: (B, K)
            # 这里利用 einsum 做批量点积: query(b, d), hard(b, k, d) -> out(b, k)
            hard_neg_logits = torch.einsum('bd,bkd->bk', query_emb, hard_neg_emb) / self.temperature

            # 3. 施加权重 (Bias Trick)
            # log(w) 加在 logit 上等价于概率乘以 w
            if self.hard_neg_weight != 1.0:
                hard_neg_logits = hard_neg_logits + torch.log(torch.tensor(self.hard_neg_weight).to(query_emb.device))

            # 4. 拼接 Logits (B, B + K)
            full_logits = torch.cat([in_batch_logits, hard_neg_logits], dim=1)
        else:
            full_logits = in_batch_logits

        # 5. 生成标签
        # 正样本永远在 index = 0, 1, ..., B-1 (即对角线位置)
        # 哪怕后面拼接了负样本，正样本的索引依然是 row_index
        labels = torch.arange(batch_size).to(query_emb.device)

        loss = self.criterion(full_logits, labels)
        return loss