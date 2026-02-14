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
