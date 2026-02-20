import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_emb, key_emb, neg_key_emb=None):
        """
        Args:
            query_emb: User Tower output (B, D)
            key_emb: Item Tower output (B, D)
            neg_key_emb: (Optional) Negative Item Tower output (B, D) or (B, K, D)
        Returns:
            loss: scalar
        """
        # 1. 归一化 (双保险)
        query_emb = F.normalize(query_emb, p=2, dim=1)
        key_emb = F.normalize(key_emb, p=2, dim=1)

        # 2. 计算 Batch 内的正负样本相似度 (B, B)
        # 每一行 i:
        # - col i: 正样本 (Positive)
        # - col j (j!=i): Batch 内负样本 (Batch Negatives)
        batch_logits = torch.matmul(query_emb, key_emb.T) / self.temperature

        # 3. 如果传入了额外的硬负样本/热度负样本
        if neg_key_emb is not None:
            neg_key_emb = F.normalize(neg_key_emb, p=2, dim=1)

            # Case A: 每个用户只有 1 个负样本 (B, D)
            if neg_key_emb.dim() == 2:
                # element-wise multiply then sum -> (B, 1)
                neg_logits = (query_emb * neg_key_emb).sum(dim=1, keepdim=True) / self.temperature

            # Case B: 每个用户有 K 个负样本 (B, K, D)
            elif neg_key_emb.dim() == 3:
                # (B, 1, D) * (B, K, D) -> (B, K)
                neg_logits = torch.bmm(neg_key_emb, query_emb.unsqueeze(2)).squeeze(2) / self.temperature

            # 4. 拼接 Logits
            # 结果形状: (B, B + K)
            # 前 B 列包含 1 个正样本和 B-1 个 Batch 负样本
            # 后 K 列包含 K 个 Hard 负样本
            logits = torch.cat([batch_logits, neg_logits], dim=1)
        else:
            logits = batch_logits

        # 5. 生成标签
        # 正样本永远在对角线位置，即 batch_logits 的第 i 列
        # 无论后面拼接了多少负样本，正样本的索引（Label）依然是 range(B)
        batch_size = query_emb.shape[0]
        labels = torch.arange(batch_size).to(query_emb.device)

        # 6. 计算 Loss
        loss = self.criterion(logits, labels)
        return loss
