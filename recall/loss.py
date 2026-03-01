import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_emb, key_emb, neg_key_emb=None, key_log_p=None, neg_log_p=None):
        query_emb = F.normalize(query_emb, p=2, dim=1)
        key_emb = F.normalize(key_emb, p=2, dim=1)

        batch_logits = torch.matmul(query_emb, key_emb.T) / self.temperature

        # === 1. In-Batch 纠偏 ===
        # key_log_p 的形状应为 (B,)，代表 Batch 内每个正样本的对数采得概率
        if key_log_p is not None:
            # 广播机制：第 j 列减去第 j 个样本的 log_p
            batch_logits = batch_logits - key_log_p.unsqueeze(0)

        if neg_key_emb is not None:
            neg_key_emb = F.normalize(neg_key_emb, p=2, dim=1)

            if neg_key_emb.dim() == 2:
                neg_logits = (query_emb * neg_key_emb).sum(dim=1, keepdim=True) / self.temperature
            elif neg_key_emb.dim() == 3:
                neg_logits = torch.bmm(neg_key_emb, query_emb.unsqueeze(2)).squeeze(2) / self.temperature

            # === 2. Hard Negative 纠偏 ===
            if neg_log_p is not None:
                # neg_log_p 若是 2D 需要变 2D, 若是 3D 的话就对应形状相减
                if neg_log_p.dim() == 1:
                    neg_log_p = neg_log_p.unsqueeze(1)  # 转为 (B, 1)
                neg_logits = neg_logits - neg_log_p

            logits = torch.cat([batch_logits, neg_logits], dim=1)
        else:
            logits = batch_logits

        batch_size = query_emb.shape[0]
        labels = torch.arange(batch_size).to(query_emb.device)

        loss = self.criterion(logits, labels)
        return loss
