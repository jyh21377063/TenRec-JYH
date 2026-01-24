import torch
import torch.nn as nn


class FMLayer(nn.Module):
    """
    Factorization Machine Layer
    计算所有 Embedding 之间的两两点积交互。
    公式: 0.5 * ( (sum(v))^2 - sum(v^2) )
    """

    def __init__(self):
        super(FMLayer, self).__init__()

    def forward(self, inputs):
        # inputs shape: [batch_size, field_size, embedding_size]

        # 1. 和的平方: (a+b)^2
        square_of_sum = torch.pow(torch.sum(inputs, dim=1), 2)

        # 2. 平方的和: a^2 + b^2
        sum_of_square = torch.sum(torch.pow(inputs, 2), dim=1)

        # 3. 结果: 0.5 * ((a+b)^2 - (a^2+b^2)) = ab
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        # 对 Embedding 维度求和，得到每个样本的一个标量分数 (或者保留维度作为特征)
        # 这里我们求和得到 (Batch, 1)，代表低阶交互的总强度
        return torch.sum(cross_term, dim=1, keepdim=True)