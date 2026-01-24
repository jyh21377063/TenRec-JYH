import torch
import torch.nn as nn

class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim

        # W: (num_layers, input_dim, input_dim)
        # 为了节省显存，通常使用低秩矩阵分解或者直接用 Linear
        # 这里为了演示清晰，使用标准矩阵形式 (也可以优化为 input_dim -> 1 的向量形式即 DCN-V1，或矩阵形式 DCN-V2)
        # 工业界常用做法：使用 nn.Linear 模拟 W * x + b
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [Batch, Input_Dim]
        x0 = x
        xi = x

        for layer in self.cross_layers:
            # Cross Net 公式: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
            # 1. 计算 feature_crossing = W * xi + b
            feature_crossing = layer(xi)

            # 2. 核心：与 x0 进行元素级相乘 (Element-wise product)
            # 注意：DCN v2 论文中通常是 x0 * (W xi + b)，这里 * 是哈达玛积
            xi = x0 * feature_crossing + xi

        return xi