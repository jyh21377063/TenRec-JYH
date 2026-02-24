import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNetV2(nn.Module):
    """ DCN-v2 的显式特征交叉层 (Matrix 形式) """
    def __init__(self, in_features, layer_num=2):
        super(CrossNetV2, self).__init__()
        self.layer_num = layer_num
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_features, in_features)) for _ in range(layer_num)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_features)) for _ in range(layer_num)
        ])
        for i in range(self.layer_num):
            nn.init.xavier_normal_(self.cross_weights[i])
            nn.init.zeros_(self.cross_bias[i])

    def forward(self, x):
        x_0 = x
        x_l = x
        for i in range(self.layer_num):
            # X_{l+1} = X_0 * (W_l * X_l + b_l) + X_l
            xl_w = torch.matmul(x_l, self.cross_weights[i]) + self.cross_bias[i]
            x_l = x_0 * xl_w + x_l
        return x_l