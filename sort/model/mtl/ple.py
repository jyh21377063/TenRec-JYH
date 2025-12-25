import torch
import torch.nn as nn
from torch.nn import functional as F

from layer.din import DINAttentionLayer,Dice


class PLE_SEQ(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128,
                 n_specific_experts=1, n_shared_experts=1,
                 mmoe_hidden_dim=128, hidden_dim=[128, 128], dropouts=[0.5, 0.5],
                 output_size=1, expert_activation=F.relu, num_task=2, device=None):
        super(PLE_SEQ, self).__init__()

        # --- 基础检查与参数设置 (与原代码一致) ---
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = num_task
        self.n_specific_experts = n_specific_experts
        self.n_shared_experts = n_shared_experts

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")

        # --- Embedding 初始化 ---
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # 计算输入维度
        input_dim = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                    (len(self.user_feature_dict) - user_cate_feature_nums) + \
                    (len(self.item_feature_dict) - item_cate_feature_nums)
        hidden_size = input_dim + emb_dim  # 加上序列 Embedding


        # 1. Shared Experts (所有任务共享)
        self.shared_experts = nn.Parameter(
            torch.rand(hidden_size, mmoe_hidden_dim, n_shared_experts), requires_grad=True
        )
        self.shared_experts.data.normal_(0, 1)
        self.shared_experts_bias = nn.Parameter(
            torch.rand(mmoe_hidden_dim, n_shared_experts), requires_grad=True
        )

        # 2. Specific Experts (每个任务有自己的一组 Experts)
        # 这是一个 ParameterList，长度为 num_task
        self.specific_experts = nn.ParameterList([
            nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_specific_experts), requires_grad=True)
            for _ in range(num_task)
        ])
        self.specific_experts_bias = nn.ParameterList([
            nn.Parameter(torch.rand(mmoe_hidden_dim, n_specific_experts), requires_grad=True)
            for _ in range(num_task)
        ])
        # 初始化
        for i in range(num_task):
            self.specific_experts[i].data.normal_(0, 1)

        # 3. Gates (每个任务有自己的 Gate)
        # Gate 的输入维度是 hidden_size
        # Gate 的输出维度是: n_specific_experts + n_shared_experts (它只能选择自己的 experts 和 共享 experts)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.rand(hidden_size, n_specific_experts + n_shared_experts))
            for _ in range(num_task)
        ])
        self.gates_bias = nn.ParameterList([
            nn.Parameter(torch.rand(n_specific_experts + n_shared_experts))
            for _ in range(num_task)
        ])
        for gate in self.gates:
            gate.data.normal_(0, 1)

        # --- Tower 层---
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                layer = getattr(self, 'task_{}_dnn'.format(i + 1))
                layer.add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                layer.add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                layer.add_module('ctr_activation_{}'.format(j),nn.ReLU())
                layer.add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # --- Embedding 处理 (与原代码一致) ---
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # 序列特征处理 (Avg Pooling)
        item_id_emb_layer = getattr(self, 'item_id')
        seq_emb = item_id_emb_layer(x_seq.long())
        mask = (x_seq > 0).float().unsqueeze(-1)
        seq_sum = torch.sum(seq_emb * mask, dim=1)
        seq_len_valid = torch.sum(mask, dim=1)
        seq_avg = seq_sum / (seq_len_valid + 1e-8)

        # 拼接 Input
        input_layer = torch.cat([user_embed, item_embed, seq_avg], axis=1).float()  # (Batch, Input_Dim)

        # --- PLE Forward 逻辑 ---

        # 1. 计算 Shared Experts 的输出
        # shared_out: (Batch, Hidden, n_shared)
        shared_out = torch.einsum('ij, jkl -> ikl', input_layer, self.shared_experts)
        shared_out += self.shared_experts_bias
        if self.expert_activation is not None:
            shared_out = self.expert_activation(shared_out)

        # 2. 计算每个 Task 的 Specific Experts 输出，并与 Shared Concat
        cgc_outputs = list()

        for i in range(self.num_task):
            # A. 计算当前 Task 的 Specific Experts 输出
            # specific_out: (Batch, Hidden, n_specific)
            specific_out = torch.einsum('ij, jkl -> ikl', input_layer, self.specific_experts[i])
            specific_out += self.specific_experts_bias[i]
            if self.expert_activation is not None:
                specific_out = self.expert_activation(specific_out)

            # B. 拼接: [Specific Experts, Shared Experts]
            # combined_experts: (Batch, Hidden, n_specific + n_shared)
            combined_experts = torch.cat([specific_out, shared_out], dim=2)

            # C. Gate 控制
            # gate_out: (Batch, n_specific + n_shared)
            gate_out = torch.einsum('ab, bc -> ac', input_layer, self.gates[i])
            if self.gates_bias:
                gate_out += self.gates_bias[i]
            gate_out = nn.Softmax(dim=-1)(gate_out)

            # D. 加权求和
            expanded_gate = torch.unsqueeze(gate_out, 1)  # (Batch, 1, n_total_experts)
            weighted_output = combined_experts * expanded_gate.expand_as(combined_experts)
            final_expert_out = torch.sum(weighted_output, 2)  # (Batch, Hidden)

            cgc_outputs.append(final_expert_out)

        # --- Task Towers---
        task_outputs = list()
        for i in range(self.num_task):
            x_tower = cgc_outputs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x_tower = mod(x_tower)
            task_outputs.append(x_tower)

        return task_outputs


class PLE_DIN(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128,
                 n_specific_experts=1, n_shared_experts=1,
                 mmoe_hidden_dim=128, hidden_dim=[128, 128], din_hidden_dim=[256, 128], dropouts=[0.5, 0.5],
                 output_size=1, expert_activation=F.relu, num_task=2, device=None):
        super(PLE_DIN, self).__init__()

        # --- 基础检查与参数设置 (与原代码一致) ---
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = num_task
        self.n_specific_experts = n_specific_experts
        self.n_shared_experts = n_shared_experts

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")

        # --- Embedding 初始化 ---
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # 计算输入维度
        input_dim = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                    (len(self.user_feature_dict) - user_cate_feature_nums) + \
                    (len(self.item_feature_dict) - item_cate_feature_nums)
        hidden_size = input_dim + emb_dim  # 加上序列 Embedding
        self.din_attention = DINAttentionLayer(emb_dim, attention_hidden_units=din_hidden_dim)


        # 1. Shared Experts (所有任务共享)
        self.shared_experts = nn.Parameter(
            torch.rand(hidden_size, mmoe_hidden_dim, n_shared_experts), requires_grad=True
        )
        self.shared_experts.data.normal_(0, 1)
        self.shared_experts_bias = nn.Parameter(
            torch.rand(mmoe_hidden_dim, n_shared_experts), requires_grad=True
        )

        # 2. Specific Experts (每个任务有自己的一组 Experts)
        # 这是一个 ParameterList，长度为 num_task
        self.specific_experts = nn.ParameterList([
            nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_specific_experts), requires_grad=True)
            for _ in range(num_task)
        ])
        self.specific_experts_bias = nn.ParameterList([
            nn.Parameter(torch.rand(mmoe_hidden_dim, n_specific_experts), requires_grad=True)
            for _ in range(num_task)
        ])
        # 初始化
        for i in range(num_task):
            self.specific_experts[i].data.normal_(0, 1)

        # 3. Gates (每个任务有自己的 Gate)
        # Gate 的输入维度是 hidden_size
        # Gate 的输出维度是: n_specific_experts + n_shared_experts (它只能选择自己的 experts 和 共享 experts)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.rand(hidden_size, n_specific_experts + n_shared_experts))
            for _ in range(num_task)
        ])
        self.gates_bias = nn.ParameterList([
            nn.Parameter(torch.rand(n_specific_experts + n_shared_experts))
            for _ in range(num_task)
        ])
        for gate in self.gates:
            gate.data.normal_(0, 1)

        # --- Tower 层---
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                layer = getattr(self, 'task_{}_dnn'.format(i + 1))
                layer.add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                layer.add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                layer.add_module('ctr_activation_{}'.format(j), nn.ReLU())
                layer.add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # --- Embedding 处理---
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # 序列特征处理 (Avg Pooling)
        item_id_emb_layer = getattr(self, 'item_id')
        seq_emb = item_id_emb_layer(x_seq.long())
        item_id_idx = self.item_feature_dict['item_id'][1]
        target_item_emb = item_id_emb_layer(x[:, item_id_idx].long())  # (Batch, Emb_Dim)

        # 3. 准备 Mask
        mask = (x_seq > 0).float()  # (Batch, Seq_Len)
        seq = self.din_attention(target_item_emb, seq_emb, mask)

        # 拼接 Input
        input_layer = torch.cat([user_embed, item_embed, seq], axis=1).float()  # (Batch, Input_Dim)

        # --- PLE Forward 逻辑 ---

        # 1. 计算 Shared Experts 的输出
        # shared_out: (Batch, Hidden, n_shared)
        shared_out = torch.einsum('ij, jkl -> ikl', input_layer, self.shared_experts)
        shared_out += self.shared_experts_bias
        if self.expert_activation is not None:
            shared_out = self.expert_activation(shared_out)

        # 2. 计算每个 Task 的 Specific Experts 输出，并与 Shared Concat
        cgc_outputs = list()

        for i in range(self.num_task):
            # A. 计算当前 Task 的 Specific Experts 输出
            # specific_out: (Batch, Hidden, n_specific)
            specific_out = torch.einsum('ij, jkl -> ikl', input_layer, self.specific_experts[i])
            specific_out += self.specific_experts_bias[i]
            if self.expert_activation is not None:
                specific_out = self.expert_activation(specific_out)

            # B. 拼接: [Specific Experts, Shared Experts]
            # combined_experts: (Batch, Hidden, n_specific + n_shared)
            combined_experts = torch.cat([specific_out, shared_out], dim=2)

            # C. Gate 控制
            # gate_out: (Batch, n_specific + n_shared)
            gate_out = torch.einsum('ab, bc -> ac', input_layer, self.gates[i])
            if self.gates_bias:
                gate_out += self.gates_bias[i]
            gate_out = nn.Softmax(dim=-1)(gate_out)

            # D. 加权求和
            expanded_gate = torch.unsqueeze(gate_out, 1)  # (Batch, 1, n_total_experts)
            weighted_output = combined_experts * expanded_gate.expand_as(combined_experts)
            final_expert_out = torch.sum(weighted_output, 2)  # (Batch, Hidden)

            cgc_outputs.append(final_expert_out)

        # --- Task Towers---
        task_outputs = list()
        for i in range(self.num_task):
            x_tower = cgc_outputs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x_tower = mod(x_tower)
            task_outputs.append(x_tower)

        return task_outputs