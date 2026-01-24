import torch
import torch.nn as nn
from torch.nn import functional as F

from layer.din import DINAttentionLayer,Dice
from layer.dcn import CrossNetV2


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


class PLE_DCN_DIN(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128,
                 n_specific_experts=1, n_shared_experts=1,
                 mmoe_hidden_dim=128, hidden_dim=[128, 128], din_hidden_dim=[256, 128], dropouts=[0.5, 0.5], dcn_layers=3,
                 output_size=1, expert_activation=F.relu, num_task=2, device=None):
        super(PLE_DCN_DIN, self).__init__()

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
        self.target_item_col = "item_id"

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.embeddings = nn.ModuleDict()
        self.dense_trans = nn.ModuleDict()  # 用于连续特征

        all_features = {**user_feature_dict, **item_feature_dict}
        self.num_sparse = 0
        self.num_dense = 0

        for name, num in all_features.items():
            if num[0] > 1:  # 离散特征
                self.embeddings[name] = nn.Embedding(num[0], emb_dim)
                self.num_sparse += 1
            else:  # 连续特征
                self.dense_trans[name] = nn.Linear(1, emb_dim)
                self.num_dense += 1

        # 计算输入维度 (所有特征都映射成了 emb_dim)
        total_feats = self.num_sparse + self.num_dense
        input_dim = total_feats * emb_dim

        # DIN 部分
        self.din_attention = DINAttentionLayer(emb_dim, attention_hidden_units=din_hidden_dim)

        # DCN 输入 = 静态特征 + DIN序列特征
        dcn_input_dim = input_dim + emb_dim

        # PLE 的核心输入维度
        ple_input_dim = dcn_input_dim
        hidden_size = input_dim + emb_dim  # 加上序列 Embedding
        self.dcn = CrossNetV2(input_dim=hidden_size, num_layers=dcn_layers)


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
        tower_input_dim = mmoe_hidden_dim + ple_input_dim
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [tower_input_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                layer = getattr(self, 'task_{}_dnn'.format(i + 1))
                layer.add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                layer.add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                layer.add_module('ctr_activation_{}'.format(j), nn.ReLU())
                layer.add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        embed_list = []
        all_features = {**self.user_feature_dict, **self.item_feature_dict}

        # 1. 统一处理 Embedding 和 Dense Layer
        for name, num in all_features.items():
            if num[0] > 1:
                embed_list.append(self.embeddings[name](x[:, num[1]].long()))
            else:
                # 连续特征过 Linear 层
                val = x[:, num[1]].unsqueeze(1).float()
                embed_list.append(self.dense_trans[name](val))

        input_embed = torch.cat(embed_list, axis=1)  # [B, Total_Feats * Emb]

        # 2. DIN 逻辑 (使用 self.target_item_col)
        item_col_idx = self.item_feature_dict[self.target_item_col][1]
        target_item_emb = self.embeddings[self.target_item_col](x[:, item_col_idx].long())
        seq_emb = self.embeddings[self.target_item_col](x_seq.long())
        mask = (x_seq > 0).float()

        seq_interest = self.din_attention(target_item_emb, seq_emb, mask)

        # 3. 拼接输入
        input_layer = torch.cat([input_embed, seq_interest], axis=1).float()

        # 4. DCN
        dcn_output = self.dcn(input_layer)

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
            ple_out = cgc_outputs[i]

            # 将 PLE 的抽象特征 与 DCN 的显式交叉特征 拼接
            tower_input = torch.cat([ple_out, dcn_output], dim=1)
            x_tower = tower_input
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x_tower = mod(x_tower)
            task_outputs.append(x_tower)

        return task_outputs