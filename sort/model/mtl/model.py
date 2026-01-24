import torch
import torch.nn as nn
from torch.nn import functional as F

from layer.din import DINAttentionLayer
from layer.fm import FMLayer
from layer.dcn import CrossNetV2


class MMOE_DCN_DIN(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, n_expert=3, mmoe_hidden_dim=128, dcn_layers=3,
                 din_hidden_dim=[256, 128], hidden_dim=[128, 64], dropouts=[0.5, 0.5],
                 output_size=1, expert_activation=F.relu, num_task=2, device=None):
        super(MMOE_DCN_DIN, self).__init__()

        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.emb_dim = emb_dim
        self.embeddings = nn.ModuleDict()
        self.dense_trans = nn.ModuleDict()
        self.expert_activation = expert_activation
        self.num_task = num_task
        if device:
            self.device = device

        all_features = {**self.user_feature_dict, **self.item_feature_dict}

        self.num_sparse_features = 0
        for name, num in all_features.items():
            if num[0] > 1:  # 离散特征
                self.embeddings[name] = nn.Embedding(num[0], emb_dim)
                self.num_sparse_features += 1
            else:  # 连续特征
                self.dense_trans[name] = nn.Linear(1, emb_dim)
                self.num_sparse_features += 1

        # A. DIN 组件
        self.din_attention = DINAttentionLayer(emb_dim, attention_hidden_units=din_hidden_dim)

        self.dcn_input_dim = self.num_sparse_features * emb_dim + emb_dim  # +emb_dim for DIN
        self.dcn = CrossNetV2(input_dim=self.dcn_input_dim, num_layers=dcn_layers)
        self.dcn_bottleneck = nn.Linear(self.dcn_input_dim, mmoe_hidden_dim)

        # DCN 输出通常还要过一个 Linear 压缩到 1 或者 task_num
        # 如果是作为 Wide 部分，通常直接映射到 output_size
        # self.dcn_output_layer = nn.Linear(self.dcn_input_dim, output_size)

        # Input Dim = (User+Item Embeddings) + (DIN Sequence Vector)
        # 注意：FM的输出通常直接加到最后，不进入DNN/Experts，或者作为一个标量特征进入
        input_dim = (self.num_sparse_features * emb_dim) + emb_dim

        # --- 3. MMOE 组件 (充当 DeepFM 中的 Deep 部分) ---

        # Experts
        self.experts = torch.nn.Parameter(torch.rand(input_dim, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 0.01)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)

        # Gates
        self.gates = nn.ParameterList([torch.nn.Parameter(torch.rand(input_dim, n_expert)) for _ in range(num_task)])
        self.gates_bias = nn.ParameterList([torch.nn.Parameter(torch.rand(n_expert)) for _ in range(num_task)])
        for gate in self.gates: gate.data.normal_(0, 0.01)

        # Towers (DNN for each task)
        tower_input_dim = mmoe_hidden_dim + mmoe_hidden_dim
        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [tower_input_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                layer = getattr(self, 'task_{}_dnn'.format(i + 1))
                layer.add_module('ctr_hidden_{}'.format(j), nn.Linear(hid_dim[j], hid_dim[j + 1]))
                layer.add_module('ctr_batchnorm_{}'.format(j), nn.BatchNorm1d(hid_dim[j + 1]))
                layer.add_module('ctr_activation_{}'.format(j), nn.ReLU())
                layer.add_module('ctr_dropout_{}'.format(j), nn.Dropout(dropouts[j]))

            # 最后一层
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        embed_list = []

        # 合并遍历，防止漏掉 item_list
        # 注意：这里假设 feature_dict 的顺序是固定的，或者你需要确保 list 顺序一致性
        # 更好的做法是遍历 self.embeddings 和 self.dense_trans 的 keys

        all_features = {**self.user_feature_dict, **self.item_feature_dict}

        for name, num in all_features.items():
            if num[0] > 1:  # 离散
                # [B, Emb_Dim]
                embed_list.append(self.embeddings[name](x[:, num[1]].long()))
            else:  # 连续
                # [B] -> [B, 1] -> [B, Emb_Dim]
                val = x[:, num[1]].unsqueeze(1).float()
                embed_list.append(self.dense_trans[name](val))

        # Flatten 供 DNN 使用
        deep_input_emb = torch.cat(embed_list, axis=1)  # [Batch, Total_Dim]

        # --- 2. DIN 处理序列特征 ---
        item_id_idx = self.item_feature_dict['item_id'][1]
        target_item_emb = self.embeddings['item_id'](x[:, item_id_idx].long())

        seq_emb = self.embeddings['item_id'](x_seq.long())
        mask = (x_seq > 0).float()

        # DIN Output

        din_out = self.din_attention(target_item_emb, seq_emb, mask)  # [Batch, Emb_Dim]

        # DCN 能够捕捉 deep_input_emb 内部的高阶显式交叉
        dcn_input = torch.cat([deep_input_emb, din_out], axis=1)
        dcn_out = self.dcn(dcn_input)
        dcn_compressed = self.dcn_bottleneck(dcn_out)  # [Batch, mmoe_hidden]

        # --- 4. 拼接输入给 MMOE ---
        # MMOE 的输入 = 原始特征 Flatten + DIN 提取的兴趣特征
        mmoe_input = torch.cat([deep_input_emb, din_out], axis=1).float()

        # --- 5. MMOE 运算 (Einsum) ---
        experts_out = torch.einsum('ij, jkl -> ikl', mmoe_input, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        task_outputs = list()
        for i in range(self.num_task):
            # A. Gate
            gate = self.gates[i]
            gate_out = torch.einsum('ab, bc -> ac', mmoe_input, gate)
            if self.gates_bias:
                gate_out += self.gates_bias[i]
            gate_out = nn.Softmax(dim=-1)(gate_out)

            # B. Weighted Sum of Experts
            expanded_gate = torch.unsqueeze(gate_out, 1)
            weighted_expert = experts_out * expanded_gate.expand_as(experts_out)
            expert_combined = torch.sum(weighted_expert, 2)

            # 让 Tower 同时看到“隐式特征(MMOE)”和“显式交叉特征(DCN)”
            tower_input = torch.cat([expert_combined, dcn_compressed], dim=1)

            # C. Task Tower
            x_tower = tower_input
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x_tower = mod(x_tower)

            # 输出即为最终 Logits，无需再加 dcn_out
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

class ESMM_DCN_DIN(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128,
                 hidden_dim=[128, 64], dropouts=[0.5, 0.5], din_hidden_dim=[256, 128],
                 dcn_layers=3, output_size=1, num_task=2):
        super(ESMM_DCN_DIN, self).__init__()

        # === 1. 参数校验与保存 ===
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("Feature dicts must not be None")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task

        self.embeddings = nn.ModuleDict()
        self.dense_trans = nn.ModuleDict()

        all_feats = {**user_feature_dict, **item_feature_dict}
        self.num_sparse = 0
        self.num_dense = 0

        for name, num in all_feats.items():
            if num[0] > 1:
                self.embeddings[name] = nn.Embedding(num[0], emb_dim)
                self.num_sparse += 1
            else:
                self.dense_trans[name] = nn.Linear(1, emb_dim)
                self.num_dense += 1

        # === 3. 核心组件初始化 ===

        # A. DIN 组件
        # 必须确保 item_id 在字典中
        if 'item_id' not in self.item_feature_dict:
            raise ValueError("DIN requires 'item_id' feature")
        self.din_attention = DINAttentionLayer(emb_dim, attention_hidden_units=din_hidden_dim)

        # B. DCN 组件
        # DCN 输入维度 = 所有特征 flattened 的长度
        total_static_dim = (self.num_sparse + self.num_dense) * emb_dim
        self.dcn_input_dim = total_static_dim + emb_dim
        self.dcn = CrossNetV2(input_dim=self.dcn_input_dim, num_layers=dcn_layers)

        # === 4. 计算 Task Tower 的输入维度 ===
        # 输入 = [原始特征 Flatten] + [DIN 序列兴趣] + [DCN 交叉特征]
        # 注意：DCN 输出维度等于输入维度
        tower_input_dim = self.dcn_input_dim + self.dcn_input_dim

        # === 5. ESMM Task Towers ===
        self.ctr_tower = self._make_tower(tower_input_dim, hidden_dim, dropouts, output_size)
        self.cvr_tower = self._make_tower(tower_input_dim, hidden_dim, dropouts, output_size)

    def _make_tower(self, input_dim, hidden_dim, dropouts, output_size):
        layers = []
        for h, drop in zip(hidden_dim, dropouts):
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
            input_dim = h
        layers.append(nn.Linear(input_dim, output_size))
        return nn.Sequential(*layers)

    def forward(self, x, x_seq):
        # x: [Batch, Num_Features] (全是离散特征的 Index)

        # === 1. 获取 Embeddings ===
        embed_list = []
        all_feats = {**self.user_feature_dict, **self.item_feature_dict}

        for name, num in all_feats.items():
            if num[0] > 1:
                embed_list.append(self.embeddings[name](x[:, num[1]].long()))
            else:
                val = x[:, num[1]].unsqueeze(1).float()
                embed_list.append(self.dense_trans[name](val))

        # [Batch, Total_Feats * Emb_Dim] -> 用于 DCN 和 拼接
        flat_input_emb = torch.cat(embed_list, dim=1)

        # --- B. DIN 序列特征提取 ---
        # 动态获取 item_id 的索引
        item_col_idx = self.item_feature_dict[self.target_item_col][1]
        target_item_emb = self.embeddings[self.target_item_col](x[:, item_col_idx].long())
        seq_emb = self.embeddings[self.target_item_col](x_seq.long())
        mask = (x_seq > 0).float()

        din_out = self.din_attention(target_item_emb, seq_emb, mask)

        dcn_in = torch.cat([flat_input_emb, din_out], dim=1)
        dcn_out = self.dcn(dcn_in)

        # --- D. 拼接最终特征 ---
        final_input = torch.cat([dcn_in, dcn_out], dim=1)

        # --- E. 双塔输出 ---
        ctr_logit = self.ctr_tower(final_input)
        cvr_logit = self.cvr_tower(final_input)

        # --- F. 返回 List (保持兼容) ---
        # 约定：第一个是 CTR，第二个是 CVR
        return [ctr_logit, cvr_logit]