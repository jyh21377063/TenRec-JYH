import torch
import torch.nn as nn
from torch.nn import functional as F

from layer.din import DINAttentionLayer
from layer.fm import FMLayer
from layer.dcn import CrossNetV2


class MMOE(nn.Module):

    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, n_expert=2, mmoe_hidden_dim=128,
                 hidden_dim=[128, 128], dropouts=[0.5, 0.5], output_size=1, expert_activation=F.relu, num_task=2, device=None):
        super(MMOE, self).__init__()
        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = num_task

        if device:
            self.device = device

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # user embedding + item embedding
        hidden_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                      (len(self.user_feature_dict) - user_cate_feature_nums) + (
                              len(self.item_feature_dict) - item_cate_feature_nums)

        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        # gates
        self.gates = nn.ParameterList([
            torch.nn.Parameter(torch.rand(hidden_size, n_expert))
            for _ in range(num_task)
        ])
        # 初始化
        for gate in self.gates:
            gate.data.normal_(0, 1)

        self.gates_bias = nn.ParameterList([
            torch.nn.Parameter(torch.rand(n_expert))
            for _ in range(num_task)
        ])

        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_activation_{}'.format(j),
                                                                      nn.ReLU())
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
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

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # hidden layer
        hidden = torch.cat([user_embed, item_embed], axis=1).float()  # batch * hidden_size

        # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)  # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gates_out = list()
        for idx, gate in enumerate(self.gates):
            # gate = gate.to(self.device)
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)  # batch * num_experts
            if self.gates_bias:
                # self.gates_bias[idx] = self.gates_bias[idx].to(self.device)
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)  # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out)  # batch * mmoe_hidden_size * num_experts
            outs.append(torch.sum(weighted_expert_output, 2))  # batch * mmoe_hidden_size

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)

        return task_outputs


class MMOE_SEQ(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, n_expert=2, mmoe_hidden_dim=128,
                 hidden_dim=[128, 128], dropouts=[0.5, 0.5], output_size=1, expert_activation=F.relu, num_task=2, device=None):
        super(MMOE_SEQ, self).__init__()
        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = num_task

        if device:
            self.device = device

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # user embedding + item embedding
        input_dim = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                    (len(self.user_feature_dict) - user_cate_feature_nums) + \
                    (len(self.item_feature_dict) - item_cate_feature_nums)

        # 加上序列带来的维度
        hidden_size = input_dim + emb_dim

        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        # gates
        self.gates = nn.ParameterList([
            torch.nn.Parameter(torch.rand(hidden_size, n_expert))
            for _ in range(num_task)
        ])
        # 初始化
        for gate in self.gates:
            gate.data.normal_(0, 1)

        self.gates_bias = nn.ParameterList([
            torch.nn.Parameter(torch.rand(n_expert))
            for _ in range(num_task)
        ])

        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_activation_{}'.format(j),
                                                                      nn.ReLU())
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
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

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        item_id_emb_layer = getattr(self, 'item_id')

        seq_emb = item_id_emb_layer(x_seq.long())  # (Batch, Seq_Len, Emb_Dim)

        # Masking: 忽略 padding (0) 的位置
        mask = (x_seq > 0).float().unsqueeze(-1)  # (Batch, Seq_Len, 1)

        # Sum Pooling over sequence length
        seq_sum = torch.sum(seq_emb * mask, dim=1)  # (Batch, Emb_Dim)

        # Compute Average (防止除以0)
        seq_len_valid = torch.sum(mask, dim=1)
        seq_avg = seq_sum / (seq_len_valid + 1e-8)  # (Batch, Emb_Dim)

        # 将 User, Item, 和 Sequence Avg 拼接
        hidden = torch.cat([user_embed, item_embed, seq_avg], axis=1).float()

        # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)  # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gates_out = list()
        for idx, gate in enumerate(self.gates):
            # gate = gate.to(self.device)
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)  # batch * num_experts
            if self.gates_bias:
                # self.gates_bias[idx] = self.gates_bias[idx].to(self.device)
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)  # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out)  # batch * mmoe_hidden_size * num_experts
            outs.append(torch.sum(weighted_expert_output, 2))  # batch * mmoe_hidden_size

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)

        return task_outputs



class MMOE_DIN(nn.Module):
    def __init__(self, user_feature_dict, item_feature_dict, emb_dim=128, n_expert=2, mmoe_hidden_dim=128, din_hidden_dim=[256, 128],
                 hidden_dim=[128, 128], dropouts=[0.5, 0.5], output_size=1, expert_activation=F.relu, num_task=2, device=None):
        super(MMOE_DIN, self).__init__()
        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = num_task

        if device:
            self.device = device

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # user embedding + item embedding
        input_dim = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                    (len(self.user_feature_dict) - user_cate_feature_nums) + \
                    (len(self.item_feature_dict) - item_cate_feature_nums)

        # 加上序列带来的维度
        hidden_size = input_dim + emb_dim
        self.din_attention = DINAttentionLayer(emb_dim, attention_hidden_units=din_hidden_dim)

        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        # gates
        self.gates = nn.ParameterList([
            torch.nn.Parameter(torch.rand(hidden_size, n_expert))
            for _ in range(num_task)
        ])
        # 初始化
        for gate in self.gates:
            gate.data.normal_(0, 1)

        self.gates_bias = nn.ParameterList([
            torch.nn.Parameter(torch.rand(n_expert))
            for _ in range(num_task)
        ])

        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_activation_{}'.format(j),
                                                                      nn.ReLU())
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, x_seq):
        # assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
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

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        item_id_emb_layer = getattr(self, 'item_id')

        seq_emb = item_id_emb_layer(x_seq.long())  # (Batch, Seq_Len, Emb_Dim)

        item_id_idx = self.item_feature_dict['item_id'][1]
        target_item_emb = item_id_emb_layer(x[:, item_id_idx].long())  # (Batch, Emb_Dim)

        # 3. 准备 Mask
        mask = (x_seq > 0).float()  # (Batch, Seq_Len)
        seq = self.din_attention(target_item_emb, seq_emb, mask)

        # 将 User, Item, 和 Sequence Avg 拼接
        hidden = torch.cat([user_embed, item_embed, seq], axis=1).float()

        # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)  # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gates_out = list()
        for idx, gate in enumerate(self.gates):
            # gate = gate.to(self.device)
            gate_out = torch.einsum('ab, bc -> ac', hidden, gate)  # batch * num_experts
            if self.gates_bias:
                # self.gates_bias[idx] = self.gates_bias[idx].to(self.device)
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)  # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out)  # batch * mmoe_hidden_size * num_experts
            outs.append(torch.sum(weighted_expert_output, 2))  # batch * mmoe_hidden_size

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)

        return task_outputs

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